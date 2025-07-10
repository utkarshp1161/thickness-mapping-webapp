from flask import Flask, render_template, request, jsonify, send_file
import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import hyperspy.api as hs
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
import cv2
import traceback

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to store data
loaded_images = None
current_image = None
original_image = None
smoothed_image = None
pixel_size = None
vertical_profiles = None
detected_peaks = []
manual_peaks = []

def normalize(img):
    """Normalize image to 0-1 range"""
    return (img - img.min()) / (img.max() - img.min())

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    try:
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
        img_buffer.seek(0)
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        plt.close(fig)
        return img_str
    except Exception as e:
        plt.close(fig)
        print(f"Error converting plot to base64: {e}")
        return None

def image_to_base64(image_array):
    """Convert numpy array to base64 string for display"""
    try:
        if image_array.dtype != np.uint8:
            img_normalized = cv2.normalize(image_array.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img_normalized = image_array
        
        _, buffer = cv2.imencode('.png', img_normalized)
        img_str = base64.b64encode(buffer).decode()
        return img_str
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None

def generate_analysis_plot():
    """Generate the main analysis plot with three panels"""
    global original_image, smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    try:
        if original_image is None or smoothed_image is None:
            return None
        
        # Combine auto-detected and manual peaks
        all_peaks = list(detected_peaks) + list(manual_peaks)
        all_peaks = sorted(set(all_peaks))  # Remove duplicates and sort
        
        fig, (ax_img_raw, ax_img, ax_prof) = plt.subplots(1, 3, figsize=(20, 7), sharey=True)
        
        # Left: Original Image
        ax_img_raw.imshow(original_image, cmap='gray', aspect='auto', origin="lower")
        ax_img_raw.set_title("Original Image")
        ax_img_raw.set_xlabel("X (pixels)")
        ax_img_raw.set_ylabel("Y (pixels)")
        
        # Middle: Smoothed Image with detected interfaces
        ax_img.imshow(smoothed_image, cmap='gray', aspect='auto', origin="lower")
        ax_img.set_title("Smoothed Image with Interfaces")
        ax_img.set_xlabel("X (pixels)")
        ax_img.set_ylabel("Y (pixels)")
        
        # Plot interfaces
        for y in all_peaks:
            if y in detected_peaks:
                color = 'cyan'
                linestyle = ':'
                linewidth = 1
            else:
                color = 'red'
                linestyle = '--'
                linewidth = 2
            ax_img.axhline(y, color=color, linestyle=linestyle, linewidth=linewidth)
        
        # Right: Vertical Profiles
        if vertical_profiles is not None:
            y_pixels = np.arange(len(vertical_profiles['mean']))
            ax_prof.plot(vertical_profiles['mean'], y_pixels, label="Mean", color='black')
            ax_prof.plot(vertical_profiles['std'], y_pixels, label="Std Dev", alpha=0.5, color='blue')
            ax_prof.plot(vertical_profiles['gradient'], y_pixels, label="Gradient", linestyle='--', alpha=0.5, color='green')
            
            # Plot interface lines
            auto_labeled = False
            manual_labeled = False
            for y in all_peaks:
                if y in detected_peaks:
                    color = 'cyan'
                    linestyle = '--'
                    label = "Auto Interfaces" if not auto_labeled else None
                    auto_labeled = True
                else:
                    color = 'red'
                    linestyle = '-'
                    label = "Manual Interfaces" if not manual_labeled else None
                    manual_labeled = True
                ax_prof.axhline(y, linestyle=linestyle, color=color, alpha=0.8, label=label)
        
        ax_prof.set_title("Vertical Profiles & Detected Interfaces")
        ax_prof.set_xlabel("Intensity")
        ax_prof.set_ylabel("Y (pixels)")
        ax_prof.invert_yaxis()
        ax_prof.legend()
        ax_prof.grid(True, linestyle=':', alpha=0.4)
        
        # Custom Y-axis ticks
        yticks = np.arange(0, original_image.shape[0], 50)
        ax_img_raw.set_yticks(yticks)
        ax_img.set_yticks(yticks)
        ax_prof.set_yticks(yticks)
        
        plt.tight_layout()
        return plot_to_base64(fig)
        
    except Exception as e:
        print(f"Error generating analysis plot: {e}")
        traceback.print_exc()
        return None

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error. Please check the console for details.'}), 500

@app.route('/')
def index():
    return render_template('thickness.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global loaded_images, pixel_size
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file selected'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and file.filename.endswith('.emd'):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            file.save(filepath)
            
            print(f"Loading EMD file: {filepath}")
            loaded_data = hs.load(filepath)
            
            if isinstance(loaded_data, list):
                loaded_images = loaded_data
            else:
                loaded_images = [loaded_data]
            
            images_info = []
            for i, img in enumerate(loaded_images):
                try:
                    if hasattr(img, 'original_metadata') and img.original_metadata:
                        try:
                            pixel_size = float(img.original_metadata["BinaryResult"]["PixelSize"]["height"]) * 1e9
                        except (KeyError, TypeError):
                            pixel_size = 1.0
                    else:
                        pixel_size = 1.0
                    
                    image_data = np.array(img.data)
                    preview = image_to_base64(image_data)
                    if preview is None:
                        continue
                        
                    images_info.append({
                        'index': i,
                        'shape': image_data.shape,
                        'dtype': str(image_data.dtype),
                        'pixel_size': pixel_size,
                        'preview': preview
                    })
                except Exception as e:
                    print(f"Error processing image {i}: {e}")
                    continue
            
            if not images_info:
                return jsonify({'error': 'No valid images found in the EMD file'})
            
            print(f"Loaded {len(images_info)} images from EMD file")
            
            return jsonify({
                'success': True,
                'images': images_info,
                'file_size': os.path.getsize(filepath)
            })
        
        return jsonify({'error': 'Please upload a .emd file'})
        
    except Exception as e:
        print(f"Error in upload_file: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error loading EMD file: {str(e)}'})

@app.route('/select_image', methods=['POST'])
def select_image():
    global current_image, original_image, pixel_size, smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    try:
        if loaded_images is None:
            return jsonify({'error': 'No images loaded'})
        
        data = request.json
        image_index = int(data.get('index', 0))
        
        if image_index < 0 or image_index >= len(loaded_images):
            return jsonify({'error': 'Invalid image index'})
        
        selected_img = loaded_images[image_index]
        
        try:
            pixel_size = float(selected_img.original_metadata["BinaryResult"]["PixelSize"]["height"]) * 1e9
        except (KeyError, TypeError):
            pixel_size = 1.0
        
        # Store original image and normalize
        raw_image = np.array(selected_img.data).astype(np.float32)
        original_image = normalize(raw_image)
        current_image = original_image.copy()
        
        # Reset analysis when new image is selected
        smoothed_image = None
        vertical_profiles = None
        detected_peaks = []
        manual_peaks = []
        
        print(f"Selected image {image_index}. Shape: {current_image.shape}")
        
        preview = image_to_base64(current_image)
        if preview is None:
            return jsonify({'error': 'Error generating image preview'})
        
        return jsonify({
            'success': True,
            'image_shape': current_image.shape,
            'pixel_size': pixel_size,
            'image_preview': preview
        })
        
    except Exception as e:
        print(f"Error in select_image: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error selecting image: {str(e)}'})

@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    global current_image, smoothed_image, vertical_profiles
    
    try:
        if current_image is None:
            return jsonify({'error': 'No image selected'})
        
        data = request.json
        sigma = float(data.get('sigma', 4.0))
        
        print(f"Preprocessing with sigma={sigma}")
        
        # Apply Gaussian smoothing
        smoothed_image = gaussian_filter(current_image, sigma=sigma)
        
        # Compute vertical profiles
        vertical_mean = np.mean(smoothed_image, axis=1)
        vertical_std = np.std(smoothed_image, axis=1)
        vertical_grad = np.gradient(vertical_mean)
        
        vertical_profiles = {
            'mean': vertical_mean,
            'std': vertical_std,
            'gradient': vertical_grad
        }
        
        # Generate analysis plot
        plot_base64 = generate_analysis_plot()
        smoothed_preview = image_to_base64(smoothed_image)
        
        if plot_base64 is None or smoothed_preview is None:
            return jsonify({'error': 'Error generating plots'})
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'smoothed_image': smoothed_preview,
            'stats': {
                'mean_intensity': float(np.mean(smoothed_image)),
                'std_intensity': float(np.std(smoothed_image)),
                'gradient_range': [float(np.min(vertical_grad)), float(np.max(vertical_grad))]
            }
        })
        
    except Exception as e:
        print(f"Error in preprocess_image: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error preprocessing image: {str(e)}'})

@app.route('/detect_peaks', methods=['POST'])
def detect_peaks():
    global vertical_profiles, detected_peaks
    
    try:
        if vertical_profiles is None:
            return jsonify({'error': 'No preprocessed image available'})
        
        data = request.json
        distance = int(data.get('distance', 30))
        height = float(data.get('height', 0.001))
        top_n = int(data.get('top_n', 100))
        
        print(f"Detecting peaks with distance={distance}, height={height}, top_n={top_n}")
        
        # Find peaks in gradient
        grad = vertical_profiles['gradient']
        peaks, _ = find_peaks(np.abs(grad), distance=distance, height=height)
        
        # Select strongest N peaks
        if len(peaks) > top_n:
            idx = np.argsort(np.abs(grad[peaks]))[::-1][:top_n]
            peaks = peaks[idx]
        
        detected_peaks = sorted(peaks.tolist())
        
        # Generate updated analysis plot
        plot_base64 = generate_analysis_plot()
        
        if plot_base64 is None:
            return jsonify({'error': 'Error generating plot'})
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'peaks_found': len(detected_peaks),
            'peaks': detected_peaks
        })
        
    except Exception as e:
        print(f"Error in detect_peaks: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error detecting peaks: {str(e)}'})

@app.route('/add_manual_peak_region', methods=['POST'])
def add_manual_peak_region():
    global manual_peaks, vertical_profiles
    
    try:
        if vertical_profiles is None:
            return jsonify({'error': 'No preprocessed image available'})
        
        data = request.json
        x_start = int(data.get('x_start', 0))
        x_end = int(data.get('x_end', 100))
        y_start = int(data.get('y_start', 0))
        y_end = int(data.get('y_end', 100))
        
        print(f"Adding manual peak in region x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        # Validate range
        if y_start >= y_end or y_start < 0 or y_end >= len(vertical_profiles['gradient']):
            return jsonify({'error': 'Invalid Y range specified'})
        
        # Find the strongest gradient in the Y range
        grad_region = vertical_profiles['gradient'][y_start:y_end]
        if len(grad_region) == 0:
            return jsonify({'error': 'Empty Y range specified'})
        
        offset = np.argmax(np.abs(grad_region))
        manual_peak = y_start + offset
        
        # Add to manual peaks if not already present
        if manual_peak not in manual_peaks:
            manual_peaks.append(manual_peak)
            manual_peaks.sort()
            
            # Generate updated analysis plot
            plot_base64 = generate_analysis_plot()
            
            if plot_base64 is None:
                return jsonify({'error': 'Error generating plot'})
            
            return jsonify({
                'success': True,
                'plot': plot_base64,
                'peak_added': manual_peak,
                'manual_peaks': manual_peaks,
                'message': f'Interface added at Y={manual_peak} (selected region: x[{x_start}:{x_end}], y[{y_start}:{y_end}])'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Interface at Y={manual_peak} already exists'
            })
            
    except Exception as e:
        print(f"Error in add_manual_peak_region: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error adding manual peak: {str(e)}'})

@app.route('/remove_manual_peak', methods=['POST'])
def remove_manual_peak():
    global manual_peaks
    
    try:
        data = request.json
        peak_to_remove = int(data.get('peak', 0))
        
        if peak_to_remove in manual_peaks:
            manual_peaks.remove(peak_to_remove)
            
            # Generate updated analysis plot
            plot_base64 = generate_analysis_plot()
            
            if plot_base64 is None:
                return jsonify({'error': 'Error generating plot'})
            
            return jsonify({
                'success': True,
                'plot': plot_base64,
                'peak_removed': peak_to_remove,
                'manual_peaks': manual_peaks
            })
        else:
            return jsonify({'error': 'Peak not found in manual peaks'})
            
    except Exception as e:
        print(f"Error in remove_manual_peak: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error removing manual peak: {str(e)}'})

@app.route('/calculate_thickness', methods=['POST'])
def calculate_thickness():
    global detected_peaks, manual_peaks, pixel_size
    
    try:
        if len(detected_peaks) == 0 and len(manual_peaks) == 0:
            return jsonify({'error': 'No peaks detected or manually added'})
        
        # Combine all peaks
        all_peaks = list(detected_peaks) + list(manual_peaks)
        all_peaks = sorted(set(all_peaks))
        
        if len(all_peaks) < 2:
            return jsonify({'error': 'At least 2 interfaces needed for thickness calculation'})
        
        # Calculate thicknesses between consecutive interfaces
        thicknesses = []
        for i in range(len(all_peaks) - 1):
            thickness_pixels = all_peaks[i+1] - all_peaks[i]
            thickness_nm = thickness_pixels * pixel_size
            thicknesses.append({
                'layer': i + 1,
                'start_interface': all_peaks[i],
                'end_interface': all_peaks[i+1],
                'thickness_pixels': thickness_pixels,
                'thickness_nm': thickness_nm
            })
        
        # Calculate statistics
        thickness_values = [t['thickness_nm'] for t in thicknesses]
        stats = {
            'mean_thickness': float(np.mean(thickness_values)),
            'std_thickness': float(np.std(thickness_values)),
            'min_thickness': float(np.min(thickness_values)),
            'max_thickness': float(np.max(thickness_values)),
            'total_layers': len(thicknesses),
            'total_interfaces': len(all_peaks),
            'auto_interfaces': len(detected_peaks),
            'manual_interfaces': len(manual_peaks),
            'pixel_size': pixel_size
        }
        
        return jsonify({
            'success': True,
            'thicknesses': thicknesses,
            'stats': stats,
            'all_peaks': all_peaks
        })
        
    except Exception as e:
        print(f"Error in calculate_thickness: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error calculating thickness: {str(e)}'})

@app.route('/reset_analysis', methods=['POST'])
def reset_analysis():
    global smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    try:
        # Reset analysis data
        smoothed_image = None
        vertical_profiles = None
        detected_peaks = []
        manual_peaks = []
        
        return jsonify({'success': True, 'message': 'Analysis reset successfully'})
        
    except Exception as e:
        print(f"Error in reset_analysis: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error resetting analysis: {str(e)}'})

@app.route('/clear_manual_peaks', methods=['POST'])
def clear_manual_peaks():
    global manual_peaks
    
    try:
        manual_peaks = []
        
        # Generate updated analysis plot
        plot_base64 = generate_analysis_plot()
        
        if plot_base64 is None:
            return jsonify({'error': 'Error generating plot'})
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'message': 'All manual interfaces cleared'
        })
        
    except Exception as e:
        print(f"Error in clear_manual_peaks: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error clearing manual peaks: {str(e)}'})

@app.route('/get_all_interfaces', methods=['GET'])
def get_all_interfaces():
    """Get all current interfaces (auto + manual)"""
    try:
        global detected_peaks, manual_peaks
        
        all_peaks = list(detected_peaks) + list(manual_peaks)
        all_peaks = sorted(set(all_peaks))
        
        return jsonify({
            'success': True,
            'all_interfaces': all_peaks,
            'auto_interfaces': detected_peaks,
            'manual_interfaces': manual_peaks
        })
        
    except Exception as e:
        print(f"Error in get_all_interfaces: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Error getting interfaces: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)