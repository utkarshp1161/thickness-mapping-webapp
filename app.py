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
import csv
import tempfile

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
detected_peaks = None
manual_peaks = []

def normalize(img):
    """Normalize image to 0-1 range"""
    return (img - img.min()) / (img.max() - img.min())

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_str = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close(fig)
    return img_str

def image_to_base64(image_array):
    """Convert numpy array to base64 string for display"""
    if image_array.dtype != np.uint8:
        img_normalized = cv2.normalize(image_array.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img_normalized = image_array
    
    _, buffer = cv2.imencode('.png', img_normalized)
    img_str = base64.b64encode(buffer).decode()
    return img_str

def generate_analysis_plot():
    """Generate the main analysis plot with three panels"""
    global original_image, smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    if original_image is None or smoothed_image is None:
        return None
    
    # Combine auto-detected and manual peaks
    all_peaks = list(detected_peaks) if detected_peaks is not None else []
    all_peaks.extend(manual_peaks)
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
    for i, y in enumerate(all_peaks):
        if detected_peaks is not None and y in detected_peaks:
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
            if detected_peaks is not None and y in detected_peaks:
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

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.route('/')
def index():
    return render_template('thickness.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global loaded_images, pixel_size
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.endswith('.emd'):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
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
                    images_info.append({
                        'index': i,
                        'shape': image_data.shape,
                        'dtype': str(image_data.dtype),
                        'pixel_size': pixel_size,
                        'preview': image_to_base64(image_data)
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
            
        except Exception as e:
            print(f"Error loading EMD file: {e}")
            return jsonify({'error': f'Error loading EMD file: {str(e)}'})
    
    return jsonify({'error': 'Please upload a .emd file'})

@app.route('/select_image', methods=['POST'])
def select_image():
    global current_image, original_image, pixel_size, smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    if loaded_images is None:
        return jsonify({'error': 'No images loaded'})
    
    data = request.json
    image_index = int(data.get('index', 0))
    
    if image_index < 0 or image_index >= len(loaded_images):
        return jsonify({'error': 'Invalid image index'})
    
    try:
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
        detected_peaks = None
        manual_peaks = []
        
        print(f"Selected image {image_index}. Shape: {current_image.shape}")
        
        return jsonify({
            'success': True,
            'image_shape': current_image.shape,
            'pixel_size': pixel_size,
            'image_preview': image_to_base64(current_image)
        })
        
    except Exception as e:
        print(f"Error selecting image: {e}")
        return jsonify({'error': f'Error selecting image: {str(e)}'})

@app.route('/preprocess', methods=['POST'])
def preprocess_image():
    global current_image, smoothed_image, vertical_profiles
    
    if current_image is None:
        return jsonify({'error': 'No image selected'})
    
    data = request.json
    sigma = float(data.get('sigma', 4.0))
    
    try:
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
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'smoothed_image': image_to_base64(smoothed_image),
            'stats': {
                'mean_intensity': float(np.mean(smoothed_image)),
                'std_intensity': float(np.std(smoothed_image)),
                'gradient_range': [float(np.min(vertical_grad)), float(np.max(vertical_grad))]
            }
        })
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return jsonify({'error': f'Error preprocessing image: {str(e)}'})

@app.route('/detect_peaks', methods=['POST'])
def detect_peaks():
    global vertical_profiles, detected_peaks
    
    if vertical_profiles is None:
        return jsonify({'error': 'No preprocessed image available'})
    
    data = request.json
    distance = int(data.get('distance', 30))
    percentile = float(data.get('percentile', 95))  # Changed from height to percentile
    top_n = int(data.get('top_n', 10))
    
    try:
        print(f"Detecting peaks with distance={distance}, percentile={percentile}, top_n={top_n}")
        
        # Find peaks in gradient using percentile-based height threshold
        grad = vertical_profiles['gradient']
        abs_grad = np.abs(grad)
        height_threshold = np.percentile(abs_grad, percentile)  # Use percentile for height
        
        peaks, _ = find_peaks(abs_grad, distance=distance, height=height_threshold)
        
        # Select strongest N peaks
        if len(peaks) > top_n:
            idx = np.argsort(abs_grad[peaks])[::-1][:top_n]
            peaks = peaks[idx]
        
        detected_peaks = sorted(peaks.tolist())
        
        # Generate updated analysis plot
        plot_base64 = generate_analysis_plot()
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'peaks_found': len(detected_peaks),
            'peaks': detected_peaks
        })
        
    except Exception as e:
        print(f"Error detecting peaks: {e}")
        return jsonify({'error': f'Error detecting peaks: {str(e)}'})
    
    
@app.route('/add_manual_peak_region', methods=['POST'])
def add_manual_peak_region():
    global manual_peaks, vertical_profiles
    
    if vertical_profiles is None:
        return jsonify({'error': 'No preprocessed image available'})
    
    data = request.json
    x_start = int(data.get('x_start', 0))
    x_end = int(data.get('x_end', 100))
    y_start = int(data.get('y_start', 0))
    y_end = int(data.get('y_end', 100))
    
    try:
        print(f"Adding manual peak in region x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        # Validate range with proper bounds checking
        max_y = len(vertical_profiles['gradient']) - 1
        if y_start >= y_end:
            return jsonify({'error': f'Y Start ({y_start}) must be less than Y End ({y_end})'})
        if y_start < 0:
            return jsonify({'error': f'Y Start ({y_start}) cannot be negative'})
        if y_end > max_y:
            return jsonify({'error': f'Y End ({y_end}) exceeds image height ({max_y})'})
        if y_start > max_y:
            return jsonify({'error': f'Y Start ({y_start}) exceeds image height ({max_y})'})

        # Ensure we have a valid range
        y_start = max(0, min(y_start, max_y))
        y_end = max(y_start + 1, min(y_end, max_y))
        
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
            
            return jsonify({
                'success': True,
                'plot': plot_base64,
                'peak_added': int(manual_peak),
                'manual_peaks': [int(p) for p in manual_peaks],
                'message': f'Interface added at Y={manual_peak} (selected region: x[{x_start}:{x_end}], y[{y_start}:{y_end}])'
            })
        else:
            return jsonify({'error': f'Interface at Y={manual_peak} already exists'})
                
    except Exception as e:
        print(f"Error adding manual peak: {e}")
        return jsonify({'error': f'Error adding manual peak: {str(e)}'})

@app.route('/remove_interface', methods=['POST'])
def remove_interface():
    global detected_peaks, manual_peaks
    
    data = request.json
    peak_to_remove = int(data.get('peak', 0))
    
    try:
        removed_from = None
        
        # Try to remove from manual peaks first
        if peak_to_remove in manual_peaks:
            manual_peaks.remove(peak_to_remove)
            removed_from = "manual"
        # Then try to remove from detected peaks
        elif detected_peaks is not None and peak_to_remove in detected_peaks:
            detected_peaks.remove(peak_to_remove)
            removed_from = "auto"
        else:
            return jsonify({'error': 'Interface not found'})
        
        # Generate updated analysis plot
        plot_base64 = generate_analysis_plot()
        
        return jsonify({
            'success': True,
            'plot': plot_base64,
            'peak_removed': peak_to_remove,
            'removed_from': removed_from,
            'manual_peaks': manual_peaks,
            'auto_peaks': detected_peaks if detected_peaks is not None else [],
            'message': f'Interface at Y={peak_to_remove} removed from {removed_from} interfaces'
        })
        
    except Exception as e:
        print(f"Error removing interface: {e}")
        return jsonify({'error': f'Error removing interface: {str(e)}'})

@app.route('/calculate_thickness', methods=['POST'])
def calculate_thickness():
    global detected_peaks, manual_peaks, pixel_size
    
    if detected_peaks is None and not manual_peaks:
        return jsonify({'error': 'No peaks detected or manually added'})
    
    try:
        # Combine all peaks
        all_peaks = list(detected_peaks) if detected_peaks is not None else []
        all_peaks.extend(manual_peaks)
        all_peaks = sorted(set(all_peaks))
        
        if len(all_peaks) < 2:
            return jsonify({'error': 'At least 2 interfaces needed for thickness calculation'})
        
        # Calculate thicknesses between consecutive interfaces
        thicknesses = []
        for i in range(len(all_peaks) - 1):
            thickness_pixels = all_peaks[i+1] - all_peaks[i]
            thickness_nm = thickness_pixels * pixel_size
            thicknesses.append({
                'layer': int(i + 1),
                'start_interface': int(all_peaks[i]),
                'end_interface': int(all_peaks[i+1]),
                'thickness_pixels': int(thickness_pixels),
                'thickness_nm': float(thickness_nm)
            })
                    
        # Calculate statistics
        thickness_values = [t['thickness_nm'] for t in thicknesses]
        stats = {
            'mean_thickness': float(np.mean(thickness_values)),
            'std_thickness': float(np.std(thickness_values)),
            'min_thickness': float(np.min(thickness_values)),
            'max_thickness': float(np.max(thickness_values)),
            'total_layers': int(len(thicknesses)),
            'total_interfaces': int(len(all_peaks)),
            'auto_interfaces': int(len(detected_peaks)) if detected_peaks is not None else 0,
            'manual_interfaces': int(len(manual_peaks)),
            'pixel_size': float(pixel_size)
        }
        all_peaks = [int(p) for p in all_peaks]
        return jsonify({
            'success': True,
            'thicknesses': thicknesses,
            'stats': stats,
            'all_peaks': all_peaks
        })
        
    except Exception as e:
        print(f"Error calculating thickness: {e}")
        return jsonify({'error': f'Error calculating thickness: {str(e)}'})

@app.route('/reset_analysis', methods=['POST'])
def reset_analysis():
    global smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    # Reset analysis data
    smoothed_image = None
    vertical_profiles = None
    detected_peaks = None
    manual_peaks = []
    
    return jsonify({'success': True, 'message': 'Analysis reset successfully'})

@app.route('/clear_manual_peaks', methods=['POST'])
def clear_manual_peaks():
    global manual_peaks
    
    manual_peaks = []
    
    # Generate updated analysis plot
    plot_base64 = generate_analysis_plot()
    
    return jsonify({
        'success': True,
        'plot': plot_base64,
        'message': 'All manual interfaces cleared'
    })

@app.route('/get_all_interfaces', methods=['GET'])
def get_all_interfaces():
    global detected_peaks, manual_peaks
    
    # Combine all peaks with their types
    all_interfaces = []
    
    if detected_peaks is not None:
        for peak in detected_peaks:
            all_interfaces.append({
                'y_position': int(peak),
                'type': 'auto',
                'color': 'cyan'
            })
    
    for peak in manual_peaks:
        all_interfaces.append({
            'y_position': int(peak),
            'type': 'manual',
            'color': 'red'
        })
    
    # Sort by Y position
    all_interfaces.sort(key=lambda x: x['y_position'])
    
    return jsonify({
        'success': True,
        'interfaces': all_interfaces,
        'total_count': len(all_interfaces)
    })

@app.route('/download_csv', methods=['POST'])
def download_csv():
    global detected_peaks, manual_peaks, pixel_size, current_image
    
    if detected_peaks is None and not manual_peaks:
        return jsonify({'error': 'No peaks detected or manually added'})
    
    try:
        # Combine all peaks
        all_peaks = list(detected_peaks) if detected_peaks is not None else []
        all_peaks.extend(manual_peaks)
        all_peaks = sorted(set(all_peaks))
        
        if len(all_peaks) < 2:
            return jsonify({'error': 'At least 2 interfaces needed for thickness calculation'})
        
        # Calculate thicknesses
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
        
        # Create CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        writer = csv.writer(temp_file)
        
        # Write headers
        writer.writerow(['Layer', 'Start_Interface', 'End_Interface', 'Thickness_Pixels', 'Thickness_nm'])
        
        # Write data
        for thickness in thicknesses:
            writer.writerow([
                thickness['layer'],
                thickness['start_interface'],
                thickness['end_interface'],
                thickness['thickness_pixels'],
                thickness['thickness_nm']
            ])
        
        temp_file.close()
        
        return send_file(temp_file.name, as_attachment=True, download_name='thickness_results.csv')
        
    except Exception as e:
        print(f"Error generating CSV: {e}")
        return jsonify({'error': f'Error generating CSV: {str(e)}'})

@app.route('/cleanup', methods=['POST'])
def cleanup():
    global loaded_images, current_image, original_image, smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    try:
        # Clean up uploaded files
        upload_folder = app.config['UPLOAD_FOLDER']
        for filename in os.listdir(upload_folder):
            file_path = os.path.join(upload_folder, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        
        # Reset all global variables
        loaded_images = None
        current_image = None
        original_image = None
        smoothed_image = None
        vertical_profiles = None
        detected_peaks = None
        manual_peaks = []
        
        return jsonify({'success': True, 'message': 'Cleanup completed successfully'})
        
    except Exception as e:
        print(f"Error during cleanup: {e}")
        return jsonify({'error': f'Error during cleanup: {str(e)}'})
    
@app.route('/download_analysis_image', methods=['POST'])
def download_analysis_image():
    global original_image, smoothed_image, vertical_profiles, detected_peaks, manual_peaks
    
    if original_image is None or smoothed_image is None:
        return jsonify({'error': 'No analysis image available'})
    
    try:
        # Generate the analysis plot (same as generate_analysis_plot but save to file)
        all_peaks = list(detected_peaks) if detected_peaks is not None else []
        all_peaks.extend(manual_peaks)
        all_peaks = sorted(set(all_peaks))
        
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
        for i, y in enumerate(all_peaks):
            if detected_peaks is not None and y in detected_peaks:
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
                if detected_peaks is not None and y in detected_peaks:
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
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        return send_file(temp_file.name, as_attachment=True, download_name='analysis_with_interfaces.png')
        
    except Exception as e:
        print(f"Error generating analysis image: {e}")
        return jsonify({'error': f'Error generating analysis image: {str(e)}'})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)