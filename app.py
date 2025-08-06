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

def create_analysis_figure(show_thickness_text=True):
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
    for y in all_peaks:
        is_auto = detected_peaks is not None and y in detected_peaks
        ax_img.axhline(y, color='cyan' if is_auto else 'red',
                          linestyle=':' if is_auto else '--',
                          linewidth=1 if is_auto else 2)

    # Annotate thickness
    if show_thickness_text and len(all_peaks) >= 2:
        for i in range(len(all_peaks) - 1):
            y_start, y_end = all_peaks[i], all_peaks[i+1]
            text_y = (y_start + y_end) / 2
            thickness_pixels = y_end - y_start
            thickness_nm = thickness_pixels * pixel_size if pixel_size else thickness_pixels
            thickness_text = f'{thickness_nm:.1f} nm' if pixel_size else f'{thickness_pixels} px'
            text_x = smoothed_image.shape[1] * (0.05 if i % 2 == 0 else 0.95)
            ha = 'left' if i % 2 == 0 else 'right'
            ax_img.text(text_x, text_y, thickness_text,
                        color='yellow', fontsize=10, fontweight='bold',
                        ha=ha, va='center',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    # Right: Vertical Profiles
    if vertical_profiles is not None:
        y_pixels = np.arange(len(vertical_profiles['mean']))
        ax_prof.plot(vertical_profiles['mean'], y_pixels, label="Mean", color='black')
        ax_prof.plot(vertical_profiles['std'], y_pixels, label="Std Dev", alpha=0.5, color='blue')
        ax_prof.plot(vertical_profiles['gradient'], y_pixels, label="Gradient", linestyle='--', alpha=0.5, color='green')

        auto_labeled = manual_labeled = False
        for y in all_peaks:
            is_auto = detected_peaks is not None and y in detected_peaks
            label = "Auto Interfaces" if is_auto and not auto_labeled else (
                    "Manual Interfaces" if not is_auto and not manual_labeled else None)
            if is_auto: auto_labeled = True
            else: manual_labeled = True
            ax_prof.axhline(y, linestyle='--' if is_auto else '-', color='cyan' if is_auto else 'red', alpha=0.8, label=label)

        ax_prof.set_title("Vertical Profiles & Detected Interfaces")
        ax_prof.set_xlabel("Intensity")
        ax_prof.set_ylabel("Y (pixels)")
        ax_prof.invert_yaxis()
        ax_prof.legend()
        ax_prof.grid(True, linestyle=':', alpha=0.4)

    # Y-ticks for all
    yticks = np.arange(0, original_image.shape[0], 50)
    for ax in (ax_img_raw, ax_img, ax_prof):
        ax.set_yticks(yticks)

    plt.tight_layout()
    return fig

def generate_analysis_plot():
    global original_image, smoothed_image
    if original_image is None or smoothed_image is None:
        return None

    fig = create_analysis_figure(show_thickness_text=True)
    return plot_to_base64(fig)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.route('/')
def index():
    return render_template('thickness.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global loaded_images, pixel_size, filename
    
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

@app.route('/add_interface_by_method', methods=['POST'])
def add_interface_by_method():
    global manual_peaks, vertical_profiles
    
    if vertical_profiles is None:
        return jsonify({'error': 'No preprocessed image available'})
    
    data = request.json
    method = data.get('method', 'minima')
    x_start = int(data.get('x_start', 0))
    x_end = int(data.get('x_end', 100))
    y_start = int(data.get('y_start', 0))
    y_end = int(data.get('y_end', 100))
    
    try:
        print(f"Adding interface using {method} in region x[{x_start}:{x_end}], y[{y_start}:{y_end}]")
        
        # Validate range
        max_y = len(vertical_profiles['mean']) - 1
        if y_start >= y_end or y_start < 0 or y_end > max_y:
            return jsonify({'error': 'Invalid Y range specified'})
        
        y_start = max(0, min(y_start, max_y))
        y_end = max(y_start + 1, min(y_end, max_y))
        
        # Get the mean profile in the specified region
        mean_region = vertical_profiles['mean'][y_start:y_end]
        if len(mean_region) == 0:
            return jsonify({'error': 'Empty Y range specified'})
        
        # Find local minima or maxima
        if method == 'minima':
            # Find local minimum
            offset = np.argmin(mean_region)
        else:  # maxima
            # Find local maximum
            offset = np.argmax(mean_region)
        
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
                'message': f'Interface added at Y={manual_peak} using local {method} (region: x[{x_start}:{x_end}], y[{y_start}:{y_end}])'
            })
        else:
            return jsonify({'error': f'Interface at Y={manual_peak} already exists'})
                
    except Exception as e:
        print(f"Error adding interface by {method}: {e}")
        return jsonify({'error': f'Error adding interface by {method}: {str(e)}'})
    
@app.route('/remove_interface', methods=['POST'])
def remove_interface():
    global detected_peaks, manual_peaks
    
    data = request.json
    peak_to_remove = int(data.get('peak', 0))
    
    try:
        print(f"Attempting to remove interface at Y={peak_to_remove}")
        removed_from = None
        
        # Try to remove from manual peaks first
        if peak_to_remove in manual_peaks:
            manual_peaks.remove(peak_to_remove)
            removed_from = "manual"
            print(f"Removed from manual peaks. Remaining: {manual_peaks}")
        # Then try to remove from detected peaks
        elif detected_peaks is not None and peak_to_remove in detected_peaks:
            detected_peaks.remove(peak_to_remove)
            removed_from = "auto"
            print(f"Removed from auto peaks. Remaining: {detected_peaks}")
        else:
            print(f"Interface at Y={peak_to_remove} not found in any list")
            return jsonify({'success': False, 'error': 'Interface not found'})
        
        # Generate updated analysis plot
        plot_base64 = generate_analysis_plot()
        
        response_data = {
            'success': True,
            'plot': plot_base64,
            'peak_removed': peak_to_remove,
            'removed_from': removed_from,
            'manual_peaks': [int(p) for p in manual_peaks],
            'auto_peaks': [int(p) for p in detected_peaks] if detected_peaks is not None else [],
            'message': f'Interface at Y={peak_to_remove} removed from {removed_from} interfaces'
        }
        
        print(f"Removal successful. Response: {response_data['message']}")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error removing interface: {e}")
        return jsonify({'success': False, 'error': f'Error removing interface: {str(e)}'})

@app.route('/calculate_thickness', methods=['POST'])
def calculate_thickness():
    global detected_peaks, manual_peaks, pixel_size, smoothed_image
    
    if detected_peaks is None and not manual_peaks:
        return jsonify({'error': 'No peaks detected or manually added'})
    
    try:
        # Combine all peaks
        all_peaks = list(detected_peaks) if detected_peaks is not None else []
        all_peaks.extend(manual_peaks)
        all_peaks = sorted(set(all_peaks))
        
        if len(all_peaks) < 2:
            return jsonify({'error': 'At least 2 interfaces needed for thickness calculation'})
        
        # Choose image source for roughness calculation
        image_source = smoothed_image if smoothed_image is not None else current_image
        
        # Calculate roughness for each interface
        interface_roughness = {}
        if image_source is not None:
            for y in all_peaks:
                roughness_data = calculate_interface_roughness(y, image_source, pixel_size, all_interfaces=all_peaks)
                if roughness_data['valid']:
                    interface_roughness[int(y)] = roughness_data
        
        # Calculate thicknesses between consecutive interfaces
        thicknesses = []
        for i in range(len(all_peaks) - 1):
            y_start = all_peaks[i]
            y_end = all_peaks[i + 1]
            thickness_pixels = y_end - y_start
            thickness_nm = thickness_pixels * pixel_size
            
            # Get roughness data
            start_roughness = interface_roughness.get(int(y_start), {})
            end_roughness = interface_roughness.get(int(y_end), {})
            
            thicknesses.append({
                'layer': int(i + 1),
                'start_interface': int(y_start),
                'end_interface': int(y_end),
                'thickness_pixels': int(thickness_pixels),
                'thickness_nm': float(thickness_nm),
                'start_roughness_nm': float(start_roughness.get('roughness_nm', 0.0)),
                'end_roughness_nm': float(end_roughness.get('roughness_nm', 0.0)),
                'start_roughness_valid': start_roughness.get('valid', False),
                'end_roughness_valid': end_roughness.get('valid', False)
            })
        
        # Calculate statistics
        thickness_values = [t['thickness_nm'] for t in thicknesses]
        valid_roughness_values = []
        for y, data in interface_roughness.items():
            if data['valid']:
                valid_roughness_values.append(data['roughness_nm'])
        
        stats = {
            'mean_thickness': float(np.mean(thickness_values)),
            'std_thickness': float(np.std(thickness_values)),
            'min_thickness': float(np.min(thickness_values)),
            'max_thickness': float(np.max(thickness_values)),
            'total_layers': int(len(thicknesses)),
            'total_interfaces': int(len(all_peaks)),
            'auto_interfaces': int(len(detected_peaks)) if detected_peaks is not None else 0,
            'manual_interfaces': int(len(manual_peaks)),
            'pixel_size': float(pixel_size),
            'mean_roughness': float(np.mean(valid_roughness_values)) if valid_roughness_values else 0.0,
            'std_roughness': float(np.std(valid_roughness_values)) if valid_roughness_values else 0.0,
            'min_roughness': float(np.min(valid_roughness_values)) if valid_roughness_values else 0.0,
            'max_roughness': float(np.max(valid_roughness_values)) if valid_roughness_values else 0.0,
            'valid_roughness_count': len(valid_roughness_values)
        }
        
        all_peaks = [int(p) for p in all_peaks]
        return jsonify({
            'success': True,
            'thicknesses': thicknesses,
            'stats': stats,
            'all_peaks': all_peaks,
            'interface_roughness': {int(k): v for k, v in interface_roughness.items()}
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
    global detected_peaks, manual_peaks, pixel_size, current_image, smoothed_image

    if detected_peaks is None and not manual_peaks:
        return jsonify({'error': 'No peaks detected or manually added'})

    try:
        # Combine all peaks
        all_peaks = list(detected_peaks) if detected_peaks is not None else []
        all_peaks.extend(manual_peaks)
        all_peaks = sorted(set(all_peaks))

        if len(all_peaks) < 2:
            return jsonify({'error': 'At least 2 interfaces needed for thickness calculation'})

        # Choose image source
        image_source = smoothed_image if smoothed_image is not None else current_image
        if image_source is None:
            return jsonify({'error': 'No image available for roughness calculation'})

        # Calculate roughness for each interface
        interface_roughness = {}
        for y in all_peaks:
            roughness_data = calculate_interface_roughness(y, image_source, pixel_size)
            if roughness_data['valid']:
                interface_roughness[int(y)] = roughness_data

        # Prepare thickness and roughness data
        thicknesses = []
        for i in range(len(all_peaks) - 1):
            y_start = all_peaks[i]
            y_end = all_peaks[i + 1]
            thickness_pixels = y_end - y_start
            thickness_nm = thickness_pixels * pixel_size
            
            start_roughness = interface_roughness.get(int(y_start), {})
            end_roughness = interface_roughness.get(int(y_end), {})

            thicknesses.append({
                'layer': i + 1,
                'start_interface': y_start,
                'end_interface': y_end,
                'thickness_pixels': thickness_pixels,
                'thickness_nm': thickness_nm,
                'start_roughness_nm': start_roughness.get('roughness_nm', 0.0),
                'end_roughness_nm': end_roughness.get('roughness_nm', 0.0)
            })

        # Create CSV file
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv', newline='')
        writer = csv.writer(temp_file)

        # Write headers
        writer.writerow([
            'Layer',
            'Start_Interface',
            'End_Interface',
            'Thickness_Pixels',
            'Thickness_nm',
            'Start_Interface_Roughness_nm',
            'End_Interface_Roughness_nm'
        ])

        # Write data
        for t in thicknesses:
            writer.writerow([
                t['layer'],
                t['start_interface'],
                t['end_interface'],
                t['thickness_pixels'],
                f"{t['thickness_nm']:.4f}",
                f"{t['start_roughness_nm']:.4f}",
                f"{t['end_roughness_nm']:.4f}"
            ])

        temp_file.close()
        return send_file(temp_file.name, as_attachment=True, download_name='thickness_roughness_results.csv')

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
    global original_image, smoothed_image, detected_peaks, manual_peaks, pixel_size

    if original_image is None or smoothed_image is None:
        return jsonify({'error': 'No analysis image available'})

    try:
        all_peaks = list(detected_peaks) if detected_peaks is not None else []
        all_peaks.extend(manual_peaks)
        all_peaks = sorted(set(all_peaks))

        height, width = smoothed_image.shape
        aspect_ratio = width / height
        fig_width = 8
        fig_height = fig_width / aspect_ratio
        # Only one panel: annotated smoothed image
        fig, ax_img = plt.subplots(figsize=(fig_width, fig_height))

        ax_img.imshow(smoothed_image, cmap='gray', aspect='equal', origin="upper")
        ax_img.set_title("Smoothed Image with Interfaces")
        # ax_img.set_xlabel("X (pixels)")
        # ax_img.set_ylabel("Y (pixels)")
        
        # Interface lines
        # for y in all_peaks:
        #     is_auto = detected_peaks is not None and y in detected_peaks
        #     ax_img.axhline(y, color='cyan' if is_auto else 'red',
        #                       linestyle=':' if is_auto else '--',
        #                       linewidth=1 if is_auto else 2)
        # Annotate thickness
        if len(all_peaks) >= 2:
            for i in range(len(all_peaks) - 1):
                y_start, y_end = all_peaks[i], all_peaks[i+1]
                text_y = (y_start + y_end) / 2
                thickness_pixels = y_end - y_start
                thickness_nm = thickness_pixels * pixel_size if pixel_size else thickness_pixels
                thickness_text = f'{thickness_nm:.1f} nm' if pixel_size else f'{thickness_pixels} px'

                text_x = width * (0.05 if i % 2 == 0 else 0.95)
                ha = 'left' if i % 2 == 0 else 'right'

                ax_img.text(text_x, text_y, thickness_text,
                            color='yellow', fontsize=10, fontweight='bold',
                            ha=ha, va='center',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

        # --- Add Scale Bar (50 nm) ---
        if pixel_size:  # Avoid error if pixel_size is None
            scale_length_nm = 50
            scale_length_px = scale_length_nm / pixel_size

            # Define position at bottom-left
            margin_y = 0.05 * height
            margin_x = 0.05 * width
            bar_y = height - margin_y
            bar_x_start = margin_x
            bar_x_end = bar_x_start + scale_length_px

            # Draw scale bar
            ax_img.hlines(bar_y, bar_x_start, bar_x_end, color='white', linewidth=3)
            ax_img.text((bar_x_start + bar_x_end) / 2, bar_y - 10, f'{scale_length_nm:.0f} nm',
                        color='white', fontsize=10, fontweight='bold',
                        ha='center', va='top',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # ax_img.set_yticks(np.arange(0, height, 50))
        ax_img.axis('off')
        fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, format='png', dpi=300)
        plt.close(fig)
        

        # return send_file(temp_file.name, as_attachment=True, download_name='smoothed_with_annotations.png')

        basename = os.path.splitext(os.path.basename(filename))[0] if filename else "thickness"
        download_name = f"thickness_{basename}.png"
        return send_file(temp_file.name, as_attachment=True, download_name=download_name)


    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'error': str(e)})


def calculate_interface_roughness(y_position, image_source, pixel_size, method='rms', window_size=15, all_interfaces=None):
    """
    Calculate geometric interface roughness by tracking interface position variations
    
    Args:
        y_position: Nominal Y coordinate of the interface
        image_source: 2D numpy array (smoothed or original image)
        pixel_size: Pixel size in nm
        method: Roughness calculation method ('rms', 'ra', 'rmax')
        window_size: Half-width of search window around nominal interface
        all_interfaces: List of all interface positions to avoid overlap
    
    Returns:
        dict: Roughness values in pixels and nm
    """
    try:
        height, width = image_source.shape
        y_pos = int(y_position)
        
        # Adaptive window size based on adjacent interfaces
        adaptive_window = window_size
        if all_interfaces is not None:
            interfaces = sorted([int(i) for i in all_interfaces])
            current_idx = None
            
            # Find current interface in the list
            for i, interface_y in enumerate(interfaces):
                if interface_y == y_pos:
                    current_idx = i
                    break
            
            if current_idx is not None:
                # Check distance to previous interface
                if current_idx > 0:
                    prev_distance = y_pos - interfaces[current_idx - 1]
                    adaptive_window = min(adaptive_window, max(3, prev_distance // 2))
                
                # Check distance to next interface
                if current_idx < len(interfaces) - 1:
                    next_distance = interfaces[current_idx + 1] - y_pos
                    adaptive_window = min(adaptive_window, max(3, next_distance // 2))
        
        # Define search window around nominal interface
        y_min = max(0, y_pos - adaptive_window)
        y_max = min(height - 1, y_pos + adaptive_window)
        
        if y_max - y_min < 3:  # Need minimum window
            return {'roughness_pixels': 0.0, 'roughness_nm': 0.0, 'valid': False}
        
        # Track actual interface position for each x coordinate
        interface_positions = []
        
        for x in range(width):
            # Extract vertical profile at this x position
            vertical_profile = image_source[y_min:y_max+1, x]
            
            # Find interface position using gradient method
            # (You can also use edge detection, threshold, or other methods)
            gradient = np.gradient(vertical_profile)
            
            # Find position of maximum absolute gradient (strongest edge)
            edge_idx = np.argmax(np.abs(gradient))
            actual_y_position = y_min + edge_idx
            
            # Store relative position from nominal interface
            interface_positions.append(actual_y_position - y_pos)
        
        interface_positions = np.array(interface_positions)
        
        # Calculate geometric roughness
        if method == 'rms':
            # Root Mean Square roughness
            roughness_pixels = np.sqrt(np.mean(interface_positions**2))
        elif method == 'ra':
            # Average roughness
            roughness_pixels = np.mean(np.abs(interface_positions))
        elif method == 'rmax':
            # Maximum peak-to-valley roughness
            roughness_pixels = np.max(interface_positions) - np.min(interface_positions)
        else:
            roughness_pixels = np.sqrt(np.mean(interface_positions**2))  # Default to RMS
        
        # Convert to nanometers
        roughness_nm = roughness_pixels * pixel_size
        
        return {
            'roughness_pixels': float(roughness_pixels),
            'roughness_nm': float(roughness_nm),
            'valid': True,
            'interface_positions': interface_positions.tolist(),
            'mean_deviation': float(np.mean(interface_positions)),
            'std_deviation': float(np.std(interface_positions)),
            'measurement_points': len(interface_positions),
            'used_window_size': int(adaptive_window)  # Added for debugging
        }
        
    except Exception as e:
        print(f"Error calculating roughness for interface at Y={y_position}: {e}")
        return {'roughness_pixels': 0.0, 'roughness_nm': 0.0, 'valid': False}
    
def create_roughness_analysis_figure():
    """Create figure showing smoothed image with roughness annotations and visual roughness traces"""
    global smoothed_image, detected_peaks, manual_peaks, pixel_size
    
    if smoothed_image is None:
        return None
    
    # Combine all peaks
    all_peaks = list(detected_peaks) if detected_peaks is not None else []
    all_peaks.extend(manual_peaks)
    all_peaks = sorted(set(all_peaks))
    
    if len(all_peaks) == 0:
        return None
    
    # Calculate roughness for each interface
    interface_roughness = {}
    for i, y in enumerate(all_peaks):
        if i !=0 or i!=2:
            roughness_data = calculate_interface_roughness(y, smoothed_image, pixel_size)
            if roughness_data['valid']:
                interface_roughness[y] = roughness_data
    
    height, width = smoothed_image.shape
    aspect_ratio = width / height
    fig_width = 14
    fig_height = fig_width / aspect_ratio
    
    # Create subplots: only the main image
    fig = plt.figure(figsize=(fig_width, fig_height))
    ax_main = plt.subplot2grid((1, 1), (0, 0))  # Removed ax_rough subplot
    
    # Display smoothed image
    ax_main.imshow(smoothed_image, cmap='gray', aspect='equal', origin="upper")
    ax_main.set_title("Interface Roughness Analysis", fontsize=14, fontweight='bold')
    
    colors = ['cyan', 'red', 'lime', 'magenta', 'yellow', 'orange', 'pink', 'lightblue']
    
    for i, y in enumerate(all_peaks):
        is_auto = detected_peaks is not None and y in detected_peaks
        base_color = 'cyan' if is_auto else 'red'
        trace_color = colors[i % len(colors)]
        linestyle = ':' if is_auto else '--'
        linewidth = 1.5 if is_auto else 2.5
        
        ax_main.axhline(y, color=base_color, linestyle=linestyle, linewidth=linewidth, alpha=0.6, label=f'Interface {i+1}')
        
        if y in interface_roughness and 'interface_positions' in interface_roughness[y]:
            interface_positions = np.array(interface_roughness[y]['interface_positions'])
            x_coords = np.arange(width)
            actual_y_positions = y + interface_positions
            
            ax_main.plot(x_coords, actual_y_positions, color=trace_color, linewidth=2, alpha=0.8)
            ax_main.fill_between(x_coords, y, actual_y_positions, alpha=0.2, color=trace_color)
            
            # --- Commented out roughness profile plot ---
            # ax_rough.plot(interface_positions, x_coords, color=trace_color, linewidth=2, 
            #              label=f'Y={int(y)} (R={interface_roughness[y]["roughness_nm"]:.2f}nm)')
        
        if y in interface_roughness:
            roughness_nm = interface_roughness[y]['roughness_nm']
            roughness_pixels = interface_roughness[y]['roughness_pixels']
            
            text_x = width * (0.02 if i % 2 == 0 else 0.98)
            ha = 'left' if i % 2 == 0 else 'right'
            roughness_text = f'R={roughness_nm:.3f}nm\n({roughness_pixels:.2f}px)'
            
            ax_main.text(text_x, y, roughness_text,
                        color='white', fontsize=8, fontweight='bold',
                        ha=ha, va='center',
                        bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor=trace_color, alpha=0.8, edgecolor='white'))
    

    
    # Add scale bar
    if pixel_size:
        scale_length_nm = 50
        scale_length_px = scale_length_nm / pixel_size
        
        margin_y = 0.05 * height
        margin_x = 0.05 * width
        bar_y = height - margin_y
        bar_x_start = margin_x
        bar_x_end = bar_x_start + scale_length_px
        
        ax_main.hlines(bar_y, bar_x_start, bar_x_end, color='white', linewidth=3)
        ax_main.text((bar_x_start + bar_x_end) / 2, bar_y - 10, f'{scale_length_nm:.0f} nm',
                    color='white', fontsize=10, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
    
    # Add legend
    legend_elements = []
    if detected_peaks and len(detected_peaks) > 0:
        legend_elements.append(plt.Line2D([0], [0], color='cyan', linestyle=':', linewidth=2, label='Auto Interfaces'))
    if manual_peaks and len(manual_peaks) > 0:
        legend_elements.append(plt.Line2D([0], [0], color='red', linestyle='--', linewidth=2, label='Manual Interfaces'))
    legend_elements.append(plt.Line2D([0], [0], color='lime', linewidth=2, label='Actual Interface Trace'))
    
    if legend_elements:# exclude 1st and 3rd
        ax_main.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=9)
    
    # if interface_roughness:
    #     all_roughness_nm = [data['roughness_nm'] for data in interface_roughness.values()]
    #     mean_roughness = np.mean(all_roughness_nm)
    #     std_roughness = np.std(all_roughness_nm)
    #     min_roughness = np.min(all_roughness_nm)
    #     max_roughness = np.max(all_roughness_nm)
        
    #     stats_text = f'ROUGHNESS STATISTICS\n' \
    #                 f'Mean: {mean_roughness:.3f} ± {std_roughness:.3f} nm\n' \
    #                 f'Range: {min_roughness:.3f} - {max_roughness:.3f} nm\n' \
    #                 f'Interfaces: {len(interface_roughness)}'
        
    #     ax_main.text(0.02, 0.02, stats_text,
    #                 transform=ax_main.transAxes,
    #                 fontsize=9, fontweight='bold',
    #                 verticalalignment='bottom',
    #                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    ax_main.axis('off')
    
    plt.tight_layout()
    return fig

@app.route('/download_roughness_image', methods=['POST'])
def download_roughness_image():
    """Download roughness analysis image"""
    global smoothed_image, detected_peaks, manual_peaks
    
    if smoothed_image is None:
        return jsonify({'error': 'No smoothed image available'})
    
    try:
        fig = create_roughness_analysis_figure()
        if fig is None:
            return jsonify({'error': 'No interfaces available for roughness analysis'})
        
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        fig.savefig(temp_file.name, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        basename = os.path.splitext(os.path.basename(filename))[0] if filename else "roughness"
        download_name = f"roughness_{basename}.png"
        return send_file(temp_file.name, as_attachment=True, download_name=download_name)
        
    except Exception as e:
        print(f"Error generating roughness image: {e}")
        return jsonify({'error': f'Error generating roughness image: {str(e)}'})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)