import pytest
import numpy as np
from scipy.ndimage import gaussian_filter

class TestImageProcessingFunctions:
    """Test image processing functionality."""
    
    def test_gaussian_smoothing(self, sample_image):
        """Test Gaussian smoothing produces expected results."""
        sigma = 2.0
        smoothed = gaussian_filter(sample_image, sigma=sigma)
        
        # Smoothed image should have same shape
        assert smoothed.shape == sample_image.shape
        
        # Smoothed image should be less noisy (lower std dev)
        assert np.std(smoothed) <= np.std(sample_image)
        
        # Values should still be in reasonable range
        assert smoothed.min() >= 0
        assert smoothed.max() <= 1
    
    def test_vertical_profile_calculation(self, sample_image):
        """Test vertical profile calculations."""
        # Calculate profiles
        vertical_mean = np.mean(sample_image, axis=1)
        vertical_std = np.std(sample_image, axis=1)
        vertical_grad = np.gradient(vertical_mean)
        
        # Check shapes
        assert len(vertical_mean) == sample_image.shape[0]
        assert len(vertical_std) == sample_image.shape[0]
        assert len(vertical_grad) == sample_image.shape[0]
        
        # Check value ranges
        assert np.all(vertical_mean >= 0)
        assert np.all(vertical_mean <= 1)
        assert np.all(vertical_std >= 0)
    
    def test_peak_detection_logic(self, sample_image):
        """Test peak detection logic."""
        from scipy.signal import find_peaks
        
        # Create vertical profile
        smoothed = gaussian_filter(sample_image, sigma=2.0)
        vertical_mean = np.mean(smoothed, axis=1)
        vertical_grad = np.gradient(vertical_mean)
        
        # Find peaks
        abs_grad = np.abs(vertical_grad)
        height_threshold = np.percentile(abs_grad, 90)
        peaks, _ = find_peaks(abs_grad, distance=20, height=height_threshold)
        
        # Should find some peaks in our synthetic layered structure
        assert len(peaks) > 0
        assert len(peaks) <= len(abs_grad)
        
        # All peak positions should be valid indices
        assert np.all(peaks >= 0)
        assert np.all(peaks < len(abs_grad))

class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_empty_image_handling(self, client):
        """Test handling of empty or invalid images."""
        # This would test what happens with malformed data
        pass
    
    def test_memory_constraints(self, client):
        """Test behavior with very large images."""
        # This would test memory management
        pass
    
    def test_invalid_parameters(self, preprocessed_session):
        """Test handling of invalid parameters."""
        # Test negative sigma
        response = preprocessed_session.post('/preprocess', json={'sigma': -1.0})
        # Should handle gracefully (might not error, but should produce valid results)
        
        # Test zero distance in peak detection
        response = preprocessed_session.post('/detect_peaks', 
                                           json={'distance': 0, 'percentile': 95, 'top_n': 10})
        # Should handle gracefully

class TestIntegrationWorkflow:
    """Test complete workflows from start to finish."""
    
    def test_complete_workflow(self, client, mock_emd_file):
        """Test complete workflow from upload to thickness calculation."""
        # 1. Upload file
        with open(mock_emd_file, 'rb') as f:
            response = client.post('/upload', 
                                 data={'file': (f, 'test_data.emd')},
                                 content_type='multipart/form-data')
        assert response.status_code == 200
        assert response.get_json()['success'] is True
        
        # 2. Select image
        response = client.post('/select_image', json={'index': 0})
        assert response.status_code == 200
        assert response.get_json()['success'] is True
        
        # 3. Preprocess
        response = client.post('/preprocess', json={'sigma': 4.0})
        assert response.status_code == 200
        assert response.get_json()['success'] is True
        
        # 4. Detect peaks
        response = client.post('/detect_peaks', 
                              json={'distance': 30, 'percentile': 95, 'top_n': 10})
        assert response.status_code == 200
        assert response.get_json()['success'] is True
        
        # 5. Add manual peak
        response = client.post('/add_manual_peak_region', 
                              json={'x_start': 10, 'x_end': 60, 'y_start': 20, 'y_end': 80})
        assert response.status_code == 200
        assert response.get_json()['success'] is True
        
        # 6. Calculate thickness
        response = client.post('/calculate_thickness')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'thicknesses' in data
        assert 'stats' in data
        
        # 7. Download CSV
        response = client.post('/download_csv')
        assert response.status_code == 200
        
        # 8. Cleanup
        response = client.post('/cleanup')
        assert response.status_code == 200
        assert response.get_json()['success'] is True