import pytest
import json
import io
import os

class TestUploadEndpoint:
    """Test file upload functionality."""
    
    def test_upload_no_file(self, client):
        """Test upload without file."""
        response = client.post('/upload')
        assert response.status_code == 200
        data = response.get_json()
        assert data['error'] == 'No file selected'
    
    def test_upload_empty_filename(self, client):
        """Test upload with empty filename."""
        response = client.post('/upload', 
                             data={'file': (io.BytesIO(b''), '')})
        assert response.status_code == 200
        data = response.get_json()
        assert data['error'] == 'No file selected'
    
    def test_upload_wrong_extension(self, client):
        """Test upload with wrong file extension."""
        response = client.post('/upload', 
                             data={'file': (io.BytesIO(b'fake data'), 'test.txt')})
        assert response.status_code == 200
        data = response.get_json()
        assert data['error'] == 'Please upload a .emd file'
    
    def test_upload_success(self, client, mock_emd_file):
        """Test successful file upload."""
        with open(mock_emd_file, 'rb') as f:
            response = client.post('/upload', 
                                 data={'file': (f, 'test_data.emd')},
                                 content_type='multipart/form-data')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'images' in data
        assert len(data['images']) > 0
        assert 'file_size' in data

class TestImageSelection:
    """Test image selection functionality."""
    
    def test_select_image_no_upload(self, client):
        """Test selecting image without uploading first."""
        response = client.post('/select_image', json={'index': 0})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'No images loaded' in data['error']
    
    def test_select_image_invalid_index(self, uploaded_session):
        """Test selecting image with invalid index."""
        response = uploaded_session.post('/select_image', json={'index': 999})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'Invalid image index' in data['error']
    
    def test_select_image_success(self, uploaded_session):
        """Test successful image selection."""
        response = uploaded_session.post('/select_image', json={'index': 0})
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'image_shape' in data
        assert 'pixel_size' in data

class TestPreprocessing:
    """Test image preprocessing functionality."""
    
    def test_preprocess_no_image(self, client):
        """Test preprocessing without selected image."""
        response = client.post('/preprocess', json={'sigma': 4.0})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'No image selected' in data['error']
    
    def test_preprocess_success(self, uploaded_session):
        """Test successful preprocessing."""
        # First select an image
        uploaded_session.post('/select_image', json={'index': 0})
        
        # Then preprocess
        response = uploaded_session.post('/preprocess', json={'sigma': 4.0})
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'plot' in data
        assert 'smoothed_image' in data
        assert 'stats' in data
    
    def test_preprocess_different_sigma(self, uploaded_session):
        """Test preprocessing with different sigma values."""
        uploaded_session.post('/select_image', json={'index': 0})
        
        for sigma in [1.0, 2.0, 5.0, 10.0]:
            response = uploaded_session.post('/preprocess', json={'sigma': sigma})
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True

class TestPeakDetection:
    """Test peak detection functionality."""
    
    def test_detect_peaks_no_preprocessing(self, client):
        """Test peak detection without preprocessing."""
        response = client.post('/detect_peaks', 
                              json={'distance': 30, 'percentile': 95, 'top_n': 10})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'No preprocessed image available' in data['error']
    
    def test_detect_peaks_success(self, preprocessed_session):
        """Test successful peak detection."""
        response = preprocessed_session.post('/detect_peaks', 
                                           json={'distance': 30, 'percentile': 95, 'top_n': 10})
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'peaks_found' in data
        assert 'peaks' in data
        assert 'plot' in data
    
    def test_detect_peaks_different_parameters(self, preprocessed_session):
        """Test peak detection with different parameters."""
        test_params = [
            {'distance': 20, 'percentile': 90, 'top_n': 5},
            {'distance': 50, 'percentile': 99, 'top_n': 20},
            {'distance': 10, 'percentile': 85, 'top_n': 3}
        ]
        
        for params in test_params:
            response = preprocessed_session.post('/detect_peaks', json=params)
            assert response.status_code == 200
            data = response.get_json()
            assert data['success'] is True

class TestManualPeakAddition:
    """Test manual peak addition functionality."""
    
    def test_add_manual_peak_no_preprocessing(self, client):
        """Test adding manual peak without preprocessing."""
        response = client.post('/add_manual_peak_region', 
                              json={'x_start': 0, 'x_end': 50, 'y_start': 0, 'y_end': 50})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
    
    def test_add_manual_peak_invalid_range(self, preprocessed_session):
        """Test adding manual peak with invalid range."""
        # Y start >= Y end
        response = preprocessed_session.post('/add_manual_peak_region', 
                                           json={'x_start': 0, 'x_end': 50, 'y_start': 50, 'y_end': 10})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
    
    def test_add_manual_peak_success(self, preprocessed_session):
        """Test successful manual peak addition."""
        response = preprocessed_session.post('/add_manual_peak_region', 
                                           json={'x_start': 10, 'x_end': 60, 'y_start': 20, 'y_end': 80})
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert 'peak_added' in data
        assert 'plot' in data

class TestInterfaceManagement:
    """Test interface management functionality."""
    
    def test_get_all_interfaces_empty(self, client):
        """Test getting interfaces when none exist."""
        response = client.get('/get_all_interfaces')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
        assert data['total_count'] == 0
        assert len(data['interfaces']) == 0
    
    def test_remove_interface_not_found(self, client):
        """Test removing non-existent interface."""
        response = client.post('/remove_interface', json={'peak': 999})
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'Interface not found' in data['error']
    
    def test_clear_manual_peaks(self, client):
        """Test clearing manual peaks."""
        response = client.post('/clear_manual_peaks')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True

class TestThicknessCalculation:
    """Test thickness calculation functionality."""
    
    def test_calculate_thickness_no_peaks(self, client):
        """Test thickness calculation without peaks."""
        response = client.post('/calculate_thickness')
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'No peaks detected' in data['error']
    
    def test_calculate_thickness_insufficient_peaks(self, preprocessed_session):
        """Test thickness calculation with insufficient peaks."""
        # Add only one manual peak
        preprocessed_session.post('/add_manual_peak_region', 
                                json={'x_start': 10, 'x_end': 60, 'y_start': 20, 'y_end': 30})
        
        response = preprocessed_session.post('/calculate_thickness')
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
        assert 'At least 2 interfaces needed' in data['error']

class TestFileOperations:
    """Test file download and cleanup operations."""
    
    def test_download_csv_no_data(self, client):
        """Test CSV download without data."""
        response = client.post('/download_csv')
        assert response.status_code == 200
        data = response.get_json()
        assert 'error' in data
    
    def test_cleanup(self, client):
        """Test cleanup operation."""
        response = client.post('/cleanup')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True
    
    def test_reset_analysis(self, client):
        """Test analysis reset."""
        response = client.post('/reset_analysis')
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True