import pytest
import json
import os
from app import app

class TestAppConfiguration:
    """Test app configuration and setup."""
    
    def test_app_exists(self):
        """Test that the app exists."""
        assert app is not None
    
    def test_app_is_in_testing_mode(self, client):
        """Test that app is in testing mode."""
        assert app.config['TESTING'] is True
    
    def test_upload_folder_exists(self, client):
        """Test that upload folder is created."""
        assert os.path.exists(app.config['UPLOAD_FOLDER'])

class TestRoutes:
    """Test basic route availability."""
    
    def test_index_route(self, client):
        """Test the main index route."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Layer Thickness Measurement' in response.data
    
    def test_404_error(self, client):
        """Test 404 error handling."""
        response = client.get('/nonexistent')
        assert response.status_code == 404

class TestUtilityFunctions:
    """Test utility functions from the app."""
    
    def test_normalize_function(self):
        """Test the normalize function."""
        from app import normalize
        import numpy as np
        
        # Test with simple array
        arr = np.array([1, 2, 3, 4, 5])
        normalized = normalize(arr)
        
        assert normalized.min() == 0.0
        assert normalized.max() == 1.0
        assert len(normalized) == len(arr)
        
        # Test with constant array
        const_arr = np.array([5, 5, 5, 5])
        normalized_const = normalize(const_arr)
        assert np.all(normalized_const == 0)  # Should be all zeros when min==max
    
    def test_plot_to_base64(self):
        """Test plot to base64 conversion."""
        from app import plot_to_base64
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 2])
        
        base64_str = plot_to_base64(fig)
        
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0
        # Base64 strings should be valid
        import base64
        try:
            base64.b64decode(base64_str)
            is_valid_base64 = True
        except:
            is_valid_base64 = False
        assert is_valid_base64
    
    def test_image_to_base64(self, sample_image):
        """Test image to base64 conversion."""
        from app import image_to_base64
        
        base64_str = image_to_base64(sample_image)
        
        assert isinstance(base64_str, str)
        assert len(base64_str) > 0