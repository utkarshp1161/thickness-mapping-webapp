import pytest
import tempfile
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from app import app
import hyperspy.api as hs

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    app.config['UPLOAD_FOLDER'] = tempfile.mkdtemp()
    
    with app.test_client() as client:
        with app.app_context():
            yield client
    
    # Cleanup
    import shutil
    if os.path.exists(app.config['UPLOAD_FOLDER']):
        shutil.rmtree(app.config['UPLOAD_FOLDER'])

@pytest.fixture
def sample_image():
    """Create a sample 2D numpy array that mimics an EMD image."""
    # Create a synthetic layered structure
    height, width = 200, 150
    image = np.zeros((height, width))
    
    # Add some layers with different intensities
    layer_positions = [20, 60, 100, 140, 180]
    for i, pos in enumerate(layer_positions):
        # Add layer with some thickness
        start = max(0, pos - 5)
        end = min(height, pos + 5)
        intensity = 0.2 + (i * 0.15)
        image[start:end, :] = intensity
    
    # Add some noise
    noise = np.random.normal(0, 0.05, (height, width))
    image += noise
    
    # Normalize to 0-1
    image = (image - image.min()) / (image.max() - image.min())
    
    return image

@pytest.fixture
def mock_emd_file(sample_image, tmp_path):
    """Create a mock EMD file for testing."""
    # Create a temporary EMD file
    emd_path = tmp_path / "test_data.emd"
    
    # Create a mock hyperspy signal
    signal = hs.signals.Signal2D(sample_image)
    signal.metadata.set_item("General.title", "Test Image")
    
    # Add mock metadata
    signal.original_metadata.set_item("BinaryResult.PixelSize.height", 1e-9)  # 1 nm
    
    # Save as EMD
    signal.save(str(emd_path))
    
    return str(emd_path)

@pytest.fixture
def uploaded_session(client, mock_emd_file):
    """Fixture that uploads a file and returns the client with session data."""
    with open(mock_emd_file, 'rb') as f:
        response = client.post('/upload', 
                             data={'file': (f, 'test_data.emd')},
                             content_type='multipart/form-data')
    
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    
    return client

@pytest.fixture
def preprocessed_session(uploaded_session):
    """Fixture with uploaded and preprocessed image."""
    client = uploaded_session
    
    # Select first image
    response = client.post('/select_image', 
                          json={'index': 0})
    assert response.status_code == 200
    
    # Preprocess image
    response = client.post('/preprocess', 
                          json={'sigma': 4.0})
    assert response.status_code == 200
    
    return client