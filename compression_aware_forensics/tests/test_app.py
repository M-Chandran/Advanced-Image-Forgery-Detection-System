"""
Integration tests for Flask application
"""
import pytest
import json
import io
from PIL import Image
import numpy as np
from app import app
class TestFlaskApp:
    """Test cases for Flask application endpoints"""
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        app.config['UPLOAD_FOLDER'] = 'static/test_uploads'
        with app.test_client() as client:
            yield client
    def test_index_page(self, client):
        """Test index page loads"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'AI Image Forensics' in response.data
    def test_upload_no_file(self, client):
        """Test upload endpoint with no file"""
        response = client.post('/upload')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'error' in data
        assert 'No file part' in data['error']
    def test_upload_empty_filename(self, client):
        """Test upload with empty filename"""
        data = {'file': (io.BytesIO(b''), '')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert 'error' in response_data
    @pytest.fixture
    def sample_image_bytes(self):
        """Create sample image bytes for testing"""
        image = Image.new('RGB', (100, 100), color='red')
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr.seek(0)
        return img_byte_arr
    def test_upload_valid_image(self, client, sample_image_bytes):
        """Test upload with valid image"""
        data = {'file': (sample_image_bytes, 'test.jpg')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert 'error' not in response_data
        expected_fields = [
            'compression_type', 'forgery_result', 'accuracy',
            'precision', 'recall', 'f1_score', 'psnr', 'ssim'
        ]
        for field in expected_fields:
            assert field in response_data
    def test_upload_large_file(self, client):
        """Test upload with file exceeding size limit"""
        large_data = b'x' * (20 * 1024 * 1024)
        data = {'file': (io.BytesIO(large_data), 'large.jpg')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 413 or 'error' in json.loads(response.data)
    def test_demo_mode_fallback(self, client, sample_image_bytes, monkeypatch):
        """Test demo mode when models are not available"""
        monkeypatch.setattr('app.TORCH_AVAILABLE', False)
        data = {'file': (sample_image_bytes, 'original_test.jpg')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert 'compression_type' in response_data
        assert response_data['compression_type'] == 'Original'
    def test_copy_move_detection(self, client, tmp_path):
        """Test copy-move detection in response"""
        image = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        source_region = image[20:70, 20:70]
        image[100:150, 100:150] = source_region
        img_path = tmp_path / "copy_move_test.jpg"
        Image.fromarray(image).save(img_path)
        with open(img_path, 'rb') as f:
            data = {'file': (f, 'copy_move_test.jpg')}
            response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert 'copy_move_detected' in response_data
        assert 'highlighted_image' in response_data
        assert 'extracted_regions' in response_data
class TestErrorHandling:
    """Test error handling scenarios"""
    @pytest.fixture
    def client(self):
        """Create test client"""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    def test_invalid_file_type(self, client):
        """Test upload with invalid file type"""
        data = {'file': (io.BytesIO(b'not an image'), 'test.txt')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        response_data = json.loads(response.data)
        assert 'error' in response_data
    def test_corrupted_image(self, client):
        """Test upload with corrupted image data"""
        corrupted_data = b'\x89PNG\r\n\x1a\n' + b'x' * 100
        data = {'file': (io.BytesIO(corrupted_data), 'corrupted.png')}
        response = client.post('/upload', data=data, content_type='multipart/form-data')
        assert response.status_code == 200
        response_data = json.loads(response.data)
if __name__ == "__main__":
    pytest.main([__file__])