"""
future ideas to be added
"""
import pytest
import io
from PIL import Image
from src.utils import generate_phash
from src.inference import _preprocess

@pytest.fixture
def dummy_image():
    """
    Creates fake, 500x500, in-memo images 
    """
    file_stream = io.BytesIO()
    image = Image.new("RGB", (500, 500), color = "red")
    image.save(file_stream, format= "PNG")
    file_stream.seek(0)
    return file_stream.read()

def test_preprocess_tensor_format(dummy_image):
    """
    Approves the conversion by the _preprocessing if it's tensor format (1, 3, 224, 224) and Float32. No model upload
    """
    image_io = io.BytesIO(dummy_image)
    
    path_str, tensor, error = _preprocess(image_io)
    
    assert error is None
    assert tensor is not None
    assert tensor.shape == (1, 3, 224, 224)
        
def test_generate_phash_consistency(dummy_image):
    """
    checks the phash values of the image
    """
    hash1 = generate_phash(dummy_image)
    hash2 = generate_phash(dummy_image)
    
    assert hash1 == hash2
    assert isinstance(hash2, str)
    assert len(hash1) > 0