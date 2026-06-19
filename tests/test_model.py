"""
future ideas to be added
"""
import pytest
import io
from PIL import Image
from unittest.mock import MagicMock
import numpy as np

from src.utils import generate_phash
from src.inference import _preprocess
from src.services.inference_service import InferenceService

@pytest.fixture
def service():
    return InferenceService(predictor= MagicMock(), threshold = 0.75, gray_area_margin = 0.35)

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
    assert tensor.dtype == np.float32
        
def test_generate_phash_consistency(dummy_image):
    """
    checks the phash values of the image
    """
    hash1 = generate_phash(dummy_image)
    hash2 = generate_phash(dummy_image)
    
    assert hash1 == hash2
    assert isinstance(hash2, str)
    assert len(hash1) > 0

@pytest.mark.parametrize("confidence, expected", [
    (0.92, "AI_GENERATED"),
    (0.25, "REAL"),
    (0.70, "UNCERTAIN_LEANING_AI"),
    (0.43, "UNCERTAIN_LEANING_REAL"),
    (0.58, "UNCERTAIN_NEUTRAL"),
    (0.75, "AI_GENERATED"),   # boundary: exactly at threshold
    (0.40, "REAL"),           # boundary: exactly at lower bound
])
def test_decide_class_boundariees(service, confidence, expected):
    assert service._decide_class(confidence).value == expected

    
"""
Preprocessing edge cases
"""
@pytest.mark.parametrize("mode, size", [
    ("L", (1, 1)),          # Çok küçük siyah-beyaz
    ("RGBA", (500, 500))    # Saydamlık içeren büyük resim
])
def test_preprocess_edge_cases(mode, size): 
    img_byte_arr = io.BytesIO()
    Image.new(mode, size).save(img_byte_arr, format = "PNG")
    img_byte_arr.seek(0)
    
    path_str, tensor, error = _preprocess(img_byte_arr)
    
    assert error is None
    assert tensor.shape == (1,3,224,224)
    
def test_process_applies_normalization(dummy_image):
    """
    After normalization values leave the [0,1] range
    """
    _, tensor, error = _preprocess(io.BytesIO(dummy_image))
    assert error is None
    assert tensor.min() < 0.0 or tensor.max() > 1.0
    
def test_preprocess_invalid_image():
    path_str, tensor, error = _preprocess(io.BytesIO(b"this is not an image"))
    assert tensor is None
    assert error is not None