import io 
import pytest
import time
from PIL import Image

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from src.main import app
from src.config import settings
from src.api.deps import get_feedback_service
from src.core.exceptions import InvalidImageFormatError, ModelInferenceError
from src.schemas.predict import InferenceStatus

VALID_API_KEY = settings.API_KEY.get_secret_value()

@pytest.fixture
def client():
    """
    Mocks the heavy loading state for the model and database services
    """
    mock_inference = AsyncMock()
    mock_inference.process_upload.return_value = {
        "filename": "test.png", "prediction": "REAL", "confidence": 0.99, "status": "SUCCESS"
    }
    app.state.inference_service = mock_inference
    mock_feedback_service = MagicMock()
    mock_feedback_service.register_feedback = MagicMock()
    
    app.dependency_overrides[get_feedback_service] = lambda: mock_feedback_service
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()
    
@pytest.fixture
def auth_headers():
    return {"X-API-Key": VALID_API_KEY}

def test_health_check_without_api_key(client):
    response = client.get("/inference/health")
    assert response.status_code == 401
    
def test_health_check_success(client, auth_headers):
    response = client.get("/inference/health", headers = auth_headers)
    assert response.status_code == 200
    
def test_create_feedback_success(client, auth_headers):
    """
    mocks the successful case
    """
    payload = {
        "filename": "image.png",
        "model_prediction": "AI_GENERATED",
        "confidence": 0.88,
        "user_correction": "REAL",
        "client_source": "API_v1"
    }
    response = client.post("/feedback", json = payload, headers = auth_headers)
    assert response.status_code == 202
    assert "queued for processing" in response.json()["message"]

@pytest.mark.parametrize("confidence,expected_label", [
    (0.95, "AI_GENERATED"),
    (0.30, "REAL"),
    (0.65, "UNCERTAIN_LEANING_AI"),
    (0.45, "UNCERTAIN_LEANING_REAL"),
    (0.58, "UNCERTAIN_NEUTRAL"),    
])    
def test_decision_boundary(confidence, expected_label):
    service = InferenceService(
        predictor=MagicMock(),
        threshold=0.75,
        gray_area_margin=0.35
    )
    
    assert service._decide_class(confidence).value == expected_label
 
    
def test_predict_image_file_too_large(client, auth_headers):
    """
    mocks the case when upload over 10MB files
    """
    large_data = b"0" * (11 * 1024 * 1024)  # FIX: 11mb too much
    files = {"file": ("large.png", large_data, "image/png")}
    
    response = client.post("/inference/predict", files = files, headers = auth_headers)
    assert response.status_code == 413
    
@pytest.mark.parametrize(
    "invalid_payload",
    [{"filename": "", "model_prediction": "REAL", "confidence": 0.9, "user_correction": "REAL"},
        {"filename": "1.png", "model_prediction": "INVALID", "confidence": 0.9, "user_correction": "REAL"},
        {"filename": "1.png", "model_prediction": "REAL", "confidence": 1.5, "user_correction": "REAL"},
    ]
)

def test_create_feedback_validation_errors(client, auth_headers, invalid_payload):
    response = client.post("/feedback", json=invalid_payload, headers=auth_headers)
    assert response.status_code == 422
    
def test_predict_success(client, auth_headers):
    """
    tests inference by sending image to /predict endpoint
    """
    img_byte_arr = io.BytesIO()
    Image.new('RGB', (224, 224), color='red').save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    files = {"file": ("test.png", img_byte_arr.read(), "image/png")}
    
    response = client.post("/inference/predict", files=files, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.png"
    assert data["prediction"] == "REAL"
    assert data["confidence"] == 0.99
    assert data["status"] == "SUCCESS"
    assert "processing_time_ms" in data

def test_rate_limiting(client, auth_headers):
    responses = [
        client.post("/inference/predict",
                    files = {"file": ("test.png", b"dummy", "image/png")},
                    headers = auth_headers)
        for _ in range(6)
    ]
    assert any(r.status_code == 429 for r in responses), \
        f"Expected 429, got: {[r.status_code for r in responses]}"
    
def test_predict_invalid_format(client, auth_headers):
    client.app.state.inference_service.process_upload.side_effect = InvalidImageFormatError("bad format")
    files = {"file": ("test.txt", b"dummy text content", "text/plain")}
    response = client.post("/inference/predict", files=files, headers=auth_headers)
    assert response.status_code == 400
    
def test_predict_model_error(client, auth_headers):
    client.app.state.inference_service.process_upload.side_effect = ModelInferenceError("Mocked inference crash")
    img_byte_arr = io.BytesIO()
    Image.new('RGB', (10, 10)).save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    files = {"file": ("test.png", img_byte_arr.read(), "image/png")}
    
    response = client.post("/inference/predict", files=files, headers=auth_headers)
    assert response.status_code == 500
    
@pytest.mark.parametrize("bad_key", ["", "wrong_key_123", "   "])
def test_auth_invalid_keys(client, bad_key):
    response = client.get("/inference/health", headers={"X-API-Key": bad_key})
    assert response.status_code == 401
    
@pytest.fixture(autouse= True)
def reset_rate_limiter():
    yield
    time.sleep(1)