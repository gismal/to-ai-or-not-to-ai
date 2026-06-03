import pytest
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from src.main import app
from src.config import settings

VALID_API_KEY = settings.API_KEY.get_secret_value()

@pytest.fixture
def client():
    """
    Mocks the heavy loading state for the model and database services
    """
    app.state.inference_service = MagicMock()
    
    mock_feedback_service = MagicMock()
    mock_feedback_service.register_feedback = MagicMock()
    
    from src.api.deps import get_feedback_service
    app.dependency_overrides[get_feedback_service] = lambda: mock_feedback_service
    
    yield TestClient(app)
    
    app.dependency_overrides.clear()
    
@pytest.fixture
def auth_headers():
    return {"X-API-Key": VALID_API_KEY}

def test_health_check_withouy_api_key(client):
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
    
def test_predict_image_file_too_large(client, auth_headers):
    """
    mocks the case when upload over 10MB files
    """
    large_data = b"0" * (11 * 1024 * 1024)
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