import io
import pytest
import time
from PIL import Image

from unittest.mock import MagicMock
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock

from src.main import app
from src.config import settings
from src.api.deps import (
    get_feedback_service,
    get_prediction_log_repository,
    get_explainability_service,
)
from src.core.exceptions import ModelInferenceError


VALID_API_KEY = settings.API_KEY.get_secret_value()


@pytest.fixture
def client():
    """
    Mocks the heavy loading state for the model and database services
    """
    mock_inference = AsyncMock()
    mock_inference.predict.return_value = {
        "filename": "test.png",
        "prediction": "REAL",
        "confidence": 0.99,
        "status": "SUCCESS",
        "processing_time_ms": 15.5,
    }
    app.state.inference_service = mock_inference
    app.state.explainability_service = MagicMock()
    app.state.arq_pool = AsyncMock()
    mock_feedback_service = MagicMock()
    mock_feedback_service.register_feedback = AsyncMock()
    mock_log_repo = MagicMock()
    mock_log_repo.create_log = AsyncMock()

    app.dependency_overrides[get_feedback_service] = lambda: mock_feedback_service
    app.dependency_overrides[get_prediction_log_repository] = lambda: mock_log_repo

    yield TestClient(app)

    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers():
    return {"X-API-Key": VALID_API_KEY}


def test_health_check_without_api_key(client):
    response = client.get("/v1/inference/health")
    assert response.status_code == 401


def test_health_check_success(client, auth_headers):
    response = client.get("/v1/inference/health", headers=auth_headers)
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
        "client_source": "API_v1",
    }
    response = client.post("/v1/feedback", json=payload, headers=auth_headers)
    assert response.status_code == 200
    assert "queued for processing" in response.json()["message"]


@pytest.mark.parametrize(
    "confidence,expected_label",
    [
        (0.95, "AI_GENERATED"),
        (0.30, "REAL"),
        (0.65, "UNCERTAIN_LEANING_AI"),
        (0.45, "UNCERTAIN_LEANING_REAL"),
        (0.58, "UNCERTAIN_NEUTRAL"),
    ],
)
@pytest.mark.parametrize(
    "invalid_payload",
    [
        {
            "filename": "",
            "model_prediction": "REAL",
            "confidence": 0.9,
            "user_correction": "REAL",
        },
        {
            "filename": "1.png",
            "model_prediction": "INVALID",
            "confidence": 0.9,
            "user_correction": "REAL",
        },
        {
            "filename": "1.png",
            "model_prediction": "REAL",
            "confidence": 1.5,
            "user_correction": "REAL",
        },
    ],
)
def test_create_feedback_validation_errors(client, auth_headers, invalid_payload):
    response = client.post("/v1/feedback", json=invalid_payload, headers=auth_headers)
    assert response.status_code == 422


def test_predict_success(client, auth_headers):
    """
    tests inference by sending image to /predict endpoint
    """
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (224, 224), color="red").save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    files = {"file": ("test.png", img_byte_arr.read(), "image/png")}

    response = client.post("/v1/inference/predict", files=files, headers=auth_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.png"
    assert data["prediction"] == "REAL"
    assert data["confidence"] == 0.99
    assert data["status"] == "SUCCESS"
    assert "processing_time_ms" in data


def test_rate_limiting(client, auth_headers):
    responses = [
        client.post(
            "/v1/inference/predict",
            files={"file": ("test.png", b"dummy", "image/png")},
            headers=auth_headers,
        )
        for _ in range(6)
    ]
    assert any(
        r.status_code == 429 for r in responses
    ), f"Expected 429, got: {[r.status_code for r in responses]}"


def test_predict_invalid_format(client, auth_headers):
    files = {"file": ("test.txt", b"dummy text content", "text/plain")}
    response = client.post("/v1/inference/predict", files=files, headers=auth_headers)
    assert response.status_code == 400


def test_predict_model_error(client, auth_headers):
    client.app.state.inference_service.predict.side_effect = ModelInferenceError(
        "Mocked inference crash"
    )
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (10, 10)).save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    files = {"file": ("test.png", img_byte_arr.read(), "image/png")}

    response = client.post("/v1/inference/predict", files=files, headers=auth_headers)
    assert response.status_code == 500


@pytest.mark.parametrize("bad_key", ["", "wrong_key_123", "   "])
def test_auth_invalid_keys(client, bad_key):
    response = client.get("/v1/inference/health", headers={"X-API-Key": bad_key})
    assert response.status_code == 401


@pytest.fixture(autouse=True)
def reset_rate_limiter():
    yield
    time.sleep(1)


def test_predict_phash_cache(client, auth_headers):
    """
    Tests if the second identical request returns cached: True
    """
    img_byte_arr = io.BytesIO()
    Image.new("RGB", (100, 100), color="blue").save(img_byte_arr, format="PNG")
    img_data = img_byte_arr.getvalue()

    client.app.state.inference_service.predict.side_effect = [
        {
            "filename": "test.png",
            "prediction": "REAL",
            "confidence": 0.99,
            "status": "SUCCESS",
            "processing_time_ms": 42.0,
        },
        {
            "filename": "test.png",
            "prediction": "REAL",
            "confidence": 0.99,
            "status": "SUCCESS",
            "processing_time_ms": 0.0,
        },
    ]

    res1 = client.post(
        "/v1/inference/predict",
        files={"file": ("test.png", img_data, "image/png")},
        headers=auth_headers,
    )
    res2 = client.post(
        "/v1/inference/predict",
        files={"file": ("test.png", img_data, "image/png")},
        headers=auth_headers,
    )

    assert res1.json().get("processing_time_ms") == 42.0
    assert res2.json().get("processing_time_ms") == 0.0


def test_explain_endpoint(client, auth_headers):
    """
    Tests the /explain endpoint structure
    """

    class MockExplainer:
        async def explain(self, *args, **kwargs):
            return {
                "filename": "test.png",
                "prediction": "REAL",
                "confidence": 0.99,
                "status": "SUCCESS",
                "processing_time_ms": 150.0,
                "heatmap_base64": "dummy_base64_string",
            }

    client.app.state.explainability_service = MockExplainer()

    img_byte_arr = io.BytesIO()
    Image.new("RGB", (10, 10), color="blue").save(img_byte_arr, format="PNG")
    files = {"file": ("test.png", img_byte_arr.getvalue(), "image/png")}

    response = client.post("/v1/inference/explain", files=files, headers=auth_headers)

    assert response.status_code == 200
    data = response.json()
    assert "heatmap_base64" in data

    client.app.dependency_overrides.pop(get_explainability_service, None)
