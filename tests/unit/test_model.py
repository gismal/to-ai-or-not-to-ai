"""
future ideas to be added
"""

import io
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from PIL import Image
from prometheus_client import REGISTRY

from src.core.enums import InferenceStatus, PredictionLabel
from src.ml.familiarity import FamiliarityCheck
from src.ml.predictor import _preprocess
from src.services.cache_service import CacheService
from src.services.inference_service import InferenceService
from src.utils import generate_phash


@pytest.fixture
def service():
    return InferenceService(
        predictor=MagicMock(),
        cache=AsyncMock(),
        threshold=0.75,
        gray_area_margin=0.35,
    )


@pytest.fixture
def dummy_image():
    """
    Creates fake, 500x500, in-memo images
    """
    file_stream = io.BytesIO()
    image = Image.new("RGB", (500, 500), color="red")
    image.save(file_stream, format="PNG")
    file_stream.seek(0)
    return file_stream.read()


def test_preprocess_tensor_format(dummy_image):
    """
    Approves the conversion by the _preprocessing if it's tensor format (1, 3, 224, 224) and Float32. No model upload
    """
    image_io = io.BytesIO(dummy_image)

    _, tensor, error = _preprocess(image_io)

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
    assert hash1 is not None
    assert len(hash1) > 0


@pytest.mark.parametrize(
    "confidence, expected",
    [
        (0.92, "AI_GENERATED"),
        (0.25, "REAL"),
        (0.70, "UNCERTAIN_LEANING_AI"),
        (0.43, "UNCERTAIN_LEANING_REAL"),
        (0.58, "UNCERTAIN_NEUTRAL"),
        (0.75, "AI_GENERATED"),  # boundary: exactly at threshold
        (0.40, "REAL"),  # boundary: exactly at lower bound
    ],
)
def test_decide_class_boundaries(service, confidence, expected):
    assert service._decide_class(confidence).value == expected


"""
Preprocessing edge cases
"""


@pytest.mark.parametrize(
    "mode, size",
    [
        ("L", (1, 1)),  # Tiny greyscale
        ("RGBA", (500, 500)),  # Large image with transparency channel
    ],
)
def test_preprocess_edge_cases(mode, size):
    img_byte_arr = io.BytesIO()
    Image.new(mode, size).save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)

    _, tensor, error = _preprocess(img_byte_arr)

    assert error is None
    assert tensor is not None
    assert tensor.shape == (1, 3, 224, 224)


def test_process_applies_normalization(dummy_image):
    """
    After normalization values leave the [0,1] range
    """
    _, tensor, error = _preprocess(io.BytesIO(dummy_image))
    assert error is None
    assert tensor is not None
    assert tensor.min() < 0.0 or tensor.max() > 1.0


def test_preprocess_invalid_image():
    _, tensor, error = _preprocess(io.BytesIO(b"this is not an image"))
    assert tensor is None
    assert error is not None


@given(st.floats(min_value=0.0, max_value=1.0))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_decide_class_property(service, confidence):
    """
    Tests the boundary logic against all possible float values between 0 and 1.
    """
    result = service._decide_class(confidence)
    assert result is not None

    if confidence >= service.threshold:
        assert result.value == "AI_GENERATED"
    elif confidence <= service.lower_bound:
        assert result.value == "REAL"
    else:
        assert "UNCERTAIN" in result.value


@given(st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=500)
def test_decide_class_covers_all_values(confidence):
    service = InferenceService(
        predictor=MagicMock(),
        cache=AsyncMock(),
        threshold=0.75,
        gray_area_margin=0.35,
    )
    label = service._decide_class(confidence)
    assert label in list(PredictionLabel)


class FakeRedis:
    """Stores strings the way real Redis does, so the JSON round trip really runs."""

    def __init__(self):
        self.store = {}

    async def get(self, key):
        return self.store.get(key)

    async def setex(self, key, ttl, value):
        assert isinstance(value, str), (
            "Redis stores strings; the cache must serialize first"
        )
        self.store[key] = value


@pytest.fixture
def image_bytes():
    buf = io.BytesIO()
    Image.linear_gradient("L").convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


async def test_second_identical_image_is_served_from_the_cache(image_bytes):
    service, predictor = make_service(CacheService(FakeRedis(), model_version="v-test"))

    first = await service.predict(image_bytes, "first.png")
    second = await service.predict(image_bytes, "second.png")

    assert first.cached is False
    assert second.cached is True
    assert second.prediction == first.prediction
    assert second.confidence == first.confidence
    assert (
        second.filename == "second.png"
    )  # the current upload's name, not the cached one
    predictor.predict.assert_called_once()  # the model ran only for the first upload


async def test_a_new_model_does_not_reuse_the_old_models_answers(image_bytes):
    redis = FakeRedis()
    old, _ = make_service(CacheService(redis, model_version="v1"))
    new, new_predictor = make_service(CacheService(redis, model_version="v2"))

    await old.predict(image_bytes, "a.png")
    answer = await new.predict(image_bytes, "a.png")

    assert answer.cached is False
    new_predictor.predict.assert_called_once()


def make_service(cache, confidence=0.92, features=None, familiarity=None, enforce=True):
    predictor = MagicMock()
    predictor.predict.return_value = SimpleNamespace(
        status=InferenceStatus.SUCCESS,
        confidence=confidence,
        error=None,
        features=features,
    )
    service = InferenceService(
        predictor=predictor,
        cache=cache,
        threshold=0.75,
        gray_area_margin=0.35,
        familiarity=familiarity,
        enforce_familiarity=enforce,
    )
    return service, predictor


def check_around_zero(threshold=1.0):
    """Familiar means: within `threshold` of the origin, in 4 dimensions."""
    return FamiliarityCheck(
        np.zeros((2, 4), np.float32), np.eye(4, dtype=np.float32), threshold
    )


async def test_an_unfamiliar_image_gets_no_definite_ai_verdict(image_bytes):
    service, _ = make_service(None, 0.92, np.full(4, 10.0), check_around_zero())
    answer = await service.predict(image_bytes, "a.png")
    assert answer.prediction == PredictionLabel.UNCERTAIN_LEANING_AI
    assert answer.familiar is False


async def test_an_unfamiliar_image_gets_no_definite_real_verdict(image_bytes):
    service, _ = make_service(None, 0.10, np.full(4, 10.0), check_around_zero())
    answer = await service.predict(image_bytes, "a.png")
    assert answer.prediction == PredictionLabel.UNCERTAIN_LEANING_REAL


async def test_a_familiar_image_keeps_its_verdict(image_bytes):
    service, _ = make_service(None, 0.92, np.zeros(4), check_around_zero())
    answer = await service.predict(image_bytes, "a.png")
    assert answer.prediction == PredictionLabel.AI_GENERATED
    assert answer.familiar is True


async def test_without_a_check_nothing_changes(image_bytes):
    service, _ = make_service(None, 0.92, np.full(4, 10.0), None)
    answer = await service.predict(image_bytes, "a.png")
    assert answer.prediction == PredictionLabel.AI_GENERATED
    assert answer.familiar is None


async def test_shadow_mode_measures_but_does_not_change_the_verdict(image_bytes):
    def unfamiliar_count():
        return (
            REGISTRY.get_sample_value(
                "ai_detector_familiarity_total", {"result": "unfamiliar"}
            )
            or 0
        )

    before = unfamiliar_count()
    service, _ = make_service(
        None, features=np.full(4, 100.0), familiarity=check_around_zero(), enforce=False
    )
    response = await service.predict(image_bytes, "x.jpg")

    assert response.prediction == PredictionLabel.AI_GENERATED
    assert response.familiar is None
    assert unfamiliar_count() == before + 1
