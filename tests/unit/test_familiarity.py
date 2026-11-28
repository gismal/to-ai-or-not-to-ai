"""The familiarity check: distances, fitting, loading, and the feature-returning model."""

import numpy as np
import pytest
import torch

from scripts.training.familiarity import WithFeatures, fit
from scripts.training.model import build_model
from src.ml.familiarity import FamiliarityCheck, mahalanobis

rng = np.random.default_rng(0)


@pytest.fixture
def clusters():
    """Two familiar classes around +3 and -3 in 8 dimensions."""
    a = rng.normal(3.0, 1.0, size=(400, 8))
    b = rng.normal(-3.0, 1.0, size=(400, 8))
    return np.vstack([a, b]), np.array([0] * 400 + [1] * 400)


def test_a_class_centre_is_at_distance_zero():
    means = np.array([[1.0, 2.0], [5.0, 5.0]])
    assert mahalanobis(means[:1], means, np.eye(2))[0] == pytest.approx(0.0)


def test_distance_is_to_the_nearest_centre():
    means = np.array([[0.0, 0.0], [10.0, 0.0]])
    assert mahalanobis(np.array([[9.0, 0.0]]), means, np.eye(2))[0] == pytest.approx(
        1.0
    )


def test_the_threshold_flags_about_the_chosen_share_of_familiar_images(clusters):
    x, y = clusters
    val = np.vstack(
        [rng.normal(3.0, 1.0, size=(500, 8)), rng.normal(-3.0, 1.0, size=(500, 8))]
    )
    stats = fit(x, y, val, percentile=95.0)
    check = FamiliarityCheck(
        stats["means"], stats["precision"], float(stats["threshold"])
    )
    flagged = (check.distance(val) > check.threshold).mean()
    assert flagged == pytest.approx(0.05, abs=0.01)


def test_something_far_away_is_unfamiliar_and_something_typical_is_not(clusters):
    x, y = clusters
    stats = fit(x, y, x)
    check = FamiliarityCheck(
        stats["means"], stats["precision"], float(stats["threshold"])
    )
    assert check.is_unfamiliar(np.full(8, 40.0))
    assert not check.is_unfamiliar(np.full(8, 3.0))


def test_the_check_round_trips_through_a_file(tmp_path, clusters):
    x, y = clusters
    path = tmp_path / "familiarity.npz"
    np.savez(path, **fit(x, y, x))
    check = FamiliarityCheck.load(path)
    assert check is not None
    assert check.means.shape == (2, 8)


@pytest.mark.parametrize("content", [None, b"not a numpy file"])
def test_a_missing_or_broken_file_means_no_check(tmp_path, content):
    path = tmp_path / "familiarity.npz"
    if content is not None:
        path.write_bytes(content)
    assert FamiliarityCheck.load(path) is None


def test_the_feature_model_gives_the_same_answer_plus_576_features():
    model = build_model(2, pretrained=False).eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        logits, features = WithFeatures(model)(x)
        assert torch.allclose(logits, model(x), atol=1e-5)
    assert features.shape == (3, 576)


def test_the_onnx_export_has_scores_and_features_that_match_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    onnxscript = pytest.importorskip("onnxscript")  # noqa: F841  (the exporter needs it)
    from scripts.training.export import export_onnx

    model = build_model(2, pretrained=False).eval()
    path = tmp_path / "model.onnx"
    export_onnx(model, path)

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["output", "features"]
    x = torch.randn(2, 3, 224, 224)
    logits, features = session.run(None, {"input": x.numpy()})
    with torch.no_grad():
        want_logits, want_features = WithFeatures(model)(x)
    assert np.allclose(logits, want_logits.numpy(), atol=1e-4)
    assert np.allclose(features, want_features.numpy(), atol=1e-4)
