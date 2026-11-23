"""The serving model's identity: read from metadata.json, never fatal when missing."""

import json
from types import SimpleNamespace

import pytest

from src.infra.model_info import UNKNOWN, ModelInfo, current_model_info, read_model_info


def _write(tmp_path, content: str):
    path = tmp_path / "metadata.json"
    path.write_text(content)
    return path


def test_reads_version_and_training_time(tmp_path) -> None:
    path = _write(
        tmp_path,
        json.dumps(
            {"model_version": "v2.0.0", "trained_at": "2026-11-23T15:00:00+00:00"}
        ),
    )
    assert read_model_info(path) == ModelInfo("v2.0.0", "2026-11-23T15:00:00+00:00")


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        "[1, 2, 3]",  # valid JSON, but not an object
        "{}",  # no fields at all
        '{"model_version": "", "trained_at": null}',
    ],
)
def test_bad_or_empty_metadata_falls_back_to_unknown(tmp_path, content: str) -> None:
    info = read_model_info(_write(tmp_path, content))
    assert info.version == "unknown"
    assert info.trained_at is None


def test_missing_file_falls_back_to_unknown(tmp_path) -> None:
    assert read_model_info(tmp_path / "nowhere.json") == UNKNOWN


def test_version_without_training_time_is_kept(tmp_path) -> None:
    info = read_model_info(_write(tmp_path, '{"model_version": "v2.0.0"}'))
    assert info == ModelInfo("v2.0.0", None)


def test_request_without_stored_info_reports_unknown() -> None:
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    assert current_model_info(request) == UNKNOWN  # type: ignore[arg-type]


def test_request_returns_the_info_stored_at_startup() -> None:
    info = ModelInfo("v2.0.0", "2026-11-23T15:00:00+00:00")
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(model_info=info))
    )
    assert current_model_info(request) is info  # type: ignore[arg-type]
