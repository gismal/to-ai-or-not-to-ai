"""
Tests for reading the current model's training time from models/metadata.json.

The drift check only counts feedback given after the serving model was trained, so
complaints about a replaced model cannot trigger another retrain. That makes this
timestamp part of the drift logic, and two things must hold:

* It is always timezone-aware (UTC). Mixing naive and aware datetimes has caused
  "can't subtract offset-naive and offset-aware datetimes" errors in this project.
* A missing or malformed metadata file never crashes the drift check; the reader
  returns None and the check falls back to the plain lookback window.
"""

import json
from datetime import UTC, datetime

import pytest

from src.services import retrain_service


def _use_metadata(tmp_path, monkeypatch, content: str | None) -> None:
    """Point the reader at a temporary metadata.json. None means the file does not exist."""
    path = tmp_path / "metadata.json"
    if content is not None:
        path.write_text(content)
    monkeypatch.setattr(retrain_service, "METADATA_PATH", path)


def test_reads_a_timezone_aware_timestamp(tmp_path, monkeypatch):
    """The ISO timestamp written by train.py is read back unchanged."""
    _use_metadata(
        tmp_path, monkeypatch, json.dumps({"trained_at": "2026-09-13T12:23:10+00:00"})
    )

    assert retrain_service._model_trained_at() == datetime(
        2026, 9, 13, 12, 23, 10, tzinfo=UTC
    )


def test_a_naive_timestamp_is_treated_as_utc(tmp_path, monkeypatch):
    """Older metadata without an offset is interpreted as UTC, so comparisons stay safe."""
    _use_metadata(
        tmp_path, monkeypatch, json.dumps({"trained_at": "2026-09-13T12:23:10"})
    )

    trained = retrain_service._model_trained_at()

    assert trained is not None
    assert trained.tzinfo is UTC


@pytest.mark.parametrize(
    "content",
    [
        pytest.param(None, id="file missing"),
        pytest.param("{not json", id="invalid json"),
        pytest.param("{}", id="no trained_at key"),
        pytest.param('{"trained_at": null}', id="trained_at is null"),
        pytest.param('{"trained_at": "yesterday"}', id="not a date"),
    ],
)
def test_unknown_when_missing_or_malformed(tmp_path, monkeypatch, content):
    """Anything unreadable gives None instead of an exception."""
    _use_metadata(tmp_path, monkeypatch, content)

    assert retrain_service._model_trained_at() is None
