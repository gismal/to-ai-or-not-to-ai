"""
Which model is serving: read once from models/metadata.json at startup.

The version goes into /health, into every prediction response and into the cache key,
so every answer can be traced to the model that produced it and a new model never serves
answers its predessor called
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fastapi import Request

from src.logger import logger

METADATA_PATH = Path("models/metadata.json")


@dataclass(frozen=True)
class ModelInfo:
    version: str
    trained_at: str | None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


UNKNOWN = ModelInfo(version="unknown", trained_at=None)


def read_model_info(path: Path = METADATA_PATH) -> ModelInfo:
    """
    The serving model's version and training time

    Never raises a missing or broken metadata file. Because it will stop the API
    it will fall back to "unknown" and log a warning instead
    """
    try:
        meta = json.loads(path.read_text())
        version = meta.get("model_version")
        trained_at = meta.get("trained_at")
        return ModelInfo(
            version=str(version) if version else UNKNOWN.version,
            trained_at=str(trained_at) if trained_at else None,
        )
    except (OSError, ValueError, AttributeError) as e:
        logger.warning(f"Could not read model metadata from {path}: {e}")
        return UNKNOWN


def current_model_info(request: Request) -> ModelInfo:
    """The ModelInfo stored at startup, or UNKNOWN if startup did not set one (e.g. tests)."""
    return getattr(request.app.state, "model_info", UNKNOWN)
