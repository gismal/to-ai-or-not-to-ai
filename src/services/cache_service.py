"""
Handles phash and Redis caching for prediction result
"""

import json
from arq import ArqRedis

from src.core.enums import PredictionLabel
from src.logger import logger
from src.utils import generate_phash


class CacheService:
    """
    Wraps Redis with image-aware caching using perceptual hashing.
    phash produces same hash for visually identical images regardless of minor compression differences
    """

    CACHE_TTL = 86_400  # 24 hours
    KEY_PREFIX = "prediction"  # "prediction: {phash}"

    def __init__(self, redis_pool: ArqRedis) -> None:
        self._redis = redis_pool

    # -- Private -----------------------
    def _make_key(self, image_bytes: bytes) -> str | None:
        """
        Returns a cache key for the image or None if hashing fails
        None means caching is skipped for this image
        """
        img_hash = generate_phash(image_bytes)
        if not img_hash:
            return None
        return f"{self.KEY_PREFIX}:{img_hash}"

    # --- Public API -------------
    async def get(self, image_bytes: bytes) -> dict | None:
        """
        Returns the cached prediction dict or None on a miss.
        Deserialized the PredictionLabel enum from its stored string value
        """
        key = self._make_key(image_bytes)
        if not key:
            return None

        try:
            raw = await self._redis.get(key)
            if not raw:
                return None

            result = json.load(raw)
            result["prediction"] = PredictionLabel(
                result["prediction"]
            )  # restoring string value from enum
            logger.debug(f"Cache hit: {key}")
            return result

        except Exception as e:
            logger.warning(f"Cache read failed for {key}: {e}")
            return None

    async def set(self, image_bytes: bytes, result: dict) -> None:
        """
        Caches a prediction result for 24 hours
        Serialized PredictionLabel enum to string value before storing
        """
        key = self._make_key(image_bytes)
        if not key:
            return

        try:
            payload = result.copy()

            if isinstance(payload.get("prediction"), PredictionLabel):
                payload["prediction"] = payload["prediction"].value

            await self._redis.setex(key, self.CACHE_TTL, json.dumps(payload))
            logger.debug(f"Cache set: {key} (TTL: {self.CACHE_TTL}s)")

        except Exception as e:
            logger.warning(f"Cache write failed for {key}: {e}")
