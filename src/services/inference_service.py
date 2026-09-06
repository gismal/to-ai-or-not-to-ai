import asyncio
import io
import time

from src.schemas.predict import PredictionResponse
from src.core.enums import InferenceStatus, PredictionLabel
from src.core.exceptions import ModelInferenceError
from src.inference import ONNXPredictor
from src.logger import logger
from src.services.cache_service import CacheService
from src.infra.metrics import CACHE_COUNTER, INFERENCE_HISTOGRAM, PREDICTION_COUNTER


class InferenceService:
    """
    Orchastrates rpediciton for a single image
    """

    def __init__(
        self,
        predictor: ONNXPredictor,
        cache: CacheService,
        threshold: float,
        gray_area_margin: float = 0.35,
    ) -> None:
        """
        Initializes the inference service with its dependencies

        Args:
            - predictor (ONNXPredictor): The loaded ONNX runtime engine
            - threshold (float): The base confidence score to flag an image as AI
            - cache (CacheService): cache service to get the images from the redis cache
            - gray_area_margin (float): The amrgin below the threshol to classify as UNCERTAIN

        """
        self.predictor = predictor
        self.threshold = threshold
        self.cache = cache
        self.lower_bound = threshold - gray_area_margin
        self.one_third = gray_area_margin / 3

    def _decide_class(self, confidence: float) -> PredictionLabel:
        """
        Maps a confidence score. There are five labels:

                The confidence score represents probability of AI_GENERATED (class 0).
        Zones:
          [0.00 - lower_bound]          → REAL
          [lower_bound - lower_bound+⅓] → UNCERTAIN_LEANING_REAL
          [lower_bound+⅓ - threshold-⅓] → UNCERTAIN_NEUTRAL
          [threshold-⅓  - threshold]    → UNCERTAIN_LEANING_AI
          [threshold - 1.00]            → AI_GENERATED

        Args:
            - confidence (float): The prediction score from the ONNX model

        Returns:
            - PredictionLabel: Classification outcome

        """
        # sharp bounds
        if confidence >= self.threshold:
            return PredictionLabel.AI_GENERATED
        if confidence <= self.lower_bound:
            return PredictionLabel.REAL

        # analysis of uncertain area
        if confidence >= (self.threshold - self.one_third):
            return PredictionLabel.UNCERTAIN_LEANING_AI
        if confidence <= (self.lower_bound + self.one_third):
            return PredictionLabel.UNCERTAIN_LEANING_REAL
        return PredictionLabel.UNCERTAIN_NEUTRAL

    # -- Prediction -------------------------
    async def predict(
        self,
        content_bytes: bytes,
        filename: str,
    ) -> PredictionResponse:
        """
        Full prediction pipeline: cache -> ONNX -> classify -> cache

        Args:
            - content_bytes: Raw, validated image bytes
            - filename: safe sanitized filename

        Returns:
            Response dict matching PredictionResponse schema

        Raises:
            - ModelInferenceError: If ONNX runtime fails
        """
        start = time.time()

        # -- 1. Cache lookup ---------------------
        cached = None
        if self.cache is not None:
            cached = await self.cache.get(content_bytes)

        if cached:
            CACHE_COUNTER.labels(result="hit").inc()
            return PredictionResponse(
                **cached,
                processing_time_ms=round((time.time() - start) * 1000, 2),
                cached=True,
            )

        CACHE_COUNTER.labels(result="miss").inc()
        # -- 2. ONNX ------------------------------
        inference_start = time.time()

        file_stream = io.BytesIO(content_bytes)
        result = await asyncio.to_thread(self.predictor.predict, file_stream, filename)

        INFERENCE_HISTOGRAM.observe(time.time() - inference_start)

        if result.status == InferenceStatus.FAILED:
            raise ModelInferenceError(str(result.confidence))

        # -- 3. Classify ------------------------
        prediction = self._decide_class(result.confidence)
        PREDICTION_COUNTER.labels(label=prediction.value).inc()

        if "UNCERTAIN" in prediction.value:
            logger.warning(
                f"UNCERTAIN prediction: {filename} "
                f"(confidence: {result.confidence:.4f})"
            )

        # -- 4. Build response --------------------
        response = {
            "filename": filename,
            "confidence": result.confidence,
            "prediction": prediction,
            "status": InferenceStatus.SUCCESS,
            "processing_time_ms": round((time.time() - start) * 1000, 2),
            "cached": False,
        }

        # -- 5. Cache the result ------------------------
        await self.cache.set(content_bytes, response)

        return response
