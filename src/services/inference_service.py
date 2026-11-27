import asyncio
import io
import time

from src.core.enums import InferenceStatus, PredictionLabel
from src.core.exceptions import ModelInferenceError
from src.familiarity import FamiliarityCheck
from src.inference import ONNXPredictor
from src.infra.metrics import CACHE_COUNTER, FAMILIARITY_COUNTER, INFERENCE_HISTOGRAM
from src.schemas.predict import PredictionResponse
from src.services.cache_service import CacheService


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
        familiarity: FamiliarityCheck | None = None,
        enforce_familiarity: bool = False,
    ) -> None:
        """
        Initializes the inference service with its dependencies

        Args:
            - predictor (ONNXPredictor): The loaded ONNX runtime engine
            - threshold (float): The base confidence score to flag an image as AI
            - cache (CacheService): cache service to get the images from the redis cache
            - gray_area_margin (float): The margin below the threshold to classify as UNCERTAIN
            - familiarity (FamiliarityCheck | None): Measures how far an image is from
            the training data. None means the check is off.
            - enforce_familiarity (bool): If True, unfamiliar images lose their definite
            verdict. If False (shadow mode), they're only counted in the metrics.
        """
        self.predictor = predictor
        self.threshold = threshold
        self.cache = cache
        self.lower_bound = threshold - gray_area_margin
        self.one_third = gray_area_margin / 3
        self.familiarity = familiarity
        self.enforce_familiarity = enforce_familiarity

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
            elapsed_ms = round((time.time() - start) * 1000, 2)
            return PredictionResponse(
                **{
                    **cached,
                    "filename": filename,
                    "processing_time_ms": elapsed_ms,
                    "cached": True,
                }
            )

        CACHE_COUNTER.labels(result="miss").inc()
        # -- 2. ONNX ------------------------------
        inference_start = time.time()

        file_stream = io.BytesIO(content_bytes)
        result = await asyncio.to_thread(self.predictor.predict, file_stream, filename)

        INFERENCE_HISTOGRAM.observe(time.time() - inference_start)

        if result.status == InferenceStatus.FAILED:
            raise ModelInferenceError(str(result.error))

        # -- 3. Classify ------------------------
        safe_confidence = float(result.confidence)

        prediction = self._decide_class(safe_confidence)

        familiar = None
        features = getattr(result, "features", None)
        if self.familiarity is not None and features is not None:
            looks_familiar = not self.familiarity.is_unfamiliar(features)
            FAMILIARITY_COUNTER.labels(
                result="familiar" if looks_familiar else "unfamiliar"
            ).inc()
            if self.enforce_familiarity:
                familiar = looks_familiar
                if not looks_familiar:
                    if prediction == PredictionLabel.AI_GENERATED:
                        prediction = PredictionLabel.UNCERTAIN_LEANING_AI
                    elif prediction == PredictionLabel.REAL:
                        prediction = PredictionLabel.UNCERTAIN_LEANING_REAL
        # -- 4. Build response --------------------
        response = {
            "filename": filename,
            "confidence": safe_confidence,
            "prediction": prediction,
            "status": InferenceStatus.SUCCESS,
            "familiar": familiar,
            "processing_time_ms": round((time.time() - start) * 1000, 2),
            "cached": False,
        }

        # -- 5. Cache the result ------------------------
        if self.cache is not None:
            await self.cache.set(content_bytes, response)

        return PredictionResponse(**response)  # type: ignore
