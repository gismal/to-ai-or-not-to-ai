import tempfile
import asyncio
import time
from pathlib import Path
from fastapi import UploadFile
from arq import ArqRedis

from src.inference import ONNXPredictor
from src.logger import logger
from src.core.exceptions import InvalidImageFormatError, ModelInferenceError
from src.core.enums import PredictionLabel, InferenceStatus


class InferenceService:
    """
    Encapsulates the core business logic, includng asynchronous file streaming, model execution and granular uncertainty classification
    """
    def __init__(self, predictor: ONNXPredictor, redis_pool: redis_pool, threshold: float, gray_area_margin: float = 0.35):
        """
        Initializes the inference service with its dependencies
        
        Args:
            - predictor (ONNXPredictor): The loaded ONNX runtime engine
            - threshold (float): The base confidence score to flag an image as AI
            - gray_area_margin (float): The amrgin below the threshol to classify as UNCERTAIN
        
        """
        self.predictor = predictor
        self.redis_pool = redis_pool
        self.threshold = threshold
        self.lower_bound = threshold - gray_area_margin
        self.one_third = gray_area_margin / 3
    
    async def trigger_retraining_process(self, dataset_path: str):
        await self.redis_pool.enqueue_job("train_model_task", dataset_path = dataset_path)
        logger.info(f"Retraining job enqueued for {dataset_path}")
            
    def _decide_class(self, confidence: float) -> PredictionLabel:
        """
        Determines the final label with detailed granularity for uncertain cases
        
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
        elif confidence <= (self.lower_bound + self.one_third):
            return PredictionLabel.UNCERTAIN_LEANING_REAL
        else:
            return PredictionLabel.UNCERTAIN_NEUTRAL
        
    async def process_upload(self, file: UploadFile) -> dict:
        """ 
        Streams the uploaded file to disk and runs inference inside an asynchronous thread pool.
        
        Args:
            file (UploadFile): The image file uploaded via the API.
        
        Returns: 
            dict: A dictionary containing the filename, confidence score, final prediction and status.
            
        Raises:
            InvalidImageFormatError: If the file extension is not JPG/PNG.
            ModelInferenceError: If the ONNX engine encounters a runtime error.
        """
        start_time = time.time()
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            raise InvalidImageFormatError("Only JPG or PNG formats are supported")
        
        import io
        import imghdr
        
        content = io.BytesIO(await file.read())
        if imghdr.what(None, h= content) not in ("jpeg", "png"):
            raise InvalidImageFormatError("File content does not match a supported image format")        
        result = await asyncio.to_thread(self.predictor.predict, content, file.filename)
            
        if result.status == InferenceStatus.FAILED:     
            raise ModelInferenceError(str(result.error))
        
        final_prediction = self._decide_class(result.confidence)
        
        if "UNCERTAIN" in final_prediction.value:
            logger.warning(f"UNCERTAIN detected: {file.filename} (Score: {result.confidence})")
        return {
            "filename": file.filename,
            "confidence": result.confidence,
            "prediction": final_prediction,
            "status": "SUCCESS",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }       
        