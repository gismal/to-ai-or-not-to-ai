import io
import json
import magic
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
from src.utils import generate_phash

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
        
        safe_filename = PurePosixPath(file.filename).name
        
        content = await file.read()
        mime = magic.from_buffer(content_bytes[:2048], mime = True)
        if mime not in ("image/jpeg", "image/png"):
            raise InvalidImageFormatError
        
        img_hash = generate_phash(content_bytes)
        cache_key= f"phash: {img_hash}" if img_hash else None
        
        if cache_key:
            cached_result = await self.redis_pool.get(cache_key)
            if cached_result:
                logger.info(f"Cache hit: {safe_filename} (pHash: {img_hash})")
                parsed_cache = json.loads(cached_result)
                parsed_cache["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
                parsed_cache["cached"] = True
                parsed_cache["prediction"] = PredictionLabel(parsed_cache["prediction"])
                return parsed_cache
        
        content = io.BytesIO(content_bytes)
        result = await asyncio.to_thread(self.predictor.predict, content, safe_filename)
             
        if result.status == InferenceStatus.FAILED:     
            raise ModelInferenceError(str(result.error))
        
        final_prediction = self._decide_class(result.confidence)
        
        if "UNCERTAIN" in final_prediction.value:
            logger.warning(f"UNCERTAIN detected: {file.filename} (Score: {result.confidence})")
        
        response_data = {
            "filename": file.filename,
            "confidence": result.confidence,
            "prediction": final_prediction,
            "status": "SUCCESS",
            "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }       
        
        if cache_key:
            cache_payload = response_data.copy()
            cache_payload["prediction"] = final_prediction.value 
            await self.redis_pool.setex(cache_key, 86400, json.dumps(cache_payload))
            
        response_data["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        response_data["cached"] = False
        return response_data
        