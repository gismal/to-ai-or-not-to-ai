from pydantic import BaseModel, Field, ConfigDict
from enum import Enum

from src.core.enums import FeedbackLabel, PredictionLabel, InferenceStatus

class PredictionResponse(BaseModel):
    """
    Data model representing the standard API response for image inference
    
    Attributes:
        - filename (str): The name of the processed image file
        - confidence (float): The model's confidence score (0.0 - 1.0)
        - prediction (PredictionLabel): The final classification ('REAL', 'AI_GENERATED', 'UNCERTAIN')
        - status (str): The operational status of the request (e.g. 'SUCCESS')
    """
    filename: str
    confidence: float = Field(ge= 0.0, le= 1.0)
    prediction: PredictionLabel
    status: InferenceStatus
    processing_time_ms: float = Field(default= 0.0, ge= 0.0)
    
    # Swagger Doc Example
    model_config = ConfigDict(
        json_schema_extra = {
            "example": {
                "filename": "image.jpg",
                "confidence": 0.92,
                "prediction": "AI_GENERATED",
                "status": "SUCCESS",
                "processing_time_ms": 145.2
        }
        }
    )
    
