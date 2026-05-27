from pydantic import BaseModel
from enum import Enum

class PredictionLabel(str, Enum):
    """ 
    Enumaration of all possible prediction outcomes
    """
    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"
    UNCERTAIN_LEANING_AI = "UNCERTAIN_LEANING_AI"
    UNCERTAIN_LEANING_REAL = "UNCERTAIN_LEANING_REAL"
    UNCERTAIN_NEUTRAL = "UNCERTAIN_NEUTRAL"

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
    confidence: float
    prediction: PredictionLabel
    status: str
    
class InferenceStatus(str, Enum):
    """Enumeration of the inference execution status."""
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"