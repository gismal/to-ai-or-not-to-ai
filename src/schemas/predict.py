from pydantic import BaseModel, ConfigDict, Field

from src.core.enums import InferenceStatus, PredictionLabel


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
    confidence: float = Field(ge=0.0, le=1.0)
    prediction: PredictionLabel
    status: InferenceStatus
    familiar: bool | None = Field(
        default=None,
        description="False if the image looks unlike the training data; the verdict is then never definite. "
        "None if the check isn't available.",
    )
    processing_time_ms: float = Field(default=0.0, ge=0.0)
    cached: bool = False
    model_version: str = Field(
        default="unknown", description="The model that produced this answer"
    )

    # Swagger Doc Example
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "image.jpg",
                "confidence": 0.92,
                "prediction": "AI_GENERATED",
                "status": "SUCCESS",
                "processing_time_ms": 145.2,
                "model_version": "v2.0.0",
            }
        }
    )
