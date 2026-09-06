from pydantic import BaseModel, ConfigDict, Field

from src.core.enums import InferenceStatus, PredictionLabel


class ExplainResponse(BaseModel):
    """
    Response model for the /explain endpoint. Extends the standard prediction response with a GradCAM heatmap
    """

    filename: str
    confidence: float = Field(ge=0.0, le=1.0)
    prediction: PredictionLabel
    status: InferenceStatus
    heatmap_base64: str = Field(
        description="Base64-encoded PNG of the GradCAM heatmap overlaid on the original image. "
        "Decode and render client-side as <img src='data:image/png;base64,...'>"
    )
    processing_time_ms: float = Field(ge=0.0)
    cached: bool = False

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "photo.jpg",
                "confidence": 0.91,
                "prediction": "AI_GENERATED",
                "status": "SUCCESS",
                "heatmap_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
                "processing_time_ms": 312.4,
                "cached": False,
            }
        }
    )
