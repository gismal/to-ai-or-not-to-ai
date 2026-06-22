from typing import Literal
from pydantic import BaseModel, Field, ConfigDict
from src.infra.feedbacks import FeedbackLabel
from src.schemas.predict import PredictionLabel


class FeedbackCreateRequest(BaseModel):
    """
    Pydantic schema for validating incoming feedback requests.
    """

    filename: str = Field(
        ..., min_length=1, description="Original name of the analyzed image file"
    )
    model_prediction: PredictionLabel
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0"
    )
    user_correction: FeedbackLabel = Field(
        ..., description="Actual label provided by the user (REAL or AI_GENERATED)"
    )
    client_source: Literal["API_v1", "WEB_UI"] = Field(
        default="API_v1", description="Source client of the feedback"
    )


class FeedbackResponse(BaseModel):
    """
    Pydantic schema for the feedback creation response.
    """

    filename: str = Field(..., description="Name of the processed file")
    message: str = Field(
        default="Feedback received successfully", description="Status message"
    )

    model_config = ConfigDict(from_attributes=True)
