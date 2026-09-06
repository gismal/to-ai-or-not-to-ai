from pydantic import BaseModel, Field

from src.core.enums import FeedbackLabel, PredictionLabel


class LabelRequest(BaseModel):
    """
    Body for POST /v1/admin/label
    Submits a human-verified label for an UNCERTAIN prediciton
    """

    prediction_log_id: int = Field(description="ID from prediction_log table")
    correct_label: FeedbackLabel = Field(description="Human-verified correct class")


class LabelResponse(BaseModel):
    """
    Confirmation that the label was saved
    """

    prediction_log_id: int
    correct_label: FeedbackLabel
    original_prediction: PredictionLabel
    message: str = "Label saved. It will be in next training run"


class UncertainPrediction(BaseModel):
    """One uncertain prediction awaiting human review"""

    id: int
    filename: str
    confidence: float
    predicted_label: PredictionLabel
    processing_time_ms: float
    created_at: str
