"""
LabeledSample: stores ground truth for UNCERTAIN predictions provided by human

This is the active learning table. When the model is uncertain:
  1. The prediction is logged to prediction_logs with an UNCERTAIN label
  2. An admin reviews it via GET /v1/admin/uncertain
  3. Admin submits the correct label via POST /v1/admin/label
  4. The label is stored here
  5. When retraining, these samples are exported to data/train/ automatically

This closes the human-in-the-loop gap in the MLOps cycle.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy import (
    Enum as SQLEnum,
)

from src.core.enums import FeedbackLabel, PredictionLabel
from src.infra.database import Base


class LabeledSample(Base):
    __tablename__ = "labeled_samples"
    __table_args__ = (
        UniqueConstraint("prediction_log_id", name="uq_labeled_sample_prediction"),
    )

    id = Column(Integer, primary_key=True, index=True)
    prediction_log_id = Column(
        Integer,
        ForeignKey("prediction_logs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename = Column(String, nullable=False, index=True)
    original_prediction = Column(SQLEnum(PredictionLabel), nullable=False)  # type: ignore
    correct_label = Column(SQLEnum(FeedbackLabel), nullable=False)  # type: ignore
    labeled_by = Column(String, default="admin", nullable=False)
    added_to_training = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        index=True,
    )
