from sqlalchemy import (
    Enum as SQLEnum,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    Boolean,
    UniqueConstraint,
)
from datetime import datetime, timezone

from src.infra.database import Base
from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel


class FeedbackItem(Base):
    """
    Bridge between Python instances and SQL tables
    """

    __tablename__ = "feedback_logs"
    __table_args__ = (
        UniqueConstraint(
            "filename", "client_source", name="uq_feedback_filename_source"
        ),
    )

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    model_prediction = Column(SQLEnum(PredictionLabel), nullable=False)  # type: ignore
    confidence = Column(Float, nullable=False)
    user_correction = Column(SQLEnum(FeedbackLabel), nullable=False)  # type: ignore
    # for future MLOps analysis
    error_type = Column(SQLEnum(ErrorType), nullable=False)  # type: ignore
    client_source = Column(String, default="API_v1")
    # soft deletion to not to delete something pyhsically and entirely
    is_deleted = Column(Boolean, default=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )


class PredictionLog(Base):
    """
    Stores every prediction made by the model for complete MLOps obervability
    """

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True, nullable=False)
    confidence = Column(Float, nullable=False)
    predicted_label = Column(SQLEnum(PredictionLabel), nullable=False)  # type: ignore
    processing_time_ms = Column(Float, nullable=True)
    client_source = Column(String, default="API_v1")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_deleted = Column(Boolean, default=False)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
