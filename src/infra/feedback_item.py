from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    UniqueConstraint,
    Enum as SQLEnum,
)
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
