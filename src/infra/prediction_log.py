from datetime import datetime, timezone
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Enum as SQLEnum,
)
from src.infra.database import Base
from src.core.enums import PredictionLabel


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
