import enum
from sqlalchemy import Enum as SQLEnum, Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime, timezone 

from src.infra.database import Base
from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel

class FeedbackItem(Base):
    """
    Bridge between Python instances and SQL tables
    """
    __tablename__ = "feedback_logs"
    
    id = Column(Integer, primary_key = True, index = True)
    filename = Column(String, index = True, nullable = False)
    model_prediction = Column(SQLEnum(PredictionLabel), nullable = False)
    confidence = Column(Float, nullable = False)
    user_correction = Column(SQLEnum(FeedbackLabel), nullable = False)
    # for future MLOps analysis 
    error_type = Column(SQLEnum(ErrorType), nullable = False)
    client_source = Column(String, default = "API_v1")
    # soft deletion to not to delete something pyhsically and entirely 
    is_deleted = Column(Boolean, default = False) 
    created_at = Column(DateTime, default = lambda: datetime.now(timezone.utc))
    
    