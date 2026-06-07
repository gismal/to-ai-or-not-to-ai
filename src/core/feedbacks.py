import enum
from sqlalchemy import Enum as SQLEnum, Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime, timezone 
from src.core.database import Base

class FeedbackLabel(str, enum.Enum):
    """
    Enum classes for the database to define feedbacks
    """
    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"
    
class ErrorType(str, enum.Enum):
    FALSE_POSITIVE = "FALSE_POSITIVE" # Model marks as AI but image is REAL
    FALSE_NEGATIVE = "FALSE_NEGATIVE"  # Model marks as REAL but image is AI
    UNCERTAIN_FAIL = "UNCERTAIN_FAIL"  # Model is indecisive

class FeedbackItem(Base):
    """
    Bridge between Python instances and SQL tables
    """
    __tablename__ = "feedback_logs"
    
    id = Column(Integer, primary_key = True, index = True)
    filename = Column(String, index = True)
    model_prediction = Column(String, nullable = False)
    confidence = Column(Float)
    user_correction = Column(SQLEnum(FeedbackLabel), nullable = False)
    # for future MLOps analysis 
    error_type = Column(SQLEnum(ErrorType), nullable = False)
    client_source = Column(String, default = "API_v1")
    # soft deletion to not to delete something pyhsically and entirely 
    is_deleted = Column(Boolean, default = False) 
    created_at = Column(DateTime, default = lambda: datetime.now(timezone.utc))
    
    