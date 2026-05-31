from abc import ABC, abstractmethod
from typing import Sequence
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from src.models.feedback import FeedbackItem, FeedbackLabel, ErrorType
from src.logger import logger
from src.core.exceptions import DatabaseError

# Dependency Inversion
class AbstractFeedbackRepository(ABC):
    """
    Creates abstract interface to ensure loose coupling in case future changes
    """
    
    @abstractmethod 
    async def create_feedback(self, filename: str, model_prediction: str, confidence: float, user_correction: FeedbackLabel, client_source: str = "API_v1") -> FeedbackItem:
        pass
    
    @abstractmethod
    async def get_recent_errors(self, limit: int = 100, offset: int = 0) -> Sequence[FeedbackItem]:
        pass
    
    @abstractmethod
    async def soft_delete(self, record_id: int) -> bool:
        pass
    
class FeedbackRepository(AbstractFeedbackRepository):
    def __init__(self, session: AsyncSession):
        self.session = session
            
    def _determine_error_type(self, model_pred: str, user_corr: FeedbackLabel) -> ErrorType:
        if "UNCERTAIN" in model_pred:
        # str for model_pred for the flexibility but can be Enum in the future if it's necesarry
            return ErrorType.UNCERTAIN_FAIL
        elif model_pred == "AI_GENERATED" and user_corr == FeedbackLabel.REAL:
            return ErrorType.FALSE_POSITIVE
        elif model_pred == "REAL" and user_corr == FeedbackLabel.AI_GENERATED:
            return ErrorType.FALSE_NEGATIVE
        return ErrorType.UNCERTAIN_FAIL
        
    async def create_feedback(
        self, filename: str, model_prediction: str, confidence: float, user_correction: FeedbackLabel, client_source: str = "API_v1") -> FeedbackItem:
            
        calculated_error = self._determine_error_type(model_prediction, user_correction)
        db_item = FeedbackItem(
            filename = filename, 
            model_prediction = model_prediction,
            confidence = confidence,
            user_correction = user_correction,
            error_type = calculated_error,
            client_source = client_source
            )            
            
        # Transaction managment and error handling
        try:
            self.session.add(db_item)
            await self.session.commit()
            await self.session.refresh(db_item)
            logger.info(f"Feedback saved: {filename} (ID: {db_item.id})")
            return db_item
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Error during saving the feedback: {str(e)}")
            raise DatabaseError("Feedback could not be saved to the database.")
            
    async def get_recent_errors(self, limit: int = 100, offset: int = 0) -> Sequence[FeedbackItem]:
        """
        Pagination and Reading method
        """
        try:
        # brings only the non-deleted items
            stmt = select(FeedbackItem).where(FeedbackItem.is_deleted == False).order_by(FeedbackItem.created_at.desc()).limit(limit).offset(offset)
            result = await self.session.execute(stmt)
            return result.scalars().all()
        except SQLAlchemyError as e:
            logger.error(f"Data reading error: {str(e)}")
            raise DatabaseError("Failed to fetch recent errors.")
            
    async def soft_delete(self, record_id: int) -> bool:
        """
        Method for soft deleting.
        """
            
        try:
            stmt = select(FeedbackItem).where(FeedbackItem.id == record_id)
            result = await self.session.execute(stmt)
            db_item = result.scalar_one_or_none()
                
            if db_item:
                db_item.is_deleted = True
                await self.session.commit()
                logger.info(f"Record has been soft deleted. ID: {record_id}")
                return True
            return False
        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Soft delete error. ID: {record_id}: {str(e)}")
            raise DatabaseError("Soft delete fails")
    