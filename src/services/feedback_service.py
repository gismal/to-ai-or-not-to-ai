from fastapi import BackgroundTasks
from src.core.database import AsyncSessionLocal
from src.repositories.feedback_repo import FeedbackRepository
from src.schemas.feedback import FeedbackCreateRequest
from src.logger import logger

class FeedbackService:
    """
    Handles all bussiness logic for user feedback operations
    """
    
    @staticmethod
    async def _save_feedback_task(payload: FeedbackCreateRequest) -> None:
        """
        Independent background database
        """
        async with AsyncSessionLocal() as session:
            repo = FeedbackRepository(session)
            try:
                await repo.create_feedback(
                    filename = payload.filename,
                    model_prediction = payload.model_prediction,
                    confidence = payload.confidence,
                    user_correction = payload.user_correction,
                    client_source = payload.client_source
                )
            except Exception as e:
                logger.error(f"Feedback background task failed: {str(e)}")
                
    def register_feedback(self, payload: FeedbackCreateRequest, background_tasks: BackgroundTasks) -> None:
        """
        Adds the tasks to the queue
       """
        background_tasks.add_task(self._save_feedback_task, payload)