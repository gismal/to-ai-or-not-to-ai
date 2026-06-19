from src.repositories.feedback_repo import FeedbackRepository
from src.schemas.feedback import FeedbackCreateRequest
from src.logger import logger

class FeedbackService:
    """
    Handles all bussiness logic for user feedback operations
    """
    def __init__(self, repo: FeedbackRepository) -> None:
        self.repo = repo
    
    async def register_feedback(self, payload: FeedbackCreateRequest) -> None:
        """
        Independent background database
        """
        try:
            await self.repo.create_feedback(
                filename = payload.filename,
                model_prediction = payload.model_prediction,
                confidence = payload.confidence,
                user_correction = payload.user_correction,
                client_source = payload.client_source
            )
        except Exception as e:
            logger.error(f"Feedback background task failed: {str(e)}", exc_info = True)
            raise
     