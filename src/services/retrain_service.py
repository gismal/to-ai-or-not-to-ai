from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from src.core.feedbacks import FeedbackItem, ErrorType
from src.worker.tasks import retrain_model
from src.config import settings
from src.logger import logger  

class RetrainService:
    
    @staticmethod
    async def check_drift_and_trigger(session: AsyncASession) -> dict:
        """
        Calculates recent error rate
        Enqueues retraining if drift exceeds threshold
        """
        total = await session.scalar(
            select(func.count().select_from(FeedbackItem))
            .where(FeedbackItem.is_deleted == False)
        )
        
        if not total:
            return {"status": "no_data", "error_rate": 0.0}
        
        errors = await session.scalar(
            select(func.count()).select_from(FeedbackItem)
            .where(FeedbackItem).error_type.in_([
                ErrorType.FALSE_POSITIVE,
                ErrorType.FALSE_NEGATIVE
            ])
        )
        
        error_rate = errors / total
        logger.info(f"Drift Check
                    /n Error Rate: {error_rate:.2%} ({errors}/{total})
                    ")
        
        if error_rate >= settings.DRIFT_THRESHOLD:
            job = retrain_model.delay()  # enqueue to Redis, return immediately
            logger.warning(
                f"Drift threshold exceed ({error_rate:.2%})"
                f"Retraining enqueued. Job ID: {job.id}"
            )
            return {"status": "retraining_enqueued", "job_id": job.id, "error_rate": error_rate}

        return {"status": "ok", "error_rate": error_rate}