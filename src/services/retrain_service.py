from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from arq import ArqRedis

from src.infra.feedbacks import FeedbackItem
from src.core.enums import ErrorType
from src.config import settings
from src.logger import logger  

class RetrainService:
    
    @staticmethod
    async def check_drift_and_trigger(
        session: AsyncSession,
        arq_pool: ArqRedis
    ) -> dict:
        """
        Calculates recent error rate
        Enqueues retraining via ARQ if drift exceeds threshold
        """
        total = await session.scalar(
            select(func.count()).select_from(FeedbackItem)
            .where(FeedbackItem.is_deleted.is_(False))
        )
        
        if not total:
            return {"status": "no_data", "error_rate": 0.0}
        
        errors = await session.scalar(
            select(func.count()).select_from(FeedbackItem)
            .where(FeedbackItem.error_type.in_([
                ErrorType.FALSE_POSITIVE,
                ErrorType.FALSE_NEGATIVE
            ]),
                   FeedbackItem.is_deleted.is_(False)
        ))
                    
        error_rate = round(errors / total, 4)
        logger.info(f"Drift Check. Error Rate: {error_rate:.2%} ({errors}/{total})")

        if error_rate >= settings.DRIFT_THRESHOLD:
            job = await arq_pool.enqueue_job('retrain_model')

            logger.warning(
                f"Drift threshold exceed ({error_rate:.2%}). "
                f"Retraining enqueued. Job ID: {job.job_id}"
            )
            return {
                "status": "retraining_enqueued",
                "job_id": job.job_id
            }

        return {
            "status": "ok",
            "error_rate": error_rate
        }