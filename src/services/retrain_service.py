from datetime import datetime, timedelta, timezone

from arq import ArqRedis
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.enums import ErrorType
from src.infra.feedback_item import FeedbackItem
from src.logger import logger

HARD_ERRORS = (ErrorType.FALSE_POSITIVE, ErrorType.FALSE_NEGATIVE)
RETRAIN_JOB = "retrain_model"


class RetrainService:
    def __init__(self, session: AsyncSession, arq_pool: ArqRedis) -> None:
        self.session = session
        self.arq_pool = arq_pool

    async def check_drift_and_trigger(
        self,
        lookback_days: int = 7,
    ) -> dict:
        """
        Calculates the recent error rate using both tables, then enqueues
        retraining if it exceeds the configured threshold.

        Args:
            session:       Active DB session (injected from get_db_session)
            arq_pool:      ARQ Redis pool for enqueuing the retrain job
            lookback_days: How many days of data to consider

        Returns:
            dict with status, error_rate, and optional job_id
        """
        cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(
            days=lookback_days
        )

        rated = (
            FeedbackItem.is_deleted.is_(False),
            FeedbackItem.created_at >= cutoff,
            FeedbackItem.error_type != ErrorType.UNKNOWN,
        )
        window = {"lookback_days": lookback_days}

        total = (
            await self.session.scalar(select(func.count(FeedbackItem.id)).where(*rated))
            or 0
        )

        if total == 0:
            logger.info(
                f"Drift check: no rated feedback in the last {lookback_days} days."
            )
            return {"status": "no_data", "error_rate": 0.0, **window}
        if total < settings.DRIFT_MIN_SAMPLES:
            return {
                "status": "insufficient_data",
                "samples": total,
                "required": settings.DRIFT_MIN_SAMPLES,
                "error_rate": 0.0,
                **window,
            }

        errors = (
            await self.session.scalar(
                select(func.count(FeedbackItem.id)).where(
                    *rated, FeedbackItem.error_type.in_(HARD_ERRORS)
                )
            )
            or 0
        )
        error_rate = round(errors / total, 4)
        logger.info(
            f"Drift check ({lookback_days}d): {error_rate:.2%} ({errors} errors / {total} rated answers)"
        )

        if error_rate < settings.DRIFT_THRESHOLD:
            return {"status": "ok", "error_rate": error_rate, **window}

        if self.arq_pool is None:
            logger.error(
                "Drift detected but the job queue is unavailable (Redis disabled)."
            )
            return {
                "status": "error",
                "message": "Job queue unavailable",
                "error_rate": error_rate,
                **window,
            }

        job = await self.arq_pool.enqueue_job(RETRAIN_JOB, _job_id=RETRAIN_JOB)
        if job is None:
            logger.info(
                "Drift detected, but a retraining job is already queued or recently finished."
            )
            return {"status": "already_queued", "error_rate": error_rate, **window}

        logger.warning(
            f"Drift threshold exceeded ({error_rate:.2%}). Retraining enqueued, job {job.job_id}."
        )
        return {
            "status": "retraining_enqueued",
            "job_id": job.job_id,
            "error_rate": error_rate,
            **window,
        }
