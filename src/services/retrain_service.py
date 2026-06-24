from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from arq import ArqRedis

from src.infra.feedbacks import FeedbackItem, PredictionLog
from src.core.enums import ErrorType
from src.config import settings
from src.logger import logger


class RetrainService:
    def __init__(self, session: AsyncSession, arq_pool: ArqRedis) -> None:
        self.session = session
        self.arq_pool = arq_pool

    LOOKBACK_DAYS: int = 7

    @staticmethod
    async def check_drift_and_trigger(
        session: AsyncSession,
        arq_pool: ArqRedis,
        lookback_days: int = LOOKBACK_DAYS,
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
        cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)

        total_with_feedback = await session.scalar(
            select(func.count(PredictionLog.id))
            .join(FeedbackItem, FeedbackItem.filename == PredictionLog.filename)
            .where(
                FeedbackItem.is_deleted.is_(False),
                PredictionLog.is_deleted.is_(False),
                PredictionLog.created_at >= cutoff,
            )
        )

        if not total_with_feedback:
            logger.info(
                f"Drift check: no predictions with feedback in the last {lookback_days} days."
            )
            return {
                "status": "no_data",
                "error_rate": 0.0,
                "lookback_days": lookback_days,
            }

        errors = await session.scalar(
            select(func.count())
            .select_from(FeedbackItem)
            .where(
                FeedbackItem.error_type.in_(
                    [ErrorType.FALSE_POSITIVE, ErrorType.FALSE_NEGATIVE]
                ),
                FeedbackItem.is_deleted.is_(False),
                FeedbackItem.created_at >= cutoff,
            )
        )

        errors = errors or 0
        total_with_feedback = total_with_feedback or 1

        error_rate = round(errors / total_with_feedback, 4)
        logger.info(
            f"Drift check ({lookback_days}d) — "
            f"error rate: {error_rate:.2%} "
            f"({errors} errors / {total_with_feedback} labelled predictions)"
        )

        # -- Trigger retraining if threshold exceeded ----------
        if error_rate >= settings.DRIFT_THRESHOLD:
            job = await arq_pool.enqueue_job("retrain_model")

            if not job:
                logger.error("Drift spotted but can't enqueued to retraining ")
                return {
                    "status": "error",
                    "message": "Failed to enqueue retraining job",
                }

            logger.warning(
                f"Drift threshold exceed ({error_rate:.2%}). "
                f"Retraining enqueued. Job ID: {job.job_id}"
            )
            return {
                "status": "retraining_enqueued",
                "job_id": job.job_id,
                "error_rate": error_rate,
                "lookback_days": lookback_days,
            }

        return {
            "status": "ok",
            "error_rate": error_rate,
            "lookback_days": lookback_days,
        }
