"""
Drift detection for the deployed model.

RetrainService measures how often users report that the model was wrong and queues a
retraining job (ARQ, in Redis) when that rate reaches DRIFT_THRESHOLD.

What counts as evidence:

* Only feedback about the model serving now. The window starts at the later of
  `lookback_days` ago and the model's training time, so complaints about a replaced
  model cannot trigger another retrain.
* Only decided answers: the model was right (CORRECT) or wrong (FALSE_POSITIVE,
  FALSE_NEGATIVE). "Not sure" answers and the model's own abstentions carry no
  accuracy signal; the abstention rate is tracked on the dashboard instead.
* Nothing is triggered below DRIFT_MIN_SAMPLES decided answers.

Feedback is voluntary, so the rate measures dissatisfaction among people who vote,
not true accuracy. It is meant to detect change over time.
"""

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, TypedDict

from arq import ArqRedis
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.core.enums import ErrorType
from src.infra.feedback_item import FeedbackItem
from src.logger import logger

HARD_ERRORS = (ErrorType.FALSE_POSITIVE, ErrorType.FALSE_NEGATIVE)
DECIDED = (ErrorType.CORRECT, *HARD_ERRORS)
RETRAIN_JOB = "retrain_model"  # fixed job id, so repeated checks never stack jobs
METADATA_PATH = Path("models/metadata.json")

Status = Literal[
    "no_data",
    "insufficient_data",
    "ok",
    "retraining_enqueued",
    "already_queued",
    "error",
]


class DriftReport(TypedDict, total=False):
    """
    The result of one drift check.

    Every report has status, error_rate, lookback_days and since. The other keys
    appear only with the status they explain: samples and required with
    insufficient_data, job_id with retraining_enqueued, message with error.
    """

    status: Status
    error_rate: float  # hard errors / decided answers, 0.0 when not computed
    lookback_days: int  # the window length that was requested
    since: str  # ISO start of the window actually used
    samples: int  # decided answers found
    required: int  # decided answers needed (DRIFT_MIN_SAMPLES)
    job_id: str  # id of the queued retraining job
    message: str  # why the check could not act


def _model_trained_at() -> datetime | None:
    """
    When the model currently serving traffic was trained, read from metadata.json.

    The result is always timezone-aware: a naive timestamp is treated as UTC.
    A missing or malformed file returns None, so the drift check falls back to the
    plain lookback window instead of failing.
    """
    try:
        raw = json.loads(METADATA_PATH.read_text())["trained_at"]
        trained = datetime.fromisoformat(raw)
    except (OSError, KeyError, TypeError, ValueError):
        return None
    return trained if trained.tzinfo else trained.replace(tzinfo=UTC)


class RetrainService:
    def __init__(self, session: AsyncSession, arq_pool: ArqRedis | None) -> None:
        self.session = session
        self.arq_pool = arq_pool  # None when Redis is disabled (REDIS_URL=none)

    async def check_drift_and_trigger(self, lookback_days: int = 7) -> DriftReport:
        """
        Measure the recent error rate and queue retraining if it is too high.

        Args:
            lookback_days: How many days of feedback to consider at most.

        Returns:
            A DriftReport whose status is one of: no_data, insufficient_data, ok,
            retraining_enqueued, already_queued (a job with the fixed id is queued,
            running or recently finished) or error (no job queue available).
        """
        cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
        trained_at = _model_trained_at()
        if trained_at and trained_at > cutoff:
            cutoff = trained_at

        def report(
            status: Status,
            error_rate: float = 0.0,
            *,
            samples: int | None = None,
            required: int | None = None,
            job_id: str | None = None,
            message: str | None = None,
        ) -> DriftReport:
            """Build a report with the common keys plus whichever extras apply."""
            out: DriftReport = {
                "status": status,
                "error_rate": error_rate,
                "lookback_days": lookback_days,
                "since": cutoff.isoformat(),
            }
            if samples is not None:
                out["samples"] = samples
            if required is not None:
                out["required"] = required
            if job_id is not None:
                out["job_id"] = job_id
            if message is not None:
                out["message"] = message
            return out

        decided = (
            FeedbackItem.is_deleted.is_(False),
            FeedbackItem.created_at >= cutoff,
            FeedbackItem.error_type.in_(DECIDED),
        )
        # One query for both numbers, so they always describe the same rows.
        total, errors = (
            await self.session.execute(
                select(
                    func.count(FeedbackItem.id),
                    func.count(case((FeedbackItem.error_type.in_(HARD_ERRORS), 1))),
                ).where(*decided)
            )
        ).one()

        if total == 0:
            logger.info(f"Drift check: no decided feedback since {cutoff.isoformat()}.")
            return report("no_data")
        if total < settings.DRIFT_MIN_SAMPLES:
            return report(
                "insufficient_data", samples=total, required=settings.DRIFT_MIN_SAMPLES
            )

        error_rate = round(errors / total, 4)
        logger.info(
            f"Drift check: {error_rate:.2%} ({errors} errors / {total} decided answers)"
        )
        if error_rate < settings.DRIFT_THRESHOLD:
            return report("ok", error_rate)

        if self.arq_pool is None:
            logger.error(
                "Drift detected but the job queue is unavailable (Redis disabled)."
            )
            return report("error", error_rate, message="Job queue unavailable")

        job = await self.arq_pool.enqueue_job(RETRAIN_JOB, _job_id=RETRAIN_JOB)
        if job is None:
            logger.info(
                "Drift detected, but a retraining job is already queued or recently finished."
            )
            return report("already_queued", error_rate)

        logger.warning(
            f"Drift threshold exceeded ({error_rate:.2%}). "
            f"Retraining enqueued, job {job.job_id}."
        )
        return report("retraining_enqueued", error_rate, job_id=job.job_id)
