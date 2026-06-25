import asyncio
from datetime import datetime, timezone
from urllib.parse import urlparse
from arq.connections import RedisSettings, create_pool
from arq import Retry, cron

from src.logger import logger
from src.infra.database import AsyncSessionLocal
from src.repositories.task_result_repo import TaskResultRepository
from src.config import settings


# -- Model Retraining --------------------------
async def retrain_model(ctx: dict) -> dict:
    """
    Runs the full training pipeline in an isolated subprocess.
    Writes a TaskResult record for observability
    """
    job_id = ctx.get("job_id", "unknown")
    logger.info(f"[{job_id}] Retraining task started.")

    async with AsyncSessionLocal() as session:
        repo = TaskResultRepository(session)
        record = await repo.create(task_name="retrain_model")

    try:
        process = await asyncio.create_subprocess_exec(
            "python",
            "scripts/train.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=3600)

        if process.returncode != 0:
            raise RuntimeError(stderr.decode().strip())

        async with AsyncSessionLocal() as session:
            await TaskResultRepository(session).mark_success(int(record.id))  # type: ignore

        logger.info(f"[{job_id}] Retraining completed successfully.")
        return {"status": "success", "task_result_id": record.id}

    except Exception as exc:
        async with AsyncSessionLocal() as session:
            await TaskResultRepository(session).mark_failed(int(record.id), str(exc))  # type: ignore

        logger.error(f"[{job_id}] Retraining failed: {exc}", exc_info=True)
        raise Retry(defer=60) from exc


# -- Scheduled Drift Check ----------------------------------------
async def check_drift_scheduled(ctx: dict) -> dict:
    """
    It will run every night at 02:00 UTC and check error rate
    """
    logger.info(
        f"Scheduled drift check starting at{datetime.now(timezone.utc).isoformat()}"
    )
    # lazy import to avoid circular dependency
    from src.services.retrain_service import RetrainService

    async with AsyncSessionLocal() as session:
        # short-lived ARQ pool special for this task
        async with await create_pool(WorkerSettings.redis_settings) as pool:
            service = RetrainService(session=session, arq_pool=pool)
            result = await service.check_drift_and_trigger()

    logger.info(f"Scheduled drift check complete: {result}")
    return result


# -- Worker config -----------------------------------------------------

_parsed = urlparse(settings.REDIS_URL)


class WorkerSettings:
    functions = [retrain_model]

    cron_jobs = [
        cron(
            check_drift_scheduled,
            hour=2,
            minute=0,
            timeout=120,
        )
    ]

    redis_settings = RedisSettings(
        host=_parsed.hostname or "redis",
        port=_parsed.port or 6379,
        database=int(_parsed.path.lstrip("/"))
        if _parsed.path and _parsed.path != "/"
        else 0,
    )
    max_tries = 3
    job_timeout = 4200

    @staticmethod
    async def on_startup(ctx: dict) -> None:
        logger.info("ARQ worker started.")

    @staticmethod
    async def on_shutdown(ctx: dict) -> None:
        logger.info("ARQ worker shutting down.")
