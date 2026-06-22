import asyncio
from urllib.parse import urlparse
from arq.connections import RedisSettings
from arq import Retry

from src.logger import logger
from src.infra.database import AsyncSessionLocal
from src.repositories.task_result_repo import TaskResultRepository
from src.config import settings


async def retrain_model(ctx: dict) -> dict:
    """
    Runs the full training pipeline in an isolated subprocess.
    Tracks start/end/error in the task_results table.
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


# ── Worker config ─────────────────────────────────────────────────────────────

parsed = urlparse(settings.REDIS_URL)


class WorkerSettings:
    functions = [retrain_model]
    redis_settings = RedisSettings(
        host=parsed.hostname or "redis",
        port=parsed.port or 6379,
        database=int(parsed.path.lstrip("/"))
        if parsed.path and parsed.path != "/"
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
