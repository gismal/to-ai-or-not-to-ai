import asyncio
from urllib.parse import urlparse
from arq.connections import RedisSettings
from arq import Retry

from src.logger import logger
from src.config import settings

async def retrain_model(ctx):
    """
    Runs the full training pipeline in an isolated subprocess
    """
    job_id = ctx.get("job_id")
    logger.info(f"{job_id} Retraining task started")
    
    try: 
        process = await asyncio.create_subprocess_exec(
            "python", "scripts/train.py",
            stdout = asyncio.subprocess.PIPE,
            stderr = asyncio.subprocess.PIPE
        )
        # wait 1h 
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout = 3600)
        
        if process.returncode != 0:
            raise RuntimeError(stderr.decode().strip())
        
        logger.info("Retraining completed successfully")
        return {"status": "success"}
    
    except Exception as exc:
        logger.error(f"Retraining failed: {exc}", exc_info= True)
        raise Retry(defer = 60) from exc
    
parsed_url = urlparse(settings.REDIS_URL)


class WorkerSettings:
    functions = [retrain_model]
    redis_settings = RedisSettings(
        host=parsed_url.hostname or 'localhost',
        port=parsed_url.port or 6379,
        database=int(parsed_url.path.lstrip('/')) if parsed_url.path and parsed_url.path != '/' else 0,
    )
    max_tries = 3      
    job_timeout = 4200
    
    async def on_startup(ctx):
        logger.info("ARQ worker started")
        
    async def on_shutdown(ctx):
        logger.info("ARQ worker shutting down")
        
    