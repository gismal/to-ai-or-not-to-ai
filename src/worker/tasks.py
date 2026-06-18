import subprocess  # isolates the training process entirely from the workers memo space
from src.worker.celery_app import celery_app
from src.logger import logger

@celery_app.task(
    bind = True,
    max_retries = 2,
    default_retry_delay = 60,   # wait 60s before retry
    name= "worker.retrain_model"
)

def retrain_model(self):
    """
    Runs the full training pipeline in an isolated subprocess
    """
    logger.info("Retraining task started")
    
    try: 
        result = subprocess.run(
            ["python", "scripts/train.py"],
            capture_output = True,
            text = True,
            timeout = 3600  # 1 hour max
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)
        
        logger.info("Retraining completed successfully")
        return {"status": "success"}
    
    except Exception as exc:
        logger.error(f"Retraining failed: {exc}", exc_info= True)
        raise self.retry(exc = exc)