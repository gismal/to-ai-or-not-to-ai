from celery import Celery
from src.config import settings

celery_app = Celery(
    "mlops_worker",
    broker = settings.REDIS_URL,
    backend = settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer = "json",
    result_expires = 3600,
    worker_prefetch_multiplier = 1
)