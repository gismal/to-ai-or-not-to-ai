import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
import contextvars

from src.config import settings

request_id_ctx: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default="N/A"
)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log: dict = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "request_id": request_id_ctx.get(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            log["exception"] = self.formatException(record.exc_info)
        return json.dumps(log)


def _build_logger() -> logging.Logger:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    handlers: list[logging.Handler] = [handler]

    if settings.DEBUG:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log")
        file_handler.setFormatter(JSONFormatter())
        handlers.append(file_handler)

    log = logging.getLogger("mlops_detector")
    log.setLevel(logging.INFO)
    log.handlers = handlers
    log.propagate = False
    return log


logger = _build_logger()
