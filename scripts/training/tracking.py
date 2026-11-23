"""
MLFlow when it is installed, a silent stand-in when it is not
"""

from contextlib import nullcontext

from src.logger import logger

try:
    import mlflow

    MLFLOW_AVAILABLE = True

except ImportError:
    MLFLOW_AVAILABLE = False


class _NoMlFlow:
    """
    Accepts the same calls as mlflow and does nothing.
    """

    @staticmethod
    def set_tracking_uri(*a, **kw):
        pass

    @staticmethod
    def set_experiment(*a, **kw):
        pass

    @staticmethod
    def log_params(*a, **kw):
        pass

    @staticmethod
    def log_metrics(*a, **kw):
        pass

    @staticmethod
    def log_artifact(*a, **kw):
        pass

    @staticmethod
    def start_run(*a, **kw):
        return nullcontext()


if not MLFLOW_AVAILABLE:
    mlflow = _NoMlFlow()  # type: ignore[assignment]
    logger.warning("mlflow not installed. training without experiment tracking")
