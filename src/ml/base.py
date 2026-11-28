from abc import ABC, abstractmethod
from pathlib import Path

from src.logger import logger


class BaseMLEngine(ABC):
    """
    Abstract base for ML inference engines loaded from disk
    It guarantees:
        * Path is validated before any loading happens, if fails fast at startup
        * _load() is always called exactly once during construction
        * subclasses expose their model_path for introspection and logging
    """

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        self._validate()
        self._load()
        logger.info(f"{self.__class__.__name__} loaded from {self.model_path}")

    def _validate(self) -> None:
        """
        Called before _load() to ensure we fail at startup with a clear message
        Raises FileNotFoundError if the model file is missing
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"{self.__class__.__name__}: model not found at {self.model_path}\n"
                "Run training first or check your volume mount in docker-compose.yml."
            )

    @abstractmethod
    def _load(self) -> None:
        """
        Load model weights and any runtime sessions
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_path={self.model_path})"
