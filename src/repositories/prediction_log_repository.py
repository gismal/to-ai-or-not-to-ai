from abc import ABC, abstractmethod

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import PredictionLabel
from src.core.exceptions import DatabaseError
from src.infra.prediction_log import PredictionLog
from src.logger import logger


class AbstractPredictionLogRepository(ABC):
    @abstractmethod
    async def create_log(
        self,
        filename: str,
        confidence: float,
        predicted_label: PredictionLabel,
        processing_time_ms: float,
        client_source: str = "API_v1",
    ) -> PredictionLog:
        pass


class PredictionLogRepository(AbstractPredictionLogRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_log(
        self,
        filename: str,
        confidence: float,
        predicted_label: PredictionLabel,
        processing_time_ms: float,
        client_source: str = "API_v1",
    ) -> PredictionLog:
        """
        Inserts one prediction record
        """
        log = PredictionLog(
            filename=filename,
            confidence=confidence,
            predicted_label=predicted_label,
            processing_time_ms=processing_time_ms,
            client_source=client_source,
        )

        try:
            self.session.add(log)
            await self.session.flush()
            await self.session.refresh(log)
            logger.debug(f"Prediction logged: {filename} -> {predicted_label.value}")
            return log

        except SQLAlchemyError as e:
            logger.error(f"Failed to log prediction for {filename}: {e}", exc_info=True)
            raise DatabaseError
