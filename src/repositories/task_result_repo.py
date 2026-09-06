"""
Creates and updates task result records
"""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.exceptions import DatabaseError
from src.infra.task_results import TaskResult, TaskStatus
from src.logger import logger


class TaskResultRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, task_name: str) -> TaskResult:
        record = TaskResult(task_name=task_name, status=TaskStatus.RUNNING)
        try:
            # emtpy object that will be added is created. status is RUNNING
            self.session.add(record)
            await self.session.flush()
            await self.session.refresh(record)
            await self.session.commit()
            return record

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(f"Failed to create task result: {e}", exc_info=True)
            raise DatabaseError

    async def mark_success(self, record_id: int) -> None:
        """Updates the record to SUCCESS with a completion timestamp"""
        await self._update(record_id, TaskStatus.SUCCESS, error=None)

    async def mark_failed(self, record_id: int, error: str | None) -> None:
        """
        Updates the record to FAILED with error message
        """
        await self._update(record_id, TaskStatus.FAILED, error=error)

    async def _update(
        self, record_id: int, status: TaskStatus, error: str | None
    ) -> None:
        try:
            result = await self.session.execute(
                select(TaskResult).where(TaskResult.id == record_id)
            )

            record = result.scalar_one_or_none()
            if not record:
                logger.warning(f"TaskResult {record_id} not found for update")
                return
            record.status = status  # type: ignore
            record.completed_at = datetime.now(timezone.utc)  # type: ignore
            record.error = error  # type: ignore
            await self.session.commit()

        except SQLAlchemyError as e:
            await self.session.rollback()
            logger.error(
                f"Failed to update task result {record_id}: {e}", exc_info=True
            )
