"""
ORM model for tracking ARQ background task outcome
Saves the job history for future Postgres queries
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy import Enum as SQLEnum

from src.infra.database import Base


class TaskStatus(str, enum.Enum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class TaskResult(Base):
    __tablename__ = "task_results"

    id = Column(Integer, primary_key=True, index=True)
    task_name = Column(String, nullable=False, index=True)
    status = Column(SQLEnum(TaskStatus), nullable=False, default=TaskStatus.RUNNING)  # type: ignore
    error = Column(Text, nullable=True)  # populated on failure
    started_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), index=True
    )
    completed_at = Column(DateTime, nullable=True)  # populated on finish
