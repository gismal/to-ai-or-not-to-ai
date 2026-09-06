import os

import pytest

os.environ["DATABASE_URL"] = "sqlite+aiosqlite:///:memory:"
os.environ["API_KEY"] = "test_gizli_anahtar_123"
os.environ["MODEL_THRESHOLD"] = "0.75"

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.core.enums import FeedbackLabel, PredictionLabel
from src.infra.database import Base
from src.repositories.feedback_repo import FeedbackRepository
from src.repositories.prediction_log_repository import (
    PredictionLogRepository,
)
from src.schemas.feedback import FeedbackCreateRequest

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestingSessionLocal = async_sessionmaker(bind=test_engine, expire_on_commit=False)


@pytest.fixture()
async def setup_db():
    """
    Creates the tables
    """
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
async def db_session(setup_db):
    """
    rollback after each test to clean up
    """
    async with TestingSessionLocal() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def feedback_repo(db_session):
    """
    Ready-to-use repo that directly call the tests
    """
    return FeedbackRepository(db_session)


@pytest.fixture
def mock_feedback_data():
    """
    Mock data for tests
    """
    return FeedbackCreateRequest(
        filename="db_test_image.png",
        model_prediction=PredictionLabel.AI_GENERATED,
        confidence=0.99,
        user_correction=FeedbackLabel.REAL,
        client_source="API_v1",
    )


@pytest.fixture
async def prediction_log_repo(db_session):
    return PredictionLogRepository(db_session)
