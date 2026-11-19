"""
Feedback votes: every real client is accepted, and one filename can be voted on many times.

Regression tests for two bugs that silently dropped feedback:
- the schema only accepted "API_v1" and "WEB_UI", so the playground ("playground_v2")
  and the extension ("extension_v1") got a 422 on every vote;
- a unique constraint on (filename, client_source) rejected every later vote on a
  common name like image.jpg.
"""

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel
from src.infra.database import Base
from src.infra.feedback_item import FeedbackItem
from src.repositories.feedback_repo import FeedbackRepository
from src.schemas.feedback import FeedbackCreateRequest

CLIENTS = ["API_v1", "WEB_UI", "playground_v2", "extension_v1"]


@pytest.fixture
async def session() -> AsyncIterator[AsyncSession]:
    """A fresh in-memory database built from the models, one per test."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async with async_sessionmaker(engine, expire_on_commit=False)() as db:
        yield db
    await engine.dispose()


async def _vote(
    repo: FeedbackRepository,
    correction: FeedbackLabel,
    filename: str = "image.jpg",
    client: str = "playground_v2",
) -> FeedbackItem:
    """One vote on an image the model called AI-generated."""
    return await repo.create_feedback(
        filename=filename,
        model_prediction=PredictionLabel.AI_GENERATED,
        confidence=0.9,
        user_correction=correction,
        error_type=(
            ErrorType.FALSE_POSITIVE
            if correction == FeedbackLabel.REAL
            else ErrorType.CORRECT
        ),
        client_source=client,
    )


# -- Schema -------------------------------------------------------------------


def _request(client: str) -> FeedbackCreateRequest:
    return FeedbackCreateRequest(
        filename="image.jpg",
        model_prediction=PredictionLabel.AI_GENERATED,
        confidence=0.9,
        user_correction=FeedbackLabel.REAL,
        client_source=client,  # type: ignore[arg-type]
    )


@pytest.mark.parametrize("client", CLIENTS)
def test_every_real_client_is_accepted(client: str) -> None:
    assert _request(client).client_source == client


def test_an_unknown_client_is_rejected() -> None:
    with pytest.raises(ValidationError):
        _request("someone_else")


# -- Repository ---------------------------------------------------------------


async def test_two_votes_on_the_same_filename_are_both_saved(
    session: AsyncSession,
) -> None:
    repo = FeedbackRepository(session)
    first = await _vote(repo, FeedbackLabel.REAL)
    second = await _vote(repo, FeedbackLabel.REAL)
    assert first.id != second.id


@pytest.mark.parametrize("client", CLIENTS)
async def test_repeat_votes_are_saved_for_every_client(
    session: AsyncSession, client: str
) -> None:
    repo = FeedbackRepository(session)
    await _vote(repo, FeedbackLabel.REAL, client=client)
    await _vote(repo, FeedbackLabel.AI_GENERATED, client=client)
    assert len(await repo.get_recent_errors()) == 2


async def test_the_latest_vote_is_returned_for_a_filename(
    session: AsyncSession,
) -> None:
    repo = FeedbackRepository(session)
    older = await _vote(repo, FeedbackLabel.REAL)
    older.created_at = datetime(2020, 1, 1, tzinfo=UTC)  # type: ignore[assignment]
    await session.flush()
    newer = await _vote(repo, FeedbackLabel.AI_GENERATED)

    latest = await repo.get_feedback_by_filename("image.jpg")

    assert latest is not None
    assert latest.id == newer.id


async def test_no_vote_returns_none(session: AsyncSession) -> None:
    repo = FeedbackRepository(session)
    assert await repo.get_feedback_by_filename("never-voted.jpg") is None
