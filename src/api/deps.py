"""
FastAPI dependencies: API key checks, database sessions, repositories and services.

Two key scopes:

* verify_api_key (admin scope) accepts only the real key, which stays secret.
* verify_public_key (public scope) also accepts the demo key, which is public by design
  because it ships in the playground and the extension. It is used for inference and
  feedback only, and is rate limited separately.
"""

import secrets
from collections.abc import AsyncGenerator
from typing import Annotated

from arq import ArqRedis
from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from src.config import settings
from src.infra.database import AsyncSessionLocal
from src.logger import logger
from src.ml.explainer import ExplainabilityService
from src.repositories.feedback_repo import FeedbackRepository
from src.repositories.prediction_log_repo import PredictionLogRepository
from src.services.cache_service import CacheService
from src.services.feedback_service import FeedbackService
from src.services.inference_service import InferenceService
from src.services.retrain_service import RetrainService

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


# -- API keys ---------------------------------------------------------------
def _matches(candidate: str, *valid: str) -> bool:
    """
    Constant-time comparison against the accepted keys.

    Empty keys never match, so an unset DEMO_API_KEY switches demo access off
    instead of letting an empty header through.
    """
    return bool(candidate) and any(
        v and secrets.compare_digest(candidate.encode(), v.encode()) for v in valid
    )


def _reject() -> HTTPException:
    logger.warning("Unauthorized API access attempt")
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """Admin scope: the real key only."""
    if not _matches(api_key, settings.API_KEY.get_secret_value()):
        raise _reject()
    return api_key


def verify_public_key(api_key: str = Security(api_key_header)) -> str:
    """Public scope (inference, feedback): the real key or the public demo key."""
    if not _matches(
        api_key, settings.API_KEY.get_secret_value(), settings.DEMO_API_KEY
    ):
        raise _reject()
    return api_key


# -- Database -----------------------------------------------------------------
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    One asynchronous database session per request.

    Commits when the request succeeds, rolls back if it raises, and always closes.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


# -- Shared services from the application state ------------------------------
def get_inference_service(request: Request) -> InferenceService:
    """The InferenceService created once at startup."""
    return request.app.state.inference_service


def get_explainability_service(request: Request) -> ExplainabilityService:
    """The ExplainabilityService (GradCAM) created once at startup."""
    return request.app.state.explainability_service


def get_cache_service(request: Request) -> CacheService:
    """The CacheService created once at startup."""
    return request.app.state.cache_service


def get_arq_pool(request: Request) -> ArqRedis | None:
    """The ARQ Redis pool, or None when Redis is disabled (REDIS_URL=none)."""
    return request.app.state.arq_pool


# -- Repositories and per-request services ----------------------------------
async def get_feedback_repository(
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackRepository:
    """A FeedbackRepository bound to this request's session."""
    return FeedbackRepository(session)


def get_feedback_service(
    repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackService:
    return FeedbackService(repo)


async def get_prediction_log_repository(
    session: AsyncSession = Depends(get_db_session),
) -> PredictionLogRepository:
    """A PredictionLogRepository bound to this request's session."""
    return PredictionLogRepository(session)


async def get_retrain_service(
    session: AsyncSession = Depends(get_db_session),
    arq_pool: ArqRedis | None = Depends(get_arq_pool),
) -> RetrainService:
    return RetrainService(session, arq_pool)
