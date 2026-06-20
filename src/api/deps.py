from typing import AsyncGenerator, Annotated
from arq import ArqRedis
from fastapi import Request, Depends, HTTPException, Security, status
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import APIKeyHeader

from src.logger import logger
from src.infra.database import AsyncSessionLocal
from src.services.inference_service import InferenceService
from src.repositories.feedback_repo import FeedbackRepository
from src.services.feedback_service import FeedbackService
from src.services.explainability_service import ExplainabilityService
from src.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    if api_key != settings.API_KEY.get_secret_value():
        logger.warning("Unauthorized API access attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key


def get_inference_service(request: Request) -> InferenceService:
    """
    Dependency injection helper to retrieve the global InferenceService instance from the FastAPI application state
    """
    return request.app.state.inference_service


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Creates and yields a new asynchronous database session for each API request. The session is automatically closed after the request is completed or if an error occurs
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_arq_pool(request: Request) -> ArqRedis:
    return request.app.state.arq_pool


async def get_feedback_repository(
    session: AsyncSession = Depends(get_db_session),
) -> FeedbackRepository:
    """
    Dependency provider for the FeedbackRepository
    Injects the active database session into the repository instance

    Args:
        session (AsyncSession): The injected database session

    Returns:
        FeedbackRepository: An initialized instance of the repository
    """
    return FeedbackRepository(session)


SessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_feedback_service(
    repo: FeedbackRepository = Depends(get_feedback_repository),
) -> FeedbackService:
    return FeedbackService(repo)


def get_explainability_service(request: Request) -> ExplainabilityService:
    """
    Retrieves the ExplainabilityService from the app state
    """
    return request.app.state.explainability_service
