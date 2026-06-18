from typing import AsyncGenerator, Annotated

from fastapi import Request, Depends, HTTPException, Security, status
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.security import APIKeyHeader

from src.infra.database import AsyncSessionLocal, get_db_session
from src.services.inference_service import InferenceService
from src.repositories.feedback_repo import FeedbackRepository 
from src.services.feedback_service import FeedbackService
from src.config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return api_key

def get_inference_service(request: Request) -> InferenceService:
    """
    Dependency injection helper to retrieve the global InferenceService instance from the FastAPI application state
    """
    return request.app.state.inference_service
    
SessionDep = Annotated[AsyncSession, Depends(get_db_session)]
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Creates and yields a new asynchronous database session for each API request. The session is automatically closed after the request is completed or if an error occurs
    """
    session = AsyncSessionLocal()
    try:
        yield session
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
        
async def get_feedback_repository(
    session: AsyncSession = Depends(get_db_session)
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

def get_feedback_service() -> FeedbackService:
    return FeedbackService()