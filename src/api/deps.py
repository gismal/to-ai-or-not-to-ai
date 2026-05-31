from typing import AsyncGenerator, Annotated

from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database import AsyncSessionLocal
from src.services.inference_service import InferenceService
from src.repositories.feedback_repo import FeedbackRepository 

def get_inference_service(request: Request) -> InferenceService:
    """
    Dependency injection helper to retrieve the gloabl InferenceService instance from the FastAPI application state
    """
    return request.app.state.inference_service

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Creates and yields a new asynchronous database session for each API request. The session is automatically closed after the request is completed or if an error occurs
    """
    SessionDep == Annotated[AsyncSession, Depends(get_db_session)]
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
        FeedbackRepository: An intialized instance of the repository
    """
    return FeedbackRepository(session)