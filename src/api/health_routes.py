"""Liveness and readiness: is the model loaded, and are the database and Redis reachable?"""

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session
from src.infra.model_info import current_model_info
from src.logger import logger

health_router = APIRouter(tags=["System"])


@health_router.get(
    "/health",
    dependencies=[],
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
)
async def health_check(
    request: Request,
    session: AsyncSession = Depends(get_db_session),
):
    inference_service = request.app.state.inference_service
    is_model_loaded = (
        inference_service is not None and inference_service.predictor is not None
    )

    db_ok = False
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except SQLAlchemyError as db_exc:
        logger.error(f"Database health check failed: {db_exc}", exc_info=True)
    except Exception as e:
        logger.error(f"Database failed: {e}", exc_info=True)

    redis_ok = (
        hasattr(request.app.state, "arq_pool")
        and request.app.state.arq_pool is not None
    )

    checks = {"model": is_model_loaded, "database": db_ok, "redis": redis_ok}

    if not all(checks.values()):
        raise HTTPException(
            status_code=503,
            detail={"status": "degraded", "checks": checks},
        )

    return {
        "status": "healthy",
        "checks": checks,
        "model": current_model_info(request).as_dict(),
    }
