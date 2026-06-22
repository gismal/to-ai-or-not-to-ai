import io

from fastapi import (
    APIRouter,
    Depends,
    UploadFile,
    File,
    Request,
    status,
    HTTPException,
    BackgroundTasks,
)
from sqlalchemy.ext.asyncio import AsyncSession  # Eksik import eklendi
from sqlalchemy import text
from arq import ArqRedis
import magic
from pathlib import PurePosixPath
from PIL import Image as PILImage

from src.api.deps import get_prediction_log_repository
from src.repositories.prediction_log_repository import PredictionLogRepository
from src.infra.limiter import limiter
from src.services.inference_service import InferenceService
from src.schemas.predict import PredictionResponse
from src.api.deps import (
    get_inference_service,
    get_feedback_service,
    verify_api_key,
    get_db_session,
    get_arq_pool,
    get_explainability_service,
)
from src.schemas.feedback import FeedbackCreateRequest, FeedbackResponse
from src.services.feedback_service import FeedbackService
from src.services.retrain_service import RetrainService
from src.services.explainability_service import ExplainabilityService
from src.schemas.explain import ExplainResponse
from src.core.exceptions import InvalidImageFormatError


"""
Inference Router
"""
router = APIRouter(
    prefix="/v1/inference", tags=["Inference"], dependencies=[Depends(verify_api_key)]
)


@router.get(
    "/health",  # dependecies = []
    status_code=status.HTTP_200_OK,
    summary="System Health Check",
)
async def health_check(
    request: Request, session: AsyncSession = Depends(get_db_session)
):
    """
    Verifies that the model is loaded and responsive, is the database reachable
    """

    inference_service = request.app.state.inference_service
    is_model_loaded = (
        inference_service is not None and inference_service.predictor is not None
    )

    db_ok = False
    try:
        await session.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass

    redis_ok = (
        hasattr(request.app.state, "arq_pool")
        and request.app.state.arq_pool is not None
    )

    checks = {
        "model": is_model_loaded,
        "database": db_ok,
        "redis": redis_ok,
    }

    if not all(checks.values()):
        raise HTTPException(
            status_code=503, detail={"status": "degraded", "checks": checks}
        )

    return {"status": "healthy", "checks": checks}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify image origin",
    responses={
        200: {"description": "Successful inference"},
        400: {"description": "Unsupported image format or bad request"},
        401: {"description": "Invalid or missing API key"},
        413: {"description": "File exceeds 10MB limit"},
        429: {"description": "Rate limit exceeded (5 requests/second)"},
        500: {"description": "Model inference failure"},
    },
    description="Accepts an image file, validates the format and executes ONNX inference",
)
@limiter.limit("5/second")
async def predict_image(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
    log_repo: PredictionLogRepository = Depends(get_prediction_log_repository),
):
    # -- Size Guard ----------------------
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB allowed")

    # -- Read -----------------------------
    content_bytes = await file.read()

    # -- Validate MIME ------------------
    mime = magic.from_buffer(content_bytes[:2048], mime=True)
    if mime not in ("image/jpeg", "image/png"):
        raise InvalidImageFormatError(
            f"Unsupported file type: {mime}. Only JPEG and PNG accepted"
        )

    with PILImage.open(io.BytesIO(content_bytes)) as img:
        width, height = img.size
        if min(width, height) < 32:
            raise InvalidImageFormatError(
                f"Image too small ({width}x{height}px). Minimum dimension is 32px."
            )
    # -- Sanitize filename --------------------------
    # path traversal protection
    safe_filename = PurePosixPath(file.filename or "unnamed").name

    # -- Predict ------------------------------------
    result = await service.predict(content_bytes, safe_filename)

    # -- asycn Log prediction ---------------------
    background_tasks.add_task(
        log_repo.create_log,
        filename=safe_filename,
        confidence=result["confidence"],
        predicted_label=result["prediction"],
        processing_time_ms=result["processing_time_ms"],
    )

    return result


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify image and return GradCAM heatmap",
    description=(
        "Runs the full inference pipeline and additionally generates a GradCAM "
        "heatmap showing which regions of the image influenced the decision. "
        "Returns the heatmap as a base64-encoded PNG overlay. "
        "Slower than /predict (~300-500ms) due to gradient computation."
    ),
    responses={
        200: {"description": "Prediction + heatmap returned successfully"},
        400: {"description": "Unsupported image format"},
        401: {"description": "Invalid or missing API key"},
        413: {"description": "File exceeds 10MB limit"},
        500: {"description": "GradCAM or model failure"},
    },
)
async def explain_image(
    request: Request,
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
    explainer: ExplainabilityService = Depends(get_explainability_service),
):
    """
    Same validation as /predict, then runs GradCAM on the PyTorch model.
    The _decide_class function is passed as a callback so uncertainty boundary
    logic lives in exactly one place (InferenceService) and isn't duplicated here.
    """
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB allowed")

    content_bytes = await file.read()

    mime = magic.from_buffer(content_bytes[:2048], mime=True)
    if mime not in ("image/jpeg", "image/png"):
        raise InvalidImageFormatError(
            f"Unsupported file type: {mime}. Only JPEG and PNG accepted."
        )

    safe_filename = PurePosixPath(file.filename or "unnamed_image").name

    return await explainer.explain(
        image_bytes=content_bytes,
        filename=safe_filename,
        decide_class_fn=service._decide_class,
    )


@router.post(
    "/admin/check-drift",
    status_code=status.HTTP_200_OK,
    summary="Check model drift and trigger retraining if needed",
    tags=["Admin"],
)
async def check_drift(
    session: AsyncSession = Depends(get_db_session),
    arq_pool: ArqRedis = Depends(get_arq_pool),
):
    return await RetrainService.check_drift_and_trigger(session, arq_pool)


"""
Feedback Router
"""
feedback_router = APIRouter(
    prefix="/v1/feedback", tags=["Feedback"], dependencies=[Depends(verify_api_key)]
)


@feedback_router.post(
    "/",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit correction for a prediction",
)
async def create_user_feedback(
    payload: FeedbackCreateRequest,
    service: FeedbackService = Depends(get_feedback_service),
):
    await service.register_feedback(payload)

    return FeedbackResponse(
        filename=payload.filename, message="Feedback received and queued for processing"
    )
