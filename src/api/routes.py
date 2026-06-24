import io
import asyncio
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

from core.enums import PredictionLabel
from schemas.batch import BatchPredictionResponse
from schemas.label import LabelRequest, LabelResponse, UncertainPrediction
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
from src.logger import logger

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


@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify multiple images in one request",
    description=(
        "Accepts up to 10 images simultaneously. "
        "Each image is validated and predicted concurrently. "
        "Failed images are included in the response with an error field rather "
        "than aborting the entire batch."
    ),
    responses={
        200: {"description": "Batch processed — check each item's status"},
        401: {"description": "Invalid or missing API key"},
    },
)
async def predict_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    service: InferenceService = Depends(get_inference_service),
    log_repo: PredictionLogRepository = Depends(get_prediction_log_repository),
):
    MAX_BATCH = 10
    if len(files) > MAX_BATCH:
        files = files[:MAX_BATCH]

    async def process_one(file: UploadFile) -> dict:
        try:
            content_bytes = await file.read()
            mime = magic.from_buffer(content_bytes[:2048], mime=True)
            if mime not in ("image/jpeg", "image/png"):
                return {
                    "filename": file.filename,
                    "error": f"Unsupported type: {mime}",
                    "status": "FAILED",
                }
            safe_name = PurePosixPath(file.filename or "unnamed").name
            result = await service.predict(content_bytes, safe_name)

            background_tasks.add_task(
                log_repo.create_log,
                filename=safe_name,
                confidence=result["confidence"],
                predicted_label=result["prediction"],
                processing_time_ms=result["processing_time_ms"],
            )
            return result

        except Exception as exc:
            return {"filename": file.filename, "error": str(exc), "status": "FAILED"}

    results = await asyncio.gather(*[process_one(f) for f in files])
    successful = sum(1 for r in results if r.get("status") != "FAILED")

    return BatchPredictionResponse(
        predictions=list(results),
        total=len(results),
        successful=successful,
        failed=len(results) - successful,
    )


@router.get(
    "/admin/uncertain",
    response_model=list[UncertainPrediction],
    status_code=status.HTTP_200_OK,
    summary="List recent UNCERTAIN predictions awaiting human review",
    tags=["Admin"],
)
async def list_uncertain(
    limit: int = 20, session: AsyncSession = Depends(get_db_session)
):
    """
    Returns the most recent predictions the model was uncertain about
    These are candidates for human labeling via POST /admin/label
    """
    from sqlalchemy import select
    from src.infra.feedbacks import PredictionLog

    uncertain_labels = [
        PredictionLabel.UNCERTAIN_LEANING_AI.value,
        PredictionLabel.UNCERTAIN_LEANING_REAL.value,
        PredictionLabel.UNCERTAIN_NEUTRAL.value,
    ]

    stmt = (
        select(PredictionLog)
        .where(
            PredictionLog.predicted_label.in_(uncertain_labels),
            PredictionLog.is_deleted == False,  # noqa: E712
        )
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
    )

    result = await session.execute(stmt)
    rows = result.scalars().all()

    return [
        UncertainPrediction(
            id=row.id,
            filename=row.filename,
            confidence=row.confidence,
            predicted_label=row.predicted_label,
            processing_time_ms=row.processing_time_ms,
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@router.post(
    "/admin/label",
    response_model=LabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a human-verified label for an UNCERTAIN prediction",
    tags=["Admin"],
)
async def submit_label(
    payload: LabelRequest, session: AsyncSession = Depends(get_db_session)
):
    """
    Saves a human-provided correct label for a prediction
    Labeled samples are automatically included in the next training run
    """
    from sqlalchemy import select
    from src.infra.feedbacks import PredictionLog
    from src.infra.labeled_samples import LabeledSample

    log_row = await session.scalar(
        select(PredictionLog).where(PredictionLog.id == payload.prediction_log_id)
    )
    if not log_row:
        raise HTTPException(
            status_code=404,
            detail=f"Prediction log {payload.prediction_log_id} not found",
        )

    label = LabeledSample(
        prediction_log_id=payload.prediction_log_id,
        filename=log_row.filename,
        original_prediction=log_row.predicted_label,
        correct_label=payload.correct_label,
    )
    session.add(label)
    await session.flush()

    logger.info(
        f"Label saved: prediction {payload.prediction_log_id} > {payload.correct_label.value}"
    )

    return LabelResponse(
        prediction_log_id=payload.prediction_log_id,
        correct_label=payload.correct_label,
        original_prediction=log_row.predicted_label,
    )
