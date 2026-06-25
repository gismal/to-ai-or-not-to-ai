"""
src/api/inference_routes.py
Inference domain: predict, predict-batch, explain, health.
"""

import asyncio
import io
import magic
from pathlib import PurePosixPath

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from PIL import Image as PILImage
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import (
    get_db_session,
    get_explainability_service,
    get_inference_service,
    get_prediction_log_repository,
    verify_api_key,
)
from src.config import settings
from src.core.exceptions import InvalidImageFormatError
from src.infra.limiter import limiter
from src.repositories.prediction_log_repository import PredictionLogRepository
from src.schemas.batch import BatchPredictionResponse
from src.schemas.explain import ExplainResponse
from src.schemas.predict import PredictionResponse
from src.services.explainability_service import ExplainabilityService
from src.services.inference_service import InferenceService


router = APIRouter(
    prefix="/v1/inference",
    tags=["Inference"],
    dependencies=[Depends(verify_api_key)],
)


# ── Health ────────────────────────────────────────────────────────────────────


@router.get(
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
    except Exception:
        pass

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

    return {"status": "healthy", "checks": checks}


# ── Predict ───────────────────────────────────────────────────────────────────


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify image origin",
    responses={
        200: {"description": "Successful inference"},
        400: {"description": "Unsupported image format or bad request"},
        401: {"description": "Invalid or missing API key"},
        413: {"description": "File exceeds size limit"},
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
    content_bytes, safe_filename = await _read_and_validate(request, file)
    result = await service.predict(content_bytes, safe_filename)

    background_tasks.add_task(
        log_repo.create_log,
        filename=safe_filename,
        confidence=result["confidence"],  # type: ignore
        predicted_label=result["prediction"],  # type: ignore
        processing_time_ms=result["processing_time_ms"],  # type: ignore
    )
    return result


# ── Predict Batch ─────────────────────────────────────────────────────────────


@router.post(
    "/predict-batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify multiple images in one request",
)
async def predict_batch(
    request: Request,
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    service: InferenceService = Depends(get_inference_service),
    log_repo: PredictionLogRepository = Depends(get_prediction_log_repository),
):
    if len(files) > settings.MAX_BATCH_SIZE:
        files = files[: settings.MAX_BATCH_SIZE]

    async def process_one(f: UploadFile) -> dict:
        try:
            content_bytes, safe_name = await _read_and_validate(request, f)
            result = await service.predict(content_bytes, safe_name)
            background_tasks.add_task(
                log_repo.create_log,
                filename=safe_name,
                confidence=result["confidence"],  # type: ignore
                predicted_label=result["prediction"],  # type: ignore
                processing_time_ms=result["processing_time_ms"],  # type: ignore
            )
            return result
        except Exception as exc:
            return {"filename": f.filename, "error": str(exc), "status": "FAILED"}

    results = await asyncio.gather(*[process_one(f) for f in files])
    successful = sum(1 for r in results if r.get("status") != "FAILED")

    return BatchPredictionResponse(
        predictions=list(results),
        total=len(results),
        successful=successful,
        failed=len(results) - successful,
    )


# ── Explain ───────────────────────────────────────────────────────────────────


@router.post(
    "/explain",
    response_model=ExplainResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify image and return GradCAM heatmap",
    description=(
        "Runs the full inference pipeline and generates a GradCAM heatmap "
        "showing which regions influenced the decision. "
        "Slower than /predict (~300-500ms) due to gradient computation."
    ),
    responses={
        200: {"description": "Prediction + heatmap returned successfully"},
        400: {"description": "Unsupported image format"},
        401: {"description": "Invalid or missing API key"},
        413: {"description": "File exceeds size limit"},
        500: {"description": "GradCAM or model failure"},
    },
)
async def explain_image(
    request: Request,
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service),
    explainer: ExplainabilityService = Depends(get_explainability_service),
):
    content_bytes, safe_filename = await _read_and_validate(request, file)
    return await explainer.explain(
        image_bytes=content_bytes,
        filename=safe_filename,
        decide_class_fn=service._decide_class,
    )


# ── Shared validation helper ──────────────────────────────────────────────────


async def _read_and_validate(
    request: Request,
    file: UploadFile,
) -> tuple[bytes, str]:
    """
    Single validation gate for all image endpoints.
    1. Content-length size guard
    2. Read bytes
    3. Real MIME type check (python-magic on actual bytes)
    4. Minimum pixel dimension check
    5. Filename sanitization
    """
    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {settings.MAX_UPLOAD_SIZE_MB} MB allowed.",
        )

    content_bytes = await file.read()

    mime = magic.from_buffer(content_bytes[:2048], mime=True)
    if mime not in ("image/jpeg", "image/png"):
        raise InvalidImageFormatError(
            f"Unsupported file type: {mime}. Only JPEG and PNG accepted."
        )

    with PILImage.open(io.BytesIO(content_bytes)) as img:
        w, h = img.size
        if min(w, h) < settings.MIN_IMAGE_DIMENSION:
            raise InvalidImageFormatError(
                f"Image too small ({w}x{h}px). "
                f"Minimum is {settings.MIN_IMAGE_DIMENSION}px."
            )

    safe_filename = PurePosixPath(file.filename or "unnamed").name
    return content_bytes, safe_filename
