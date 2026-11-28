import uuid
from contextlib import asynccontextmanager

import redis.exceptions
from arq import create_pool
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.api.admin_routes import admin_router
from src.api.feedback_routes import feedback_router
from src.api.health_routes import health_router
from src.api.inference_routes import router as inference_router
from src.config import settings
from src.core.exceptions import (
    DatabaseError,
    InvalidImageFormatError,
    ModelInferenceError,
)
from src.infra.limiter import limiter
from src.infra.model_info import read_model_info
from src.infra.redis_client import create_cache_client
from src.logger import logger, request_id_ctx
from src.ml.explainer import ExplainabilityService
from src.ml.familiarity import FamiliarityCheck
from src.ml.predictor import ONNXPredictor
from src.services.cache_service import CacheService
from src.services.inference_service import InferenceService
from src.worker.tasks import WorkerSettings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup:
      1. Model info: read the serving model's version from models/metadata.json
      2. ARQ pool: job queue (Redis)
      3. Cache client: pHash cache, keyed by model version (same Redis)
      4. ONNX predictor + familiarity check
      5. InferenceService (receives cache, not arq_pool)
      6. ExplainabilityService
    Shutdown:
      - Close both Redis clients
    """
    logger.info("Starting up API, loading ONNX model")
    app.state.model_info = read_model_info()
    logger.info(
        f"Serving model {app.state.model_info.version} "
        f"(trained {app.state.model_info.trained_at or 'unknown'})"
    )

    try:
        redis_url = settings.REDIS_URL

        if redis_url and redis_url != "none":
            # -- Redis: job queue (db 0 ) -------------
            app.state.arq_pool = await create_pool(WorkerSettings.redis_settings)

            # -- Redis: cache client (db 0) seperate from job queue --------------
            app.state.cache_client = await create_cache_client()
            app.state.cache_service = CacheService(
                redis_client=app.state.cache_client,
                model_version=app.state.model_info.version,
            )
            logger.info("Redis connected")
        else:
            app.state.arq_pool = None
            app.state.cache_service = None

    except (
        redis.exceptions.ConnectionError,
        redis.exceptions.TimeoutError,
    ) as redis_exc:  # todo: special exception for redis and migration
        logger.warning(f"Redis unavailable ({redis_exc}) - caching disabled")
        app.state.arq_pool = None
        app.state.cache_service = None

    try:
        # -- ONNX Inference -----------------------
        logger.info("Loading AI models")
        predictor = ONNXPredictor()

        # -- Familiarity check (optional: verdicts aren't downgraded without it) --
        familiarity = FamiliarityCheck.load()
        logger.info(
            "Familiarity check loaded"
            if familiarity
            else "Familiarity check not found; verdicts won't be downgraded"
        )

        app.state.inference_service = InferenceService(
            predictor=predictor,
            cache=app.state.cache_service,  # type: ignore
            threshold=settings.MODEL_THRESHOLD,
            gray_area_margin=settings.GRAY_AREA_MARGIN,
            familiarity=familiarity,
            enforce_familiarity=settings.FAMILIARITY_ENFORCE,
        )

        # -- GradCAM explainability --------------------------
        app.state.explainability_service = ExplainabilityService(
            checkpoint_path="models/best_checkpoint.pt",
            num_classes=2,
        )
        logger.info("All services ready")

    except Exception as e:
        logger.critical(f"Startup failed, can't load model: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down")
    if app.state.arq_pool:
        await app.state.arq_pool.aclose()
    if hasattr(app.state, "cache_client") and app.state.cache_client:
        await app.state.cache_client.aclose()


app = FastAPI(title="To AI or Not to AI", version="1.0.0", lifespan=lifespan)
app.include_router(inference_router)
app.include_router(health_router)
app.include_router(feedback_router)
app.include_router(admin_router)


@app.get("/privacy", include_in_schema=False)
async def privacy():
    return FileResponse("frontend/privacy.html")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_ctx.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    request_id_ctx.reset(token)
    return response


Instrumentator().instrument(app).expose(app, include_in_schema=False)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.exception_handler(InvalidImageFormatError)
async def invalid_image_handler(request: Request, exc: InvalidImageFormatError):
    """Handles unsupported file format errors globally"""
    return JSONResponse(status_code=400, content={"detail": f"{exc!s}"})


@app.exception_handler(ModelInferenceError)
async def model_error_handler(request: Request, exc: ModelInferenceError):
    """Handles model inference errors"""
    logger.error(f"Model Inference Error: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"Model Error: {exc!s}"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles errors related to Pydantic and logs them user-friendly
    """
    logger.warning(f"Unvalid data entry: {exc.errors()} - Endpoint: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "details": "Invalid request data submitted",
            "errors": exc.errors(),
        },
    )


@app.exception_handler(DatabaseError)
async def database_error_handler(request: Request, exc: DatabaseError):
    logger.error(f"Database error: {exc} — Endpoint: {request.url.path}")
    return JSONResponse(
        status_code=500, content={"detail": "A database error occurred."}
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handles all errors can occur unexpected and logs all of them
    """
    logger.error(f"Unexpected system error: {exc!s} - Endpoint: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Unexpected error occured"},
    )


app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", include_in_schema=False)
async def serve_playground():
    return FileResponse("frontend/index.html")
