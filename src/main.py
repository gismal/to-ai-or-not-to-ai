from arq import create_pool
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import uuid

from src.config import settings
from src.logger import logger, request_id_ctx
from src.inference import ONNXPredictor
from src.infra.limiter import limiter
from src.core.exceptions import (
    InvalidImageFormatError,
    ModelInferenceError,
    DatabaseError,
)
from src.services.inference_service import InferenceService
from src.services.cache_service import CacheService
from src.worker.tasks import WorkerSettings
from src.services.explainability_service import ExplainabilityService
from src.api.routes import router as inference_router
from src.api.routes import feedback_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle events of the FastAPI application. Loads the ONNX model into memo on startup and cleans up temporary directories on
    shutdown to prevent memo leaks and disk clutter
    """
    logger.info("Starting up API, loading ONNX model")
    try:
        app.state.arq_pool = await create_pool(WorkerSettings.redis_settings)

        predictor = ONNXPredictor()
        app.state.cache_service = CacheService(redis_pool=app.state.arq_pool)

        app.state.inference_service = InferenceService(
            predictor=predictor,
            cache=app.state.cache_service,
            threshold=settings.MODEL_THRESHOLD,
            gray_area_margin=settings.GRAY_AREA_MARGIN,
        )

        app.state.explainability_service = ExplainabilityService(
            checkpoint_path="models/best_checkpoint.pt", num_classes=2
        )
        logger.info("Model loaded successfully")

    except Exception as e:
        logger.critical(f"Startup failed, can't load model: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down, releasing model session")
    await app.state.arq_pool.aclose()


app = FastAPI(title="To AI or Not to AI", version="1.0.0", lifespan=lifespan)
app.include_router(inference_router)
app.include_router(feedback_router)

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

app.mount("/static", StaticFiles(directory="frontend"), name="static")


@app.get("/", include_in_schema=False)
async def serve_playground():
    return FileResponse("frontend/index.html")


@app.get("/health", tags=["System"], status_code=status.HTTP_200_OK)
async def health_check():
    return {"status": "ok"}


@app.exception_handler(InvalidImageFormatError)
async def invalid_image_handler(request: Request, exc: InvalidImageFormatError):
    """Handles unsupported file format errors globally"""
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(ModelInferenceError)
async def model_error_handler(request: Request, exc: ModelInferenceError):
    """Handles model inference errors"""
    logger.error(f"Model Inference Error: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"Model Error: {str(exc)}"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles errors related to Pydantic and logs them user-friendly
    """
    logger.warning(f"Unvalid data entry: {exc.errors()} - Endpoint: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
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
    logger.error(f"Unexpected system error: {str(exc)} - Endpoint: {request.url.path}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Unexpected error occured"},
    )


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    token = request_id_ctx.set(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    request_id_ctx.reset(token)
    return response


Instrumentator().instrument(app).expose(app)
