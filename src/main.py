from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.config import settings
from src.logger import logger
from src.inference import ONNXPredictor
from src.api.routes import router, feedback_router
from src.infra.limiter import limiter
from src.infra.exceptions import InvalidImageFormatError, ModelInferenceError
from src.services.inference_service import InferenceService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle events of the FastAPI application. Loads the ONNX model into memo on startup and cleans up temporary directories on
    shutdown to prevent memo leaks and disk clutter
    """
    logger.info("Starting up API, loading ONNX model")
    
    try:
        predictor = ONNXPredictor()
        app.state.inference_service = InferenceService(
            predictor = predictor,
            threshold = settings.MODEL_THRESHOLD,
            gray_area_margin = settings.GRAY_AREA_MARGIN
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Startup failed, can't load model: {e}", exc_info = True)
        raise
    
    yield
    logger.info("Shutting down, realing model session")
    app.state.inference_service.predictor.close()
    
app = FastAPI(title= "To AI or Not to AI", version= "1.0.0", lifespan= lifespan)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware for future possinle frontend connections
ALLOWED_ORIGINS: list[str] = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins = settings.ALLOWED_ORIGINS,
    allow_credentials = False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

@app.exception_handler(InvalidImageFormatError)
async def invalid_image_handler(request: Request, exc: InvalidImageFormatError):
    """ Handles unsupported file format errors globally """
    return JSONResponse(status_code= 400, content= {"detail": str(exc)})

@app.exception_handler(ModelInferenceError)
async def model_error_handler(request: Request, exc: ModelInferenceError):
    """ Handles model inference errors """
    logger.error(f"Model Inference Error: {exc}")
    return JSONResponse(status_code= 500, content= {"detail": f"Model Error: {str(exc)}"})

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles errors related to Pydantic and logs them user-friendly
    """
    logger.warning(f"Unvalid data entry: {exc.errors()} - Endpoint: {request.url.path}")
    return JSONResponse(
        status_code = status.HTTP_422_UNPROCESSABLE_ENTITY,
        content = {"details": "Invalid request data submitted", "errors":exc.errors(include_url = False)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Handles all errors can occur unexpected and logs all of them
    """
    logger.error(f"Unexpected system error: {str(exc)} - Endpoint: {request.url.path}")
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content = {"detail": "Unexpected error occured"}
    )
    
app.include_router(router)
app.include_router(feedback_router)

