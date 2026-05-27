from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.response import JSONResponse

from src.config import settings
from src.logger import logger
from src.inference import ONNXPredictor
from src.api.routes import router
from src.core.exceptions import InvalidImageFormatError, ModelInferenceError
from src.services.inference_service import InferenceService

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the lifecycle events of the FastAPI application. Loads the ONNX model into memo on startup and cleans up temporary directories on
    shutdown to prevent memo leaks and disk clutter
    """
    logger.info("Starting up API, loading ONNX model")
    predictor = ONNXPredictor()
    
    app.state.inference_service = InferenceService(
        predictor = predictor,
        threshold = settings.MODEL_THRESHOLD
    )
    yield
    
    logger.info("Shutting down API, cleaning up resources")
    temp_dir = Path("temp_uploads")
    if temp_dir.exists():
        for f in temp_dir.glob("*"):
            f.unlink()
    
app = FastAPI(title= "To AI or Not to AI", version= "1.0.0", lifespan= lifespan)

@app.exception_handler(InvalidImageFormatError)
async def invalid_image_handler(request: Request, exc: InvalidImageFormatError):
    """ Handles unsupported file format errors globally """
    return JSONResponse(status_code= 400, content= {"detail": str(exc)})

@app.exception_handler(ModelInferenceError)
async def model_error_handler(request: Request, exc: ModelInferenceError):
    """ Handles model inference errors """
    logger.error(f"Model Inference Error: {exc}")
    return JSONResponse(status_code= 500, content= {"detail": f"Model Error: {str(exc)}"})

app.include_router(router)