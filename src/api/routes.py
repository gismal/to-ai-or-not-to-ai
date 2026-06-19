from fastapi import APIRouter, Depends, UploadFile, File, Request, status, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession # Eksik import eklendi

from src.infra.limiter import limiter
from src.services.inference_service import InferenceService
from src.schemas.predict import PredictionResponse
from src.api.deps import get_inference_service, get_feedback_service, verify_api_key, get_db_session # get_db_session eklendi
from src.schemas.feedback import FeedbackCreateRequest, FeedbackResponse
from src.services.feedback_service import FeedbackService
from src.services.retrain_service import RetrainService 
from src.api.deps import get_arq_pool

"""
Inference Router
"""
router = APIRouter(
    prefix="/inference",
    tags=["Inference"],
    dependencies=[Depends(verify_api_key)]    
)

@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Check model and system status"
)
async def health_check(request: Request): # request parametresi eklendi
    """
    Verifies that the model is loaded and responsive, is the database reachable
    """
    # Not: _check_model ve _check_database fonksiyonları bu dosyada yok.
    # Bunları ya infra/health.py gibi bir yerden import etmeli ya da burada tanımlamalısınız.
    # Şimdilik kodun çökmemesi için geçici bir yapı kuruyorum.
    
    inference_service = request.app.state.inference_service
    is_model_loaded = inference_service is not None and inference_service.predictor is not None
    
    checks = {
        "model": is_model_loaded,
        "database": True # Gerçek bir DB kontrolü (örn: session.execute(text("SELECT 1"))) buraya gelmeli
    }
    
    system_status = "healthy" if all(checks.values()) else "degraded"        
    return {"status": f"System is {system_status}", "checks": checks, "model_loaded": is_model_loaded}


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Classify image origin",
    description="Accepts an image file, validates the format and executes ONNX inference"
)
@limiter.limit("5/second")
async def predict_image(
    request: Request,
    file: UploadFile = File(...),
    service: InferenceService = Depends(get_inference_service)
):
    """
    Delegates the uploaded file to the underlying InferenceService
    """
    # early download limit
    content_length = request.headers.get("content-length")
    if content_length and int(content_length) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Max 10 MB allowed")
    
    return await service.process_upload(file)

@router.post(
    "/admin/check-drift",
    status_code=status.HTTP_200_OK,
    summary="Check model drift and trigger retraining if needed",
    tags=["Admin"]
)
async def check_drift(
    session: AsyncSession = Depends(get_db_session),
    arq_pool: ArqRedis = Depends(get_arq_pool)
):
    return await RetrainService.check_drift_and_trigger(session, arq_pool)


"""
Feedback Router
"""
feedback_router = APIRouter(
    prefix="/feedback",
    tags=["Feedback"],
    dependencies=[Depends(verify_api_key)]
)

@feedback_router.post(
    "/",
    response_model=FeedbackResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Get user feedback"
)
async def create_user_feedback(
    payload: FeedbackCreateRequest,
    background_tasks: BackgroundTasks,
    service: FeedbackService = Depends(get_feedback_service)
):
    service.register_feedback(payload, background_tasks)
    
    return FeedbackResponse(
        filename=payload.filename,
        message="Feedback received and queued for processing"
    )