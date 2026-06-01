from fastapi import APIRouter, Depends, UploadFile, File, Request, status, HTTPException, BackgroundTasks
from src.api.security import verify_api_key
from src.services.inference_service import InferenceService
from src.schemas.predict import PredictionResponse
from src.api.deps import get_inference_service, get_feedback_service
from src.schemas.feedback import FeedbackCreateRequest, FeedbackResponse
from src.services.feedback_service import FeedbackService
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func = get_remote_address)

"""
Inference Router
"""
router = APIRouter(
    prefix = "/inference",
    tags = ["Inference"],
    dependencies = [Depends(verify_api_key)]    
)

class InferenceRouter:
    """
    Class-based structure encapsulating the inference routes to ensure OOP principles within the API layer 
    """

    @staticmethod
    @router.get(
        "/health",
        status_code= status.HTTP_200_OK,
        summary= "Check model and system status")
    def health_check():
        """
        Verifies that the API and the inference engine are running
        """
        return {"status": "System is running optimally", "model_loaded": True}

    @staticmethod
    @router.post(
        "/predict",
        response_model = PredictionResponse,
        status_code = status.HTTP_200_OK,
        summary= "Classify image origin",
        description= "Accepts an image file, validates the format and executes ONNX inference"
    )
    @limiter.limit("5/second")
    async def predict_image(
        request : Request,
        file: UploadFile = File(...),
        service: InferenceService = Depends(get_inference_service)
        ):
        """
        Delegates the uploaded file to the underlying InferenceService
        """
        # early download limit
        if request.headers.get("content-length"):
            if int(request.headers.get("content-length")) > 10 * 1024 * 1024:
                raise HTTPException(status_code = 413, detail = "File too large. Max 10 MB allowed")
        
        return await service.process_upload(file)
    
    """
    Feedback Router
    """
feedback_router = APIRouter(
    prefix = "/feedback",
    tags = ["Feedback"],
    dependencies = [Depends(verity_api_key)]
    )
    
class FeedbackRouter:
    """
    Class-based structure encapsulating feedback routes
    """
    @staticmethod
    @feedback_router.post(
        "/",
        response_model = FeedbackResponse,
        status_code = status.HTTP_202_ACCEPTED,
        summary = "Get user feedback"
        )
    
    async def create_user_feedback(
        payload: FeedbackCreateRequest,
        background_tasks: BackgroundTasks,
        service: FeedbackService = Depends(get_feedback_service)
        ):
        service.register_feedback(payload, background_tasks)
            
        return FeedbackResponse(
            filename = payload.filename,
            message = "Feedback received and queued for processing"
            )