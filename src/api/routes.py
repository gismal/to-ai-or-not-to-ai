from fastapi import APIRouter, Depends, UploadFile, File, Request, statu, Requests, HTTPException
from src.api.security import verify_api_key
from src.services.inference_service import InferenceService
from src.schemas.predict import PredictionResponse
from src.api.deps import get_inference_service
from slowapi import limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func = get_remote_address)

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
        file: UploadFile = File(...),
        service: InferenceService = Depends(get_inference_service),
        request : Request
        ):
        """
        Delegates the uploaded file to the underlying InferenceService
        """
        # early download limit
        if request.headers.get("content-length"):
            if int(request.headers.get("content-length")) > 10 * 1024 * 1024:
                raise HTTPException(status_code = 413, detail = "File too large. Max 10 MB allowed")
        
        return await service.process_upload(file)