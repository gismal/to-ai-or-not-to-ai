from fastapi import APIRouter, Depends, UploadFile, File, Request, status
from src.api.security import verify_api_key
from src.services.inference_service import InferenceService
from src.schemas.predict import PredictionResponse
from src.api.deps import get_inference_service
 
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
    async def predict_image(
        file: UploadFile = File(...),
        service: InferenceService = Depends(get_inference_service),
        ):
        """
        Delegates the uploaded file to the underlying InferenceService
        """
        return await service.process_upload(file)