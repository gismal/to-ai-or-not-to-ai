from fastapi import Request
from src.services.inference_service import InferenceService

def get_inference_service(request: Request) -> InferenceService:
    """
    Dependency injection helper to retrieve the gloabl InferenceService instance from the FastAPI application state
    """
    return request.app.state.inference_service