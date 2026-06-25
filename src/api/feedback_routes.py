"""
Feedback domain: submit user corrections
"""

from fastapi import APIRouter, Depends, status

from src.api.deps import get_feedback_service, verify_api_key
from src.schemas.feedback import FeedbackCreateRequest, FeedbackResponse
from src.services.feedback_service import FeedbackService

feedback_router = APIRouter(
    prefix="/v1/feedback",
    tags=["Feedback"],
    dependencies=[Depends(verify_api_key)],
)


@feedback_router.post(
    "/",
    response_model=FeedbackResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a correction for a prediction",
    responses={
        200: {"description": "Feedback saved"},
        401: {"description": "Invalid or missing API key"},
        422: {"description": "Validation error"},
    },
)
async def create_user_feedback(
    payload: FeedbackCreateRequest,
    service: FeedbackService = Depends(get_feedback_service),
):
    await service.register_feedback(payload)
    return FeedbackResponse(
        filename=payload.filename, message="Feedback received and saved"
    )
