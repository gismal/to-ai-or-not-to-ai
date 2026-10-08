from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel
from src.logger import logger
from src.repositories.feedback_repo import FeedbackRepository
from src.schemas.feedback import FeedbackCreateRequest


class FeedbackService:
    def __init__(self, repo: FeedbackRepository) -> None:
        self.repo = repo

    async def register_feedback(self, payload: FeedbackCreateRequest) -> None:
        try:
            error_type = classify_error(
                model_prediction=payload.model_prediction,
                user_correction=payload.user_correction,
            )
            await self.repo.create_feedback(
                filename=payload.filename,
                model_prediction=payload.model_prediction,
                confidence=payload.confidence,
                user_correction=payload.user_correction,
                client_source=payload.client_source,
                error_type=error_type,
            )
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}", exc_info=True)
            raise


def classify_error(
    model_prediction: PredictionLabel, user_correction: FeedbackLabel
) -> ErrorType:
    """
    Compares the model prediction to user correction

    FALSE_POSITIVE: model said AI, user says human made
    FALSE_NEGATIVE: model said REAL, user says AI generated
    UNCERTAIN_FAIL: model was uncertain, user had a clear answer
    CORRECT: model and user agree
    """
    if user_correction == FeedbackLabel.UNKNOWN:
        return ErrorType.UNKNOWN

    pred = model_prediction.value
    truth = user_correction.value

    if pred == truth:
        return ErrorType.CORRECT

    if "UNCERTAIN" in pred:
        return ErrorType.UNCERTAIN_FAIL

    if pred == "AI_GENERATED" and truth == "REAL":
        return ErrorType.FALSE_POSITIVE

    if pred == "REAL" and truth == "AI_GENERATED":
        return ErrorType.FALSE_NEGATIVE

    return ErrorType.UNCERTAIN_FAIL
