# clean home for shared enums
from enum import Enum


class InferenceStatus(str, Enum):
    """Enumeration of the inference execution status."""

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class PredictionLabel(str, Enum):
    """
    Enumaration of all possible prediction outcomes
    """

    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"
    UNCERTAIN_LEANING_AI = "UNCERTAIN_LEANING_AI"
    UNCERTAIN_LEANING_REAL = "UNCERTAIN_LEANING_REAL"
    UNCERTAIN_NEUTRAL = "UNCERTAIN_NEUTRAL"


class FeedbackLabel(str, Enum):
    """
    Enum classes for the database to define feedbacks
    """

    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"


class ErrorType(str, Enum):
    FALSE_POSITIVE = "FALSE_POSITIVE"  # Model marks as AI but image is REAL
    FALSE_NEGATIVE = "FALSE_NEGATIVE"  # Model marks as REAL but image is AI
    UNCERTAIN_FAIL = "UNCERTAIN_FAIL"  # Model is indecisive
    CORRECT = "CORRECT"  # User confirms the model was rigth
