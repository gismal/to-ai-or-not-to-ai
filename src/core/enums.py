"""
Shared enums used across the application.
All centralized here to ensure consistency in database records, logic routing and API responses
"""

from enum import Enum


class InferenceStatus(str, Enum):
    """
    Represents the results of the inference task.
    """

    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


class PredictionLabel(str, Enum):
    """
    All possible prediction outcomes by the AI model
    """

    REAL = "REAL"  # confidently human made
    AI_GENERATED = "AI_GENERATED"  # confidently AI generated
    UNCERTAIN_LEANING_AI = "UNCERTAIN_LEANING_AI"  # undecided but closer to AI side
    UNCERTAIN_LEANING_REAL = (
        "UNCERTAIN_LEANING_REAL"  # undecided but closer to human made
    )
    UNCERTAIN_NEUTRAL = (
        "UNCERTAIN_NEUTRAL"  # undecided but leans to nothing, absolute uncertainity
    )


class FeedbackLabel(str, Enum):
    """
    User feedback options to decide the origin of the image
    """

    REAL = "REAL"
    AI_GENERATED = "AI_GENERATED"


class ErrorType(str, Enum):
    """
    Classifies the model's prediction accuracy based on user feedback
    """

    FALSE_POSITIVE = "FALSE_POSITIVE"  # Model marks as AI but image is REAL
    FALSE_NEGATIVE = "FALSE_NEGATIVE"  # Model marks as REAL but image is AI
    UNCERTAIN_FAIL = "UNCERTAIN_FAIL"  # Model is indecisive
    CORRECT = "CORRECT"  # User confirms the model was rigth
