"""
This unit test module is designed to validate the logic of the feedback classification of the model.
It ensures that the classify_error function correctly identifies the type of error based on the model's prediction and the user's feedback.
"""

import itertools

import pytest

from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel
from src.services.feedback_service import classify_error

# Aliases for easier reference in the tests
P, F, E = PredictionLabel, FeedbackLabel, ErrorType


@pytest.mark.parametrize(
    "prediction, truth, expected",
    [
        (P.AI_GENERATED, F.REAL, E.FALSE_POSITIVE),
        (P.REAL, F.AI_GENERATED, E.FALSE_NEGATIVE),
        (P.UNCERTAIN_LEANING_AI, F.REAL, E.UNCERTAIN_FAIL),
        (P.UNCERTAIN_LEANING_REAL, F.AI_GENERATED, E.UNCERTAIN_FAIL),
        (P.UNCERTAIN_NEUTRAL, F.REAL, E.UNCERTAIN_FAIL),
        (P.UNCERTAIN_NEUTRAL, F.AI_GENERATED, E.UNCERTAIN_FAIL),
        (P.AI_GENERATED, F.AI_GENERATED, E.CORRECT),
        (P.REAL, F.REAL, E.CORRECT),
        (P.AI_GENERATED, F.UNKNOWN, E.UNKNOWN),
        (P.REAL, F.UNKNOWN, E.UNKNOWN),
    ],
)
def test_classify_error(prediction, truth, expected):
    """
    Testing the coverage of the classify_error function to ensure that all combinations of
    predictions and feedback labels are correctly classified into the appropriate error types.
    This helps maintain the integrity of the feedback system and ensures accurate drift detection in the model's performance.

    Args:
        prediction (PredictionLabel): The model's prediction label.
        truth (FeedbackLabel): The user's feedback label.
        expected (ErrorType): The expected error type classification based on the prediction and feedback.

    Asssertions:
        - The classify_error function returns the expected error type for each combination of prediction and feedback labels
    """
    assert classify_error(prediction, truth) is expected


@pytest.mark.parametrize("predicted", list(PredictionLabel))
def test_unknown_answer_is_never_an_error(predicted):
    """
    Testing the situation where the feedback label is UNKNOWN, ensuring that the classify_error function always returns UNKNOWN error type regardless of the model's prediction.

    Args
        predicted (PredictionLabel): The model's prediction label.

    Asssertions:
        - The classify_error function returns UNKNOWN error type when the feedback label is UNKNOWN, regardless of
    """

    assert classify_error(predicted, F.UNKNOWN) is E.UNKNOWN


def test_every_combination_is_classified():
    """
    Testing that every combination of prediction and feedback labels is linked into a valid error definition.

    Assertions:
        - The classify_error function returns an instance of ErrorType for every combination of PredictionLabel and FeedbackLabel.
        It ensures comprehensive coverage of all possible scenarios in the feedback classification process.

    """
    for predicted, truth in itertools.product(PredictionLabel, FeedbackLabel):
        assert isinstance(classify_error(predicted, truth), ErrorType)
