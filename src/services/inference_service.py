from src.inference import ONNXPredictor
from src.core.enums import PredictionLabel

class InferenceService:
    """
    Encapsulates the core business logic, includng asynchronous file streaming, model execution and granular uncertainty classification
    """

    def __init__(
        self,
        predictor: ONNXPredictor,
        threshold: float,
        gray_area_margin: float = 0.35,
    ):
        """
        Initializes the inference service with its dependencies

        Args:
            - predictor (ONNXPredictor): The loaded ONNX runtime engine
            - threshold (float): The base confidence score to flag an image as AI
            - gray_area_margin (float): The amrgin below the threshol to classify as UNCERTAIN

        """
        self.predictor = predictor
        self.threshold = threshold
        self.lower_bound = threshold - gray_area_margin
        self.one_third = gray_area_margin / 3

    
    def _decide_class(self, confidence: float) -> PredictionLabel:
        """
        Determines the final label with detailed granularity for uncertain cases

        Args:
            - confidence (float): The prediction score from the ONNX model

        Returns:
            - PredictionLabel: Classification outcome

        """
        # sharp bounds
        if confidence >= self.threshold:
            return PredictionLabel.AI_GENERATED
        if confidence <= self.lower_bound:
            return PredictionLabel.REAL

        # analysis of uncertain area
        if confidence >= (self.threshold - self.one_third):
            return PredictionLabel.UNCERTAIN_LEANING_AI
        elif confidence <= (self.lower_bound + self.one_third):
            return PredictionLabel.UNCERTAIN_LEANING_REAL
        else:
            return PredictionLabel.UNCERTAIN_NEUTRAL

    