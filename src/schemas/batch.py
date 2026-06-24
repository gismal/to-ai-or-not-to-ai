from pydantic import BaseModel, Field
from src.schemas.predict import PredictionResponse


class BatchPredictionResponse(BaseModel):
    """ "
    Response for POST /v1/inference/predict-batch
    """

    predictions: list[PredictionResponse | dict]  # dict for failed items
    total: int = Field(description="Total files received")
    successful: int = Field(description="Successfully analysed")
    failed: int = Field(description="Files that could not be processed")
