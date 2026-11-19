"""
admin domain: drift detection, active learning, retraining
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.deps import get_db_session, get_retrain_service, verify_api_key
from src.core.enums import PredictionLabel
from src.infra.labeled_sample import LabeledSample
from src.infra.prediction_log import PredictionLog
from src.logger import logger
from src.schemas.label import LabelRequest, LabelResponse, UncertainPrediction
from src.services.retrain_service import RetrainService

admin_router = APIRouter(
    prefix="/v1/admin", tags=["Admin"], dependencies=[Depends(verify_api_key)]
)

_UNCERTAIN_LABELS = [
    PredictionLabel.UNCERTAIN_LEANING_AI.value,
    PredictionLabel.UNCERTAIN_LEANING_REAL.value,
    PredictionLabel.UNCERTAIN_NEUTRAL.value,
]


@admin_router.post(
    "/check-drift",
    status_code=status.HTTP_200_OK,
    summary=" Check model drift and trigger retraining if needed",
)
async def check_drift(retrain_service: RetrainService = Depends(get_retrain_service)):
    return await retrain_service.check_drift_and_trigger()


@admin_router.get(
    "/uncertain",
    response_model=list[UncertainPrediction],
    status_code=status.HTTP_200_OK,
    summary="List recent uncertain predictions awaiting human review",
)
async def list_uncertain(
    limit: int = 20, session: AsyncSession = Depends(get_db_session)
):
    stmt = (
        select(PredictionLog)
        .where(
            PredictionLog.predicted_label.in_(_UNCERTAIN_LABELS),
            PredictionLog.is_deleted.is_(False),
        )
        .order_by(PredictionLog.created_at.desc())
        .limit(limit)
    )
    rows = (await session.execute(stmt)).scalars().all()

    return [
        UncertainPrediction(
            id=int(row.id),
            filename=str(row.filename),
            confidence=float(row.confidence),
            predicted_label=PredictionLabel(row.predicted_label),
            processing_time_ms=float(row.processing_time_ms or 0),
            created_at=row.created_at.isoformat(),
        )
        for row in rows
    ]


@admin_router.post(
    "/label",
    response_model=LabelResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit a human-verified label for an uncertain prediction",
    responses={
        404: {"description": "No prediction log with this id"},
        409: {"description": "This prediction already has a label"},
    },
)
async def submit_label(
    payload: LabelRequest,
    session: AsyncSession = Depends(get_db_session),
):
    log_id = payload.prediction_log_id
    already_labelled = HTTPException(
        status_code=409, detail=f"Prediction log {log_id} is already labelled."
    )

    log_row = await session.scalar(
        select(PredictionLog).where(PredictionLog.id == log_id)
    )
    if not log_row:
        raise HTTPException(
            status_code=404, detail=f"Prediction log {log_id} not found."
        )

    existing = await session.scalar(
        select(LabeledSample.id).where(LabeledSample.prediction_log_id == log_id)
    )
    if existing is not None:
        raise already_labelled

    label = LabeledSample(
        prediction_log_id=log_id,
        filename=log_row.filename,
        original_prediction=PredictionLabel(log_row.predicted_label),
        correct_label=payload.correct_label,
    )
    session.add(label)
    try:
        await session.flush()
    except IntegrityError:
        # Two requests passed the check at the same moment; the database's
        # one-label-per-prediction rule stopped the second one.
        await session.rollback()
        raise already_labelled

    logger.info(f"Label saved: log {log_id} → {payload.correct_label.value}")

    return LabelResponse(
        prediction_log_id=log_id,
        correct_label=payload.correct_label,
        original_prediction=log_row.predicted_label,  # type: ignore
    )
