"""
Admin review routes: listing uncertain predictions and labelling them.

The route functions are called directly with the test session, so these tests check
the handlers' own logic. Who may call them is covered in test_api.py
(test_demo_key_cannot_reach_admin).
"""

import pytest
from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.admin_routes import list_uncertain, submit_label
from src.core.enums import FeedbackLabel, PredictionLabel
from src.infra.prediction_log import PredictionLog
from src.schemas.label import LabelRequest


async def _log(
    session: AsyncSession,
    label: PredictionLabel,
    filename: str = "image.jpg",
    deleted: bool = False,
) -> PredictionLog:
    row = PredictionLog(
        filename=filename,
        confidence=0.6,
        predicted_label=label,
        processing_time_ms=10.0,
        is_deleted=deleted,
    )
    session.add(row)
    await session.flush()
    return row


def _label(log_id: int) -> LabelRequest:
    return LabelRequest(prediction_log_id=log_id, correct_label=FeedbackLabel.REAL)


# -- GET /v1/admin/uncertain --------------------------------------------------


async def test_uncertain_lists_only_uncertain_live_predictions(
    db_session: AsyncSession,
) -> None:
    await _log(db_session, PredictionLabel.REAL, "sure.jpg")
    await _log(db_session, PredictionLabel.UNCERTAIN_NEUTRAL, "unsure.jpg")
    await _log(
        db_session, PredictionLabel.UNCERTAIN_LEANING_AI, "gone.jpg", deleted=True
    )

    rows = await list_uncertain(limit=20, session=db_session)

    assert [r.filename for r in rows] == ["unsure.jpg"]


async def test_uncertain_respects_the_limit(db_session: AsyncSession) -> None:
    for i in range(3):
        await _log(db_session, PredictionLabel.UNCERTAIN_LEANING_REAL, f"u{i}.jpg")

    rows = await list_uncertain(limit=2, session=db_session)

    assert len(rows) == 2


# -- POST /v1/admin/label -----------------------------------------------------


async def test_a_label_is_saved(db_session: AsyncSession) -> None:
    row = await _log(db_session, PredictionLabel.UNCERTAIN_NEUTRAL)

    response = await submit_label(_label(int(row.id)), session=db_session)

    assert response.prediction_log_id == row.id
    assert response.correct_label == FeedbackLabel.REAL


async def test_a_second_label_is_refused_with_409(db_session: AsyncSession) -> None:
    row = await _log(db_session, PredictionLabel.UNCERTAIN_NEUTRAL)
    await submit_label(_label(int(row.id)), session=db_session)

    with pytest.raises(HTTPException) as exc:
        await submit_label(_label(int(row.id)), session=db_session)

    assert exc.value.status_code == 409


async def test_labelling_an_unknown_prediction_gives_404(
    db_session: AsyncSession,
) -> None:
    with pytest.raises(HTTPException) as exc:
        await submit_label(_label(999_999), session=db_session)

    assert exc.value.status_code == 404
