"""
This test module debugs the whole file from the retraining.

"""

import inspect
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel
from src.infra.feedback_item import FeedbackItem
from src.services import retrain_service


async def test_debug(db_session, feedback_repo):
    """
    Debugs the file row by row, controls, counts errors, deleted items, timing.

    args:
        db_sessions (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing.

    Returns:
        Prints the count results

    """

    print("\nservice file:", retrain_service.__file__)
    print(
        "old query still present:",
        "PredictionLog" in inspect.getsource(retrain_service),
    )

    for i in range(3):
        await feedback_repo.create_feedback(
            filename=f"d{i}.png",
            model_prediction=PredictionLabel.REAL,
            confidence=0.1,
            user_correction=FeedbackLabel.REAL,
            error_type=ErrorType.CORRECT,
        )

    cutoff = datetime.now(timezone.utc) - timedelta(days=7)

    def count(*w):
        return db_session.scalar(select(func.count(FeedbackItem.id)).where(*w))

    print("all rows       :", await count())
    print("not deleted    :", await count(FeedbackItem.is_deleted.is_(False)))
    print("in time window :", await count(FeedbackItem.created_at >= cutoff))
    print("not UNKNOWN    :", await count(FeedbackItem.error_type != ErrorType.UNKNOWN))
    row = (await db_session.execute(select(FeedbackItem))).scalars().first()
    print("sample created_at / is_deleted:", repr(row.created_at), repr(row.is_deleted))
