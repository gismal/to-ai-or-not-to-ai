"""
This test module tests the retraining logic. It simulates fake ARQ pool to test enqueue method.
It defines the prediction and feedback labels and error types to be tested.


"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.config import settings
from src.core.enums import ErrorType, FeedbackLabel, PredictionLabel
from src.services.retrain_service import RetrainService

CORRECT = (PredictionLabel.REAL, FeedbackLabel.REAL, ErrorType.CORRECT)
FALSE_POSITIVE = (
    PredictionLabel.AI_GENERATED,
    FeedbackLabel.REAL,
    ErrorType.FALSE_POSITIVE,
)
FALSE_NEGATIVE = (
    PredictionLabel.REAL,
    FeedbackLabel.AI_GENERATED,
    ErrorType.FALSE_NEGATIVE,
)
UNKNOWN = (PredictionLabel.UNCERTAIN_NEUTRAL, FeedbackLabel.UNKNOWN, ErrorType.UNKNOWN)


class FakePool:
    """
    Simulates a fake ARQ pool for testing the enqueue_job method of the RetrainService class.
    Tests the enqueue_job with a fake job ID and records the calls.
    Attributes:
        job_id (str): A fake job ID to simulate the enqueued job.
        calls (list): A list to record the calls made to enqueue_job. It stores the name and kwargs of each call.
    Methods:
        enqueue_job(name, **kwargs): Simulates enqueuing a job by recording the call and returning a SimpleNamespace with the fake job ID.
    """

    def __init__(self, job_id="fake_job_id"):
        self.job_id, self.calls = job_id, []

    async def enqueue_job(self, name, **kwargs):
        self.calls.append((name, kwargs))
        return SimpleNamespace(job_id=self.job_id) if self.job_id else None


@pytest.fixture(autouse=True)
def drift_settings(monkeypatch):
    """
    Sets up the environment for drift detection tests.

    Args:
        monkeypatch: Sets the DRIFT_THRESHOLD and DRIFT_MIN_SAMPLES settings to specific values for testing purposes.
    """
    monkeypatch.setattr(settings, "DRIFT_THRESHOLD", 0.15)
    monkeypatch.setattr(settings, "DRIFT_MIN_SAMPLES", 10)


async def add(repo, n, kind, tag="x"):
    """
    Add feedback items to repo for testing drift detection. It creates feedback items with prediction, truth and error type (provided in kind)
    The filename is generated using the tag and error type, and the items are created with a confidence of 0.9.

    Args:
        repo (FeedbackRepositories): Repo for the test session
        n (int): Number of rows to create
        kind (tuple): It has predictive label, user's answer, error type respectively

    Returns:
        list[FeedbackItem] same kind of feedback
    """

    prediction, truth, error = kind
    return [
        await repo.create_feedback(
            filename=f"{tag}_{error.value}{i}.jpg",
            model_prediction=prediction,
            confidence=0.9,
            user_correction=truth,
            error_type=error,
        )
        for i in range(n)
    ]


async def test_error_rate_below_threshold_is_ok(db_session, feedback_repo):
    """
    Testing a scenario where the error rate is below the configured threshold for drift detection.
    This test ensures that the RetrainService correctly identifies that the error rate is acceptable and does not trigger a retraining job.

    Args:
        db_session (AsyncSession): Acts like a database session
        feedback_repo (FeedbackRepository): Ready to use repository to test the threshold

    Assertions:
        - Result is passed as okay
        - Result has 0.1 error rate
        - Nothing is queued
    """
    await add(feedback_repo, 18, CORRECT)
    await add(feedback_repo, 2, FALSE_POSITIVE)
    pool = FakePool()
    result = await RetrainService(db_session, pool).check_drift_and_trigger()  # type: ignore
    assert result["status"] == "ok" and result["error_rate"] == 0.1 and not pool.calls


async def test_drift_enqueues_one_job_with_a_stable_id(db_session, feedback_repo):
    """
    Testing the enqueue behaviour with a job with one stable ID.

    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertions:
        Results are enqueued for retraining
        Jod_id set for job-42 (42 is the answer to everything)
    """
    await add(feedback_repo, 16, CORRECT)
    await add(feedback_repo, 4, FALSE_NEGATIVE)
    pool = FakePool("job-42")
    result = await RetrainService(db_session, pool).check_drift_and_trigger()  # type: ignore
    assert result["status"] == "retraining_enqueued" and result["job_id"] == "job-42"
    assert pool.calls == [("retrain_model", {"_job_id": "retrain_model"})]


async def test_unknown_answers_do_not_dilute_the_error_rate(db_session, feedback_repo):
    """
    Testing if the unknown answers can find a way into error rate calculations. This test will create a scenario that is correct, false positive and unknown entries.
    The entries are sent to a session to control whether they are trigger the drift control.

    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertions:
        Error rates of the results are set to 0.1

    """

    await add(feedback_repo, 18, CORRECT)
    await add(feedback_repo, 2, FALSE_POSITIVE)
    await add(feedback_repo, 200, UNKNOWN)

    result = await RetrainService(db_session, FakePool()).check_drift_and_trigger()  # type: ignore
    assert result["error_rate"] == 0.1


async def test_a_handful_of_false_positive_cannot_trigger_retraining(
    db_session, feedback_repo
):
    """
    Testing the scenario if a bunch of (10) false positive results can trigger a retraining.

    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertions:
        Insufficient results
        10 sample results
        Don't evoke the pool calls
    """

    await add(feedback_repo, settings.DRIFT_MIN_SAMPLES - 1, FALSE_POSITIVE)
    pool = FakePool()
    result = await RetrainService(db_session, pool).check_drift_and_trigger()  # type: ignore
    assert (
        result["status"] == "insufficient_data"
        and result["samples"] == settings.DRIFT_MIN_SAMPLES - 1
        and not pool.calls
    )


async def test_old_feedback_is_outside_the_window(db_session, feedback_repo):
    """
    Testing the case if the old feedback would  join the retraining.

    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertions:
        Results are passed as ok
        Error rate set to 0
    """

    old = await add(feedback_repo, 20, FALSE_NEGATIVE, tag="old")
    for item in old:
        item.created_at = datetime.now(timezone.utc) - timedelta(days=30)
    await db_session.flush()
    await add(feedback_repo, 10, CORRECT)
    result = await RetrainService(db_session, FakePool()).check_drift_and_trigger()  # type: ignore
    assert result["status"] == "ok" and result["error_rate"] == 0.0


async def test_soft_deleted_feedback_is_ignored(db_session, feedback_repo):
    """
    Testing the case when soft deleted entries are ignored for retraining. This ensures none of the soft deleted feedback isn't counted in the error rate.
    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertions:
        Results passed as ok.
        Error rate is 0
    """

    await add(feedback_repo, 10, CORRECT)
    for item in await add(feedback_repo, 20, FALSE_NEGATIVE, tag="gone"):
        item.is_deleted = True
    await db_session.flush()
    result = await RetrainService(db_session, FakePool()).check_drift_and_trigger()  # type: ignore
    assert result["status"] == "ok" and result["error_rate"] == 0.0


async def test_already_queued_job_is_not_an_error(db_session, feedback_repo):
    """
    Tries to queue a job that's already there.
    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertions:
        Result is already queued
    """

    await add(feedback_repo, settings.DRIFT_MIN_SAMPLES, FALSE_POSITIVE)
    result = await RetrainService(
        db_session,
        FakePool(job_id=None),  # type: ignore
    ).check_drift_and_trigger()
    assert result["status"] == "already_queued"


async def test_missing_job_queue_is_reported_not_raised(db_session, feedback_repo):
    """
    Testing the logic with a missing job queue to see if they will be reported as error.

    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertion:
        Result stastus passed as error
    """

    await add(feedback_repo, 10, FALSE_POSITIVE)
    result = await RetrainService(db_session, None).check_drift_and_trigger()  # type: ignore
    assert result["status"] == "error"


async def test_exactly_the_minimum_is_enough(db_session, feedback_repo):
    """
    Testing the  minimum limit of the samples if retraining would queue them when they are exactly bare minimum.

    Args:
        db_session (AsyncSession): Acts like database session
        feedback_repo (FeedbackRepository): Ready to use repository for testing the drift.

    Assertion:
        Result passed as enqueued for retraining

    """
    await add(feedback_repo, settings.DRIFT_MIN_SAMPLES, FALSE_POSITIVE)
    result = await RetrainService(db_session, FakePool()).check_drift_and_trigger()  # type: ignore
    assert result["status"] == "retraining_enqueued"
