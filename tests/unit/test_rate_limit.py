"""
This unit test modules is testing the rate limiting functionality of the API.
It ensures that the defined rate limits are correctly enforces for both demo and real API keys and the unauthorized access is properly handled
"""

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.rate_limits import demo_limit
from src.config import settings


def _client(scope: str, limit: int) -> TestClient:
    """
    Setting up a FastAPI test client with a single route that is protected by the demo_limit dependency
    """
    app = FastAPI()

    @app.get("/x", dependencies=[Depends(demo_limit(scope, limit))])
    def x():
        return {"ok": True}

    return TestClient(app)


def test_demo_key_is_limited_per_window():
    """
    Testing the rate limiting for demo API keys.
    This test ensures that the defined rate limits are correctly enforced, allowing only a specified number of requests within a given time window.

    Assertions:
        - The first two requests return a 200 status code, indicating successful access.
        - The third request returns a 429 status code, indicating that the rate limit has been exceeded.
        - The response headers include a "retry-after" field, indicating when the client can
    """

    c = _client("t_limited", 2)
    h = {"X-API-Key": settings.DEMO_API_KEY}
    codes = [c.get("/x", headers=h).status_code for _ in range(3)]

    assert codes == [200, 200, 429]
    assert "retry-after" in c.get("/x", headers=h).headers


def test_real_key_is_not_limited():
    """
    Testing how users with real API keys are not subject to the same rate limiting as demo users.

    Args:
        scope (str): The scope of the API key (e.g., "t_real", "t_limited")

    Assertions:
        - All requests return a 200 status code, indicating successful access without any rate limiting.
    """
    c = _client("t_real", 2)
    h = {"X-API-Key": settings.API_KEY.get_secret_value()}
    assert all(c.get("/x", headers=h).status_code == 200 for _ in range(10))


def test_bad_key_is_rejected_before_counting():
    """
    Testing handling of unauthorized access attempts with invalid API keys.
    This test ensures that requests with invalid API keys are rejected before they are counted towards the rate limit.

    Assertions:
        - The response status code is 401 for unauthorized access.
    """

    c = _client("t_bad", 2)
    assert c.get("/x", headers={"X-API-Keys": "nope"}).status_code == 401
