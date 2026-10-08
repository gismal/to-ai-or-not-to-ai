from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.rate_limits import demo_limit
from src.config import settings


def _client(scope: str, limit: int) -> TestClient:
    app = FastAPI()

    @app.get("/x", dependencies=[Depends(demo_limit(scope, limit))])
    def x():
        return {"ok": True}

    return TestClient(app)


def test_demo_key_is_limited_per_window():
    c = _client("t_limited", 2)
    h = {"X-API-Key": settings.DEMO_API_KEY}
    codes = [c.get("/x", headers=h).status_code for _ in range(3)]

    assert codes == [200, 200, 429]
    assert "retry-after" in c.get("/x", headers=h).headers


def test_real_key_is_not_limited():
    c = _client("t_real", 2)
    h = {"X-API-Key": settings.API_KEY.get_secret_value()}
    assert all(c.get("/x", headers=h).status_code == 200 for _ in range(10))


def test_bad_key_is_rejected_before_counting():
    c = _client("t_bad", 2)
    assert c.get("/x", headers={"X-API-Keys": "nope"}).status_code == 401
