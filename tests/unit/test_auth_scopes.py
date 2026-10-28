"""
Unit test case for the authentication scopes.
This test suite challanges the authentication system by verifying that different types of API keys (real and demo) and
fifferent situations are correctly handled.
"""

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from src.api.deps import verify_api_key, verify_public_key
from src.config import settings

REAL = {"X-API-Key": settings.API_KEY.get_secret_value()}
DEMO = {"X-API-Key": settings.DEMO_API_KEY}


@pytest.fixture
def client() -> TestClient:
    """
    This fixture sets up a FastAPI test client with two routes: one for admin access and one for public access.
    The admin route requires a valid API key, while the public route requires a valid demo key
    """

    app = FastAPI()

    @app.get("/admin", dependencies=[Depends(verify_api_key)])
    def real():
        return {"ok": True}

    @app.get("/public", dependencies=[Depends(verify_public_key)])
    def demo():
        return {"ok": True}

    return TestClient(app)


def test_real_key_opens_both_routes(client):
    """
    Testing a real key has access to both the admin and public routes, ensuring that users with
    valid credentials can access all necessary endpoints. This test verifies that the authentication system correctly differentiates
    between real and demo keys, allowing full access for authorized users while maintaining security for restricted areas.

    Args:
        client (TestClient): A test client instance for making requests to the FastAPI application.

    Assertions:
        - The response status code is 200 for both admin and public routes, means successful access
    """
    assert client.get("/admin", headers=REAL).status_code == 200
    assert client.get("/public", headers=REAL).status_code == 200


def test_demo_key_opens_only_public_route(client):
    """
    Testing a demo key has access to only the public route. This test ensures that users with demo credentials are, restricted, maintaining the integrity of the system's access control.
    It verifies that the authentication mechanism correctly enforces access limitations based on the type of API key provided, allowing for a safe and controlled user experience.
    """
    assert client.get("/admin", headers=DEMO).status_code == 401
    assert client.get("/public", headers=DEMO).status_code == 200


@pytest.mark.parametrize("scope", ["/admin", "/public"])
def test_missing_header_is_rejected(client, scope):
    """
    Testing requests that are missing the API key header are rejected for both admin and public routes.

    Args:
        client (TestClient): A test client instance for making requests to the FastAPI application.
        scope (str): The route scope to test.

    Assertions:
        - The response status code is either 401 or 403, indicating unauthorized access due to missing credentials.
    """

    assert client.get(scope).status_code in (401, 403)


@pytest.mark.parametrize("scope", ["/admin", "/public"])
@pytest.mark.parametrize("key", ["wrong", " ", ""])
def test_bad_key_is_rejected(client, scope, key):
    """
    Testing the invalid API keys are rejected for both admin and public routes.
    This test ensures that the authentication system correctly identifies and denies access to requests with incorrect or malformed API keys,
    maintaining the security and integrity of the application. It verifies that only valid credentials are accepted.

    Args:
        client (TestClient): A test client instance for making requests to the FastAPI application.
        scope (str): The route scope to test.
        key (str): The API key to test.

    Asssertions:
        - The response status code is 401 for invalid API keys, prepresenting unauthorized access.
    """
    assert client.get(scope, headers={"X-API-Key": key}).status_code == 401


def test_empty_demo_key_switches_demo_access_off(client, monkeypatch):
    """
    Testing a empty demo key disables access to the public route.
    This test ensures that when the demo API key is not set, the system correctly denies access to the public endpoint.
    It verifies that the application behaves as expected in scenarios where demo access is intentionally disabled, maintaining security and proper access control.
    Args:
        client (TestClient): A test client instance for making requests to the FastAPI application.
        monkeypatch: A pytest fixture for temporarily modifying settings during the test.

    Asssertions:
        - The public route returns a 401 status code when the demo key is empty.
    """

    monkeypatch.setattr(settings, "DEMO_API_KEY", "")
    assert client.get("/public", headers={"X-API-Key": ""}).status_code == 401
    assert client.get("/public", headers=DEMO).status_code == 401
    assert client.get("/public", headers=REAL).status_code == 200
