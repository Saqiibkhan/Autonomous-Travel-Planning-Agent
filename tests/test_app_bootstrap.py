"""
Task 1 smoke tests.

These don't test any travel-planning logic yet (there isn't any) -- they
prove the scaffolding itself is sound: settings load with sane defaults,
the FastAPI app builds, and /health + /status respond correctly. Every
later task builds on this; if this file ever breaks, nothing else will work.
"""

from fastapi.testclient import TestClient

from src.config import Settings
from src.main import app

client = TestClient(app)


def test_settings_load_with_defaults():
    s = Settings(_env_file=None)  # ignore any local .env for this test
    assert s.app_name == "Autonomous Travel Planning Agent"
    assert s.llm_model == "openai/gpt-oss-20b:free"
    assert s.max_tool_retries == 3
    assert s.max_reflection_retries == 3


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_status_endpoint_shape():
    response = client.get("/status")
    assert response.status_code == 200
    body = response.json()

    assert "app_name" in body
    assert "version" in body
    assert "configured_keys" in body

    # Keys must be booleans (presence flags), never the actual secret values
    for key_name, is_configured in body["configured_keys"].items():
        assert isinstance(is_configured, bool)
