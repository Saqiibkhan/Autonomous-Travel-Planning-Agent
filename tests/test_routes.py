"""
Unit tests for routing_tool.py.

get_route() makes two POST calls: driving-car, then foot-walking. Both
are mocked via httpx.AsyncClient.post.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.routing_tool import get_route
from src.utils.exceptions import RoutingToolError


def _response(payload) -> httpx.Response:
    return httpx.Response(
        200, json=payload, request=httpx.Request("POST", "https://api.openrouteservice.org")
    )


DRIVING_PAYLOAD = {"routes": [{"summary": {"distance": 12500.0, "duration": 900.0}}]}  # 12.5km, 15min
WALKING_PAYLOAD = {"routes": [{"summary": {"distance": 5000.0, "duration": 3600.0}}]}  # 5km, 60min

ORIGIN = (36.30, 74.65)
DESTINATION = (36.32, 74.66)


async def test_get_route_success(mocker):
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.side_effect = [_response(DRIVING_PAYLOAD), _response(WALKING_PAYLOAD)]

    result = await get_route(ORIGIN, DESTINATION, api_key="fake-key")

    assert result.driving_distance_km == 12.5
    assert result.driving_duration_min == 15.0
    assert result.walking_distance_km == 5.0
    assert result.walking_duration_min == 60.0
    assert result.warnings == []
    assert mock_post.call_count == 2


async def test_get_route_missing_api_key(monkeypatch):
    monkeypatch.setattr("src.config.settings.routing_api_key", "")
    with pytest.raises(RoutingToolError, match="ROUTING_API_KEY"):
        await get_route(ORIGIN, DESTINATION, api_key=None)


async def test_get_route_driving_failure_raises(mocker):
    mocker.patch("src.services.retry.asyncio.sleep", new_callable=AsyncMock)
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    error_response = httpx.Response(
        500, json={"error": "server error"}, request=httpx.Request("POST", "https://api.openrouteservice.org")
    )
    mock_post.return_value = error_response

    with pytest.raises(RoutingToolError, match="500"):
        await get_route(ORIGIN, DESTINATION, api_key="fake-key")


async def test_get_route_walking_failure_degrades_gracefully(mocker):
    mocker.patch("src.services.retry.asyncio.sleep", new_callable=AsyncMock)
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.side_effect = [
        _response(DRIVING_PAYLOAD),
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
    ]

    result = await get_route(ORIGIN, DESTINATION, api_key="fake-key")

    # Driving succeeded, so the call as a whole does NOT raise.
    assert result.driving_distance_km == 12.5
    assert result.walking_distance_km is None
    assert len(result.warnings) == 1
    assert "Walking route unavailable" in result.warnings[0]
