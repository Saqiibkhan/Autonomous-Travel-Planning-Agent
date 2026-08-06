"""
Unit tests for places_tool.py.

Nominatim (GET, geocoding) and Overpass (POST, POI query) are mocked
separately -- the tool makes one call to each. No real network access
required.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.places_tool import find_places
from src.utils.exceptions import PlacesToolError


def _json_response(method: str, url: str, payload) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request(method, url))


NOMINATIM_PAYLOAD = [{"lat": "36.3167", "lon": "74.65"}]

OVERPASS_PAYLOAD = {
    "elements": [
        {
            "type": "node",
            "lat": 36.318,
            "lon": 74.652,
            "tags": {"name": "Baltit Fort", "historic": "castle"},
        },
        {
            "type": "node",
            "lat": 36.32,
            "lon": 74.66,
            "tags": {"name": "Cafe de Hunza", "amenity": "cafe"},
        },
        {
            "type": "node",
            "lat": 36.30,
            "lon": 74.64,
            "tags": {"name": "Hunza Museum", "tourism": "museum"},
        },
        {
            # No name tag -- should be skipped
            "type": "node",
            "lat": 36.31,
            "lon": 74.63,
            "tags": {"tourism": "attraction"},
        },
    ]
}


async def test_find_places_success(mocker):
    mocker.patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=_json_response(
            "GET", "https://nominatim.openstreetmap.org/search", NOMINATIM_PAYLOAD
        ),
    )
    mocker.patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=_json_response(
            "POST", "https://overpass-api.de/api/interpreter", OVERPASS_PAYLOAD
        ),
    )

    result = await find_places("Hunza")

    assert result.destination == "Hunza"
    assert result.latitude == pytest.approx(36.3167)
    assert result.longitude == pytest.approx(74.65)
    # The unnamed node must be filtered out
    assert len(result.places) == 3

    names = {p.name: p.category for p in result.places}
    assert names["Baltit Fort"] == "landmark"
    assert names["Cafe de Hunza"] == "restaurant"
    assert names["Hunza Museum"] == "museum"


async def test_find_places_empty_destination():
    with pytest.raises(PlacesToolError, match="non-empty"):
        await find_places("   ")


async def test_find_places_geocode_fails(mocker):
    mocker.patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=_json_response(
            "GET", "https://nominatim.openstreetmap.org/search", []
        ),
    )

    with pytest.raises(PlacesToolError, match="Could not geocode"):
        await find_places("Nowhereland")


async def test_find_places_no_results_near_destination(mocker):
    mocker.patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=_json_response(
            "GET", "https://nominatim.openstreetmap.org/search", NOMINATIM_PAYLOAD
        ),
    )
    mocker.patch(
        "httpx.AsyncClient.post",
        new_callable=AsyncMock,
        return_value=_json_response(
            "POST", "https://overpass-api.de/api/interpreter", {"elements": []}
        ),
    )

    with pytest.raises(PlacesToolError, match="No places found"):
        await find_places("Hunza")


async def test_find_places_overpass_timeout_retries(mocker):
    mocker.patch("src.services.retry.asyncio.sleep", new_callable=AsyncMock)
    mocker.patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=_json_response(
            "GET", "https://nominatim.openstreetmap.org/search", NOMINATIM_PAYLOAD
        ),
    )
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.side_effect = httpx.TimeoutException("timed out")

    with pytest.raises(PlacesToolError, match="unreachable"):
        await find_places("Hunza")

    assert mock_post.call_count == 3  # default max_tool_retries
