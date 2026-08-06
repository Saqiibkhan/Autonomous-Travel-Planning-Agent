"""Unit tests for src/services/geocoding.py."""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.services.geocoding import GeocodingError, geocode


def _response(payload) -> httpx.Response:
    return httpx.Response(
        200, json=payload, request=httpx.Request("GET", "https://nominatim.openstreetmap.org/search")
    )


async def test_geocode_success(mocker):
    mocker.patch(
        "httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=_response([{"lat": "36.3167", "lon": "74.65"}]),
    )
    lat, lon = await geocode("Hunza")
    assert lat == pytest.approx(36.3167)
    assert lon == pytest.approx(74.65)


async def test_geocode_not_found(mocker):
    mocker.patch(
        "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_response([])
    )
    with pytest.raises(GeocodingError, match="Could not geocode"):
        await geocode("Nowhereland")
