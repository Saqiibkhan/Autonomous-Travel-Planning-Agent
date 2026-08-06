"""
Unit tests for weather_tool.py.

get_weather() makes two GET calls in sequence: geocode (Nominatim), then
the forecast fetch (OpenWeatherMap). Both are mocked via a single ordered
side_effect list on httpx.AsyncClient.get.
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.weather_tool import get_weather
from src.utils.exceptions import WeatherToolError


def _response(payload) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", "https://example.com"))


NOMINATIM_PAYLOAD = [{"lat": "36.3167", "lon": "74.65"}]

FORECAST_PAYLOAD = {
    "list": [
        {"dt_txt": "2026-08-05 00:00:00", "main": {"temp": 15.0}, "weather": [{"main": "Clear"}], "pop": 0.1},
        {"dt_txt": "2026-08-05 12:00:00", "main": {"temp": 25.0}, "weather": [{"main": "Clear"}], "pop": 0.2},
        {"dt_txt": "2026-08-05 21:00:00", "main": {"temp": 18.0}, "weather": [{"main": "Clouds"}], "pop": 0.3},
        {"dt_txt": "2026-08-06 00:00:00", "main": {"temp": -1.0}, "weather": [{"main": "Snow"}], "pop": 0.8},
        {"dt_txt": "2026-08-06 12:00:00", "main": {"temp": 5.0}, "weather": [{"main": "Snow"}], "pop": 0.9},
    ]
}


async def test_get_weather_success(mocker):
    mock_get = mocker.patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    mock_get.side_effect = [_response(NOMINATIM_PAYLOAD), _response(FORECAST_PAYLOAD)]

    result = await get_weather("Hunza", duration_days=2, api_key="fake-key")

    assert result.destination == "Hunza"
    assert len(result.forecast) == 2
    assert result.notes == []  # duration fits within the 5-day window

    day1, day2 = result.forecast
    assert day1.date == "2026-08-05"
    assert day1.temperature_min_c == 15.0
    assert day1.temperature_max_c == 25.0
    assert day1.condition == "Clear"
    assert day1.warnings == []  # nothing extreme

    assert day2.date == "2026-08-06"
    assert day2.temperature_min_c == -1.0
    assert day2.rainfall_probability == 0.9
    assert any("rain" in w.lower() for w in day2.warnings)
    assert any("freezing" in w.lower() for w in day2.warnings)


async def test_get_weather_beyond_forecast_window_adds_note(mocker):
    mock_get = mocker.patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    mock_get.side_effect = [_response(NOMINATIM_PAYLOAD), _response(FORECAST_PAYLOAD)]

    result = await get_weather("Hunza", duration_days=10, api_key="fake-key")

    assert len(result.notes) == 1
    assert "5" in result.notes[0]


async def test_get_weather_missing_api_key(monkeypatch):
    monkeypatch.setattr("src.config.settings.weather_api_key", "")
    with pytest.raises(WeatherToolError, match="WEATHER_API_KEY"):
        await get_weather("Hunza", duration_days=3, api_key=None)


async def test_get_weather_invalid_duration():
    with pytest.raises(WeatherToolError, match="duration_days"):
        await get_weather("Hunza", duration_days=0, api_key="fake-key")


async def test_get_weather_geocode_fails(mocker):
    mocker.patch(
        "httpx.AsyncClient.get", new_callable=AsyncMock, return_value=_response([])
    )
    with pytest.raises(WeatherToolError, match="Could not geocode"):
        await get_weather("Nowhereland", duration_days=3, api_key="fake-key")


async def test_get_weather_forecast_unreachable_retries(mocker):
    mocker.patch("src.services.retry.asyncio.sleep", new_callable=AsyncMock)
    mock_get = mocker.patch("httpx.AsyncClient.get", new_callable=AsyncMock)
    mock_get.side_effect = [
        _response(NOMINATIM_PAYLOAD),
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
        httpx.TimeoutException("timed out"),
    ]

    with pytest.raises(WeatherToolError, match="unreachable"):
        await get_weather("Hunza", duration_days=3, api_key="fake-key")

    assert mock_get.call_count == 4  # 1 geocode + 3 forecast attempts
