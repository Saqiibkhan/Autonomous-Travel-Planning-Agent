"""
Weather Tool.

Returns a day-by-day forecast (temperature range, rainfall probability,
condition, and any warnings) for the trip.

Why OpenWeatherMap's free "5 day / 3 hour forecast" endpoint over
WeatherAPI.com / OWM's own One Call 3.0:
- It needs only a free API key, no credit card. OWM's newer One Call 3.0
  now requires a payment method on file even to use its free monthly
  quota -- a dealbreaker for a student project.
- It already returns `pop` (probability of precipitation) per 3-hour
  block, which is exactly the "rainfall probability" field this tool
  must return -- no extra calls or estimation needed.
Trade-off: this endpoint only forecasts 5 days out. For trips longer than
that, days beyond the window are simply not returned, and a note is
attached explaining why -- better to say "we don't know" than to fake a
forecast for three weeks from now, which no free API can actually do.
"""

import statistics
from collections import defaultdict
from typing import Dict, List, Optional

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.services.api_clients import get_http_client
from src.services.geocoding import GeocodingError, geocode
from src.services.retry import async_retry
from src.utils.exceptions import WeatherToolError
from src.utils.logger import get_logger

logger = get_logger(__name__)

FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
MAX_FORECAST_DAYS = 5

HIGH_RAIN_PROBABILITY_THRESHOLD = 0.5
EXTREME_COLD_C = 0.0
EXTREME_HOT_C = 38.0


class DailyWeather(BaseModel):
    date: str
    temperature_min_c: float
    temperature_max_c: float
    condition: str
    rainfall_probability: float  # 0.0-1.0
    warnings: List[str] = Field(default_factory=list)


class WeatherResult(BaseModel):
    destination: str
    forecast: List[DailyWeather] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


@async_retry()
async def _fetch_forecast(lat: float, lon: float, api_key: str) -> dict:
    client = get_http_client()
    response = await client.get(
        FORECAST_URL,
        params={"lat": lat, "lon": lon, "appid": api_key, "units": "metric"},
    )
    response.raise_for_status()
    return response.json()


def _warnings_for_day(temp_min: float, temp_max: float, rain_prob: float, condition: str) -> List[str]:
    warnings: List[str] = []
    if rain_prob >= HIGH_RAIN_PROBABILITY_THRESHOLD:
        warnings.append(f"High chance of rain ({rain_prob:.0%}) -- plan an indoor alternative.")
    if temp_min <= EXTREME_COLD_C:
        warnings.append(f"Near/below freezing overnight ({temp_min:.1f}°C) -- pack warm layers.")
    if temp_max >= EXTREME_HOT_C:
        warnings.append(f"Extreme heat expected ({temp_max:.1f}°C) -- avoid midday outdoor activity.")
    if "storm" in condition.lower() or "thunder" in condition.lower():
        warnings.append("Thunderstorms possible -- keep outdoor plans flexible.")
    return warnings


async def get_weather(
    destination: str, duration_days: int, api_key: Optional[str] = None
) -> WeatherResult:
    """
    Get a day-by-day forecast for a destination, up to the first
    min(duration_days, 5) days -- the limit of OWM's free forecast window.
    """
    key = api_key or settings.weather_api_key
    if not key:
        raise WeatherToolError("WEATHER_API_KEY is not configured.")
    if not destination or not destination.strip():
        raise WeatherToolError("destination must be a non-empty string.")
    if duration_days <= 0:
        raise WeatherToolError("duration_days must be a positive integer.")

    try:
        lat, lon = await geocode(destination)
        data = await _fetch_forecast(lat, lon, key)
    except GeocodingError as exc:
        raise WeatherToolError(str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise WeatherToolError(
            f"OpenWeatherMap API returned an error: {exc.response.status_code}"
        ) from exc
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise WeatherToolError(f"OpenWeatherMap API unreachable: {exc}") from exc

    entries = data.get("list", [])
    if not entries:
        raise WeatherToolError(f"No forecast data returned for '{destination}'.")

    # OWM gives 3-hour blocks; group them into calendar days.
    by_date: Dict[str, list] = defaultdict(list)
    for entry in entries:
        day = entry["dt_txt"].split(" ")[0]
        by_date[day].append(entry)

    sorted_days = sorted(by_date.keys())[: min(duration_days, MAX_FORECAST_DAYS)]

    forecast: List[DailyWeather] = []
    for day in sorted_days:
        day_entries = by_date[day]
        temps = [e["main"]["temp"] for e in day_entries]
        pops = [e.get("pop", 0.0) for e in day_entries]
        conditions = [e["weather"][0]["main"] for e in day_entries if e.get("weather")]

        dominant_condition = statistics.mode(conditions) if conditions else "Unknown"
        temp_min, temp_max = min(temps), max(temps)
        rain_prob = max(pops) if pops else 0.0

        forecast.append(
            DailyWeather(
                date=day,
                temperature_min_c=round(temp_min, 1),
                temperature_max_c=round(temp_max, 1),
                condition=dominant_condition,
                rainfall_probability=round(rain_prob, 2),
                warnings=_warnings_for_day(temp_min, temp_max, rain_prob, dominant_condition),
            )
        )

    notes: List[str] = []
    if duration_days > MAX_FORECAST_DAYS:
        notes.append(
            f"Forecast only covers the first {MAX_FORECAST_DAYS} of {duration_days} "
            f"trip days (free-tier limit) -- plan remaining days using seasonal norms "
            f"from the Search Tool rather than a specific forecast."
        )

    logger.info(
        "weather_tool: destination=%s days_returned=%d notes=%d",
        destination,
        len(forecast),
        len(notes),
    )
    return WeatherResult(destination=destination, forecast=forecast, notes=notes)
