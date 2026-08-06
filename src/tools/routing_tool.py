"""
Routing Tool.

Returns driving and walking distance/duration between two coordinates.

Why OpenRouteService over Mapbox Directions / Google Directions:
- Free tier (2,000 requests/day) needs just a free account -- no credit
  card, unlike Google Directions, which requires billing enabled from
  day one even for its free monthly credit.
- Supports both driving-car and foot-walking profiles directly, which is
  exactly what this tool needs to return.
Trade-off: no live traffic data (Google's does have this) -- irrelevant
for trip *planning*, which is what this agent does; it's not turn-by-turn
navigation.

Design note: a failed walking route does NOT fail the whole call. Some
routes (e.g. mountain roads) genuinely have no walkable path OSM knows
about -- that's a real, expected outcome, not a bug. Walking failure is
recorded as a warning so the planner can still use the driving numbers;
only a failed *driving* route (the more fundamental case) raises.
"""

from typing import List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.services.api_clients import get_http_client
from src.services.retry import async_retry
from src.utils.exceptions import RoutingToolError
from src.utils.logger import get_logger

logger = get_logger(__name__)

DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions"

Coordinates = Tuple[float, float]  # (latitude, longitude)


class RouteResult(BaseModel):
    origin: Coordinates
    destination: Coordinates
    driving_distance_km: Optional[float] = None
    driving_duration_min: Optional[float] = None
    walking_distance_km: Optional[float] = None
    walking_duration_min: Optional[float] = None
    warnings: List[str] = Field(default_factory=list)


@async_retry()
async def _fetch_route(
    profile: str, origin: Coordinates, destination: Coordinates, api_key: str
) -> dict:
    client = get_http_client()
    lat1, lon1 = origin
    lat2, lon2 = destination
    response = await client.post(
        f"{DIRECTIONS_URL}/{profile}",
        headers={"Authorization": api_key, "Content-Type": "application/json"},
        # ORS wants [longitude, latitude] pairs, the opposite order of
        # everything else in this project -- easy bug to reintroduce, so
        # it's called out explicitly here.
        json={"coordinates": [[lon1, lat1], [lon2, lat2]]},
    )
    response.raise_for_status()
    return response.json()


def _extract_summary(data: dict) -> Tuple[float, float]:
    """Returns (distance_km, duration_min) from an ORS directions response."""
    summary = data["routes"][0]["summary"]
    return round(summary["distance"] / 1000, 2), round(summary["duration"] / 60, 1)


async def get_route(
    origin: Coordinates,
    destination: Coordinates,
    api_key: Optional[str] = None,
) -> RouteResult:
    """Get driving and walking distance/duration between two lat/lon points."""
    key = api_key or settings.routing_api_key
    if not key:
        raise RoutingToolError("ROUTING_API_KEY is not configured.")

    result = RouteResult(origin=origin, destination=destination)

    try:
        driving_data = await _fetch_route("driving-car", origin, destination, key)
        result.driving_distance_km, result.driving_duration_min = _extract_summary(driving_data)
    except httpx.HTTPStatusError as exc:
        raise RoutingToolError(
            f"OpenRouteService API returned an error: {exc.response.status_code}"
        ) from exc
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise RoutingToolError(f"OpenRouteService API unreachable: {exc}") from exc

    try:
        walking_data = await _fetch_route("foot-walking", origin, destination, key)
        result.walking_distance_km, result.walking_duration_min = _extract_summary(walking_data)
    except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as exc:
        warning = f"Walking route unavailable: {exc}"
        result.warnings.append(warning)
        logger.warning(
            "routing_tool: walking route failed for %s -> %s: %s", origin, destination, exc
        )

    logger.info(
        "routing_tool: %s -> %s driving=%s walking=%s",
        origin,
        destination,
        result.driving_distance_km,
        result.walking_distance_km,
    )
    return result
