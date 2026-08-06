"""
Shared geocoding service.

The Places, Weather, and Routing tools all need to turn a destination
name into coordinates. Centralizing that here means one Nominatim
integration and one retry policy, instead of three near-identical copies
scattered across tools -- and one place to swap providers later if needed.
"""

from typing import Tuple

from src.services.api_clients import get_http_client
from src.services.retry import async_retry

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Nominatim's usage policy requires an identifying User-Agent (no real
# contact email needed for low-volume educational use, but a generic
# browser-like UA would get us blocked).
NOMINATIM_USER_AGENT = "AutonomousTravelPlanningAgent/0.1 (student internship project)"


class GeocodingError(Exception):
    def __init__(self, destination: str):
        self.destination = destination
        super().__init__(f"Could not geocode destination '{destination}'.")


@async_retry()
async def geocode(destination: str) -> Tuple[float, float]:
    """Resolve a destination name to (latitude, longitude)."""
    client = get_http_client()
    response = await client.get(
        NOMINATIM_URL,
        params={"q": destination, "format": "json", "limit": 1},
        headers={"User-Agent": NOMINATIM_USER_AGENT},
    )
    response.raise_for_status()
    data = response.json()
    if not data:
        raise GeocodingError(destination)
    return float(data[0]["lat"]), float(data[0]["lon"])
