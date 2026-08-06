"""
Places Tool.

Returns attractions, restaurants, museums, and landmarks near a
destination, each with name, category, coordinates, and address when
available.

Why OpenStreetMap (Nominatim + Overpass) over Google Places:
- Zero API key, zero billing setup. Google Places' free tier requires a
  Google Cloud project with billing enabled even to get the free credits
  -- a real barrier for a student project with no card on file.
- Overpass lets us query multiple categories (tourism attractions,
  museums, historic landmarks, restaurants) in a single request via
  Overpass QL, instead of separate paid calls per category.
Trade-off: no star ratings or photos (OSM doesn't track those), and data
completeness varies by region -- well-mapped cities are great, remote
areas can be sparse. `rating` is kept as an optional field on PlaceItem
so a Google Places fallback (using PLACES_API_KEY, currently unused)
could be added later without changing the tool's interface.
Documented properly in docs/api_research.md (Task 7).
"""

from typing import List, Optional

import httpx
from pydantic import BaseModel, Field

from src.services.api_clients import get_http_client
from src.services.geocoding import GeocodingError, geocode
from src.services.retry import async_retry
from src.utils.exceptions import PlacesToolError
from src.utils.logger import get_logger

logger = get_logger(__name__)

OVERPASS_URL = "https://lz4.overpass-api.de/api/interpreter"
OVERPASS_USER_AGENT = "AutonomousTravelPlanningAgent/0.1 (student internship project)"

DEFAULT_RADIUS_METERS = 15_000
MAX_PLACES = 40


class PlaceItem(BaseModel):
    name: str
    category: str  # "attraction" | "restaurant" | "museum" | "landmark"
    latitude: float
    longitude: float
    address: Optional[str] = None
    rating: Optional[float] = None  # OSM has no ratings; reserved for a future API fallback


class PlacesResult(BaseModel):
    destination: str
    latitude: float
    longitude: float
    places: List[PlaceItem] = Field(default_factory=list)


@async_retry()
async def _query_overpass(lat: float, lon: float, radius_m: int) -> dict:
    query = f"""
    [out:json][timeout:25];
    (
      node["tourism"="attraction"](around:{radius_m},{lat},{lon});
      node["tourism"="museum"](around:{radius_m},{lat},{lon});
      node["tourism"="viewpoint"](around:{radius_m},{lat},{lon});
      node["historic"](around:{radius_m},{lat},{lon});
      node["amenity"="restaurant"](around:{radius_m},{lat},{lon});
      node["amenity"="cafe"](around:{radius_m},{lat},{lon});
    );
    out center {MAX_PLACES};
    """
    client = get_http_client()
    response = await client.post(
        OVERPASS_URL,
        data={"data": query},
        headers={"User-Agent": OVERPASS_USER_AGENT},
    )
    response.raise_for_status()
    return response.json()


def _categorize(tags: dict) -> str:
    if tags.get("amenity") in ("restaurant", "cafe"):
        return "restaurant"
    if tags.get("tourism") == "museum":
        return "museum"
    if "historic" in tags:
        return "landmark"
    return "attraction"


def _build_address(tags: dict) -> Optional[str]:
    parts = [
        tags.get(key)
        for key in ("addr:housenumber", "addr:street", "addr:city")
        if tags.get(key)
    ]
    return ", ".join(parts) if parts else None


async def find_places(
    destination: str, radius_meters: int = DEFAULT_RADIUS_METERS
) -> PlacesResult:
    """
    Find attractions, restaurants, museums, and landmarks near a destination.

    Two-step process: geocode the destination name into coordinates via
    Nominatim, then query points of interest around those coordinates
    via Overpass. Nodes without a `name` tag are skipped -- an unnamed
    point on a map isn't a usable recommendation.
    """
    if not destination or not destination.strip():
        raise PlacesToolError("destination must be a non-empty string.")

    try:
        lat, lon = await geocode(destination)
        overpass_data = await _query_overpass(lat, lon, radius_meters)
    except GeocodingError as exc:
        raise PlacesToolError(str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise PlacesToolError(
            f"OpenStreetMap API returned an error: {exc.response.status_code}"
        ) from exc
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise PlacesToolError(f"OpenStreetMap API unreachable: {exc}") from exc

    places: List[PlaceItem] = []
    for element in overpass_data.get("elements", []):
        tags = element.get("tags", {})
        name = tags.get("name")
        if not name:
            continue

        place_lat = element.get("lat") or element.get("center", {}).get("lat")
        place_lon = element.get("lon") or element.get("center", {}).get("lon")
        if place_lat is None or place_lon is None:
            continue

        places.append(
            PlaceItem(
                name=name,
                category=_categorize(tags),
                latitude=place_lat,
                longitude=place_lon,
                address=_build_address(tags),
                rating=None,
            )
        )

    if not places:
        raise PlacesToolError(f"No places found near '{destination}'.")

    logger.info("places_tool: destination=%s found=%d places", destination, len(places))
    return PlacesResult(destination=destination, latitude=lat, longitude=lon, places=places)
