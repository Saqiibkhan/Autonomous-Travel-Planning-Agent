"""
Search Tool.

Returns general destination research: an overview, cultural notes, the
best season to visit, practical travel tips, and attraction names
mentioned in the write-ups it finds. Coordinates/categories for those
attractions come from the Places Tool, not this one -- this tool answers
"what should I know about this place", not "where exactly is it".

Why Tavily over Serper.dev / DuckDuckGo:
- Tavily returns a synthesized `answer` field built from multiple sources,
  not just a list of links. That means this tool can stay a *pure* data
  tool -- no LLM call needed here just to summarize search results. Serper
  and DuckDuckGo would hand back raw snippets that something would then
  have to summarize (likely burning an LLM call per search).
- Free tier: 1,000 searches/month, no credit card required -- plenty for
  a project like this.
- Simple REST API, API-key auth, no SDK lock-in.
Trade-off: it does require signing up for a key (unlike DuckDuckGo, which
needs none at all) -- documented properly in docs/api_research.md (Task 7).
"""

import re
from typing import List, Optional

import httpx
from pydantic import BaseModel, Field

from src.config import settings
from src.services.api_clients import get_http_client
from src.services.retry import async_retry
from src.utils.exceptions import SearchToolError
from src.utils.logger import get_logger

logger = get_logger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


class SearchResult(BaseModel):
    destination: str
    overview: str
    culture: str
    best_season: str
    travel_tips: List[str] = Field(default_factory=list)
    attractions_mentioned: List[str] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)


@async_retry()
async def _tavily_query(query: str, api_key: str) -> dict:
    """One Tavily call. Retried automatically on timeout/connection/HTTP errors."""
    client = get_http_client()
    response = await client.post(
        TAVILY_URL,
        json={
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "include_answer": True,
            "max_results": 5,
        },
    )
    response.raise_for_status()
    return response.json()


def _split_into_tips(text: str, max_tips: int = 5) -> List[str]:
    """
    Break a synthesized answer into short, roughly self-contained
    sentences so the planner has discrete tips to cite instead of one
    wall of text. Deliberately simple (sentence-boundary regex, length
    filter) -- no LLM call, this tool just returns data.
    """
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15][:max_tips]


async def search_destination(destination: str, api_key: Optional[str] = None) -> SearchResult:
    """
    Research a destination: overview + attractions mentioned, culture,
    and best season + travel tips.

    Runs three targeted Tavily queries rather than one broad query,
    because a single query tends to synthesize an answer biased toward
    either "what to see" or "when to go" -- rarely both well. Three
    focused queries map cleanly onto the fields this tool must return.
    """
    key = api_key or settings.search_api_key
    if not key:
        raise SearchToolError("SEARCH_API_KEY is not configured.")
    if not destination or not destination.strip():
        raise SearchToolError("destination must be a non-empty string.")

    try:
        overview_data = await _tavily_query(
            f"{destination} travel guide and top tourist attractions", key
        )
        culture_data = await _tavily_query(
            f"{destination} local culture, customs and traditions", key
        )
        season_data = await _tavily_query(
            f"best time to visit {destination}, weather and practical travel tips", key
        )
    except httpx.HTTPStatusError as exc:
        raise SearchToolError(
            f"Tavily API returned an error: {exc.response.status_code}"
        ) from exc
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        raise SearchToolError(f"Tavily API unreachable: {exc}") from exc

    overview_answer = overview_data.get("answer") or ""
    culture_answer = culture_data.get("answer") or ""
    season_answer = season_data.get("answer") or ""

    if not overview_answer and not overview_data.get("results"):
        raise SearchToolError(f"No search results found for '{destination}'.")

    attractions_mentioned = [
        r["title"] for r in overview_data.get("results", []) if r.get("title")
    ][:8]

    all_results = (
        overview_data.get("results", [])
        + culture_data.get("results", [])
        + season_data.get("results", [])
    )
    sources = list({r["url"] for r in all_results if r.get("url")})

    result = SearchResult(
        destination=destination,
        overview=overview_answer or "No overview available.",
        culture=culture_answer or "No culture information available.",
        best_season=season_answer or "No seasonal information available.",
        travel_tips=_split_into_tips(season_answer),
        attractions_mentioned=attractions_mentioned,
        sources=sources,
    )

    logger.info(
        "search_tool: destination=%s tips=%d attractions_mentioned=%d sources=%d",
        destination,
        len(result.travel_tips),
        len(result.attractions_mentioned),
        len(result.sources),
    )
    return result
