"""
Unit tests for search_tool.py.

Every test mocks httpx.AsyncClient.post -- we never hit the real Tavily
API in unit tests. This matters for two reasons: tests must run without
network access or a real API key, and they must run fast/deterministically
(no flaky network calls in CI).
"""

from unittest.mock import AsyncMock

import httpx
import pytest

from src.tools.search_tool import search_destination
from src.utils.exceptions import SearchToolError


def _tavily_response(payload: dict) -> httpx.Response:
    return httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", "https://api.tavily.com/search"),
    )


OVERVIEW_PAYLOAD = {
    "answer": (
        "Hunza is a mountainous valley in Gilgit-Baltistan, Pakistan, known "
        "for dramatic peaks, ancient forts, and apricot orchards."
    ),
    "results": [
        {"title": "Top 10 Attractions in Hunza Valley", "url": "https://example.com/1"},
        {"title": "Baltit Fort Guide", "url": "https://example.com/2"},
    ],
}
CULTURE_PAYLOAD = {
    "answer": "Hunza is known for its Wakhi and Burushaski-speaking communities and Ismaili traditions.",
    "results": [{"title": "Hunza Culture", "url": "https://example.com/3"}],
}
SEASON_PAYLOAD = {
    "answer": (
        "The best time to visit Hunza is April to October. Pack warm layers for "
        "evenings. Carry cash, as ATMs are scarce in the valley."
    ),
    "results": [{"title": "Best Time to Visit Hunza", "url": "https://example.com/4"}],
}


async def test_search_destination_success(mocker):
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.side_effect = [
        _tavily_response(OVERVIEW_PAYLOAD),
        _tavily_response(CULTURE_PAYLOAD),
        _tavily_response(SEASON_PAYLOAD),
    ]

    result = await search_destination("Hunza", api_key="fake-key")

    assert result.destination == "Hunza"
    assert "Hunza" in result.overview
    assert "Ismaili" in result.culture
    assert "April" in result.best_season
    assert len(result.travel_tips) > 0
    assert "Top 10 Attractions in Hunza Valley" in result.attractions_mentioned
    assert len(result.sources) == 4
    assert mock_post.call_count == 3


async def test_search_destination_missing_api_key(monkeypatch):
    monkeypatch.setattr("src.config.settings.search_api_key", "")
    with pytest.raises(SearchToolError, match="SEARCH_API_KEY"):
        await search_destination("Hunza", api_key=None)


async def test_search_destination_empty_name():
    with pytest.raises(SearchToolError, match="non-empty"):
        await search_destination("   ", api_key="fake-key")


async def test_search_destination_no_results_raises(mocker):
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    empty_payload = {"answer": "", "results": []}
    mock_post.side_effect = [_tavily_response(empty_payload)] * 3

    with pytest.raises(SearchToolError, match="No search results"):
        await search_destination("Nowhereland", api_key="fake-key")


async def test_search_destination_retries_then_raises(mocker):
    mocker.patch("src.services.retry.asyncio.sleep", new_callable=AsyncMock)
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    mock_post.side_effect = httpx.TimeoutException("timed out")

    with pytest.raises(SearchToolError, match="unreachable"):
        await search_destination("Hunza", api_key="fake-key")

    # 3 attempts for the first query (max_tool_retries default = 3), then it
    # gives up entirely -- it never gets to the culture/season queries.
    assert mock_post.call_count == 3


async def test_search_destination_http_error(mocker):
    mocker.patch("src.services.retry.asyncio.sleep", new_callable=AsyncMock)
    mock_post = mocker.patch("httpx.AsyncClient.post", new_callable=AsyncMock)
    error_response = httpx.Response(
        401, json={"error": "unauthorized"}, request=httpx.Request("POST", "https://api.tavily.com/search")
    )
    mock_post.return_value = error_response

    with pytest.raises(SearchToolError, match="401"):
        await search_destination("Hunza", api_key="bad-key")
