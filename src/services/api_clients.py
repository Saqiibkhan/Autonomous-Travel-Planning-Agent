"""
Shared async HTTP client.

One httpx.AsyncClient, reused across every tool call, instead of each
tool creating (and forgetting to close) its own client. This is a real
performance thing, not just tidiness -- reusing a client means connection
pooling actually works when the LangGraph workflow calls multiple tools
back to back.
"""

import httpx

from src.config import settings

_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)
    return _client


async def close_http_client() -> None:
    """Call this on app shutdown so we don't leak the connection pool."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
