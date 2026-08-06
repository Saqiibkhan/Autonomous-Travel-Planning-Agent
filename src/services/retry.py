"""
Generic retry decorator for async tool calls.

Wraps a function so that transient network/HTTP failures get retried
`settings.max_tool_retries` times with a growing delay between attempts,
logging every attempt. Every tool's low-level API call gets wrapped with
this instead of hand-rolling its own retry loop.

Note: right now this treats ALL httpx errors (timeouts, connection drops,
AND bad status codes like a 401 or 429) as retryable. That's intentionally
naive for this task -- a 401 (bad API key) will never succeed no matter
how many times you retry it. Status-code-aware handling (don't retry 4xx,
do retry 5xx/429) is a natural refinement for the reflection node in
Task 6, once we're actually distinguishing failure types for the user.
"""

import asyncio
import functools
from typing import Awaitable, Callable, TypeVar

import httpx

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")

RETRYABLE_EXCEPTIONS = (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)


def async_retry(
    max_attempts: int | None = None,
    backoff_seconds: float = 1.0,
) -> Callable[[Callable[..., Awaitable[T]]], Callable[..., Awaitable[T]]]:
    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            attempts = max_attempts or settings.max_tool_retries
            last_exc: Exception | None = None

            for attempt in range(1, attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except RETRYABLE_EXCEPTIONS as exc:
                    last_exc = exc
                    logger.warning(
                        "%s: attempt %d/%d failed (%s)",
                        func.__name__,
                        attempt,
                        attempts,
                        exc,
                    )
                    if attempt < attempts:
                        await asyncio.sleep(backoff_seconds * attempt)

            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator
