"""Unit tests for the shared HTTP client singleton."""

import src.services.api_clients as api_clients


async def test_get_http_client_returns_same_instance():
    api_clients._client = None  # ensure a clean slate regardless of test order
    client_a = api_clients.get_http_client()
    client_b = api_clients.get_http_client()
    assert client_a is client_b
    await api_clients.close_http_client()


async def test_close_http_client_resets_singleton():
    api_clients.get_http_client()
    await api_clients.close_http_client()
    assert api_clients._client is None


async def test_close_http_client_when_never_created_is_a_noop():
    api_clients._client = None
    await api_clients.close_http_client()  # should not raise
    assert api_clients._client is None
