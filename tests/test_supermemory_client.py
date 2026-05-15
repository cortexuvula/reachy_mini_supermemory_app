"""Tests for the shared supermemory HTTP client and helpers."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from reachy_mini_supermemory_app import _supermemory_client as smc
from reachy_mini_supermemory_app._supermemory_client import (
    DEFAULT_BASE_URL,
    aclose_client_for_current_loop,
    derive_container_tag,
    is_configured,
    post_json,
    _client_for_current_loop,
)
from reachy_mini_supermemory_app._testing import reset_supermemory_clients


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)


@pytest.fixture(autouse=True)
def _clean_clients() -> None:
    reset_supermemory_clients()


def test_derive_container_tag_default() -> None:
    assert derive_container_tag(None) == "reachy-mini:default"


def test_derive_container_tag_passthrough() -> None:
    assert derive_container_tag("supermemory") == "reachy-mini:supermemory"


def test_derive_container_tag_sanitizes_disallowed_chars() -> None:
    # Spaces, slashes, and dots all violate ^[A-Za-z0-9_:-]+$
    assert derive_container_tag("hello world/v2.0") == "reachy-mini:hello_world_v2_0"


def test_derive_container_tag_collapses_to_default_when_empty() -> None:
    assert derive_container_tag("---") == "reachy-mini:default"


def test_is_configured_reads_env(monkeypatch: pytest.MonkeyPatch) -> None:
    assert not is_configured()
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-test")
    assert is_configured()
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "   ")
    assert not is_configured()


@pytest.mark.asyncio
async def test_post_json_returns_error_when_api_key_missing() -> None:
    result = await post_json("/v4/memories", {"hello": "world"})
    assert "error" in result
    assert "API key" in result["error"]


@pytest.mark.asyncio
@respx.mock
async def test_post_json_sends_bearer_auth_and_returns_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-abc")
    route = respx.post(f"{DEFAULT_BASE_URL}/v4/memories").mock(
        return_value=httpx.Response(201, json={"documentId": "d1", "memories": [{"id": "m1"}]})
    )

    result = await post_json("/v4/memories", {"memories": [{"content": "x"}], "containerTag": "t"})

    assert route.called
    request = route.calls.last.request
    assert request.headers["authorization"] == "Bearer sk-abc"
    assert request.headers["content-type"] == "application/json"
    assert result == {"documentId": "d1", "memories": [{"id": "m1"}]}


@pytest.mark.asyncio
@respx.mock
async def test_post_json_honors_custom_base_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-abc")
    monkeypatch.setenv("SUPERMEMORY_BASE_URL", "https://proxy.example.com")
    respx.post("https://proxy.example.com/v4/search").mock(return_value=httpx.Response(200, json={"results": []}))

    result = await post_json("/v4/search", {"query": "hello"})

    assert result == {"results": []}


@pytest.mark.asyncio
@respx.mock
async def test_post_json_maps_http_4xx_to_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-abc")
    respx.post(f"{DEFAULT_BASE_URL}/v4/memories").mock(return_value=httpx.Response(401, text="invalid token"))

    result = await post_json("/v4/memories", {})

    assert "error" in result
    assert "401" in result["error"]


@pytest.mark.asyncio
@respx.mock
async def test_post_json_maps_timeout_to_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-abc")
    respx.post(f"{DEFAULT_BASE_URL}/v4/memories").mock(side_effect=httpx.TimeoutException("slow"))

    result = await post_json("/v4/memories", {})

    assert result == {"error": "Supermemory request timed out."}


@pytest.mark.asyncio
@respx.mock
async def test_post_json_handles_non_json_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-abc")
    respx.post(f"{DEFAULT_BASE_URL}/v4/memories").mock(return_value=httpx.Response(200, text="not json"))

    result = await post_json("/v4/memories", {})

    assert "error" in result
    assert "non-JSON" in result["error"]


# ---------- per-loop client cache ----------


@pytest.mark.asyncio
async def test_client_for_current_loop_returns_same_instance_within_one_loop() -> None:
    """Repeated calls inside one loop must hit the cache — that's the whole point."""
    a = _client_for_current_loop()
    b = _client_for_current_loop()
    assert a is b
    assert isinstance(a, httpx.AsyncClient)


@pytest.mark.asyncio
async def test_client_for_current_loop_recreates_after_close(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If an external caller aclose()s the client and we resurface, we get a fresh one."""
    first = _client_for_current_loop()
    await aclose_client_for_current_loop()
    second = _client_for_current_loop()
    assert second is not first


def test_client_for_current_loop_evicts_closed_loops() -> None:
    """A loop that closed between requests must have its entry dropped on next lookup.

    Simulates the old auto-digest 'asyncio.run per iteration' pattern: each
    iteration was a fresh loop, so the cache would have leaked entries
    without stale eviction.
    """
    # Build a real loop, register a client, close the loop.
    loop_a = asyncio.new_event_loop()
    try:
        client_a = loop_a.run_until_complete(_register_and_return_client())
    finally:
        loop_a.run_until_complete(loop_a.shutdown_asyncgens())
        loop_a.close()

    # Cache still holds an entry keyed to the now-closed loop_a.
    assert any(loop is loop_a for loop, _ in smc._default_client._clients.values())

    # New loop's lookup must evict loop_a's stale entry and create a fresh client.
    loop_b = asyncio.new_event_loop()
    try:
        client_b = loop_b.run_until_complete(_register_and_return_client())
        assert client_b is not client_a
        # loop_a's entry is gone.
        assert not any(loop is loop_a for loop, _ in smc._default_client._clients.values())
    finally:
        loop_b.run_until_complete(loop_b.shutdown_asyncgens())
        loop_b.close()


async def _register_and_return_client() -> httpx.AsyncClient:
    return _client_for_current_loop()


@pytest.mark.asyncio
async def test_aclose_client_for_current_loop_is_noop_when_no_client() -> None:
    """Cleanup must tolerate being called when nothing was cached for this loop."""
    reset_supermemory_clients()
    await aclose_client_for_current_loop()  # must not raise


# ---------- multi-instance isolation ----------


@pytest.mark.asyncio
async def test_instances_have_independent_client_caches() -> None:
    """A non-default ``SupermemoryClient`` must not share state with the singleton.

    The class refactor exists so tests can build hermetic instances when they
    want isolation. This test pins that contract — two instances each maintain
    their own per-loop pool.
    """
    a = smc.SupermemoryClient()
    b = smc.SupermemoryClient()
    client_a = a.client_for_current_loop()
    client_b = b.client_for_current_loop()
    assert client_a is not client_b
    # And the default singleton stays clean.
    smc._default_client.reset_clients()
    assert smc._default_client._clients == {}


def test_instances_have_independent_tag_caches() -> None:
    """Tag-cache mutations on one instance don't leak to another."""
    a = smc.SupermemoryClient()
    b = smc.SupermemoryClient()
    a._tag_cache["tags"] = ["one", "two"]
    a._tag_cache["expires_at"] = 1e9
    assert b._tag_cache["tags"] is None
    b.invalidate_tag_cache()
    # a's cache unaffected by b's invalidation.
    assert a._tag_cache["tags"] == ["one", "two"]


def test_module_facade_routes_to_default_singleton() -> None:
    """The module-level helpers delegate to ``_default_client``, not a fresh instance."""
    smc._default_client.reset_tag_cache()
    smc._default_client._tag_cache["tags"] = ["sentinel"]
    smc._default_client._tag_cache["expires_at"] = 1e9
    # invalidate_tag_cache facade goes through _default_client, so it should wipe.
    smc.invalidate_tag_cache()
    assert smc._default_client._tag_cache["tags"] is None


@pytest.mark.asyncio
@respx.mock
async def test_post_json_reuses_pooled_client_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sequential POSTs go through the same AsyncClient instance.

    Before this refactor each call built a new AsyncClient (new TLS handshake);
    here we verify the cache binds a single instance to the running loop.
    """
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-abc")
    respx.post(f"{DEFAULT_BASE_URL}/v4/memories").mock(
        return_value=httpx.Response(200, json={"ok": True})
    )

    await post_json("/v4/memories", {"a": 1})
    client_after_first = _client_for_current_loop()
    await post_json("/v4/memories", {"a": 2})
    client_after_second = _client_for_current_loop()

    assert client_after_first is client_after_second
