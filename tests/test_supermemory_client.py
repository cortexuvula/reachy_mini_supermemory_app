"""Tests for the shared supermemory HTTP client and helpers."""

from __future__ import annotations

import httpx
import pytest
import respx

from reachy_mini_supermemory_app._supermemory_client import (
    DEFAULT_BASE_URL,
    derive_container_tag,
    is_configured,
    post_json,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    monkeypatch.delenv("SUPERMEMORY_BASE_URL", raising=False)


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
