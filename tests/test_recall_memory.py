"""Tests for the recall_memory tool."""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from reachy_mini_supermemory_app._supermemory_client import (
    RECALL_EXCLUDED_TAGS_ENV,
    RECALL_TAGS_ENV,
)
from reachy_mini_supermemory_app._testing import reset_supermemory_tag_cache
from recall_memory import (  # type: ignore[import-not-found]
    DEFAULT_LIMIT,
    MAX_LIMIT,
    RecallMemory,
    _parse_matches,
    _resolve_recall_tags,
)


@pytest.fixture(autouse=True)
def _clean_tag_cache() -> None:
    reset_supermemory_tag_cache()


def _patched_post_json(mock: AsyncMock) -> ExitStack:
    """Patch post_json everywhere the recall path could hit it.

    Three patch sites: the module-level facade, the per-tool import in
    recall_memory, and the singleton's method (discover_container_tags uses
    ``self`` after the SupermemoryClient class refactor).
    """
    stack = ExitStack()
    stack.enter_context(patch("recall_memory.post_json", new=mock))
    stack.enter_context(patch("reachy_mini_supermemory_app._supermemory_client.post_json", new=mock))
    stack.enter_context(
        patch("reachy_mini_supermemory_app._supermemory_client._default_client.post_json", new=mock)
    )
    return stack


def _patched_get_json(mock: AsyncMock) -> ExitStack:
    """Patch get_json on both the module facade and the singleton's method."""
    stack = ExitStack()
    stack.enter_context(patch("reachy_mini_supermemory_app._supermemory_client.get_json", new=mock))
    stack.enter_context(
        patch("reachy_mini_supermemory_app._supermemory_client._default_client.get_json", new=mock)
    )
    return stack


@pytest.mark.asyncio
async def test_recall_memory_rejects_empty_query() -> None:
    tool = RecallMemory()
    result = await tool(deps=None, query="   ")  # type: ignore[arg-type]
    assert result == {"error": "query is required"}


@pytest.mark.asyncio
async def test_recall_memory_passes_query_and_clamps_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "reachy-mini:supermemory")
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        return {"results": []}

    with _patched_post_json(AsyncMock(side_effect=fake_post_json)):
        tool = RecallMemory()
        await tool(deps=None, query="favorite book", limit=999)  # type: ignore[arg-type]

    assert captured["path"] == "/v3/search"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["q"] == "favorite book"
    assert body["containerTags"] == ["reachy-mini:supermemory"]
    assert body["limit"] == MAX_LIMIT
    assert 0 < body["threshold"] <= 1


@pytest.mark.asyncio
async def test_recall_memory_uses_default_limit_when_unspecified(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "reachy-mini:supermemory")
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, body: dict) -> dict:
        captured["body"] = body
        return {"results": []}

    with _patched_post_json(AsyncMock(side_effect=fake_post_json)):
        tool = RecallMemory()
        await tool(deps=None, query="anything")  # type: ignore[arg-type]

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["limit"] == DEFAULT_LIMIT


@pytest.mark.asyncio
async def test_recall_memory_returns_parsed_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "reachy-mini:supermemory")
    api_response = {
        "results": [
            {"memory": "User likes lobster jokes.", "score": 0.92},
            {"content": "User finished Project Hail Mary.", "relevance": 0.81},
            {"unrelated": "garbage"},
        ]
    }

    with _patched_post_json(AsyncMock(return_value=api_response)):
        tool = RecallMemory()
        result = await tool(deps=None, query="user")  # type: ignore[arg-type]

    matches = result["matches"]
    assert len(matches) == 2
    assert matches[0]["memory"] == "User likes lobster jokes."
    assert matches[0]["score"] == pytest.approx(0.92)
    assert matches[1]["memory"] == "User finished Project Hail Mary."
    assert matches[1]["score"] == pytest.approx(0.81)


@pytest.mark.asyncio
async def test_recall_memory_returns_empty_when_no_matches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "reachy-mini:supermemory")
    with _patched_post_json(AsyncMock(return_value={"results": []})):
        tool = RecallMemory()
        result = await tool(deps=None, query="x")  # type: ignore[arg-type]

    assert result == {"matches": []}


@pytest.mark.asyncio
async def test_recall_memory_propagates_error_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "reachy-mini:supermemory")
    with _patched_post_json(AsyncMock(return_value={"error": "down"})):
        tool = RecallMemory()
        result = await tool(deps=None, query="x")  # type: ignore[arg-type]

    assert result == {"error": "down"}


def test_parse_matches_supports_multiple_response_shapes() -> None:
    assert _parse_matches({"matches": [{"memory": "a", "score": 0.5}]}) == [{"memory": "a", "score": 0.5}]
    assert _parse_matches({"memories": [{"text": "b"}]}) == [{"memory": "b"}]
    assert _parse_matches({"results": []}) == []
    assert _parse_matches({}) == []


def test_parse_matches_extracts_v3_chunks_shape() -> None:
    v3_payload = {
        "results": [
            {
                "documentId": "abc",
                "title": "Hermes",
                "score": 0.65,
                "chunks": [
                    {"content": "user said hi", "score": 0.65},
                    {"content": "assistant replied", "score": 0.6},
                ],
            }
        ]
    }
    matches = _parse_matches(v3_payload)
    assert matches == [{"memory": "user said hi assistant replied", "score": pytest.approx(0.65)}]


@pytest.mark.asyncio
async def test_resolve_recall_tags_prefers_env_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "alpha,beta")
    monkeypatch.setenv(RECALL_EXCLUDED_TAGS_ENV, "alpha")  # ignored when pinned
    with _patched_get_json(AsyncMock(side_effect=AssertionError("should not call API"))):
        assert await _resolve_recall_tags() == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_resolve_recall_tags_filters_excluded_after_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECALL_TAGS_ENV, raising=False)
    monkeypatch.setenv(RECALL_EXCLUDED_TAGS_ENV, "hermes,sm_project_twitter_bookmarks")
    payload: List[Dict[str, Any]] = [
        {"containerTag": "hermes"},
        {"containerTag": "reachy-mini:supermemory"},
        {"containerTag": "sm_project_twitter_bookmarks"},
    ]

    with _patched_get_json(AsyncMock(return_value=payload)):
        tags = await _resolve_recall_tags()
    assert tags == ["reachy-mini:supermemory"]


@pytest.mark.asyncio
async def test_resolve_recall_tags_falls_back_when_excluding_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECALL_TAGS_ENV, raising=False)
    monkeypatch.setenv(RECALL_EXCLUDED_TAGS_ENV, "hermes,reachy-mini:supermemory")
    payload: List[Dict[str, Any]] = [
        {"containerTag": "hermes"},
        {"containerTag": "reachy-mini:supermemory"},
    ]

    with _patched_get_json(AsyncMock(return_value=payload)):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tags = await _resolve_recall_tags()
    assert tags == ["reachy-mini:supermemory"]


@pytest.mark.asyncio
async def test_resolve_recall_tags_handles_discovery_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RECALL_TAGS_ENV, raising=False)
    monkeypatch.delenv(RECALL_EXCLUDED_TAGS_ENV, raising=False)
    with _patched_get_json(AsyncMock(return_value={"error": "boom"})):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tags = await _resolve_recall_tags()
    assert tags == ["reachy-mini:supermemory"]


@pytest.mark.asyncio
async def test_tag_discovery_negative_caches_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed /container-tags/list response must not re-hit the API on every recall.

    Before the negative-cache fix, an auth error or outage meant one tag-list
    round-trip per recall_memory call (a tight feedback loop with the model
    asking for memories).
    """
    from reachy_mini_supermemory_app import _supermemory_client as smc

    mock = AsyncMock(return_value={"error": "401 unauthorized"})
    with _patched_get_json(mock):
        first = await smc.discover_container_tags()
        second = await smc.discover_container_tags()
    assert first == []
    assert second == []
    # Two recalls, but the API was only hit once thanks to the negative cache.
    assert mock.await_count == 1


@pytest.mark.asyncio
async def test_tag_discovery_negative_cache_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative cache must release after the short TTL so recovery is fast."""
    from reachy_mini_supermemory_app import _supermemory_client as smc

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(smc.time, "monotonic", lambda: fake_now["t"])

    mock = AsyncMock(return_value={"error": "503 service unavailable"})
    with _patched_get_json(mock):
        await smc.discover_container_tags()  # caches negative
        # Inside the negative TTL window: still cached.
        fake_now["t"] += smc.TAG_CACHE_NEGATIVE_TTL_S - 1
        await smc.discover_container_tags()
        assert mock.await_count == 1
        # Past the TTL: cache expires, API is hit again.
        fake_now["t"] += 2
        await smc.discover_container_tags()
        assert mock.await_count == 2


@pytest.mark.asyncio
async def test_tag_discovery_caches_successful_empty_with_long_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 200 OK with zero tags (new account) must use the LONG TTL, not the short one."""
    from reachy_mini_supermemory_app import _supermemory_client as smc

    fake_now = {"t": 1000.0}
    monkeypatch.setattr(smc.time, "monotonic", lambda: fake_now["t"])

    mock = AsyncMock(return_value=[])  # successful but empty
    with _patched_get_json(mock):
        await smc.discover_container_tags()
        # Way past the SHORT TTL — long TTL must still be in effect.
        fake_now["t"] += smc.TAG_CACHE_NEGATIVE_TTL_S * 2
        await smc.discover_container_tags()
    assert mock.await_count == 1
