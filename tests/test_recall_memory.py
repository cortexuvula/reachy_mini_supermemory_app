"""Tests for the recall_memory tool."""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest
from recall_memory import (  # type: ignore[import-not-found]
    DEFAULT_LIMIT,
    MAX_LIMIT,
    RECALL_TAGS_ENV,
    RecallMemory,
    _discover_container_tags,
    _override_tags_from_env,
    _parse_matches,
    _reset_tag_cache_for_tests,
    _resolve_recall_tags,
)


@pytest.fixture(autouse=True)
def _clean_tag_cache() -> None:
    _reset_tag_cache_for_tests()


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

    with patch("recall_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
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

    with patch("recall_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
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

    with patch("recall_memory.post_json", new=AsyncMock(return_value=api_response)):
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
    with patch("recall_memory.post_json", new=AsyncMock(return_value={"results": []})):
        tool = RecallMemory()
        result = await tool(deps=None, query="x")  # type: ignore[arg-type]

    assert result == {"matches": []}


@pytest.mark.asyncio
async def test_recall_memory_propagates_error_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "reachy-mini:supermemory")
    with patch("recall_memory.post_json", new=AsyncMock(return_value={"error": "down"})):
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


def test_override_tags_from_env_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "hermes, reachy-mini:supermemory ,reachy_mini")
    assert _override_tags_from_env() == ["hermes", "reachy-mini:supermemory", "reachy_mini"]


def test_override_tags_from_env_blank_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RECALL_TAGS_ENV, raising=False)
    assert _override_tags_from_env() == []
    monkeypatch.setenv(RECALL_TAGS_ENV, "   ,  ")
    assert _override_tags_from_env() == []


@pytest.mark.asyncio
async def test_resolve_recall_tags_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(RECALL_TAGS_ENV, "alpha,beta")
    # post_json must NOT be called when env override is present.
    with patch("recall_memory.post_json", new=AsyncMock(side_effect=AssertionError("should not call API"))):
        assert await _resolve_recall_tags() == ["alpha", "beta"]


@pytest.mark.asyncio
async def test_resolve_recall_tags_runs_discovery_then_caches(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RECALL_TAGS_ENV, raising=False)
    pages: List[Dict[str, Any]] = [
        {
            "memories": [
                {"containerTags": ["hermes"]},
                {"containerTags": ["reachy-mini:supermemory"]},
            ],
            "pagination": {"totalPages": 2, "currentPage": 1},
        },
        {
            "memories": [{"containerTags": ["reachy_mini", "hermes"]}],
            "pagination": {"totalPages": 2, "currentPage": 2},
        },
    ]
    call_count = {"n": 0}

    async def fake_post_json(path: str, body: dict) -> dict:
        assert path == "/v3/documents/list"
        page = body["page"] - 1
        call_count["n"] += 1
        return pages[page] if page < len(pages) else {"memories": []}

    with patch("recall_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
        first = await _resolve_recall_tags()
    assert first == ["hermes", "reachy-mini:supermemory", "reachy_mini"]
    assert call_count["n"] == 2  # paginated until totalPages reached

    # Second call should be served from cache — no further post_json invocations.
    with patch("recall_memory.post_json", new=AsyncMock(side_effect=AssertionError("cache miss"))):
        second = await _resolve_recall_tags()
    assert second == first


@pytest.mark.asyncio
async def test_resolve_recall_tags_falls_back_to_own_scope_on_empty_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(RECALL_TAGS_ENV, raising=False)
    with patch("recall_memory.post_json", new=AsyncMock(return_value={"memories": []})):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tags = await _resolve_recall_tags()
    assert tags == ["reachy-mini:supermemory"]


@pytest.mark.asyncio
async def test_discover_container_tags_stops_on_error() -> None:
    async def fake_post_json(path: str, body: dict) -> dict:
        if body["page"] == 1:
            return {
                "memories": [{"containerTags": ["hermes"]}],
                "pagination": {"totalPages": 5, "currentPage": 1},
            }
        return {"error": "boom"}

    with patch("recall_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
        tags = await _discover_container_tags()
    assert tags == ["hermes"]
