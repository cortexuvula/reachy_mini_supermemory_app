"""Tests for the recall_memory tool."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from recall_memory import (  # type: ignore[import-not-found]
    DEFAULT_LIMIT,
    MAX_LIMIT,
    RecallMemory,
    _parse_matches,
)


@pytest.mark.asyncio
async def test_recall_memory_rejects_empty_query() -> None:
    tool = RecallMemory()
    result = await tool(deps=None, query="   ")  # type: ignore[arg-type]
    assert result == {"error": "query is required"}


@pytest.mark.asyncio
async def test_recall_memory_passes_query_and_clamps_limit() -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        return {"results": []}

    with patch("recall_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = RecallMemory()
            await tool(deps=None, query="favorite book", limit=999)  # type: ignore[arg-type]

    assert captured["path"] == "/v4/search"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["query"] == "favorite book"
    assert body["containerTag"] == "reachy-mini:supermemory"
    assert body["limit"] == MAX_LIMIT
    assert 0 < body["threshold"] <= 1


@pytest.mark.asyncio
async def test_recall_memory_uses_default_limit_when_unspecified() -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, body: dict) -> dict:
        captured["body"] = body
        return {"results": []}

    with patch("recall_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = RecallMemory()
            await tool(deps=None, query="anything")  # type: ignore[arg-type]

    body = captured["body"]
    assert isinstance(body, dict)
    assert body["limit"] == DEFAULT_LIMIT


@pytest.mark.asyncio
async def test_recall_memory_returns_parsed_matches() -> None:
    api_response = {
        "results": [
            {"memory": "User likes lobster jokes.", "score": 0.92},
            {"content": "User finished Project Hail Mary.", "relevance": 0.81},
            {"unrelated": "garbage"},
        ]
    }

    with patch("recall_memory.post_json", new=AsyncMock(return_value=api_response)):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = RecallMemory()
            result = await tool(deps=None, query="user")  # type: ignore[arg-type]

    matches = result["matches"]
    assert len(matches) == 2
    assert matches[0]["memory"] == "User likes lobster jokes."
    assert matches[0]["score"] == pytest.approx(0.92)
    assert matches[1]["memory"] == "User finished Project Hail Mary."
    assert matches[1]["score"] == pytest.approx(0.81)


@pytest.mark.asyncio
async def test_recall_memory_returns_empty_when_no_matches() -> None:
    with patch("recall_memory.post_json", new=AsyncMock(return_value={"results": []})):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = RecallMemory()
            result = await tool(deps=None, query="x")  # type: ignore[arg-type]

    assert result == {"matches": []}


@pytest.mark.asyncio
async def test_recall_memory_propagates_error_dict() -> None:
    with patch("recall_memory.post_json", new=AsyncMock(return_value={"error": "down"})):
        with patch("recall_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = RecallMemory()
            result = await tool(deps=None, query="x")  # type: ignore[arg-type]

    assert result == {"error": "down"}


def test_parse_matches_supports_multiple_response_shapes() -> None:
    assert _parse_matches({"matches": [{"memory": "a", "score": 0.5}]}) == [{"memory": "a", "score": 0.5}]
    assert _parse_matches({"memories": [{"text": "b"}]}) == [{"memory": "b"}]
    assert _parse_matches({"results": []}) == []
    assert _parse_matches({}) == []
