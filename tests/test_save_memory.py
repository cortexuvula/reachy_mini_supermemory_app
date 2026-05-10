"""Tests for the save_memory tool."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

# The profile-local tool path is added by conftest.
from save_memory import SaveMemory  # type: ignore[import-not-found]


@pytest.mark.asyncio
async def test_save_memory_rejects_empty_content() -> None:
    tool = SaveMemory()
    result = await tool(deps=None, content="   ")  # type: ignore[arg-type]
    assert result == {"error": "content is required"}


@pytest.mark.asyncio
async def test_save_memory_posts_correct_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "sk-test")
    monkeypatch.setenv("REACHY_MINI_CUSTOM_PROFILE", "supermemory")

    fake_response = {"documentId": "d1", "memories": [{"id": "m-42"}]}
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        return fake_response

    with patch("save_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
        # Force the container tag to read the env-set profile
        with patch("save_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = SaveMemory()
            result = await tool(deps=None, content="User likes lobster jokes.", kind="preference")  # type: ignore[arg-type]

    assert captured["path"] == "/v4/memories"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["containerTag"] == "reachy-mini:supermemory"
    memories = body["memories"]
    assert isinstance(memories, list) and len(memories) == 1
    assert memories[0]["content"] == "User likes lobster jokes."
    assert memories[0]["metadata"] == {"kind": "preference"}
    # Ensure JSON-serializable
    json.dumps(body)

    assert result == {"saved": True, "memory_id": "m-42"}


@pytest.mark.asyncio
async def test_save_memory_omits_metadata_when_kind_blank() -> None:
    captured: dict[str, object] = {}

    async def fake_post_json(path: str, body: dict) -> dict:
        captured["body"] = body
        return {"memories": [{"id": "x"}]}

    with patch("save_memory.post_json", new=AsyncMock(side_effect=fake_post_json)):
        with patch("save_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = SaveMemory()
            await tool(deps=None, content="Plain fact.")  # type: ignore[arg-type]

    body = captured["body"]
    assert isinstance(body, dict)
    assert "metadata" not in body["memories"][0]


@pytest.mark.asyncio
async def test_save_memory_propagates_error_dict() -> None:
    with patch("save_memory.post_json", new=AsyncMock(return_value={"error": "boom"})):
        with patch("save_memory.derive_container_tag", return_value="reachy-mini:supermemory"):
            tool = SaveMemory()
            result = await tool(deps=None, content="x")  # type: ignore[arg-type]

    assert result == {"error": "boom"}
