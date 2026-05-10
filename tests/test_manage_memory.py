"""Tests for the manage_memory tool dispatch."""

from __future__ import annotations

from pathlib import Path

import pytest
from manage_memory import ManageMemory  # type: ignore[import-not-found]
from reachy_mini_supermemory_app._inline_memory import (
    INLINE_MEMORY_CHAR_LIMIT_ENV,
    INLINE_MEMORY_FILE_ENV,
)


@pytest.fixture
def memory_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "inline-memory.json"
    monkeypatch.setenv(INLINE_MEMORY_FILE_ENV, str(path))
    monkeypatch.delenv(INLINE_MEMORY_CHAR_LIMIT_ENV, raising=False)
    return path


@pytest.mark.asyncio
async def test_add_dispatch(memory_path: Path) -> None:
    tool = ManageMemory()
    result = await tool(deps=None, action="add", content="Andre lives in Kelowna.")  # type: ignore[arg-type]
    assert result["ok"] is True
    assert result["entries"] == ["Andre lives in Kelowna."]


@pytest.mark.asyncio
async def test_list_dispatch_returns_metadata(memory_path: Path) -> None:
    tool = ManageMemory()
    await tool(deps=None, action="add", content="fact one")  # type: ignore[arg-type]
    await tool(deps=None, action="add", content="fact two")  # type: ignore[arg-type]

    listed = await tool(deps=None, action="list")  # type: ignore[arg-type]
    assert listed == {
        "entries": ["fact one", "fact two"],
        "chars_used": len("fact one") + len("fact two"),
        "char_limit": 3000,
    }


@pytest.mark.asyncio
async def test_replace_dispatch(memory_path: Path) -> None:
    tool = ManageMemory()
    await tool(deps=None, action="add", content="Andre likes slow dances.")  # type: ignore[arg-type]
    result = await tool(  # type: ignore[arg-type]
        deps=None,
        action="replace",
        old_text="slow",
        content="Andre prefers upbeat dances.",
    )
    assert result["ok"] is True
    assert result["entries"] == ["Andre prefers upbeat dances."]


@pytest.mark.asyncio
async def test_remove_dispatch(memory_path: Path) -> None:
    tool = ManageMemory()
    await tool(deps=None, action="add", content="forgettable")  # type: ignore[arg-type]
    await tool(deps=None, action="add", content="keepable")  # type: ignore[arg-type]
    result = await tool(deps=None, action="remove", old_text="forgettable")  # type: ignore[arg-type]
    assert result["ok"] is True
    assert result["entries"] == ["keepable"]


@pytest.mark.asyncio
async def test_unknown_action_returns_error(memory_path: Path) -> None:
    tool = ManageMemory()
    result = await tool(deps=None, action="explode")  # type: ignore[arg-type]
    assert "error" in result
    assert "explode" in result["error"]


@pytest.mark.asyncio
async def test_action_normalizes_case(memory_path: Path) -> None:
    tool = ManageMemory()
    result = await tool(deps=None, action=" ADD ", content="upper-case action")  # type: ignore[arg-type]
    assert result.get("ok") is True
