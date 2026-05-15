"""Tests for the local-JSON inline memory store."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from reachy_mini_supermemory_app._inline_memory import (
    INLINE_MEMORY_CHAR_LIMIT_ENV,
    INLINE_MEMORY_FILE_ENV,
    add_entry,
    char_limit,
    inline_memory_file,
    load_entries,
    remove_entry,
    render_block,
    replace_entry,
    total_chars,
)


@pytest.fixture
def memory_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "inline-memory.json"
    monkeypatch.setenv(INLINE_MEMORY_FILE_ENV, str(path))
    monkeypatch.delenv(INLINE_MEMORY_CHAR_LIMIT_ENV, raising=False)
    return path


def test_inline_memory_file_honours_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    target = tmp_path / "custom.json"
    monkeypatch.setenv(INLINE_MEMORY_FILE_ENV, str(target))
    assert inline_memory_file() == target


def test_inline_memory_file_default_uses_xdg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv(INLINE_MEMORY_FILE_ENV, raising=False)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    path = inline_memory_file()
    assert path.parent == tmp_path / "reachy_mini_supermemory_app"
    assert path.name == "inline-memory.json"


def test_load_entries_when_missing_returns_empty(memory_path: Path) -> None:
    assert not memory_path.exists()
    assert load_entries() == []


def test_add_entry_persists_and_returns_metadata(memory_path: Path) -> None:
    result = add_entry("User's name is Andre.")
    assert result["ok"] is True
    assert result["entries"] == ["User's name is Andre."]
    assert result["chars_used"] == len("User's name is Andre.")
    assert result["char_limit"] == 3000

    on_disk = json.loads(memory_path.read_text(encoding="utf-8"))
    assert on_disk["entries"] == ["User's name is Andre."]
    assert "updated_at" in on_disk


def test_add_entry_rejects_blank_content(memory_path: Path) -> None:
    assert add_entry("   ") == {"error": "content is required"}


def test_add_entry_enforces_char_limit(monkeypatch: pytest.MonkeyPatch, memory_path: Path) -> None:
    monkeypatch.setenv(INLINE_MEMORY_CHAR_LIMIT_ENV, "120")
    assert add_entry("a" * 100).get("ok") is True
    overflow = add_entry("b" * 50)
    assert "error" in overflow
    assert "120" in overflow["error"]
    # the overflow should not have been persisted
    assert load_entries() == ["a" * 100]


def test_replace_entry_updates_matching_lines(memory_path: Path) -> None:
    add_entry("User likes slow dances.")
    add_entry("User has a dog named Charlie.")

    result = replace_entry("slow dances", "User prefers fast dances over slow ones.")
    assert result["ok"] is True
    assert result["replaced"] == 1
    assert load_entries() == [
        "User prefers fast dances over slow ones.",
        "User has a dog named Charlie.",
    ]


def test_replace_entry_errors_when_no_match(memory_path: Path) -> None:
    add_entry("Existing fact.")
    result = replace_entry("nothing", "replacement")
    assert "error" in result
    assert load_entries() == ["Existing fact."]


def test_replace_entry_errors_on_ambiguous_match(memory_path: Path) -> None:
    """Previously collapsed both entries to the same content; now must refuse."""
    add_entry("Daughter Mia, 7")
    add_entry("Daughter Mira, 5")
    result = replace_entry("Daughter", "Daughter (unspecified)")
    assert "error" in result
    # The matching entries are returned so the model can refine.
    assert "matches" in result
    assert set(result["matches"]) == {"Daughter Mia, 7", "Daughter Mira, 5"}
    # Nothing changed on disk.
    assert load_entries() == ["Daughter Mia, 7", "Daughter Mira, 5"]


def test_replace_entry_requires_old_text_and_content(memory_path: Path) -> None:
    add_entry("anything")
    assert "error" in replace_entry("", "replacement")
    assert "error" in replace_entry("any", "")


def test_remove_entry_drops_matching_lines(memory_path: Path) -> None:
    add_entry("User likes coffee.")
    add_entry("User has a dog named Charlie.")
    add_entry("User likes tea.")

    result = remove_entry("coffee")
    assert result["ok"] is True
    assert result["removed"] == 1
    assert load_entries() == [
        "User has a dog named Charlie.",
        "User likes tea.",
    ]


def test_remove_entry_errors_when_no_match(memory_path: Path) -> None:
    add_entry("anything")
    result = remove_entry("nothing")
    assert "error" in result


def test_render_block_empty_when_no_entries(memory_path: Path) -> None:
    assert render_block() == ""


def test_render_block_uses_bullets_with_delimiters(memory_path: Path) -> None:
    add_entry("User's name is Andre.")
    add_entry("User has a dog named Charlie.")
    rendered = render_block()
    assert rendered.startswith("=== Things Reachy already knows about this user")
    assert "- User's name is Andre." in rendered
    assert "- User has a dog named Charlie." in rendered
    assert rendered.rstrip().endswith("=== end of memory ===")


def test_render_block_truncates_when_over_char_limit(
    monkeypatch: pytest.MonkeyPatch, memory_path: Path
) -> None:
    """Render must respect char_limit even when the on-disk file exceeds it.

    Writes refuse to push the store over the limit, but a manual JSON edit or a
    post-hoc-lowered limit can leave it over — render shouldn't then dump the
    whole thing into the system prompt.
    """
    # Bypass the write-time cap by lowering the limit after the entries are saved.
    add_entry("a" * 80)
    add_entry("b" * 80)
    add_entry("c" * 80)
    monkeypatch.setenv(INLINE_MEMORY_CHAR_LIMIT_ENV, "150")

    rendered = render_block()
    assert len(rendered) <= 200  # limit + small fudge for omission notice
    assert "omitted" in rendered
    assert rendered.startswith("=== Things Reachy already knows")
    assert rendered.rstrip().endswith("=== end of memory ===")


def test_render_block_singular_when_one_entry_omitted(
    monkeypatch: pytest.MonkeyPatch, memory_path: Path
) -> None:
    add_entry("a" * 80)
    add_entry("b" * 80)
    # Header (67) + footer (22) + first bullet (83) ≈ 173 → fits in 200,
    # second bullet would push to ~256 → omitted.
    monkeypatch.setenv(INLINE_MEMORY_CHAR_LIMIT_ENV, "200")
    rendered = render_block()
    assert "1 more entry" in rendered  # singular


def test_load_entries_recovers_from_corrupt_file(memory_path: Path) -> None:
    memory_path.parent.mkdir(parents=True, exist_ok=True)
    memory_path.write_text("not json", encoding="utf-8")
    assert load_entries() == []


def test_total_chars_sums_entries() -> None:
    assert total_chars(["abc", "de"]) == 5
    assert total_chars([]) == 0


def test_char_limit_honours_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(INLINE_MEMORY_CHAR_LIMIT_ENV, "500")
    assert char_limit() == 500
    monkeypatch.setenv(INLINE_MEMORY_CHAR_LIMIT_ENV, "abc")
    assert char_limit() == 3000  # falls back to default on parse error
