"""Tests for the .env lifecycle helpers extracted into ``_env_paths``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from reachy_mini_supermemory_app import _env_paths


@pytest.fixture
def isolated_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    """Route persistent + legacy to throwaway paths under ``tmp_path``."""
    persistent = tmp_path / "config" / ".env"
    legacy = tmp_path / "install" / ".env"
    monkeypatch.setattr(_env_paths, "persistent_env_path", lambda: persistent)
    monkeypatch.setattr(_env_paths, "package_env_path", lambda: legacy)
    return persistent, legacy


# ---------- _read_env_file ----------


def test_read_env_file_parses_plain_assignments(tmp_path: Path) -> None:
    target = tmp_path / "test.env"
    target.write_text("FOO=bar\nBAZ=qux\n", encoding="utf-8")
    assert _env_paths._read_env_file(target) == {"FOO": "bar", "BAZ": "qux"}


def test_read_env_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    target = tmp_path / "test.env"
    target.write_text("# top comment\nFOO=bar\n\n# mid\nBAZ=qux\n", encoding="utf-8")
    assert _env_paths._read_env_file(target) == {"FOO": "bar", "BAZ": "qux"}


def test_read_env_file_missing_returns_empty(tmp_path: Path) -> None:
    assert _env_paths._read_env_file(tmp_path / "absent.env") == {}


def test_read_env_file_skips_lines_without_eq(tmp_path: Path) -> None:
    target = tmp_path / "malformed.env"
    target.write_text("nope\nFOO=bar\n=val\n", encoding="utf-8")
    assert _env_paths._read_env_file(target) == {"FOO": "bar"}


# ---------- migrate_legacy_to_persistent_if_needed ----------


def test_migrate_copies_when_only_legacy_exists(isolated_paths: tuple[Path, Path]) -> None:
    persistent, legacy = isolated_paths
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("KEY=v1\n", encoding="utf-8")

    _env_paths.migrate_legacy_to_persistent_if_needed(persistent, legacy)
    assert persistent.exists()
    assert persistent.read_text(encoding="utf-8") == "KEY=v1\n"


def test_migrate_noop_when_persistent_exists(isolated_paths: tuple[Path, Path]) -> None:
    persistent, legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("KEEP=true\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("OLD=value\n", encoding="utf-8")

    _env_paths.migrate_legacy_to_persistent_if_needed(persistent, legacy)
    # Persistent untouched — migration is a one-time bootstrap, not a sync.
    assert persistent.read_text(encoding="utf-8") == "KEEP=true\n"


# ---------- capture_legacy_extras_into_persistent ----------


def test_capture_appends_keys_only_in_legacy(isolated_paths: tuple[Path, Path]) -> None:
    """Upstream-console writes to legacy must NOT be lost when we sync the other direction.

    Pre-fix, the next ``sync_persistent_to_legacy`` would overwrite legacy and
    erase any key set via upstream's own UI. Capture now rescues them by
    appending to persistent first.
    """
    persistent, legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("SUPERMEMORY_API_KEY=ours\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "SUPERMEMORY_API_KEY=ours\nHF_TOKEN=set-via-upstream-console\n", encoding="utf-8"
    )

    _env_paths.capture_legacy_extras_into_persistent(persistent, legacy)

    parsed = _env_paths._read_env_file(persistent)
    assert parsed["SUPERMEMORY_API_KEY"] == "ours"
    assert parsed["HF_TOKEN"] == "set-via-upstream-console"


def test_capture_does_not_overwrite_existing_keys(isolated_paths: tuple[Path, Path]) -> None:
    """If a key is set in BOTH, persistent wins — capture only fills gaps."""
    persistent, legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("KEY=new-from-persistent\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("KEY=stale-from-legacy\n", encoding="utf-8")

    _env_paths.capture_legacy_extras_into_persistent(persistent, legacy)
    parsed = _env_paths._read_env_file(persistent)
    assert parsed["KEY"] == "new-from-persistent"


def test_capture_noop_when_legacy_missing(isolated_paths: tuple[Path, Path]) -> None:
    persistent, legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("KEY=value\n", encoding="utf-8")

    _env_paths.capture_legacy_extras_into_persistent(persistent, legacy)
    assert persistent.read_text(encoding="utf-8") == "KEY=value\n"


def test_capture_creates_persistent_when_only_legacy_has_extras(
    isolated_paths: tuple[Path, Path],
) -> None:
    persistent, legacy = isolated_paths
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("ONLY=in-legacy\n", encoding="utf-8")

    _env_paths.capture_legacy_extras_into_persistent(persistent, legacy)
    assert persistent.exists()
    assert _env_paths._read_env_file(persistent) == {"ONLY": "in-legacy"}


# ---------- sync_persistent_to_legacy ----------


def test_sync_overwrites_legacy(isolated_paths: tuple[Path, Path]) -> None:
    persistent, legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("KEY=current\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("KEY=stale\nDROPPED=yes\n", encoding="utf-8")

    _env_paths.sync_persistent_to_legacy(persistent, legacy)
    # Legacy is REPLACED, not merged — that's why capture must run first.
    assert legacy.read_text(encoding="utf-8") == "KEY=current\n"


def test_sync_noop_when_persistent_missing(isolated_paths: tuple[Path, Path]) -> None:
    persistent, legacy = isolated_paths
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("KEEP=true\n", encoding="utf-8")

    _env_paths.sync_persistent_to_legacy(persistent, legacy)
    # Nothing happens when there's no source of truth.
    assert legacy.read_text(encoding="utf-8") == "KEEP=true\n"


# ---------- load_package_dotenv (orchestrator) ----------


def test_load_package_dotenv_full_flow_rescues_upstream_console_writes(
    isolated_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: a key written to legacy ends up in os.environ and in persistent."""
    persistent, legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("SUPERMEMORY_API_KEY=ours\n", encoding="utf-8")
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text(
        "SUPERMEMORY_API_KEY=ours\nHF_TOKEN=set-via-upstream\n", encoding="utf-8"
    )

    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    _env_paths.load_package_dotenv()

    # 1. Both keys landed in os.environ.
    assert os.environ["SUPERMEMORY_API_KEY"] == "ours"
    assert os.environ["HF_TOKEN"] == "set-via-upstream"
    # 2. Persistent now has BOTH keys (rescue worked).
    parsed = _env_paths._read_env_file(persistent)
    assert parsed["SUPERMEMORY_API_KEY"] == "ours"
    assert parsed["HF_TOKEN"] == "set-via-upstream"
    # 3. Legacy mirrors persistent (so upstream's load_dotenv override=True is a no-op).
    assert _env_paths._read_env_file(legacy) == parsed


def test_load_package_dotenv_existing_env_wins_over_persistent(
    isolated_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """systemd Environment= overrides must NOT be clobbered by persistent."""
    persistent, _legacy = isolated_paths
    persistent.parent.mkdir(parents=True, exist_ok=True)
    persistent.write_text("SUPERMEMORY_API_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "from-process-env")

    _env_paths.load_package_dotenv()
    assert os.environ["SUPERMEMORY_API_KEY"] == "from-process-env"


def test_load_package_dotenv_handles_fresh_install(
    isolated_paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No files anywhere — must not raise, must leave os.environ alone."""
    monkeypatch.delenv("SUPERMEMORY_API_KEY", raising=False)
    _env_paths.load_package_dotenv()
    assert "SUPERMEMORY_API_KEY" not in os.environ
