"""Tests for the reminders store."""

from __future__ import annotations

import datetime
import json
from pathlib import Path

import pytest

from reachy_mini_supermemory_app import _reminders as r


@pytest.fixture
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Route the store to a throwaway path under tmp_path."""
    path = tmp_path / "reminders.json"
    monkeypatch.setenv(r.REMINDERS_FILE_ENV, str(path))
    return path


def _iso(offset_seconds: int) -> str:
    """Build an ISO string ``offset_seconds`` from now (positive = future)."""
    when = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=offset_seconds)
    return when.isoformat()


# ---------- _parse_iso ----------


def test_parse_iso_accepts_z_suffix() -> None:
    dt = r._parse_iso("2026-05-17T21:00:00Z")
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_iso_accepts_explicit_offset() -> None:
    dt = r._parse_iso("2026-05-17T17:00:00-04:00")
    assert dt is not None
    assert dt.utcoffset() == datetime.timedelta(hours=-4)


def test_parse_iso_treats_naive_as_utc() -> None:
    dt = r._parse_iso("2026-05-17T17:00:00")
    assert dt is not None
    assert dt.utcoffset() == datetime.timedelta(0)


def test_parse_iso_rejects_garbage() -> None:
    assert r._parse_iso("not a datetime") is None
    assert r._parse_iso("") is None
    assert r._parse_iso(None) is None  # type: ignore[arg-type]


# ---------- add ----------


def test_add_persists_pending_entry(store_path: Path) -> None:
    result = r.add("call mom", _iso(3600))
    assert "error" not in result
    assert result["status"] == r.STATUS_PENDING
    assert result["text"] == "call mom"
    assert "id" in result
    # File on disk has it.
    on_disk = json.loads(store_path.read_text())
    assert any(e["id"] == result["id"] for e in on_disk["entries"])


def test_add_rejects_blank_text(store_path: Path) -> None:
    assert "error" in r.add("   ", _iso(60))


def test_add_rejects_no_time_args(store_path: Path) -> None:
    """Neither in_seconds nor when_iso → clear error."""
    result = r.add("anything")
    assert "error" in result
    assert "in_seconds" in result["error"] or "when_iso" in result["error"]


def test_add_rejects_bad_iso(store_path: Path) -> None:
    result = r.add("anything", when_iso="not iso")
    assert "error" in result


def test_add_rejects_past_time(store_path: Path) -> None:
    result = r.add("anything", when_iso=_iso(-60))
    assert "error" in result
    assert "future" in result["error"]


def test_add_assigns_unique_ids(store_path: Path) -> None:
    a = r.add("first", when_iso=_iso(60))
    b = r.add("second", when_iso=_iso(120))
    assert a["id"] != b["id"]


# ---------- relative-time path ----------


def test_add_with_in_seconds_resolves_against_wall_clock(store_path: Path) -> None:
    """The trustworthy path — model just says '60' for 'in one minute'."""
    before = datetime.datetime.now(datetime.timezone.utc)
    result = r.add("call mom", in_seconds=60)
    assert "error" not in result
    fire_at = datetime.datetime.fromisoformat(result["fire_at"])
    delta = (fire_at - before).total_seconds()
    # Should be ~60s; allow 5s tolerance for test slowness.
    assert 55 < delta < 65


def test_add_in_seconds_rejects_zero_and_negative(store_path: Path) -> None:
    assert "error" in r.add("x", in_seconds=0)
    assert "error" in r.add("x", in_seconds=-30)


def test_add_in_seconds_rejects_non_int(store_path: Path) -> None:
    """The model might emit a string. Reject cleanly."""
    assert "error" in r.add("x", in_seconds="sixty")  # type: ignore[arg-type]


def test_in_seconds_takes_precedence_when_both_supplied(store_path: Path) -> None:
    """If the model accidentally passes both, the trustworthy path wins."""
    far_future_iso = _iso(99999)
    result = r.add("x", when_iso=far_future_iso, in_seconds=60)
    assert "error" not in result
    fire_at = datetime.datetime.fromisoformat(result["fire_at"])
    delta = (fire_at - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert delta < 120  # ≈ 60s, not the far-future ISO


# ---------- list_all ----------


def test_list_all_returns_pending_sorted(store_path: Path) -> None:
    later = r.add("later", _iso(3600))
    sooner = r.add("sooner", _iso(60))
    out = r.list_all()
    assert [e["id"] for e in out] == [sooner["id"], later["id"]]


def test_list_all_excludes_terminal_by_default(store_path: Path) -> None:
    a = r.add("first", _iso(60))
    r.cancel(a["id"])
    r.add("second", _iso(120))
    out = r.list_all()
    assert len(out) == 1
    assert out[0]["text"] == "second"


def test_list_all_includes_terminal_when_requested(store_path: Path) -> None:
    a = r.add("first", _iso(60))
    r.cancel(a["id"])
    out = r.list_all(include_terminal=True)
    assert len(out) == 1
    assert out[0]["status"] == r.STATUS_CANCELLED


# ---------- cancel ----------


def test_cancel_marks_pending_entry(store_path: Path) -> None:
    entry = r.add("call dentist", _iso(60))
    result = r.cancel(entry["id"])
    assert "error" not in result
    assert result["status"] == r.STATUS_CANCELLED
    # No longer in pending list.
    assert all(e["id"] != entry["id"] for e in r.list_all())


def test_cancel_rejects_unknown_id(store_path: Path) -> None:
    assert "error" in r.cancel("not-a-real-id")


def test_cancel_refuses_already_terminal(store_path: Path) -> None:
    entry = r.add("call dentist", _iso(60))
    r.cancel(entry["id"])
    second = r.cancel(entry["id"])
    assert "error" in second


# ---------- due_reminders + pop_first_due ----------


def test_due_reminders_returns_only_past_pending(store_path: Path) -> None:
    """Past-time entries directly inserted into the store should surface as due."""
    # We can't use r.add for past times (it rejects). Hand-write the store.
    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
    future = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=3600)
    payload = {
        "entries": [
            {"id": "p1", "text": "past", "fire_at": past.isoformat(), "status": r.STATUS_PENDING, "created_at": ""},
            {"id": "f1", "text": "future", "fire_at": future.isoformat(), "status": r.STATUS_PENDING, "created_at": ""},
        ]
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload), encoding="utf-8")
    due = r.due_reminders()
    assert [e["id"] for e in due] == ["p1"]


def test_pop_first_due_marks_fired_and_returns_copy(store_path: Path) -> None:
    past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=5)
    payload = {
        "entries": [
            {"id": "p1", "text": "past1", "fire_at": past.isoformat(), "status": r.STATUS_PENDING, "created_at": ""},
        ]
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload), encoding="utf-8")

    claimed = r.pop_first_due()
    assert claimed is not None
    assert claimed["id"] == "p1"
    # Second call returns nothing — the entry was atomically marked FIRED.
    assert r.pop_first_due() is None
    # And the on-disk status now reads FIRED.
    on_disk = json.loads(store_path.read_text())
    [entry] = on_disk["entries"]
    assert entry["status"] == r.STATUS_FIRED


def test_pop_first_due_picks_oldest_when_multiple_due(store_path: Path) -> None:
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "entries": [
            {"id": "older", "text": "older", "fire_at": (now - datetime.timedelta(seconds=120)).isoformat(),
             "status": r.STATUS_PENDING, "created_at": ""},
            {"id": "newer", "text": "newer", "fire_at": (now - datetime.timedelta(seconds=10)).isoformat(),
             "status": r.STATUS_PENDING, "created_at": ""},
        ]
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload), encoding="utf-8")
    claimed = r.pop_first_due()
    assert claimed is not None and claimed["id"] == "older"


def test_pop_first_due_returns_none_when_nothing_due(store_path: Path) -> None:
    r.add("future", _iso(3600))
    assert r.pop_first_due() is None


# ---------- mark_fired ----------


def test_mark_fired_flips_status(store_path: Path) -> None:
    entry = r.add("future", _iso(60))
    assert r.mark_fired(entry["id"]) is True
    assert r.mark_fired(entry["id"]) is False  # already terminal


def test_mark_fired_rejects_unknown(store_path: Path) -> None:
    assert r.mark_fired("nothing") is False


# ---------- GC ----------


def test_gc_drops_old_terminal_entries(store_path: Path) -> None:
    """FIRED/CANCELLED entries older than RETENTION should be culled on read."""
    long_ago = (
        datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(seconds=r.RETENTION_SECONDS + 60)
    ).isoformat()
    recent = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "entries": [
            {"id": "old", "text": "old", "fire_at": long_ago, "status": r.STATUS_FIRED,
             "terminal_at": long_ago, "created_at": long_ago},
            {"id": "new", "text": "new", "fire_at": recent, "status": r.STATUS_FIRED,
             "terminal_at": recent, "created_at": recent},
        ]
    }
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(json.dumps(payload), encoding="utf-8")
    out = r.list_all(include_terminal=True)
    assert [e["id"] for e in out] == ["new"]


# ---------- file resilience ----------


def test_load_recovers_from_corrupt_file(store_path: Path) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text("not json", encoding="utf-8")
    assert r.list_all() == []


def test_atomic_write_replaces_via_tmp(store_path: Path) -> None:
    """Write path should use a .tmp rename so a crash mid-write can't truncate the store."""
    r.add("first", _iso(60))
    # A .tmp file should NOT linger after a successful write.
    assert not store_path.with_suffix(store_path.suffix + ".tmp").exists()
