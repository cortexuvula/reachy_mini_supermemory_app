"""Tests for the add_reminder / list_reminders / cancel_reminder tools."""

from __future__ import annotations

import datetime
from pathlib import Path

import pytest

from add_reminder import AddReminder  # type: ignore[import-not-found]
from cancel_reminder import CancelReminder  # type: ignore[import-not-found]
from list_reminders import ListReminders  # type: ignore[import-not-found]
from reachy_mini_supermemory_app import _reminders as r


@pytest.fixture(autouse=True)
def store_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "reminders.json"
    monkeypatch.setenv(r.REMINDERS_FILE_ENV, str(path))
    return path


def _iso(offset_seconds: int) -> str:
    when = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(seconds=offset_seconds)
    return when.isoformat()


# ---------- add_reminder (absolute path: when_iso) ----------


@pytest.mark.asyncio
async def test_add_reminder_happy_path_with_iso() -> None:
    result = await AddReminder()(deps=None, text="call mom", when_iso=_iso(3600))  # type: ignore[arg-type]
    assert result["scheduled"] is True
    assert result["text"] == "call mom"
    assert "id" in result
    assert "fire_at" in result


@pytest.mark.asyncio
async def test_add_reminder_rejects_blank_text() -> None:
    result = await AddReminder()(deps=None, text="  ", when_iso=_iso(60))  # type: ignore[arg-type]
    assert "error" in result


@pytest.mark.asyncio
async def test_add_reminder_rejects_bad_iso() -> None:
    result = await AddReminder()(deps=None, text="hi", when_iso="five o'clock")  # type: ignore[arg-type]
    assert "error" in result


@pytest.mark.asyncio
async def test_add_reminder_rejects_past_time() -> None:
    result = await AddReminder()(deps=None, text="hi", when_iso=_iso(-30))  # type: ignore[arg-type]
    assert "error" in result
    assert "future" in result["error"]


@pytest.mark.asyncio
async def test_add_reminder_rejects_missing_time_args() -> None:
    """If the model forgets both args, return a clear error not a crash."""
    result = await AddReminder()(deps=None, text="hi")  # type: ignore[arg-type]
    assert "error" in result


# ---------- add_reminder (relative path: in_seconds) ----------


@pytest.mark.asyncio
async def test_add_reminder_in_seconds_happy_path() -> None:
    result = await AddReminder()(deps=None, text="put the kettle on", in_seconds=60)  # type: ignore[arg-type]
    assert result["scheduled"] is True
    assert result["text"] == "put the kettle on"
    # fire_at should be ~60s from now.
    import datetime as _dt

    fire_at = _dt.datetime.fromisoformat(result["fire_at"])
    delta = (fire_at - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    assert 50 < delta < 70


@pytest.mark.asyncio
async def test_add_reminder_in_seconds_rejects_zero() -> None:
    result = await AddReminder()(deps=None, text="hi", in_seconds=0)  # type: ignore[arg-type]
    assert "error" in result


@pytest.mark.asyncio
async def test_add_reminder_in_seconds_rejects_negative() -> None:
    result = await AddReminder()(deps=None, text="hi", in_seconds=-5)  # type: ignore[arg-type]
    assert "error" in result


@pytest.mark.asyncio
async def test_add_reminder_in_seconds_takes_precedence() -> None:
    """When both are passed, the wall-clock path wins (it can't be wrong)."""
    far_future = _iso(99999)
    result = await AddReminder()(  # type: ignore[arg-type]
        deps=None, text="hi", when_iso=far_future, in_seconds=60
    )
    assert result["scheduled"] is True
    import datetime as _dt

    fire_at = _dt.datetime.fromisoformat(result["fire_at"])
    delta = (fire_at - _dt.datetime.now(_dt.timezone.utc)).total_seconds()
    assert delta < 120  # ≈60s, not the far-future ISO


# ---------- list_reminders ----------


@pytest.mark.asyncio
async def test_list_reminders_empty_by_default() -> None:
    result = await ListReminders()(deps=None)  # type: ignore[arg-type]
    assert result == {"reminders": [], "count": 0}


@pytest.mark.asyncio
async def test_list_reminders_returns_pending_sorted() -> None:
    await AddReminder()(deps=None, text="later", when_iso=_iso(7200))  # type: ignore[arg-type]
    await AddReminder()(deps=None, text="sooner", when_iso=_iso(60))  # type: ignore[arg-type]
    result = await ListReminders()(deps=None)  # type: ignore[arg-type]
    assert result["count"] == 2
    assert [e["text"] for e in result["reminders"]] == ["sooner", "later"]


@pytest.mark.asyncio
async def test_list_reminders_include_terminal_flag() -> None:
    entry = await AddReminder()(deps=None, text="cancel me", when_iso=_iso(60))  # type: ignore[arg-type]
    await CancelReminder()(deps=None, reminder_id=entry["id"])  # type: ignore[arg-type]

    pending = await ListReminders()(deps=None)  # type: ignore[arg-type]
    assert pending["count"] == 0

    with_terminal = await ListReminders()(deps=None, include_terminal=True)  # type: ignore[arg-type]
    assert with_terminal["count"] == 1
    assert with_terminal["reminders"][0]["status"] == r.STATUS_CANCELLED


# ---------- cancel_reminder ----------


@pytest.mark.asyncio
async def test_cancel_reminder_marks_pending_entry() -> None:
    entry = await AddReminder()(deps=None, text="cancel me", when_iso=_iso(60))  # type: ignore[arg-type]
    result = await CancelReminder()(deps=None, reminder_id=entry["id"])  # type: ignore[arg-type]
    assert result["cancelled"] is True
    assert result["id"] == entry["id"]


@pytest.mark.asyncio
async def test_cancel_reminder_rejects_unknown_id() -> None:
    result = await CancelReminder()(deps=None, reminder_id="not-an-id")  # type: ignore[arg-type]
    assert "error" in result


@pytest.mark.asyncio
async def test_cancel_reminder_rejects_blank_id() -> None:
    result = await CancelReminder()(deps=None, reminder_id="  ")  # type: ignore[arg-type]
    assert "error" in result


# ---------- tool schemas ----------


def test_add_reminder_schema_is_well_formed() -> None:
    tool = AddReminder()
    assert tool.name == "add_reminder"
    # Only `text` is required; the model picks either in_seconds or when_iso.
    assert tool.parameters_schema["required"] == ["text"]
    props = tool.parameters_schema["properties"]
    assert "in_seconds" in props
    assert "when_iso" in props


def test_list_reminders_has_no_required_params() -> None:
    tool = ListReminders()
    assert tool.parameters_schema.get("required") == []
    assert "include_terminal" in tool.parameters_schema["properties"]


def test_cancel_reminder_requires_id() -> None:
    tool = CancelReminder()
    assert tool.parameters_schema["required"] == ["reminder_id"]
