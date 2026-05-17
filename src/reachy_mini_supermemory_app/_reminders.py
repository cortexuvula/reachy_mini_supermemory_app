"""Local JSON-backed reminders store.

One-shot reminders only in v1: each entry records text + ISO fire-time +
status. The realtime ``emit()`` hook polls ``due_reminders()`` every audio
tick and ``mark_fired()`` the entries it actually delivered, so a reminder
fires at most once even if multiple ticks see it as due simultaneously.

Storage is a JSON file at ``$XDG_DATA_HOME/reachy_mini_supermemory_app/
reminders.json``. Atomic-rename writes guard against truncation on crash.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import secrets
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

REMINDERS_FILE_ENV = "REACHY_MINI_REMINDERS_FILE"
DEFAULT_FILE_NAME = "reminders.json"

# Status values stored alongside each reminder. PENDING is the only state
# the model needs to know about; FIRED entries linger briefly so we can
# answer "did my reminder go off?" before being garbage-collected.
STATUS_PENDING = "pending"
STATUS_FIRED = "fired"
STATUS_CANCELLED = "cancelled"

# How long a FIRED or CANCELLED entry stays in the store before being
# garbage-collected. Long enough that "did I get my 5pm reminder?" still
# returns useful info; short enough that the store doesn't grow forever.
RETENTION_SECONDS = 7 * 24 * 3600  # 7 days

_io_lock = Lock()


def reminders_file() -> Path:
    """Resolve the on-disk path for the reminders store."""
    override = (os.environ.get(REMINDERS_FILE_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "reachy_mini_supermemory_app" / DEFAULT_FILE_NAME


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _parse_iso(value: str) -> Optional[datetime.datetime]:
    """Parse an ISO 8601 datetime, returning None on failure.

    Tolerates trailing ``Z`` (turns into ``+00:00``) since some models emit it.
    Returns a tz-aware datetime; naive inputs are treated as UTC so we can
    compare across the store without surprise.
    """
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.datetime.fromisoformat(s)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def _load_unsafe(path: Path) -> List[Dict[str, Any]]:
    """Read and return the stored list, tolerating absent / malformed files."""
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Reminders file malformed at %s: %s — treating as empty", path, e)
        return []
    entries = data.get("entries") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _save_unsafe(path: Path, entries: List[Dict[str, Any]]) -> None:
    payload = {"entries": entries, "updated_at": _now_utc().isoformat()}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _new_id() -> str:
    """Short, URL-safe identifier. ~8 chars of entropy is plenty for a personal store."""
    return secrets.token_urlsafe(6)


def _gc(entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop FIRED / CANCELLED entries older than RETENTION_SECONDS."""
    cutoff = _now_utc() - datetime.timedelta(seconds=RETENTION_SECONDS)
    kept: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") == STATUS_PENDING:
            kept.append(entry)
            continue
        terminal_at = _parse_iso(entry.get("terminal_at") or entry.get("fire_at") or "")
        if terminal_at is None or terminal_at >= cutoff:
            kept.append(entry)
    return kept


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def add(text: str, when_iso: str) -> Dict[str, Any]:
    """Schedule a new reminder. Returns the stored entry or ``{"error": ...}``."""
    text = (text or "").strip()
    if not text:
        return {"error": "text is required"}
    fire_at = _parse_iso(when_iso or "")
    if fire_at is None:
        return {"error": f"when_iso is not a valid ISO 8601 datetime: {when_iso!r}"}
    # Naive past-time check — refuse explicitly so the model gets a clear
    # error instead of a silently-fired reminder.
    if fire_at <= _now_utc():
        return {"error": "when_iso must be in the future"}

    entry: Dict[str, Any] = {
        "id": _new_id(),
        "text": text,
        "fire_at": fire_at.isoformat(),
        "status": STATUS_PENDING,
        "created_at": _now_utc().isoformat(),
    }
    with _io_lock:
        path = reminders_file()
        entries = _gc(_load_unsafe(path))
        entries.append(entry)
        _save_unsafe(path, entries)
    return entry


def list_all(include_terminal: bool = False) -> List[Dict[str, Any]]:
    """Return reminders, sorted by fire-time ascending.

    By default returns only pending entries (what the model usually wants).
    ``include_terminal=True`` returns FIRED / CANCELLED too, useful for the
    "did my reminder go off?" lookup.
    """
    with _io_lock:
        entries = _gc(_load_unsafe(reminders_file()))
        if not include_terminal:
            entries = [e for e in entries if e.get("status") == STATUS_PENDING]
    entries.sort(key=lambda e: e.get("fire_at") or "")
    return entries


def cancel(reminder_id: str) -> Dict[str, Any]:
    """Mark a pending reminder as cancelled. Returns the new entry or error."""
    reminder_id = (reminder_id or "").strip()
    if not reminder_id:
        return {"error": "reminder_id is required"}
    with _io_lock:
        path = reminders_file()
        entries = _gc(_load_unsafe(path))
        for entry in entries:
            if entry.get("id") == reminder_id:
                if entry.get("status") != STATUS_PENDING:
                    return {"error": f"reminder {reminder_id!r} is not pending"}
                entry["status"] = STATUS_CANCELLED
                entry["terminal_at"] = _now_utc().isoformat()
                _save_unsafe(path, entries)
                return dict(entry)
        return {"error": f"no reminder with id {reminder_id!r}"}


def due_reminders() -> List[Dict[str, Any]]:
    """Return pending reminders whose fire-time has passed, sorted by fire-time.

    Does NOT mutate state — the emit-hook caller decides which entries it
    successfully delivered, then calls ``mark_fired(id)`` for each.
    """
    now = _now_utc()
    with _io_lock:
        entries = _gc(_load_unsafe(reminders_file()))
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if entry.get("status") != STATUS_PENDING:
            continue
        fire_at = _parse_iso(entry.get("fire_at") or "")
        if fire_at is None or fire_at > now:
            continue
        out.append(entry)
    out.sort(key=lambda e: e.get("fire_at") or "")
    return out


def pop_first_due() -> Optional[Dict[str, Any]]:
    """Atomically claim the oldest due reminder.

    Marks the entry FIRED under ``_io_lock`` and returns a copy. Returns
    None when no reminder is due. Used by the realtime emit hook so two
    consecutive ticks can't fire the same reminder twice (which would
    happen if we used ``due_reminders()`` + ``mark_fired()`` separately
    and yielded to the loop between them).
    """
    now = _now_utc()
    with _io_lock:
        path = reminders_file()
        entries = _gc(_load_unsafe(path))
        candidate: Optional[Dict[str, Any]] = None
        candidate_when: Optional[datetime.datetime] = None
        for entry in entries:
            if entry.get("status") != STATUS_PENDING:
                continue
            fire_at = _parse_iso(entry.get("fire_at") or "")
            if fire_at is None or fire_at > now:
                continue
            if candidate_when is None or fire_at < candidate_when:
                candidate = entry
                candidate_when = fire_at
        if candidate is None:
            return None
        candidate["status"] = STATUS_FIRED
        candidate["terminal_at"] = _now_utc().isoformat()
        _save_unsafe(path, entries)
        return dict(candidate)


def mark_fired(reminder_id: str) -> bool:
    """Flip a pending reminder to FIRED. Returns True if the entry was found."""
    reminder_id = (reminder_id or "").strip()
    if not reminder_id:
        return False
    with _io_lock:
        path = reminders_file()
        entries = _gc(_load_unsafe(path))
        for entry in entries:
            if entry.get("id") == reminder_id and entry.get("status") == STATUS_PENDING:
                entry["status"] = STATUS_FIRED
                entry["terminal_at"] = _now_utc().isoformat()
                _save_unsafe(path, entries)
                return True
    return False


def _reset_for_tests() -> None:
    """Delete the store file. Used by test fixtures to isolate state."""
    try:
        reminders_file().unlink()
    except FileNotFoundError:
        pass
    except Exception as e:
        logger.warning("Could not reset reminders file: %s", e)
