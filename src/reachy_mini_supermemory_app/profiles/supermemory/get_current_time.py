"""get_current_time tool — return the precise current local time.

Companion to the ``<<CURRENT_DATETIME>>`` substitution in the session
prompt: that block gives the model an ambient sense of "today is X",
useful for date-based questions ("is the event next week?"). This tool
covers the precise-minute case ("what time is it right now?") and the
"what time is it in Tokyo?" cross-zone case.

Stateless and local — no HTTP, sub-millisecond latency.
"""

from __future__ import annotations

import datetime
import logging
import os
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

logger = logging.getLogger(__name__)

USER_TIMEZONE_ENV = "SUPERMEMORY_USER_TIMEZONE"


class GetCurrentTime(Tool):
    """Return the precise current local date and time."""

    name = "get_current_time"
    description = (
        "Return the current local date and time. Use this when the user asks for "
        "the precise time, what minute it is, or the time in a specific city / "
        "timezone. The session prompt already includes an approximate datetime "
        "stamped at session start — for casual date awareness (\"is your "
        "birthday next week?\") that ambient context is enough; call this tool "
        "only when the answer needs to be accurate to the minute or when "
        "checking a non-local timezone."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "Optional IANA timezone name (e.g. 'America/New_York', "
                    "'Europe/London', 'Asia/Tokyo'). Omit for the user's "
                    "configured local timezone."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the current datetime in the requested or configured timezone."""
        requested_tz = (kwargs.get("timezone") or "").strip()
        env_tz = (os.environ.get(USER_TIMEZONE_ENV) or "").strip()
        # Explicit param wins; otherwise fall back to env override; otherwise OS local.
        tz_target = requested_tz or env_tz

        now, tz_display, tz_warning = _resolve_now(tz_target)

        result: Dict[str, Any] = {
            "weekday": now.strftime("%A"),
            "date": _format_date(now),
            "time": _format_time(now),
            "timezone": tz_display,
            "iso": now.isoformat(timespec="seconds"),
        }
        if tz_warning:
            result["warning"] = tz_warning
        return result


def _resolve_now(tz_target: str) -> tuple[datetime.datetime, str, str | None]:
    """Return ``(now, tz_display, warning_or_none)``.

    Falls back through: requested IANA → OS-local → UTC. The warning is
    populated only when the caller asked for a specific timezone that
    couldn't be loaded, so the model can tell the user "I don't know that
    timezone — here's UTC instead" rather than silently lying.
    """
    if tz_target:
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(tz_target)
            return datetime.datetime.now(tz), tz_target, None
        except Exception:
            try:
                now = datetime.datetime.now().astimezone()
                return (
                    now,
                    str(now.tzinfo) if now.tzinfo else "local",
                    f"Unknown timezone {tz_target!r}; returning local time instead.",
                )
            except Exception:
                return (
                    datetime.datetime.now(datetime.timezone.utc),
                    "UTC",
                    f"Unknown timezone {tz_target!r}; falling back to UTC.",
                )
    try:
        now = datetime.datetime.now().astimezone()
        return now, str(now.tzinfo) if now.tzinfo else "local", None
    except Exception:
        return datetime.datetime.now(datetime.timezone.utc), "UTC", None


def _format_date(now: datetime.datetime) -> str:
    """Render the date as 'May 16, 2026' (locale-independent, no leading zero on day)."""
    try:
        return now.strftime("%B %-d, %Y")
    except Exception:
        # GNU %-d unavailable on this platform — strip leading zero manually.
        return now.strftime("%B %d, %Y").replace(" 0", " ")


def _format_time(now: datetime.datetime) -> str:
    """Render the time as '2:18 PM' (12-hour clock, no leading zero on hour)."""
    return now.strftime("%I:%M %p").lstrip("0")
