"""add_reminder tool — schedule a one-shot reminder for a future time."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._reminders import add as add_to_store

logger = logging.getLogger(__name__)


class AddReminder(Tool):
    """Schedule a one-shot reminder."""

    name = "add_reminder"
    description = (
        "Schedule a one-shot reminder. The robot speaks the reminder when the "
        "time arrives (if a session is active; otherwise on next reconnect). "
        "Two ways to specify the time:\n"
        "  • For RELATIVE requests (\"in 1 minute\", \"in 25 minutes\", "
        "\"in 2 hours\") pass in_seconds — just the integer offset. STRONGLY "
        "PREFERRED for relative phrases: the robot computes the absolute fire "
        "time from its own wall clock, so it can't drift or be confused about "
        "what 'now' is.\n"
        "  • For ABSOLUTE clock-time requests (\"at 5 pm\", \"tomorrow at 9 "
        "am\", \"on Tuesday at noon\") pass when_iso as a full ISO 8601 string "
        "with timezone offset (e.g. '2026-05-17T17:00:00-04:00'). Use the "
        "date/time stamp at the top of your prompt to compute the date and "
        "include the timezone offset that matches it.\n"
        "If you pass both, in_seconds wins. Confirm what you scheduled in one "
        "short sentence (\"OK, I'll remind you in one minute to put the kettle "
        "on\")."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "What to remind the user about. Keep it short and "
                    "concrete — this gets spoken aloud verbatim or "
                    "paraphrased (\"call mom\", \"take the bread out\")."
                ),
            },
            "in_seconds": {
                "type": "integer",
                "description": (
                    "Seconds from now until the reminder fires. Use for any "
                    "\"in N <unit>\" request: minutes → ×60, hours → ×3600. "
                    "Must be positive. Preferred over when_iso for relative "
                    "phrases because the server uses the wall clock — no "
                    "date math, no drift."
                ),
            },
            "when_iso": {
                "type": "string",
                "description": (
                    "Absolute ISO 8601 datetime with timezone offset, e.g. "
                    "'2026-05-17T17:00:00-04:00' or "
                    "'2026-05-17T21:00:00Z'. Must be in the future. Use ONLY "
                    "when the user specified a clock time or date, not for "
                    "\"in N minutes\" style requests."
                ),
            },
        },
        "required": ["text"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Persist the reminder via either in_seconds or when_iso path."""
        result = add_to_store(
            text=kwargs.get("text") or "",
            when_iso=kwargs.get("when_iso"),
            in_seconds=kwargs.get("in_seconds"),
        )
        if "error" in result:
            return result
        return {
            "scheduled": True,
            "id": result["id"],
            "text": result["text"],
            "fire_at": result["fire_at"],
        }
