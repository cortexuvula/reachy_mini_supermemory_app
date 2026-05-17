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
        "Schedule a one-shot reminder for a future time. The robot will speak the "
        "reminder text when the time arrives, provided a session is active "
        "(reminders due while the user is away fire on next reconnect). Compute "
        "the absolute ISO 8601 datetime yourself using the date/time stamped at "
        "the top of your prompt — for 'in 25 minutes' add 25 minutes to that "
        "stamp; for 'tomorrow at 8 am' bump the date and zero the time. Include "
        "timezone offset (e.g. '2026-05-17T17:00:00-04:00') so the reminder fires "
        "at the right local moment. Confirm what you scheduled in one short "
        "sentence so the user knows it landed (\"OK, I'll remind you at 5 PM "
        "to call mom\")."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": (
                    "What to remind the user about. Keep it short and "
                    "concrete — this gets spoken aloud verbatim or paraphrased "
                    "(\"call mom\", \"take the bread out of the oven\")."
                ),
            },
            "when_iso": {
                "type": "string",
                "description": (
                    "Absolute ISO 8601 datetime with timezone offset, e.g. "
                    "'2026-05-17T17:00:00-04:00' or '2026-05-17T21:00:00Z'. "
                    "Must be in the future."
                ),
            },
        },
        "required": ["text", "when_iso"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Persist the reminder."""
        result = add_to_store(kwargs.get("text") or "", kwargs.get("when_iso") or "")
        if "error" in result:
            return result
        return {
            "scheduled": True,
            "id": result["id"],
            "text": result["text"],
            "fire_at": result["fire_at"],
        }
