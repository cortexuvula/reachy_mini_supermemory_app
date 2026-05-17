"""cancel_reminder tool — cancel a pending reminder by id."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._reminders import cancel as cancel_in_store

logger = logging.getLogger(__name__)


class CancelReminder(Tool):
    """Cancel a pending reminder."""

    name = "cancel_reminder"
    description = (
        "Cancel a pending reminder by its id. Call list_reminders first if "
        "you don't already have the id from a prior add_reminder confirmation. "
        "Confirm with the user before cancelling something the user didn't "
        "explicitly name (\"Cancel the 5 PM reminder to call mom?\"). After "
        "successful cancel, say so in one short sentence."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "reminder_id": {
                "type": "string",
                "description": (
                    "The id returned by add_reminder or surfaced by "
                    "list_reminders."
                ),
            },
        },
        "required": ["reminder_id"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Mark the reminder cancelled in the store."""
        result = cancel_in_store(kwargs.get("reminder_id") or "")
        if "error" in result:
            return result
        return {
            "cancelled": True,
            "id": result["id"],
            "text": result["text"],
            "fire_at": result["fire_at"],
        }
