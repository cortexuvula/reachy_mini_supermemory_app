"""list_reminders tool — show pending (or all recent) reminders."""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._reminders import list_all

logger = logging.getLogger(__name__)


class ListReminders(Tool):
    """List pending reminders, or all recent reminders if requested."""

    name = "list_reminders"
    description = (
        "List the user's reminders. By default returns only PENDING ones (not "
        "yet fired). Pass include_terminal=true to also surface recently FIRED "
        "or CANCELLED ones — useful when the user asks 'did my reminder go off?' "
        "or 'what was that thing I set for 5 pm?'. Use when the user asks "
        "'what reminders do I have?' / 'do I have anything scheduled?' / "
        "before cancelling so you can confirm the right one."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "include_terminal": {
                "type": "boolean",
                "description": (
                    "Set true to include FIRED and CANCELLED entries from the "
                    "last 7 days. Default false."
                ),
            },
        },
        "required": [],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Return the reminder list."""
        include_terminal = bool(kwargs.get("include_terminal"))
        entries: List[Dict[str, Any]] = list_all(include_terminal=include_terminal)
        return {"reminders": entries, "count": len(entries)}
