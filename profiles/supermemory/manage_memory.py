"""manage_memory tool — curate Reachy's always-loaded inline memory."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._inline_memory import (
    add_entry,
    char_limit,
    load_entries,
    remove_entry,
    replace_entry,
    total_chars,
)

logger = logging.getLogger(__name__)


class ManageMemory(Tool):
    """Curate Reachy's always-loaded inline memory of durable facts about the user."""

    name = "manage_memory"
    description = (
        "Curate Reachy's always-loaded inline memory: short durable facts about the user that get "
        "injected into every session's prompt so the user never has to re-explain them. "
        "Use ONLY for high-signal facts: name, preferences, recurring people, decisions, environment "
        "specifics. NEVER for small talk, transient mood, task state, or anything easily re-discovered. "
        "Test before saving: 'will this still matter in 3 sessions?'. There is a hard character limit; "
        "remove or replace older entries before adding new ones if you hit it. Editing is silent — "
        "don't announce additions to the user."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "list"],
                "description": (
                    "add: append a new entry. replace: find an existing entry containing old_text "
                    "and swap it for content. remove: delete entries containing old_text. list: see "
                    "current entries before deciding what to change."
                ),
            },
            "content": {
                "type": "string",
                "description": "Entry text. Required for add and replace. One concise declarative fact.",
            },
            "old_text": {
                "type": "string",
                "description": "Substring to match against existing entries. Required for replace and remove.",
            },
        },
        "required": ["action"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Dispatch on action to the inline-memory helpers."""
        action = (kwargs.get("action") or "").strip().lower()
        if action == "add":
            return add_entry(kwargs.get("content") or "")
        if action == "replace":
            return replace_entry(kwargs.get("old_text") or "", kwargs.get("content") or "")
        if action == "remove":
            return remove_entry(kwargs.get("old_text") or "")
        if action == "list":
            entries = load_entries()
            return {
                "entries": entries,
                "chars_used": total_chars(entries),
                "char_limit": char_limit(),
            }
        return {"error": f"unknown action: {action!r}. Use add, replace, remove, or list."}
