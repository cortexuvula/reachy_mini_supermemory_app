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
        "Curate Reachy's always-loaded inline memory — the small block of durable facts injected "
        "into every session's prompt so the user never has to re-explain them. ALTERNATIVE TO "
        "save_memory: pick one store per fact, never call both for the same content. ALWAYS use "
        "this (not save_memory) for: the user's name, top preferences, AND every named person the "
        "user introduces (children, partner, parents, siblings, close colleagues). Those people "
        "must be in the prompt every session — supermemory is searchable but doesn't auto-load. "
        "For one-off anecdotes, places, dates, or stories, use save_memory instead. Wait until the "
        "user has finished spelling/clarifying before saving — never call this multiple times for "
        "the same fact during a clarification. Use action='replace' to fix a previously-saved "
        "entry. There is a hard character limit; remove or replace older entries before adding "
        "new ones if hit. Editing is silent — don't announce additions."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "replace", "remove", "list"],
                "description": (
                    "add: append a new entry. replace: find the SINGLE existing entry containing "
                    "old_text and swap it for content — if old_text matches more than one entry, "
                    "this returns an error with the matching entries so you can refine old_text. "
                    "remove: delete entries containing old_text. list: see current entries before "
                    "deciding what to change."
                ),
            },
            "content": {
                "type": "string",
                "description": "Entry text. Required for add and replace. One concise declarative fact.",
            },
            "old_text": {
                "type": "string",
                "description": (
                    "Substring to match against existing entries. Required for replace and remove. "
                    "For replace, must uniquely identify exactly one entry; pick something distinctive "
                    "(a name, a specific number) rather than a generic word like 'Daughter' that may "
                    "match multiple."
                ),
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
