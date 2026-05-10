"""save_memory tool — persist a durable fact to supermemory.ai."""

from __future__ import annotations

import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._supermemory_client import (
    derive_container_tag,
    post_json,
)

logger = logging.getLogger(__name__)


class SaveMemory(Tool):
    """Save a durable fact to supermemory.ai for future recall."""

    name = "save_memory"
    description = (
        "Save a durable fact about the user (preference, identity, decision, recurring person/place) "
        "to long-term memory. Don't use for small talk, transient mood, or things the user did not "
        "actually say. Saving is silent — do not announce it to the user."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The fact to remember, written in plain prose, third-person about the user.",
            },
            "kind": {
                "type": "string",
                "description": "Optional label: preference, identity, decision, fact, relationship.",
            },
        },
        "required": ["content"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Send the memory to supermemory.ai."""
        content = (kwargs.get("content") or "").strip()
        if not content:
            return {"error": "content is required"}

        kind = (kwargs.get("kind") or "").strip() or None
        memory: Dict[str, Any] = {"content": content}
        if kind:
            memory["metadata"] = {"kind": kind}

        body = {
            "memories": [memory],
            "containerTag": derive_container_tag(),
        }

        result = await post_json("/v4/memories", body)
        if "error" in result:
            return result

        memories = result.get("memories") or []
        memory_id = memories[0].get("id") if memories else None
        logger.info("save_memory ok: id=%s kind=%s", memory_id, kind)
        return {"saved": True, "memory_id": memory_id}
