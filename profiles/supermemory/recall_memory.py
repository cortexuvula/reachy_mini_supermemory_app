"""recall_memory tool — search supermemory.ai for prior memories."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._supermemory_client import (
    derive_container_tag,
    post_json,
)

logger = logging.getLogger(__name__)

DEFAULT_LIMIT = 5
MAX_LIMIT = 20
DEFAULT_THRESHOLD = 0.6
RECALL_TAGS_ENV = "SUPERMEMORY_RECALL_CONTAINER_TAGS"


def _recall_container_tags() -> List[str]:
    """Resolve the list of containerTags to search across.

    Honours ``SUPERMEMORY_RECALL_CONTAINER_TAGS`` (comma-separated). When unset,
    falls back to the active profile's own scope so recall doesn't leak across
    personas by default.
    """
    raw = os.environ.get(RECALL_TAGS_ENV, "").strip()
    if raw:
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        if tags:
            return tags
    return [derive_container_tag()]


def _parse_matches(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Best-effort parse of supermemory's search response into a flat list."""
    candidates = payload.get("results") or payload.get("matches") or payload.get("memories") or []
    out: List[Dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        text = entry.get("memory") or entry.get("content") or entry.get("text")
        if not text and isinstance(entry.get("chunks"), list):
            chunks = [
                c.get("content")
                for c in entry["chunks"]
                if isinstance(c, dict) and isinstance(c.get("content"), str)
            ]
            if chunks:
                text = " ".join(chunks)
        if not text:
            continue
        item: Dict[str, Any] = {"memory": text}
        score = entry.get("score") or entry.get("similarity") or entry.get("relevance")
        if isinstance(score, (int, float)):
            item["score"] = float(score)
        out.append(item)
    return out


class RecallMemory(Tool):
    """Search supermemory.ai for relevant prior memories."""

    name = "recall_memory"
    description = (
        "Search long-term memory when the user references prior context not in this conversation. "
        "Use only when needed — not on every turn. Phrase the query as a topic, not a question."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Topic or keywords to search for.",
            },
            "limit": {
                "type": "integer",
                "description": f"Max matches to return (default {DEFAULT_LIMIT}, max {MAX_LIMIT}).",
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Search supermemory.ai for memories matching the query."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}

        limit_raw = kwargs.get("limit")
        try:
            limit = int(limit_raw) if limit_raw is not None else DEFAULT_LIMIT
        except (TypeError, ValueError):
            limit = DEFAULT_LIMIT
        limit = max(1, min(limit, MAX_LIMIT))

        body = {
            "q": query,
            "containerTags": _recall_container_tags(),
            "threshold": DEFAULT_THRESHOLD,
            "limit": limit,
        }

        result = await post_json("/v3/search", body)
        if "error" in result:
            return result

        matches = _parse_matches(result)
        logger.info("recall_memory: query=%r returned %d matches", query, len(matches))
        return {"matches": matches}
