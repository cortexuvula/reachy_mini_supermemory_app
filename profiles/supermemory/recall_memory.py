"""recall_memory tool — search supermemory.ai for prior memories."""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Set

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
DISCOVERY_PAGE_SIZE = 100
DISCOVERY_MAX_PAGES = 10
TAG_CACHE_TTL_S = 600.0

_tag_cache: Dict[str, Any] = {"tags": None, "expires_at": 0.0}


def _override_tags_from_env() -> List[str]:
    raw = os.environ.get(RECALL_TAGS_ENV, "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


async def _discover_container_tags() -> List[str]:
    """Enumerate distinct containerTags by scanning ``/v3/documents/list``.

    Bounded by ``DISCOVERY_MAX_PAGES`` so cold-recall latency stays predictable
    on accounts with very large document counts. Tags in older pages may be
    missed; users can always override via ``SUPERMEMORY_RECALL_CONTAINER_TAGS``.
    """
    seen: Set[str] = set()
    for page in range(1, DISCOVERY_MAX_PAGES + 1):
        payload = await post_json("/v3/documents/list", {"limit": DISCOVERY_PAGE_SIZE, "page": page})
        if "error" in payload:
            logger.warning("Tag discovery aborted on page %d: %s", page, payload.get("error"))
            break
        memories = payload.get("memories") or []
        if not memories:
            break
        for m in memories:
            if not isinstance(m, dict):
                continue
            for tag in m.get("containerTags") or []:
                if isinstance(tag, str) and tag.strip():
                    seen.add(tag.strip())
        pagination = payload.get("pagination") or {}
        total_pages = pagination.get("totalPages")
        if isinstance(total_pages, int) and page >= total_pages:
            break
    return sorted(seen)


async def _resolve_recall_tags() -> List[str]:
    """Resolve the containerTag list for a recall call.

    Order: explicit env override → cached discovery → fresh discovery →
    own-scope fallback. The own-scope fallback ensures we never search a
    completely empty list (which the API rejects anyway).
    """
    override = _override_tags_from_env()
    if override:
        return override

    now = time.monotonic()
    cached_tags = _tag_cache.get("tags")
    if cached_tags and now < float(_tag_cache.get("expires_at", 0.0)):
        return list(cached_tags)

    discovered = await _discover_container_tags()
    if discovered:
        _tag_cache["tags"] = discovered
        _tag_cache["expires_at"] = now + TAG_CACHE_TTL_S
        logger.info("Discovered %d containerTags: %s", len(discovered), discovered)
        return discovered

    return [derive_container_tag()]


def _reset_tag_cache_for_tests() -> None:
    _tag_cache["tags"] = None
    _tag_cache["expires_at"] = 0.0


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
            "containerTags": await _resolve_recall_tags(),
            "threshold": DEFAULT_THRESHOLD,
            "limit": limit,
        }

        result = await post_json("/v3/search", body)
        if "error" in result:
            return result

        matches = _parse_matches(result)
        logger.info("recall_memory: query=%r returned %d matches", query, len(matches))
        return {"matches": matches}
