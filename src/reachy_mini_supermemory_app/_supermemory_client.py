"""HTTP client + helpers for supermemory.ai integration.

Reads SUPERMEMORY_API_KEY and SUPERMEMORY_BASE_URL at call time (not import
time) so the headless settings UI can provision credentials after launch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any, Dict, List, Set

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.supermemory.ai"
REQUEST_TIMEOUT_S = 10.0
CONTAINER_TAG_RE = re.compile(r"[^A-Za-z0-9_:-]+")

RECALL_TAGS_ENV = "SUPERMEMORY_RECALL_CONTAINER_TAGS"
RECALL_EXCLUDED_TAGS_ENV = "SUPERMEMORY_RECALL_EXCLUDED_TAGS"
TAG_CACHE_TTL_S = 600.0

_tag_cache: Dict[str, Any] = {"tags": None, "expires_at": 0.0}
_tag_cache_lock = asyncio.Lock()


class SupermemoryConfigError(RuntimeError):
    """Raised when required configuration is missing."""


def _get_api_key() -> str:
    key = (os.environ.get("SUPERMEMORY_API_KEY") or "").strip()
    if not key:
        raise SupermemoryConfigError(
            "Supermemory API key not configured. Set SUPERMEMORY_API_KEY in .env or visit /supermemory/."
        )
    return key


def _get_base_url() -> str:
    return (os.environ.get("SUPERMEMORY_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def _sanitize_tag_segment(value: str) -> str:
    """Strip characters supermemory rejects, collapse runs, trim length."""
    cleaned = CONTAINER_TAG_RE.sub("_", value).strip("_:-")
    return cleaned[:80] or "default"


def derive_container_tag(profile: str | None = None) -> str:
    """Return ``reachy-mini:<profile>``, sanitized for the supermemory regex.

    Reads ``config.REACHY_MINI_CUSTOM_PROFILE`` lazily so live profile switches
    in the personality UI take effect on the next call.
    """
    if profile is None:
        try:
            from reachy_mini_conversation_app.config import config

            profile = config.REACHY_MINI_CUSTOM_PROFILE
        except Exception:
            profile = None
    return f"reachy-mini:{_sanitize_tag_segment(profile or 'default')}"


def _format_http_error(response: httpx.Response) -> str:
    body = response.text or ""
    excerpt = body[:200].replace("\n", " ").strip()
    if excerpt:
        return f"Supermemory HTTP {response.status_code}: {excerpt}"
    return f"Supermemory HTTP {response.status_code}"


async def get_json(path: str) -> Any:
    """GET path from supermemory and return parsed JSON, or ``{"error": ...}``.

    Returns whatever shape supermemory sends — list, dict, scalar — so callers
    must validate before indexing.
    """
    try:
        api_key = _get_api_key()
    except SupermemoryConfigError as e:
        return {"error": str(e)}

    url = f"{_get_base_url()}{path}"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            response = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        logger.warning("Supermemory request timed out: %s", path)
        return {"error": "Supermemory request timed out."}
    except httpx.HTTPError as e:
        logger.warning("Supermemory request failed: %s", e)
        return {"error": f"Supermemory request failed: {e}"}

    if response.status_code >= 400:
        message = _format_http_error(response)
        logger.warning(message)
        return {"error": message}

    try:
        return response.json()
    except ValueError:
        return {"error": "Supermemory returned non-JSON response."}


async def post_json(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """POST JSON to supermemory and return the parsed response or an error dict.

    Caller is responsible for handling the ``error`` key. We never raise on
    network/HTTP failures — tools want to relay a friendly message to the
    model, not crash the conversation.
    """
    try:
        api_key = _get_api_key()
    except SupermemoryConfigError as e:
        return {"error": str(e)}

    url = f"{_get_base_url()}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
            response = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException:
        logger.warning("Supermemory request timed out: %s", path)
        return {"error": "Supermemory request timed out."}
    except httpx.HTTPError as e:
        logger.warning("Supermemory request failed: %s", e)
        return {"error": f"Supermemory request failed: {e}"}

    if response.status_code >= 400:
        message = _format_http_error(response)
        logger.warning(message)
        return {"error": message}

    try:
        return response.json()  # type: ignore[no-any-return]
    except ValueError:
        return {"error": "Supermemory returned non-JSON response."}


def is_configured() -> bool:
    """Return True when SUPERMEMORY_API_KEY is set."""
    return bool((os.environ.get("SUPERMEMORY_API_KEY") or "").strip())


def _parse_csv_env(env_var: str) -> List[str]:
    raw = (os.environ.get(env_var) or "").strip()
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def recall_pinned_tags() -> List[str]:
    """Return the explicit pin list from ``SUPERMEMORY_RECALL_CONTAINER_TAGS``."""
    return _parse_csv_env(RECALL_TAGS_ENV)


def recall_excluded_tags() -> List[str]:
    """Return the exclusion list from ``SUPERMEMORY_RECALL_EXCLUDED_TAGS``."""
    return _parse_csv_env(RECALL_EXCLUDED_TAGS_ENV)


async def discover_container_tags() -> List[str]:
    """Enumerate distinct containerTags via ``GET /v3/container-tags/list``.

    Cached for ``TAG_CACHE_TTL_S`` seconds. Returns only containers the API key
    can actually search; ad-hoc-tagged docs without a registered Space won't
    show up here, so callers can still override via
    ``SUPERMEMORY_RECALL_CONTAINER_TAGS``.

    Guarded by ``_tag_cache_lock`` so a burst of concurrent recall/refresh
    requests after a cold start (or cache invalidation) collapses into one
    network round-trip instead of stampeding the supermemory API.
    """
    async with _tag_cache_lock:
        now = time.monotonic()
        cached = _tag_cache.get("tags")
        if cached and now < float(_tag_cache.get("expires_at", 0.0)):
            return list(cached)

        payload = await get_json("/v3/container-tags/list")
        if isinstance(payload, dict) and "error" in payload:
            logger.warning("Tag discovery failed: %s", payload.get("error"))
            return []
        if not isinstance(payload, list):
            logger.warning("Tag discovery returned unexpected shape: %r", type(payload))
            return []

        seen: Set[str] = set()
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            tag = entry.get("containerTag")
            if isinstance(tag, str) and tag.strip():
                seen.add(tag.strip())

        discovered = sorted(seen)
        if discovered:
            _tag_cache["tags"] = discovered
            _tag_cache["expires_at"] = now + TAG_CACHE_TTL_S
            logger.info("Discovered %d containerTags: %s", len(discovered), discovered)
        return discovered


def invalidate_tag_cache() -> None:
    """Drop the cached tag list so the next ``discover_container_tags`` call refetches."""
    _tag_cache["tags"] = None
    _tag_cache["expires_at"] = 0.0


def _reset_tag_cache_for_tests() -> None:
    invalidate_tag_cache()
