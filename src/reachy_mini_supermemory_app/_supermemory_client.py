"""HTTP client + helpers for supermemory.ai integration.

Reads SUPERMEMORY_API_KEY and SUPERMEMORY_BASE_URL at call time (not import
time) so the headless settings UI can provision credentials after launch.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.supermemory.ai"
REQUEST_TIMEOUT_S = 10.0
CONTAINER_TAG_RE = re.compile(r"[^A-Za-z0-9_:-]+")


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
