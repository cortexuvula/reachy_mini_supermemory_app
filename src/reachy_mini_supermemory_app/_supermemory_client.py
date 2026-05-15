"""HTTP client + helpers for supermemory.ai integration.

The bulk of the state — the discovered-containerTag cache and the per-event-
loop ``httpx.AsyncClient`` pool — is owned by ``SupermemoryClient``. The
module exposes a ``_default_client`` singleton plus a layer of one-line
facade functions (``get_json``, ``post_json``, ``discover_container_tags``,
…) so existing callers don't have to thread an instance through their
code, while tests can spin up an isolated instance when they need parallel
or hermetic state.

Configuration (``SUPERMEMORY_API_KEY``, ``SUPERMEMORY_BASE_URL``) is read
at call time, not import time, so the headless settings UI can provision
credentials after launch.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import httpx

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.supermemory.ai"
REQUEST_TIMEOUT_S = 10.0
CONTAINER_TAG_RE = re.compile(r"[^A-Za-z0-9_:-]+")

RECALL_TAGS_ENV = "SUPERMEMORY_RECALL_CONTAINER_TAGS"
RECALL_EXCLUDED_TAGS_ENV = "SUPERMEMORY_RECALL_EXCLUDED_TAGS"
TAG_CACHE_TTL_S = 600.0
# Short negative TTL for failed/empty discovery — prevents hammering the API
# when the key is bad or supermemory is down, but recovers quickly once fixed.
TAG_CACHE_NEGATIVE_TTL_S = 30.0


class SupermemoryConfigError(RuntimeError):
    """Raised when required configuration is missing."""


# ---------------------------------------------------------------------------
# Stateless helpers (read env at call time, no caching beyond the regex)
# ---------------------------------------------------------------------------


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


def _format_http_error(response: httpx.Response) -> str:
    body = response.text or ""
    excerpt = body[:200].replace("\n", " ").strip()
    if excerpt:
        return f"Supermemory HTTP {response.status_code}: {excerpt}"
    return f"Supermemory HTTP {response.status_code}"


# ---------------------------------------------------------------------------
# Stateful client
# ---------------------------------------------------------------------------


class SupermemoryClient:
    """Owns the tag-discovery cache and per-loop ``httpx.AsyncClient`` pool.

    Two state clusters live here:

    - ``_tag_cache`` / ``_tag_cache_lock`` — discovered containerTag list with
      a positive TTL (``TAG_CACHE_TTL_S``) for successful responses and a short
      negative TTL (``TAG_CACHE_NEGATIVE_TTL_S``) for failures so an outage
      doesn't translate into one round-trip per ``recall_memory`` call.

    - ``_clients`` / ``_clients_lock`` — per-event-loop ``httpx.AsyncClient``
      cache. ``httpx.AsyncClient`` binds its anyio transport to the loop that
      first issues a request; sharing a single instance across loops crashes.
      Stale-loop entries are evicted on every lookup so dead-loop refs don't
      leak.

    Both clusters use ``threading.Lock`` (not ``asyncio.Lock``) because the
    critical sections are short and non-awaiting; this keeps the locks safe
    across event loops.

    The module exposes a process-wide ``_default_client`` singleton for the
    common case; instantiating directly gives tests hermetic state.
    """

    def __init__(self) -> None:
        self._tag_cache: Dict[str, Any] = {"tags": None, "expires_at": 0.0}
        self._tag_cache_lock = threading.Lock()
        self._clients: Dict[int, Tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = {}
        self._clients_lock = threading.Lock()

    # -- client pool -----------------------------------------------------

    def client_for_current_loop(self) -> httpx.AsyncClient:
        """Return an ``httpx.AsyncClient`` bound to the running event loop.

        Reusing a client across requests preserves the underlying TCP+TLS
        connection pool — a fresh client per call paid a full TLS handshake
        on every recall/save/tag-discovery hit. The lookup also evicts entries
        whose owning loop has since closed.

        Must be called from inside a coroutine — ``asyncio.get_running_loop()``
        raises if there is no running loop.
        """
        current_loop = asyncio.get_running_loop()
        current_id = id(current_loop)
        with self._clients_lock:
            stale = [lid for lid, (loop, _) in self._clients.items() if loop.is_closed()]
            for lid in stale:
                self._clients.pop(lid, None)
            entry = self._clients.get(current_id)
            if entry is None:
                client = httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
                self._clients[current_id] = (current_loop, client)
                return client
            return entry[1]

    async def aclose_client_for_current_loop(self) -> None:
        """Close the cached client bound to the current event loop, if any.

        Called by long-running owner threads (e.g. the auto-digest persistent
        loop) on shutdown so pooled connections are released cleanly instead
        of being abandoned in a soon-to-close loop.
        """
        current_id = id(asyncio.get_running_loop())
        with self._clients_lock:
            entry = self._clients.pop(current_id, None)
        if entry is not None:
            try:
                await entry[1].aclose()
            except Exception:
                pass

    def reset_clients(self) -> None:
        """Drop all cached httpx clients without aclose()ing them.

        Test helper. Stale-loop eviction handles this automatically once a
        new loop makes a request, but explicit reset keeps the cache small
        in test suites with many loops.
        """
        with self._clients_lock:
            self._clients.clear()

    # -- HTTP -----------------------------------------------------------

    async def _request(self, method: str, path: str, *, body: Optional[Dict[str, Any]] = None) -> Any:
        """Issue an authenticated request to supermemory; map all failures to ``{"error": ...}``.

        Single chokepoint for auth, error mapping, JSON parsing, and the
        "never raise" contract for tool callers. Future cross-cutting
        changes (retry, request-ID logging, 429 backoff) land here.
        """
        try:
            api_key = _get_api_key()
        except SupermemoryConfigError as e:
            return {"error": str(e)}

        url = f"{_get_base_url()}{path}"
        headers: Dict[str, str] = {"Authorization": f"Bearer {api_key}"}
        if body is not None:
            headers["Content-Type"] = "application/json"

        try:
            client = self.client_for_current_loop()
            response = await client.request(method, url, headers=headers, json=body)
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

    async def get_json(self, path: str) -> Any:
        """GET ``path`` and return parsed JSON, or ``{"error": ...}``."""
        return await self._request("GET", path)

    async def post_json(self, path: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """POST JSON to ``path`` and return the parsed response, or ``{"error": ...}``."""
        return await self._request("POST", path, body=body)  # type: ignore[no-any-return]

    # -- tag discovery --------------------------------------------------

    async def discover_container_tags(self) -> List[str]:
        """Enumerate distinct containerTags via ``GET /v3/container-tags/list``.

        Cached for ``TAG_CACHE_TTL_S`` (success) or
        ``TAG_CACHE_NEGATIVE_TTL_S`` (failure / unexpected shape). Returns
        only containers the API key can actually search; ad-hoc-tagged docs
        without a registered Space won't show up here, so callers can still
        override via ``SUPERMEMORY_RECALL_CONTAINER_TAGS``.

        The cache is mutated under ``self._tag_cache_lock``. The HTTP call
        itself runs without the lock to keep us safe across event loops; a
        cold-start race may produce one extra fetch but the result is
        convergent (both writers write the same list).
        """
        now = time.monotonic()
        with self._tag_cache_lock:
            cached = self._tag_cache.get("tags")
            expires_at = float(self._tag_cache.get("expires_at", 0.0))
            if cached is not None and now < expires_at:
                return list(cached)

        payload = await self.get_json("/v3/container-tags/list")

        if isinstance(payload, dict) and "error" in payload:
            logger.warning("Tag discovery failed: %s", payload.get("error"))
            with self._tag_cache_lock:
                self._tag_cache["tags"] = []
                self._tag_cache["expires_at"] = now + TAG_CACHE_NEGATIVE_TTL_S
            return []
        if not isinstance(payload, list):
            logger.warning("Tag discovery returned unexpected shape: %r", type(payload))
            with self._tag_cache_lock:
                self._tag_cache["tags"] = []
                self._tag_cache["expires_at"] = now + TAG_CACHE_NEGATIVE_TTL_S
            return []

        seen: Set[str] = set()
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            tag = entry.get("containerTag")
            if isinstance(tag, str) and tag.strip():
                seen.add(tag.strip())

        discovered = sorted(seen)
        with self._tag_cache_lock:
            self._tag_cache["tags"] = discovered
            # Successful-but-empty (new account) gets the same long TTL as a
            # populated result — recall falls back to the own-tag default,
            # and we don't want to re-hit the API every recall waiting for
            # the user to actually save something.
            self._tag_cache["expires_at"] = now + TAG_CACHE_TTL_S
        if discovered:
            logger.info("Discovered %d containerTags: %s", len(discovered), discovered)
        return discovered

    def invalidate_tag_cache(self) -> None:
        """Drop the cached tag list so the next ``discover_container_tags`` call refetches."""
        with self._tag_cache_lock:
            self._tag_cache["tags"] = None
            self._tag_cache["expires_at"] = 0.0

    def reset_tag_cache(self) -> None:
        """Test helper alias for ``invalidate_tag_cache``."""
        self.invalidate_tag_cache()


# ---------------------------------------------------------------------------
# Process-wide singleton + module-level facades
# ---------------------------------------------------------------------------

# Process-wide instance. Tools, auto-digest, and the settings UI all go
# through this; tests can either reset its state via ``_testing`` or
# construct a fresh ``SupermemoryClient()`` for hermetic per-test isolation.
_default_client = SupermemoryClient()


def _client_for_current_loop() -> httpx.AsyncClient:
    """Module-level facade. Delegates to the default singleton."""
    return _default_client.client_for_current_loop()


async def aclose_client_for_current_loop() -> None:
    """Module-level facade. Delegates to the default singleton."""
    await _default_client.aclose_client_for_current_loop()


async def get_json(path: str) -> Any:
    """Module-level facade. Delegates to the default singleton."""
    return await _default_client.get_json(path)


async def post_json(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Module-level facade. Delegates to the default singleton."""
    return await _default_client.post_json(path, body)


async def discover_container_tags() -> List[str]:
    """Module-level facade. Delegates to the default singleton."""
    return await _default_client.discover_container_tags()


def invalidate_tag_cache() -> None:
    """Module-level facade. Delegates to the default singleton."""
    _default_client.invalidate_tag_cache()


def _reset_tag_cache_for_tests() -> None:
    _default_client.reset_tag_cache()


def _reset_clients_for_tests() -> None:
    _default_client.reset_clients()
