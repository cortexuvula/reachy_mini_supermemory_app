"""web_search tool — query Tavily for current information.

Voice-tuned wrapper around Tavily's ``POST /search`` endpoint. The model
calls this when the user asks about anything outside its training cutoff
or the user's own past sessions (which belong to ``recall_memory``).

The tool returns a compact dict with ``answer`` (Tavily's pre-summarised
response, when available) and ``results`` (title / url / truncated content
per hit). URLs are intentionally preserved so the robot can cite sources
aloud — "according to nytimes.com…".

Configuration: ``TAVILY_API_KEY`` env var (set via the settings UI or .env).
Optional overrides: ``TAVILY_BASE_URL`` for self-hosted / proxied
deployments.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List

import httpx
from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies

from reachy_mini_supermemory_app._supermemory_client import _client_for_current_loop

logger = logging.getLogger(__name__)

TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
TAVILY_BASE_URL_ENV = "TAVILY_BASE_URL"
USER_TIMEZONE_ENV = "SUPERMEMORY_USER_TIMEZONE"
DEFAULT_BASE_URL = "https://api.tavily.com"
SEARCH_PATH = "/search"

DEFAULT_MAX_RESULTS = 3
MAX_MAX_RESULTS = 10
DEFAULT_SEARCH_DEPTH = "basic"
SEARCH_DEPTHS = ("basic", "advanced")
TIME_RANGES = ("day", "week", "month", "year")
# Voice context — the model has to read the result aloud, so a full page
# of content is counterproductive. ~400 chars ≈ one sustained sentence
# fragment per result, leaving the model room to compose its own framing.
RESULT_CONTENT_MAX_CHARS = 400
# Tavily's basic search returns in ~1 s typically; advanced can take 3-5 s.
# Cap at 15 s so a stalled call doesn't leave the speaker hanging.
REQUEST_TIMEOUT_S = 15.0


class WebSearch(Tool):
    """Search the web for information outside the model's training data and the user's memory."""

    name = "web_search"
    description = (
        "Search the public web for current information the model doesn't already know — "
        "news, recent events, weather, prices, definitions, anything outside the training "
        "cutoff or the user's personal memory. NOT for the user's own past context — "
        "for that, use recall_memory. Use sparingly: each call hits a paid API and adds "
        "1-3 seconds of latency to the spoken response. Phrase the query as topic + key "
        "terms (e.g. 'World Cup 2026 final result'), not as a full question. For "
        "time-sensitive queries ('tonight', 'today', 'this week'), set the time_range "
        "parameter — otherwise stale results may surface. Cite the source domain when "
        "speaking the answer ('according to bbc.com…')."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Plain-text search query. Topic + key terms, not a question.",
            },
            "max_results": {
                "type": "integer",
                "description": (
                    f"How many results to fetch (default {DEFAULT_MAX_RESULTS}, max "
                    f"{MAX_MAX_RESULTS}). Voice context — keep small unless the user "
                    "asks for an exhaustive sweep."
                ),
            },
            "search_depth": {
                "type": "string",
                "enum": list(SEARCH_DEPTHS),
                "description": (
                    "'basic' is fast and cheap — use for most queries. 'advanced' "
                    "returns deeper content but costs more credits and adds 1-2 s of "
                    "latency; only use when the user explicitly asks for in-depth info."
                ),
            },
            "time_range": {
                "type": "string",
                "enum": list(TIME_RANGES),
                "description": (
                    "Restrict results to recent content only. Use 'day' for 'tonight' "
                    "/ 'today' queries (sports scores, breaking news), 'week' for "
                    "'this week' / 'recent', 'month' for 'recent months', 'year' for "
                    "'this year'. OMIT for historical or factual queries (e.g. 'when "
                    "was the Eiffel Tower built') so the answer isn't artificially "
                    "filtered out."
                ),
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Run a Tavily search and return ``{"answer", "results"}`` or ``{"error": ...}``."""
        query = (kwargs.get("query") or "").strip()
        if not query:
            return {"error": "query is required"}

        api_key = (os.environ.get(TAVILY_API_KEY_ENV) or "").strip()
        if not api_key:
            return {
                "error": (
                    "Tavily API key not configured. Set TAVILY_API_KEY in .env "
                    "or via /supermemory/."
                )
            }

        max_results = _coerce_max_results(kwargs.get("max_results"))
        depth = _coerce_search_depth(kwargs.get("search_depth"))
        time_range = _coerce_time_range(kwargs.get("time_range"))

        body: Dict[str, Any] = {
            # Prefix the user's query with the current date/time so Tavily's
            # answer-summariser resolves relative phrases ("tonight",
            # "today", "this week") against the actual calendar instead of
            # its own training cutoff. Without this anchor, time-sensitive
            # queries like "Fever vs Mystics tonight's score" come back
            # with last-season's game.
            "query": _build_anchored_query(query),
            "max_results": max_results,
            "search_depth": depth,
            "include_answer": True,
            "include_raw_content": False,
            "include_images": False,
        }
        if time_range:
            body["time_range"] = time_range
        url = f"{_get_base_url()}{SEARCH_PATH}"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            client = _client_for_current_loop()
            response = await client.post(url, headers=headers, json=body, timeout=REQUEST_TIMEOUT_S)
        except httpx.TimeoutException:
            logger.warning("Tavily search timed out: %s", query)
            return {"error": "Search request timed out."}
        except httpx.HTTPError as e:
            logger.warning("Tavily search failed: %s", e)
            return {"error": f"Search request failed: {e}"}

        if response.status_code >= 400:
            excerpt = (response.text or "")[:200].replace("\n", " ").strip()
            message = (
                f"Tavily HTTP {response.status_code}: {excerpt}"
                if excerpt
                else f"Tavily HTTP {response.status_code}"
            )
            logger.warning(message)
            return {"error": message}

        try:
            payload = response.json()
        except ValueError:
            return {"error": "Search returned non-JSON response."}

        return _format_response(payload, max_results)


def _get_base_url() -> str:
    return (os.environ.get(TAVILY_BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")


def _coerce_max_results(raw: Any) -> int:
    try:
        n = int(raw) if raw is not None else DEFAULT_MAX_RESULTS
    except (TypeError, ValueError):
        n = DEFAULT_MAX_RESULTS
    return max(1, min(n, MAX_MAX_RESULTS))


def _coerce_search_depth(raw: Any) -> str:
    if not isinstance(raw, str):
        return DEFAULT_SEARCH_DEPTH
    candidate = raw.strip().lower()
    return candidate if candidate in SEARCH_DEPTHS else DEFAULT_SEARCH_DEPTH


def _coerce_time_range(raw: Any) -> str:
    """Validate the ``time_range`` parameter, returning '' when not set or invalid.

    Empty string skips the field in the request body entirely so Tavily
    doesn't filter results — important for historical queries.
    """
    if not isinstance(raw, str):
        return ""
    candidate = raw.strip().lower()
    return candidate if candidate in TIME_RANGES else ""


def _build_anchored_query(query: str) -> str:
    """Prefix ``query`` with the current date/time/timezone for Tavily.

    Tavily's ``include_answer`` summariser is itself an LLM with a training
    cutoff; without an explicit date anchor it interprets "tonight" /
    "today" / "this week" against that cutoff. Empirically the answer for
    sports scores or breaking-news queries comes back months or seasons
    out of date.

    The prefix gives the summariser everything it needs to resolve
    relative time: ISO date for unambiguous parsing, weekday for
    "tonight" vs "yesterday" reasoning, local time for hour-of-day
    queries, and timezone so it knows what "tonight" means relative to
    the user's locale.
    """
    import datetime

    tz_name = (os.environ.get(USER_TIMEZONE_ENV) or "").strip()
    if tz_name:
        try:
            import zoneinfo

            now = datetime.datetime.now(zoneinfo.ZoneInfo(tz_name))
            tz_display = tz_name
        except Exception:
            now = datetime.datetime.now().astimezone()
            tz_display = str(now.tzinfo) if now.tzinfo else "local"
    else:
        try:
            now = datetime.datetime.now().astimezone()
            tz_display = str(now.tzinfo) if now.tzinfo else "local"
        except Exception:
            now = datetime.datetime.now(datetime.timezone.utc)
            tz_display = "UTC"

    anchor = (
        f"As of {now.strftime('%Y-%m-%d %H:%M')} "
        f"({now.strftime('%A')}, {tz_display})"
    )
    return f"{anchor}: {query}"


def _format_response(payload: Dict[str, Any], max_results: int) -> Dict[str, Any]:
    """Trim Tavily's response to what's useful for spoken output.

    - Keep ``answer`` (pre-summarised by Tavily, ideal for voice).
    - Keep ``title`` + ``url`` + truncated ``content`` per result.
    - Drop ``score``, ``raw_content``, ``response_time``, ``images`` — the
      model doesn't need them, and shorter tool output means less context
      pressure on the realtime session.
    """
    results_raw = payload.get("results") if isinstance(payload, dict) else None
    results: List[Dict[str, Any]] = []
    if isinstance(results_raw, list):
        for entry in results_raw[:max_results]:
            if not isinstance(entry, dict):
                continue
            content = entry.get("content") or ""
            if isinstance(content, str) and len(content) > RESULT_CONTENT_MAX_CHARS:
                content = content[:RESULT_CONTENT_MAX_CHARS].rstrip() + "…"
            results.append(
                {
                    "title": entry.get("title"),
                    "url": entry.get("url"),
                    "content": content,
                }
            )

    out: Dict[str, Any] = {"results": results}
    answer = payload.get("answer") if isinstance(payload, dict) else None
    if isinstance(answer, str) and answer.strip():
        out["answer"] = answer.strip()
    return out
