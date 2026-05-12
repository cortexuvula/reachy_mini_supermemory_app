"""Opportunistic auto-digest of conversation transcripts to supermemory.

Captures user/assistant transcript lines off the conversation app's ``console``
logger, waits for idle (no new turns for N minutes), summarises the
accumulated turns via HF's OpenAI-compatible chat-completions endpoint, and
writes the summary to supermemory as a single ``save_memory``-style entry.

Opt-in via ``SUPERMEMORY_AUTO_DIGEST=true``. The summariser is instructed to
return the literal string ``skip`` when there's nothing worth saving, so
quiet sessions don't pollute the archive.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)

AUTO_DIGEST_ENABLED_ENV = "SUPERMEMORY_AUTO_DIGEST"
IDLE_MINUTES_ENV = "SUPERMEMORY_DIGEST_IDLE_MINUTES"
MIN_TURNS_ENV = "SUPERMEMORY_DIGEST_MIN_TURNS"
DIGEST_MODEL_ENV = "SUPERMEMORY_DIGEST_MODEL"
DIGEST_API_URL_ENV = "SUPERMEMORY_DIGEST_API_URL"
HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

DEFAULT_IDLE_MINUTES = 10
DEFAULT_MIN_TURNS = 4
DEFAULT_DIGEST_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DIGEST_API_URL = "https://router.huggingface.co/v1/chat/completions"
CHECK_INTERVAL_S = 60.0
SUMMARY_MAX_CHARS = 600
SKIP_SENTINEL = "skip"

_TRANSCRIPT_RE = re.compile(r"role=(\w+)\s+content=(.*)", re.DOTALL)
_INTERESTING_ROLES = frozenset({"user", "assistant"})

_SYSTEM_PROMPT = (
    "You summarise voice conversations between a user and Reachy Mini (a small desk robot). "
    "Given a transcript, write ONE compact paragraph (~400 chars max) capturing what the USER "
    "shared that's worth remembering across future sessions: durable facts, preferences, "
    "decisions, recurring people, plans. Skip small talk, greetings, and the robot's own "
    "responses. Use past tense, third person ('the user said …'). Return the literal string "
    f"'{SKIP_SENTINEL}' (no quotes, no punctuation) when the transcript has no durable content."
)


def _get_token() -> Optional[str]:
    for name in HF_TOKEN_ENVS:
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


def is_enabled() -> bool:
    """Return True when auto-digest should run (env-var gated)."""
    return (os.environ.get(AUTO_DIGEST_ENABLED_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


class TranscriptCapture(logging.Handler):
    """Logging handler that buffers user/assistant transcript lines in memory.

    Attaches to ``reachy_mini_conversation_app.console`` (where the conversation
    app emits ``role=… content=…`` per turn). The handler is additive — normal
    log output still propagates to the journal.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self._buffer: List[Tuple[float, str, str]] = []
        self._lock = threading.Lock()
        self._last_activity_at: float = time.monotonic()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
        except Exception:
            return
        match = _TRANSCRIPT_RE.match(msg)
        if not match:
            return
        role, content = match.group(1), match.group(2).strip()
        if role not in _INTERESTING_ROLES or not content:
            return
        # Skip tool-result JSON payloads — they're framework chatter, not transcript.
        if content.startswith("{") and content.rstrip().endswith("}"):
            return
        with self._lock:
            self._buffer.append((time.monotonic(), role, content))
            self._last_activity_at = time.monotonic()

    def snapshot(self) -> Tuple[int, float]:
        """Return ``(buffered_turn_count, seconds_since_last_turn)`` without draining."""
        with self._lock:
            n = len(self._buffer)
            elapsed = time.monotonic() - self._last_activity_at if n else float("inf")
            return n, elapsed

    def drain(self) -> List[Tuple[float, str, str]]:
        """Atomically pull all buffered turns and reset the buffer."""
        with self._lock:
            items = list(self._buffer)
            self._buffer.clear()
            return items


def _format_transcript(items: List[Tuple[float, str, str]]) -> str:
    return "\n".join(f"{role}: {content}" for _ts, role, content in items)


async def summarise(transcript: str) -> Optional[str]:
    """Call HF's OpenAI-compatible chat-completions endpoint to summarise a transcript.

    Returns ``None`` when summarisation can't run (no token, transport error,
    bad response) or when the model says ``skip``.
    """
    token = _get_token()
    if not token:
        logger.warning("Auto-digest skipped: no HF token in env (%s)", "/".join(HF_TOKEN_ENVS))
        return None
    api_url = (os.environ.get(DIGEST_API_URL_ENV) or DEFAULT_DIGEST_API_URL).strip()
    model = (os.environ.get(DIGEST_MODEL_ENV) or DEFAULT_DIGEST_MODEL).strip()

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        "max_tokens": 220,
        "temperature": 0.3,
    }
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(api_url, headers=headers, json=body)
    except httpx.HTTPError as e:
        logger.warning("Auto-digest summariser HTTP error: %s", e)
        return None

    if response.status_code >= 400:
        logger.warning("Auto-digest summariser %s -> HTTP %s: %s", model, response.status_code, response.text[:200])
        return None

    try:
        payload = response.json()
    except ValueError:
        return None
    try:
        text = payload["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError):
        return None
    if not text or text.lower() == SKIP_SENTINEL:
        return None
    return text[:SUMMARY_MAX_CHARS]


async def save_digest(summary: str) -> bool:
    """Write the digest as a single supermemory entry under the active profile's tag."""
    # Imported lazily so this module is testable without the rest of the client
    # being fully set up (and so the import cycle stays one-way).
    from ._supermemory_client import derive_container_tag, post_json

    body: Dict[str, Any] = {
        "memories": [
            {
                "content": summary,
                "metadata": {
                    "kind": "digest",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                },
            }
        ],
        "containerTag": derive_container_tag(),
    }
    result = await post_json("/v4/memories", body)
    if "error" in result:
        logger.warning("Auto-digest save failed: %s", result.get("error"))
        return False
    logger.info("Auto-digest saved (%d chars)", len(summary))
    return True


def _digest_loop(
    capture: TranscriptCapture,
    idle_seconds: float,
    min_turns: int,
    stop_event: threading.Event,
) -> None:
    """Background loop: wait for idle, summarise, save, repeat."""
    while not stop_event.is_set():
        if stop_event.wait(CHECK_INTERVAL_S):
            return
        n, elapsed = capture.snapshot()
        if n < min_turns or elapsed < idle_seconds:
            continue
        items = capture.drain()
        if len(items) < min_turns:
            continue
        try:
            asyncio.run(_run_digest_once(items))
        except Exception as e:
            logger.warning("Auto-digest loop iteration failed: %s", e)


async def _run_digest_once(items: List[Tuple[float, str, str]]) -> None:
    transcript = _format_transcript(items)
    summary = await summarise(transcript)
    if not summary:
        logger.info("Auto-digest produced no summary for %d turns; nothing saved", len(items))
        return
    await save_digest(summary)


def install(target_logger_name: str = "reachy_mini_conversation_app.console") -> Optional[TranscriptCapture]:
    """Attach the capture handler and start the digest loop. Returns the handler.

    No-op (returns None) when ``SUPERMEMORY_AUTO_DIGEST`` is not truthy.
    """
    if not is_enabled():
        return None
    target = logging.getLogger(target_logger_name)
    # Idempotent: only one capture per logger.
    for h in target.handlers:
        if isinstance(h, TranscriptCapture):
            return h
    capture = TranscriptCapture()
    target.addHandler(capture)
    idle_seconds = _env_int(IDLE_MINUTES_ENV, DEFAULT_IDLE_MINUTES) * 60
    min_turns = _env_int(MIN_TURNS_ENV, DEFAULT_MIN_TURNS)
    stop_event = threading.Event()
    capture._stop_event = stop_event  # type: ignore[attr-defined]  # for testability
    thread = threading.Thread(
        target=_digest_loop,
        args=(capture, idle_seconds, min_turns, stop_event),
        name="supermemory-auto-digest",
        daemon=True,
    )
    thread.start()
    # print() rather than logger.info() because install() runs before upstream
    # configures the root logger, so an INFO log here would go to the void.
    print(
        f"Auto-digest enabled: idle={idle_seconds}s, min_turns={min_turns}, "
        f"model={os.environ.get(DIGEST_MODEL_ENV, DEFAULT_DIGEST_MODEL)}"
    )
    return capture
