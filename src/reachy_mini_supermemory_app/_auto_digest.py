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

from ._log_utils import startup_log
from ._supermemory_client import _client_for_current_loop

logger = logging.getLogger(__name__)

AUTO_DIGEST_ENABLED_ENV = "SUPERMEMORY_AUTO_DIGEST"
IDLE_MINUTES_ENV = "SUPERMEMORY_DIGEST_IDLE_MINUTES"
MIN_TURNS_ENV = "SUPERMEMORY_DIGEST_MIN_TURNS"
DIGEST_MODEL_ENV = "SUPERMEMORY_DIGEST_MODEL"
DIGEST_API_URL_ENV = "SUPERMEMORY_DIGEST_API_URL"
SUMMARY_MAX_CHARS_ENV = "SUPERMEMORY_DIGEST_SUMMARY_MAX_CHARS"
TRANSCRIPT_MAX_CHARS_ENV = "SUPERMEMORY_DIGEST_TRANSCRIPT_MAX_CHARS"
MAX_BUFFER_TURNS_ENV = "SUPERMEMORY_DIGEST_MAX_BUFFER_TURNS"
HF_TOKEN_ENVS = ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN")

DEFAULT_IDLE_MINUTES = 10
DEFAULT_MIN_TURNS = 4
DEFAULT_DIGEST_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DIGEST_API_URL = "https://router.huggingface.co/v1/chat/completions"
CHECK_INTERVAL_S = 60.0
SKIP_SENTINEL = "skip"

# Defaults for the env-tunable knobs below. The constants remain module-level
# so tests can monkeypatch them directly; the helpers consult the env first
# and fall back to these. Run-time edits via the settings UI therefore take
# effect without a restart, and unit tests can dial values down without
# touching the environment.
SUMMARY_MAX_CHARS = 600
# Conservative cap on transcript chars sent to the summariser. Llama-3.1-8B
# handles 128k tokens, but the HF Router default is lower and other free-tier
# models cap around 8k. ~32 KB ≈ 8k tokens — well within any reasonable model
# context, and keeps any pathological buffer growth from rejecting the whole
# digest (which would then re-queue → re-fail forever).
TRANSCRIPT_MAX_CHARS = 32_000
# Hard cap on retained turns: when summarisation stays broken (e.g. HF token
# revoked) we requeue items to retry next idle window, but the buffer must not
# grow without bound. We drop the oldest turns past this cap — losing the tail
# is preferable to OOM in a long-running daemon.
MAX_BUFFER_TURNS = 500


def summary_max_chars() -> int:
    """Cap on chars the digest summariser is allowed to write to supermemory."""
    return _env_int(SUMMARY_MAX_CHARS_ENV, SUMMARY_MAX_CHARS)


def transcript_max_chars() -> int:
    """Cap on chars fed into the HF Router summarisation call."""
    return _env_int(TRANSCRIPT_MAX_CHARS_ENV, TRANSCRIPT_MAX_CHARS)


def max_buffer_turns() -> int:
    """Cap on retained transcript turns awaiting a successful digest."""
    return _env_int(MAX_BUFFER_TURNS_ENV, MAX_BUFFER_TURNS)
# Hard cap on retained turns: when summarisation stays broken (e.g. HF token
# revoked) we requeue items to retry next idle window, but the buffer must not
# grow without bound. We drop the oldest turns past this cap — losing the tail
# is preferable to OOM in a long-running daemon.
MAX_BUFFER_TURNS = 500


class _SummariseTransientError(RuntimeError):
    """Raised when summarisation fails for a reason worth retrying (network, 5xx, 429).

    Callers re-queue the drained transcript so a transient HF outage doesn't
    silently discard a conversation. Permanent failures (missing token, 4xx,
    bad JSON, skip sentinel) keep the existing 'return None' contract so the
    items are dropped as intentional.
    """

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
        # Drop transcripts while the user has privacy mode on — side
        # conversations shouldn't end up in the supermemory archive.
        try:
            from ._privacy_mode import is_privacy_active

            if is_privacy_active():
                return
        except Exception:
            pass
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

    def requeue(self, items: List[Tuple[float, str, str]]) -> None:
        """Re-insert items at the head of the buffer after a transient failure.

        Caps total buffer at ``max_buffer_turns()`` so a stuck summariser
        can't grow the buffer indefinitely; oldest items are dropped on
        overflow.
        """
        if not items:
            return
        cap = max_buffer_turns()
        with self._lock:
            combined = items + self._buffer
            if len(combined) > cap:
                combined = combined[-cap:]
            self._buffer = combined


def _format_transcript(items: List[Tuple[float, str, str]]) -> str:
    """Render the turns as a chat transcript, capped at ``TRANSCRIPT_MAX_CHARS``.

    Truncation drops the OLDEST turns: the digest is meant to capture durable
    user facts, and recent turns are more likely to contain crisp statements
    than the warm-up of a long session. A truncation marker is prepended so
    the summariser knows context is missing and won't over-anchor on the
    first preserved line.
    """
    cap = transcript_max_chars()
    full = "\n".join(f"{role}: {content}" for _ts, role, content in items)
    if len(full) <= cap:
        return full
    marker = "[...earlier turns elided due to transcript-size cap...]\n"
    keep = cap - len(marker)
    # Snap to the next newline so we don't slice mid-line.
    tail = full[-keep:]
    nl = tail.find("\n")
    if nl != -1 and nl < len(tail) - 1:
        tail = tail[nl + 1:]
    return marker + tail


async def summarise(transcript: str) -> Optional[str]:
    """Call HF's OpenAI-compatible chat-completions endpoint to summarise a transcript.

    Returns ``None`` for intentional drops: no token, malformed/empty model
    response, or the model returning the ``skip`` sentinel. Raises
    ``_SummariseTransientError`` for transient transport failures (network,
    5xx, 429) so the caller can re-queue the transcript and retry next idle
    window. 4xx responses other than 429 stay as ``None`` because retrying
    won't fix an auth or schema problem.
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
        client = _client_for_current_loop()
        # 30 s is generous compared to supermemory's 10 s default — the HF
        # Router can stall on the first cold inference for a couple of
        # seconds. Per-request override keeps the rest of the shared client
        # at its tighter default.
        response = await client.post(api_url, headers=headers, json=body, timeout=30.0)
    except httpx.HTTPError as e:
        logger.warning("Auto-digest summariser HTTP error: %s", e)
        raise _SummariseTransientError(str(e)) from e

    if response.status_code == 429 or response.status_code >= 500:
        logger.warning(
            "Auto-digest summariser %s -> HTTP %s (transient): %s",
            model,
            response.status_code,
            response.text[:200],
        )
        raise _SummariseTransientError(f"HTTP {response.status_code}")
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
    return text[: summary_max_chars()]


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
    """Background loop: wait for idle, summarise, save, repeat.

    On transient failure (network blip, HF 5xx, supermemory save error) the
    drained turns get re-queued so the next idle window retries them — without
    this, a single failed POST silently discarded the whole conversation.

    Runs a single persistent event loop for the lifetime of the thread (rather
    than ``asyncio.run`` per iteration). That lets the supermemory_client's
    per-loop httpx client survive between iterations, so the digest path
    pools its TCP+TLS connection to ``api.supermemory.ai`` instead of paying a
    fresh handshake every time.
    """
    loop = asyncio.new_event_loop()
    try:
        while not stop_event.is_set():
            if stop_event.wait(CHECK_INTERVAL_S):
                break
            n, elapsed = capture.snapshot()
            if n < min_turns or elapsed < idle_seconds:
                continue
            items = capture.drain()
            if len(items) < min_turns:
                continue
            try:
                keep = loop.run_until_complete(_run_digest_once(items))
            except Exception as e:
                logger.warning("Auto-digest loop iteration failed: %s", e)
                keep = False  # unknown failure → safer to retry than to drop
            if not keep:
                capture.requeue(items)
    finally:
        # Close the cached httpx client before tearing down the loop, so the
        # pool releases its sockets cleanly instead of getting GC'd later.
        from ._supermemory_client import aclose_client_for_current_loop

        try:
            loop.run_until_complete(aclose_client_for_current_loop())
        except Exception:
            pass
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
        except Exception:
            pass
        loop.close()


async def _run_digest_once(items: List[Tuple[float, str, str]]) -> bool:
    """Summarise + save one batch.

    Returns ``True`` when the items should be discarded (success, or
    intentional skip — e.g. model returned 'skip', or no HF token), and
    ``False`` when the caller should re-queue them for a retry on the next
    idle window (transient HF / supermemory failure).
    """
    transcript = _format_transcript(items)
    try:
        summary = await summarise(transcript)
    except _SummariseTransientError as e:
        logger.info("Auto-digest summariser unavailable (%s); re-queuing %d turns", e, len(items))
        return False
    if not summary:
        logger.info("Auto-digest produced no summary for %d turns; nothing saved", len(items))
        return True
    saved = await save_digest(summary)
    if not saved:
        logger.info("Auto-digest save failed; re-queuing %d turns", len(items))
        return False
    return True


_TRANSCRIPT_PROBE_USER = "__supermemory_probe_user__"
_TRANSCRIPT_PROBE_CONTENT = "__probe_content__"


def _probe_transcript_regex() -> bool:
    """Return True if ``_TRANSCRIPT_RE`` matches the documented upstream log format.

    Upstream emits per-turn log lines like ``role=user content=<text>`` on the
    ``reachy_mini_conversation_app.console`` logger; ``_TRANSCRIPT_RE`` parses
    that. If upstream ever changes the format (added quoting, structured
    record, new field separator), auto-digest would silently buffer nothing.
    This probe verifies the assumption against the literal format the regex
    expects so a drift fails LOUDLY at install instead of producing zero
    digests over weeks.
    """
    sample = f"role={_TRANSCRIPT_PROBE_USER} content={_TRANSCRIPT_PROBE_CONTENT}"
    match = _TRANSCRIPT_RE.match(sample)
    if match is None:
        return False
    role, content = match.group(1), match.group(2).strip()
    return role == _TRANSCRIPT_PROBE_USER and content == _TRANSCRIPT_PROBE_CONTENT


def install(target_logger_name: str = "reachy_mini_conversation_app.console") -> Optional[TranscriptCapture]:
    """Attach the capture handler and start the digest loop. Returns the handler.

    No-op (returns None) when ``SUPERMEMORY_AUTO_DIGEST`` is not truthy.
    """
    if not is_enabled():
        return None
    if not _probe_transcript_regex():
        # Loud failure: if the regex has drifted (someone edited it, upstream
        # changed format and we adapted the regex incorrectly), the install
        # banner should tell the operator instead of letting the daemon
        # appear to work while quietly capturing nothing.
        startup_log(
            "Auto-digest WARNING: transcript regex self-probe failed — "
            "log format may have drifted; capture may produce empty digests",
            logger=logger,
        )
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
    startup_log(
        f"Auto-digest enabled: idle={idle_seconds}s, min_turns={min_turns}, "
        f"model={os.environ.get(DIGEST_MODEL_ENV, DEFAULT_DIGEST_MODEL)}",
        logger=logger,
    )
    return capture
