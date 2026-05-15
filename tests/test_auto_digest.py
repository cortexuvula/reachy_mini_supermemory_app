"""Tests for the auto-digest pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest
from reachy_mini_supermemory_app import _auto_digest as ad


# ---------- env-gate ----------


def test_is_enabled_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ad.AUTO_DIGEST_ENABLED_ENV, raising=False)
    assert ad.is_enabled() is False
    for truthy in ("true", "1", "yes", "ON"):
        monkeypatch.setenv(ad.AUTO_DIGEST_ENABLED_ENV, truthy)
        assert ad.is_enabled() is True, truthy
    for falsy in ("false", "0", "no", "", "maybe"):
        monkeypatch.setenv(ad.AUTO_DIGEST_ENABLED_ENV, falsy)
        assert ad.is_enabled() is False, falsy


def test_install_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ad.AUTO_DIGEST_ENABLED_ENV, raising=False)
    target = logging.getLogger("test.fake.console.disabled")
    target.handlers.clear()
    assert ad.install(target.name) is None
    assert not any(isinstance(h, ad.TranscriptCapture) for h in target.handlers)


# ---------- TranscriptCapture ----------


def test_capture_picks_up_user_and_assistant_turns() -> None:
    logger_name = "test.fake.console.cap1"
    target = logging.getLogger(logger_name)
    target.handlers.clear()
    target.setLevel(logging.INFO)
    cap = ad.TranscriptCapture()
    target.addHandler(cap)
    try:
        target.info("role=user content=Hello Reachy")
        target.info("role=assistant content=Hey there")
        target.info("role=user_partial content=ignor")  # partials excluded
        target.info("unrelated log line about audio")
    finally:
        target.removeHandler(cap)

    items = cap.drain()
    assert [(r, c) for _ts, r, c in items] == [
        ("user", "Hello Reachy"),
        ("assistant", "Hey there"),
    ]


def test_capture_snapshot_and_drain() -> None:
    logger_name = "test.fake.console.cap2"
    target = logging.getLogger(logger_name)
    target.handlers.clear()
    target.setLevel(logging.INFO)
    cap = ad.TranscriptCapture()
    target.addHandler(cap)
    try:
        target.info("role=user content=one")
        target.info("role=assistant content=two")
        n, elapsed = cap.snapshot()
        assert n == 2
        assert elapsed >= 0
        drained = cap.drain()
        assert len(drained) == 2
        # After drain, snapshot reports zero.
        n2, _ = cap.snapshot()
        assert n2 == 0
    finally:
        target.removeHandler(cap)


def test_capture_ignores_lines_without_role_content() -> None:
    target = logging.getLogger("test.fake.console.cap3")
    target.handlers.clear()
    target.setLevel(logging.INFO)
    cap = ad.TranscriptCapture()
    target.addHandler(cap)
    try:
        target.info("Realtime session initialized with profile='supermemory'")
        target.info("Tool call received - tool_name='dance'")
    finally:
        target.removeHandler(cap)
    assert cap.drain() == []


def test_capture_filters_tool_result_json_payloads() -> None:
    target = logging.getLogger("test.fake.console.cap4")
    target.handlers.clear()
    target.setLevel(logging.INFO)
    cap = ad.TranscriptCapture()
    target.addHandler(cap)
    try:
        target.info('role=assistant content={"status": "queued", "emotion": "shy1"}')
        target.info('role=assistant content={"ok": true, "memory_id": "abc"}')
        target.info("role=assistant content=That's sweet of you — thank you.")
    finally:
        target.removeHandler(cap)

    items = cap.drain()
    assert [(r, c) for _ts, r, c in items] == [
        ("assistant", "That's sweet of you — thank you."),
    ]


# ---------- summarise ----------


@pytest.mark.asyncio
async def test_summarise_returns_none_when_no_token(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ad.HF_TOKEN_ENVS:
        monkeypatch.delenv(name, raising=False)
    assert await ad.summarise("user: hi") is None


class _FakeResp:
    """Minimal stand-in for httpx.Response used by the summarise tests."""

    def __init__(self, status_code: int, json_body: Optional[Dict[str, Any]] = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._json


class _FakeClient:
    """Stand-in for the cached httpx.AsyncClient. Returns ``response``, or raises it if it's an Exception."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def post(self, *a: Any, **kw: Any) -> Any:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, response: Any) -> None:
    """Replace the per-loop cache lookup with a single shared FakeClient instance."""
    fake = _FakeClient(response)
    monkeypatch.setattr(ad, "_client_for_current_loop", lambda: fake)


@pytest.mark.asyncio
async def test_summarise_returns_none_on_skip_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _install_fake_client(
        monkeypatch,
        _FakeResp(200, {"choices": [{"message": {"content": "skip"}}]}, text="skip"),
    )
    assert await ad.summarise("user: hi") is None


@pytest.mark.asyncio
async def test_summarise_returns_text_trimmed_to_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    long = "Andre mentioned X. " * 100  # ~1900 chars
    _install_fake_client(
        monkeypatch, _FakeResp(200, {"choices": [{"message": {"content": long}}]})
    )
    result = await ad.summarise("user: stuff")
    assert result is not None
    assert len(result) <= ad.SUMMARY_MAX_CHARS


@pytest.mark.asyncio
async def test_summarise_returns_none_on_http_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _install_fake_client(monkeypatch, _FakeResp(401, text="unauthorized"))
    assert await ad.summarise("user: hi") is None


@pytest.mark.asyncio
async def test_summarise_raises_transient_on_http_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """5xx is retryable — must raise, not silently drop."""
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _install_fake_client(monkeypatch, _FakeResp(503, text="service unavailable"))
    with pytest.raises(ad._SummariseTransientError):
        await ad.summarise("user: hi")


@pytest.mark.asyncio
async def test_summarise_raises_transient_on_429(monkeypatch: pytest.MonkeyPatch) -> None:
    """Rate-limit is retryable — must raise."""
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _install_fake_client(monkeypatch, _FakeResp(429, text="rate limited"))
    with pytest.raises(ad._SummariseTransientError):
        await ad.summarise("user: hi")


@pytest.mark.asyncio
async def test_summarise_raises_transient_on_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    _install_fake_client(monkeypatch, ad.httpx.ConnectError("connection refused"))
    with pytest.raises(ad._SummariseTransientError):
        await ad.summarise("user: hi")


# ---------- save_digest ----------


@pytest.mark.asyncio
async def test_save_digest_posts_to_v4_memories(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(path: str, body: dict) -> dict:
        captured["path"] = path
        captured["body"] = body
        return {"memories": [{"id": "abc"}]}

    monkeypatch.setattr("reachy_mini_supermemory_app._supermemory_client.post_json", AsyncMock(side_effect=fake_post))
    monkeypatch.setattr(
        "reachy_mini_supermemory_app._supermemory_client.derive_container_tag",
        lambda profile=None: "reachy-mini:test",
    )
    ok = await ad.save_digest("summary text about Andre's day")
    assert ok is True
    assert captured["path"] == "/v4/memories"
    assert captured["body"]["containerTag"] == "reachy-mini:test"
    memories = captured["body"]["memories"]
    assert len(memories) == 1
    assert memories[0]["content"] == "summary text about Andre's day"
    assert memories[0]["metadata"]["kind"] == "digest"
    assert "generated_at" in memories[0]["metadata"]


@pytest.mark.asyncio
async def test_save_digest_returns_false_on_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "reachy_mini_supermemory_app._supermemory_client.post_json",
        AsyncMock(return_value={"error": "boom"}),
    )
    monkeypatch.setattr(
        "reachy_mini_supermemory_app._supermemory_client.derive_container_tag",
        lambda profile=None: "reachy-mini:test",
    )
    assert await ad.save_digest("anything") is False


# ---------- transcript regex self-probe ----------


def test_probe_transcript_regex_returns_true_for_current_format() -> None:
    """Default regex must match the literal upstream format we expect."""
    assert ad._probe_transcript_regex() is True


def test_probe_transcript_regex_detects_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    """A regex that won't match the sample must fail the probe."""
    import re

    drifted = re.compile(r"^WILL_NEVER_MATCH")
    monkeypatch.setattr(ad, "_TRANSCRIPT_RE", drifted)
    assert ad._probe_transcript_regex() is False


def test_install_warns_when_probe_fails(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Install must emit a startup warning when the probe finds drift."""
    import re

    monkeypatch.setenv(ad.AUTO_DIGEST_ENABLED_ENV, "true")
    monkeypatch.setattr(ad, "_TRANSCRIPT_RE", re.compile(r"^DRIFTED"))
    target = logging.getLogger("test.fake.console.probe-fail")
    target.handlers.clear()

    ad.install(target.name)

    err = capsys.readouterr().err
    assert "transcript regex self-probe failed" in err


# ---------- env-tunable knobs ----------


def test_summary_max_chars_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ad.SUMMARY_MAX_CHARS_ENV, "42")
    assert ad.summary_max_chars() == 42
    monkeypatch.delenv(ad.SUMMARY_MAX_CHARS_ENV, raising=False)
    assert ad.summary_max_chars() == ad.SUMMARY_MAX_CHARS


def test_transcript_max_chars_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ad.TRANSCRIPT_MAX_CHARS_ENV, "1000")
    assert ad.transcript_max_chars() == 1000


def test_max_buffer_turns_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ad.MAX_BUFFER_TURNS_ENV, "7")
    assert ad.max_buffer_turns() == 7


def test_env_overrides_invalid_value_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bad input keeps the default rather than raising — daemon must not crash on a typo."""
    monkeypatch.setenv(ad.TRANSCRIPT_MAX_CHARS_ENV, "not-an-int")
    assert ad.transcript_max_chars() == ad.TRANSCRIPT_MAX_CHARS


# ---------- transcript cap ----------


def test_format_transcript_passthrough_when_under_cap() -> None:
    items = [(0.0, "user", "hi"), (0.0, "assistant", "hello")]
    out = ad._format_transcript(items)
    assert out == "user: hi\nassistant: hello"


def test_format_transcript_caps_at_max_chars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ad, "TRANSCRIPT_MAX_CHARS", 200)
    items: List[Any] = []
    for i in range(50):
        items.append((float(i), "user", f"turn-{i:04d} " + "x" * 20))
    out = ad._format_transcript(items)
    assert len(out) <= 200
    assert out.startswith("[...earlier turns elided")
    # Most recent turn must still be present.
    assert "turn-0049" in out


def test_format_transcript_truncation_snaps_to_newline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tail must start at a turn boundary so the summariser doesn't see half a line."""
    monkeypatch.setattr(ad, "TRANSCRIPT_MAX_CHARS", 200)
    items = [(0.0, "user", "x" * 50), (0.0, "assistant", "y" * 50), (0.0, "user", "tail")]
    out = ad._format_transcript(items)
    body = out.split("\n", 1)[1]  # drop the elided-marker line
    # Body must start with a complete role tag, not mid-content.
    assert body.split(":", 1)[0] in {"user", "assistant"}


# ---------- requeue / data-loss protection ----------


def test_requeue_prepends_items_into_buffer() -> None:
    cap = ad.TranscriptCapture()
    target = logging.getLogger("test.fake.console.requeue1")
    target.handlers.clear()
    target.setLevel(logging.INFO)
    target.addHandler(cap)
    try:
        target.info("role=user content=newest")
    finally:
        target.removeHandler(cap)

    cap.requeue([(0.0, "user", "older1"), (0.0, "assistant", "older2")])
    items = cap.drain()
    assert [(r, c) for _ts, r, c in items] == [
        ("user", "older1"),
        ("assistant", "older2"),
        ("user", "newest"),
    ]


def test_requeue_caps_at_max_buffer_turns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stuck summariser must not grow the buffer unboundedly."""
    monkeypatch.setattr(ad, "MAX_BUFFER_TURNS", 4)
    cap = ad.TranscriptCapture()
    # Pretend a long conversation is already buffered.
    cap.requeue([(0.0, "user", f"old{i}") for i in range(3)])
    # Now requeue a previous batch on top — overflow drops oldest.
    cap.requeue([(0.0, "user", f"older{i}") for i in range(3)])
    items = cap.drain()
    assert len(items) == 4
    # Newest items in the buffer survive (the FIFO tail). The two oldest of
    # the older batch get dropped because they push past MAX_BUFFER_TURNS.
    contents = [c for _ts, _r, c in items]
    assert contents == ["older2", "old0", "old1", "old2"]


def test_requeue_noop_on_empty_list() -> None:
    cap = ad.TranscriptCapture()
    cap.requeue([])
    assert cap.drain() == []


@pytest.mark.asyncio
async def test_run_digest_once_returns_false_on_transient_summarise_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _raises(_t: str) -> str:
        raise ad._SummariseTransientError("boom")

    monkeypatch.setattr(ad, "summarise", _raises)
    keep = await ad._run_digest_once([(0.0, "user", "hi")])
    assert keep is False


@pytest.mark.asyncio
async def test_run_digest_once_returns_true_on_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _skip(_t: str) -> Any:
        return None  # model said skip / empty / no token

    monkeypatch.setattr(ad, "summarise", _skip)
    keep = await ad._run_digest_once([(0.0, "user", "hi")])
    assert keep is True


@pytest.mark.asyncio
async def test_run_digest_once_returns_false_on_save_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _summary(_t: str) -> str:
        return "the user said something durable"

    async def _save_fails(_s: str) -> bool:
        return False

    monkeypatch.setattr(ad, "summarise", _summary)
    monkeypatch.setattr(ad, "save_digest", _save_fails)
    keep = await ad._run_digest_once([(0.0, "user", "hi")])
    assert keep is False


@pytest.mark.asyncio
async def test_run_digest_once_returns_true_on_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _summary(_t: str) -> str:
        return "the user said something durable"

    async def _save_ok(_s: str) -> bool:
        return True

    monkeypatch.setattr(ad, "summarise", _summary)
    monkeypatch.setattr(ad, "save_digest", _save_ok)
    keep = await ad._run_digest_once([(0.0, "user", "hi")])
    assert keep is True


# ---------- persistent-loop client pooling ----------


def test_digest_loop_pools_httpx_client_across_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The persistent loop must reuse one supermemory_client across iterations.

    The pre-fix loop called ``asyncio.run`` per iteration, which created a
    fresh event loop and (post-bug-2-fix) a fresh httpx client every time —
    defeating the connection-pool optimisation. Here we run two iterations
    inside a real persistent loop and verify the client cache stays put.
    """
    import threading
    import time as time_mod

    from reachy_mini_supermemory_app import _supermemory_client as smc
    from reachy_mini_supermemory_app._testing import reset_supermemory_clients

    reset_supermemory_clients()
    seen_client_ids: List[int] = []

    async def _fake_run_digest_once(items: List[Any]) -> bool:
        # Stand-in for what save_digest -> post_json would do: ask for the
        # cached client on the current loop. The persistent loop must yield
        # the same instance every iteration.
        seen_client_ids.append(id(smc._client_for_current_loop()))
        return True

    monkeypatch.setattr(ad, "_run_digest_once", _fake_run_digest_once)
    monkeypatch.setattr(ad, "CHECK_INTERVAL_S", 0.01)

    cap = ad.TranscriptCapture()
    # Two iterations' worth of buffered turns, idle "now" so each tick fires.
    cap._buffer = [(0.0, "user", "hi"), (0.0, "assistant", "hello")]
    cap._last_activity_at = time_mod.monotonic() - 9999

    stop = threading.Event()
    t = threading.Thread(target=ad._digest_loop, args=(cap, 0.0, 1, stop), daemon=True)
    t.start()

    # Wait for at least two iterations to record their client id, then stop.
    deadline = time_mod.monotonic() + 2.0
    while time_mod.monotonic() < deadline and len(seen_client_ids) < 2:
        # Refill the buffer so each idle check has work to do.
        with cap._lock:
            if not cap._buffer:
                cap._buffer = [(0.0, "user", "again"), (0.0, "assistant", "ack")]
                cap._last_activity_at = time_mod.monotonic() - 9999
        time_mod.sleep(0.02)
    stop.set()
    t.join(timeout=2.0)

    assert len(seen_client_ids) >= 2, "expected at least two digest iterations"
    assert len(set(seen_client_ids)) == 1, (
        f"client churned across iterations: {seen_client_ids}"
    )
