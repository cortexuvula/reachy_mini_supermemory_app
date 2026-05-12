"""Tests for the auto-digest pipeline."""

from __future__ import annotations

import logging
from typing import Any, Dict, List
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


@pytest.mark.asyncio
async def test_summarise_returns_none_on_skip_sentinel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    class FakeResp:
        status_code = 200

        def json(self) -> Dict[str, Any]:
            return {"choices": [{"message": {"content": "skip"}}]}

        text = "skip"

    class FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, *a: Any, **kw: Any) -> FakeResp:
            return FakeResp()

    monkeypatch.setattr(ad.httpx, "AsyncClient", FakeClient)
    assert await ad.summarise("user: hi") is None


@pytest.mark.asyncio
async def test_summarise_returns_text_trimmed_to_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    long = "Andre mentioned X. " * 100  # ~1900 chars

    class FakeResp:
        status_code = 200
        text = ""

        def json(self) -> Dict[str, Any]:
            return {"choices": [{"message": {"content": long}}]}

    class FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, *a: Any, **kw: Any) -> FakeResp:
            return FakeResp()

    monkeypatch.setattr(ad.httpx, "AsyncClient", FakeClient)
    result = await ad.summarise("user: stuff")
    assert result is not None
    assert len(result) <= ad.SUMMARY_MAX_CHARS


@pytest.mark.asyncio
async def test_summarise_returns_none_on_http_4xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")

    class FakeResp:
        status_code = 401
        text = "unauthorized"

        def json(self) -> Dict[str, Any]:
            return {}

    class FakeClient:
        def __init__(self, *a: Any, **kw: Any) -> None:
            pass

        async def __aenter__(self) -> "FakeClient":
            return self

        async def __aexit__(self, *a: Any) -> None:
            return None

        async def post(self, *a: Any, **kw: Any) -> FakeResp:
            return FakeResp()

    monkeypatch.setattr(ad.httpx, "AsyncClient", FakeClient)
    assert await ad.summarise("user: hi") is None


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
