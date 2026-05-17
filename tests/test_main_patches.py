"""Tests for the runtime monkey-patches in ``reachy_mini_supermemory_app.main``.

These patches mutate ``sys.modules`` and module attributes, so the tests use
fake upstream stubs registered in ``sys.modules`` instead of relying on the
real conversation_app being installed.
"""

from __future__ import annotations

import importlib.util
import sys
import threading
import time
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


# Lazy imports so the module-load-time stubs from conftest.py are in place first.
def _import_main():
    from reachy_mini_supermemory_app import main as m

    return m


# ============================================================================
# _preload_unlocked_upstream_config
# ============================================================================


@pytest.fixture
def restore_upstream_config():
    """Snapshot+restore the upstream config sys.modules entry around each test."""
    name = "reachy_mini_conversation_app.config"
    saved = sys.modules.get(name)
    sys.modules.pop(name, None)
    yield
    sys.modules.pop(name, None)
    if saved is not None:
        sys.modules[name] = saved


def _fake_config_spec(tmp_path: Path, source: str) -> importlib.util.spec_from_file_location:
    fake_src = tmp_path / "fake_upstream_config.py"
    fake_src.write_text(source, encoding="utf-8")
    return importlib.util.spec_from_file_location(
        "reachy_mini_conversation_app.config", str(fake_src)
    )


def test_preload_swaps_locked_profile_to_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_upstream_config: None
) -> None:
    main = _import_main()
    spec = _fake_config_spec(
        tmp_path,
        'LOCKED_PROFILE: str | None = "companion"\nMARKER = "loaded"\n',
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "reachy_mini_conversation_app.config" else None
    )

    main._preload_unlocked_upstream_config()

    mod = sys.modules["reachy_mini_conversation_app.config"]
    assert mod.LOCKED_PROFILE is None
    assert mod.MARKER == "loaded"


def test_preload_is_noop_when_already_unlocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_upstream_config: None
) -> None:
    main = _import_main()
    spec = _fake_config_spec(tmp_path, "LOCKED_PROFILE: str | None = None\n")
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "reachy_mini_conversation_app.config" else None
    )

    main._preload_unlocked_upstream_config()

    # When already None, we should NOT have created a sys.modules entry.
    assert "reachy_mini_conversation_app.config" not in sys.modules


def test_preload_replaces_all_locked_profile_assignments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_upstream_config: None
) -> None:
    main = _import_main()
    spec = _fake_config_spec(
        tmp_path,
        'LOCKED_PROFILE: str | None = "first"\n'
        "X = 1\n"
        'LOCKED_PROFILE = "second"\n',  # second top-level assignment
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "reachy_mini_conversation_app.config" else None
    )

    main._preload_unlocked_upstream_config()
    mod = sys.modules["reachy_mini_conversation_app.config"]
    assert mod.LOCKED_PROFILE is None  # second assignment also neutralised


def test_preload_cleans_up_sys_modules_on_exec_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore_upstream_config: None
) -> None:
    main = _import_main()
    spec = _fake_config_spec(
        tmp_path,
        'LOCKED_PROFILE: str | None = "x"\n'
        "raise RuntimeError('boom from upstream config init')\n",
    )
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name: spec if name == "reachy_mini_conversation_app.config" else None
    )

    with pytest.raises(RuntimeError, match="boom from upstream config init"):
        main._preload_unlocked_upstream_config()

    assert "reachy_mini_conversation_app.config" not in sys.modules


# ============================================================================
# _patch_inline_memory_into_prompt
# ============================================================================


@pytest.fixture
def fake_prompts_module():
    """Install a fake reachy_mini_conversation_app.prompts module."""
    name = "reachy_mini_conversation_app.prompts"
    saved = sys.modules.get(name)
    mod = types.ModuleType(name)
    mod.get_session_instructions = lambda: "BASE PROMPT\nwith multiple lines"
    sys.modules[name] = mod
    yield mod
    sys.modules.pop(name, None)
    if saved is not None:
        sys.modules[name] = saved


@pytest.fixture
def temp_inline_memory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "inline-memory.json"
    monkeypatch.setenv("REACHY_MINI_INLINE_MEMORY_FILE", str(path))
    return path


def test_inline_memory_patch_appends_block_when_no_placeholder(
    fake_prompts_module: types.ModuleType, temp_inline_memory: Path
) -> None:
    from reachy_mini_supermemory_app._inline_memory import add_entry

    add_entry("User's name is Andre.")
    main = _import_main()
    main._patch_inline_memory_into_prompt()

    rendered = fake_prompts_module.get_session_instructions()
    assert rendered.startswith("BASE PROMPT")
    assert "User's name is Andre." in rendered


def test_inline_memory_patch_substitutes_first_placeholder_only(
    fake_prompts_module: types.ModuleType, temp_inline_memory: Path
) -> None:
    from reachy_mini_supermemory_app._inline_memory import add_entry

    add_entry("Andre likes coffee.")
    fake_prompts_module.get_session_instructions = lambda: (
        "intro\n<<INLINE_MEMORY>>\nmid\n<<INLINE_MEMORY>>\nend"
    )
    main = _import_main()
    main._patch_inline_memory_into_prompt()
    out = fake_prompts_module.get_session_instructions()

    assert out.count("Andre likes coffee.") == 1
    # The second sentinel must remain literal so accidental duplicates are visible.
    assert "<<INLINE_MEMORY>>" in out


def test_inline_memory_patch_removes_placeholder_when_empty(
    fake_prompts_module: types.ModuleType, temp_inline_memory: Path
) -> None:
    fake_prompts_module.get_session_instructions = lambda: "before\n<<INLINE_MEMORY>>\nafter"
    main = _import_main()
    main._patch_inline_memory_into_prompt()
    out = fake_prompts_module.get_session_instructions()

    assert "<<INLINE_MEMORY>>" not in out
    assert "before" in out and "after" in out


def test_inline_memory_patch_is_idempotent(
    fake_prompts_module: types.ModuleType, temp_inline_memory: Path
) -> None:
    main = _import_main()
    main._patch_inline_memory_into_prompt()
    first = fake_prompts_module.get_session_instructions
    main._patch_inline_memory_into_prompt()
    assert fake_prompts_module.get_session_instructions is first


# ============================================================================
# _patch_realtime_vad_defaults
# ============================================================================


@pytest.fixture
def fake_realtime_modules():
    """Install fake huggingface_realtime + openai_realtime modules with a callable ServerVad."""
    names = (
        "reachy_mini_conversation_app.huggingface_realtime",
        "reachy_mini_conversation_app.openai_realtime",
    )
    saved = {n: sys.modules.get(n) for n in names}
    modules = {}
    for n in names:
        mod = types.ModuleType(n)
        mod.ServerVad = lambda **kw: dict(kw)  # callable that just echoes kwargs
        sys.modules[n] = mod
        modules[n] = mod
    yield modules
    for n in names:
        sys.modules.pop(n, None)
        if saved[n] is not None:
            sys.modules[n] = saved[n]


def test_vad_patch_injects_default_thresholds(
    fake_realtime_modules: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SUPERMEMORY_VAD_THRESHOLD", raising=False)
    monkeypatch.delenv("SUPERMEMORY_VAD_SILENCE_MS", raising=False)
    monkeypatch.delenv("SUPERMEMORY_VAD_PREFIX_PADDING_MS", raising=False)

    main = _import_main()
    main._patch_realtime_vad_defaults()

    hf = fake_realtime_modules["reachy_mini_conversation_app.huggingface_realtime"]
    result = hf.ServerVad(type="server_vad", interrupt_response=True)
    assert result["threshold"] == 0.7
    assert result["silence_duration_ms"] == 700
    assert result["prefix_padding_ms"] == 400


def test_vad_patch_respects_env_overrides(
    fake_realtime_modules: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SUPERMEMORY_VAD_THRESHOLD", "0.85")
    monkeypatch.setenv("SUPERMEMORY_VAD_SILENCE_MS", "1200")
    monkeypatch.setenv("SUPERMEMORY_VAD_PREFIX_PADDING_MS", "250")

    main = _import_main()
    main._patch_realtime_vad_defaults()

    hf = fake_realtime_modules["reachy_mini_conversation_app.huggingface_realtime"]
    result = hf.ServerVad(type="server_vad")
    assert result["threshold"] == 0.85
    assert result["silence_duration_ms"] == 1200
    assert result["prefix_padding_ms"] == 250


def test_vad_patch_preserves_caller_supplied_values(fake_realtime_modules: dict) -> None:
    main = _import_main()
    main._patch_realtime_vad_defaults()

    hf = fake_realtime_modules["reachy_mini_conversation_app.huggingface_realtime"]
    # Caller passes threshold explicitly — must not be overridden.
    result = hf.ServerVad(type="server_vad", threshold=0.95)
    assert result["threshold"] == 0.95


def test_vad_patch_is_idempotent(fake_realtime_modules: dict) -> None:
    main = _import_main()
    main._patch_realtime_vad_defaults()
    hf = fake_realtime_modules["reachy_mini_conversation_app.huggingface_realtime"]
    wrapped_once = hf.ServerVad

    main._patch_realtime_vad_defaults()
    assert hf.ServerVad is wrapped_once  # not re-wrapped


def test_apply_all_patches_emits_rollup(
    fake_realtime_modules: dict, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The roll-up line must report how many patches landed."""
    # Stub the upstream surfaces the four post-preload patches reach into.
    prompts = types.ModuleType("reachy_mini_conversation_app.prompts")
    prompts.get_session_instructions = lambda: "base"  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.prompts"] = prompts

    gradio_personality = types.ModuleType("reachy_mini_conversation_app.gradio_personality")

    class PersonalityUI:
        def _list_personalities(self):  # type: ignore[no-untyped-def]
            return []

    gradio_personality.PersonalityUI = PersonalityUI  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.gradio_personality"] = gradio_personality

    headless = types.ModuleType("reachy_mini_conversation_app.headless_personality")
    headless.available_tools_for = lambda _selected: ["dance", "background_tool_manager"]  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.headless_personality"] = headless

    # Stub the realtime module too so the reminders-emit patch lands in the rollup.
    base_realtime = types.ModuleType("reachy_mini_conversation_app.base_realtime")

    class _FakeRealtime:
        async def emit(self, *a, **kw):  # type: ignore[no-untyped-def]
            return None

    base_realtime.BaseRealtimeHandler = _FakeRealtime  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.base_realtime"] = base_realtime

    main = _import_main()
    try:
        main.apply_all_patches()
        err = capsys.readouterr().err
        assert "Upstream patches:" in err
        # 6 of the 7 patches always land in this fixture (locked-profile is
        # content-conditional and won't fire without a real config module).
        assert "/7 applied" in err
    finally:
        sys.modules.pop("reachy_mini_conversation_app.prompts", None)
        sys.modules.pop("reachy_mini_conversation_app.gradio_personality", None)
        sys.modules.pop("reachy_mini_conversation_app.headless_personality", None)
        sys.modules.pop("reachy_mini_conversation_app.base_realtime", None)


def test_available_tools_patch_filters_non_tool_modules() -> None:
    """available_tools_for must no longer return background_tool_manager / tool_constants."""
    headless = types.ModuleType("reachy_mini_conversation_app.headless_personality")
    # Pretend upstream's pre-patch behavior — leaks both non-tool modules.
    headless.available_tools_for = lambda _selected: [  # type: ignore[attr-defined]
        "dance",
        "background_tool_manager",
        "play_emotion",
        "tool_constants",
        "task_status",
    ]
    sys.modules["reachy_mini_conversation_app.headless_personality"] = headless

    main = _import_main()
    try:
        main._patch_filter_available_tools()
        # Pull the now-patched function and call it.
        filtered = sys.modules[
            "reachy_mini_conversation_app.headless_personality"
        ].available_tools_for("supermemory")
        assert "background_tool_manager" not in filtered
        assert "tool_constants" not in filtered
        # Real tools survive.
        assert "dance" in filtered
        assert "play_emotion" in filtered
        assert "task_status" in filtered
    finally:
        sys.modules.pop("reachy_mini_conversation_app.headless_personality", None)


def test_available_tools_patch_is_idempotent() -> None:
    """Re-running the patch must not double-wrap the function."""
    headless = types.ModuleType("reachy_mini_conversation_app.headless_personality")
    headless.available_tools_for = lambda _selected: ["dance", "background_tool_manager"]  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.headless_personality"] = headless

    main = _import_main()
    try:
        main._patch_filter_available_tools()
        first_wrapped = sys.modules[
            "reachy_mini_conversation_app.headless_personality"
        ].available_tools_for
        main._patch_filter_available_tools()
        assert (
            sys.modules["reachy_mini_conversation_app.headless_personality"].available_tools_for
            is first_wrapped
        )
    finally:
        sys.modules.pop("reachy_mini_conversation_app.headless_personality", None)


def test_available_tools_patch_warns_when_module_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If headless_personality is absent, we must surface a startup warning."""
    sys.modules.pop("reachy_mini_conversation_app.headless_personality", None)
    main = _import_main()
    main._patch_filter_available_tools()
    err = capsys.readouterr().err
    # Either an import error or a missing-symbol warning — both acceptable.
    assert "Available-tools patch WARNING" in err


# ============================================================================
# _patch_datetime_into_prompt
# ============================================================================


@pytest.fixture
def fake_prompts_for_datetime():
    """Stub the upstream prompts module so the datetime patch has something to wrap."""
    saved = sys.modules.get("reachy_mini_conversation_app.prompts")
    mod = types.ModuleType("reachy_mini_conversation_app.prompts")
    mod.get_session_instructions = lambda: "head\n<<CURRENT_DATETIME>>\ntail"  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.prompts"] = mod
    yield mod
    sys.modules.pop("reachy_mini_conversation_app.prompts", None)
    if saved is not None:
        sys.modules["reachy_mini_conversation_app.prompts"] = saved


def test_datetime_patch_substitutes_placeholder(fake_prompts_for_datetime: Any) -> None:
    main = _import_main()
    main._patch_datetime_into_prompt()
    rendered = fake_prompts_for_datetime.get_session_instructions()
    assert "<<CURRENT_DATETIME>>" not in rendered
    assert "Current date and time:" in rendered
    assert "get_current_time" in rendered  # the in-prompt hint


def test_datetime_patch_passes_through_when_placeholder_absent(monkeypatch: pytest.MonkeyPatch) -> None:
    saved = sys.modules.get("reachy_mini_conversation_app.prompts")
    mod = types.ModuleType("reachy_mini_conversation_app.prompts")
    mod.get_session_instructions = lambda: "no placeholder here"  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.prompts"] = mod
    try:
        main = _import_main()
        main._patch_datetime_into_prompt()
        assert mod.get_session_instructions() == "no placeholder here"
    finally:
        sys.modules.pop("reachy_mini_conversation_app.prompts", None)
        if saved is not None:
            sys.modules["reachy_mini_conversation_app.prompts"] = saved


def test_datetime_patch_is_idempotent(fake_prompts_for_datetime: Any) -> None:
    main = _import_main()
    main._patch_datetime_into_prompt()
    first = fake_prompts_for_datetime.get_session_instructions
    main._patch_datetime_into_prompt()
    assert fake_prompts_for_datetime.get_session_instructions is first


def test_datetime_patch_warns_when_prompts_missing(capsys: pytest.CaptureFixture[str]) -> None:
    sys.modules.pop("reachy_mini_conversation_app.prompts", None)
    main = _import_main()
    main._patch_datetime_into_prompt()
    err = capsys.readouterr().err
    assert "Datetime patch WARNING" in err


def test_render_current_datetime_uses_env_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    from reachy_mini_supermemory_app import _patches

    monkeypatch.setenv("SUPERMEMORY_USER_TIMEZONE", "Asia/Tokyo")
    rendered = _patches._render_current_datetime()
    assert "Asia/Tokyo" in rendered


def test_render_current_datetime_falls_back_on_invalid_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bogus IANA name must not crash; fall through to local time."""
    from reachy_mini_supermemory_app import _patches

    monkeypatch.setenv("SUPERMEMORY_USER_TIMEZONE", "Not/A/Real/Zone")
    rendered = _patches._render_current_datetime()
    # Should still produce a date+time line — just not in the bogus zone.
    assert "Current date and time:" in rendered
    assert "Not/A/Real/Zone" not in rendered


def test_datetime_and_inline_memory_compose_in_either_order(
    fake_prompts_for_datetime: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both patches must coexist regardless of which wrapped first."""
    fake_prompts_for_datetime.get_session_instructions = lambda: (
        "head\n<<CURRENT_DATETIME>>\n<<INLINE_MEMORY>>\ntail"
    )
    main = _import_main()
    main._patch_inline_memory_into_prompt()
    main._patch_datetime_into_prompt()
    rendered = fake_prompts_for_datetime.get_session_instructions()
    assert "<<CURRENT_DATETIME>>" not in rendered
    assert "<<INLINE_MEMORY>>" not in rendered
    assert "Current date and time:" in rendered


# ============================================================================
# _patch_realtime_emit_with_reminders
# ============================================================================


@pytest.fixture
def fake_base_realtime():
    """Stub upstream base_realtime so the emit patch has something to wrap."""
    saved = sys.modules.get("reachy_mini_conversation_app.base_realtime")
    mod = types.ModuleType("reachy_mini_conversation_app.base_realtime")

    class _Fake:
        async def emit(self, *a, **kw):  # type: ignore[no-untyped-def]
            return "original-emit-result"

    mod.BaseRealtimeHandler = _Fake  # type: ignore[attr-defined]
    sys.modules["reachy_mini_conversation_app.base_realtime"] = mod
    yield mod
    sys.modules.pop("reachy_mini_conversation_app.base_realtime", None)
    if saved is not None:
        sys.modules["reachy_mini_conversation_app.base_realtime"] = saved


def test_reminders_patch_wraps_emit(fake_base_realtime: Any) -> None:
    """After patching, emit gains the marker attribute."""
    main = _import_main()
    main._patch_realtime_emit_with_reminders()
    assert getattr(fake_base_realtime.BaseRealtimeHandler.emit, "_supermemory_reminders_patched", False)


def test_reminders_patch_is_idempotent(fake_base_realtime: Any) -> None:
    main = _import_main()
    main._patch_realtime_emit_with_reminders()
    first = fake_base_realtime.BaseRealtimeHandler.emit
    main._patch_realtime_emit_with_reminders()
    assert fake_base_realtime.BaseRealtimeHandler.emit is first


def test_reminders_patch_warns_when_module_missing(capsys: pytest.CaptureFixture[str]) -> None:
    sys.modules.pop("reachy_mini_conversation_app.base_realtime", None)
    main = _import_main()
    main._patch_realtime_emit_with_reminders()
    err = capsys.readouterr().err
    assert "Reminders patch WARNING" in err


@pytest.mark.asyncio
async def test_patched_emit_fires_due_reminder(
    fake_base_realtime: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a reminder is due and the session is idle, the patched emit injects it."""
    import asyncio
    import datetime
    import json as _json

    from reachy_mini_supermemory_app import _reminders as r

    store_path = tmp_path / "reminders.json"
    monkeypatch.setenv(r.REMINDERS_FILE_ENV, str(store_path))
    past = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
    ).isoformat()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        _json.dumps(
            {
                "entries": [
                    {
                        "id": "p1",
                        "text": "call mom",
                        "fire_at": past,
                        "status": r.STATUS_PENDING,
                        "created_at": past,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    # Wire up a fake realtime instance with the minimal interface the patch needs.
    injected: list[dict] = []
    response_created: list[bool] = []

    class _FakeItem:
        async def create(self, *, item: dict) -> None:
            injected.append(item)

    class _FakeConversation:
        item = _FakeItem()

    class _FakeConnection:
        conversation = _FakeConversation()

    class _Instance:
        connection = _FakeConnection()
        _response_done_event = asyncio.Event()

        async def emit(self, *a, **kw):  # original; will be wrapped
            return "ok"

        async def _safe_response_create(self, *a, **kw):  # type: ignore[no-untyped-def]
            response_created.append(True)

    _Instance._response_done_event.set()  # idle
    fake_base_realtime.BaseRealtimeHandler = _Instance  # type: ignore[attr-defined]

    main = _import_main()
    main._patch_realtime_emit_with_reminders()

    instance = _Instance()
    result = await instance.emit()
    assert result == "ok"  # original return surfaced
    assert len(injected) == 1
    assert "call mom" in injected[0]["content"][0]["text"]
    assert response_created == [True]
    # And the reminder is now marked fired in the store.
    on_disk = _json.loads(store_path.read_text())
    assert on_disk["entries"][0]["status"] == r.STATUS_FIRED


@pytest.mark.asyncio
async def test_patched_emit_skips_when_response_in_flight(
    fake_base_realtime: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Don't interrupt the model mid-response — reminder waits for the next idle tick."""
    import asyncio
    import datetime
    import json as _json

    from reachy_mini_supermemory_app import _reminders as r

    store_path = tmp_path / "reminders.json"
    monkeypatch.setenv(r.REMINDERS_FILE_ENV, str(store_path))
    past = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
    ).isoformat()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        _json.dumps(
            {
                "entries": [
                    {"id": "p1", "text": "x", "fire_at": past, "status": r.STATUS_PENDING, "created_at": past}
                ]
            }
        ),
        encoding="utf-8",
    )

    injected: list[dict] = []

    class _FakeItem:
        async def create(self, *, item: dict) -> None:
            injected.append(item)

    class _FakeConnection:
        conversation = type("c", (), {"item": _FakeItem()})()

    class _Instance:
        connection = _FakeConnection()
        _response_done_event = asyncio.Event()
        # done_event NOT set → response is in flight

        async def emit(self, *a, **kw):
            return "ok"

        async def _safe_response_create(self):
            pass

    fake_base_realtime.BaseRealtimeHandler = _Instance  # type: ignore[attr-defined]

    main = _import_main()
    main._patch_realtime_emit_with_reminders()
    await _Instance().emit()
    assert injected == []
    # Reminder is STILL pending — we didn't claim it.
    on_disk = _json.loads(store_path.read_text())
    assert on_disk["entries"][0]["status"] == r.STATUS_PENDING


@pytest.mark.asyncio
async def test_patched_emit_skips_when_no_connection(
    fake_base_realtime: Any, tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Don't try to inject when the session isn't connected."""
    import asyncio
    import datetime
    import json as _json

    from reachy_mini_supermemory_app import _reminders as r

    store_path = tmp_path / "reminders.json"
    monkeypatch.setenv(r.REMINDERS_FILE_ENV, str(store_path))
    past = (
        datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=10)
    ).isoformat()
    store_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.write_text(
        _json.dumps(
            {
                "entries": [
                    {"id": "p1", "text": "x", "fire_at": past, "status": r.STATUS_PENDING, "created_at": past}
                ]
            }
        ),
        encoding="utf-8",
    )

    class _Instance:
        connection = None  # not connected
        _response_done_event = asyncio.Event()

        async def emit(self, *a, **kw):
            return "ok"

        async def _safe_response_create(self):
            pass

    _Instance._response_done_event.set()
    fake_base_realtime.BaseRealtimeHandler = _Instance  # type: ignore[attr-defined]

    main = _import_main()
    main._patch_realtime_emit_with_reminders()
    await _Instance().emit()
    # Reminder NOT claimed — still pending.
    on_disk = _json.loads(store_path.read_text())
    assert on_disk["entries"][0]["status"] == r.STATUS_PENDING


def test_reminders_patch_in_rollup(monkeypatch: pytest.MonkeyPatch) -> None:
    """apply_all_patches now reports a 7th patch."""
    # Lazy import to be sure we get the freshly-patched _patches module.
    from reachy_mini_supermemory_app import _patches

    assert hasattr(_patches, "_patch_realtime_emit_with_reminders")
    # The rollup line in apply_all_patches references "reminders-emit-hook".
    import inspect

    src = inspect.getsource(_patches.apply_all_patches)
    assert "reminders-emit-hook" in src
    assert "/7 applied" not in src  # the literal text isn't in source; check structure
    # The number of "checks = (...) items" should be 7.
    assert src.count("(\"") == 7  # noqa: B015 — sanity heuristic on the checks tuple


def test_vad_patch_warns_when_servervad_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """If neither realtime module exposes ``ServerVad``, the operator must learn about it.

    Previously this silently no-op'd — users heard constant mid-sentence
    interruptions and had no diagnostic to chase.
    """
    names = (
        "reachy_mini_conversation_app.huggingface_realtime",
        "reachy_mini_conversation_app.openai_realtime",
    )
    # Install modules WITHOUT a ServerVad attribute.
    saved = {n: sys.modules.get(n) for n in names}
    for n in names:
        sys.modules[n] = types.ModuleType(n)  # no ServerVad
    try:
        main = _import_main()
        main._patch_realtime_vad_defaults()
        err = capsys.readouterr().err
        assert "VAD patch WARNING" in err
        assert "SUPERMEMORY_VAD_" in err
    finally:
        for n in names:
            sys.modules.pop(n, None)
            if saved[n] is not None:
                sys.modules[n] = saved[n]


# ============================================================================
# _wake_up_robot_async
# ============================================================================


def _wait_for(predicate, timeout_s: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


def test_wake_up_posts_to_daemon(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)  # skip the 3s settle
    main = _import_main()

    calls: list[str] = []

    class _Resp:
        status_code = 200

    def fake_post(url: str, timeout: float = 5.0) -> _Resp:
        calls.append(url)
        return _Resp()

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.post = fake_post
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    main._wake_up_robot_async()
    assert _wait_for(lambda: calls), "wake_up POST was never issued"
    assert calls[0].endswith("/api/move/play/wake_up")


def test_wake_up_retries_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    main = _import_main()

    attempts: list[int] = []

    class _OkResp:
        status_code = 200

    def flaky_post(url: str, timeout: float = 5.0) -> Any:
        attempts.append(1)
        if len(attempts) < 3:
            raise RuntimeError("connection refused")
        return _OkResp()

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.post = flaky_post
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    main._wake_up_robot_async()
    assert _wait_for(lambda: len(attempts) >= 3, timeout_s=3.0)


def test_wake_up_honors_daemon_base_url_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    monkeypatch.setenv("REACHY_MINI_DAEMON_API_BASE", "http://10.0.0.5:9000/")
    main = _import_main()

    calls: list[str] = []

    class _Resp:
        status_code = 200

    fake_httpx = types.ModuleType("httpx")
    fake_httpx.post = lambda url, timeout=5.0: (calls.append(url) or _Resp())
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    main._wake_up_robot_async()
    assert _wait_for(lambda: calls)
    assert calls[0] == "http://10.0.0.5:9000/api/move/play/wake_up"


# ============================================================================
# _persistent_instance_dir mkdir-failure visibility
# ============================================================================


def test_persistent_instance_dir_warns_once_on_mkdir_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Read-only / unwritable config base must surface a single stderr warning.

    Pre-fix the failure was swallowed by a bare except, so users saw silent
    settings-save no-ops with no diagnostic.
    """
    from reachy_mini_supermemory_app import _env_paths

    _env_paths._reset_persistent_dir_warning_for_tests()
    main = _import_main()
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    def _boom(self: Path, *_a: Any, **_kw: Any) -> None:
        raise PermissionError("read-only fs")

    monkeypatch.setattr(Path, "mkdir", _boom)
    main._persistent_instance_dir()
    main._persistent_instance_dir()  # second call must NOT re-warn

    err = capsys.readouterr().err
    # exactly one occurrence of the warning text
    assert err.count("cannot create config dir") == 1
    assert "read-only fs" in err
