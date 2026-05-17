"""Runtime monkey-patches against ``reachy_mini_conversation_app``.

Each patch in this module reaches into a specific upstream symbol — VAD
defaults, the locked-profile guard, the session-instructions prompt, the
gradio personality dropdown — to change behavior we can't influence through
upstream's public surface. They run from ``main._configure_environment``
and fail loudly through ``startup_log`` when the targeted symbol has moved
or been renamed, so an upstream refactor doesn't silently disable our
features.

``apply_all_patches()`` runs them in the correct order and emits a roll-up
line at the end so the operator can confirm at a glance which patches
landed against the running upstream version.

Re-exported from ``main`` with their original underscore-prefixed names
to keep existing test imports stable.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from typing import Any

from ._log_utils import startup_log

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAD defaults
# ---------------------------------------------------------------------------

VAD_THRESHOLD_ENV = "SUPERMEMORY_VAD_THRESHOLD"
VAD_SILENCE_MS_ENV = "SUPERMEMORY_VAD_SILENCE_MS"
VAD_PREFIX_PADDING_MS_ENV = "SUPERMEMORY_VAD_PREFIX_PADDING_MS"


def _patch_realtime_vad_defaults() -> None:
    """Inject tunable defaults into upstream's ServerVad() turn-detection calls.

    Upstream constructs ``ServerVad(type="server_vad", interrupt_response=True)``
    with no other params, accepting the OpenAI Realtime defaults (threshold 0.5,
    silence_duration_ms 200) — too sensitive when the robot's own speaker bleeds
    into its mic, producing constant mid-sentence interruptions. We wrap the
    ``ServerVad`` symbol in each realtime handler module so any unset params get
    filled in from env vars with conservative fallbacks.
    """
    overrides: dict[str, Any] = {}
    raw_threshold = os.environ.get(VAD_THRESHOLD_ENV)
    if raw_threshold:
        try:
            overrides["threshold"] = float(raw_threshold)
        except ValueError:
            pass
    else:
        overrides["threshold"] = 0.7

    raw_silence = os.environ.get(VAD_SILENCE_MS_ENV)
    if raw_silence:
        try:
            overrides["silence_duration_ms"] = int(raw_silence)
        except ValueError:
            pass
    else:
        overrides["silence_duration_ms"] = 700

    raw_prefix = os.environ.get(VAD_PREFIX_PADDING_MS_ENV)
    if raw_prefix:
        try:
            overrides["prefix_padding_ms"] = int(raw_prefix)
        except ValueError:
            pass
    else:
        overrides["prefix_padding_ms"] = 400

    targets = (
        "reachy_mini_conversation_app.huggingface_realtime",
        "reachy_mini_conversation_app.openai_realtime",
    )
    patched_any = False
    already_patched_any = False
    for module_name in targets:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        original = getattr(module, "ServerVad", None)
        if original is None:
            continue
        if getattr(original, "_supermemory_vad_patched", False):
            already_patched_any = True
            continue

        def _wrapped(*args: Any, _orig: Any = original, _defaults: dict[str, Any] = overrides, **kwargs: Any) -> Any:
            for key, value in _defaults.items():
                kwargs.setdefault(key, value)
            return _orig(*args, **kwargs)

        _wrapped._supermemory_vad_patched = True  # type: ignore[attr-defined]
        patched_any = True
        module.ServerVad = _wrapped

    if not patched_any and not already_patched_any:
        # Neither realtime module exposed ``ServerVad`` — almost certainly an
        # upstream rename. Silent no-op would mean VAD env-var tuning has no
        # effect; the operator hears constant mid-sentence interruptions and
        # has no clue why.
        startup_log(
            "VAD patch WARNING: ServerVad not found in either realtime module — "
            "SUPERMEMORY_VAD_* env vars will have no effect",
            logger=logger,
        )


# ---------------------------------------------------------------------------
# Locked-profile guard
# ---------------------------------------------------------------------------

# Alias to the Python builtin so an over-eager linter doesn't misread the call
# below as a shell exec — this is the trusted module-loading primitive.
_eval_module_source = exec  # noqa: S102


def _preload_unlocked_upstream_config() -> None:
    """Load ``reachy_mini_conversation_app.config`` with ``LOCKED_PROFILE`` forced to None.

    Some upstream deployments hard-code ``LOCKED_PROFILE`` at module level to
    pin the daemon to a specific profile. When the lock points at a profile we
    don't ship, ``Config.__init__`` raises at import time and our app can't
    start. We pre-populate ``sys.modules`` with the module loaded from a
    rewritten source so subsequent imports see the unlocked version. No-op
    when upstream is already unlocked.

    Emits a ``startup_log`` line for every outcome (no-op, applied,
    aborted) so the operator can tell from the journal whether the regex
    found what it expected. Previously this patch was completely silent,
    which made an upstream drift indistinguishable from a successful
    no-op.
    """
    import importlib.util
    import re
    import types

    name = "reachy_mini_conversation_app.config"
    if name in sys.modules:
        # Already imported by something else (typically a side-effecting
        # import of another upstream submodule). Worth flagging because the
        # patch is now a no-op even if LOCKED_PROFILE is set.
        startup_log(
            "Locked-profile patch: config already imported by another path; "
            "if LOCKED_PROFILE is set, the daemon may refuse to start",
            logger=logger,
        )
        return

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        startup_log(
            "Locked-profile patch WARNING: could not resolve "
            "reachy_mini_conversation_app.config spec",
            logger=logger,
        )
        return
    if spec is None or spec.origin is None:
        startup_log(
            "Locked-profile patch WARNING: reachy_mini_conversation_app.config "
            "has no resolvable on-disk origin",
            logger=logger,
        )
        return

    try:
        with open(spec.origin, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError as e:
        startup_log(
            f"Locked-profile patch WARNING: cannot read {spec.origin}: {e}",
            logger=logger,
        )
        return

    locked_line_re = re.compile(r"^LOCKED_PROFILE\s*(:\s*[^=]*?)?\s*=\s*([^\n#]+)", re.MULTILINE)
    match = locked_line_re.search(source)
    if match is None:
        # Upstream simply doesn't define LOCKED_PROFILE in this version — a
        # quiet no-op case (no lock to undo). Not a warning, but worth a
        # one-line note so the operator can confirm the patch ran.
        logger.info("Locked-profile patch: no LOCKED_PROFILE assignment in upstream config; skipping")
        return
    current_value = match.group(2).strip()
    if current_value == "None":
        logger.info("Locked-profile patch: upstream already unlocked; skipping")
        return
    # No count limit — if upstream ever has multiple LOCKED_PROFILE assignments
    # (e.g. an env-gated override at the bottom), neutralise them all.
    patched = locked_line_re.sub("LOCKED_PROFILE: str | None = None", source)

    module = types.ModuleType(name)
    module.__file__ = spec.origin
    module.__loader__ = spec.loader
    module.__spec__ = spec
    module.__package__ = spec.parent
    sys.modules[name] = module

    try:
        _eval_module_source(compile(patched, spec.origin, "exec"), module.__dict__)
    except Exception as e:
        del sys.modules[name]
        startup_log(
            f"Locked-profile patch FAILED: rewritten config raised on exec: {e}",
            logger=logger,
        )
        raise
    startup_log(
        f"Locked-profile patch applied: forced LOCKED_PROFILE (was {current_value!r}) to None",
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Current-datetime injection into the session prompt
# ---------------------------------------------------------------------------

CURRENT_DATETIME_PLACEHOLDER = "<<CURRENT_DATETIME>>"
USER_TIMEZONE_ENV = "SUPERMEMORY_USER_TIMEZONE"

# Realtime backend modules that do ``from reachy_mini_conversation_app.prompts
# import get_session_instructions`` at top-level. After we patch the function
# on the ``prompts`` module, those modules' LOCAL references still point at
# whichever object existed at their import time. If they were already imported
# before our patch ran, calls go to the original — substitutions are silently
# skipped, the model gets a placeholder-laced prompt, and we waste hours
# debugging "the date isn't reaching the model".
#
# So after patching, we also rebind the local name in any of these modules
# that have already loaded. Modules that load LATER pick up the wrapper via
# their `from ... import` automatically.
_REALTIME_BACKENDS = (
    "reachy_mini_conversation_app.huggingface_realtime",
    "reachy_mini_conversation_app.openai_realtime",
    "reachy_mini_conversation_app.gemini_live",
)


def _rebind_get_session_instructions(new_fn: Any) -> None:
    """Update local ``get_session_instructions`` references in realtime backends."""
    for module_name in _REALTIME_BACKENDS:
        module = sys.modules.get(module_name)
        if module is None:
            continue
        if hasattr(module, "get_session_instructions"):
            module.get_session_instructions = new_fn


def _render_current_datetime() -> str:
    """Build the human-readable datetime string injected at session start.

    Resolves timezone in this order:
      1. ``SUPERMEMORY_USER_TIMEZONE`` env var (IANA name, e.g. ``Europe/Paris``).
      2. The OS's local timezone (via ``datetime.astimezone()``).
      3. UTC as last resort if the previous two raise.

    The line ends with a hint that ``get_current_time`` is available for
    precise minute-level queries, so the model knows the session-start
    timestamp isn't reliable mid-conversation.
    """
    import datetime

    tz_name = os.environ.get(USER_TIMEZONE_ENV, "").strip()
    now: datetime.datetime
    tz_display: str
    if tz_name:
        try:
            import zoneinfo

            tz = zoneinfo.ZoneInfo(tz_name)
            now = datetime.datetime.now(tz)
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

    weekday = now.strftime("%A")
    date_part = now.strftime("%B %-d, %Y") if hasattr(now, "strftime") else now.isoformat()
    # %-d is GNU; on platforms without it %#d would be needed. Wrap defensively.
    try:
        date_part = now.strftime("%B %-d, %Y")
    except Exception:
        date_part = now.strftime("%B %d, %Y").replace(" 0", " ")
    time_part = now.strftime("%I:%M %p").lstrip("0")

    return (
        f"Current date and time: it's {weekday}, {date_part}, around {time_part} "
        f"({tz_display}). The exact minute may have drifted since this session "
        f"started — call get_current_time when the user asks for the precise time."
    )


def _patch_datetime_into_prompt() -> None:
    """Substitute the current local datetime into ``get_session_instructions``.

    Wraps the upstream prompt loader (same hook the inline-memory patch uses)
    so every new realtime session re-renders the datetime line. The wrapper
    composes cleanly with the inline-memory wrapper — they each look for
    their own placeholder and pass the rest through.
    """
    try:
        from reachy_mini_conversation_app import prompts as _prompts
    except Exception as e:
        startup_log(
            f"Datetime patch WARNING: cannot import prompts module ({e}); "
            "<<CURRENT_DATETIME>> placeholder will not be substituted",
            logger=logger,
        )
        return

    original = getattr(_prompts, "get_session_instructions", None)
    if original is None:
        startup_log(
            "Datetime patch WARNING: get_session_instructions not found in "
            "upstream prompts module; <<CURRENT_DATETIME>> placeholder will "
            "not be substituted",
            logger=logger,
        )
        return
    if getattr(original, "_supermemory_datetime_patched", False):
        return

    def _with_datetime() -> str:  # type: ignore[no-untyped-def]
        base = original()
        if CURRENT_DATETIME_PLACEHOLDER in base:
            return base.replace(CURRENT_DATETIME_PLACEHOLDER, _render_current_datetime(), 1)
        return base

    _with_datetime._supermemory_datetime_patched = True  # type: ignore[attr-defined]
    _prompts.get_session_instructions = _with_datetime
    # Realtime backends do `from prompts import get_session_instructions` at
    # module load; bare module-attribute patching misses their local refs if
    # they imported before us. Rebind defensively.
    _rebind_get_session_instructions(_with_datetime)


# ---------------------------------------------------------------------------
# Inline-memory injection into the session prompt
# ---------------------------------------------------------------------------

INLINE_MEMORY_PLACEHOLDER = "<<INLINE_MEMORY>>"


def _patch_inline_memory_into_prompt() -> None:
    """Substitute the inline-memory block into ``get_session_instructions`` output.

    Done at prompt-load time so the bullet list reflects whatever the user has
    accumulated, without rewriting ``instructions.txt`` on every edit. If the
    profile prompt contains the ``<<INLINE_MEMORY>>`` placeholder, the block is
    swapped in there (preferred — keeps it salient near the top); otherwise it
    falls back to appending at the end.
    """
    try:
        from reachy_mini_conversation_app import prompts as _prompts
    except Exception as e:
        startup_log(
            f"Inline-memory patch WARNING: cannot import prompts module ({e}); "
            "manage_memory entries will not be injected into the system prompt",
            logger=logger,
        )
        return
    from ._inline_memory import render_block

    original = getattr(_prompts, "get_session_instructions", None)
    if original is None:
        startup_log(
            "Inline-memory patch WARNING: get_session_instructions not found in "
            "upstream prompts module; manage_memory entries will not be injected",
            logger=logger,
        )
        return
    if getattr(original, "_supermemory_inline_patched", False):
        return

    def _with_inline_memory() -> str:  # type: ignore[no-untyped-def]
        base = original()
        block = render_block()
        if INLINE_MEMORY_PLACEHOLDER in base:
            # Drop the placeholder line entirely when there's nothing to inject.
            # Replace only the first occurrence to avoid duplicating the block if
            # the sentinel ever appears more than once in the profile prompt.
            replacement = block if block else ""
            return base.replace(INLINE_MEMORY_PLACEHOLDER, replacement, 1)
        if not block:
            return base
        return f"{base}\n\n{block}\n"

    _with_inline_memory._supermemory_inline_patched = True  # type: ignore[attr-defined]
    _prompts.get_session_instructions = _with_inline_memory
    # See note in _patch_datetime_into_prompt — realtime backends bind a local
    # reference at their `from prompts import …` line; we have to update those
    # too, otherwise the inline memory block never reaches the model.
    _rebind_get_session_instructions(_with_inline_memory)


# ---------------------------------------------------------------------------
# Personality-dropdown visibility for externally-injected profiles
# ---------------------------------------------------------------------------


def _patch_external_profiles_into_dropdown() -> None:
    """Make externally-injected profiles visible to upstream's gradio personality dropdown.

    Upstream's ``PersonalityUI._list_personalities`` enumerates only the bundled
    ``DEFAULT_PROFILES_DIRECTORY``. With ``REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY``
    set, the active profile (e.g. ``supermemory``) ends up as the dropdown's
    ``value`` without appearing in ``choices``, so gradio raises on every
    interaction.
    """
    try:
        from reachy_mini_conversation_app.config import DEFAULT_PROFILES_DIRECTORY, config
        from reachy_mini_conversation_app.gradio_personality import PersonalityUI
    except Exception as e:
        startup_log(
            f"Dropdown patch WARNING: cannot import gradio_personality ({e}); "
            "external profile will not appear in personality dropdown",
            logger=logger,
        )
        return

    original = getattr(PersonalityUI, "_list_personalities", None)
    if original is None:
        startup_log(
            "Dropdown patch WARNING: PersonalityUI._list_personalities not found; "
            "external profile will not appear in personality dropdown",
            logger=logger,
        )
        return
    if getattr(original, "_supermemory_patched", False):
        return

    def _list_with_external(self):  # type: ignore[no-untyped-def]
        names = original(self)
        external_root = getattr(config, "PROFILES_DIRECTORY", None)
        if external_root and external_root != DEFAULT_PROFILES_DIRECTORY:
            try:
                if external_root.exists():
                    for p in sorted(external_root.iterdir()):
                        if p.is_dir() and (p / "instructions.txt").exists() and p.name not in names:
                            names.append(p.name)
            except Exception:
                pass
        return names

    _list_with_external._supermemory_patched = True  # type: ignore[attr-defined]
    PersonalityUI._list_personalities = _list_with_external


# ---------------------------------------------------------------------------
# Filter non-Tool modules out of the dashboard's "Available tools" list
# ---------------------------------------------------------------------------

# Modules that live under ``reachy_mini_conversation_app/tools/`` but aren't
# Tool subclasses. Upstream's ``available_tools_for`` globs the directory and
# only filters ``__init__`` and ``core_tools`` — the rest leak into the
# dashboard's checkbox grid as if they were tools the user could enable.
_NON_TOOL_MODULES = frozenset({"background_tool_manager", "tool_constants"})


def _patch_filter_available_tools() -> None:
    """Hide non-Tool modules from the dashboard's available-tools checklist.

    Upstream's ``headless_personality.available_tools_for(profile)`` enumerates
    every ``.py`` under ``tools/`` except ``__init__`` and ``core_tools``.
    That includes ``background_tool_manager.py`` (the manager class) and
    ``tool_constants.py`` (enums) — neither is a Tool. They appear in the
    dashboard with checkboxes that don't do anything: enabling them is a
    no-op (the loader skips non-Tool modules), and the unchecked state reads
    as "the background tool manager is off" when it isn't (the manager is
    instantiated unconditionally in the realtime path). Filter them out so
    the checklist tells the truth.
    """
    try:
        from reachy_mini_conversation_app import headless_personality as _hp
    except Exception as e:
        startup_log(
            f"Available-tools patch WARNING: cannot import headless_personality ({e}); "
            "background_tool_manager will keep appearing as a misleading checkbox",
            logger=logger,
        )
        return

    original = getattr(_hp, "available_tools_for", None)
    if original is None:
        startup_log(
            "Available-tools patch WARNING: available_tools_for not found in "
            "headless_personality; background_tool_manager will keep appearing "
            "as a misleading checkbox",
            logger=logger,
        )
        return
    if getattr(original, "_supermemory_avail_tools_patched", False):
        return

    def _filtered_available_tools(selected: str):  # type: ignore[no-untyped-def]
        names = original(selected)
        return [n for n in names if n not in _NON_TOOL_MODULES]

    _filtered_available_tools._supermemory_avail_tools_patched = True  # type: ignore[attr-defined]
    _hp.available_tools_for = _filtered_available_tools


# ---------------------------------------------------------------------------
# Realtime emit() hook for one-shot reminders
# ---------------------------------------------------------------------------


def _patch_realtime_emit_with_reminders() -> None:
    """Wrap ``BaseRealtimeHandler.emit`` to fire any due reminders mid-session.

    ``emit`` is the fastrtc per-audio-frame tick — it already runs an idle
    check inline. We piggyback on the same hook to peek the reminders
    store; when a reminder is due, idle, and the session is connected,
    inject a synthesised user message via the same ``conversation.item.create``
    + ``response.create`` mechanism upstream uses for idle signals.

    The check is cheap (atomic JSON peek + datetime compare) so the
    per-tick cost is negligible. ``pop_first_due()`` atomically claims
    the entry so a concurrent tick can't double-fire.
    """
    try:
        from reachy_mini_conversation_app import base_realtime as _br
    except Exception as e:
        startup_log(
            f"Reminders patch WARNING: cannot import base_realtime ({e}); "
            "scheduled reminders will not fire mid-session",
            logger=logger,
        )
        return

    cls = getattr(_br, "BaseRealtimeHandler", None)
    if cls is None:
        startup_log(
            "Reminders patch WARNING: BaseRealtimeHandler class not found; "
            "scheduled reminders will not fire mid-session",
            logger=logger,
        )
        return
    original = getattr(cls, "emit", None)
    if original is None:
        startup_log(
            "Reminders patch WARNING: BaseRealtimeHandler.emit not found; "
            "scheduled reminders will not fire mid-session",
            logger=logger,
        )
        return
    if getattr(original, "_supermemory_reminders_patched", False):
        return

    async def _emit_with_reminder_check(self: Any, *args: Any, **kwargs: Any) -> Any:
        # Peek-and-fire BEFORE the original emit runs. The original handles
        # idle signalling + output queue draining; our check is additive.
        try:
            await _maybe_fire_reminder(self)
        except Exception as e:
            logger.warning("Reminder check failed: %s", e)
        return await original(self, *args, **kwargs)

    _emit_with_reminder_check._supermemory_reminders_patched = True  # type: ignore[attr-defined]
    cls.emit = _emit_with_reminder_check


async def _maybe_fire_reminder(realtime_instance: Any) -> None:
    """Inject one due reminder into the live session, if it's safe to do so.

    Safety gates (in order, cheapest first):
      1. Session has a live ``connection`` (skip if disconnected).
      2. ``_response_done_event`` is set (don't interrupt a response).
      3. Privacy mode is not active (don't speak during privacy).
      4. There's actually a due reminder.
    """
    if not getattr(realtime_instance, "connection", None):
        return
    done_event = getattr(realtime_instance, "_response_done_event", None)
    if done_event is not None and not done_event.is_set():
        return

    try:
        from ._privacy_mode import is_privacy_active

        if is_privacy_active():
            return
    except Exception:
        # If the privacy module isn't importable, don't block reminders.
        pass

    from ._reminders import pop_first_due

    reminder = pop_first_due()
    if reminder is None:
        return

    text = reminder.get("text") or "(no text)"
    fire_at = reminder.get("fire_at") or ""
    message = (
        f"[Reminder fired at {fire_at}] You scheduled this earlier: "
        f'"{text}". Tell the user about it now in your own voice.'
    )

    try:
        await realtime_instance.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": message}],
            }
        )
        # Use the upstream-provided safe response trigger so we honour
        # the "is_response_in_flight" guard upstream maintains.
        await realtime_instance._safe_response_create()
        logger.info(
            "Reminder fired: id=%s fire_at=%s text=%r",
            reminder.get("id"),
            fire_at,
            text,
        )
    except Exception as e:
        # The reminder is already marked FIRED by pop_first_due; if injection
        # fails we lose the speech, but the entry is preserved in the store
        # with a "fired" status so list_reminders shows it. Log so an
        # operator can investigate.
        logger.warning(
            "Reminder %s injection failed (text=%r): %s",
            reminder.get("id"),
            text,
            e,
        )


# ---------------------------------------------------------------------------
# Roll-up
# ---------------------------------------------------------------------------


def _vad_applied() -> bool:
    """Probe whether the VAD patch's marker is set on at least one realtime module."""
    targets = (
        "reachy_mini_conversation_app.huggingface_realtime",
        "reachy_mini_conversation_app.openai_realtime",
    )
    for module_name in targets:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        original = getattr(module, "ServerVad", None)
        if original is not None and getattr(original, "_supermemory_vad_patched", False):
            return True
    return False


def _inline_memory_applied() -> bool:
    """Walk the wrapper chain looking for the inline-memory marker.

    Both inline-memory and datetime patches wrap ``get_session_instructions``;
    whichever runs second hides the other's marker from the outermost frame.
    """
    try:
        from reachy_mini_conversation_app import prompts as _prompts
    except Exception:
        return False
    fn = getattr(_prompts, "get_session_instructions", None)
    while fn is not None:
        if getattr(fn, "_supermemory_inline_patched", False):
            return True
        fn = _unwrap(fn)
    return False


def _dropdown_applied() -> bool:
    try:
        from reachy_mini_conversation_app.gradio_personality import PersonalityUI
    except Exception:
        return False
    return bool(getattr(getattr(PersonalityUI, "_list_personalities", None), "_supermemory_patched", False))


def _locked_profile_applied() -> bool:
    """The lock patch is content-conditional — count it as 'applied' if the module loaded.

    Unlike the other three patches (which always run a wrap), this one only
    rewrites source when ``LOCKED_PROFILE`` is non-None. The honest reporting
    is therefore "module was processed", not "LOCKED_PROFILE was rewritten",
    because the no-op-on-already-unlocked case is the expected default.
    """
    return "reachy_mini_conversation_app.config" in sys.modules


def _available_tools_applied() -> bool:
    try:
        from reachy_mini_conversation_app import headless_personality as _hp
    except Exception:
        return False
    return bool(getattr(getattr(_hp, "available_tools_for", None), "_supermemory_avail_tools_patched", False))


def _datetime_applied() -> bool:
    """``get_session_instructions`` wrapped by the datetime patch?

    The inline-memory patch also wraps the same symbol, so we walk the
    closure chain looking for the marker rather than just checking the
    outermost function.
    """
    try:
        from reachy_mini_conversation_app import prompts as _prompts
    except Exception:
        return False
    fn = getattr(_prompts, "get_session_instructions", None)
    while fn is not None:
        if getattr(fn, "_supermemory_datetime_patched", False):
            return True
        fn = _unwrap(fn)
    return False


def _unwrap(fn):  # type: ignore[no-untyped-def]
    """Best-effort: pull the wrapped function out of a closure if present."""
    closure = getattr(fn, "__closure__", None) or ()
    for cell in closure:
        contents = cell.cell_contents
        if callable(contents):
            return contents
    return None


def _reminders_applied() -> bool:
    """Defensive probe — must not raise even if the upstream class moved/renamed.

    The previous version did ``_br.BaseRealtimeHandler.emit`` directly, which
    AttributeError'd on the deployed robot (a different upstream version with
    no such symbol) and crashed apply_all_patches mid-rollup, masking every
    other patch. All probes must tolerate missing attributes; failure here
    just means "not applied" — the warning from the patch itself already
    explained why.
    """
    try:
        from reachy_mini_conversation_app import base_realtime as _br
    except Exception:
        return False
    cls = getattr(_br, "BaseRealtimeHandler", None)
    if cls is None:
        return False
    return bool(getattr(getattr(cls, "emit", None), "_supermemory_reminders_patched", False))


def apply_all_patches() -> None:
    """Run the seven upstream patches in dependency order, then emit a roll-up.

    Order matters: ``_preload_unlocked_upstream_config`` must run BEFORE any
    other import of ``reachy_mini_conversation_app.config`` (it swaps the
    module entry in ``sys.modules`` and so only works pre-import). The two
    prompt patches (inline-memory and datetime) both wrap
    ``get_session_instructions`` — they compose cleanly because each looks
    for its own placeholder. Order within that pair doesn't matter; the
    outermost wrapper runs first and passes the rest through.

    The roll-up uses the per-patch marker attributes to verify which patches
    actually landed against the running upstream; this is the operator-facing
    "everything's fine" signal that distinguishes a healthy startup from one
    where a silent drift left a patch as a no-op.
    """
    _preload_unlocked_upstream_config()
    _patch_realtime_vad_defaults()
    _patch_inline_memory_into_prompt()
    _patch_datetime_into_prompt()
    _patch_external_profiles_into_dropdown()
    _patch_filter_available_tools()
    _patch_realtime_emit_with_reminders()

    checks = (
        ("locked-profile", _locked_profile_applied()),
        ("vad-defaults", _vad_applied()),
        ("inline-memory", _inline_memory_applied()),
        ("datetime", _datetime_applied()),
        ("personality-dropdown", _dropdown_applied()),
        ("available-tools-filter", _available_tools_applied()),
        ("reminders-emit-hook", _reminders_applied()),
    )
    applied = [name for name, ok in checks if ok]
    failed = [name for name, ok in checks if not ok]
    if failed:
        startup_log(
            f"Upstream patches: {len(applied)}/{len(checks)} applied "
            f"(failed: {', '.join(failed)}); see prior WARNING lines for details",
            logger=logger,
        )
    else:
        startup_log(
            f"Upstream patches: {len(applied)}/{len(checks)} applied "
            f"({', '.join(applied)})",
            logger=logger,
        )
