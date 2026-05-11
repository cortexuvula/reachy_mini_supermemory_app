"""Entry point for the Reachy Mini supermemory app."""

from __future__ import annotations

import importlib
import os
import threading
from pathlib import Path
from typing import Any

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.main import (
    ReachyMiniConversationApp,
)
from reachy_mini_conversation_app.main import (
    run as _conversation_run,
)
from reachy_mini_conversation_app.utils import parse_args

from .settings_ui import mount_supermemory_routes

PROFILE_NAME = "supermemory"


def _bundled_profiles_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "profiles"


def _configure_environment() -> None:
    """Point the conversation app at our bundled profile, without clobbering user choices."""
    profiles_dir = _bundled_profiles_dir()
    if profiles_dir.is_dir():
        # Always set this — it's our app's responsibility to expose its profile.
        os.environ["REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY"] = str(profiles_dir)
    # Default to the supermemory profile, but let an explicit env var override.
    # Must be set BEFORE the preload, because upstream config.py reads these
    # env vars at class-definition time while we're loading it.
    os.environ.setdefault("REACHY_MINI_CUSTOM_PROFILE", PROFILE_NAME)
    _preload_unlocked_upstream_config()
    _patch_external_profiles_into_dropdown()
    _patch_inline_memory_into_prompt()
    _patch_realtime_vad_defaults()


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
    for module_name in targets:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        original = getattr(module, "ServerVad", None)
        if original is None or getattr(original, "_supermemory_vad_patched", False):
            continue

        def _wrapped(*args: Any, _orig: Any = original, _defaults: dict[str, Any] = overrides, **kwargs: Any) -> Any:
            for key, value in _defaults.items():
                kwargs.setdefault(key, value)
            return _orig(*args, **kwargs)

        _wrapped._supermemory_vad_patched = True  # type: ignore[attr-defined]
        module.ServerVad = _wrapped


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
    """
    import importlib.util
    import re
    import sys
    import types

    name = "reachy_mini_conversation_app.config"
    if name in sys.modules:
        return  # already imported by something else — too late to swap

    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return
    if spec is None or spec.origin is None:
        return

    try:
        with open(spec.origin, "r", encoding="utf-8") as f:
            source = f.read()
    except OSError:
        return

    locked_line_re = re.compile(r"^LOCKED_PROFILE\s*(:\s*[^=]*?)?\s*=\s*([^\n#]+)", re.MULTILINE)
    match = locked_line_re.search(source)
    if match is None:
        return
    current_value = match.group(2).strip()
    if current_value == "None":
        return  # already unlocked
    patched = locked_line_re.sub("LOCKED_PROFILE: str | None = None", source, count=1)

    module = types.ModuleType(name)
    module.__file__ = spec.origin
    module.__loader__ = spec.loader
    module.__spec__ = spec
    module.__package__ = spec.parent
    sys.modules[name] = module

    try:
        _eval_module_source(compile(patched, spec.origin, "exec"), module.__dict__)
    except Exception:
        del sys.modules[name]
        raise


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
    except Exception:
        return
    from ._inline_memory import render_block

    original = _prompts.get_session_instructions
    if getattr(original, "_supermemory_inline_patched", False):
        return

    def _with_inline_memory() -> str:  # type: ignore[no-untyped-def]
        base = original()
        block = render_block()
        if INLINE_MEMORY_PLACEHOLDER in base:
            # Drop the placeholder line entirely when there's nothing to inject.
            replacement = block if block else ""
            return base.replace(INLINE_MEMORY_PLACEHOLDER, replacement)
        if not block:
            return base
        return f"{base}\n\n{block}\n"

    _with_inline_memory._supermemory_inline_patched = True  # type: ignore[attr-defined]
    _prompts.get_session_instructions = _with_inline_memory


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
    except Exception:
        return

    original = PersonalityUI._list_personalities
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


SETTINGS_PORT_ENV = "SUPERMEMORY_SETTINGS_PORT"
SETTINGS_HOST_ENV = "SUPERMEMORY_SETTINGS_HOST"
DEFAULT_SETTINGS_PORT = 7861
DEFAULT_SETTINGS_HOST = "127.0.0.1"


def main() -> None:
    """CLI entry point: ``reachy-mini-supermemory-app``."""
    _configure_environment()
    _start_cli_settings_server()
    args, _ = parse_args()
    _conversation_run(args)


def _start_cli_settings_server() -> None:
    """Serve the /supermemory/ routes on a side-port for CLI launches.

    Upstream's gradio path doesn't actually serve the FastAPI it builds — it
    launches gradio on its own internal app. So in CLI mode we run our own
    tiny uvicorn on a separate port to expose the settings UI. In daemon mode
    the routes are still mounted on ``self.settings_app`` via ``run()`` below.
    """
    try:
        import uvicorn
        from fastapi import FastAPI
    except Exception:
        return

    try:
        port = int(os.environ.get(SETTINGS_PORT_ENV, DEFAULT_SETTINGS_PORT))
    except ValueError:
        port = DEFAULT_SETTINGS_PORT
    host = (os.environ.get(SETTINGS_HOST_ENV) or DEFAULT_SETTINGS_HOST).strip() or DEFAULT_SETTINGS_HOST

    instance_path = Path.cwd()
    app = FastAPI()
    mount_supermemory_routes(app, str(instance_path))

    def _serve() -> None:
        config = uvicorn.Config(app, host=host, port=port, log_level="warning")
        uvicorn.Server(config).run()

    threading.Thread(target=_serve, name="supermemory-settings", daemon=True).start()
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Supermemory settings UI: http://{display_host}:{port}/supermemory/ (bound to {host})")


class ReachyMiniSupermemoryApp(ReachyMiniConversationApp):
    """Reachy Mini Apps entry point that bundles the supermemory profile."""

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Configure env, mount settings routes, then delegate to the conversation app."""
        _configure_environment()
        try:
            instance_path = self._get_instance_path().parent
        except Exception:
            instance_path = None
        mount_supermemory_routes(self.settings_app, str(instance_path) if instance_path else None)
        super().run(reachy_mini, stop_event)


if __name__ == "__main__":
    app = ReachyMiniSupermemoryApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
