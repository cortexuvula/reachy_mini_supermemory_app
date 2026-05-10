"""Entry point for the Reachy Mini supermemory app."""

from __future__ import annotations

import os
import threading
from pathlib import Path

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
    os.environ.setdefault("REACHY_MINI_CUSTOM_PROFILE", PROFILE_NAME)
    _patch_external_profiles_into_dropdown()
    _patch_inline_memory_into_prompt()


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
DEFAULT_SETTINGS_PORT = 7861


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

    instance_path = Path.cwd()
    app = FastAPI()
    mount_supermemory_routes(app, str(instance_path))

    def _serve() -> None:
        config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
        uvicorn.Server(config).run()

    threading.Thread(target=_serve, name="supermemory-settings", daemon=True).start()
    print(f"Supermemory settings UI: http://127.0.0.1:{port}/supermemory/")


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
