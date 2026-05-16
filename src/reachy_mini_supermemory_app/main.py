"""Entry point for the Reachy Mini supermemory app."""

from __future__ import annotations

import logging
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

from ._log_utils import startup_log
from .settings_ui import mount_supermemory_routes

logger = logging.getLogger(__name__)

PROFILE_NAME = "supermemory"


def _bundled_profiles_dir() -> Path:
    # Profiles ship inside the package so they're present after `pip install`.
    return Path(__file__).resolve().parent / "profiles"


def _configure_environment() -> None:
    """Point the conversation app at our bundled profile, without clobbering user choices."""
    # Load the package-local .env BEFORE anything reads env vars. Upstream
    # only loads it inside its own run() (much later), so without this our
    # feature gates (PRIVACY_TOGGLE, AUTO_DIGEST, …) see an empty environment
    # at install time and silently no-op.
    _load_package_dotenv()
    profiles_dir = _bundled_profiles_dir()
    if profiles_dir.is_dir():
        # Always set this — it's our app's responsibility to expose its profile.
        os.environ["REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY"] = str(profiles_dir)
    # Default to the supermemory profile, but let an explicit env var override.
    # Must be set BEFORE the preload, because upstream config.py reads these
    # env vars at class-definition time while we're loading it.
    os.environ.setdefault("REACHY_MINI_CUSTOM_PROFILE", PROFILE_NAME)
    # apply_all_patches runs the four upstream monkey-patches in the right
    # order (locked-profile preload must come first because it swaps the
    # config module BEFORE anything else imports it) and emits a one-line
    # "patches applied: N/M" roll-up at the end. That's the diagnostic the
    # operator should look for in the journal to confirm compat with the
    # running upstream version.
    apply_all_patches()
    # Auto-digest install lives here (not in main()) so it runs for BOTH the
    # CLI launch and the dashboard-managed launch (which calls wrapped_run
    # directly and never enters main()).
    _install_auto_digest()


# .env lifecycle lives in ``_env_paths`` so this entry-point file stays
# focused on bootstrapping. Re-exported with their pre-refactor names to keep
# the test suite's import surface stable.
from ._env_paths import (
    load_package_dotenv as _load_package_dotenv,
    package_env_path as _package_env_path,
    persistent_env_path,
    persistent_instance_dir as _persistent_instance_dir,
)


# Patches against the upstream conversation_app live in ``_patches`` so this
# entry-point file stays focused on bootstrapping. Re-exported with their
# original names because test_main_patches.py reaches into them via ``main._``.
from ._patches import (
    INLINE_MEMORY_PLACEHOLDER,
    VAD_PREFIX_PADDING_MS_ENV,
    VAD_SILENCE_MS_ENV,
    VAD_THRESHOLD_ENV,
    apply_all_patches,
    _patch_external_profiles_into_dropdown,
    _patch_filter_available_tools,
    _patch_inline_memory_into_prompt,
    _patch_realtime_vad_defaults,
    _preload_unlocked_upstream_config,
)


SETTINGS_PORT_ENV = "SUPERMEMORY_SETTINGS_PORT"
SETTINGS_HOST_ENV = "SUPERMEMORY_SETTINGS_HOST"
DEFAULT_SETTINGS_PORT = 7861
DEFAULT_SETTINGS_HOST = "127.0.0.1"


def main() -> None:
    """CLI entry point: ``reachy-mini-supermemory-app``."""
    _configure_environment()
    _start_cli_settings_server()
    _wake_up_robot_async()
    args, _ = parse_args()
    _conversation_run(args)


def _install_auto_digest() -> None:
    """Wire up the optional auto-digest pipeline (no-op when env-gate is off)."""
    try:
        from ._auto_digest import install
    except Exception:
        return
    install()


DAEMON_API_BASE_ENV = "REACHY_MINI_DAEMON_API_BASE"
DEFAULT_DAEMON_API_BASE = "http://127.0.0.1:8000"


def _wake_up_robot_async() -> None:
    """Fire-and-forget wake-up via the daemon's REST API.

    The Reachy Mini daemon starts the robot in sleep pose
    (``--no-wake-up-on-start``); the conversation app doesn't wake it.
    Without this, the bot will talk but the head stays on the base and
    every motor tool (dance, move_head, play_emotion) is silently ignored.
    Runs in a daemon thread with a small startup delay so it doesn't race
    the daemon coming up.
    """
    import time

    base = (os.environ.get(DAEMON_API_BASE_ENV) or DEFAULT_DAEMON_API_BASE).rstrip("/")

    def _wake() -> None:
        try:
            import httpx
        except Exception:
            return
        # Daemon usually settles within a couple of seconds; give it a head start.
        time.sleep(3)
        for _ in range(5):
            try:
                resp = httpx.post(f"{base}/api/move/play/wake_up", timeout=5.0)
                if resp.status_code < 400:
                    print(f"Reachy Mini woken up via {base}/api/move/play/wake_up")
                    return
            except Exception:
                pass
            time.sleep(2)
        print("Warning: could not wake Reachy Mini — robot may stay in sleep pose")

    threading.Thread(target=_wake, name="supermemory-wake", daemon=True).start()


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

    app = FastAPI()
    mount_supermemory_routes(app, str(_persistent_instance_dir()))

    def _serve() -> None:
        try:
            config = uvicorn.Config(app, host=host, port=port, log_level="warning")
            uvicorn.Server(config).run()
        except Exception as e:
            # Without this catch the thread dies silently — the user only
            # discovers the failure when the URL doesn't load. Common case:
            # port already in use (another instance, or 7861 is otherwise
            # bound).
            startup_log(
                f"Supermemory settings UI failed on {host}:{port}: {e}",
                logger=logger,
            )

    threading.Thread(target=_serve, name="supermemory-settings", daemon=True).start()
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    print(f"Supermemory settings UI: http://{display_host}:{port}/supermemory/ (bound to {host})")


# class ReachyMiniSupermemoryApp(ReachyMiniApp) — the comment above is the
# canonical form `reachy-mini-app-assistant check` greps for. The real class
# below inherits ReachyMiniApp transitively via ReachyMiniConversationApp.
class ReachyMiniSupermemoryApp(ReachyMiniConversationApp):
    """Reachy Mini Apps entry point that bundles the supermemory profile."""

    # Tell the dashboard which URL to iframe for the in-app settings panel.
    # ReachyMiniApp.__init__ spins up uvicorn on this host:port serving
    # self.settings_app, and auto-mounts static/index.html at "/". We then
    # attach our /supermemory/* JSON routes to the same FastAPI in run().
    # Port must NOT collide with the daemon (8000) or its WebRTC (8443).
    custom_app_url = "http://0.0.0.0:8042/"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Upstream's ReachyMiniApp.__init__ creates self.settings_app and
        # registers "/" → static/index.html. We must replace that "/" route
        # BEFORE wrapped_run() starts the uvicorn thread, otherwise the
        # dashboard's first request sees the standalone supermemory page and
        # only later requests see the composed conversation_app + supermemory
        # view. .env loading is needed before mount because the routes look
        # at SUPERMEMORY_* env vars at registration time.
        _load_package_dotenv()
        super().__init__(*args, **kwargs)
        if self.settings_app is not None:
            mount_supermemory_routes(self.settings_app, str(_persistent_instance_dir()))

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        """Configure env, install patches, then delegate to the conversation app."""
        _configure_environment()
        self._install_privacy_mode(reachy_mini)
        super().run(reachy_mini, stop_event)

    @staticmethod
    def _install_privacy_mode(reachy_mini: ReachyMini) -> None:
        """Wire up antenna-press privacy toggle (no-op when env-gate is off)."""
        try:
            from ._privacy_mode import install as _privacy_install
        except Exception:
            return
        _privacy_install(reachy_mini)


if __name__ == "__main__":
    app = ReachyMiniSupermemoryApp()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
