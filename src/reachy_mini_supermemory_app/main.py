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


def main() -> None:
    """CLI entry point: ``reachy-mini-supermemory-app``."""
    _configure_environment()
    args, _ = parse_args()
    _conversation_run(args)


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
