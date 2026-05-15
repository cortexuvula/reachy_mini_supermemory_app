"""``.env`` file lifecycle for the supermemory app.

Two physical files cooperate:

- **persistent** (``~/.config/reachy_mini_supermemory_app/.env``) is the
  source of truth. The settings UI writes here, the persistent location
  survives ``pip uninstall && pip install`` cycles the Reachy daemon performs
  during updates.

- **legacy** (``<install_dir>/.env``) is a mirror. It exists ONLY because the
  upstream ``reachy_mini_conversation_app`` calls
  ``load_dotenv(_get_instance_path().parent / .env, override=True)`` deep in
  its own ``run()`` — that override=True would clobber whatever we loaded into
  ``os.environ`` unless the file on disk holds the same values.

The functions here keep the two files in sync without losing data written via
either path:

1. ``migrate_legacy_to_persistent_if_needed`` — first-run only: pre-existing
   users had their config in the install dir. Move it to the persistent
   location so the next reinstall doesn't wipe it.
2. ``capture_legacy_extras_into_persistent`` — bidirectional bridge for keys
   set by upstream's own console UI (which writes to legacy). Any key in
   legacy that's not in persistent is added to persistent BEFORE we sync the
   other direction — without this step we'd silently lose upstream-side edits
   every time the daemon restarted.
3. ``load_persistent_into_env`` — populate ``os.environ`` from persistent.
4. ``sync_persistent_to_legacy`` — overwrite legacy with persistent so
   upstream's load_dotenv(override=True) is a no-op against our values.

``load_package_dotenv`` is the public orchestrator that runs them in order.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_persistent_dir_warned: bool = False


def package_env_path() -> Path:
    """The legacy install-dir .env that upstream's load_dotenv reads."""
    return Path(__file__).resolve().parent / ".env"


def persistent_instance_dir() -> Path:
    """Return the user-config-dir that survives app updates / reinstalls.

    Honours ``XDG_CONFIG_HOME``; falls back to ``~/.config``. The site-packages
    install directory is unsafe for user settings because the daemon's
    update flow runs ``pip uninstall`` and reinstalls the package — anything
    we wrote there can be wiped.

    If mkdir fails (read-only FS, permission denied), the returned path is
    unusable for writes; downstream callers will fail individually. We surface
    the failure ONCE to stderr so the user knows persistence is broken,
    rather than silently watching every settings save no-op.
    """
    # Imported here to avoid a top-level cycle (_log_utils doesn't import this
    # module but the loader at startup wants minimal import-time work).
    from ._log_utils import startup_log

    global _persistent_dir_warned
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    p = Path(base) / "reachy_mini_supermemory_app"
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        if not _persistent_dir_warned:
            _persistent_dir_warned = True
            startup_log(
                f"Warning: cannot create config dir {p}: {e} — settings will not persist across restarts",
                logger=logger,
            )
    return p


def persistent_env_path() -> Path:
    """The persistent .env the settings UI writes to."""
    return persistent_instance_dir() / ".env"


# ---------------------------------------------------------------------------
# Lifecycle steps (each one is independently testable)
# ---------------------------------------------------------------------------


def migrate_legacy_to_persistent_if_needed(persistent: Path, legacy: Path) -> None:
    """First-run-only: copy legacy → persistent if only legacy exists.

    Users who installed the app before the persistent directory existed had
    their .env in the package install dir. Move it to the durable location so
    the next ``pip uninstall`` (which the daemon's update flow triggers) doesn't
    wipe their API key.
    """
    if persistent.exists() or not legacy.exists():
        return
    try:
        persistent.parent.mkdir(parents=True, exist_ok=True)
        persistent.write_bytes(legacy.read_bytes())
        try:
            persistent.chmod(0o600)
        except Exception:
            pass
    except Exception:
        pass


def capture_legacy_extras_into_persistent(persistent: Path, legacy: Path) -> None:
    """Merge legacy keys missing from persistent INTO persistent.

    Upstream's own console UI writes to legacy directly. Without this step,
    those writes would be wiped on the next ``sync_persistent_to_legacy``
    call. Persistent stays the source of truth; we just "rescue" any keys
    set on the other side before overwriting.
    """
    if not legacy.exists():
        return
    legacy_values = _read_env_file(legacy)
    if not legacy_values:
        return
    persistent_values = _read_env_file(persistent) if persistent.exists() else {}
    extras = {k: v for k, v in legacy_values.items() if k not in persistent_values}
    if not extras:
        return
    # Append the missing keys at the end of persistent. Use the same plain
    # KEY=value writer we use for our own settings UI — no fancy escaping
    # required (matching ``_persist_to_env_file`` semantics).
    try:
        persistent.parent.mkdir(parents=True, exist_ok=True)
        existing = persistent.read_text(encoding="utf-8") if persistent.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        appended = existing + "\n".join(f"{k}={v}" for k, v in extras.items()) + "\n"
        persistent.write_text(appended, encoding="utf-8")
        try:
            persistent.chmod(0o600)
        except Exception:
            pass
        logger.info(
            "Captured %d key(s) from legacy .env into persistent: %s",
            len(extras),
            sorted(extras),
        )
    except Exception as e:
        logger.warning("Failed to capture legacy extras into persistent: %s", e)


def load_persistent_into_env(persistent: Path) -> None:
    """Populate ``os.environ`` from the persistent .env (override=False).

    No-op if the file doesn't exist (fresh install) or python-dotenv isn't
    importable. ``override=False`` so explicit env vars already set in the
    process (e.g. systemd Environment= directives) win.
    """
    if not persistent.exists():
        return
    try:
        from dotenv import load_dotenv
    except Exception:
        return
    try:
        load_dotenv(dotenv_path=str(persistent), override=False)
    except Exception:
        pass


def sync_persistent_to_legacy(persistent: Path, legacy: Path) -> None:
    """Overwrite legacy with persistent so upstream's reload is a no-op.

    Upstream calls ``load_dotenv(legacy_path, override=True)`` later in its
    own bootstrap. If legacy has stale values, that call would clobber what
    ``load_persistent_into_env`` just wrote. Mirroring guarantees idempotence.
    """
    if not persistent.exists():
        return
    try:
        legacy.parent.mkdir(parents=True, exist_ok=True)
        legacy.write_bytes(persistent.read_bytes())
        try:
            legacy.chmod(0o600)
        except Exception:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def load_package_dotenv() -> None:
    """Run the full .env lifecycle in the right order.

    Called at every entry point (CLI ``main()``, daemon ``__init__`` /
    ``run``) so the env state is consistent regardless of how the app boots.
    """
    persistent = persistent_env_path()
    legacy = package_env_path()
    migrate_legacy_to_persistent_if_needed(persistent, legacy)
    capture_legacy_extras_into_persistent(persistent, legacy)
    load_persistent_into_env(persistent)
    sync_persistent_to_legacy(persistent, legacy)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _read_env_file(path: Path) -> Dict[str, str]:
    """Parse a .env file into a dict, tolerating comments and blank lines.

    Intentionally simple — does NOT handle quoted values or variable
    interpolation. The settings UI writes plain ``KEY=value`` lines and
    upstream's console UI uses python-dotenv's writer (which preserves
    plain assignments for the keys we care about). If anyone ever drops a
    fancy multi-line value into a .env file, the rescue path will skip it
    and python-dotenv's load_dotenv will still pick it up — worst case is
    we lose the cross-side mirroring for that specific key.
    """
    out: Dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return out
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        out[key] = value
    return out


def _reset_persistent_dir_warning_for_tests() -> None:
    """Drop the once-warned guard so tests can re-exercise the warning path."""
    global _persistent_dir_warned
    _persistent_dir_warned = False
