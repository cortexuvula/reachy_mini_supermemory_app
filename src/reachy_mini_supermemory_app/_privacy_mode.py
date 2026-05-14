"""Privacy mode: push an antenna to mute Reachy mid-conversation.

A background thread polls the antenna positions and toggles a global
"privacy active" flag whenever a sudden deflection larger than
``SUPERMEMORY_PRIVACY_DEVIATION_DEG`` is observed (debounced to avoid
multi-fire on a single press). When privacy turns on, audio capture is
stopped at the gstreamer level and the speaker queue is flushed —
nothing reaches the realtime backend, the bot goes silent mid-sentence.
The antennas fold down as the visible status indicator. Pressing again
restores the previous antenna pose and resumes capture.

Opt-in via ``SUPERMEMORY_PRIVACY_TOGGLE=true``. Auto-digest reads
``is_privacy_active()`` and drops transcript lines while it's on.
"""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLED_ENV = "SUPERMEMORY_PRIVACY_TOGGLE"
DEVIATION_DEG_ENV = "SUPERMEMORY_PRIVACY_DEVIATION_DEG"
DEBOUNCE_MS_ENV = "SUPERMEMORY_PRIVACY_DEBOUNCE_MS"

DEFAULT_DEVIATION_DEG = 25.0
DEFAULT_DEBOUNCE_MS = 1500  # generous: covers motor-settling motion after a toggle
POLL_INTERVAL_S = 0.05  # 50 ms
# EMA smoothing for the "what counts as resting position" baseline. The slow
# alpha lets the baseline drift to track motor poses and emote sweeps over a
# couple of seconds without absorbing a brief press. The fast alpha kicks in
# during debounce so the baseline catches up to whatever pose the user has
# pushed the antenna into — that way their release is absorbed into the new
# baseline instead of being detected as a second press.
BASELINE_ALPHA_SLOW = 0.1
BASELINE_ALPHA_FAST = 0.5

# Antenna pose used to signal "privacy is on". Symmetric, well outside the
# resting range (~±0.2 rad) so it's visually unambiguous.
PRIVACY_POSE: Tuple[float, float] = (1.5, -1.5)

# Module-level flag so _auto_digest can ask `is_privacy_active()` without
# circular imports.
_active: bool = False
_state_lock = threading.Lock()


def is_enabled() -> bool:
    """Return True when the env var opts the feature in."""
    return (os.environ.get(ENABLED_ENV) or "").strip().lower() in ("1", "true", "yes", "on")


def is_privacy_active() -> bool:
    """Read the global privacy state (used by auto-digest and tests)."""
    with _state_lock:
        return _active


def _set_privacy_active(value: bool) -> None:
    global _active
    with _state_lock:
        _active = value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(0.0, float(raw))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        return default


class PrivacyController:
    """Polls antenna positions, detects presses, calls on_activate / on_deactivate."""

    def __init__(
        self,
        get_present_positions: Callable[[], Optional[Tuple[float, float]]],
        on_activate: Callable[[], None],
        on_deactivate: Callable[[], None],
        deviation_deg: Optional[float] = None,
        debounce_ms: Optional[int] = None,
        on_tick: Optional[Callable[[], None]] = None,
    ) -> None:
        self._get_positions = get_present_positions
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        self._on_tick = on_tick
        deg = deviation_deg if deviation_deg is not None else _env_float(DEVIATION_DEG_ENV, DEFAULT_DEVIATION_DEG)
        self._deviation_rad = math.radians(deg)
        debounce = debounce_ms if debounce_ms is not None else _env_int(DEBOUNCE_MS_ENV, DEFAULT_DEBOUNCE_MS)
        self._debounce_s = debounce / 1000.0
        self._baseline: Optional[Tuple[float, float]] = None
        self._last_toggle_at: float = 0.0
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True, name="privacy-mode")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            if self._stop_event.wait(POLL_INTERVAL_S):
                return
            try:
                self.tick()
            except Exception as e:
                logger.warning("privacy-mode tick failed: %s", e)
            if self._on_tick is not None:
                try:
                    self._on_tick()
                except Exception as e:
                    logger.warning("privacy-mode on_tick failed: %s", e)

    def tick(self) -> None:
        """One poll: track a slow-moving baseline, fire on any sharp deviation.

        A press shows up as a brief excursion AWAY from wherever the antenna
        had been resting — could be 0° when privacy is off, ~86° when it's
        on, or anywhere mid-emote. Anything slow enough to be motor drive
        or an idle/emote sweep gets absorbed into the EMA baseline before
        it can fire. During debounce we crank up the baseline alpha so the
        user's release (return to rest) is absorbed into the new baseline
        rather than firing a second toggle as soon as they let go.

        Earlier detectors used a sliding-window comparison of oldest vs
        newest, then a reversal check, then "trajectory must return near
        start." All of those required the press to complete within a fixed
        time window — they couldn't catch a normal-length tap+hold+release.
        Baseline tracking has no window: any tap duration works.
        """
        positions = self._get_positions()
        if positions is None or len(positions) < 2:
            return
        r, l = float(positions[0]), float(positions[1])

        if self._baseline is None:
            self._baseline = (r, l)
            return

        in_debounce = (time.monotonic() - self._last_toggle_at) < self._debounce_s
        alpha = BASELINE_ALPHA_FAST if in_debounce else BASELINE_ALPHA_SLOW
        br, bl = self._baseline
        self._baseline = (
            (1.0 - alpha) * br + alpha * r,
            (1.0 - alpha) * bl + alpha * l,
        )

        if in_debounce:
            return

        if abs(r - br) > self._deviation_rad or abs(l - bl) > self._deviation_rad:
            self._maybe_toggle()

    def _maybe_toggle(self) -> None:
        now = time.monotonic()
        if now - self._last_toggle_at < self._debounce_s:
            return
        self._last_toggle_at = now
        new_state = not is_privacy_active()
        _set_privacy_active(new_state)
        try:
            if new_state:
                self._on_activate()
            else:
                self._on_deactivate()
        except Exception as e:
            logger.warning("privacy-mode handler raised: %s", e)


class PrivacyMode:
    """Glue: wires a PrivacyController to a real ReachyMini instance."""

    def __init__(self, mini: Any) -> None:
        self.mini = mini
        self._saved_antennas: Optional[Tuple[float, float]] = None
        self.controller = PrivacyController(
            get_present_positions=self._get_positions,
            on_activate=self._on_activate,
            on_deactivate=self._on_deactivate,
            on_tick=self._on_tick,
        )

    def _on_tick(self) -> None:
        """Re-assert PRIVACY_POSE every poll while active, so the antennas stay
        folded as the visual indicator. Safe with the baseline-tracking detector:
        steady motor force at PRIVACY_POSE settles into the baseline, only a
        sharp deflection against motor torque (= a real press) exceeds threshold.
        """
        if not is_privacy_active():
            return
        try:
            self.mini.set_target_antenna_joint_positions(PRIVACY_POSE)
        except Exception:
            pass

    def start(self) -> None:
        self.controller.start()

    def stop(self) -> None:
        self.controller.stop()

    def _get_positions(self) -> Optional[Tuple[float, float]]:
        try:
            positions = self.mini.get_present_antenna_joint_positions()
        except Exception:
            return None
        if positions is None:
            return None
        return tuple(positions)

    def _on_activate(self) -> None:
        logger.info("Privacy mode ON — muting mic + flushing speaker")
        try:
            self._saved_antennas = self._get_positions()
            self.mini.set_target_antenna_joint_positions(PRIVACY_POSE)
        except Exception as e:
            logger.warning("privacy ON antenna pose failed: %s", e)
        try:
            self.mini.media.audio.stop_recording()
        except Exception as e:
            logger.warning("privacy ON stop_recording failed: %s", e)
        try:
            self.mini.media.audio.clear_player()
        except Exception as e:
            logger.warning("privacy ON clear_player failed: %s", e)

    def _on_deactivate(self) -> None:
        logger.info("Privacy mode OFF — resuming mic")
        try:
            if self._saved_antennas is not None:
                self.mini.set_target_antenna_joint_positions(self._saved_antennas)
        except Exception as e:
            logger.warning("privacy OFF antenna restore failed: %s", e)
        try:
            self.mini.media.audio.start_recording()
        except Exception as e:
            logger.warning("privacy OFF start_recording failed: %s", e)


def _patch_moves_manager_for_privacy() -> None:
    """Force antenna writes to PRIVACY_POSE while privacy is active.

    Upstream's MovementManager._issue_control_command (the single set_target
    call point — line 851 of moves.py) writes head + antennas + body_yaw
    every tick at ~100 Hz. Our 20 Hz on_tick can't reliably win that race
    — idle breathing, emotes, and the listening-antennas blend keep
    overriding our PRIVACY_POSE. Patching the single write point so that
    privacy ALWAYS wins is deterministic and only ever costs a function
    call when privacy is off (the flag check returns immediately).
    """
    try:
        from reachy_mini_conversation_app import moves as _moves  # type: ignore[import-not-found]
    except Exception as e:
        print(f"Privacy patch: cannot import moves module: {e}", file=sys.stderr, flush=True)
        return

    # NB the class is ``MovementManager`` (the file is ``moves.py``). The
    # first attempt at this patch grepped for ``MovesManager`` and silently
    # no-op'd — fail loud now if the class disappears or gets renamed.
    cls = getattr(_moves, "MovementManager", None)
    if cls is None:
        print("Privacy patch: MovementManager class not found in moves module", file=sys.stderr, flush=True)
        return
    original = getattr(cls, "_issue_control_command", None)
    if original is None:
        print("Privacy patch: _issue_control_command not found on MovementManager", file=sys.stderr, flush=True)
        return
    if getattr(original, "_privacy_patched", False):
        return  # already patched, idempotent

    def patched(self: Any, head: Any, antennas: Any, body_yaw: Any) -> Any:
        if is_privacy_active():
            antennas = PRIVACY_POSE
        return original(self, head, antennas, body_yaw)

    patched._privacy_patched = True  # type: ignore[attr-defined]
    cls._issue_control_command = patched
    print("Privacy patch: MovementManager._issue_control_command intercepted", file=sys.stderr, flush=True)


def install(mini: Any) -> Optional[PrivacyMode]:
    """Attach a PrivacyMode if env-gate is on. Returns the instance or None."""
    if not is_enabled():
        return None
    _patch_moves_manager_for_privacy()
    pm = PrivacyMode(mini)
    pm.start()
    # stderr + flush so the line lands in the systemd journal immediately —
    # logger.info would be dropped at this point (root logger isn't yet
    # configured) and stdout is block-buffered under the daemon's pipe.
    deg = _env_float(DEVIATION_DEG_ENV, DEFAULT_DEVIATION_DEG)
    debounce = _env_int(DEBOUNCE_MS_ENV, DEFAULT_DEBOUNCE_MS)
    print(
        f"Privacy mode enabled: deviation={deg}°, debounce={debounce}ms, poll={int(POLL_INTERVAL_S*1000)}ms",
        file=sys.stderr,
        flush=True,
    )
    return pm


def _reset_for_tests() -> None:
    """Drop the module-level privacy flag so tests start clean."""
    _set_privacy_active(False)
