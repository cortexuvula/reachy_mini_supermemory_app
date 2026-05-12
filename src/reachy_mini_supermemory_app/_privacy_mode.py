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
import threading
import time
from collections import deque
from typing import Any, Callable, Deque, Optional, Tuple

logger = logging.getLogger(__name__)

ENABLED_ENV = "SUPERMEMORY_PRIVACY_TOGGLE"
DEVIATION_DEG_ENV = "SUPERMEMORY_PRIVACY_DEVIATION_DEG"
DEBOUNCE_MS_ENV = "SUPERMEMORY_PRIVACY_DEBOUNCE_MS"

DEFAULT_DEVIATION_DEG = 25.0
DEFAULT_DEBOUNCE_MS = 500
POLL_INTERVAL_S = 0.05  # 50 ms
WINDOW_SIZE = 4  # ~200 ms history — a press creates a big oldest→newest delta within this

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
    ) -> None:
        self._get_positions = get_present_positions
        self._on_activate = on_activate
        self._on_deactivate = on_deactivate
        deg = deviation_deg if deviation_deg is not None else _env_float(DEVIATION_DEG_ENV, DEFAULT_DEVIATION_DEG)
        self._deviation_rad = math.radians(deg)
        debounce = debounce_ms if debounce_ms is not None else _env_int(DEBOUNCE_MS_ENV, DEFAULT_DEBOUNCE_MS)
        self._debounce_s = debounce / 1000.0
        self._window: Deque[Tuple[float, float]] = deque(maxlen=WINDOW_SIZE)
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

    def tick(self) -> None:
        """One poll: read antenna positions, decide if a press should toggle."""
        positions = self._get_positions()
        if positions is None or len(positions) < 2:
            return
        sample = (float(positions[0]), float(positions[1]))
        self._window.append(sample)
        if len(self._window) < WINDOW_SIZE:
            return
        oldest = self._window[0]
        newest = self._window[-1]
        delta_right = abs(newest[0] - oldest[0])
        delta_left = abs(newest[1] - oldest[1])
        if delta_right > self._deviation_rad or delta_left > self._deviation_rad:
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
        # Drop the window so the new pose (e.g. antennas folded down) isn't
        # itself interpreted as the next press.
        self._window.clear()


class PrivacyMode:
    """Glue: wires a PrivacyController to a real ReachyMini instance."""

    def __init__(self, mini: Any) -> None:
        self.mini = mini
        self._saved_antennas: Optional[Tuple[float, float]] = None
        self.controller = PrivacyController(
            get_present_positions=self._get_positions,
            on_activate=self._on_activate,
            on_deactivate=self._on_deactivate,
        )

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


def install(mini: Any) -> Optional[PrivacyMode]:
    """Attach a PrivacyMode if env-gate is on. Returns the instance or None."""
    if not is_enabled():
        return None
    pm = PrivacyMode(mini)
    pm.start()
    return pm


def _reset_for_tests() -> None:
    """Drop the module-level privacy flag so tests start clean."""
    _set_privacy_active(False)
