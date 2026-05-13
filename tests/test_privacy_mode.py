"""Tests for the antenna-press privacy mode."""

from __future__ import annotations

import math
from typing import Any, List, Optional, Tuple
from unittest.mock import MagicMock

import pytest
from reachy_mini_supermemory_app import _privacy_mode as pm


@pytest.fixture(autouse=True)
def _reset() -> Any:
    pm._reset_for_tests()
    yield
    # Also reset on the way out so the module-level flag doesn't leak into
    # later test modules (auto-digest reads is_privacy_active()).
    pm._reset_for_tests()


def _make_controller(
    positions: List[Optional[Tuple[float, float]]],
    deviation_deg: float = 25.0,
    debounce_ms: int = 0,
) -> Tuple[pm.PrivacyController, List[str]]:
    """Build a controller whose get_positions plays back the given samples in order."""
    events: List[str] = []
    iterator = iter(positions)

    def _next_pos() -> Optional[Tuple[float, float]]:
        try:
            return next(iterator)
        except StopIteration:
            return None

    controller = pm.PrivacyController(
        get_present_positions=_next_pos,
        on_activate=lambda: events.append("on"),
        on_deactivate=lambda: events.append("off"),
        deviation_deg=deviation_deg,
        debounce_ms=debounce_ms,
    )
    return controller, events


def _rad(degrees: float) -> float:
    return math.radians(degrees)


def test_small_jitter_does_not_trigger() -> None:
    samples = [
        (0.0, 0.0),
        (0.05, -0.05),
        (-0.03, 0.04),
        (0.02, 0.0),
    ]
    controller, events = _make_controller(samples)
    for _ in samples:
        controller.tick()
    assert events == []
    assert not pm.is_privacy_active()


def test_large_deflection_triggers_activate() -> None:
    # Fill the window with calm samples then deflect one antenna sharply.
    samples = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (_rad(40), 0.0),  # right antenna pushed ~40°
    ]
    controller, events = _make_controller(samples)
    for _ in samples:
        controller.tick()
    assert events == ["on"]
    assert pm.is_privacy_active()


def test_second_press_deactivates() -> None:
    # 8 samples: first 4 set up + trigger ON; clear window then 4 more trigger OFF.
    samples = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (_rad(40), 0.0),
        # After the ON toggle the window is cleared. Refill with calm, then deflect.
        (_rad(40), 0.0),
        (_rad(40), 0.0),
        (_rad(40), 0.0),
        (0.0, _rad(40)),  # left antenna pushed
    ]
    controller, events = _make_controller(samples)
    for _ in samples:
        controller.tick()
    assert events == ["on", "off"]
    assert not pm.is_privacy_active()


def test_debounce_blocks_immediate_retrigger() -> None:
    samples = [
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (_rad(40), 0.0),
        (_rad(40), 0.0),
        (_rad(40), 0.0),
        (_rad(40), 0.0),
        (0.0, _rad(40)),  # would be a second press but should be debounced
    ]
    # 5s debounce — second press happens within the window
    controller, events = _make_controller(samples, debounce_ms=5000)
    for _ in samples:
        controller.tick()
    assert events == ["on"]  # only one toggle
    assert pm.is_privacy_active()


def test_partial_window_does_not_trigger() -> None:
    # Fewer than WINDOW_SIZE samples should never trigger.
    samples = [(0.0, 0.0), (_rad(40), _rad(40))]
    controller, events = _make_controller(samples)
    for _ in samples:
        controller.tick()
    assert events == []


def test_none_positions_are_safe() -> None:
    samples: List[Optional[Tuple[float, float]]] = [None, None, None, (0.0, 0.0)]
    controller, events = _make_controller(samples)
    for _ in samples:
        controller.tick()
    assert events == []


def test_install_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(pm.ENABLED_ENV, raising=False)
    fake_mini = MagicMock()
    assert pm.install(fake_mini) is None


def test_install_starts_controller_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(pm.ENABLED_ENV, "true")
    fake_mini = MagicMock()
    fake_mini.get_present_antenna_joint_positions.return_value = (0.0, 0.0)
    instance = pm.install(fake_mini)
    try:
        assert instance is not None
        assert instance.controller._thread is not None
        assert instance.controller._thread.is_alive()
    finally:
        if instance is not None:
            instance.stop()


def test_privacy_mode_activate_calls_audio_and_antenna_methods() -> None:
    mini = MagicMock()
    mini.get_present_antenna_joint_positions.return_value = (0.1, -0.1)
    instance = pm.PrivacyMode(mini)

    instance._on_activate()
    # Antennas folded to the privacy pose
    mini.set_target_antenna_joint_positions.assert_called_with(pm.PRIVACY_POSE)
    # Mic stopped, speaker queue flushed
    mini.media.audio.stop_recording.assert_called_once()
    mini.media.audio.clear_player.assert_called_once()
    # Saved the previous antenna pose for restore
    assert instance._saved_antennas == (0.1, -0.1)


def test_privacy_mode_deactivate_restores() -> None:
    mini = MagicMock()
    instance = pm.PrivacyMode(mini)
    instance._saved_antennas = (0.2, -0.2)

    instance._on_deactivate()
    mini.set_target_antenna_joint_positions.assert_called_with((0.2, -0.2))
    mini.media.audio.start_recording.assert_called_once()


def test_handler_errors_dont_propagate() -> None:
    # If on_activate raises, the controller should swallow it and not crash the thread.
    samples = [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (_rad(40), 0.0)]

    def _boom() -> None:
        raise RuntimeError("boom")

    iterator = iter(samples)
    controller = pm.PrivacyController(
        get_present_positions=lambda: next(iterator, None),
        on_activate=_boom,
        on_deactivate=lambda: None,
    )
    # Should not raise even though on_activate did
    for _ in samples:
        controller.tick()
    # State still flipped despite the handler error (caller's responsibility)
    assert pm.is_privacy_active()
