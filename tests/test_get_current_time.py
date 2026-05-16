"""Tests for the get_current_time tool."""

from __future__ import annotations

import re
from typing import Any

import pytest

# Profile-local tool path is added by conftest.
from get_current_time import (  # type: ignore[import-not-found]
    USER_TIMEZONE_ENV,
    GetCurrentTime,
    _format_date,
    _format_time,
)


# ---------- formatting helpers ----------


def test_format_date_strips_leading_zero_on_day() -> None:
    import datetime

    d = datetime.datetime(2026, 5, 6, 14, 30)
    assert _format_date(d) == "May 6, 2026"


def test_format_time_renders_12hour() -> None:
    import datetime

    morning = datetime.datetime(2026, 5, 16, 9, 5)
    afternoon = datetime.datetime(2026, 5, 16, 14, 18)
    assert _format_time(morning).endswith("AM")
    assert _format_time(afternoon).endswith("PM")
    # No leading zero on hour:
    assert _format_time(morning).startswith("9:")
    assert _format_time(afternoon).startswith("2:")


# ---------- GetCurrentTime.__call__ ----------


@pytest.mark.asyncio
async def test_returns_full_payload_for_local_time(monkeypatch: pytest.MonkeyPatch) -> None:
    """No timezone param + no env override → use OS-local time."""
    monkeypatch.delenv(USER_TIMEZONE_ENV, raising=False)
    result = await GetCurrentTime()(deps=None)  # type: ignore[arg-type]
    assert {"weekday", "date", "time", "timezone", "iso"} <= result.keys()
    # ISO is parseable.
    import datetime

    datetime.datetime.fromisoformat(result["iso"])
    # No spurious warning on the happy path.
    assert "warning" not in result


@pytest.mark.asyncio
async def test_respects_env_timezone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_TIMEZONE_ENV, "Asia/Tokyo")
    result = await GetCurrentTime()(deps=None)  # type: ignore[arg-type]
    assert result["timezone"] == "Asia/Tokyo"


@pytest.mark.asyncio
async def test_explicit_param_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Per-call ``timezone`` arg wins over the env-configured default."""
    monkeypatch.setenv(USER_TIMEZONE_ENV, "Asia/Tokyo")
    result = await GetCurrentTime()(deps=None, timezone="Europe/London")  # type: ignore[arg-type]
    assert result["timezone"] == "Europe/London"


@pytest.mark.asyncio
async def test_unknown_timezone_returns_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(USER_TIMEZONE_ENV, raising=False)
    result = await GetCurrentTime()(deps=None, timezone="Not/A/Real/Zone")  # type: ignore[arg-type]
    assert "warning" in result
    assert "Not/A/Real/Zone" in result["warning"]


@pytest.mark.asyncio
async def test_empty_timezone_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(USER_TIMEZONE_ENV, raising=False)
    result = await GetCurrentTime()(deps=None, timezone="   ")  # type: ignore[arg-type]
    assert "warning" not in result


@pytest.mark.asyncio
async def test_weekday_is_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_TIMEZONE_ENV, "Europe/Paris")
    result = await GetCurrentTime()(deps=None)  # type: ignore[arg-type]
    assert result["weekday"] in {
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }


@pytest.mark.asyncio
async def test_time_format_matches_12hour_pattern(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(USER_TIMEZONE_ENV, "UTC")
    result = await GetCurrentTime()(deps=None)  # type: ignore[arg-type]
    assert re.match(r"^\d{1,2}:\d{2} (AM|PM)$", result["time"])


@pytest.mark.asyncio
async def test_tool_name_and_schema() -> None:
    """Sanity check on the tool's declared contract."""
    tool = GetCurrentTime()
    assert tool.name == "get_current_time"
    assert tool.parameters_schema["type"] == "object"
    # timezone is optional, not required.
    assert tool.parameters_schema.get("required") == []
    assert "timezone" in tool.parameters_schema["properties"]
