"""Tests for the web_search tool (Tavily adapter)."""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import patch

import httpx
import pytest

# The profile-local tool path is added to sys.path by conftest.
from web_search import (  # type: ignore[import-not-found]
    DEFAULT_MAX_RESULTS,
    DEFAULT_SEARCH_DEPTH,
    MAX_MAX_RESULTS,
    RESULT_CONTENT_MAX_CHARS,
    SEARCH_DEPTHS,
    TAVILY_API_KEY_ENV,
    WebSearch,
    _coerce_max_results,
    _coerce_search_depth,
    _format_response,
)


# ---------- helpers ----------


class _FakeResp:
    """httpx.Response stand-in for tests."""

    def __init__(self, status_code: int, json_body: Optional[Dict[str, Any]] = None, text: str = "") -> None:
        self.status_code = status_code
        self._json = json_body if json_body is not None else {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._json


class _FakeClient:
    """``httpx.AsyncClient`` stand-in. Returns ``response`` or raises if it's an exception."""

    def __init__(self, response: Any) -> None:
        self._response = response

    async def post(self, *a: Any, **kw: Any) -> Any:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _install_fake_client(monkeypatch: pytest.MonkeyPatch, response: Any) -> _FakeClient:
    fake = _FakeClient(response)
    monkeypatch.setattr("web_search._client_for_current_loop", lambda: fake)
    return fake


def _tavily_payload(*, answer: Optional[str] = None, results: Optional[list] = None) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"query": "x", "response_time": 0.5}
    if answer is not None:
        payload["answer"] = answer
    payload["results"] = results if results is not None else []
    return payload


# ---------- coercion helpers ----------


def test_coerce_max_results_defaults_when_none() -> None:
    assert _coerce_max_results(None) == DEFAULT_MAX_RESULTS


def test_coerce_max_results_clamps_above_max() -> None:
    assert _coerce_max_results(999) == MAX_MAX_RESULTS


def test_coerce_max_results_clamps_below_one() -> None:
    assert _coerce_max_results(0) == 1
    assert _coerce_max_results(-5) == 1


def test_coerce_max_results_tolerates_garbage() -> None:
    assert _coerce_max_results("not a number") == DEFAULT_MAX_RESULTS


def test_coerce_search_depth_defaults_when_unknown() -> None:
    assert _coerce_search_depth("garbage") == DEFAULT_SEARCH_DEPTH
    assert _coerce_search_depth(None) == DEFAULT_SEARCH_DEPTH
    assert _coerce_search_depth(42) == DEFAULT_SEARCH_DEPTH


def test_coerce_search_depth_accepts_valid_values() -> None:
    for depth in SEARCH_DEPTHS:
        assert _coerce_search_depth(depth) == depth
        assert _coerce_search_depth(depth.upper()) == depth


# ---------- _format_response ----------


def test_format_response_truncates_long_content() -> None:
    long = "x" * (RESULT_CONTENT_MAX_CHARS + 200)
    payload = _tavily_payload(
        answer="brief summary",
        results=[{"title": "T", "url": "https://example.com", "content": long}],
    )
    out = _format_response(payload, max_results=5)
    assert out["answer"] == "brief summary"
    [result] = out["results"]
    assert result["title"] == "T"
    assert result["url"] == "https://example.com"
    assert len(result["content"]) <= RESULT_CONTENT_MAX_CHARS + 1  # +1 for the ellipsis
    assert result["content"].endswith("…")


def test_format_response_caps_at_max_results() -> None:
    payload = _tavily_payload(
        results=[
            {"title": str(i), "url": f"https://e/{i}", "content": "ok"} for i in range(10)
        ]
    )
    out = _format_response(payload, max_results=3)
    assert len(out["results"]) == 3
    assert [r["title"] for r in out["results"]] == ["0", "1", "2"]


def test_format_response_omits_answer_when_missing_or_blank() -> None:
    payload = _tavily_payload(answer="   ", results=[])
    out = _format_response(payload, max_results=3)
    assert "answer" not in out

    payload2 = _tavily_payload(results=[])
    out2 = _format_response(payload2, max_results=3)
    assert "answer" not in out2


def test_format_response_skips_non_dict_results() -> None:
    payload = _tavily_payload(results=["nope", {"title": "T", "url": "u", "content": "c"}, 42])
    out = _format_response(payload, max_results=5)
    assert len(out["results"]) == 1


def test_format_response_drops_score_and_raw_content() -> None:
    """Pruning keeps tool output compact for the realtime session."""
    payload = _tavily_payload(
        results=[
            {
                "title": "T",
                "url": "u",
                "content": "c",
                "score": 0.99,
                "raw_content": "huge blob",
            }
        ]
    )
    out = _format_response(payload, max_results=1)
    assert set(out["results"][0].keys()) == {"title", "url", "content"}


# ---------- WebSearch.__call__ ----------


@pytest.mark.asyncio
async def test_rejects_empty_query() -> None:
    tool = WebSearch()
    result = await tool(deps=None, query="   ")  # type: ignore[arg-type]
    assert result == {"error": "query is required"}


@pytest.mark.asyncio
async def test_errors_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TAVILY_API_KEY_ENV, raising=False)
    tool = WebSearch()
    result = await tool(deps=None, query="weather")  # type: ignore[arg-type]
    assert "error" in result
    assert "TAVILY_API_KEY" in result["error"]


@pytest.mark.asyncio
async def test_successful_search_returns_formatted_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")
    _install_fake_client(
        monkeypatch,
        _FakeResp(
            200,
            _tavily_payload(
                answer="Tomorrow it's 18°C.",
                results=[
                    {
                        "title": "Cape Town weather",
                        "url": "https://example.com/weather",
                        "content": "Cape Town forecast: 18°C with light cloud cover.",
                    }
                ],
            ),
        ),
    )

    result = await WebSearch()(deps=None, query="Cape Town weather")  # type: ignore[arg-type]
    assert result["answer"] == "Tomorrow it's 18°C."
    assert len(result["results"]) == 1
    assert result["results"][0]["url"] == "https://example.com/weather"


@pytest.mark.asyncio
async def test_request_body_sets_required_tavily_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the wire format actually matches Tavily's contract."""
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")
    captured: Dict[str, Any] = {}

    class _CapturingClient:
        async def post(self, url: str, *, headers: Dict[str, str], json: Dict[str, Any], **_kw: Any) -> _FakeResp:
            captured["url"] = url
            captured["headers"] = headers
            captured["body"] = json
            return _FakeResp(200, _tavily_payload(results=[]))

    monkeypatch.setattr("web_search._client_for_current_loop", lambda: _CapturingClient())

    await WebSearch()(deps=None, query="elections 2026", max_results=5, search_depth="advanced")  # type: ignore[arg-type]

    assert captured["url"].endswith("/search")
    assert captured["headers"]["Authorization"] == "Bearer tvly-test"
    assert captured["headers"]["Content-Type"] == "application/json"
    body = captured["body"]
    assert body["query"] == "elections 2026"
    assert body["max_results"] == 5
    assert body["search_depth"] == "advanced"
    assert body["include_answer"] is True
    assert body["include_raw_content"] is False
    assert body["include_images"] is False


@pytest.mark.asyncio
async def test_http_4xx_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")
    _install_fake_client(monkeypatch, _FakeResp(401, text="invalid key"))
    result = await WebSearch()(deps=None, query="anything")  # type: ignore[arg-type]
    assert "error" in result
    assert "401" in result["error"]


@pytest.mark.asyncio
async def test_timeout_returns_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")
    _install_fake_client(monkeypatch, httpx.TimeoutException("slow"))
    result = await WebSearch()(deps=None, query="x")  # type: ignore[arg-type]
    assert result == {"error": "Search request timed out."}


@pytest.mark.asyncio
async def test_network_error_returns_friendly_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")
    _install_fake_client(monkeypatch, httpx.ConnectError("dns fail"))
    result = await WebSearch()(deps=None, query="x")  # type: ignore[arg-type]
    assert "error" in result
    assert "Search request failed" in result["error"]


@pytest.mark.asyncio
async def test_non_json_response_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")

    class _BadJsonResp:
        status_code = 200
        text = "<html>oops</html>"

        def json(self) -> Dict[str, Any]:
            raise ValueError("not json")

    _install_fake_client(monkeypatch, _BadJsonResp())
    result = await WebSearch()(deps=None, query="x")  # type: ignore[arg-type]
    assert result == {"error": "Search returned non-JSON response."}


@pytest.mark.asyncio
async def test_base_url_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Self-hosted / proxied deployments can override the Tavily endpoint."""
    monkeypatch.setenv(TAVILY_API_KEY_ENV, "tvly-test")
    monkeypatch.setenv("TAVILY_BASE_URL", "https://proxy.example.com")
    captured: Dict[str, Any] = {}

    class _CapturingClient:
        async def post(self, url: str, **_kw: Any) -> _FakeResp:
            captured["url"] = url
            return _FakeResp(200, _tavily_payload(results=[]))

    monkeypatch.setattr("web_search._client_for_current_loop", lambda: _CapturingClient())
    await WebSearch()(deps=None, query="x")  # type: ignore[arg-type]
    assert captured["url"] == "https://proxy.example.com/search"
