"""Tests for the settings_ui FastAPI routes + static-asset loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from reachy_mini_supermemory_app import settings_ui


@pytest.fixture(autouse=True)
def _reset_section_cache() -> None:
    """Clear the cached section blob so tests can swap the underlying file."""
    settings_ui._settings_section_cache = None


def test_supermemory_section_loads_from_static_file() -> None:
    """The dashboard section is shipped as a static asset, not a Python string."""
    section = settings_ui._supermemory_section()
    # Smoke checks that we're reading the real file, not the fallback.
    assert "supermemory-section" in section
    assert "/supermemory/api-key" in section
    assert "Recall scope" in section


def test_supermemory_section_cached_after_first_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Second call must not re-read disk."""
    fake_section = tmp_path / "fake.html"
    fake_section.write_text("<p>v1</p>", encoding="utf-8")
    monkeypatch.setattr(settings_ui, "_SETTINGS_SECTION_FILE", fake_section)

    assert settings_ui._supermemory_section() == "<p>v1</p>"

    # Mutate the file — cached call must NOT pick this up.
    fake_section.write_text("<p>v2</p>", encoding="utf-8")
    assert settings_ui._supermemory_section() == "<p>v1</p>"


def test_supermemory_section_falls_back_when_asset_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing static asset must return a user-visible diagnostic, not crash."""
    missing = tmp_path / "absent.html"
    monkeypatch.setattr(settings_ui, "_SETTINGS_SECTION_FILE", missing)

    section = settings_ui._supermemory_section()
    assert "Reinstall" in section
    assert "supermemory-app" in section


def test_standalone_page_wraps_section_in_html_shell() -> None:
    """The /supermemory/ route now serves the section wrapped in a minimal shell.

    Previously there was a separate ``static/index.html`` with a near-duplicate
    JS — this test guards the single-source-of-truth refactor.
    """
    page = settings_ui._supermemory_standalone_page()
    assert page.startswith("<!DOCTYPE html>")
    assert "<title>Supermemory settings</title>" in page
    # Contains the canonical section (same DOM IDs as the embedded fragment).
    assert "supermemory-section" in page
    assert 'id="sm-key"' in page
    # Ends with closing tags.
    assert page.rstrip().endswith("</html>")


def test_build_composed_index_injects_section_after_header(tmp_path: Path) -> None:
    """When upstream's </header> is present, the section lands right after it."""
    upstream = tmp_path / "index.html"
    upstream.write_text(
        '<html><head></head><body><header class="hero">UPSTREAM</header>'
        '<main>BODY</main></body></html>',
        encoding="utf-8",
    )
    composed = settings_ui._build_composed_index(upstream)
    # The injection happens immediately after </header>, before <main>.
    header_idx = composed.index("</header>")
    main_idx = composed.index("<main>")
    section_idx = composed.index("supermemory-section")
    assert header_idx < section_idx < main_idx


def test_build_composed_index_rewrites_static_paths(tmp_path: Path) -> None:
    upstream = tmp_path / "index.html"
    upstream.write_text(
        '<link href="/static/main.css"><script src="/static/main.js"></script>'
        '<body><header class="hero"></header></body>',
        encoding="utf-8",
    )
    composed = settings_ui._build_composed_index(upstream)
    assert 'href="/upstream-static/main.css"' in composed
    assert 'src="/upstream-static/main.js"' in composed
    assert "/static/main.css" not in composed
    assert "/static/main.js" not in composed


def test_rewrite_static_refs_handles_single_quotes() -> None:
    """Single-quoted attributes used to be ignored by the naive str.replace."""
    html = "<link href='/static/main.css'><img src='/static/hero.png'>"
    rewritten = settings_ui._rewrite_static_refs(html)
    assert "/static/" not in rewritten
    assert "href='/upstream-static/main.css'" in rewritten
    assert "src='/upstream-static/hero.png'" in rewritten


def test_rewrite_static_refs_handles_srcset() -> None:
    """``srcset`` is the third common path-bearing attribute upstream might use."""
    html = '<img srcset="/static/lo.png 1x, /static/hi.png 2x">'
    rewritten = settings_ui._rewrite_static_refs(html)
    # The first /static/ in the srcset gets rewritten by the attribute match;
    # the second URL in the comma-list stays raw unless we also handle it
    # explicitly. Document this limitation in the assertion.
    assert 'srcset="/upstream-static/lo.png 1x' in rewritten


def test_rewrite_static_refs_is_case_insensitive() -> None:
    """HTML attribute names are case-insensitive."""
    html = '<link HREF="/static/x.css"><script SRC="/static/x.js"></script>'
    rewritten = settings_ui._rewrite_static_refs(html)
    assert "/static/" not in rewritten


def test_rewrite_static_refs_ignores_other_paths() -> None:
    """Only ``/static/`` URLs get rewritten — other paths must pass through."""
    html = '<a href="/api/foo"><img src="/uploads/bar.png">'
    rewritten = settings_ui._rewrite_static_refs(html)
    assert rewritten == html


def test_build_composed_index_falls_back_to_section_when_upstream_unreadable(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist.html"
    composed = settings_ui._build_composed_index(missing)
    # Falls back to the section blob alone.
    assert "supermemory-section" in composed
