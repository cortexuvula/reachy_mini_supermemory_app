"""FastAPI routes for managing the supermemory API key from the headless UI.

Mirrors the env-write pattern used by ``reachy_mini_conversation_app.console``
without reaching into private API: the env-write logic is duplicated locally.
"""

import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, List, Optional

from ._supermemory_client import (
    RECALL_EXCLUDED_TAGS_ENV,
    discover_container_tags,
    invalidate_tag_cache,
    is_configured,
    recall_excluded_tags,
    recall_pinned_tags,
)

logger = logging.getLogger(__name__)

_API_KEY_ENV = "SUPERMEMORY_API_KEY"
_env_file_lock = threading.Lock()


def _persist_to_env_file(env_path: Path, key: str, value: str) -> bool:
    """Insert or replace ``KEY=value`` in ``env_path`` (creating the file if needed).

    Returns True on success, False if the value was rejected or the write
    failed. Rejects values containing newlines because a multi-line value
    silently corrupts the .env on the next read (subsequent lines parse as
    standalone KEY=VAL pairs). All occurrences of ``KEY=`` are removed before
    the new line is appended, so a manually-edited .env with duplicate keys
    is reduced to a single canonical entry.

    Held under ``_env_file_lock`` so concurrent POSTs (e.g. saving the API key
    and updating tag excludes at the same time) can't interleave their
    read-modify-write and silently drop one of the keys.
    """
    if "\n" in value or "\r" in value:
        logger.warning("Refusing to persist %s: value contains newline", key)
        return False

    with _env_file_lock:
        try:
            lines: list[str]
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
            else:
                lines = []

            prefix = f"{key}="
            # Drop ALL existing assignments for this key — duplicates from a
            # hand-edited .env would otherwise shadow our write on the next
            # python-dotenv load (last assignment wins).
            kept = [line for line in lines if not line.strip().startswith(prefix)]
            kept.append(f"{key}={value}")

            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
            logger.info("Persisted %s to %s", key, env_path)
            return True
        except Exception as e:
            logger.warning("Failed to persist %s to %s: %s", key, env_path, e)
            return False


# The dashboard supermemory section is shipped as a static asset so the
# HTML/CSS/JS gets proper editor support (syntax highlighting, formatting,
# linting). Loaded lazily on first use and cached for the process lifetime.
_SETTINGS_SECTION_FILE = Path(__file__).parent / "static" / "settings_section.html"
_settings_section_cache: Optional[str] = None
_FALLBACK_SECTION = (
    "<p>Supermemory settings UI assets missing. "
    "Reinstall the reachy-mini-supermemory-app package.</p>"
)


def _supermemory_section() -> str:
    """Return the dashboard supermemory section HTML, cached after first read."""
    global _settings_section_cache
    if _settings_section_cache is not None:
        return _settings_section_cache
    try:
        _settings_section_cache = _SETTINGS_SECTION_FILE.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning("Could not read %s: %s — falling back", _SETTINGS_SECTION_FILE, e)
        _settings_section_cache = _FALLBACK_SECTION
    return _settings_section_cache


_STANDALONE_PAGE_SHELL = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no" />
  <title>Supermemory settings</title>
  <style>:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; }} body {{ margin: 0; }}</style>
</head>
<body>
{section}
</body>
</html>
"""


def _supermemory_standalone_page() -> str:
    """Return the standalone ``/supermemory/`` page (section wrapped in an HTML shell).

    Previously this view lived in a separate ``static/index.html`` whose
    JS was a hand-maintained duplicate of the embedded section. Single source
    of truth now: the section file is the canonical UI, and the standalone
    view is just a small chrome wrapper around it.
    """
    return _STANDALONE_PAGE_SHELL.format(section=_supermemory_section())


def _compose_dashboard_index(app: Any, **types: Any) -> None:
    """Replace the dashboard's "/" page with upstream conversation_app's index
    plus our supermemory section, so the dashboard panel shows both at once.

    Without this the dashboard iframe only sees our standalone supermemory
    page — the upstream Headless control UI (backend selection, profile
    tweaks, etc.) is invisible while our app is the active one.

    The composed HTML is built once at mount and served from a closure so the
    hot dashboard-refresh path doesn't re-read the file from disk and re-run
    string rewrites on every GET.
    """
    HTMLResponse = types["HTMLResponse"]
    StaticFiles = types["StaticFiles"]

    try:
        import reachy_mini_conversation_app  # type: ignore[import-not-found]
    except Exception:
        return

    upstream_static = Path(reachy_mini_conversation_app.__file__).resolve().parent / "static"
    upstream_index = upstream_static / "index.html"
    if not upstream_index.exists():
        return

    # Drop any "/" route upstream already registered (ReachyMiniApp.__init__
    # adds one that serves our standalone static/index.html) so ours wins.
    app.routes[:] = [r for r in app.routes if not (getattr(r, "path", None) == "/")]
    # Make upstream's main.js / style.css reachable so the combined page works.
    try:
        app.mount("/upstream-static", StaticFiles(directory=upstream_static), name="upstream-static")
    except Exception:
        pass

    composed_html = _build_composed_index(upstream_index)

    @app.get("/")
    async def _combined() -> Any:  # type: ignore[misc]
        return HTMLResponse(composed_html)


_STATIC_REF_RE = re.compile(
    r"""(?P<attr>href|src|srcset)\s*=\s*(?P<quote>['"])/static/""",
    re.IGNORECASE,
)


def _rewrite_static_refs(html: str) -> str:
    """Rewrite upstream's ``/static/`` URLs to our ``/upstream-static/`` mount.

    The previous version used naive ``str.replace`` against double-quoted
    forms only. A regex covers single quotes, attribute-value whitespace,
    and ``srcset`` — anything beyond that (CSS ``url()`` references in inline
    styles, JS string literals) would still need separate handling, but
    none of those appear in upstream's current index.html.
    """
    return _STATIC_REF_RE.sub(
        lambda m: f'{m.group("attr")}={m.group("quote")}/upstream-static/',
        html,
    )


def _build_composed_index(upstream_index: Path) -> str:
    """Read upstream's index.html, rewrite static refs, inject our section.

    Pure function so the result can be cached at mount time. Falls back to
    serving just our supermemory section if upstream's file can't be read.
    """
    section = _supermemory_section()
    try:
        html = upstream_index.read_text(encoding="utf-8")
    except Exception:
        return section
    html = _rewrite_static_refs(html)
    # Inject our supermemory section at the TOP of the container (right after
    # upstream's hero), so it's visible in the dashboard iframe without
    # scrolling past upstream's long settings list to find it.
    for marker in ('</header>',):  # closes upstream's <header class="hero">
        if marker in html:
            return html.replace(marker, marker + section, 1)
    # Fallback: append at end if upstream's layout changed.
    return html.replace("</body>", section + "</body>", 1)


def mount_supermemory_routes(app: object, instance_path: Optional[str] = None) -> None:
    """Mount /supermemory/ routes on the provided FastAPI settings app.

    The routes are best-effort: if FastAPI/pydantic isn't importable in this
    environment (e.g. unit tests with no daemon), the function silently no-ops.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except Exception:  # pragma: no cover - settings app only present at runtime
        logger.debug("FastAPI not available; skipping supermemory settings routes.")
        return

    if not isinstance(app, FastAPI):
        logger.debug("Settings app is not a FastAPI instance; skipping supermemory routes.")
        return

    _compose_dashboard_index(app, HTMLResponse=HTMLResponse, StaticFiles=StaticFiles)

    class ApiKeyPayload(BaseModel):
        key: str

    class ExcludesPayload(BaseModel):
        excluded: List[str]

    standalone_page = _supermemory_standalone_page()

    @app.get("/supermemory/")
    def _index() -> HTMLResponse:  # type: ignore[misc]
        return HTMLResponse(standalone_page)

    @app.get("/supermemory/status")
    def _status() -> JSONResponse:  # type: ignore[misc]
        return JSONResponse({"configured": is_configured()})

    @app.post("/supermemory/api-key")
    def _set_key(payload: ApiKeyPayload) -> JSONResponse:  # type: ignore[misc]
        value = (payload.key or "").strip()
        if not value:
            return JSONResponse({"ok": False, "error": "API key is empty."}, status_code=400)
        if "\n" in value or "\r" in value:
            return JSONResponse(
                {"ok": False, "error": "API key cannot contain newlines."}, status_code=400
            )
        os.environ[_API_KEY_ENV] = value
        if instance_path:
            _persist_to_env_file(Path(instance_path) / ".env", _API_KEY_ENV, value)
        return JSONResponse({"ok": True})

    @app.get("/supermemory/tags")
    async def _list_tags() -> JSONResponse:  # type: ignore[misc]
        if not is_configured():
            return JSONResponse({"configured": False, "pinned": None, "discovered": [], "excluded": []})
        pinned = recall_pinned_tags()
        if pinned:
            return JSONResponse(
                {"configured": True, "pinned": pinned, "discovered": [], "excluded": recall_excluded_tags()}
            )
        discovered = await discover_container_tags()
        return JSONResponse(
            {"configured": True, "pinned": None, "discovered": discovered, "excluded": recall_excluded_tags()}
        )

    @app.post("/supermemory/tags")
    def _set_tag_excludes(payload: ExcludesPayload) -> JSONResponse:  # type: ignore[misc]
        excluded = sorted({t.strip() for t in (payload.excluded or []) if isinstance(t, str) and t.strip()})
        value = ",".join(excluded)
        if value:
            os.environ[RECALL_EXCLUDED_TAGS_ENV] = value
        else:
            os.environ.pop(RECALL_EXCLUDED_TAGS_ENV, None)
        if instance_path:
            _persist_to_env_file(Path(instance_path) / ".env", RECALL_EXCLUDED_TAGS_ENV, value)
        return JSONResponse({"ok": True, "excluded": excluded})

    @app.post("/supermemory/tags/refresh")
    async def _refresh_tags() -> JSONResponse:  # type: ignore[misc]
        invalidate_tag_cache()
        if not is_configured():
            return JSONResponse({"configured": False, "discovered": []})
        discovered = await discover_container_tags()
        return JSONResponse({"configured": True, "discovered": discovered})
