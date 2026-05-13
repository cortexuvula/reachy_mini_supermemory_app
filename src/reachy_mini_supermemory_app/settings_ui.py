"""FastAPI routes for managing the supermemory API key from the headless UI.

Mirrors the env-write pattern used by ``reachy_mini_conversation_app.console``
without reaching into private API: the env-write logic is duplicated locally.
"""

import logging
import os
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


def _persist_to_env_file(env_path: Path, key: str, value: str) -> None:
    """Insert or replace ``KEY=value`` in ``env_path`` (creating the file if needed).

    Held under ``_env_file_lock`` so concurrent POSTs (e.g. saving the API key
    and updating tag excludes at the same time) can't interleave their
    read-modify-write and silently drop one of the keys.
    """
    with _env_file_lock:
        try:
            lines: list[str]
            if env_path.exists():
                lines = env_path.read_text(encoding="utf-8").splitlines()
            else:
                lines = []

            replaced = False
            for i, line in enumerate(lines):
                if line.strip().startswith(f"{key}="):
                    lines[i] = f"{key}={value}"
                    replaced = True
                    break
            if not replaced:
                lines.append(f"{key}={value}")

            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Persisted %s to %s", key, env_path)
        except Exception as e:
            logger.warning("Failed to persist %s to %s: %s", key, env_path, e)


_SUPERMEMORY_SECTION = """
<section id="supermemory-section" style="max-width:32rem;margin:2rem auto;padding:0 1rem;font-family:system-ui,sans-serif;color:inherit;">
  <h2 style="font-size:1.25rem;margin-bottom:0.5rem;">Supermemory</h2>
  <p style="color:#888;">API key used by <code>save_memory</code> and <code>recall_memory</code>. Get one at <a href="https://supermemory.ai" target="_blank" rel="noreferrer">supermemory.ai</a>.</p>
  <div id="sm-status" class="sm-status">Checking…</div>
  <label for="sm-key" style="display:block;margin:1rem 0 0.25rem;font-weight:600;">API key</label>
  <input id="sm-key" type="password" autocomplete="off" placeholder="sk-..." style="width:100%;padding:0.5rem;font-size:1rem;box-sizing:border-box;" />
  <button id="sm-save" style="margin-top:1rem;padding:0.6rem 1.2rem;font-size:1rem;cursor:pointer;">Save</button>

  <h3 style="font-size:1.05rem;margin:2rem 0 0.5rem;">Recall scope</h3>
  <p style="color:#888;">Tags <code>recall_memory</code> searches across. Saves always go to <code>reachy-mini:&lt;profile&gt;</code>. Uncheck any tag you don't want recall to read.</p>
  <div id="sm-tags-status" class="sm-status sm-info">Loading tags…</div>
  <ul id="sm-tag-list" style="list-style:none;padding:0;margin:0.5rem 0;"></ul>
  <div style="display:flex;gap:0.5rem;flex-wrap:wrap;">
    <button id="sm-apply-tags" style="padding:0.5rem 1rem;cursor:pointer;">Apply</button>
    <button id="sm-refresh-tags" style="padding:0.5rem 1rem;cursor:pointer;background:transparent;border:1px solid #888;">Refresh tags</button>
  </div>
  <style>
    #supermemory-section .sm-status { margin-top:0.5rem;padding:0.5rem 0.75rem;border-radius:0.25rem; }
    #supermemory-section .sm-ok { background:#d1f0d1;color:#1e5e1e; }
    #supermemory-section .sm-err { background:#f0d1d1;color:#5e1e1e; }
    #supermemory-section .sm-info { background:#e6e6e6;color:#333; }
    #supermemory-section #sm-tag-list li { padding:0.35rem 0;border-bottom:1px solid rgba(127,127,127,0.2);display:flex;align-items:center;gap:0.5rem; }
    #supermemory-section #sm-tag-list code { font-family:ui-monospace,monospace;font-size:0.95rem; }
  </style>
  <script>
  (function(){
    const $ = (id) => document.getElementById(id);
    const statusEl = $("sm-status"), keyEl = $("sm-key"), saveBtn = $("sm-save");
    const tagsStatusEl = $("sm-tags-status"), tagListEl = $("sm-tag-list");
    const applyBtn = $("sm-apply-tags"), refreshBtn = $("sm-refresh-tags");
    async function refresh(){ try { const j = await (await fetch("/supermemory/status")).json(); statusEl.className = "sm-status " + (j.configured?"sm-ok":"sm-err"); statusEl.textContent = j.configured?"API key is configured.":"No API key set."; } catch(e) { statusEl.className = "sm-status sm-err"; statusEl.textContent = "Could not reach settings backend."; } }
    function renderTags(p){ tagListEl.replaceChildren(); if (p.configured===false){ tagsStatusEl.className="sm-status sm-info"; tagsStatusEl.textContent="Set an API key first."; applyBtn.disabled=true; refreshBtn.disabled=true; return; } if (p.pinned){ tagsStatusEl.className="sm-status sm-info"; tagsStatusEl.textContent="Pinned via SUPERMEMORY_RECALL_CONTAINER_TAGS: "+p.pinned.join(", "); applyBtn.disabled=true; refreshBtn.disabled=true; return; } const discovered=p.discovered||[]; const excluded=new Set(p.excluded||[]); if (!discovered.length){ tagsStatusEl.className="sm-status sm-info"; tagsStatusEl.textContent="No tags discovered yet."; applyBtn.disabled=true; refreshBtn.disabled=false; return; } tagsStatusEl.className="sm-status sm-ok"; tagsStatusEl.textContent=discovered.length+" tag"+(discovered.length===1?"":"s")+" discovered."; for (const tag of discovered){ const li=document.createElement("li"); const cb=document.createElement("input"); cb.type="checkbox"; cb.id="sm-tag-"+tag; cb.checked=!excluded.has(tag); cb.dataset.tag=tag; const lbl=document.createElement("label"); lbl.htmlFor=cb.id; lbl.style.fontWeight="normal"; lbl.style.margin="0"; const c=document.createElement("code"); c.textContent=tag; lbl.appendChild(c); li.appendChild(cb); li.appendChild(lbl); tagListEl.appendChild(li); } applyBtn.disabled=false; refreshBtn.disabled=false; }
    async function loadTags(){ try { renderTags(await (await fetch("/supermemory/tags")).json()); } catch(e) { tagsStatusEl.className="sm-status sm-err"; tagsStatusEl.textContent="Could not load tags."; } }
    saveBtn.addEventListener("click", async () => { const key=keyEl.value.trim(); if (!key) return; saveBtn.disabled=true; try { const j=await (await fetch("/supermemory/api-key",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({key})})).json(); if (j.ok){ keyEl.value=""; await refresh(); await loadTags(); } else { statusEl.className="sm-status sm-err"; statusEl.textContent=j.error||"Save failed."; } } finally { saveBtn.disabled=false; } });
    applyBtn.addEventListener("click", async () => { const cbs=tagListEl.querySelectorAll("input[type=checkbox]"); const excluded=[]; cbs.forEach(cb=>{ if (!cb.checked) excluded.push(cb.dataset.tag); }); applyBtn.disabled=true; try { const j=await (await fetch("/supermemory/tags",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({excluded})})).json(); tagsStatusEl.className="sm-status "+(j.ok?"sm-ok":"sm-err"); tagsStatusEl.textContent=j.ok?(excluded.length===0?"Saved — searching all tags.":"Saved — excluding "+excluded.length+"."):(j.error||"Save failed."); } catch(e) { tagsStatusEl.className="sm-status sm-err"; tagsStatusEl.textContent="Save failed."; } finally { applyBtn.disabled=false; } });
    refreshBtn.addEventListener("click", async () => { refreshBtn.disabled=true; tagsStatusEl.className="sm-status sm-info"; tagsStatusEl.textContent="Re-scanning…"; try { await fetch("/supermemory/tags/refresh",{method:"POST"}); await loadTags(); } catch(e) { tagsStatusEl.className="sm-status sm-err"; tagsStatusEl.textContent="Refresh failed."; } finally { refreshBtn.disabled=false; } });
    refresh(); loadTags();
  })();
  </script>
</section>
"""


def _compose_dashboard_index(app: Any, **types: Any) -> None:
    """Replace the dashboard's "/" page with upstream conversation_app's index
    plus our supermemory section, so the dashboard panel shows both at once.

    Without this the dashboard iframe only sees our standalone supermemory
    page — the upstream Headless control UI (backend selection, profile
    tweaks, etc.) is invisible while our app is the active one.
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

    @app.get("/")
    async def _combined() -> Any:  # type: ignore[misc]
        try:
            html = upstream_index.read_text(encoding="utf-8")
        except Exception:
            return HTMLResponse(_SUPERMEMORY_SECTION)
        # Upstream's HTML refs /static/* — rewrite to our /upstream-static/ mount.
        html = html.replace('href="/static/', 'href="/upstream-static/')
        html = html.replace('src="/static/', 'src="/upstream-static/')
        # Inject our supermemory section just before </body>.
        if "</body>" in html:
            html = html.replace("</body>", _SUPERMEMORY_SECTION + "</body>", 1)
        else:
            html = html + _SUPERMEMORY_SECTION
        return HTMLResponse(html)


def mount_supermemory_routes(app: object, instance_path: Optional[str] = None) -> None:
    """Mount /supermemory/ routes on the provided FastAPI settings app.

    The routes are best-effort: if FastAPI/pydantic isn't importable in this
    environment (e.g. unit tests with no daemon), the function silently no-ops.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from pydantic import BaseModel
    except Exception:  # pragma: no cover - settings app only present at runtime
        logger.debug("FastAPI not available; skipping supermemory settings routes.")
        return

    if not isinstance(app, FastAPI):
        logger.debug("Settings app is not a FastAPI instance; skipping supermemory routes.")
        return

    static_index = Path(__file__).parent / "static" / "index.html"
    _compose_dashboard_index(app, HTMLResponse=HTMLResponse, StaticFiles=StaticFiles, FileResponse=FileResponse)

    class ApiKeyPayload(BaseModel):
        key: str

    class ExcludesPayload(BaseModel):
        excluded: List[str]

    @app.get("/supermemory/")
    def _index() -> FileResponse:  # type: ignore[misc]
        return FileResponse(str(static_index))

    @app.get("/supermemory/status")
    def _status() -> JSONResponse:  # type: ignore[misc]
        return JSONResponse({"configured": is_configured()})

    @app.post("/supermemory/api-key")
    def _set_key(payload: ApiKeyPayload) -> JSONResponse:  # type: ignore[misc]
        value = (payload.key or "").strip()
        if not value:
            return JSONResponse({"ok": False, "error": "API key is empty."}, status_code=400)
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
