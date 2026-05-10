"""FastAPI routes for managing the supermemory API key from the headless UI.

Mirrors the env-write pattern used by ``reachy_mini_conversation_app.console``
without reaching into private API: the env-write logic is duplicated locally.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from ._supermemory_client import is_configured

logger = logging.getLogger(__name__)

_API_KEY_ENV = "SUPERMEMORY_API_KEY"


def _persist_to_env_file(env_path: Path, key: str, value: str) -> None:
    """Insert or replace ``KEY=value`` in ``env_path`` (creating the file if needed)."""
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


def mount_supermemory_routes(app: object, instance_path: Optional[str] = None) -> None:
    """Mount /supermemory/ routes on the provided FastAPI settings app.

    The routes are best-effort: if FastAPI/pydantic isn't importable in this
    environment (e.g. unit tests with no daemon), the function silently no-ops.
    """
    try:
        from fastapi import FastAPI
        from fastapi.responses import FileResponse, JSONResponse
        from pydantic import BaseModel
    except Exception:  # pragma: no cover - settings app only present at runtime
        logger.debug("FastAPI not available; skipping supermemory settings routes.")
        return

    if not isinstance(app, FastAPI):
        logger.debug("Settings app is not a FastAPI instance; skipping supermemory routes.")
        return

    static_index = Path(__file__).parent / "static" / "index.html"

    class ApiKeyPayload(BaseModel):
        key: str

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
