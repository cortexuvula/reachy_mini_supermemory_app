"""Local-JSON inline memory store.

Mirrors the hermes "inline memory" pattern: a small curated bullet list of
durable facts about the user, injected into Reachy's system prompt at session
start so the user never has to re-explain them. Distinct from supermemory's
semantic store (managed by save_memory/recall_memory) — this one is
always-loaded, hard-capped, and edited by the LLM via the manage_memory tool.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

INLINE_MEMORY_FILE_ENV = "REACHY_MINI_INLINE_MEMORY_FILE"
INLINE_MEMORY_CHAR_LIMIT_ENV = "REACHY_MINI_INLINE_MEMORY_CHAR_LIMIT"
DEFAULT_CHAR_LIMIT = 3000
DEFAULT_FILE_NAME = "inline-memory.json"

_io_lock = Lock()


def inline_memory_file() -> Path:
    """Resolve the on-disk path for the inline memory store."""
    override = (os.environ.get(INLINE_MEMORY_FILE_ENV) or "").strip()
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_DATA_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return base / "reachy_mini_supermemory_app" / DEFAULT_FILE_NAME


def char_limit() -> int:
    raw = os.environ.get(INLINE_MEMORY_CHAR_LIMIT_ENV)
    if raw:
        try:
            return max(100, int(raw))
        except ValueError:
            pass
    return DEFAULT_CHAR_LIMIT


def _load_unsafe(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Inline memory file malformed at %s: %s — treating as empty", path, e)
        return {"entries": []}
    entries = data.get("entries")
    if not isinstance(entries, list):
        return {"entries": []}
    return {"entries": [e.strip() for e in entries if isinstance(e, str) and e.strip()]}


def _save_unsafe(path: Path, entries: List[str]) -> None:
    payload = {
        "entries": entries,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def load_entries() -> List[str]:
    with _io_lock:
        return list(_load_unsafe(inline_memory_file())["entries"])


def total_chars(entries: List[str]) -> int:
    return sum(len(e) for e in entries)


def render_block() -> str:
    """Render the inline memory as a labeled bullet block, or empty when nothing stored.

    Enforces ``char_limit()`` at render time so a manually-edited JSON or a
    post-hoc-lowered limit can't blow up the system prompt. Writes already
    refuse to exceed the cap, so this only fires defensively.
    """
    entries = load_entries()
    if not entries:
        return ""
    header = "=== Things Reachy already knows about this user (always remember) ==="
    footer = "=== end of memory ==="
    limit = char_limit()
    lines = [header]
    used = len(header) + len(footer) + 1  # +1 for the newline between header and footer
    kept = 0
    for entry in entries:
        line = f"- {entry}"
        # +1 accounts for the newline we'll add when joining.
        if used + len(line) + 1 > limit:
            break
        lines.append(line)
        used += len(line) + 1
        kept += 1
    omitted = len(entries) - kept
    if omitted > 0:
        suffix = "y" if omitted == 1 else "ies"
        lines.append(f"- … ({omitted} more entr{suffix} omitted; over {limit}-char inline memory limit)")
    lines.append(footer)
    return "\n".join(lines)


def add_entry(content: str) -> Dict[str, Any]:
    content = (content or "").strip()
    if not content:
        return {"error": "content is required"}
    with _io_lock:
        path = inline_memory_file()
        entries = list(_load_unsafe(path)["entries"]) + [content]
        used = total_chars(entries)
        limit = char_limit()
        if used > limit:
            return {
                "error": (
                    f"Adding this entry would put inline memory at {used}/{limit} chars. "
                    "Remove or replace older entries first."
                )
            }
        _save_unsafe(path, entries)
        return {"ok": True, "entries": entries, "chars_used": used, "char_limit": limit}


def replace_entry(old_text: str, content: str) -> Dict[str, Any]:
    """Replace a single entry whose text contains ``old_text``.

    Errors if ``old_text`` matches more than one entry — previously, an
    ambiguous match silently rewrote both to the same content (the LLM would
    accidentally collapse "Daughter Mia, 7" and "Daughter Mira, 5" into the
    same line if it used "Daughter" as old_text). The error response includes
    the matching entries so the model can retry with a more specific
    substring.
    """
    old_text = (old_text or "").strip()
    content = (content or "").strip()
    if not old_text:
        return {"error": "old_text is required for replace"}
    if not content:
        return {"error": "content is required for replace"}
    with _io_lock:
        path = inline_memory_file()
        entries = list(_load_unsafe(path)["entries"])
        matches = [i for i, e in enumerate(entries) if old_text in e]
        if not matches:
            return {"error": f"no entry contains {old_text!r}"}
        if len(matches) > 1:
            return {
                "error": (
                    f"{len(matches)} entries contain {old_text!r}; refine old_text "
                    "to uniquely identify the one to replace."
                ),
                "matches": [entries[i] for i in matches],
            }
        entries[matches[0]] = content
        used = total_chars(entries)
        limit = char_limit()
        if used > limit:
            return {
                "error": (
                    f"Replacement would put inline memory at {used}/{limit} chars. "
                    "Shorten the new content or remove other entries first."
                )
            }
        _save_unsafe(path, entries)
        return {
            "ok": True,
            "replaced": 1,
            "entries": entries,
            "chars_used": used,
            "char_limit": limit,
        }


def remove_entry(old_text: str) -> Dict[str, Any]:
    old_text = (old_text or "").strip()
    if not old_text:
        return {"error": "old_text is required for remove"}
    with _io_lock:
        path = inline_memory_file()
        before = list(_load_unsafe(path)["entries"])
        after = [e for e in before if old_text not in e]
        removed = len(before) - len(after)
        if removed == 0:
            return {"error": f"no entry contains {old_text!r}"}
        _save_unsafe(path, after)
        return {
            "ok": True,
            "removed": removed,
            "entries": after,
            "chars_used": total_chars(after),
            "char_limit": char_limit(),
        }
