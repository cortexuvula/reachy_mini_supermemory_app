# reachy_mini_supermemory_app — plan

## What this is

A Reachy Mini app that extends `reachy_mini_conversation_app` with two LLM-driven tools:

- `save_memory(content, kind?)` → `POST https://api.supermemory.ai/v4/memories`
- `recall_memory(query, limit?)` → `POST https://api.supermemory.ai/v4/search`

Save and recall happen only when the LLM decides it needs them — there is no auto-save on every turn and no auto-recall before every response. The profile's `instructions.txt` is what teaches the model when each is appropriate.

## Layout

```
reachy_mini_supermemory_app/
├── plan.md                                     # this file
├── pyproject.toml                              # entry point + deps
├── README.md
├── .env.example                                # SUPERMEMORY_API_KEY
├── .gitignore
├── LICENSE                                     # Apache-2.0 (matches parent)
├── src/reachy_mini_supermemory_app/
│   ├── __init__.py
│   ├── main.py                                 # ReachyMiniSupermemoryApp + CLI
│   ├── _supermemory_client.py                  # httpx wrapper, container tag, error mapping
│   ├── settings_ui.py                          # /supermemory/* FastAPI routes
│   └── static/index.html                       # password input for the API key
├── profiles/supermemory/
│   ├── instructions.txt                        # teaches save/recall discipline
│   ├── tools.txt                               # default tools + save_memory + recall_memory
│   ├── save_memory.py                          # Tool subclass
│   └── recall_memory.py                        # Tool subclass
└── tests/
    ├── conftest.py
    ├── test_supermemory_client.py
    ├── test_save_memory.py
    └── test_recall_memory.py
```

## How it plugs in

`ReachyMiniSupermemoryApp(ReachyMiniConversationApp)` (in `main.py`):

1. Sets `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` to point at our `profiles/` directory **before** any `reachy_mini_conversation_app.config` import happens. Verified safe: `reachy_mini_conversation_app.main` only imports `config` lazily from inside `run()`.
2. Defaults `REACHY_MINI_CUSTOM_PROFILE` to `supermemory` (does not override an existing setting — user can still pick another profile via the personality UI).
3. Calls `super().run(reachy_mini, stop_event)` — the conversation app's full handler/movement/audio stack runs unchanged.
4. Mounts our settings routes onto `self.settings_app` (provided by the Reachy Mini Apps runtime) so the API key can be entered from a browser without touching `.env`.

The package registers itself in `pyproject.toml` under `[project.entry-points."reachy_mini_apps"]` so the daemon discovers it as a separate app alongside the conversation app.

A console script `reachy-mini-supermemory-app` is also provided for users who want to launch via CLI without the daemon — it sets the env vars and delegates to `reachy_mini_conversation_app.main:main`.

## Tools

Both subclass `reachy_mini_conversation_app.tools.core_tools.Tool`. Loaded as profile-local tools via the profile-first lookup in `core_tools._load_profile_tools`.

### `save_memory`

```python
parameters_schema = {
    "type": "object",
    "properties": {
        "content": {"type": "string", "description": "Durable fact worth remembering across sessions."},
        "kind": {"type": "string", "description": "Optional category: preference, identity, decision, fact, ..."},
    },
    "required": ["content"],
}
```

Calls `POST /v4/memories` with body:
```json
{
  "memories": [{"content": "<content>", "metadata": {"kind": "<kind?>"}}],
  "containerTag": "reachy-mini:<profile>"
}
```
Returns `{"saved": true, "memory_id": "..."}` on success, `{"error": "..."}` otherwise.

### `recall_memory`

```python
parameters_schema = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "What to look up in long-term memory."},
        "limit": {"type": "integer", "description": "Max matches to return (default 5)."},
    },
    "required": ["query"],
}
```

Calls `POST /v4/search` with body:
```json
{"query": "<query>", "containerTag": "reachy-mini:<profile>", "threshold": 0.6, "limit": <limit>}
```
Returns `{"matches": [{"memory": "...", "score": 0.81}, ...]}`, `{"matches": []}` if nothing crosses the threshold, or `{"error": "..."}` on failure.

## Shared client (`_supermemory_client.py`)

- Single async helper `post_json(path, body) -> dict` using `httpx.AsyncClient` with `timeout=10.0`.
- Reads `SUPERMEMORY_API_KEY` and `SUPERMEMORY_BASE_URL` (default `https://api.supermemory.ai`) from `os.environ` **at call time**, not import time, so settings-UI updates take effect immediately.
- `derive_container_tag()` reads `config.REACHY_MINI_CUSTOM_PROFILE` at call time and returns `f"reachy-mini:{profile or 'default'}"`, sanitized to match `^[a-zA-Z0-9_:-]+$`.
- Error mapping: missing API key → friendly message; HTTP 4xx/5xx → `{"error": "<status>: <body excerpt>"}`; network timeout → `{"error": "Supermemory request timed out."}`. The LLM hears these strings as tool output and can speak gracefully.

## API key plumbing

- `.env.example` documents `SUPERMEMORY_API_KEY` (and optional `SUPERMEMORY_BASE_URL` for proxies).
- `settings_ui.py` mounts:
  - `GET /supermemory/status` → `{"configured": bool}` (does not return the key).
  - `POST /supermemory/api-key` body `{"key": "<value>"}` → writes to `os.environ["SUPERMEMORY_API_KEY"]` and persists to `<instance_path>/.env`. Mirrors the existing `LocalStream._persist_env_value` pattern; the env-write logic is duplicated locally rather than reaching into private API.
  - `GET /supermemory/` → serves `static/index.html`, a single password input + save button.
- If the key is missing when a tool is called, both tools return `{"error": "Supermemory API key not configured. Set SUPERMEMORY_API_KEY in .env or visit /supermemory/."}`. The model speaks that, the user knows what to do.

## ContainerTag

Per profile, derived as `f"reachy-mini:{config.REACHY_MINI_CUSTOM_PROFILE or 'default'}"` and read fresh on every call. Live profile switches in the personality UI immediately reroute writes/reads — switching to `mars_rover` and back to `supermemory` keeps memories isolated. Sanitized to drop characters outside `[a-zA-Z0-9_:-]`.

## `instructions.txt` (the prompt that drives discipline)

Tells the model:
- You have a long-term memory store across sessions.
- **Save** when the user shares something durable (preferences, names, decisions, recurring people/places). Don't save small talk or transient facts.
- **Recall** only when the user references prior context ("what did I tell you about…", "do you remember…", or a name/topic surfaces that isn't in the current session). Not on every turn.
- If recall returns nothing, say so honestly instead of guessing.

## Tests

`pytest` + `pytest-asyncio`. Mock `httpx.AsyncClient.post` via `respx` or a small fake. Cover:

- `save_memory` happy path: correct URL, headers (`Authorization: Bearer ...`), body shape, containerTag derivation.
- `save_memory` returns `{"error": ...}` when API key is missing.
- `save_memory` maps HTTP 401/500 to `{"error": ...}`.
- `recall_memory` happy path: parses `matches`, filters/passes through scores.
- `recall_memory` empty body and nothing-crosses-threshold case.
- Container tag sanitization: profile names with spaces/punct become safe tags.
- `derive_container_tag()` reads live config (changing `config.REACHY_MINI_CUSTOM_PROFILE` mid-test changes the tag).

`tests/conftest.py` sets `REACHY_MINI_SKIP_DOTENV=1` (matches parent project's pattern) so no developer `.env` leaks into tests.

## Out of scope

- No changes to `reachy_mini_conversation_app` itself.
- No HF Space publish.
- No automatic save/recall hooks — tools-only, by design.
- No memory listing/deletion UI — supermemory.ai's dashboard handles that.

## Risks / known unknowns

- Supermemory API response shape for `/v4/search` is documented partially; the `matches` structure is inferred. Will adjust the parser after a real first call if fields differ.
- The Reachy Mini Apps daemon may serve static files under a path prefix specific to each app — if `GET /supermemory/` doesn't reach our routes, the fallback is to instruct users to set `SUPERMEMORY_API_KEY` in `.env` and the tools work fine without the UI.
