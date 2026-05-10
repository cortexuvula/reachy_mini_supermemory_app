# reachy_mini_supermemory_app

Reachy Mini conversation app with long-term memory backed by [supermemory.ai](https://supermemory.ai).

Extends [`reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app) with two complementary memory layers:

- **Inline memory** — a small always-loaded bullet list of curated facts (under ~3000 chars) injected into Reachy's system prompt at every session start, so the user never has to re-explain things like their name or top preferences. Stored as local JSON. Managed by the `manage_memory(action, content?, old_text?)` tool with actions `add` / `replace` / `remove` / `list`.
- **Long-term memory** (supermemory.ai) — a larger searchable store accessed via two tools: `save_memory(content, kind?)` writes a durable fact, `recall_memory(query, limit?)` semantic-searches prior memories across every tag the API key can reach.

All four tools are LLM-driven; the model decides when to use them based on `instructions.txt`. There is no auto-save and no auto-recall.

## Install

```bash
uv venv --python python3.12 .venv
source .venv/bin/activate
uv pip install -e .
```

`reachy_mini_conversation_app` must be importable in the same environment. If you have it checked out as a sibling directory:

```bash
uv pip install -e ../reachy_mini_conversation_app
uv pip install -e .
```

## Configure

```bash
cp .env.example .env
# edit .env and add SUPERMEMORY_API_KEY=...
```

Or launch and visit `http://127.0.0.1:7861/supermemory/` (CLI launches) or `http://<robot>:8000/supermemory/` (daemon-registered launches) to paste the key into the settings UI.

## Run

```bash
reachy-mini-supermemory-app                    # console mode
reachy-mini-supermemory-app --gradio           # web UI on http://127.0.0.1:7860/
```

The app sets `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` to its bundled `profiles/` and defaults the active profile to `supermemory`. All other CLI flags from `reachy_mini_conversation_app` (e.g. `--head-tracker`, `--no-camera`, `--debug`) work unchanged. CLI launches additionally serve the supermemory settings UI on a side port (default `7861`, override with `SUPERMEMORY_SETTINGS_PORT`).

## How memories are scoped

Saves go under a `containerTag` of `reachy-mini:<profile>`, so switching profiles in the personality UI isolates writes per persona. Memories saved while the `supermemory` profile is active live under `reachy-mini:supermemory`.

Recall auto-discovers all containerTags in your supermemory project (via `GET /v3/container-tags/list`, cached for ten minutes) so the bot can surface memories written by other agents on the same account. The `/supermemory/` settings page lists every discovered tag with a checkbox; uncheck any you don't want recall to read. Set `SUPERMEMORY_RECALL_CONTAINER_TAGS` (comma-separated) to hard-pin recall to a specific scope and disable the UI controls.

## Tests

```bash
uv pip install --group dev
uv run pytest tests/ -v
```

## License

Apache-2.0.
