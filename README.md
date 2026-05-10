# reachy_mini_supermemory_app

Reachy Mini conversation app with long-term memory backed by [supermemory.ai](https://supermemory.ai).

Extends [`reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app) with two LLM-callable tools:

- `save_memory(content, kind?)` — write a durable fact to supermemory.
- `recall_memory(query, limit?)` — search prior memories.

Both are tool calls only. The model decides when to use them based on `instructions.txt`. There is no auto-save and no auto-recall.

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

Or launch and visit `http://<robot>:8000/supermemory/` to paste the key into the settings UI.

## Run

```bash
reachy-mini-supermemory-app                    # console mode
reachy-mini-supermemory-app --gradio           # web UI on http://127.0.0.1:7860/
```

The app sets `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` to its bundled `profiles/` and defaults the active profile to `supermemory`. All other CLI flags from `reachy_mini_conversation_app` (e.g. `--head-tracker`, `--no-camera`, `--debug`) work unchanged.

## How memories are scoped

Memories are stored under a `containerTag` of `reachy-mini:<profile>`, so switching profiles in the personality UI isolates memories per persona. Memories saved while the `supermemory` profile is active live under `reachy-mini:supermemory`.

## Tests

```bash
uv pip install --group dev
uv run pytest tests/ -v
```

## License

Apache-2.0.
