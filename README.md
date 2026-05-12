---
title: Reachy Mini Supermemory
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: static
pinned: false
short_description: Reachy Mini voice companion with persistent memory
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# reachy_mini_supermemory_app

Reachy Mini conversation app with long-term memory backed by [supermemory.ai](https://supermemory.ai).

Extends [`reachy_mini_conversation_app`](https://github.com/pollen-robotics/reachy_mini_conversation_app) with two complementary memory layers:

- **Inline memory** — a small always-loaded bullet list of curated facts (under ~3000 chars) injected into Reachy's system prompt at every session start, so the user never has to re-explain things like their name or top preferences. Stored as local JSON. Managed by the `manage_memory(action, content?, old_text?)` tool with actions `add` / `replace` / `remove` / `list`.
- **Long-term memory** (supermemory.ai) — a larger searchable store accessed via two tools: `save_memory(content, kind?)` writes a durable fact, `recall_memory(query, limit?)` semantic-searches prior memories across every tag the API key can reach.
- **Auto-digest (opt-in)** — when `SUPERMEMORY_AUTO_DIGEST=true`, idle conversations get summarised by a Hugging Face chat model and written to supermemory as a single entry. Captures nuance the LLM-driven save/manage tools miss. See the configuration reference for tuning.

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

The app sets `REACHY_MINI_EXTERNAL_PROFILES_DIRECTORY` to its bundled `profiles/` and defaults the active profile to `supermemory`. All other CLI flags from `reachy_mini_conversation_app` (e.g. `--head-tracker`, `--no-camera`, `--debug`) work unchanged. On startup the app also POSTs `/api/move/play/wake_up` to the Reachy Mini daemon so the head lifts off the base before the first turn (no manual wake needed).

CLI launches additionally serve the supermemory settings UI on a side port (default `127.0.0.1:7861`). When running on the robot, set `SUPERMEMORY_SETTINGS_HOST=0.0.0.0` so a LAN client can reach `http://<robot>:7861/supermemory/` (or front it with Caddy / nginx for HTTPS so the browser will grant mic/cam access).

## How memories are scoped

Saves go under a `containerTag` of `reachy-mini:<profile>`, so switching profiles in the personality UI isolates writes per persona. Memories saved while the `supermemory` profile is active live under `reachy-mini:supermemory`.

Recall auto-discovers all containerTags in your supermemory project (via `GET /v3/container-tags/list`, cached for ten minutes) so the bot can surface memories written by other agents on the same account. The `/supermemory/` settings page lists every discovered tag with a checkbox; uncheck any you don't want recall to read. Set `SUPERMEMORY_RECALL_CONTAINER_TAGS` (comma-separated) to hard-pin recall to a specific scope and disable the UI controls.

## Configuration reference

All env vars below can go in `.env` (see `.env.example`) or be exported by your service unit. Bold means required.

| Variable | Default | Purpose |
| --- | --- | --- |
| **`SUPERMEMORY_API_KEY`** | *(required)* | Bearer token for supermemory.ai. Can also be saved via `/supermemory/` UI. |
| `SUPERMEMORY_BASE_URL` | `https://api.supermemory.ai` | API base override (e.g. for a proxy). |
| `SUPERMEMORY_RECALL_CONTAINER_TAGS` | *auto-discover* | Comma-separated tags to pin recall scope. When set, the UI tag controls become read-only. |
| `SUPERMEMORY_RECALL_EXCLUDED_TAGS` | *(none)* | Tags to exclude from recall (managed by the `/supermemory/` UI). Ignored when the pin list above is set. |
| `SUPERMEMORY_SETTINGS_HOST` | `127.0.0.1` | Side-port bind for the `/supermemory/` UI. Set `0.0.0.0` to expose on the LAN. |
| `SUPERMEMORY_SETTINGS_PORT` | `7861` | Side-port port. |
| `SUPERMEMORY_VAD_THRESHOLD` | `0.7` | Realtime VAD threshold (0–1). Higher = less sensitive (mic less likely to trip on speaker bleed). |
| `SUPERMEMORY_VAD_SILENCE_MS` | `700` | Silence duration (ms) before end-of-turn is committed. Higher = more tolerant of pauses. |
| `SUPERMEMORY_VAD_PREFIX_PADDING_MS` | `400` | Audio (ms) included before speech onset is detected. |
| `REACHY_MINI_INLINE_MEMORY_FILE` | `$XDG_DATA_HOME/reachy_mini_supermemory_app/inline-memory.json` | Path to the always-loaded inline memory JSON. |
| `REACHY_MINI_INLINE_MEMORY_CHAR_LIMIT` | `3000` | Hard cap on total inline-memory characters. Min 100. |
| `REACHY_MINI_DAEMON_API_BASE` | `http://127.0.0.1:8000` | Reachy Mini daemon REST base, used by the auto-wake call at startup. |
| `SUPERMEMORY_AUTO_DIGEST` | `false` | Opt-in. When `true`, conversation turns are buffered in memory, and after idle a Hugging Face chat model produces a one-paragraph digest written to supermemory as a single entry. Requires `HF_TOKEN`. |
| `SUPERMEMORY_DIGEST_IDLE_MINUTES` | `10` | Minutes of conversation silence before a digest is triggered. |
| `SUPERMEMORY_DIGEST_MIN_TURNS` | `4` | Minimum buffered turns needed before a digest is attempted (skips trivially short exchanges). |
| `SUPERMEMORY_DIGEST_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | HF model used for the digest summarisation. |
| `SUPERMEMORY_DIGEST_API_URL` | `https://router.huggingface.co/v1/chat/completions` | Chat-completions endpoint for the digest call. |

Upstream `reachy_mini_conversation_app` env vars (e.g. `BACKEND_PROVIDER`, `OPENAI_API_KEY`, `GEMINI_API_KEY`, `HF_REALTIME_CONNECTION_MODE`, `GRADIO_SERVER_NAME`) work unchanged — see the upstream `.env.example` for the full list.

## Tests

```bash
uv pip install --group dev
uv run pytest tests/ -v
```

## Publish

After committing changes, re-publish to the Hugging Face Space:

```bash
./scripts/publish.sh "your commit message"
# or skip the slow pre-flight check:
./scripts/publish.sh --skip-check "your commit message"
```

The script reads `HF_TOKEN` from `.env` (use a fine-grained write token scoped to the Space repo only). `HF_SPACE_REPO` defaults to `RoamingH/reachy_mini_supermemory_app`; set in `.env` to override.

## License

Apache-2.0.
