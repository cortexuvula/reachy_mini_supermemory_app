#!/usr/bin/env bash
# Publish the app to its Hugging Face Space.
#
# Reads HF_TOKEN from the project's .env (must have write access to the Space).
# Pass --skip-check to skip the slow pre-flight install round-trip.

set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -f .env ]; then
  echo "error: no .env found in $(pwd)" >&2
  exit 1
fi

# shellcheck disable=SC1091
set -a; . ./.env; set +a

if [ -z "${HF_TOKEN:-}" ]; then
  echo "error: HF_TOKEN not set in .env" >&2
  exit 1
fi

# Prefer the local venv if present.
PY="${PYTHON:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  PY="$(command -v python3)"
fi

SKIP_CHECK=0
COMMIT_MSG="${1:-Update app}"
if [ "${1:-}" = "--skip-check" ]; then
  SKIP_CHECK=1
  COMMIT_MSG="${2:-Update app}"
fi

if [ "$SKIP_CHECK" = "0" ]; then
  echo ">>> running reachy-mini-app-assistant check"
  "$PY" -m reachy_mini.apps.app check . || {
    echo "error: check failed; fix the issues or re-run with --skip-check" >&2
    exit 1
  }
fi

REPO_ID="${HF_SPACE_REPO:-RoamingH/reachy_mini_supermemory_app}"

echo ">>> uploading folder to $REPO_ID"
"$PY" - <<PY
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
result = api.upload_folder(
    folder_path=".",
    repo_id="$REPO_ID",
    repo_type="space",
    commit_message="""$COMMIT_MSG""",
    ignore_patterns=[
        ".git/**", ".github/**",
        ".venv/**", "**/__pycache__/**", "*.pyc",
        ".pytest_cache/**", ".ruff_cache/**", ".mypy_cache/**",
        "build/**", "dist/**", "*.egg-info/**",
        ".remember/**", ".env", "*.env",
    ],
)
print(f"OK -> {result}")
PY
