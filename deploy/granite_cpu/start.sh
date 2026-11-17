#!/usr/bin/env bash
# Serve the plain IBM Granite 3.1 1B-A400M BASE model (Q4_K_M GGUF) on CPU only (no VRAM), behind the producer shield.
# It is a base model: the chat template just passes your text through, so the model CONTINUES it. It does not follow instructions.
# Usage: GGUF=/path/base-q4km.gguf bash start.sh   (then run hivemind_producer_agent.py against the shield with its endpoint token)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
: "${GGUF:?set GGUF to the Q4_K_M file}"
LLAMA_SERVER="${LLAMA_SERVER:-llama-server}"
"$LLAMA_SERVER" -m "$GGUF" --host 127.0.0.1 --port "${UPSTREAM_PORT:-8391}" -c 2048 -ngl 0 -t "${THREADS:-6}" \
  --alias granite-1b-base --jinja --chat-template-file "$HERE/completion.jinja" &
exec python3 "$HERE/../../scripts/hivemind_producer_shield.py" --upstream "http://127.0.0.1:${UPSTREAM_PORT:-8391}" \
  --host 127.0.0.1 --port "${SHIELD_PORT:-8791}" --allow-model granite-1b-base
