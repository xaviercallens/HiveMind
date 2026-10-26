#!/bin/bash
# Serve the Bee expert (qwen2.5-coder-7b-instruct Q4 + stars LoRA) on the RTX 2070 (8 GB) with llama.cpp. Run inside the copied bundle directory:
#   LLAMA_SERVER=/path/to/llama-server ./serve_bee.sh [PORT]
# Needs a CUDA build of llama.cpp (llama-server). ~5 GB VRAM at -c 4096, so it fits 8 GB with room for the display. Listens on loopback only; put your own
# TLS/proxy in front if it should be reachable, and register it with the gateway with hivemind_producer_agent.py (alias hivemind-bee-stars, key from your own file).
set -euo pipefail
cd "$(dirname "$0")"
sha256sum -c SHA256SUMS >/dev/null || { echo "bundle checksum mismatch: re-copy the files" >&2; exit 1; }
SERVER="${LLAMA_SERVER:-$(command -v llama-server || true)}"
[ -x "$SERVER" ] || { echo "set LLAMA_SERVER to a CUDA llama-server binary" >&2; exit 1; }
PORT="${1:-8294}"
exec "$SERVER" -m bee-base-qwen2.5-coder-7b-instruct-q4.gguf --lora stars.gguf --alias hivemind-bee-stars \
  -c 4096 -ngl 99 -np 1 --host 127.0.0.1 --port "$PORT"
