#!/bin/bash
# Build a portable bundle of the Bee expert (base GGUF + stars LoRA + serve script) to copy to the RTX 2070 (8 GB) machine. Run on the T4 node:
#   bash deploy/rtx2070/make_bee_bundle.sh [OUT_DIR]
# It only reads local files and writes OUT_DIR (default: <data disk>/hivemind-bee-bundle); nothing is sent anywhere. Copy the directory yourself (scp/rsync).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DISK=/mnt/disks/disk-socrateai-local-1
BASE="$DISK/leanmaster/ollama/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463"   # qwen2.5-coder:7b-instruct Q4 (4.68 GB), same as hivemind-live-bee-llama.service
LORA="$DISK/hivemind-train/adapters/stars.gguf"                                                                  # the Bee "stars" LoRA (161 MB)
OUT="${1:-$DISK/hivemind-bee-bundle}"
for f in "$BASE" "$LORA"; do [ -r "$f" ] || { echo "missing or unreadable: $f" >&2; exit 1; }; done
mkdir -p "$OUT"
cp --reflink=auto "$BASE" "$OUT/bee-base-qwen2.5-coder-7b-instruct-q4.gguf"
cp --reflink=auto "$LORA" "$OUT/stars.gguf"
cp "$HERE/serve_bee.sh" "$OUT/serve_bee.sh"
( cd "$OUT" && sha256sum *.gguf > SHA256SUMS && chmod 0755 serve_bee.sh )
echo "bundle ready: $OUT"; du -sh "$OUT"; cat "$OUT/SHA256SUMS"
