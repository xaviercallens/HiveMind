#!/bin/bash
# Build the three artefacts a volunteer's stock BOINC client downloads for the HiveMind QLoRA-MoE apps. Reads from this host, writes only to OUT.
#   hivemind_runtime_<sha12>.tar.xz  relocatable standalone CPython 3.10 + the pinned libraries (deploy/boinc_dist/runtime-requirements.txt)
#   hivemind_src_<sha12>.tar.xz      the hivemind python package (small; changes with every code update)
#   hivemind_model_<name>_<sha12>.tar  the base model in Hugging Face cache layout (safetensors do not compress)
# Names carry the content hash because BOINC files are immutable by name: an unchanged runtime/model is not downloaded again after a code-only update.
#   bash scripts/boinc_build_distribution.sh [OUT_DIR] [MODEL_DIR_NAME]      (default model: models--ibm-granite--granite-3.1-1b-a400m-base)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/mnt/disks/disk-socrateai-local-1/hivemind-dist}"
MODEL="${2:-models--ibm-granite--granite-3.1-1b-a400m-base}"
HF="$HOME/.cache/huggingface/hub"
mkdir -p "$OUT/build"

# 1. runtime -------------------------------------------------------------------------------------------------------------------------------------------
PY="$(uv python find 3.10)"
if [ ! -x "$OUT/build/runtime/bin/python3.10" ]; then
  rsync -a "$(dirname "$(dirname "$(readlink -f "$PY")")")/" "$OUT/build/runtime/"
  rm -f "$OUT/build/runtime/lib/python3.10/EXTERNALLY-MANAGED"
fi
uv pip install --python "$OUT/build/runtime/bin/python3.10" -r "$REPO/deploy/boinc_dist/runtime-requirements.txt" \
  --index-url https://download.pytorch.org/whl/cu118 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
find "$OUT/build/runtime" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
rm -rf "$OUT/build/runtime/lib/python3.10/site-packages/torch/test" "$OUT/build/runtime/lib/python3.10/site-packages/torch/include" 2>/dev/null || true

# 2. source --------------------------------------------------------------------------------------------------------------------------------------------
rm -rf "$OUT/build/src" && mkdir -p "$OUT/build/src"
git -C "$REPO" archive HEAD src/hivemind | tar -x -C "$OUT/build/src" --strip-components=1   # committed content only, like the node installer

# 3. pack ----------------------------------------------------------------------------------------------------------------------------------------------
pack() {  # pack <tar flags/dir...> -> name with content hash
  local kind="$1"; shift; local tmp="$OUT/$kind.tmp"
  tar -C "$1" -cf - "${@:2}" | nice -n 19 xz -T"${XZ_THREADS:-2}" -6 > "$tmp"   # few threads: this runs on the inference node
  local h; h="$(sha256sum "$tmp" | cut -c1-12)"; mv "$tmp" "$OUT/hivemind_${kind}_${h}.tar.xz"; echo "hivemind_${kind}_${h}.tar.xz"
}
if [ "${ONLY:-}" != "model" ]; then   # ONLY=model rebuilds just the model archive (the runtime pack takes ~25 min)
  pack runtime "$OUT/build/runtime" .
  pack src "$OUT/build/src" .
fi
# The cached weights are symlinks into a shared blob store: stage a self-contained cache directory (real files, refs/main) and archive that.
stage="$OUT/build/model"; rm -rf "$stage"; mkdir -p "$stage/$MODEL/snapshots" "$stage/$MODEL/refs"
cp -rL "$HF/$MODEL/snapshots/." "$stage/$MODEL/snapshots/"
cp -L "$HF/$MODEL/refs/"* "$stage/$MODEL/refs/"
tmp="$OUT/model.tmp"; tar -C "$stage" -cf "$tmp" "$MODEL"; h="$(sha256sum "$tmp" | cut -c1-12)"
mv "$tmp" "$OUT/hivemind_model_${MODEL#models--}_${h}.tar"; echo "hivemind_model_${MODEL#models--}_${h}.tar"
ls -la "$OUT" | grep hivemind_
