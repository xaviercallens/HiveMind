#!/usr/bin/env bash
# ==============================================================================
# HiveMind: Dyson & Exoplanet Odyssey — Local RTX 2070 (8 GB) + La Ruche Launcher
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PORT="${PORT:-8790}"
HOST="${HOST:-127.0.0.1}"

cd "${REPO_ROOT}"
echo "=============================================================================="
echo "  HIVEMIND: DYSON & EXOPLANET ODYSSEY (STITCH STEAM UI + UNITY + ANTIGRAVITY)"
echo "  Target Hardware: NVIDIA GeForce RTX 2070 (8 GB VRAM, <1.5s WDDM Preemption)"
echo "  La Ruche Cloud:  https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"
echo "  Local Game HUD:  http://${HOST}:${PORT}/scientific-game"
echo "=============================================================================="

if command -v uv >/dev/null 2>&1; then
  PY_CMD=(uv run python)
elif [ -x "${REPO_ROOT}/.venv/bin/python" ]; then
  PY_CMD=("${REPO_ROOT}/.venv/bin/python")
else
  PY_CMD=(python3)
fi

PYTHONPATH=src "${PY_CMD[@]}" -m hivemind.game.scientific_discovery_game --sync-la-ruche || true
exec env PYTHONPATH=src "${PY_CMD[@]}" -m hivemind.game.scientific_discovery_game --serve --host "${HOST}" --port "${PORT}"
