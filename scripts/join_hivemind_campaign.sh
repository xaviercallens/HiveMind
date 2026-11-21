#!/usr/bin/env bash
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# ============================================================================
# Project HiveMind - Turnkey Citizen-Science Volunteer Onboarding (macOS / Linux)
# Runs real hardware/VRAM preflight -> Quick-Connect / Gamer-Whisper / OpenTrainer
# with strict free-VRAM reserve & thermal guard (auto-yield < 1.5s).
#
# PERSONAL HOME PC & NON-MONETARY NOTICE (0.00 EUR / 0.00 USD):
# Personal home machines and personal hobby projects ONLY (never work laptops).
# No financial payment, money, or cryptocurrency is ever offered or paid.
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GATEWAY_URL="${HIVEMIND_GATEWAY_URL:-https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app}"
RESERVE_MIB="${HIVEMIND_RESERVE_MIB:-1536}"
HOT_C="${HIVEMIND_HOT_C:-82}"
COOL_C="${HIVEMIND_COOL_C:-74}"
MODE="${HIVEMIND_JOIN_MODE:-opentrainer}" # connect | gamer-whisper | opentrainer | train | crowdgpt | legacy-demo
ONCE_FLAG="--once"
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            cat <<'EOF'
Usage: ./scripts/join_hivemind_campaign.sh [options]

Turnkey 1-command HiveMind Personal Onboarding (Personal Home PCs Only • 0.00 EUR):
  1. Quick 60s Developer Connect (`--connect`): gets/loads ~/.hivemind/credentials.json & VS Code config
  2. Quiet PC Gamer Mode (`--gamer-whisper`): <= 70 C cap, 2048 MiB free VRAM, < 1.5s stop on Steam game
  3. OpenTrainer / CrowdGPT GPU contributor (`--train` or `--crowdgpt`)

Options:
  --connect        Register/load personal API key in <60s and print VS Code (Continue.dev) snippet
  --gamer-whisper  Quiet home gaming PC mode (70 C cap, 2048 MiB reserve, instant stop on game launch)
  --train          Run a real OpenTrainer round (auto-loads ~/.hivemind/credentials.json if present)
  --crowdgpt       Contribute to Crowd-v1 via CrowdGPT client under HiveMind VRAM & thermal guards
  --continuous     Keep running heartbeats/training loop instead of single `--once` pass
  --legacy-demo    Run legacy local Season 9/10 simulation demo
  --reserve-mib N  Free VRAM MiB always reserved for your desktop/games (default: 1536, gamer-whisper: 2048)
  -h, --help       Show this help and exit
EOF
            exit 0
            ;;
        --connect) MODE="connect" ;;
        --gamer-whisper) MODE="gamer-whisper" ;;
        --train) MODE="train" ;;
        --crowdgpt) MODE="crowdgpt" ;;
        --continuous) ONCE_FLAG="" ;;
        --legacy-demo) MODE="legacy-demo" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

echo "============================================================================"
echo "  PROJECT HIVEMIND - TURNKEY PERSONAL ONBOARDING"
echo "  Gateway: ${GATEWAY_URL}"
echo "============================================================================"
echo "[RULE] Personal home PC & personal hobby code ONLY (never work laptops)."
echo "[LEGAL NOTICE] 0.00 EUR / 0.00 USD: No money is paid; points confer reciprocal compute."
echo ""

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [ -x "$ROOT_DIR/../../../.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/../../../.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if [ "$MODE" = "connect" ]; then
    exec "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_quickstart.py" \
        --gateway "$GATEWAY_URL" \
        --write-continue-config \
        "${EXTRA_ARGS[@]}"
fi

if [ "$MODE" = "gamer-whisper" ]; then
    WHISPER_RESERVE="${HIVEMIND_RESERVE_MIB:-2048}"
    WHISPER_HOT="${HIVEMIND_HOT_C:-70}"
    exec "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_quickstart.py" \
        --gateway "$GATEWAY_URL" \
        --gamer-whisper \
        --reserve-mib "$WHISPER_RESERVE" \
        --hot-c "$WHISPER_HOT" \
        ${ONCE_FLAG:+"$ONCE_FLAG"} \
        "${EXTRA_ARGS[@]}"
fi

if [ "$MODE" = "legacy-demo" ]; then
    HANDLE="${USER:-citizen-scientist}"
    mkdir -p "$ROOT_DIR/artifacts"
    PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_multi_moe_hf_campaign.py" \
        --contributor-handle "$HANDLE" --tier standard
    exit 0
fi

# Auto-load saved key from ~/.hivemind/credentials.json if HIVEMIND_API_KEY is not exported
if [ -z "${HIVEMIND_API_KEY:-}" ] && [ -f "$HOME/.hivemind/credentials.json" ]; then
    SAVED_KEY="$("$PYTHON_BIN" -c 'import json, os; p=os.path.expanduser("~/.hivemind/credentials.json"); d=json.load(open(p)); print(d.get("api_key",""))' 2>/dev/null || true)"
    if [[ "$SAVED_KEY" == hm-* ]]; then
        export HIVEMIND_API_KEY="$SAVED_KEY"
    fi
fi

echo "[Step 1/2] Running volunteer hardware & VRAM preflight..."
set +e
PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_volunteer_preflight.py" \
    --project-url "$GATEWAY_URL"
PREFLIGHT_RC=$?
set -e
if [ "$PREFLIGHT_RC" -eq 2 ]; then
    echo "Preflight reported not_ready (exit 2). Fix the FAIL items above before joining." >&2
    exit 2
fi

echo ""
echo "[Step 2/2] Running OpenTrainer volunteer check / contributor (${MODE})..."
if [ "$MODE" = "crowdgpt" ]; then
    exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_opentrainer_agent.py" \
        --gateway "$GATEWAY_URL" \
        --crowdgpt \
        --reserve-mib "$RESERVE_MIB" \
        --hot-c "$HOT_C" \
        --cool-c "$COOL_C" \
        ${ONCE_FLAG:+"$ONCE_FLAG"} \
        "${EXTRA_ARGS[@]}"
fi

if [ "$MODE" = "train" ]; then
    if [ -z "${HIVEMIND_API_KEY:-}" ]; then
        "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_quickstart.py" --gateway "$GATEWAY_URL" --json >/dev/null
        SAVED_KEY="$("$PYTHON_BIN" -c 'import json, os; p=os.path.expanduser("~/.hivemind/credentials.json"); d=json.load(open(p)); print(d.get("api_key",""))' 2>/dev/null || true)"
        export HIVEMIND_API_KEY="$SAVED_KEY"
    fi
    exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_opentrainer_agent.py" \
        --gateway "$GATEWAY_URL" \
        --max-vram-gb 5.5 \
        --train \
        ${ONCE_FLAG:+"$ONCE_FLAG"} \
        "${EXTRA_ARGS[@]}"
fi

if [ -n "${HIVEMIND_API_KEY:-}" ]; then
    exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_opentrainer_agent.py" \
        --gateway "$GATEWAY_URL" \
        --check-firewall \
        ${ONCE_FLAG:+"$ONCE_FLAG"} \
        "${EXTRA_ARGS[@]}"
fi

echo "No HIVEMIND_API_KEY set; running outbound firewall traversal check only."
echo "Run ./scripts/join_hivemind_campaign.sh --connect to register your personal key in <60s."
exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_opentrainer_agent.py" \
    --gateway "$GATEWAY_URL" \
    --check-firewall \
    --once \
    "${EXTRA_ARGS[@]}"
