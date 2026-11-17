#!/usr/bin/env bash
# ============================================================================
# Project HiveMind - Turnkey Citizen-Science Volunteer Onboarding (macOS / Linux)
# Runs real hardware/VRAM preflight -> OpenTrainer / CrowdGPT GPU contributor
# with strict free-VRAM reserve & thermal guard (auto-yield < 1.5s).
#
# NON-MONETARY NOTICE (0.00 EUR / 0.00 USD):
# No financial payment, money, or cryptocurrency is ever offered or paid.
# Symbolic points (PTS / HMT) grant exclusively reciprocal computing benefits
# on the HiveMind OpenRouter network.
# ============================================================================
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

GATEWAY_URL="${HIVEMIND_GATEWAY_URL:-https://hivemind-openrouter-1003063861791.us-central1.run.app}"
RESERVE_MIB="${HIVEMIND_RESERVE_MIB:-1536}"
HOT_C="${HIVEMIND_HOT_C:-82}"
COOL_C="${HIVEMIND_COOL_C:-74}"
MODE="${HIVEMIND_JOIN_MODE:-opentrainer}" # opentrainer | crowdgpt | legacy-demo
ONCE_FLAG="--once"
EXTRA_ARGS=()

for arg in "$@"; do
    case "$arg" in
        --help|-h)
            cat <<'EOF'
Usage: ./scripts/join_hivemind_campaign.sh [options]

Turnkey 1-command HiveMind GPU contributor onboarding:
  1. Hardware, NVIDIA driver, disk & endpoint preflight (`scripts/hivemind_volunteer_preflight.py`)
  2. Volunteer node heartbeat / firewall check or CrowdGPT co-training (`scripts/hivemind_opentrainer_agent.py`)

Options:
  --crowdgpt       Contribute to Crowd-v1 via CrowdGPT client under HiveMind VRAM & thermal guards
  --continuous     Keep running heartbeats/training loop instead of single `--once` pass
  --legacy-demo    Run legacy local Season 9/10 simulation demo
  --reserve-mib N  Free VRAM MiB always reserved for your desktop/games (default: 1536)
  -h, --help       Show this help and exit
EOF
            exit 0
            ;;
        --crowdgpt) MODE="crowdgpt" ;;
        --continuous) ONCE_FLAG="" ;;
        --legacy-demo) MODE="legacy-demo" ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

echo "============================================================================"
echo "  PROJECT HIVEMIND - TURNKEY VOLUNTEER GPU ONBOARDING"
echo "  Gateway: ${GATEWAY_URL}"
echo "============================================================================"
echo "[LEGAL NOTICE] 0.00 EUR / 0.00 USD: No money is paid; points confer reciprocal compute."
echo ""

if [ -x "$ROOT_DIR/.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
elif [ -x "$ROOT_DIR/../../../.venv/bin/python" ]; then
    PYTHON_BIN="$ROOT_DIR/../../../.venv/bin/python"
else
    PYTHON_BIN="python3"
fi

if [ "$MODE" = "legacy-demo" ]; then
    HANDLE="${USER:-citizen-scientist}"
    mkdir -p "$ROOT_DIR/artifacts"
    PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/run_multi_moe_hf_campaign.py" \
        --contributor-handle "$HANDLE" --tier standard
    exit 0
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

if [ -n "${HIVEMIND_API_KEY:-}" ]; then
    exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_opentrainer_agent.py" \
        --gateway "$GATEWAY_URL" \
        --check-firewall \
        ${ONCE_FLAG:+"$ONCE_FLAG"} \
        "${EXTRA_ARGS[@]}"
fi

echo "No HIVEMIND_API_KEY set; running outbound firewall traversal check only."
echo "Get your key at ${GATEWAY_URL}/signup and run with HIVEMIND_API_KEY=hm-... to register."
exec env PYTHONPATH="$ROOT_DIR/src" "$PYTHON_BIN" "$ROOT_DIR/scripts/hivemind_opentrainer_agent.py" \
    --gateway "$GATEWAY_URL" \
    --check-firewall \
    --once \
    "${EXTRA_ARGS[@]}"
