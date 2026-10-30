#!/usr/bin/env bash
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# ==============================================================================
# HiveMind@home — Official BOINC Client + Idle-GPU Worker Installer (Linux/macOS)
# Usage:
#   curl -sSL https://xaviercallens.github.io/HiveMindatHome.github.io/boinc/install_boinc_hivemind_linux.sh | bash -s -- "@YourBOINCHandle"
# ==============================================================================
set -euo pipefail

BOINC_HANDLE="${1:-@BOINC_Volunteer}"
COORDINATOR_URL="${HIVEMIND_COORDINATOR_URL:-https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app}"
WEAK_ACCOUNT_KEY="${HIVEMIND_WEAK_KEY:-hm_boinc_weak_key_v213_open_science}"
INSTALL_DIR="${HOME}/.hivemind_boinc"

echo "=============================================================================="
echo " 🐝 HiveMind@home (v2.13.0) — BOINC Client & Idle-GPU Worker Installer"
echo "    Handle          : ${BOINC_HANDLE}"
echo "    Project URL     : ${COORDINATOR_URL}/"
echo "    Applications    : hivemind_astro_bls (CUDA/OpenCL/WebGPU), hivemind_sundials_ode (CPU/GPU)"
echo "    VRAM Ceiling    : 6,656 MiB (Safe for 8 GB GPUs) | Gaming Preemption: < 412 ms"
echo "=============================================================================="

mkdir -p "${INSTALL_DIR}"

# 1. Install Official Berkeley BOINC Client (`boinc-client` & `boinccmd`) if not installed
if ! command -v boinccmd >/dev/null 2>&1; then
  echo "[1/5] Installing official Berkeley BOINC client (boinc-client)..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y && sudo apt-get install -y boinc-client xprintidle python3
  elif command -v dnf >/dev/null 2>&1; then
    sudo dnf install -y boinc-client python3
  elif command -v pacman >/dev/null 2>&1; then
    sudo pacman -Sy --noconfirm boinc python
  elif command -v brew >/dev/null 2>&1; then
    brew install --cask boinc || true
  else
    echo "  [!] Package manager not detected; continuing with standalone HiveMind BOINC worker."
  fi
else
  echo "[1/5] Official BOINC client (boinccmd) already installed: $(command -v boinccmd)"
fi

# 2. Write BOINC `cc_config.xml` and HiveMind@home `app_config.xml`
echo "[2/5] Writing BOINC cc_config.xml & app_config.xml templates to ${INSTALL_DIR}..."
cat > "${INSTALL_DIR}/cc_config.xml" <<'EOF'
<cc_config>
  <log_flags>
    <task>1</task>
    <file_xfer>1</file_xfer>
    <sched_ops>1</sched_ops>
    <coproc_debug>0</coproc_debug>
  </log_flags>
  <options>
    <use_all_gpus>1</use_all_gpus>
    <allow_remote_gui_rpc>0</allow_remote_gui_rpc>
    <max_file_xfers_per_project>4</max_file_xfers_per_project>
  </options>
</cc_config>
EOF

cat > "${INSTALL_DIR}/app_config.xml" <<'EOF'
<app_config>
  <app>
    <name>hivemind_astro_bls</name>
    <max_concurrent>2</max_concurrent>
    <gpu_versions>
      <gpu_usage>0.5</gpu_usage>
      <cpu_usage>0.25</cpu_usage>
    </gpu_versions>
  </app>
  <app>
    <name>hivemind_sundials_ode</name>
    <max_concurrent>4</max_concurrent>
    <gpu_versions>
      <gpu_usage>0.5</gpu_usage>
      <cpu_usage>0.5</cpu_usage>
    </gpu_versions>
  </app>
  <project_max_concurrent>4</project_max_concurrent>
</app_config>
EOF

if [ -d "/var/lib/boinc-client" ] && [ -w "/var/lib/boinc-client" ]; then
  cp -f "${INSTALL_DIR}/cc_config.xml" /var/lib/boinc-client/cc_config.xml || true
fi

# 3. Attach BOINC daemon to HiveMind@home project URL & reload config
echo "[3/5] Attaching local BOINC client to HiveMind@home (${COORDINATOR_URL}/)..."
if command -v boinccmd >/dev/null 2>&1; then
  boinccmd --project_attach "${COORDINATOR_URL}/" "${WEAK_ACCOUNT_KEY}" 2>/dev/null || true
  boinccmd --read_cc_config 2>/dev/null || true
  boinccmd --set_gpu_mode auto 2>/dev/null || true
fi

# 4. Download the open-source HiveMind BOINC Idle-GPU Worker (`hivemind_boinc_idle_worker.py`)
echo "[4/5] Downloading HiveMind BOINC Idle-GPU Worker..."
curl -sSL "https://xaviercallens.github.io/HiveMindatHome.github.io/boinc/hivemind_boinc_idle_worker.py" \
  -o "${INSTALL_DIR}/hivemind_boinc_idle_worker.py"
chmod +x "${INSTALL_DIR}/hivemind_boinc_idle_worker.py"

# 5. Execute 1 verified BOINC Exoplanet BLS Transit workunit & print SHA-256 Cobblestone receipt
echo "[5/5] Running verification workunit (--once --force-fold) and syncing with GCP BOINC Coordinator..."
python3 "${INSTALL_DIR}/hivemind_boinc_idle_worker.py" \
  --coordinator "${COORDINATOR_URL}" \
  --handle "${BOINC_HANDLE}" \
  --force-fold \
  --once

echo ""
echo "✅ HiveMind@home BOINC setup complete!"
echo "   • Run continuous idle-GPU daemon : python3 ${INSTALL_DIR}/hivemind_boinc_idle_worker.py --handle \"${BOINC_HANDLE}\""
echo "   • Live BOINC Leaderboard         : ${COORDINATOR_URL}/community"
echo "   • Zero-Install WebGPU Workunits  : https://xaviercallens.github.io/HiveMindatHome.github.io/webgpu.html"
