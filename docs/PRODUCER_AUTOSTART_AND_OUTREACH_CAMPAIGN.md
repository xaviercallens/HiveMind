# Producer Auto-Start Background Service & End-User GPU Reconnection Campaign

This guide documents:
1. **Honest Physical GPU Telemetry (`#402`)**: how the HiveMind OpenRouter gateway counts distinct physical GPU hosts (`distinct_physical_gpus`) separately from logical producer accounts (`live_producer_accounts`) and endpoint paths (`live_endpoints`).
2. **Cross-Platform Producer Auto-Start Service (`scripts/hivemind_producer_autostart.py`)**: how volunteer HiveMiners on **Windows 10/11**, **Linux (`systemd --user`)**, and **macOS (`launchd`)** keep their local GPU producer + Cloudflare tunnel + authenticating shield connected across reboots and terminal closures.
3. **End-User GPU Reconnection & Onboarding Campaign (`scripts/hivemind_gpu_outreach_campaign.py`)**: how to generate personalized bilingual (FR/EN) outreach invitations for registered prosumer/producer accounts (read from a router.db snapshot).

---

## 1. Honest Physical GPU Counting (`distinct_physical_gpus`)

Every live endpoint registered in `/v1/hivemind/producer/heartbeat` has its hostname normalized via `RouterStore._endpoint_host_root(endpoint_url)`.
- When multiple logical producer services run on the same physical machine (for example `https://34-48-48-154.sslip.io` for `hivemind/coder` and `https://34-48-48-154.sslip.io/bee` for `hivemind/bee-stars` on `socreateai-agora-hermes-node1`), `/v1/hivemind/producers/scale-100-status` and `/v1/hivemind/producers/fleet` report:
  - `distinct_physical_gpus`: count of unique normalized hostnames across live endpoints (`1` for a single VM)
  - `live_endpoints`: count of active service endpoints (`2+`)
  - `shared_host_endpoints`: count of endpoints sharing a physical host with another endpoint
  - `scale_readiness_pct`: computed strictly from `distinct_physical_gpus / target_producers`
- In `/v1/hivemind/t4/status`, `hardware.secondary_gpu` is omitted whenever only one physical GPU host is active (`sole_active_gpu: true`).

---

## 2. Auto-Start Background Service Installer (`scripts/hivemind_producer_autostart.py`)

When a volunteer runs `hivemind_producer_shield.py`, `cloudflared tunnel`, and `hivemind_producer_agent.py` manually inside a terminal window, closing the terminal or rebooting the PC drops the Cloudflare tunnel (`HTTP 000`).

Use `scripts/hivemind_producer_autostart.py` to generate an auto-reconnecting background service for your operating system:

### A. Linux (`systemd --user`)

```bash
mkdir -p ~/.config/hivemind
cat > ~/.config/hivemind/producer.env <<'EOF'
HIVEMIND_API_KEY=hm-YOUR_PRODUCER_KEY
EOF
chmod 600 ~/.config/hivemind/producer.env

python3 scripts/hivemind_producer_autostart.py linux \
  --model coder=qwen2.5-coder:7b \
  --apply ~/.local/share/hivemind

mkdir -p ~/.config/systemd/user
cp ~/.local/share/hivemind/hivemind-producer.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now hivemind-producer.service
```

### B. Windows 10 / 11 (PowerShell Scheduled Task — e.g. RTX 4070 / RTX 3080 / RTX 2070)

```powershell
New-Item -ItemType Directory -Force -Path "$HOME\.hivemind" | Out-Null
Set-Content -Path "$HOME\.hivemind\producer.env.ps1" -Value '$env:HIVEMIND_API_KEY = "hm-YOUR_PRODUCER_KEY"'

python scripts\hivemind_producer_autostart.py windows `
  --model coder=qwen2.5-coder:7b `
  --apply .\producer-autostart

powershell -ExecutionPolicy Bypass -File .\producer-autostart\install-producer-service.ps1
```

### C. macOS (`launchd` LaunchAgent — Apple Silicon M1–M4)

```bash
python3 scripts/hivemind_producer_autostart.py macos \
  --model coder=qwen2.5-coder:7b \
  --apply ~/.local/share/hivemind

cp ~/.local/share/hivemind/org.hivemind.producer.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/org.hivemind.producer.plist
```

---

## 3. Organizing the End-User GPU Reconnection Campaign

Run `scripts/hivemind_gpu_outreach_campaign.py` to generate ready-to-send bilingual (FR/EN) outreach messages for the registered prosumer/producer cohort (read from a SQLite `router.db` snapshot; optional per-recipient hardware notes come from an untracked `--cohort-file` JSON, never from the repository):

```bash
# Generate Markdown outreach kit for all registered prosumers/producers:
python3 scripts/hivemind_gpu_outreach_campaign.py --db /path/to/router.db --format markdown

# Or inspect structured JSON from a specific router.db snapshot:
python3 scripts/hivemind_gpu_outreach_campaign.py --db /path/to/router.db --format json
```
