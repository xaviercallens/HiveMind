#!/bin/bash
# Boot orchestrator for the T4 node. Idempotent. Run at every boot (GCE startup-script metadata calls it as root) and safe to
# run by hand. Order: serving first (inference), then the BOINC training layer, then the governor and monitor. It does not
# start anything that is not installed and enabled, and it never kills a process.
LOG=/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/startup.log
mkdir -p "$(dirname "$LOG")"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$LOG"; }
unit_ok() { systemctl cat "$1.service" >/dev/null 2>&1 && [ "$(systemctl is-enabled "$1.service" 2>/dev/null)" = "enabled" ]; }
start() { if unit_ok "$1"; then systemctl start "$1.service" && log "started $1" || log "FAILED to start $1"; else log "skip $1 (not installed/enabled)"; fi; }

log "startup begin (boot_id $(cat /proc/sys/kernel/random/boot_id))"
for i in $(seq 1 60); do [ -d /mnt/disks/disk-socrateai-local-1/boinc ] && break; sleep 2; done   # the data disk
[ -d /mnt/disks/disk-socrateai-local-1/boinc ] || log "WARNING: data disk not mounted after 120 s"
install -d -o callensxavier_gmail_com -g callensxavier_gmail_com /run/mysqld
systemctl daemon-reload

for u in ollama nginx; do start "$u"; done
# inference producers: the owner's account, then the older coder and bee producers
for u in hivemind-live-shield-user hivemind-live-agent-user hivemind-live-bee-llama hivemind-live-shield-bee hivemind-live-agent-bee \
         hivemind-live-shield-coder hivemind-live-agent-coder; do start "$u"; done
# batch layer
for u in hivemind-boinc-db hivemind-boinc-apache hivemind-boinc-daemons hivemind-boinc-client hivemind-t4-governor hivemind-t4-workgen hivemind-t4-monitor; do start "$u"; done

# verify what end users see (the gateway lists a tier only while a producer for it is live)
for i in $(seq 1 30); do
  tiers=$(curl -s -m 8 https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/models | tr ',' '\n' | grep -o 'hivemind/[a-z-]*' | sort -u | tr '\n' ' ')
  case "$tiers" in *hivemind/coder*) log "gateway lists: $tiers"; break;; esac
  sleep 10
done
log "startup end (gateway tiers: ${tiers:-none})"
