#!/bin/bash
# Probe several candidate base models one after another (one process each, so GPU memory is fully released between them) and append one JSON line per model.
#   bash scripts/t4_model_probe_all.sh OUT.jsonl model1 model2 ...
# Skips a model if a chat request reached the node in the last 60 s (the GPU belongs to inference first). Downloads missing weights from the Hub.
set -uo pipefail
OUT="$1"; shift
PY=/mnt/disks/disk-socrateai-local-1/boinc/venv/bin/python
HERE="$(cd "$(dirname "$0")" && pwd)"
for m in "$@"; do
  if [ "$(sudo tail -n 200 /var/log/nginx/access.log | grep -c "$(date -u -d '60 seconds ago' '+%d/%b/%Y:%H:%M')")" -gt 0 ] && sudo tail -n 200 /var/log/nginx/access.log | grep "$(date -u '+%d/%b/%Y:%H:%M')" | grep -q "chat/completions"; then
    echo "{\"model_id\": \"$m\", \"ok\": false, \"error\": \"skipped: inference traffic\"}" >> "$OUT"; continue
  fi
  HF_HUB_OFFLINE=0 "$PY" -W ignore "$HERE/t4_model_probe.py" "$m" 2>/dev/null | tail -n 1 >> "$OUT"
done
