#!/bin/bash
# Start or stop the three Spot Kafka voters (and the T4 broker that needs them) so they only cost money while used. AGENTS.md: stopped when idle.
#   bash scripts/kafka_voters.sh status | stop | start
# `stop` refuses while a topic has received data in the last IDLE_MIN minutes (default 120). The Cloud Scheduler keepalive jobs restart preempted voters every 5 minutes,
# so they are paused on stop and resumed on start. KRaft keeps its metadata and topic data on the persistent disks: a stop loses nothing committed.
set -uo pipefail
PROJECT=gen-lang-client-0625573011
REGION="${SCHEDULER_REGION:-us-central1}"
IDLE_MIN="${IDLE_MIN:-120}"
DATA=/mnt/disks/disk-socrateai-local-1/amcp/kafka-node4
VOTERS=("hm-kafka-1 us-east4-a" "hm-kafka-2 us-east4-b" "hm-kafka-3 us-east4-c")

scheduler_location() {
  for r in us-east4 us-central1; do
    if gcloud scheduler jobs describe hm-kafka-1-keepalive --location "$r" --project "$PROJECT" >/dev/null 2>&1; then echo "$r"; return; fi
  done
  echo "$REGION"
}

case "${1:-status}" in
  status)
    for v in "${VOTERS[@]}"; do set -- $v; printf '%s %s\n' "$1" "$(gcloud compute instances describe "$1" --zone "$2" --project "$PROJECT" --format='value(status)')"; done
    systemctl is-active hivemind-kafka-t4-node 2>/dev/null | sed 's/^/t4 broker: /'
    ;;
  stop)
    recent=$(find "$DATA" -name '*.log' -not -path '*__cluster_metadata*' -mmin "-$IDLE_MIN" -size +0 2>/dev/null | head -5)
    if [ -n "$recent" ]; then echo "NOT stopping: topic data written in the last $IDLE_MIN min:"; echo "$recent"; exit 2; fi
    loc=$(scheduler_location)
    for i in 1 2 3; do gcloud scheduler jobs pause "hm-kafka-$i-keepalive" --location "$loc" --project "$PROJECT" --quiet; done
    sudo systemctl stop hivemind-kafka-t4-node
    for v in "${VOTERS[@]}"; do set -- $v; gcloud compute instances stop "$1" --zone "$2" --project "$PROJECT" --quiet; done
    ;;
  start)
    loc=$(scheduler_location)
    for v in "${VOTERS[@]}"; do set -- $v; gcloud compute instances start "$1" --zone "$2" --project "$PROJECT" --quiet; done
    for i in 1 2 3; do gcloud scheduler jobs resume "hm-kafka-$i-keepalive" --location "$loc" --project "$PROJECT" --quiet; done
    sudo systemctl start hivemind-kafka-t4-node
    ;;
  *) echo "usage: $0 status|stop|start"; exit 64 ;;
esac
