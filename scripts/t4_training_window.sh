#!/bin/bash
# Run the T4 VM (Spot) only during a nightly training window, using a GCE instance schedule. AGENTS.md: "a few hours a day or night for training only".
#   bash scripts/t4_training_window.sh status
#   CONFIRM=yes START=22 STOP=2 bash scripts/t4_training_window.sh apply     # hours in Europe/Paris
#   bash scripts/t4_training_window.sh remove
# While the VM is off, EVERYTHING on it is off: T4 producers (the gateway keeps only the RTX 2070 and volunteers), the BOINC project server (volunteers get no work and cannot report),
# the local Kafka, Redis, and every Claude session running on the VM. Read docs/T4_TRAINING_WINDOW.md first. A Spot VM preempted inside the window waits for the next scheduled start.
set -euo pipefail
PROJECT=gen-lang-client-0625573011
REGION=us-east4
ZONE=us-east4-b
VM=socreateai-agora-hermes-node1
POLICY=hivemind-t4-training-window
START="${START:-22}"
STOP="${STOP:-2}"

case "${1:-status}" in
  status)
    gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" --format='value(status,resourcePolicies.list())'
    gcloud compute resource-policies describe "$POLICY" --region "$REGION" --project "$PROJECT" --format='value(instanceSchedulePolicy.vmStartSchedule.schedule,instanceSchedulePolicy.vmStopSchedule.schedule,instanceSchedulePolicy.timeZone)' 2>/dev/null || echo "no schedule"
    ;;
  apply)
    [ "${CONFIRM:-}" = yes ] || { echo "refusing: set CONFIRM=yes after checking the list in docs/T4_TRAINING_WINDOW.md"; exit 2; }
    gcloud compute resource-policies describe "$POLICY" --region "$REGION" --project "$PROJECT" >/dev/null 2>&1 || \
      gcloud compute resource-policies create instance-schedule "$POLICY" --region "$REGION" --project "$PROJECT" --timezone Europe/Paris \
        --vm-start-schedule "0 $START * * *" --vm-stop-schedule "0 $STOP * * *" --description "T4 training window (owner decision 2026-11-04)"
    gcloud compute instances add-resource-policies "$VM" --zone "$ZONE" --project "$PROJECT" --resource-policies "$POLICY"
    ;;
  remove)
    gcloud compute instances remove-resource-policies "$VM" --zone "$ZONE" --project "$PROJECT" --resource-policies "$POLICY"
    gcloud compute resource-policies delete "$POLICY" --region "$REGION" --project "$PROJECT" --quiet
    ;;
  *) echo "usage: $0 status|apply|remove"; exit 64 ;;
esac
