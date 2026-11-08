#!/usr/bin/env bash
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
# Provisioning & Keepalive Script for the Low-Cost Permanent OpenTrainer Orchestrator VM (`hm-opentrainer-1`).
# Enforces AGENTS.md GCP Cost Governance:
#   - e2-small (2 vCPU, 2 GiB RAM, 0 GPUs, ~$3.65/mo Spot)
#   - --provisioning-model=SPOT --instance-termination-action=STOP
#   - Private VPC IP 10.150.0.14 (no paid unused static external IP)
#   - 5-minute Cloud Scheduler keepalive (`hm-opentrainer-1-keepalive`)

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-gen-lang-client-0625573011}"
ZONE="${ZONE:-us-east4-b}"
VM_NAME="${VM_NAME:-hm-opentrainer-1}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-small}"
PRIVATE_IP="${PRIVATE_IP:-10.150.0.14}"

echo "[opentrainer-vm] Creating low-cost Spot VM ${VM_NAME} (${MACHINE_TYPE}) in ${PROJECT_ID}/${ZONE}..."
gcloud compute instances create "${VM_NAME}" \
  --project="${PROJECT_ID}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --network-interface="subnet=default,private-network-ip=${PRIVATE_IP}" \
  --tags="hm-kafka,hm-opentrainer" \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --boot-disk-type=pd-standard \
  --no-service-account \
  --no-scopes

echo "[opentrainer-vm] Ensuring 5-minute Cloud Scheduler keepalive (${VM_NAME}-keepalive)..."
gcloud scheduler jobs create http "${VM_NAME}-keepalive" \
  --project="${PROJECT_ID}" \
  --location="us-central1" \
  --schedule="*/5 * * * *" \
  --uri="https://compute.googleapis.com/compute/v1/projects/${PROJECT_ID}/zones/${ZONE}/instances/${VM_NAME}/start" \
  --http-method=POST \
  --oauth-service-account-email="1003063861791-compute@developer.gserviceaccount.com" || true

echo "[opentrainer-vm] Done. Install deploy/opentrainer_vm/hivemind-opentrainer.service on ${VM_NAME}."
