#!/usr/bin/env bash
# Deploy the HiveMind@home & La Ruche V3.1.1 Web Interface (hivemind-at-home-ruche) to Cloud Run.
# Enforces AGENTS.md Tier 1 Stateless Control Plane & HTTP Services autoscaler rules:
#   --min-instances 0 --max-instances 2 --cpu-throttling --cpu-boost --cpu 1 --memory 1Gi --concurrency 80
set -euo pipefail

GCP_PROJECT="${GCP_PROJECT:-gen-lang-client-0625573011}"
REGION="${REGION:-us-central1}"
SERVICE_NAME="${SERVICE_NAME:-hivemind-at-home-ruche}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -r "$repo/src" "$stage/src"
mkdir -p "$stage/scripts" "$stage/artifacts/mvp" "$stage/artifacts/experiments" "$stage/docs"
cp -f "$repo/scripts/adapter_to_served.py" "$repo/scripts/hivemind_"*.py "$stage/scripts/" 2>/dev/null || true
cp -f "$repo/artifacts/mvp/"*.json "$stage/artifacts/mvp/" 2>/dev/null || true
cp -f "$repo/artifacts/experiments/GRANITE_ADAPTER_"*.json "$stage/artifacts/experiments/" 2>/dev/null || true
cp -f "$repo/docs/MVP_SCORECARD.md" "$repo/docs/MVP_OWNER_DECISIONS.md" "$repo/docs/BOINC_ALWAYS_ON_PLAN.md" "$stage/docs/" 2>/dev/null || true
find "$stage/src" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$here/Dockerfile" "$here/server.py" "$here/requirements.txt" "$stage/"

cd "$stage"
gcloud run deploy "$SERVICE_NAME" \
  --project "$GCP_PROJECT" \
  --source . \
  --region "$REGION" \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 2 \
  --cpu-throttling \
  --cpu-boost \
  --cpu 1 \
  --memory 1Gi \
  --concurrency 80 \
  --timeout 300 \
  --set-env-vars "HIVEMIND_FINOPS_MIN_INSTANCES=0,HIVEMIND_PUBLIC_DNS_DOMAIN=hivemind.socrate-ai-lab.com,HIVEMIND_CORE_MVP_SSO_PROVIDERS=google,reddit,steam,github,HIVEMIND_PASSWORD_KDF=PBKDF2-HMAC-SHA256:200000"
