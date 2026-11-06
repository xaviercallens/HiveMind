#!/usr/bin/env bash
# Deploy the HiveMind OpenRouter gateway to Cloud Run as a PUBLIC service (the app authenticates every request itself
# with hm- API keys; the owner approved public exposure on 2026-10-24).
#
# Required environment:
#   GCP_PROJECT                      Google Cloud project id
#   HIVEMIND_ROUTER_ADMIN_KEY_HASH   HMAC-SHA256 hex of the admin key, keyed with the router secret (see docs)
# Optional environment:
#   REGION (default us-central1), SECRET_NAME (default hivemind-router-secret),
#   SERVICE_ACCOUNT (default hivemind-openrouter@$GCP_PROJECT.iam.gserviceaccount.com),
#   LEDGER_BUCKET (default hivemind-openrouter-ledger-<project number>)
#   HIVEMIND_GOOGLE_CLIENT_ID  OAuth web client id that enables "Sign in with Google" at /signup (passed via
#                              --update-env-vars so extra variables are preserved). HIVEMIND_OPEN_REGISTRATION=0 closes
#                              handle-only registration once Google sign-in works.
#
# Pre-provisioned (one time): the service account, the Secret Manager secret (accessor role for the account) and the
# private versioned GCS bucket (objectAdmin for the account).
#
# Durability: the SQLite ledger is replicated to gs://$LEDGER_BUCKET/router by Litestream and restored on every cold
# start. Scale to zero (--min-instances 0) for cost control (owner decision 2026-11-04, AGENTS.md); while an instance lives its CPU stays allocated (--no-cpu-throttling)
# so Litestream keeps replicating between requests. These flags match the live service.
set -euo pipefail

: "${GCP_PROJECT:?set GCP_PROJECT to the Google Cloud project id}"
: "${HIVEMIND_ROUTER_ADMIN_KEY_HASH:?set HIVEMIND_ROUTER_ADMIN_KEY_HASH to the admin key HMAC hex}"
REGION="${REGION:-us-central1}"
SECRET_NAME="${SECRET_NAME:-hivemind-router-secret}"
SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-hivemind-openrouter@${GCP_PROJECT}.iam.gserviceaccount.com}"
PROJECT_NUMBER="$(gcloud projects describe "$GCP_PROJECT" --format 'value(projectNumber)')"
LEDGER_BUCKET="${LEDGER_BUCKET:-hivemind-openrouter-ledger-${PROJECT_NUMBER}}"

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo="$(cd "$here/../.." && pwd)"

# Assemble a clean build context: Dockerfile, server.py, entrypoint.sh, requirements.txt and the src/ tree.
stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT
cp -r "$repo/src" "$stage/src"
mkdir -p "$stage/artifacts/mvp" "$stage/artifacts/experiments" "$stage/docs"
cp -f "$repo/artifacts/mvp/"*.json "$stage/artifacts/mvp/" 2>/dev/null || true
cp -f "$repo/artifacts/experiments/GRANITE_ADAPTER_"*.json "$stage/artifacts/experiments/" 2>/dev/null || true
cp -f "$repo/docs/MVP_SCORECARD.md" "$repo/docs/MVP_OWNER_DECISIONS.md" "$stage/docs/" 2>/dev/null || true
find "$stage/src" -name __pycache__ -type d -prune -exec rm -rf {} +
cp "$here/Dockerfile" "$here/server.py" "$here/entrypoint.sh" "$here/requirements.txt" "$stage/"
chmod +x "$stage/entrypoint.sh"

ENV_VARS="HIVEMIND_ROUTER_DB=/data/router.db,HIVEMIND_ROUTER_REPLICA_URL=gcs://${LEDGER_BUCKET}/router,HIVEMIND_ROUTER_ADMIN_KEY_HASH=${HIVEMIND_ROUTER_ADMIN_KEY_HASH}"
[ -n "${HIVEMIND_GOOGLE_CLIENT_ID:-}" ] && ENV_VARS="${ENV_VARS},HIVEMIND_GOOGLE_CLIENT_ID=${HIVEMIND_GOOGLE_CLIENT_ID}"
[ -n "${HIVEMIND_OPEN_REGISTRATION:-}" ] && ENV_VARS="${ENV_VARS},HIVEMIND_OPEN_REGISTRATION=${HIVEMIND_OPEN_REGISTRATION}"

CPU="${CPU:-1}"
MEMORY="${MEMORY:-1Gi}"

cd "$stage"
gcloud run deploy hivemind-openrouter \
  --project "$GCP_PROJECT" \
  --source . \
  --region "$REGION" \
  --service-account "$SERVICE_ACCOUNT" \
  --allow-unauthenticated \
  --min-instances 0 \
  --max-instances 1 \
  --no-cpu-throttling \
  --cpu-boost \
  --cpu "$CPU" \
  --concurrency 80 \
  --memory "$MEMORY" \
  --timeout 600 \
  --update-env-vars "${ENV_VARS}" \
  --set-secrets "HIVEMIND_ROUTER_SECRET=${SECRET_NAME}:latest"
