#!/bin/sh
# Stage the overlay context and build it with Cloud Build.
# Usage: deploy/router_overlay/build.sh <image-tag> [--deploy]
set -eu
TAG="${1:-i18n-5lang}"
DEPLOY_FLAG="${2:-}"
IMAGE="us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter:${TAG}"
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
cp "$here/Dockerfile" "$here/server.py" "$stage/"
mkdir -p "$stage/hivemind" "$stage/gallery"
cp -r "$here/../../src/hivemind/"* "$stage/hivemind/"
cp "$here/../../public/gallery/"*.jpg "$here/../../public/gallery/"*.svg "$stage/gallery/"
find "$stage/hivemind" -name __pycache__ -type d -prune -exec rm -rf {} +
gcloud builds submit --region us-central1 --tag "$IMAGE" "$stage"
rm -r "$stage"
if [ "$DEPLOY_FLAG" = "--deploy" ]; then
  gcloud run services update hivemind-openrouter \
    --project gen-lang-client-0625573011 \
    --region us-central1 \
    --image "$IMAGE"
fi
