#!/bin/sh
# Stage the overlay context and build it with Cloud Build. Usage: deploy/router_overlay/build.sh <image-tag>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
cp "$here/Dockerfile" "$here/server.py" "$stage/"
mkdir -p "$stage/hivemind" "$stage/gallery"
cp -r "$here/../../src/hivemind/"* "$stage/hivemind/"
cp "$here/../../public/gallery/"*.jpg "$here/../../public/gallery/"*.svg "$stage/gallery/"
find "$stage/hivemind" -name __pycache__ -type d -prune -exec rm -rf {} +
gcloud builds submit --region us-central1 --tag "us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter:$1" "$stage"
rm -r "$stage"
