#!/bin/sh
# Stage the overlay context and build it with Cloud Build. Usage: deploy/router_overlay/build.sh <image-tag>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
cp "$here/Dockerfile" "$here/server.py" "$stage/"
mkdir -p "$stage/router"
cp -r "$here/../../src/hivemind/router/"* "$stage/router/"
gcloud builds submit --region us-central1 --tag "us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter:$1" "$stage"
rm -r "$stage"
