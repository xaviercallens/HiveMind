#!/bin/sh
# Stage the overlay context and build it with Cloud Build. Usage: deploy/router_overlay/build.sh <image-tag>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
cp "$here/Dockerfile" "$here/server.py" "$stage/"
cp "$here/../../src/hivemind/router/points_wording.py" "$stage/"
cp "$here/../../src/hivemind/router/join_page.py" "$stage/"
gcloud builds submit --region us-central1 --tag "us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter:$1" "$stage"
rm -r "$stage"
