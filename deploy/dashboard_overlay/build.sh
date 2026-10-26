#!/bin/sh
# Stage and build the dashboard overlay on the image a service currently serves.
# Usage: deploy/dashboard_overlay/build.sh <base-image@sha256:...> <image-tag>
set -eu
here=$(cd "$(dirname "$0")" && pwd)
stage=$(mktemp -d)
printf 'FROM %s\nUSER root\nRUN mv /app/server.py /app/server_orig.py\nCOPY points_wording.py /app/src/hivemind/router/points_wording.py\nCOPY server.py /app/server.py\n' "$1" > "$stage/Dockerfile"
cp "$here/server.py" "$stage/"
cp "$here/../../src/hivemind/router/points_wording.py" "$stage/"
gcloud builds submit --region us-central1 --tag "us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-at-home-ruche:$2" "$stage"
rm -r "$stage"
