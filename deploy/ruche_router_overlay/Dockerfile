FROM us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-at-home-ruche@sha256:0afc7419911fb10a6b99535fbdd4f3af938006d7fd08c3d050fd3d115ff3ebd1
COPY night_moe_page.py /app/src/hivemind/boinc/night_moe_page.py
COPY artifacts/experiments/ /app/artifacts/experiments/
# join_page.py: copy src/hivemind/router/join_page.py into this directory before building (same convention as night_moe_page.py)
COPY join_page.py /app/join_page.py
COPY scripts/ /app/scripts/
COPY server.py /app/server.py
