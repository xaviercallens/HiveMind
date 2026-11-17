# Overlay on the live HiveMind OpenRouter image: adds the activity window on consumer bids (PR #182),
# the points-only display wording, full src/hivemind modules, and the upgraded /join, /rejoindre,
# and /life-as-a-bee#image-moe-gallery pages.
FROM us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter@sha256:a75897200a7c3fe18de81962491654e455138b87de1e1c3053e2f4ff3c4c4b88
USER root
RUN [ -f /app/server_orig.py ] || mv /app/server.py /app/server_orig.py
COPY hivemind/ /app/src/hivemind/
COPY gallery/ /app/public/gallery/
COPY server.py /app/server.py
USER app
