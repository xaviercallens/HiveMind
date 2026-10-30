# Overlay on the live HiveMind OpenRouter image: adds the activity window on consumer bids (PR #182),
# the points-only display wording, full src/hivemind modules, the /life-as-a-bee#image-moe-gallery assets,
# and full 5-language (EN, FR, ES, DE, ZH) support across /, /playground, /fr, /aire-de-jeu, /es, /de, /zh,
# /signup, /inscription, /join, /rejoindre, and /v1/hivemind/i18n.
FROM us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-openrouter@sha256:a612ff4036b785c90a1e441f473ef832c43f2b6e9244560167136cbff07107cc
USER root
RUN mv /app/server.py /app/server_orig.py
COPY hivemind/ /app/src/hivemind/
COPY gallery/ /app/public/gallery/
COPY server.py /app/server.py
USER app
