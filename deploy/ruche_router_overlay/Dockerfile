FROM us-central1-docker.pkg.dev/gen-lang-client-0625573011/cloud-run-source-deploy/hivemind-at-home-ruche:join-community-20261029
USER root
RUN rm -f /app/src/hivemind/boinc/stitch_portal.py
COPY hivemind_at_home_dashboard.py /app/src/hivemind/boinc/hivemind_at_home_dashboard.py
COPY night_moe_page.py /app/src/hivemind/boinc/night_moe_page.py
COPY game_init.py /app/src/hivemind/game/__init__.py
COPY swarm_init.py /app/src/hivemind/swarm/__init__.py
COPY webgpu_galaxy_cockpit.py /app/src/hivemind/game/webgpu_galaxy_cockpit.py
COPY stitch_v46_cockpit_template.py /app/src/hivemind/game/stitch_v46_cockpit_template.py
COPY stitch_sync_v2/ /app/steam_dist/HiveMind_Astrophysics_Exoplanet_Hunter/stitch_sync_v2/
COPY international_launch_model.py /app/src/hivemind/economy/international_launch_model.py
COPY international_launch_views.py /app/src/hivemind/economy/international_launch_views.py
COPY international_launch_scaler.py /app/src/hivemind/swarm/international_launch_scaler.py
COPY paper/ /app/paper/
COPY artifacts/ /app/artifacts/
# join_page.py: copy src/hivemind/router/join_page.py into this directory before building (same convention as night_moe_page.py)
COPY join_page.py /app/join_page.py
COPY join_page.py /app/src/hivemind/router/join_page.py
COPY scripts/ /app/scripts/
COPY server.py /app/server.py
