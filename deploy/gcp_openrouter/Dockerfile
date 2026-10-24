# HiveMind OpenRouter gateway image.
# Build context is a staging directory assembled by deploy.sh: src/, server.py, requirements.txt, Dockerfile.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS=1

WORKDIR /app

# Litestream streams the SQLite ledger to a GCS bucket so accounts and HMT balances survive instance restarts.
# The release tarball is pinned by version and verified against a SHA-256 recorded when this file was written.
ARG LITESTREAM_VERSION=0.3.13
ARG LITESTREAM_SHA256=eb75a3de5cab03875cdae9f5f539e6aedadd66607003d9b1e7a9077948818ba0
ADD https://github.com/benbjohnson/litestream/releases/download/v${LITESTREAM_VERSION}/litestream-v${LITESTREAM_VERSION}-linux-amd64.tar.gz /tmp/litestream.tar.gz
RUN echo "${LITESTREAM_SHA256}  /tmp/litestream.tar.gz" | sha256sum -c - \
    && tar -C /usr/local/bin -xzf /tmp/litestream.tar.gz litestream \
    && rm /tmp/litestream.tar.gz

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY src /app/src
COPY server.py /app/server.py
COPY entrypoint.sh /app/entrypoint.sh

RUN useradd --system --uid 10001 --no-create-home app && mkdir /data && chown app /data
USER app

# Cloud Run's front end appends the caller's address to X-Forwarded-For. uvicorn is deliberately NOT started with
# --proxy-headers/--forwarded-allow-ips='*': that trusts the left-most (client-forgeable) entry. The gateway instead
# reads the right-most hop itself (HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS), so the registration rate limit is per real caller.
# Cloud Run injects PORT; default 8080 for local runs.
ENTRYPOINT ["/app/entrypoint.sh"]
