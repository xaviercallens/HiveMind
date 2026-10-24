#!/usr/bin/env python3
"""HiveMind producer shield: a small authenticating proxy in front of your local model server.

Why: to earn points your model server must be reachable from the internet (through a tunnel or a port forward), and
Ollama, vLLM and llama.cpp have no login of their own. Anyone who learns the URL could use your GPU, change or delete
your models (Ollama's management API is on the same port) and read what you serve. The shield publishes only what the
gateway needs and nothing else:

  * ``GET /v1/models`` and ``POST /v1/chat/completions`` are the only routes; everything else is 404.
  * Every request needs ``Authorization: Bearer <your endpoint token>`` (constant-time comparison); the gateway
    presents the token you gave the producer agent in HIVEMIND_ENDPOINT_TOKEN.
  * Request bodies are capped, concurrent requests are capped (extra ones get 429 at once, so the gateway tries
    another producer instead of queueing on your GPU), and ``--allow-model`` can pin the models that may be run.

Usage (point your tunnel or port forward at the shield's port, not at the model server):
    export HIVEMIND_ENDPOINT_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
    python hivemind_producer_shield.py --upstream http://127.0.0.1:11434 --port 8787 --allow-model qwen2.5-coder:7b
Needs: pip install fastapi uvicorn httpx
"""

from __future__ import annotations

import argparse
import hmac
import json
import os
import itertools
import sys
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

ENDPOINT_TOKEN_ENV: str = "HIVEMIND_ENDPOINT_TOKEN"
MIN_TOKEN_CHARS: int = 16
DEFAULT_MAX_BODY_BYTES: int = 1_000_000
DEFAULT_MAX_CONCURRENT: int = 2
UPSTREAM_TIMEOUT_S: float = 300.0


def _error(status: int, message: str, code: str) -> JSONResponse:
    return JSONResponse({"error": {"message": message, "type": "shield_error", "code": code}}, status_code=status)


def create_shield_app(
    upstream: str,
    token: str,
    *,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    allowed_models: Sequence[str] = (),
    transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """Build the proxy app. ``transport`` lets tests stand in for the model server."""
    if len(token) < MIN_TOKEN_CHARS:
        raise ValueError(f"the endpoint token must be at least {MIN_TOKEN_CHARS} characters")
    if max_concurrent < 1 or max_body_bytes < 1:
        raise ValueError("max_concurrent and max_body_bytes must be >= 1")
    base = upstream.rstrip("/")
    allowed = frozenset(allowed_models)
    expected = f"Bearer {token}".encode()
    inflight: dict[int, float] = {}  # request number -> monotonic start; entries expire so a lost disconnect cannot jam the shield
    numbers = itertools.count()
    client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S, transport=transport)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def authorised(request: Request) -> bool:
        supplied = request.headers.get("authorization", "").encode()
        return hmac.compare_digest(supplied, expected)

    @app.get("/v1/models")
    async def models(request: Request) -> Any:
        if not authorised(request):
            return _error(401, "Missing or wrong endpoint token.", "unauthorized")
        try:
            upstream_response = await client.get(base + "/v1/models")
            if upstream_response.status_code == 200:
                return JSONResponse(upstream_response.json())
            # Older Ollama releases have no /v1/models; present their own model list in the OpenAI shape.
            tags = await client.get(base + "/api/tags")
        except (httpx.HTTPError, ValueError):
            return _error(502, "The model server is not reachable.", "upstream_unreachable")
        if tags.status_code != 200:
            return _error(502, "The model server did not list its models.", "upstream_models_unavailable")
        try:
            names = [str(m["name"]) for m in tags.json().get("models", []) if isinstance(m, dict) and "name" in m]
        except ValueError:
            return _error(502, "The model server did not list its models.", "upstream_models_unavailable")
        return JSONResponse({"object": "list", "data": [{"id": name, "object": "model", "owned_by": "local"} for name in names]})

    @app.post("/v1/chat/completions")
    async def chat(request: Request) -> Any:
        if not authorised(request):
            return _error(401, "Missing or wrong endpoint token.", "unauthorized")
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > max_body_bytes:
            return _error(413, "Request body too large.", "body_too_large")
        chunks: list[bytes] = []
        total = 0
        async for chunk in request.stream():
            total += len(chunk)
            if total > max_body_bytes:
                return _error(413, "Request body too large.", "body_too_large")
            chunks.append(chunk)
        raw = b"".join(chunks)
        try:
            body = json.loads(raw)
        except ValueError:
            return _error(400, "Body must be JSON.", "invalid_json")
        if not isinstance(body, dict):
            return _error(400, "Body must be a JSON object.", "invalid_json")
        if allowed and body.get("model") not in allowed:
            return _error(403, "This model is not offered by this producer.", "model_not_allowed")
        now = time.monotonic()
        for stale in [n for n, started in inflight.items() if now - started > UPSTREAM_TIMEOUT_S + 30.0]:
            del inflight[stale]
        if len(inflight) >= max_concurrent:
            return _error(429, "Producer is busy.", "busy")
        number = next(numbers)
        inflight[number] = now
        try:
            upstream_request = client.build_request(
                "POST", base + "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
            )
            upstream_response = await client.send(upstream_request, stream=True)
        except httpx.HTTPError:
            inflight.pop(number, None)
            return _error(502, "The model server is not reachable.", "upstream_unreachable")

        async def relay() -> AsyncIterator[bytes]:
            try:
                async for part in upstream_response.aiter_raw():
                    yield part
            finally:
                await upstream_response.aclose()
                inflight.pop(number, None)

        media_type = upstream_response.headers.get("content-type", "application/json")
        return StreamingResponse(relay(), status_code=upstream_response.status_code, media_type=media_type)

    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Authenticating proxy in front of a local OpenAI-compatible model server.")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434", help="Your model server (default: Ollama on this machine)")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to listen on; keep 127.0.0.1 and point your tunnel at it")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    parser.add_argument("--allow-model", action="append", default=[], metavar="NAME",
                        help="Only run this local model name (repeat). Recommended: it keeps your GPU for the model you advertise.")
    args = parser.parse_args(argv)
    token = os.environ.get(ENDPOINT_TOKEN_ENV, "")
    if len(token) < MIN_TOKEN_CHARS:
        parser.error(f"set {ENDPOINT_TOKEN_ENV} to a random secret of at least {MIN_TOKEN_CHARS} characters")
    import uvicorn  # imported late so the module can be imported (and tested) without a server

    app = create_shield_app(args.upstream, token, max_concurrent=args.max_concurrent,
                            max_body_bytes=args.max_body_bytes, allowed_models=args.allow_model)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
