#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
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
import itertools
import json
import os
import sys
import time
from collections.abc import AsyncIterator, Iterable, Mapping, Sequence
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
    if isinstance(status, bool) or not isinstance(status, int) or not (400 <= status <= 599):
        raise ValueError("status must be an HTTP error status integer in [400, 599]")
    if not isinstance(message, str) or not message.strip() or not isinstance(code, str) or not code.strip():
        raise ValueError("message and code must be non-empty strings")
    return JSONResponse({"error": {"message": message, "type": "shield_error", "code": code}}, status_code=status)


MAX_TOKENS_CAP = 16384  # per-request ceiling for max_tokens / max_completion_tokens


def create_shield_app(
    upstream: str,
    token: str,
    *,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    allowed_models: Sequence[str] = (),
    transport: httpx.AsyncBaseTransport | None = None,
    routes: Mapping[str, str] | None = None,
    control_state_file: str | None = None,
) -> FastAPI:
    """Build the proxy app. ``transport`` lets tests stand in for the model server.

    ``routes`` maps a model name to the base URL of the server that runs it, for a producer account that advertises models from two servers behind ONE
    endpoint (the gateway keeps one endpoint per account). Models not listed go to ``upstream``; routed models are allowed implicitly and are listed in
    ``/v1/models`` only while their server answers.
    """
    # ``control_state_file`` is the small JSON file the producer agent writes (``--control-state-file``): while it says ``"paused": true`` chat requests get
    # 503 ``producer_paused``. A missing or unreadable file means "not paused" (the behaviour without remote control); ``/v1/models`` is never affected.
    if (
        not isinstance(upstream, str)
        or not upstream.strip().startswith(("http://", "https://"))
        or any(c in upstream for c in ("\r", "\n", "\x00"))
    ):
        raise ValueError("upstream must be an http:// or https:// URL without control characters")
    if not isinstance(token, str) or len(token) < MIN_TOKEN_CHARS or any(c in token for c in ("\r", "\n", "\x00")):
        raise ValueError(f"the endpoint token must be at least {MIN_TOKEN_CHARS} characters without control characters")
    if (
        isinstance(max_concurrent, bool)
        or not isinstance(max_concurrent, int)
        or isinstance(max_body_bytes, bool)
        or not isinstance(max_body_bytes, int)
        or max_concurrent < 1
        or max_body_bytes < 1
    ):
        raise ValueError("max_concurrent and max_body_bytes must be >= 1")
    if (
        isinstance(allowed_models, (str, bytes))
        or not isinstance(allowed_models, Iterable)
        or any(not isinstance(m, str) or not m.strip() or any(c in m for c in ("\r", "\n", "\x00")) for m in allowed_models)
    ):
        raise ValueError("allowed_models must be a sequence of non-empty model strings without control characters")
    if routes is not None and (
        not isinstance(routes, Mapping)
        or any(
            not isinstance(m, str)
            or not m.strip()
            or any(c in m for c in ("\r", "\n", "\x00"))
            or not isinstance(u, str)
            or not u.strip().startswith(("http://", "https://"))
            or any(c in u for c in ("\r", "\n", "\x00"))
            for m, u in routes.items()
        )
    ):
        raise ValueError("routes must map non-empty model strings to http:// or https:// URLs without control characters")
    if control_state_file is not None and (
        not isinstance(control_state_file, str)
        or not control_state_file.strip()
        or any(c in control_state_file for c in ("\r", "\n", "\x00"))
    ):
        raise ValueError("control_state_file must be a non-empty string path without control characters or None")
    base = upstream.rstrip("/")
    route_base = {m: u.rstrip("/") for m, u in (routes or {}).items()}
    allowed = frozenset(allowed_models) | (frozenset(route_base) if allowed_models else frozenset())
    expected = f"Bearer {token}".encode()
    inflight: dict[int, float] = {}  # request number -> monotonic start; entries expire so a lost disconnect cannot jam the shield
    numbers = itertools.count()
    client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT_S, transport=transport)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await client.aclose()

    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, lifespan=lifespan)

    def paused() -> bool:
        if not control_state_file:
            return False
        try:
            with open(control_state_file, encoding="utf-8") as handle:
                return json.loads(handle.read(4096)).get("paused") is True
        except (OSError, ValueError, AttributeError):
            return False

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
                listing = upstream_response.json()
                for routed, target in route_base.items():
                    try:
                        if (await client.get(target + "/v1/models", timeout=3.0)).status_code == 200:
                            listing.setdefault("data", []).append({"id": routed, "object": "model", "owned_by": "local"})
                    except httpx.HTTPError:
                        pass  # that server is down: do not advertise its model
                return JSONResponse(listing)
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
        if paused():
            return _error(503, "This producer is paused by its owner.", "producer_paused")
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
        model = body.get("model")
        if not isinstance(model, str):  # a list or object would raise TypeError in the set/dict lookups below (a 500)
            return _error(400, "model must be a string.", "invalid_request")
        clamped = False
        for limit_key in ("max_tokens", "max_completion_tokens"):  # a consumer can not make one request run for ever
            limit = body.get(limit_key)
            if isinstance(limit, (int, float)) and not isinstance(limit, bool) and limit > MAX_TOKENS_CAP:
                body[limit_key] = MAX_TOKENS_CAP
                clamped = True
        if clamped:
            raw = json.dumps(body).encode("utf-8")
        if allowed and model not in allowed:
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
                "POST", route_base.get(model, base) + "/v1/chat/completions", content=raw, headers={"Content-Type": "application/json"}
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


POC_NOTICE = (
    "Proof of concept (PoC): not a production service. Until HiveMind is fully secured, prompts you serve and requests you make "
    "must contain only public, non-sensitive data (no secrets, personal, private or enterprise data); the people using your node "
    "are anonymous, and nothing yet verifies producers or their output."
)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the authenticating producer shield."""
    parser = argparse.ArgumentParser(description="Authenticating proxy in front of a local OpenAI-compatible model server.")
    parser.add_argument("--upstream", default="http://127.0.0.1:11434", help="Your model server (default: Ollama on this machine)")
    parser.add_argument("--host", default="127.0.0.1", help="Interface to listen on; keep 127.0.0.1 and point your tunnel at it")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--max-concurrent", type=int, default=DEFAULT_MAX_CONCURRENT)
    parser.add_argument("--max-body-bytes", type=int, default=DEFAULT_MAX_BODY_BYTES)
    parser.add_argument("--allow-model", action="append", default=[], metavar="NAME",
                        help="Only run this local model name (repeat). Recommended: it keeps your GPU for the model you advertise.")
    parser.add_argument("--route", action="append", default=[], metavar="MODEL=URL",
                        help="Send this model to another local server (repeat), e.g. hivemind-bee-stars=http://127.0.0.1:8294. Implies --allow-model for it.")
    parser.add_argument("--control-state-file", metavar="PATH", help="State file written by the producer agent (--control-state-file); while it says paused, chat requests get 503 producer_paused")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    print(f"NOTICE: {POC_NOTICE}", file=sys.stderr)
    if not args.upstream.startswith(("http://127.0.0.1", "http://localhost")):
        parser.error(f"--upstream expects a local loopback URL (http://127.0.0.1:PORT or http://localhost:PORT), got {args.upstream!r}")
    routes: dict[str, str] = {}
    for item in args.route:
        model, sep, url = item.partition("=")
        if not sep or not model.strip() or not url.startswith(("http://127.0.0.1", "http://localhost")):
            parser.error(f"--route expects MODEL=http://127.0.0.1:PORT, got {item!r}")
        routes[model.strip()] = url.strip()
    token = os.environ.get(ENDPOINT_TOKEN_ENV, "")
    if len(token) < MIN_TOKEN_CHARS:
        parser.error(f"set {ENDPOINT_TOKEN_ENV} to a random secret of at least {MIN_TOKEN_CHARS} characters")
    import uvicorn  # imported late so the module can be imported (and tested) without a server

    app = create_shield_app(args.upstream, token, max_concurrent=args.max_concurrent,
                            max_body_bytes=args.max_body_bytes, allowed_models=args.allow_model, routes=routes,
                            control_state_file=args.control_state_file)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
