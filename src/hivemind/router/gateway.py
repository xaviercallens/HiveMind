"""OpenAI-compatible FastAPI gateway of the HiveMind OpenRouter.

Consumers call ``/v1/chat/completions`` with a ``Bearer hm-...`` key; the gateway clears the per-model market
(:func:`hivemind.router.market.clear_market`), picks a live producer by smooth weighted round-robin, forwards the
request, meters the usage defensively (a producer can never bill more than the bytes that actually crossed the wire)
and settles in HMT through :class:`hivemind.router.accounts.RouterStore`.

Security notes
--------------
* API keys, admin keys and prompt text are never logged; log lines carry request ids, model ids and status only.
* Producer endpoints must be ``https`` with only globally routable addresses (SSRF guard); the check runs at
  heartbeat time and again immediately before every forward.
* Registration is rate limited in memory (10 / minute / client address). Behind a proxy run uvicorn with
  ``--proxy-headers --forwarded-allow-ips`` so ``request.client`` is the real caller.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import logging
import math
import re
import secrets
import socket
import threading
import time
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from hivemind.router.accounts import (
    Account,
    DuplicateRequest,
    HandleTaken,
    InsufficientBalance,
    ProducerRecord,
    ReservedHandle,
    RouterStore,
    TooManyEndpoints,
    TooManyHolds,
    UnknownAccount,
)
from hivemind.router.catalog import PREWIRED_PARTNER_MODELS, catalog_payload, normalize_model_alias
from hivemind.router.google_auth import GoogleAuthError, GoogleIdTokenVerifier
from hivemind.router.oidc_auth import OidcAuthError, OidcIdTokenVerifier
from hivemind.router.portal_page import portal_content_security_policy, render_portal_page
from hivemind.router.signup_page import content_security_policy, render_signup_page
from hivemind.router.market import (
    MECHANISM_NO_TRADE,
    ConsumerBid,
    FeePolicy,
    MarketEquilibrium,
    ProducerOffer,
    certify,
    clear_market,
)
from hivemind.security._exceptions import SwarmSecurityViolationError

LOGGER = logging.getLogger("hivemind.router.gateway")

MODEL_PREFIX = "hivemind/"
AUTO_MODEL = "auto"
PRODUCER_TTL_S = 120.0
PRODUCER_TIMEOUT_S = 120.0
DEFAULT_MAX_TOKENS = 1024
REGISTER_LIMIT = 10
REGISTER_WINDOW_S = 60.0
MAX_BODY_BYTES = 4_000_000
MAX_PRICE_CEILING = 1_000.0
MAX_PRODUCER_RESPONSE_BYTES = 8_000_000
MAX_SSE_LINE_BYTES = 1_000_000
PRODUCER_DEADLINE_S = 300.0
MAX_PRODUCER_CAPACITY_1K = 100_000.0
PROMPT_TOKEN_FACTOR = 2
MARKET_CACHE_EPOCH_S = 10.0
LIMIT_FIELDS: tuple[str, ...] = ("max_completion_tokens", "max_tokens")
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")

Clock = Callable[[], float]
Resolver = Callable[[str], Sequence[str]]


class RouterError(Exception):
    """An error rendered in the OpenAI error shape ``{"error": {"message", "type", "code"}}``."""

    def __init__(self, status: int, message: str, error_type: str, code: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code


def _error_response(status: int, message: str, error_type: str, code: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type, "code": code}})


# ---------------------------------------------------------------------------------------------------------
# pure helpers: metering, text extraction, SSRF guard, scheduling
# ---------------------------------------------------------------------------------------------------------


def estimate_tokens(num_bytes: int) -> int:
    """Conservative token estimate (4 bytes per token) used only when a producer reports no usage."""
    return (max(0, num_bytes) + 3) // 4


def meter_usage(
    reported_prompt: int, reported_completion: int, prompt_bytes: int, completion_bytes: int, max_tokens: int | None
) -> tuple[int, int, int]:
    """Anti-inflation metering: returns ``(prompt_tokens, completion_tokens, billed_tokens)``.

    A token is at least one UTF-8 byte, so the prompt part is capped by ``PROMPT_TOKEN_FACTOR`` times the 4-bytes-per-token
    estimate (and never more than the prompt bytes) and the completion part by the completion bytes and by
    ``max_tokens`` (``DEFAULT_MAX_TOKENS`` when the consumer set none, so billing stays inside the checked reserve).
    Hence ``billed <= min(reported_prompt + reported_completion, prompt_bytes + completion_bytes)``.
    """
    prompt_cap = min(int(prompt_bytes), estimate_tokens(int(prompt_bytes)) * PROMPT_TOKEN_FACTOR)
    prompt = max(0, min(int(reported_prompt), prompt_cap))
    completion = max(0, min(int(reported_completion), int(completion_bytes)))
    completion = min(completion, max(0, DEFAULT_MAX_TOKENS if max_tokens is None else int(max_tokens)))
    return prompt, completion, prompt + completion


def _part_text(part: object) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        text = part.get("text")
        return text if isinstance(text, str) else json.dumps(part, sort_keys=True, separators=(",", ":"))
    return ""


def content_text(content: object) -> str:
    """Flatten an OpenAI message ``content`` (string or list of parts) to text."""
    if isinstance(content, list):
        return "".join(_part_text(p) for p in content)
    return _part_text(content)


# Only these top-level request fields reach a producer. Anything else (legacy functions, vendor extras such as
# documents or chat_template, response_format schemas) could carry unmetered prompt material, so it is dropped.
FORWARDED_REQUEST_FIELDS: frozenset[str] = frozenset({
    "model", "messages", "stream", "stream_options", "temperature", "top_p", "stop", "seed", "tools", "tool_choice",
    "presence_penalty", "frequency_penalty", "max_tokens", "max_completion_tokens", "user", "options",
})


# Reasoning models (the moe-general tier) think before they answer and the thinking counts against max_tokens, so a consumer
# must be able to ask for less of it. Only these exact values are forwarded.
REASONING_EFFORTS: frozenset[str] = frozenset({"low", "medium", "high"})


def prompt_byte_count(body: Mapping[str, Any]) -> int:
    """UTF-8 bytes of everything the consumer sent as prompt: message contents, tool calls and tool definitions."""
    total = 0
    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            total += len(content_text(message.get("content")).encode("utf-8"))
            if message.get("tool_calls"):
                total += len(json.dumps(message["tool_calls"], sort_keys=True).encode("utf-8"))
    if body.get("tools"):
        total += len(json.dumps(body["tools"], sort_keys=True).encode("utf-8"))
    return total


def _message_text(message: object) -> str:
    """Text generated by the model in a response ``message`` or a stream ``delta``."""
    if not isinstance(message, dict):
        return ""
    pieces = [content_text(message.get("content"))]
    for key in ("reasoning_content", "reasoning"):
        value = message.get(key)
        if isinstance(value, str):
            pieces.append(value)
    calls = message.get("tool_calls")
    if isinstance(calls, list):
        for call in calls:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict):
                for key in ("name", "arguments"):
                    value = function.get(key)
                    if isinstance(value, str):
                        pieces.append(value)
    return "".join(pieces)


def completion_byte_count(payload: Mapping[str, Any], *, delta: bool) -> int:
    """UTF-8 bytes generated in a chat completion (``delta=False``) or in one stream chunk (``delta=True``)."""
    total = 0
    choices = payload.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                total += len(_message_text(choice.get("delta" if delta else "message")).encode("utf-8"))
                if not delta and isinstance(choice.get("text"), str):
                    total += len(choice["text"].encode("utf-8"))
    return total


def has_completion_choice(payload: Mapping[str, Any]) -> bool:
    """True when the payload carries at least one choice with a message (or text) and a finish_reason."""
    choices = payload.get("choices")
    if not isinstance(choices, list):
        return False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        has_body = isinstance(choice.get("message"), dict) or isinstance(choice.get("text"), str)
        if has_body and choice.get("finish_reason") is not None:
            return True
    return False


def request_token_limit(body: Mapping[str, Any]) -> int:
    """The single output limit billing and forwarding both use: the minimum of every limit field present.

    ``None`` (or a missing field) everywhere means ``DEFAULT_MAX_TOKENS``; a present field that is null, boolean,
    non-integer or below 1 is rejected, so no field can raise the producer's limit above the billed one.
    """
    limits: list[int] = []
    for name in LIMIT_FIELDS:
        if name in body:
            value = body[name]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise RouterError(400, f"'{name}' must be a positive integer.", "invalid_request_error", "invalid_request")
            limits.append(value)
    for name in ("n", "best_of"):
        value = body.get(name)
        if value is not None and (isinstance(value, bool) or value != 1):
            raise RouterError(400, f"'{name}' other than 1 is not supported.", "invalid_request_error", "invalid_request")
    return min(limits) if limits else DEFAULT_MAX_TOKENS


class UpstreamLimitError(Exception):
    """A producer exceeded the response size, line length or wall-clock budget."""


def read_limited(response: httpx.Response, *, max_bytes: int, deadline: float) -> bytes:
    """Read a whole producer body, aborting above ``max_bytes`` or after the monotonic ``deadline``."""
    chunks: list[bytes] = []
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise UpstreamLimitError("producer response too large")
        if time.monotonic() > deadline:
            raise UpstreamLimitError("producer response too slow")
        chunks.append(chunk)
    return b"".join(chunks)


def limited_lines(response: httpx.Response, *, max_bytes: int, max_line: int, deadline: float) -> Iterator[str]:
    """Yield decoded lines of a producer stream; abort on an over-long line, too many bytes or a missed deadline."""
    buffer = b""
    total = 0
    for chunk in response.iter_bytes():
        total += len(chunk)
        if total > max_bytes:
            raise UpstreamLimitError("producer stream too large")
        if time.monotonic() > deadline:
            raise UpstreamLimitError("producer stream too slow")
        buffer += chunk
        while True:
            newline = buffer.find(b"\n")
            if newline < 0:
                break
            line, buffer = buffer[:newline], buffer[newline + 1 :]
            if len(line) > max_line:
                raise UpstreamLimitError("producer stream line too long")
            yield line.decode("utf-8", errors="replace").rstrip("\r")
        if len(buffer) > max_line:
            raise UpstreamLimitError("producer stream line too long")
    if buffer:
        yield buffer.decode("utf-8", errors="replace").rstrip("\r")


def _usage_pair(payload: Mapping[str, Any]) -> tuple[int, int] | None:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None
    prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if isinstance(prompt, bool) or isinstance(completion, bool):
        return None
    if isinstance(prompt, (int, float)) and isinstance(completion, (int, float)) and prompt >= 0 and completion >= 0:
        return int(prompt), int(completion)
    return None


def _default_resolver(host: str) -> list[str]:
    return sorted({str(info[4][0]) for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)})


_NAT64_PREFIX = ipaddress.ip_network("64:ff9b::/96")
_NAT64_LOCAL_PREFIX = ipaddress.ip_network("64:ff9b:1::/48")


def _embedded_ipv4(address: ipaddress.IPv6Address) -> ipaddress.IPv4Address | None:
    """The IPv4 address an IPv6 form carries (mapped, NAT64, IPv4-compatible, 6to4, Teredo server), if any."""
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address in _NAT64_PREFIX:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    if address.sixtofour is not None:
        return address.sixtofour
    if address.teredo is not None:
        return address.teredo[1]
    if int(address) >> 32 == 0 and int(address) > 1:  # ::a.b.c.d (deprecated IPv4-compatible)
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def _ip_is_global(text: str) -> bool:
    """True only for a public address; IPv6 forms that smuggle an IPv4 address are judged by that IPv4 address."""
    address = ipaddress.ip_address(text.split("%", 1)[0])
    if isinstance(address, ipaddress.IPv6Address):
        if address in _NAT64_LOCAL_PREFIX:
            return False
        embedded = _embedded_ipv4(address)
        if embedded is not None:
            return embedded.is_global
    return address.is_global


def producer_url_problem(url: str, *, allow_private: bool, resolver: Resolver = _default_resolver) -> str | None:
    """Return a reason the producer URL is unacceptable, or ``None``. ``allow_private`` skips every check."""
    if allow_private:
        return None
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return "endpoint_url is not a valid URL"
    if parts.scheme != "https":
        return "endpoint_url must use https"
    host = parts.hostname
    if not host or parts.username is not None or parts.password is not None:
        return "endpoint_url must have a host and no credentials"
    if port is not None and not 0 < port < 65536:
        return "endpoint_url has an invalid port"
    try:
        addresses = [host] if _is_ip_literal(host) else list(resolver(host))
    except OSError:
        return "endpoint_url host does not resolve"
    if not addresses:
        return "endpoint_url host does not resolve"
    for address in addresses:
        try:
            if not _ip_is_global(address):
                return "endpoint_url resolves to a non-public address"
        except ValueError:
            return "endpoint_url resolved to an invalid address"
    return None


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        return False
    return True


class WeightedRoundRobin:
    """Deterministic smooth weighted round-robin (the nginx algorithm), one rotation per key."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current: dict[str, dict[str, float]] = {}

    def order(self, key: str, weights: Mapping[str, float]) -> list[str]:
        """Candidates in preference order: the rotation's pick first, the others by descending credit."""
        ids = sorted(weights)
        if not ids:
            return []
        with self._lock:
            state = {i: self._current.get(key, {}).get(i, 0.0) for i in ids}
            total = 0.0
            for i in ids:
                weight = max(float(weights[i]), 1e-12)
                state[i] += weight
                total += weight
            pick = max(ids, key=lambda i: (state[i], -ids.index(i)))
            state[pick] -= total
            self._current[key] = state
            rest = sorted((i for i in ids if i != pick), key=lambda i: (-state[i], i))
        return [pick, *rest]


class SlidingWindowLimiter:
    """In-memory ``limit`` events per ``window_s`` per key."""

    def __init__(self, limit: int, window_s: float) -> None:
        self._limit = limit
        self._window = window_s
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = {}

    def allow(self, key: str, now: float) -> bool:
        with self._lock:
            queue = self._events.setdefault(key, deque())
            while queue and queue[0] <= now - self._window:
                queue.popleft()
            if len(queue) >= self._limit:
                return False
            queue.append(now)
            if len(self._events) > 10_000:
                for stale in [k for k, q in self._events.items() if not q or q[-1] <= now - self._window]:
                    del self._events[stale]
            return True


# ---------------------------------------------------------------------------------------------------------
# request models
# ---------------------------------------------------------------------------------------------------------


class RegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    handle: str = Field(min_length=1, max_length=128)
    role: Literal["consumer", "producer", "prosumer"]
    test: bool = False  # test/probe account: needs the admin key, never bids in the market, counted apart from real users


class GoogleRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_token: str = Field(min_length=1, max_length=4096)
    role: Literal["consumer", "producer", "prosumer"] = "prosumer"


class MaxPriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_price_per_1k: float = Field(gt=0.0, le=MAX_PRICE_CEILING, allow_inf_nan=False)


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    local_name: str = Field(min_length=1, max_length=256)


class HeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint_url: str = Field(min_length=1, max_length=512)
    models: list[ModelEntry] = Field(min_length=1, max_length=64)
    cost_per_1k: float = Field(ge=0.0, le=MAX_PRICE_CEILING, allow_inf_nan=False)
    capacity_1k: float = Field(ge=0.0, le=1e12, allow_inf_nan=False)
    execution_class: str = Field(default="unknown", max_length=64)
    # Secret the gateway presents to the producer's own server (``Authorization: Bearer``) so that only the gateway can
    # use it. Optional for compatibility; without it the producer's endpoint is open to anyone who learns its URL.
    endpoint_token: str | None = Field(default=None, min_length=16, max_length=200, pattern=r"^[A-Za-z0-9._~+/=-]+$")


class GrantRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str | None = Field(default=None, max_length=128)
    handle: str | None = Field(default=None, max_length=128)
    amount: float = Field(gt=0.0, le=1e12, allow_inf_nan=False)
    ref: str = Field(default="admin-grant", min_length=1, max_length=256)


class AccountFlagRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str | None = Field(default=None, max_length=128)
    handle: str | None = Field(default=None, max_length=128)
    test: bool


@dataclass(frozen=True)
class _Plan:
    """Everything the router decided about one chat request before contacting a producer."""

    request_id: str
    model_id: str
    price_per_1k: float
    mechanism: str
    candidates: tuple[ProducerRecord, ...]
    prompt_bytes: int
    max_tokens: int


def _public_model(model_id: str) -> str:
    return MODEL_PREFIX + model_id


def _strip_prefix(model: str) -> str:
    return normalize_model_alias(model.removeprefix(MODEL_PREFIX))


def _sse(payload: Mapping[str, Any]) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n\n"


ASGIMessage = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[ASGIMessage]]
Send = Callable[[ASGIMessage], Awaitable[None]]


class BodyLimitMiddleware:
    """Pure ASGI middleware: refuse request bodies above ``max_bytes`` with 413 before any routing or JSON parsing.

    The declared ``Content-Length`` is checked first; chunked bodies (or a lying header) are counted while they are
    received, so memory use is bounded by ``max_bytes`` whatever the client claims.
    """

    def __init__(self, app: Callable[[ASGIMessage, Receive, Send], Awaitable[None]], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: ASGIMessage, receive: Receive, send: Send) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        for name, value in scope.get("headers", []):
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                await self._reject(scope, receive, send)
                return
        buffered: list[ASGIMessage] = []
        total = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                break
            total += len(message.get("body", b""))
            if total > self.max_bytes:
                await self._reject(scope, receive, send)
                return
            if not message.get("more_body", False):
                break
        replay = iter(buffered)

        async def replayed() -> ASGIMessage:
            try:
                return next(replay)
            except StopIteration:
                return await receive()

        await self.app(scope, replayed, send)

    @staticmethod
    async def _reject(scope: ASGIMessage, receive: Receive, send: Send) -> None:
        response = _error_response(413, "Request body too large.", "invalid_request_error", "request_too_large")
        await response(scope, receive, send)


def create_router_app(
    store: RouterStore,
    *,
    http_client: httpx.Client | None = None,
    fee: FeePolicy = FeePolicy(),  # noqa: B008 - frozen dataclass, the spec's signature fixes this default
    allow_private_producers: bool = False,
    admin_key_hash: str | None = None,
    clock: Clock = time.time,
    resolver: Resolver | None = None,
    trusted_proxy_hops: int = 0,
    max_producer_capacity_1k: float = MAX_PRODUCER_CAPACITY_1K,
    google_client_id: str | None = None,
    google_verifier: GoogleIdTokenVerifier | None = None,
    oidc_verifiers: Mapping[str, OidcIdTokenVerifier] | None = None,
    open_registration: bool = True,
    bid_activity_window_s: float = 0.0,
) -> FastAPI:
    """Build the gateway app. ``resolver`` (host -> IP strings) is injectable for tests of the SSRF guard.

    ``trusted_proxy_hops`` is the number of reverse proxies in front of the app that append to ``X-Forwarded-For``
    (0 = use the TCP peer; 1 on Cloud Run). The rate limiter keys on the entry that many hops from the right, which a
    client cannot forge. ``max_producer_capacity_1k`` is the operator clamp on self-declared producer capacity.

    ``google_client_id`` turns on "Sign in with Google" registration (``/signup`` and ``/v1/hivemind/register/google``);
    without it both are disabled. ``google_verifier`` lets tests inject keys. ``open_registration=False`` closes the
    handle-only ``/v1/hivemind/register`` so that Google sign-in is the only way in.
    """
    client = http_client if http_client is not None else httpx.Client(timeout=PRODUCER_TIMEOUT_S, follow_redirects=False)
    resolve: Resolver = resolver if resolver is not None else _default_resolver
    app = FastAPI(title="HiveMind OpenRouter", version="1.0.0")
    rotation = WeightedRoundRobin()
    register_limiter = SlidingWindowLimiter(REGISTER_LIMIT, REGISTER_WINDOW_S)
    market_cache: dict[tuple[str, int], dict[str, Any]] = {}
    market_lock = threading.Lock()
    capacity_clamp = float(max_producer_capacity_1k)
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    # ------------------------------------------------------------------ errors
    @app.exception_handler(RouterError)
    def _router_error(_: Request, exc: RouterError) -> JSONResponse:
        return _error_response(exc.status, exc.message, exc.error_type, exc.code)

    @app.exception_handler(RequestValidationError)
    def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        first = exc.errors()[0] if exc.errors() else {}
        location = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
        return _error_response(400, f"invalid request: {location or 'body'}: {first.get('msg', 'invalid')}", "invalid_request_error", "invalid_request")

    @app.exception_handler(StarletteHTTPException)
    def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        kind = "not_found_error" if exc.status_code == 404 else "invalid_request_error"
        return _error_response(exc.status_code, str(exc.detail), kind, f"http_{exc.status_code}")

    # ------------------------------------------------------------------ helpers
    def authenticate(request: Request) -> Account:
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        account = store.authenticate(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
        if account is None:
            raise RouterError(401, "Invalid or missing API key. Send 'Authorization: Bearer hm-...'.", "authentication_error", "invalid_api_key")
        return account

    def require_admin(request: Request) -> None:
        supplied = request.headers.get("x-admin-key", "")
        if admin_key_hash is None or not supplied or not hmac.compare_digest(store.key_digest(supplied), admin_key_hash):
            raise RouterError(403, "Admin key required.", "permission_error", "admin_forbidden")

    def account_target(account_id: str | None, handle: str | None) -> str:
        try:
            target = account_id or (store.account_id_for(handle) if handle else None)
        except SwarmSecurityViolationError:
            raise RouterError(400, "Invalid handle.", "invalid_request_error", "invalid_handle") from None
        if target is None:
            raise RouterError(400, "Give 'account_id' or 'handle'.", "invalid_request_error", "invalid_request")
        return target

    def clamped(record: ProducerRecord) -> float:
        return min(record.capacity_1k, capacity_clamp)

    def offers_for(records: Sequence[ProducerRecord]) -> list[ProducerOffer]:
        return [ProducerOffer(r.account_id, r.cost_per_1k, clamped(r)) for r in records]

    def bids_for(extra: Account | None) -> list[ConsumerBid]:
        """Bids of funded accounts only, each capped by what its balance can pay for one default-sized request."""
        active_since = clock() - bid_activity_window_s if bid_activity_window_s > 0.0 else None
        prices = store.funded_consumer_bids(DEFAULT_MAX_TOKENS, active_since=active_since)
        if extra is not None and extra.role != "producer" and not extra.is_test and extra.account_id not in prices:
            balance = store.balance(extra.account_id)
            if balance > 0.0:
                prices[extra.account_id] = min(extra.max_price_per_1k, balance * 1000.0 / DEFAULT_MAX_TOKENS)
        return [ConsumerBid(a, p) for a, p in prices.items()]

    def client_key(request: Request) -> str:
        """Rate-limit key: the TCP peer, or the XFF entry ``trusted_proxy_hops`` from the right when behind proxies."""
        if trusted_proxy_hops > 0:
            hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
            if len(hops) >= trusted_proxy_hops:
                return hops[-trusted_proxy_hops]
        return request.client.host if request.client is not None else "unknown"

    def equilibrium(model_id: str, records: Sequence[ProducerRecord], extra: Account | None) -> MarketEquilibrium:
        return clear_market(model_id, bids_for(extra), offers_for(records), fee)

    def resolve_model(requested: str) -> tuple[str, list[ProducerRecord]]:
        now = clock()
        name = _strip_prefix(requested.strip())
        if name == AUTO_MODEL:
            best: tuple[tuple[int, float], str] | None = None
            for candidate in store.live_models(now, PRODUCER_TTL_S):
                live = store.active_producers(candidate, now, PRODUCER_TTL_S)
                # distinct live producers first: a single producer cannot steer 'auto' by declaring huge capacity
                supply = (len({r.account_id for r in live}), sum(clamped(r) for r in live))
                if best is None or supply > best[0]:
                    best = (supply, candidate)
            if best is None:
                raise RouterError(503, "No HiveMind producer is currently live.", "service_unavailable", "no_live_producers")
            name = best[1]
        records = store.active_producers(name, now, PRODUCER_TTL_S)
        if not records:
            raise RouterError(404, f"The model '{requested}' is not served by any live producer.", "invalid_request_error", "model_not_found")
        return name, records

    def plan_request(account: Account, body: Mapping[str, Any]) -> _Plan:
        requested = body.get("model")
        messages = body.get("messages")
        if not isinstance(requested, str) or not requested.strip():
            raise RouterError(400, "'model' is required.", "invalid_request_error", "invalid_request")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise RouterError(400, "'messages' must be a non-empty list of message objects.", "invalid_request_error", "invalid_request")
        max_tokens = request_token_limit(body)
        model_id, records = resolve_model(requested)
        if store.balance(account.account_id) <= 0.0:
            raise RouterError(402, "Balance is empty; ask an admin for a grant or earn HMT as a producer.", "insufficient_quota", "insufficient_balance")
        eq = equilibrium(model_id, records, account)
        if eq.mechanism == MECHANISM_NO_TRADE and eq.price_per_1k <= 0.0:
            raise RouterError(503, f"No capacity is offered for '{requested}' right now.", "service_unavailable", "no_capacity")
        if account.max_price_per_1k < eq.price_per_1k:
            raise RouterError(
                402,
                f"Current price {eq.price_per_1k:.6f} HMT/1k tokens exceeds your max price {account.max_price_per_1k:.6f}. "
                "Raise it with POST /v1/hivemind/consumer/max-price.",
                "insufficient_quota",
                "price_above_max_price",
            )
        prompt_bytes = prompt_byte_count(body)
        reserve = eq.price_per_1k * (prompt_bytes + max_tokens) / 1000.0
        if store.balance(account.account_id) < reserve:
            # a requester too poor for the reserve is told so, not that the market has no capacity for its (capped) bid
            raise RouterError(
                402,
                f"Balance {store.balance(account.account_id):.6f} HMT is below the {reserve:.6f} HMT reserve for this request.",
                "insufficient_quota",
                "insufficient_balance",
            )
        if eq.mechanism == MECHANISM_NO_TRADE or not eq.active_producers:
            raise RouterError(503, f"No producer is willing to serve '{requested}' at the current price.", "service_unavailable", "no_capacity")
        request_id = "req-" + uuid.uuid4().hex
        try:
            store.hold(request_id, account.account_id, reserve, now_utc=clock())
        except InsufficientBalance:
            raise RouterError(
                402,
                f"Balance {store.balance(account.account_id):.6f} HMT is below the {reserve:.6f} HMT reserve for this request.",
                "insufficient_quota",
                "insufficient_balance",
            ) from None
        except TooManyHolds:
            raise RouterError(429, "Too many requests in flight for this account.", "rate_limit_error", "too_many_inflight") from None
        by_id = {r.account_id: r for r in records}
        weights = {a: clamped(by_id[a]) for a in eq.active_producers if a in by_id}
        order = rotation.order(model_id, weights)
        candidates: list[ProducerRecord] = []
        for a in order:
            servers = by_id[a].per_endpoint()
            if len(servers) > 1:  # one account, several servers: spread load over them by capacity, keep the rest as fallbacks
                by_url = {s.endpoint_url: s for s in servers}
                shares = {u: min(s.capacity_1k, capacity_clamp) for u, s in by_url.items()}
                candidates.extend(by_url[u] for u in rotation.order(f"{model_id}|{a}", shares))
            else:
                candidates.append(servers[0])
        return _Plan(
            request_id=request_id,
            model_id=model_id,
            price_per_1k=eq.price_per_1k,
            mechanism=eq.mechanism,
            candidates=tuple(candidates),
            prompt_bytes=prompt_bytes,
            max_tokens=max_tokens,
        )

    def producer_request(record: ProducerRecord, plan: _Plan, body: Mapping[str, Any], *, stream: bool) -> httpx.Request | None:
        if producer_url_problem(record.endpoint_url, allow_private=allow_private_producers, resolver=resolve) is not None:
            LOGGER.warning("producer %s skipped: endpoint failed the SSRF re-check", record.account_id)
            return None
        local = record.local_name(plan.model_id)
        if local is None:
            return None
        outgoing = {k: v for k, v in body.items() if k in FORWARDED_REQUEST_FIELDS}
        effort = body.get("reasoning_effort")
        if isinstance(effort, str) and effort in REASONING_EFFORTS:  # an enum, so it cannot carry prompt material; anything else is dropped
            outgoing["reasoning_effort"] = effort
        outgoing["model"] = local
        outgoing["max_tokens"] = plan.max_tokens
        outgoing["max_completion_tokens"] = plan.max_tokens
        outgoing.pop("max_output_tokens", None)
        options = outgoing.get("options")
        if isinstance(options, dict):
            outgoing["options"] = {k: v for k, v in options.items() if k not in ("num_predict", "max_tokens")}
        if stream:
            options = outgoing.get("stream_options")
            merged = dict(options) if isinstance(options, dict) else {}
            merged["include_usage"] = True
            outgoing["stream_options"] = merged
        url = record.endpoint_url.rstrip("/") + "/v1/chat/completions"
        headers = {"Accept": "text/event-stream" if stream else "application/json"}
        if record.endpoint_token:
            headers["Authorization"] = f"Bearer {record.endpoint_token}"
        return client.build_request("POST", url, json=outgoing, headers=headers)

    def settle(plan: _Plan, account: Account, producer_id: str, reported: tuple[int, int], completion_bytes: int) -> dict[str, Any]:
        """Settle against the escrow taken at plan time: the metered charge can never exceed it, so it is never free."""
        prompt, completion, billed = meter_usage(reported[0], reported[1], plan.prompt_bytes, completion_bytes, plan.max_tokens)
        receipt = store.settle(plan.request_id, account.account_id, producer_id, plan.model_id, prompt, completion, billed, plan.price_per_1k, fee)
        billed = receipt.billed_tokens
        prompt = min(prompt, billed)
        completion = billed - prompt
        return {
            "price_per_1k": plan.price_per_1k,
            "charged_hmt": receipt.charge,
            "producer_id": producer_id,
            "mechanism": plan.mechanism,
            "request_id": plan.request_id,
            "billed_tokens": billed,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
        }

    def chat_blocking(plan: _Plan, account: Account, body: Mapping[str, Any]) -> JSONResponse:
        for record in plan.candidates:
            request = producer_request(record, plan, body, stream=False)
            if request is None:
                continue
            payload: Any = None
            try:
                upstream = client.send(request, stream=True)
                try:
                    if upstream.status_code == 200:
                        raw = read_limited(upstream, max_bytes=MAX_PRODUCER_RESPONSE_BYTES, deadline=time.monotonic() + PRODUCER_DEADLINE_S)
                        payload = json.loads(raw)
                finally:
                    upstream.close()
            except (httpx.HTTPError, ValueError, UpstreamLimitError):
                payload = None
            if not isinstance(payload, dict) or not has_completion_choice(payload):
                LOGGER.warning("producer %s failed request %s", record.account_id, plan.request_id)
                continue
            reported = _usage_pair(payload)
            cbytes = completion_byte_count(payload, delta=False)
            if reported is None:
                reported = (estimate_tokens(plan.prompt_bytes), estimate_tokens(cbytes))
            try:
                info = settle(plan, account, record.account_id, reported, cbytes)
            except (InsufficientBalance, DuplicateRequest, UnknownAccount):
                raise RouterError(402, "Balance changed while the request was running.", "insufficient_quota", "insufficient_balance") from None
            payload["model"] = _public_model(plan.model_id)
            payload["usage"] = {
                "prompt_tokens": info["prompt_tokens"],
                "completion_tokens": info["completion_tokens"],
                "total_tokens": info["billed_tokens"],
            }
            payload["hivemind"] = {k: info[k] for k in ("price_per_1k", "charged_hmt", "producer_id", "mechanism", "request_id")}
            return JSONResponse(payload)
        raise RouterError(502, "Every producer for this model failed; you were not charged.", "api_error", "upstream_failure")

    def chat_streaming(plan: _Plan, account: Account, body: Mapping[str, Any]) -> StreamingResponse:
        for record in plan.candidates:
            request = producer_request(record, plan, body, stream=True)
            if request is None:
                continue
            try:
                upstream = client.send(request, stream=True)
            except httpx.HTTPError:
                LOGGER.warning("producer %s failed request %s", record.account_id, plan.request_id)
                continue
            if upstream.status_code != 200:
                upstream.close()
                LOGGER.warning("producer %s answered %s for %s", record.account_id, upstream.status_code, plan.request_id)
                continue
            return StreamingResponse(_stream_body(plan, account, record.account_id, upstream), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        raise RouterError(502, "Every producer for this model failed; you were not charged.", "api_error", "upstream_failure")

    def _stream_body(plan: _Plan, account: Account, producer_id: str, upstream: httpx.Response) -> Iterator[bytes]:
        public = _public_model(plan.model_id)
        completion_bytes = 0
        reported: tuple[int, int] | None = None
        completion_id = "chatcmpl-" + plan.request_id
        settled = False
        failed = False
        produced = False

        def finish() -> dict[str, Any] | None:
            nonlocal settled
            settled = True
            usage = reported if reported is not None else (estimate_tokens(plan.prompt_bytes), estimate_tokens(completion_bytes))
            if not produced:
                usage = (0, 0)  # no choice chunk ever arrived: the producer did no work, so nothing is billed
            try:
                return settle(plan, account, producer_id, usage, completion_bytes)
            except (InsufficientBalance, DuplicateRequest, UnknownAccount):
                LOGGER.warning("settlement failed for %s", plan.request_id)
                return None

        try:
            try:
                deadline = time.monotonic() + PRODUCER_DEADLINE_S
                for line in limited_lines(upstream, max_bytes=MAX_PRODUCER_RESPONSE_BYTES, max_line=MAX_SSE_LINE_BYTES, deadline=deadline):
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except ValueError:
                        continue
                    if not isinstance(chunk, dict):
                        continue
                    if isinstance(chunk.get("choices"), list) and chunk["choices"]:
                        produced = True
                    completion_id = str(chunk.get("id", completion_id))
                    completion_bytes += completion_byte_count(chunk, delta=True)
                    pair = _usage_pair(chunk)
                    if pair is not None:
                        reported = pair
                    chunk["model"] = public
                    yield _sse(chunk)
            except (httpx.HTTPError, UpstreamLimitError):
                failed = True
            info = finish()
            if failed:
                yield _sse({"error": {"message": "The producer connection broke mid-stream.", "type": "api_error", "code": "upstream_failure"}})
            summary = (
                {k: info[k] for k in ("price_per_1k", "charged_hmt", "producer_id", "mechanism", "request_id")}
                if info is not None
                else {"price_per_1k": plan.price_per_1k, "charged_hmt": 0.0, "producer_id": producer_id, "mechanism": plan.mechanism, "request_id": plan.request_id}
            )
            yield _sse({"id": completion_id, "object": "chat.completion.chunk", "created": int(clock()), "model": public, "choices": [], "hivemind": summary})
            yield b"data: [DONE]\n\n"
        finally:
            upstream.close()
            try:
                if not settled:
                    finish()
            finally:
                store.release_hold(plan.request_id)

    # ------------------------------------------------------------------ endpoints
    @app.api_route("/health", methods=["GET", "HEAD"])  # Cloud Run's front end swallows /healthz, so the same probe is also served here
    @app.api_route("/healthz", methods=["GET", "HEAD"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def portal() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        page = render_portal_page(google.client_id if google else None, nonce)
        return HTMLResponse(
            page,
            headers={
                "Content-Security-Policy": portal_content_security_policy(nonce),
                "Cache-Control": "public, max-age=60",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "strict-origin-when-cross-origin",
            },
        )

    @app.get("/v1/hivemind/catalog")
    def catalog() -> dict[str, Any]:
        """Public list of capability tiers (which model fits which GPU); static data, no account needed."""
        return catalog_payload()

    @app.get("/v1/hivemind/partner-models")
    def partner_models() -> dict[str, Any]:
        """Pre-wired federated partner models (e.g. CrowdGPT Vxtzq/Crowd-v1) ready for producer & Playground routing."""
        now = clock()
        live_set = set(store.live_models(now, PRODUCER_TTL_S))
        items = []
        for partner in PREWIRED_PARTNER_MODELS:
            entry = partner.to_dict()
            entry["live_on_swarm"] = partner.tier_id in live_set
            items.append(entry)
        return {"object": "hivemind.partner_models", "models": items}

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        created = int(clock())
        ids = [_public_model(AUTO_MODEL), *(_public_model(m) for m in store.live_models(clock(), PRODUCER_TTL_S))]
        return {"object": "list", "data": [{"id": i, "object": "model", "created": created, "owned_by": "hivemind"} for i in ids]}

    def handle_chat(account: Account, raw: bytes) -> JSONResponse | StreamingResponse:
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request") from None
        if not isinstance(body, dict):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request")
        plan = plan_request(account, body)  # takes the escrow hold
        if body.get("stream") is True:
            try:
                return chat_streaming(plan, account, body)
            except BaseException:
                store.release_hold(plan.request_id)
                raise
        try:
            return chat_blocking(plan, account, body)
        finally:
            store.release_hold(plan.request_id)

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        # authenticate first, then read the body (already bounded by BodyLimitMiddleware): anonymous callers cost nothing
        account = await run_in_threadpool(authenticate, request)
        raw = await request.body()
        return await run_in_threadpool(handle_chat, account, raw)

    google = google_verifier or (GoogleIdTokenVerifier(google_client_id, clock=clock) if google_client_id else None)

    @app.api_route("/signup", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def signup() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        page = render_signup_page(google.client_id if google else None, nonce)
        return HTMLResponse(page, headers={"Content-Security-Policy": content_security_policy(nonce), "Cache-Control": "no-store",
                                           "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    @app.post("/v1/hivemind/register/google")
    def register_google(request: Request, payload: GoogleRegisterRequest) -> JSONResponse:
        if google is None:
            raise RouterError(503, "Sign in with Google is not enabled on this gateway.", "service_unavailable", "google_signin_disabled")
        if not register_limiter.allow(client_key(request), clock()):
            raise RouterError(429, "Too many sign-ins; try again in a minute.", "rate_limit_error", "rate_limited")
        try:
            identity = google.verify(payload.id_token)
        except GoogleAuthError as exc:
            if exc.code == "keys_unavailable":
                raise RouterError(503, "Could not reach Google to check the sign-in; try again.", "service_unavailable", "google_unreachable") from None
            if exc.code == "busy":
                raise RouterError(503, "Too many sign-ins in progress; try again.", "service_unavailable", "busy") from None
            LOGGER.info("google sign-in refused: %s", exc.code)  # reason code only: never the token
            raise RouterError(401, "Google sign-in failed. Sign in again.", "authentication_error", f"google_{exc.code}") from None
        account_id, api_key, created = store.sign_in_with_identity("google", identity.subject, payload.role)
        account = store.get_account(account_id)
        role = account.role if account is not None else payload.role
        return JSONResponse({"account_id": account_id, "api_key": api_key, "role": role, "created": created},
                            headers={"Cache-Control": "no-store"})

    oidc = dict(oidc_verifiers or {})
    for name in oidc:
        if name == "google":
            raise ValueError("configure Google with google_client_id; 'google' is not an extra OIDC provider")
        store.allow_identity_provider(name)

    @app.get("/v1/hivemind/signin/providers")
    def signin_providers() -> dict[str, Any]:
        """Which sign-in methods this gateway accepts (public). Clients send a provider ID token to the matching endpoint."""
        providers = ([{"name": "google", "register": "/v1/hivemind/register/google", "client_id": google.client_id}] if google else [])
        providers += [{"name": n, "register": f"/v1/hivemind/register/oidc/{n}", "client_id": v.client_id} for n, v in sorted(oidc.items())]
        return {"object": "hivemind.signin_providers", "providers": providers, "handle_registration": open_registration}

    @app.post("/v1/hivemind/register/oidc/{provider}")
    def register_oidc(provider: str, request: Request, payload: GoogleRegisterRequest) -> JSONResponse:
        verifier = oidc.get(provider)
        if verifier is None:
            raise RouterError(404, "Unknown sign-in provider.", "invalid_request_error", "unknown_provider")
        if not register_limiter.allow(client_key(request), clock()):
            raise RouterError(429, "Too many sign-ins; try again in a minute.", "rate_limit_error", "rate_limited")
        try:
            identity = verifier.verify(payload.id_token)
        except OidcAuthError as exc:
            if exc.code == "keys_unavailable":
                raise RouterError(503, "Could not reach the identity provider to check the sign-in; try again.", "service_unavailable", "provider_unreachable") from None
            if exc.code == "busy":
                raise RouterError(503, "Too many sign-ins in progress; try again.", "service_unavailable", "busy") from None
            LOGGER.info("%s sign-in refused: %s", provider, exc.code)  # reason code only: never the token
            raise RouterError(401, "Sign-in failed. Sign in again.", "authentication_error", f"{provider}_{exc.code}") from None
        account_id, api_key, created = store.sign_in_with_identity(provider, identity.subject, payload.role)
        account = store.get_account(account_id)
        role = account.role if account is not None else payload.role
        return JSONResponse({"account_id": account_id, "api_key": api_key, "role": role, "created": created},
                            headers={"Cache-Control": "no-store"})

    @app.post("/v1/hivemind/register")
    def register(request: Request, payload: RegisterRequest) -> dict[str, Any]:
        if not open_registration:
            raise RouterError(403, "Handle registration is closed. Sign in at /signup or see /v1/hivemind/signin/providers.", "permission_error", "registration_closed")
        if payload.test:
            require_admin(request)  # refused, never silently ignored: a caller that asked for a test account must learn it did not get one
        if not register_limiter.allow(client_key(request), clock()):
            raise RouterError(429, "Too many registrations; try again in a minute.", "rate_limit_error", "rate_limited")
        try:
            account_id, api_key = store.register(payload.handle, payload.role, is_test=payload.test)
        except HandleTaken:
            raise RouterError(409, "That handle is already registered.", "invalid_request_error", "handle_taken") from None
        except (SwarmSecurityViolationError, ReservedHandle):
            raise RouterError(400, "Handle rejected: use 1-64 ASCII letters, digits and . _ @ + - characters, not starting with a sign-in provider prefix such as 'google.'.", "invalid_request_error", "invalid_handle") from None
        reply: dict[str, Any] = {"account_id": account_id, "api_key": api_key, "role": payload.role}
        if payload.test:
            reply["test"] = True
        return reply

    @app.get("/v1/hivemind/me")
    def me(request: Request) -> dict[str, Any]:
        account = authenticate(request)
        return {
            "account_id": account.account_id,
            "handle": account.handle,
            "role": account.role,
            "balance": store.balance(account.account_id),
            "max_price_per_1k": account.max_price_per_1k,
            "test": account.is_test,
            "recent_usage": store.recent_usage(account.account_id, 20),
        }

    @app.post("/v1/hivemind/consumer/max-price")
    def set_max_price(request: Request, payload: MaxPriceRequest) -> dict[str, Any]:
        account = authenticate(request)
        store.set_max_price(account.account_id, payload.max_price_per_1k)
        return {"account_id": account.account_id, "max_price_per_1k": payload.max_price_per_1k}

    @app.post("/v1/hivemind/producer/heartbeat")
    def heartbeat(request: Request, payload: HeartbeatRequest) -> dict[str, Any]:
        account = authenticate(request)
        if account.role not in ("producer", "prosumer"):
            raise RouterError(403, "Only producer or prosumer accounts may send heartbeats.", "permission_error", "role_forbidden")
        problem = producer_url_problem(payload.endpoint_url, allow_private=allow_private_producers, resolver=resolve)
        if problem is not None:
            raise RouterError(400, problem, "invalid_request_error", "invalid_endpoint")
        models: list[dict[str, str]] = []
        for entry in payload.models:
            public = _strip_prefix(entry.id)
            if not _MODEL_ID_RE.match(public) or public == AUTO_MODEL:
                raise RouterError(400, f"Invalid model id: {entry.id!r}", "invalid_request_error", "invalid_model_id")
            models.append({"id": public, "local_name": entry.local_name})
        try:
            endpoints = store.upsert_producer(
                account.account_id, payload.endpoint_url, models, payload.cost_per_1k, payload.capacity_1k, payload.execution_class,
                now_utc=clock(), endpoint_token=payload.endpoint_token,
            )
        except TooManyEndpoints as exc:
            raise RouterError(400, str(exc), "invalid_request_error", "too_many_endpoints") from None
        return {"account_id": account.account_id, "status": "live", "models": [m["id"] for m in models], "ttl_s": PRODUCER_TTL_S, "endpoints": endpoints}

    @app.get("/v1/hivemind/market")
    def market(request: Request, model: str = AUTO_MODEL) -> dict[str, Any]:
        authenticate(request)
        name, records = resolve_model(model)
        epoch = int(clock() // MARKET_CACHE_EPOCH_S)
        key = (name, epoch)
        with market_lock:
            cached = market_cache.get(key)
            if cached is None:
                # certify() is quadratic in the bidders: compute once per model and epoch, serve the cached result
                bids = bids_for(None)
                offers = offers_for(records)
                eq = clear_market(name, bids, offers, fee)
                cert = certify(eq, bids, offers)
                shape = eq.to_dict()
                # consumer-keyed maps would reveal each account's private max price: publish aggregates only
                shape["consumer_shares"] = sorted(eq.consumer_shares.values(), reverse=True)
                shape["total_consumer_surplus"] = sum(eq.consumer_surplus.values())
                del shape["consumer_surplus"]
                shape["num_consumers"] = len(bids)
                cached = {"model": _public_model(name), "equilibrium": shape, "certificate": cert.to_dict()}
                for stale in [k for k in market_cache if k[1] != epoch]:
                    del market_cache[stale]
                market_cache[key] = cached
        return cached

    @app.post("/v1/hivemind/admin/grant")
    def admin_grant(request: Request, payload: GrantRequest) -> dict[str, Any]:
        require_admin(request)
        target = account_target(payload.account_id, payload.handle)
        try:
            store.grant(target, payload.amount, payload.ref)
        except UnknownAccount:
            raise RouterError(404, "Unknown account.", "invalid_request_error", "account_not_found") from None
        return {"account_id": target, "granted": payload.amount, "balance": store.balance(target)}

    @app.post("/v1/hivemind/admin/test-flag")
    def admin_test_flag(request: Request, payload: AccountFlagRequest) -> dict[str, Any]:
        """Mark an existing account as a test/probe account (or restore it to a real one): it stops (or starts) bidding."""
        require_admin(request)
        target = account_target(payload.account_id, payload.handle)
        try:
            store.set_test_flag(target, payload.test)
        except UnknownAccount:
            raise RouterError(404, "Unknown account.", "invalid_request_error", "account_not_found") from None
        with market_lock:
            market_cache.clear()  # the published market was computed with the old bid set
        return {"account_id": target, "test": payload.test}

    @app.get("/v1/hivemind/admin/users")
    def admin_users(request: Request, window_s: float = 3600.0) -> dict[str, Any]:
        """Real and test accounts counted apart: ``total``, ``created`` and ``active`` within the last ``window_s`` seconds."""
        require_admin(request)
        if not math.isfinite(window_s) or window_s < 0.0:
            raise RouterError(400, "window_s must be a finite number of seconds, zero or more.", "invalid_request_error", "invalid_request")
        return {"window_s": window_s, **store.user_counts(clock() - window_s)}

    return app
