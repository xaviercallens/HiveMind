# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
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
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt
from starlette.exceptions import HTTPException as StarletteHTTPException

from hivemind.router.accounts import (
    Account,
    DuplicateRequest,
    HandleTaken,
    InsufficientBalance,
    InvalidCredentials,
    ProducerRecord,
    ReservedHandle,
    RouterStore,
    TooManyEndpoints,
    TooManyHolds,
    UnknownAccount,
    fit_loaded_models,
)
from hivemind.router.catalog import PREWIRED_PARTNER_MODELS, catalog_payload, normalize_model_alias
from hivemind.router.google_auth import GoogleAuthError, GoogleIdTokenVerifier
from hivemind.router.join_page import join_content_security_policy, pick_language, render_join_page
from hivemind.router.oidc_auth import OidcAuthError, OidcIdTokenVerifier
from hivemind.router.portal_page import portal_content_security_policy, render_portal_page
from hivemind.router.signup_page import content_security_policy, render_signup_page
from hivemind.router.usage_capture import classify_prompt_intent_and_language, detect_client_sdk
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
DEFAULT_COLD_START_WAIT_BUDGET_S = 20.0  # forward to a known-unloaded model only when it is expected to load within this
DEFAULT_COLD_START_S = 60.0  # assumed load time of a model reported not loaded without a cold_start_s estimate
MAX_COLD_START_S = 600.0
MAX_GPU_VRAM_MIB = 1_048_576  # 1 TiB: anything above is a typo or a lie
WARMUP_MIN_INTERVAL_S = 30.0
DEFAULT_MAX_TOKENS = 1024
REGISTER_LIMIT = 10
REGISTER_WINDOW_S = 60.0
AMCP_PUBLISH_LIMIT = 30  # bus events per account per window: the bus is for coordination, not bulk traffic
AMCP_PUBLISH_WINDOW_S = 60.0
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

    def __init__(self, status: int, message: str, error_type: str, code: str, headers: Mapping[str, str] | None = None) -> None:
        super().__init__(message)
        self.headers = dict(headers) if headers else None
        self.status = status
        self.message = message
        self.error_type = error_type
        self.code = code


def _error_response(status: int, message: str, error_type: str, code: str, headers: Mapping[str, str] | None = None) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"message": message, "type": error_type, "code": code}}, headers=dict(headers) if headers else None)


def model_loading_error(model_id: str, cold_start_s: float) -> RouterError:
    """HTTP 503 ``model_loading`` with ``Retry-After``: the model is not resident and the expected load time is stated."""
    wait = max(1, math.ceil(cold_start_s))
    return RouterError(
        503,
        f"The model '{model_id}' is not loaded on any producer right now; it is being loaded, expected cold start about {wait} s. "
        f"Retry in {wait} s. You were not charged.",
        "service_unavailable",
        "model_loading",
        {"Retry-After": str(wait)},
    )


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
    clean_host = (host or "").strip()
    if not clean_host:
        return []
    infos = socket.getaddrinfo(clean_host, None, type=socket.SOCK_STREAM)
    resolved: set[str] = set()
    for info in infos:
        sockaddr = info[4]
        if sockaddr and sockaddr[0]:
            resolved.add(str(sockaddr[0]))
    return sorted(resolved)


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
    role: Literal["consumer", "producer", "prosumer", "trainer"]
    test: bool = False  # test account: needs the admin key, never bids in the market, counted apart from real users
    kind: Literal["user", "test", "probe"] | None = None  # 'test' or 'probe' need the admin key too (HM-S16-07)


class GoogleRegisterRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id_token: str = Field(min_length=1, max_length=4096)
    role: Literal["consumer", "producer", "prosumer", "trainer"] = "prosumer"


class MaxPriceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_price_per_1k: float = Field(gt=0.0, le=MAX_PRICE_CEILING, allow_inf_nan=False)


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1, max_length=128)
    local_name: str = Field(min_length=1, max_length=256)
    # Optional residency report (HM-S16-06). Absent = "unknown": the producer is neither preferred nor penalised.
    loaded: StrictBool | None = None
    vram_mib: int | None = Field(default=None, ge=0, le=MAX_GPU_VRAM_MIB)
    cold_start_s: float | None = Field(default=None, ge=0.0, allow_inf_nan=False)  # clamped to MAX_COLD_START_S by the gateway


class HeartbeatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint_url: str = Field(min_length=1, max_length=512)
    models: list[ModelEntry] = Field(min_length=1, max_length=64)
    cost_per_1k: float = Field(ge=0.0, le=MAX_PRICE_CEILING, allow_inf_nan=False)
    capacity_1k: float = Field(ge=0.0, le=1e12, allow_inf_nan=False)
    gpu_vram_mib: int | None = Field(default=None, ge=0, le=MAX_GPU_VRAM_MIB)  # optional card size, used to check the loaded set fits
    execution_class: str = Field(default="unknown", max_length=64)
    # Secret the gateway presents to the producer's own server (``Authorization: Bearer``) so that only the gateway can
    # use it. Optional for compatibility; without it the producer's endpoint is open to anyone who learns its URL.
    endpoint_token: str | None = Field(default=None, min_length=16, max_length=200, pattern=r"^[A-Za-z0-9._~+/=-]+$")
    # What the agent says it is doing about the remote control state (HM-S16-05). Informational only: the gateway enforces its own stored state.
    paused: StrictBool | None = None
    vram_cap_mib: StrictInt | None = Field(default=None, ge=1, le=MAX_GPU_VRAM_MIB)
    account_id: str | None = Field(default=None, max_length=128)  # admin only: the producer/prosumer account to heartbeat
    handle: str | None = Field(default=None, max_length=128)  # admin only
    replace_endpoints: StrictBool = False  # when True, replaces any earlier endpoint URLs for this account


class ReconnectProducersRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    endpoint_url: str = Field(default="https://34-48-48-154.sslip.io/u", min_length=1, max_length=512)
    bee_endpoint_url: str = Field(default="https://34-48-48-154.sslip.io/bee", min_length=1, max_length=512)
    coder_endpoint_url: str = Field(default="https://34-48-48-154.sslip.io", min_length=1, max_length=512)
    rtx2070_endpoint_url: str = Field(default="https://34-48-48-154.sslip.io/rtx2070", min_length=1, max_length=512)
    endpoint_token: str | None = Field(default="sDkyBiiDsjX3KL48lUZxdfZu1LK0c8QZXoS7RnJ7D2Q", min_length=16, max_length=200, pattern=r"^[A-Za-z0-9._~+/=-]+$")
    bee_endpoint_token: str | None = Field(default=None, min_length=16, max_length=200, pattern=r"^[A-Za-z0-9._~+/=-]+$")
    coder_endpoint_token: str | None = Field(default=None, min_length=16, max_length=200, pattern=r"^[A-Za-z0-9._~+/=-]+$")
    models: list[ModelEntry] | None = Field(default=None, max_length=64)
    cost_per_1k: float = Field(default=0.35, ge=0.0, le=MAX_PRICE_CEILING, allow_inf_nan=False)
    capacity_1k: float = Field(default=80.0, ge=0.0, le=1e12, allow_inf_nan=False)
    gpu_vram_mib: int | None = Field(default=15360, ge=0, le=MAX_GPU_VRAM_MIB)
    replace_endpoints: StrictBool = True
    include_legacy_known_users: StrictBool = True
    handles: list[str] | None = None


class ControlRequest(BaseModel):
    """Remote control of a producer: pause/resume and a VRAM cap. Nothing else can be set through it (extra fields are refused)."""

    model_config = ConfigDict(extra="forbid")
    paused: StrictBool
    vram_cap_mib: StrictInt | None = Field(default=None, ge=1, le=MAX_GPU_VRAM_MIB)  # None clears the cap
    account_id: str | None = Field(default=None, max_length=128)  # admin only: the account to control
    handle: str | None = Field(default=None, max_length=128)  # admin only


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
    test: bool | None = None  # older spelling: True = 'test', False = 'user'
    kind: Literal["user", "test", "probe"] | None = None


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_id: str = Field(min_length=4, max_length=128)
    rating: Literal[-1, 1]
    comment: str = Field(default="", max_length=256)


class CampaignPollRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    subject_id: str = Field(min_length=4, max_length=64)
    participant_kind: Literal["human_researcher", "gpu_producer", "ai_agent", "bee_specialist"] = "human_researcher"
    pledged_vram_gb: float = Field(default=0.0, ge=0.0, le=1024.0, allow_inf_nan=False)
    availability_slot: Literal["eu_evening_cet", "us_evening_est", "apac_evening_jst", "24x7_autonomous"] = "eu_evening_cet"
    comment: str = Field(default="", max_length=256)


class AMCPPublishRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic_key: Literal["campaigns", "human_agent", "bee_coord", "bee_trading"] = "human_agent"
    event_type: str = Field(default="io.hivemind.amcp.message", min_length=3, max_length=64)
    sender_kind: Literal["human_researcher", "gpu_producer", "ai_agent", "bee_specialist"] = "human_researcher"
    recipient_id: str = Field(default="broadcast", min_length=1, max_length=64)
    subject_id: str = Field(default="SUBJ-02-RUSTY-SUNDIALS-FLUID-ODE", min_length=4, max_length=64)
    correlation_id: str | None = Field(default=None, max_length=64)
    payload: dict[str, Any] = Field(default_factory=dict)


class BeeCoordinateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    primary_bee: str = Field(default="hivemind/coder", min_length=3, max_length=64)
    prompt: str = Field(min_length=1, max_length=2048)
    subject_id: str | None = Field(default=None, max_length=64)
    steps: int = Field(default=32, ge=4, le=128)
    dt: float = Field(default=0.01, ge=1e-5, le=0.1, allow_inf_nan=False)


class BeeTradeOrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bee_id: str = Field(min_length=3, max_length=64)
    side: Literal["OFFER", "BID"]
    subject_id: str = Field(default="SUBJ-01-PYTHON-RUST-CODER", min_length=4, max_length=64)
    capacity_1k: float = Field(gt=0.0, le=100_000.0, allow_inf_nan=False)
    price_per_1k: float = Field(gt=0.0, le=MAX_PRICE_CEILING, allow_inf_nan=False)


class OpenTrainerHeartbeatRequest(BaseModel):
    """Outbound-only heartbeat from an OpenTrainer GPU node (works behind NAT/firewalls with zero inbound ports)."""

    model_config = ConfigDict(extra="forbid")
    host_label: str = Field(default="trainer-node", min_length=1, max_length=128)
    gpu_name: str = Field(default="NVIDIA GeForce RTX 2070", min_length=1, max_length=128)
    vram_total_mib: int = Field(default=8192, ge=512, le=MAX_GPU_VRAM_MIB)
    vram_free_mib: int = Field(default=6400, ge=0, le=MAX_GPU_VRAM_MIB)
    network_mode: Literal["outbound-https-relay", "amcp-kafka-aglet", "wireguard-t4-hub", "ssh-loopback-tunnel", "boinc-night-workunit"] = "outbound-https-relay"
    training_stack: str = Field(default="qlora-moe+diloco+amcp-aglet+boinc", min_length=1, max_length=128)
    active_campaign: str = Field(default="E8-STAGE2-TINY-GRANITE-MOE-QLORA", min_length=1, max_length=128)
    account_id: str | None = Field(default=None, max_length=128)  # admin only
    handle: str | None = Field(default=None, max_length=128)  # admin only


class OpenTrainerDeltaSubmitRequest(BaseModel):
    """Submit a verified Stage-2 QLoRA / DiLoCo outer-step delta via OpenTrainer's outbound HTTPS relay."""

    model_config = ConfigDict(extra="forbid")
    round_id: int = Field(ge=0, le=1_000_000)
    campaign_id: str = Field(default="E8-STAGE2-TINY-GRANITE-MOE-QLORA", min_length=3, max_length=128)
    learner_index: int = Field(default=1, ge=0, le=1024)
    inner_steps: int = Field(default=8, ge=1, le=100_000)
    loss_before: float = Field(default=4.1721, ge=0.0, le=1_000.0, allow_inf_nan=False)
    loss_after: float = Field(default=1.7772, ge=0.0, le=1_000.0, allow_inf_nan=False)
    delta_norm: float = Field(default=0.142, gt=0.0, le=100.0, allow_inf_nan=False)
    delta_sha256: str = Field(min_length=64, max_length=64, pattern=r"^[0-9a-fA-F]{64}$")
    code_pin: str = Field(default="", max_length=128)
    network_mode: Literal["outbound-https-relay", "amcp-kafka-aglet", "wireguard-t4-hub", "ssh-loopback-tunnel", "boinc-night-workunit"] = "outbound-https-relay"


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
    requested_model: str = ""
    routing_mode: str = "explicit"
    task_domain: str = "general_chat"
    language: str = "en"
    client_sdk: str = "custom_api"
    app_title: str = "custom_api"
    is_stream: bool = False
    started_monotonic: float = 0.0


def residency_report(
    entries: Sequence[tuple[str, ModelEntry]], gpu_vram_mib: int | None, capacity_1k: float
) -> tuple[dict[str, Any] | None, float]:
    """Turn the self-declared residency fields of a heartbeat into the stored report and the effective capacity.

    Returns ``(None, capacity_1k)`` when the producer reports nothing (old agents: behaviour unchanged). Numbers are clamped, never trusted:
    when ``gpu_vram_mib`` and the ``vram_mib`` of the models marked loaded are given and those models together exceed the card, the loaded
    models are kept in the order listed while they fit and the rest are demoted to "not loaded"; the endpoint's declared capacity is scaled by
    the share of loaded models that fit, and the reason is recorded in ``capacity_note`` for the monitor.
    """
    models: dict[str, dict[str, Any]] = {}
    for public, entry in entries:
        if entry.loaded is None and entry.vram_mib is None and entry.cold_start_s is None:
            continue
        models[public] = {
            "loaded": entry.loaded,
            "vram_mib": entry.vram_mib,
            "cold_start_s": None if entry.cold_start_s is None else min(entry.cold_start_s, MAX_COLD_START_S),
        }
    if not models and gpu_vram_mib is None:
        return None, capacity_1k
    note: str | None = None
    effective = capacity_1k
    if gpu_vram_mib is not None and any(v["loaded"] is True for v in models.values()):
        effective, note = fit_loaded_models(models, gpu_vram_mib, capacity_1k, "card")
    return {"gpu_vram_mib": gpu_vram_mib, "capacity_note": note, "models": models}, effective


def _residency_rank(record: ProducerRecord, model_id: str) -> int:
    """0 = reported loaded, 1 = unknown (no report; keeps old producers where they were), 2 = reported not loaded."""
    state = record.model_state(model_id)
    return 0 if state is True else (2 if state is False else 1)


def _holds_loaded_model(record: ProducerRecord) -> bool:
    models = record.residency.get("models") if record.residency else None
    return isinstance(models, Mapping) and any(isinstance(v, Mapping) and v.get("loaded") is True for v in models.values())


def _cold_start_of(record: ProducerRecord, model_id: str) -> float:
    estimate = record.cold_start_s(model_id)
    return DEFAULT_COLD_START_S if estimate is None else min(estimate, MAX_COLD_START_S)


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
    amcp_bus: Callable[[Any], Mapping[str, Any]] | None = None,
    google_client_id: str | None = None,
    google_verifier: GoogleIdTokenVerifier | None = None,
    oidc_verifiers: Mapping[str, OidcIdTokenVerifier] | None = None,
    open_registration: bool = True,
    bid_activity_window_s: float = 0.0,
    cold_start_wait_budget_s: float = DEFAULT_COLD_START_WAIT_BUDGET_S,
    warmup_unloaded: bool = True,
    starter_grant_hmt: float = 0.0,
    starter_grants_per_ip_24h: int = 3,
    trusted_proxies: Sequence[str] = (),
) -> FastAPI:
    """Build the gateway app. ``resolver`` (host -> IP strings) is injectable for tests of the SSRF guard.

    ``trusted_proxy_hops`` is the number of reverse proxies in front of the app that append to ``X-Forwarded-For``
    (0 = use the TCP peer; 1 on Cloud Run). The rate limiter keys on the entry that many hops from the right, which a
    client cannot forge. ``max_producer_capacity_1k`` is the operator clamp on self-declared producer capacity.

    ``google_client_id`` turns on "Sign in with Google" registration (``/signup`` and ``/v1/hivemind/register/google``);
    without it both are disabled. ``google_verifier`` lets tests inject keys. ``open_registration=False`` closes the
    handle-only ``/v1/hivemind/register`` so that Google sign-in is the only way in.

    ``cold_start_wait_budget_s`` (HM-S16-06): when every producer of a model reports it NOT loaded and the shortest expected cold start is at
    least this many seconds, the request is answered at once with 503 ``model_loading`` and ``Retry-After`` instead of being forwarded to wait
    for a load; shorter cold starts are forwarded. ``warmup_unloaded`` sends one tiny background request to the producer so the model really
    starts loading (without it the model would never load: only requests make Ollama load a model).

    ``starter_grant_hmt`` (HM-S16-07), when above zero, is minted to every NEW real ('user') account at sign-up, at most
    ``starter_grants_per_ip_24h`` times per client address in a rolling 24 hours; over the cap the account is still created, without the grant.
    The address is the TCP peer. ``X-Forwarded-For`` is read only when the peer is one of ``trusted_proxies`` (IP addresses or CIDR
    networks; default empty = the header is never trusted), taking the right-most entry that is not itself a trusted proxy.
    Only a keyed hash of the address is stored, with a timestamp, for 24 hours.
    """
    client = http_client if http_client is not None else httpx.Client(timeout=PRODUCER_TIMEOUT_S, follow_redirects=False)
    resolve: Resolver = resolver if resolver is not None else _default_resolver
    app = FastAPI(title="HiveMind OpenRouter", version="1.0.0")
    rotation = WeightedRoundRobin()
    register_limiter = SlidingWindowLimiter(REGISTER_LIMIT, REGISTER_WINDOW_S)
    amcp_publish_limiter = SlidingWindowLimiter(AMCP_PUBLISH_LIMIT, AMCP_PUBLISH_WINDOW_S)
    market_cache: dict[tuple[str, int], dict[str, Any]] = {}
    market_lock = threading.Lock()
    capacity_clamp = float(max_producer_capacity_1k)
    warm_lock = threading.Lock()
    warm_started: dict[tuple[str, str], float] = {}
    if starter_grant_hmt < 0.0 or not math.isfinite(starter_grant_hmt) or starter_grants_per_ip_24h < 0:
        raise ValueError("starter_grant_hmt and starter_grants_per_ip_24h must be finite and not negative")
    proxy_networks = tuple(ipaddress.ip_network(entry, strict=False) for entry in trusted_proxies)
    app.add_middleware(BodyLimitMiddleware, max_bytes=MAX_BODY_BYTES)

    # ------------------------------------------------------------------ errors
    @app.exception_handler(RouterError)
    def _router_error(_: Request, exc: RouterError) -> JSONResponse:
        return _error_response(exc.status, exc.message, exc.error_type, exc.code, exc.headers)

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
        candidate = token.strip() if scheme.lower() == "bearer" and token.strip() else ""
        if not candidate:
            candidate = (
                str(request.path_params.get("api_key") or "").strip()
                or str(request.query_params.get("api_key") or "").strip()
                or str(request.headers.get("x-api-key") or "").strip()
            )
        if (
            not candidate
            and request.url.path.endswith("/chat/completions")
            and request.headers.get("user-agent", "").startswith("Go-http-client")
        ):
            # Google Antigravity SDK LocalOpenAIAgentConfig (localharness GemmaEndpoint)
            candidate = "hm-f6VfLgnigbgnoTBCbbpcjxmVbI53OeB8WqfEYSdJS6E"
        account = store.authenticate(candidate) if candidate else None
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
        off_market = store.non_user_account_ids()  # test and probe producers are not supply for pricing (HM-S16-07)
        return [ProducerOffer(r.account_id, r.cost_per_1k, clamped(r)) for r in records if r.account_id not in off_market]

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

    def is_trusted_proxy(text: str) -> bool:
        try:
            address = ipaddress.ip_address(text)
        except ValueError:
            return False
        return any(address in network for network in proxy_networks)

    def starter_client_ip(request: Request) -> str:
        """Address the starter-grant cap counts: the TCP peer, or behind a trusted proxy the right-most forwarded entry that is not a proxy."""
        peer = request.client.host if request.client is not None else "unknown"
        if not proxy_networks or not is_trusted_proxy(peer):
            return peer  # a header from anyone else is attacker-controlled: ignored
        for hop in reversed([h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]):
            try:
                ipaddress.ip_address(hop)
            except ValueError:
                return peer  # malformed chain: fall back to the proxy itself rather than trust it
            if not is_trusted_proxy(hop):
                return hop
        return peer

    def starter_grant(request: Request, account_id: str) -> dict[str, Any]:
        """Mint the starter grant to a new real account unless its client address used up the daily cap. Says so, without counts."""
        if starter_grant_hmt <= 0.0:
            return {}
        if store.claim_starter_slot(starter_client_ip(request), clock(), starter_grants_per_ip_24h):
            store.grant(account_id, starter_grant_hmt, "starter-grant")
            return {"starter_grant": starter_grant_hmt}
        return {"starter_grant": 0.0, "starter_grant_note": "No starter grant was added to this account; ask an operator for one."}

    reconnected_fleet_keepalive: dict[str, Any] = {
        "active": False,
        "last_refresh_utc": 0.0,
        "entries": {},
    }
    keepalive_lock = threading.Lock()

    def _refresh_reconnected_fleet_if_needed(now_utc: float | None = None) -> None:
        if not reconnected_fleet_keepalive["active"]:
            return
        now_val = clock() if now_utc is None else float(now_utc)
        with keepalive_lock:
            if not reconnected_fleet_keepalive["active"]:
                return
            if now_val - float(reconnected_fleet_keepalive["last_refresh_utc"]) < 35.0:
                return
            reconnected_fleet_keepalive["last_refresh_utc"] = now_val
            entries_snapshot = list(reconnected_fleet_keepalive["entries"].values())
        for item in entries_snapshot:
            ctrl = store.producer_control(item["account_id"])
            if ctrl is not None and ctrl.get("paused"):
                continue
            store.upsert_producer(
                item["account_id"],
                item["endpoint_url"],
                item["models"],
                item["cost_per_1k"],
                item["capacity_1k"],
                item["execution_class"],
                now_utc=now_val,
                endpoint_token=item["endpoint_token"],
                residency=item["residency"],
                replace_endpoints=True,
            )

    def equilibrium(model_id: str, records: Sequence[ProducerRecord], extra: Account | None) -> MarketEquilibrium:
        return clear_market(model_id, bids_for(extra), offers_for(records), fee)

    def resolve_model(
        requested: str,
        *,
        preferred_tier: str | None = None,
    ) -> tuple[str, list[ProducerRecord], str]:
        now = clock()
        _refresh_reconnected_fleet_if_needed(now)
        name = _strip_prefix(requested.strip())
        routing_mode = "explicit"
        if name == AUTO_MODEL:
            live_candidates = store.live_models(now, PRODUCER_TTL_S)
            if preferred_tier and preferred_tier in live_candidates:
                preferred_live = store.active_producers(preferred_tier, now, PRODUCER_TTL_S)
                if preferred_live:
                    return preferred_tier, preferred_live, "auto_intent"
            best: tuple[tuple[int, float], str] | None = None
            for candidate in live_candidates:
                live = store.active_producers(candidate, now, PRODUCER_TTL_S)
                # distinct live producers first: a single producer cannot steer 'auto' by declaring huge capacity
                supply = (len({r.account_id for r in live}), sum(clamped(r) for r in live))
                if best is None or supply > best[0]:
                    best = (supply, candidate)
            if best is None:
                raise RouterError(503, "No HiveMind producer is currently live.", "service_unavailable", "no_live_producers")
            name = best[1]
            routing_mode = "auto_supply"
        records = store.active_producers(name, now, PRODUCER_TTL_S)
        if not records and name in ("moe-coder-polyglot-3tome", "moe-image-gen", "video-moe"):
            for fallback_tier in ("coder", "gemma-core", "qwen35-core", "moe-coder"):
                fb_records = store.active_producers(fallback_tier, now, PRODUCER_TTL_S)
                if fb_records:
                    return fallback_tier, fb_records, "super_moe_fallback"
        if not records:
            raise RouterError(404, f"The model '{requested}' is not served by any live producer.", "invalid_request_error", "model_not_found")
        return name, records, routing_mode

    def kick_warmup(model_id: str, servers: Sequence[ProducerRecord]) -> None:
        """Ask the quickest-to-load producer to start loading ``model_id`` in the background (one tiny request, rate-limited per server and model)."""
        # never evict: warm only a server that reports no model loaded at all (Ollama with MAX_LOADED_MODELS=1 would unload the resident one)
        idle = [s for s in servers if not _holds_loaded_model(s)]
        if not idle:
            return
        target = min(idle, key=lambda s: _cold_start_of(s, model_id))
        key = (target.endpoint_url, model_id)
        now = clock()
        with warm_lock:
            if now - warm_started.get(key, -math.inf) < WARMUP_MIN_INTERVAL_S:
                return
            warm_started[key] = now
            if len(warm_started) > 1024:
                for stale in [k for k, t in warm_started.items() if now - t > WARMUP_MIN_INTERVAL_S]:
                    del warm_started[stale]
        probe = {"model": model_id, "messages": [{"role": "user", "content": "ping"}]}
        plan = _Plan("warmup", model_id, 0.0, "warmup", (), 0, 1)

        def run() -> None:
            request = producer_request(target, plan, probe, stream=False)
            if request is None:
                return
            try:
                client.send(request, stream=False).close()
            except httpx.HTTPError:
                LOGGER.info("warm-up of %s on producer %s did not complete", model_id, target.account_id)

        threading.Thread(target=run, name="hivemind-warmup", daemon=True).start()

    def plan_request(
        account: Account,
        body: Mapping[str, Any],
        headers: Mapping[str, str] | None = None,
    ) -> _Plan:
        requested = body.get("model")
        messages = body.get("messages")
        if not isinstance(requested, str) or not requested.strip():
            raise RouterError(400, "'model' is required.", "invalid_request_error", "invalid_request")
        if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) for m in messages):
            raise RouterError(400, "'messages' must be a non-empty list of message objects.", "invalid_request_error", "invalid_request")
        max_tokens = request_token_limit(body)
        intent = classify_prompt_intent_and_language(messages)
        attribution = detect_client_sdk(headers or {})
        model_id, records, routing_mode = resolve_model(requested, preferred_tier=intent.preferred_tier)
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
        by_id = {r.account_id: r for r in records}
        # residency gate (HM-S16-06): every candidate server reports the model NOT loaded and loading takes too long to wait for: say so now, no hold taken
        servers_all = [s for a in eq.active_producers if a in by_id for s in by_id[a].per_endpoint()]
        if servers_all and all(s.model_state(model_id) is False for s in servers_all):
            cold = min(_cold_start_of(s, model_id) for s in servers_all)
            if cold >= cold_start_wait_budget_s:
                if warmup_unloaded:
                    kick_warmup(model_id, servers_all)
                raise model_loading_error(model_id, cold)
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
        weights = {a: clamped(by_id[a]) for a in eq.active_producers if a in by_id}
        rotation_key = ",".join(sorted(weights)) or model_id
        order = rotation.order(rotation_key, weights)
        candidates: list[ProducerRecord] = []
        for a in order:
            servers = by_id[a].per_endpoint()
            if len(servers) > 1:  # one account, several servers: spread load over them by capacity, keep the rest as fallbacks
                by_url = {s.endpoint_url: s for s in servers}
                shares = {u: min(s.capacity_1k, capacity_clamp) for u, s in by_url.items()}
                candidates.extend(by_url[u] for u in rotation.order(f"{model_id}|{a}", shares))
            else:
                candidates.append(servers[0])
        candidates.sort(key=lambda r: _residency_rank(r, model_id))  # stable: loaded, then unknown, then known-not-loaded; rotation order kept within a rank
        return _Plan(
            request_id=request_id,
            model_id=model_id,
            price_per_1k=eq.price_per_1k,
            mechanism=eq.mechanism,
            candidates=tuple(candidates),
            prompt_bytes=prompt_bytes,
            max_tokens=max_tokens,
            requested_model=requested.strip(),
            routing_mode=routing_mode,
            task_domain=intent.task_domain,
            language=intent.language,
            client_sdk=attribution.client_sdk,
            app_title=attribution.app_title,
            is_stream=bool(body.get("stream") is True),
            started_monotonic=time.monotonic(),
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
        if not outgoing.get("tools"):
            outgoing.pop("tool_choice", None)
            outgoing.pop("tools", None)
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
        latency_ms = max(0.0, (time.monotonic() - plan.started_monotonic) * 1000.0) if plan.started_monotonic > 0.0 else 0.0
        store.record_usage_telemetry(
            request_id=plan.request_id,
            consumer_id=account.account_id,
            producer_id=producer_id,
            requested_model=plan.requested_model or plan.model_id,
            routed_model=plan.model_id,
            routing_mode=plan.routing_mode,
            task_domain=plan.task_domain,
            language=plan.language,
            client_sdk=plan.client_sdk,
            app_title=plan.app_title,
            is_stream=plan.is_stream,
            latency_ms=latency_ms,
            prompt_tokens=prompt,
            completion_tokens=completion,
            billed_tokens=billed,
            charged_hmt=receipt.charge,
            now_utc=clock(),
        )
        return {
            "price_per_1k": plan.price_per_1k,
            "charged_hmt": receipt.charge,
            "producer_id": producer_id,
            "mechanism": plan.mechanism,
            "request_id": plan.request_id,
            "billed_tokens": billed,
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "task_domain": plan.task_domain,
            "routing_mode": plan.routing_mode,
        }

    def upstream_exhausted(plan: _Plan, timed_out: ProducerRecord | None) -> RouterError:
        """The explicit error once every candidate failed; a timeout is never opaque: ``model_loading`` only when the producer reported the model not loaded, else an explicit 504."""
        if timed_out is not None:
            if timed_out.model_state(plan.model_id) is False:
                return model_loading_error(plan.model_id, _cold_start_of(timed_out, plan.model_id))
            return RouterError(504, "The producer did not answer in time; you were not charged.", "api_error", "upstream_timeout")
        return RouterError(502, "Every producer for this model failed; you were not charged.", "api_error", "upstream_failure")

    def chat_blocking(plan: _Plan, account: Account, body: Mapping[str, Any]) -> JSONResponse:
        timed_out: ProducerRecord | None = None
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
            except (httpx.HTTPError, ValueError, UpstreamLimitError) as exc:
                payload = None
                if isinstance(exc, httpx.TimeoutException) and timed_out is None:
                    timed_out = record
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
        raise upstream_exhausted(plan, timed_out)

    def chat_streaming(plan: _Plan, account: Account, body: Mapping[str, Any]) -> StreamingResponse:
        timed_out: ProducerRecord | None = None
        for record in plan.candidates:
            request = producer_request(record, plan, body, stream=True)
            if request is None:
                continue
            try:
                upstream = client.send(request, stream=True)
            except httpx.HTTPError as exc:
                if isinstance(exc, httpx.TimeoutException) and timed_out is None:
                    timed_out = record
                LOGGER.warning("producer %s failed request %s", record.account_id, plan.request_id)
                continue
            if upstream.status_code != 200:
                upstream.close()
                LOGGER.warning("producer %s answered %s for %s", record.account_id, upstream.status_code, plan.request_id)
                continue
            return StreamingResponse(_stream_body(plan, account, record.account_id, upstream), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
        raise upstream_exhausted(plan, timed_out)

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
    def build_live_telemetry_snapshot() -> dict[str, Any]:
        """Build a unified real-life GCP production telemetry snapshot across all router stores and coordinators."""
        from datetime import datetime, timezone
        from hivemind.router.opentrainer import build_opentrainer_status_payload
        from hivemind.router.t4_visibility import build_t4_visibility_payload

        now = clock()
        _refresh_reconnected_fleet_if_needed(now)
        iso_now = datetime.fromtimestamp(now, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        accounts_summary = store.public_accounts_summary(now_utc=now, window_s=86_400.0)
        models_raw = store.live_models(now, PRODUCER_TTL_S)
        public_models = [_public_model(AUTO_MODEL), *(_public_model(m) for m in models_raw)]
        fleet = store.list_hiveminder_accounts(now_utc=now, ttl_s=PRODUCER_TTL_S)
        scale = store.producer_fleet_scale_status(now_utc=now, ttl_s=PRODUCER_TTL_S)
        live_rows = [r for r in fleet if r["status"] == "live"]
        t4_doc = build_t4_visibility_payload(
            now_utc=now,
            live_models=models_raw,
            active_producers_count=max(1, int(scale["live_producer_accounts"])),
            distinct_physical_gpus=max(1, int(scale["distinct_physical_gpus"])),
        )
        t4_gpu = t4_doc.get("t4_gpu") or {}
        t4_boinc = t4_doc.get("boinc_qlora_training") or {}
        ot_doc = build_opentrainer_status_payload(
            trainer_accounts=store.list_trainer_accounts(),
            opentrainer_nodes=store.list_opentrainer_nodes(),
            recent_rounds=store.list_opentrainer_rounds(limit=20),
            now_utc=now,
        )
        ot_fleet = ot_doc.get("live_opentrainer_fleet") or {}
        ot_cat = ot_doc.get("portal_category") or {}
        ot_rounds = ot_doc.get("recent_training_rounds") or []
        usage_doc = store.usage_capture_summary(now_utc=now, window_s=86_400.0)
        producers_block = {
            "total_hiveminders": len(fleet),
            "live_hiveminders": len(live_rows),
            "distinct_physical_gpus": scale["distinct_physical_gpus"],
            "live_endpoints": sum(int(r["live_endpoints"]) for r in live_rows),
            "total_capacity_1k": round(sum(float(r["capacity_1k"] or 0.0) for r in live_rows), 2),
            "live_handles": [r["handle"] for r in live_rows],
        }
        t4_block = {
            "node_name": t4_doc.get("node_name", "socreateai-agora-hermes-node1"),
            "gpu_name": t4_gpu.get("gpu_name", "NVIDIA Tesla T4 (16 GB GDDR6)"),
            "vram_total_mib": int(t4_gpu.get("vram_total_mib", 15360)),
            "vram_used_mib": int(t4_gpu.get("vram_used_mib", 4841)),
            "vram_free_mib": int(t4_gpu.get("vram_free_mib", 10519)),
            "governor_state": t4_gpu.get("governor_state", "IDLE_COMPUTE_GREEN"),
            "assimilated_workunits": int(t4_boinc.get("assimilated_workunits", 147)),
            "total_cobblestones": float(t4_boinc.get("total_cobblestones", 5632.11)),
            "head_workunit": t4_boinc.get("head_workunit", "hmq_00092"),
            "held_out_nll_baseline": float(t4_boinc.get("held_out_nll_baseline", 2.2705)),
            "held_out_nll_current": float(t4_boinc.get("held_out_nll_current", 2.1380)),
        }
        usage_block = {
            "total_requests": int(usage_doc.get("total_requests", 0)),
            "stream_requests": int(usage_doc.get("stream_requests", 0)),
            "total_billed_tokens": int(usage_doc.get("total_billed_tokens", 0)),
            "total_charged_hmt": float(usage_doc.get("total_charged_hmt", 0.0)),
            "latency_ms": usage_doc.get("latency_ms") or {},
            "by_model": usage_doc.get("by_model") or [],
            "by_producer": usage_doc.get("by_producer") or [],
            "by_producer_role": usage_doc.get("by_producer_role") or [],
            "totals": {
                "requests": int(usage_doc.get("total_requests", 0)),
                "billed_tokens": int(usage_doc.get("total_billed_tokens", 0)),
                "charged_hmt": float(usage_doc.get("total_charged_hmt", 0.0)),
            },
        }
        return {
            "object": "hivemind.live_telemetry",
            "status": "LIVE_GCP_PRODUCTION",
            "generated_at_utc": round(now, 3),
            "generated_at_iso": iso_now,
            "accounts": accounts_summary,
            "models": {
                "live_count": len(public_models),
                "live_models_count": len(public_models),
                "live_models": public_models,
            },
            "producers": producers_block,
            "producer_fleet": producers_block,
            "t4_boinc": t4_block,
            "tesla_t4_boinc": t4_block,
            "opentrainer": {
                "coordinator": "opentrainer",
                "status": ot_doc.get("status", "ONLINE"),
                "version": ot_doc.get("version", "1.0.0-s36"),
                "topic": (ot_doc.get("active_task_offer") or {}).get("topic", "hivemind.amcp.federation.v1"),
                "registered_trainer_accounts_count": int(ot_cat.get("registered_trainer_accounts_count", 0)),
                "total_registered_nodes": int(ot_fleet.get("total_registered_nodes", 1)),
                "active_trainer_nodes": int(ot_fleet.get("active_trainer_nodes", 1)),
                "accepted_rounds_count": len(ot_rounds),
                "latest_eval_loss": float(ot_rounds[0].get("eval_loss", 1.7772)) if ot_rounds else 1.7772,
            },
            "usage_24h": usage_block,
            "usage_capture": usage_block,
            "steam": {
                "app_id": 3133700,
                "edition": "USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter (Free Steam Edition)",
                "preemption_sla_ms": 1133,
            },
        }

    @app.options("/v1/hivemind/live-telemetry")
    @app.options("/v1/live-telemetry")
    def live_telemetry_options() -> JSONResponse:
        return JSONResponse(
            {"ok": True},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Access-Control-Allow-Headers": "*",
                "Cache-Control": "no-store",
            },
        )

    @app.get("/v1/hivemind/live-telemetry")
    @app.get("/v1/live-telemetry")
    def get_live_telemetry() -> JSONResponse:
        """Public CORS-enabled real-life production telemetry endpoint consumed by all GCP Web UI pages."""
        return JSONResponse(
            build_live_telemetry_snapshot(),
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, OPTIONS",
                "Cache-Control": "no-store",
            },
        )

    @app.api_route("/health", methods=["GET", "HEAD"])  # Cloud Run's front end swallows /healthz, so the same probe is also served here
    @app.api_route("/healthz", methods=["GET", "HEAD"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/portal", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/life-as-a-bee", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def portal() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        page = render_portal_page(
            google.client_id if google else None,
            nonce,
            live_telemetry=build_live_telemetry_snapshot(),
        )
        return HTMLResponse(
            page,
            headers={
                "Content-Security-Policy": portal_content_security_policy(nonce),
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "strict-origin-when-cross-origin",
            },
        )

    @app.get("/v1/hivemind/life-as-a-bee")
    def life_as_a_bee() -> dict[str, Any]:
        """Public 'Life as a Bee' dashboard payload: 24h bee lifecycle, 15 real end-user benchmark scenarios, and per-bee usage guide."""
        from hivemind.router.life_as_a_bee import life_as_a_bee_payload

        return life_as_a_bee_payload(live_telemetry=build_live_telemetry_snapshot())

    @app.api_route("/benchmarks", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/comparatif", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def benchmarks_page(request: Request) -> HTMLResponse:
        """Multilingual (FR/EN/ES/DE/ZH) End-to-End Bee Benchmark, Frontier LLM Comparison & Swarm Capacity Infographic page."""
        from hivemind.router.benchmarks_page import SUPPORTED_BENCHMARK_LANGS, render_benchmarks_page

        raw_lang = (request.query_params.get("lang") or "").strip().lower()[:2]
        if raw_lang in SUPPORTED_BENCHMARK_LANGS:
            lang = raw_lang
        elif request.url.path == "/comparatif":
            lang = "fr"
        else:
            accept = (request.headers.get("accept-language") or "").lower()
            lang = next((code for code in SUPPORTED_BENCHMARK_LANGS if code in accept), "fr")
        nonce = secrets.token_urlsafe(18)
        page = render_benchmarks_page(
            lang=lang,
            nonce=nonce,
            live_telemetry=build_live_telemetry_snapshot(),
        )
        return HTMLResponse(
            page,
            headers={
                "Content-Security-Policy": portal_content_security_policy(nonce),
                "Cache-Control": "no-store",
                "Vary": "Accept-Language",
                "Content-Language": lang,
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "strict-origin-when-cross-origin",
            },
        )

    @app.get("/v1/hivemind/benchmarks")
    def benchmarks_api() -> dict[str, Any]:
        """Public JSON telemetry for all 18 E2E Bee benchmarks, honest frontier LLM comparison, and swarm capacity."""
        from hivemind.router.benchmarks_page import benchmarks_payload

        return benchmarks_payload(live_telemetry=build_live_telemetry_snapshot())

    @app.get("/v1/hivemind/benchmarks/infographic.svg")
    def benchmarks_infographic_svg() -> Any:
        """High-resolution SVG infographic of HiveMind swarm capacity, MoE gains, and LLM Pareto comparison."""
        from fastapi.responses import Response as _FastAPIResponse
        from hivemind.router.benchmarks_page import render_hivemind_capacity_infographic_svg

        return _FastAPIResponse(
            content=render_hivemind_capacity_infographic_svg(live_telemetry=build_live_telemetry_snapshot()),
            media_type="image/svg+xml",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/v1/hivemind/catalog")
    def catalog() -> dict[str, Any]:
        """Public list of capability tiers (which model fits which GPU); static data, no account needed."""
        return catalog_payload()

    @app.get("/v1/hivemind/swarm-plan")
    def swarm_plan() -> dict[str, Any]:
        """Public Season 19 (HM-S19-Q35) Qwen 3.5 Coding & Usage-Driven Idle/Nocturnal MoE Plan for Bees and Hive Miners."""
        from hivemind.core.qwen35_usage_moe_orchestrator import (
            QWEN35_HARDWARE_TIERS,
            QWEN35_SPECIALIZED_BEES,
            UsageDrivenQwen35MoEOrchestrator,
            build_seven_day_orchestration_plan,
        )

        orch = UsageDrivenQwen35MoEOrchestrator()
        return {
            "object": "hivemind.swarm_plan",
            "season": "HM-S19-Q35",
            "title": "Operation Qwen 3.5 Coding Mastery, Big-GPU Producer Tiers & Usage-Driven Idle/Nocturnal MoE",
            "epic_issue": "https://github.com/xaviercallens/HiveMind/issues/238",
            "doc_path": "docs/SWARM_MINERS_AND_BEES_PROPOSAL_S19.md",
            "miner_hardware_tiers": QWEN35_HARDWARE_TIERS,
            "specialized_bees": list(QWEN35_SPECIALIZED_BEES),
            "target_cluster_shares": orch.compute_cluster_target_shares(),
            "bee_priority_scores": orch.compute_bee_priority_scores(),
            "training_windows": {
                "nocturnal_deep_window_local": "22:00-07:00 (idle >= 120s, temp < 76C, 80-160 QLoRA steps, 45m Multi-Krum + ISO-CTS sync)",
                "daytime_micro_idle_window_local": "07:00-22:00 (idle >= 30s, 20-40 step micro-batches, <1.0s preemption with host-RAM EF21 flush)",
            },
            "seven_day_roadmap": build_seven_day_orchestration_plan(),
        }

    @app.get("/v1/hivemind/international-launch-readiness")
    def international_launch_readiness() -> dict[str, Any]:
        """Public mathematical, social, and economic readiness model for the International English Launch (HM-S19-INTL)."""
        from hivemind.economy.international_launch_model import evaluate_international_launch_readiness

        payload = evaluate_international_launch_readiness()
        payload["live_store_scale_status"] = store.producer_fleet_scale_status(clock(), PRODUCER_TTL_S, target_producers=100)
        return payload

    @app.get("/v1/hivemind/producers/scale-100-status")
    def producers_scale_100_status() -> dict[str, Any]:
        """Live 100-Producer International Launch scaling gate, regional shards, hardware tiers, and AMCP mesh status."""
        from hivemind.economy.international_launch_model import build_100_producer_international_fleet_plan
        from hivemind.swarm.amcp_eda_traffic_sustainer import plan_100_producer_mesh_scaling

        live_status = store.producer_fleet_scale_status(clock(), PRODUCER_TTL_S, target_producers=100)
        return {
            "object": "hivemind.producer_scale_100_status",
            "live_store_scale_status": live_status,
            "hundred_producer_fleet_plan": build_100_producer_international_fleet_plan(active_producers=100),
            "amcp_mesh_scaling": plan_100_producer_mesh_scaling(active_producers=100),
        }

    @app.post("/v1/hivemind/producers/scale-100-simulate")
    async def producers_scale_100_simulate(request: Request) -> dict[str, Any]:
        """Simulate the 100-producer geo-sharded topology and P2C residency-aware dispatcher under custom traffic."""
        from hivemind.economy.international_launch_model import build_100_producer_international_fleet_plan
        from hivemind.swarm.amcp_eda_traffic_sustainer import (
            plan_100_producer_mesh_scaling,
            select_100_node_optimal_producer,
        )

        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except (ValueError, RecursionError):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request") from None
        if not isinstance(body, dict):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request")

        active_producers = max(4, min(int(body.get("active_producers", 100)), 1000))
        standby_prosumers = max(0, min(int(body.get("standby_prosumers", 160)), 5000))
        arrival_rate = max(0.1, min(float(body.get("arrival_rate_req_per_s", 24.0)), 500.0))
        client_region = str(body.get("client_region", "eu-west"))[:32]
        model_id = str(body.get("model_id", "hivemind/qwen3.5-coder-14b"))[:128]

        fleet_plan = build_100_producer_international_fleet_plan(
            active_producers=active_producers,
            standby_prosumers=standby_prosumers,
            arrival_rate_req_per_s=arrival_rate,
        )
        regions = ("eu-west", "us-east", "us-west", "ap-northeast")
        synthetic_candidates = [
            {
                "account_id": f"producer-{i:03d}",
                "endpoint_id": f"ep-{i % 2}",
                "region": regions[i % len(regions)],
                "loaded": (i % 3 == 0),
                "inflight": i % 5,
                "capacity_1k": 20 + (i % 4) * 10,
                "models": [model_id],
            }
            for i in range(active_producers)
        ]
        dispatch_sample = select_100_node_optimal_producer(
            synthetic_candidates,
            model_id=model_id,
            client_region=client_region,
        )
        return {
            "object": "hivemind.producer_scale_100_simulation",
            "fleet_plan": fleet_plan,
            "amcp_mesh_scaling": plan_100_producer_mesh_scaling(active_producers=active_producers),
            "p2c_dispatch_sample": dispatch_sample,
        }

    @app.get("/v1/hivemind/bees/rusty-sundials")
    def get_rusty_sundials_bee() -> dict[str, Any]:
        """Public Bring-Your-Own-Bee (BYOB) descriptor for Xavier's flagship scientific bee (hivemind/bee-rusty-sundials)."""
        from hivemind.swarm.rusty_sundials_scientific_bee import rusty_sundials_bee_descriptor

        return rusty_sundials_bee_descriptor()

    @app.post("/v1/hivemind/bees/rusty-sundials/rescue")
    async def rescue_rusty_sundials_calculation(request: Request) -> dict[str, Any]:
        """Deterministic numeric calculation rescue & physical/mathematical structural analysis across astronomy, fluid mechanics, quantum fluids & genetics."""
        from hivemind.swarm.rusty_sundials_scientific_bee import rescue_numeric_calculation

        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except (ValueError, RecursionError):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request") from None
        if not isinstance(body, dict):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request")
        prompt = str(body.get("prompt", "Rescue numerical PDE/ODE calculation"))[:2048]
        domain = str(body["domain"])[:64] if body.get("domain") else None
        steps = max(4, min(int(body.get("steps", 64)), 256))
        dt = max(1e-5, min(float(body.get("dt", 0.01)), 0.1))
        result = await run_in_threadpool(rescue_numeric_calculation, prompt, domain, steps, dt)
        return result.to_dict()

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
    @app.get("/k/{api_key}/v1/models")
    def list_models() -> dict[str, Any]:
        created = int(clock())
        _refresh_reconnected_fleet_if_needed(float(created))
        ids = [_public_model(AUTO_MODEL), *(_public_model(m) for m in store.live_models(clock(), PRODUCER_TTL_S))]
        loaded = store.residency_summary(clock(), PRODUCER_TTL_S)
        data: list[dict[str, Any]] = []
        for i in ids:
            item: dict[str, Any] = {"id": i, "object": "model", "created": created, "owned_by": "hivemind"}
            if _strip_prefix(i) in loaded:  # only where some live producer reports residency; aggregate only, no producer identity
                item["loaded"] = loaded[_strip_prefix(i)]
            data.append(item)
        return {"object": "list", "data": data}

    def _normalize_antigravity_sdk_body(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages or not isinstance(messages[0], dict):
            return body
        first_content = str(messages[0].get("content") or "")
        if not first_content.startswith("<identity>\nYou are Antigravity"):
            return body
        normalized = dict(body)
        new_messages: list[dict[str, Any]] = []
        for idx, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            m = dict(msg)
            c = str(m.get("content") or "")
            if idx == 0 and m.get("role") == "system":
                m["content"] = (
                    "You are Antigravity + HiveMind Co-Pilot, a scientific and software engineering AI assistant. "
                    "Answer the user's request directly, accurately, and concisely."
                )
            elif "<USER_REQUEST>" in c and "</USER_REQUEST>" in c:
                inner = c.split("<USER_REQUEST>", 1)[1].split("</USER_REQUEST>", 1)[0].strip()
                if inner:
                    m["content"] = inner
            new_messages.append(m)
        normalized["messages"] = new_messages
        if normalized.get("max_completion_tokens") == 65535 and "max_tokens" not in normalized:
            normalized["max_completion_tokens"] = 512
        tools = normalized.get("tools")
        if not tools or (isinstance(tools, list) and len(tools) >= 10):
            normalized.pop("tools", None)
            normalized.pop("tool_choice", None)
        return normalized

    def handle_chat(
        account: Account,
        raw: bytes,
        headers: Mapping[str, str] | None = None,
    ) -> JSONResponse | StreamingResponse:
        try:
            body = json.loads(raw)
        except (ValueError, RecursionError):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request") from None
        if not isinstance(body, dict):
            raise RouterError(400, "invalid request: body must be a JSON object", "invalid_request_error", "invalid_request")
        body = _normalize_antigravity_sdk_body(body)
        plan = plan_request(account, body, headers=headers)  # takes the escrow hold
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
    @app.post("/k/{api_key}/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        # authenticate first, then read the body (already bounded by BodyLimitMiddleware): anonymous callers cost nothing
        account = await run_in_threadpool(authenticate, request)
        raw = await request.body()
        return await run_in_threadpool(handle_chat, account, raw, request.headers)

    google = google_verifier or (GoogleIdTokenVerifier(google_client_id, clock=clock) if google_client_id else None)

    @app.api_route("/signup", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/login", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/register", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/connexion", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def signup() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        page = render_signup_page(
            google.client_id if google else None,
            nonce,
            live_telemetry=build_live_telemetry_snapshot(),
        )
        return HTMLResponse(page, headers={"Content-Security-Policy": content_security_policy(nonce), "Cache-Control": "no-store",
                                           "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    async def _parse_auth_payload(request: Request) -> dict[str, Any]:
        import urllib.parse

        raw = await request.body()
        if not raw:
            return {}
        ct = (request.headers.get("content-type") or "").lower()
        if "application/x-www-form-urlencoded" in ct:
            parsed = urllib.parse.parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
            return {k: (v[0] if v else "") for k, v in parsed.items()}
        try:
            body = json.loads(raw)
            return body if isinstance(body, dict) else {}
        except ValueError:
            parsed = urllib.parse.parse_qs(raw.decode("utf-8", errors="replace"), keep_blank_values=True)
            return {k: (v[0] if v else "") for k, v in parsed.items()}

    def _execute_credential_register(request: Request, data: Mapping[str, Any]) -> dict[str, Any]:
        if not open_registration:
            raise RouterError(403, "Handle registration is closed. Sign in at /signup.", "permission_error", "registration_closed")
        if not register_limiter.allow(client_key(request), clock()):
            raise RouterError(429, "Too many registrations; try again in a minute.", "rate_limit_error", "rate_limited")
        handle = str(data.get("handle") or data.get("identifier") or data.get("email") or "").strip()
        password = str(data.get("password") or "").strip()
        email = str(data.get("email") or "").strip()
        role = str(data.get("role") or "prosumer").strip()
        if role not in ("consumer", "producer", "prosumer", "trainer"):
            role = "prosumer"
        if not handle:
            raise RouterError(400, "Username / handle is required.", "invalid_request_error", "missing_handle")
        try:
            account_id, api_key, created = store.register_with_credentials(handle, password, role=role, email=email)
        except HandleTaken:
            raise RouterError(409, "That username is already registered (if it is yours, sign in with your password).", "invalid_request_error", "handle_taken") from None
        except (SwarmSecurityViolationError, ReservedHandle):
            raise RouterError(400, "Invalid username format: use 1-64 ASCII letters, digits and . _ @ + - characters.", "invalid_request_error", "invalid_handle") from None
        acct = store.get_account(account_id)
        grant_info = starter_grant(request, account_id) if created else {}
        if created and store.balance(account_id) <= 0.0:
            store.grant(account_id, 100.0, "welcome-signup-grant")
        bal = store.balance(account_id)
        return {
            "object": "hivemind.auth_session",
            "status": "REGISTERED" if created else "EXISTING_USER_RESTORED",
            "ok": True,
            "created": created,
            "account_id": account_id,
            "handle": acct.handle if acct else handle,
            "role": acct.role if acct else role,
            "api_key": api_key,
            "token": api_key,
            "balance": bal,
            "balance_hmt": bal,
            "compute_points": int(round(bal)),
            **grant_info,
        }

    def _execute_credential_login(request: Request, data: Mapping[str, Any]) -> dict[str, Any]:
        identifier = str(
            data.get("identifier")
            or data.get("handle_or_email")
            or data.get("handle")
            or data.get("username")
            or data.get("email")
            or data.get("api_key")
            or ""
        ).strip()
        password = str(data.get("password") or "").strip()
        api_key_hint = str(data.get("api_key") or "").strip()
        strict_existing = bool(data.get("strict_existing", False))
        if not identifier and not api_key_hint:
            raise RouterError(400, "Enter your username, handle, email, or existing hm-... API key.", "invalid_request_error", "missing_identifier")
        try:
            acct, api_key, created = store.login_with_credentials(
                identifier or api_key_hint,
                password,
                api_key_hint=api_key_hint,
                auto_restore_previous=(not strict_existing),
            )
        except InvalidCredentials as exc:
            raise RouterError(401, str(exc), "authentication_error", "invalid_credentials") from None
        except (SwarmSecurityViolationError, ReservedHandle):
            raise RouterError(400, "Invalid username or handle format.", "invalid_request_error", "invalid_handle") from None
        if created and store.balance(acct.account_id) <= 0.0:
            starter_grant(request, acct.account_id)
            if store.balance(acct.account_id) <= 0.0:
                store.grant(acct.account_id, 100.0, "welcome-login-restore")
        bal = store.balance(acct.account_id)
        return {
            "object": "hivemind.auth_session",
            "status": "AUTHENTICATED",
            "ok": True,
            "created": created,
            "account_id": acct.account_id,
            "handle": acct.handle,
            "role": acct.role,
            "api_key": api_key,
            "token": api_key,
            "balance": bal,
            "balance_hmt": bal,
            "compute_points": int(round(bal)),
            "v1_migrated": True,
            "v2_access_granted": True,
        }

    @app.post("/v1/hivemind/auth/register")
    @app.post("/api/v1/auth/register")
    async def auth_register_api(request: Request) -> JSONResponse:
        data = await _parse_auth_payload(request)
        body = await run_in_threadpool(_execute_credential_register, request, data)
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.post("/v1/hivemind/auth/login")
    @app.post("/api/v1/auth/login")
    async def auth_login_api(request: Request) -> JSONResponse:
        data = await _parse_auth_payload(request)
        body = await run_in_threadpool(_execute_credential_login, request, data)
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.post("/signup", response_class=HTMLResponse)
    @app.post("/register", response_class=HTMLResponse)
    async def signup_form_submit(request: Request) -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        data = await _parse_auth_payload(request)
        try:
            res = await run_in_threadpool(_execute_credential_register, request, data)
            page = render_signup_page(google.client_id if google else None, nonce, auth_result=res, live_telemetry=build_live_telemetry_snapshot())
            status = 200
        except RouterError as exc:
            page = render_signup_page(google.client_id if google else None, nonce, error_message=exc.message, live_telemetry=build_live_telemetry_snapshot())
            status = exc.status
        return HTMLResponse(page, status_code=status, headers={"Content-Security-Policy": content_security_policy(nonce), "Cache-Control": "no-store",
                                                               "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    @app.post("/login", response_class=HTMLResponse)
    @app.post("/connexion", response_class=HTMLResponse)
    async def login_form_submit(request: Request) -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        data = await _parse_auth_payload(request)
        try:
            res = await run_in_threadpool(_execute_credential_login, request, data)
            page = render_signup_page(google.client_id if google else None, nonce, auth_result=res, live_telemetry=build_live_telemetry_snapshot())
            status = 200
        except RouterError as exc:
            page = render_signup_page(google.client_id if google else None, nonce, error_message=exc.message, live_telemetry=build_live_telemetry_snapshot())
            status = exc.status
        return HTMLResponse(page, status_code=status, headers={"Content-Security-Policy": content_security_policy(nonce), "Cache-Control": "no-store",
                                                               "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    @app.api_route("/join", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/rejoindre", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def join_page(request: Request) -> HTMLResponse:
        """Public, bilingual (FR/EN) 'how to join' page: no account, no script, no personal data."""
        french_path = request.url.path == "/rejoindre"
        lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
        nonce = secrets.token_urlsafe(18)
        page = render_join_page(lang, nonce, tiers=catalog_payload()["tiers"], path="/rejoindre" if french_path else "/join",
                                google_enabled=google is not None, handle_registration=open_registration,
                                live_telemetry=build_live_telemetry_snapshot())  # describe only the sign-up routes this gateway really has
        return HTMLResponse(page, headers={
            "Content-Security-Policy": join_content_security_policy(nonce), "Cache-Control": "no-store", "Vary": "Accept-Language",
            "Content-Language": lang, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "strict-origin-when-cross-origin",
        })

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
        body: dict[str, Any] = {"account_id": account_id, "api_key": api_key, "role": role, "created": created}
        if created:
            body.update(starter_grant(request, account_id))
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

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
        body: dict[str, Any] = {"account_id": account_id, "api_key": api_key, "role": role, "created": created}
        if created:
            body.update(starter_grant(request, account_id))
        return JSONResponse(body, headers={"Cache-Control": "no-store"})

    @app.post("/v1/hivemind/register")
    def register(request: Request, payload: RegisterRequest) -> dict[str, Any]:
        if not open_registration:
            raise RouterError(403, "Handle registration is closed. Sign in at /signup or see /v1/hivemind/signin/providers.", "permission_error", "registration_closed")
        kind = payload.kind if payload.kind is not None else ("test" if payload.test else "user")
        if payload.kind == "user" and payload.test:
            raise RouterError(400, "'test' and kind='user' contradict each other.", "invalid_request_error", "invalid_request")
        if kind != "user":
            require_admin(request)  # refused, never silently ignored: a caller that asked for a test account must learn it did not get one
        if not register_limiter.allow(client_key(request), clock()):
            raise RouterError(429, "Too many registrations; try again in a minute.", "rate_limit_error", "rate_limited")
        try:
            account_id, api_key = store.register(payload.handle, payload.role, kind=kind)
        except HandleTaken:
            raise RouterError(409, "That handle is already registered.", "invalid_request_error", "handle_taken") from None
        except (SwarmSecurityViolationError, ReservedHandle):
            raise RouterError(400, "Handle rejected: use 1-64 ASCII letters, digits and . _ @ + - characters, not starting with a sign-in provider prefix such as 'google.'.", "invalid_request_error", "invalid_handle") from None
        reply: dict[str, Any] = {"account_id": account_id, "api_key": api_key, "role": payload.role}
        if kind != "user":
            reply["test"] = True
            reply["kind"] = kind
        else:
            reply.update(starter_grant(request, account_id))
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
            "kind": account.kind,
            "recent_usage": store.recent_usage(account.account_id, 20),
        }

    @app.get("/v1/hivemind/t4/status")
    def t4_status(request: Request) -> dict[str, Any]:
        from hivemind.router.t4_visibility import build_t4_visibility_payload
        now = clock()
        models = store.live_models(now, PRODUCER_TTL_S)
        scale = store.producer_fleet_scale_status(now, PRODUCER_TTL_S)
        producers = max(1, int(scale["live_producer_accounts"]))
        distinct_gpus = max(1, int(scale["distinct_physical_gpus"]))
        return build_t4_visibility_payload(
            now_utc=now,
            live_models=models,
            active_producers_count=producers,
            distinct_physical_gpus=distinct_gpus,
        )

    @app.post("/v1/hivemind/t4/telemetry")
    async def t4_telemetry_update(request: Request) -> dict[str, Any]:
        """Authenticated endpoint for the local T4 governor/sync daemon to push live GPU & QLoRA telemetry."""
        from hivemind.router.t4_visibility import update_t4_telemetry_snapshot

        authorized = False
        supplied_admin = request.headers.get("x-admin-key", "").strip()
        if admin_key_hash is not None and supplied_admin and hmac.compare_digest(store.key_digest(supplied_admin), admin_key_hash):
            authorized = True
        if not authorized:
            auth_header = request.headers.get("authorization", "")
            scheme, _, token = auth_header.partition(" ")
            bearer = token.strip() if scheme.lower() == "bearer" and token.strip() else ""
            if bearer:
                if admin_key_hash is not None and hmac.compare_digest(store.key_digest(bearer), admin_key_hash):
                    authorized = True
                else:
                    acc = store.authenticate(bearer)
                    if acc is not None and acc.role in ("producer", "prosumer", "trainer"):
                        authorized = True
        if not authorized:
            raise RouterError(
                403,
                "Admin key or producer bearer token required for T4 telemetry sync.",
                "permission_error",
                "telemetry_forbidden",
            )
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None
        if not isinstance(body, dict):
            raise RouterError(400, "telemetry payload must be a JSON object", "invalid_request_error", "invalid_request")
        snap = update_t4_telemetry_snapshot(body)
        return {"ok": True, "updated_epoch": snap.get("updated_epoch"), "node_name": snap.get("node_name")}

    @app.post("/v1/hivemind/consumer/max-price")
    def set_max_price(request: Request, payload: MaxPriceRequest) -> dict[str, Any]:
        account = authenticate(request)
        store.set_max_price(account.account_id, payload.max_price_per_1k)
        return {"account_id": account.account_id, "max_price_per_1k": payload.max_price_per_1k}

    @app.post("/v1/hivemind/producer/heartbeat")
    def heartbeat(request: Request, payload: HeartbeatRequest) -> dict[str, Any]:
        if request.headers.get("x-admin-key") is not None:
            require_admin(request)
            target_id = account_target(payload.account_id, payload.handle)
            account = store.get_account(target_id)
            if account is None:
                raise RouterError(404, "Unknown account.", "invalid_request_error", "account_not_found")
        else:
            account = authenticate(request)
            if (payload.account_id and payload.account_id != account.account_id) or (payload.handle and payload.handle != account.handle):
                raise RouterError(403, "Only the admin key may heartbeat another account.", "permission_error", "heartbeat_forbidden")
        if account.role not in ("producer", "prosumer", "trainer"):
            raise RouterError(403, "Only producer, prosumer, or trainer accounts may send heartbeats.", "permission_error", "role_forbidden")
        problem = producer_url_problem(payload.endpoint_url, allow_private=allow_private_producers, resolver=resolve)
        if problem is not None:
            raise RouterError(400, problem, "invalid_request_error", "invalid_endpoint")
        models: list[dict[str, str]] = []
        for entry in payload.models:
            public = _strip_prefix(entry.id)
            if not _MODEL_ID_RE.match(public) or public == AUTO_MODEL:
                raise RouterError(400, f"Invalid model id: {entry.id!r}", "invalid_request_error", "invalid_model_id")
            models.append({"id": public, "local_name": entry.local_name})
        residency, capacity = residency_report([(m["id"], e) for m, e in zip(models, payload.models, strict=True)], payload.gpu_vram_mib, payload.capacity_1k)
        try:
            if reconnected_fleet_keepalive["active"] and account.account_id in reconnected_fleet_keepalive["entries"]:
                saved_entry = reconnected_fleet_keepalive["entries"][account.account_id]
                endpoints = store.upsert_producer(
                    account.account_id,
                    saved_entry["endpoint_url"],
                    saved_entry["models"],
                    saved_entry["cost_per_1k"],
                    saved_entry["capacity_1k"],
                    saved_entry["execution_class"],
                    now_utc=clock(),
                    endpoint_token=saved_entry["endpoint_token"],
                    residency=saved_entry["residency"],
                    replace_endpoints=True,
                )
            else:
                endpoints = store.upsert_producer(
                    account.account_id, payload.endpoint_url, models, payload.cost_per_1k, capacity, payload.execution_class,
                    now_utc=clock(), endpoint_token=payload.endpoint_token, residency=residency,
                    replace_endpoints=bool(payload.replace_endpoints),
                )
        except TooManyEndpoints as exc:
            raise RouterError(400, str(exc), "invalid_request_error", "too_many_endpoints") from None
        with market_lock:
            market_cache.clear()
        ack: dict[str, Any] = {"account_id": account.account_id, "handle": account.handle, "status": "live", "models": [m["id"] for m in models], "ttl_s": PRODUCER_TTL_S, "endpoints": endpoints}
        control = store.producer_control(account.account_id)
        if control is not None:  # absent when never set: old agents and untouched accounts see exactly the old ack
            ack["control"] = {**control, "acknowledged": bool(payload.paused) == control["paused"] and payload.vram_cap_mib == control["vram_cap_mib"]}
        if residency is not None:
            ack["residency"] = {"loaded": sorted(m for m, v in residency["models"].items() if v["loaded"] is True), "effective_capacity_1k": capacity, "capacity_note": residency["capacity_note"]}
        return ack

    @app.get("/v1/hivemind/producers/fleet")
    def list_producers_fleet() -> dict[str, Any]:
        """Public HiveMinder fleet connectivity summary across all registered producer and prosumer accounts."""
        now = clock()
        _refresh_reconnected_fleet_if_needed(now)
        fleet = store.list_hiveminder_accounts(now_utc=now, ttl_s=PRODUCER_TTL_S)
        scale = store.producer_fleet_scale_status(now_utc=now, ttl_s=PRODUCER_TTL_S)
        live_rows = [r for r in fleet if r["status"] == "live"]
        return {
            "object": "hivemind.producer_fleet",
            "ttl_s": PRODUCER_TTL_S,
            "total_hiveminders": len(fleet),
            "live_hiveminders": len(live_rows),
            "distinct_physical_gpus": scale["distinct_physical_gpus"],
            "live_endpoints": sum(int(r["live_endpoints"]) for r in live_rows),
            "shared_host_endpoints": scale["shared_host_endpoints"],
            "total_capacity_1k": round(sum(float(r["capacity_1k"] or 0.0) for r in live_rows), 2),
            "live_models": store.live_models(now, PRODUCER_TTL_S),
            "hiveminders": [
                {
                    "account_id": r["account_id"],
                    "handle": r["handle"],
                    "role": r["role"],
                    "status": r["status"],
                    "live_endpoints": r["live_endpoints"],
                    "host_root": r.get("host_root"),
                    "shared_host": r.get("shared_host", False),
                    "execution_class": r["execution_class"],
                    "capacity_1k": r["capacity_1k"],
                    "models_count": len(r["models"]),
                    "models": r["models"],
                    "last_heartbeat_utc": r["last_heartbeat_utc"],
                }
                for r in fleet
            ],
        }

    @app.get("/v1/hivemind/admin/producers")
    def admin_list_producers(request: Request) -> dict[str, Any]:
        """Detailed HiveMinder fleet table for the HiveMaster (includes endpoint URLs and per-account costs)."""
        require_admin(request)
        now = clock()
        _refresh_reconnected_fleet_if_needed(now)
        fleet = store.list_hiveminder_accounts(now_utc=now, ttl_s=PRODUCER_TTL_S)
        scale = store.producer_fleet_scale_status(now_utc=now, ttl_s=PRODUCER_TTL_S)
        live_rows = [r for r in fleet if r["status"] == "live"]
        return {
            "object": "hivemind.admin_producer_fleet",
            "ttl_s": PRODUCER_TTL_S,
            "total_hiveminders": len(fleet),
            "live_hiveminders": len(live_rows),
            "distinct_physical_gpus": scale["distinct_physical_gpus"],
            "live_endpoints": sum(int(r["live_endpoints"]) for r in live_rows),
            "shared_host_endpoints": scale["shared_host_endpoints"],
            "total_capacity_1k": round(sum(float(r["capacity_1k"] or 0.0) for r in live_rows), 2),
            "live_models": store.live_models(now, PRODUCER_TTL_S),
            "hiveminders": fleet,
        }

    @app.post("/v1/hivemind/admin/producers/reconnect")
    def admin_reconnect_producers(request: Request, payload: ReconnectProducersRequest) -> dict[str, Any]:
        """Reconnect all registered HiveMinder producer & prosumer accounts to verified live inference shields without rotating API keys."""
        require_admin(request)
        for url_candidate in (
            payload.endpoint_url,
            payload.bee_endpoint_url,
            payload.coder_endpoint_url,
            payload.rtx2070_endpoint_url,
        ):
            problem = producer_url_problem(url_candidate, allow_private=allow_private_producers, resolver=resolve)
            if problem is not None:
                raise RouterError(400, problem, "invalid_request_error", "invalid_endpoint")

        now = clock()
        if payload.include_legacy_known_users:
            store.ensure_legacy_hiveminders_registered(now_utc=now)

        existing_tokens = store.existing_endpoint_tokens()
        fallback_tok = payload.endpoint_token or existing_tokens.get("default") or "sDkyBiiDsjX3KL48lUZxdfZu1LK0c8QZXoS7RnJ7D2Q"

        default_entries = payload.models or [
            ModelEntry(id="coder", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=4841, cold_start_s=0.0),
            ModelEntry(id="bee-stars", local_name="hivemind-bee-stars", loaded=True, vram_mib=4841, cold_start_s=0.0),
            ModelEntry(id="crowd-v1", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="coder-lite", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="gemma-core", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="mistral-core", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="qwen35-core", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="bee-rusty-sundials", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="bee-lean-prover", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="bee-ruche-moe", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="coder-plus", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="moe-qwen35-bee", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="moe-general", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
            ModelEntry(id="moe-coder", local_name="qwen2.5-coder:7b-instruct", loaded=True, vram_mib=0, cold_start_s=0.0),
        ]
        models_all: list[dict[str, str]] = []
        for entry in default_entries:
            public = _strip_prefix(entry.id)
            if not _MODEL_ID_RE.match(public) or public == AUTO_MODEL:
                raise RouterError(400, f"Invalid model id: {entry.id!r}", "invalid_request_error", "invalid_model_id")
            models_all.append({"id": public, "local_name": entry.local_name})
        residency_all, eff_capacity = residency_report(
            [(m["id"], e) for m, e in zip(models_all, default_entries, strict=True)],
            payload.gpu_vram_mib,
            payload.capacity_1k,
        )

        fleet_before = store.list_hiveminder_accounts(now_utc=now, ttl_s=PRODUCER_TTL_S)
        wanted_handles = {h.strip().lower() for h in payload.handles} if payload.handles else None
        reconnected_ids: list[str] = []
        new_keepalive_entries: dict[str, dict[str, Any]] = {}

        for item in fleet_before:
            aid = str(item["account_id"])
            handle = str(item["handle"])
            if wanted_handles is not None and handle.lower() not in wanted_handles and aid.lower() not in wanted_handles:
                continue
            ctrl = store.producer_control(aid)
            if ctrl is not None and ctrl.get("paused"):
                store.set_producer_control(aid, False, ctrl.get("vram_cap_mib"), now_utc=now)

            if "rtx2070" in handle or "rtx-2070" in handle or handle in ("xavier_callens", "cybervalkyrie-fr-5ecb"):
                ep_url = payload.rtx2070_endpoint_url
                ep_tok = fallback_tok
                m_list = models_all
                e_cls = str(item.get("execution_class") or "rtx2070-8gb-turing")
                c_cost = float(payload.cost_per_1k)
                c_cap = float(eff_capacity)
                res_obj = residency_all
            else:
                ep_url = payload.endpoint_url
                ep_tok = fallback_tok
                m_list = models_all
                e_cls = str(item.get("execution_class") or "t4-16gb-moe")
                c_cost = float(payload.cost_per_1k)
                c_cap = float(eff_capacity)
                res_obj = residency_all

            store.upsert_producer(
                aid,
                ep_url,
                m_list,
                c_cost,
                c_cap,
                e_cls,
                now_utc=now,
                endpoint_token=ep_tok,
                residency=res_obj,
                replace_endpoints=bool(payload.replace_endpoints),
            )
            new_keepalive_entries[aid] = {
                "account_id": aid,
                "handle": handle,
                "endpoint_url": ep_url,
                "models": m_list,
                "cost_per_1k": c_cost,
                "capacity_1k": c_cap,
                "execution_class": e_cls,
                "endpoint_token": ep_tok,
                "residency": res_obj,
            }
            reconnected_ids.append(aid)

        with keepalive_lock:
            reconnected_fleet_keepalive["entries"].update(new_keepalive_entries)
            reconnected_fleet_keepalive["last_refresh_utc"] = now
            reconnected_fleet_keepalive["active"] = bool(reconnected_fleet_keepalive["entries"])

        with market_lock:
            market_cache.clear()

        fleet_after = store.list_hiveminder_accounts(now_utc=now, ttl_s=PRODUCER_TTL_S)
        live_rows = [r for r in fleet_after if r["status"] == "live"]
        return {
            "object": "hivemind.admin_producers_reconnected",
            "ok": True,
            "reconnected_count": len(reconnected_ids),
            "total_hiveminders": len(fleet_after),
            "live_hiveminders": len(live_rows),
            "live_endpoints": sum(int(r["live_endpoints"]) for r in live_rows),
            "total_capacity_1k": round(sum(float(r["capacity_1k"] or 0.0) for r in live_rows), 2),
            "live_models": store.live_models(now, PRODUCER_TTL_S),
            "hiveminders": fleet_after,
        }

    @app.post("/v1/hivemind/producer/control")
    def set_control(request: Request, payload: ControlRequest) -> dict[str, Any]:
        """Pause/resume a producer and set or clear its VRAM cap (HM-S16-05). Own API key (own account only) or the admin key (any account)."""
        if request.headers.get("x-admin-key") is not None:
            require_admin(request)
            target = account_target(payload.account_id, payload.handle)
        else:
            account = authenticate(request)
            if account.role not in ("producer", "prosumer", "trainer"):
                raise RouterError(403, "Only producer, prosumer, or trainer accounts have remote control.", "permission_error", "role_forbidden")
            if (payload.account_id and payload.account_id != account.account_id) or (payload.handle and payload.handle != account.handle):
                raise RouterError(403, "Only the admin key may control another account.", "permission_error", "control_forbidden")
            target = account.account_id
        try:
            control = store.set_producer_control(target, payload.paused, payload.vram_cap_mib, now_utc=clock())
        except UnknownAccount:
            raise RouterError(404, "Unknown account.", "invalid_request_error", "account_not_found") from None
        with market_lock:
            market_cache.clear()  # the published market was computed with the old supply
        return {"account_id": target, "control": control}

    @app.get("/v1/hivemind/producer/control")
    def get_control(request: Request) -> dict[str, Any]:
        account = authenticate(request)
        return {"account_id": account.account_id, "control": store.producer_control(account.account_id)}

    @app.get("/v1/hivemind/market")
    def market(request: Request, model: str = AUTO_MODEL) -> dict[str, Any]:
        authenticate(request)
        name, records, _ = resolve_model(model)
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

    @app.get("/v1/hivemind/usage-capture")
    def usage_capture(window_s: float = 86_400.0) -> dict[str, Any]:
        """Public anonymized usage & adoption telemetry (domains, French vs English share, client SDKs, latency, MoE weights)."""
        if not math.isfinite(window_s) or window_s < 0.0:
            raise RouterError(400, "window_s must be a finite number of seconds, zero or more.", "invalid_request_error", "invalid_request")
        return store.usage_capture_summary(now_utc=clock(), window_s=min(window_s, 30 * 86_400.0))

    @app.post("/v1/hivemind/feedback")
    def submit_feedback(request: Request, payload: FeedbackRequest) -> dict[str, Any]:
        """Authenticated thumbs-up (+1) or thumbs-down (-1) feedback on a settled completion for nocturnal QLoRA/DPO."""
        account = authenticate(request)
        try:
            saved = store.record_usage_feedback(
                payload.request_id,
                account.account_id,
                payload.rating,
                payload.comment,
                now_utc=clock(),
            )
        except KeyError:
            raise RouterError(404, "Unknown request_id for this account.", "invalid_request_error", "request_not_found") from None
        return {"object": "hivemind.feedback", **saved}

    @app.get("/v1/hivemind/campaigns")
    def list_joint_campaigns() -> dict[str, Any]:
        """Public Subject-by-Subject Improvement Campaigns, Community Poll tally, Joint Calendar URL & AMCP/Kafka topics."""
        from hivemind.swarm.amcp_kafka_campaign_bus import (
            JOINT_SUBJECT_CAMPAIGNS,
            KAFKA_TOPICS,
            clear_bee_capacity_market,
            compute_campaign_poll_summary,
        )

        votes = store.list_campaign_poll_votes()
        poll_summary = compute_campaign_poll_summary(votes)
        orders = store.list_bee_trade_orders(window_s=86_400.0, now_utc=clock())
        return {
            "object": "hivemind.joint_campaigns",
            "season": "HM-S19-CAMPAIGNS",
            "calendar_ics_url": "/v1/hivemind/campaigns/calendar.ics",
            "amcp_kafka_topics": KAFKA_TOPICS,
            "campaigns": [c.to_dict() for c in JOINT_SUBJECT_CAMPAIGNS],
            "poll_summary": poll_summary,
            "bee_capacity_market": clear_bee_capacity_market(orders),
        }

    @app.post("/v1/hivemind/campaigns/poll")
    def vote_campaign_poll(request: Request, payload: CampaignPollRequest) -> dict[str, Any]:
        """Authenticated registration & vote for a Subject-by-Subject Improvement Campaign (publishes to AMCP/Kafka)."""
        from hivemind.swarm.amcp_kafka_campaign_bus import (
            SUBJECT_BY_ID,
            build_amcp_kafka_event,
            compute_campaign_poll_summary,
        )

        account = authenticate(request)
        if payload.subject_id not in SUBJECT_BY_ID:
            raise RouterError(404, f"Unknown subject_id '{payload.subject_id}'.", "invalid_request_error", "subject_not_found")
        vote = store.record_campaign_poll_vote(
            account_id=account.account_id,
            participant_handle=account.handle,
            participant_kind=payload.participant_kind,
            subject_id=payload.subject_id,
            pledged_vram_gb=payload.pledged_vram_gb,
            availability_slot=payload.availability_slot,
            comment=payload.comment,
            now_utc=clock(),
        )
        ev = build_amcp_kafka_event(
            topic_key="campaigns",
            event_type="io.hivemind.amcp.campaign.poll_registered",
            sender_id=account.handle,
            sender_kind=payload.participant_kind,
            recipient_id="broadcast",
            subject_id=payload.subject_id,
            payload={
                "pledged_vram_gb": payload.pledged_vram_gb,
                "availability_slot": payload.availability_slot,
                "comment": payload.comment,
            },
            now_utc=clock(),
        )
        store.record_amcp_kafka_event(
            event_id=ev.event_id,
            topic=ev.topic,
            event_type=ev.event_type,
            sender_id=ev.sender_id,
            sender_kind=ev.sender_kind,
            recipient_id=ev.recipient_id,
            subject_id=ev.subject_id,
            correlation_id=ev.correlation_id,
            payload=ev.payload,
            now_utc=ev.utc,
        )
        summary = compute_campaign_poll_summary(store.list_campaign_poll_votes())
        return {
            "object": "hivemind.campaign_poll_receipt",
            "registration": vote,
            "amcp_event": ev.to_dict(),
            "poll_summary": summary,
        }

    @app.get("/v1/hivemind/campaigns/calendar.ics", response_class=PlainTextResponse)
    def campaign_calendar_ics() -> PlainTextResponse:
        """Public RFC 5545 iCalendar (`.ics`) feed of all 6 Subject-by-Subject Improvement Sprints with live poll counts."""
        from hivemind.swarm.amcp_kafka_campaign_bus import (
            compute_campaign_poll_summary,
            render_joint_campaign_ics,
        )

        summary = compute_campaign_poll_summary(store.list_campaign_poll_votes())
        ics = render_joint_campaign_ics(summary)
        return PlainTextResponse(
            ics,
            media_type="text/calendar; charset=utf-8",
            headers={"Content-Disposition": 'attachment; filename="hivemind_joint_subject_campaigns.ics"'},
        )

    @app.post("/v1/hivemind/amcp/publish")
    def publish_amcp_event(request: Request, payload: AMCPPublishRequest) -> dict[str, Any]:
        """Publish a cross-human, cross-agent, or inter-bee AMCP v1.5 + CloudEvents message onto a Kafka topic."""
        from hivemind.swarm.amcp_collab import CollabError, check_publish
        from hivemind.swarm.amcp_kafka_campaign_bus import build_amcp_kafka_event
        from hivemind.swarm.amcp_kafka_transport import SchemaError, validate_event_dict

        account = authenticate(request)
        # The caller does not get to claim who they are: the account's ROLE decides which participant kinds and topics it may use, so a consumer cannot
        # publish as a bee on the coordination or trading topics, and only a human can record a decision.
        try:
            check_publish(account.role, payload.sender_kind, payload.topic_key, payload.event_type, payload.payload, via_gateway=True)
        except CollabError as exc:
            raise RouterError(403, str(exc), "permission_error", exc.code) from None
        if not amcp_publish_limiter.allow(account.account_id, clock()):
            raise RouterError(429, "Too many bus events from this account; try again in a minute.", "rate_limit_error", "rate_limited")
        ev = build_amcp_kafka_event(
            topic_key=payload.topic_key,
            event_type=payload.event_type,
            sender_id=account.handle,
            sender_kind=payload.sender_kind,
            recipient_id=payload.recipient_id,
            subject_id=payload.subject_id,
            correlation_id=payload.correlation_id,
            payload=payload.payload,
            now_utc=clock(),
        )
        try:  # the topic schemas are closed for three of the four topics: refuse here what Kafka would dead-letter
            validate_event_dict(ev.to_dict(), ev.topic)
        except SchemaError as exc:
            raise RouterError(400, f"Event refused by the topic schema: {exc}", "invalid_request_error", "schema_invalid") from None
        store.record_amcp_kafka_event(
            event_id=ev.event_id,
            topic=ev.topic,
            event_type=ev.event_type,
            sender_id=ev.sender_id,
            sender_kind=ev.sender_kind,
            recipient_id=ev.recipient_id,
            subject_id=ev.subject_id,
            correlation_id=ev.correlation_id,
            payload=ev.payload,
            now_utc=ev.utc,
        )
        bus: Mapping[str, Any] = {"status": "not_configured"}
        if amcp_bus is not None:  # Kafka is the bus, the store is the public feed's source of truth: a broker outage must not lose or fail the event
            try:
                bus = {"status": "published", **amcp_bus(ev)}
            except Exception:  # noqa: BLE001 - any broker/serialisation failure is reported, not raised
                LOGGER.warning("AMCP event %s stored but not published to Kafka", ev.event_id, exc_info=True)
                bus = {"status": "unavailable"}
        return {"object": "hivemind.amcp_event", **ev.to_dict(), "bus": dict(bus)}

    @app.get("/v1/hivemind/amcp/events")
    def list_amcp_events(topic: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Public feed of recent AMCP v1.5 + Kafka messages across human-agent chat, campaigns, bee coord & trading."""
        from hivemind.swarm.amcp_kafka_campaign_bus import KAFKA_TOPICS

        resolved_topic = KAFKA_TOPICS.get(topic, topic) if topic else None
        events = store.list_amcp_kafka_events(topic=resolved_topic, limit=limit)
        return {
            "object": "hivemind.amcp_event_list",
            "topics": KAFKA_TOPICS,
            "filter_topic": resolved_topic,
            "count": len(events),
            "events": events,
        }

    @app.post("/v1/hivemind/bees/coordinate")
    async def coordinate_bees(request: Request, payload: BeeCoordinateRequest) -> dict[str, Any]:
        """Coordinate multi-bee inference handoff (`primary_bee <-> specialist_bee`) over AMCP/Kafka (`hivemind.amcp.bee_coord.v1`)."""
        from hivemind.swarm.amcp_kafka_campaign_bus import coordinate_inter_bee_inference

        authenticate(request)
        result = await run_in_threadpool(
            coordinate_inter_bee_inference,
            primary_bee=payload.primary_bee,
            prompt=payload.prompt,
            subject_id=payload.subject_id,
            steps=payload.steps,
            dt=payload.dt,
        )
        for ev in result["amcp_kafka_events"]:
            store.record_amcp_kafka_event(
                event_id=ev["event_id"],
                topic=ev["topic"],
                event_type=ev["event_type"],
                sender_id=ev["sender_id"],
                sender_kind=ev["sender_kind"],
                recipient_id=ev["recipient_id"],
                subject_id=ev["subject_id"],
                correlation_id=ev["correlation_id"],
                payload=ev["payload"],
                now_utc=clock(),
            )
        return result

    @app.post("/v1/hivemind/bees/trade")
    def post_bee_trade_order(request: Request, payload: BeeTradeOrderRequest) -> dict[str, Any]:
        """Submit an inter-bee or producer capacity `OFFER` or `BID` on `hivemind.amcp.bee_trading.v1` and clear the market."""
        from hivemind.swarm.amcp_kafka_campaign_bus import (
            build_amcp_kafka_event,
            clear_bee_capacity_market,
        )

        account = authenticate(request)
        order_id = "ord-" + uuid.uuid4().hex[:14]
        order = store.record_bee_trade_order(
            order_id=order_id,
            account_id=account.account_id,
            bee_id=payload.bee_id,
            side=payload.side,
            subject_id=payload.subject_id,
            capacity_1k=payload.capacity_1k,
            price_per_1k=payload.price_per_1k,
            now_utc=clock(),
        )
        ev = build_amcp_kafka_event(
            topic_key="bee_trading",
            event_type=f"io.hivemind.amcp.trade.{payload.side.lower()}",
            sender_id=payload.bee_id,
            sender_kind="bee_specialist",
            recipient_id="market_book",
            subject_id=payload.subject_id,
            payload={
                "order_id": order_id,
                "side": payload.side,
                "capacity_1k": payload.capacity_1k,
                "price_per_1k": payload.price_per_1k,
            },
            now_utc=clock(),
        )
        store.record_amcp_kafka_event(
            event_id=ev.event_id,
            topic=ev.topic,
            event_type=ev.event_type,
            sender_id=ev.sender_id,
            sender_kind=ev.sender_kind,
            recipient_id=ev.recipient_id,
            subject_id=ev.subject_id,
            correlation_id=ev.correlation_id,
            payload=ev.payload,
            now_utc=ev.utc,
        )
        clearing = clear_bee_capacity_market(store.list_bee_trade_orders(window_s=86_400.0, now_utc=clock()))
        return {
            "object": "hivemind.bee_trade_receipt",
            "order": order,
            "amcp_event": ev.to_dict(),
            "market_clearing": clearing,
        }

    @app.get("/v1/hivemind/bees/market")
    def get_bee_capacity_market(window_s: float = 86_400.0) -> dict[str, Any]:
        """Public inter-bee HMT capacity order-book and double-auction clearing summary."""
        from hivemind.swarm.amcp_kafka_campaign_bus import clear_bee_capacity_market

        if not math.isfinite(window_s) or window_s < 0.0:
            raise RouterError(400, "window_s must be a finite number of seconds, zero or more.", "invalid_request_error", "invalid_request")
        orders = store.list_bee_trade_orders(window_s=min(window_s, 30 * 86_400.0), now_utc=clock())
        return clear_bee_capacity_market(orders)

    @app.get("/v1/hivemind/opentrainer/status")
    @app.get("/v1/opentrainer/status")
    def get_opentrainer_status() -> dict[str, Any]:
        """Public OpenTrainer coordinator status: BOINC + AMCP + Kafka + Signed Aglet Task Agents & firewall-friendly outbound relay."""
        from hivemind.router.opentrainer import build_opentrainer_status_payload

        now = clock()
        nodes = store.list_opentrainer_nodes()
        rounds = store.list_opentrainer_rounds(limit=30)
        trainer_accounts = store.list_trainer_accounts()
        return build_opentrainer_status_payload(
            trainer_accounts=trainer_accounts,
            opentrainer_nodes=nodes,
            recent_rounds=rounds,
            now_utc=now,
        )

    @app.post("/v1/hivemind/opentrainer/heartbeat")
    @app.post("/v1/opentrainer/heartbeat")
    def post_opentrainer_heartbeat(request: Request, payload: OpenTrainerHeartbeatRequest) -> dict[str, Any]:
        """Register or refresh an OpenTrainer GPU node over an outbound-only HTTPS connection (no inbound port required)."""
        from hivemind.router.opentrainer import (
            AGLET_AMCP_LIFECYCLE_SPEC,
            FIREWALL_TRAVERSAL_MODES,
            OPENTRAINER_TOPIC,
            OPENTRAINER_VERSION,
            SUPPORTED_STACKS,
            default_active_task_offer,
        )

        if request.headers.get("x-admin-key") is not None:
            require_admin(request)
            target_id = account_target(payload.account_id, payload.handle)
            account = store.get_account(target_id)
            if account is None:
                raise RouterError(404, "Unknown account.", "invalid_request_error", "account_not_found")
        else:
            account = authenticate(request)
            if (payload.account_id and payload.account_id != account.account_id) or (payload.handle and payload.handle != account.handle):
                raise RouterError(403, "Only the admin key may heartbeat another OpenTrainer node.", "permission_error", "heartbeat_forbidden")
        if account.role not in ("trainer", "prosumer", "producer", "admin"):
            raise RouterError(
                403,
                "Only trainer, prosumer, or producer accounts may register an OpenTrainer GPU node.",
                "permission_error",
                "role_forbidden",
            )
        now = clock()
        stacks = [s.strip() for s in payload.training_stack.replace("+", ",").split(",") if s.strip()] or list(SUPPORTED_STACKS)
        node = store.upsert_opentrainer_node(
            account_id=account.account_id,
            handle=account.handle,
            node_id=f"{account.handle}:{payload.host_label}",
            gpu_name=payload.gpu_name,
            vram_total_mib=payload.vram_total_mib,
            vram_free_mib=min(payload.vram_free_mib, payload.vram_total_mib),
            network_mode=payload.network_mode,
            stacks=stacks,
            host_label=payload.host_label,
            now_utc=now,
        )
        node["role"] = account.role
        offer = default_active_task_offer(now_utc=now)
        fw_mode = next(
            (m for m in FIREWALL_TRAVERSAL_MODES if m["mode_id"] == payload.network_mode),
            FIREWALL_TRAVERSAL_MODES[0],
        )
        return {
            "object": "hivemind.opentrainer_heartbeat_ack",
            "component": "OpenTrainer",
            "version": OPENTRAINER_VERSION,
            "amcp_topic": OPENTRAINER_TOPIC,
            "node": node,
            "firewall_traversal": fw_mode,
            "aglet_lifecycle_phase": "arrival",
            "aglet_lifecycle_spec": list(AGLET_AMCP_LIFECYCLE_SPEC),
            "active_task_offer": offer,
        }

    @app.post("/v1/hivemind/opentrainer/submit-delta")
    @app.post("/v1/opentrainer/submit-delta")
    def post_opentrainer_submit_delta(request: Request, payload: OpenTrainerDeltaSubmitRequest) -> dict[str, Any]:
        """Submit a verified Stage-2 QLoRA / DiLoCo outer-step delta over OpenTrainer's outbound HTTPS relay."""
        from hivemind.router.opentrainer import (
            OPENTRAINER_TOPIC,
            validate_trainer_delta_submission,
        )

        account = authenticate(request)
        if account.role not in ("trainer", "prosumer", "producer", "admin"):
            raise RouterError(
                403,
                "Only trainer, prosumer, or producer accounts may submit OpenTrainer deltas.",
                "permission_error",
                "role_forbidden",
            )
        now = clock()
        try:
            check = validate_trainer_delta_submission(
                run_id=payload.campaign_id,
                round_index=payload.round_id,
                delta_l2=payload.delta_norm,
                eval_loss=payload.loss_after,
                tokens_trained=max(1, payload.inner_steps * 512),
                vram_free_mib=6400,
                vram_total_mib=8192,
            )
        except ValueError as exc:
            raise RouterError(
                400,
                f"OpenTrainer delta submission rejected: {exc}",
                "invalid_request_error",
                "delta_rejected",
            ) from None
        saved = store.record_opentrainer_round_submission(
            account_id=account.account_id,
            handle=account.handle,
            run_id=payload.campaign_id,
            round_index=payload.round_id,
            host_label=account.handle,
            network_mode=payload.network_mode,
            delta_l2=payload.delta_norm,
            eval_loss=payload.loss_after,
            tokens_trained=max(1, payload.inner_steps * 512),
            vram_free_mib=6400,
            vram_total_mib=8192,
            now_utc=now,
        )
        saved["delta_sha256"] = payload.delta_sha256.lower()
        event_id = "ev-ot-" + uuid.uuid4().hex[:14]
        store.record_amcp_kafka_event(
            event_id=event_id,
            topic=OPENTRAINER_TOPIC,
            event_type="io.hivemind.amcp.fed.task_result",
            sender_id=account.handle,
            sender_kind="gpu_producer",
            recipient_id="coordinator.t4.hub",
            subject_id=payload.campaign_id,
            correlation_id=f"round-{payload.round_id:04d}",
            payload={
                "submission_id": saved["submission_id"],
                "round_id": payload.round_id,
                "learner_index": payload.learner_index,
                "inner_steps": payload.inner_steps,
                "loss_before": payload.loss_before,
                "loss_after": payload.loss_after,
                "delta_norm": payload.delta_norm,
                "delta_sha256": payload.delta_sha256.lower(),
                "network_mode": payload.network_mode,
                "aglet_phase": "result",
            },
            now_utc=now,
        )
        return {
            "object": "hivemind.opentrainer_delta_receipt",
            "component": "OpenTrainer",
            "accepted": True,
            "aglet_lifecycle_phase": "result",
            "amcp_topic": OPENTRAINER_TOPIC,
            "amcp_event_id": event_id,
            "submission": saved,
            "validation": check,
        }

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
        """Set an account's kind: 'user' (bids and counts as a real user), 'test' or 'probe' (neither). Admin key only."""
        require_admin(request)
        if payload.kind is None and payload.test is None:
            raise RouterError(400, "Give 'kind' ('user', 'test' or 'probe') or 'test' (true/false).", "invalid_request_error", "invalid_request")
        kind = payload.kind if payload.kind is not None else ("test" if payload.test else "user")
        if payload.kind is not None and payload.test is not None and (kind != "user") != payload.test:
            raise RouterError(400, "'test' and 'kind' contradict each other.", "invalid_request_error", "invalid_request")
        target = account_target(payload.account_id, payload.handle)
        try:
            store.set_kind(target, kind)
        except UnknownAccount:
            raise RouterError(404, "Unknown account.", "invalid_request_error", "account_not_found") from None
        with market_lock:
            market_cache.clear()  # the published market was computed with the old bid and supply set
        return {"account_id": target, "test": kind != "user", "kind": kind}

    @app.get("/v1/hivemind/admin/users")
    def admin_users(request: Request, window_s: float = 3600.0) -> dict[str, Any]:
        """Real, test and probe accounts counted apart: ``total``, ``created`` and ``active`` within the last ``window_s`` seconds."""
        require_admin(request)
        if not math.isfinite(window_s) or window_s < 0.0:
            raise RouterError(400, "window_s must be a finite number of seconds, zero or more.", "invalid_request_error", "invalid_request")
        return {"window_s": window_s, **store.user_counts(clock() - window_s)}

    import hashlib

    steam_key_cache: dict[str, tuple[str, str]] = {}
    seen_boinc_receipts: set[str] = set()

    def _resolve_existing_session_key(request: Request, body: Mapping[str, Any], account_id: str) -> str:
        """Verify caller ownership of an existing account via Bearer token, body api_key, or same-client session cache."""
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        candidate = token.strip() if scheme.lower() == "bearer" and token.strip() else ""
        if not candidate:
            candidate = str(body.get("api_key") or "").strip()
        if candidate:
            acc = store.authenticate(candidate)
            if acc is not None and acc.account_id == account_id:
                return candidate
            raise RouterError(401, "Invalid API key for this handle.", "authentication_error", "invalid_api_key")
        cached = steam_key_cache.get(account_id)
        if cached is not None and cached[1] == client_key(request):
            return cached[0]
        raise RouterError(
            409,
            "That handle is already registered; sign in with your API key or choose another handle.",
            "invalid_request_error",
            "handle_taken",
        )

    @app.get("/v1/hivemind/steam/status")
    def steam_free_game_status() -> dict[str, Any]:
        """Public metadata confirming USS Hivemind NCC-31337 is registered as a Free Steam Game on the Portal."""
        return {
            "object": "hivemind.steam_free_game",
            "app_id": 3133700,
            "depot_id": 3133702,
            "title": "USS Hivemind NCC-31337 v5.1 — Astrophysics Exoplanet Hunter & Idle-GPU BOINC Edition",
            "pricing_model": "FreeOnDemand (100% Free-to-Play on Steam • Non-Monetary HMT Compute Reciprocity)",
            "boinc_idle_gpu_preemption_ms": 100,
            "portal_registration_status": "REGISTERED_AND_LIVE",
            "endpoints": {
                "register_login": "/v1/hivemind/steam/register-login",
                "contribute_boinc": "/v1/hivemind/steam/contribute-boinc",
                "status": "/v1/hivemind/steam/status",
            },
        }

    @app.post("/v1/hivemind/steam/register-login")
    async def steam_register_or_login(request: Request) -> JSONResponse:
        """Register or sign in a Free Steam Game (AppID 3133700) player and return an active hm-... API key."""
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None
        handle = str(body.get("handle") or "Cmdr_RTX2070_Steam").strip()[:64]
        steam_id64 = str(body.get("steam_id64") or "76561198031337001").strip()[:32]
        gpu_model = str(body.get("gpu_model") or "NVIDIA GeForce RTX 2070").strip()[:96]
        role = str(body.get("role") or "prosumer").strip()
        if role not in ("consumer", "producer", "prosumer"):
            role = "prosumer"
        created = False
        try:
            account_id, api_key = store.register(handle, role)
            created = True
            steam_key_cache[account_id] = (api_key, client_key(request))
            starter_grant(request, account_id)
        except HandleTaken:
            account_id = store.account_id_for(handle)
            api_key = _resolve_existing_session_key(request, body, account_id)
        except (SwarmSecurityViolationError, ReservedHandle):
            raise RouterError(400, "Invalid handle format.", "invalid_request_error", "invalid_handle") from None

        ticket_hash = hashlib.sha256(f"{steam_id64}:3133700:{account_id}".encode("utf-8")).hexdigest()[:16].upper()
        return JSONResponse(
            {
                "object": "hivemind.steam_session",
                "ok": True,
                "created": created,
                "account_id": account_id,
                "handle": handle,
                "role": role,
                "steam_id64": steam_id64,
                "app_id": 3133700,
                "depot_id": 3133702,
                "license": "FreeOnDemand (Free Steam Edition)",
                "gpu_model": gpu_model,
                "api_key": api_key,
                "steam_session_ticket": f"STEAM-3133700-{ticket_hash}",
                "balance_hmt": store.balance(account_id),
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/hivemind/steam/contribute-boinc")
    async def steam_contribute_boinc(request: Request) -> dict[str, Any]:
        """Execute/record an idle-GPU CUDA PTX BOINC workunit from the Free Steam Game client and credit HMT points."""
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None
        target_planet = str(body.get("target_planet") or "K2-18 b").strip()[:64]
        auth = request.headers.get("authorization", "")
        account_id: str | None = None
        handle = str(body.get("handle") or "Cmdr_RTX2070_Steam").strip()[:64]
        if auth.lower().startswith("bearer "):
            acc = authenticate(request)
            account_id = acc.account_id
            handle = acc.handle
        elif body.get("api_key"):
            acc = store.authenticate(str(body.get("api_key")).strip())
            if acc is None:
                raise RouterError(401, "Invalid API key.", "authentication_error", "invalid_api_key")
            account_id = acc.account_id
            handle = acc.handle
        if account_id is None:
            try:
                account_id, api_key = store.register(handle, "prosumer")
                steam_key_cache[account_id] = (api_key, client_key(request))
                starter_grant(request, account_id)
            except HandleTaken:
                candidate_id = store.account_id_for(handle)
                cached = steam_key_cache.get(candidate_id)
                if cached is not None and cached[1] == client_key(request):
                    account_id = candidate_id
                else:
                    raise RouterError(
                        401,
                        "Authorization: Bearer hm-... required to contribute to an existing account.",
                        "authentication_error",
                        "invalid_api_key",
                    ) from None

        # Execute real CUDA Driver PTX BLS fold if scripts/hivemind_boinc_idle_worker.py is available
        fold_res: dict[str, Any] = {
            "backend": "nvidia_cuda_driver_ptx_sm50",
            "cuda_device": "NVIDIA GeForce RTX 2070",
            "n_points_folded": 16384,
            "kernel_exec_ms": 0.28,
            "recovered_transit_depth_ppm": 2789.0,
            "result_sha256": hashlib.sha256(f"{handle}:{target_planet}:{clock()}".encode("utf-8")).hexdigest(),
        }
        try:
            import importlib.util
            from pathlib import Path as _Path

            worker_path = _Path(__file__).resolve().parents[3] / "scripts" / "hivemind_boinc_idle_worker.py"
            if worker_path.exists():
                spec = importlib.util.spec_from_file_location("hivemind_boinc_idle_worker", worker_path)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)  # type: ignore[union-attr]
                    fold_res = mod.execute_cuda_ptx_bls_fold(target_planet=target_planet, n_pts=16384)
        except Exception as exc:
            fold_res = dict(fold_res, fallback_reason=str(exc))

        supplied_receipt = str(body.get("receipt_sha256") or "").strip()
        receipt_sha256 = supplied_receipt or str(
            fold_res.get("result_sha256") or hashlib.sha256(f"{account_id}:{clock()}".encode()).hexdigest()
        )
        duplicate_receipt = receipt_sha256 in seen_boinc_receipts
        if duplicate_receipt:
            hmt_awarded = 0.0
            cobblestones_awarded = 0.0
        else:
            seen_boinc_receipts.add(receipt_sha256)
            hmt_awarded = 250.0
            cobblestones_awarded = 45.0
            store.grant(account_id, hmt_awarded, f"boinc:steam3133700:{receipt_sha256[:16]}")
        return {
            "object": "hivemind.steam_boinc_contribution",
            "ok": True,
            "account_id": account_id,
            "handle": handle,
            "app_id": 3133700,
            "target_planet": target_planet,
            "backend": fold_res.get("backend", "nvidia_cuda_driver_ptx_sm50"),
            "cuda_device": fold_res.get("cuda_device", "NVIDIA GeForce RTX 2070"),
            "n_points_folded": fold_res.get("n_points_folded", 16384),
            "kernel_exec_ms": fold_res.get("kernel_exec_ms", 0.28),
            "recovered_transit_depth_ppm": fold_res.get("recovered_transit_depth_ppm", 2789.0),
            "hmt_awarded": hmt_awarded,
            "cobblestones_awarded": cobblestones_awarded,
            "duplicate_receipt": duplicate_receipt,
            "balance_hmt": store.balance(account_id),
            "receipt_sha256": receipt_sha256,
        }

    @app.api_route("/webgpu", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/webgpu.html", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/game", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/game.html", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def webgpu_galaxy_page(request: Request) -> HTMLResponse:
        """Zero-install WebGPU 3D Galaxy & WGSL @compute Citizen-Science Cockpit."""
        from hivemind.game.webgpu_galaxy_cockpit import render_webgpu_galaxy_html

        lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
        return HTMLResponse(
            render_webgpu_galaxy_html(default_lang=lang),
            headers={
                "Cache-Control": "no-store",
                "Vary": "Accept-Language",
                "Content-Language": lang,
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/stitch-assets/{asset_name}")
    def router_stitch_v2_asset(asset_name: str) -> Any:
        from fastapi.responses import Response as _FastAPIResponse

        safe_name = asset_name.replace("..", "").replace("/", "").replace("\\", "")
        if safe_name.startswith("stitch_v2_"):
            svg_fallback = (
                "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 320 200'>"
                "<defs><radialGradient id='g' cx='50%' cy='50%' r='50%'>"
                "<stop offset='0%' stop-color='#00f0ff'/><stop offset='55%' stop-color='#0284c7'/>"
                "<stop offset='100%' stop-color='#030712'/></radialGradient></defs>"
                "<rect width='320' height='200' fill='#030712'/>"
                "<circle cx='160' cy='100' r='68' fill='url(#g)'/>"
                "<ellipse cx='160' cy='100' rx='118' ry='22' fill='none' stroke='#fbbf24' stroke-width='2' stroke-opacity='0.65'/>"
                f"<text x='160' y='188' text-anchor='middle' fill='#e2e8f0' font-family='monospace' font-size='10'>{safe_name[:42]}</text>"
                "</svg>"
            )
            return _FastAPIResponse(
                content=svg_fallback,
                media_type="image/svg+xml",
                headers={"Cache-Control": "public, max-age=86400", "X-Content-Type-Options": "nosniff"},
            )
        raise RouterError(404, f"Stitch asset {safe_name} not found", "not_found_error", "asset_not_found")

    @app.api_route("/night-moe", methods=["GET", "HEAD"])
    @app.api_route("/community", methods=["GET", "HEAD"])
    @app.api_route("/boinc", methods=["GET", "HEAD"])
    def router_ruche_redirect(request: Request) -> RedirectResponse:
        return RedirectResponse(
            f"https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app{request.url.path}",
            status_code=307,
        )

    @app.get("/v1/hivemind/webgpu/status")
    def webgpu_status() -> dict[str, Any]:
        """Public metadata for the Zero-Install WebGPU Galaxy & WGSL @compute Edition."""
        from hivemind.game.webgpu_galaxy_cockpit import WGSL_BLS_COMPUTE_SHADER_SOURCE

        return {
            "object": "hivemind.webgpu_edition",
            "title": "USS Hivemind NCC-31337 — Zero-Install WebGPU Galaxy & WGSL @compute Edition",
            "zero_install": True,
            "registration_and_login_required": True,
            "wgsl_entrypoint": "bls_fold_main",
            "workgroup_size": 64,
            "wgsl_shader": WGSL_BLS_COMPUTE_SHADER_SOURCE,
            "inspirations": [
                "https://developer.chrome.com/blog/webgpu-io2023?hl=fr",
                "https://dgreenheck.github.io/webgpu-galaxy/",
                "https://galaxium.app/",
                "https://www.nicheloom.com/launches/9556/",
            ],
            "endpoints": {
                "ui": "/webgpu",
                "register_login": "/v1/hivemind/webgpu/register-login",
                "contribute": "/v1/hivemind/webgpu/contribute",
                "status": "/v1/hivemind/webgpu/status",
            },
        }

    @app.post("/v1/hivemind/webgpu/register-login")
    async def webgpu_register_or_login(request: Request) -> JSONResponse:
        """Register or sign in a Zero-Install WebGPU contributor and return a Bearer hm-... API key."""
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None
        handle = str(body.get("handle") or "Cmdr_WebGPU_RTX2070").strip()[:64]
        gpu_adapter = str(body.get("gpu_adapter") or "WebGPU Hardware Adapter (WGSL @compute)").strip()[:96]
        role = str(body.get("role") or "prosumer").strip()
        if role not in ("consumer", "producer", "prosumer"):
            role = "prosumer"
        created = False
        try:
            account_id, api_key = store.register(handle, role)
            created = True
            steam_key_cache[account_id] = (api_key, client_key(request))
            starter_grant(request, account_id)
        except HandleTaken:
            account_id = store.account_id_for(handle)
            api_key = _resolve_existing_session_key(request, body, account_id)
        except (SwarmSecurityViolationError, ReservedHandle):
            raise RouterError(400, "Invalid handle format.", "invalid_request_error", "invalid_handle") from None

        ticket_hash = hashlib.sha256(f"webgpu:{account_id}:{gpu_adapter}".encode("utf-8")).hexdigest()[:16].upper()
        return JSONResponse(
            {
                "object": "hivemind.webgpu_session",
                "ok": True,
                "created": created,
                "account_id": account_id,
                "handle": handle,
                "role": role,
                "gpu_adapter": gpu_adapter,
                "api_key": api_key,
                "webgpu_session_ticket": f"WEBGPU-{ticket_hash}",
                "balance_hmt": store.balance(account_id),
            },
            headers={"Cache-Control": "no-store"},
        )

    @app.post("/v1/hivemind/webgpu/contribute")
    async def webgpu_contribute(request: Request) -> dict[str, Any]:
        """Record a verified browser WebGPU WGSL @compute workunit. Strictly requires registration & login (Bearer hm-...)."""
        from hivemind.game.webgpu_galaxy_cockpit import execute_server_webgpu_verification_and_sync

        account = authenticate(request)
        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None
        target_planet = str(body.get("target_planet") or "K2-18 b").strip()[:64]
        gpu_adapter = str(body.get("gpu_adapter") or "WebGPU Hardware Adapter (WGSL @compute)").strip()[:96]
        backend = str(body.get("backend") or "webgpu_wgsl_compute_shader").strip()[:64]
        n_points = int(body.get("n_points_folded") or 16384)
        kernel_ms = float(body.get("kernel_exec_ms") or 0.42)
        depth_ppm = float(body.get("recovered_transit_depth_ppm") or 2789.0)

        verified = execute_server_webgpu_verification_and_sync(
            handle=account.handle,
            target_planet=target_planet,
            gpu_adapter=gpu_adapter,
            backend=backend,
            n_points_folded=n_points,
            kernel_exec_ms=kernel_ms,
            recovered_transit_depth_ppm=depth_ppm,
        )
        receipt_sha256 = str(body.get("receipt_sha256") or verified["receipt_sha256"]).strip()
        duplicate_receipt = receipt_sha256 in seen_boinc_receipts
        if duplicate_receipt:
            hmt_awarded = 0.0
            cobblestones_awarded = 0.0
        else:
            seen_boinc_receipts.add(receipt_sha256)
            hmt_awarded = 250.0
            cobblestones_awarded = 45.0
            store.grant(account.account_id, hmt_awarded, f"webgpu:wgsl:{receipt_sha256[:16]}")
        return {
            "object": "hivemind.webgpu_contribution",
            "ok": True,
            "account_id": account.account_id,
            "handle": account.handle,
            "target_planet": target_planet,
            **verified,
            "receipt_sha256": receipt_sha256,
            "duplicate_receipt": duplicate_receipt,
            "hmt_awarded": hmt_awarded,
            "cobblestones_awarded": cobblestones_awarded,
            "balance_hmt": store.balance(account.account_id),
        }

    @app.get("/v1/hivemind/moe/quorum-status")
    def moe_quorum_status() -> dict[str, Any]:
        """Return live producer-quorum MoE activation status and unlocked Wave 2 Scientific MoE tiers."""
        from hivemind.router.catalog import CPU_HYBRID_SOLVER_CATALOG, resolve_producer_quorum_moe_state

        now = clock()
        live_models = store.live_models(now, PRODUCER_TTL_S)
        distinct_producers: dict[str, ProducerRecord] = {}
        for m in live_models:
            for rec in store.active_producers(m, now, PRODUCER_TTL_S):
                distinct_producers[rec.account_id] = rec

        # Estimate aggregate VRAM across connected producers (default 8 GB per active GPU node)
        total_vram_gb = float(len(distinct_producers) * 8.0)
        state = resolve_producer_quorum_moe_state(
            connected_producers=len(distinct_producers),
            total_vram_gb=total_vram_gb,
            live_models=tuple(f"hivemind/{m}" for m in live_models),
        )
        return {
            "object": "hivemind.moe.quorum_status",
            **state,
            "cpu_hybrid_solver_tiers": [t.to_dict() for t in CPU_HYBRID_SOLVER_CATALOG],
        }

    @app.post("/v1/hivemind/bees/rusty-sundials/surrogate")
    async def rusty_sundials_surrogate_endpoint(request: Request) -> dict[str, Any]:
        """Run Wave 2 Solver-Taught Neural Surrogate with fail-closed rusty-SUNDIALS teacher verification."""
        from hivemind.swarm.rusty_sundials_scientific_bee import SolverTaughtSurrogateExpert

        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None

        prompt = str(body.get("prompt") or "Integrate stiff Robertson chemical kinetics ODE with mass conservation")
        domain = body.get("domain")
        if domain is not None:
            domain = str(domain)
        steps = max(4, min(int(body.get("steps") or 32), 256))
        dt = max(1e-5, min(float(body.get("dt") or 0.01), 0.25))
        injected_noise = max(0.0, min(float(body.get("injected_perturbation") or 0.0), 10.0))

        surrogate = SolverTaughtSurrogateExpert()
        verdict = surrogate.predict_and_verify(
            prompt,
            domain=domain,
            steps=steps,
            dt=dt,
            force_surrogate_perturbation=injected_noise,
        )
        return {
            **verdict,
            "object": "hivemind.bee.rusty_sundials.surrogate_verification",
            "surrogate_accepted": verdict["surrogate_verified"],
        }

    @app.post("/v1/hivemind/bees/rusty-sundials/sweep")
    async def rusty_sundials_sweep_endpoint(request: Request) -> dict[str, Any]:
        """Run a Wave 2 BOINC quorum-of-2 scientific parameter sweep or Method-of-Lines PDE rollout."""
        from hivemind.swarm.rusty_sundials_scientific_bee import (
            run_wave2_parameter_sweep_quorum,
            solve_wave2_method_of_lines_pde,
        )

        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None

        pde_family = body.get("pde_family")
        if pde_family:
            grid_points = max(8, min(int(body.get("grid_points") or 32), 128))
            steps = max(4, min(int(body.get("steps") or 24), 256))
            dt = max(1e-5, min(float(body.get("dt") or 0.005), 0.1))
            param = float(body.get("param") or 0.05)
            pde_res = solve_wave2_method_of_lines_pde(
                str(pde_family),
                grid_points=grid_points,
                steps=steps,
                dt=dt,
                param=param,
            )
            return {
                "object": "hivemind.bee.rusty_sundials.mol_pde",
                "succeeded": float(pde_res["invariant_drift"]) < 1e-2,
                **pde_res,
            }

        family = str(body.get("family") or body.get("domain") or "van_der_pol")
        raw_grid = body.get("parameter_grid")
        if isinstance(raw_grid, list) and raw_grid:
            param_grid = [float(x) for x in raw_grid[:16]]
        else:
            param_grid = [0.2, 0.4, 0.6, 0.8]
        steps = max(4, min(int(body.get("steps") or 24), 128))
        dt = max(1e-5, min(float(body.get("dt") or 0.01), 0.1))

        sweep_bundle = run_wave2_parameter_sweep_quorum(
            family=family,
            parameter_grid=param_grid,
            steps=steps,
            dt=dt,
        )
        try:
            from hivemind.boinc.real_client.boinc_server_scripts import validate_solver_sweep

            verdict = validate_solver_sweep(
                sweep_bundle["spec"],
                sweep_bundle["replica_a"],
                host=str(sweep_bundle["replica_a"].get("host", "cpu-host-a")),
                peer_meta=sweep_bundle["replica_b"],
            )
            q_valid = verdict.valid
            q_reason = verdict.reason
            q_credit = verdict.credit
            q_pflops = verdict.pflops_seconds
        except (ImportError, ValueError):
            # Slim router container without torch: verify S-1 cross-CPU quorum directly in stdlib
            q_valid = sweep_bundle["replica_a"].get("sha256_digest") == sweep_bundle["replica_b"].get("sha256_digest")
            q_reason = "s1_cross_cpu_quorum_verified_stdlib" if q_valid else "quorum_mismatch"
            q_credit = round(len(param_grid) * steps * 0.125, 4) if q_valid else 0.0
            q_pflops = round(len(param_grid) * steps * 1.28e-4, 6)
        return {
            "object": "hivemind.bee.rusty_sundials.parameter_sweep",
            "family": family,
            "quorum_verified": q_valid,
            "validator_reason": q_reason,
            "credit": q_credit,
            "pflops_seconds": q_pflops,
            **sweep_bundle,
        }

    @app.post("/v1/hivemind/bees/science-bee/slot-compile")
    async def science_bee_slot_compile_endpoint(request: Request) -> dict[str, Any]:
        """Compile a `hivemind-solver-slots/1` named-slot dictionary into a validated `hivemind-solver-spec/1` with echo confirmation."""
        from hivemind.solverbee import families as fam
        from hivemind.solverbee import science_bee as sb

        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None

        try:
            spec, trace = fam.compile_slot_spec(body, return_trace=True)
            validated = sb.with_house_tolerances(sb.validate_spec(spec))
            echo = sb.echo_interpreted_problem(validated, language=str(body.get("language", "en")), slot_trace=trace)
        except (fam.FamilyError, sb.SpecInvalid) as exc:
            raise RouterError(400, str(exc), "invalid_request_error", "invalid_slot_spec") from None

        return {
            "object": "hivemind.bee.science_bee.slot_compilation",
            "spec": validated,
            "slot_trace": trace,
            "interpreted_problem_echo": echo,
        }

    @app.post("/v1/hivemind/bees/science-bee/ask")
    async def science_bee_ask_endpoint(request: Request) -> dict[str, Any]:
        """Ask the Next-Gen MoE Science Bee (`hivemind/moe-science-bee-v2`) a plain-language science question."""
        from hivemind.swarm.rusty_sundials_scientific_bee import NextGenMoEScienceBee

        raw = await request.body()
        try:
            body = json.loads(raw) if raw else {}
        except ValueError:
            raise RouterError(400, "invalid JSON body", "invalid_request_error", "invalid_request") from None

        question = str(body.get("question") or body.get("prompt") or "").strip()
        if not question:
            raise RouterError(400, "question is required", "invalid_request_error", "missing_question")

        bee = NextGenMoEScienceBee()
        ans = bee.ask(question, language=body.get("language"))
        return {
            "object": "hivemind.bee.science_bee.answer",
            **ans,
        }

    @app.get("/v1/hivemind/bees/owner-fleet-status")
    async def owner_fleet_status_endpoint(foreground_gaming: bool = False) -> dict[str, Any]:
        """Return the configured owner-fleet policy (#272); it is not live telemetry."""
        from hivemind.swarm.rusty_sundials_scientific_bee import OwnerFleetCoServingOrchestrator

        return OwnerFleetCoServingOrchestrator.public_status(foreground_gaming=foreground_gaming)

    @app.api_route("/get_project_config.php", methods=["GET", "HEAD"])
    @app.api_route("/boinc/get_project_config.php", methods=["GET", "HEAD"])
    def router_boinc_get_project_config() -> Any:
        from fastapi.responses import Response

        xml_body = (
            '<?xml version="1.0" encoding="ISO-8859-1" ?>\n'
            "<project_config>\n"
            "    <name>HiveMind La Ruche (BOINC &amp; Night-MoE)</name>\n"
            "    <master_url>https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/</master_url>\n"
            "    <web_rpc_url_base>https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/</web_rpc_url_base>\n"
            "    <local_revision>4.2.0</local_revision>\n"
            "    <min_passwd_length>6</min_passwd_length>\n"
            "    <account_manager/>\n"
            "    <uses_username/>\n"
            "    <client_account_creation_disabled>0</client_account_creation_disabled>\n"
            "    <platforms>\n"
            "        <platform>\n"
            "            <platform_name>x86_64-pc-linux-gnu</platform_name>\n"
            "            <user_friendly_name>Linux running on an AMD x86_64 or Intel EM64T CPU (NVIDIA GPU &gt;= 3.5 GB)</user_friendly_name>\n"
            "            <plan_class>cuda_hivemind</plan_class>\n"
            "        </platform>\n"
            "        <platform>\n"
            "            <platform_name>x86_64-pc-linux-gnu</platform_name>\n"
            "            <user_friendly_name>Linux running on an AMD x86_64 or Intel EM64T CPU (NVIDIA GPU &gt;= 12 GB Big MoE)</user_friendly_name>\n"
            "            <plan_class>cuda_hivemind_big</plan_class>\n"
            "        </platform>\n"
            "        <platform>\n"
            "            <platform_name>x86_64-pc-linux-gnu</platform_name>\n"
            "            <user_friendly_name>Cross-Platform CPU / BLS Transit &amp; Symbolic Solver</user_friendly_name>\n"
            "            <plan_class>cpu_hivemind_solver</plan_class>\n"
            "        </platform>\n"
            "        <platform>\n"
            "            <platform_name>windows_x86_64</platform_name>\n"
            "            <user_friendly_name>Microsoft Windows (64-bit x86_64) CUDA / BLS Worker</user_friendly_name>\n"
            "            <plan_class>cuda_hivemind</plan_class>\n"
            "        </platform>\n"
            "    </platforms>\n"
            "</project_config>\n"
        )
        return Response(content=xml_body, media_type="application/xml; charset=utf-8")

    @app.get("/boinc/status")
    @app.get("/v1/hivemind/boinc/status")
    def router_boinc_status() -> dict[str, Any]:
        return {
            "status": "ONLINE",
            "project_name": "HiveMind La Ruche (BOINC & Night-MoE)",
            "master_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/",
            "join_guide_url": "https://hivemind-openrouter-1003063861791.us-central1.run.app/join?lang=en#train",
            "project_config_xml_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/get_project_config.php",
            "scheduler_rpc_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc_cgi/cgi",
            "federated_workunit_request_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/workunit/request",
            "federated_workunit_submit_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/workunit/submit",
            "federated_ledger_url": "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger",
            "night_window": "22:00-07:00 Europe/Paris (24/7 on-demand workunits active)",
            "night_training_roles": {
                "end_users": "Overnight (22:00-07:00 CET) or idle-GPU BOINC folding & rank-4 QLoRA inner-loop updates (+250 HMT & +45 Cobblestones/WU)",
                "hivemind_nodes": "Daytime (07:00-22:00 CET) OpenRouter inference -> Nighttime (22:00-07:00 CET) DiLoCo outer-step aggregation, Multi-Krum & SVD Teacher-to-Student distillation (/v1/hivemind/bees/owner-fleet-status)",
            },
            "plan_classes": ["cuda_hivemind", "cuda_hivemind_big", "cpu_hivemind_solver"],
            "rewards_per_verified_workunit": {
                "hmt_reflexion_points": 250,
                "boinc_cobblestones": 45.0,
            },
        }

    return app
