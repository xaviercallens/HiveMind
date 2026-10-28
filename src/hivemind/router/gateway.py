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
    test: bool = False  # test account: needs the admin key, never bids in the market, counted apart from real users
    kind: Literal["user", "test", "probe"] | None = None  # 'test' or 'probe' need the admin key too (HM-S16-07)


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

    def equilibrium(model_id: str, records: Sequence[ProducerRecord], extra: Account | None) -> MarketEquilibrium:
        return clear_market(model_id, bids_for(extra), offers_for(records), fee)

    def resolve_model(
        requested: str,
        *,
        preferred_tier: str | None = None,
    ) -> tuple[str, list[ProducerRecord], str]:
        now = clock()
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
    @app.api_route("/health", methods=["GET", "HEAD"])  # Cloud Run's front end swallows /healthz, so the same probe is also served here
    @app.api_route("/healthz", methods=["GET", "HEAD"])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/life-as-a-bee", methods=["GET", "HEAD"], response_class=HTMLResponse)
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

    @app.get("/v1/hivemind/life-as-a-bee")
    def life_as_a_bee() -> dict[str, Any]:
        """Public 'Life as a Bee' dashboard payload: 24h bee lifecycle, 15 real end-user benchmark scenarios, and per-bee usage guide."""
        from hivemind.router.life_as_a_bee import life_as_a_bee_payload

        return life_as_a_bee_payload()

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

        return evaluate_international_launch_readiness()

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
    def list_models() -> dict[str, Any]:
        created = int(clock())
        ids = [_public_model(AUTO_MODEL), *(_public_model(m) for m in store.live_models(clock(), PRODUCER_TTL_S))]
        loaded = store.residency_summary(clock(), PRODUCER_TTL_S)
        data: list[dict[str, Any]] = []
        for i in ids:
            item: dict[str, Any] = {"id": i, "object": "model", "created": created, "owned_by": "hivemind"}
            if _strip_prefix(i) in loaded:  # only where some live producer reports residency; aggregate only, no producer identity
                item["loaded"] = loaded[_strip_prefix(i)]
            data.append(item)
        return {"object": "list", "data": data}

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
    async def chat_completions(request: Request) -> JSONResponse | StreamingResponse:
        # authenticate first, then read the body (already bounded by BodyLimitMiddleware): anonymous callers cost nothing
        account = await run_in_threadpool(authenticate, request)
        raw = await request.body()
        return await run_in_threadpool(handle_chat, account, raw, request.headers)

    google = google_verifier or (GoogleIdTokenVerifier(google_client_id, clock=clock) if google_client_id else None)

    @app.api_route("/signup", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def signup() -> HTMLResponse:
        nonce = secrets.token_urlsafe(18)
        page = render_signup_page(google.client_id if google else None, nonce)
        return HTMLResponse(page, headers={"Content-Security-Policy": content_security_policy(nonce), "Cache-Control": "no-store",
                                           "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    @app.api_route("/join", methods=["GET", "HEAD"], response_class=HTMLResponse)
    @app.api_route("/rejoindre", methods=["GET", "HEAD"], response_class=HTMLResponse)
    def join_page(request: Request) -> HTMLResponse:
        """Public, bilingual (FR/EN) 'how to join' page: no account, no script, no personal data."""
        french_path = request.url.path == "/rejoindre"
        lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
        nonce = secrets.token_urlsafe(18)
        page = render_join_page(lang, nonce, tiers=catalog_payload()["tiers"], path="/rejoindre" if french_path else "/join",
                                google_enabled=google is not None, handle_registration=open_registration)  # describe only the sign-up routes this gateway really has
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
        models = store.live_models(clock(), PRODUCER_TTL_S)
        producers = len(store.active_producers("coder", clock(), PRODUCER_TTL_S))
        return build_t4_visibility_payload(now_utc=clock(), live_models=models, active_producers_count=producers)

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
        residency, capacity = residency_report([(m["id"], e) for m, e in zip(models, payload.models, strict=True)], payload.gpu_vram_mib, payload.capacity_1k)
        try:
            endpoints = store.upsert_producer(
                account.account_id, payload.endpoint_url, models, payload.cost_per_1k, capacity, payload.execution_class,
                now_utc=clock(), endpoint_token=payload.endpoint_token, residency=residency,
            )
        except TooManyEndpoints as exc:
            raise RouterError(400, str(exc), "invalid_request_error", "too_many_endpoints") from None
        ack: dict[str, Any] = {"account_id": account.account_id, "status": "live", "models": [m["id"] for m in models], "ttl_s": PRODUCER_TTL_S, "endpoints": endpoints}
        control = store.producer_control(account.account_id)
        if control is not None:  # absent when never set: old agents and untouched accounts see exactly the old ack
            ack["control"] = {**control, "acknowledged": bool(payload.paused) == control["paused"] and payload.vram_cap_mib == control["vram_cap_mib"]}
        if residency is not None:
            ack["residency"] = {"loaded": sorted(m for m, v in residency["models"].items() if v["loaded"] is True), "effective_capacity_1k": capacity, "capacity_note": residency["capacity_note"]}
        return ack

    @app.post("/v1/hivemind/producer/control")
    def set_control(request: Request, payload: ControlRequest) -> dict[str, Any]:
        """Pause/resume a producer and set or clear its VRAM cap (HM-S16-05). Own API key (own account only) or the admin key (any account)."""
        if request.headers.get("x-admin-key") is not None:
            require_admin(request)
            target = account_target(payload.account_id, payload.handle)
        else:
            account = authenticate(request)
            if account.role not in ("producer", "prosumer"):
                raise RouterError(403, "Only producer or prosumer accounts have remote control.", "permission_error", "role_forbidden")
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

    return app
