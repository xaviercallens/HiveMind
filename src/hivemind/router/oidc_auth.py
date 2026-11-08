# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Verification of OpenID Connect ID tokens from ANY provider (Google, Microsoft, Auth0, Keycloak, GitLab, ...) for registration.
Torch-free; needs only ``cryptography`` and ``httpx``. ``google_auth`` is this engine with Google's settings.

An ID token is a JWT signed with RS256 by a key the provider publishes at its JWKS URL. We accept a token only
when ALL of these hold, and each has a negative test:

* the header says ``RS256`` (never read the algorithm from the token: no ``none``, no HS256-with-public-key confusion),
* the signature verifies with the RSA key (at least 2048 bits) whose ``kid`` the provider's key set lists,
* ``iss`` is one of the provider's issuers, ``aud`` is exactly our client id, ``exp`` is in the future,
* ``iat`` is recent (a sign-in token is a bearer credential, so a stale one is refused), and not from the future,
* ``sub`` is present and, unless the provider is configured otherwise, ``email_verified`` is true (a real, verified account),
* the same token has not been presented before (single use, remembered until it expires).

Nothing about the user except the opaque ``sub`` is kept (hashed together with the provider name); the e-mail address is never stored.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding, rsa

PROVIDER_NAME_RE = re.compile(r"^[a-z][a-z0-9]{1,15}$")
MAX_TOKEN_CHARS: int = 4096
MIN_RSA_BITS: int = 2048
DEFAULT_MAX_AGE_S: float = 600.0
DEFAULT_LEEWAY_S: float = 30.0
MIN_JWKS_TTL_S: float = 300.0
MAX_JWKS_TTL_S: float = 86_400.0
UNKNOWN_KID_REFETCH_S: float = 60.0
MAX_REMEMBERED_TOKENS: int = 50_000
JWKS_TIMEOUT_S: float = 5.0

# (key set, seconds it may be cached)
JwksFetcher = Callable[[], tuple[Mapping[str, Any], float]]


class OidcAuthError(Exception):
    """The token was refused. ``code`` is a stable machine-readable reason; the message never echoes the token."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OidcIdentity:
    subject: str
    issued_at: float
    expires_at: float


@dataclass(frozen=True)
class OidcProvider:
    """One identity provider. ``name`` becomes the account-handle prefix (``<name>.<keyed hash>``)."""

    name: str
    issuers: frozenset[str]
    jwks_url: str
    client_id: str
    require_verified_email: bool = True

    def __post_init__(self) -> None:
        if not PROVIDER_NAME_RE.match(self.name):
            raise ValueError("provider name must be 2-16 lowercase letters or digits, starting with a letter")
        if not self.issuers or not all(isinstance(i, str) and i for i in self.issuers):
            raise ValueError("at least one non-empty issuer is required")
        if not self.jwks_url.startswith("https://"):
            raise ValueError("jwks_url must be an https URL")
        if not self.client_id or len(self.client_id) > 256:
            raise ValueError("client_id must be 1-256 characters")


def fetch_jwks(url: str, client: httpx.Client | None = None) -> tuple[Mapping[str, Any], float]:
    """Download a provider's current signing keys; the cache lifetime follows ``Cache-Control: max-age``, clamped."""
    owns = client is None
    http = client or httpx.Client(timeout=JWKS_TIMEOUT_S)
    try:
        response = http.get(url, timeout=JWKS_TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
    finally:
        if owns:
            http.close()
    match = re.search(r"max-age=(\d+)", response.headers.get("cache-control", ""))
    ttl = float(match.group(1)) if match else MIN_JWKS_TTL_S
    return body, min(max(ttl, MIN_JWKS_TTL_S), MAX_JWKS_TTL_S)


_B64URL_PART = re.compile(r"[A-Za-z0-9_-]+")


def _b64url(part: str) -> bytes:
    """Strict, canonical unpadded base64url: padding, other characters or unused trailing bits are refused, so one signed
    token has exactly one accepted spelling (otherwise ``tok + "=="`` or a flipped last bit replays a used token)."""
    if not _B64URL_PART.fullmatch(part):
        raise OidcAuthError("malformed_token")
    try:
        raw = base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
    except (binascii.Error, ValueError):
        raise OidcAuthError("malformed_token") from None
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii") != part:
        raise OidcAuthError("malformed_token")
    return raw


def _json_object(part: str) -> dict[str, Any]:
    try:
        value = json.loads(_b64url(part))
    except ValueError:
        raise OidcAuthError("malformed_token") from None
    if not isinstance(value, dict):
        raise OidcAuthError("malformed_token")
    return value


def _rsa_key(jwk: Mapping[str, Any]) -> rsa.RSAPublicKey:
    if jwk.get("kty") != "RSA":
        raise OidcAuthError("unusable_key")
    try:
        numbers = rsa.RSAPublicNumbers(int.from_bytes(_b64url(str(jwk["e"])), "big"), int.from_bytes(_b64url(str(jwk["n"])), "big"))
        key = numbers.public_key()
    except (KeyError, ValueError):
        raise OidcAuthError("unusable_key") from None
    if key.key_size < MIN_RSA_BITS:
        raise OidcAuthError("unusable_key")
    return key


class OidcIdTokenVerifier:
    """Verify ID tokens for one provider and client id. Thread-safe; ``clock`` and ``jwks_fetcher`` are injectable."""

    def __init__(
        self,
        provider: OidcProvider,
        *,
        jwks_fetcher: JwksFetcher | None = None,
        clock: Callable[[], float] = time.time,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        leeway_s: float = DEFAULT_LEEWAY_S,
    ) -> None:
        if max_age_s <= 0 or leeway_s < 0:
            raise ValueError("max_age_s must be > 0 and leeway_s >= 0")
        self.provider = provider
        self.client_id = provider.client_id
        self._fetch = jwks_fetcher or (lambda: fetch_jwks(provider.jwks_url))
        self._clock = clock
        self._max_age = max_age_s
        self._leeway = leeway_s
        self._lock = threading.Lock()
        self._keys: dict[str, Mapping[str, Any]] = {}
        self._keys_valid_until = 0.0
        self._last_fetch = float("-inf")
        self._seen: dict[str, float] = {}  # sha256(token) -> expiry, so a token works once

    # ------------------------------------------------------------------ keys
    def _refresh(self, now: float) -> None:
        try:
            body, ttl = self._fetch()
            keys = {str(k["kid"]): k for k in body.get("keys", []) if isinstance(k, Mapping) and "kid" in k}
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError):
            raise OidcAuthError("keys_unavailable") from None
        self._keys, self._keys_valid_until, self._last_fetch = keys, now + ttl, now

    def _key_for(self, kid: str, now: float) -> rsa.RSAPublicKey:
        with self._lock:
            if now >= self._keys_valid_until:
                self._refresh(now)
            if kid not in self._keys and now - self._last_fetch >= UNKNOWN_KID_REFETCH_S:
                self._refresh(now)  # the provider rotated keys: look again, but never more than once a minute
            jwk = self._keys.get(kid)
        if jwk is None:
            raise OidcAuthError("unknown_key")
        return _rsa_key(jwk)

    # ------------------------------------------------------------------ verify
    def verify(self, token: str) -> OidcIdentity:
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_CHARS:
            raise OidcAuthError("malformed_token")
        parts = token.split(".")
        if len(parts) != 3 or not all(parts):
            raise OidcAuthError("malformed_token")
        header, claims = _json_object(parts[0]), _json_object(parts[1])
        if header.get("alg") != "RS256":
            raise OidcAuthError("bad_algorithm")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise OidcAuthError("malformed_token")
        now = self._clock()
        key = self._key_for(kid, now)
        try:
            key.verify(_b64url(parts[2]), f"{parts[0]}.{parts[1]}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
        except (InvalidSignature, UnicodeEncodeError):
            raise OidcAuthError("bad_signature") from None

        if claims.get("iss") not in self.provider.issuers:
            raise OidcAuthError("bad_issuer")
        if claims.get("aud") != self.client_id:
            raise OidcAuthError("bad_audience")
        exp, iat = claims.get("exp"), claims.get("iat")
        if isinstance(exp, bool) or isinstance(iat, bool) or not isinstance(exp, (int, float)) or not isinstance(iat, (int, float)):
            raise OidcAuthError("malformed_token")
        if now > exp + self._leeway:
            raise OidcAuthError("expired")
        if iat > now + self._leeway or now - iat > self._max_age + self._leeway:
            raise OidcAuthError("not_fresh")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > 255:
            raise OidcAuthError("malformed_token")
        if self.provider.require_verified_email and claims.get("email_verified") is not True:
            raise OidcAuthError("email_not_verified")

        # keyed on the signed bytes (header.claims), not the raw string: no re-encoding of a used token gets a second use
        digest = hashlib.sha256(f"{parts[0]}.{parts[1]}".encode("ascii")).hexdigest()
        with self._lock:
            for old in [d for d, until in self._seen.items() if until < now]:
                del self._seen[old]
            if digest in self._seen:
                raise OidcAuthError("replayed")
            if len(self._seen) >= MAX_REMEMBERED_TOKENS:
                raise OidcAuthError("busy")
            self._seen[digest] = float(exp) + self._leeway
        return OidcIdentity(subject=subject, issued_at=float(iat), expires_at=float(exp))


def providers_from_json(text: str, *, clock: Callable[[], float] = time.time) -> dict[str, OidcIdTokenVerifier]:
    """Build verifiers from the operator's ``HIVEMIND_OIDC_PROVIDERS`` JSON: a list of objects

    ``{"name": "acme", "issuer": "https://login.example.com/" | "issuers": [...], "jwks_url": "https://...", "client_id": "...",
    "require_verified_email": true}``. Raises ``ValueError`` on anything malformed, a duplicate name or the reserved name
    ``google`` (Google has its own setting), so a typo stops the deploy instead of silently disabling sign-in.
    """
    try:
        entries = json.loads(text)
    except ValueError as exc:
        raise ValueError("HIVEMIND_OIDC_PROVIDERS is not valid JSON") from exc
    if not isinstance(entries, list):
        raise ValueError("HIVEMIND_OIDC_PROVIDERS must be a JSON list")
    out: dict[str, OidcIdTokenVerifier] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each OIDC provider must be a JSON object")
        unknown = set(entry) - {"name", "issuer", "issuers", "jwks_url", "client_id", "require_verified_email",
                                 "authorization_endpoint", "token_endpoint", "label"}  # the last three are read by the web portal
        if unknown:
            raise ValueError(f"unknown OIDC provider field(s): {sorted(unknown)}")
        name = entry.get("name")
        if not isinstance(name, str) or name == "google" or name in out:
            raise ValueError("OIDC provider names must be unique strings and not 'google'")
        issuers = entry.get("issuers", [entry["issuer"]] if "issuer" in entry else [])
        if not isinstance(issuers, list):
            raise ValueError("issuers must be a list")
        for iss in issuers:
            # Microsoft's multi-tenant endpoints put each user's tenant in "iss": a fixed-issuer check would reject everyone,
            # and matching any tenant would accept every Microsoft account. Only tenant-specific issuers are accepted.
            if not isinstance(iss, str) or "{" in iss or "/common/" in iss or "/organizations/" in iss or "/consumers/" in iss:
                raise ValueError(f"issuer must be a fixed, tenant-specific URL: {iss!r}")
        verify_email = entry.get("require_verified_email", True)
        if not isinstance(verify_email, bool):
            raise ValueError("require_verified_email must be true or false")
        provider = OidcProvider(name, frozenset(issuers), str(entry.get("jwks_url", "")), str(entry.get("client_id", "")), verify_email)
        out[name] = OidcIdTokenVerifier(provider, clock=clock)
    return out
