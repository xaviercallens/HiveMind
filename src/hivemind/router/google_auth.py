"""Verification of Google ID tokens ("Sign in with Google") for registration. Torch-free; needs only ``cryptography`` and ``httpx``.

A Google ID token is a JWT signed with RS256 by a key Google publishes at ``GOOGLE_JWKS_URL``. We accept a token only
when ALL of these hold, and each has a negative test:

* the header says ``RS256`` (never read the algorithm from the token: no ``none``, no HS256-with-public-key confusion),
* the signature verifies with the RSA key (at least 2048 bits) whose ``kid`` Google's key set lists,
* ``iss`` is Google, ``aud`` is exactly our OAuth client id, ``exp`` is in the future,
* ``iat`` is recent (a sign-in token is a bearer credential, so a stale one is refused), and not from the future,
* ``sub`` is present and ``email_verified`` is true (a real, verified Google account),
* the same token has not been presented before (single use, remembered until it expires).

Nothing about the user except the opaque ``sub`` is kept; the e-mail address is never stored.
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

GOOGLE_JWKS_URL: str = "https://www.googleapis.com/oauth2/v3/certs"
GOOGLE_ISSUERS: frozenset[str] = frozenset({"https://accounts.google.com", "accounts.google.com"})
CLIENT_ID_RE = re.compile(r"^[0-9]+-[a-z0-9]+\.apps\.googleusercontent\.com$")
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


class GoogleAuthError(Exception):
    """The token was refused. ``code`` is a stable machine-readable reason; the message never echoes the token."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class GoogleIdentity:
    subject: str
    issued_at: float
    expires_at: float


def fetch_google_jwks(client: httpx.Client | None = None) -> tuple[Mapping[str, Any], float]:
    """Download Google's current signing keys; the cache lifetime follows ``Cache-Control: max-age``, clamped."""
    owns = client is None
    http = client or httpx.Client(timeout=JWKS_TIMEOUT_S)
    try:
        response = http.get(GOOGLE_JWKS_URL, timeout=JWKS_TIMEOUT_S)
        response.raise_for_status()
        body = response.json()
    finally:
        if owns:
            http.close()
    match = re.search(r"max-age=(\d+)", response.headers.get("cache-control", ""))
    ttl = float(match.group(1)) if match else MIN_JWKS_TTL_S
    return body, min(max(ttl, MIN_JWKS_TTL_S), MAX_JWKS_TTL_S)


def _b64url(part: str) -> bytes:
    try:
        return base64.urlsafe_b64decode(part + "=" * (-len(part) % 4))
    except (binascii.Error, ValueError):
        raise GoogleAuthError("malformed_token") from None


def _json_object(part: str) -> dict[str, Any]:
    try:
        value = json.loads(_b64url(part))
    except ValueError:
        raise GoogleAuthError("malformed_token") from None
    if not isinstance(value, dict):
        raise GoogleAuthError("malformed_token")
    return value


def _rsa_key(jwk: Mapping[str, Any]) -> rsa.RSAPublicKey:
    if jwk.get("kty") != "RSA":
        raise GoogleAuthError("unusable_key")
    try:
        numbers = rsa.RSAPublicNumbers(int.from_bytes(_b64url(str(jwk["e"])), "big"), int.from_bytes(_b64url(str(jwk["n"])), "big"))
        key = numbers.public_key()
    except (KeyError, ValueError):
        raise GoogleAuthError("unusable_key") from None
    if key.key_size < MIN_RSA_BITS:
        raise GoogleAuthError("unusable_key")
    return key


class GoogleIdTokenVerifier:
    """Verify Google ID tokens for one OAuth client id. Thread-safe; ``clock`` and ``jwks_fetcher`` are injectable."""

    def __init__(
        self,
        client_id: str,
        *,
        jwks_fetcher: JwksFetcher = fetch_google_jwks,
        clock: Callable[[], float] = time.time,
        max_age_s: float = DEFAULT_MAX_AGE_S,
        leeway_s: float = DEFAULT_LEEWAY_S,
    ) -> None:
        if not CLIENT_ID_RE.match(client_id):
            raise ValueError("client_id must look like 123456789-abc.apps.googleusercontent.com")
        if max_age_s <= 0 or leeway_s < 0:
            raise ValueError("max_age_s must be > 0 and leeway_s >= 0")
        self.client_id = client_id
        self._fetch = jwks_fetcher
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
            raise GoogleAuthError("keys_unavailable") from None
        self._keys, self._keys_valid_until, self._last_fetch = keys, now + ttl, now

    def _key_for(self, kid: str, now: float) -> rsa.RSAPublicKey:
        with self._lock:
            if now >= self._keys_valid_until:
                self._refresh(now)
            if kid not in self._keys and now - self._last_fetch >= UNKNOWN_KID_REFETCH_S:
                self._refresh(now)  # Google rotated keys: look again, but never more than once a minute
            jwk = self._keys.get(kid)
        if jwk is None:
            raise GoogleAuthError("unknown_key")
        return _rsa_key(jwk)

    # ------------------------------------------------------------------ verify
    def verify(self, token: str) -> GoogleIdentity:
        if not isinstance(token, str) or not token or len(token) > MAX_TOKEN_CHARS:
            raise GoogleAuthError("malformed_token")
        parts = token.split(".")
        if len(parts) != 3 or not all(parts):
            raise GoogleAuthError("malformed_token")
        header, claims = _json_object(parts[0]), _json_object(parts[1])
        if header.get("alg") != "RS256":
            raise GoogleAuthError("bad_algorithm")
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            raise GoogleAuthError("malformed_token")
        now = self._clock()
        key = self._key_for(kid, now)
        try:
            key.verify(_b64url(parts[2]), f"{parts[0]}.{parts[1]}".encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
        except (InvalidSignature, UnicodeEncodeError):
            raise GoogleAuthError("bad_signature") from None

        if claims.get("iss") not in GOOGLE_ISSUERS:
            raise GoogleAuthError("bad_issuer")
        if claims.get("aud") != self.client_id:
            raise GoogleAuthError("bad_audience")
        exp, iat = claims.get("exp"), claims.get("iat")
        if isinstance(exp, bool) or isinstance(iat, bool) or not isinstance(exp, (int, float)) or not isinstance(iat, (int, float)):
            raise GoogleAuthError("malformed_token")
        if now > exp + self._leeway:
            raise GoogleAuthError("expired")
        if iat > now + self._leeway or now - iat > self._max_age + self._leeway:
            raise GoogleAuthError("not_fresh")
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > 255:
            raise GoogleAuthError("malformed_token")
        if claims.get("email_verified") is not True:
            raise GoogleAuthError("email_not_verified")

        digest = hashlib.sha256(token.encode("ascii")).hexdigest()
        with self._lock:
            for old in [d for d, until in self._seen.items() if until < now]:
                del self._seen[old]
            if digest in self._seen:
                raise GoogleAuthError("replayed")
            if len(self._seen) >= MAX_REMEMBERED_TOKENS:
                raise GoogleAuthError("busy")
            self._seen[digest] = float(exp) + self._leeway
        return GoogleIdentity(subject=subject, issued_at=float(iat), expires_at=float(exp))
