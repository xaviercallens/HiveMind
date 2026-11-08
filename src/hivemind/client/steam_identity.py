# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Proof of control of a SteamID64, and the facts the Sybil gate needs, taken from Steam instead of from the client.

Why this exists. The coordinator used to take ``steam_id64`` (and an ``is_non_limited`` flag that defaulted to ``True``) from the request body, and the
"Steam auth ticket" was an HMAC over values the client chose, signed with a secret published in this repository. Anyone could claim any identity and any spend.
Here:

* **Who is it?** "Sign in with Steam" (OpenID 2.0). Steam signs an assertion for the user's SteamID64; we hand the assertion back to Steam
  (``openid.mode=check_authentication``) and accept it only if Steam says ``is_valid:true`` and the signed fields name our own return URL, a fresh and
  unused nonce, and the claimed identity. No Steam Web API key and no Steam App ID are needed.
* **Is the account worth a work unit?** ``isLimitedAccount`` and ``vacBanned`` are read from the public community profile XML (checked against the live
  service 2026-10-28: both fields are present even for friends-only profiles). They come from Steam, not from the caller.
* **Afterwards.** A short-lived HMAC token binds the verified SteamID64 so the coordinator does not call Steam on every request.

Everything fails closed: any network error, malformed answer or mismatch raises :class:`IdentityError`. Standard library only; HTTP is injectable for tests.
"""

from __future__ import annotations

import calendar
import hashlib
import hmac
import math
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass

OPENID_ENDPOINT = "https://steamcommunity.com/openid/login"
OPENID_NS = "http://specs.openid.net/auth/2.0"
CLAIMED_ID_PREFIXES = ("https://steamcommunity.com/openid/id/", "http://steamcommunity.com/openid/id/")
PROFILE_XML_URL = "https://steamcommunity.com/profiles/{steam_id64}?xml=1"
STEAMID64_RE = re.compile(r"^7656119\d{10}$")  # individual accounts: 76561197960265728 + account id
REQUIRED_SIGNED = frozenset({"op_endpoint", "claimed_id", "identity", "return_to", "response_nonce", "assoc_handle"})
MAX_BODY_BYTES = 262_144
HTTP_TIMEOUT_S = 10.0

# (method, url, form fields or None) -> (status code, response text). Injectable so tests never touch the network.
Http = Callable[[str, str, Mapping[str, str] | None], tuple[int, str]]


class IdentityError(Exception):
    """The identity could not be proven or the account is not acceptable; ``str(exc)`` is safe to show to the caller."""


def is_valid_steamid64(value: object) -> bool:
    return isinstance(value, str) and STEAMID64_RE.fullmatch(value) is not None


def default_http(method: str, url: str, form: Mapping[str, str] | None = None) -> tuple[int, str]:
    """HTTPS to steamcommunity.com only, bounded time and size, no redirects followed."""
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or parts.hostname != "steamcommunity.com":
        raise IdentityError("refusing to contact anything but https://steamcommunity.com")

    class _NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            return None

    data = urllib.parse.urlencode(form).encode() if form is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"User-Agent": "hivemind-identity/1"})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=HTTP_TIMEOUT_S) as response:
            return response.status, response.read(MAX_BODY_BYTES).decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except (OSError, ValueError) as exc:
        raise IdentityError("Steam could not be reached") from exc


def login_url(return_to: str, realm: str) -> str:
    """Where to send the user's browser to sign in with Steam."""
    if not isinstance(return_to, str) or not return_to.strip() or not isinstance(realm, str) or not realm.strip():
        raise ValueError("return_to and realm must be non-empty strings")
    query = {
        "openid.ns": OPENID_NS, "openid.mode": "checkid_setup", "openid.return_to": return_to, "openid.realm": realm,
        "openid.identity": f"{OPENID_NS}/identifier_select", "openid.claimed_id": f"{OPENID_NS}/identifier_select",
    }
    return OPENID_ENDPOINT + "?" + urllib.parse.urlencode(query)


class SteamOpenIDVerifier:
    """Verifies the assertion Steam appends to ``return_to`` after the user signs in; returns the SteamID64 or raises :class:`IdentityError`."""

    def __init__(self, return_to: str, *, http: Http = default_http, clock: Callable[[], float] = time.time,
                 nonce_ttl_s: float = 300.0, max_nonces: int = 4096, max_attempts_per_ip: int = 64) -> None:
        if not isinstance(return_to, str) or (not return_to.startswith("https://") and not return_to.startswith("http://127.0.0.1") and not return_to.startswith("http://localhost")):
            raise ValueError("return_to must be https (or loopback http for development)")
        if isinstance(nonce_ttl_s, bool) or not isinstance(nonce_ttl_s, (int, float)) or not math.isfinite(float(nonce_ttl_s)) or float(nonce_ttl_s) <= 0.0:
            raise ValueError("nonce_ttl_s must be finite and > 0")
        if isinstance(max_nonces, bool) or not isinstance(max_nonces, int) or int(max_nonces) <= 0:
            raise ValueError("max_nonces must be > 0")
        if isinstance(max_attempts_per_ip, bool) or not isinstance(max_attempts_per_ip, int) or int(max_attempts_per_ip) <= 0:
            raise ValueError("max_attempts_per_ip must be > 0")
        self.return_to, self._http, self._clock, self._ttl, self._max = return_to, http, clock, float(nonce_ttl_s), int(max_nonces)
        self._max_attempts_per_ip = int(max_attempts_per_ip)
        self._seen: dict[str, float] = {}
        self._ip_attempts: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise IdentityError("system clock reading is non-finite")
        return now

    def _check_ip_rate_limit(self, client_ip: str | None) -> None:
        if not client_ip:
            return
        now = self._now()
        cutoff = now - self._ttl
        with self._lock:
            recent = [ts for ts in self._ip_attempts.get(client_ip, []) if ts >= cutoff]
            if len(recent) >= self._max_attempts_per_ip:
                self._ip_attempts[client_ip] = recent
                raise IdentityError("too many sign-in attempts from this IP; try again shortly")
            recent.append(now)
            self._ip_attempts[client_ip] = recent

    def _check_fresh_unused_nonce(self, nonce: str) -> None:
        match = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})Z[\x21-\x7e]{0,64}", nonce)
        if not match:
            raise IdentityError("malformed response nonce")
        stamp = calendar.timegm(tuple(int(g) for g in match.groups()))  # the nonce time is UTC
        now = self._now()
        if stamp < now - self._ttl or stamp > now + 60.0:
            raise IdentityError("stale or future response nonce")
        with self._lock:
            for key in [k for k, until in self._seen.items() if until < now]:
                del self._seen[key]
            if nonce in self._seen:
                raise IdentityError("response nonce was already used")
            if len(self._seen) >= self._max:
                raise IdentityError("too many sign-ins in flight; try again shortly")

    def _commit_nonce(self, nonce: str) -> None:
        now = self._now()
        with self._lock:
            for key in [k for k, until in self._seen.items() if until < now]:
                del self._seen[key]
            if nonce in self._seen:
                raise IdentityError("response nonce was already used")
            if len(self._seen) >= self._max:
                raise IdentityError("too many sign-ins in flight; try again shortly")
            self._seen[nonce] = now + self._ttl + 60.0

    def verify_assertion(self, params: Mapping[str, str], *, client_ip: str | None = None) -> str:
        self._check_ip_rate_limit(client_ip)
        if params.get("openid.ns") != OPENID_NS or params.get("openid.mode") != "id_res":
            raise IdentityError("not an OpenID 2.0 positive assertion")
        if params.get("openid.op_endpoint") != OPENID_ENDPOINT:
            raise IdentityError("assertion was not issued by Steam")
        if params.get("openid.return_to") != self.return_to:
            raise IdentityError("assertion was issued for a different return URL")
        signed = set(params.get("openid.signed", "").split(","))
        if not REQUIRED_SIGNED <= signed:
            raise IdentityError("assertion does not sign the fields that bind it to this login")
        ids = []
        for key in ("openid.claimed_id", "openid.identity"):
            value = params.get(key, "")
            prefix = next((p for p in CLAIMED_ID_PREFIXES if value.startswith(p)), None)
            if prefix is None or not is_valid_steamid64(value[len(prefix):]):
                raise IdentityError("assertion does not name a Steam individual account")
            ids.append(value[len(prefix):])
        if ids[0] != ids[1]:
            raise IdentityError("claimed_id and identity disagree")
        nonce = params.get("openid.response_nonce", "")
        self._check_fresh_unused_nonce(nonce)
        check = {k: v for k, v in params.items() if k.startswith("openid.")}
        check["openid.mode"] = "check_authentication"
        status, body = self._http("POST", OPENID_ENDPOINT, check)
        if status != 200 or not re.search(r"(?m)^is_valid:true\s*$", body):
            raise IdentityError("Steam did not confirm the assertion")
        self._commit_nonce(nonce)
        return ids[0]


@dataclass(frozen=True)
class AccountFacts:
    steam_id64: str
    limited: bool  # Steam's own flag: the account has spent less than the minimum (USD 5)
    vac_banned: bool


def fetch_account_facts(steam_id64: str, http: Http = default_http) -> AccountFacts:
    """Limited/VAC-ban flags straight from Steam's community profile XML. Regex, not an XML parser: this is a network answer and entity expansion is not worth risking."""
    if not is_valid_steamid64(steam_id64):
        raise IdentityError("not a SteamID64")
    status, body = http("GET", PROFILE_XML_URL.format(steam_id64=steam_id64), None)
    if status != 200:
        raise IdentityError("Steam profile could not be read")
    fields = {name: re.search(rf"<{name}>\s*([^<\s]*)\s*</{name}>", body) for name in ("steamID64", "isLimitedAccount", "vacBanned")}
    if not all(fields.values()) or fields["steamID64"].group(1) != steam_id64:  # type: ignore[union-attr]
        raise IdentityError("Steam profile answer is incomplete or for another account")
    flags = {k: m.group(1) for k, m in fields.items() if k != "steamID64"}  # type: ignore[union-attr]
    if any(v not in ("0", "1") for v in flags.values()):
        raise IdentityError("Steam profile flags are malformed")
    return AccountFacts(steam_id64, flags["isLimitedAccount"] == "1", flags["vacBanned"] == "1")


class IdentityTokens:
    """``v1.<steam_id64>.<expiry>.<hmac>``: proof that this process verified the identity recently. The secret never leaves the process unless configured."""

    def __init__(self, secret: bytes | None = None, ttl_s: float = 3600.0, clock: Callable[[], float] = time.time) -> None:
        if secret is not None and not isinstance(secret, (bytes, bytearray)):
            raise ValueError("identity secret must be bytes")
        env = os.environ.get("HIVEMIND_IDENTITY_SECRET", "")
        self._secret = bytes(secret) if secret else (bytes.fromhex(env) if re.fullmatch(r"[0-9a-fA-F]{64,}", env or "") else secrets.token_bytes(32))
        if len(self._secret) < 32:
            raise ValueError("identity secret must be at least 32 bytes")
        if isinstance(ttl_s, bool) or not isinstance(ttl_s, (int, float)) or not math.isfinite(float(ttl_s)) or float(ttl_s) <= 0.0:
            raise ValueError("ttl_s must be finite and > 0")
        self.ttl_s, self._clock = float(ttl_s), clock

    def _now(self) -> float:
        now = float(self._clock())
        if not math.isfinite(now):
            raise IdentityError("system clock reading is non-finite")
        return now

    def _mac(self, body: str) -> str:
        return hmac.new(self._secret, body.encode(), hashlib.sha256).hexdigest()

    def issue(self, steam_id64: str) -> str:
        if not is_valid_steamid64(steam_id64):
            raise IdentityError("not a SteamID64")
        body = f"v1.{steam_id64}.{int(self._now() + self.ttl_s)}"
        return f"{body}.{self._mac(body)}"

    def verify(self, token: str) -> str:
        parts = token.split(".") if isinstance(token, str) else []
        if len(parts) != 4 or parts[0] != "v1" or not is_valid_steamid64(parts[1]) or not parts[2].isdigit():
            raise IdentityError("malformed identity token")
        if not hmac.compare_digest(parts[3], self._mac(".".join(parts[:3]))):
            raise IdentityError("identity token signature is invalid")
        if int(parts[2]) < self._now():
            raise IdentityError("identity token expired; sign in with Steam again")
        return parts[1]


class SteamIdentityGate:
    """What the coordinator needs: complete a Steam sign-in into a token, and turn a token plus a claimed id into a trusted id."""

    def __init__(self, return_to: str, *, http: Http = default_http, tokens: IdentityTokens | None = None,
                 verifier: SteamOpenIDVerifier | None = None, allow_limited: bool = False) -> None:
        if not isinstance(allow_limited, bool):
            raise ValueError("allow_limited must be a boolean")
        self.return_to, self._http, self.allow_limited = return_to, http, allow_limited
        self.verifier = verifier or SteamOpenIDVerifier(return_to, http=http)
        self.tokens = tokens or IdentityTokens()

    def login_redirect(self) -> str:
        parts = urllib.parse.urlsplit(self.return_to)
        return login_url(self.return_to, f"{parts.scheme}://{parts.netloc}")

    def complete_login(self, params: Mapping[str, str], *, client_ip: str | None = None) -> tuple[str, str]:
        """Returns ``(steam_id64, token)`` for an account Steam vouches for and the policy accepts; raises :class:`IdentityError` otherwise."""
        steam_id64 = (
            self.verifier.verify_assertion(params, client_ip=client_ip)
            if client_ip is not None
            else self.verifier.verify_assertion(params)
        )
        facts = fetch_account_facts(steam_id64, self._http)
        if facts.vac_banned:
            raise IdentityError("this Steam account has an active VAC/game ban")
        if facts.limited and not self.allow_limited:
            raise IdentityError("this Steam account is Limited (it has spent less than the USD 5 minimum)")
        return steam_id64, self.tokens.issue(steam_id64)

    def identify(self, token: str | None) -> str:
        """The SteamID64 a valid token belongs to (for endpoints that need a signed-in node but do not name one)."""
        if not token:
            raise IdentityError("an identity token is required: sign in with Steam first")
        return self.tokens.verify(token)

    def authorize(self, token: str | None, claimed_steam_id64: str) -> str:
        if not token:
            raise IdentityError("an identity token is required: sign in with Steam first")
        steam_id64 = self.tokens.verify(token)
        if steam_id64 != claimed_steam_id64:
            raise IdentityError("the token belongs to a different Steam account")
        return steam_id64
