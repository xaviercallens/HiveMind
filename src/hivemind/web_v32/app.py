# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind web portal v3.2.

Design rules (see docs/WEB_V3_2.md):

* **One account store.** Every account operation goes to the gateway (``HIVEMIND_OPENROUTER_URL``), the Litestream-backed
  store where previous users' passwords already are. This service keeps no user file and seeds no user.
* **No auth state in process memory.** The service runs up to two instances, so the session and the OAuth state live in
  encrypted, authenticated cookies (Fernet, key derived from ``HIVEMIND_WEB_SESSION_SECRET``). Without that secret every
  sign-in route answers 503: it fails closed instead of issuing sessions one instance cannot read.
* **The browser never holds the API key for a session.** The session cookie is HttpOnly, Secure, SameSite=Lax and carries
  the key encrypted; a new key is shown once, on registration or rotation, for the user to copy.
* **CSRF.** Every state-changing request needs the ``X-CSRF-Token`` header equal to the ``hm_csrf`` cookie and an
  ``Origin`` (or ``Referer``) equal to this site's origin.
* **SSO is real or not shown.** "Sign in with Google" is an authorization-code flow with PKCE (S256), ``state`` and
  ``nonce`` kept server-side in the encrypted cookie; the code is exchanged with the client secret on the server, the ID
  token is verified here (signature, issuer, audience, expiry, ``email_verified``, nonce) and again by the gateway. The
  button appears only when ``HIVEMIND_GOOGLE_CLIENT_ID``, ``HIVEMIND_GOOGLE_CLIENT_SECRET`` and ``HIVEMIND_PUBLIC_WEB_URL``
  are set. No other provider is offered until it is verified the same way.
* **Only sourced facts.** ``/v32/api/facts`` returns each number with the URL it came from and the time it was read; a
  source that fails is shown as unavailable, never replaced by a stored number.
* **Strict CSP.** Pages carry ``script-src 'self' 'nonce-…'`` with no ``unsafe-inline``; scripts and styles are files.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

import httpx
from cryptography.fernet import Fernet, InvalidToken
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response

STATIC_DIR = Path(__file__).resolve().parent / "static"
SESSION_COOKIE = "hm_s"
CSRF_COOKIE = "hm_csrf"
OIDC_COOKIE = "hm_oidc"
SESSION_TTL_S = 12 * 3600
OIDC_TTL_S = 600
GOOGLE_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
PAPER_DOI = "10.5281/zenodo.24043016"
UI_HEADER = "X-HiveMind-UI"
UI_VERSION = "v3.2"
GENERIC_LOGIN_ERROR = "Wrong e-mail, username or password."
# Every error has a stable code; the page shows it in the visitor's language (static/i18n.json, keys "err_<code>"),
# API clients get the English text. tests/test_web_v33_i18n.py checks both languages cover every code.
ERRORS: dict[str, str] = {
    "not_configured": "Sign-in is not configured on this deployment (missing session secret).",
    "csrf": "Request refused (CSRF check). Reload the page and try again.",
    "wrong_host": "Use the official address of this site.",
    "bad_login": GENERIC_LOGIN_ERROR,
    "rate_limited": "Too many attempts; wait a minute.",
    "unavailable": "Sign-in is temporarily unavailable. Try again in a minute.",
    "register_input": "Enter a valid e-mail and a password of at least 12 characters.",
    "register_refused": "This e-mail or username cannot be registered. If it is yours, sign in or reset the password.",
    "sign_in_first": "Sign in first.",
    "weak_password": "The new password must be 12 to 256 characters.",
    "session_expired": "Your session has expired. Sign in again.",
    "reset_unavailable": "Password reset by e-mail is not available yet on this deployment.",
    "reset_invalid": "This reset link is invalid or has expired. Ask for a new one.",
}
LANGUAGES = ("en", "fr")


class GatewayClient:
    """Thin server-to-server client of the gateway. Injectable in tests."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 10.0,
        client: httpx.Client | None = None,
        forward_secret: bytes | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s, headers={"User-Agent": "HiveMind-Web/3.2"})
        self._forward_secret = forward_secret
        self._clock = clock

    def post(
        self, path: str, payload: Mapping[str, Any], bearer: str | None = None, client_ip: str | None = None
    ) -> tuple[int, dict[str, Any]]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        if client_ip and self._forward_secret:
            from hivemind.router import forwarded_client

            headers[forwarded_client.HEADER] = forwarded_client.sign(self._forward_secret, client_ip, self._clock())
        try:
            r = self._client.post(f"{self.base_url}{path}", json=dict(payload), headers=headers)
        except httpx.HTTPError:
            return 502, {}
        try:
            body = r.json()
        except ValueError:
            body = {}
        return r.status_code, body if isinstance(body, dict) else {}

    def get(self, path: str, bearer: str | None = None) -> tuple[int, dict[str, Any]]:
        headers = {"Accept": "application/json"}
        if bearer:
            headers["Authorization"] = f"Bearer {bearer}"
        try:
            r = self._client.get(f"{self.base_url}{path}", headers=headers)
        except httpx.HTTPError:
            return 502, {}
        try:
            body = r.json()
        except ValueError:
            body = {}
        return r.status_code, body if isinstance(body, dict) else {}


def _fernet(secret: str) -> Fernet:
    key = base64.urlsafe_b64encode(hashlib.sha256(b"hivemind-web-v32-session:" + secret.encode("utf-8")).digest())
    return Fernet(key)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _jwt_claims_unverified(token: str) -> dict[str, Any]:
    """Payload of a JWT whose signature has ALREADY been verified by the caller (used only to read ``nonce``)."""
    try:
        part = token.split(".")[1]
        return json.loads(base64.urlsafe_b64decode(part + "=" * (-len(part) % 4)))
    except Exception:
        return {}


def _eq(a: str, b: str) -> bool:
    """Constant-time equality for strings that may contain any character (``hmac.compare_digest`` raises on non-ASCII
    ``str``, which turned a crafted ``state`` or header into a server error)."""
    return hmac.compare_digest(str(a).encode("utf-8", "surrogatepass"), str(b).encode("utf-8", "surrogatepass"))


def page_csp(nonce: str) -> str:
    return (
        "default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}'; "
        "style-src 'self'; "
        "img-src 'self' data:; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'; "
        "object-src 'none'; "
        "form-action 'self' https://accounts.google.com"
    )


def mount_web_v32(
    app: FastAPI,
    *,
    env: Mapping[str, str] | None = None,
    gateway: GatewayClient | None = None,
    google_verifier: Any | None = None,
    token_exchanger: Callable[[Mapping[str, str]], dict[str, Any]] | None = None,
    clock: Callable[[], float] = time.time,
    oidc_verifiers: Mapping[str, Any] | None = None,
) -> None:
    """Register the v3.2 routes on ``app``. Call it before the legacy routes so that, with ``HIVEMIND_WEB_V32=1``, the
    v3.2 page answers ``/``, ``/auth``, ``/login`` and ``/signup``. ``/v32`` always serves it (for a canary)."""
    e = os.environ if env is None else env
    session_secret = e.get("HIVEMIND_WEB_SESSION_SECRET", "")
    fernet = _fernet(session_secret) if len(session_secret) >= 32 else None
    public_url = e.get("HIVEMIND_PUBLIC_WEB_URL", "").strip().rstrip("/")
    router_secret = e.get("HIVEMIND_ROUTER_SECRET", "").encode("utf-8") or None
    gw = gateway or GatewayClient(
        e.get("HIVEMIND_OPENROUTER_URL", "").strip() or "https://hivemind-openrouter-1003063861791.us-central1.run.app",
        forward_secret=router_secret,
        clock=clock,
    )
    public_host = urlsplit(public_url).netloc.lower() if public_url else ""
    take_over_root = e.get("HIVEMIND_WEB_V32", "").strip() == "1"
    log = logging.getLogger("hivemind.web_v32")
    https_site = public_url.startswith("https://")

    # ---- sign-in providers: each appears only when fully configured -------------------------------------------------
    # kind "oidc": an ID token is verified here (signature, issuer, audience, expiry, nonce, e-mail rule) and again by the
    # gateway. kind "github": GitHub has no ID token; the gateway exchanges the code itself with its own client secret.
    providers: dict[str, dict[str, Any]] = {}
    google_id = e.get("HIVEMIND_GOOGLE_CLIENT_ID", "").strip()
    google_secret = e.get("HIVEMIND_GOOGLE_CLIENT_SECRET", "").strip()
    if google_id and google_secret and https_site:
        if google_verifier is None:
            from hivemind.router.google_auth import GoogleIdTokenVerifier

            try:
                google_verifier = GoogleIdTokenVerifier(google_id)
            except ValueError:
                # a malformed client id disables Google sign-in only; it must not take the whole portal down
                log.warning("HIVEMIND_GOOGLE_CLIENT_ID is malformed: Google sign-in disabled")
        if google_verifier is not None:
            providers["google"] = {
                "kind": "oidc", "label": "Google", "client_id": google_id, "client_secret": google_secret,
                "authorize_url": GOOGLE_AUTHORIZE_URL, "token_url": GOOGLE_TOKEN_URL, "scope": "openid email profile",
                "verifier": google_verifier, "register": "/v1/hivemind/register/google", "link": "/api/v1/auth/link/google",
                "extra": {"prompt": "select_account"},
            }
    oidc_text = e.get("HIVEMIND_OIDC_PROVIDERS", "").strip()
    if oidc_text and https_site:
        try:
            entries = json.loads(oidc_text)
            verifiers = dict(oidc_verifiers) if oidc_verifiers is not None else {}
            if oidc_verifiers is None:
                from hivemind.router.oidc_auth import providers_from_json

                verifiers = dict(providers_from_json(oidc_text))
            for entry in entries if isinstance(entries, list) else []:
                name = str(entry.get("name", ""))
                if name in ("google", "github"):
                    continue  # reserved: those providers have their own configuration and identity namespace
                secret_value = e.get(f"HIVEMIND_OIDC_{name.upper()}_CLIENT_SECRET", "").strip()
                auth_url, token_url = str(entry.get("authorization_endpoint", "")), str(entry.get("token_endpoint", ""))
                if name in verifiers and name not in providers and secret_value and auth_url.startswith("https://") and token_url.startswith("https://"):
                    providers[name] = {
                        "kind": "oidc", "label": str(entry.get("label") or name.capitalize())[:40], "client_id": str(entry.get("client_id", "")),
                        "client_secret": secret_value, "authorize_url": auth_url, "token_url": token_url, "scope": "openid email profile",
                        "verifier": verifiers[name], "register": f"/v1/hivemind/register/oidc/{name}", "link": f"/api/v1/auth/link/{name}",
                        "extra": {},
                    }
        except (ValueError, TypeError, AttributeError) as exc:
            # a broken provider list disables those providers only; it must not take the portal down
            log.warning("HIVEMIND_OIDC_PROVIDERS ignored: %s", type(exc).__name__)
    github_id = e.get("HIVEMIND_GITHUB_CLIENT_ID", "").strip()
    if github_id and https_site:
        providers["github"] = {
            "kind": "github", "label": "GitHub", "client_id": github_id, "authorize_url": GITHUB_AUTHORIZE_URL,
            "scope": "read:user user:email", "register": "/v1/hivemind/register/github", "link": "/api/v1/auth/link/github", "extra": {},
        }

    def _exchange(form: Mapping[str, str], token_url: str = GOOGLE_TOKEN_URL) -> dict[str, Any]:
        if token_exchanger is not None:
            return token_exchanger(form)
        try:
            r = httpx.post(token_url, data=dict(form), headers={"Accept": "application/json"}, timeout=10.0)
            return r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            return {}

    # ---- helpers -----------------------------------------------------------------------------------------------------
    def _client_ip(request: Request) -> str:
        """The right-most X-Forwarded-For entry is the one Cloud Run's front end appended; else the TCP peer."""
        hops = [h.strip() for h in request.headers.get("x-forwarded-for", "").split(",") if h.strip()]
        return hops[-1] if hops else (request.client.host if request.client else "")

    def _gw_post(request: Request, path: str, payload: Mapping[str, Any], bearer: str | None = None) -> tuple[int, dict[str, Any]]:
        return gw.post(path, payload, bearer=bearer, client_ip=_client_ip(request))

    def _origin(request: Request) -> str:
        if public_url:
            return public_url
        return f"{request.url.scheme}://{request.url.netloc}"

    def _security(resp: Response, *, nonce: str | None = None) -> Response:
        resp.headers[UI_HEADER] = UI_VERSION
        resp.headers["Cache-Control"] = "no-store"
        if nonce is not None:
            resp.headers["Content-Security-Policy"] = page_csp(nonce)
        else:
            resp.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Referrer-Policy"] = "no-referrer"
        return resp

    def _cookie(resp: Response, name: str, value: str, max_age: int, *, http_only: bool = True, path: str = "/", same_site: str = "lax") -> None:
        resp.set_cookie(name, value, max_age=max_age, path=path, secure=True, httponly=http_only, samesite=same_site)

    def _read_session(request: Request) -> dict[str, Any] | None:
        raw = request.cookies.get(SESSION_COOKIE, "")
        if not raw or fernet is None:
            return None
        try:
            data = json.loads(fernet.decrypt_at_time(raw.encode("ascii"), SESSION_TTL_S, int(clock())))
        except (InvalidToken, ValueError):
            return None
        return data if isinstance(data, dict) and str(data.get("k", "")).startswith("hm-") else None

    def _write_session(resp: Response, *, key: str, handle: str, role: str) -> None:
        assert fernet is not None
        token = fernet.encrypt_at_time(json.dumps({"k": key, "h": handle, "r": role, "iat": int(clock())}).encode("utf-8"), int(clock())).decode("ascii")
        _cookie(resp, SESSION_COOKIE, token, SESSION_TTL_S)

    def _json(body: Mapping[str, Any], status: int = 200) -> Response:
        return _security(JSONResponse(dict(body), status_code=status))

    def _err(code: str, status: int) -> Response:
        return _json({"error": ERRORS[code], "code": code}, status)

    def _not_configured() -> Response:
        return _err("not_configured", 503)

    def _csrf_ok(request: Request) -> bool:
        cookie = request.cookies.get(CSRF_COOKIE, "")
        header = request.headers.get("x-csrf-token", "")
        if not cookie or not header or not _eq(cookie, header):
            return False
        origin = request.headers.get("origin") or ""
        if not origin:
            ref = request.headers.get("referer") or ""
            parts = urlsplit(ref)
            origin = f"{parts.scheme}://{parts.netloc}" if parts.scheme and parts.netloc else ""
        return _eq(origin.rstrip("/"), _origin(request))

    async def _body(request: Request) -> dict[str, Any]:
        try:
            data = await request.json()
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    def _guard(request: Request) -> Response | None:
        if fernet is None:
            return _not_configured()
        if not _csrf_ok(request):
            return _err("csrf", 403)
        return None

    # ---- page and static files ---------------------------------------------------------------------------------------
    template = (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    def _language(request: Request) -> str:
        asked = (request.query_params.get("lang") or request.cookies.get("hm_lang") or "").lower()[:2]
        if asked in LANGUAGES:
            return asked
        for part in (request.headers.get("accept-language") or "").split(","):
            code = part.split(";")[0].strip().lower()[:2]
            if code in LANGUAGES:
                return code
        return "en"

    def _page(request: Request) -> Response:
        nonce = secrets.token_urlsafe(18)
        html = template.replace("{{NONCE}}", nonce).replace("{{PAPER_DOI}}", PAPER_DOI).replace("{{LANG}}", _language(request))
        resp = _security(HTMLResponse(html), nonce=nonce)
        if not request.cookies.get(CSRF_COOKIE):
            _cookie(resp, CSRF_COOKIE, secrets.token_urlsafe(24), SESSION_TTL_S, http_only=False, same_site="strict")
        return resp

    v32_prefixes = ("/v32", "/auth/")  # every /auth/<provider>/... path is the portal's, configured or not
    root_pages = ("/", "/auth", "/login", "/signup") if take_over_root else ()

    @app.middleware("http")
    async def _canonical_host(request: Request, call_next: Any) -> Response:
        """One origin only: cookies, the CSRF Origin check and the Google redirect all depend on it."""
        path = request.url.path
        mine = path.startswith(v32_prefixes) or path in root_pages
        host = (request.headers.get("host") or "").lower()
        if mine and public_host and host and host != public_host:
            if request.method in ("GET", "HEAD"):
                target = public_url + path + (f"?{request.url.query}" if request.url.query else "")
                return _security(RedirectResponse(target, status_code=301))
            return _json({"error": f"Use {public_url}.", "code": "wrong_host"}, 421)
        response = await call_next(request)
        if mine and response.headers.get(UI_HEADER) != UI_VERSION:
            # responses produced by the framework itself (404, 405, 422, 500) on a v3 path get the strict policy too
            _security(response)
        return response

    page_paths = ["/v32", "/v32/", "/auth/reset"] + (["/", "/auth", "/login", "/signup"] if take_over_root else [])
    for p in page_paths:
        app.add_api_route(p, _page, methods=["GET"], response_class=HTMLResponse, include_in_schema=False)

    @app.get("/v32/static/{name}", include_in_schema=False)
    def v32_static(name: str) -> Response:
        allowed = {"app.js": "text/javascript", "app.css": "text/css", "i18n.json": "application/json"}
        if name not in allowed:
            return _security(Response(status_code=404))
        resp = FileResponse(STATIC_DIR / name, media_type=allowed[name])
        resp.headers[UI_HEADER] = UI_VERSION
        resp.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["Cache-Control"] = "public, max-age=300"
        return resp

    # ---- facts: each with its source -----------------------------------------------------------------------------------
    @app.get("/v32/api/facts")
    def v32_facts() -> Response:
        read_at = int(clock())
        facts: list[dict[str, Any]] = []
        status, tel = gw.get("/v1/hivemind/live-telemetry")
        src = f"{gw.base_url}/v1/hivemind/live-telemetry"
        acc = tel.get("accounts", {}) if status == 200 else {}
        prod = tel.get("producers", {}) if status == 200 else {}
        models = tel.get("models", {}) if status == 200 else {}

        def fact(key: str, label: str, value: Any, source: str, note: str = "") -> None:
            ok = isinstance(value, (int, float)) and not isinstance(value, bool)
            facts.append({"key": key, "label": label, "value": value if ok else None, "available": ok, "source": source, "read_at": read_at, "note": note})

        fact("accounts", "Registered accounts", acc.get("total_registered_accounts"), src, "includes test and probe accounts")
        fact("active_24h", "Accounts active in the last 24 h", acc.get("active_accounts_24h"), src)
        fact("gpus", "Volunteer GPUs serving now", prod.get("distinct_physical_gpus"), src)
        fact("models", "Models served now", models.get("live_models_count"), src, "open-weight models run by volunteers; none is trained by HiveMind yet")
        return _json({
            "facts": facts,
            "statements": [
                {"key": "poc", "text": "HiveMind is a proof of concept, not a production service.", "source": "https://github.com/xaviercallens/HiveMind"},
                {"key": "prompts", "text": "A volunteer who serves a request sees its full prompt. Send only public, non-sensitive text.", "source": "https://github.com/xaviercallens/HiveMind/blob/main/docs/HIVEMIND_OPENROUTER.md"},
                {"key": "chain", "text": "The training chain has made held-out loss worse since round 171; the cause is being investigated.", "source": f"https://doi.org/{PAPER_DOI}"},
            ],
            "paper_doi": PAPER_DOI,
        })

    # ---- capabilities and session ----------------------------------------------------------------------------------------
    @app.get("/v32/api/capabilities")
    def v32_capabilities() -> Response:
        status, caps = gw.get("/api/v1/auth/capabilities")
        caps = caps if status == 200 and isinstance(caps, dict) else {}
        reset_email = bool(caps.get("password_reset_email"))
        gateway_oidc = caps.get("oidc") if isinstance(caps.get("oidc"), list) else []

        def gateway_accepts(name: str) -> bool:
            # show only buttons the gateway can complete: a provider configured here but not there always fails
            return bool(caps.get(name)) if name in ("google", "github") else name in gateway_oidc

        offered = [{"name": n, "label": c["label"]} for n, c in providers.items() if gateway_accepts(n)] if fernet is not None else []
        return _json({
            "sessions": fernet is not None,
            "password_login": fernet is not None,
            "password_reset_email": reset_email and fernet is not None,
            "google": any(p["name"] == "google" for p in offered),
            "providers": offered,
        })

    @app.get("/v32/api/account")
    def v32_account(request: Request) -> Response:
        s = _read_session(request)
        if s is None:
            return _err("sign_in_first", 401)
        status, me = gw.get("/v1/hivemind/me", bearer=str(s["k"]))
        if status == 401:
            return _err("session_expired", 401)
        if status != 200:
            return _err("unavailable", 503)
        balance = me.get("balance")
        usage = []
        for row in (me.get("recent_usage") or [])[:10]:
            if isinstance(row, dict):
                usage.append({
                    "utc": row.get("utc"), "model": str(row.get("model_id", ""))[:80],
                    "tokens": row.get("billed_tokens"), "charge": row.get("charge"),
                    "as": "consumer" if row.get("consumer_id") == me.get("account_id") else "producer",
                })
        return _json({
            "handle": me.get("handle", s.get("h", "")), "role": me.get("role", ""),
            "balance": balance if isinstance(balance, (int, float)) and not isinstance(balance, bool) else None,
            "usage": usage, "source": f"{gw.base_url}/v1/hivemind/me",
        })

    @app.get("/v32/api/session")
    def v32_session(request: Request) -> Response:
        s = _read_session(request)
        if s is None:
            return _json({"authenticated": False})
        return _json({"authenticated": True, "handle": s.get("h", ""), "role": s.get("r", "")})

    # ---- password sign-in, registration, sign-out ------------------------------------------------------------------------
    @app.post("/v32/api/login")
    async def v32_login(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        data = await _body(request)
        ident = str(data.get("identifier") or "").strip()[:254]
        password = str(data.get("password") or "")[:256]
        if not ident or not password:
            return _err("bad_login", 401)
        status, body = _gw_post(request, "/api/v1/auth/login", {"handle_or_email": ident, "password": password})
        if status == 429:
            return _err("rate_limited", 429)
        if status >= 500:
            return _err("unavailable", 503)
        key = str(body.get("api_key") or body.get("token") or "")
        if status != 200 or not key.startswith("hm-"):
            return _err("bad_login", 401)
        resp = _json({"authenticated": True, "handle": body.get("handle", ""), "role": body.get("role", "")})
        _write_session(resp, key=key, handle=str(body.get("handle", "")), role=str(body.get("role", "")))
        return resp

    @app.post("/v32/api/register")
    async def v32_register(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        data = await _body(request)
        email = str(data.get("email") or "").strip().lower()[:254]
        handle = str(data.get("handle") or "").strip()[:64] or email.split("@")[0]
        password = str(data.get("password") or "")
        if "@" not in email or len(password) < 12 or len(password) > 256:
            return _err("register_input", 400)
        status, body = _gw_post(request, "/api/v1/auth/register", {"handle": handle, "email": email, "password": password})
        key = str(body.get("api_key") or body.get("token") or "")
        if status == 429:
            return _err("rate_limited", 429)
        if status != 200 or not key.startswith("hm-"):
            return _err("register_refused", 409)
        created = body.get("created") is True
        out: dict[str, Any] = {"authenticated": True, "handle": body.get("handle", handle), "role": body.get("role", "")}
        if created:
            out["api_key_once"] = key  # shown once so the user can copy it; never stored by the page
        resp = _json(out)
        _write_session(resp, key=key, handle=str(out["handle"]), role=str(out["role"]))
        return resp

    @app.post("/v32/api/logout")
    async def v32_logout(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        resp = _json({"authenticated": False})
        resp.delete_cookie(SESSION_COOKIE, path="/", secure=True, httponly=True, samesite="lax")
        return resp

    # ---- account actions -----------------------------------------------------------------------------------------------
    @app.post("/v32/api/change-password")
    async def v32_change_password(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        s = _read_session(request)
        if s is None:
            return _err("sign_in_first", 401)
        data = await _body(request)
        status, body = _gw_post(request, "/api/v1/auth/change-password", {"new_password": str(data.get("new_password") or "")}, bearer=str(s["k"]))
        if status == 400:
            return _err("weak_password", 400)
        if status != 200:
            return _err("session_expired", 401)
        new_key = str(body.get("api_key") or "")
        if not new_key.startswith("hm-"):
            return _json({"status": "PASSWORD_SET"})
        # the gateway replaced the API key: this session follows it, every other session and copy of the old key stops
        resp = _json({"status": "PASSWORD_SET", "api_key_once": new_key})
        _write_session(resp, key=new_key, handle=str(s.get("h", "")), role=str(s.get("r", "")))
        return resp

    @app.post("/v32/api/rotate-key")
    async def v32_rotate_key(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        s = _read_session(request)
        if s is None:
            return _err("sign_in_first", 401)
        status, body = _gw_post(request, "/api/v1/auth/rotate-key", {}, bearer=str(s["k"]))
        key = str(body.get("api_key") or body.get("token") or "")
        if status != 200 or not body.get("old_token_revoked") or not key.startswith("hm-"):
            return _err("session_expired", 401)
        resp = _json({"status": "KEY_ROTATED", "api_key_once": key})
        _write_session(resp, key=key, handle=str(s.get("h", "")), role=str(s.get("r", "")))
        return resp

    # ---- password reset ------------------------------------------------------------------------------------------------
    @app.post("/v32/api/password-reset/request")
    async def v32_reset_request(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        data = await _body(request)
        status, body = _gw_post(request, "/api/v1/auth/password-reset/request", {"email": str(data.get("email") or "")[:254]})
        if status == 503:
            return _err("reset_unavailable", 503)
        if status == 429:
            return _err("rate_limited", 429)
        return _json({"message": "If an account uses this e-mail address, a reset link is on its way. It is valid for 30 minutes."}, 202)

    @app.post("/v32/api/password-reset/confirm")
    async def v32_reset_confirm(request: Request) -> Response:
        if (bad := _guard(request)) is not None:
            return bad
        data = await _body(request)
        status, body = _gw_post(request, 
            "/api/v1/auth/password-reset/confirm",
            {"token": str(data.get("token") or "")[:128], "new_password": str(data.get("new_password") or "")[:256]},
        )
        if status != 200:
            return _err("reset_invalid", 400)
        return _json({"status": "PASSWORD_RESET", "message": "Your password is changed. Sign in with it."})

    # ---- single sign-on (authorization code + PKCE S256 + state, and nonce for OpenID Connect; server side) -------------
    @app.get("/auth/{provider}/start", include_in_schema=False)
    def sso_start(provider: str, request: Request, link: int = 0) -> Response:
        cfg = providers.get(provider)
        if cfg is None or fernet is None:
            return _security(Response(status_code=404))
        state, nonce = secrets.token_urlsafe(24), secrets.token_urlsafe(24)
        verifier = secrets.token_urlsafe(48)
        challenge = _b64url(hashlib.sha256(verifier.encode("ascii")).digest())
        want_link = bool(link) and _read_session(request) is not None
        params = {
            "client_id": cfg["client_id"],
            "redirect_uri": f"{public_url}/auth/{provider}/callback",
            "response_type": "code",
            "scope": cfg["scope"],
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            **cfg["extra"],
        }
        if cfg["kind"] == "oidc":
            params["nonce"] = nonce
        resp = _security(RedirectResponse(f"{cfg['authorize_url']}?{urlencode(params)}", status_code=302))
        sealed = fernet.encrypt_at_time(
            json.dumps({"p": provider, "s": state, "n": nonce, "v": verifier, "l": want_link}).encode("utf-8"), int(clock())
        ).decode("ascii")
        _cookie(resp, OIDC_COOKIE, sealed, OIDC_TTL_S, path=f"/auth/{provider}")
        return resp

    @app.get("/auth/{provider}/callback", include_in_schema=False)
    def sso_callback(provider: str, request: Request, state: str = "", code: str = "", error: str = "") -> Response:
        cfg = providers.get(provider)

        def done(fragment: str) -> Response:
            resp = _security(RedirectResponse(f"/v32#{fragment}", status_code=302))
            resp.delete_cookie(OIDC_COOKIE, path=f"/auth/{provider}", secure=True, httponly=True, samesite="lax")
            return resp

        if cfg is None or fernet is None:
            return _security(Response(status_code=404))
        try:
            sealed = json.loads(fernet.decrypt_at_time(request.cookies.get(OIDC_COOKIE, "").encode("ascii"), OIDC_TTL_S, int(clock())))
        except (InvalidToken, ValueError):
            return done("signin-error=expired")
        if str(sealed.get("p", provider)) != provider:
            return done("signin-error=state")
        if error or not code or not state or not _eq(state, str(sealed.get("s", ""))):
            return done("signin-error=state")
        redirect_uri = f"{public_url}/auth/{provider}/callback"
        session = _read_session(request)
        want_link = bool(sealed.get("l")) and session is not None

        if cfg["kind"] == "github":
            payload = {"code": code[:200], "code_verifier": str(sealed.get("v", "")), "redirect_uri": redirect_uri}
        else:
            tokens = _exchange({
                "code": code,
                "client_id": cfg["client_id"],
                "client_secret": cfg["client_secret"],
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
                "code_verifier": str(sealed.get("v", "")),
            }, cfg["token_url"])
            id_token = str(tokens.get("id_token") or "")
            if not id_token:
                return done("signin-error=exchange")
            try:
                cfg["verifier"].verify(id_token)  # signature, issuer, audience, expiry, e-mail rule
            except Exception:
                return done("signin-error=token")
            if not _eq(str(_jwt_claims_unverified(id_token).get("nonce", "")), str(sealed.get("n", ""))):
                return done("signin-error=nonce")
            payload = {"id_token": id_token}

        if want_link:
            status, _ = _gw_post(request, cfg["link"], payload, bearer=str(session["k"]))
            return done("linked" if status == 200 else "signin-error=link")
        status, body = _gw_post(request, cfg["register"], payload)
        key = str(body.get("api_key") or "")
        if status != 200 or not key.startswith("hm-"):
            return done("signin-error=gateway")
        resp = done("account")
        _write_session(resp, key=key, handle=f"{cfg['label']} account", role=str(body.get("role", "")))
        return resp
