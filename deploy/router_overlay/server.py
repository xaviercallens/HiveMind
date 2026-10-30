# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Router entrypoint overlay: price among recently active consumers, show points only, and serve 5-language portal, signup & join.

1. `funded_consumer_bids` counts every funded account as a bidder in the Kelly game. With a starter grant on every sign-up,
   idle accounts that once raised their max price keep the clearing price above the default max price of new accounts,
   so a new user's first request is refused (402). This keeps only accounts with a request or a hold in the last
   HIVEMIND_BID_WINDOW_S seconds (default 900; 0 restores the original rule), as PR #182 does in the source.
2. Pages and account JSON are passed through `to_points_wording`: no euro or dollar amount is shown, only HiveMind
   Reflexion Points (HMT). Completions (`/v1/chat/...`, `/v1/completions`) and `/gallery/` are never rewritten.
3. `/`, `/playground`, `/life-as-a-bee`, `/fr`, `/aire-de-jeu`, `/es`, `/de`, `/zh`, `/signup`, `/inscription`, `/join`,
   `/rejoindre`, and `/v1/hivemind/i18n` serve the 5-language (`en`, `fr`, `es`, `de`, `zh`) OpenRouter experience.
"""

from __future__ import annotations

import os
import runpy
import secrets
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from hivemind.router.accounts import RouterStore
from hivemind.router.catalog import catalog_payload
from hivemind.router.join_page import DEFAULT_API_URL, join_content_security_policy, pick_language, render_join_page
from hivemind.router.points_wording import to_points_wording
from hivemind.router.portal_page import (
    PORTAL_I18N,
    PORTAL_LANG_META,
    PORTAL_LANGS,
    portal_content_security_policy,
    render_portal_page,
)
from hivemind.router.signup_page import SIGNUP_I18N, content_security_policy, render_signup_page

BID_WINDOW_S: float = float(os.environ.get("HIVEMIND_BID_WINDOW_S", "900") or "900")
REWRITTEN_TYPES: tuple[str, ...] = ("text/html", "application/json", "application/javascript", "text/javascript")
NEVER_REWRITTEN: tuple[str, ...] = ("/v1/chat/", "/v1/completions", "/gallery/")
GALLERY_DIR: Path = Path(os.environ.get("HIVEMIND_GALLERY_DIR", "/app/public/gallery"))
_REPLACED_PATHS: set[str] = {
    "/",
    "/playground",
    "/life-as-a-bee",
    "/fr",
    "/aire-de-jeu",
    "/es",
    "/de",
    "/zh",
    "/signup",
    "/inscription",
    "/join",
    "/rejoindre",
    "/v1/hivemind/i18n",
}
_funded_consumer_bids = RouterStore.funded_consumer_bids


def active_funded_consumer_bids(self: RouterStore, expected_tokens: int, *args: Any, **kwargs: Any) -> dict[str, float]:
    bids = _funded_consumer_bids(self, expected_tokens, *args, **kwargs)
    if BID_WINDOW_S <= 0:
        return bids
    since = time.time() - BID_WINDOW_S
    with self._lock:
        rows = self._conn.execute(
            "SELECT consumer_id AS a FROM usage WHERE utc >= ? UNION SELECT account_id AS a FROM holds WHERE utc >= ?", (since, since)
        ).fetchall()
    active = {str(r["a"]) for r in rows}
    return {a: p for a, p in bids.items() if a in active}


RouterStore.funded_consumer_bids = active_funded_consumer_bids
app: FastAPI = runpy.run_path(os.environ.get("HIVEMIND_ORIG_SERVER", "/app/server_orig.py"), run_name="hivemind_router_server")["app"]
app.router.routes = [r for r in app.router.routes if getattr(r, "path", "") not in _REPLACED_PATHS]


def _google_client_id() -> str | None:
    raw = os.environ.get("HIVEMIND_GOOGLE_CLIENT_ID", "").strip()
    return raw or None


@app.api_route("/", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/life-as-a-bee", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/fr", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/aire-de-jeu", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/es", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/de", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/zh", methods=["GET", "HEAD"], response_class=HTMLResponse)
def portal_overlay(request: Request) -> HTMLResponse:
    path_lang_defaults = {
        "/fr": "fr",
        "/aire-de-jeu": "fr",
        "/es": "es",
        "/de": "de",
        "/zh": "zh",
    }
    default_lang = path_lang_defaults.get(request.url.path, "en")
    explicit_lang = request.query_params.get("lang")
    if explicit_lang is None and request.url.path in path_lang_defaults:
        lang = default_lang
    else:
        lang = pick_language(explicit_lang, request.headers.get("accept-language"), default=default_lang)
    client_id = _google_client_id()
    nonce = secrets.token_urlsafe(18)
    return HTMLResponse(
        render_portal_page(client_id, nonce, lang=lang),
        headers={
            "Content-Security-Policy": portal_content_security_policy(nonce),
            "Cache-Control": "public, max-age=60",
            "Vary": "Accept-Language",
            "Content-Language": lang,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        },
    )


@app.api_route("/signup", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/inscription", methods=["GET", "HEAD"], response_class=HTMLResponse)
def signup_overlay(request: Request) -> HTMLResponse:
    french_path = request.url.path == "/inscription"
    default_lang = "fr" if french_path else "en"
    explicit_lang = request.query_params.get("lang")
    if explicit_lang is None and french_path:
        lang = "fr"
    else:
        lang = pick_language(explicit_lang, request.headers.get("accept-language"), default=default_lang)
    client_id = _google_client_id()
    nonce = secrets.token_urlsafe(18)
    return HTMLResponse(
        render_signup_page(client_id, nonce, lang=lang),
        headers={
            "Content-Security-Policy": content_security_policy(nonce),
            "Cache-Control": "no-store",
            "Vary": "Accept-Language",
            "Content-Language": lang,
            "Cross-Origin-Opener-Policy": "same-origin-allow-popups",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        },
    )


@app.api_route("/join", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/rejoindre", methods=["GET", "HEAD"], response_class=HTMLResponse)
def join_page_overlay(request: Request) -> HTMLResponse:
    french_path = request.url.path == "/rejoindre"
    lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
    nonce = secrets.token_urlsafe(18)
    api_url = os.environ.get("HIVEMIND_PUBLIC_URL", DEFAULT_API_URL).rstrip("/")
    google_enabled = bool(_google_client_id())
    handle_reg = os.environ.get("HIVEMIND_OPEN_REGISTRATION", "1").strip().lower() not in ("0", "false", "no")
    return HTMLResponse(
        render_join_page(
            lang,
            nonce,
            tiers=catalog_payload()["tiers"],
            api_url=api_url,
            path="/rejoindre" if french_path else "/join",
            google_enabled=google_enabled,
            handle_registration=handle_reg,
        ),
        headers={
            "Content-Security-Policy": join_content_security_policy(nonce),
            "Cache-Control": "no-store",
            "Vary": "Accept-Language",
            "Content-Language": lang,
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "strict-origin-when-cross-origin",
        },
    )


@app.api_route("/gallery/{filename}", methods=["GET", "HEAD"])
def gallery_asset_overlay(filename: str) -> FileResponse:
    if "/" in filename or ".." in filename or not filename.endswith((".jpg", ".svg")):
        raise HTTPException(status_code=404, detail="Not found")
    target = (GALLERY_DIR / filename).resolve()
    if not target.is_file() or GALLERY_DIR.resolve() not in target.parents:
        raise HTTPException(status_code=404, detail="Not found")
    media_type = "image/jpeg" if filename.endswith(".jpg") else "image/svg+xml"
    return FileResponse(target, media_type=media_type, headers={"Cache-Control": "public, max-age=3600"})


@app.get("/v1/hivemind/i18n")
def i18n_catalog_overlay(lang: str | None = None) -> dict[str, Any]:
    selected = lang if lang in PORTAL_LANGS else None
    return {
        "object": "hivemind.i18n_catalog",
        "supported_languages": list(PORTAL_LANGS),
        "language_metadata": PORTAL_LANG_META,
        "official_points_names": {
            "en": "HiveMind Reflexion Points (HMT)",
            "fr": "Points de Réflexion HiveMind (HMT)",
            "es": "Puntos de Reflexión HiveMind (HMT)",
            "de": "HiveMind-Reflexionspunkte (HMT)",
            "zh": "HiveMind 反思积分 (HMT)",
        },
        "routes": {
            "en": "/?lang=en",
            "fr": "/fr",
            "es": "/es",
            "de": "/de",
            "zh": "/zh",
            "join": "/join",
            "rejoindre": "/rejoindre",
            "signup": "/signup",
        },
        "portal": {selected: PORTAL_I18N[selected]} if selected else PORTAL_I18N,
        "signup": {selected: SIGNUP_I18N[selected]} if selected else SIGNUP_I18N,
    }


@app.middleware("http")
async def points_only(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    response = await call_next(request)
    if request.url.path.startswith(NEVER_REWRITTEN) or not response.headers.get("content-type", "").startswith(REWRITTEN_TYPES):
        return response
    body = b"".join([chunk async for chunk in response.body_iterator])  # type: ignore[attr-defined]
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError:
        new_body = body
    else:
        new_body = to_points_wording(text).encode("utf-8")
    headers = {k: v for k, v in response.headers.items() if k.lower() != "content-length"}
    return Response(content=new_body, status_code=response.status_code, headers=headers)
