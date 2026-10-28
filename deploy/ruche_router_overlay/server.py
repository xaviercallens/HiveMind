"""Cloud Run entrypoint: HiveMind@home dashboard, with sign-up and the OpenAI-compatible API served by the real router.

The dashboard image carries its own `/v1/*`, `/signup`, `/login` and `/openrouter` routes, whose completions come from a
template generator and whose accounts live in a separate ledger. Those routes are removed here and replaced by:
- `/v1/*`: forwarded unchanged (headers, body, streaming) to the HiveMind OpenRouter, where completions are produced by
  live producers and settled in the real HMT ledger;
- `/signup`, `/login`: redirect to the router's sign-up page;
- `/openrouter`: live model list from the router and the measured result of the served Bee expert;
- `/night-moe`, `/api/v1/night-moe`: the NIGHT-MOE-1 results page.
"""

from __future__ import annotations

import html
import json
import os
import secrets
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from starlette.background import BackgroundTask

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app
from hivemind.boinc.night_moe_page import load_night_moe, render_night_moe_html

try:  # the Dockerfile copies src/hivemind/router/join_page.py next to this file (it has no imports from the rest of hivemind)
    from join_page import join_content_security_policy, pick_language, render_join_page
except ImportError:  # running from a source checkout
    from hivemind.router.join_page import join_content_security_policy, pick_language, render_join_page

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EXPERIMENTS = ARTIFACTS_DIR / "experiments"
ROUTER = os.environ.get("HIVEMIND_ROUTER_URL", "https://hivemind-openrouter-1003063861791.us-central1.run.app").rstrip("/")
PUBLIC_BASE = os.environ.get("HIVEMIND_PUBLIC_BASE", "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app").rstrip("/")
REPLACED_PREFIXES = ("/v1/",)
REPLACED_PATHS = {"/signup", "/login", "/openrouter", "/playground", "/join", "/rejoindre"}
HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te", "trailers", "transfer-encoding", "upgrade",
              "host", "content-length", "content-encoding"}

app = create_hivemind_at_home_app(
    gpu_campaigns_report_path=ARTIFACTS_DIR / "season9_10_gpu_swarm_campaigns_report.json",
    mining_ruche_report_path=ARTIFACTS_DIR / "dataset_mining_and_ruche_report.json",
    vault_dir=ARTIFACTS_DIR / "la_ruche_vault",
)
app.router.routes = [r for r in app.router.routes
                     if not (getattr(r, "path", "").startswith(REPLACED_PREFIXES) or getattr(r, "path", "") in REPLACED_PATHS)]
_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=600.0, write=60.0, pool=10.0))


@app.api_route("/v1/{path:path}", methods=["GET", "POST", "OPTIONS"])
async def forward_to_router(path: str, request: Request) -> Response:
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
    upstream = _client.build_request(request.method, f"{ROUTER}/v1/{path}", params=request.query_params, headers=headers,
                                     content=await request.body())
    try:
        resp = await _client.send(upstream, stream=True)
    except httpx.HTTPError:
        return JSONResponse(status_code=502, content={"error": {"message": "The HiveMind router is not reachable.", "type": "api_error",
                                                                "code": "router_unreachable"}})
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
    return StreamingResponse(resp.aiter_raw(), status_code=resp.status_code, headers=out_headers, background=BackgroundTask(resp.aclose))


@app.api_route("/signup", methods=["GET", "HEAD"])
@app.api_route("/login", methods=["GET", "HEAD"])
def signup_redirect() -> RedirectResponse:
    return RedirectResponse(f"{ROUTER}/signup", status_code=307)


@app.api_route("/join", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/rejoindre", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def join_page(request: Request) -> HTMLResponse:
    """Public bilingual (FR/EN) 'how to join' page. The model table comes from the router's public catalog; if the router is down the page still renders."""
    french_path = request.url.path == "/rejoindre"
    lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr" if french_path else "en")
    try:
        tiers = (await _client.get(f"{ROUTER}/v1/hivemind/catalog", timeout=4.0)).json()["tiers"]
    except (httpx.HTTPError, ValueError, KeyError):
        tiers = None
    google, handles = False, True  # what the router offers for creating an account; if it cannot be asked, assume handle registration only
    try:
        providers = (await _client.get(f"{ROUTER}/v1/hivemind/signin/providers", timeout=4.0)).json()
        google = any(p.get("name") == "google" for p in providers.get("providers", []))
        handles = bool(providers.get("handle_registration", True))
    except (httpx.HTTPError, ValueError, AttributeError):
        pass
    nonce = secrets.token_urlsafe(18)
    return HTMLResponse(
        render_join_page(lang, nonce, tiers=tiers, api_url=ROUTER, path="/rejoindre" if french_path else "/join", google_enabled=google, handle_registration=handles),
        headers={"Content-Security-Policy": join_content_security_policy(nonce), "Cache-Control": "no-store", "Vary": "Accept-Language",
                 "Content-Language": lang, "X-Content-Type-Options": "nosniff", "Referrer-Policy": "strict-origin-when-cross-origin"},
    )


def _load(name: str) -> dict[str, Any] | None:
    p = EXPERIMENTS / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@app.api_route("/openrouter", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/playground", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def openrouter_page() -> HTMLResponse:
    try:
        models = [m["id"] for m in (await _client.get(f"{ROUTER}/v1/models")).json()["data"]]
        status = "live"
    except (httpx.HTTPError, ValueError, KeyError):
        models, status = [], "unreachable"
    direct, routed = _load("serve_stars_summary.json"), _load("router_stars_summary.json")
    rows = "".join(f"<li><code>{html.escape(m)}</code></li>" for m in models) or "<li>none right now</li>"
    bee = ""
    if direct and routed:
        bee = (f"<h2>Bee expert on the T4: <code>hivemind/bee-stars</code></h2><p>Qwen2.5-Coder-7B-Instruct with the NIGHT-MOE-1 stars "
               f"adapter, served by llama.cpp on a Tesla T4. Stellar-classification pack, n={direct['n']}: served directly "
               f"{direct['correct']}/{direct['n']} ({direct['accuracy']:.4f}), identical item by item to the night evaluation; "
               f"through this router {routed['correct']}/{routed['n']} ({routed['accuracy']:.4f}). The base model scored 0.489 on "
               f"the same items. It is one specialist expert, not a routed mixture, and it has not yet passed the regression "
               f"gate on other packs; see <a href='/night-moe'>/night-moe</a>.</p>")
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HiveMind OpenRouter</title><style>body{{font-family:system-ui,sans-serif;max-width:760px;margin:2rem auto;padding:0 16px;
line-height:1.5;background:#fff;color:#111}}code,pre{{background:#f3f3f3;padding:2px 4px;border-radius:4px}}pre{{padding:12px;overflow-x:auto}}
@media (prefers-color-scheme:dark){{body{{background:#111;color:#eee}}code,pre{{background:#222}}}}</style></head><body>
<h1>HiveMind OpenRouter</h1><p>Router status: <strong>{status}</strong>. OpenAI-compatible base URL: <code>{PUBLIC_BASE}/v1</code>
(same API as <code>{ROUTER}/v1</code>).</p>
<p><a href="{ROUTER}/signup">Create an account and get an API key</a>. HMT are points with no cash value; producers see the prompts they serve.
New here? <a href="/join?lang=en">How to join</a> · <a href="/rejoindre">Comment participer (français)</a>.</p>
<h2>Models with a live producer</h2><ul>{rows}</ul>{bee}
<h2>Quick start</h2><pre>curl {PUBLIC_BASE}/v1/chat/completions \\
  -H "Authorization: Bearer $HIVEMIND_API_KEY" -H "content-type: application/json" \\
  -d '{{"model":"hivemind/bee-stars","messages":[{{"role":"user","content":"Spectrum shows strong Balmer lines, weak Ca II. Class?"}}]}}'</pre>
</body></html>"""
    return HTMLResponse(page)


@app.get("/night-moe", response_class=HTMLResponse)
def night_moe_page() -> HTMLResponse:
    return HTMLResponse(content=render_night_moe_html(load_night_moe(EXPERIMENTS / "NIGHT_MOE_results.json")))


@app.get("/api/v1/night-moe", response_class=JSONResponse)
def api_night_moe() -> JSONResponse:
    report = load_night_moe(EXPERIMENTS / "NIGHT_MOE_results.json")
    if report is None:
        return JSONResponse(status_code=404, content={"detail": "NIGHT-MOE-1 results are not deployed"})
    return JSONResponse(content=report)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
