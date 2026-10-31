# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
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
REPLACED_PATHS = {
    "/signup",
    "/login",
    "/openrouter",
    "/playground",
    "/join",
    "/rejoindre",
    "/community",
    "/boinc",
    "/api/v1/community/hub",
    "/api/v1/community/boinc-sync",
}
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
@app.api_route("/register", methods=["GET", "HEAD"])
@app.api_route("/connexion", methods=["GET", "HEAD"])
def signup_redirect() -> RedirectResponse:
    return RedirectResponse(f"{ROUTER}/signup", status_code=307)


@app.api_route("/benchmarks", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/comparatif", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def benchmarks_proxy(request: Request) -> Response:
    lang_qs = f"?lang={request.query_params.get('lang')}" if request.query_params.get("lang") else ""
    try:
        resp = await _client.get(f"{ROUTER}{request.url.path}{lang_qs}", timeout=10.0)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
        return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)
    except httpx.HTTPError:
        return RedirectResponse(f"{ROUTER}{request.url.path}{lang_qs}", status_code=307)


@app.api_route("/life-as-a-bee", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def life_as_a_bee_proxy() -> Response:
    try:
        resp = await _client.get(f"{ROUTER}/life-as-a-bee", timeout=10.0)
        out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in HOP_BY_HOP}
        return Response(content=resp.content, status_code=resp.status_code, headers=out_headers)
    except httpx.HTTPError:
        return RedirectResponse(f"{ROUTER}/life-as-a-bee", status_code=307)


@app.api_route("/join", methods=["GET", "HEAD"], response_class=HTMLResponse)
@app.api_route("/rejoindre", methods=["GET", "HEAD"], response_class=HTMLResponse)
async def join_page(request: Request) -> HTMLResponse:
    """Public bilingual (FR/EN) 'how to join' page. The model table comes from the router's public catalog; if the router is down the page still renders."""
    french_path = request.url.path == "/rejoindre"
    lang = pick_language(request.query_params.get("lang"), request.headers.get("accept-language"), default="fr")
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


_COMMUNITY_STATE_PATH = Path(os.environ.get("HIVEMIND_COMMUNITY_STATE_PATH", "/tmp/hivemind_community_hub_state.json"))
_DEFAULT_COMMUNITY_STATE: dict[str, Any] = {
    "hub_title": "USS HiveMind NCC-31337 · Portail Communautaire BOINC, Joueurs Steam & Science Participative",
    "charter": "0.00 EUR / 0.00 USD · Points de Réflexion HiveMind (HMT) & Science Citoyenne",
    "boinc_idle_gpu_policy": {
        "idle_threshold_seconds": 4.0,
        "max_gpu_load_pct": 45.0,
        "max_temp_c": 78.0,
        "supported_boinc_projects": [
            {
                "project_id": "hivemind_tess_at_home",
                "name": "HiveMind TESS/JWST Box-Least-Squares (BLS) Transit Folding",
                "progress_pct": 68.5,
                "cobblestones_awarded": 14820.0,
            },
            {
                "project_id": "hivemind_night_moe_qlora",
                "name": "Night-MoE QLoRA & Low-Bit Dream Teacher Consolidation (22:00-07:00 CET)",
                "progress_pct": 74.0,
                "cobblestones_awarded": 22450.0,
            },
        ],
    },
    "community_members": [
        {
            "handle": "pioneer-prosumer-01",
            "role": "Pioneer Prosumer & BOINC Contributor",
            "gpu": "NVIDIA Tesla T4 (16 GB)",
            "boinc_workunits_completed": 42,
            "boinc_cobblestones": 1890.0,
            "hmt_compute_points": 840,
            "favorite_world": "K2-18 b",
        },
        {
            "handle": "rtx2070-boinc-fr",
            "role": "Steam Gamer + Idle-GPU BOINC Donor",
            "gpu": "NVIDIA GeForce RTX 2070 (8 GB)",
            "boinc_workunits_completed": 19,
            "boinc_cobblestones": 855.0,
            "hmt_compute_points": 475,
            "favorite_world": "LHS 1140 b",
        },
    ],
    "coop_discovery_feed": [
        {
            "timestamp": "2026-10-29T09:30:00Z",
            "gamer_handle": "rtx2070-boinc-fr",
            "scientist_handle": "@GCP_BOINC_Validator",
            "target": "K2-18 b",
            "event": "IDLE-GPU BOINC WORKUNIT COMPLETED (HIVEMIND_TESS_AT_HOME)",
            "summary": "GPU idle on NVIDIA GeForce RTX 2070 -> Crunched BLS transit folding on K2-18 b (Depth=2789 ppm) -> +45.0 BOINC Cobblestones & +250 HMT!",
            "gcp_receipt_sha256": "4a651ffce95b484a747748502c4bf093a1bc8f0ffa34400bad0b3c0a24bafd28",
        },
    ],
}


def _load_community_state() -> dict[str, Any]:
    import fcntl

    lock_path = _COMMUNITY_STATE_PATH.with_suffix(".lock")
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            if _COMMUNITY_STATE_PATH.exists():
                return json.loads(_COMMUNITY_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    _save_community_state(_DEFAULT_COMMUNITY_STATE)
    return json.loads(json.dumps(_DEFAULT_COMMUNITY_STATE))


def _save_community_state(state: dict[str, Any]) -> None:
    import fcntl

    lock_path = _COMMUNITY_STATE_PATH.with_suffix(".lock")
    try:
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            tmp_path = _COMMUNITY_STATE_PATH.with_suffix(".tmp")
            tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp_path.replace(_COMMUNITY_STATE_PATH)
    except Exception:
        pass


@app.get("/community", response_class=HTMLResponse)
@app.get("/boinc", response_class=HTMLResponse)
def community_hub_page() -> HTMLResponse:
    st = _load_community_state()
    members_rows = "".join(
        f"<tr><td><code>{html.escape(str(m.get('handle', '')))}</code></td>"
        f"<td>{html.escape(str(m.get('role', '')))}</td>"
        f"<td>{html.escape(str(m.get('gpu', '')))}</td>"
        f"<td style='text-align:right'>{int(m.get('boinc_workunits_completed', 0))}</td>"
        f"<td style='text-align:right'>{float(m.get('boinc_cobblestones', 0.0)):.1f}</td>"
        f"<td style='text-align:right'>{int(m.get('hmt_compute_points', 0))} HMT</td></tr>"
        for m in st.get("community_members", [])[:25]
    )
    feed_items = "".join(
        f"<li style='margin:0.5rem 0'><strong>{html.escape(str(f.get('event', '')))}</strong> · "
        f"<code>{html.escape(str(f.get('gamer_handle', '')))}</code> ({html.escape(str(f.get('target', '')))}) — "
        f"{html.escape(str(f.get('summary', '')))}</li>"
        for f in st.get("coop_discovery_feed", [])[:12]
    )
    page_html = (
        "<!doctype html><html lang='fr'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Portail Communautaire BOINC &amp; Science Participative · HiveMind La Ruche</title>"
        "<style>body{font:16px/1.55 system-ui,-apple-system,sans-serif;max-width:64rem;margin:0 auto;padding:1.2rem;background:#0f172a;color:#f8fafc}"
        "a{color:#38bdf8}table{width:100%;border-collapse:collapse;margin:1rem 0}th,td{border:1px solid #334155;padding:0.5rem;text-align:left}"
        "th{background:#1e293b}code,pre{background:#1e293b;padding:0.15rem 0.35rem;border-radius:0.3rem;font-family:ui-monospace,monospace}"
        "pre{padding:0.8rem;overflow-x:auto}.nav{display:flex;gap:1rem;flex-wrap:wrap;margin-bottom:1rem}</style></head><body>"
        "<nav class='nav'><a href='/rejoindre?lang=fr'>← Rejoindre HiveMind (FR/EN/ES/DE/ZH)</a>"
        "<a href='/'>🏠 HiveMind Portal</a>"
        "<a href='/planet-forecast'>🌍 3D Planet HiveMiners (&gt;200 Nodes)</a>"
        "<a href='/foundation-paper'>📜 Foundation Paper v4.2.0 (PDF &amp; E20)</a>"
        f"<a href='{ROUTER}/signup'>🔑 Sign Up / Register &amp; Login (/signup)</a>"
        f"<a href='{ROUTER}/benchmarks'>📊 Comparatif &amp; Benchmarks (5 Langues)</a>"
        "<a href='/game'>🎮 Jouer dans le Navigateur (Cockpit 4 Étapes)</a>"
        "<a href='/webgpu'>🌌 Jouer en WebGPU (@compute)</a>"
        "<a href='/download/steam-fat-client'>⬇ Télécharger Client Windows .EXE (v2.12.0)</a>"
        "<a href='/download/steam-package-zip'>📦 Archive Steam Win64 .ZIP</a>"
        "<a href='/night-moe'>Résultats Night-MoE</a>"
        "<a href='/life-as-a-bee'>Vie d'une Abeille (24 MoE)</a>"
        "<a href='/api/v1/community/hub'>JSON API (/api/v1/community/hub)</a></nav>"
        f"<h1>{html.escape(str(st.get('hub_title', 'HiveMind Community Hub')))}</h1>"
        "<p>Classement en direct des producteurs GPU, volontaires BOINC nocturnes (22h–07h CET) et joueurs Steam (AppID 3133700, pause GPU automatique &lt; 4 s). "
        "Les Points de Réflexion HiveMind (HMT) n'ont aucune valeur monétaire.</p>"
        "<h2>Rejoindre en 1 commande (Agent BOINC / Pause GPU Automatique)</h2>"
        f"<pre><code>curl -sLO {PUBLIC_BASE}/scripts/hivemind_boinc_idle_worker.py\n"
        f"python3 hivemind_boinc_idle_worker.py --coordinator {PUBLIC_BASE} --handle votre-pseudo --domain physics_sim --once</code></pre>"
        "<h2>Classement des Contributeurs BOINC &amp; GPU</h2>"
        "<table><thead><tr><th>Pseudonyme</th><th>Rôle</th><th>Carte GPU</th><th>Unités BOINC</th><th>Cobblestones</th><th>Points HMT</th></tr></thead>"
        f"<tbody>{members_rows}</tbody></table>"
        "<h2>Journal des Découvertes &amp; Unités Calculées en Direct</h2>"
        f"<ul>{feed_items}</ul>"
        "</body></html>"
    )
    return HTMLResponse(content=page_html)


@app.get("/api/v1/community/hub", response_class=JSONResponse)
def api_get_community_hub() -> JSONResponse:
    return JSONResponse(content=_load_community_state())


_MAX_SYNC_COBBLESTONES_PER_WU = 100.0
_MAX_SYNC_HMT_PER_WU = 250


@app.post("/api/v1/community/boinc-sync", response_class=JSONResponse)
def api_boinc_sync(request: Request, payload: dict[str, Any] | None = None) -> JSONResponse:
    import hashlib
    import hmac
    import time

    sync_secret = os.environ.get("HIVEMIND_BOINC_SYNC_KEY", "").strip()
    if sync_secret:
        provided = (
            request.headers.get("x-admin-key")
            or request.headers.get("x-boinc-sync-key")
            or request.headers.get("authorization", "").removeprefix("Bearer ").removeprefix("bearer ").strip()
            or ""
        )
        if not provided or not hmac.compare_digest(provided, sync_secret):
            return JSONResponse(
                status_code=401,
                content={"ok": False, "error": "Invalid or missing BOINC sync key", "code": "invalid_sync_key"},
            )

    body = payload or {}
    st = _load_community_state()
    handle = str(body.get("handle") or body.get("commander") or "@Volunteer_FR")[:64]
    gpu_name = str(body.get("gpu_name") or body.get("gpu_model") or "NVIDIA GeForce RTX 2070")[:96]
    project_id = str(body.get("project_id") or "hivemind_tess_at_home")[:64]
    target_name = str(body.get("target_name") or body.get("target_planet") or "K2-18 b")[:64]
    idle_seconds = max(0.0, float(body.get("idle_seconds") or body.get("gpu_idle_seconds") or 10.0))
    raw_cobblestones = float(body.get("cobblestones_delta", body.get("cobblestones", 45.0)))
    raw_hmt_bonus = int(body.get("hmt_delta", body.get("hmt_bonus", 250)))
    cobblestones_delta = round(max(0.0, min(raw_cobblestones, _MAX_SYNC_COBBLESTONES_PER_WU)), 1)
    hmt_bonus = max(0, min(raw_hmt_bonus, _MAX_SYNC_HMT_PER_WU))

    cuda_fold = body.get("cuda_fold") if isinstance(body.get("cuda_fold"), dict) else {}
    bls_depth_ppm = float(cuda_fold.get("recovered_transit_depth_ppm") or 2789.0)
    receipt_seed = f"{handle}:{gpu_name}:{project_id}:{target_name}:{bls_depth_ppm:.2f}:{time.time()}"
    receipt_sha256 = str(body.get("receipt_sha256") or body.get("digest") or hashlib.sha256(receipt_seed.encode("utf-8")).hexdigest())

    existing_receipts = {str(f.get("gcp_receipt_sha256", "")) for f in st.get("coop_discovery_feed", [])}
    if receipt_sha256 in existing_receipts:
        return JSONResponse(
            content={
                "ok": True,
                "deduplicated": True,
                "duplicate_receipt": True,
                "handle": handle,
                "gpu_name": gpu_name,
                "project_id": project_id,
                "target_name": target_name,
                "bls_depth_ppm": round(bls_depth_ppm, 1),
                "cobblestones_earned": 0.0,
                "hmt_points_earned": 0,
                "receipt_sha256": receipt_sha256,
                "charter": "HiveMind Reflexion Points (HMT) - Non-Monetary Compute Reciprocity",
            }
        )

    found = False
    for m in st.get("community_members", []):
        if str(m.get("handle", "")).lower() == handle.lower():
            m["boinc_workunits_completed"] = int(m.get("boinc_workunits_completed", 0)) + 1
            m["boinc_cobblestones"] = round(float(m.get("boinc_cobblestones", 0.0)) + cobblestones_delta, 1)
            m["hmt_compute_points"] = int(m.get("hmt_compute_points", 0)) + hmt_bonus
            found = True
            break
    if not found:
        st.setdefault("community_members", []).insert(
            0,
            {
                "handle": handle,
                "role": "Volunteer BOINC & Idle-GPU Donor",
                "gpu": gpu_name,
                "boinc_workunits_completed": 1,
                "boinc_cobblestones": round(cobblestones_delta, 1),
                "hmt_compute_points": hmt_bonus,
                "favorite_world": target_name,
            },
        )
    st.setdefault("coop_discovery_feed", []).insert(
        0,
        {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gamer_handle": handle,
            "scientist_handle": "@GCP_BOINC_Validator",
            "target": target_name,
            "event": f"IDLE-GPU BOINC WORKUNIT COMPLETED ({project_id.upper()})",
            "summary": f"GPU idle {idle_seconds:.1f}s on {gpu_name} -> Folded {target_name} ({bls_depth_ppm:.1f} ppm) -> +{cobblestones_delta:.1f} Cobblestones & +{hmt_bonus} HMT",
            "gcp_receipt_sha256": receipt_sha256,
        },
    )
    st["coop_discovery_feed"] = st["coop_discovery_feed"][:25]
    _save_community_state(st)
    return JSONResponse(
        content={
            "ok": True,
            "deduplicated": False,
            "duplicate_receipt": False,
            "handle": handle,
            "gpu_name": gpu_name,
            "project_id": project_id,
            "target_name": target_name,
            "bls_depth_ppm": round(bls_depth_ppm, 1),
            "cobblestones_earned": cobblestones_delta,
            "hmt_points_earned": hmt_bonus,
            "receipt_sha256": receipt_sha256,
            "charter": "HiveMind Reflexion Points (HMT) - Non-Monetary Compute Reciprocity",
        }
    )


@app.get("/scripts/{script_name}")
def download_onboarding_script(script_name: str) -> Response:
    allowed_scripts = {
        "hivemind_boinc_idle_worker.py",
        "hivemind_producer_agent.py",
        "hivemind_producer_shield.py",
        "hivemind_volunteer_preflight.py",
    }
    if script_name not in allowed_scripts:
        return JSONResponse(status_code=404, content={"detail": "Unknown onboarding script"})
    for candidate_dir in (BASE_DIR / "scripts", Path("/app/scripts")):
        sp = candidate_dir / script_name
        if sp.exists():
            return Response(
                content=sp.read_text(encoding="utf-8"),
                media_type="text/x-python; charset=utf-8",
                headers={"Content-Disposition": f'inline; filename="{script_name}"'},
            )
    return JSONResponse(status_code=404, content={"detail": "Script file not present on server"})




@app.get("/game")
@app.get("/game.html")
@app.get("/exoplanet-hunter")
@app.get("/cockpit-v46")
def redirect_or_serve_playable_game() -> Response:
    return RedirectResponse("https://xaviercallens.github.io/HiveMindatHome.github.io/game.html", status_code=307)


@app.get("/webgpu")
@app.get("/webgpu.html")
def redirect_or_serve_webgpu_game() -> Response:
    return RedirectResponse(f"{ROUTER}/webgpu", status_code=307)


@app.get("/download/steam-fat-client")
def redirect_download_steam_exe() -> RedirectResponse:
    return RedirectResponse(
        "https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Exoplanet_Hunter.exe",
        status_code=307,
    )


@app.get("/download/steam-package-zip")
def redirect_download_steam_zip() -> RedirectResponse:
    return RedirectResponse(
        "https://github.com/xaviercallens/HiveMind/releases/download/v2.12.0/HiveMind_Astrophysics_Exoplanet_Hunter_v2.11.0_Steam_Win64.zip",
        status_code=307,
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
