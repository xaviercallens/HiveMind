# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified Login & Registration screen served at ``/signup``, ``/login``, ``/register``, and ``/connexion``.

Supports:
1. **Returning / Previous User Login** (Username / Handle / Email + Password, or existing ``hm-...`` / ``hm_live_...`` API key)
   via ``POST /v1/hivemind/auth/login`` (and ``POST /login`` HTML form fallback).
2. **New User Registration** (Username / Handle + Password + optional Email + Role ``prosumer`` / ``consumer`` / ``producer`` / ``trainer``)
   via ``POST /v1/hivemind/auth/register`` (and ``POST /signup`` HTML form fallback).
3. **Google Sign-In** (when ``client_id`` is configured) via ``POST /v1/hivemind/register/google``.

All dynamic DOM updates use ``textContent`` (never ``innerHTML``), and the Content-Security-Policy enforces per-request nonces.
"""

from __future__ import annotations

import html
import math
import re
from typing import Any, Mapping

from hivemind.router.google_auth import CLIENT_ID_RE

# Origins Google documents for the Identity Services button (script, frame, connect, style).
GSI = "https://accounts.google.com/gsi/"
_NONCE_RE = re.compile(r"^[A-Za-z0-9+/=_-]{8,128}$")


def content_security_policy(nonce: str) -> str:
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        raise ValueError("nonce must be an 8-128 char URL/base64-safe token")
    return "; ".join([
        "default-src 'none'",
        f"script-src 'nonce-{nonce}' {GSI}client",
        f"style-src 'nonce-{nonce}' {GSI}style",
        f"frame-src {GSI}",
        f"connect-src 'self' {GSI}",
        "img-src 'self' data:",
        "base-uri 'none'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ])


_CSS = """
:root{color-scheme:light dark;--bg:#0b0f19;--fg:#f1f5f9;--mut:#94a3b8;--card:#131b2e;--input:#0f172a;--line:#26334d;--acc:#38bdf8;--amber:#fbbf24;--green:#34d399;--err:#f87171}
@media (prefers-color-scheme:light){:root{--bg:#f8fafc;--fg:#0f172a;--mut:#475569;--card:#ffffff;--input:#f1f5f9;--line:#cbd5e1;--acc:#0284c7;--amber:#d97706;--green:#059669;--err:#dc2626}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,sans-serif}
main{max-width:54rem;margin:0 auto;padding:1.75rem 1rem 3.5rem}
header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.75rem;padding-bottom:1rem;border-bottom:1px solid var(--line);margin-bottom:1.25rem}
h1{font-size:1.55rem;margin:.2rem 0}
h2{font-size:1.12rem;margin:0 0 .5rem;color:var(--amber)}
.nav-links a{color:var(--acc);text-decoration:none;margin-left:.85rem;font-size:.9rem;font-weight:600}
.nav-links a:hover{text-decoration:underline}
.grid-auth{display:grid;grid-template-columns:1fr 1fr;gap:1.15rem;margin:1rem 0}
@media (max-width:740px){.grid-auth{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:.7rem;padding:1.2rem;margin:.85rem 0}
fieldset{border:0;padding:0;margin:.5rem 0 .85rem}legend{font-weight:600;margin-bottom:.35rem;font-size:.92rem}
label{display:block;padding:.25rem 0;font-size:.88rem;font-weight:600}
.radio-label{font-weight:400;display:flex;align-items:center;gap:.45rem;padding:.2rem 0}
input[type=text],input[type=password],input[type=email]{width:100%;padding:.55rem .7rem;margin-top:.2rem;margin-bottom:.55rem;border:1px solid var(--line);border-radius:.45rem;background:var(--input);color:var(--fg);font:inherit}
input:focus{outline:2px solid var(--acc);border-color:transparent}
code,pre{font:.88rem ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all;white-space:pre-wrap}
pre{background:var(--input);border:1px solid var(--line);border-radius:.45rem;padding:.7rem;margin:.5rem 0}
button,.btn-link{font:inherit;font-weight:600;padding:.55rem 1rem;border-radius:.45rem;border:1px solid var(--line);background:var(--acc);color:#06101e;cursor:pointer;text-decoration:none;display:inline-block}
button.secondary,.btn-link.secondary{background:var(--card);color:var(--fg)}
button:hover,.btn-link:hover{filter:brightness(1.08)}
.mut{color:var(--mut);font-size:.86rem}
.ok-banner{border-color:var(--green);background:rgba(52,211,153,.08)}
.err-text{color:var(--err);font-weight:600}
[hidden]{display:none !important}
"""

_AUTH_JS = r"""
const $ = (id) => document.getElementById(id);
function setStatus(text, isErr) {
  const el = $('status');
  if (!el) return;
  el.textContent = text;
  el.className = isErr ? 'err-text' : 'mut';
}
function persistUnifiedSession(body) {
  try {
    const tok = body.api_key || body.token || '';
    const h = body.handle || '';
    if (tok) localStorage.setItem('hivemind_api_key', tok);
    if (h) localStorage.setItem('hivemind_user_handle', h);
    if (tok && h) {
      const unified = {
        handle: h,
        email: body.email || (h + '@hivemind.network'),
        provider: body.provider || 'OPENROUTER_AUTH',
        linkedProviders: body.linked_providers || [body.provider || 'OPENROUTER_AUTH'],
        role: body.role || 'prosumer',
        targetProfile: body.target_profile || 'developer',
        points: typeof body.compute_points === 'number' ? body.compute_points : Math.round(Number(body.balance_hmt || body.balance || 0)),
        token: tok,
        nodeId: body.node_id || '0xHIVEMIND',
        loginCount: body.login_count || 1,
        isNewUser: Boolean(body.created),
        isGuest: false,
        userState: body.created ? 'NEW_USER' : 'EXISTING_USER',
      };
      localStorage.setItem('hivemind_user', JSON.stringify(unified));
    }
  } catch (e) {}
}
function showSessionResult(body) {
  const tok = body.api_key || body.token || '';
  if ($('key')) $('key').textContent = tok;
  if ($('acct')) $('acct').textContent = (body.account_id || '') + (body.handle ? ' (' + body.handle + ')' : '');
  if ($('role')) $('role').textContent = body.role || 'prosumer';
  if ($('bal')) $('bal').textContent = (typeof body.balance_hmt === 'number' ? body.balance_hmt.toFixed(2) : (typeof body.balance === 'number' ? body.balance.toFixed(2) : (typeof body.compute_points === 'number' ? body.compute_points.toFixed(2) : 'n/a'))) + ' HMT';
  if ($('kind')) {
    $('kind').textContent = body.created
      ? 'Account created & authenticated! Your API key is saved in this browser for the Playground and Unified v3.1 Portal.'
      : 'Welcome back! Your previous account & HMT balance have been restored and authenticated.';
  }
  persistUnifiedSession(body);
  if ($('result')) $('result').hidden = false;
  setStatus('');
}
function chosenRegRole() {
  const checked = document.querySelector('input[name=role]:checked');
  return checked ? checked.value : 'prosumer';
}
try {
  const qRole = (new URLSearchParams(window.location.search).get('role') || '').trim().toLowerCase();
  if (['prosumer', 'consumer', 'producer', 'trainer'].includes(qRole)) {
    const radio = document.querySelector('input[name=role][value="' + qRole + '"]');
    if (radio) radio.checked = true;
  }
} catch (e) {}
if ($('loginForm')) {
  $('loginForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const identifier = ($('loginIdentifier').value || '').trim();
    const password = ($('loginPassword').value || '').trim();
    if (!identifier) {
      setStatus('Please enter your username, handle, email, or existing hm-... API key.', true);
      return;
    }
    setStatus('Signing in & restoring your HiveMind account...', false);
    try {
      const reply = await fetch('/v1/hivemind/auth/login', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({identifier: identifier, password: password}),
      });
      const body = await reply.json().catch(() => ({}));
      if (!reply.ok) {
        setStatus((body.error && body.error.message) || 'Sign-in failed. Check your username/password.', true);
        return;
      }
      showSessionResult(body);
    } catch (err) {
      setStatus('Could not reach the gateway. Try again.', true);
    }
  });
}
if ($('registerForm')) {
  $('registerForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const handle = ($('regHandle').value || '').trim();
    const email = ($('regEmail').value || '').trim();
    const password = ($('regPassword').value || '').trim();
    const role = chosenRegRole();
    if (!handle) {
      setStatus('Please choose a username / handle (e.g. alice_gpu or @alice).', true);
      return;
    }
    setStatus('Creating your HiveMind account...', false);
    try {
      const reply = await fetch('/v1/hivemind/auth/register', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({handle: handle, email: email, password: password, role: role}),
      });
      const body = await reply.json().catch(() => ({}));
      if (!reply.ok) {
        setStatus((body.error && body.error.message) || 'Registration failed.', true);
        return;
      }
      showSessionResult(body);
    } catch (err) {
      setStatus('Could not reach the gateway. Try again.', true);
    }
  });
}
if ($('copy')) {
  $('copy').addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText($('key').textContent);
      $('copy').textContent = 'Copied!';
    } catch (err) {
      $('copy').textContent = 'Select and copy the key manually';
    }
  });
}

async function syncSignupLiveTelemetry() {
  try {
    const res = await fetch('/v1/hivemind/live-telemetry', { headers: { 'Accept': 'application/json' } });
    if (!res.ok) return;
    const d = await res.json();
    if (!d || !d.accounts) return;
    const acct = d.accounts || {};
    const byRole = acct.by_role || {};
    const models = d.models || {};
    const t4 = d.t4_boinc || {};
    const ot = d.opentrainer || {};
    if ($('signupLiveTelemetry')) {
      const rd = function(v, thousands) { return (v === null || v === undefined) ? 'n/a' : (thousands ? Number(v).toLocaleString() : String(v)); };
      const partial = [acct.total_registered_accounts, models.live_models_count, t4.vram_free_mib].some(function(v) { return v === null || v === undefined; });
      $('signupLiveTelemetry').textContent =
        '🟢 ' + (partial ? 'GCP Swarm (live telemetry partly unavailable)' : 'Live GCP Swarm') + ': ' + rd(acct.total_registered_accounts) + ' Registered Accounts (' +
        rd(byRole.prosumer) + ' prosumers, ' +
        rd(byRole.producer) + ' producers, ' +
        rd(byRole.consumer) + ' consumers, ' +
        rd(byRole.trainer) + ' trainers) • ' +
        rd(models.live_models_count) + ' Live Models • T4 Free VRAM ' +
        rd(t4.vram_free_mib, true) + ' MiB • OpenTrainer ' +
        rd(ot.total_registered_nodes) + ' Trainer GPU(s) / ' +
        rd(ot.accepted_rounds_count) + ' Accepted Round(s)';
    }
  } catch (e) {}
}
async function checkActiveBrowserSession() {
  try {
    let tok = localStorage.getItem('hivemind_api_key') || '';
    if (!tok) {
      const rawUser = localStorage.getItem('hivemind_user');
      if (rawUser) {
        const parsed = JSON.parse(rawUser);
        if (parsed && parsed.token) tok = parsed.token;
      }
    }
    if ($('key') && $('key').textContent && $('key').textContent.trim()) {
      const serverTok = $('key').textContent.trim();
      const acctText = ($('acct') ? $('acct').textContent : '') || '';
      const hMatch = acctText.match(/\(([^)]+)\)/);
      persistUnifiedSession({
        api_key: serverTok,
        handle: hMatch ? hMatch[1] : acctText,
        role: ($('role') ? $('role').textContent : 'prosumer'),
      });
      tok = serverTok;
    }
    if (!tok) return;
    const res = await fetch('/api/v1/auth/session', { headers: { 'Authorization': 'Bearer ' + tok, 'Accept': 'application/json' } });
    if (!res.ok) return;
    const s = await res.json();
    if (!s || !s.authenticated || !s.user) return;
    persistUnifiedSession({
      api_key: tok,
      handle: s.user.handle,
      role: s.user.role,
      balance_hmt: s.user.balance_hmt,
      compute_points: s.user.compute_points,
      created: false,
    });
    if ($('activeSessionHandle')) $('activeSessionHandle').textContent = '@' + (s.user.handle || 'operator');
    if ($('activeSessionBalance')) $('activeSessionBalance').textContent = (typeof s.user.balance_hmt === 'number' ? s.user.balance_hmt.toFixed(2) : String(s.user.compute_points || 0)) + ' HMT';
    if ($('existingBrowserSessionBanner')) $('existingBrowserSessionBanner').hidden = false;
  } catch (e) {}
}
if ($('rotateKeyBtn')) {
  $('rotateKeyBtn').addEventListener('click', async () => {
    try {
      const tok = ($('key') && $('key').textContent.trim()) || localStorage.getItem('hivemind_api_key') || '';
      if (!tok) return;
      const res = await fetch('/api/v1/auth/rotate-key', { method: 'POST', headers: { 'Authorization': 'Bearer ' + tok } });
      if (!res.ok) return;
      const d = await res.json();
      if (d && d.token) {
        if ($('key')) $('key').textContent = d.token;
        persistUnifiedSession({ api_key: d.token, handle: d.handle || localStorage.getItem('hivemind_user_handle') || 'operator' });
        setStatus('API key rotated! Your old key is revoked and your new key is saved in this browser.', false);
      }
    } catch (e) {}
  });
}
if ($('signOutSessionBtn')) {
  $('signOutSessionBtn').addEventListener('click', async () => {
    try {
      const tok = localStorage.getItem('hivemind_api_key') || '';
      if (tok) await fetch('/api/v1/auth/logout', { method: 'POST', headers: { 'Authorization': 'Bearer ' + tok } });
    } catch (e) {}
    try {
      localStorage.removeItem('hivemind_api_key');
      localStorage.removeItem('hivemind_user_handle');
      localStorage.removeItem('hivemind_user');
    } catch (e) {}
    if ($('existingBrowserSessionBanner')) $('existingBrowserSessionBanner').hidden = true;
    if ($('result')) $('result').hidden = true;
    setStatus('Signed out of browser session.', false);
  });
}
syncSignupLiveTelemetry();
checkActiveBrowserSession();
"""

_GOOGLE_JS = """
const CLIENT_ID = %CLIENT_ID%;
async function onCredential(response) {
  setStatus('Checking your Google sign-in...', false);
  try {
    const reply = await fetch('/v1/hivemind/register/google', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id_token: response.credential, role: chosenRegRole()}),
    });
    const body = await reply.json().catch(() => ({}));
    if (!reply.ok) { setStatus((body.error && body.error.message) || 'Sign-in failed.', true); return; }
    $('key').textContent = body.api_key;
    $('acct').textContent = body.account_id;
    $('kind').textContent = body.created ? 'Your account was created.' : 'Welcome back: your previous key no longer works.';
    $('role').textContent = body.role;
    persistUnifiedSession({api_key: body.api_key, handle: body.handle || body.account_id, role: body.role, provider: 'GOOGLE_OIDC', created: body.created});
    $('result').hidden = false; if ($('signin')) $('signin').hidden = true; setStatus('', false);
  } catch (err) { setStatus('Could not reach the gateway. Try again.', true); }
}
function init() {
  google.accounts.id.initialize({client_id: CLIENT_ID, callback: onCredential, auto_select: false});
  google.accounts.id.renderButton($('gbtn'), {theme: 'outline', size: 'large', text: 'signin_with', shape: 'pill'});
}
window.onGoogleLibraryLoad = init;
"""


def _reading(source: Mapping[str, Any], key: str, *, thousands: bool = False) -> str:
    """The integer reading at ``key`` as text, or "n/a" when the live snapshot has none."""
    value = source.get(key)
    if value is None:
        return "n/a"
    number = int(value)
    return f"{number:,}" if thousands else str(number)


def render_signup_page(
    client_id: str | None,
    nonce: str,
    *,
    auth_result: dict[str, Any] | None = None,
    error_message: str | None = None,
    live_telemetry: dict[str, Any] | None = None,
    starter_grant_hmt: float = 0.0,
    default_role: str = "prosumer",
) -> str:
    """Render the unified Login & Registration page at ``/signup`` and ``/login``."""
    if not isinstance(nonce, str) or not nonce.strip():
        raise ValueError("nonce must be a non-empty string")
    if client_id is not None and (not isinstance(client_id, str) or not CLIENT_ID_RE.match(client_id)):
        raise ValueError("invalid Google client id")
    if isinstance(starter_grant_hmt, bool) or not isinstance(starter_grant_hmt, (int, float)) or not math.isfinite(float(starter_grant_hmt)) or float(starter_grant_hmt) < 0.0:
        raise ValueError("starter_grant_hmt must be a finite non-negative float")
    if auth_result is not None and not isinstance(auth_result, Mapping):
        raise ValueError("auth_result must be a mapping or None")
    if live_telemetry is not None and not isinstance(live_telemetry, Mapping):
        raise ValueError("live_telemetry must be a mapping or None")
    role_norm = str(default_role or "prosumer").strip().lower()
    selected_role = role_norm if role_norm in ("prosumer", "consumer", "producer", "trainer") else "prosumer"
    chk_prosumer = " checked" if selected_role == "prosumer" else ""
    chk_consumer = " checked" if selected_role == "consumer" else ""
    chk_producer = " checked" if selected_role == "producer" else ""
    chk_trainer = " checked" if selected_role == "trainer" else ""
    lt = live_telemetry or {}
    acct_summary = lt.get("accounts") or {}
    by_role = acct_summary.get("by_role") or {}
    models_live = lt.get("models") or {}
    t4_live = lt.get("t4_boinc") or {}
    ot_live = lt.get("opentrainer") or {}
    # A missing reading is shown as "n/a", never replaced by an invented number.
    total_accts = _reading(acct_summary, "total_registered_accounts")
    prosumers_cnt = _reading(by_role, "prosumer")
    producers_cnt = _reading(by_role, "producer")
    consumers_cnt = _reading(by_role, "consumer")
    trainers_cnt = _reading(by_role, "trainer")
    live_models_cnt = _reading(models_live, "live_models_count")
    t4_free_mib = _reading(t4_live, "vram_free_mib", thousands=True)
    ot_nodes = _reading(ot_live, "total_registered_nodes")
    ot_rounds = _reading(ot_live, "accepted_rounds_count")
    telemetry_label = "Live GCP Swarm" if "n/a" not in (total_accts, live_models_cnt, t4_free_mib) else "GCP Swarm (live telemetry partly unavailable)"

    starter_line = (
        f"Starter grant on this gateway: {starter_grant_hmt:.2f} HMT per new account, at most a few per address per day."
        if starter_grant_hmt > 0.0
        else "A new account starts at 0.00 HMT on this gateway: points come from sharing a GPU or from an operator grant."
    )
    result_hidden = "" if auth_result else " hidden"
    res_kind = ""
    res_key = ""
    res_acct = ""
    res_role = ""
    res_bal = "n/a"
    if auth_result:
        res_kind = (
            "Your account was created."
            if auth_result.get("created")
            else "Welcome back! Your previous account & HMT balance have been restored."
        )
        res_key = html.escape(str(auth_result.get("api_key", "")))
        res_acct = html.escape(
            f"{auth_result.get('account_id', '')}"
            + (f" ({auth_result.get('handle')})" if auth_result.get("handle") else "")
        )
        res_role = html.escape(str(auth_result.get("role", "prosumer")))
        bal_val = float(auth_result.get("balance_hmt", auth_result.get("balance", 0.0)))
        res_bal = f"{bal_val:.2f} HMT"

    err_html = html.escape(error_message) if error_message else ""
    err_cls = "err-text" if error_message else "mut"

    if client_id is None:
        google_section = (
            '<div class="card">'
            '<p class="mut">Sign in with Google is not enabled on this gateway yet. '
            'Use your <strong>Username / Email &amp; Password</strong>, your <strong>existing API key (<code>hm-...</code>)</strong>, '
            'or create a new account below.</p>'
            '</div>'
        )
        scripts_html = f'<script nonce="{nonce}">{_AUTH_JS}</script>'
    else:
        g_script = _GOOGLE_JS.replace("%CLIENT_ID%", '"' + client_id + '"')
        google_section = (
            '<section id="signin" class="card">'
            '<h2>Option 3 · Sign in with Google</h2>'
            '<div id="gbtn"></div>'
            '<noscript><p>This page needs JavaScript to show the Google button.</p></noscript>'
            '<p class="mut">We keep only an anonymous hash of your Google account id. Your e-mail address is not stored.</p>'
            '</section>'
        )
        scripts_html = (
            f'<script nonce="{nonce}">{_AUTH_JS}\n{g_script}</script>\n'
            f'<script nonce="{nonce}" src="https://accounts.google.com/gsi/client" async defer></script>'
        )

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Join &amp; Sign In — HiveMind OpenRouter</title><style nonce="{nonce}">{_CSS}</style></head><body><p role="note" id="poc-notice"><strong>&#9888; Proof of concept (PoC)</strong> &mdash; not a production service yet. Until HiveMind is fully secured, send only public, non-sensitive data: no secrets, passwords, API keys, personal information, or private or enterprise data. Prompts are processed by volunteer-run nodes whose operators are not verified. Do not reuse a password from another site; if you reused a password here, change it everywhere you used it.</p><main>
<header>
  <div>
    <h1>🐝 Join or Sign In to HiveMind</h1>
    <p class="mut">Use open coding, scientific &amp; visual MoE models from your editor, or share your GPU and earn HMT points (0.00 EUR/USD).</p>
    <p class="mut" id="starter-grant">{starter_line}</p>
  </div>
  <nav class="nav-links" data-testid="signup-quick-nav">
    <a href="/v3.1">🐝 Unified v3.1 Portal</a>
    <a href="/">⚡ AI Playground</a>
    <a href="/manifesto">📜 Manifesto</a>
    <a href="/benchmarks">📊 Benchmarks</a>
    <a href="/life-as-a-bee">🐝 Life as a Bee</a>
    <a href="/webgpu">🌌 WebGPU</a>
    <a href="/join">How to join</a>
    <a href="/rejoindre">Comment participer</a>
  </nav>
</header>

<div id="signupLiveTelemetry" class="card" style="padding:.65rem .9rem;margin-bottom:.9rem;border-color:#065f46;background:#022c22;color:#a7f3d0;font-family:ui-monospace,monospace;font-size:.82rem;">
  🟢 {telemetry_label}: {total_accts} Registered Accounts ({prosumers_cnt} prosumers, {producers_cnt} producers, {consumers_cnt} consumers, {trainers_cnt} trainers) • {live_models_cnt} Live Models • T4 Free VRAM {t4_free_mib} MiB • OpenTrainer {ot_nodes} Trainer GPU(s) / {ot_rounds} Accepted Round(s)
</div>

<section id="existingBrowserSessionBanner" data-testid="existing-browser-session-banner" class="card ok-banner" hidden>
  <h2>🟢 Active HiveMind Session Detected</h2>
  <p class="mut">Signed in as <strong id="activeSessionHandle">@operator</strong> · Balance <strong id="activeSessionBalance">0.00 HMT</strong></p>
  <p>
    <a href="/v3.1" id="bannerOpenV31Btn" data-testid="banner-open-v31-btn" class="btn-link">🐝 Continue to Unified v3.1 Portal</a>
    <a href="/#playground" class="btn-link secondary">🚀 Open AI Playground</a>
    <button id="signOutSessionBtn" data-testid="signup-signout-btn" type="button" class="secondary">🚪 Sign Out</button>
  </p>
</section>

<section id="result" class="card ok-banner"{result_hidden}>
  <h2>✅ Authenticated HiveMind Account</h2>
  <p id="kind">{res_kind}</p>
  <p>Your active API key (automatically saved in this browser for the Playground and Unified v3.1 Portal):</p>
  <pre><code id="key">{res_key}</code></pre>
  <p>
    <button id="copy" type="button">📋 Copy API Key</button>
    <a href="/v3.1" id="openV31Btn" data-testid="btn-open-v31-authenticated" class="btn-link">🐝 Open Unified v3.1 Portal</a>
    <a href="/#playground" class="btn-link secondary">🚀 Open AI Playground</a>
    <a href="/webgpu" class="btn-link secondary">🌌 Open WebGPU Cockpit</a>
    <button id="rotateKeyBtn" data-testid="signup-rotate-key-btn" type="button" class="secondary">🔄 Rotate API Key</button>
  </p>
  <p class="mut">Account <code id="acct">{res_acct}</code> · Role <code id="role">{res_role}</code> · Balance <code id="bal">{res_bal}</code></p>
  <p>Next, in a terminal or IDE (Continue / Cline / Aider / OpenCode):</p>
  <pre><code>export HIVEMIND_API_KEY=&lt;the key above&gt;
python hivemind_connect.py check
python hivemind_connect.py snippet continue --model coder</code></pre>
</section>

<p id="status" class="{err_cls}" role="status">{err_html}</p>

<div class="grid-auth">
  <section id="loginCard" class="card">
    <h2>🔑 Existing User Login (Se connecter)</h2>
    <p class="mut">Sign in with your previous <strong>username / handle / email</strong> and <strong>password</strong>, or paste an existing <code>hm-...</code> / <code>hm_live_...</code> key.</p>
    <form id="loginForm" method="post" action="/login">
      <label for="loginIdentifier">Username, Handle, Email, or API Key</label>
      <input id="loginIdentifier" name="identifier" type="text" required autocomplete="username" placeholder="e.g. xavier_callens, @Vxtzq, email, or hm-..." />
      <label for="loginPassword">Password (required for password accounts; omit if pasting an <code>hm-...</code> API key)</label>
      <input id="loginPassword" name="password" type="password" autocomplete="current-password" placeholder="Your account password" />
      <p style="margin-top:.65rem"><button id="loginSubmitBtn" type="submit">Sign In &amp; Restore Account · Se connecter</button></p>
    </form>
  </section>

  <section id="registerCard" class="card">
    <h2>🐝 New User Registration (Créer un compte)</h2>
    <p class="mut">Create a new HiveMind account with a username and password to receive your <code>hm-...</code> API key. The starting balance is stated above.</p>
    <form id="registerForm" method="post" action="/signup">
      <label for="regHandle">Username / Handle</label>
      <input id="regHandle" name="handle" type="text" required autocomplete="username" placeholder="e.g. alice_gpu or dev-paris" />
      <label for="regEmail">Email (optional — for account recovery)</label>
      <input id="regEmail" name="email" type="email" autocomplete="email" placeholder="alice@example.org (optional)" />
      <label for="regPassword">Password</label>
      <input id="regPassword" name="password" type="password" autocomplete="new-password" placeholder="Choose a password" />
      <fieldset><legend>What do you want to do?</legend>
        <label class="radio-label"><input type="radio" name="role" value="prosumer"{chk_prosumer}> Both: use models and share my GPU (Prosumer)</label>
        <label class="radio-label"><input type="radio" name="role" value="consumer"{chk_consumer}> Use models in my editor / Playground (Consumer)</label>
        <label class="radio-label"><input type="radio" name="role" value="producer"{chk_producer}> Share my GPU with the swarm (Producer)</label>
        <label class="radio-label"><input type="radio" name="role" value="trainer"{chk_trainer}> Provide my GPU for intense training via OpenTrainer — BOINC, AMCP, Kafka &amp; Aglets (Trainer)</label>
      </fieldset>
      <p><button id="regSubmitBtn" type="submit">Create Account &amp; Get API Key · S'inscrire</button></p>
    </form>
  </section>
</div>

<section id="opentrainerCard" class="card">
  <h2>🔥 OpenTrainer — Intense GPU Co-Training (BOINC + AMCP + Kafka + Signed Aglets, Firewall-Friendly)</h2>
  <p class="mut">Choosing the <strong>Trainer (<code>trainer</code>)</strong> role registers your GPU for intense distributed training across the <strong>RTX 2070 (<code>XAVGAMSRV</code>, 8 GB, Windows 11) ↔ GCP Tesla T4 (<code>socreateai-agora-hermes-node1</code>, 16 GB, Linux)</strong> federation cell: Stage-2 Tiny Granite-MoE NF4 QLoRA, DiLoCo 2D-DCT outer steps, signed AMCP/Kafka Task Agents (IBM Aglet life cycle without mobile code), and BOINC nocturnal workunits.</p>
  <p class="mut"><strong>Folding@home Privacy Guarantee (Zero Private Prompts on Trainers):</strong> Zero private user prompts ever touch a Trainer GPU &mdash; your node only pulls signed 4-bit NF4 training micro-batches from curated public scientific and coding datasets over outbound HTTPS (<code>TCP/443</code>), capped under 5.5 GB VRAM and 68&deg;C with &lt; 1.5s Steam auto-pause.</p>
  <p class="mut"><strong>Easy Network Integration Despite Firewalls:</strong> <code>OpenTrainer</code> uses outbound HTTPS (<code>TCP/443</code>) via <code>/v1/hivemind/opentrainer/heartbeat</code> and <code>/v1/hivemind/opentrainer/submit-delta</code> (zero inbound ports, zero <code>0.0.0.0</code> listeners, works behind residential NAT, CGNAT, and Windows Firewall):</p>
  <pre><code>export HIVEMIND_API_KEY=&lt;your hm-... trainer key&gt;
python scripts/hivemind_opentrainer_agent.py --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app --network-mode outbound-https-relay --check-firewall --once</code></pre>
</section>

{google_section}

{scripts_html}
</main></body></html>"""
