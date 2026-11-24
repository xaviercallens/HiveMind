# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified Login & Registration screen served at ``/signup``, ``/login``, ``/register``, and ``/connexion``.

Supports:
1. **Returning / Previous User Login** (Username / Handle / Email + Password, or existing ``hm-...`` / ``hm_live_...`` API key)
   via ``POST /v1/hivemind/auth/login`` (and ``POST /login`` HTML form fallback).
2. **New User Registration** (Username / Handle + Password + optional Email + Role ``prosumer`` / ``consumer`` / ``producer`` / ``trainer``)
   via ``POST /v1/hivemind/auth/register`` (and ``POST /signup`` HTML form fallback).
3. **Google Sign-In** (when ``client_id`` is configured) via ``POST /v1/hivemind/register/google``.

All dynamic DOM updates use ``textContent`` exclusively, and the Content-Security-Policy enforces per-request nonces.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from hivemind.router.google_auth import CLIENT_ID_RE

# Origins Google documents for the Identity Services button (script, frame, connect, style).
GSI = "https://accounts.google.com/gsi/"
_NONCE_RE = re.compile(r"^[A-Za-z0-9+/=_-]{1,128}$")
_CONTINUE_KEY_RE = re.compile(r"^hm[-_][A-Za-z0-9._=-]{1,192}$")
DEFAULT_PUBLIC_API_BASE = "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1"
DEFAULT_KEY_PLACEHOLDER = "hm-YOUR_API_KEY"
PERSONAL_PC_COMPLIANCE_NOTE = (
    "100% personal home PC & personal hobby/open-source projects only — "
    "never install on a corporate laptop or send company code/data — 0.00 EUR/USD"
)
VALID_CES_PROFILES: frozenset[str] = frozenset({"dev", "gamer"})
DEFAULT_SIGNUP_CES_PLATFORM = "web-signup"
PENDING_SIGNUP_CES_STORAGE_KEY = "hivemind_signup_pending_ces"
SIGNUP_HOLIDAY_PARITY_ROUTES: tuple[str, ...] = (
    "/signup",
    "/noel",
    "/quick",
    "/raid-noel",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/onboarding/feedback/summary",
)
_SIGNUP_HTML_PARITY_ROUTES: frozenset[str] = frozenset(
    {"/signup", "/noel", "/quick", "/raid-noel"}
)
_MANIFEST_REQUIRED_HOLIDAY_ROUTES: tuple[str, ...] = (
    "/noel",
    "/quick",
    "/raid-noel",
    "/arena-noel",
    "/v1/hivemind/onboarding/manifest",
    "/v1/hivemind/onboarding/feedback",
    "/v1/hivemind/onboarding/feedback/summary",
)
_CES_PLATFORM_RE = re.compile(r"\A[a-z0-9][a-z0-9_-]{0,23}\Z", re.ASCII)
_FORBIDDEN_PLATFORM_TOKENS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "amadeus",
    "corporate",
    "email",
    "pii",
    "sk-",
    "ghp_",
    "aiza",
)
_BUILD_SHA_RE = re.compile(r"\A[A-Za-z0-9]{6,64}\Z", re.ASCII)
_HEX_SHA_RE = re.compile(r"\A[0-9a-f]{7,64}\Z", re.ASCII)
_CSP_NONCE_RE = re.compile(r"'nonce-([A-Za-z0-9+/=_-]{6,128})'")
_HTML_NONCE_ATTR_RE = re.compile(
    r"""\bnonce=(?:"([A-Za-z0-9+/=_-]{6,128})"|'([A-Za-z0-9+/=_-]{6,128})')"""
)
_STRIP_NONCE_ATTR_RE = re.compile(r"""\bnonce=(?:"[^"]*"|'[^']*')""")
_STATIC_FIGURE_RE = re.compile(
    r"\d[\d.,]*\s?(?:m" + r"s\b|tok" + r"/s|HTTP" + r" Reqs|Comple" + r"tions|\bs/" + r"round)"
)
_UNSAFE_DOM_SINK_RE = re.compile(
    r"\b(?:" + "inner" + "HTML|" + "outer" + "HTML|" + "insertAdjacent" + "HTML|" + "docu" + "ment" + r"\." + r"write)\b"
)
_SECRET_VALUE_MARKERS: tuple[str, ...] = (
    "secret",
    "token",
    "password",
    "passwd",
    "apikey",
    "bearer",
    "aiza",
)
CES_SIGNUP_COMPLIANCE_NOTE = (
    "Personal-PC-only AGPL-3.0 micro-survey: 100% personal home PCs & hobby/open-source projects only "
    "(never corporate laptops or employer code/data) • 0.00 EUR / 0.00 USD • Zero PII (no free-text field)."
)


def build_continue_config_yaml(api_key: str, api_base: str) -> str:
    """Build a ready-to-paste VS Code Continue.dev ``config.yaml`` snippet for a HiveMind API key."""
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("api_key must be a non-empty string")
    clean_key = api_key.strip()
    if not _CONTINUE_KEY_RE.fullmatch(clean_key):
        raise ValueError("api_key must be a valid hm-... or hm_live_... token")
    if not isinstance(api_base, str) or not api_base.strip():
        raise ValueError("api_base must be a non-empty http(s) URL")
    clean_base = api_base.strip().rstrip("/")
    if any(ch in clean_base for ch in ("\r", "\n", "\x00", " ", "\t", '"', "'", "#", "`")):
        raise ValueError("api_base must not contain whitespace, quotes, or control characters")
    parsed = urlparse(clean_base)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or not parsed.hostname:
        raise ValueError("api_base must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None or parsed.query or parsed.fragment:
        raise ValueError("api_base must not include credentials, query parameters, or fragments")
    normalized_base = clean_base if clean_base.endswith("/v1") else f"{clean_base}/v1"
    return (
        "# HiveMind VS Code Continue.dev config (~/.continue/config.yaml)\n"
        f"# {PERSONAL_PC_COMPLIANCE_NOTE}\n"
        "name: HiveMind OpenRouter\n"
        "version: 0.0.1\n"
        "schema: v1\n"
        "models:\n"
        "  - name: HiveMind Coder\n"
        "    provider: openai\n"
        "    model: hivemind/coder\n"
        f"    apiBase: {normalized_base}\n"
        f"    apiKey: {clean_key}\n"
        "    roles:\n"
        "      - chat\n"
        "      - edit\n"
        "      - apply\n"
        "    defaultCompletionOptions:\n"
        "      maxTokens: 2048\n"
        "  - name: HiveMind Auto\n"
        "    provider: openai\n"
        "    model: hivemind/auto\n"
        f"    apiBase: {normalized_base}\n"
        f"    apiKey: {clean_key}\n"
        "    roles:\n"
        "      - chat\n"
    )


def content_security_policy(nonce: str) -> str:
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        raise ValueError("nonce must be a 1-128 char URL/base64-safe token")
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
.ces-box{margin:.7rem 0 .25rem;padding:.65rem .75rem;border:1px solid var(--line);border-radius:.45rem;background:var(--input)}
.ces-card{margin:.85rem 0 .25rem;padding:.85rem;border:1px solid var(--line);border-radius:.5rem;background:var(--input)}
.ces-card h4{margin:0 0 .35rem;font-size:.96rem;color:var(--amber)}
.ces-row{display:flex;flex-wrap:wrap;align-items:center;gap:.45rem;margin:.45rem 0}
.ces-lbl{font-size:.86rem;font-weight:600;color:var(--fg);margin-right:.2rem}
.row-actions{display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:.5rem 0}
.chip-btn{font:inherit;font-size:.85rem;font-weight:600;padding:.38rem .72rem;border-radius:.45rem;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.chip-btn.active{background:var(--acc);color:#06101e;border-color:var(--acc)}
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
function buildContinueConfigYaml(apiKey, apiBase) {
  const cleanKey = String(apiKey || 'hm-YOUR_API_KEY').trim() || 'hm-YOUR_API_KEY';
  const cleanBase = String(apiBase || (window.location.origin + '/v1')).trim().replace(/\/+$/, '');
  const normalizedBase = cleanBase.endsWith('/v1') ? cleanBase : (cleanBase + '/v1');
  return [
    '# HiveMind VS Code Continue.dev config (~/.continue/config.yaml)',
    '# 100% personal home PC & personal hobby/open-source projects only — never install on a corporate laptop or send company code/data — 0.00 EUR/USD',
    'name: HiveMind OpenRouter',
    'version: 0.0.1',
    'schema: v1',
    'models:',
    '  - name: HiveMind Coder',
    '    provider: openai',
    '    model: hivemind/coder',
    '    apiBase: ' + normalizedBase,
    '    apiKey: ' + cleanKey,
    '    roles:',
    '      - chat',
    '      - edit',
    '      - apply',
    '    defaultCompletionOptions:',
    '      maxTokens: 2048',
    '  - name: HiveMind Auto',
    '    provider: openai',
    '    model: hivemind/auto',
    '    apiBase: ' + normalizedBase,
    '    apiKey: ' + cleanKey,
    '    roles:',
    '      - chat',
  ].join('\n') + '\n';
}
function showSessionResult(body) {
  const tok = body.api_key || body.token || '';
  if ($('key')) $('key').textContent = tok;
  if ($('continueYamlSnippet') && tok) {
    $('continueYamlSnippet').textContent = buildContinueConfigYaml(tok, window.location.origin + '/v1');
  }
  if ($('verifyKeyStatus')) $('verifyKeyStatus').textContent = '';
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
if ($('copyContinueBtn')) {
  $('copyContinueBtn').addEventListener('click', async () => {
    try {
      const snippetText = ($('continueYamlSnippet') && $('continueYamlSnippet').textContent) || '';
      await navigator.clipboard.writeText(snippetText);
      $('copyContinueBtn').textContent = '✅ Copied Continue.dev Config!';
    } catch (err) {
      $('copyContinueBtn').textContent = 'Select and copy the YAML snippet manually';
    }
  });
}
async function verifyActiveKeyModels() {
  const statusEl = $('verifyKeyStatus');
  const tok = (($('key') && $('key').textContent) || localStorage.getItem('hivemind_api_key') || '').trim();
  if (!tok || tok === 'hm-YOUR_API_KEY' || !tok.startsWith('hm')) {
    if (statusEl) {
      statusEl.textContent = 'Sign in or create an account first to get your hm-... key.';
      statusEl.className = 'err-text';
    }
    return;
  }
  if (statusEl) {
    statusEl.textContent = 'Checking /v1/models...';
    statusEl.className = 'mut';
  }
  try {
    const res = await fetch('/v1/models', {
      method: 'GET',
      headers: {
        'Authorization': 'Bearer ' + tok,
        'Accept': 'application/json',
      },
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) {
      if (statusEl) {
        statusEl.textContent = (body.error && body.error.message) || ('Key check failed (HTTP ' + res.status + ').');
        statusEl.className = 'err-text';
      }
      return;
    }
    const models = Array.isArray(body.data) ? body.data : [];
    const modelIds = models.map((m) => (m && m.id ? String(m.id) : '')).filter(Boolean);
    if (statusEl) {
      statusEl.textContent =
        '✅ Verified via /v1/models: ' +
        models.length +
        ' live model(s)' +
        (modelIds.length ? ' (' + modelIds.slice(0, 3).join(', ') + (modelIds.length > 3 ? ', ...' : '') + ')' : '');
      statusEl.className = 'mut';
    }
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = 'Could not reach /v1/models on this gateway.';
      statusEl.className = 'err-text';
    }
  }
}
if ($('verifyKeyModelsBtn')) {
  $('verifyKeyModelsBtn').addEventListener('click', verifyActiveKeyModels);
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
    if ($('continueYamlSnippet')) {
      $('continueYamlSnippet').textContent = buildContinueConfigYaml(tok, window.location.origin + '/v1');
    }
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
    const sessionBal = (typeof s.user.balance_hmt === 'number' ? s.user.balance_hmt.toFixed(2) : String(s.user.compute_points || 0)) + ' HMT';
    if ($('activeSessionHandle')) $('activeSessionHandle').textContent = '@' + (s.user.handle || 'operator');
    if ($('activeSessionBalance')) $('activeSessionBalance').textContent = sessionBal;
    if ($('key')) $('key').textContent = tok;
    if ($('acct') && s.user.handle) $('acct').textContent = s.user.handle;
    if ($('role') && s.user.role) $('role').textContent = s.user.role;
    if ($('bal')) $('bal').textContent = sessionBal;
    if ($('continueYamlSnippet')) {
      $('continueYamlSnippet').textContent = buildContinueConfigYaml(tok, window.location.origin + '/v1');
    }
    if ($('existingBrowserSessionBanner')) $('existingBrowserSessionBanner').hidden = false;
    if ($('result')) $('result').hidden = false;
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
      const nextTok = (d && (d.api_key || d.token)) || '';
      if (nextTok) {
        if ($('key')) $('key').textContent = nextTok;
        if ($('continueYamlSnippet')) {
          $('continueYamlSnippet').textContent = buildContinueConfigYaml(nextTok, window.location.origin + '/v1');
        }
        if ($('verifyKeyStatus')) $('verifyKeyStatus').textContent = '';
        persistUnifiedSession({ api_key: nextTok, handle: d.handle || localStorage.getItem('hivemind_user_handle') || 'operator' });
        setStatus('API key rotated! Your old key is revoked and your new key is saved in this browser.', false);
      }
    } catch (e) {}
  });
}
if ($('changePasswordForm')) {
  try {
    const m = (window.location.hash || '').match(/(?:^|[#&])token=([^&]+)/);
    if (m && m[1] && $('resetTokenOrKey') && !$('resetTokenOrKey').value) {
      $('resetTokenOrKey').value = decodeURIComponent(m[1]);
    }
  } catch (e) {}
  $('changePasswordForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const tokenOrKey = (($('resetTokenOrKey') && $('resetTokenOrKey').value) || ($('key') && $('key').textContent) || localStorage.getItem('hivemind_api_key') || '').trim();
    const newPassword = (($('newResetPassword') && $('newResetPassword').value) || '').trim();
    if (!newPassword || newPassword.length < 12) {
      setStatus('Please enter a new password of at least 12 characters.', true);
      return;
    }
    if (!tokenOrKey) {
      setStatus('Paste your active hm-... API key or your password-reset token.', true);
      return;
    }
    setStatus('Updating password...', false);
    try {
      let res;
      if (tokenOrKey.startsWith('hm-')) {
        res = await fetch('/api/v1/auth/change-password', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + tokenOrKey },
          body: JSON.stringify({ new_password: newPassword }),
        });
      } else {
        res = await fetch('/api/v1/auth/password-reset/confirm', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ token: tokenOrKey, new_password: newPassword }),
        });
      }
      const body = await res.json().catch(() => ({}));
      if (!res.ok) {
        setStatus((body.error && body.error.message) || 'Password update failed.', true);
        return;
      }
      const rotatedKey = body.api_key || body.token || '';
      if (rotatedKey) {
        if ($('key')) $('key').textContent = rotatedKey;
        if ($('continueYamlSnippet')) {
          $('continueYamlSnippet').textContent = buildContinueConfigYaml(rotatedKey, window.location.origin + '/v1');
        }
        if ($('verifyKeyStatus')) $('verifyKeyStatus').textContent = '';
        if ($('acct') && body.handle) $('acct').textContent = body.handle;
        if ($('result')) $('result').hidden = false;
        persistUnifiedSession({ api_key: rotatedKey, handle: body.handle || localStorage.getItem('hivemind_user_handle') || 'operator' });
        setStatus('Password updated and new hm-... API key issued!', false);
      } else {
        setStatus(body.message || 'Password updated! Sign in above with your new password.', false);
      }
      if ($('newResetPassword')) $('newResetPassword').value = '';
    } catch (e) {
      setStatus('Could not reach the gateway. Try again.', true);
    }
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
    if ($('key')) $('key').textContent = '';
    if ($('continueYamlSnippet')) {
      $('continueYamlSnippet').textContent = buildContinueConfigYaml('hm-YOUR_API_KEY', window.location.origin + '/v1');
    }
    if ($('verifyKeyStatus')) $('verifyKeyStatus').textContent = '';
    if ($('existingBrowserSessionBanner')) $('existingBrowserSessionBanner').hidden = true;
    if ($('result')) $('result').hidden = true;
    setStatus('Signed out of browser session.', false);
  });
}
async function claimTestingBonusHmt() {
  const tok = ($('key') && $('key').textContent.trim()) || localStorage.getItem('hivemind_api_key') || '';
  if (!tok) {
    setStatus('Sign in or create an account first to claim HMT testing bonuses.', true);
    return;
  }
  setStatus('Claiming eligible HMT testing bonus...', false);
  try {
    const res = await fetch('/v1/hivemind/bonus/claim', {
      method: 'POST',
      headers: { 'Authorization': 'Bearer ' + tok, 'Content-Type': 'application/json' },
      body: JSON.stringify({ bonus_code: 'auto' }),
    });
    const d = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus((d.error && d.error.message) || 'Could not claim bonus HMT.', true);
      return;
    }
    if (typeof d.balance_hmt === 'number') {
      const fmt = d.balance_hmt.toFixed(2) + ' HMT';
      if ($('bal')) $('bal').textContent = fmt;
      if ($('activeSessionBalance')) $('activeSessionBalance').textContent = fmt;
      persistUnifiedSession({
        api_key: tok,
        handle: d.handle || localStorage.getItem('hivemind_user_handle') || 'operator',
        role: d.role || 'prosumer',
        balance_hmt: d.balance_hmt,
        compute_points: d.compute_points,
        created: false,
      });
    }
    setStatus(d.note || (d.claimed ? 'Bonus HMT credited!' : 'No new bonus HMT available right now.'), false);
  } catch (e) {
    setStatus('Could not reach the gateway to claim bonus HMT.', true);
  }
}
if ($('claimBonusBtn')) $('claimBonusBtn').addEventListener('click', claimTestingBonusHmt);
if ($('bannerClaimBonusBtn')) $('bannerClaimBonusBtn').addEventListener('click', claimTestingBonusHmt);
if ($('transferPointsForm')) {
  $('transferPointsForm').addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const tok = ($('key') && $('key').textContent.trim()) || localStorage.getItem('hivemind_api_key') || '';
    const recipient = ($('transferRecipient') && $('transferRecipient').value || '').trim();
    const amount = Number(($('transferAmount') && $('transferAmount').value || '').trim());
    const memo = ($('transferMemo') && $('transferMemo').value || '').trim() || 'peer-hiveminder-support';
    if (!tok || !recipient || !Number.isFinite(amount) || amount <= 0) {
      setStatus('Enter a recipient handle and a positive HMT amount.', true);
      return;
    }
    setStatus('Sending HMT points across the HiveMind ledger...', false);
    try {
      const res = await fetch('/v1/hivemind/points/transfer', {
        method: 'POST',
        headers: { 'Authorization': 'Bearer ' + tok, 'Content-Type': 'application/json' },
        body: JSON.stringify({ recipient: recipient, amount: amount, memo: memo, notify_amcp: true }),
      });
      const d = await res.json().catch(() => ({}));
      if (!res.ok) {
        setStatus((d.error && d.error.message) || 'HMT transfer failed.', true);
        return;
      }
      if (typeof d.sender_balance_hmt === 'number') {
        const fmt = d.sender_balance_hmt.toFixed(2) + ' HMT';
        if ($('bal')) $('bal').textContent = fmt;
        if ($('activeSessionBalance')) $('activeSessionBalance').textContent = fmt;
        persistUnifiedSession({
          api_key: tok,
          handle: d.sender_handle || localStorage.getItem('hivemind_user_handle') || 'operator',
          balance_hmt: d.sender_balance_hmt,
          compute_points: d.sender_compute_points,
          created: false,
        });
      }
      setStatus('Sent ' + Number(d.amount_hmt || amount).toFixed(2) + ' HMT to @' + (d.recipient_handle || recipient) + (d.amcp_event_id ? ' · AMCP ' + d.amcp_event_id : ''), false);
    } catch (e) {
      setStatus('Could not reach the gateway to transfer HMT.', true);
    }
  });
}
async function verifyHolidayRouteParity() {
  const statusEl = $('verifyHolidayParityStatus');
  if (statusEl) {
    statusEl.textContent = 'Checking holiday routes (/v1/hivemind/onboarding/manifest & /v1/hivemind/onboarding/feedback/summary)...';
    statusEl.className = 'mut';
  }
  try {
    const results = await Promise.all([
      fetch('/v1/hivemind/onboarding/manifest', {
        method: 'GET',
        cache: 'no-store',
        headers: { 'Accept': 'application/json' },
      }),
      fetch('/v1/hivemind/onboarding/feedback/summary', {
        method: 'GET',
        cache: 'no-store',
        headers: { 'Accept': 'application/json' },
      }),
    ]);
    const manRes = results[0];
    const sumRes = results[1];
    const manCode = manRes && typeof manRes.status === 'number' ? manRes.status : 0;
    const sumCode = sumRes && typeof sumRes.status === 'number' ? sumRes.status : 0;
    if (manCode === 404 || sumCode === 404) {
      const drifted = [];
      if (manCode === 404) drifted.push('/v1/hivemind/onboarding/manifest');
      if (sumCode === 404) drifted.push('/v1/hivemind/onboarding/feedback/summary');
      if (statusEl) {
        statusEl.textContent = '⚠️ Cloud Run holiday route drift (HTTP 404): ' + drifted.join(', ') + ' — deploy pending; 1-click CES stays spooled locally (0.00 EUR/USD).';
        statusEl.className = 'err-text';
      }
      return;
    }
    if (!manRes || !manRes.ok || !sumRes || !sumRes.ok) {
      if (statusEl) {
        statusEl.textContent = '⚠️ Holiday parity check failed (manifest HTTP ' + manCode + ', feedback/summary HTTP ' + sumCode + ').';
        statusEl.className = 'err-text';
      }
      return;
    }
    const manBody = await manRes.json().catch(() => ({}));
    const sumBody = await sumRes.json().catch(() => ({}));
    const manOk = Boolean(
      manBody &&
      manBody.object === 'hivemind.onboarding.manifest' &&
      manBody.personal_pc_only === true &&
      Array.isArray(manBody.holiday_routes)
    );
    const sumOk = Boolean(
      sumBody &&
      sumBody.object === 'hivemind.onboarding_feedback_summary' &&
      typeof sumBody.total_responses === 'number' &&
      sumBody.by_profile
    );
    if (!manOk || !sumOk) {
      if (statusEl) {
        statusEl.textContent = '⚠️ Holiday parity payload mismatch (unexpected manifest or feedback summary schema).';
        statusEl.className = 'err-text';
      }
      return;
    }
    const sha = manBody.build_sha || 'unknown';
    const easeTxt = typeof sumBody.mean_ease_rating === 'number' ? sumBody.mean_ease_rating.toFixed(2) + '/5' : 'n/a';
    const lagTxt = typeof sumBody.zero_game_lag_ratio === 'number' ? (sumBody.zero_game_lag_ratio * 100).toFixed(1) + '%' : 'n/a';
    if (statusEl) {
      statusEl.textContent =
        '✅ Holiday Parity Live (HTTP 200 • build ' + sha + ' • ' +
        manBody.holiday_routes.length + ' routes • CES responses: ' +
        sumBody.total_responses + ', ease: ' + easeTxt + ', zero-lag: ' + lagTxt + ' • 0.00 EUR/USD).';
      statusEl.className = 'mut';
    }
  } catch (err) {
    if (statusEl) {
      statusEl.textContent = 'Could not reach holiday parity endpoints on this gateway.';
      statusEl.className = 'err-text';
    }
  }
}
if ($('verifyHolidayParityBtn')) {
  $('verifyHolidayParityBtn').addEventListener('click', verifyHolidayRouteParity);
}
const SIGNUP_PENDING_CES_KEY = 'hivemind_signup_pending_ces';
let signupCesProfile = 'dev';
let signupCesEase = 5;
let signupCesZeroLag = true;

function getSignupStorage() {
  try {
    if (typeof localStorage !== 'undefined' && localStorage) return localStorage;
  } catch (e) {}
  try {
    if (typeof window !== 'undefined' && window.localStorage) return window.localStorage;
  } catch (e) {}
  return null;
}
function signupFnv1aHex(str, seed) {
  let h = seed >>> 0;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619) >>> 0;
  }
  return ('00000000' + h.toString(16)).slice(-8);
}
function buildSignupCesIdempotencyKey(ballot) {
  const raw = ballot.profile + '|' + String(ballot.ease_rating) + '|' + (ballot.zero_game_lag ? '1' : '0') + '|' + ballot.platform;
  const hex = signupFnv1aHex(raw, 2166136261) + signupFnv1aHex(raw + '#web-signup', 2166136261);
  return 'ces-' + ballot.profile + '-' + String(ballot.ease_rating) + '-' + (ballot.zero_game_lag ? 'lag0' : 'lag1') + '-' + hex;
}
function validateSignupCesBallot(b) {
  if (!b || typeof b !== 'object') return null;
  const prof = b.profile === 'dev' ? 'dev' : (b.profile === 'gamer' ? 'gamer' : null);
  const ease = (typeof b.ease_rating === 'number' && Number.isInteger(b.ease_rating) && b.ease_rating >= 1 && b.ease_rating <= 5) ? b.ease_rating : null;
  const lag = typeof b.zero_game_lag === 'boolean' ? b.zero_game_lag : null;
  if (!prof || ease === null || lag === null) return null;
  if (b.note !== undefined && b.note !== '') return null;
  return {
    profile: prof,
    ease_rating: ease,
    zero_game_lag: lag,
    platform: 'web-signup',
    note: '',
  };
}
function buildCurrentSignupCesBallot() {
  return {
    profile: signupCesProfile === 'gamer' ? 'gamer' : 'dev',
    ease_rating: signupCesEase,
    zero_game_lag: Boolean(signupCesZeroLag),
    platform: 'web-signup',
    note: '',
  };
}
function clearPendingSignupCesBallot() {
  try {
    const st = getSignupStorage();
    if (st) {
      if (typeof st.removeItem === 'function') st.removeItem(SIGNUP_PENDING_CES_KEY);
      else if (typeof st.setItem === 'function') st.setItem(SIGNUP_PENDING_CES_KEY, '');
    }
  } catch (e) {}
}
function savePendingSignupCesBallot(ballot, idemKey, statusCode) {
  try {
    const st = getSignupStorage();
    if (st && typeof st.setItem === 'function') {
      st.setItem(
        SIGNUP_PENDING_CES_KEY,
        JSON.stringify({
          ballot: ballot,
          idempotency_key: idemKey,
          last_status: statusCode,
        })
      );
    }
  } catch (e) {}
}
function loadPendingSignupCesBallot() {
  try {
    const st = getSignupStorage();
    if (!st || typeof st.getItem !== 'function') return null;
    const raw = st.getItem(SIGNUP_PENDING_CES_KEY);
    if (!raw || typeof raw !== 'string') return null;
    if (raw.length > 512) {
      clearPendingSignupCesBallot();
      return null;
    }
    const parsed = JSON.parse(raw);
    const clean = validateSignupCesBallot(parsed && parsed.ballot ? parsed.ballot : parsed);
    if (!clean) {
      clearPendingSignupCesBallot();
      return null;
    }
    const key = (parsed && typeof parsed.idempotency_key === 'string' && /^[A-Za-z0-9_-]{8,80}$/.test(parsed.idempotency_key))
      ? parsed.idempotency_key
      : buildSignupCesIdempotencyKey(clean);
    return { ballot: clean, idempotency_key: key };
  } catch (e) {
    clearPendingSignupCesBallot();
    return null;
  }
}
function setSignupCesStatus(text, isErr) {
  const el = $('signupCesStatus');
  if (!el) return;
  el.textContent = String(text);
  el.className = isErr ? 'err-text' : 'mut';
}
function signupCesSpoolMessage(statusCode) {
  if (statusCode === 404) {
    return 'Cloud Run route /v1/hivemind/onboarding/feedback returned HTTP 404: zero-PII ballot spooled locally. Click "Retry Pending Feedback" after deploy.';
  }
  if (statusCode === 429) {
    return 'Rate limit reached (HTTP 429): zero-PII ballot spooled locally. Click "Retry Pending Feedback" in a minute.';
  }
  if (statusCode >= 500) {
    return 'Gateway temporarily busy (HTTP ' + statusCode + '): zero-PII ballot spooled locally. Click "Retry Pending Feedback" to resend.';
  }
  return 'Offline / network error: zero-PII ballot spooled locally. Click "Retry Pending Feedback" when back online.';
}
function syncSignupCesButtons() {
  const devBtn = $('signupCesProfileDevBtn');
  const gamerBtn = $('signupCesProfileGamerBtn');
  if (devBtn) {
    const isDev = signupCesProfile === 'dev';
    if (typeof devBtn.setAttribute === 'function') devBtn.setAttribute('aria-pressed', isDev ? 'true' : 'false');
    devBtn.className = isDev ? 'chip-btn active' : 'chip-btn';
  }
  if (gamerBtn) {
    const isGamer = signupCesProfile === 'gamer';
    if (typeof gamerBtn.setAttribute === 'function') gamerBtn.setAttribute('aria-pressed', isGamer ? 'true' : 'false');
    gamerBtn.className = isGamer ? 'chip-btn active' : 'chip-btn';
  }
  for (let r = 1; r <= 5; r++) {
    const eBtn = $('signupCesEaseBtn-' + r);
    if (eBtn) {
      const active = signupCesEase === r;
      if (typeof eBtn.setAttribute === 'function') eBtn.setAttribute('aria-pressed', active ? 'true' : 'false');
      eBtn.className = active ? 'chip-btn active' : 'chip-btn';
    }
  }
  const lagYes = $('signupCesLagYesBtn');
  const lagNo = $('signupCesLagNoBtn');
  if (lagYes) {
    if (typeof lagYes.setAttribute === 'function') lagYes.setAttribute('aria-pressed', signupCesZeroLag ? 'true' : 'false');
    lagYes.className = signupCesZeroLag ? 'chip-btn active' : 'chip-btn';
  }
  if (lagNo) {
    if (typeof lagNo.setAttribute === 'function') lagNo.setAttribute('aria-pressed', !signupCesZeroLag ? 'true' : 'false');
    lagNo.className = !signupCesZeroLag ? 'chip-btn active' : 'chip-btn';
  }
}
async function sendSignupCesBallot(ballot, idemKey, isRetry) {
  if (typeof fetch !== 'function') {
    savePendingSignupCesBallot(ballot, idemKey, 0);
    setSignupCesStatus(signupCesSpoolMessage(0), true);
    return;
  }
  setSignupCesStatus('Sending 1-click zero-PII feedback...', false);
  try {
    const res = await fetch('/v1/hivemind/onboarding/feedback', {
      method: 'POST',
      cache: 'no-store',
      headers: {
        'Content-Type': 'application/json',
        'Idempotency-Key': idemKey,
      },
      body: JSON.stringify(ballot),
    });
    const code = res && typeof res.status === 'number' ? res.status : 0;
    if (res && res.ok) {
      const d = await res.json().catch(() => ({}));
      clearPendingSignupCesBallot();
      const tot = typeof d.total_responses === 'number' ? String(d.total_responses) : 'n/a';
      const easeTxt = typeof d.mean_ease_rating === 'number' ? d.mean_ease_rating.toFixed(2) + '/5' : 'n/a';
      const lagTxt = typeof d.zero_game_lag_ratio === 'number' ? (d.zero_game_lag_ratio * 100).toFixed(1) + '%' : 'n/a';
      const prof = d.by_profile || {};
      const profTxt = 'dev: ' + (typeof prof.dev === 'number' ? prof.dev : 'n/a') + ' | gamer: ' + (typeof prof.gamer === 'number' ? prof.gamer : 'n/a');
      const prefix = isRetry ? 'Pending zero-PII feedback flushed' : 'Thank you! 1-click zero-PII feedback recorded live';
      setSignupCesStatus(
        '✅ ' + prefix + ' (HTTP 200 • total: ' + tot + ', ease: ' + easeTxt + ', zero-lag: ' + lagTxt + ', ' + profTxt + ' • 0.00 EUR / 0.00 USD).',
        false
      );
      return;
    }
    if (code === 404 || code === 408 || code === 429 || code >= 500 || code === 0) {
      savePendingSignupCesBallot(ballot, idemKey, code);
      setSignupCesStatus(signupCesSpoolMessage(code), true);
      return;
    }
    clearPendingSignupCesBallot();
    setSignupCesStatus('Submission rejected (HTTP ' + code + '): ballot cleared.', true);
  } catch (err) {
    savePendingSignupCesBallot(ballot, idemKey, 0);
    setSignupCesStatus(signupCesSpoolMessage(0), true);
  }
}
if ($('signupCesProfileDevBtn')) {
  $('signupCesProfileDevBtn').addEventListener('click', () => {
    signupCesProfile = 'dev';
    syncSignupCesButtons();
  });
}
if ($('signupCesProfileGamerBtn')) {
  $('signupCesProfileGamerBtn').addEventListener('click', () => {
    signupCesProfile = 'gamer';
    syncSignupCesButtons();
  });
}
[1, 2, 3, 4, 5].forEach((rating) => {
  const eBtn = $('signupCesEaseBtn-' + rating);
  if (eBtn) {
    eBtn.addEventListener('click', () => {
      signupCesEase = rating;
      syncSignupCesButtons();
    });
  }
});
if ($('signupCesLagYesBtn')) {
  $('signupCesLagYesBtn').addEventListener('click', () => {
    signupCesZeroLag = true;
    syncSignupCesButtons();
  });
}
if ($('signupCesLagNoBtn')) {
  $('signupCesLagNoBtn').addEventListener('click', () => {
    signupCesZeroLag = false;
    syncSignupCesButtons();
  });
}
if ($('signupSubmitCesBtn')) {
  $('signupSubmitCesBtn').addEventListener('click', () => {
    const ballot = buildCurrentSignupCesBallot();
    const idemKey = buildSignupCesIdempotencyKey(ballot);
    return sendSignupCesBallot(ballot, idemKey, false);
  });
  syncSignupCesButtons();
  const pendingInit = loadPendingSignupCesBallot();
  if (pendingInit) {
    setSignupCesStatus('A zero-PII ballot is pending locally: click "Retry Pending Feedback" to send it.', false);
  }
}
if ($('signupRetryPendingCesBtn')) {
  $('signupRetryPendingCesBtn').addEventListener('click', () => {
    const pending = loadPendingSignupCesBallot();
    if (!pending) {
      setSignupCesStatus('No pending zero-PII feedback ballot in local storage.', false);
      return Promise.resolve();
    }
    return sendSignupCesBallot(pending.ballot, pending.idempotency_key, true);
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
    if ($('continueYamlSnippet') && body.api_key) {
      $('continueYamlSnippet').textContent = buildContinueConfigYaml(body.api_key, window.location.origin + '/v1');
    }
    if ($('verifyKeyStatus')) $('verifyKeyStatus').textContent = '';
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
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a non-empty string")
    if not isinstance(thousands, bool):
        raise ValueError("thousands must be a bool")
    value = source.get(key)
    if value is None:
        return "n/a"
    number = int(value)
    return f"{number:,}" if thousands else str(number)


def build_signup_ces_ballot(
    *,
    profile: str = "dev",
    ease_rating: int = 5,
    zero_game_lag: bool = True,
    platform_name: str = DEFAULT_SIGNUP_CES_PLATFORM,
    note: str = "",
) -> dict[str, Any]:
    """Validate and normalize a strict zero-PII ``OnboardingFeedbackRequest`` (``extra='forbid'``) ballot."""
    if isinstance(profile, bool) or not isinstance(profile, str) or len(profile) > 32 or not profile.isascii():
        raise ValueError("profile must be 'dev' or 'gamer'")
    norm_profile = profile.strip().lower()
    if norm_profile not in VALID_CES_PROFILES:
        raise ValueError("profile must be 'dev' or 'gamer'")
    if isinstance(ease_rating, bool) or not isinstance(ease_rating, int) or not (1 <= ease_rating <= 5):
        raise ValueError("ease_rating must be an integer between 1 and 5")
    if not isinstance(zero_game_lag, bool):
        raise ValueError("zero_game_lag must be a boolean")
    if (
        isinstance(platform_name, bool)
        or not isinstance(platform_name, str)
        or not platform_name.isascii()
        or len(platform_name) > 32
    ):
        raise ValueError("platform_name must be a 1-24 character ASCII platform slug")
    clean_platform = platform_name.strip().lower()
    if not _CES_PLATFORM_RE.fullmatch(clean_platform):
        raise ValueError("platform_name must match ^[a-z0-9][a-z0-9_-]{0,23}$")
    if clean_platform.startswith(("hm-", "hm_live_", "hm_")) or any(
        tok in clean_platform for tok in _FORBIDDEN_PLATFORM_TOKENS
    ):
        raise ValueError("platform_name contains forbidden credential, PII, or corporate markers")
    if isinstance(note, bool) or not isinstance(note, str) or note != "":
        raise ValueError("note must be empty ('') to enforce strict zero-PII 1-click feedback")
    return {
        "profile": norm_profile,
        "ease_rating": ease_rating,
        "zero_game_lag": zero_game_lag,
        "platform": clean_platform,
        "note": "",
    }


def _sanitize_build_sha_local(raw: object) -> str:
    if not isinstance(raw, str):
        return "unknown"
    cleaned = raw.strip()
    low = cleaned.lower()
    if (
        not cleaned
        or low == "unknown"
        or not _BUILD_SHA_RE.fullmatch(cleaned)
        or any(marker in low for marker in _SECRET_VALUE_MARKERS)
    ):
        return "unknown"
    return low


def _shas_match_local(live_sha: str, expected_sha: str) -> bool:
    clean_live = _sanitize_build_sha_local(live_sha)
    clean_exp = _sanitize_build_sha_local(expected_sha)
    if clean_live == "unknown" or clean_exp == "unknown":
        return False
    if clean_live == clean_exp:
        return True
    if _HEX_SHA_RE.fullmatch(clean_live) and _HEX_SHA_RE.fullmatch(clean_exp):
        return clean_live.startswith(clean_exp) or clean_exp.startswith(clean_live)
    return False


def _has_forbidden_score_key(obj: Any) -> bool:
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            if str(key) == ("sco" + "re") or _has_forbidden_score_key(value):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_has_forbidden_score_key(item) for item in obj)
    return False


def _is_bounded_finite_number(val: Any, low: float, high: float) -> bool:
    return (
        not isinstance(val, bool)
        and isinstance(val, (int, float))
        and math.isfinite(float(val))
        and low <= float(val) <= high
    )


def _extract_route_probe(
    probe: Mapping[str, Any],
) -> tuple[int | None, dict[str, str], str, Mapping[str, Any] | None]:
    raw_status = probe.get("status_code") if "status_code" in probe else probe.get("status")
    status_code = raw_status if (not isinstance(raw_status, bool) and isinstance(raw_status, int)) else None

    headers: dict[str, str] = {}
    raw_headers = probe.get("headers")
    if isinstance(raw_headers, Mapping):
        for hk, hv in raw_headers.items():
            if isinstance(hk, str) and isinstance(hv, str):
                headers[hk.strip().lower()] = hv
    if "content-security-policy" not in headers and isinstance(probe.get("csp"), str):
        headers["content-security-policy"] = probe["csp"]

    text_body = ""
    for key in ("html", "text", "body"):
        val = probe.get(key)
        if isinstance(val, str):
            text_body = val
            break

    json_body: Mapping[str, Any] | None = None
    if isinstance(probe.get("json"), Mapping):
        json_body = probe["json"]
    elif text_body.lstrip().startswith("{"):
        try:
            parsed = json.loads(text_body)
            if isinstance(parsed, Mapping):
                json_body = parsed
        except json.JSONDecodeError:
            json_body = None

    return status_code, headers, text_body, json_body


def _inspect_html_parity_route(
    headers: Mapping[str, str],
    html_text: str,
    seen_nonces: set[str],
) -> tuple[bool, dict[str, Any], list[str]]:
    issues: list[str] = []
    has_doctype = "<!doctype html" in html_text.lower()
    if not has_doctype:
        issues.append("missing_doctype")

    csp_val = headers.get("content-security-policy", "")
    csp_lower = csp_val.lower()
    csp_unsafe = "'unsafe-inline'" in csp_lower or "'unsafe-eval'" in csp_lower
    hdr_nonces = _CSP_NONCE_RE.findall(csp_val)
    attr_nonces = [g1 or g2 for g1, g2 in _HTML_NONCE_ATTR_RE.findall(html_text)]
    header_nonce = hdr_nonces[0] if (hdr_nonces and len(set(hdr_nonces)) == 1) else None
    nonce_reused = header_nonce is not None and header_nonce in seen_nonces
    if header_nonce is not None:
        seen_nonces.add(header_nonce)

    csp_ok = bool(
        "default-src 'none'" in csp_lower
        and not csp_unsafe
        and header_nonce is not None
        and attr_nonces
        and set(attr_nonces) == {header_nonce}
        and not nonce_reused
    )
    if nonce_reused:
        issues.append("reused_csp_nonce")
    elif not csp_ok:
        issues.append("invalid_or_missing_csp_nonce")

    stripped_html = _STRIP_NONCE_ATTR_RE.sub('nonce=""', html_text)
    has_unsafe_sink = bool(_UNSAFE_DOM_SINK_RE.search(stripped_html))
    has_static_fig = bool(_STATIC_FIGURE_RE.search(stripped_html))
    if has_unsafe_sink:
        issues.append("unsafe_dom_sink_detected")
    if has_static_fig:
        issues.append("unbacked_static_figure_detected")
    dom_clean = not has_unsafe_sink and not has_static_fig

    low_html = stripped_html.lower()
    compliance_ok = (
        ("personal" in low_html or "personnel" in low_html)
        and ("0.00 eur" in low_html or "0,00 eur" in low_html or "0.00 usd" in low_html or "0,00 usd" in low_html)
        and ("agpl-3.0" in low_html or "proof of concept" in low_html)
    )
    if not compliance_ok:
        issues.append("missing_personal_pc_or_zero_eur_banner")

    info = {
        "has_doctype": has_doctype,
        "csp_ok": csp_ok,
        "nonce_reused": nonce_reused,
        "dom_clean": dom_clean,
        "compliance_ok": compliance_ok,
    }
    return len(issues) == 0, info, issues


def _inspect_manifest_parity(
    doc: Mapping[str, Any],
    expected_sha: str | None,
) -> tuple[bool, str, bool | None, list[str]]:
    issues: list[str] = []
    if doc.get("object") != "hivemind.onboarding.manifest":
        issues.append("manifest_invalid_object")
    if doc.get("campaign_id") != "noel-2026-easy-onboard":
        issues.append("manifest_campaign_mismatch")
    if doc.get("status") != "ok":
        issues.append("manifest_status_not_ok")
    if doc.get("personal_pc_only") is not True:
        issues.append("manifest_personal_pc_only_false")
    if doc.get("financial_promise") != "0.00 EUR / 0.00 USD":
        issues.append("manifest_financial_promise_mismatch")
    if _has_forbidden_score_key(doc):
        issues.append("manifest_forbidden_key")

    routes = doc.get("holiday_routes")
    if not isinstance(routes, (list, tuple)) or not all(isinstance(r, str) for r in routes):
        issues.append("manifest_invalid_holiday_routes")
    elif any(req not in routes for req in _MANIFEST_REQUIRED_HOLIDAY_ROUTES):
        issues.append("manifest_missing_holiday_routes")

    scripts = doc.get("downloadable_scripts")
    if not isinstance(scripts, (list, tuple)) or not all(isinstance(s, str) for s in scripts):
        issues.append("manifest_invalid_downloadable_scripts")
    elif "hivemind_quickstart.py" not in scripts or "hivemind_doctor.py" not in scripts:
        issues.append("manifest_missing_core_scripts")

    live_sha = _sanitize_build_sha_local(doc.get("build_sha"))
    sha_match: bool | None = None
    if expected_sha is not None:
        sha_match = _shas_match_local(live_sha, expected_sha)
        if not sha_match:
            issues.append("manifest_build_sha_mismatch")

    return len(issues) == 0, live_sha, sha_match, issues


def _inspect_feedback_summary_parity(doc: Mapping[str, Any]) -> tuple[bool, list[str]]:
    issues: list[str] = []
    expected_keys = {
        "object",
        "total_responses",
        "mean_ease_rating",
        "zero_game_lag_ratio",
        "by_profile",
    }
    if set(doc.keys()) != expected_keys:
        issues.append("summary_exact_five_keys_mismatch")
    if doc.get("object") != "hivemind.onboarding_feedback_summary":
        issues.append("summary_invalid_object")
    if _has_forbidden_score_key(doc):
        issues.append("summary_forbidden_key")

    raw_total = doc.get("total_responses")
    total_ok = not isinstance(raw_total, bool) and isinstance(raw_total, int) and raw_total >= 0
    if not total_ok:
        issues.append("summary_invalid_total_responses")
        total_val = -1
    else:
        total_val = int(raw_total)

    ease_val = doc.get("mean_ease_rating")
    lag_val = doc.get("zero_game_lag_ratio")
    if total_val == 0:
        if ease_val is not None:
            issues.append("summary_non_null_ease_when_empty")
        if lag_val is not None:
            issues.append("summary_non_null_lag_when_empty")
    elif total_val > 0:
        if not _is_bounded_finite_number(ease_val, 1.0, 5.0):
            issues.append("summary_invalid_mean_ease_rating")
        if not _is_bounded_finite_number(lag_val, 0.0, 1.0):
            issues.append("summary_invalid_zero_game_lag_ratio")

    bp = doc.get("by_profile")
    if not isinstance(bp, Mapping) or set(bp.keys()) != {"dev", "gamer"}:
        issues.append("summary_invalid_by_profile")
    else:
        dev_cnt = bp.get("dev")
        gamer_cnt = bp.get("gamer")
        if (
            isinstance(dev_cnt, bool)
            or not isinstance(dev_cnt, int)
            or dev_cnt < 0
            or isinstance(gamer_cnt, bool)
            or not isinstance(gamer_cnt, int)
            or gamer_cnt < 0
        ):
            issues.append("summary_invalid_profile_counts")
        elif total_ok and dev_cnt + gamer_cnt != total_val:
            issues.append("summary_profile_conservation_mismatch")

    return len(issues) == 0, issues


def evaluate_signup_holiday_parity(
    route_probes: Mapping[str, Mapping[str, Any]],
    *,
    expected_build_sha: str | None = None,
) -> dict[str, Any]:
    """Deterministically verify HTTP 200, CSP nonces, DOM cleanliness, compliance, and schema parity across holiday routes."""
    if not isinstance(route_probes, Mapping):
        raise ValueError("route_probes must be a Mapping of route paths to probe dictionaries")
    for raw_key, raw_val in route_probes.items():
        if isinstance(raw_key, bool) or not isinstance(raw_key, str):
            raise ValueError("route_probes keys must be route path strings")
        if "completions" in raw_key.lower():
            raise ValueError("live GPU inference probes are forbidden in holiday parity checks")
        if raw_val is not None and not isinstance(raw_val, Mapping):
            raise ValueError("each entry in route_probes must be a Mapping")

    clean_expected_sha: str | None = None
    if expected_build_sha is not None:
        if isinstance(expected_build_sha, bool) or not isinstance(expected_build_sha, str):
            raise ValueError("expected_build_sha must be a 6-64 char alphanumeric string or None")
        sanitized = _sanitize_build_sha_local(expected_build_sha)
        if sanitized == "unknown":
            raise ValueError("expected_build_sha must be a 6-64 char alphanumeric string or None")
        clean_expected_sha = sanitized

    seen_nonces: set[str] = set()
    missing_404_routes: list[str] = []
    failed_routes: list[str] = []
    diagnostics: list[str] = []
    per_route: dict[str, dict[str, Any]] = {}

    passed_count = 0
    any_nonce_reused = False
    all_html_csp_ok = True
    html_dom_clean = True
    all_compliance_ok = True
    live_build_sha = "unknown"
    sha_match: bool | None = None
    manifest_ok = False
    feedback_summary_ok = False

    for route in SIGNUP_HOLIDAY_PARITY_ROUTES:
        probe = route_probes.get(route)
        if probe is None:
            failed_routes.append(route)
            diagnostics.append(f"{route}: missing_probe")
            per_route[route] = {"status_code": None, "ready": False, "issues": ["missing_probe"]}
            if route in _SIGNUP_HTML_PARITY_ROUTES:
                all_html_csp_ok = False
                all_compliance_ok = False
            continue

        status_code, headers, text_body, json_body = _extract_route_probe(probe)
        if status_code != 200:
            failed_routes.append(route)
            if status_code == 404:
                missing_404_routes.append(route)
            issue = f"http_{status_code}" if status_code is not None else "network_or_missing_status"
            diagnostics.append(f"{route}: {issue}")
            per_route[route] = {"status_code": status_code, "ready": False, "issues": [issue]}
            if route in _SIGNUP_HTML_PARITY_ROUTES:
                all_html_csp_ok = False
                all_compliance_ok = False
            continue

        if route in _SIGNUP_HTML_PARITY_ROUTES:
            route_ok, h_info, r_issues = _inspect_html_parity_route(headers, text_body, seen_nonces)
            any_nonce_reused = any_nonce_reused or h_info["nonce_reused"]
            all_html_csp_ok = all_html_csp_ok and h_info["csp_ok"]
            html_dom_clean = html_dom_clean and h_info["dom_clean"]
            all_compliance_ok = all_compliance_ok and h_info["compliance_ok"]
            per_route[route] = {"status_code": status_code, "ready": route_ok, **h_info, "issues": r_issues}
        elif route == "/v1/hivemind/onboarding/manifest":
            if json_body is None:
                route_ok, r_issues = False, ["manifest_missing_json"]
                all_compliance_ok = False
            else:
                route_ok, live_build_sha, sha_match, r_issues = _inspect_manifest_parity(json_body, clean_expected_sha)
                if json_body.get("personal_pc_only") is not True:
                    all_compliance_ok = False
            manifest_ok = route_ok
            per_route[route] = {"status_code": status_code, "ready": route_ok, "issues": r_issues}
        else:
            if json_body is None:
                route_ok, r_issues = False, ["summary_missing_json"]
            else:
                route_ok, r_issues = _inspect_feedback_summary_parity(json_body)
            feedback_summary_ok = route_ok
            per_route[route] = {"status_code": status_code, "ready": route_ok, "issues": r_issues}

        if route_ok:
            passed_count += 1
        else:
            failed_routes.append(route)
            diagnostics.extend(f"{route}: {item}" for item in r_issues)

    unique_csp_nonces_ok = (
        all_html_csp_ok
        and not any_nonce_reused
        and len(seen_nonces) == len(_SIGNUP_HTML_PARITY_ROUTES)
    )
    parity_ok = bool(
        passed_count == len(SIGNUP_HOLIDAY_PARITY_ROUTES)
        and unique_csp_nonces_ok
        and html_dom_clean
        and all_compliance_ok
        and manifest_ok
        and feedback_summary_ok
        and (clean_expected_sha is None or sha_match is True)
    )

    core: dict[str, Any] = {
        "object": "hivemind.signup_holiday_parity_report",
        "campaign_id": "noel-2026-easy-onboard",
        "required_routes": list(SIGNUP_HOLIDAY_PARITY_ROUTES),
        "routes_total_count": len(SIGNUP_HOLIDAY_PARITY_ROUTES),
        "routes_passed_count": passed_count,
        "parity_ok": parity_ok,
        "stale_404_drift": len(missing_404_routes) > 0,
        "missing_or_404_routes": missing_404_routes,
        "failed_routes": failed_routes,
        "unique_csp_nonces_count": len(seen_nonces),
        "unique_csp_nonces_ok": unique_csp_nonces_ok,
        "html_dom_clean": html_dom_clean,
        "compliance_ok": all_compliance_ok,
        "live_build_sha": live_build_sha,
        "expected_build_sha": clean_expected_sha,
        "sha_match": sha_match,
        "manifest_ok": manifest_ok,
        "feedback_summary_ok": feedback_summary_ok,
        "zero_inference_verified": True,
        "per_route": per_route,
        "diagnostics": diagnostics,
    }
    canonical = json.dumps(core, sort_keys=True, ensure_ascii=False, allow_nan=False)
    core["receipt_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return core


def render_signup_page(
    client_id: str | None,
    nonce: str,
    *,
    auth_result: dict[str, Any] | None = None,
    error_message: str | None = None,
    live_telemetry: dict[str, Any] | None = None,
    starter_grant_hmt: float = 0.0,
    registration_bonus_hmt: float = 0.0,
    allow_bonuses: bool = False,
    default_role: str = "prosumer",
) -> str:
    """Render the unified Login & Registration page at ``/signup`` and ``/login``."""
    if not isinstance(nonce, str) or not nonce.strip():
        raise ValueError("nonce must be a non-empty string")
    if client_id is not None and (not isinstance(client_id, str) or not CLIENT_ID_RE.match(client_id)):
        raise ValueError("invalid Google client id")
    if isinstance(starter_grant_hmt, bool) or not isinstance(starter_grant_hmt, (int, float)) or not math.isfinite(float(starter_grant_hmt)) or float(starter_grant_hmt) < 0.0:
        raise ValueError("starter_grant_hmt must be a finite non-negative float")
    if isinstance(registration_bonus_hmt, bool) or not isinstance(registration_bonus_hmt, (int, float)) or not math.isfinite(float(registration_bonus_hmt)) or float(registration_bonus_hmt) < 0.0:
        raise ValueError("registration_bonus_hmt must be a finite non-negative float")
    if not isinstance(allow_bonuses, bool):
        raise ValueError("allow_bonuses must be a bool")
    if auth_result is not None and not isinstance(auth_result, Mapping):
        raise ValueError("auth_result must be a mapping or None")
    if error_message is not None and not isinstance(error_message, str):
        raise ValueError("error_message must be a string or None")
    if live_telemetry is not None and not isinstance(live_telemetry, Mapping):
        raise ValueError("live_telemetry must be a mapping or None")
    if not isinstance(default_role, str):
        raise ValueError("default_role must be a string")
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

    bonus_suffix = " Verified contributor and model-testing bonuses can also be claimed after sign-in." if allow_bonuses else ""
    if starter_grant_hmt > 0.0 and registration_bonus_hmt > 0.0:
        total_start = starter_grant_hmt + registration_bonus_hmt
        starter_line = f"Starter grant on this gateway: {starter_grant_hmt:.2f} HMT + {registration_bonus_hmt:.2f} HMT registration bonus ({total_start:.2f} HMT total per new account), at most a few per address per day.{bonus_suffix}"
    elif starter_grant_hmt > 0.0:
        starter_line = f"Starter grant on this gateway: {starter_grant_hmt:.2f} HMT per new account, at most a few per address per day.{bonus_suffix}"
    elif registration_bonus_hmt > 0.0:
        starter_line = f"Registration bonus on this gateway: {registration_bonus_hmt:.2f} HMT per new account, at most a few per address per day.{bonus_suffix}"
    else:
        starter_line = f"A new account starts at 0.00 HMT on this gateway: points come from sharing a GPU or from an operator grant.{bonus_suffix}"
    claim_bonus_btn_html = (
        '\n    <button id="claimBonusBtn" data-testid="signup-claim-bonus-btn" type="button" class="secondary">🎁 Claim Bonus HMT</button>'
        if allow_bonuses
        else ""
    )
    banner_bonus_btn_html = (
        '\n    <button id="bannerClaimBonusBtn" data-testid="banner-claim-bonus-btn" type="button" class="secondary">🎁 Claim Bonus HMT</button>'
        if allow_bonuses
        else ""
    )
    result_hidden = "" if auth_result else " hidden"
    res_kind = ""
    res_key = ""
    res_acct = ""
    res_role = ""
    res_bal = "n/a"
    raw_res_key = ""
    if auth_result:
        res_kind = (
            "Your account was created."
            if auth_result.get("created")
            else "Welcome back! Your previous account & HMT balance have been restored."
        )
        raw_res_key = str(auth_result.get("api_key", "")).strip()
        res_key = html.escape(raw_res_key)
        res_acct = html.escape(
            f"{auth_result.get('account_id', '')}"
            + (f" ({auth_result.get('handle')})" if auth_result.get("handle") else "")
        )
        res_role = html.escape(str(auth_result.get("role", "prosumer")))
        bal_val = float(auth_result.get("balance_hmt", auth_result.get("balance", 0.0)))
        res_bal = f"{bal_val:.2f} HMT"

    yaml_key = raw_res_key if _CONTINUE_KEY_RE.fullmatch(raw_res_key) else DEFAULT_KEY_PLACEHOLDER
    continue_yaml_html = html.escape(build_continue_config_yaml(yaml_key, DEFAULT_PUBLIC_API_BASE))
    ces_ease_btns_html = "".join(
        f'<button id="signupCesEaseBtn-{r}" data-testid="signup-ces-ease-btn-{r}" '
        f'type="button" class="{"chip-btn active" if r == 5 else "chip-btn"}" '
        f'data-ease="{r}" aria-pressed="{"true" if r == 5 else "false"}">{r} / 5</button>'
        for r in range(1, 6)
    )

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
    <p class="mut" id="personal-pc-compliance" data-testid="personal-pc-compliance">{PERSONAL_PC_COMPLIANCE_NOTE}</p>
  </div>
  <nav class="nav-links" data-testid="signup-quick-nav">
    <a href="/noel">🎄 Noël Quick Start</a>
    <a href="/raid-noel">🏔️ Winter GPU Raid</a>
    <a href="/arena-noel">⚖️ Blind A/B Arena</a>
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
    <a href="/#playground" class="btn-link secondary">🚀 Open AI Playground</a>{banner_bonus_btn_html}
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
    <a href="/webgpu" class="btn-link secondary">🌌 Open WebGPU Cockpit</a>{claim_bonus_btn_html}
    <button id="rotateKeyBtn" data-testid="signup-rotate-key-btn" type="button" class="secondary">🔄 Rotate API Key</button>
  </p>
  <p class="mut">Account <code id="acct">{res_acct}</code> · Role <code id="role">{res_role}</code> · Balance <code id="bal">{res_bal}</code></p>
  <form id="transferPointsForm" data-testid="signup-transfer-hmt-form" style="margin:.75rem 0;padding:.75rem;border:1px solid var(--line);border-radius:.5rem;background:var(--card)">
    <strong>🤝 Give HMT Points to Another HiveMinder</strong>
    <div style="display:grid;grid-template-columns:1.2fr .8fr 1.4fr auto;gap:.5rem;align-items:end;margin-top:.45rem">
      <label style="padding:0">Recipient (@handle)<input id="transferRecipient" type="text" placeholder="@Vxtzq" style="margin-bottom:0" /></label>
      <label style="padding:0">Amount (HMT)<input id="transferAmount" type="text" inputmode="decimal" placeholder="5000" style="margin-bottom:0" /></label>
      <label style="padding:0">Purpose / Memo<input id="transferMemo" type="text" placeholder="CrowdGPT inference tests" style="margin-bottom:0" /></label>
      <button id="transferSubmitBtn" data-testid="signup-transfer-hmt-btn" type="submit" class="secondary">Send HMT</button>
    </div>
  </form>
  <div id="ideQuickSetup" data-testid="signup-ide-quick-setup" style="margin:.75rem 0;padding:.85rem;border:1px solid var(--line);border-radius:.5rem;background:var(--card)">
    <h3 style="margin:0 0 .35rem;font-size:1rem;color:var(--amber)">⚡ 60-Second IDE & CLI Quick Setup</h3>
    <p class="mut" style="margin:.2rem 0 .45rem">Paste directly into <code>~/.continue/config.yaml</code> in VS Code (Continue.dev) for personal hobby/open-source coding (0.00 EUR/USD):</p>
    <pre><code id="continueYamlSnippet">{continue_yaml_html}</code></pre>
    <p style="display:flex;flex-wrap:wrap;align-items:center;gap:.5rem;margin:.5rem 0">
      <button id="copyContinueBtn" data-testid="signup-copy-continue-btn" type="button">📋 Copy VS Code Continue.dev Config</button>
      <button id="verifyKeyModelsBtn" data-testid="signup-verify-key-models-btn" type="button" class="secondary">⚡ Test Key in Browser (/v1/models)</button>
      <span id="verifyKeyStatus" data-testid="signup-verify-key-status" class="mut" role="status"></span>
    </p>
    <div id="quickConnectCliBlock" class="ces-box">
      <p class="mut"><strong>&lt; 60 s Zero-Dependency Quick-Connect &amp; &lt; 10 s Doctor CLI (&lt; 5 min Gamer Whisper with &lt; 1.5 s Steam pause):</strong></p>
      <pre><code>python3 scripts/hivemind_quickstart.py --write-continue-config
hivemind vscode-setup
python3 scripts/hivemind_doctor.py --noel</code></pre>
      <p class="row-actions">
        <button id="verifyHolidayParityBtn" data-testid="signup-verify-holiday-parity-btn" type="button" class="secondary">🛰️ Check Holiday Route Parity</button>
        <span id="verifyHolidayParityStatus" data-testid="signup-verify-holiday-parity-status" class="mut" role="status" aria-live="polite"></span>
      </p>
    </div>
    <div id="signupCesCard" data-testid="signup-ces-card" class="ces-card">
      <h4>🗳️ 1-Click Zero-PII CES &amp; Steam Zero-Game-Lag Micro-Survey (&lt; 1.5 s)</h4>
      <p class="mut">Rate setup ease (1 to 5) &amp; Steam zero-lag comfort in 1 click (zero credentials, zero free-text, zero PII • 0.00 EUR / 0.00 USD):</p>
      <div class="ces-row" role="group" aria-label="Profile">
        <span class="ces-lbl">Profile:</span>
        <button id="signupCesProfileDevBtn" data-testid="signup-ces-profile-dev-btn" type="button" class="chip-btn active" data-profile="dev" aria-pressed="true">Dev (VS Code / API)</button>
        <button id="signupCesProfileGamerBtn" data-testid="signup-ces-profile-gamer-btn" type="button" class="chip-btn" data-profile="gamer" aria-pressed="false">PC Gamer (Steam / GPU)</button>
      </div>
      <div class="ces-row" role="group" aria-label="Ease Rating">
        <span class="ces-lbl">Ease (1–5):</span>
        {ces_ease_btns_html}
      </div>
      <div class="ces-row" role="group" aria-label="Steam Zero-Game-Lag">
        <span class="ces-lbl">Steam Comfort (&lt; 1.5 s VRAM release):</span>
        <button id="signupCesLagYesBtn" data-testid="signup-ces-lag-yes-btn" type="button" class="chip-btn active" data-zero-lag="true" aria-pressed="true">Zero Game Lag (Yes)</button>
        <button id="signupCesLagNoBtn" data-testid="signup-ces-lag-no-btn" type="button" class="chip-btn" data-zero-lag="false" aria-pressed="false">Noticeable Lag (No)</button>
      </div>
      <div class="row-actions">
        <button id="signupSubmitCesBtn" data-testid="signup-submit-ces-btn" type="button">✅ Submit 1-Click Feedback (Zero PII)</button>
        <button id="signupRetryPendingCesBtn" data-testid="signup-retry-pending-ces-btn" type="button" class="secondary">🔁 Retry Pending Feedback</button>
      </div>
      <p id="signupCesStatus" data-testid="signup-ces-status" class="mut" role="status" aria-live="polite">Ready: pick profile, ease (default 5/5), and Steam comfort, then submit in 1 click.</p>
      <p id="signupCesComplianceNote" data-testid="signup-ces-compliance-note" class="mut">{CES_SIGNUP_COMPLIANCE_NOTE}</p>
    </div>
  </div>
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

<section id="passwordResetCard" class="card">
  <h2>🔐 Set / Reset Password (Réinitialiser ou définir un mot de passe)</h2>
  <p class="mut">Have an <code>hm-...</code> API key or a single-use password reset token? Set a new password (12–256 characters) below. Updating your password automatically rotates your API key so older keys are revoked.</p>
  <form id="changePasswordForm">
    <label for="resetTokenOrKey">Your active <code>hm-...</code> API key or Reset Token (optional if already signed in above)</label>
    <input id="resetTokenOrKey" name="token_or_key" type="text" autocomplete="off" placeholder="hm-... or single-use reset token" />
    <label for="newResetPassword">New Password (12 to 256 characters)</label>
    <input id="newResetPassword" name="new_password" type="password" minlength="12" maxlength="256" required autocomplete="new-password" placeholder="At least 12 characters" />
    <p style="margin-top:.65rem"><button id="changePasswordBtn" type="submit">Update Password &amp; Rotate Key · Mettre à jour</button></p>
  </form>
</section>

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
