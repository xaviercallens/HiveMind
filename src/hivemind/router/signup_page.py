# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unified Login & Registration screen served at ``/signup``, ``/login``, ``/register``, and ``/connexion``.

Supports:
1. **Returning / Previous User Login** (Username / Handle / Email + Password, or existing ``hm-...`` / ``hm_live_...`` API key)
   via ``POST /v1/hivemind/auth/login`` (and ``POST /login`` HTML form fallback).
2. **New User Registration** (Username / Handle + Password + optional Email + Role ``prosumer`` / ``consumer`` / ``producer``)
   via ``POST /v1/hivemind/auth/register`` (and ``POST /signup`` HTML form fallback).
3. **Google Sign-In** (when ``client_id`` is configured) via ``POST /v1/hivemind/register/google``.

All dynamic DOM updates use ``textContent`` (never ``innerHTML``), and the Content-Security-Policy enforces per-request nonces.
"""

from __future__ import annotations

import html
from typing import Any

from hivemind.router.google_auth import CLIENT_ID_RE

# Origins Google documents for the Identity Services button (script, frame, connect, style).
GSI = "https://accounts.google.com/gsi/"


def content_security_policy(nonce: str) -> str:
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

_AUTH_JS = """
const $ = (id) => document.getElementById(id);
function setStatus(text, isErr) {
  const el = $('status');
  if (!el) return;
  el.textContent = text;
  el.className = isErr ? 'err-text' : 'mut';
}
function showSessionResult(body) {
  if ($('key')) $('key').textContent = body.api_key || '';
  if ($('acct')) $('acct').textContent = (body.account_id || '') + (body.handle ? ' (' + body.handle + ')' : '');
  if ($('role')) $('role').textContent = body.role || 'prosumer';
  if ($('bal')) $('bal').textContent = (typeof body.balance_hmt === 'number' ? body.balance_hmt.toFixed(2) : (typeof body.balance === 'number' ? body.balance.toFixed(2) : '100.00')) + ' HMT';
  if ($('kind')) {
    $('kind').textContent = body.created
      ? 'Account created & authenticated! Your API key is saved in this browser for the Playground.'
      : 'Welcome back! Your previous account & HMT balance have been restored and authenticated.';
  }
  try {
    if (body.api_key) localStorage.setItem('hivemind_api_key', body.api_key);
    if (body.handle) localStorage.setItem('hivemind_user_handle', body.handle);
  } catch (e) {}
  if ($('result')) $('result').hidden = false;
  setStatus('');
}
function chosenRegRole() {
  const checked = document.querySelector('input[name=role]:checked');
  return checked ? checked.value : 'prosumer';
}
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
    setStatus('Creating your HiveMind account & minting starter HMT...', false);
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
    try { if (body.api_key) localStorage.setItem('hivemind_api_key', body.api_key); } catch (e) {}
    $('result').hidden = false; if ($('signin')) $('signin').hidden = true; setStatus('', false);
  } catch (err) { setStatus('Could not reach the gateway. Try again.', true); }
}
function init() {
  google.accounts.id.initialize({client_id: CLIENT_ID, callback: onCredential, auto_select: false});
  google.accounts.id.renderButton($('gbtn'), {theme: 'outline', size: 'large', text: 'signin_with', shape: 'pill'});
}
window.onGoogleLibraryLoad = init;
"""


SIGNUP_I18N: dict[str, dict[str, str]] = {
    "en": {
        "title": "Join HiveMind",
        "lead": "Use open coding models from your editor, or share your GPU and earn points. HiveMind points have no cash value.",
        "disabled_notice": "Sign in with Google is not enabled on this gateway yet.",
        "disabled_hint": "You can still register with a handle through the API. See the getting-started guide.",
        "legend": "What do you want to do?",
        "role_prosumer": "Both: use models and share my GPU",
        "role_consumer": "Use models in my editor",
        "role_producer": "Share my GPU",
        "privacy": "We keep only an anonymous hash of your Google account id. Your e-mail address is not stored.",
        "key_label": "Your API key (shown once; signing in again replaces it):",
        "copy_btn": "Copy key",
        "next_label": "Next, in a terminal:",
        "zero_points": "New accounts hold 0 points. Share a GPU to earn some, or ask the operator for a grant.",
        "portal_link": "← Back to OpenRouter Playground",
    },
    "fr": {
        "title": "Rejoindre HiveMind",
        "lead": "Utilisez des modèles de code ouverts depuis votre éditeur, ou partagez votre GPU et gagnez des Points de Réflexion HiveMind (HMT). Les points n'ont aucune valeur monétaire.",
        "disabled_notice": "La connexion avec Google n'est pas encore activée sur cette passerelle (Sign in with Google is not enabled on this gateway yet).",
        "disabled_hint": "Vous pouvez toujours vous inscrire avec un pseudonyme via l'API. Consultez le guide de démarrage.",
        "legend": "Que souhaitez-vous faire ?",
        "role_prosumer": "Les deux : utiliser les modèles et partager mon GPU",
        "role_consumer": "Utiliser les modèles dans mon éditeur",
        "role_producer": "Partager mon GPU",
        "privacy": "Nous ne conservons qu'un hachage anonyme de votre identifiant Google. Votre adresse e-mail n'est jamais stockée.",
        "key_label": "Votre clé API (affichée une seule fois ; une reconnexion la remplace) :",
        "copy_btn": "Copier la clé",
        "next_label": "Ensuite, dans un terminal :",
        "zero_points": "Les nouveaux comptes ont 0 point. Partagez un GPU pour en gagner ou demandez un don de démarrage.",
        "portal_link": "← Retour au Playground OpenRouter",
    },
    "es": {
        "title": "Unirse a HiveMind",
        "lead": "Usa modelos abiertos de programación desde tu editor o comparte tu GPU y gana Puntos de Reflexión HiveMind (HMT). Los puntos no tienen valor monetario.",
        "disabled_notice": "El inicio de sesión con Google aún no está habilitado en esta pasarela (Sign in with Google is not enabled on this gateway yet).",
        "disabled_hint": "Aún puedes registrarte con un seudónimo mediante la API. Consulta la guía de inicio.",
        "legend": "¿Qué quieres hacer?",
        "role_prosumer": "Ambos: usar modelos y compartir mi GPU",
        "role_consumer": "Usar modelos en mi editor",
        "role_producer": "Compartir mi GPU",
        "privacy": "Solo guardamos un hash anónimo de tu ID de Google. Tu correo electrónico no se almacena.",
        "key_label": "Tu clave API (se muestra una sola vez; volver a iniciar sesión la reemplaza):",
        "copy_btn": "Copiar clave",
        "next_label": "A continuación, en una terminal:",
        "zero_points": "Las cuentas nuevas tienen 0 puntos. Comparte una GPU para ganar puntos o solicita una concesión inicial.",
        "portal_link": "← Volver al Playground OpenRouter",
    },
    "de": {
        "title": "HiveMind beitreten",
        "lead": "Nutzen Sie offene Coding-Modelle in Ihrem Editor oder teilen Sie Ihre GPU und sammeln Sie HiveMind-Reflexionspunkte (HMT). Punkte haben keinen Geldwert.",
        "disabled_notice": "Die Anmeldung mit Google ist auf diesem Gateway noch nicht aktiviert (Sign in with Google is not enabled on this gateway yet).",
        "disabled_hint": "Sie können sich weiterhin mit einem Pseudonym über die API registrieren. Siehe Schnellstart-Anleitung.",
        "legend": "Was möchten Sie tun?",
        "role_prosumer": "Beides: Modelle nutzen und meine GPU teilen",
        "role_consumer": "Modelle in meinem Editor nutzen",
        "role_producer": "Meine GPU teilen",
        "privacy": "Wir speichern nur einen anonymen Hash Ihrer Google-ID. Ihre E-Mail-Adresse wird nicht gespeichert.",
        "key_label": "Ihr API-Schlüssel (wird nur einmal angezeigt; erneutes Anmelden ersetzt ihn):",
        "copy_btn": "Schlüssel kopieren",
        "next_label": "Als Nächstes im Terminal:",
        "zero_points": "Neue Konten starten mit 0 Punkten. Teilen Sie eine GPU oder fragen Sie nach einem Startguthaben.",
        "portal_link": "← Zurück zum OpenRouter Playground",
    },
    "zh": {
        "title": "加入 HiveMind",
        "lead": "在代码编辑器中使用开源编程模型，或共享您的 GPU 赚取 HiveMind 反思积分 (HMT)。积分没有任何货币价值。",
        "disabled_notice": "此网关尚未启用 Google 登录 (Sign in with Google is not enabled on this gateway yet)。",
        "disabled_hint": "您仍然可以通过 API 使用昵称注册账户，请参阅入门指南。",
        "legend": "您希望如何参与？",
        "role_prosumer": "两者兼顾：使用模型并共享我的 GPU",
        "role_consumer": "在编辑器中使用 AI 模型",
        "role_producer": "共享我的 GPU 算力",
        "privacy": "我们仅保存您 Google 账户 ID 的匿名哈希值，绝不存储您的电子邮箱地址。",
        "key_label": "您的 API 密钥（仅显示一次，重新登录将生成新密钥）：",
        "copy_btn": "复制密钥",
        "next_label": "接下来，在终端中运行：",
        "zero_points": "新账户初始为 0 积分。共享 GPU 即可赚取积分，或向管理员申请初始试用积分。",
        "portal_link": "← 返回 OpenRouter 演练场",
    },
}


def render_signup_page(
    client_id: str | None,
    nonce: str,
    *,
    lang: str = "en",
    auth_result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> str:
    """Render the unified 5-language Login & Registration page at ``/signup``, ``/inscription``, ``/login``, ``/register``, and ``/connexion``."""
    if client_id is not None and not CLIENT_ID_RE.match(client_id):
        raise ValueError("invalid Google client id")

    active_lang = lang if lang in SIGNUP_I18N else "en"
    t = SIGNUP_I18N[active_lang]
    lang_nav = (
        '<nav aria-label="Language" style="margin-bottom:0.8rem;font-size:0.9rem;">'
        '<a href="/signup?lang=en">🇬🇧 EN</a> · '
        '<a href="/signup?lang=fr">🇫🇷 FR</a> · '
        '<a href="/signup?lang=es">🇪🇸 ES</a> · '
        '<a href="/signup?lang=de">🇩🇪 DE</a> · '
        '<a href="/signup?lang=zh">🇨🇳 中文</a> · '
        f'<a href="/?lang={active_lang}">{t["portal_link"]}</a>'
        '</nav>'
    )

    result_hidden = "" if auth_result else " hidden"
    res_kind = ""
    res_key = ""
    res_acct = ""
    res_role = ""
    res_bal = "100.00 HMT"
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
        bal_val = float(auth_result.get("balance_hmt", auth_result.get("balance", 100.0)))
        res_bal = f"{bal_val:.2f} HMT"

    err_html = html.escape(error_message) if error_message else ""
    err_cls = "err-text" if error_message else "mut"

    if client_id is None:
        google_section = (
            '<div class="card">'
            f'<p>{t["disabled_notice"]}</p>'
            f'<p class="mut">{t["disabled_hint"]}</p>'
            '</div>'
        )
        scripts_html = f'<script nonce="{nonce}">{_AUTH_JS}</script>'
    else:
        g_script = _GOOGLE_JS.replace("%CLIENT_ID%", '"' + client_id + '"')
        google_section = (
            '<section id="signin" class="card">'
            '<h2>Option 3 · Sign in with Google</h2>'
            f'<fieldset><legend>{t["legend"]}</legend>'
            f'<label><input type="radio" name="g_role" value="prosumer" checked> {t["role_prosumer"]}</label>'
            f'<label><input type="radio" name="g_role" value="consumer"> {t["role_consumer"]}</label>'
            f'<label><input type="radio" name="g_role" value="producer"> {t["role_producer"]}</label>'
            '</fieldset>'
            '<div id="gbtn"></div>'
            '<noscript><p>This page needs JavaScript to show the Google button.</p></noscript>'
            f'<p class="mut">{t["privacy"]}</p>'
            '</section>'
        )
        scripts_html = (
            f'<script nonce="{nonce}">{_AUTH_JS}\n{g_script}</script>\n'
            f'<script nonce="{nonce}" src="https://accounts.google.com/gsi/client" async defer></script>'
        )

    return f"""<!doctype html><html lang="{active_lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{t["title"]} — HiveMind OpenRouter</title><style nonce="{nonce}">{_CSS}</style></head><body><main>
{lang_nav}
<header>
  <div>
    <h1>🐝 {t["title"]}</h1>
    <p class="mut">{t["lead"]}</p>
  </div>
  <nav class="nav-links">
    <a href="/">⚡ AI Playground</a>
    <a href="/benchmarks">📊 Benchmarks</a>
    <a href="/life-as-a-bee">🐝 Life as a Bee</a>
    <a href="/webgpu">🌌 WebGPU</a>
    <a href="/join">How to join</a> · <a href="/rejoindre">Comment participer</a> · <a href="/join?lang=es">Español</a> · <a href="/join?lang=de">Deutsch</a> · <a href="/join?lang=zh">中文</a>
  </nav>
</header>

<section id="result" class="card ok-banner"{result_hidden}>
  <h2>✅ Authenticated HiveMind Account</h2>
  <p id="kind">{res_kind}</p>
  <p>{t["key_label"]}</p>
  <pre><code id="key">{res_key}</code></pre>
  <p>
    <button id="copy" type="button">{t["copy_btn"]}</button>
    <a href="/#playground" class="btn-link secondary">🚀 Open AI Playground</a>
    <a href="/webgpu" class="btn-link secondary">🌌 Open WebGPU Cockpit</a>
  </p>
  <p class="mut">Account <code id="acct">{res_acct}</code> · Role <code id="role">{res_role}</code> · Balance <code id="bal">{res_bal}</code></p>
  <p>{t["next_label"]}</p>
  <pre><code>export HIVEMIND_API_KEY=&lt;the key above&gt;
python hivemind_connect.py check
python hivemind_connect.py snippet continue --model coder</code></pre>
  <p class="mut">{t["zero_points"]}</p>
</section>

<p id="status" class="{err_cls}" role="status">{err_html}</p>

<div class="grid-auth">
  <section id="loginCard" class="card">
    <h2>🔑 Existing User Login (Se connecter)</h2>
    <p class="mut">Sign in with your previous <strong>username / handle / email</strong> and <strong>password</strong>, or paste an existing <code>hm-...</code> / <code>hm_live_...</code> key.</p>
    <form id="loginForm" method="post" action="/login">
      <label for="loginIdentifier">Username, Handle, Email, or API Key</label>
      <input id="loginIdentifier" name="identifier" type="text" required autocomplete="username" placeholder="e.g. xavier_callens, @Vxtzq, email, or hm-..." />
      <label for="loginPassword">Password (optional if using API key or legacy handle)</label>
      <input id="loginPassword" name="password" type="password" autocomplete="current-password" placeholder="Your password (sets password on first login)" />
      <p style="margin-top:.65rem"><button id="loginSubmitBtn" type="submit">Sign In &amp; Restore Account · Se connecter</button></p>
    </form>
  </section>

  <section id="registerCard" class="card">
    <h2>🐝 New User Registration (Créer un compte)</h2>
    <p class="mut">Create a new HiveMind account with a username and password to receive your <code>hm-...</code> API key and starter HMT points.</p>
    <form id="registerForm" method="post" action="/signup">
      <label for="regHandle">Username / Handle</label>
      <input id="regHandle" name="handle" type="text" required autocomplete="username" placeholder="e.g. alice_gpu or dev-paris" />
      <label for="regEmail">Email (optional — for account recovery)</label>
      <input id="regEmail" name="email" type="email" autocomplete="email" placeholder="alice@example.org (optional)" />
      <label for="regPassword">Password</label>
      <input id="regPassword" name="password" type="password" autocomplete="new-password" placeholder="Choose a password" />
      <fieldset><legend>What do you want to do?</legend>
        <label class="radio-label"><input type="radio" name="role" value="prosumer" checked> Both: use models and share my GPU (Prosumer)</label>
        <label class="radio-label"><input type="radio" name="role" value="consumer"> Use models in my editor / Playground (Consumer)</label>
        <label class="radio-label"><input type="radio" name="role" value="producer"> Share my GPU with the swarm (Producer)</label>
      </fieldset>
      <p><button id="regSubmitBtn" type="submit">Create Account &amp; Get API Key · S'inscrire</button></p>
    </form>
  </section>
</div>

{google_section}

{scripts_html}
</main></body></html>"""
