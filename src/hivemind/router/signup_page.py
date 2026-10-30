# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The one-page "Sign in with Google" registration screen served at ``/signup``. Pure function, no I/O.

The page loads Google's Identity Services script, sends the ID token to ``/v1/hivemind/register/google`` and shows the API
key once. Everything dynamic is written with ``textContent`` (never ``innerHTML``), the only server-inserted value is the
OAuth client id (validated against a strict pattern), and the content-security policy allows scripts only from this page's
nonce and Google. Verify the exact Google origins against Google's current Identity Services CSP documentation.
"""

from __future__ import annotations

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
        "form-action 'none'",
        "frame-ancestors 'none'",
    ])


_CSS = """
:root{color-scheme:light dark;--bg:#fafafa;--fg:#1a1a1a;--mut:#555;--card:#fff;--line:#d8d8d8;--acc:#1a56db}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#ececec;--mut:#a8a8a8;--card:#1c1c1c;--line:#3a3a3a;--acc:#8ab4f8}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:34rem;margin:0 auto;padding:1.5rem 1rem 3rem}
h1{font-size:1.5rem;margin:.2rem 0 .6rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:.6rem;padding:1rem;margin:1rem 0}
fieldset{border:0;padding:0;margin:0 0 1rem}legend{font-weight:600;margin-bottom:.4rem}
label{display:block;padding:.25rem 0}
code,pre{font:.9rem ui-monospace,monospace;word-break:break-all;white-space:pre-wrap}
button{font:inherit;padding:.5rem .9rem;border-radius:.4rem;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.mut{color:var(--mut);font-size:.9rem}[hidden]{display:none}
"""

_JS = """
const CLIENT_ID = %CLIENT_ID%;
const $ = (id) => document.getElementById(id);
function setStatus(text) { $('status').textContent = text; }
function chosenRole() { return document.querySelector('input[name=role]:checked').value; }
async function onCredential(response) {
  setStatus('Checking your Google sign-in...');
  try {
    const reply = await fetch('/v1/hivemind/register/google', {
      method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id_token: response.credential, role: chosenRole()}),
    });
    const body = await reply.json().catch(() => ({}));
    if (!reply.ok) { setStatus((body.error && body.error.message) || 'Sign-in failed.'); return; }
    $('key').textContent = body.api_key;
    $('acct').textContent = body.account_id;
    $('kind').textContent = body.created ? 'Your account was created.' : 'Welcome back: your previous key no longer works.';
    $('role').textContent = body.role;
    $('result').hidden = false; $('signin').hidden = true; setStatus('');
  } catch (err) { setStatus('Could not reach the gateway. Try again.'); }
}
function init() {
  google.accounts.id.initialize({client_id: CLIENT_ID, callback: onCredential, auto_select: false});
  google.accounts.id.renderButton($('gbtn'), {theme: 'outline', size: 'large', text: 'signin_with', shape: 'pill'});
}
window.onGoogleLibraryLoad = init;
$('copy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('key').textContent); $('copy').textContent = 'Copied'; }
  catch (err) { $('copy').textContent = 'Select and copy the key manually'; }
});
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


def render_signup_page(client_id: str | None, nonce: str, *, lang: str = "en") -> str:
    """HTML for ``/signup``; ``client_id`` ``None`` renders the "not enabled" notice with no script at all."""
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
    head = (f'<!doctype html><html lang="{active_lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{t["title"]}</title><style nonce="{nonce}">{_CSS}</style></head><body><main>{lang_nav}')
    if client_id is None:
        return head + (f'<h1>{t["title"]}</h1><div class="card"><p>{t["disabled_notice"]}</p>'
                       '<p><a href="/join">How to join</a> · <a href="/rejoindre">Comment participer</a> · '
                       '<a href="/join?lang=es">Español</a> · <a href="/join?lang=de">Deutsch</a> · <a href="/join?lang=zh">中文</a></p>'
                       f'<p class="mut">{t["disabled_hint"]}</p></div></main></body></html>')
    if not CLIENT_ID_RE.match(client_id):
        raise ValueError("invalid Google client id")
    script = _JS.replace("%CLIENT_ID%", '"' + client_id + '"')  # the id matches [0-9a-z.-]+, so it cannot break out of the string
    return head + f"""
<h1>{t["title"]}</h1>
<p>{t["lead"]}
<a href="/join">How to join</a> · <a href="/rejoindre">Comment participer</a> · <a href="/join?lang=es">Español</a> · <a href="/join?lang=de">Deutsch</a> · <a href="/join?lang=zh">中文</a></p>
<section id="signin" class="card">
  <fieldset><legend>{t["legend"]}</legend>
    <label><input type="radio" name="role" value="prosumer" checked> {t["role_prosumer"]}</label>
    <label><input type="radio" name="role" value="consumer"> {t["role_consumer"]}</label>
    <label><input type="radio" name="role" value="producer"> {t["role_producer"]}</label>
  </fieldset>
  <div id="gbtn"></div>
  <noscript><p>This page needs JavaScript to show the Google button.</p></noscript>
  <p class="mut">{t["privacy"]}</p>
</section>
<section id="result" class="card" hidden>
  <p id="kind"></p>
  <p>{t["key_label"]}</p>
  <pre><code id="key"></code></pre>
  <p><button id="copy" type="button">{t["copy_btn"]}</button></p>
  <p class="mut">Account <code id="acct"></code>, role <code id="role"></code>.</p>
  <p>{t["next_label"]}</p>
  <pre><code>export HIVEMIND_API_KEY=&lt;the key above&gt;
python hivemind_connect.py check
python hivemind_connect.py snippet continue --model coder</code></pre>
  <p class="mut">{t["zero_points"]}</p>
</section>
<p id="status" class="mut" role="status"></p>
<script nonce="{nonce}">{script}</script>
<script nonce="{nonce}" src="https://accounts.google.com/gsi/client" async defer></script>
</main></body></html>"""
