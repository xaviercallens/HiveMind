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


def render_signup_page(client_id: str | None, nonce: str) -> str:
    """HTML for ``/signup``; ``client_id`` ``None`` renders the "not enabled" notice with no script at all."""
    head = (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Join HiveMind</title><style nonce="{nonce}">{_CSS}</style></head><body><main>')
    if client_id is None:
        return head + ('<h1>Join HiveMind</h1><div class="card"><p>Sign in with Google is not enabled on this gateway yet.</p>'
                       '<p><a href="/join">How to join</a> · <a href="/rejoindre">Comment participer</a></p>'
                       '<p class="mut">You can still register with a handle through the API. See the getting-started guide.</p></div></main></body></html>')
    if not CLIENT_ID_RE.match(client_id):
        raise ValueError("invalid Google client id")
    script = _JS.replace("%CLIENT_ID%", '"' + client_id + '"')  # the id matches [0-9a-z.-]+, so it cannot break out of the string
    return head + f"""
<h1>Join HiveMind</h1>
<p>Use open coding models from your editor, or share your GPU and earn points. HiveMind points have no cash value.
<a href="/join">How to join</a> · <a href="/rejoindre">Comment participer</a></p>
<section id="signin" class="card">
  <fieldset><legend>What do you want to do?</legend>
    <label><input type="radio" name="role" value="prosumer" checked> Both: use models and share my GPU</label>
    <label><input type="radio" name="role" value="consumer"> Use models in my editor</label>
    <label><input type="radio" name="role" value="producer"> Share my GPU</label>
  </fieldset>
  <div id="gbtn"></div>
  <noscript><p>This page needs JavaScript to show the Google button.</p></noscript>
  <p class="mut">We keep only an anonymous hash of your Google account id. Your e-mail address is not stored.</p>
</section>
<section id="result" class="card" hidden>
  <p id="kind"></p>
  <p>Your API key (shown once; signing in again replaces it):</p>
  <pre><code id="key"></code></pre>
  <p><button id="copy" type="button">Copy key</button></p>
  <p class="mut">Account <code id="acct"></code>, role <code id="role"></code>.</p>
  <p>Next, in a terminal:</p>
  <pre><code>export HIVEMIND_API_KEY=&lt;the key above&gt;
python hivemind_connect.py check
python hivemind_connect.py snippet continue --model coder</code></pre>
  <p class="mut">New accounts hold 0 points. Share a GPU to earn some, or ask the operator for a grant.</p>
</section>
<p id="status" class="mut" role="status"></p>
<script nonce="{nonce}">{script}</script>
<script nonce="{nonce}" src="https://accounts.google.com/gsi/client" async defer></script>
</main></body></html>"""
