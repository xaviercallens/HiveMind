# HiveMind web portal v3.3 — hardening, French/English, real account panel, intensive tests in CI

Builds on v3.2 (`docs/WEB_V3_2.md`, released as v3.2.0). Nothing here is deployed; the portal still needs the owner's
configuration (§5).

## 1. Security changes
- **One modern password format.** New passwords and registrations use `pbkdf2_sha256$600000$<salt>$<hash>`
  (PBKDF2-HMAC-SHA256, 600,000 iterations, the OWASP 2023 recommendation). Any successful sign-in with a legacy format
  (salted PBKDF2 at 100k–200k, the secret-salted handle hash, legacy credential rows) rewrites the stored hash in the modern
  format, so previous users migrate silently. Modern hashes with an iteration count outside 100k–5M are refused.
- **A password change replaces the API key.** A stolen key or session no longer survives a password change; the portal moves
  the current session to the new key and shows it once.
- **Sandbox SSO deleted** in both the web service and the gateway. It verified nothing with any provider; the gateway copy
  created password-less accounts under a provider label, and let anyone attach a not-yet-registered e-mail to one. Both
  routes answer 410; Google sign-in with a verified ID token is the only SSO. The `HIVEMIND_LEGACY_SANDBOX_SSO` re-enable flag
  is gone.
- **Bugs found by the new tests and fixed:**
  - a `state`, CSRF header or cookie containing a non-ASCII character made `hmac.compare_digest` raise, which was a server
    error; all such comparisons now run on bytes;
  - responses produced by the framework itself (404, 405, 422) on portal paths, and the static files, did not carry the
    strict policy in the full app; they do now;
  - session and OAuth-state expiry used the system clock instead of the injected one (untestable); fixed;
  - a malformed Google client id would have stopped the whole portal at start; it now disables Google only.

## 2. Interface
- French and English. The language comes from `?lang=`, then the `hm_lang` cookie, then `Accept-Language`, and a toggle
  switches it. Every error has a stable code that the page translates (`static/i18n.json`).
- Accessibility: a skip link, the ARIA tab pattern with arrow, Home and End keys, and focus moved to the status message after
  each action. An axe-core scan in Chromium finds no serious or critical issue.
- Account panel: the real balance and the last ten usage records (model, tokens, points, user or volunteer), read
  server-side from the gateway's `/v1/hivemind/me` with the session's key, each with its source.

## 3. Tests

| File | What it proves | Required job | Intense workflow |
|---|---|---|---|
| `tests/test_auth_v33_password_properties.py` | for random accounts in every stored format: the right password opens, others never do, a failure changes nothing stored, a legacy sign-in upgrades the hash, another account's password never opens this one | 6 examples each (deterministic) | 300 examples each |
| `tests/test_web_v33_intense.py` | routes discovered from `app.routes`; strict headers on every response including 404/405/422 in the full dashboard; CSRF and Origin matrix over every state-changing route; fuzzing of every POST with arbitrary JSON (never a 5xx, never a leaked key); non-JSON bodies; any change to the session cookie signs out; a cookie from another secret is refused; the Google callback only redirects inside the site; the Host header never controls a redirect; static checks of the page (no HTML-injection sink, no inline handler, every input labelled, same-origin scripts only) | 30 examples per property | 1,500 per property |
| `tests/test_web_v33_i18n.py` | identical French and English keys; every key used by the page and script exists; every error code, fact and statement translated; language choice; account panel | yes | yes |
| `tests/test_web_v33_browser.py` | Chromium end to end: no CSP violation or console error, facts, French toggle, keyboard tabs, sign-in with balance and usage, password change shows the new key; axe-core scan | skipped (no browser) | yes |
| Bandit (medium and above) on `src/hivemind/web_v32` and the new auth modules | static security scan | — | yes |

The intense workflow is `.github/workflows/web_auth_intense.yml`. It runs on pull requests that touch the portal or the
account code, on pushes to `main`, nightly at 02:17 UTC, and on demand. It is read-only and uses no secrets.

## 4. Owner decision
Making the intense workflow a **required** check is a branch-protection change: the owner adds its three job names under
the `main` protection rule if wanted. Until then it reports on every relevant PR without blocking the merge.

## 5. Still needed to switch the portal on
Unchanged from v3.2: `HIVEMIND_WEB_SESSION_SECRET` (Secret Manager), `HIVEMIND_PUBLIC_WEB_URL=https://hivemind-at-home-1003063861791.us-central1.run.app`,
Google OAuth client id and secret, SMTP settings, then `HIVEMIND_WEB_V32=1` after testing `/v32` on a tagged revision.
