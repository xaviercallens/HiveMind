# HiveMind web portal v3.2 — sign-in that works, enterprise-grade security, only real claims

Status: **proposal on branch `feat/web-ui-v3.2`, not deployed.** Built on the hotfix branch `fix/gateway-password-check-scope`.

## 1. Review findings (2026-11-07)

Reviewed: the web service `hivemind-at-home` (code `src/hivemind/boinc/hivemind_at_home_dashboard.py`, 12,342 lines, built from
`deploy/gcp_cloud_run/`), the gateway `hivemind-openrouter` (`src/hivemind/router/accounts.py`, `gateway.py`), and the live
services with read-only requests.

| # | Severity | Finding | Where | v3.2 |
|---|---|---|---|---|
| 1 | Critical | The gateway's password check accepted evidence not bound to the account: a special case for one handle that accepted any 8-character password while its hash was a placeholder (and rewrote the hash), a table-wide fallback over legacy credential rows, and caller-supplied e-mails selecting credentials. Introduced by `316c106`, live from revision `00067` | `accounts.py` `_verify_row_password` | fixed in the hotfix commit, with tests; recovery route `change-password` |
| 2 | High | "SSO" is a sandbox: no provider verifies anything, the server returns the PKCE `code_verifier` to the browser, and without an existing account any caller gets a new account with 2,500 points labelled as a Google sign-in | dashboard `/api/v1/auth/sso/*` | retired (410); real Google sign-in added |
| 3 | High | Web accounts are stored in `/tmp/ruche_users_auth.json` inside the container: lost on every restart and scale-to-zero, and different on each of the two instances | dashboard, env `HIVEMIND_RUCHE_AUTH_FILE` | v3.2 keeps no user file; the gateway is the only account store |
| 4 | High | No password reset; no e-mail delivery configured anywhere | gateway env | reset by e-mail added (needs SMTP); honest "not available" until then |
| 5 | Medium | Portal CSP allows `'unsafe-inline'` scripts and a third-party CDN; API keys are handed to page JavaScript | dashboard `portal_csp` | v3.2 page: nonce CSP, no inline code, key never in the browser for a session |
| 6 | Medium | Seeded fictional users (for example `elena@berkeley.edu`, `gamer_pro@steam.federation.hivemind`) and demo globe data on public pages; training telemetry block marked `NO_TELEMETRY_RECEIVED` still shows improving numbers | dashboard, gateway `/v1/hivemind/live-telemetry` `t4_boinc` | v3.2 shows only sourced live counts; the `t4_boinc` block is not shown |
| 7 | Medium | Behind the portal, every visitor reached the gateway from the portal's own address and would have shared one rate-limit bucket. (The gateway image itself sets `HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS=1`, so direct callers are keyed correctly.) | portal → gateway | signed forwarded client address (§2) |
| 8 | Low | Guest and upgrade-guest endpoints still issue points without verification; legacy portal remains reachable | dashboard | not changed in v3.2 (listed in §5) |

Exposure check for finding 1 (Cloud Logging, read-only, every POST to the gateway since the vulnerable revision went live at
16:52 UTC, 1,133 requests): the only requests that reach the password check are two successful logins on
`/v1/hivemind/auth/login`, at 17:36 and 19:56 UTC, both from the same household connection and the same browser. No form, register, SSO or guest route was called. Request bodies are not logged, so the handle used
is unknown; the owner should confirm both were theirs.

**Interim step before any deploy (owner only):** while the vulnerable code is live, signing in once to the owner account with a
new strong password replaces the placeholder hash, which closes path 1 for that account. Paths 2 and 3 stay open until the
hotfix is deployed.

## 2. What v3.2 changes

**Gateway (single account store)**
- Password check bound to the account (hotfix). Previous users keep their password in every stored format: salted PBKDF2 at
  100k, 120k and 200k iterations, the secret-salted handle hash, and legacy credential rows tied to the account id or its own
  e-mail (tests).
- `POST /api/v1/auth/change-password` (needs the account's key).
- `POST /api/v1/auth/password-reset/request` (identical answer whether or not the account exists; per-IP and per-e-mail limits)
  and `/confirm` (256-bit single-use token stored only as an HMAC, 30-minute expiry, revokes the previous key).
- `POST /api/v1/auth/link/google` (link a verified Google identity to an existing account, with that account's key).
- `GET /api/v1/auth/capabilities` (only the methods that really work).
- `password_reset_email.py`: SMTP over TLS only; the token is in the URL fragment, so it never reaches a log or a Referer.

**Web (`src/hivemind/web_v32/`)**
- One sober page (`/v32`, and `/`, `/auth`, `/login`, `/signup` when `HIVEMIND_WEB_V32=1`): what HiveMind is, the privacy
  notice, live facts with their sources, and the account panel. No demo data, no invented numbers, no provider that is not
  verified.
- Sessions: encrypted, authenticated cookie (Fernet), HttpOnly, Secure, SameSite=Lax, 12 h; the API key stays server-side.
  Fails closed (503) without `HIVEMIND_WEB_SESSION_SECRET`.
- CSRF: double-submit token plus Origin check on every state-changing request.
- Google: authorization-code flow with PKCE S256, `state` and `nonce` sealed in an encrypted cookie, code exchanged server-side
  with the client secret, ID token verified (signature, issuer, audience, expiry, `email_verified`, nonce) here and again by
  the gateway.
- Generic login error (no account enumeration); strict CSP (`script-src 'self' 'nonce-…'`, no `unsafe-inline`); the legacy
  header middleware no longer overwrites it.
- Per-visitor rate limits at the gateway: the portal forwards the visitor's address in `X-HiveMind-Forwarded-Client`, signed
  with HMAC-SHA256 under the secret both services share (`HIVEMIND_ROUTER_SECRET`) and valid 60 s; the gateway uses it only
  when it verifies. Without it every visitor would share the portal's bucket and one attacker could lock everyone out.
- One origin: any v3.2 request on another host is redirected (GET, 301) or refused (POST, 421) to `HIVEMIND_PUBLIC_WEB_URL`,
  because cookies, the CSRF Origin check and the Google redirect depend on a single origin. Owner decision 2026-11-07:
  `HIVEMIND_PUBLIC_WEB_URL=https://hivemind-at-home-1003063861791.us-central1.run.app` for now.
- Reset reaches previous users whose e-mail exists only in the legacy credentials table or the known-legacy list.
- Next to the Google button: users with a password account sign in first and link Google, instead of opening a new account.

**Tests:** `tests/test_gateway_password_check_scope.py` (hotfix), `tests/test_auth_v32_gateway.py`, `tests/test_web_v32.py`,
and `tests/test_web_v32_end_to_end.py`, which wires the portal to the real gateway in process (register, login, change
password, reset by e-mail, login again; a legacy-credentials user resets; two visitors get separate rate-limit buckets; forged
or stale forwarded headers are ignored; non-canonical hosts are redirected). Five legacy tests that exercise the retired
sandbox SSO now opt in with `HIVEMIND_LEGACY_SANDBOX_SSO=1`; one owner-account test accepts 410. Results on the branch: 52
passed in the four v3.2 files; 459 passed / 1 skipped in the gateway, router and auth files; 187 passed in the dashboard and
portal files with one failure (`test_ip203_stage9…`) that fails identically without these changes.

## 3. Configuration only the owner can provide

| Variable | Service | Secret? | Purpose |
|---|---|---|---|
| `HIVEMIND_WEB_SESSION_SECRET` (>= 32 random bytes) | web | yes (Secret Manager) | session and OAuth-state encryption; without it sign-in is closed |
| `HIVEMIND_PUBLIC_WEB_URL` (`https://…`) | web and gateway | no | origin for CSRF checks, Google redirect URI, links in reset e-mails |
| `HIVEMIND_GOOGLE_CLIENT_ID` | web and gateway | no | Google sign-in (OAuth web client) |
| `HIVEMIND_GOOGLE_CLIENT_SECRET` | web | yes (Secret Manager) | server-side code exchange |
| `HIVEMIND_SMTP_HOST`, `_PORT`, `_USER`, `_PASSWORD`, `_FROM` | gateway | password yes | reset e-mails |
| `HIVEMIND_WEB_V32=1` | web | no | serve v3.2 at `/`, `/auth`, `/login`, `/signup` |

Google console: authorised redirect URI `https://<public web origin>/auth/google/callback`.

## 4. Deployment plan (owner go required at each step)

1. Hotfix first, alone: **done 2026-11-07** on the owner's go. Revision `hivemind-openrouter-00081-qom` (commit `c7d2fa2`,
   PR #733) deployed as `--no-traffic --tag fix`; read-only checks on the tagged URL and then on the public URL (random
   passwords on the owner's handle and e-mail refused, change-password refuses a missing or fake key); traffic shifted from
   `00069-hz7`. Still to do by the owner: rotate their key with the admin route and set a password with `change-password`.
2. v3.2 gateway routes: same canary procedure; set the variables of §3 with `--update-env-vars` (never `--set-env-vars`).
3. v3.2 web: deploy with `HIVEMIND_WEB_V32` unset and test `/v32` on a tagged revision; then set `HIVEMIND_WEB_V32=1`.
4. Rollback at each step: shift traffic back to the recorded revision.

## 5. Not in v3.2 (deliberately)

- The legacy portal pages and their demo content remain reachable at their old paths until the owner retires them; v3.2
  replaces the entry points only.
- The legacy dashboard auth (guest accounts with points, the `/tmp` user file, the sandbox SSO code behind its off-by-default
  flag) is not deleted; it should be removed once v3.2 has run for a week.
- Reddit, Steam and GitHub sign-in: not offered until each is verified as Google is (GitHub has no ID token and needs a
  verified-e-mail userinfo check).
- Rehashing every password to a stronger format on next sign-in (scrypt or PBKDF2 at 600k) is the next step after the hotfix.
