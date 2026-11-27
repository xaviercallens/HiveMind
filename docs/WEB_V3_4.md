# Web portal v3.4: Google sign-in and single sign-on

v3.4 replaces the Google-only sign-in flow with one flow for every provider. A provider button appears only when that
provider is fully configured on the portal **and** the gateway reports it in `/api/v1/auth/capabilities`, so the page
never offers a sign-in that would fail. With no configuration the portal behaves exactly like v3.3 (password sign-in
only). The names `google` and `github` are reserved: an OpenID Connect entry using either is refused by the gateway and
ignored by the portal.

| Provider | How it is verified | Configured with |
|---|---|---|
| Google | ID token checked by the portal and again by the gateway (RS256, issuer, audience, expiry, nonce, verified e-mail, single use) | `HIVEMIND_GOOGLE_CLIENT_ID` (both services), `HIVEMIND_GOOGLE_CLIENT_SECRET` (portal, Secret Manager) |
| Any OpenID Connect provider (Microsoft Entra ID, Okta, GitLab, Keycloak...) | Same checks, with the gateway's existing `OidcIdTokenVerifier` | `HIVEMIND_OIDC_PROVIDERS` (same JSON on both services), `HIVEMIND_OIDC_<NAME>_CLIENT_SECRET` (portal) |
| GitHub | GitHub has no ID token. The gateway exchanges the authorization code itself with its own client secret, so it never accepts a bare access token. The identity is the numeric user id, and the account needs a verified e-mail | `HIVEMIND_GITHUB_CLIENT_ID` (both services), `HIVEMIND_GITHUB_CLIENT_SECRET` (gateway, Secret Manager), `HIVEMIND_PUBLIC_WEB_URL` (gateway) |

Every flow uses the authorization code with PKCE (S256) and a `state` value sealed in an encrypted, HttpOnly,
path-scoped cookie; OpenID Connect flows also use a `nonce`. GitHub has supported PKCE for OAuth Apps since July 2025.

## Existing password accounts

Signing in with a provider for the first time opens a **new, separate** account. To reach an existing password account
with a provider, sign in with the password first, then use "Link my ... account" on the account page
(`/api/v1/auth/link/{provider}`, which requires the account's key). Identities are keyed on the provider's subject, never
on the e-mail address, so a provider that does not verify e-mail cannot take over an account.

## Microsoft Entra ID

Only tenant-specific issuers are accepted (`https://login.microsoftonline.com/<tenant-id>/v2.0`). The multi-tenant
`common`, `organizations` and `consumers` endpoints are refused at start-up: their tokens carry each user's own tenant,
and accepting any tenant would accept every Microsoft account. Entra tokens usually omit `email_verified`, so set
`"require_verified_email": false` for it. Example entry:

```json
[{"name": "entra", "label": "Microsoft",
  "issuer": "https://login.microsoftonline.com/<tenant-id>/v2.0",
  "jwks_url": "https://login.microsoftonline.com/<tenant-id>/discovery/v2.0/keys",
  "authorization_endpoint": "https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/authorize",
  "token_endpoint": "https://login.microsoftonline.com/<tenant-id>/oauth2/v2.0/token",
  "client_id": "<application-id>", "require_verified_email": false}]
```

The gateway refuses to start if an existing account handle already uses a provider's name as its prefix
(`<name>.`); a no-traffic canary shows this before any traffic moves.

## Owner steps to turn on Google sign-in

1. Google Cloud Console, project `gen-lang-client-0625573011`, **APIs & Services → OAuth consent screen**: user type
   External, app name "HiveMind", support e-mail, scopes `openid`, `email`, `profile`. Publish it ("In production");
   in "Testing" only listed test users can sign in. These three scopes are non-sensitive, so Google normally requires no
   app verification for them (adding a logo can trigger brand verification).
2. **APIs & Services → Credentials → Create credentials → OAuth client ID**, type *Web application*:
   - Authorized JavaScript origin: `https://hivemind-at-home-1003063861791.us-central1.run.app`
   - Authorized redirect URI: `https://hivemind-at-home-1003063861791.us-central1.run.app/auth/google/callback`
3. Store the client secret in Secret Manager as `hivemind-google-client-secret`, and grant
   `1003063861791-compute@developer.gserviceaccount.com` the *Secret Manager Secret Accessor* role on it.
4. Send the client ID (it is public). The deploy then sets `HIVEMIND_GOOGLE_CLIENT_ID` on both services, mounts the
   secret on the portal, and deploys both as no-traffic canaries first.
5. Do one real Google sign-in yourself: automated checks can verify the redirect, but not a completed Google login.

## Owner steps to turn on GitHub sign-in (optional)

GitHub → Settings → Developer settings → **OAuth Apps → New OAuth App**: homepage
`https://hivemind-at-home-1003063861791.us-central1.run.app`, callback
`https://hivemind-at-home-1003063861791.us-central1.run.app/auth/github/callback`. Store the client secret as
`hivemind-github-client-secret` with the same accessor grant, and send the client ID.

## Tests

`tests/test_sso_v34.py` runs the production verifiers, not fakes: tokens are signed with a test RSA key, and the
attacks covered are wrong issuer, wrong audience, expired token, `alg: none`, HS256 signed with the public key, unknown
key id, a different signing key, unverified e-mail, wrong nonce, replay, tenant-wildcard issuers, a state cookie moved
between providers, a bare GitHub access token, a replayed GitHub code, a foreign redirect URI and GitHub outages.
