# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Sign in with GitHub, verified by the gateway itself (v3.4).

GitHub OAuth is not OpenID Connect: there is no signed ID token to check. A bare access token proves nothing about *our*
application (a token issued to any other GitHub app would also answer ``/user``), so the gateway never accepts one. The
portal forwards the authorization ``code`` with its PKCE ``code_verifier``; the gateway exchanges it at GitHub with its own
client secret (which only the gateway holds), so the resulting token is ours by construction. The identity is GitHub's
numeric user id (a login can be renamed), and the account must have at least one verified e-mail address.

Configuration (gateway): ``HIVEMIND_GITHUB_CLIENT_ID`` and ``HIVEMIND_GITHUB_CLIENT_SECRET`` (Secret Manager); the only
accepted ``redirect_uri`` is ``<HIVEMIND_PUBLIC_WEB_URL>/auth/github/callback``.
"""
from __future__ import annotations

from dataclasses import dataclass

import httpx

TOKEN_URL = "https://github.com/login/oauth/access_token"
USER_URL = "https://api.github.com/user"
EMAILS_URL = "https://api.github.com/user/emails"


class GitHubAuthError(Exception):
    """``code`` is a short machine reason ("exchange_failed", "no_verified_email", ...); never contains a secret."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class GitHubIdentity:
    subject: str  # numeric GitHub user id, as a string
    login: str


class GitHubCodeVerifier:
    def __init__(self, client_id: str, client_secret: str, redirect_uri: str, http: httpx.Client | None = None) -> None:
        if not client_id or not client_secret:
            raise ValueError("GitHub sign-in needs a client id and a client secret")
        if not redirect_uri.startswith("https://"):
            raise ValueError("GitHub redirect_uri must be https")
        self.client_id = client_id
        self._secret = client_secret
        self.redirect_uri = redirect_uri
        self._http = http or httpx.Client(timeout=10.0, headers={"User-Agent": "HiveMind-Gateway/3.4"})

    def verify_code(self, code: str, code_verifier: str, redirect_uri: str) -> GitHubIdentity:
        if not code or len(code) > 200 or not code_verifier or not (43 <= len(code_verifier) <= 128):
            raise GitHubAuthError("bad_request")
        if redirect_uri != self.redirect_uri:
            raise GitHubAuthError("redirect_mismatch")
        try:
            r = self._http.post(
                TOKEN_URL,
                data={"client_id": self.client_id, "client_secret": self._secret, "code": code,
                      "code_verifier": code_verifier, "redirect_uri": redirect_uri},
                headers={"Accept": "application/json"},
            )
            token_body = r.json() if r.status_code == 200 else {}
        except (httpx.HTTPError, ValueError):
            raise GitHubAuthError("github_unreachable") from None
        token = token_body.get("access_token") if isinstance(token_body, dict) else None
        if not isinstance(token, str) or not token:
            raise GitHubAuthError("exchange_failed")  # bad, replayed or expired code, or wrong verifier
        auth = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        try:
            user = self._http.get(USER_URL, headers=auth)
            emails = self._http.get(EMAILS_URL, headers=auth)
            user_body = user.json() if user.status_code == 200 else {}
            email_body = emails.json() if emails.status_code == 200 else []
        except (httpx.HTTPError, ValueError):
            raise GitHubAuthError("github_unreachable") from None
        uid = user_body.get("id") if isinstance(user_body, dict) else None
        if not isinstance(uid, int) or isinstance(uid, bool) or uid <= 0:
            raise GitHubAuthError("no_user")
        if not isinstance(email_body, list) or not any(isinstance(e, dict) and e.get("verified") is True for e in email_body):
            raise GitHubAuthError("no_verified_email")
        return GitHubIdentity(subject=str(uid), login=str(user_body.get("login", ""))[:64])
