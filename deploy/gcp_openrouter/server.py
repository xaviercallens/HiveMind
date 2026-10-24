"""Cloud Run entrypoint for the HiveMind OpenRouter gateway.

Configuration (environment):
    HIVEMIND_ROUTER_SECRET          required; HMAC secret for API keys and account ids (>= 32 bytes).
                                    Mount from Secret Manager, never bake into the image.
    HIVEMIND_ROUTER_DB              SQLite path, default /tmp/router.db. On Cloud Run /tmp is
                                    ephemeral: ledger and accounts vanish on instance restart.
                                    Production needs Cloud SQL or Firestore (task HM-S13-03).
    HIVEMIND_ROUTER_ADMIN_KEY_HASH  optional; HMAC hex matching the admin key, enables admin grants.
    HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS
                                    reverse proxies in front of the app that append to X-Forwarded-For (Cloud Run: 1).
                                    The registration limiter keys on the entry that many hops from the right, which a
                                    client cannot forge. 0 (default) keys on the TCP peer.
    HIVEMIND_GOOGLE_CLIENT_ID       optional; the OAuth web client id (a public identifier, not a secret) that turns on
                                    "Sign in with Google" registration at /signup. Unset = disabled.
    HIVEMIND_OPEN_REGISTRATION      "0" closes handle-only registration so that Google sign-in is the only way in.
                                    Default "1". Closing it without a Google client id is refused at start-up.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from fastapi import FastAPI

from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app

DEFAULT_DB_PATH: str = "/tmp/router.db"
MEMORY_DB: str = ":memory:"


def build_app(env: Mapping[str, str]) -> FastAPI:
    """Construct the gateway app from environment variables. Fails closed when the secret is absent."""
    secret_text = env.get("HIVEMIND_ROUTER_SECRET", "")
    if not secret_text:
        raise RuntimeError("HIVEMIND_ROUTER_SECRET is not set; refusing to start without it")
    db_setting = env.get("HIVEMIND_ROUTER_DB", DEFAULT_DB_PATH)
    db_target: Path | str = db_setting if db_setting == MEMORY_DB else Path(db_setting)
    admin_key_hash = env.get("HIVEMIND_ROUTER_ADMIN_KEY_HASH", "") or None
    store = RouterStore(db_target, secret_text.encode("utf-8"))
    hops = int(env.get("HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS", "0") or "0")
    if hops < 0:
        raise RuntimeError("HIVEMIND_ROUTER_TRUSTED_PROXY_HOPS must be >= 0")
    google_client_id = env.get("HIVEMIND_GOOGLE_CLIENT_ID", "").strip() or None
    open_registration = env.get("HIVEMIND_OPEN_REGISTRATION", "1").strip() != "0"
    if not open_registration and google_client_id is None:
        raise RuntimeError("HIVEMIND_OPEN_REGISTRATION=0 needs HIVEMIND_GOOGLE_CLIENT_ID: nobody could register otherwise")
    return create_router_app(store, admin_key_hash=admin_key_hash, trusted_proxy_hops=hops,
                             google_client_id=google_client_id, open_registration=open_registration)


app: FastAPI = build_app(os.environ)
