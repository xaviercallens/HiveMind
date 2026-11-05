#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Create, rotate or check a HiveMind gateway API key (proof of concept: send only public, non-sensitive data).

    python scripts/hivemind_key.py new --handle my-name [--role consumer]   # a NEW account with a new key; open to anyone; its balance starts low or at 0
    python scripts/hivemind_key.py check                                    # reads the key from HIVEMIND_API_KEY
    HIVEMIND_ADMIN_KEY=... python scripts/hivemind_key.py rotate --account-id <id>   # same account and balance, new key; every earlier key stops working

The web portal (/signup) can create a new account but has no rotate button: rotation is admin-only. A new key is shown once and cannot be recovered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

GATEWAY = "https://hivemind-openrouter-1003063861791.us-central1.run.app"


def call(path: str, *, body: dict | None = None, key: str | None = None, admin: str | None = None, gateway: str = GATEWAY) -> tuple[int, dict]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    if admin:
        headers["x-admin-key"] = admin
    req = urllib.request.Request(gateway + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - fixed https gateway or an explicit --gateway
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except ValueError:
            return exc.code, {}


def summary(key: str, gateway: str) -> str:
    code, me = call("/v1/hivemind/me", key=key, gateway=gateway)
    if code != 200:
        return f"key rejected (HTTP {code})"
    return f"valid: handle={me.get('handle')} role={me.get('role')} balance={me.get('balance')}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default=GATEWAY)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new")
    n.add_argument("--handle", required=True)
    n.add_argument("--role", default="consumer", choices=["consumer", "producer", "prosumer", "trainer"])
    sub.add_parser("check")
    r = sub.add_parser("rotate")
    r.add_argument("--account-id")
    r.add_argument("--handle")
    a = ap.parse_args(argv)
    if a.cmd == "check":
        key = os.environ.get("HIVEMIND_API_KEY", "")
        print(summary(key, a.gateway) if key else "set HIVEMIND_API_KEY")
        return 0
    if a.cmd == "new":
        code, out = call("/v1/hivemind/register", body={"handle": a.handle, "role": a.role}, gateway=a.gateway)
    else:
        admin = os.environ.get("HIVEMIND_ADMIN_KEY", "")
        if not admin or not (a.account_id or a.handle):
            print("rotate needs HIVEMIND_ADMIN_KEY and --account-id or --handle", file=sys.stderr)
            return 2
        target = {"account_id": a.account_id} if a.account_id else {"handle": a.handle}
        code, out = call("/v1/hivemind/admin/accounts/rotate-key", body=target, admin=admin, gateway=a.gateway)
    if code != 200 or "api_key" not in out:
        print(f"refused (HTTP {code}): {(out.get('error') or {}).get('message', out)}", file=sys.stderr)
        return 1
    print("API key (shown once):", out["api_key"])
    print("account:", out["account_id"])
    print(summary(out["api_key"], a.gateway))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
