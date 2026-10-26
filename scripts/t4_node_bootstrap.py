#!/usr/bin/env python3
"""One-time: create the node's BOINC account (through the project's web stack) and its persistent client data directory.

Idempotent. The project server must be up (hivemind-boinc-db / -apache / -daemons). The account authenticator is kept in
``<node>/account.key`` (mode 0600). The client dir has NO app_info.xml: the client downloads the project's signed app versions.
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.client_setup import prepare_data_dir, project_dir_name  # noqa: E402
from hivemind.boinc.real_client.server_stack import Stack  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-dir", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/node"))
    ap.add_argument("--stack-base", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/stack"))
    ap.add_argument("--boinc-src", type=Path, default=Path("/mnt/disks/disk-socrateai-local-1/boinc/src"))
    a = ap.parse_args()
    st = Stack(a.stack_base, a.boinc_src, sys.executable, ROOT / "src")
    client = a.node_dir / "client"
    if (client / f"account_{project_dir_name(st.master_url)}.xml").exists():
        print("already bootstrapped:", client)
        return 0
    if not st.wait_http(15):
        print("the BOINC project web is not answering", file=sys.stderr)
        return 2
    auth = st.create_account(email=f"t4-node-{secrets.token_hex(4)}@local.invalid", name="hivemind-t4-node")
    a.node_dir.mkdir(parents=True, exist_ok=True)
    key = a.node_dir / "account.key"
    key.write_text(auth)
    key.chmod(0o600)
    prepare_data_dir(client, st.master_url, ROOT / "src", authenticator=auth, install_apps=False)
    print("bootstrapped node client:", client)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
