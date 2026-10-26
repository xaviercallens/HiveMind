#!/usr/bin/env python3
"""Create / start / stop a real BOINC server (built from source) on loopback for the HiveMind QLoRA MoE apps.

Prerequisites (see docs/BOINC_PRODUCTION_SERVER.md): BOINC server built in --boinc-src, a private MariaDB listening on
/run/mysqld/mysqld.sock, apt packages apache2 + libapache2-mod-php + php-mysql, the venv with peft + bitsandbytes.

    python3 scripts/boinc_server_stack.py setup --base .../boinc/stack --boinc-src .../boinc/src --venv-python .../venv/bin/python
    python3 scripts/boinc_server_stack.py start --base ... (same flags)
    python3 scripts/boinc_server_stack.py stop  --base ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.server_stack import Stack, launcher  # noqa: E402

APPS = {
    "hivemind_qlora_moe": {"args": ["--mode", "train"], "n_inputs": 1, "outputs": ["out.json", "out.pt"], "max_nbytes": 200_000_000},
    "hivemind_moe_infer": {"args": ["--mode", "infer"], "n_inputs": 2, "outputs": ["out.json"], "max_nbytes": 5_000_000},
}


def shim(stack: Stack, cmd: str, venv_python: str) -> str:
    return (
        "#!/bin/bash\nunset LD_LIBRARY_PATH\n"
        f"export PYTHONPATH={ROOT / 'src'}\nexport HF_HUB_OFFLINE=1\n"
        f"export HIVEMIND_SPEC_DIR={stack.root}/hivemind_specs\nexport HIVEMIND_LEDGER_DIR={stack.root}/hivemind_ledger\n"
        f"exec {venv_python} -W ignore -m hivemind.boinc.real_client.boinc_server_scripts {cmd} \"$@\"\n"
    )


def setup(stack: Stack, venv_python: str) -> None:
    stack.make_project()
    stack.write_apache()
    stack.write_plan_class_spec()
    stack.fix_scheduler_list()
    for d in ("hivemind_specs", "hivemind_ledger"):
        (stack.root / d).mkdir(exist_ok=True)
    for name, cmd in (("hivemind_validate", "validate"), ("hivemind_assimilate", "assimilate")):
        p = stack.root / "bin" / name
        p.write_text(shim(stack, cmd, venv_python))
        p.chmod(0o755)
    for app, spec in APPS.items():
        body = launcher(venv_python, ROOT / "src", "hivemind.boinc.real_client.moe_app", spec["args"], {"HF_HUB_OFFLINE": "1"})
        stack.install_app(app, body, spec["n_inputs"], spec["outputs"], spec["max_nbytes"])
    stack.configure("hivemind_validate", "hivemind_assimilate", list(APPS))
    stack.register_apps(list(APPS))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["setup", "start", "stop", "apps"])
    ap.add_argument("--version", type=int, default=101, help="app version to publish with the 'apps' action")
    ap.add_argument("--code-src", type=Path, default=Path("/opt/hivemind/t4/src"), help="stable code path baked into the 'apps' launchers")
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--boinc-src", type=Path, required=True)
    ap.add_argument("--venv-python", required=True)
    ap.add_argument("--port", type=int, default=8081)
    a = ap.parse_args()
    st = Stack(a.base, a.boinc_src, a.venv_python, ROOT / "src", a.port)
    if a.action == "setup":
        setup(st, a.venv_python)
        print("setup done:", st.master_url)
    elif a.action == "apps":
        # publish a new app version whose launcher points at a stable code path (not an ephemeral git worktree); older versions are deprecated
        for app, spec in APPS.items():
            body = launcher(a.venv_python, a.code_src, "hivemind.boinc.real_client.moe_app", spec["args"], {"HF_HUB_OFFLINE": "1"})
            st.install_app(app, body, spec["n_inputs"], spec["outputs"], spec["max_nbytes"], version=a.version)
        st.ensure_chain_template()
        print(st.update_versions())
        st.deprecate_versions_below(a.version)
        print("published app version", a.version, "-> code at", a.code_src)
    elif a.action == "start":
        print("db up:", st.start_db())
        st.start_apache()
        print("apache up:", st.wait_http(30))
        print(st.start_daemons())
    else:
        st.stop_daemons()
        st.stop_apache()
        st.stop_db()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
