#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Turnkey CLI for RTX BOINC Server & Client Deployment (`scripts/hivemind_rtx_boinc_deploy.py`).

Subcommands:
  - `detect`        : Probe NVIDIA RTX GPU, VRAM headroom, and BOINC plan class (`cuda_hivemind` / `cuda_hivemind_big`).
  - `server`        : Launch the zero-MariaDB SQLite RTX BOINC Server (BOINC XML RPC + JSON/HTTPS Workunit API).
  - `deploy-client` : Generate cross-platform BOINC client XMLs (`cc_config.xml`, `global_prefs_override.xml`,
                      `app_info.xml`, `app_config.xml`, `account_*.xml`) + Windows/Linux service units.
  - `client`        : Run the RTX BOINC CUDA worker (`--once` or continuous idle-GPU loop).
  - `self-test`     : End-to-end verification of Server + Client + CUDA PTX/QLoRA + SQLite credit on the local RTX GPU.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for p in (str(SRC_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivemind.boinc.rtx_boinc_deployer import (
    RTXBoincServer,
    RTXWorkunitLeaseRequest,
    RTXWorkunitSubmitRequest,
    build_rtx_boinc_fastapi_router,
    deploy_rtx_boinc_client_files,
    detect_rtx_boinc_profile,
    execute_rtx_boinc_workunit,
    run_rtx_boinc_client_cycle,
)
from hivemind.config import ARTIFACTS_ROOT
from hivemind.router.accounts import RouterStore


def cmd_detect(args: argparse.Namespace) -> int:
    prof = detect_rtx_boinc_profile()
    out = prof.to_dict()
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if (prof.gpu_detected or args.allow_cpu) else 1


def cmd_server(args: argparse.Namespace) -> int:
    import uvicorn

    router_db = Path(args.router_db) if args.router_db else (ARTIFACTS_ROOT / "local_portal_router.db")
    store = RouterStore(
        router_db,
        secret=os.environ.get("HIVEMIND_ROUTER_SECRET", "hivemind-local-portal-secret-v1-32bytes-minimum").encode("utf-8"),
    )
    master_url = args.master_url or f"http://{args.host}:{args.port}/"
    boinc_srv = RTXBoincServer(
        db_path=Path(args.db_path) if args.db_path else (ARTIFACTS_ROOT / "rtx_boinc_server.db"),
        boinc_ledger_dir=Path(args.ledger_dir) if args.ledger_dir else (ARTIFACTS_ROOT / "boinc_ledger"),
        router_store=store,
        master_url=master_url,
        require_cuda=not args.allow_cpu,
    )
    app = FastAPI(title="HiveMind@home Turnkey RTX BOINC Server", version="2.13.0")
    app.include_router(build_rtx_boinc_fastapi_router(boinc_srv))

    @app.get("/healthz")
    async def healthz() -> dict[str, Any]:
        return boinc_srv.status_summary(limit=5)

    print(
        f"[RTX-BOINC-Server] Listening on http://{args.host}:{args.port} "
        f"(DB: {boinc_srv.db_path}, RouterStore: {router_db})",
        flush=True,
    )
    uvicorn.run(app, host=args.host, port=int(args.port), log_level="info")
    return 0


def cmd_deploy_client(args: argparse.Namespace) -> int:
    install_dir = Path(args.data_dir).expanduser().resolve()
    manifest = deploy_rtx_boinc_client_files(
        install_dir,
        server_url=args.server,
        handle=args.handle,
        node_id=args.node_id,
        authenticator=args.authenticator,
        games=args.game or None,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def cmd_client(args: argparse.Namespace) -> int:
    from scripts.hivemind_boinc_idle_worker import get_user_idle_seconds

    prof = detect_rtx_boinc_profile()
    node_id = args.node_id or f"rtx-{platform.node() or 'node'}-{prof.plan_class}"
    while True:
        idle_s = get_user_idle_seconds()
        if not args.force and idle_s < float(args.idle_threshold_seconds):
            print(
                f"[RTX-BOINC-Client] User active (idle={idle_s:.1f}s < {args.idle_threshold_seconds}s); "
                "pausing RTX GPU workunit lease.",
                flush=True,
            )
        else:
            try:
                res = run_rtx_boinc_client_cycle(
                    server_url=args.server,
                    handle=args.handle,
                    node_id=node_id,
                    preferred_domain=args.domain,
                    device=args.device,
                )
                print(json.dumps(res, indent=2, sort_keys=True), flush=True)
            except Exception as exc:
                print(f"[RTX-BOINC-Client] Cycle error: {exc}", file=sys.stderr, flush=True)
                if args.once:
                    return 1
        if args.once:
            break
        time.sleep(max(5.0, float(args.poll_seconds)))
    return 0


def cmd_self_test(args: argparse.Namespace) -> int:
    prof = detect_rtx_boinc_profile()
    router_db = ARTIFACTS_ROOT / "local_portal_router.db"
    store = RouterStore(
        router_db,
        secret=os.environ.get("HIVEMIND_ROUTER_SECRET", "hivemind-local-portal-secret-v1-32bytes-minimum").encode("utf-8"),
    )
    boinc_srv = RTXBoincServer(
        db_path=ARTIFACTS_ROOT / "rtx_boinc_server.db",
        boinc_ledger_dir=ARTIFACTS_ROOT / "boinc_ledger",
        router_store=store,
        master_url="http://127.0.0.1:8095/",
        require_cuda=(args.device == "cuda"),
    )
    app = FastAPI()
    app.include_router(build_rtx_boinc_fastapi_router(boinc_srv))
    client = TestClient(app)

    # 1. Verify BOINC XML RPC endpoints
    cfg_resp = client.get("/get_project_config.php")
    assert cfg_resp.status_code == 200 and "<project_config>" in cfg_resp.text
    acct_resp = client.get("/create_account.php?email_addr=rtx2070@hivemind.local&user_name=xaviercallens-rtx2070")
    assert acct_resp.status_code == 200 and "<authenticator>" in acct_resp.text
    sched_xml = (
        "<scheduler_request>"
        "<hostid>2070</hostid>"
        "<platform_name>windows_x86_64</platform_name>"
        "<work_req_seconds>60</work_req_seconds>"
        "<gpu_req_seconds>60</gpu_req_seconds>"
        f"<coprocs><coproc_cuda><name>{prof.gpu_name}</name>"
        f"<available_ram>{prof.vram_total_mib * 1024 * 1024}</available_ram></coproc_cuda></coprocs>"
        "</scheduler_request>"
    )
    sched_resp = client.post("/boinc_cgi/cgi", content=sched_xml.encode("utf-8"))
    assert sched_resp.status_code == 200 and "<scheduler_reply>" in sched_resp.text and "<workunit>" in sched_resp.text

    # 2. Deploy client files into artifacts/rtx_boinc_client_bundle
    client_dir = ARTIFACTS_ROOT / "rtx_boinc_client_bundle"
    manifest = deploy_rtx_boinc_client_files(
        client_dir,
        server_url="http://127.0.0.1:8095",
        handle=args.handle,
        node_id=f"rtx2070-{platform.node() or 'XAVGAMSRV'}",
        profile=prof,
    )

    # 3. Lease, execute on CUDA RTX GPU, and submit via REST API
    lease_resp = client.post(
        "/api/v1/boinc/rtx/lease",
        json={
            "handle": args.handle,
            "node_id": f"rtx2070-{platform.node() or 'XAVGAMSRV'}",
            "gpu_name": prof.gpu_name,
            "vram_total_mib": prof.vram_total_mib,
            "vram_free_mib": prof.vram_free_mib,
            "preferred_domain": "code",
            "workunit_kind": "qlora_moe",
        },
    )
    assert lease_resp.status_code == 200
    spec = lease_resp.json()
    exec_out = execute_rtx_boinc_workunit(spec, device=args.device)
    sub_resp = client.post(
        "/api/v1/boinc/rtx/submit",
        json={
            **exec_out,
            "handle": args.handle,
            "node_id": f"rtx2070-{platform.node() or 'XAVGAMSRV'}",
        },
    )
    assert sub_resp.status_code == 200
    verdict = sub_resp.json()
    status_out = client.get("/api/v1/boinc/rtx/status").json()

    report = {
        "ok": bool(verdict.get("valid")),
        "timestamp": time.time(),
        "hardware_profile": prof.to_dict(),
        "boinc_xml_rpc_verified": True,
        "client_deployment_manifest": manifest,
        "workunit_execution": exec_out,
        "server_verdict": verdict,
        "server_status_summary": status_out,
    }
    out_path = ARTIFACTS_ROOT / "rtx_boinc_deployment_verification.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Turnkey RTX BOINC Server & Client Deployment CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_det = sub.add_parser("detect", help="Detect NVIDIA RTX GPU and BOINC plan class")
    p_det.add_argument("--allow-cpu", action="store_true", help="Exit 0 even if no NVIDIA GPU is detected")

    p_srv = sub.add_parser("server", help="Start the SQLite-backed RTX BOINC Server")
    p_srv.add_argument("--host", default="0.0.0.0")
    p_srv.add_argument("--port", type=int, default=8097)
    p_srv.add_argument("--master-url", default="")
    p_srv.add_argument("--db-path", default="")
    p_srv.add_argument("--ledger-dir", default="")
    p_srv.add_argument("--router-db", default="")
    p_srv.add_argument("--allow-cpu", action="store_true")

    for name in ("deploy-client", "install-client"):
        p_dep = sub.add_parser(name, help="Generate BOINC client configs, app_info.xml, and OS service scripts")
        p_dep.add_argument("--data-dir", default=str(Path.home() / ".hivemind_boinc"))
        p_dep.add_argument("--server", default="http://127.0.0.1:8095")
        p_dep.add_argument("--handle", default="xaviercallens-rtx2070")
        p_dep.add_argument("--node-id", default="")
        p_dep.add_argument("--authenticator", default="hm_boinc_weak_key_v213_open_science")
        p_dep.add_argument("--game", action="append", default=[])

    for name in ("client", "run-worker"):
        p_cli = sub.add_parser(name, help="Run the RTX BOINC CUDA workunit worker")
        p_cli.add_argument("--server", default="http://127.0.0.1:8095")
        p_cli.add_argument("--handle", default="xaviercallens-rtx2070")
        p_cli.add_argument("--node-id", default="")
        p_cli.add_argument("--domain", default=None)
        p_cli.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
        p_cli.add_argument("--idle-threshold-seconds", type=float, default=4.0)
        p_cli.add_argument("--poll-seconds", type=float, default=30.0)
        p_cli.add_argument("--force", action="store_true", help="Run even if user input is active")
        p_cli.add_argument("--once", action="store_true", help="Execute a single workunit and exit")

    p_test = sub.add_parser("self-test", help="Run full RTX BOINC Server + Client + CUDA verification")
    p_test.add_argument("--handle", default="xaviercallens-rtx2070")
    p_test.add_argument("--device", default="cuda", choices=("auto", "cuda", "cpu"))

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "detect":
        return cmd_detect(args)
    if args.cmd == "server":
        return cmd_server(args)
    if args.cmd in ("deploy-client", "install-client"):
        return cmd_deploy_client(args)
    if args.cmd in ("client", "run-worker"):
        return cmd_client(args)
    if args.cmd == "self-test":
        return cmd_self_test(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
