# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit and integration tests for turnkey RTX BOINC Server & Client deployment."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from hivemind.boinc.rtx_boinc_deployer import (
    PLAN_CLASS_BIG,
    PLAN_CLASS_CPU,
    PLAN_CLASS_STANDARD,
    PLATFORM_LINUX,
    PLATFORM_WINDOWS,
    RTXBoincServer,
    RTXWorkunitLeaseRequest,
    RTXWorkunitSubmitRequest,
    build_rtx_boinc_fastapi_router,
    deploy_rtx_boinc_client_files,
    detect_rtx_boinc_profile,
    execute_rtx_boinc_workunit,
)
from hivemind.router import RouterStore, create_router_app


def test_detect_rtx_boinc_profiles_across_8gb_12gb_16gb_and_cpu() -> None:
    # 1. 8 GB RTX 2070 on Windows -> cuda_hivemind, WDDM ceiling <= 6656 MiB
    p_2070 = detect_rtx_boinc_profile(
        override_vram_total_mib=8192,
        override_vram_free_mib=6800,
        override_gpu_name="NVIDIA GeForce RTX 2070",
        sys_platform="win32",
    )
    assert p_2070.gpu_detected is True
    assert p_2070.boinc_platform == PLATFORM_WINDOWS
    assert p_2070.plan_class == PLAN_CLASS_STANDARD
    assert p_2070.app_name == "hivemind_qlora_moe"
    assert p_2070.wddm_ceiling_mib == 6656
    assert p_2070.min_free_vram_mib == 1536
    assert p_2070.qlora_rank == 8
    assert p_2070.vram_headroom_ok is True

    # 2. 12 GB RTX 4070 on Linux -> cuda_hivemind_big, rank 16
    p_4070 = detect_rtx_boinc_profile(
        override_vram_total_mib=12288,
        override_vram_free_mib=10500,
        override_gpu_name="NVIDIA GeForce RTX 4070",
        sys_platform="linux",
    )
    assert p_4070.boinc_platform == PLATFORM_LINUX
    assert p_4070.plan_class == PLAN_CLASS_BIG
    assert p_4070.app_name == "hivemind_qlora_moe_big"
    assert p_4070.qlora_rank == 16
    assert p_4070.min_free_vram_mib == 2560
    assert p_4070.vram_headroom_ok is True

    # 3. 16 GB Tesla T4 -> enforces >= 4300 MiB free VRAM floor
    p_t4 = detect_rtx_boinc_profile(
        override_vram_total_mib=15360,
        override_vram_free_mib=5000,
        override_gpu_name="Tesla T4",
        sys_platform="linux",
    )
    assert p_t4.plan_class == PLAN_CLASS_BIG
    assert p_t4.min_free_vram_mib == 4300
    assert p_t4.vram_headroom_ok is True

    # 4. CPU fallback
    p_cpu = detect_rtx_boinc_profile(
        override_vram_total_mib=0,
        override_vram_free_mib=0,
        override_gpu_name="CPU-Only",
        sys_platform="linux",
    )
    assert p_cpu.gpu_detected is False
    assert p_cpu.plan_class == PLAN_CLASS_CPU


def test_deploy_rtx_boinc_client_files_writes_valid_xmls_and_launchers(tmp_path: Path) -> None:
    prof = detect_rtx_boinc_profile(
        override_vram_total_mib=8192,
        override_vram_free_mib=6800,
        override_gpu_name="NVIDIA GeForce RTX 2070",
        sys_platform="win32",
    )
    manifest = deploy_rtx_boinc_client_files(
        tmp_path / "boinc_client",
        server_url="http://127.0.0.1:8095",
        handle="xaviercallens-rtx2070",
        node_id="rtx2070-XAVGAMSRV",
        profile=prof,
    )
    assert manifest["ok"] is True
    for fpath in manifest["files_written"]:
        assert Path(fpath).is_file(), f"Missing generated file: {fpath}"

    # Verify XML files parse cleanly
    pdir = Path(manifest["project_dir"])
    app_info_root = ET.fromstring((pdir / "app_info.xml").read_text(encoding="utf-8"))
    assert app_info_root.find(".//app_version/plan_class").text == PLAN_CLASS_STANDARD
    assert app_info_root.find(".//app_version/platform").text == PLATFORM_WINDOWS


def test_rtx_boinc_server_xml_rpc_and_json_workunit_lifecycle(tmp_path: Path) -> None:
    router_store = RouterStore(tmp_path / "router.db", secret=b"test-secret-0123456789-abcdef-0123456789")
    srv = RTXBoincServer(
        db_path=tmp_path / "rtx_boinc.db",
        boinc_ledger_dir=tmp_path / "boinc_ledger",
        router_store=router_store,
        master_url="http://127.0.0.1:8097/",
        require_cuda=False,
    )
    app = FastAPI()
    app.include_router(build_rtx_boinc_fastapi_router(srv))
    client = TestClient(app)

    # 1. XML RPC: get_project_config.php, create_account.php, lookup_account.php, /boinc_cgi/cgi
    cfg = client.get("/get_project_config.php")
    assert cfg.status_code == 200
    assert "windows_x86_64" in cfg.text and "cuda_hivemind" in cfg.text

    acc = client.get("/create_account.php?email_addr=test@hivemind.local&user_name=tester")
    assert acc.status_code == 200
    auth = ET.fromstring(acc.text).findtext("authenticator")
    assert auth and len(auth) >= 16

    lookup = client.get("/lookup_account.php?email_addr=test@hivemind.local")
    assert lookup.status_code == 200
    assert ET.fromstring(lookup.text).findtext("authenticator") == auth

    sched_req = (
        "<scheduler_request><hostid>42</hostid><platform_name>windows_x86_64</platform_name>"
        "<work_req_seconds>120</work_req_seconds><gpu_req_seconds>120</gpu_req_seconds>"
        "<coprocs><coproc_cuda><name>NVIDIA GeForce RTX 2070</name>"
        "<available_ram>8589934592</available_ram></coproc_cuda></coprocs></scheduler_request>"
    )
    sched_reply = client.post("/boinc_cgi/cgi", content=sched_req.encode("utf-8"))
    assert sched_reply.status_code == 200
    assert "<workunit>" in sched_reply.text and "<result>" in sched_reply.text

    # 2. JSON Workunit lease -> execute -> submit -> verify SQLite & ledger
    lease_resp = client.post(
        "/api/v1/boinc/rtx/lease",
        json={
            "handle": "xaviercallens-rtx2070",
            "node_id": "rtx2070-XAVGAMSRV",
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "vram_total_mib": 8192,
            "vram_free_mib": 6800,
            "preferred_domain": "code",
        },
    )
    assert lease_resp.status_code == 200
    spec = lease_resp.json()
    exec_res = execute_rtx_boinc_workunit(spec, device="cpu")
    assert exec_res["eval_loss_after"] < exec_res["eval_loss_before"]

    sub_resp = client.post(
        "/api/v1/boinc/rtx/submit",
        json={
            **exec_res,
            "handle": "xaviercallens-rtx2070",
            "node_id": "rtx2070-XAVGAMSRV",
        },
    )
    assert sub_resp.status_code == 200
    verdict = sub_resp.json()
    assert verdict["valid"] is True
    assert verdict["grant"]["hmt_awarded"] == 100

    # Verify rejection when monotone ratchet fails (eval_loss_after > eval_loss_before)
    lease2 = client.post(
        "/api/v1/boinc/rtx/lease",
        json={
            "handle": "bad-node",
            "node_id": "rtx-bad",
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "vram_total_mib": 8192,
            "vram_free_mib": 6800,
        },
    ).json()
    bad_sub = client.post(
        "/api/v1/boinc/rtx/submit",
        json={
            "wu_id": lease2["wu_id"],
            "handle": "bad-node",
            "node_id": "rtx-bad",
            "gpu_name": "NVIDIA GeForce RTX 2070",
            "plan_class": PLAN_CLASS_STANDARD,
            "app_name": "hivemind_qlora_moe",
            "domain": lease2["domain"],
            "cuda_verified": True,
            "eval_loss_before": 2.0,
            "eval_loss_after": 2.5,
            "state_sha256": ["abc123"],
            "hmac_signature": lease2["hmac_signature"],
        },
    ).json()
    assert bad_sub["valid"] is False
    assert "Monotone validation ratchet failed" in bad_sub["reason"]

    status = client.get("/api/v1/boinc/rtx/status").json()
    assert status["workunits_validated"] >= 1
    assert status["workunits_invalid"] >= 1


def test_gateway_exposes_rtx_boinc_server_endpoints(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "gateway.db", secret=b"test-secret-0123456789-abcdef-0123456789")
    app = create_router_app(store)
    client = TestClient(app)

    st = client.get("/api/v1/boinc/rtx/status")
    assert st.status_code == 200
    assert st.json()["ok"] is True

    acct = client.get("/create_account.php?email_addr=gw@hivemind.local")
    assert acct.status_code == 200
    assert "<authenticator>" in acct.text
