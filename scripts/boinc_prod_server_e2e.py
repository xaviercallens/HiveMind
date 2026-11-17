#!/usr/bin/env python3
"""End-to-end: stock BOINC client + the real (source-built) BOINC server, HiveMind QLoRA MoE apps on the T4.

Criteria: artifacts/boinc_prod_server/ACCEPTANCE.md (C1-C8). The server stack must already be set up and running
(scripts/boinc_server_stack.py). Run with the venv that has peft + bitsandbytes.
"""

from __future__ import annotations

import argparse
import json
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from hivemind.boinc.real_client.client_setup import Client, prepare_data_dir  # noqa: E402
from hivemind.boinc.real_client.moe_work import build_infer_spec, build_train_spec  # noqa: E402
from hivemind.boinc.real_client.server_stack import Stack  # noqa: E402

FPOPS = 4.8e14
PROMPTS = ["The Mixture of Experts model", "Volunteer computing with BOINC means"]
COLS = "name,server_state,outcome,client_state,validate_state,granted_credit,app_version_id,exit_status"


def results(st: Stack) -> dict[str, dict]:
    out = st.sql(f"use {st.dbname}; select {COLS} from result order by id;").strip().splitlines()
    rows = {}
    for ln in out[1:]:
        v = ln.split("\t")
        rows[v[0]] = dict(zip(COLS.split(","), v, strict=False))
    return rows


def wait_validated(st: Stack, names: list[str], timeout: float) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        r = results(st)
        if all(n in r and r[n]["validate_state"] in ("1", "2", "3") and r[n]["server_state"] == "5" for n in names):
            return True
        time.sleep(10)
    return False


def listeners() -> list[str]:
    return [ln for ln in subprocess.run(["ss", "-ltn"], capture_output=True, text=True).stdout.splitlines()[1:] if ln.strip()]


def put_work(st: Stack, app: str, spec: dict, command_line: str = "", extra: dict[str, bytes] | None = None, adapter_bytes: bytes | None = None) -> None:
    (st.root / "hivemind_specs" / f"{spec['wu_id']}.json").write_text(json.dumps(spec))
    if adapter_bytes is not None:
        (st.root / "hivemind_specs" / f"{spec['wu_id']}.adapter.pt").write_bytes(adapter_bytes)
    inputs = {"in.json": json.dumps(spec, sort_keys=True).encode(), **(extra or {})}
    st.create_work(app, spec["wu_id"], inputs, FPOPS, command_line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boinc-root", type=Path, required=True)
    ap.add_argument("--stack-base", type=Path, required=True)
    ap.add_argument("--boinc-src", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--core-report", type=Path, default=ROOT / "artifacts/boinc_core_node/e2e_report.json")
    ap.add_argument("--core-infer", type=Path, required=True, help="the emulator run's uploaded inference result (hm_moe_infer_001_0_0)")
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/boinc_prod_server/e2e_report.json")
    ap.add_argument("--rpc-port", type=int, default=31418)
    a = ap.parse_args()
    st = Stack(a.stack_base, a.boinc_src, sys.executable, ROOT / "src")
    rep: dict = {"criteria": {}, "master_url": st.master_url}
    if not st.wait_http(10):
        print("the BOINC server is not answering on", st.master_url, file=sys.stderr)
        return 2

    auth = st.create_account(email=f"node-{secrets.token_hex(4)}@local.invalid", name=f"hivemind-t4-{secrets.token_hex(2)}")
    prepare_data_dir(a.work / "client", st.master_url, ROOT / "src", authenticator=auth, install_apps=False)
    client = Client(a.boinc_root / "usr/bin/boinc", a.boinc_root / "usr/lib/x86_64-linux-gnu", a.work / "client", a.rpc_port)

    put_work(st, "hivemind_qlora_moe", build_train_spec("hmp_good_001"))
    put_work(st, "hivemind_qlora_moe", build_train_spec("hmp_lazy_001", shard_index=1), "--lazy")
    t0 = time.time()
    client.start()
    rep["client_rpc_up"] = client.wait_rpc(90)
    rep["stage_train_validated"] = wait_validated(st, ["hmp_good_001_0", "hmp_lazy_001_0"], 2400)
    rep["stage_train_s"] = round(time.time() - t0)
    time.sleep(15)  # let the assimilator finish

    ledger = st.root / "hivemind_ledger"
    adapter_pt = ledger / "adapter_hmp_good_001.pt"
    if adapter_pt.exists():
        put_work(st, "hivemind_moe_infer", build_infer_spec("hmp_infer_001", PROMPTS), extra={"adapter.pt": adapter_pt.read_bytes()}, adapter_bytes=adapter_pt.read_bytes())
        client.cmd("--project", st.master_url, "update")
        rep["stage_infer_validated"] = wait_validated(st, ["hmp_infer_001_0"], 1500)
        time.sleep(15)
    rep["client_messages"] = client.cmd("--get_messages")[-2500:]
    client.quit()
    time.sleep(2)

    log = client.log_path.read_text(errors="replace")
    rep["client_log_lines"] = [ln for ln in log.splitlines() if any(k in ln for k in ("anonymous", "Starting task", "Finished upload", "Computation for", "Reporting", "CUDA:"))][:30]
    rep["results"] = results(st)
    rep["app_versions"] = st.sql(f"use {st.dbname}; select id,appid,version_num,plan_class,platformid from app_version;")
    rep["wu_assimilate_states"] = st.sql(f"use {st.dbname}; select name,assimilate_state,canonical_resultid from workunit order by id;")
    rep["verdicts"] = {p.name: json.loads(p.read_text()) for p in sorted(ledger.glob("verdict_*.json"))}
    rep["assimilated"] = [json.loads(ln) for ln in (ledger / "assimilated.jsonl").read_text().splitlines()] if (ledger / "assimilated.jsonl").exists() else []
    rep["listeners"] = listeners()

    # ---- score
    c, r = rep["criteria"], rep["results"]
    core = json.loads(a.core_report.read_text())
    core_final_sha = core["good_meta"]["state_sha256"][-1]
    c["C1"] = bool(rep["client_rpc_up"] and not any("anonymous" in ln for ln in log.splitlines()) and "hmp_good_001_0" in r)
    avid = {ln.split("\t")[0]: ln.split("\t")[3] for ln in rep["app_versions"].strip().splitlines()[1:]}
    c["C2"] = bool(r.get("hmp_good_001_0", {}).get("app_version_id") in avid and any("Starting task hmp_good_001_0" in ln and "cuda_hivemind" in ln for ln in rep["client_log_lines"]))
    # the real server names output files hmp_good_001_0_r<random>_<n> (the emulator used <result>_<n>)
    c["C3"] = all(re.search(rf"Finished upload of hmp_good_001_0_r\d+_{i}\b", log) for i in (0, 1))
    g = r.get("hmp_good_001_0", {})
    c["C4"] = bool(g.get("validate_state") == "1" and float(g.get("granted_credit", 0) or 0) > 0)
    lz = r.get("hmp_lazy_001_0", {})
    c["C5"] = bool(lz.get("validate_state") == "2" and float(lz.get("granted_credit", 0) or 0) == 0)
    ad = next((e for e in rep["assimilated"] if e["wu_id"] == "hmp_good_001"), None)
    c["C6"] = bool(ad and ad.get("adapter_sha256") == core_final_sha)
    inf = next((e for e in rep["assimilated"] if e["wu_id"] == "hmp_infer_001"), None)
    core_inf = [x["token_ids"] for x in json.loads(a.core_infer.read_text())["completions"]]
    c["C7"] = bool(inf and [x["token_ids"] for x in inf["completions"]] == core_inf and r.get("hmp_infer_001_0", {}).get("validate_state") == "1")
    stopped = subprocess.run(["bash", "/home/callensxavier_gmail_com/.claude/jobs/5c0bcf79/tmp/pj.sh", "bin/stop"], capture_output=True, text=True)
    time.sleep(3)
    left = subprocess.run(["pgrep", "-f", "feeder|transitioner|file_deleter|script_validator|script_assimilator|usr/bin/boinc|moe_app"], capture_output=True, text=True).stdout.split()
    nonloop = [ln for ln in rep["listeners"] if (":8081 " in ln or ":3306 " in ln) and "127.0.0.1" not in ln]
    rep["after_stop_processes"], rep["non_loopback_listeners"], rep["bin_stop_rc"] = left, nonloop, stopped.returncode
    c["C8"] = bool(not left and not nonloop)
    subprocess.run(["bash", "/home/callensxavier_gmail_com/.claude/jobs/5c0bcf79/tmp/pj.sh", "bin/start"], capture_output=True, text=True)  # leave the server running
    rep["PASS"] = all(c.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=2, sort_keys=True, default=str))
    (a.out.parent / "client_stdout.log").write_text(log[-80000:])
    print(json.dumps({"criteria": c, "PASS": rep["PASS"]}, indent=2))
    return 0 if rep["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
