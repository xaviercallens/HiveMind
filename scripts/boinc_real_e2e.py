#!/usr/bin/env python3
"""End-to-end test of a real, stock BOINC client on this machine's GPU (criteria: artifacts/boinc_real/ACCEPTANCE.md).

Needs the unpacked Ubuntu ``boinc-client`` + ``libboinc7`` packages (no system install, no daemon):

    boinc/root/usr/bin/boinc   boinc/root/usr/lib/x86_64-linux-gnu/libboinc.so.7

    python3 scripts/boinc_real_e2e.py --boinc-root /mnt/disks/.../boinc/root --work /mnt/disks/.../boinc/run1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.boinc.real_client.app import run  # noqa: E402
from hivemind.boinc.real_client.client_setup import Client, prepare_data_dir  # noqa: E402
from hivemind.boinc.real_client.project_server import ProjectServer  # noqa: E402

BASE = {"seed": 7, "d_in": 512, "d_out": 512, "rank": 8, "batch": 256, "lr": 2.0, "spot_check_interval": 50}


def gpu_busy() -> list[str]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name,used_memory", "--format=csv,noheader"], capture_output=True, text=True).stdout
    # the runner's own CUDA context (the validator replays on the GPU in-process) is not a leftover of the client
    return [l for l in out.splitlines() if l.strip() and not l.startswith(f"{os.getpid()},")]


def flops(spec: dict) -> float:
    return 6.0 * spec["batch"] * (spec["d_in"] * spec["d_out"] + spec["rank"] * (spec["d_in"] + spec["d_out"])) * spec["steps"]


def wait_for(cond, timeout: float, every: float = 2.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(every)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boinc-root", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "artifacts/boinc_real/e2e_report.json")
    ap.add_argument("--rpc-port", type=int, default=31416)
    ap.add_argument("--skip-preemption", action="store_true")
    a = ap.parse_args()

    pre = gpu_busy()
    if pre:
        print("GPU is in use by other processes; refusing to share it silently:", pre, file=sys.stderr)
    report: dict = {"gpu_busy_before": pre, "criteria": {}}
    work = a.work
    work.mkdir(parents=True, exist_ok=True)
    srv = ProjectServer(work / "server")
    srv.start()
    setup = prepare_data_dir(work / "client", srv.master_url, ROOT / "src")
    client = Client(a.boinc_root / "usr/bin/boinc", a.boinc_root / "usr/lib/x86_64-linux-gnu", work / "client", a.rpc_port)

    good = {**BASE, "wu_id": "hm_good_001", "steps": 400}
    fault = {**BASE, "wu_id": "hm_fault_001", "steps": 400, "seed": 11}
    srv.create_work(good, flops_est=flops(good))
    srv.create_work(fault, command_line="--inject-fault=0.01", flops_est=flops(fault))
    report["server_url"] = srv.master_url

    # ---- stage 1: GPU detection, download, run, upload, validate, credit, red-team control
    client.start()
    ok = client.wait_rpc(90)
    report["client_rpc_up"] = ok
    done = srv.wait_all(900) if ok else False
    report["stage1_all_finished"] = done
    log = client.log_path.read_text(errors="replace")
    report["client_startup_log"] = [l for l in log.splitlines() if any(k in l for k in ("BOINC client", "NVIDIA", "CUDA", "Found app_info", "anonymous", "Starting task", "Computation for task", "Finished upload"))][:40]
    report["a2_scheduler_coprocs"] = next((r["coprocs_xml"][:600] for r in srv.sched_requests if r.get("coprocs_xml")), "")
    report["verdicts"] = {k: v.to_dict() for k, v in srv.verdicts.items()}
    report["client_messages"] = client.cmd("--get_messages")[-3000:]
    results = {}
    for wu in srv.workunits:
        p = srv.dir / "upload" / f"{wu}_0_0"
        if p.exists():
            d = json.loads(p.read_text())
            results[wu] = {k: d[k] for k in ("device_name", "initial_loss", "final_loss", "final_digest", "wall_seconds", "flops_claimed", "inject_fault")}
    report["uploaded_results"] = results

    # ---- stage 2: preemption (SIGKILL the client mid-task, restart, resume from the app checkpoint)
    if not a.skip_preemption and ok:
        long_ = {**BASE, "wu_id": "hm_preempt_001", "steps": 3000, "seed": 13}
        with tempfile.TemporaryDirectory() as td:
            reference = run(long_, "cuda", Path(td) / "ref.pt")
        srv.create_work(long_, flops_est=flops(long_))
        client.cmd("--project", srv.master_url, "update")
        ckpt_seen = wait_for(lambda: any((client.dir / "slots").glob("*/hivemind_ckpt.pt")), 240, 0.5)
        time.sleep(3)  # let a few more checkpoints land, still well before the end of 3000 steps
        report["preempt_checkpoint_seen"] = ckpt_seen
        client.kill()
        time.sleep(2)
        report["preempt_app_survivors"] = subprocess.run(["pgrep", "-f", "hivemind.boinc.real_client.app"], capture_output=True, text=True).stdout.split()
        client.start()
        report["preempt_client_restarted"] = client.wait_rpc(90)
        fin = srv.wait_all(900)
        p = srv.dir / "upload" / "hm_preempt_001_0_0"
        if p.exists():
            d = json.loads(p.read_text())
            report["preempt"] = {
                "finished": fin,
                "resumed_from_step": d["resumed_from_step"],
                "digest_equals_uninterrupted_reference": d["final_digest"] == reference["final_digest"],
                "verdict": srv.verdicts["hm_preempt_001"].to_dict() if "hm_preempt_001" in srv.verdicts else None,
            }
    client.quit()
    srv.stop()
    time.sleep(2)
    report["gpu_busy_after"] = gpu_busy()
    report["stray_boinc_processes"] = subprocess.run(["pgrep", "-f", "usr/bin/boinc"], capture_output=True, text=True).stdout.split()

    # ---- score against ACCEPTANCE.md
    v = srv.verdicts
    c = report["criteria"]
    c["A1"] = any("Tesla T4" in l for l in report["client_startup_log"]) and any("CUDA" in l or "NVIDIA" in l for l in report["client_startup_log"])
    c["A2"] = "Tesla T4" in report["a2_scheduler_coprocs"]
    c["A3"] = results.get("hm_good_001", {}).get("device_name") == "Tesla T4"
    g = v.get("hm_good_001")
    c["A4"] = bool(g and g.valid and g.same_class_exact is True and (g.cross_class_max_ulp or 0) <= 1)
    c["A5"] = bool(g and g.grant["cobblestones"] > 0 and g.grant["capped_by"] == "verified")
    f = v.get("hm_fault_001")
    c["A6"] = bool(f and not f.valid and f.grant["cobblestones"] == 0)
    if not a.skip_preemption:
        pr = report.get("preempt", {})
        c["A7"] = bool(pr.get("resumed_from_step", 0) > 0 and pr.get("digest_equals_uninterrupted_reference") and pr.get("verdict", {}).get("valid"))
    c["A8"] = not report["gpu_busy_after"] and not report["stray_boinc_processes"]
    report["PASS"] = all(c.values())
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    (a.out.parent / "client_stdout.log").write_text(client.log_path.read_text(errors="replace")[-60000:])
    print(json.dumps({"criteria": c, "PASS": report["PASS"]}, indent=2))
    return 0 if report["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
