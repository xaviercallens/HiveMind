#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion E1 check: turnkey volunteer onboarding + GPU yield to owner within seconds.

Exercises real repo onboarding and preemption code paths cleanly without requiring physical CUDA hardware:
  1. `join_hivemind_campaign.sh` passes shell syntax validation (`bash -n`) and `--help` exits 0.
  2. `hivemind_volunteer_preflight.py` validates an 8 GiB NVIDIA Linux node as `ready` (exit 0), and refuses
     a 2,000 MB card below the `cuda_hivemind` 3,500 MB floor as `not_ready` (exit 2).
  3. `hivemind_opentrainer_agent.py --crowdgpt --dry-run` honors `--reserve-mib 1536` on an 8 GiB card (exit 0)
     and refuses cleanly when free VRAM is below the reserve guard (negative control).
  4. `t4_governor.terminate_tree` preempts an active background child process and verifies full termination
     within <= 1.5s (`E1` "GPU yielded to the owner within seconds").

Writes ``artifacts/mvp/E1_onboarding.json``.

    python scripts/mvp_e1_check.py [--out artifacts/mvp/E1_onboarding.json]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import hivemind_opentrainer_agent as agent  # noqa: E402
import hivemind_volunteer_preflight as preflight  # noqa: E402
import t4_governor  # noqa: E402


def _make_fake_nvidia_runner(*, total_mib: int, free_mib: int, compute_cap: str = "7.5"):
    used_mib = max(0, total_mib - free_mib)

    def _run(cmd, timeout=10.0):
        joined = " ".join(cmd)
        if "compute_cap" in joined:
            return 0, f"NVIDIA GeForce RTX 2070, {total_mib}, {used_mib}, 560.35.03, {compute_cap}"
        if "memory.used" in joined:
            return 0, f"{used_mib}, {total_mib}, 62, 12"
        if "boinc" in joined:
            return 0, "7.24.1 x86_64-pc-linux-gnu"
        return 0, ""

    return _run


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts/mvp/E1_onboarding.json"))
    args = ap.parse_args(argv)

    work = Path(tempfile.mkdtemp(prefix="e1_"))
    join_sh = ROOT / "scripts/join_hivemind_campaign.sh"
    syntax_proc = subprocess.run(["bash", "-n", str(join_sh)], capture_output=True, text=True, timeout=10)
    help_proc = subprocess.run(["bash", str(join_sh), "--help"], capture_output=True, text=True, timeout=10)

    # 1. Preflight positive & negative control (synthetic nvidia-smi runner for deterministic CI)
    ready_checks = preflight.run_checks(
        system="Linux",
        machine="x86_64",
        runner=_make_fake_nvidia_runner(total_mib=8192, free_mib=7500),
        disk_free=lambda _p: int(30e9),
        cpus=8,
        ram_bytes=lambda: int(16 * 1024**3),
        which=lambda name: f"/usr/bin/{name}" if name in ("tar", "xz", "flock", "boinc", "boinccmd") else None,
        exists=lambda _p: True,
        battery=lambda: {"present": False, "on_battery": False},
        py_version=(3, 12, 0),
        data_dir=str(work),
    )
    ready_rep = preflight.build_report(ready_checks, preflight.Scrubber())

    small_checks = preflight.run_checks(
        system="Linux",
        machine="x86_64",
        runner=_make_fake_nvidia_runner(total_mib=2000, free_mib=1800),
        disk_free=lambda _p: int(30e9),
        cpus=8,
        ram_bytes=lambda: int(16 * 1024**3),
        which=lambda name: f"/usr/bin/{name}" if name in ("tar", "xz", "flock", "boinc", "boinccmd") else None,
        exists=lambda _p: True,
        battery=lambda: {"present": False, "on_battery": False},
        py_version=(3, 12, 0),
        data_dir=str(work),
    )
    small_rep = preflight.build_report(small_checks, preflight.Scrubber())

    # 2. Agent CrowdGPT dry-run plan (8 GiB with 1536 MiB reserve passes; 1024 MiB free < reserve refuses)
    crowd_dir = work / "crowdgpt"
    crowd_dir.mkdir(parents=True, exist_ok=True)
    (crowd_dir / "client.py").write_text(
        "import json, sys\n"
        "if '--doctor' in sys.argv:\n"
        "    print(json.dumps({'client': {'version': '3.4.0'}, 'update': {'checked': True, 'signature': True, 'client_matches': True, 'remote_version': '3.4.0'}}))\n",
        encoding="utf-8",
    )

    orig_detect = agent.detect_local_gpu
    from hivemind.opentrainer import crowdgpt_contributor as cc

    orig_query = cc.query_nvidia_gpu
    try:
        def _snap_8gb():
            return cc.GpuSnapshot(total_mib=8192, used_mib=692, free_mib=7500, util_pct=5.0, temp_c=58.0, name="NVIDIA GeForce RTX 2070")

        def _snap_low_vram():
            return cc.GpuSnapshot(total_mib=8192, used_mib=7168, free_mib=1024, util_pct=45.0, temp_c=68.0, name="NVIDIA GeForce RTX 2070")

        agent.detect_local_gpu = lambda: ("NVIDIA GeForce RTX 2070", 8192, 7500)
        cc.query_nvidia_gpu = lambda *a, **kw: _snap_8gb()
        agent_ok_rc = agent.main([
            "--crowdgpt", "--crowdgpt-dir", str(crowd_dir), "--data-dir", str(work / "data_ok"),
            "--reserve-mib", "1536", "--dry-run", "--once",
        ])

        cc.query_nvidia_gpu = lambda *a, **kw: _snap_low_vram()
        vram_low, vram_low_reason = cc.resolve_vram_gb(0.0, _snap_low_vram(), cc.GuardConfig(reserve_mib=1536))
    finally:
        agent.detect_local_gpu = orig_detect
        cc.query_nvidia_gpu = orig_query

    # 3. Real process yield check via t4_governor.terminate_tree (must yield < 1.5s)
    dummy_worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    time.sleep(0.05)
    t0 = time.monotonic()
    terminated = t4_governor.terminate_tree(dummy_worker.pid, grace_s=0.4)
    dummy_worker.wait(timeout=2.0)
    yield_s = round(time.monotonic() - t0, 4)

    checks = {
        "join_script_syntax_ok": syntax_proc.returncode == 0,
        "join_script_help_ok": help_proc.returncode == 0 and "--crowdgpt" in help_proc.stdout,
        "preflight_8gb_ready": ready_rep["verdict"] == "ready" and preflight.EXIT[ready_rep["verdict"]] == 0,
        "preflight_2gb_refused": small_rep["verdict"] == "not_ready" and preflight.EXIT[small_rep["verdict"]] == 2,
        "agent_crowdgpt_8gb_dry_run_ok": agent_ok_rc == 0,
        "agent_crowdgpt_low_vram_refused": vram_low is None and "below reserve" in vram_low_reason,
        "governor_yield_terminated": bool(terminated) and (not t4_governor.pid_alive(dummy_worker.pid)),
        "governor_yield_under_1_5s": yield_s <= 1.5,
    }

    result = {
        "criterion": "E1",
        "what": "One-command onboarding + preflight VRAM/driver checks + GPU yield within seconds",
        "environment": "CPU host with deterministic nvidia-smi stubs for GPU queries + real child-process SIGTERM yield",
        "yield_elapsed_s": yield_s,
        "checks": checks,
        "passed": bool(checks) and all(checks.values()),
        "reason": "" if all(checks.values()) else f"failed checks: {[k for k, v in checks.items() if not v]}",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "yield_elapsed_s": yield_s, "checks": checks}, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
