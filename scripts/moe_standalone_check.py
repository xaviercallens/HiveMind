#!/usr/bin/env python3
"""Tier-2 check of the QLoRA MoE app without BOINC: determinism and kill/resume.

Run with the venv that has peft + bitsandbytes:  <venv>/bin/python scripts/moe_standalone_check.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.boinc.real_client.moe_work import build_train_spec  # noqa: E402


def run_app(wd: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), HF_HUB_OFFLINE="1")
    cmd = [sys.executable, "-W", "ignore", "-m", "hivemind.boinc.real_client.moe_app", "--mode", "train", *extra]
    return subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True)


def main() -> int:
    spec = build_train_spec("standalone_001")
    out: dict = {}
    with tempfile.TemporaryDirectory() as td:
        ref, pre = Path(td) / "ref", Path(td) / "pre"
        for d in (ref, pre):
            d.mkdir()
            (d / "in.json").write_text(json.dumps(spec))
        r = run_app(ref)
        out["ref_rc"] = r.returncode
        out["ref_stderr_tail"] = r.stderr[-400:] if r.returncode else ""
        a = json.loads((ref / "out.json").read_text())
        keys = ("state_sha256", "eval_loss_before", "eval_loss_after", "wall_seconds", "peak_vram_mib", "trainable_params")
        out["ref"] = {k: a[k] for k in keys}
        c = run_app(pre, "--crash-after-segment=2")
        out["crash_rc"] = c.returncode
        out["ckpt_exists"] = (pre / "hivemind_moe_ckpt.pt").exists()
        run_app(pre)
        b = json.loads((pre / "out.json").read_text())
        out["resumed_from_segment"] = b["resumed_from_segment"]
        out["resume_bit_identical"] = b["state_sha256"] == a["state_sha256"]
    print(json.dumps(out, indent=2))
    return 0 if out.get("resume_bit_identical") and out["crash_rc"] == 75 else 1


if __name__ == "__main__":
    raise SystemExit(main())
