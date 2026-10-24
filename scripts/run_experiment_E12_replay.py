"""E12-pilot: does the Byzantine spot-check's replay rule (||theta_validator - theta_worker||_inf <= 1e-4 after one
50-step segment) survive a change of thread count, precision class or host on CPU?
Pre-registered in ``artifacts/experiments/E12_pilot_prereg.json`` (written before any run).

  run:      python scripts/run_experiment_E12_replay.py run --cell ref --threads 4 --out artifacts/experiments/e12/ref.pt
  analyze:  python scripts/run_experiment_E12_replay.py analyze

The segment is Qwen2.5-0.5B + LoRA(r=8, q_proj/v_proj), AdamW 1e-4, batch 1 x 64 tokens, 50 steps on 50 windows of
``docs/RESEARCH_PROGRAM.md`` (shipped alongside so a remote host sees identical token ids). Every cell starts from
the same saved initial LoRA state. ``--tamper`` flips the sign of the LoRA update at step 25 (positive control).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
EXP_DIR = PROJECT_ROOT / "artifacts" / "experiments"
OUT_DIR = EXP_DIR / "e12"
PREREG = EXP_DIR / "E12_pilot_prereg.json"
REPORT = EXP_DIR / "E12_pilot_report.json"
TEXT_FILE = HERE / "e12_text.md"  # copied from docs/RESEARCH_PROGRAM.md by `prepare`
MODEL = "Qwen/Qwen2.5-0.5B"
STEPS, SEQ, LR, EPS = 50, 64, 1e-4, 1e-4


def _lora_state(model) -> Dict[str, Any]:
    import torch

    return {n: p.detach().clone().to(torch.float32) for n, p in model.named_parameters() if p.requires_grad}


def run(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
    init_path = OUT_DIR / "lora_init.pt"
    if init_path.exists():
        model.load_state_dict({k: v for k, v in torch.load(init_path).items()}, strict=False)
    else:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save(_lora_state(model), init_path)
    text = TEXT_FILE.read_text(encoding="utf-8")
    ids = tok(text, return_tensors="pt").input_ids[0]
    assert ids.numel() >= STEPS * SEQ, f"text too short: {ids.numel()} tokens"
    windows = [ids[i * SEQ : (i + 1) * SEQ].unsqueeze(0) for i in range(STEPS)]
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    losses: List[float] = []
    model.train()
    for step, x in enumerate(windows):
        opt.zero_grad(set_to_none=True)
        if args.dtype == "bf16":
            with torch.autocast("cpu", dtype=torch.bfloat16):
                out = model(input_ids=x, labels=x)
        else:
            out = model(input_ids=x, labels=x)
        out.loss.backward()
        if args.tamper and step == 25:
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    p.grad.neg_()
        opt.step()
        losses.append(float(out.loss.item()))
    state = _lora_state(model)
    meta = {
        "cell": args.cell, "threads": args.threads, "dtype": args.dtype, "tamper": bool(args.tamper), "host": platform.node(),
        "machine": platform.machine(), "processor": platform.processor(), "torch": str(torch.__version__),
        "transformers": __import__("transformers").__version__, "peft": __import__("peft").__version__,
        "python": platform.python_version(), "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "n_tokens": int(ids.numel()), "steps": STEPS, "losses": losses, "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "lora": state}, Path(args.out))
    print(json.dumps({k: v for k, v in meta.items() if k != "losses"} | {"final_loss": losses[-1]}, indent=1))
    return 0


def analyze(_args: argparse.Namespace) -> int:
    import torch

    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    # our own files from our own two hosts (the meta of early cells holds a TorchVersion object, hence weights_only=False)
    files = {p.stem: torch.load(p, weights_only=False) for p in sorted(OUT_DIR.glob("*.pt")) if p.stem != "lora_init"}
    ref = files["ref"]
    rows: Dict[str, Any] = {}
    for name, d in files.items():
        diffs = {k: float((d["lora"][k] - ref["lora"][k]).abs().max()) for k in ref["lora"]}
        linf = max(diffs.values())
        loss_diff = [abs(a - b) for a, b in zip(d["meta"]["losses"], ref["meta"]["losses"])]
        first = next((i for i, v in enumerate(loss_diff) if v > 1e-6), None)
        rows[name] = {"meta": {k: v for k, v in d["meta"].items() if k != "losses"}, "linf_vs_ref": linf, "passes_eps": linf <= EPS,
                      "n_params_above_eps": sum(1 for v in diffs.values() if v > EPS), "n_param_tensors": len(diffs),
                      "max_loss_diff": max(loss_diff), "first_step_loss_diff_gt_1e-6": first, "final_loss": d["meta"]["losses"][-1]}
    def st(cell: str) -> float:
        return rows[cell]["linf_vs_ref"] if cell in rows else float("nan")
    analysis = {
        "P12a": {"same_process_repeat_linf": st("same_process_repeat"), "status": "supported" if st("same_process_repeat") == 0.0 else "refuted"},
        "P12b": {"threads_1": st("threads_1"), "threads_8": st("threads_8"),
                 "status": ("supported" if all(0.0 < st(c) <= EPS for c in ("threads_1", "threads_8") if c in rows) and any(c in rows for c in ("threads_1", "threads_8")) else "refuted")},
        "P12c": {"remote_cpu": st("remote_cpu"), "exceeds_thread_cells": st("remote_cpu") > max(st("threads_1"), st("threads_8")) if "remote_cpu" in rows else None,
                 "within_eps": (st("remote_cpu") <= EPS) if "remote_cpu" in rows else None,
                 "status": ("not run" if "remote_cpu" not in rows else ("supported (diverges more than thread cells, > 1e-6)" if st("remote_cpu") > 1e-6 and st("remote_cpu") > max(st("threads_1"), st("threads_8")) else "refuted"))},
        "P12d": {"bf16_autocast": st("bf16_autocast"), "status": "not run" if "bf16_autocast" not in rows else ("supported" if st("bf16_autocast") > EPS else "refuted")},
    }
    controls = {"negative_repeat_exactly_zero": st("same_process_repeat") == 0.0, "positive_tampered_exceeds_eps": st("tampered") > EPS if "tampered" in rows else False}
    report = {"schema_version": "hivemind-experiment-E12-pilot/v1.0", "experiment_id": "E12-pilot", "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
              "epsilon": EPS, "cells": rows, "analysis": analysis, "controls": controls, "controls_all_passed": all(controls.values()),
              "scope_note": prereg["scope"], "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode()
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for n, r in rows.items():
        print(f"{n:22s} linf {r['linf_vs_ref']:.3e} passes_eps={r['passes_eps']!s:5s} max_loss_diff {r['max_loss_diff']:.2e} first>1e-6 at step {r['first_step_loss_diff_gt_1e-6']}")
    print(json.dumps({k: v["status"] for k, v in analysis.items()} | {"controls_all_passed": report["controls_all_passed"], "receipt": report["receipt_sha256"]}, indent=1))
    return 0 if report["controls_all_passed"] else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    r = sub.add_parser("run")
    r.add_argument("--cell", required=True); r.add_argument("--threads", type=int, default=4); r.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    r.add_argument("--tamper", action="store_true"); r.add_argument("--out", required=True)
    sub.add_parser("analyze")
    args = ap.parse_args()
    return run(args) if args.mode == "run" else analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
