#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Close the train-to-serve loop for the BOINC chain: validated adapter -> GGUF LoRA -> measured on held-out text -> gate.

    python scripts/adapter_to_served.py prepare --ledger <ledger dir> --work <dir>      # PEFT dir, base GGUF (f16), LoRA GGUF
    python scripts/adapter_to_served.py evaluate --work <dir> --perplexity <llama-perplexity> --out <artifact.json>

The ledger is only read. The gate is fixed BEFORE any run: the adapter is "servable" only if, on the SERVED files (GGUF, not PyTorch), its mean
held-out negative log-likelihood is lower than the base model's on the pooled held-out text, the paired per-chunk bootstrap 95% interval of the
improvement excludes 0, and no domain gets worse by more than 1% (``MAX_DOMAIN_REGRESSION``). A base model has no instruction tuning: a tier built
from this adapter is a text-completion model and must be labelled that way.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

LLAMA_SRC = Path("/home/callensxavier_gmail_com/gwaya-lite/llama.cpp")
BASE_ID = "ibm-granite/granite-3.1-1b-a400m-base"
EVAL_DOMAINS = ["general", "french", "edu_web", "news", "biomed", "math"]  # the six the chain reports per round
CHUNK = 128  # the chain's seq_len
CHUNKS_PER_DOMAIN = 64
MAX_DOMAIN_REGRESSION = 0.01  # a domain may not get worse than +1% NLL
BOOTSTRAP = 2000


def snapshot_dir() -> Path:
    import os

    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub" / f"models--{BASE_ID.replace('/', '--')}" / "snapshots"
    snaps = sorted(p for p in hub.glob("*") if (p / "config.json").exists() and (p / "model.safetensors.index.json").exists())
    if not snaps:
        raise FileNotFoundError(f"{BASE_ID} is not in the local cache under {hub}")
    return snaps[-1]


def prepare(args: argparse.Namespace) -> int:
    import torch
    from safetensors.torch import save_file

    ledger, work = Path(args.ledger), Path(args.work)
    head = json.loads((ledger / "latest_adapter.json").read_text())
    src = ledger / f"adapter_{head['wu_id']}.pt"
    obj = torch.load(src, map_location="cpu", weights_only=True)
    if obj["sha256"] != head["sha256"]:
        print("adapter hash does not match latest_adapter.json: refusing", file=sys.stderr)
        return 1
    peft_dir = work / "peft"
    peft_dir.mkdir(parents=True, exist_ok=True)
    tensors = {k.replace(".default", ""): v.contiguous().to(torch.float32) for k, v in obj["state"].items()}
    save_file(tensors, str(peft_dir / "adapter_model.safetensors"))
    targets = sorted({k.split(".lora_")[0].split(".")[-1] for k in tensors})
    rank = next(v.shape[0] for k, v in tensors.items() if "lora_A" in k)
    snap = snapshot_dir()
    (peft_dir / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": rank, "lora_alpha": args.alpha, "lora_dropout": 0.0, "bias": "none",
        "target_modules": targets, "base_model_name_or_path": str(snap), "inference_mode": True,
    }, indent=2))
    (work / "adapter_source.json").write_text(json.dumps({"wu_id": head["wu_id"], "sha256": head["sha256"], "chain_eval_loss_before": head["eval_loss_before"],
                                                           "chain_eval_loss_after": head["eval_loss_after"], "rank": rank, "alpha": args.alpha, "targets": targets}, indent=2))
    base_gguf, lora_gguf = work / "base-f16.gguf", work / "adapter-f16.gguf"
    env = {"PYTHONPATH": str(LLAMA_SRC / "gguf-py"), "PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(Path.home()), "HF_HUB_OFFLINE": "1"}
    if not base_gguf.exists():
        subprocess.run([sys.executable, str(LLAMA_SRC / "convert_hf_to_gguf.py"), str(snap), "--outfile", str(base_gguf), "--outtype", "f16"], check=True, env=env)
    subprocess.run([sys.executable, str(LLAMA_SRC / "convert_lora_to_gguf.py"), str(peft_dir), "--base", str(snap), "--outfile", str(lora_gguf), "--outtype", "f16"], check=True, env=env)
    print(json.dumps({"adapter": head["wu_id"], "base_gguf": str(base_gguf), "lora_gguf": str(lora_gguf), "rank": rank}))
    return 0


def heldout_text(work: Path) -> dict[str, Path]:
    """Held-out text per domain from the chain's own never-trained rows (moe_work), as plain files for llama-perplexity."""
    from transformers import AutoTokenizer

    from hivemind.boinc.real_client import moe_work as mw

    tok = AutoTokenizer.from_pretrained(str(snapshot_dir()))
    out: dict[str, Path] = {}
    for d in EVAL_DOMAINS:
        ids = mw.domain_ids(tok, d, "eval", 0, CHUNK * CHUNKS_PER_DOMAIN)[: CHUNK * CHUNKS_PER_DOMAIN]
        p = work / f"heldout_{d}.txt"
        p.write_text(tok.decode(ids), encoding="utf-8")
        out[d] = p
        print(f"{d}: {len(ids)} held-out tokens", flush=True)
    return out


def run_ppl(exe: str, model: Path, text: Path, lora: Path | None, threads: int) -> list[float]:
    """Per-chunk NLL (nats/token), recovered from llama-perplexity's running perplexity estimates."""
    cmd = [exe, "-m", str(model), "-f", str(text), "-c", str(CHUNK), "-b", str(CHUNK), "-t", str(threads), "-ngl", "0", "--no-warmup"]
    if lora is not None:
        cmd += ["--lora", str(lora)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    running = [float(x) for x in re.findall(r"\[\d+\](\d+\.\d+)", res.stdout + res.stderr)]
    if not running:
        raise RuntimeError("no perplexity output: " + (res.stderr or res.stdout)[-400:])
    nll, prev = [], 0.0
    for i, r in enumerate(running, start=1):
        cur = i * math.log(r)
        nll.append(cur - prev)
        prev = cur
    return nll


def evaluate(args: argparse.Namespace) -> int:
    work = Path(args.work)
    texts = heldout_text(work)
    base, lora = work / args.base_gguf, work / "adapter-f16.gguf"
    per: dict[str, dict[str, list[float]]] = {}
    for d, t in texts.items():
        b = run_ppl(args.perplexity, base, t, None, args.threads)
        a = run_ppl(args.perplexity, base, t, lora, args.threads)
        n = min(len(a), len(b))
        per[d] = {"base": b[:n], "adapter": a[:n]}
        print(f"{d}: n_chunks={n} base_nll={sum(b[:n]) / n:.4f} adapter_nll={sum(a[:n]) / n:.4f}", flush=True)
    pairs = [(b, a) for v in per.values() for b, a in zip(v["base"], v["adapter"])]
    diffs = [b - a for b, a in pairs]  # positive = adapter better
    mean_gain = sum(diffs) / len(diffs)
    rng = random.Random(7)
    boots = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(BOOTSTRAP))
    lo, hi = boots[int(0.025 * BOOTSTRAP)], boots[int(0.975 * BOOTSTRAP)]
    domains = {}
    worst = 0.0
    for d, v in per.items():
        mb, ma = sum(v["base"]) / len(v["base"]), sum(v["adapter"]) / len(v["adapter"])
        rel = (ma - mb) / mb
        worst = max(worst, rel)
        domains[d] = {"n_chunks": len(v["base"]), "base_nll": round(mb, 4), "adapter_nll": round(ma, 4), "relative_change": round(rel, 4),
                      "base_ppl": round(math.exp(mb), 3), "adapter_ppl": round(math.exp(ma), 3)}
    passed = mean_gain > 0 and lo > 0 and worst <= MAX_DOMAIN_REGRESSION
    src = json.loads((work / "adapter_source.json").read_text())
    result = {
        "experiment": "Does the BOINC-trained adapter still beat its base model after conversion to GGUF (the files that would be served)?",
        "adapter": src, "base": BASE_ID, "engine": f"llama.cpp llama-perplexity, CPU, base {args.base_gguf} + f16 LoRA GGUF", "label": args.label,
        "held_out": f"{CHUNKS_PER_DOMAIN} chunks x {CHUNK} tokens per domain from moe_work's never-trained eval rows; domains {EVAL_DOMAINS}",
        "gate": {"rule": "pooled mean NLL gain > 0, paired bootstrap 95% interval of the gain excludes 0, no domain worse than +1%", "fixed_before_running": True},
        "pooled": {"n_chunks": len(diffs), "mean_nll_gain_nats": round(mean_gain, 5), "bootstrap95": [round(lo, 5), round(hi, 5)],
                   "relative_gain": round(mean_gain / (sum(b for b, _ in pairs) / len(pairs)), 4)},
        "domains": domains, "worst_domain_regression": round(worst, 4), "servable": passed,
        "caveat": "Language-model loss on text, not task accuracy. The adapter was trained on an NF4 base and is measured here on an f16 base. A base model has no chat tuning.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("pooled", "worst_domain_regression", "servable")}, indent=2))
    return 0 if passed else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--ledger", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--alpha", type=int, default=16)
    e = sub.add_parser("evaluate")
    e.add_argument("--work", required=True)
    e.add_argument("--perplexity", default=shutil.which("llama-perplexity") or "llama-perplexity")
    e.add_argument("--out", required=True)
    e.add_argument("--threads", type=int, default=6)
    e.add_argument("--base-gguf", default="base-f16.gguf", help="base GGUF inside --work (e.g. base-q4km.gguf)")
    e.add_argument("--label", default="primary")
    args = ap.parse_args(argv)
    return prepare(args) if args.cmd == "prepare" else evaluate(args)


if __name__ == "__main__":
    raise SystemExit(main())
