#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP criterion A2: one MoE GGUF split across two hosts with llama.cpp RPC, compared token by token with the same model on one host.

    # on the second host, bound to loopback only (rpc-server has NO authentication), reached through an SSH tunnel:
    #   ggml-rpc-server -H 127.0.0.1 -p 50052
    python scripts/a2_split_moe_rpc.py --gguf base-q4km.gguf --llama-server <rpc build>/llama-server --rpc 127.0.0.1:50052 --ngl 12 \
        --remote-host-label <vm name> --out artifacts/mvp/A2_split_moe.json

Both runs use the same binary, the same file, temperature 0, a fixed seed and no prompt cache. ``passed`` needs: the server log shows layers really assigned to the RPC device (a silent
fall-back to local CPU must not count as a split), every prompt token-identical to the single-host run, and tokens/s recorded for both.
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

PROMPTS = [
    "The capital of France is", "Water boils at a temperature of", "The largest planet in the solar system is", "In computer science, a hash table is",
    "The theory of general relativity was developed by", "A prime number is a natural number that", "The chemical symbol for gold is", "def fibonacci(n):\n",
    "The Mixture of Experts model works by", "Photosynthesis is the process by which", "The speed of light in a vacuum is", "SELECT name FROM users WHERE",
    "The mitochondria is known as", "La capitale de l'Espagne est", "The first person to walk on the moon was", "An adapter in machine learning is",
    "The derivative of x squared is", "Volunteer computing allows", "The Pacific Ocean is", "To sort a list in Python, you can",
]
N_PREDICT = 32


def compare_runs(single: list[dict[str, Any]], split: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Token-by-token comparison. A missing split run is 'not met', never a pass."""
    if not split:
        return {"compared": 0, "identical": 0, "all_identical": False, "first_mismatch": None, "reason": "no split run"}
    if len(single) != len(split):
        return {"compared": min(len(single), len(split)), "identical": 0, "all_identical": False, "first_mismatch": None, "reason": f"{len(single)} single vs {len(split)} split results"}
    identical, first = 0, None
    for i, (a, b) in enumerate(zip(single, split)):
        same = bool(a["tokens"]) and a["tokens"] == b["tokens"]
        identical += same
        if not same and first is None:
            pos = next((j for j, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), min(len(a["tokens"]), len(b["tokens"])))
            first = {"prompt_index": i, "token_position": pos, "single": a["text"][:80], "split": b["text"][:80]}
    return {"compared": len(single), "identical": identical, "all_identical": identical == len(single) and identical > 0, "first_mismatch": first, "reason": ""}


def token_agreement(single: list[dict[str, Any]], split: list[dict[str, Any]]) -> float:
    """Fraction of generated tokens before the first divergence, over all prompts: how late runs differ (reported only; the criterion stays token-identical)."""
    total = sum(len(a["tokens"]) for a in single)
    if not total or not split:
        return 0.0
    same = 0
    for a, b in zip(single, split):
        for x, y in zip(a["tokens"], b["tokens"]):
            if x != y:
                break
            same += 1
    return round(same / total, 4)


def rpc_layers_from_log(log: str) -> int:
    """How many layers the server says it put on an RPC device (0 when it silently stayed local)."""
    return len(set(re.findall(r"layer\s+(\d+)\s+assigned to device RPC", log)))


def tokens_per_second(results: list[dict[str, Any]]) -> float:
    n, s = sum(r["predicted_n"] for r in results), sum(r["predicted_ms"] for r in results) / 1000.0
    return round(n / s, 2) if s > 0 else 0.0


def decide(cmp: dict[str, Any], rpc_layers: int, tps_single: float, tps_split: float, need: int = 20) -> tuple[bool, str]:
    problems = []
    if rpc_layers <= 0:
        problems.append("the log shows no layer on an RPC device")
    if cmp["compared"] < need or not cmp["all_identical"]:
        problems.append(f"{cmp['identical']}/{cmp['compared']} prompts token-identical (need {need}/{need})")
    if tps_single <= 0 or tps_split <= 0:
        problems.append("tokens/s missing")
    return not problems, "; ".join(problems)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _run(server: str, gguf: str, extra: list[str], threads: int, log_path: Path) -> list[dict[str, Any]]:
    port = _free_port()
    with open(log_path, "wb") as log:
        proc = subprocess.Popen([server, "-m", gguf, "--host", "127.0.0.1", "--port", str(port), "-c", "2048", "-t", str(threads), "-v", *extra], stdout=log, stderr=subprocess.STDOUT)
    try:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"llama-server exited early with code {proc.returncode}; see {log_path}")
            try:
                if httpx.get(url + "/health", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(1)
        else:
            raise RuntimeError(f"llama-server not healthy; see {log_path}")
        out = []
        for p in PROMPTS:
            r = httpx.post(url + "/completion", timeout=600, json={"prompt": p, "n_predict": N_PREDICT, "temperature": 0, "seed": 7, "cache_prompt": False, "return_tokens": True})
            r.raise_for_status()
            d = r.json()
            out.append({"prompt": p, "text": d["content"], "tokens": d.get("tokens") or [], "predicted_n": d["timings"]["predicted_n"], "predicted_ms": d["timings"]["predicted_ms"]})
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--llama-server", required=True)
    ap.add_argument("--rpc", required=True, help="host:port of the RPC server (keep it on loopback through a tunnel)")
    ap.add_argument("--ngl", type=int, default=12, help="layers to place on the RPC device; the rest stay local")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--no-repack", action="store_true", help="disable weight repacking on BOTH runs, so local and remote layers use the same CPU kernels")
    ap.add_argument("--remote-host-label", required=True)
    ap.add_argument("--local-host-label", default=socket.gethostname())
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent.parent / "artifacts/mvp/A2_split_moe.json")
    args = ap.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="a2_"))
    common = ["--no-repack"] if args.no_repack else []
    single = _run(args.llama_server, args.gguf, ["-ngl", "0", *common], args.threads, work / "single.log")
    control = _run(args.llama_server, args.gguf, ["-ngl", "0", *common], args.threads, work / "single_repeat.log")  # is the reference itself reproducible?
    split_log = work / "split.log"
    split, err = None, None
    try:
        split = _run(args.llama_server, args.gguf, ["--rpc", args.rpc, "-ngl", str(args.ngl), *common], args.threads, split_log)
    except (RuntimeError, httpx.HTTPError) as exc:
        err = f"{type(exc).__name__}: {exc}"
    log_text = split_log.read_text(encoding="utf-8", errors="ignore") if split_log.exists() else ""
    cmp = compare_runs(single, split)
    layers = rpc_layers_from_log(log_text)
    tps_a, tps_b = tokens_per_second(single), tokens_per_second(split or [])
    passed, reason = decide(cmp, layers, tps_a, tps_b)
    if err:
        reason = f"{err}; {reason}"
    result = {
        "criterion": "A2", "model_file": Path(args.gguf).name, "hosts": [args.local_host_label, args.remote_host_label], "ngl_on_rpc": args.ngl, "rpc_layers_in_log": layers,
        "prompts": len(PROMPTS), "n_predict": N_PREDICT, "no_repack": args.no_repack, "comparison": cmp, "control_single_host_repeated": compare_runs(single, control),
        "token_agreement_fraction": token_agreement(single, split or []), "tokens_per_s": {"single_host": tps_a, "split": tps_b},
        "setup": "CPU on both hosts (llama.cpp RPC backend over an SSH tunnel): shows the model split and the token identity, not GPU speed.",
        "passed": passed, "reason": reason, "logs": str(work),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("hosts", "rpc_layers_in_log", "no_repack", "comparison", "control_single_host_repeated", "token_agreement_fraction", "tokens_per_s", "passed", "reason", "logs")}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
