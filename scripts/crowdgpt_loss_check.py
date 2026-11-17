#!/usr/bin/env python3
"""CrowdGPT held-out loss check: compute it on a GPU, verify it on a CPU-only backend.

  evaluate   (GPU host)  losses of base and candidate on N held-out windows -> signed receipt JSON
  verify     (backend)   signature, hashes, window choice, plausibility, k CPU spot checks, then the ratchet rule
  calibrate  (GPU host)  the same windows on the GPU and on the CPU: the measured disagreement that `--tol` must exceed

Weights are CrowdGPT `model_*.safetensors` files, given as a local path or hf://REPO@REVISION/FILE. Held-out tokens are a
raw uint32 file (a slice of a dataset chunk). Keys are read from files (--key-file, --secret-file), never from arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.swarm import crowdgpt_loss_check as lc  # noqa: E402


def parse_hf_spec(spec: str) -> tuple[str, str, str]:
    """hf://OWNER/NAME@REVISION/FILE -> (repo, revision, file). The revision is required: no moving targets."""
    repo, _, rest = spec[len("hf://"):].partition("@")
    rev, _, filename = rest.partition("/")
    if not repo or not rev or not filename:
        raise SystemExit(f"hf:// spec needs OWNER/NAME@REVISION/FILE, got {spec!r}")
    return repo, rev, filename


def resolve(spec: str) -> Path:
    """Local path, or an hf:// spec downloaded into the Hugging Face cache."""
    if not spec.startswith("hf://"):
        return Path(spec)
    from huggingface_hub import hf_hub_download

    repo, rev, filename = parse_hf_spec(spec)
    return Path(hf_hub_download(repo, filename, revision=rev))


def load_tokens(path: Path) -> np.ndarray:
    return np.frombuffer(path.read_bytes(), dtype=lc.TOKEN_DTYPE)


def pick_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]


def load_weights(spec: str, engram_spec: str | None) -> torch.Tensor:
    """Trunk only, or trunk ‖ engram when an engram file is given (current Crowd-v1 checkpoints use the engram)."""
    if engram_spec:
        return lc.load_crowd_v1_with_engram(resolve(spec), resolve(engram_spec))
    return lc.load_flat_safetensors(resolve(spec))


def model_factory(a: argparse.Namespace):
    if getattr(a, "base_engram", None):
        return lambda: lc.CrowdV1(lc.CROWD_V1_ENGRAM_CONFIG)
    return lc.CrowdV1


def cmd_evaluate(a: argparse.Namespace) -> int:
    base, cand = load_weights(a.base, a.base_engram), load_weights(a.candidate, a.candidate_engram)
    held = load_tokens(Path(a.heldout))
    t = time.time()
    r = lc.make_receipt(evaluator_id=a.evaluator_id, key=Path(a.key_file).read_bytes().strip(), base=base, candidate=cand, heldout=held,
                        window=a.window, n=a.n, seed=a.seed, model_factory=model_factory(a), device=a.device, dtype=pick_dtype(a.dtype),
                        round_id=a.round_id, batch=a.batch)
    Path(a.out).write_text(r.to_json())
    print(json.dumps({"out": a.out, "seconds": round(time.time() - t, 1), "mean_before": r.mean_before, "mean_after": r.mean_after,
                      "delta": r.mean_after - r.mean_before, "commitment": r.commitment()}))
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    r = lc.LossCheckReceipt.from_json(Path(a.receipt).read_text())
    keys = {}
    for item in a.evaluator:
        name, _, path = item.partition("=")
        keys[name] = Path(path).read_bytes().strip()
    base, cand = load_weights(a.base, a.base_engram), load_weights(a.candidate, a.candidate_engram)
    held = load_tokens(Path(a.heldout))
    torch.set_num_threads(a.threads)
    t = time.time()
    v = lc.verify_receipt(r, evaluator_keys=keys, server_secret=Path(a.secret_file).read_bytes().strip(),
                          expected_base_sha256=lc.weights_sha256(base), expected_candidate_sha256=lc.weights_sha256(cand), heldout=held,
                          recompute=lc.cpu_recompute_fn(model_factory(a), base, cand, held, r.window), k=a.k, tol=a.tol,
                          max_regression_tolerance=a.max_regression, min_windows=a.min_windows)
    print(json.dumps({"accepted": v.accepted, "reason": v.reason, "loss_before": v.loss_before, "loss_after": v.loss_after,
                      "delta_loss": v.delta_loss, "checked": list(v.checked), "detail": v.detail, "seconds": round(time.time() - t, 1)}))
    return 0 if v.accepted else 1


def cmd_calibrate(a: argparse.Namespace) -> int:
    flat = load_weights(a.base, a.base_engram)
    held = load_tokens(Path(a.heldout))
    starts = lc.heldout_windows(len(held), a.window, a.n, a.seed)
    out = {"window": a.window, "starts": starts, "device": a.device, "dtype": a.dtype}
    m = model_factory(a)()
    m.load_flat(flat)
    t = time.time()
    dev = lc.window_losses(m, held, starts, a.window, device=a.device, dtype=pick_dtype(a.dtype))
    out["device_seconds"] = round(time.time() - t, 2)
    t = time.time()
    cpu = lc.window_losses(m, held, starts, a.window, device="cpu", dtype=torch.float32)
    out["cpu_seconds"] = round(time.time() - t, 2)
    out["device_losses"], out["cpu_losses"] = dev, cpu
    out["max_abs_diff"] = max(abs(x - y) for x, y in zip(dev, cpu))
    print(json.dumps(out))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--base", required=True)
    e.add_argument("--candidate", required=True)
    e.add_argument("--base-engram", default=None, help="engram_bf16.safetensors (path or hf:// spec) for current Crowd-v1 checkpoints; omit for trunk-only models")
    e.add_argument("--candidate-engram", default=None, help="engram_bf16.safetensors (path or hf:// spec) for current Crowd-v1 checkpoints; omit for trunk-only models")
    e.add_argument("--heldout", required=True)
    e.add_argument("--evaluator-id", required=True)
    e.add_argument("--key-file", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--window", type=int, default=2048)
    e.add_argument("--n", type=int, default=32)
    e.add_argument("--seed", type=int, required=True, help="published by the backend for this round")
    e.add_argument("--round-id", default="")
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    e.add_argument("--batch", type=int, default=1)
    v = sub.add_parser("verify")
    v.add_argument("--receipt", required=True)
    v.add_argument("--base", required=True)
    v.add_argument("--candidate", required=True)
    v.add_argument("--base-engram", default=None, help="engram_bf16.safetensors (path or hf:// spec) for current Crowd-v1 checkpoints; omit for trunk-only models")
    v.add_argument("--candidate-engram", default=None, help="engram_bf16.safetensors (path or hf:// spec) for current Crowd-v1 checkpoints; omit for trunk-only models")
    v.add_argument("--heldout", required=True)
    v.add_argument("--evaluator", action="append", required=True, help="NAME=KEYFILE, repeatable")
    v.add_argument("--secret-file", required=True)
    v.add_argument("--k", type=int, default=2)
    v.add_argument("--tol", type=float, default=2e-3)
    v.add_argument("--max-regression", type=float, default=0.0)
    v.add_argument("--min-windows", type=int, default=8)
    v.add_argument("--threads", type=int, default=2)
    c = sub.add_parser("calibrate")
    c.add_argument("--base", required=True)
    c.add_argument("--base-engram", default=None, help="engram_bf16.safetensors (path or hf:// spec) for current Crowd-v1 checkpoints; omit for trunk-only models")
    c.add_argument("--heldout", required=True)
    c.add_argument("--window", type=int, default=2048)
    c.add_argument("--n", type=int, default=2)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--device", default="cuda")
    c.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    a = ap.parse_args(argv)
    return {"evaluate": cmd_evaluate, "verify": cmd_verify, "calibrate": cmd_calibrate}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
