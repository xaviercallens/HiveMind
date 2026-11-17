#!/usr/bin/env python3
"""HiveMind GPU evaluator for the CrowdGPT loss check: fetch a job from the CrowdGPT backend, compute, post the receipt.

  python scripts/crowdgpt_loss_check_agent.py --server https://server.crowdgpt.net --evaluator-id hivemind-t4 \\
      --key-file ~/.config/hivemind/crowdgpt_evaluator_key --governor-state /path/governor_state.json --once

Shared GPU: with --governor-state, the agent works only while the governor reports a state in --allowed-states (default
"training": the GPU is idle by the owner's policy) and the state file is fresh, and it stops between windows as soon as that
changes, so inference keeps priority. Every downloaded file is checked against the SHA-256 the job names before use; a
mismatch refuses the job. The key is read from a file and never printed.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.swarm import crowdgpt_loss_check as lc  # noqa: E402
from hivemind.swarm.crowdgpt_loss_check_service import LossCheckJob  # noqa: E402

UA = "hivemind-crowdgpt-loss-check/1"


def http_get(url: str, headers: dict[str, str] | None = None, timeout: float = 60.0) -> tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def http_post_json(url: str, body: str, timeout: float = 600.0) -> tuple[int, dict]:
    req = urllib.request.Request(url, data=body.encode(), method="POST", headers={"User-Agent": UA, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            return e.code, json.loads(raw or b"{}")
        except ValueError:
            return e.code, {"error": raw[:300].decode(errors="replace")}


def download(url: str, dest: Path, byte_range: str = "") -> Path:
    """Stream url (optionally a byte range) to dest atomically."""
    headers = {"User-Agent": UA}
    if byte_range:
        headers["Range"] = f"bytes={byte_range}"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=3600) as r, \
            tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        while chunk := r.read(8 << 20):
            tmp.write(chunk)
    Path(tmp.name).replace(dest)
    return dest


def fetch_weights(url: str, sha256: str, cache: Path) -> torch.Tensor:
    path = cache / f"weights_{sha256}.safetensors"
    if not path.exists():
        download(url, path)
    flat = lc.load_flat_safetensors(path)
    if lc.weights_sha256(flat) != sha256:
        path.unlink()
        raise ValueError(f"weights from {url} do not match the job's SHA-256")
    return flat


def fetch_heldout(url: str, byte_range: str, sha256: str, cache: Path) -> np.ndarray:
    path = cache / f"heldout_{sha256}.bin"
    if not path.exists():
        download(url, path, byte_range)
    tokens = np.frombuffer(path.read_bytes(), dtype=lc.TOKEN_DTYPE)
    if lc.tokens_sha256(tokens) != sha256:
        path.unlink()
        raise ValueError(f"held-out tokens from {url} do not match the job's SHA-256")
    return tokens


def governor_gate(state_file: Path | None, allowed: set[str], max_age_s: float, now: Callable[[], float] = time.time) -> tuple[bool, str]:
    """(may the GPU be used now?, why). No state file configured -> always allowed (a dedicated GPU)."""
    if state_file is None:
        return True, "no governor configured"
    try:
        st = json.loads(state_file.read_text())
    except (OSError, ValueError) as e:
        return False, f"governor state unreadable ({type(e).__name__}): not using a shared GPU blind"
    age = now() - state_file.stat().st_mtime
    if age > max_age_s:
        return False, f"governor state is {age:.0f} s old (limit {max_age_s:.0f} s)"
    state = str(st.get("state", ""))
    return (state in allowed), f"governor state {state!r}"


def run_once(a: argparse.Namespace, get: Callable = http_get, post: Callable = http_post_json) -> dict:
    gate = lambda: governor_gate(a.governor_state, set(a.allowed_states), a.max_state_age_s)  # noqa: E731
    ok, why = gate()
    if not ok:
        return {"status": "waiting", "why": why}
    status, raw = get(f"{a.server.rstrip('/')}/fl/loss_check/job?evaluator_id={a.evaluator_id}")
    if status == 204:
        return {"status": "no_job"}
    if status != 200:
        return {"status": "error", "why": f"GET job -> HTTP {status}"}
    job = LossCheckJob.from_wire(json.loads(raw))
    base = fetch_weights(job.base_url, job.base_sha256, a.cache_dir)
    cand = fetch_weights(job.candidate_url, job.candidate_sha256, a.cache_dir)
    held = fetch_heldout(job.heldout_url, job.heldout_range, job.heldout_sha256, a.cache_dir)
    t = time.time()
    try:
        r = lc.make_receipt(evaluator_id=a.evaluator_id, key=Path(a.key_file).read_bytes().strip(), base=base, candidate=cand, heldout=held,
                            window=job.window, n=job.n, seed=job.seed, model_factory=lc.CrowdV1, device=a.device,
                            dtype={"float32": torch.float32, "bfloat16": torch.bfloat16}[a.dtype], round_id=job.job_id, batch=a.batch,
                            stop=lambda: not gate()[0])
    except lc.Interrupted as e:
        return {"status": "yielded", "job_id": job.job_id, "why": str(e)}
    finally:
        if str(a.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
    code, verdict = post(f"{a.server.rstrip('/')}/fl/loss_check/receipt", r.to_json())
    return {"status": "submitted", "job_id": job.job_id, "http": code, "seconds": round(time.time() - t, 1), "mean_before": r.mean_before,
            "mean_after": r.mean_after, "response": verdict}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", required=True)
    ap.add_argument("--evaluator-id", required=True)
    ap.add_argument("--key-file", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16"))
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "hivemind-crowdgpt")
    ap.add_argument("--governor-state", type=Path, default=None)
    ap.add_argument("--allowed-states", nargs="+", default=["training"])
    ap.add_argument("--max-state-age-s", type=float, default=30.0)
    ap.add_argument("--poll-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    while True:
        try:
            out = run_once(a)
        except (OSError, ValueError, RuntimeError) as e:
            out = {"status": "error", "why": f"{type(e).__name__}: {e}"}
        print(json.dumps(out | {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}), flush=True)
        if a.once:
            return 0 if out["status"] in ("submitted", "no_job", "waiting", "yielded") else 1
        time.sleep(a.poll_s)


if __name__ == "__main__":
    raise SystemExit(main())
