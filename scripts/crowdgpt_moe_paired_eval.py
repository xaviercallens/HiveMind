#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""IP-300: independent, pre-registered paired held-out evaluation of base Crowd-v1 vs base + MoE LoRA add-on.

The frozen spec is ``docs/E_B3_MOE_PAIRED_PREREGISTRATION.json``. This script follows it:

* test rows start at each source's ``train_rows[1]`` in the training run's ``real_data_manifest.json``. Those rows were
  never read by training (rows ``[40, train_end)``) nor by the training ratchet's evaluation (rows ``[0, 40)``);
* one 64-token chunk per row (the row's first 64 token ids), ``chunks_per_expert`` per trained expert;
* per-chunk mean next-token NLL for the dense base, then for the same model upcycled in place with the checkpoint's
  buckets (built and loaded with the worker's code path), paired per chunk;
* a paired bootstrap 95 % CI of ``mean(NLL_moe - NLL_base)`` decides PASS / HARM / NO_GAIN / INSUFFICIENT_N.

CPU only: the script refuses to run if CUDA is visible. It reads the training run's files, it never writes to them.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from hivemind.swarm.crowdgpt_real_shards import (  # noqa: E402
    ROWS_PER_REQUEST,
    _cached_rows,
    row_to_text,
)

SPEC_PATH = REPO_ROOT / "docs" / "E_B3_MOE_PAIRED_PREREGISTRATION.json"
BANNED_BENCHMARKS = frozenset({"allenai/ai2_arc", "google-research-datasets/mbpp", "openai/openai_humaneval"})

RowFetcher = Callable[[str, str, str, int, int], list[Mapping[str, Any]]]


# ---------------------------------------------------------------------------
# chunk selection
# ---------------------------------------------------------------------------

def select_test_chunks(
    manifest: Mapping[str, Any],
    fetchers: Mapping[str, RowFetcher],
    encode: Callable[[str], Sequence[int]],
    *,
    chunks_per_expert: int = 40,
    chunk_tokens: int = 64,
    scan_cap_rows: int = 2000,
    banned: frozenset[str] = BANNED_BENCHMARKS,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Chunks from rows after each source's training range, one chunk per row, plus a per-expert report.

    ``fetchers`` maps a manifest ``origin`` (``local_hf_cache`` / ``datasets_server``) to a row fetcher
    ``(dataset_id, config, split, offset, length) -> rows``. A later source of an expert is read only if the earlier
    ones gave too few chunks. Experts that were not trained, or whose sources have no rows left, are reported as skipped.
    """
    chunks: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    for expert in manifest.get("experts", []):
        eid = str(expert.get("expert_id"))
        entry: dict[str, Any] = {"expert_id": eid, "sources": [], "chunks": 0, "skipped_reason": None}
        report.append(entry)
        if expert.get("skipped_reason") or not expert.get("sources") or not expert.get("train_tokens"):
            entry["skipped_reason"] = f"not trained ({expert.get('skipped_reason') or 'no sources'})"
            continue
        got: list[dict[str, Any]] = []
        for src in expert["sources"]:
            if len(got) >= chunks_per_expert:
                break
            ds = str(src["dataset_id"])
            if ds in banned:
                raise ValueError(f"{eid}: {ds} is a decontamination benchmark and must never be used")
            cfg, split = str(src.get("config") or "default"), str(src.get("split") or "train")
            origin = str(src.get("origin") or "datasets_server")
            if origin not in fetchers:
                raise ValueError(f"{eid}: no fetcher for origin {origin!r}")
            start = int(src["train_rows"][1])
            if start < int(src["heldout_rows"][1]):
                raise ValueError(f"{eid}/{ds}: train_rows end {start} precedes the held-out end")
            used = {"dataset_id": ds, "origin": origin, "first_row": start, "rows_scanned": 0, "rows_used": [],
                    "rows_too_short": 0, "exhausted": False}
            off = start
            while len(got) < chunks_per_expert and off < start + scan_cap_rows:
                length = min(ROWS_PER_REQUEST, start + scan_cap_rows - off)
                rows = fetchers[origin](ds, cfg, split, off, length)
                if not rows:
                    used["exhausted"] = True
                    break
                for i, row in enumerate(rows):
                    if len(got) >= chunks_per_expert:
                        break
                    used["rows_scanned"] += 1
                    ids = list(encode(row_to_text(row)))
                    if len(ids) < chunk_tokens:
                        used["rows_too_short"] += 1
                        continue
                    got.append({"expert_id": eid, "dataset_id": ds, "row": off + i, "ids": [int(t) for t in ids[:chunk_tokens]]})
                    used["rows_used"].append(off + i)
                if len(rows) < length:
                    used["exhausted"] = len(got) < chunks_per_expert
                    break
                off += len(rows)
            entry["sources"].append(used)
        entry["chunks"] = len(got)
        if not got:
            entry["skipped_reason"] = "no unseen rows after the training range"
        chunks.extend(got)
    return chunks, report


def chunk_ids_sha256(chunks: Sequence[Mapping[str, Any]]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(json.dumps([c["expert_id"], c["dataset_id"], c["row"], c["ids"]]).encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# model evaluation
# ---------------------------------------------------------------------------

def chunk_nll(model: torch.nn.Module, ids: Sequence[int]) -> float:
    """Mean next-token NLL (nats) of one chunk: the worker's forward, batch size 1, over len(ids) - 1 predictions."""
    from hivemind.swarm.gcp_t4_crowd_diloco import forward_crowd_v1_checkpointed

    x = torch.tensor([list(ids)], dtype=torch.long)
    with torch.no_grad():
        _, loss = forward_crowd_v1_checkpointed(model, x, labels=x, use_grad_checkpointing=False,
                                                token_chunk_size=max(2, len(ids)))
    return float(loss.item())


def eval_arm(model: torch.nn.Module, chunks: Sequence[Mapping[str, Any]], *, log: Callable[[str], None] | None = None) -> list[float]:
    model.eval()
    out: list[float] = []
    for i, c in enumerate(chunks):
        out.append(chunk_nll(model, c["ids"]))
        if log and (i + 1) % 20 == 0:
            log(f"  {i + 1}/{len(chunks)} chunks")
    return out


def upcycle_and_load(model: torch.nn.Module, split: Mapping[str, Any], buckets_path: str | None,
                     load_init_buckets: Callable[[str, dict], dict]) -> dict[str, Any]:
    """Upcycle ``model`` in place with the registered split; load ``buckets_path`` if given (else zero-delta start)."""
    from hivemind.swarm.crowdgpt_moe_opendiloco import (
        extract_moe_trainable_buckets,
        load_moe_trainable_buckets,
        upcycle_crowd_v1_to_sparse_moe,
    )

    rep = upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode=str(split["split_mode"]),
        num_experts=int(split["num_experts"]),
        top_k=int(split["top_k"]),
        lora_rank=int(split["lora_rank"]),
        routing_mode=str(split["routing_mode"]),
        resident_expert_indices=tuple(int(i) for i in split["resident_expert_indices"]),
    )
    template = extract_moe_trainable_buckets(model, num_router_fragments=int(split["num_router_fragments"]))
    if buckets_path:
        buckets = load_init_buckets(buckets_path, template)
        load_moe_trainable_buckets(model, buckets, num_router_fragments=int(split["num_router_fragments"]))
    return {"num_layers_upcycled": getattr(rep, "num_layers_upcycled", None),
            "trainable_parameters": int(sum(v.numel() for v in template.values()))}


# ---------------------------------------------------------------------------
# statistics and decision
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(diffs: Sequence[float], *, n_resamples: int = 10_000, seed: int = 20261118,
                        alpha: float = 0.05) -> tuple[float, float, float]:
    """``(mean, lo, hi)``: percentile CI of the mean of ``diffs`` under i.i.d. resampling with replacement."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        raise ValueError("no paired differences")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_resamples, d.size))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(d.mean()), float(lo), float(hi)


def sign_test(diffs: Sequence[float]) -> dict[str, Any]:
    """Exact two-sided sign test (ties dropped)."""
    neg = sum(1 for x in diffs if x < 0)
    pos = sum(1 for x in diffs if x > 0)
    n = neg + pos
    k = min(neg, pos)
    p = 1.0 if n == 0 else min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"moe_lower": neg, "moe_higher": pos, "ties": len(diffs) - n, "p_two_sided": p}


def decide(n: int, ci_lo: float, ci_hi: float, *, min_n: int = 200) -> str:
    if n < min_n:
        return "INSUFFICIENT_N"
    if ci_hi < 0:
        return "PASS"
    if ci_lo > 0:
        return "HARM"
    return "NO_GAIN"


def per_expert_summary(chunks: Sequence[Mapping[str, Any]], base: Sequence[float], moe: Sequence[float]) -> list[dict[str, Any]]:
    order: list[str] = []
    for c in chunks:
        if c["expert_id"] not in order:
            order.append(c["expert_id"])
    out = []
    for eid in order:
        ix = [i for i, c in enumerate(chunks) if c["expert_id"] == eid]
        b = float(np.mean([base[i] for i in ix]))
        m = float(np.mean([moe[i] for i in ix]))
        out.append({"expert_id": eid, "n": len(ix), "base_nll": b, "moe_nll": m, "mean_diff": m - b,
                    "relative_change_pct": 100.0 * (m - b) / b, **{"sign_test": sign_test([moe[i] - base[i] for i in ix])}})
    return out


def domain_guard(per_expert: Sequence[Mapping[str, Any]], *, max_ratio: float = 1.01) -> bool:
    return all(float(e["moe_nll"]) <= max_ratio * float(e["base_nll"]) for e in per_expert)


def b3_supported(verdict: str, guard_ok: bool) -> bool:
    return verdict == "PASS" and guard_ok


# ---------------------------------------------------------------------------
# integrity checks
# ---------------------------------------------------------------------------

def sha256_file(path: str | os.PathLike[str]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def check_local_row_order(manifest: Mapping[str, Any], train_cache_dir: Path, fetch: RowFetcher) -> list[dict[str, Any]]:
    """Re-read the last 100 training rows of every local-cache source and compare with the training run's cached copy."""
    out = []
    for expert in manifest.get("experts", []):
        for src in expert.get("sources", []):
            if src.get("origin") != "local_hf_cache":
                continue
            ds, cfg, split = src["dataset_id"], src.get("config") or "default", src.get("split") or "train"
            end = int(src["train_rows"][1])
            off = max(int(src["train_rows"][0]), end - ROWS_PER_REQUEST)
            key = hashlib.sha256(f"{ds}|{cfg}|{split}|{off}|{ROWS_PER_REQUEST}".encode()).hexdigest()[:24]
            path = train_cache_dir / f"rows_{key}.json"
            cached = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
            fresh = json.loads(json.dumps(fetch(ds, cfg, split, off, ROWS_PER_REQUEST), ensure_ascii=False, default=str))
            nkey = hashlib.sha256(f"{ds}|{cfg}|{split}|{end}|{ROWS_PER_REQUEST}".encode()).hexdigest()[:24]
            out.append({"dataset_id": ds, "rows_checked": [off, end], "match": cached is not None and cached == fresh,
                        "block_at_train_end_cached": (train_cache_dir / f"rows_{nkey}.json").exists()})
    return out


def _load_worker():
    spec = importlib.util.spec_from_file_location("gcp_t4_crowd_diloco_worker", REPO_ROOT / "scripts" / "gcp_t4_crowd_diloco_worker.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--spec", default=str(SPEC_PATH))
    ap.add_argument("--rows-cache-dir", required=True, help="scratch cache for test rows (never the training run's cache)")
    ap.add_argument("--train-rows-cache-dir", default="/mnt/disks/disk-socrateai-local-1/gcpcrowd_moe/rows",
                    help="the training run's row cache, read only, for the row-order check")
    ap.add_argument("--tokenizer-json", required=True)
    ap.add_argument("--base-cache-dir", default="/mnt/disks/disk-socrateai-local-1/crowd_v1_cache")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prereg-commit", default="")
    args = ap.parse_args(argv)

    started = _now()
    t0 = time.time()

    def log(msg: str) -> None:
        print(f"[{_now()}] {msg}", file=sys.stderr, flush=True)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "" or torch.cuda.is_available():
        print("refused: run with CUDA_VISIBLE_DEVICES= (CPU only; the T4 is serving and training)", file=sys.stderr)
        return 2
    torch.set_num_threads(int(args.threads))

    spec_bytes = Path(args.spec).read_bytes()
    spec = json.loads(spec_bytes)
    base_spec, moe_spec, chunking = spec["arms"]["base"], spec["arms"]["moe"], spec["chunking"]
    stats_seed, n_resamples = 20261118, 10_000
    deviations: list[str] = []

    for path, want in ((moe_spec["checkpoint"], moe_spec["checkpoint_sha256"]),
                       (spec["data"]["manifest"], spec["data"]["manifest_sha256"]),
                       (base_spec["weights_file"], base_spec["weights_sha256"]),
                       (args.tokenizer_json, base_spec["tokenizer_sha256"])):
        got = sha256_file(path)
        if got != want:
            print(f"refused: {path} sha256 {got} != registered {want}", file=sys.stderr)
            return 3
    log("hashes match the pre-registration")

    from tokenizers import Tokenizer

    from hivemind.swarm.crowdgpt_real_shards import LocalHFCacheFetcher, fetch_rows_datasets_server

    manifest = json.loads(Path(spec["data"]["manifest"]).read_text(encoding="utf-8"))
    local = LocalHFCacheFetcher()
    order = check_local_row_order(manifest, Path(args.train_rows_cache_dir), local)
    if not all(o["match"] and not o["block_at_train_end_cached"] for o in order):
        print(f"refused: local row order differs from training: {order}", file=sys.stderr)
        return 4
    log(f"row-order check passed on {len(order)} local sources")

    cache = Path(args.rows_cache_dir)
    if cache.resolve() == Path(args.train_rows_cache_dir).resolve():
        print("refused: --rows-cache-dir must not be the training run's cache", file=sys.stderr)
        return 2

    def _local(ds: str, cfg: str, split: str, off: int, n: int) -> list[Mapping[str, Any]]:
        if not local.origin_of(ds) == "local_hf_cache":
            raise RuntimeError(f"{ds}: manifest origin is local_hf_cache but no local files are present")
        return _cached_rows(cache, local, ds, cfg, split, off, n)

    def _remote(ds: str, cfg: str, split: str, off: int, n: int) -> list[Mapping[str, Any]]:
        return _cached_rows(cache, fetch_rows_datasets_server, ds, cfg, split, off, n)

    tok = Tokenizer.from_file(str(args.tokenizer_json))
    chunks, selection = select_test_chunks(
        manifest, {"local_hf_cache": _local, "datasets_server": _remote}, lambda s: tok.encode(s).ids,
        chunks_per_expert=int(chunking["chunks_per_expert"]), chunk_tokens=int(chunking["chunk_tokens"]),
        scan_cap_rows=2000)
    log(f"selected {len(chunks)} chunks: " + ", ".join(f"{e['expert_id']}={e['chunks']}" for e in selection))

    worker = _load_worker()
    model, rev_meta = worker._load_or_build_model(cache_dir=Path(args.base_cache_dir), use_small_arch=False,
                                                  direct_trunk_device="cpu", direct_trunk_dtype=torch.float32)
    if Path(str(rev_meta.get("source"))).resolve() != Path(base_spec["weights_file"]).resolve():
        print(f"refused: base weights loaded from {rev_meta.get('source')}, registered {base_spec['weights_file']}", file=sys.stderr)
        return 3
    vocab = int(model.cfg.vocab_size)
    if any(t < 0 or t >= vocab for c in chunks for t in c["ids"]):
        print("refused: token id outside the vocabulary", file=sys.stderr)
        return 3
    log("base arm (dense Crowd-v1)")
    base_nll = eval_arm(model, chunks, log=log) if chunks else []

    eq: dict[str, Any] = {"tolerance_nats": 1e-3}
    if chunks:
        upcycle_and_load(model, moe_spec["split"], None, worker.load_init_buckets)
        eq["zero_delta_chunk0_nll"] = chunk_nll(model, chunks[0]["ids"])
        eq["dense_chunk0_nll"] = base_nll[0]
        eq["passed"] = abs(eq["zero_delta_chunk0_nll"] - base_nll[0]) <= 1e-3
        if not eq["passed"]:
            print(f"refused: zero-delta upcycled model differs from dense on chunk 0: {eq}", file=sys.stderr)
            return 5
        log(f"base-equivalence gate passed: {eq}")
        from hivemind.swarm.crowdgpt_moe_opendiloco import extract_moe_trainable_buckets, load_moe_trainable_buckets

        template = extract_moe_trainable_buckets(model, num_router_fragments=int(moe_spec["split"]["num_router_fragments"]))
        buckets = worker.load_init_buckets(moe_spec["checkpoint"], template)
        load_moe_trainable_buckets(model, buckets, num_router_fragments=int(moe_spec["split"]["num_router_fragments"]))
        eq["moe_trainable_parameters"] = int(sum(v.numel() for v in template.values()))
        log("MoE arm (checkpoint loaded)")
        moe_nll = eval_arm(model, chunks, log=log)
    else:
        moe_nll = []

    diffs = [m - b for m, b in zip(moe_nll, base_nll)]
    n = len(diffs)
    mean_d, lo, hi = paired_bootstrap_ci(diffs, n_resamples=n_resamples, seed=stats_seed) if n else (float("nan"),) * 3
    verdict = decide(n, lo, hi, min_n=int(chunking["min_total_chunks"]))
    experts = per_expert_summary(chunks, base_nll, moe_nll) if n else []
    guard = domain_guard(experts) if experts else False
    finished = _now()
    receipt = {
        "schema": "hivemind.e_b3_moe_paired.v1",
        "evidence": "measured",
        "experiment_id": spec["experiment_id"],
        "roadmap_item": "IP-300",
        "preregistration": {"spec_path": "docs/E_B3_MOE_PAIRED_PREREGISTRATION.json",
                            "spec_sha256": hashlib.sha256(spec_bytes).hexdigest(),
                            "commit": args.prereg_commit},
        "script_commit": _git("rev-parse", "HEAD"),
        "started_utc": started,
        "finished_utc": finished,
        "cpu_seconds_wall": round(time.time() - t0, 1),
        "compute": {"device": "cpu", "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "threads": torch.get_num_threads(), "torch": torch.__version__, "numpy": np.__version__,
                    "python": platform.python_version(), "trunk_dtype": "float32"},
        "base": {**base_spec, "loaded": {k: rev_meta.get(k) for k in ("source", "hf_revision", "weights_matched_keys", "weights_total_keys")}},
        "moe": {"checkpoint_sha256": moe_spec["checkpoint_sha256"], "split": moe_spec["split"]},
        "row_order_check": order,
        "selection": selection,
        "n_chunks": n,
        "chunk_tokens": int(chunking["chunk_tokens"]),
        "chunk_ids_sha256": chunk_ids_sha256(chunks),
        "base_equivalence_gate": eq,
        "mean_base_nll": float(np.mean(base_nll)) if n else None,
        "mean_moe_nll": float(np.mean(moe_nll)) if n else None,
        "mean_paired_diff": mean_d,
        "bootstrap95": [lo, hi],
        "bootstrap": {"resamples": n_resamples, "seed": stats_seed, "method": "iid paired, percentile"},
        "sign_test": sign_test(diffs),
        "per_expert": experts,
        "verdict": verdict,
        "domain_guard_ok": guard,
        "b3_supported": b3_supported(verdict, guard),
        "b3_note": "B3 counts this as passing only if verdict == PASS and the +1% domain guard holds; otherwise B3 stays failing",
        "chunks": [{"expert_id": c["expert_id"], "dataset_id": c["dataset_id"], "row": c["row"],
                    "base_nll": base_nll[i], "moe_nll": moe_nll[i]} for i, c in enumerate(chunks)] if n else [],
        "deviations": deviations,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(receipt, indent=1) + "\n", encoding="utf-8")
    log(f"verdict {verdict}: n={n} mean_d={mean_d:.6f} ci=[{lo:.6f}, {hi:.6f}] guard={guard} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
