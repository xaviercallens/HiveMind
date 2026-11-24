# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Score many MoE checkpoints against the dense base on ONE chunk split, loading the model and the base arm once.

    CUDA_VISIBLE_DEVICES= python scripts/crowdgpt_moe_multi_eval.py --spec FROZEN_SPEC.json --skip-eligible 40 \\
        --candidates cands.json --out-dir OUT --rows-cache-dir R --tokenizer-json T

Same chunks, scoring, hash checks, row-order check and base-equivalence gate as ``crowdgpt_moe_paired_eval.py`` (it
imports its functions). It drops the same-row row-start control, so a candidate costs one MoE arm (about 9 min on 4
CPU threads) instead of two arms plus the base. Resumable: the base arm and every finished candidate are written to
``OUT`` and skipped on restart. Before each candidate it waits until enough RAM is available, because other projects
share this machine. Per-candidate receipts hold per-chunk ``base_nll`` and ``moe_nll`` for the decision code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]


def _pe():
    spec = importlib.util.spec_from_file_location("crowdgpt_moe_paired_eval", REPO / "scripts" / "crowdgpt_moe_paired_eval.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def mem_available_gb(meminfo_text: str) -> float:
    """MemAvailable of a /proc/meminfo text, in GiB (0.0 if absent)."""
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024 * 1024)
    return 0.0


def signature(chunks: list[dict]) -> str:
    """Stable identity of a chunk selection: which (expert, dataset, row) were scored, in order."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(f"{c['expert_id']}|{c['dataset_id']}|{c['row']};".encode())
    return h.hexdigest()


def pending(candidates: dict[str, str], out_dir: Path, skip: int) -> list[str]:
    """Candidate names without a finished receipt for this split, in the given order."""
    return [n for n in candidates if not (out_dir / f"{n}_skip{skip}.json").exists()]


def _sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--skip-eligible", required=True, type=int)
    ap.add_argument("--candidates", required=True, type=Path, help="JSON {name: checkpoint path}")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--rows-cache-dir", required=True, type=Path)
    ap.add_argument("--train-rows-cache-dir", default=Path("/mnt/disks/disk-socrateai-local-1/gcpcrowd_moe/rows"), type=Path)
    ap.add_argument("--tokenizer-json", required=True, type=Path)
    ap.add_argument("--base-cache-dir", default=Path("/mnt/disks/disk-socrateai-local-1/crowd_v1_cache"), type=Path)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--min-avail-gb", type=float, default=9.0)
    a = ap.parse_args(argv)

    def log(m: str) -> None:
        print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {m}", file=sys.stderr, flush=True)

    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "" or torch.cuda.is_available():
        print("refused: run with CUDA_VISIBLE_DEVICES= (CPU only)", file=sys.stderr)
        return 2
    torch.set_num_threads(a.threads)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    pe = _pe()
    spec = json.loads(a.spec.read_text())
    base_spec, moe_spec, chunking = spec["arms"]["base"], spec["arms"]["moe"], spec["chunking"]
    cands: dict[str, str] = json.loads(a.candidates.read_text())
    todo = pending(cands, a.out_dir, a.skip_eligible)
    base_path = a.out_dir / f"base_skip{a.skip_eligible}.json"
    if not todo and base_path.exists():
        log("nothing to do")
        return 0
    for path, want in [(spec["data"]["manifest"], spec["data"]["manifest_sha256"]),
                       (base_spec["weights_file"], base_spec["weights_sha256"]),
                       (str(a.tokenizer_json), base_spec["tokenizer_sha256"])]:
        if _sha(path) != want:
            print(f"refused: {path} does not match the registered sha256", file=sys.stderr)
            return 3
    from tokenizers import Tokenizer

    from hivemind.swarm.crowdgpt_real_shards import LocalHFCacheFetcher, _cached_rows, fetch_rows_datasets_server

    manifest = json.loads(Path(spec["data"]["manifest"]).read_text())
    local = LocalHFCacheFetcher()
    order = pe.check_local_row_order(manifest, a.train_rows_cache_dir, local)
    if not all(o["match"] and not o["block_at_train_end_cached"] for o in order):
        print(f"refused: local row order differs from training: {order}", file=sys.stderr)
        return 4
    if a.rows_cache_dir.resolve() == a.train_rows_cache_dir.resolve():
        print("refused: --rows-cache-dir must not be the training run's cache", file=sys.stderr)
        return 2
    tok = Tokenizer.from_file(str(a.tokenizer_json))
    chunk_offset = int(chunking.get("chunk_offset", 0))
    chunks, selection = pe.select_test_chunks(
        manifest,
        {"local_hf_cache": lambda ds, cfg, sp, off, n: _cached_rows(a.rows_cache_dir, local, ds, cfg, sp, off, n),
         "datasets_server": lambda ds, cfg, sp, off, n: _cached_rows(a.rows_cache_dir, fetch_rows_datasets_server, ds, cfg, sp, off, n)},
        lambda s: tok.encode(s).ids, chunks_per_expert=int(chunking["chunks_per_expert"]),
        chunk_tokens=int(chunking["chunk_tokens"]), scan_cap_rows=int(chunking.get("scan_cap_rows_per_source", 2000)),
        chunk_offset=chunk_offset, skip_eligible=a.skip_eligible)
    sig = signature(chunks)
    log(f"split skip={a.skip_eligible}: {len(chunks)} chunks " + ", ".join(f"{e['expert_id'][:2]}={e['chunks']}" for e in selection))

    def wait_ram(label: str) -> bool:
        waited = 0
        while mem_available_gb(Path("/proc/meminfo").read_text()) < a.min_avail_gb:
            if waited % 600 == 0:
                log(f"waiting for RAM (>= {a.min_avail_gb} GiB available) before {label}")
            time.sleep(60)
            waited += 60
            if waited > 6 * 3600:
                return False
        return True

    if not wait_ram("loading the model"):
        print("refused: RAM stayed short for 6 h", file=sys.stderr)
        return 6
    worker = pe._load_worker()
    model, rev_meta = worker._load_or_build_model(cache_dir=a.base_cache_dir, use_small_arch=False,
                                                  direct_trunk_device="cpu", direct_trunk_dtype=torch.float32)
    if Path(str(rev_meta.get("source"))).resolve() != Path(base_spec["weights_file"]).resolve():
        print("refused: base weights loaded from an unregistered file", file=sys.stderr)
        return 3
    base_nll: list[float]
    if base_path.exists() and json.loads(base_path.read_text())["signature"] == sig:
        base_nll = json.loads(base_path.read_text())["base_nll"]
        log("base arm reused from disk")
    else:
        log("base arm")
        base_nll = pe.eval_arm(model, chunks, log=log)
        base_path.write_text(json.dumps({"signature": sig, "n": len(chunks), "base_nll": base_nll}))
    pe.upcycle_and_load(model, moe_spec["split"], None, worker.load_init_buckets)
    c0 = chunks[0]
    z = pe.chunk_nll(model, c0["ids"], score_from=int(c0.get("score_from", 0)))
    if abs(z - base_nll[0]) > 1e-3:
        print(f"refused: zero-delta upcycled model differs from dense on chunk 0: {z} vs {base_nll[0]}", file=sys.stderr)
        return 5
    log(f"base-equivalence gate passed ({z:.6f} vs {base_nll[0]:.6f})")
    from hivemind.swarm.crowdgpt_moe_opendiloco import extract_moe_trainable_buckets, load_moe_trainable_buckets

    nfrag = int(moe_spec["split"]["num_router_fragments"])
    template = extract_moe_trainable_buckets(model, num_router_fragments=nfrag)
    for name in todo:
        if not wait_ram(name):
            print("refused: RAM stayed short for 6 h", file=sys.stderr)
            return 6
        load_moe_trainable_buckets(model, worker.load_init_buckets(cands[name], template), num_router_fragments=nfrag)
        log(f"scoring {name}")
        moe_nll = pe.eval_arm(model, chunks, log=None)
        rec = {"name": name, "checkpoint": cands[name], "checkpoint_sha256": _sha(cands[name]), "skip_eligible": a.skip_eligible,
               "signature": sig, "n_chunks": len(chunks),
               "chunks": [{"expert_id": c["expert_id"], "dataset_id": c["dataset_id"], "row": c["row"],
                           "base_nll": b, "moe_nll": m} for c, b, m in zip(chunks, base_nll, moe_nll)]}
        (a.out_dir / f"{name}_skip{a.skip_eligible}.json").write_text(json.dumps(rec))
        log(f"{name}: mean diff {sum(m - b for b, m in zip(base_nll, moe_nll)) / len(chunks):+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
