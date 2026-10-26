"""Server-side creation of QLoRA MoE workunits: tokenised, signed data shards (BOINC ``create_work`` for HiveMind).

Training data comes from Hugging Face datasets that are already in the local cache (no downloads, ``HF_HUB_OFFLINE=1``), one
domain per round, cycling through ``DOMAIN_ORDER``. Every domain has a fixed held-out evaluation slice that is never used for
training, so ``eval_by_domain`` is comparable from round to round. Benchmarks the repository uses for evaluation (MBPP,
HumanEval) are deliberately NOT training sources.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from hivemind.boinc.workunit import sign_shard_bytes

DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
DOMAIN_ORDER = ["general", "french", "edu_web", "news", "biomed", "math"]

# domain -> (dataset dir, train file glob, eval file glob, text columns, rows per training shard, eval start (negative = from the end))
_SOURCES: dict[str, dict[str, Any]] = {
    "general": {"ds": "Salesforce--wikitext", "train": "wikitext-2-raw-v1/train-*.parquet", "eval": "wikitext-2-raw-v1/validation-*.parquet", "cols": ["text"], "rows": 3000, "eval_start": 0},
    "french": {"ds": "wikimedia--wikipedia", "train": "20231101.fr/train-*.parquet", "eval": "20231101.fr/train-*.parquet", "cols": ["text"], "rows": 40, "eval_start": -200},
    "edu_web": {"ds": "HuggingFaceFW--fineweb-edu", "train": "sample/10BT/*.parquet", "eval": "sample/10BT/*.parquet", "cols": ["text"], "rows": 40, "eval_start": -200},
    "news": {"ds": "abisee--cnn_dailymail", "train": "3.0.0/validation-*.parquet", "eval": "3.0.0/test-*.parquet", "cols": ["article"], "rows": 40, "eval_start": 0},
    "biomed": {"ds": "ccdv--pubmed-summarization", "train": "section/validation-*.parquet", "eval": "section/test-*.parquet", "cols": ["article"], "rows": 12, "eval_start": 0},
    "math": {"ds": "openai--gsm8k", "train": "main/train-*.parquet", "eval": "main/test-*.parquet", "cols": ["question", "answer"], "rows": 120, "eval_start": 0},
}


def _file(ds: str, pattern: str) -> Path:
    return sorted((HUB / f"datasets--{ds}" / "snapshots").glob(f"*/{pattern}"))[0]


def _rows(path: Path, cols: list[str], start: int, n: int) -> list[dict[str, Any]]:
    """``n`` rows from ``start`` (negative = from the end), reading only the row groups that overlap (1000 rows each)."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    start = total + start if start < 0 else start
    out: list[dict[str, Any]] = []
    seen = 0
    for g in range(pf.num_row_groups):
        size = pf.metadata.row_group(g).num_rows
        if seen + size > start:
            tbl = pf.read_row_group(g, columns=cols).to_pylist()
            lo = max(start - seen, 0)
            out += tbl[lo : lo + (n - len(out))]
        seen += size
        if len(out) >= n:
            break
    return out


def _ids(tok, docs: list[str], need: int, max_doc_tokens: int = 3000) -> list[int]:
    """Concatenate documents (blank-line separated) until ``need`` tokens are available."""
    ids: list[int] = []
    for d in docs:
        if not d or not d.strip():
            continue
        ids += tok(d.strip() + "\n\n").input_ids[:max_doc_tokens]
        if len(ids) >= need:
            break
    return ids


def _docs(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    return ["\n".join(str(r[c]) for c in cols) for r in rows]


def domain_ids(tok, domain: str, role: str, shard_index: int, need: int) -> list[int]:
    s = _SOURCES[domain]
    if role == "train":
        rows = _rows(_file(s["ds"], s["train"]), s["cols"], shard_index * s["rows"], s["rows"])
    else:
        rows = _rows(_file(s["ds"], s["eval"]), s["cols"], s["eval_start"], 200)
    return _ids(tok, _docs(rows, s["cols"]), need)


def _chunks(ids: list[int], seq_len: int, n: int) -> list[list[int]]:
    return [ids[i * seq_len : (i + 1) * seq_len] for i in range(min(n, len(ids) // seq_len))]


def build_train_spec(
    wu_id: str,
    shard_index: int = 0,
    model_id: str = DEFAULT_MODEL,
    inner_steps: int = 40,
    segment_steps: int = 10,
    seq_len: int = 128,
    n_seq: int = 64,
    n_eval: int = 8,
    micro_batch: int = 2,
    lr: float = 2e-4,
    rank: int = 8,
    alpha: int = 16,
    rng_seed: int = 7,
    init_seed: int = 42,
    start_adapter_sha256: str | None = None,
    domain: str = "general",
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from hivemind.boinc.real_client.moe_app import HF_CACHE

    if domain not in _SOURCES:
        raise ValueError(f"unknown domain {domain!r}; choose from {DOMAIN_ORDER}")
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    train = _chunks(domain_ids(tok, domain, "train", shard_index, n_seq * seq_len), seq_len, n_seq)
    # the held-out set of EVERY domain is always evaluated, so a round trained on French also shows what it did to English
    eval_sets = {d: _chunks(domain_ids(tok, d, "eval", 0, n_eval * seq_len), seq_len, n_eval) for d in DOMAIN_ORDER}
    spec = {
        "wu_id": wu_id, "model_id": model_id, "inner_steps": inner_steps, "segment_steps": segment_steps,
        "micro_batch": micro_batch, "lr": lr, "weight_decay": 0.01, "rank": rank, "alpha": alpha,
        "rng_seed": rng_seed, "init_seed": init_seed, "shard_index": shard_index, "domain": domain,
        "train_tokens": train, "eval_tokens": eval_sets["general"], "eval_sets": eval_sets,
    }
    if start_adapter_sha256:
        spec["start_adapter_sha256"] = start_adapter_sha256
    sha, sig = sign_shard_bytes(json.dumps(train, separators=(",", ":")).encode(), rng_seed)
    spec["shard_sha256"], spec["shard_hmac_signature"] = sha, sig
    return spec


def build_infer_spec(wu_id: str, prompts: list[str], model_id: str = DEFAULT_MODEL, max_new_tokens: int = 24, rank: int = 8, alpha: int = 16, init_seed: int = 42) -> dict[str, Any]:
    return {"wu_id": wu_id, "model_id": model_id, "prompts": prompts, "max_new_tokens": max_new_tokens, "rank": rank, "alpha": alpha, "init_seed": init_seed}
