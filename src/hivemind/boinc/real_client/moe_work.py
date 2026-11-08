# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
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

from hivemind.boinc.real_client.moe_profiles import PROFILES
from hivemind.boinc.workunit import sign_shard_bytes

BASE_SHARD_TOKENS = 64 * 128  # tokens per shard the per-domain row counts below were sized for
DEFAULT_MODEL = "ibm-granite/granite-3.1-1b-a400m-base"
HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
DOMAIN_ORDER = ["general", "french", "edu_web", "news", "biomed", "math",
                "science", "medqa", "legal", "finance", "arxiv", "code", "cyber", "physics", "chemistry", "olympiad_math"]
# the first six are the original cycle; new domains are only ever appended, so a recorded round still names the same domain

# domain -> (dataset dir, train file glob, eval file glob, text columns, rows per training shard, eval start (negative = from the end))
_SOURCES: dict[str, dict[str, Any]] = {
    "general": {"ds": "Salesforce--wikitext", "train": "wikitext-2-raw-v1/train-*.parquet", "eval": "wikitext-2-raw-v1/validation-*.parquet", "cols": ["text"], "rows": 3000, "eval_start": 0},
    "french": {"ds": "wikimedia--wikipedia", "train": "20231101.fr/train-*.parquet", "eval": "20231101.fr/train-*.parquet", "cols": ["text"], "rows": 40, "eval_start": -200},
    "edu_web": {"ds": "HuggingFaceFW--fineweb-edu", "train": "sample/10BT/*.parquet", "eval": "sample/10BT/*.parquet", "cols": ["text"], "rows": 40, "eval_start": -200},
    "news": {"ds": "abisee--cnn_dailymail", "train": "3.0.0/validation-*.parquet", "eval": "3.0.0/test-*.parquet", "cols": ["article"], "rows": 40, "eval_start": 0},
    "biomed": {"ds": "ccdv--pubmed-summarization", "train": "section/validation-*.parquet", "eval": "section/test-*.parquet", "cols": ["article"], "rows": 12, "eval_start": 0},
    "math": {"ds": "openai--gsm8k", "train": "main/train-*.parquet", "eval": "main/test-*.parquet", "cols": ["question", "answer"], "rows": 120, "eval_start": 0},
    # added domains: files come from scripts/moe_fetch_domains.py (pinned commits). Single-file sources hold out their last 200 rows.
    "science": {"ds": "allenai--sciq", "train": "data/train-*.parquet", "eval": "data/test-*.parquet", "cols": ["support", "question", "correct_answer"], "rows": 300, "eval_start": 0},
    "medqa": {"ds": "openlifescienceai--medmcqa", "train": "data/train-*.parquet", "eval": "data/test-*.parquet", "cols": ["question", "exp"], "rows": 200, "eval_start": 0},
    "legal": {"ds": "lex_glue", "train": "case_hold/train-*.parquet", "eval": "case_hold/test-*.parquet", "cols": ["context"], "rows": 300, "eval_start": 0},
    "finance": {"ds": "FinGPT--fingpt-fiqa_qa", "train": "data/train-*.parquet", "eval": "data/train-*.parquet", "cols": ["input", "output"], "rows": 150, "eval_start": -200},
    "arxiv": {"ds": "ccdv--arxiv-summarization", "train": "document/validation-*.parquet", "eval": "document/test-*.parquet", "cols": ["article"], "rows": 6, "eval_start": 0},
    "code": {"ds": "flytech--python-codes-25k", "train": "default/train/0000.parquet", "eval": "default/train/0000.parquet", "cols": ["instruction", "output"], "rows": 100, "eval_start": -200},
    "cyber": {"ds": "CyberNative--Code_Vulnerability_Security_DPO", "train": "default/train/0000.parquet", "eval": "default/train/0000.parquet", "cols": ["question", "chosen"], "rows": 100, "eval_start": -200},
    "physics": {"ds": "camel-ai--physics", "train": "default/train/0000.parquet", "eval": "default/train/0000.parquet", "cols": ["message_1", "message_2"], "rows": 60, "eval_start": -200},
    "chemistry": {"ds": "camel-ai--chemistry", "train": "default/train/0000.parquet", "eval": "default/train/0000.parquet", "cols": ["message_1", "message_2"], "rows": 60, "eval_start": -200},
    "olympiad_math": {"ds": "AI-MO--NuminaMath-CoT", "train": "data/train-00000-of-00005.parquet", "eval": "data/test-*.parquet", "cols": ["problem", "solution"], "rows": 100, "eval_start": 0},
}


def _file(ds: str, pattern: str) -> Path:
    return sorted((HUB / f"datasets--{ds}" / "snapshots").glob(f"*/{pattern}"))[0]


def _total_rows(path: Path) -> int:
    import pyarrow.parquet as pq

    return int(pq.ParquetFile(path).metadata.num_rows)


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
    return ["\n".join(str(r[c]) for c in cols if r.get(c) is not None) for r in rows]  # a null column (e.g. medmcqa has no explanation) is skipped, not written as "None"


def domain_ids(tok, domain: str, role: str, shard_index: int, need: int) -> list[int]:
    if domain not in _SOURCES:
        raise ValueError(f"unknown domain {domain!r}")
    if role not in ("train", "eval"):
        raise ValueError(f"role must be 'train' or 'eval', got {role!r}")
    if isinstance(shard_index, bool) or not isinstance(shard_index, int) or shard_index < 0:
        raise ValueError("shard_index must be a non-negative integer")
    if isinstance(need, bool) or not isinstance(need, int) or need <= 0:
        raise ValueError("need must be a positive integer")
    s = _SOURCES[domain]
    if role == "train":
        path = _file(s["ds"], s["train"])
        usable = _total_rows(path)
        if s["eval_start"] < 0 and path == _file(s["ds"], s["eval"]):
            usable += s["eval_start"]  # train and eval share a file: the held-out tail is never a training shard
        per_shard = s["rows"] * max(1, -(-need // BASE_SHARD_TOKENS))  # a shard of more tokens takes proportionally more rows
        shard_index %= max(usable // per_shard, 1)  # a cursor past the last whole shard wraps instead of reading held-out rows
        rows = _rows(path, s["cols"], shard_index * per_shard, per_shard)
    else:
        rows = _rows(_file(s["ds"], s["eval"]), s["cols"], s["eval_start"], 200)
    return _ids(tok, _docs(rows, s["cols"]), need)


def _chunks(ids: list[int], seq_len: int, n: int) -> list[list[int]]:
    if isinstance(seq_len, bool) or not isinstance(seq_len, int) or seq_len < 1:
        raise ValueError("seq_len must be a positive integer")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a non-negative integer")
    return [ids[i * seq_len : (i + 1) * seq_len] for i in range(min(n, len(ids) // seq_len))]


MIX = "mix"
MIX_DOMAINS = tuple(DOMAIN_ORDER[:6])  # the six the chain evaluates every round: general, french, edu_web, news, biomed, math


def mix_counts(n: int, k: int) -> list[int]:
    """Split ``n`` sequences over ``k`` domains as evenly as possible (the first domains take the remainder); sums to ``n``."""
    if isinstance(n, bool) or not isinstance(n, int) or isinstance(k, bool) or not isinstance(k, int):
        raise ValueError("n and k must be integers")
    if n < k or k <= 0:
        raise ValueError(f"need at least one sequence per domain: n={n}, k={k}")
    base, extra = divmod(n, k)
    return [base + (1 if i < extra else 0) for i in range(k)]


def interleave(parts: list[list[list[int]]]) -> list[list[int]]:
    """Round-robin the per-domain chunk lists so no domain forms a contiguous block of the shard."""
    out: list[list[int]] = []
    for i in range(max((len(p) for p in parts), default=0)):
        out += [p[i] for p in parts if i < len(p)]
    return out


def load_custom_shards(custom_dir: Path | str, secret: bytes | None = None) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    """Read shards written by ``scripts/make_custom_shards.py`` and verify them against the manifest (sha256 of both files, HMAC of the training shard).
    Any mismatch raises: a tampered or truncated shard never becomes a workunit."""
    import hashlib
    import hmac

    d = Path(custom_dir)
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    train_bytes, held_bytes = (d / "train.json").read_bytes(), (d / "heldout.json").read_bytes()
    if hashlib.sha256(train_bytes).hexdigest() != manifest["train_sha256"] or hashlib.sha256(held_bytes).hexdigest() != manifest["heldout_sha256"]:
        raise ValueError(f"custom shards in {d} do not match their manifest (sha256)")
    if secret is not None:
        _, sig = sign_shard_bytes(train_bytes, int(manifest["seed"]), secret)
        if not hmac.compare_digest(sig, manifest["train_hmac_signature"]):
            raise ValueError(f"custom shards in {d} fail the HMAC check (wrong secret or tampered)")
    return json.loads(train_bytes), json.loads(held_bytes), manifest


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
    profile: str | None = None,
    custom_dir: str | None = None,
    custom_secret: bytes | None = None,
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from hivemind.boinc.real_client.moe_app import HF_CACHE

    custom: tuple[list[list[int]], list[list[int]], dict[str, Any]] | None = None
    if custom_dir:  # the user's own dataset, built and signed by scripts/make_custom_shards.py; verified before use
        custom = load_custom_shards(custom_dir, custom_secret)
        domain = str(custom[2]["domain"])
    elif domain not in _SOURCES and domain != MIX:
        raise ValueError(f"unknown domain {domain!r}; choose from {DOMAIN_ORDER} or {MIX!r}")
    prof = PROFILES[profile] if profile else None
    if prof:  # a profile fixes the sizes; explicit arguments are ignored so a big workunit cannot be built half-sized
        seq_len, n_seq, micro_batch, rank, alpha = prof["seq_len"], prof["n_seq"], prof["micro_batch"], prof["rank"], prof["alpha"]
        inner_steps, segment_steps = prof["inner_steps"], prof["segment_steps"]
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_CACHE)
    if custom is not None:
        if int(custom[2]["seq_len"]) != seq_len:
            raise ValueError(f"custom shards were cut at seq_len {custom[2]['seq_len']}, this workunit uses {seq_len}")
        pool = custom[0]
        start = (shard_index * n_seq) % len(pool)
        train = [pool[(start + i) % len(pool)] for i in range(min(n_seq, len(pool)))]  # a cursor past the end wraps; the held-out chunks are a separate file
    elif domain == MIX:
        # a stratified round: every round sees all of MIX_DOMAINS in equal share (one-domain rounds turned the adapter into a math/biomed specialist that lost on the rest)
        counts = mix_counts(n_seq, len(MIX_DOMAINS))
        parts = [_chunks(domain_ids(tok, d, "train", shard_index, c * seq_len), seq_len, c) for d, c in zip(MIX_DOMAINS, counts)]
        train = interleave(parts)
    else:
        train = _chunks(domain_ids(tok, domain, "train", shard_index, n_seq * seq_len), seq_len, n_seq)
    # the held-out set of EVERY domain is always evaluated, so a round trained on French also shows what it did to English
    # a stratified round is judged on the domains it trains on: the chain's ratchet compares means, so the set must match the head's (the six the chain has always reported)
    eval_domains = MIX_DOMAINS if domain == MIX else DOMAIN_ORDER
    eval_sets = {d: _chunks(domain_ids(tok, d, "eval", 0, n_eval * seq_len), seq_len, n_eval) for d in eval_domains}
    if custom is not None:
        eval_sets[domain] = custom[1][:n_eval]  # the user's own held-out chunks are reported next to the built-in domains
    spec = {
        "wu_id": wu_id, "model_id": model_id, "inner_steps": inner_steps, "segment_steps": segment_steps,
        "micro_batch": micro_batch, "lr": lr, "weight_decay": 0.01, "rank": rank, "alpha": alpha,
        "rng_seed": rng_seed, "init_seed": init_seed, "shard_index": shard_index, "domain": domain,
        "train_tokens": train, "eval_tokens": eval_sets["general"], "eval_sets": eval_sets,
    }
    if prof:
        spec.update({"app": prof["app"], "profile": profile, "min_gpu_ram_mb": prof["min_gpu_ram_mb"], "seq_len": seq_len})
    if start_adapter_sha256:
        spec["start_adapter_sha256"] = start_adapter_sha256
    sha, sig = sign_shard_bytes(json.dumps(train, separators=(",", ":")).encode(), rng_seed)
    spec["shard_sha256"], spec["shard_hmac_signature"] = sha, sig
    return spec


def build_infer_spec(wu_id: str, prompts: list[str], model_id: str = DEFAULT_MODEL, max_new_tokens: int = 24, rank: int = 8, alpha: int = 16, init_seed: int = 42) -> dict[str, Any]:
    if not isinstance(wu_id, str) or not wu_id.strip():
        raise ValueError("wu_id must be a non-empty string")
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be a non-empty string")
    if not isinstance(prompts, list) or not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError("prompts must be a non-empty list of non-empty strings")
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 1:
        raise ValueError("rank must be a positive integer")
    if isinstance(alpha, bool) or not isinstance(alpha, int) or alpha < 1:
        raise ValueError("alpha must be a positive integer")
    if isinstance(init_seed, bool) or not isinstance(init_seed, int):
        raise ValueError("init_seed must be an integer")
    return {"wu_id": wu_id, "model_id": model_id, "prompts": prompts, "max_new_tokens": max_new_tokens, "rank": rank, "alpha": alpha, "init_seed": init_seed}
