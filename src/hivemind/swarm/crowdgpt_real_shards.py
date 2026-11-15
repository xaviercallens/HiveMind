# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Real token shards for the CrowdGPT MoE OpenDiLoCo round, built from the verified 8-expert dataset plan.

The weekend worker used to train on a repeating synthetic 16-token pattern and evaluate on a slice of the same stream,
which measures nothing. This module samples real rows for each expert from the plan's allow-listed sources
(``load_crowdgpt_moe_dataset_plan``), tokenizes them with the Crowd-v1 tokenizer, and returns:

* two non-IID learner shards: experts assigned to a Tesla T4 cohort go to learner 0, the others to learner 1;
* a held-out stream made of DIFFERENT rows of every expert (never used for training);
* a manifest recording, per expert, the sources, row ranges and token counts actually used.

Rules: sources on the plan's decontamination benchmark list are never used (an expert left with no source is skipped
and reported); rows are fetched read-only through the public datasets-server API and cached on disk; nothing is uploaded.
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"
METADATA_KEYS = frozenset({"id", "url", "file_path", "dump", "date", "uuid", "language", "language_score", "token_count", "score", "int_score"})
MAX_FIELD_CHARS = 4000
ROWS_PER_REQUEST = 100  # the datasets-server maximum

RowFetcher = Callable[[str, str, str, int, int], list[Mapping[str, Any]]]


def fetch_rows_datasets_server(dataset_id: str, config: str, split: str, offset: int, length: int,
                               *, timeout_s: float = 60.0, retries: int = 4, backoff_s: float = 2.0) -> list[Mapping[str, Any]]:
    """Read ``length`` rows from the public Hugging Face datasets-server (read-only, no token)."""
    q = urllib.parse.urlencode({"dataset": dataset_id, "config": config, "split": split, "offset": int(offset),
                                "length": int(min(length, ROWS_PER_REQUEST))})
    req = urllib.request.Request(f"{DATASETS_SERVER_ROWS}?{q}", headers={"User-Agent": "HiveMind-CrowdGPT-RealShards/1"})
    body: Any = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310 - fixed https host
                body = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            # 5xx and 429 from the datasets-server are usually transient; 4xx (wrong config, gated) is not
            if attempt >= retries or not (exc.code >= 500 or exc.code == 429):
                raise
        except urllib.error.URLError:
            if attempt >= retries:
                raise
        time.sleep(backoff_s * (2 ** attempt))
    if not isinstance(body, dict) or not isinstance(body.get("rows"), list):
        raise ValueError(f"unexpected datasets-server answer for {dataset_id}:{config}")
    return [r["row"] for r in body["rows"] if isinstance(r, dict) and isinstance(r.get("row"), dict)]


def row_to_text(row: Mapping[str, Any]) -> str:
    """Plain text of one dataset row: its ``text`` field if present, else 'key: value' lines of its content fields."""
    text = row.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()[:MAX_FIELD_CHARS * 2]
    parts: list[str] = []
    for key, value in row.items():
        if key in METADATA_KEYS or value is None:
            continue
        if isinstance(value, (list, tuple)):
            if len(value) > 64:  # long numeric arrays (e.g. spectra) carry no text
                continue
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        s = str(value).strip()
        if s:
            parts.append(f"{key}: {s[:MAX_FIELD_CHARS]}")
    return "\n".join(parts)


@dataclass
class ExpertSample:
    expert_id: str
    learner: int
    sources: list[dict[str, Any]] = field(default_factory=list)
    train_tokens: int = 0
    heldout_tokens: int = 0
    skipped_reason: str | None = None


def _cached_rows(cache_dir: Path, fetch: RowFetcher, dataset_id: str, config: str, split: str,
                 offset: int, length: int) -> list[Mapping[str, Any]]:
    key = hashlib.sha256(f"{dataset_id}|{config}|{split}|{offset}|{length}".encode()).hexdigest()[:24]
    path = cache_dir / f"rows_{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = list(fetch(dataset_id, config, split, offset, length))
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return rows


def build_real_expert_shards(
    plan: Mapping[str, Any],
    encode: Callable[[str], Sequence[int]],
    vocab_size: int,
    *,
    cache_dir: Path,
    train_tokens_per_expert: int = 16_384,
    heldout_tokens_per_expert: int = 2_048,
    fetch: RowFetcher = fetch_rows_datasets_server,
    split: str = "train",
    heldout_rows: int = 40,
    max_rows_per_source: int = 600,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Return ``(shard_learner0, shard_learner1, held_out, manifest)`` from real rows of the plan's experts.

    Held-out rows are the FIRST ``heldout_rows`` rows of each source and training rows come after them, so the two never
    share a row. Tokens must fall inside ``vocab_size``.
    """
    if train_tokens_per_expert < 1 or heldout_tokens_per_expert < 1:
        raise ValueError("token budgets must be >= 1")
    banned = {str(b) for b in plan.get("decontamination_benchmarks", [])}
    shards: tuple[list[int], list[int]] = ([], [])
    held: list[int] = []
    experts: list[ExpertSample] = []
    for expert in plan.get("experts", []):
        cohort = str(expert.get("assigned_cohort", ""))
        sample = ExpertSample(str(expert.get("expert_id")), 0 if "T4" in cohort else 1)
        sources = [s for s in expert.get("accepted_sources", []) if str(s.get("dataset_id")) not in banned]
        if not sources:
            sample.skipped_reason = "every source is on the decontamination benchmark list"
            experts.append(sample)
            continue
        train_ids: list[int] = []
        held_ids: list[int] = []
        for src in sources:
            ds, cfg = str(src["dataset_id"]), str(src.get("config") or "default")
            used = {"dataset_id": ds, "config": cfg, "split": split, "license": src.get("license"),
                    "heldout_rows": [0, 0], "train_rows": [heldout_rows, heldout_rows]}
            try:
                for off in range(0, heldout_rows, ROWS_PER_REQUEST):
                    if len(held_ids) >= heldout_tokens_per_expert:
                        break
                    rows = _cached_rows(cache_dir, fetch, ds, cfg, split, off, min(ROWS_PER_REQUEST, heldout_rows - off))
                    for r in rows:
                        held_ids.extend(encode(row_to_text(r)))
                    used["heldout_rows"][1] = off + len(rows)
                off = heldout_rows
                while len(train_ids) < train_tokens_per_expert and off < heldout_rows + max_rows_per_source:
                    rows = _cached_rows(cache_dir, fetch, ds, cfg, split, off, ROWS_PER_REQUEST)
                    if not rows:
                        break
                    for r in rows:
                        train_ids.extend(encode(row_to_text(r)))
                    off += len(rows)
                    used["train_rows"][1] = off
            except Exception as exc:  # noqa: BLE001 - one unreachable source must not abort the round
                used["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            sample.sources.append(used)
            if len(train_ids) >= train_tokens_per_expert and len(held_ids) >= heldout_tokens_per_expert:
                break
        train_ids, held_ids = train_ids[:train_tokens_per_expert], held_ids[:heldout_tokens_per_expert]
        if any(t < 0 or t >= vocab_size for t in train_ids + held_ids):
            raise ValueError(f"{sample.expert_id}: tokenizer produced ids outside the model vocabulary ({vocab_size})")
        if not train_ids:
            sample.skipped_reason = "no training rows could be read"
            experts.append(sample)
            continue
        sample.train_tokens, sample.heldout_tokens = len(train_ids), len(held_ids)
        shards[sample.learner].extend(train_ids)
        held.extend(held_ids)
        experts.append(sample)
    if not shards[0] or not shards[1] or not held:
        raise ValueError("real data gave an empty learner shard or held-out stream; see the manifest")
    manifest = {
        "schema": "hivemind.crowdgpt.real_shards.v1",
        "evidence": "measured",
        "data": "real rows from the S21 dataset plan via datasets-server (read-only)",
        "decontamination_excluded": sorted(banned),
        "split_rule": "held-out rows precede training rows in every source; no row is shared",
        "learner_assignment": "learner 0 = experts in a Tesla T4 cohort, learner 1 = the others",
        "learner_tokens": [len(shards[0]), len(shards[1])],
        "heldout_tokens": len(held),
        "experts": [vars(e) for e in experts],
    }
    return (torch.tensor(shards[0], dtype=torch.long), torch.tensor(shards[1], dtype=torch.long),
            torch.tensor(held, dtype=torch.long), manifest)
