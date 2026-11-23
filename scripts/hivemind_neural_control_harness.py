#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Pre-registered 4-split anti-leakage ANSE neural-control benchmark harness and receipt verifier.

Implements the cleanroom evaluation, anti-leakage verification, and thermodynamic ratchet
protocol for open-weight Sparse MoE neural control (``NEURAL-CTRL-05``):

1. **4-Split Pairwise Anti-Leakage Verification** (``verify_pairwise_split_disjointness``):
   Enforces zero duplicate or overlapping ``(dataset_id, row)`` keys across all four splits
   (``probe``, ``selection``, ``benchmark``, ``virgin_vault``), failing closed on any leakage.
2. **Paired Bootstrap Receipt Evaluation & ANSE Energy** (``evaluate_neural_control_receipt``):
   Computes paired bootstrap 95% confidence intervals (default 10,000 resamples, configurable
   seed), per-domain relative NLL drift, the ``+1.0%`` per-domain regression guard,
   ``MOSTLY_CONTENT`` survival (``mean_diff <= -0.1653`` nats), and the objective ANSE
   energy function (``1e6`` Maximum Pain on guard failure, ``+0.05`` soft penalty above
   ``+0.85%`` general-domain drift).
3. **Multi-Generation ANSE Ratchet & Single-Shot Virgin Vault** (``run_anse_ratchet_selection``):
   Progresses across ordered generations (``S1 -> S2 -> S3 -> ... -> finalize_vault``),
   accepting a candidate only when it clears the guard, improves monotonically in ANSE energy,
   triggers the stopping gate (0 benchmark / 0 virgin-vault queries burned) if selection fails,
   and promotes at most one champion for single-shot virgin-vault verification with
   deterministic SHA-256 provenance receipts (``--json`` / ``--markdown`` CLI).

Strict CPU-only deterministic execution with zero outbound network calls and zero GPU usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "hivemind.neural_control_harness/1"
RECEIPT_OBJECT = "hivemind.neural_control_receipt"
RATCHET_OBJECT = "hivemind.anse_ratchet_receipt"
SPLITS_OBJECT = "hivemind.split_disjointness_receipt"

SPLIT_NAMES: tuple[str, str, str, str] = ("probe", "selection", "benchmark", "virgin_vault")
DEFAULT_SPLIT_SKIP_POLICY: dict[str, Any] = {
    "probe": 120,
    "selection": 40,
    "benchmark": 80,
    "virgin_vault": 160,
    "frozen_0_untouched": True,
}

GENERAL_EXPERT_PREFIX = "E0_"
MOSTLY_CONTENT_THRESHOLD_NATS = -0.1653
DOMAIN_GUARD_MAX_RATIO = 1.01
ANSE_SAFE_E0_RATIO = 1.0085
ANSE_MAXIMUM_PAIN = 1e6
ANSE_SOFT_MARGIN_PENALTY = 0.05

DEFAULT_BOOTSTRAP_RESAMPLES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20261123
DEFAULT_IP300_BOOTSTRAP_SEED = 20261118
DEFAULT_MIN_CHUNKS_PER_EXPERT = 20
DEFAULT_MIN_TOTAL_CHUNKS = 200

_FORBIDDEN_OUTPUT_RE = re.compile(
    r'"' + "sco" + r're"\s*:\s*\d|referee_' + r'score|is_peer_' + r'review"?\s*:\s*[Tt]rue'
)
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class NeuralControlHarnessError(ValueError):
    """Fail-closed validation, anti-leakage, or provenance error in the neural-control harness."""


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_sha256_hex(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _attach_provenance_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Attach a deterministic ``receipt_sha256`` over the canonical receipt body."""
    core = {k: v for k, v in payload.items() if k != "receipt_sha256"}
    core["receipt_sha256"] = compute_sha256_hex(core)
    return core


def verify_receipt_provenance(receipt_doc: Mapping[str, Any]) -> bool:
    """Verify that ``receipt_doc['receipt_sha256']`` matches its canonical SHA-256 digest."""
    if not isinstance(receipt_doc, Mapping):
        raise NeuralControlHarnessError("Receipt document must be a JSON object.")
    claimed = receipt_doc.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    core = {k: v for k, v in receipt_doc.items() if k != "receipt_sha256"}
    try:
        expected = compute_sha256_hex(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def assert_safe_output(text: str) -> str:
    """Reject any serialized output that matches forbidden gatekeeper patterns."""
    if _FORBIDDEN_OUTPUT_RE.search(text):
        raise NeuralControlHarnessError("Output safety guard blocked forbidden metric key pattern.")
    return text


def _validate_positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise NeuralControlHarnessError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_nonneg_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NeuralControlHarnessError(f"{name} must be a non-negative integer, got {value!r}")
    return int(value)


def _validate_finite_float(value: float, name: str, *, strictly_positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeuralControlHarnessError(f"{name} must be a real number, got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise NeuralControlHarnessError(f"{name} must be finite, got {value!r}")
    if strictly_positive and out <= 0.0:
        raise NeuralControlHarnessError(f"{name} must be > 0.0, got {value!r}")
    return out


def _extract_split_rows(
    split_data: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    split_name: str,
) -> list[Mapping[str, Any]]:
    """Normalize a split sequence or container mapping into a validated list of row mappings."""
    if isinstance(split_data, Mapping):
        raw_rows = split_data.get("chunks") if "chunks" in split_data else split_data.get("rows")
        if raw_rows is None:
            raise NeuralControlHarnessError(
                f"Split {split_name!r} mapping must contain 'chunks' or 'rows'."
            )
    elif isinstance(split_data, Sequence) and not isinstance(split_data, (str, bytes, bytearray)):
        raw_rows = split_data
    else:
        raise NeuralControlHarnessError(
            f"Split {split_name!r} must be a sequence of row mappings or a mapping with 'chunks'/'rows'."
        )

    rows = list(raw_rows)
    for idx, entry in enumerate(rows):
        if not isinstance(entry, Mapping):
            raise NeuralControlHarnessError(
                f"Split {split_name!r} entry #{idx} must be a mapping, got {type(entry).__name__}."
            )
    return rows


def _extract_row_key(entry: Mapping[str, Any], context_name: str, idx: int) -> tuple[str, int]:
    """Extract canonical ``(dataset_id, row)`` key, failing closed on missing or invalid fields."""
    if "dataset_id" not in entry or "row" not in entry:
        raise NeuralControlHarnessError(
            f"{context_name} entry #{idx} is missing required 'dataset_id' or 'row' provenance key."
        )
    raw_ds = entry["dataset_id"]
    if not isinstance(raw_ds, str) or not raw_ds.strip():
        raise NeuralControlHarnessError(
            f"{context_name} entry #{idx} has invalid 'dataset_id': {raw_ds!r}"
        )
    raw_row = entry["row"]
    if isinstance(raw_row, bool):
        raise NeuralControlHarnessError(
            f"{context_name} entry #{idx} has invalid boolean 'row': {raw_row!r}"
        )
    if isinstance(raw_row, int):
        row_idx = int(raw_row)
    elif isinstance(raw_row, str) and raw_row.strip().lstrip("+-").isdigit():
        row_idx = int(raw_row.strip())
    else:
        raise NeuralControlHarnessError(
            f"{context_name} entry #{idx} has non-integer 'row': {raw_row!r}"
        )
    if row_idx < 0:
        raise NeuralControlHarnessError(
            f"{context_name} entry #{idx} has negative 'row': {row_idx}"
        )
    return (raw_ds.strip(), row_idx)


def verify_pairwise_split_disjointness(
    probe: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    selection: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    benchmark: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    virgin_vault: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    require_non_empty: bool = True,
) -> dict[str, Any]:
    """Fail-closed pairwise disjointness verifier over ``(dataset_id, row)`` across all 4 splits."""
    named_inputs = (
        ("probe", probe),
        ("selection", selection),
        ("benchmark", benchmark),
        ("virgin_vault", virgin_vault),
    )
    key_sets: dict[str, set[tuple[str, int]]] = {}
    split_counts: dict[str, int] = {}
    split_fingerprints: dict[str, str] = {}

    for split_name, split_data in named_inputs:
        rows = _extract_split_rows(split_data, split_name)
        if require_non_empty and not rows:
            raise NeuralControlHarnessError(f"Split {split_name!r} must be non-empty.")

        seen: set[tuple[str, int]] = set()
        for idx, entry in enumerate(rows):
            key = _extract_row_key(entry, f"Split {split_name!r}", idx)
            if key in seen:
                raise NeuralControlHarnessError(
                    f"Split {split_name!r} contains duplicate (dataset_id, row) entry: {key!r}."
                )
            seen.add(key)

        for prev_name, prev_keys in key_sets.items():
            overlap = seen & prev_keys
            if overlap:
                sample = sorted(overlap)[:3]
                raise NeuralControlHarnessError(
                    f"Data leakage detected between {prev_name!r} and {split_name!r}: "
                    f"{len(overlap)} overlapping (dataset_id, row) entries {sample}"
                )

        key_sets[split_name] = seen
        split_counts[split_name] = len(seen)
        split_fingerprints[split_name] = compute_sha256_hex([[ds, r] for ds, r in sorted(seen)])

    return _attach_provenance_receipt(
        {
            "object": SPLITS_OBJECT,
            "schema": SCHEMA,
            "disjoint": True,
            "splits_verified": list(SPLIT_NAMES),
            "split_counts": split_counts,
            "total_unique_rows": sum(split_counts.values()),
            "split_fingerprints_sha256": split_fingerprints,
            "splits_manifest_sha256": compute_sha256_hex(split_fingerprints),
            "skip_policy": dict(DEFAULT_SPLIT_SKIP_POLICY),
        }
    )


def paired_bootstrap_ci95(
    diffs: np.ndarray | Sequence[float],
    *,
    n_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Compute paired bootstrap 95% CI ``[p2.5, p97.5]`` of mean NLL difference deterministically."""
    resamples = _validate_positive_int(n_resamples, "n_resamples")
    seed_i = _validate_nonneg_int(seed, "seed")
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0 or not bool(np.isfinite(arr).all()):
        raise NeuralControlHarnessError("diffs must be a non-empty finite 1D array.")

    rng = np.random.default_rng(seed_i)
    if arr.size <= 2000:
        idx = rng.integers(0, arr.size, size=(resamples, arr.size))
        means = arr[idx].mean(axis=1)
    else:
        means = np.empty(resamples, dtype=np.float64)
        batch = 500
        for s0 in range(0, resamples, batch):
            s1 = min(resamples, s0 + batch)
            idx = rng.integers(0, arr.size, size=(s1 - s0, arr.size))
            means[s0:s1] = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _verdict_from_ci(ci: tuple[float, float], n_total: int, min_total_chunks: int) -> str:
    if n_total < min_total_chunks:
        return "INSUFFICIENT_N"
    if ci[1] < 0.0:
        return "PASS"
    if ci[0] > 0.0:
        return "HARM"
    return "NO_GAIN"


def evaluate_neural_control_receipt(
    receipt: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    *,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ip300_seed: int = DEFAULT_IP300_BOOTSTRAP_SEED,
    min_chunks_per_expert: int = DEFAULT_MIN_CHUNKS_PER_EXPERT,
    min_total_chunks: int = DEFAULT_MIN_TOTAL_CHUNKS,
    domain_guard_max_ratio: float = DOMAIN_GUARD_MAX_RATIO,
    mostly_content_threshold_nats: float = MOSTLY_CONTENT_THRESHOLD_NATS,
    anse_safe_e0_ratio: float = ANSE_SAFE_E0_RATIO,
    expected_split_name: str | None = None,
    allowed_split_keys: set[tuple[str, int]] | None = None,
    forbidden_split_keys: Mapping[str, set[tuple[str, int]]] | None = None,
) -> dict[str, Any]:
    """Score a multi-domain neural-control receipt against the ANSE, domain-guard, and IP-300 rules."""
    resamples = _validate_positive_int(bootstrap_resamples, "bootstrap_resamples")
    seed_primary = _validate_nonneg_int(bootstrap_seed, "bootstrap_seed")
    seed_ip300 = _validate_nonneg_int(ip300_seed, "ip300_seed")
    min_per_exp = _validate_positive_int(min_chunks_per_expert, "min_chunks_per_expert")
    min_total = _validate_positive_int(min_total_chunks, "min_total_chunks")
    guard_ratio = _validate_finite_float(
        domain_guard_max_ratio, "domain_guard_max_ratio", strictly_positive=True
    )
    safe_e0_ratio = _validate_finite_float(
        anse_safe_e0_ratio, "anse_safe_e0_ratio", strictly_positive=True
    )
    mostly_thresh = _validate_finite_float(
        mostly_content_threshold_nats, "mostly_content_threshold_nats"
    )

    arm_name: str | None = None
    declared_split: str | None = None
    if isinstance(receipt, Mapping):
        if "chunks" not in receipt:
            raise NeuralControlHarnessError("Receipt mapping must contain 'chunks'.")
        raw_rows = receipt["chunks"]
        if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
            raise NeuralControlHarnessError("receipt['chunks'] must be a sequence of chunk mappings.")
        raw_name = receipt.get("name") or receipt.get("arm_name") or receipt.get("candidate")
        if raw_name is not None:
            arm_name = str(raw_name)
        if receipt.get("split") is not None:
            declared_split = str(receipt["split"])
    elif isinstance(receipt, Sequence) and not isinstance(receipt, (str, bytes, bytearray)):
        raw_rows = receipt
    else:
        raise NeuralControlHarnessError(
            "receipt must be a mapping with 'chunks' or a sequence of chunk mappings."
        )

    if expected_split_name is not None and declared_split is not None and declared_split != expected_split_name:
        raise NeuralControlHarnessError(
            f"Receipt declared split {declared_split!r}, expected {expected_split_name!r}."
        )

    rows = list(raw_rows)
    if not rows:
        raise NeuralControlHarnessError("receipt['chunks'] must be non-empty.")

    counts: dict[str, int] = {}
    order: list[str] = []
    row_eids: list[str] = []
    row_base: list[float] = []
    row_moe: list[float] = []
    seen_receipt_keys: set[tuple[str, int]] = set()
    canonical_chunk_records: list[list[Any]] = []

    for idx, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NeuralControlHarnessError(f"Chunk #{idx} must be a mapping.")
        raw_eid = row.get("expert_id")
        if not isinstance(raw_eid, str) or not raw_eid.strip():
            raise NeuralControlHarnessError(f"Chunk #{idx} has invalid or missing 'expert_id'.")
        eid = raw_eid.strip()
        if "base_nll" not in row or "moe_nll" not in row:
            raise NeuralControlHarnessError(f"Chunk #{idx} must contain 'base_nll' and 'moe_nll'.")
        b_nll = _validate_finite_float(row["base_nll"], f"chunks[{idx}].base_nll", strictly_positive=True)
        m_nll = _validate_finite_float(row["moe_nll"], f"chunks[{idx}].moe_nll")
        if m_nll < 0.0:
            raise NeuralControlHarnessError(f"chunks[{idx}].moe_nll must be >= 0.0, got {m_nll!r}")

        if "dataset_id" in row or "row" in row:
            key = _extract_row_key(row, "Receipt chunk", idx)
            if key in seen_receipt_keys:
                raise NeuralControlHarnessError(
                    f"Duplicate (dataset_id, row) detected inside receipt: {key!r}."
                )
            seen_receipt_keys.add(key)
            if forbidden_split_keys:
                for bad_split, bad_keys in forbidden_split_keys.items():
                    if key in bad_keys:
                        raise NeuralControlHarnessError(
                            f"Data leakage detected: receipt chunk {key!r} belongs to forbidden split {bad_split!r}."
                        )
            if allowed_split_keys is not None and key not in allowed_split_keys:
                raise NeuralControlHarnessError(
                    f"Split provenance mismatch: receipt chunk {key!r} is not in registered split "
                    f"{expected_split_name or '<target>'!r}."
                )
            canonical_chunk_records.append([eid, key[0], key[1], b_nll, m_nll])
        else:
            canonical_chunk_records.append([eid, b_nll, m_nll])

        if eid not in counts:
            order.append(eid)
            counts[eid] = 0
        counts[eid] += 1
        row_eids.append(eid)
        row_base.append(b_nll)
        row_moe.append(m_nll)

    qualifying = {eid for eid, cnt in counts.items() if cnt >= min_per_exp}
    excluded_lt_min = [eid for eid in order if eid not in qualifying]
    keep_ix = [i for i, eid in enumerate(row_eids) if eid in qualifying]

    per_expert: dict[str, dict[str, Any]] = {}
    for eid in order:
        ix = [i for i, row_eid in enumerate(row_eids) if row_eid == eid]
        b_m = float(np.mean([row_base[i] for i in ix]))
        m_m = float(np.mean([row_moe[i] for i in ix]))
        ratio_e = m_m / b_m
        pct_e = 100.0 * (m_m - b_m) / b_m
        per_expert[eid] = {
            "n": len(ix),
            "qualifies": eid in qualifying,
            "base_nll": b_m,
            "moe_nll": m_m,
            "mean_diff": m_m - b_m,
            "pct_vs_base": pct_e,
            "ratio": ratio_e,
            "guard_ok": bool(m_m <= guard_ratio * b_m),
        }

    per_expert_pct_vs_base = {k: round(v["pct_vs_base"], 3) for k, v in per_expert.items()}
    qual_entries = [(eid, info) for eid, info in per_expert.items() if info["qualifies"]]

    e0_entry = next(
        (v for k, v in per_expert.items() if k.startswith(GENERAL_EXPERT_PREFIX) and v["qualifies"]),
        next((v for k, v in per_expert.items() if k.startswith(GENERAL_EXPERT_PREFIX)), None),
    )
    e0_ratio = float(e0_entry["ratio"]) if e0_entry is not None else 1.0
    e0_pct = float(e0_entry["pct_vs_base"]) if e0_entry is not None else 0.0

    n_total = len(keep_ix)
    if n_total == 0:
        all_base = np.asarray(row_base, dtype=np.float64)
        all_moe = np.asarray(row_moe, dtype=np.float64)
        all_diffs = all_moe - all_base
        mean_base = float(all_base.mean())
        j_val = float(all_moe.mean())
        mean_diff = float(all_diffs.mean())
        ci_primary = (0.0, 0.0)
        ci_ip300 = (0.0, 0.0)
        guard_violating: list[str] = []
        guard_ok = False
        verdict_primary = "INSUFFICIENT_N"
        verdict_ip300 = "INSUFFICIENT_N"
        corrects = False
        worst_dom_pct = max(float(v["pct_vs_base"]) for v in per_expert.values())
        worst_dom_ratio = max(float(v["ratio"]) for v in per_expert.values())
        sign_test = {
            "moe_lower": int(np.sum(all_diffs < 0.0)),
            "moe_higher": int(np.sum(all_diffs > 0.0)),
            "ties": int(np.sum(all_diffs == 0.0)),
        }
    else:
        base_arr = np.asarray([row_base[i] for i in keep_ix], dtype=np.float64)
        moe_arr = np.asarray([row_moe[i] for i in keep_ix], dtype=np.float64)
        diffs = moe_arr - base_arr
        ci_primary = paired_bootstrap_ci95(diffs, n_resamples=resamples, seed=seed_primary)
        ci_ip300 = (
            ci_primary
            if seed_ip300 == seed_primary
            else paired_bootstrap_ci95(diffs, n_resamples=resamples, seed=seed_ip300)
        )

        guard_violating = [eid for eid, info in qual_entries if not info["guard_ok"]]
        guard_ok = len(guard_violating) == 0
        mean_base = float(base_arr.mean())
        j_val = float(moe_arr.mean())
        mean_diff = float(diffs.mean())
        verdict_primary = _verdict_from_ci(ci_primary, n_total, min_total)
        verdict_ip300 = _verdict_from_ci(ci_ip300, n_total, min_total)
        corrects = bool((n_total >= min_total) and guard_ok and (ci_primary[1] < 0.0))
        worst_dom_pct = max(float(info["pct_vs_base"]) for _, info in qual_entries)
        worst_dom_ratio = max(float(info["ratio"]) for _, info in qual_entries)
        sign_test = {
            "moe_lower": int(np.sum(diffs < 0.0)),
            "moe_higher": int(np.sum(diffs > 0.0)),
            "ties": int(np.sum(diffs == 0.0)),
        }

    mostly_content = bool(
        (verdict_ip300 == "PASS") and (verdict_primary == "PASS") and (mean_diff <= mostly_thresh)
    )
    mostly_content_survived = bool(guard_ok and corrects and mostly_content)
    midrow_label = (
        "MOSTLY_CONTENT"
        if (verdict_ip300 == "PASS" and mean_diff <= mostly_thresh)
        else ("MOSTLY_FORMAT_SOME_CONTENT" if verdict_ip300 == "PASS" else "FORMAT")
    )

    pain = ANSE_MAXIMUM_PAIN if not guard_ok else (ANSE_SOFT_MARGIN_PENALTY if e0_ratio > safe_e0_ratio else 0.0)
    anse_energy = float(mean_diff + pain)

    payload: dict[str, Any] = {
        "object": RECEIPT_OBJECT,
        "schema": SCHEMA,
        "n": n_total,
        "total_rows_input": len(rows),
        "J": j_val,
        "nll_mean": j_val,
        "mean_base_nll": mean_base,
        "mean_moe_nll": j_val,
        "mean_diff_vs_base": mean_diff,
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed_primary,
        "ip300_seed": seed_ip300,
        "ci95": [ci_primary[0], ci_primary[1]],
        f"ci95_seed{seed_primary}": [ci_primary[0], ci_primary[1]],
        f"ci95_seed{seed_ip300}": [ci_ip300[0], ci_ip300[1]],
        "guard_ok": guard_ok,
        "guard_max_ratio": guard_ratio,
        "guard_violating_experts": guard_violating,
        "corrects": corrects,
        "verdict_primary": verdict_primary,
        "verdict_ip300": verdict_ip300,
        "b3_supported": bool((verdict_ip300 == "PASS") and guard_ok),
        "midrow_label": midrow_label,
        "mostly_content": mostly_content,
        "mostly_content_survived": mostly_content_survived,
        "mostly_content_threshold_nats": mostly_thresh,
        "e0_ratio": e0_ratio,
        "e0_pct_vs_base": e0_pct,
        "worst_domain_ratio": worst_dom_ratio,
        "worst_domain_pct_vs_base": worst_dom_pct,
        "anse_energy": anse_energy,
        "excluded_lt_min": excluded_lt_min,
        "sign_test": sign_test,
        "per_expert": per_expert,
        "per_expert_pct_vs_base": per_expert_pct_vs_base,
        "chunks_sha256": compute_sha256_hex(canonical_chunk_records),
    }
    if arm_name is not None:
        payload["arm_name"] = arm_name
    if declared_split is not None:
        payload["split"] = declared_split
    return _attach_provenance_receipt(payload)


def _normalize_generations(
    generations: Sequence[Any] | Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Normalize generation definitions into an ordered ``[(gen_name, candidates_map), ...]`` list."""
    out: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(generations, Mapping):
        for gen_name, cands in generations.items():
            if not isinstance(gen_name, str) or not gen_name.strip():
                raise NeuralControlHarnessError(f"Generation name must be non-empty string, got {gen_name!r}")
            if not isinstance(cands, Mapping) or not cands:
                raise NeuralControlHarnessError(f"Generation {gen_name!r} must map to non-empty candidates.")
            out.append((gen_name.strip(), cands))
    elif isinstance(generations, Sequence) and not isinstance(generations, (str, bytes, bytearray)):
        for idx, item in enumerate(generations):
            if isinstance(item, tuple) and len(item) == 2:
                gen_name, cands = item
            elif isinstance(item, Mapping) and "generation" in item and "candidates" in item:
                gen_name, cands = item["generation"], item["candidates"]
            else:
                raise NeuralControlHarnessError(
                    f"Generation entry #{idx} must be (name, candidates) or {{'generation', 'candidates'}}."
                )
            if not isinstance(gen_name, str) or not gen_name.strip():
                raise NeuralControlHarnessError(f"Generation #{idx} name must be non-empty string.")
            if not isinstance(cands, Mapping) or not cands:
                raise NeuralControlHarnessError(f"Generation {gen_name!r} must contain >= 1 candidate.")
            out.append((gen_name.strip(), cands))
    else:
        raise NeuralControlHarnessError("generations must be a mapping or sequence of generations.")

    if not out:
        raise NeuralControlHarnessError("generations must be non-empty.")

    seen_gens: set[str] = set()
    seen_arms: set[str] = set()
    for gen_name, cands in out:
        if gen_name in seen_gens:
            raise NeuralControlHarnessError(f"Duplicate generation name {gen_name!r}.")
        seen_gens.add(gen_name)
        for arm_name in cands:
            if not isinstance(arm_name, str) or not arm_name.strip():
                raise NeuralControlHarnessError(f"Invalid arm name {arm_name!r} in generation {gen_name!r}.")
            if arm_name in seen_arms:
                raise NeuralControlHarnessError(f"Duplicate candidate arm name {arm_name!r} across generations.")
            seen_arms.add(arm_name)
    return out


def _resolve_candidate_receipt(
    evaluator: Callable[[str, Sequence[Mapping[str, Any]]], Mapping[str, Any]] | Mapping[str, Any],
    arm_name: str,
    split_rows: Sequence[Mapping[str, Any]],
    split_label: str,
) -> Mapping[str, Any]:
    """Resolve one candidate receipt from either a callable evaluator or precomputed mapping."""
    if callable(evaluator):
        receipt = evaluator(arm_name, tuple(split_rows))
    elif isinstance(evaluator, Mapping):
        if arm_name not in evaluator:
            raise NeuralControlHarnessError(
                f"Missing {split_label} receipt for selected champion {arm_name!r}."
            )
        receipt = evaluator[arm_name]
    else:
        raise NeuralControlHarnessError(
            f"{split_label}_evaluator must be a callable or a mapping of arm_name -> receipt."
        )
    if not isinstance(receipt, Mapping):
        raise NeuralControlHarnessError(
            f"{split_label} evaluator for {arm_name!r} returned non-mapping {type(receipt).__name__}."
        )
    return receipt


def run_anse_ratchet_selection(
    probe: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    selection: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    benchmark: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    virgin_vault: Sequence[Mapping[str, Any]] | Mapping[str, Any],
    generations: Sequence[Any] | Mapping[str, Any],
    *,
    benchmark_evaluator: Callable[[str, Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None = None,
    vault_evaluator: Callable[[str, Sequence[Mapping[str, Any]]], Mapping[str, Any]]
    | Mapping[str, Mapping[str, Any]]
    | None = None,
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ip300_seed: int = DEFAULT_IP300_BOOTSTRAP_SEED,
    min_chunks_per_expert: int = DEFAULT_MIN_CHUNKS_PER_EXPERT,
    min_total_chunks: int = DEFAULT_MIN_TOTAL_CHUNKS,
    domain_guard_max_ratio: float = DOMAIN_GUARD_MAX_RATIO,
    mostly_content_threshold_nats: float = MOSTLY_CONTENT_THRESHOLD_NATS,
    anse_safe_e0_ratio: float = ANSE_SAFE_E0_RATIO,
    defer_virgin_vault: bool = False,
    require_benchmark_gate_before_vault: bool = False,
    benchmark_baseline_diff_nats: float | None = None,
    preregistration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute the 4-split anti-leakage ANSE Ratchet progression and single-shot vault promotion."""
    split_verification = verify_pairwise_split_disjointness(probe, selection, benchmark, virgin_vault)
    normalized_gens = _normalize_generations(generations)

    if benchmark_baseline_diff_nats is not None:
        bench_base_diff = _validate_finite_float(benchmark_baseline_diff_nats, "benchmark_baseline_diff_nats")
    else:
        bench_base_diff = None

    split_rows_map = {
        name: _extract_split_rows(data, name)
        for name, data in (
            ("probe", probe),
            ("selection", selection),
            ("benchmark", benchmark),
            ("virgin_vault", virgin_vault),
        )
    }
    split_keys_map = {
        name: {_extract_row_key(entry, f"Split {name!r}", idx) for idx, entry in enumerate(rows)}
        for name, rows in split_rows_map.items()
    }

    eval_kwargs = {
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "ip300_seed": ip300_seed,
        "min_chunks_per_expert": min_chunks_per_expert,
        "min_total_chunks": min_total_chunks,
        "domain_guard_max_ratio": domain_guard_max_ratio,
        "mostly_content_threshold_nats": mostly_content_threshold_nats,
        "anse_safe_e0_ratio": anse_safe_e0_ratio,
    }

    champion_name: str | None = None
    champion_gen: str | None = None
    champion_eval: dict[str, Any] | None = None
    generation_history: list[dict[str, Any]] = []
    selection_results: dict[str, dict[str, Any]] = {}

    sel_forbidden = {
        "probe": split_keys_map["probe"],
        "benchmark": split_keys_map["benchmark"],
        "virgin_vault": split_keys_map["virgin_vault"],
    }

    for gen_name, cands in normalized_gens:
        gen_scored: dict[str, dict[str, Any]] = {}
        for arm_name, cand_source in cands.items():
            if callable(cand_source):
                cand_receipt = cand_source(tuple(split_rows_map["selection"]))
            elif isinstance(cand_source, Mapping):
                cand_receipt = cand_source
            else:
                raise NeuralControlHarnessError(
                    f"Candidate {arm_name!r} in generation {gen_name!r} must be a receipt mapping or callable."
                )

            scored = evaluate_neural_control_receipt(
                cand_receipt,
                **eval_kwargs,
                expected_split_name="selection",
                allowed_split_keys=split_keys_map["selection"],
                forbidden_split_keys=sel_forbidden,
            )
            scored["arm_name"] = arm_name
            scored["generation"] = gen_name
            scored = _attach_provenance_receipt(scored)
            gen_scored[arm_name] = scored
            selection_results[arm_name] = scored

        correcting = [
            (name, res)
            for name, res in gen_scored.items()
            if bool(res["corrects"]) and bool(res["guard_ok"])
        ]
        correcting.sort(key=lambda kv: (float(kv[1]["anse_energy"]), float(kv[1]["J"]), kv[0]))

        gen_winner = correcting[0][0] if correcting else None
        ratchet_improved = False
        delta_energy: float | None = None
        if gen_winner is not None:
            cand_res = gen_scored[gen_winner]
            if champion_eval is None:
                ratchet_improved = True
                champion_name = gen_winner
                champion_gen = gen_name
                champion_eval = cand_res
            else:
                cand_key = (float(cand_res["anse_energy"]), float(cand_res["J"]), gen_winner)
                champ_key = (
                    float(champion_eval["anse_energy"]),
                    float(champion_eval["J"]),
                    str(champion_name),
                )
                delta_energy = float(cand_res["anse_energy"]) - float(champion_eval["anse_energy"])
                if cand_key < champ_key:
                    ratchet_improved = True
                    champion_name = gen_winner
                    champion_gen = gen_name
                    champion_eval = cand_res

        generation_history.append(
            {
                "generation": gen_name,
                "candidates_evaluated": len(gen_scored),
                "candidates_correcting": len(correcting),
                "generation_winner": gen_winner,
                "generation_winner_anse_energy": (
                    float(gen_scored[gen_winner]["anse_energy"]) if gen_winner is not None else None
                ),
                "generation_winner_mean_diff": (
                    float(gen_scored[gen_winner]["mean_diff_vs_base"]) if gen_winner is not None else None
                ),
                "ratchet_improved": ratchet_improved,
                "delta_energy_vs_prev": delta_energy,
                "ratchet_champion_after_gen": champion_name,
                "ratchet_champion_generation": champion_gen,
                "ratchet_champion_anse_energy": (
                    float(champion_eval["anse_energy"]) if champion_eval is not None else None
                ),
            }
        )

    total_correcting = sum(
        1 for res in selection_results.values() if bool(res["corrects"]) and bool(res["guard_ok"])
    )
    benchmark_queries_burned = 0
    virgin_vault_queries_burned = 0
    benchmark_result: dict[str, Any] | None = None
    virgin_vault_result: dict[str, Any] | None = None

    if champion_name is None or champion_eval is None:
        stopping_gate_triggered = True
        vault_status = "ABORTED_SELECTION_STOPPING_GATE"
        promoted = False
        selection_decision: dict[str, Any] = {
            "status": "NO_CORRECTING_ARM",
            "winner": None,
            "winner_generation": None,
            "candidates_evaluated": len(selection_results),
            "candidates_correcting": 0,
        }
    else:
        stopping_gate_triggered = False
        selection_decision = {
            "status": "WINNER_SELECTED",
            "winner": champion_name,
            "winner_generation": champion_gen,
            "candidates_evaluated": len(selection_results),
            "candidates_correcting": total_correcting,
            "winner_anse_energy": champion_eval["anse_energy"],
            "winner_mean_diff": champion_eval["mean_diff_vs_base"],
            "winner_J": champion_eval["J"],
            "winner_mostly_content": champion_eval["mostly_content"],
        }

        if benchmark_evaluator is not None:
            bench_receipt = _resolve_candidate_receipt(
                benchmark_evaluator, champion_name, split_rows_map["benchmark"], "benchmark"
            )
            benchmark_queries_burned = 1
            benchmark_result = evaluate_neural_control_receipt(
                bench_receipt,
                **eval_kwargs,
                expected_split_name="benchmark",
                allowed_split_keys=split_keys_map["benchmark"],
                forbidden_split_keys={
                    "probe": split_keys_map["probe"],
                    "selection": split_keys_map["selection"],
                    "virgin_vault": split_keys_map["virgin_vault"],
                },
            )
            benchmark_result["arm_name"] = champion_name
            benchmark_result["generation"] = champion_gen
            if bench_base_diff is not None:
                benchmark_result["benchmark_baseline_diff_nats"] = bench_base_diff
                benchmark_result["beats_baseline"] = bool(
                    benchmark_result["guard_ok"]
                    and benchmark_result["corrects"]
                    and (float(benchmark_result["mean_diff_vs_base"]) < bench_base_diff)
                )
            benchmark_result = _attach_provenance_receipt(benchmark_result)

        benchmark_gate_blocked = bool(
            require_benchmark_gate_before_vault
            and (
                benchmark_result is None
                or not (bool(benchmark_result["corrects"]) and bool(benchmark_result["guard_ok"]))
            )
        )

        if defer_virgin_vault or vault_evaluator is None:
            vault_status = "DEFERRED"
            promoted = False
        elif benchmark_gate_blocked:
            vault_status = "ABORTED_BENCHMARK_GATE"
            promoted = False
        else:
            vault_receipt = _resolve_candidate_receipt(
                vault_evaluator, champion_name, split_rows_map["virgin_vault"], "virgin_vault"
            )
            virgin_vault_queries_burned = 1
            virgin_vault_result = evaluate_neural_control_receipt(
                vault_receipt,
                **eval_kwargs,
                expected_split_name="virgin_vault",
                allowed_split_keys=split_keys_map["virgin_vault"],
                forbidden_split_keys={
                    "probe": split_keys_map["probe"],
                    "selection": split_keys_map["selection"],
                    "benchmark": split_keys_map["benchmark"],
                },
            )
            virgin_vault_result["arm_name"] = champion_name
            virgin_vault_result["generation"] = champion_gen
            virgin_vault_result = _attach_provenance_receipt(virgin_vault_result)
            promoted = bool(virgin_vault_result["corrects"] and virgin_vault_result["guard_ok"])
            vault_status = "PROMOTED" if promoted else "REJECTED_ON_VAULT"

    prereg_dict = dict(preregistration) if preregistration is not None else {}
    report: dict[str, Any] = {
        "object": RATCHET_OBJECT,
        "schema": SCHEMA,
        "preregistration": prereg_dict,
        "preregistration_sha256": compute_sha256_hex(prereg_dict),
        "split_verification": split_verification,
        "generation_history": generation_history,
        "selection_results": selection_results,
        "selection_decision": selection_decision,
        "stopping_gate_triggered": stopping_gate_triggered,
        "benchmark_queries_burned": benchmark_queries_burned,
        "virgin_vault_queries_burned": virgin_vault_queries_burned,
        "single_shot_vault_invariant_ok": bool(
            virgin_vault_queries_burned <= 1
            and (not stopping_gate_triggered or virgin_vault_queries_burned == 0)
        ),
        "benchmark_result": benchmark_result,
        "virgin_vault_result": virgin_vault_result,
        "vault_status": vault_status,
        "promoted": promoted,
        "champion_mostly_content_survived": bool(
            virgin_vault_result is not None and virgin_vault_result["mostly_content_survived"]
        ),
    }
    return _attach_provenance_receipt(report)


def _format_receipt_markdown_section(title: str, eval_doc: Mapping[str, Any]) -> list[str]:
    ci = eval_doc.get("ci95", [0.0, 0.0])
    lines = [
        f"## {title}",
        "",
        "| Metric | Value |",
        "| :--- | :--- |",
        f"| `arm_name` | `{eval_doc.get('arm_name', 'unknown')}` |",
        f"| `n` | `{eval_doc.get('n')}` |",
        f"| `J` (`nll_mean`) | `{eval_doc.get('J'):.4f}` (base `{eval_doc.get('mean_base_nll'):.4f}`) |",
        f"| `mean_diff_vs_base` | `{eval_doc.get('mean_diff_vs_base'):+.4f}` nats |",
        f"| `ci95` (seed `{eval_doc.get('bootstrap_seed')}`, `{eval_doc.get('bootstrap_resamples')}` resamples) | `[{ci[0]:+.4f}, {ci[1]:+.4f}]` |",
        f"| `guard_ok` (+1.0% cap) | `{eval_doc.get('guard_ok')}` (E0 `{eval_doc.get('e0_pct_vs_base'):+.3f}%`, worst `{eval_doc.get('worst_domain_pct_vs_base'):+.3f}%`) |",
        f"| `midrow_label` | `{eval_doc.get('midrow_label')}` (`mostly_content_survived={eval_doc.get('mostly_content_survived')}`) |",
        f"| `anse_energy` | `{eval_doc.get('anse_energy'):.6f}` |",
        f"| `corrects` / `verdict_ip300` | `{eval_doc.get('corrects')}` / `{eval_doc.get('verdict_ip300')}` |",
        "",
        "| Domain | n | Qualifies | Base NLL | MoE NLL | Diff (nats) | Drift (%) | Guard |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    per_exp = eval_doc.get("per_expert", {})
    for eid, info in per_exp.items():
        lines.append(
            f"| `{eid}` | `{info['n']}` | `{info['qualifies']}` | `{info['base_nll']:.4f}` | "
            f"`{info['moe_nll']:.4f}` | `{info['mean_diff']:+.4f}` | `{info['pct_vs_base']:+.3f}%` | `{info['guard_ok']}` |"
        )
    return lines


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render deterministic Markdown provenance report for a ratchet, receipt, or split check."""
    obj_type = report.get("object")
    if obj_type == SPLITS_OBJECT:
        counts = report.get("split_counts", {})
        lines = [
            "# HiveMind 4-Split Anti-Leakage Verification Report",
            "",
            f"- **Disjoint**: `{report.get('disjoint')}`",
            f"- **Total Unique Rows**: `{report.get('total_unique_rows')}`",
            f"- **Splits Manifest SHA-256**: `{report.get('splits_manifest_sha256')}`",
            "",
            "| Split | Rows | Fingerprint SHA-256 |",
            "| :--- | :---: | :--- |",
        ]
        fps = report.get("split_fingerprints_sha256", {})
        for name in SPLIT_NAMES:
            lines.append(f"| `{name}` | `{counts.get(name, 0)}` | `{fps.get(name, '')[:16]}...` |")
        lines.extend(["", f"**SHA-256 Provenance Receipt**: `{report.get('receipt_sha256')}`"])
        return assert_safe_output("\n".join(lines))

    if obj_type == RECEIPT_OBJECT:
        lines = ["# HiveMind Neural-Control Receipt Evaluation", ""]
        lines.extend(_format_receipt_markdown_section("Receipt Metrics & Per-Domain Guard", report))
        lines.extend(["", f"**SHA-256 Provenance Receipt**: `{report.get('receipt_sha256')}`"])
        return assert_safe_output("\n".join(lines))

    if obj_type != RATCHET_OBJECT:
        raise NeuralControlHarnessError(f"Unsupported report object type: {obj_type!r}")

    split_ver = report.get("split_verification", {})
    counts = split_ver.get("split_counts", {})
    sel_dec = report.get("selection_decision", {})
    lines = [
        "# HiveMind ANSE Neural-Control 4-Split Ratchet Report (`NEURAL-CTRL-05`)",
        "",
        f"- **Vault Status**: `{report.get('vault_status')}` (`promoted={report.get('promoted')}`)",
        f"- **Selection Decision**: `{sel_dec.get('status')}` (winner=`{sel_dec.get('winner')}`, generation=`{sel_dec.get('winner_generation')}`)",
        f"- **Stopping Gate Triggered**: `{report.get('stopping_gate_triggered')}`",
        f"- **Benchmark / Virgin-Vault Queries Burned**: `{report.get('benchmark_queries_burned')}` / `{report.get('virgin_vault_queries_burned')}`",
        f"- **4-Split Disjoint Counts**: `probe={counts.get('probe', 0)}`, `selection={counts.get('selection', 0)}`, `benchmark={counts.get('benchmark', 0)}`, `virgin_vault={counts.get('virgin_vault', 0)}` (`manifest={split_ver.get('splits_manifest_sha256', '')[:16]}...`)",
        "",
        "## Generation Ratchet Progression",
        "",
        "| Generation | Evaluated | Correcting | Gen Winner | Ratchet Improved | Champion | ANSE Energy | Mean Diff (nats) |",
        "| :--- | :---: | :---: | :--- | :---: | :--- | :---: | :---: |",
    ]
    for step in report.get("generation_history", []):
        e_val = step.get("ratchet_champion_anse_energy")
        e_str = f"{e_val:.6f}" if isinstance(e_val, (int, float)) else "N/A"
        gw_diff = step.get("generation_winner_mean_diff")
        d_str = f"{gw_diff:+.4f}" if isinstance(gw_diff, (int, float)) else "N/A"
        lines.append(
            f"| `{step.get('generation')}` | `{step.get('candidates_evaluated')}` | `{step.get('candidates_correcting')}` | "
            f"`{step.get('generation_winner')}` | `{step.get('ratchet_improved')}` | "
            f"`{step.get('ratchet_champion_after_gen')}` | `{e_str}` | `{d_str}` |"
        )

    bench_res = report.get("benchmark_result")
    if isinstance(bench_res, Mapping):
        lines.append("")
        lines.extend(_format_receipt_markdown_section("Out-of-Sample Benchmark Confirmation (`benchmark`)", bench_res))

    vault_res = report.get("virgin_vault_result")
    if isinstance(vault_res, Mapping):
        lines.append("")
        lines.extend(_format_receipt_markdown_section("Single-Shot Virgin Vault Promotion (`virgin_vault`)", vault_res))

    lines.extend(["", f"**SHA-256 Provenance Receipt**: `{report.get('receipt_sha256')}`"])
    return assert_safe_output("\n".join(lines))


def _load_json_file(path: Path, label: str) -> Any:
    if not path.is_file():
        raise NeuralControlHarnessError(f"{label} file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NeuralControlHarnessError(f"Failed to parse {label} JSON ({path}): {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for ``scripts/hivemind_neural_control_harness.py``."""
    ap = argparse.ArgumentParser(
        description=(
            "Pre-registered 4-split anti-leakage ANSE neural-control benchmark harness and receipt verifier "
            "(CPU-only deterministic execution, zero outbound network calls)."
        )
    )
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle", type=Path, help="Path to 4-split ANSE ratchet JSON bundle.")
    mode.add_argument("--receipt", type=Path, help="Path to a single neural-control receipt JSON.")
    mode.add_argument("--verify-receipt", type=Path, help="Verify deterministic SHA-256 receipt provenance.")
    mode.add_argument(
        "--splits-only",
        type=Path,
        help="Verify 4-split pairwise (dataset_id, row) disjointness on a splits JSON file.",
    )
    ap.add_argument(
        "--splits",
        type=Path,
        default=None,
        help="Optional 4-split manifest JSON used with --receipt to assert no probe leakage.",
    )
    ap.add_argument("--bootstrap-resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    ap.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    ap.add_argument("--ip300-seed", type=int, default=DEFAULT_IP300_BOOTSTRAP_SEED)
    ap.add_argument("--min-chunks-per-expert", type=int, default=DEFAULT_MIN_CHUNKS_PER_EXPERT)
    ap.add_argument("--min-total-chunks", type=int, default=DEFAULT_MIN_TOTAL_CHUNKS)
    ap.add_argument("--defer-virgin-vault", action="store_true")
    ap.add_argument("--require-benchmark-gate", action="store_true")
    ap.add_argument("--benchmark-baseline-diff", type=float, default=None)
    ap.add_argument("--strict", action="store_true", help="Exit 2 unless promoted / guard+corrects pass.")
    ap.add_argument("--out-json", type=Path, default=None, help="Optional file path to write JSON receipt.")
    ap.add_argument("--out-markdown", type=Path, default=None, help="Optional file path to write Markdown report.")
    ap.add_argument("--json", action="store_true", help="Print JSON provenance receipt to stdout.")
    ap.add_argument("--markdown", action="store_true", help="Print Markdown report to stdout (default).")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning ``0`` on clean verification (or ``2`` on fail-closed error)."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    args = build_parser().parse_args(argv)

    try:
        if args.verify_receipt is not None:
            doc = _load_json_file(args.verify_receipt, "receipt")
            ok = verify_receipt_provenance(doc)
            result: dict[str, Any] = {
                "object": "hivemind.provenance_check",
                "schema": SCHEMA,
                "verified": ok,
                "claimed_sha256": doc.get("receipt_sha256"),
                "target_object": doc.get("object"),
            }
            out_str = assert_safe_output(json.dumps(result, indent=2, sort_keys=True))
            if args.out_json is not None:
                args.out_json.parent.mkdir(parents=True, exist_ok=True)
                args.out_json.write_text(out_str + "\n", encoding="utf-8")
            if args.json:
                print(out_str)
            else:
                print(f"provenance_verified={ok} sha256={doc.get('receipt_sha256')}")
            return 0 if ok else 2

        if args.splits_only is not None:
            splits_doc = _load_json_file(args.splits_only, "splits")
            if not isinstance(splits_doc, Mapping) or any(k not in splits_doc for k in SPLIT_NAMES):
                raise NeuralControlHarnessError(
                    f"Splits file must be a JSON mapping containing all of {SPLIT_NAMES}."
                )
            report = verify_pairwise_split_disjointness(
                splits_doc["probe"],
                splits_doc["selection"],
                splits_doc["benchmark"],
                splits_doc["virgin_vault"],
            )
            strict_ok = bool(report["disjoint"])
        elif args.receipt is not None:
            rec_doc = _load_json_file(args.receipt, "receipt")
            forbidden_keys: dict[str, set[tuple[str, int]]] | None = None
            if args.splits is not None:
                splits_doc = _load_json_file(args.splits, "splits")
                if not isinstance(splits_doc, Mapping) or any(k not in splits_doc for k in SPLIT_NAMES):
                    raise NeuralControlHarnessError(
                        f"Splits file must contain all of {SPLIT_NAMES}."
                    )
                verify_pairwise_split_disjointness(
                    splits_doc["probe"],
                    splits_doc["selection"],
                    splits_doc["benchmark"],
                    splits_doc["virgin_vault"],
                )
                probe_rows = _extract_split_rows(splits_doc["probe"], "probe")
                forbidden_keys = {
                    "probe": {_extract_row_key(r, "Split 'probe'", i) for i, r in enumerate(probe_rows)}
                }
            report = evaluate_neural_control_receipt(
                rec_doc,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
                ip300_seed=args.ip300_seed,
                min_chunks_per_expert=args.min_chunks_per_expert,
                min_total_chunks=args.min_total_chunks,
                forbidden_split_keys=forbidden_keys,
            )
            strict_ok = bool(report["corrects"] and report["guard_ok"])
        else:
            bundle = _load_json_file(args.bundle, "bundle")
            if not isinstance(bundle, Mapping):
                raise NeuralControlHarnessError("Bundle JSON must be an object.")
            splits_doc = bundle.get("splits")
            if not isinstance(splits_doc, Mapping) or any(k not in splits_doc for k in SPLIT_NAMES):
                raise NeuralControlHarnessError(
                    f"Bundle 'splits' must be an object containing all of {SPLIT_NAMES}."
                )
            if "generations" not in bundle:
                raise NeuralControlHarnessError("Bundle JSON must contain 'generations'.")

            report = run_anse_ratchet_selection(
                splits_doc["probe"],
                splits_doc["selection"],
                splits_doc["benchmark"],
                splits_doc["virgin_vault"],
                bundle["generations"],
                benchmark_evaluator=bundle.get("benchmark_receipts"),
                vault_evaluator=bundle.get("virgin_vault_receipts"),
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed,
                ip300_seed=args.ip300_seed,
                min_chunks_per_expert=args.min_chunks_per_expert,
                min_total_chunks=args.min_total_chunks,
                defer_virgin_vault=bool(args.defer_virgin_vault),
                require_benchmark_gate_before_vault=bool(args.require_benchmark_gate),
                benchmark_baseline_diff_nats=args.benchmark_baseline_diff,
                preregistration=bundle.get("preregistration"),
            )
            strict_ok = bool(report["promoted"])

        json_text = assert_safe_output(json.dumps(report, indent=2, sort_keys=True))
        md_text = render_markdown_report(report)

        if args.out_json is not None:
            args.out_json.parent.mkdir(parents=True, exist_ok=True)
            args.out_json.write_text(json_text + "\n", encoding="utf-8")
        if args.out_markdown is not None:
            args.out_markdown.parent.mkdir(parents=True, exist_ok=True)
            args.out_markdown.write_text(md_text + "\n", encoding="utf-8")

        if args.json:
            print(json_text)
        else:
            print(md_text)

        if args.strict and not strict_ok:
            return 2
        return 0
    except NeuralControlHarnessError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
