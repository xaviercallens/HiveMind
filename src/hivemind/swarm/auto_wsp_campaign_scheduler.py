# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Autonomous Open-Weight WSP Campaign Ledger, Disjoint Split Allocator, and Drift Scheduler.

Implements neural-control slice ``NEURAL-CTRL-DYN-25`` for multi-generation
nocturnal Weight-Space Programming (WSP) campaigns without holdout contamination
or GPU VRAM contention:

1. **Disjoint Split Allocation & Permanent Virgin-Vault Sealing**
   (:class:`DisjointSplitAllocator`, :class:`SplitQuadruple`):
   Deterministically reserves pairwise non-overlapping
   ``(probe_skip, selection_skip, confirm_skip, virgin_vault_skip)`` shard
   quadruples across generations ``G_1..G_k`` above the protected frozen base
   window ``[0, frozen_skip_floor)`` and permanently seals consumed virgin
   vault skip offsets so no future generation can re-allocate or re-evaluate a
   spent holdout vault.
2. **Deterministic Offline CPU S5 Recalibration Trigger**
   (:func:`evaluate_drift_recalibration_trigger`, :class:`DriftTelemetryWindow`,
   :class:`DriftTriggerDecision`):
   Evaluates live general-domain relative NLL drift (``drift_ratio``) and rogue
   expert LoRA/delta norm ratios against configurable thresholds, gating
   execution on overnight scheduling windows, CPU-only isolation, and minimum
   available host RAM (default ``11.0 GiB``).
3. **Hash-Chained ANSE Monotone Campaign Ledger**
   (:class:`AutoWSPCampaignLedger`, :class:`WSPGenerationReceipt`,
   :func:`verify_ledger_chain`):
   Records canonical SHA-256 hash-chained generation receipts
   (``prev_receipt_sha256 -> receipt_sha256``), enforces strict cross-generation
   ANSE energy monotone improvement under the ``+1.0%`` domain guard, and
   detects any receipt, link, split, or monotonicity tampering fail-closed.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import torch

SCHEMA_VERSION: str = "hivemind.auto_wsp_campaign_scheduler/1"
SPLIT_ALLOCATION_OBJECT: str = "hivemind.wsp_split_allocation_receipt"
VAULT_SEAL_OBJECT: str = "hivemind.wsp_virgin_vault_seal_receipt"
DRIFT_TRIGGER_OBJECT: str = "hivemind.wsp_drift_recalibration_decision"
GENERATION_RECEIPT_OBJECT: str = "hivemind.wsp_campaign_generation_receipt"
LEDGER_SNAPSHOT_OBJECT: str = "hivemind.wsp_campaign_ledger_snapshot"

GENESIS_RECEIPT_SHA256: str = "0" * 64
SPLIT_ROLES: tuple[str, str, str, str] = ("probe", "selection", "confirm", "virgin_vault")
RECOMMENDED_S5_STAGES: tuple[str, ...] = (
    "s5_1_anchor_ghost_sink",
    "s5_2_oof_orthogonal_multigate",
    "s5_3_tail_qr_svd_nullspace",
    "s5_4_folded_norm_temperature",
)

DEFAULT_START_SKIP: int = 40
DEFAULT_SPLIT_STRIDE: int = 40
DEFAULT_FROZEN_SKIP_FLOOR: int = 40
DEFAULT_GENERAL_DRIFT_TRIGGER_REL: float = 0.008
DEFAULT_ROGUE_EXPERT_NORM_RATIO: float = 2.5
DEFAULT_MIN_FREE_RAM_GIB: float = 11.0
DEFAULT_GENERAL_GUARD_REL: float = 0.01
DEFAULT_ANSE_SAFE_MARGIN_FRAC: float = 0.85
DEFAULT_ANSE_SOFT_MARGIN_PENALTY: float = 0.05
ANSE_MAXIMUM_PAIN: float = 1_000_000.0
GENERAL_EXPERT_PREFIX: str = "E0_"

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class SplitContaminationError(ValueError):
    """Raised when a split allocation overlaps an existing or frozen shard range."""

    def __init__(self, message: str, *, reason: str = "split_contamination") -> None:
        super().__init__(message)
        self.reason = reason


class VaultSealedError(SplitContaminationError):
    """Raised when attempting to re-allocate or re-evaluate a permanently sealed virgin vault."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="virgin_vault_sealed")


class LedgerChainError(ValueError):
    """Raised when campaign ledger hash-chaining, split binding, or provenance verification fails."""

    def __init__(self, message: str, *, reason: str = "ledger_chain_error") -> None:
        super().__init__(message)
        self.reason = reason


class AnseMonotonicityViolationError(LedgerChainError):
    """Raised when a generation fails the ANSE guard or fails to strictly lower ANSE energy."""

    def __init__(self, message: str) -> None:
        super().__init__(message, reason="anse_monotonicity_violation")


# ---------------------------------------------------------------------------
# 1. Validation & Canonical Provenance Helpers
# ---------------------------------------------------------------------------


def _validate_bool(value: Any, name: str) -> bool:
    """Validate that ``value`` is strictly a boolean."""
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
    return value


def _validate_nonempty_str(value: Any, name: str) -> str:
    """Validate that ``value`` is a printable non-empty string without control characters."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    stripped = value.strip()
    if not stripped or any(ord(ch) < 32 or ord(ch) == 127 for ch in stripped):
        raise ValueError(f"{name} must be a non-empty printable string")
    return stripped


def _validate_nonneg_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a non-negative integer (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _validate_positive_int(value: Any, name: str) -> int:
    """Validate that ``value`` is a strictly positive integer (excluding booleans)."""
    out = _validate_nonneg_int(value, name)
    if out < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return out


def _validate_finite_float(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite real number (excluding booleans)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_nonneg_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite non-negative real number."""
    out = _validate_finite_float(value, name)
    if out < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {value!r}")
    return out


def _validate_positive_finite(value: Any, name: str) -> float:
    """Validate that ``value`` is a finite strictly positive real number."""
    out = _validate_finite_float(value, name)
    if out <= 0.0:
        raise ValueError(f"{name} must be > 0.0, got {value!r}")
    return out


def _validate_sha256_hex(value: Any, name: str) -> str:
    """Validate a lowercase 64-character SHA-256 hex digest."""
    if not isinstance(value, str) or not _SHA256_HEX_RE.match(value):
        raise ValueError(f"{name} must be a 64-char lowercase hex SHA-256 digest, got {value!r}")
    return value


def _json_normalize(value: Any) -> Any:
    """Recursively normalize read-only containers into canonical JSON-serializable values."""
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Canonical JSON payload cannot contain non-finite float {value!r}")
        return value
    if isinstance(value, (Mapping, MappingProxyType)):
        return {str(k): _json_normalize(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(v) for v in value]
    raise TypeError(f"Unsupported value type in canonical JSON payload: {type(value).__name__}")


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize ``payload`` deterministically with strict finite-number enforcement."""
    normalized = _json_normalize(payload)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def compute_payload_sha256(payload: Any) -> str:
    """Return lowercase SHA-256 hex digest of ``canonical_json_bytes(payload)``."""
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def compute_state_sha256(state_dict: Mapping[str, torch.Tensor]) -> str:
    """Compute a deterministic CPU bitwise SHA-256 digest over a PyTorch tensor mapping."""
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"state_dict must be a mapping of str -> torch.Tensor, got {type(state_dict).__name__}")
    if not state_dict:
        raise ValueError("state_dict must be a non-empty mapping of str -> torch.Tensor")
    digest = hashlib.sha256()
    for key in sorted(state_dict.keys()):
        k_str = _validate_nonempty_str(key, "state_dict_key")
        val = state_dict[key]
        if not isinstance(val, torch.Tensor):
            raise TypeError(f"state_dict[{key!r}] must be a torch.Tensor, got {type(val).__name__}")
        cpu_t = val.detach().to(device="cpu").contiguous()
        if cpu_t.is_floating_point() and not bool(torch.isfinite(cpu_t).all().item()):
            raise ValueError(f"state_dict[{key!r}] contains non-finite values")
        digest.update(f"{k_str}:{cpu_t.dtype}:{tuple(cpu_t.shape)}:".encode())
        if cpu_t.numel() > 0:
            elem_bytes = cpu_t.element_size()
            view_dt = (
                torch.uint8
                if elem_bytes == 1
                else (torch.int16 if elem_bytes == 2 else (torch.int32 if elem_bytes == 4 else torch.int64))
            )
            digest.update(cpu_t.view(view_dt).reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# 2. Quantitative Telemetry, Norm-Ratio & ANSE Energy Helpers
# ---------------------------------------------------------------------------


def compute_expert_norm_ratio(
    expert_weights_or_norms: Mapping[str | int, torch.Tensor | float] | Sequence[torch.Tensor | float],
) -> tuple[float, dict[str, float]]:
    """Compute ``max_norm / median_active_norm`` across experts on CPU ``float64``.

    Intentionally excludes zeroed ghost-sink experts (norm ``<= 1e-12``) from the
    baseline median so intentional FoMoE sinking does not fabricate rogue-norm
    false alarms.
    """
    items: list[tuple[str, Any]]
    if isinstance(expert_weights_or_norms, Mapping):
        if len(expert_weights_or_norms) < 2:
            raise ValueError("expert_weights_or_norms must contain at least 2 experts")
        items = [(str(k), expert_weights_or_norms[k]) for k in sorted(expert_weights_or_norms.keys(), key=str)]
    elif isinstance(expert_weights_or_norms, Sequence) and not isinstance(
        expert_weights_or_norms, (str, bytes, bytearray)
    ):
        if len(expert_weights_or_norms) < 2:
            raise ValueError("expert_weights_or_norms must contain at least 2 experts")
        items = [(f"expert_{idx}", val) for idx, val in enumerate(expert_weights_or_norms)]
    else:
        raise TypeError("expert_weights_or_norms must be a mapping or sequence of norms/tensors")

    norms: dict[str, float] = {}
    for name, raw in items:
        key = _validate_nonempty_str(name, "expert_id")
        if isinstance(raw, torch.Tensor):
            if not raw.is_floating_point() or raw.numel() == 0:
                raise ValueError(f"Expert tensor {key!r} must be a non-empty floating-point tensor")
            t64 = raw.detach().to(device="cpu", dtype=torch.float64)
            if not bool(torch.isfinite(t64).all().item()):
                raise ValueError(f"Expert tensor {key!r} contains non-finite values")
            norms[key] = float(torch.linalg.vector_norm(t64).item())
        else:
            norms[key] = _validate_nonneg_finite(raw, f"expert_norms[{key!r}]")

    values = list(norms.values())
    max_norm = max(values)
    active = sorted(v for v in values if v > 1e-12)
    if not active:
        return 0.0, norms
    mid = len(active) // 2
    if len(active) % 2 == 1:
        median_active = active[mid]
    else:
        median_active = 0.5 * (active[mid - 1] + active[mid])
    return float(max_norm / median_active), norms


def compute_wsp_anse_energy(
    *,
    nll_mean: float,
    base_nll_mean: float,
    drift_ratio: float,
    worst_domain_drift_ratio: float | None = None,
    general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
    safe_margin_frac: float = DEFAULT_ANSE_SAFE_MARGIN_FRAC,
    soft_margin_penalty: float = DEFAULT_ANSE_SOFT_MARGIN_PENALTY,
    maximum_pain: float = ANSE_MAXIMUM_PAIN,
) -> tuple[float, bool, float]:
    """Compute ``(anse_energy, guard_ok, mean_diff_vs_base)`` for a WSP candidate.

    Enforces the ``+1.0%`` general and worst-domain drift ceiling, applying
    ``ANSE_MAXIMUM_PAIN`` (``10^6``) upon guard breach and a smooth boundary
    margin penalty above ``safe_margin_frac * general_guard_rel`` (default
    ``+0.85%``).
    """
    nll_f = _validate_positive_finite(nll_mean, "nll_mean")
    base_f = _validate_positive_finite(base_nll_mean, "base_nll_mean")
    drift_f = _validate_finite_float(drift_ratio, "drift_ratio")
    worst_f = (
        _validate_finite_float(worst_domain_drift_ratio, "worst_domain_drift_ratio")
        if worst_domain_drift_ratio is not None
        else drift_f
    )
    guard_f = _validate_positive_finite(general_guard_rel, "general_guard_rel")
    safe_f = _validate_nonneg_finite(safe_margin_frac, "safe_margin_frac")
    soft_f = _validate_nonneg_finite(soft_margin_penalty, "soft_margin_penalty")
    pain_f = _validate_positive_finite(maximum_pain, "maximum_pain")

    mean_diff = nll_f - base_f
    effective_worst = max(drift_f, worst_f)
    guard_ok = (drift_f <= guard_f + 1e-12) and (effective_worst <= guard_f + 1e-12)
    if not guard_ok:
        return pain_f, False, mean_diff

    safe_threshold = safe_f * guard_f
    penalty = soft_f if drift_f > safe_threshold + 1e-12 else 0.0
    return float(mean_diff + penalty), True, float(mean_diff)


def verify_shard_rows_disjoint(
    named_splits: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    """Certify pairwise zero overlap of ``(dataset_id, row)`` keys across all named splits."""
    if isinstance(named_splits, (str, bytes)) or not isinstance(named_splits, Sequence) or not named_splits:
        raise ValueError("named_splits must be a non-empty sequence of (split_name, rows) pairs")

    seen_per_split: dict[str, set[tuple[str, int]]] = {}
    counts: dict[str, int] = {}
    fingerprints: dict[str, str] = {}

    for item in named_splits:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("Each entry in named_splits must be a (split_name, rows) tuple")
        raw_name, rows = item
        split_name = _validate_nonempty_str(raw_name, "split_name")
        if split_name in seen_per_split:
            raise SplitContaminationError(f"Duplicate split name {split_name!r}")
        if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence) or not rows:
            raise ValueError(f"Split {split_name!r} rows must be a non-empty sequence")

        keys: set[tuple[str, int]] = set()
        for idx, row_entry in enumerate(rows):
            if not isinstance(row_entry, Mapping):
                raise TypeError(f"Split {split_name!r} entry #{idx} must be a mapping")
            ds_id = _validate_nonempty_str(row_entry.get("dataset_id"), f"{split_name}[{idx}].dataset_id")
            row_idx = _validate_nonneg_int(row_entry.get("row"), f"{split_name}[{idx}].row")
            key = (ds_id, row_idx)
            if key in keys:
                raise SplitContaminationError(
                    f"Split {split_name!r} contains duplicate (dataset_id, row) entry {key!r}"
                )
            keys.add(key)

        for prev_name, prev_keys in seen_per_split.items():
            overlap = keys & prev_keys
            if overlap:
                sample = sorted(overlap)[:3]
                raise SplitContaminationError(
                    f"Shard contamination detected between {prev_name!r} and {split_name!r}: "
                    f"{len(overlap)} overlapping rows {sample}"
                )
        seen_per_split[split_name] = keys
        counts[split_name] = len(keys)
        fingerprints[split_name] = compute_payload_sha256([[ds, r] for ds, r in sorted(keys)])

    manifest_sha = compute_payload_sha256(fingerprints)
    return {
        "disjoint": True,
        "split_count": len(counts),
        "total_unique_rows": sum(counts.values()),
        "split_counts": counts,
        "split_fingerprints_sha256": fingerprints,
        "splits_manifest_sha256": manifest_sha,
    }


# ---------------------------------------------------------------------------
# 3. DisjointSplitAllocator & Permanent Virgin-Vault Sealing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitQuadruple:
    """Non-overlapping ``(probe, selection, confirm, virgin_vault)`` skip allocation for one generation."""

    generation_id: str
    generation_index: int
    probe_skip: int
    selection_skip: int
    confirm_skip: int
    virgin_vault_skip: int
    split_stride: int
    vault_id: str
    allocation_sha256: str

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return ``(probe_skip, selection_skip, confirm_skip, virgin_vault_skip)``."""
        return (self.probe_skip, self.selection_skip, self.confirm_skip, self.virgin_vault_skip)

    def skip_offsets(self) -> dict[str, int]:
        """Return mapping from split role name to integer skip offset."""
        return {
            "probe": self.probe_skip,
            "selection": self.selection_skip,
            "confirm": self.confirm_skip,
            "virgin_vault": self.virgin_vault_skip,
        }

    def intervals(self) -> dict[str, tuple[int, int]]:
        """Return half-open shard intervals ``[skip, skip + split_stride)`` per role."""
        return {role: (offset, offset + self.split_stride) for role, offset in self.skip_offsets().items()}

    def to_dict(self) -> dict[str, Any]:
        """Serialize allocation quadruple to a canonical JSON-safe dict."""
        return {
            "object": SPLIT_ALLOCATION_OBJECT,
            "schema": SCHEMA_VERSION,
            **asdict(self),
        }


def _build_split_quadruple(
    *,
    generation_id: str,
    generation_index: int,
    probe_skip: int,
    selection_skip: int,
    confirm_skip: int,
    virgin_vault_skip: int,
    split_stride: int,
) -> SplitQuadruple:
    vault_id = f"{generation_id}:virgin_vault@{virgin_vault_skip}"
    core = {
        "object": SPLIT_ALLOCATION_OBJECT,
        "schema": SCHEMA_VERSION,
        "generation_id": generation_id,
        "generation_index": generation_index,
        "probe_skip": probe_skip,
        "selection_skip": selection_skip,
        "confirm_skip": confirm_skip,
        "virgin_vault_skip": virgin_vault_skip,
        "split_stride": split_stride,
        "vault_id": vault_id,
    }
    return SplitQuadruple(
        generation_id=generation_id,
        generation_index=generation_index,
        probe_skip=probe_skip,
        selection_skip=selection_skip,
        confirm_skip=confirm_skip,
        virgin_vault_skip=virgin_vault_skip,
        split_stride=split_stride,
        vault_id=vault_id,
        allocation_sha256=compute_payload_sha256(core),
    )


class DisjointSplitAllocator:
    """Deterministic cross-generation shard-skip allocator with permanent virgin-vault sealing."""

    def __init__(
        self,
        *,
        start_skip: int = DEFAULT_START_SKIP,
        split_stride: int = DEFAULT_SPLIT_STRIDE,
        frozen_skip_floor: int = DEFAULT_FROZEN_SKIP_FLOOR,
    ) -> None:
        self._split_stride = _validate_positive_int(split_stride, "split_stride")
        self._frozen_skip_floor = _validate_nonneg_int(frozen_skip_floor, "frozen_skip_floor")
        self._start_skip = _validate_nonneg_int(start_skip, "start_skip")
        if self._start_skip < self._frozen_skip_floor:
            raise SplitContaminationError(
                f"start_skip ({self._start_skip}) cannot intrude into frozen window [0, {self._frozen_skip_floor})"
            )
        self._allocations: dict[str, SplitQuadruple] = {}
        self._generation_order: list[str] = []
        self._reserved_intervals: dict[tuple[int, int], tuple[str, str]] = {}
        self._sealed_vault_intervals: dict[tuple[int, int], dict[str, Any]] = {}
        self._sealed_generation_ids: set[str] = set()
        self._cursor: int = self._start_skip

    @property
    def start_skip(self) -> int:
        return self._start_skip

    @property
    def split_stride(self) -> int:
        return self._split_stride

    @property
    def frozen_skip_floor(self) -> int:
        return self._frozen_skip_floor

    @property
    def generations(self) -> tuple[SplitQuadruple, ...]:
        return tuple(self._allocations[gid] for gid in self._generation_order)

    @property
    def sealed_vault_skips(self) -> frozenset[int]:
        return frozenset(start for start, _end in self._sealed_vault_intervals)

    @property
    def sealed_vault_receipts(self) -> dict[str, dict[str, Any]]:
        return {
            str(doc["vault_key"]): dict(doc)
            for doc in self._sealed_vault_intervals.values()
        }

    @staticmethod
    def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
        return start_a < end_b and start_b < end_a

    def _validate_interval_available(self, skip_offset: int, role: str, generation_id: str) -> None:
        if skip_offset < self._frozen_skip_floor:
            raise SplitContaminationError(
                f"{generation_id}.{role} skip={skip_offset} intrudes into frozen window [0, {self._frozen_skip_floor})"
            )
        cand_end = skip_offset + self._split_stride
        for (s_start, s_end), seal_doc in self._sealed_vault_intervals.items():
            if self._intervals_overlap(skip_offset, cand_end, s_start, s_end):
                raise VaultSealedError(
                    f"Refusing {generation_id}.{role} skip={skip_offset} ([{skip_offset}, {cand_end})): "
                    f"overlaps permanently sealed virgin vault [{s_start}, {s_end}) ({seal_doc['vault_key']})"
                )
        for (r_start, r_end), (owner_gen, owner_role) in self._reserved_intervals.items():
            if self._intervals_overlap(skip_offset, cand_end, r_start, r_end):
                raise SplitContaminationError(
                    f"Refusing {generation_id}.{role} skip={skip_offset} ([{skip_offset}, {cand_end})): "
                    f"overlaps existing {owner_gen}.{owner_role} [{r_start}, {r_end})"
                )

    def _first_conflict_end(self, skip_offset: int) -> int | None:
        if skip_offset < self._frozen_skip_floor:
            return self._frozen_skip_floor
        cand_end = skip_offset + self._split_stride
        conflict_ends: list[int] = []
        for s_start, s_end in self._sealed_vault_intervals:
            if self._intervals_overlap(skip_offset, cand_end, s_start, s_end):
                conflict_ends.append(s_end)
        for r_start, r_end in self._reserved_intervals:
            if self._intervals_overlap(skip_offset, cand_end, r_start, r_end):
                conflict_ends.append(r_end)
        return max(conflict_ends) if conflict_ends else None

    def _next_free_offset(self, start_from: int) -> int:
        pos = max(start_from, self._frozen_skip_floor)
        while True:
            jump_to = self._first_conflict_end(pos)
            if jump_to is None:
                return pos
            pos = jump_to

    def allocate_generation(
        self,
        generation_id: str,
        *,
        probe_skip: int | None = None,
        selection_skip: int | None = None,
        confirm_skip: int | None = None,
        virgin_vault_skip: int | None = None,
    ) -> SplitQuadruple:
        """Reserve a non-overlapping ``(probe, selection, confirm, virgin_vault)`` quadruple."""
        gid = _validate_nonempty_str(generation_id, "generation_id")
        if gid in self._allocations:
            raise SplitContaminationError(f"Generation {gid!r} already has reserved splits")

        explicit = (probe_skip, selection_skip, confirm_skip, virgin_vault_skip)
        if all(v is None for v in explicit):
            p_sk = self._next_free_offset(self._cursor)
            s_sk = self._next_free_offset(p_sk + self._split_stride)
            c_sk = self._next_free_offset(s_sk + self._split_stride)
            v_sk = self._next_free_offset(c_sk + self._split_stride)
            self._cursor = v_sk + self._split_stride
        elif all(v is not None for v in explicit):
            p_sk = _validate_nonneg_int(probe_skip, "probe_skip")
            s_sk = _validate_nonneg_int(selection_skip, "selection_skip")
            c_sk = _validate_nonneg_int(confirm_skip, "confirm_skip")
            v_sk = _validate_nonneg_int(virgin_vault_skip, "virgin_vault_skip")
            role_pairs = list(zip(SPLIT_ROLES, (p_sk, s_sk, c_sk, v_sk), strict=True))
            for role_name, sk_val in role_pairs:
                self._validate_interval_available(sk_val, role_name, gid)
            for idx_a in range(len(role_pairs)):
                role_a, sk_a = role_pairs[idx_a]
                for idx_b in range(idx_a + 1, len(role_pairs)):
                    role_b, sk_b = role_pairs[idx_b]
                    if self._intervals_overlap(
                        sk_a, sk_a + self._split_stride, sk_b, sk_b + self._split_stride
                    ):
                        raise SplitContaminationError(
                            f"Intra-generation split overlap in {gid}: {role_a}={sk_a} overlaps {role_b}={sk_b}"
                        )
        else:
            raise ValueError(
                "Either provide all four explicit skip offsets or leave all four as None for automatic allocation"
            )

        gen_index = len(self._generation_order) + 1
        quad = _build_split_quadruple(
            generation_id=gid,
            generation_index=gen_index,
            probe_skip=p_sk,
            selection_skip=s_sk,
            confirm_skip=c_sk,
            virgin_vault_skip=v_sk,
            split_stride=self._split_stride,
        )
        for role_name, sk_val in quad.skip_offsets().items():
            self._reserved_intervals[(sk_val, sk_val + self._split_stride)] = (gid, role_name)
        self._allocations[gid] = quad
        self._generation_order.append(gid)
        return quad

    def seal_virgin_vault(
        self,
        generation_id: str,
        *,
        evaluation_receipt_sha256: str | None = None,
        outcome: str = "CONSUMED",
    ) -> dict[str, Any]:
        """Permanently consume and seal ``generation_id``'s virgin vault skip offset."""
        gid = _validate_nonempty_str(generation_id, "generation_id")
        if gid not in self._allocations:
            raise SplitContaminationError(f"Unknown generation_id {gid!r}; allocate splits before sealing")
        quad = self._allocations[gid]
        v_interval = (quad.virgin_vault_skip, quad.virgin_vault_skip + self._split_stride)
        if gid in self._sealed_generation_ids or v_interval in self._sealed_vault_intervals:
            raise VaultSealedError(
                f"Virgin vault for {gid} (skip={quad.virgin_vault_skip}) is already permanently sealed"
            )

        eval_sha = (
            _validate_sha256_hex(evaluation_receipt_sha256, "evaluation_receipt_sha256")
            if evaluation_receipt_sha256 is not None
            else None
        )
        clean_outcome = _validate_nonempty_str(outcome, "outcome")
        core = {
            "object": VAULT_SEAL_OBJECT,
            "schema": SCHEMA_VERSION,
            "vault_key": quad.vault_id,
            "generation_id": gid,
            "generation_index": quad.generation_index,
            "virgin_vault_skip": quad.virgin_vault_skip,
            "split_stride": self._split_stride,
            "interval_start": v_interval[0],
            "interval_end": v_interval[1],
            "outcome": clean_outcome,
            "evaluation_receipt_sha256": eval_sha,
            "sealed": True,
        }
        core["receipt_sha256"] = compute_payload_sha256(core)
        self._sealed_vault_intervals[v_interval] = core
        self._sealed_generation_ids.add(gid)
        return dict(core)

    def seal_consumed_skip_offset(
        self,
        skip_offset: int,
        *,
        reason: str = "externally_consumed_vault",
        evaluation_receipt_sha256: str | None = None,
    ) -> dict[str, Any]:
        """Permanently seal an arbitrary consumed virgin-vault skip offset against reuse."""
        sk = _validate_nonneg_int(skip_offset, "skip_offset")
        if sk < self._frozen_skip_floor:
            raise SplitContaminationError(
                f"skip_offset ({sk}) intrudes into frozen window [0, {self._frozen_skip_floor})"
            )
        clean_reason = _validate_nonempty_str(reason, "reason")
        eval_sha = (
            _validate_sha256_hex(evaluation_receipt_sha256, "evaluation_receipt_sha256")
            if evaluation_receipt_sha256 is not None
            else None
        )
        interval = (sk, sk + self._split_stride)
        for s_start, s_end in self._sealed_vault_intervals:
            if self._intervals_overlap(interval[0], interval[1], s_start, s_end):
                raise VaultSealedError(
                    f"Skip offset {sk} ([{interval[0]}, {interval[1]})) overlaps already sealed vault [{s_start}, {s_end})"
                )
        for (r_start, r_end), (owner_gen, owner_role) in self._reserved_intervals.items():
            if self._intervals_overlap(interval[0], interval[1], r_start, r_end):
                raise SplitContaminationError(
                    f"Skip offset {sk} overlaps already reserved split {owner_gen}.{owner_role} [{r_start}, {r_end})"
                )

        vault_key = f"external:virgin_vault@{sk}"
        core = {
            "object": VAULT_SEAL_OBJECT,
            "schema": SCHEMA_VERSION,
            "vault_key": vault_key,
            "generation_id": None,
            "generation_index": None,
            "virgin_vault_skip": sk,
            "split_stride": self._split_stride,
            "interval_start": interval[0],
            "interval_end": interval[1],
            "outcome": clean_reason,
            "evaluation_receipt_sha256": eval_sha,
            "sealed": True,
        }
        core["receipt_sha256"] = compute_payload_sha256(core)
        self._sealed_vault_intervals[interval] = core
        return dict(core)

    def is_vault_sealed(self, skip_offset: int) -> bool:
        """Return True iff ``[skip_offset, skip_offset + split_stride)`` touches a sealed vault."""
        sk = _validate_nonneg_int(skip_offset, "skip_offset")
        cand_end = sk + self._split_stride
        return any(self._intervals_overlap(sk, cand_end, a, b) for a, b in self._sealed_vault_intervals)

    def is_generation_sealed(self, generation_id: str) -> bool:
        """Return True iff ``generation_id`` has had its virgin vault permanently sealed."""
        gid = _validate_nonempty_str(generation_id, "generation_id")
        return gid in self._sealed_generation_ids

    def verify_all_splits_disjoint(self) -> dict[str, Any]:
        """Verify 0 pairwise overlap across all allocated splits and return a signed manifest."""
        all_intervals: list[tuple[int, int, str, str]] = []
        all_offsets: list[int] = []
        for gid in self._generation_order:
            quad = self._allocations[gid]
            expected = _build_split_quadruple(
                generation_id=quad.generation_id,
                generation_index=quad.generation_index,
                probe_skip=quad.probe_skip,
                selection_skip=quad.selection_skip,
                confirm_skip=quad.confirm_skip,
                virgin_vault_skip=quad.virgin_vault_skip,
                split_stride=quad.split_stride,
            )
            if quad.allocation_sha256 != expected.allocation_sha256:
                raise SplitContaminationError(f"Tampered SplitQuadruple digest detected for {gid!r}")
            for role, offset in quad.skip_offsets().items():
                if offset < self._frozen_skip_floor:
                    raise SplitContaminationError(
                        f"{gid}.{role} offset {offset} violates frozen floor {self._frozen_skip_floor}"
                    )
                all_intervals.append((offset, offset + self._split_stride, gid, role))
                all_offsets.append(offset)

        sorted_iv = sorted(all_intervals, key=lambda t: (t[0], t[1]))
        for idx in range(len(sorted_iv) - 1):
            a_s, a_e, g_a, r_a = sorted_iv[idx]
            b_s, b_e, g_b, r_b = sorted_iv[idx + 1]
            if a_e > b_s:
                raise SplitContaminationError(
                    f"Overlap detected between {g_a}.{r_a}[{a_s},{a_e}) and {g_b}.{r_b}[{b_s},{b_e})"
                )

        manifest_core = {
            "total_generations": len(self._generation_order),
            "total_splits": len(all_offsets),
            "frozen_skip_floor": self._frozen_skip_floor,
            "split_stride": self._split_stride,
            "all_split_offsets": sorted(all_offsets),
            "sealed_vault_skips": sorted(self.sealed_vault_skips),
        }
        manifest_sha = compute_payload_sha256(manifest_core)
        out = {
            "disjoint": True,
            **manifest_core,
            "splits_manifest_sha256": manifest_sha,
        }
        out["receipt_sha256"] = compute_payload_sha256(out)
        return out

    def export_state(self) -> dict[str, Any]:
        """Export full allocator reservations and permanent vault seals with SHA-256 verification."""
        core = {
            "schema": SCHEMA_VERSION,
            "start_skip": self._start_skip,
            "split_stride": self._split_stride,
            "frozen_skip_floor": self._frozen_skip_floor,
            "cursor": self._cursor,
            "allocations": [self._allocations[gid].to_dict() for gid in self._generation_order],
            "sealed_vaults": [
                dict(self._sealed_vault_intervals[iv]) for iv in sorted(self._sealed_vault_intervals)
            ],
        }
        core["state_sha256"] = compute_payload_sha256(core)
        return core

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> DisjointSplitAllocator:
        """Reconstruct a verified :class:`DisjointSplitAllocator` from :meth:`export_state`."""
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        claimed = state.get("state_sha256")
        _validate_sha256_hex(claimed, "state_sha256")
        core = {k: v for k, v in state.items() if k != "state_sha256"}
        if compute_payload_sha256(core) != claimed:
            raise SplitContaminationError("DisjointSplitAllocator state_sha256 verification failed")

        alloc = cls(
            start_skip=state["start_skip"],
            split_stride=state["split_stride"],
            frozen_skip_floor=state["frozen_skip_floor"],
        )
        raw_allocs = state.get("allocations")
        raw_seals = state.get("sealed_vaults")
        if not isinstance(raw_allocs, Sequence) or not isinstance(raw_seals, Sequence):
            raise SplitContaminationError("Invalid allocations or sealed_vaults sequence in allocator state")

        for item in raw_allocs:
            if not isinstance(item, Mapping):
                raise SplitContaminationError("Allocation entry must be a mapping")
            quad = alloc.allocate_generation(
                item["generation_id"],
                probe_skip=item["probe_skip"],
                selection_skip=item["selection_skip"],
                confirm_skip=item["confirm_skip"],
                virgin_vault_skip=item["virgin_vault_skip"],
            )
            if quad.allocation_sha256 != item.get("allocation_sha256"):
                raise SplitContaminationError(f"Allocation hash mismatch for {quad.generation_id!r}")

        for seal in raw_seals:
            if not isinstance(seal, Mapping):
                raise SplitContaminationError("Sealed vault entry must be a mapping")
            seal_claimed = seal.get("receipt_sha256")
            _validate_sha256_hex(seal_claimed, "sealed_vault.receipt_sha256")
            seal_core = {k: v for k, v in seal.items() if k != "receipt_sha256"}
            if compute_payload_sha256(seal_core) != seal_claimed:
                raise VaultSealedError("Tampered virgin-vault seal receipt detected during state restore")
            gid = seal.get("generation_id")
            if gid is not None:
                alloc.seal_virgin_vault(
                    str(gid),
                    evaluation_receipt_sha256=seal.get("evaluation_receipt_sha256"),
                    outcome=str(seal.get("outcome", "CONSUMED")),
                )
            else:
                alloc.seal_consumed_skip_offset(
                    seal["virgin_vault_skip"],
                    reason=str(seal.get("outcome", "externally_consumed_vault")),
                    evaluation_receipt_sha256=seal.get("evaluation_receipt_sha256"),
                )
        alloc._cursor = max(alloc._cursor, _validate_nonneg_int(state["cursor"], "cursor"))
        return alloc


# ---------------------------------------------------------------------------
# 4. Deterministic Offline CPU Drift-Triggered Recalibration Scheduler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftTelemetryWindow:
    """Measured telemetry snapshot governing offline CPU S5 recalibration scheduling."""

    drift_ratio: float
    observed_rogue_norm_ratio: float
    free_ram_gib: float
    nll_mean: float | None = None
    base_nll_mean: float | None = None
    window_id: str = "telemetry_window_default"
    eval_lock_held: bool = False
    foreground_game_active: bool = False
    cuda_disabled_for_s5: bool = True

    def __post_init__(self) -> None:
        _validate_finite_float(self.drift_ratio, "drift_ratio")
        _validate_nonneg_finite(self.observed_rogue_norm_ratio, "observed_rogue_norm_ratio")
        _validate_nonneg_finite(self.free_ram_gib, "free_ram_gib")
        if self.nll_mean is not None:
            _validate_positive_finite(self.nll_mean, "nll_mean")
        if self.base_nll_mean is not None:
            _validate_positive_finite(self.base_nll_mean, "base_nll_mean")
        _validate_nonempty_str(self.window_id, "window_id")
        _validate_bool(self.eval_lock_held, "eval_lock_held")
        _validate_bool(self.foreground_game_active, "foreground_game_active")
        _validate_bool(self.cuda_disabled_for_s5, "cuda_disabled_for_s5")


@dataclass(frozen=True)
class DriftTriggerDecision:
    """Deterministic scheduling decision emitted by :func:`evaluate_drift_recalibration_trigger`."""

    object: str
    schema: str
    window_id: str
    should_schedule: bool
    trigger_fired: bool
    deferred: bool
    status: str
    execution_mode: str
    drift_ratio: float
    general_drift_trigger_rel: float
    observed_rogue_norm_ratio: float
    rogue_expert_norm_ratio_threshold: float
    free_ram_gib: float
    min_free_ram_gib: float
    is_overnight_window: bool
    nll_mean: float | None
    trigger_reasons: tuple[str, ...]
    defer_reasons: tuple[str, ...]
    recommended_s5_stages: tuple[str, ...]
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize trigger decision to a canonical JSON-safe dictionary."""
        return {
            "object": self.object,
            "schema": self.schema,
            "window_id": self.window_id,
            "should_schedule": self.should_schedule,
            "trigger_fired": self.trigger_fired,
            "deferred": self.deferred,
            "status": self.status,
            "execution_mode": self.execution_mode,
            "drift_ratio": self.drift_ratio,
            "general_drift_trigger_rel": self.general_drift_trigger_rel,
            "observed_rogue_norm_ratio": self.observed_rogue_norm_ratio,
            "rogue_expert_norm_ratio_threshold": self.rogue_expert_norm_ratio_threshold,
            "free_ram_gib": self.free_ram_gib,
            "min_free_ram_gib": self.min_free_ram_gib,
            "is_overnight_window": self.is_overnight_window,
            "nll_mean": self.nll_mean,
            "trigger_reasons": list(self.trigger_reasons),
            "defer_reasons": list(self.defer_reasons),
            "recommended_s5_stages": list(self.recommended_s5_stages),
            "receipt_sha256": self.receipt_sha256,
        }


def _extract_drift_ratio_from_mapping(window: Mapping[str, Any]) -> tuple[float, float | None, float | None]:
    """Resolve ``(drift_ratio, nll_mean, base_nll_mean)`` from a telemetry mapping."""
    nll_mean: float | None = (
        _validate_positive_finite(window["nll_mean"], "nll_mean")
        if window.get("nll_mean") is not None
        else None
    )
    base_nll_mean: float | None = (
        _validate_positive_finite(window["base_nll_mean"], "base_nll_mean")
        if window.get("base_nll_mean") is not None
        else None
    )

    if "drift_ratio" in window:
        return _validate_finite_float(window["drift_ratio"], "drift_ratio"), nll_mean, base_nll_mean
    if "general_drift_rel" in window:
        return _validate_finite_float(window["general_drift_rel"], "general_drift_rel"), nll_mean, base_nll_mean
    if "current_general_nll" in window and "base_general_nll" in window:
        cur_g = _validate_positive_finite(window["current_general_nll"], "current_general_nll")
        base_g = _validate_positive_finite(window["base_general_nll"], "base_general_nll")
        return (cur_g - base_g) / base_g, nll_mean, base_nll_mean

    domain_nlls = window.get("domain_nlls")
    base_domain_nlls = window.get("base_domain_nlls")
    if isinstance(domain_nlls, Mapping) and isinstance(base_domain_nlls, Mapping) and domain_nlls:
        if set(domain_nlls.keys()) != set(base_domain_nlls.keys()):
            raise ValueError("domain_nlls and base_domain_nlls must contain identical domain keys")
        gen_keys = [
            str(k)
            for k in sorted(domain_nlls.keys(), key=str)
            if str(k).startswith(GENERAL_EXPERT_PREFIX) or str(k) in ("general", "general_nll")
        ]
        if not gen_keys:
            raise ValueError("domain_nlls must include a general domain key (e.g. 'E0_general_fineweb')")
        cur_vals = [_validate_positive_finite(domain_nlls[k], f"domain_nlls[{k!r}]") for k in domain_nlls]
        base_vals = [_validate_positive_finite(base_domain_nlls[k], f"base_domain_nlls[{k!r}]") for k in base_domain_nlls]
        drifts = [
            (
                _validate_positive_finite(domain_nlls[gk], f"domain_nlls[{gk!r}]")
                - _validate_positive_finite(base_domain_nlls[gk], f"base_domain_nlls[{gk!r}]")
            )
            / _validate_positive_finite(base_domain_nlls[gk], f"base_domain_nlls[{gk!r}]")
            for gk in gen_keys
        ]
        eff_nll_mean = nll_mean if nll_mean is not None else float(sum(cur_vals) / len(cur_vals))
        eff_base_mean = base_nll_mean if base_nll_mean is not None else float(sum(base_vals) / len(base_vals))
        return max(drifts), eff_nll_mean, eff_base_mean

    raise ValueError(
        "telemetry_window must provide 'drift_ratio', 'general_drift_rel', "
        "('current_general_nll', 'base_general_nll'), or ('domain_nlls', 'base_domain_nlls')"
    )


def _coerce_telemetry_window(telemetry_window: DriftTelemetryWindow | Mapping[str, Any]) -> DriftTelemetryWindow:
    """Validate and normalize ``telemetry_window`` into a :class:`DriftTelemetryWindow`."""
    if isinstance(telemetry_window, DriftTelemetryWindow):
        return telemetry_window
    if not isinstance(telemetry_window, Mapping):
        raise TypeError(
            f"telemetry_window must be a DriftTelemetryWindow or Mapping, got {type(telemetry_window).__name__}"
        )

    drift_ratio, nll_mean, base_nll_mean = _extract_drift_ratio_from_mapping(telemetry_window)

    if "observed_rogue_norm_ratio" in telemetry_window:
        rogue_ratio = _validate_nonneg_finite(
            telemetry_window["observed_rogue_norm_ratio"], "observed_rogue_norm_ratio"
        )
    elif "rogue_expert_norm_ratio" in telemetry_window:
        rogue_ratio = _validate_nonneg_finite(
            telemetry_window["rogue_expert_norm_ratio"], "rogue_expert_norm_ratio"
        )
    elif "max_expert_norm_ratio" in telemetry_window:
        rogue_ratio = _validate_nonneg_finite(
            telemetry_window["max_expert_norm_ratio"], "max_expert_norm_ratio"
        )
    elif "expert_norms" in telemetry_window:
        rogue_ratio, _ = compute_expert_norm_ratio(telemetry_window["expert_norms"])
    else:
        raise ValueError(
            "telemetry_window must provide 'observed_rogue_norm_ratio', 'rogue_expert_norm_ratio', "
            "'max_expert_norm_ratio', or 'expert_norms'"
        )

    free_ram_key = next(
        (k for k in ("free_ram_gib", "available_ram_gib", "mem_available_gib") if k in telemetry_window),
        None,
    )
    if free_ram_key is None:
        raise ValueError("telemetry_window must provide 'free_ram_gib' (or 'available_ram_gib')")
    free_ram = _validate_nonneg_finite(telemetry_window[free_ram_key], free_ram_key)

    return DriftTelemetryWindow(
        drift_ratio=drift_ratio,
        observed_rogue_norm_ratio=rogue_ratio,
        free_ram_gib=free_ram,
        nll_mean=nll_mean,
        base_nll_mean=base_nll_mean,
        window_id=_validate_nonempty_str(telemetry_window.get("window_id", "telemetry_window_default"), "window_id"),
        eval_lock_held=_validate_bool(telemetry_window.get("eval_lock_held", False), "eval_lock_held"),
        foreground_game_active=_validate_bool(
            telemetry_window.get("foreground_game_active", False), "foreground_game_active"
        ),
        cuda_disabled_for_s5=_validate_bool(
            telemetry_window.get("cuda_disabled_for_s5", True), "cuda_disabled_for_s5"
        ),
    )


def evaluate_drift_recalibration_trigger(
    telemetry_window: DriftTelemetryWindow | Mapping[str, Any],
    *,
    general_drift_trigger_rel: float = DEFAULT_GENERAL_DRIFT_TRIGGER_REL,
    rogue_expert_norm_ratio: float = DEFAULT_ROGUE_EXPERT_NORM_RATIO,
    min_free_ram_gib: float = DEFAULT_MIN_FREE_RAM_GIB,
    is_overnight_window: bool = True,
) -> DriftTriggerDecision:
    """Decide deterministically whether to schedule an offline CPU-only ``S5`` surgery pass.

    Triggers when general-domain drift reaches ``general_drift_trigger_rel`` OR an
    active expert exhibits a norm spike ``>= rogue_expert_norm_ratio``, and defers
    unless running inside the overnight window with ``free_ram_gib >= min_free_ram_gib``,
    zero active evaluation lock, zero foreground game preemption, and CPU-only mode.
    """
    drift_thresh = _validate_positive_finite(general_drift_trigger_rel, "general_drift_trigger_rel")
    rogue_thresh = _validate_positive_finite(rogue_expert_norm_ratio, "rogue_expert_norm_ratio")
    min_ram = _validate_nonneg_finite(min_free_ram_gib, "min_free_ram_gib")
    overnight = _validate_bool(is_overnight_window, "is_overnight_window")
    win = _coerce_telemetry_window(telemetry_window)

    trigger_reasons: list[str] = []
    if win.drift_ratio >= drift_thresh:
        trigger_reasons.append("general_drift_threshold_exceeded")
    if win.observed_rogue_norm_ratio >= rogue_thresh:
        trigger_reasons.append("rogue_expert_norm_ratio_exceeded")
    trigger_fired = bool(trigger_reasons)

    defer_reasons: list[str] = []
    if trigger_fired:
        if not overnight:
            defer_reasons.append("outside_overnight_window")
        if win.free_ram_gib < min_ram:
            defer_reasons.append("insufficient_free_ram")
        if win.eval_lock_held:
            defer_reasons.append("active_eval_lock_held")
        if win.foreground_game_active:
            defer_reasons.append("foreground_game_preempted")
        if not win.cuda_disabled_for_s5:
            defer_reasons.append("cpu_only_isolation_required")

    should_schedule = trigger_fired and not defer_reasons
    deferred = trigger_fired and bool(defer_reasons)

    if not trigger_fired:
        status = "NO_ACTION_HEALTHY"
    elif should_schedule:
        status = "SCHEDULED_OFFLINE_CPU_S5"
    elif defer_reasons == ["outside_overnight_window"]:
        status = "DEFERRED_DAYTIME_WINDOW"
    elif defer_reasons == ["insufficient_free_ram"]:
        status = "DEFERRED_LOW_RAM"
    elif set(defer_reasons) == {"outside_overnight_window", "insufficient_free_ram"}:
        status = "DEFERRED_DAYTIME_AND_LOW_RAM"
    else:
        status = "DEFERRED_EXECUTION_GUARD"

    execution_mode = "cpu_only_offline_s5" if should_schedule else "none"
    stages = RECOMMENDED_S5_STAGES if should_schedule else ()

    core = {
        "object": DRIFT_TRIGGER_OBJECT,
        "schema": SCHEMA_VERSION,
        "window_id": win.window_id,
        "should_schedule": should_schedule,
        "trigger_fired": trigger_fired,
        "deferred": deferred,
        "status": status,
        "execution_mode": execution_mode,
        "drift_ratio": win.drift_ratio,
        "general_drift_trigger_rel": drift_thresh,
        "observed_rogue_norm_ratio": win.observed_rogue_norm_ratio,
        "rogue_expert_norm_ratio_threshold": rogue_thresh,
        "free_ram_gib": win.free_ram_gib,
        "min_free_ram_gib": min_ram,
        "is_overnight_window": overnight,
        "nll_mean": win.nll_mean,
        "trigger_reasons": list(trigger_reasons),
        "defer_reasons": list(defer_reasons),
        "recommended_s5_stages": list(stages),
    }
    digest = compute_payload_sha256(core)
    return DriftTriggerDecision(
        object=DRIFT_TRIGGER_OBJECT,
        schema=SCHEMA_VERSION,
        window_id=win.window_id,
        should_schedule=should_schedule,
        trigger_fired=trigger_fired,
        deferred=deferred,
        status=status,
        execution_mode=execution_mode,
        drift_ratio=win.drift_ratio,
        general_drift_trigger_rel=drift_thresh,
        observed_rogue_norm_ratio=win.observed_rogue_norm_ratio,
        rogue_expert_norm_ratio_threshold=rogue_thresh,
        free_ram_gib=win.free_ram_gib,
        min_free_ram_gib=min_ram,
        is_overnight_window=overnight,
        nll_mean=win.nll_mean,
        trigger_reasons=tuple(trigger_reasons),
        defer_reasons=tuple(defer_reasons),
        recommended_s5_stages=stages,
        receipt_sha256=digest,
    )


# ---------------------------------------------------------------------------
# 5. Hash-Chained Campaign Ledger & ANSE Monotone Improvement Ratchet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WSPGenerationReceipt:
    """Immutable hash-chained cryptographic receipt for one committed WSP generation."""

    object: str
    schema: str
    campaign_id: str
    generation_index: int
    generation_id: str
    arm_name: str
    probe_skip: int
    selection_skip: int
    confirm_skip: int
    virgin_vault_skip: int
    split_stride: int
    allocation_sha256: str
    nll_mean: float
    base_nll_mean: float
    mean_diff_vs_base: float
    drift_ratio: float
    worst_domain_drift_ratio: float
    guard_ok: bool
    anse_energy: float
    delta_anse_energy: float | None
    checkpoint_sha256: str
    parent_checkpoint_sha256: str
    virgin_vault_sealed: bool
    vault_seal_receipt_sha256: str | None
    trigger_decision_sha256: str | None
    prev_receipt_sha256: str
    receipt_sha256: str

    def core_payload(self) -> dict[str, Any]:
        """Return canonical JSON core excluding ``receipt_sha256``."""
        raw = asdict(self)
        raw.pop("receipt_sha256", None)
        return raw

    def to_dict(self) -> dict[str, Any]:
        """Serialize receipt to a canonical JSON-safe dictionary."""
        return asdict(self)


def _receipt_to_dict(receipt: WSPGenerationReceipt | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(receipt, WSPGenerationReceipt):
        return receipt.to_dict()
    if isinstance(receipt, Mapping):
        return dict(receipt)
    raise TypeError(f"receipt must be a WSPGenerationReceipt or Mapping, got {type(receipt).__name__}")


def verify_generation_receipt_hash(receipt: WSPGenerationReceipt | Mapping[str, Any]) -> bool:
    """Independently verify schema tags and canonical SHA-256 of a single generation receipt."""
    doc = _receipt_to_dict(receipt)
    claimed = doc.get("receipt_sha256")
    if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
        return False
    if doc.get("object") != GENERATION_RECEIPT_OBJECT or doc.get("schema") != SCHEMA_VERSION:
        return False
    core = {k: v for k, v in doc.items() if k != "receipt_sha256"}
    try:
        expected = compute_payload_sha256(core)
    except (TypeError, ValueError):
        return False
    return claimed == expected


def verify_ledger_chain(
    entries: Sequence[WSPGenerationReceipt | Mapping[str, Any]],
    *,
    campaign_id: str | None = None,
    min_energy_improvement: float = 0.0,
    split_stride: int = DEFAULT_SPLIT_STRIDE,
    frozen_skip_floor: int = DEFAULT_FROZEN_SKIP_FLOOR,
    require_monotone_anse: bool = True,
    raise_on_error: bool = False,
) -> dict[str, Any]:
    """Audit hash chaining, split disjointness, and ANSE monotonicity across generation receipts."""
    if isinstance(entries, (str, bytes)) or not isinstance(entries, Sequence):
        raise TypeError("entries must be a sequence of WSPGenerationReceipt or Mapping")
    min_imp = _validate_nonneg_finite(min_energy_improvement, "min_energy_improvement")
    stride = _validate_positive_int(split_stride, "split_stride")
    floor = _validate_nonneg_int(frozen_skip_floor, "frozen_skip_floor")
    _validate_bool(require_monotone_anse, "require_monotone_anse")
    _validate_bool(raise_on_error, "raise_on_error")
    expected_campaign = _validate_nonempty_str(campaign_id, "campaign_id") if campaign_id is not None else None

    violations: list[str] = []
    expected_prev = GENESIS_RECEIPT_SHA256
    prev_energy: float | None = None
    reserved_intervals: list[tuple[int, int, str, str]] = []
    sealed_vault_intervals: list[tuple[int, int, str]] = []
    monotone_ok = True
    last_hash = GENESIS_RECEIPT_SHA256

    for idx, raw_entry in enumerate(entries):
        doc = _receipt_to_dict(raw_entry)
        label = f"entry[{idx}]"

        if doc.get("object") != GENERATION_RECEIPT_OBJECT or doc.get("schema") != SCHEMA_VERSION:
            violations.append(f"{label}: invalid object/schema tag")

        entry_campaign = doc.get("campaign_id")
        if not isinstance(entry_campaign, str) or not entry_campaign.strip():
            violations.append(f"{label}: missing campaign_id")
        else:
            if expected_campaign is None:
                expected_campaign = entry_campaign
            elif entry_campaign != expected_campaign:
                violations.append(f"{label}: campaign_id {entry_campaign!r} != {expected_campaign!r}")

        gen_idx = doc.get("generation_index")
        if isinstance(gen_idx, bool) or gen_idx != idx + 1:
            violations.append(f"{label}: expected generation_index={idx + 1}, got {gen_idx!r}")

        prev_sha = doc.get("prev_receipt_sha256")
        if prev_sha != expected_prev:
            violations.append(f"{label}: prev_receipt_sha256 mismatch ({prev_sha!r} != {expected_prev!r})")

        if not verify_generation_receipt_hash(doc):
            violations.append(f"{label}: receipt_sha256 verification failed")

        for sha_field in ("checkpoint_sha256", "parent_checkpoint_sha256", "allocation_sha256", "receipt_sha256"):
            val = doc.get(sha_field)
            if not isinstance(val, str) or not _SHA256_HEX_RE.match(val):
                violations.append(f"{label}: invalid {sha_field}")

        # Split non-overlap & permanent vault sealing checks
        entry_stride = doc.get("split_stride", stride)
        if isinstance(entry_stride, bool) or not isinstance(entry_stride, int) or entry_stride != stride:
            violations.append(f"{label}: split_stride {entry_stride!r} != expected {stride}")
            entry_stride = stride

        gid = str(doc.get("generation_id", f"gen_{idx + 1}"))
        entry_skips: dict[str, int] = {}
        for role in SPLIT_ROLES:
            sk_val = doc.get(f"{role}_skip")
            if isinstance(sk_val, bool) or not isinstance(sk_val, int) or sk_val < floor:
                violations.append(f"{label}: {role}_skip {sk_val!r} must be int >= {floor}")
                continue
            entry_skips[role] = sk_val
            iv_s, iv_e = sk_val, sk_val + entry_stride
            for sv_s, sv_e, sv_owner in sealed_vault_intervals:
                if iv_s < sv_e and sv_s < iv_e:
                    violations.append(
                        f"{label}: {role}_skip={sk_val} reuses permanently sealed virgin vault from {sv_owner}"
                    )
            for rv_s, rv_e, rv_gen, rv_role in reserved_intervals:
                if iv_s < rv_e and rv_s < iv_e:
                    violations.append(
                        f"{label}: {role}_skip={sk_val} overlaps {rv_gen}.{rv_role}[{rv_s},{rv_e})"
                    )
            reserved_intervals.append((iv_s, iv_e, gid, role))

        if len(entry_skips) == 4:
            iv_list = [(sk, sk + entry_stride, role) for role, sk in entry_skips.items()]
            for a_i in range(len(iv_list)):
                a_s, a_e, r_a = iv_list[a_i]
                for b_i in range(a_i + 1, len(iv_list)):
                    b_s, b_e, r_b = iv_list[b_i]
                    if a_s < b_e and b_s < a_e:
                        violations.append(f"{label}: intra-generation overlap {r_a}={a_s} vs {r_b}={b_s}")
            if bool(doc.get("virgin_vault_sealed")):
                vs = entry_skips["virgin_vault"]
                sealed_vault_intervals.append((vs, vs + entry_stride, gid))

        # ANSE guard & strict cross-generation monotone improvement checks
        guard_ok = doc.get("guard_ok")
        raw_energy = doc.get("anse_energy")
        if guard_ok is not True:
            violations.append(f"{label}: guard_ok must be True")
            monotone_ok = False
        if isinstance(raw_energy, bool) or not isinstance(raw_energy, (int, float)) or not math.isfinite(float(raw_energy)):
            violations.append(f"{label}: anse_energy must be finite, got {raw_energy!r}")
            monotone_ok = False
        else:
            cur_energy = float(raw_energy)
            if cur_energy >= ANSE_MAXIMUM_PAIN * 0.5:
                violations.append(f"{label}: anse_energy {cur_energy} hit ANSE Maximum Pain")
                monotone_ok = False
            if prev_energy is None:
                if doc.get("delta_anse_energy") is not None:
                    violations.append(f"{label}: G_1 delta_anse_energy must be None")
            else:
                expected_delta = cur_energy - prev_energy
                claimed_delta = doc.get("delta_anse_energy")
                if (
                    isinstance(claimed_delta, bool)
                    or not isinstance(claimed_delta, (int, float))
                    or not math.isfinite(float(claimed_delta))
                    or abs(float(claimed_delta) - expected_delta) > 1e-9
                ):
                    violations.append(
                        f"{label}: delta_anse_energy {claimed_delta!r} != expected {expected_delta:.12g}"
                    )
                if require_monotone_anse and cur_energy >= prev_energy - min_imp:
                    violations.append(
                        f"{label}: ANSE monotonicity violated ({cur_energy:.6f} >= {prev_energy:.6f} - {min_imp:.6g})"
                    )
                    monotone_ok = False
            prev_energy = cur_energy

        if isinstance(doc.get("receipt_sha256"), str):
            expected_prev = str(doc["receipt_sha256"])
            last_hash = expected_prev

    valid = len(violations) == 0
    if raise_on_error and not valid:
        raise LedgerChainError(f"Ledger chain verification failed: {violations[0]}")

    return {
        "valid": valid,
        "entry_count": len(entries),
        "monotone_anse": monotone_ok and valid,
        "total_splits_verified": 4 * len(entries),
        "chain_head_sha256": last_hash,
        "violations": violations,
    }


class AutoWSPCampaignLedger:
    """Hash-chained generation ledger enforcing disjoint splits and strict ANSE monotone improvement."""

    def __init__(
        self,
        campaign_id: str,
        *,
        split_allocator: DisjointSplitAllocator | None = None,
        general_guard_rel: float = DEFAULT_GENERAL_GUARD_REL,
        min_energy_improvement: float = 0.0,
        baseline_anse_energy: float | None = None,
    ) -> None:
        self._campaign_id = _validate_nonempty_str(campaign_id, "campaign_id")
        if split_allocator is not None and not isinstance(split_allocator, DisjointSplitAllocator):
            raise TypeError("split_allocator must be a DisjointSplitAllocator or None")
        self._split_allocator = split_allocator if split_allocator is not None else DisjointSplitAllocator()
        self._general_guard_rel = _validate_positive_finite(general_guard_rel, "general_guard_rel")
        self._min_energy_improvement = _validate_nonneg_finite(min_energy_improvement, "min_energy_improvement")
        self._baseline_anse_energy = (
            _validate_finite_float(baseline_anse_energy, "baseline_anse_energy")
            if baseline_anse_energy is not None
            else None
        )
        self._entries: list[WSPGenerationReceipt] = []
        self._rejected_count: int = 0

    @property
    def campaign_id(self) -> str:
        return self._campaign_id

    @property
    def split_allocator(self) -> DisjointSplitAllocator:
        return self._split_allocator

    @property
    def general_guard_rel(self) -> float:
        return self._general_guard_rel

    @property
    def min_energy_improvement(self) -> float:
        return self._min_energy_improvement

    @property
    def baseline_anse_energy(self) -> float | None:
        return self._baseline_anse_energy

    @property
    def entries(self) -> tuple[WSPGenerationReceipt, ...]:
        return tuple(self._entries)

    @property
    def head_receipt_sha256(self) -> str:
        return GENESIS_RECEIPT_SHA256 if not self._entries else self._entries[-1].receipt_sha256

    @property
    def champion_receipt(self) -> WSPGenerationReceipt | None:
        return self._entries[-1] if self._entries else None

    @property
    def rejected_count(self) -> int:
        return self._rejected_count

    def _resolve_generation_quadruple(
        self,
        generation_id: str,
        split_quadruple: SplitQuadruple | None,
    ) -> SplitQuadruple:
        existing_map = {q.generation_id: q for q in self._split_allocator.generations}
        if split_quadruple is None:
            if generation_id in existing_map:
                return existing_map[generation_id]
            return self._split_allocator.allocate_generation(generation_id)

        if not isinstance(split_quadruple, SplitQuadruple):
            raise TypeError("split_quadruple must be a SplitQuadruple or None")
        if split_quadruple.generation_id != generation_id:
            raise SplitContaminationError(
                f"split_quadruple.generation_id ({split_quadruple.generation_id!r}) != {generation_id!r}"
            )
        if split_quadruple.split_stride != self._split_allocator.split_stride:
            raise SplitContaminationError(
                f"split_quadruple.split_stride ({split_quadruple.split_stride}) != allocator stride ({self._split_allocator.split_stride})"
            )
        expected = _build_split_quadruple(
            generation_id=split_quadruple.generation_id,
            generation_index=split_quadruple.generation_index,
            probe_skip=split_quadruple.probe_skip,
            selection_skip=split_quadruple.selection_skip,
            confirm_skip=split_quadruple.confirm_skip,
            virgin_vault_skip=split_quadruple.virgin_vault_skip,
            split_stride=split_quadruple.split_stride,
        )
        if split_quadruple.allocation_sha256 != expected.allocation_sha256:
            raise SplitContaminationError(f"Forged SplitQuadruple allocation_sha256 for {generation_id!r}")

        if generation_id in existing_map:
            reg = existing_map[generation_id]
            if reg.as_tuple() != split_quadruple.as_tuple() or reg.allocation_sha256 != split_quadruple.allocation_sha256:
                raise SplitContaminationError(
                    f"SplitQuadruple for {generation_id!r} mismatches allocator reservation"
                )
            return reg

        return self._split_allocator.allocate_generation(
            generation_id,
            probe_skip=split_quadruple.probe_skip,
            selection_skip=split_quadruple.selection_skip,
            confirm_skip=split_quadruple.confirm_skip,
            virgin_vault_skip=split_quadruple.virgin_vault_skip,
        )

    def record_generation(
        self,
        *,
        generation_id: str,
        arm_name: str,
        nll_mean: float,
        base_nll_mean: float,
        drift_ratio: float,
        checkpoint_sha256: str,
        parent_checkpoint_sha256: str,
        split_quadruple: SplitQuadruple | None = None,
        worst_domain_drift_ratio: float | None = None,
        anse_energy: float | None = None,
        consume_virgin_vault: bool = True,
        trigger_decision: DriftTriggerDecision | Mapping[str, Any] | None = None,
    ) -> WSPGenerationReceipt:
        """Validate, ANSE-ratchet, hash-chain, and commit one WSP generation receipt."""
        gid = _validate_nonempty_str(generation_id, "generation_id")
        arm = _validate_nonempty_str(arm_name, "arm_name")
        nll_f = _validate_positive_finite(nll_mean, "nll_mean")
        base_f = _validate_positive_finite(base_nll_mean, "base_nll_mean")
        drift_f = _validate_finite_float(drift_ratio, "drift_ratio")
        worst_f = (
            _validate_finite_float(worst_domain_drift_ratio, "worst_domain_drift_ratio")
            if worst_domain_drift_ratio is not None
            else drift_f
        )
        ckpt_sha = _validate_sha256_hex(checkpoint_sha256, "checkpoint_sha256")
        parent_sha = _validate_sha256_hex(parent_checkpoint_sha256, "parent_checkpoint_sha256")
        consume_vault = _validate_bool(consume_virgin_vault, "consume_virgin_vault")

        if any(e.generation_id == gid for e in self._entries):
            raise LedgerChainError(f"Generation {gid!r} is already recorded in ledger {self._campaign_id!r}")

        trigger_sha: str | None = None
        if trigger_decision is not None:
            if isinstance(trigger_decision, DriftTriggerDecision):
                trigger_sha = _validate_sha256_hex(trigger_decision.receipt_sha256, "trigger_decision.receipt_sha256")
            elif isinstance(trigger_decision, Mapping):
                trigger_sha = _validate_sha256_hex(trigger_decision.get("receipt_sha256"), "trigger_decision.receipt_sha256")
            else:
                raise TypeError("trigger_decision must be a DriftTriggerDecision, Mapping, or None")

        computed_energy, guard_ok, mean_diff = compute_wsp_anse_energy(
            nll_mean=nll_f,
            base_nll_mean=base_f,
            drift_ratio=drift_f,
            worst_domain_drift_ratio=worst_f,
            general_guard_rel=self._general_guard_rel,
        )
        eff_energy = _validate_finite_float(anse_energy, "anse_energy") if anse_energy is not None else computed_energy

        if not guard_ok or eff_energy >= ANSE_MAXIMUM_PAIN * 0.5:
            self._rejected_count += 1
            raise AnseMonotonicityViolationError(
                f"Generation {gid!r} failed domain guard (drift={drift_f:+.4%}, worst={worst_f:+.4%}, anse_energy={eff_energy})"
            )

        if not self._entries and self._baseline_anse_energy is not None:
            if eff_energy >= self._baseline_anse_energy - self._min_energy_improvement:
                self._rejected_count += 1
                raise AnseMonotonicityViolationError(
                    f"Generation {gid!r} failed baseline ANSE improvement ({eff_energy:.6f} >= {self._baseline_anse_energy:.6f})"
                )

        delta_energy: float | None = None
        if self._entries:
            prev_energy = self._entries[-1].anse_energy
            delta_energy = eff_energy - prev_energy
            if eff_energy >= prev_energy - self._min_energy_improvement:
                self._rejected_count += 1
                raise AnseMonotonicityViolationError(
                    f"Generation {gid!r} rejected: anse_energy={eff_energy:.6f} does not strictly improve "
                    f"prior generation energy={prev_energy:.6f} by > {self._min_energy_improvement:.6g}"
                )

        quad = self._resolve_generation_quadruple(gid, split_quadruple)

        vault_sealed = False
        seal_sha: str | None = None
        if consume_vault:
            if self._split_allocator.is_generation_sealed(gid):
                seal_doc = self._split_allocator.sealed_vault_receipts.get(quad.vault_id)
                seal_sha = str(seal_doc["receipt_sha256"]) if seal_doc is not None else None
                vault_sealed = True
            else:
                seal_doc = self._split_allocator.seal_virgin_vault(gid, outcome="COMMITTED_GENERATION")
                seal_sha = str(seal_doc["receipt_sha256"])
                vault_sealed = True

        gen_idx = len(self._entries) + 1
        prev_sha = self.head_receipt_sha256
        core = {
            "object": GENERATION_RECEIPT_OBJECT,
            "schema": SCHEMA_VERSION,
            "campaign_id": self._campaign_id,
            "generation_index": gen_idx,
            "generation_id": gid,
            "arm_name": arm,
            "probe_skip": quad.probe_skip,
            "selection_skip": quad.selection_skip,
            "confirm_skip": quad.confirm_skip,
            "virgin_vault_skip": quad.virgin_vault_skip,
            "split_stride": quad.split_stride,
            "allocation_sha256": quad.allocation_sha256,
            "nll_mean": nll_f,
            "base_nll_mean": base_f,
            "mean_diff_vs_base": mean_diff,
            "drift_ratio": drift_f,
            "worst_domain_drift_ratio": max(drift_f, worst_f),
            "guard_ok": True,
            "anse_energy": eff_energy,
            "delta_anse_energy": delta_energy,
            "checkpoint_sha256": ckpt_sha,
            "parent_checkpoint_sha256": parent_sha,
            "virgin_vault_sealed": vault_sealed,
            "vault_seal_receipt_sha256": seal_sha,
            "trigger_decision_sha256": trigger_sha,
            "prev_receipt_sha256": prev_sha,
        }
        receipt_sha = compute_payload_sha256(core)
        receipt = WSPGenerationReceipt(**core, receipt_sha256=receipt_sha)
        self._entries.append(receipt)
        return receipt

    def verify_chain(self, *, raise_on_error: bool = False) -> dict[str, Any]:
        """Verify cryptographic hash-chaining, split disjointness, and ANSE monotonicity."""
        return verify_ledger_chain(
            self._entries,
            campaign_id=self._campaign_id,
            min_energy_improvement=self._min_energy_improvement,
            split_stride=self._split_allocator.split_stride,
            frozen_skip_floor=self._split_allocator.frozen_skip_floor,
            require_monotone_anse=True,
            raise_on_error=raise_on_error,
        )

    def export_ledger(self) -> dict[str, Any]:
        """Serialize the complete hash-chained campaign ledger with snapshot SHA-256."""
        core = {
            "object": LEDGER_SNAPSHOT_OBJECT,
            "schema": SCHEMA_VERSION,
            "campaign_id": self._campaign_id,
            "general_guard_rel": self._general_guard_rel,
            "min_energy_improvement": self._min_energy_improvement,
            "baseline_anse_energy": self._baseline_anse_energy,
            "head_receipt_sha256": self.head_receipt_sha256,
            "rejected_count": self._rejected_count,
            "entries": [entry.to_dict() for entry in self._entries],
            "split_allocator_state": self._split_allocator.export_state(),
        }
        core["snapshot_sha256"] = compute_payload_sha256(core)
        return core

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AutoWSPCampaignLedger:
        """Restore and cryptographically verify an :class:`AutoWSPCampaignLedger` snapshot."""
        if not isinstance(payload, Mapping):
            raise TypeError("payload must be a mapping")
        if payload.get("object") != LEDGER_SNAPSHOT_OBJECT or payload.get("schema") != SCHEMA_VERSION:
            raise LedgerChainError("Invalid ledger snapshot object or schema tag")

        claimed = payload.get("snapshot_sha256")
        if not isinstance(claimed, str) or not _SHA256_HEX_RE.match(claimed):
            raise LedgerChainError("Missing or malformed snapshot_sha256")
        core = {k: v for k, v in payload.items() if k != "snapshot_sha256"}
        if compute_payload_sha256(core) != claimed:
            raise LedgerChainError("Ledger snapshot_sha256 tamper detected")

        try:
            allocator = DisjointSplitAllocator.from_state(payload["split_allocator_state"])
        except (KeyError, TypeError, ValueError) as exc:
            raise LedgerChainError(f"Invalid split_allocator_state in ledger snapshot: {exc}") from exc

        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, Sequence) or isinstance(raw_entries, (str, bytes)):
            raise LedgerChainError("Ledger entries must be a sequence")

        verify_ledger_chain(
            raw_entries,
            campaign_id=str(payload.get("campaign_id", "")),
            min_energy_improvement=payload.get("min_energy_improvement", 0.0),
            split_stride=allocator.split_stride,
            frozen_skip_floor=allocator.frozen_skip_floor,
            require_monotone_anse=True,
            raise_on_error=True,
        )

        ledger = cls(
            campaign_id=payload["campaign_id"],
            split_allocator=allocator,
            general_guard_rel=payload["general_guard_rel"],
            min_energy_improvement=payload["min_energy_improvement"],
            baseline_anse_energy=payload.get("baseline_anse_energy"),
        )
        ledger._rejected_count = _validate_nonneg_int(payload.get("rejected_count", 0), "rejected_count")
        for doc in raw_entries:
            ledger._entries.append(WSPGenerationReceipt(**dict(doc)))

        if ledger.head_receipt_sha256 != payload.get("head_receipt_sha256"):
            raise LedgerChainError("head_receipt_sha256 mismatch in ledger snapshot")
        return ledger
