# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S36-02 (Rupture 2 / HM-S30-02 / E21) — Crossover Benchmark: Multi-Krum + 10% Replay vs ZK-OPoI v5.

Pre-registered experiment ``E21`` comparing the validation cost of:
1. Legacy ``Multi-Krum O(N^2 * d) + 10% Deterministic GPU Replay``
2. Succinct ``ZK-OPoI v5 O(1) Per-Receipt CPU Verification (< 5 ms)``
across swarm sizes ``N in {100, 250, 500, 1000, 2500, 5000, 10000}``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Sequence

from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import ZKOPoIV5Int8TransitionVerifier

DEFAULT_SWARM_SIZES: tuple[int, ...] = (100, 250, 500, 1000, 2500, 5000, 10000)


@dataclass(frozen=True)
class E21CrossoverRow:
    """Validation cost comparison at a given swarm node count N."""

    swarm_nodes_n: int
    multikrum_pairwise_ms: float
    gpu_replay_10pct_ms: float
    legacy_total_ms: float
    zk_opoi_v5_total_ms: float
    speedup_ratio: float
    energy_reduction_pct: float
    zk_dominates: bool

    def __post_init__(self) -> None:
        if isinstance(self.swarm_nodes_n, bool) or not isinstance(self.swarm_nodes_n, int) or self.swarm_nodes_n < 2:
            raise ValueError("swarm_nodes_n must be an integer >= 2")
        for f_field in (
            "multikrum_pairwise_ms",
            "gpu_replay_10pct_ms",
            "legacy_total_ms",
            "zk_opoi_v5_total_ms",
            "speedup_ratio",
        ):
            val = getattr(self, f_field)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) < 0.0:
                raise ValueError(f"{f_field} must be a finite non-negative number")
        if (
            isinstance(self.energy_reduction_pct, bool)
            or not isinstance(self.energy_reduction_pct, (int, float))
            or not math.isfinite(float(self.energy_reduction_pct))
            or not (0.0 <= float(self.energy_reduction_pct) <= 100.0)
        ):
            raise ValueError("energy_reduction_pct must be a finite number in [0.0, 100.0]")
        if not isinstance(self.zk_dominates, bool):
            raise ValueError("zk_dominates must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class E21CrossoverBenchmarkReport:
    """Full E21 crossover benchmark report across N = 100 .. 10,000 nodes."""

    experiment_id: str
    estimation_mode: str
    measured_flop_ms_coeff: float
    measured_single_zk_verify_ms: float
    measured_proof_bytes: int
    crossover_node_count_n: int
    energy_reduction_at_10k_pct: float
    rows: tuple[E21CrossoverRow, ...]
    gate_passed: bool

    def __post_init__(self) -> None:
        for s_field in ("experiment_id", "estimation_mode"):
            val = getattr(self, s_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{s_field} must be a non-empty string")
        for p_field in ("measured_flop_ms_coeff", "measured_single_zk_verify_ms"):
            val = getattr(self, p_field)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) <= 0.0:
                raise ValueError(f"{p_field} must be a positive finite number")
        if isinstance(self.measured_proof_bytes, bool) or not isinstance(self.measured_proof_bytes, int) or self.measured_proof_bytes < 1:
            raise ValueError("measured_proof_bytes must be an integer >= 1")
        if isinstance(self.crossover_node_count_n, bool) or not isinstance(self.crossover_node_count_n, int) or self.crossover_node_count_n < 2:
            raise ValueError("crossover_node_count_n must be an integer >= 2")
        if (
            isinstance(self.energy_reduction_at_10k_pct, bool)
            or not isinstance(self.energy_reduction_at_10k_pct, (int, float))
            or not math.isfinite(float(self.energy_reduction_at_10k_pct))
            or not (0.0 <= float(self.energy_reduction_at_10k_pct) <= 100.0)
        ):
            raise ValueError("energy_reduction_at_10k_pct must be a finite number in [0.0, 100.0]")
        if not isinstance(self.rows, tuple) or len(self.rows) == 0 or not all(isinstance(r, E21CrossoverRow) for r in self.rows):
            raise ValueError("rows must be a non-empty tuple of E21CrossoverRow instances")
        if not isinstance(self.gate_passed, bool):
            raise ValueError("gate_passed must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "estimation_mode": self.estimation_mode,
            "measured_flop_ms_coeff": self.measured_flop_ms_coeff,
            "measured_single_zk_verify_ms": self.measured_single_zk_verify_ms,
            "measured_proof_bytes": self.measured_proof_bytes,
            "crossover_node_count_n": self.crossover_node_count_n,
            "energy_reduction_at_10k_pct": self.energy_reduction_at_10k_pct,
            "rows": [r.to_dict() for r in self.rows],
            "gate_passed": self.gate_passed,
        }


def _measure_pairwise_l2_flop_ms_coeff(k_calib: int = 8, d_calib: int = 256) -> float:
    """Measure pairwise squared-L2 distance cost per coordinate on the current CPU (IP-116 / IP-137)."""
    if isinstance(k_calib, bool) or not isinstance(k_calib, int) or k_calib < 2:
        raise ValueError("k_calib must be an integer >= 2")
    if isinstance(d_calib, bool) or not isinstance(d_calib, int) or d_calib < 1:
        raise ValueError("d_calib must be a positive integer")
    vecs = [[0.01 * ((i + j) % 11 - 5) for j in range(d_calib)] for i in range(k_calib)]
    t0 = time.perf_counter()
    acc = 0.0
    pairs = 0
    for i in range(k_calib):
        vi = vecs[i]
        for j in range(i + 1, k_calib):
            vj = vecs[j]
            acc += sum((a - b) * (a - b) for a, b in zip(vi, vj))
            pairs += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    if acc < 0.0:
        elapsed_ms += 0.0
    total_coord_ops = max(1, pairs * d_calib)
    return max(1e-10, elapsed_ms / float(total_coord_ops))


def run_e21_multikrum_vs_zk_crossover(
    swarm_sizes: Sequence[int] = DEFAULT_SWARM_SIZES,
    *,
    lora_dim_d: int = 1_376_256,
    gpu_step_replay_ms_per_wu: float = 420.0,
    spot_check_ratio: float = 0.10,
    single_zk_verify_ms_override: float | None = None,
    flop_ms_coeff_override: float | None = None,
    forged_proof_control: bool = False,
) -> E21CrossoverBenchmarkReport:
    """Execute the E21 validation cost crossover benchmark (model-based estimate calibrated by measured CPU timings)."""
    if (
        isinstance(swarm_sizes, (str, bytes))
        or not isinstance(swarm_sizes, Sequence)
        or not swarm_sizes
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 2 for n in swarm_sizes)
    ):
        raise ValueError("swarm_sizes must be a non-empty sequence of integers >= 2")
    if isinstance(lora_dim_d, bool) or not isinstance(lora_dim_d, int) or lora_dim_d < 1:
        raise ValueError("lora_dim_d must be a positive integer")
    if (
        isinstance(gpu_step_replay_ms_per_wu, bool)
        or not isinstance(gpu_step_replay_ms_per_wu, (int, float))
        or not math.isfinite(float(gpu_step_replay_ms_per_wu))
        or float(gpu_step_replay_ms_per_wu) <= 0.0
    ):
        raise ValueError("gpu_step_replay_ms_per_wu must be a finite positive number")
    if (
        isinstance(spot_check_ratio, bool)
        or not isinstance(spot_check_ratio, (int, float))
        or not (0.0 < float(spot_check_ratio) <= 1.0)
    ):
        raise ValueError("spot_check_ratio must be a finite number in (0, 1]")
    if single_zk_verify_ms_override is not None:
        if (
            isinstance(single_zk_verify_ms_override, bool)
            or not isinstance(single_zk_verify_ms_override, (int, float))
            or not math.isfinite(float(single_zk_verify_ms_override))
            or float(single_zk_verify_ms_override) <= 0.0
        ):
            raise ValueError("single_zk_verify_ms_override must be a finite positive number")
    if flop_ms_coeff_override is not None:
        if (
            isinstance(flop_ms_coeff_override, bool)
            or not isinstance(flop_ms_coeff_override, (int, float))
            or not math.isfinite(float(flop_ms_coeff_override))
            or float(flop_ms_coeff_override) <= 0.0
        ):
            raise ValueError("flop_ms_coeff_override must be a finite positive number")
    if not isinstance(forged_proof_control, bool):
        raise ValueError("forged_proof_control must be a boolean")
    verifier = ZKOPoIV5Int8TransitionVerifier()
    theta0 = [0.02 * ((i % 13) - 6) for i in range(256)]
    delta = [0.005 * ((i % 7) - 3) for i in range(256)]
    shard_hash = "e21_shard_sha256_canonical_0001"
    receipt = verifier.generate_proof(
        worker_id="node-e21-probe",
        workunit_id="wu-e21-0001",
        shard_sha256=shard_hash,
        theta0=theta0,
        delta_weights=delta,
    )
    expected_shard = "e21_forged_shard_digest" if forged_proof_control else shard_hash
    ver = verifier.verify_proof(
        receipt,
        expected_shard_sha256=expected_shard,
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    single_zk_ms = (
        float(single_zk_verify_ms_override)
        if single_zk_verify_ms_override is not None
        else max(0.02, ver.verify_cpu_ms)
    )
    within_5ms_sla = single_zk_ms < 5.0 and ver.within_5ms_sla

    flop_ms_coeff = (
        float(flop_ms_coeff_override)
        if flop_ms_coeff_override is not None
        else _measure_pairwise_l2_flop_ms_coeff()
    )
    rows: list[E21CrossoverRow] = []
    crossover_n = swarm_sizes[0] if swarm_sizes else 0
    found_crossover = False

    for n in swarm_sizes:
        pairs = n * (n - 1) // 2
        mk_ms = pairs * lora_dim_d * flop_ms_coeff
        replay_ms = n * spot_check_ratio * gpu_step_replay_ms_per_wu
        legacy_ms = mk_ms + replay_ms
        zk_ms = n * single_zk_ms
        speedup = legacy_ms / max(zk_ms, 1e-6)
        # GPU replay draws ~70W vs 15W single-core CPU check -> weighted joule reduction
        legacy_joules = (mk_ms * 0.035) + (replay_ms * 0.070)
        zk_joules = zk_ms * 0.015
        energy_red_pct = max(0.0, (1.0 - (zk_joules / max(legacy_joules, 1e-9))) * 100.0)
        dominates = zk_ms < legacy_ms
        if dominates and not found_crossover:
            crossover_n = n
            found_crossover = True

        rows.append(
            E21CrossoverRow(
                swarm_nodes_n=int(n),
                multikrum_pairwise_ms=round(mk_ms, 3),
                gpu_replay_10pct_ms=round(replay_ms, 3),
                legacy_total_ms=round(legacy_ms, 3),
                zk_opoi_v5_total_ms=round(zk_ms, 3),
                speedup_ratio=round(speedup, 2),
                energy_reduction_pct=round(energy_red_pct, 4),
                zk_dominates=dominates,
            )
        )

    energy_10k = rows[-1].energy_reduction_pct if rows else 0.0
    gate_passed = (
        ver.verified
        and within_5ms_sla
        and ver.within_32kb_bound
        and energy_10k >= 99.0
    )
    return E21CrossoverBenchmarkReport(
        experiment_id="E21-MULTIKRUM-VS-ZK-OPOI-V5",
        estimation_mode="model-based estimate",
        measured_flop_ms_coeff=round(flop_ms_coeff, 12),
        measured_single_zk_verify_ms=round(single_zk_ms, 4),
        measured_proof_bytes=ver.proof_bytes,
        crossover_node_count_n=crossover_n,
        energy_reduction_at_10k_pct=round(energy_10k, 4),
        rows=tuple(rows),
        gate_passed=gate_passed,
    )
