# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S36-02 (Rupture 2 / HM-S30-02 / E21) — Crossover Benchmark: Multi-Krum + 10% Replay vs ZK-OPoI v5.

Pre-registered experiment ``E21`` comparing the validation cost of:
1. Legacy ``Multi-Krum O(N^2 * d) + 10% Deterministic GPU Replay``
2. Succinct ``ZK-OPoI v5 O(1) Per-Receipt CPU Verification (< 5 ms)``
across swarm sizes ``N in {100, 250, 500, 1000, 2500, 5000, 10000}``.
"""

from __future__ import annotations

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

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class E21CrossoverBenchmarkReport:
    """Full E21 crossover benchmark report across N = 100 .. 10,000 nodes."""

    experiment_id: str
    measured_single_zk_verify_ms: float
    measured_proof_bytes: int
    crossover_node_count_n: int
    energy_reduction_at_10k_pct: float
    rows: tuple[E21CrossoverRow, ...]
    gate_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "experiment_id": self.experiment_id,
            "measured_single_zk_verify_ms": self.measured_single_zk_verify_ms,
            "measured_proof_bytes": self.measured_proof_bytes,
            "crossover_node_count_n": self.crossover_node_count_n,
            "energy_reduction_at_10k_pct": self.energy_reduction_at_10k_pct,
            "rows": [r.to_dict() for r in self.rows],
            "gate_passed": self.gate_passed,
        }


def run_e21_multikrum_vs_zk_crossover(
    swarm_sizes: Sequence[int] = DEFAULT_SWARM_SIZES,
    *,
    lora_dim_d: int = 1_376_256,
    gpu_step_replay_ms_per_wu: float = 420.0,
    spot_check_ratio: float = 0.10,
) -> E21CrossoverBenchmarkReport:
    """Execute the E21 validation cost crossover benchmark."""
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
    ver = verifier.verify_proof(
        receipt,
        expected_shard_sha256=shard_hash,
        expected_theta0_sha256=receipt.theta0_sha256,
    )
    single_zk_ms = max(0.02, ver.verify_cpu_ms)

    # Pairwise L2 distance cost in Multi-Krum: N*(N-1)/2 * d * 1.2e-9 ms/flop
    flop_ms_coeff = 1.2e-9
    rows: list[E21CrossoverRow] = []
    crossover_n = swarm_sizes[0]
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
        and ver.within_5ms_sla
        and ver.within_32kb_bound
        and energy_10k >= 99.0
    )
    return E21CrossoverBenchmarkReport(
        experiment_id="E21-MULTIKRUM-VS-ZK-OPOI-V5",
        measured_single_zk_verify_ms=round(single_zk_ms, 4),
        measured_proof_bytes=ver.proof_bytes,
        crossover_node_count_n=crossover_n,
        energy_reduction_at_10k_pct=round(energy_10k, 4),
        rows=tuple(rows),
        gate_passed=gate_passed,
    )
