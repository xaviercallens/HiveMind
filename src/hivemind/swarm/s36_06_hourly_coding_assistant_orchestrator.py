# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S36-06 — Hourly Improvement Plan & Roadmap Orchestrator using HiveMind as Coding & Unit-Test Assistant.

Executes the hourly self-improvement cycle across the Improvement Plan (`docs/IMPROVEMENT_PLAN_2026-10-31.md`)
and the Season 36 Roadmap (`HM-S36-01`..`HM-S36-06`), using ``hivemind/coder`` and ``hivemind/bee-stars``
via the Google Antigravity SDK (``LocalOpenAIAgentConfig.lightweight()``) for Python task & unit-test
co-generation, supervised by the ``GWAYA`` Zero-Stub AST gate (`E = 10^6` fail-closed).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from hivemind.swarm.s32_01_verlet_kepler442b_copilot import (
    DEFAULT_HIVEMIND_BASE_URL,
    AntigravityHiveMindCoPilot,
    ensure_antigravity_sdk_lightweight_compat,
)
from hivemind.swarm.s34_01_speculative_swarm_decoding import sample_residual_token_distribution
from hivemind.swarm.s35_02_gwaya_laya_failclosed_gate import (
    MAXIMUM_PAIN_ENERGY,
    inspect_python_ast_anti_stub,
)
from hivemind.swarm.s35_05_unified_xdev_ecosystem_bridge import build_xdev_portfolio_catalog
from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import ZKOPoIV5Int8TransitionVerifier
from hivemind.swarm.s36_02_e21_multikrum_vs_zk_crossover import run_e21_multikrum_vs_zk_crossover
from hivemind.swarm.s36_03_geocryptographic_neural_mesh import GeocryptographicNeuralMeshRouter
from hivemind.swarm.s36_04_fuzzy_subswarm_consensus import (
    LocalizedFuzzySubSwarmConsensus,
    SubSwarmVote,
)
from hivemind.swarm.s36_05_darl_cas_shard_leasing import DARLShardLeaseCoordinator


def resolve_hivemind_api_key(explicit_key: str | None = None) -> str | None:
    """Resolve the HiveMind API key from argument, environment, or user config without source literals."""
    if explicit_key and explicit_key.strip():
        return explicit_key.strip()
    for env_var in ("HIVEMIND_API_KEY", "OPENAI_API_KEY"):
        val = os.environ.get(env_var, "").strip()
        if val:
            return val
    cred_path = Path.home() / ".config" / "hivemind" / "credentials.json"
    if cred_path.exists():
        try:
            data = json.loads(cred_path.read_text(encoding="utf-8"))
            key = str(data.get("api_key") or "").strip()
            if key:
                return key
        except (OSError, ValueError):
            return None
    return None


@dataclass(frozen=True)
class CodingAssistantTaskReceipt:
    """Receipt from a single HiveMind coding assistant task + GWAYA AST verification."""

    task_id: str
    title: str
    model: str
    live_completion_id: str
    live_routed_model: str
    assistant_excerpt: str
    gwaya_zero_stub_passed: bool
    unit_verification_passed: bool
    anse_energy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class HourlyImprovementCycleReport:
    """Full report of an hourly roadmap & improvement-plan cycle."""

    timestamp_utc: float
    antigravity_sdk_compat: bool
    live_openrouter_called: bool
    tasks_verified: int
    tasks_total: int
    s36_01_zk_verify_ms: float
    s36_02_e21_energy_reduction_pct: float
    s36_03_geomesh_p95_reduction_pct: float
    s36_04_subswarm_agreement: float
    s36_05_darl_conservation_holds: bool
    ip21_residual_sampling_verified: bool
    ip67_no_hardcoded_xdev_verified: bool
    task_receipts: tuple[CodingAssistantTaskReceipt, ...]
    merkle_root_sha256: str
    all_gates_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "antigravity_sdk_compat": self.antigravity_sdk_compat,
            "live_openrouter_called": self.live_openrouter_called,
            "tasks_verified": self.tasks_verified,
            "tasks_total": self.tasks_total,
            "s36_01_zk_verify_ms": self.s36_01_zk_verify_ms,
            "s36_02_e21_energy_reduction_pct": self.s36_02_e21_energy_reduction_pct,
            "s36_03_geomesh_p95_reduction_pct": self.s36_03_geomesh_p95_reduction_pct,
            "s36_04_subswarm_agreement": self.s36_04_subswarm_agreement,
            "s36_05_darl_conservation_holds": self.s36_05_darl_conservation_holds,
            "ip21_residual_sampling_verified": self.ip21_residual_sampling_verified,
            "ip67_no_hardcoded_xdev_verified": self.ip67_no_hardcoded_xdev_verified,
            "task_receipts": [r.to_dict() for r in self.task_receipts],
            "merkle_root_sha256": self.merkle_root_sha256,
            "all_gates_passed": self.all_gates_passed,
        }


class HourlyHiveMindCodingAssistantOrchestrator:
    """Supervises hourly roadmap & improvement-plan tasks with `hivemind/coder` and `hivemind/bee-stars`."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_HIVEMIND_BASE_URL,
    ) -> None:
        self.api_key = resolve_hivemind_api_key(api_key)
        self.base_url = base_url
        self.sdk_compat = ensure_antigravity_sdk_lightweight_compat()

    def _query_coding_bee(
        self,
        *,
        model: str,
        prompt: str,
        call_live: bool,
    ) -> dict[str, Any]:
        if call_live and self.api_key:
            try:
                copilot = AntigravityHiveMindCoPilot(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=model,
                )
                res = copilot.query_openrouter_direct(prompt, model=model, max_tokens=200, timeout_s=30.0)
                return {
                    "live": True,
                    "completion_id": str(res.get("completion_id") or ""),
                    "routed_model": str(res.get("routed_model") or model),
                    "excerpt": str(res.get("content") or "")[:240],
                }
            except Exception as exc:
                return {
                    "live": False,
                    "completion_id": "offline-fallback",
                    "routed_model": model,
                    "excerpt": f"offline_fallback ({type(exc).__name__})",
                }
        return {
            "live": False,
            "completion_id": "deterministic-local-check",
            "routed_model": model,
            "excerpt": "deterministic local supervision check",
        }

    def run_hourly_improvement_cycle(self, *, call_live_openrouter: bool = False) -> HourlyImprovementCycleReport:
        """Execute and verify all Season 36 & Improvement-Plan checks with HiveMind coding assistant."""
        repo_root = Path(__file__).resolve().parents[3]
        swarm_dir = repo_root / "src" / "hivemind" / "swarm"

        # 1. HM-S36-01: ZK-OPoI v5 INT8 fixed-point verifier
        zk = ZKOPoIV5Int8TransitionVerifier()
        theta0 = [0.05 * ((i % 9) - 4) for i in range(128)]
        delta = [0.01 * ((i % 5) - 2) for i in range(128)]
        proof = zk.generate_proof(
            worker_id="bee-miner-01",
            workunit_id="wu-s36-01",
            shard_sha256="shard_digest_s36_01_valid",
            theta0=theta0,
            delta_weights=delta,
        )
        ver_ok = zk.verify_proof(
            proof,
            expected_shard_sha256="shard_digest_s36_01_valid",
            expected_theta0_sha256=proof.theta0_sha256,
        )
        ver_tampered = zk.verify_proof(
            proof,
            expected_shard_sha256="shard_digest_s36_01_tampered",
            expected_theta0_sha256=proof.theta0_sha256,
        )
        s36_01_ok = ver_ok.verified and ver_ok.within_5ms_sla and ver_ok.within_32kb_bound and (not ver_tampered.verified)

        # 2. HM-S36-02: E21 Multi-Krum vs ZK-OPoI v5 crossover
        e21_report = run_e21_multikrum_vs_zk_crossover()
        s36_02_ok = e21_report.gate_passed

        # 3. HM-S36-03: Geocryptographic P2P Neural Mesh
        mesh = GeocryptographicNeuralMeshRouter()
        mesh_eval = mesh.evaluate_multi_region_p95_reduction()
        s36_03_ok = bool(mesh_eval["meets_40pct_reduction_gate"])

        # 4. HM-S36-04: Localized 15-20 Node Fuzzy Sub-Swarm Consensus
        consensus = LocalizedFuzzySubSwarmConsensus()
        votes = [
            SubSwarmVote(
                node_id=f"peer-{i:02d}",
                answer_digest="ans-canonical-kepler442b" if i < 15 else "ans-byzantine",
                semantic_confidence=0.92 if i < 15 else 0.40,
                observer_reputation=0.95 if i < 15 else 0.25,
            )
            for i in range(18)
        ]
        cons_rep = consensus.evaluate_subswarm(votes)
        s36_04_ok = cons_rep.quorum_reached and cons_rep.within_50ms_sla and cons_rep.winning_digest == "ans-canonical-kepler442b"

        # 5. HM-S36-05: DARL Heartbeat-Backed CAS Shard Leasing
        darl = DARLShardLeaseCoordinator(["shard-01", "shard-02", "shard-03"])
        l1 = darl.grant_lease("w1", now_s=10.0)
        l2 = darl.grant_lease("w2", now_s=10.0)
        assert l1 is not None and l2 is not None
        darl.commit_lease(l1.lease_id, "w1", now_s=12.0)
        # w2 churns; lease expires and is reclaimed at t=25.0
        darl.reclaim_expired(now_s=25.0)
        l2_retry = darl.grant_lease("w3", now_s=26.0)
        assert l2_retry is not None
        darl.commit_lease(l2_retry.lease_id, "w3", now_s=28.0)
        l3 = darl.grant_lease("w1", now_s=28.5)
        assert l3 is not None
        darl.commit_lease(l3.lease_id, "w1", now_s=30.0)
        darl_snap = darl.snapshot()
        s36_05_ok = darl_snap.conservation_holds and darl_snap.committed == 3 and darl_snap.reclaimed == 1

        # 6. Improvement Plan IP-21 (residual sampling) & IP-67 (no hardcoded xdev path in s35_05)
        res_tok_low = sample_residual_token_distribution(
            target_probs={10: 0.10, 20: 0.70, 30: 0.20},
            draft_probs={10: 0.60, 20: 0.20, 30: 0.20},
            uniform_draw=0.35,
        )
        ip21_ok = res_tok_low == 20
        s35_05_source = (swarm_dir / "s35_05_unified_xdev_ecosystem_bridge.py").read_text(encoding="utf-8")
        cat = build_xdev_portfolio_catalog()
        ip67_ok = ("/home/" not in s35_05_source) and len(cat) == 19

        task_specs = [
            (
                "HM-S36-01",
                "Fixed-Point INT8 ZK-OPoI v5 Transition Prover & <5ms CPU Verifier",
                "hivemind/coder",
                swarm_dir / "s36_01_zk_opoi_v5_int8_verifier.py",
                s36_01_ok,
                "Generate pytest assertions for HM-S36-01 INT8 ZK-OPoI v5 receipt verification (<5ms CPU, <32KB proof, tampering rejection).",
            ),
            (
                "HM-S36-02",
                "E21 Crossover Benchmark: Multi-Krum + 10% Replay vs ZK-OPoI v5 (N=100..10000)",
                "hivemind/bee-stars",
                swarm_dir / "s36_02_e21_multikrum_vs_zk_crossover.py",
                s36_02_ok,
                "Generate pytest assertions for E21 Multi-Krum O(N^2 d) vs ZK-OPoI v5 O(1) validation energy crossover.",
            ),
            (
                "HM-S36-03",
                "Geocryptographic P2P Neural Mesh Routing Table (>=40% p95 RTT Reduction)",
                "hivemind/coder",
                swarm_dir / "s36_03_geocryptographic_neural_mesh.py",
                s36_03_ok,
                "Generate pytest assertions for Geocryptographic Vivaldi 3D routing & SHA-256 adapter verification.",
            ),
            (
                "HM-S36-04",
                "Localized 15-20 Node Fuzzy Sub-Swarm Consensus (<50ms & Collusion Bound)",
                "hivemind/bee-stars",
                swarm_dir / "s36_04_fuzzy_subswarm_consensus.py",
                s36_04_ok,
                "Generate pytest assertions for 15-20 node waggle-dance fuzzy consensus and hypergeometric collusion bound.",
            ),
            (
                "HM-S36-05",
                "DARL Heartbeat-Backed CAS Shard Leasing (committed = granted - reclaimed)",
                "hivemind/coder",
                swarm_dir / "s36_05_darl_cas_shard_leasing.py",
                s36_05_ok,
                "Generate pytest assertions for DARL shard lease TTL expiration, reclamation, and zero duplicate commits.",
            ),
            (
                "IP-21+IP-67",
                "Improvement Plan IP-21 Residual Speculative Sampling & IP-67 Path Hygiene",
                "hivemind/coder",
                swarm_dir / "s34_01_speculative_swarm_decoding.py",
                ip21_ok and ip67_ok,
                "Verify Leviathan residual distribution sampling max(0, p-q) on speculative rejection (IP-21).",
            ),
        ]

        receipts: list[CodingAssistantTaskReceipt] = []
        any_live = False
        for task_id, title, model, module_path, unit_ok, prompt in task_specs:
            src_code = module_path.read_text(encoding="utf-8")
            violations = inspect_python_ast_anti_stub(src_code)
            ast_clean = len(violations) == 0
            bee_res = self._query_coding_bee(
                model=model,
                prompt=prompt,
                call_live=call_live_openrouter,
            )
            if bee_res["live"]:
                any_live = True
            energy = 0.0 if (ast_clean and unit_ok) else MAXIMUM_PAIN_ENERGY
            receipts.append(
                CodingAssistantTaskReceipt(
                    task_id=task_id,
                    title=title,
                    model=model,
                    live_completion_id=bee_res["completion_id"],
                    live_routed_model=bee_res["routed_model"],
                    assistant_excerpt=bee_res["excerpt"],
                    gwaya_zero_stub_passed=ast_clean,
                    unit_verification_passed=unit_ok,
                    anse_energy=energy,
                )
            )

        verified_cnt = sum(
            1 for r in receipts if r.gwaya_zero_stub_passed and r.unit_verification_passed and r.anse_energy < 1_000_000.0
        )
        all_passed = verified_cnt == len(receipts) and self.sdk_compat
        merkle_payload = json.dumps([r.to_dict() for r in receipts], sort_keys=True).encode("utf-8")
        merkle_root = hashlib.sha256(merkle_payload).hexdigest()

        return HourlyImprovementCycleReport(
            timestamp_utc=round(time.time(), 3),
            antigravity_sdk_compat=self.sdk_compat,
            live_openrouter_called=any_live,
            tasks_verified=verified_cnt,
            tasks_total=len(receipts),
            s36_01_zk_verify_ms=ver_ok.verify_cpu_ms,
            s36_02_e21_energy_reduction_pct=e21_report.energy_reduction_at_10k_pct,
            s36_03_geomesh_p95_reduction_pct=float(mesh_eval["p95_reduction_pct"]),
            s36_04_subswarm_agreement=cons_rep.fuzzy_agreement_score,
            s36_05_darl_conservation_holds=darl_snap.conservation_holds,
            ip21_residual_sampling_verified=ip21_ok,
            ip67_no_hardcoded_xdev_verified=ip67_ok,
            task_receipts=tuple(receipts),
            merkle_root_sha256=merkle_root,
            all_gates_passed=all_passed,
        )
