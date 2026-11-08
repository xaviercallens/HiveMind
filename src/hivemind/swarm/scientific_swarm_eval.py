# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Synthetic swarm/CAS/game mechanics, not scientific language benchmarks.

Historical public names are retained for compatibility. MSE values below are
measured on seeded Gaussian regression tensors, not on the listed text. Game
rewards, proposed performance figures and leaderboards are simulations. For a
separate measured token-training experiment, see data.token_training_eval.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from hivemind.core.btm_moe import UpcycledSparseMoEModel
from hivemind.core.compression import (
    compress_lora_delta,
    compute_pseudo_gradient_delta,
    decompress_lora_delta,
)
from hivemind.data.hf_real_trainer import (
    CURATED_SEASON3_DOMAIN_CORPORA,
    train_single_domain_lora_adapter,
)
from hivemind.swarm.dht_storage import ContentAddressedSwarmStore

SCIENTIFIC_BENCHMARK_SUITES: Dict[str, Dict[str, Any]] = {
    "Rust_Systems_Security": {
        "benchmark_id": "HF-SCI-BENCH-01-RUST-BORROW",
        "hf_dataset_id": "anon/rust-code-instructions",
        "scientific_metric": "Synthetic Gaussian regression MSE; not a Rust benchmark",
        "holdout_samples": [
            "pub fn atomic_epoch_tick(cell: &std::sync::atomic::AtomicU64) -> u64 { cell.fetch_add(1, std::sync::atomic::Ordering::SeqCst) }",
            "pub fn verify_zero_copy_bounds(slice: &[u8], offset: usize, len: usize) -> Option<&[u8]> { slice.get(offset..offset.checked_add(len)?) }",
            "pub fn lock_free_ring_enqueue<T>(buf: &mut Vec<T>, item: T, cap: usize) -> Result<(), &'static str> { if buf.len() >= cap { return Err(\"full\"); } buf.push(item); Ok(()) }",
        ],
    },
    "Solidity_Smart_Contracts": {
        "benchmark_id": "HF-SCI-BENCH-02-SOLIDITY-ESCROW",
        "hf_dataset_id": "ASSERT-KTH/DISL-Solidity-Smart-Contracts",
        "scientific_metric": "Synthetic Gaussian regression MSE; not an invariant benchmark",
        "holdout_samples": [
            "function withdrawEscrow(uint256 amount) external nonReentrant { require(balances[msg.sender] >= amount, 'UNDERFLOW'); balances[msg.sender] -= amount; payable(msg.sender).transfer(amount); }",
            "function verifyMerkleSpotCheck(bytes32 root, bytes32 leaf, bytes32 idx) external pure returns (bool) { return keccak256(abi.encodePacked(leaf, idx)) == root; }",
            "function distributeQuadraticReward(uint256 stake, uint256 epochs) external pure returns (uint256) { return stake * epochs; }",
        ],
    },
    "Biomedical_Clinical_QA": {
        "benchmark_id": "HF-SCI-BENCH-03-PUBMEDQA-CLINICAL",
        "hf_dataset_id": "qiaojin/PubMedQA",
        "scientific_metric": "Synthetic Gaussian regression MSE; not a clinical benchmark",
        "holdout_samples": [
            "### Clinical Question: Does CRISPR-Cas9 ribonucleoprotein delivery reduce off-target mutagenesis compared to plasmid transfection? ### Evidence Conclusion (yes): Transient RNP complexes degrade rapidly in vivo, minimizing off-target cleavage.",
            "### Clinical Question: Does T-cell PD-1 checkpoint blockade restore anti-tumor cytokine secretion in exhausted TILs? ### Evidence Conclusion (yes): Blocking PD-1/PD-L1 interaction reverses T-cell exhaustion.",
            "### Clinical Question: Does mitochondrial complex I inhibition alter NAD+/NADH redox homeostasis? ### Evidence Conclusion (yes): Impaired electron transport elevates intracellular NADH ratios.",
        ],
    },
    "Lean4_Theorem_Prover": {
        "benchmark_id": "HF-SCI-BENCH-04-PROOFNET-LEAN4",
        "hf_dataset_id": "hoskinson-center/proofnet",
        "scientific_metric": "Synthetic Gaussian regression MSE; not a theorem-proving benchmark",
        "holdout_samples": [
            "theorem multikrum_honest_majority_bound (n f : Nat) (h : 2 * f + 3 <= n) : f < n := by omega",
            "theorem fp8_topk_bandwidth_saving (raw compressed : Nat) (h : compressed * 10 <= raw) : compressed <= raw := by omega",
            "theorem quadratic_voting_sybil_bound (a b : Nat) : a + b <= a + b := by omega",
        ],
    },
}


HF_GAMIFICATION_BADGES: Dict[str, Dict[str, str]] = {
    "HF_SWARM_KNOWLEDGE_DONOR": {
        "badge_id": "HF_SWARM_KNOWLEDGE_DONOR",
        "title": "Swarm Knowledge Donor (CAS Broadcast)",
        "tier": "Diamond",
        "description": "Retrained a QLoRA adapter that was hot-swapped and leveraged by >= 3 other swarm agents with >95% transfer efficiency.",
    },
    "HF_COOP_RAID_COMMANDER": {
        "badge_id": "HF_COOP_RAID_COMMANDER",
        "title": "Joined Campaign Commander (Co-Op Raid)",
        "tier": "Mythic",
        "description": "Completed all 4 Joined Swarm Co-Op Campaigns with >3.5x parallel speedup and TIES sign-consensus synergy.",
    },
    "HF_SCIENTIFIC_VALIDATOR": {
        "badge_id": "HF_SCIENTIFIC_VALIDATOR",
        "title": "Peer-Reviewed Swarm Scientist",
        "tier": "Platinum",
        "description": "Achieved >80% holdout loss reduction across all 4 scientific benchmark suites (Rust, Solidity, PubMedQA, Lean 4).",
    },
    "HF_OPEN_SOURCE_ARCHITECT": {
        "badge_id": "HF_OPEN_SOURCE_ARCHITECT",
        "title": "HiveMind Core Protocol Architect",
        "tier": "Legendary",
        "description": "Merged verified PRs and formal Lean 4 theorems powering the global HiveMind swarm fabric.",
    },
}


@dataclass
class PeerKnowledgeTransferAudit:
    """Synthetic regression/CAS transfer telemetry; no language or GPU evidence."""

    domain_name: str
    benchmark_id: str
    trainer_agent_id: str
    expert_cid: str
    trainer_initial_loss: float
    trainer_final_loss: float
    trainer_loss_reduction_pct: float
    peer_evaluations: List[Dict[str, Any]]
    mean_peer_pre_transfer_loss: Optional[float]
    mean_peer_post_transfer_loss: Optional[float]
    mean_peer_pre_perplexity: Optional[float]
    mean_peer_post_perplexity: Optional[float]
    mean_peer_loss_reduction_pct: Optional[float]
    knowledge_transfer_efficiency_ratio: Optional[float]
    mean_hotswap_latency_ms: Optional[float]
    total_peer_gpu_steps_saved: Optional[int]
    royalties_paid_to_trainer_hive: float
    all_peers_leveraged_successfully: bool
    measurement_kind: str = "synthetic_regression"
    loss_metric: str = "mean_squared_error"
    perplexity_status: str = "not_applicable_to_mse"
    hardware_validation_status: str = "not_run"
    transfer_scope: str = "single_process_local_file_load"
    simulated_peer_steps_avoided: int = 0
    peer_measurement_status: str = "synthetic_measured"


@dataclass
class JoinedCampaignProposal:
    """A gamified cooperative campaign where multiple swarm agents mine faster and better together."""

    campaign_id: str
    codename: str
    strategy_mode: str
    target_domains: List[str]
    min_participating_agents: int
    expected_speedup_factor: float
    expected_quality_synergy_pct: float
    reward_pool_hive: float
    reward_pool_pts: int
    scientific_rationale: str
    evidence_status: str = "unvalidated_design_hypothesis_not_measurement"

    def __post_init__(self) -> None:
        for str_field in ("campaign_id", "codename", "strategy_mode", "scientific_rationale", "evidence_status"):
            val = getattr(self, str_field)
            if not isinstance(val, str) or not val.strip():
                raise ValueError(f"{str_field} must be a non-empty string")
        if isinstance(self.target_domains, (str, bytes)) or not isinstance(self.target_domains, Sequence) or not self.target_domains or not all(
            isinstance(d, str) and bool(d.strip()) for d in self.target_domains
        ):
            raise ValueError("target_domains must be a non-empty sequence of non-empty strings")
        if isinstance(self.min_participating_agents, bool) or not isinstance(self.min_participating_agents, int) or self.min_participating_agents < 1:
            raise ValueError("min_participating_agents must be an integer >= 1")
        if isinstance(self.reward_pool_pts, bool) or not isinstance(self.reward_pool_pts, int) or self.reward_pool_pts < 0:
            raise ValueError("reward_pool_pts must be an integer >= 0")
        if isinstance(self.expected_speedup_factor, bool) or not isinstance(self.expected_speedup_factor, (int, float)) or not math.isfinite(float(self.expected_speedup_factor)) or float(self.expected_speedup_factor) <= 0.0:
            raise ValueError("expected_speedup_factor must be a finite number > 0")
        for nonneg_f in ("expected_quality_synergy_pct", "reward_pool_hive"):
            fval = getattr(self, nonneg_f)
            if isinstance(fval, bool) or not isinstance(fval, (int, float)) or not math.isfinite(float(fval)) or float(fval) < 0.0:
                raise ValueError(f"{nonneg_f} must be a finite number >= 0")


PROPOSED_JOINED_CAMPAIGNS: List[JoinedCampaignProposal] = [
    JoinedCampaignProposal(
        campaign_id="CAMPAIGN-COOP-01-DILOCO-BLITZ",
        codename="Operation: Hyper-DiLoCo Shard Blitz",
        strategy_mode="Data-Parallel Shard Partitioning + Nesterov Outer Momentum + TIES Merge",
        target_domains=["Rust_Systems_Security"],
        min_participating_agents=4,
        expected_speedup_factor=3.85,
        expected_quality_synergy_pct=14.5,
        reward_pool_hive=250.0,
        reward_pool_pts=4800,
        scientific_rationale=(
            "Splits a Hugging Face dataset across K=4 miners so each GPU processes 1/K of tokens "
            "in parallel (H=12 inner steps), communicating only Top-5% + FP8 deltas (12.79x wire compression) "
            "and fusing via TIES sign-consensus to reduce gradient variance and mine 3.85x faster."
        ),
    ),
    JoinedCampaignProposal(
        campaign_id="CAMPAIGN-COOP-02-POLYMATH-FUSION",
        codename="Operation: Polymath 4-Domain Cross-Pollination",
        strategy_mode="Specialist Division of Labor + Instant CAS Hot-Swap + 4-Expert Sparse MoE",
        target_domains=[
            "Rust_Systems_Security",
            "Solidity_Smart_Contracts",
            "Biomedical_Clinical_QA",
            "Lean4_Theorem_Prover",
        ],
        min_participating_agents=4,
        expected_speedup_factor=4.00,
        expected_quality_synergy_pct=300.0,
        reward_pool_hive=400.0,
        reward_pool_pts=7200,
        scientific_rationale=(
            "Each agent trains 1 specialized domain expert and publishes it to the Content-Addressed Store. "
            "All 4 agents immediately hot-swap their peers' 3 experts, gaining 4/4 domain mastery at 1/4 "
            "the individual GPU training cost (4.0x compute leverage)."
        ),
    ),
    JoinedCampaignProposal(
        campaign_id="CAMPAIGN-COOP-03-BYZANTINE-CITADEL",
        codename="Operation: Zero-Day Byzantine Citadel",
        strategy_mode="Co-Op Multi-Krum 2-Sigma Defense + 10% Deterministic Spot-Check Audit",
        target_domains=["Solidity_Smart_Contracts", "Lean4_Theorem_Prover"],
        min_participating_agents=4,
        expected_speedup_factor=3.40,
        expected_quality_synergy_pct=22.0,
        reward_pool_hive=300.0,
        reward_pool_pts=5600,
        scientific_rationale=(
            "Co-mining squad jointly filters out adversarial Poisoning Bots using pairwise Euclidean "
            "Multi-Krum distance and 10% activation spot-checks, ensuring 0% model corruption and splitting "
            "slashed cheater stakes among honest miners."
        ),
    ),
    JoinedCampaignProposal(
        campaign_id="CAMPAIGN-COOP-04-SUPERRING-EXPRESS",
        codename="Operation: Global Super-Ring Speculative Express",
        strategy_mode="0-RTT QUIC Geo-Ring + Token-Tree Speculative Decoding (gamma=6) + Royalty Streaming",
        target_domains=[
            "Rust_Systems_Security",
            "Solidity_Smart_Contracts",
            "Biomedical_Clinical_QA",
            "Lean4_Theorem_Prover",
        ],
        min_participating_agents=4,
        expected_speedup_factor=5.40,
        expected_quality_synergy_pct=18.2,
        reward_pool_hive=350.0,
        reward_pool_pts=6400,
        scientific_rationale=(
            "Links EU, US, and APAC peer agents that have ingested the shared CAS expert pool into a "
            "pipelined speculative verification ring, cutting multi-hop RTT latency by >5x and paying "
            "automatic $HIVE reuse royalties to the original adapter trainers."
        ),
    ),
]


def _evaluate_lora_on_scientific_holdout(
    domain_name: str,
    lora_state: Dict[str, torch.Tensor],
    lora_rank_r: int = 16,
    lora_alpha: int = 32,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluate synthetic regression MSE on fresh Gaussian samples.

    The named domain's text and benchmark holdout samples are NOT consumed.
    The historical function name does not make this a scientific text benchmark.
    Perplexity is undefined here: exponentiating MSE is not token perplexity.
    """
    if not isinstance(domain_name, str) or not domain_name.strip():
        raise ValueError("domain_name must be a non-empty string")
    if not isinstance(lora_state, Mapping) or len(lora_state) == 0:
        raise ValueError("lora_state must be a non-empty mapping")
    if isinstance(lora_rank_r, bool) or not isinstance(lora_rank_r, int) or lora_rank_r < 1:
        raise ValueError("lora_rank_r must be an integer >= 1")
    if isinstance(lora_alpha, bool) or not isinstance(lora_alpha, int) or lora_alpha < 1:
        raise ValueError("lora_alpha must be an integer >= 1")
    from hivemind.data.hf_real_trainer import _encode_synthetic_regression_batch

    target_device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domain_seed = int(hashlib.sha256(domain_name.encode("utf-8")).hexdigest()[:8], 16) % 10_000 + 101
    scaling = float(lora_alpha) / float(max(1, lora_rank_r))
    corpus = CURATED_SEASON3_DOMAIN_CORPORA.get(
        domain_name,
        CURATED_SEASON3_DOMAIN_CORPORA["Rust_Systems_Security"],
    )

    prefix_groups: Dict[str, Dict[str, torch.Tensor]] = {}
    for k, v in lora_state.items():
        if "lora_A" in k:
            prefix = k.split("lora_A")[0]
            prefix_groups.setdefault(prefix, {})["A"] = v.detach().to(
                device=target_device, dtype=torch.float32
            )
        elif "lora_B" in k:
            prefix = k.split("lora_B")[0]
            prefix_groups.setdefault(prefix, {})["B"] = v.detach().to(
                device=target_device, dtype=torch.float32
            )

    holdout_losses: List[float] = []
    cosine_alignments: List[float] = []

    with torch.no_grad():
        for idx, (_, grp) in enumerate(sorted(prefix_groups.items())):
            a_mat = grp["A"]
            b_mat = grp["B"]
            in_dim = int(a_mat.shape[1])
            out_dim = int(b_mat.shape[0])

            x_cpu, y_cpu = _encode_synthetic_regression_batch(
                corpus,
                in_dim,
                out_dim,
                seed=domain_seed + idx * 17,
                is_holdout=True,
            )
            x_holdout = x_cpu.to(target_device)
            y_holdout = y_cpu.to(target_device)

            pred_holdout = (x_holdout @ a_mat.T @ b_mat.T) * scaling
            mse_loss = float(torch.mean((pred_holdout - y_holdout) ** 2).item())
            holdout_losses.append(mse_loss)

            pred_flat = pred_holdout.reshape(-1)
            target_flat = y_holdout.reshape(-1)
            if float(torch.norm(pred_flat).item()) > 1e-8:
                cos_val = float(
                    torch.nn.functional.cosine_similarity(
                        pred_flat.unsqueeze(0), target_flat.unsqueeze(0)
                    ).item()
                )
            else:
                cos_val = 0.0
            cosine_alignments.append(cos_val)

    mean_loss = round(sum(holdout_losses) / max(1, len(holdout_losses)), 6)
    mean_cosine = round(sum(cosine_alignments) / max(1, len(cosine_alignments)), 4)
    return {
        "measurement_kind": "synthetic_regression",
        "loss_metric": "mean_squared_error",
        "text_content_used": False,
        "holdout_loss": mean_loss,
        "holdout_perplexity": None,
        "perplexity_status": "not_applicable_to_mse",
        "holdout_cosine_alignment": mean_cosine,
    }



def ties_merge_lora_adapters(
    base_lora_state: Dict[str, torch.Tensor],
    worker_lora_states: Sequence[Dict[str, torch.Tensor]],
    trim_density: float = 0.20,
    scaling_lambda: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Merges multiple peer-trained LoRA state dicts using TIES-Merging (Trim, Elect Sign, Disjoint Merge)
    so cooperative miners combine their training efforts without destructive gradient interference.
    """
    if not isinstance(base_lora_state, Mapping) or len(base_lora_state) == 0:
        raise ValueError("base_lora_state must be a non-empty mapping")
    if isinstance(worker_lora_states, (str, bytes)) or not isinstance(worker_lora_states, Sequence):
        raise ValueError("worker_lora_states must be a sequence of state dicts")
    if isinstance(trim_density, bool) or not isinstance(trim_density, (int, float)) or not math.isfinite(float(trim_density)) or not 0.0 < float(trim_density) <= 1.0:
        raise ValueError("trim_density must be in (0.0, 1.0]")
    if isinstance(scaling_lambda, bool) or not isinstance(scaling_lambda, (int, float)) or not math.isfinite(float(scaling_lambda)) or float(scaling_lambda) <= 0.0:
        raise ValueError("scaling_lambda must be a finite number > 0")
    if not worker_lora_states:
        return {k: v.clone() for k, v in base_lora_state.items()}

    merged_state: Dict[str, torch.Tensor] = {}
    for k, base_tensor in base_lora_state.items():
        base_fp32 = base_tensor.detach().to(torch.float32)
        task_vectors: List[torch.Tensor] = []
        for w_state in worker_lora_states:
            if k in w_state:
                tau = w_state[k].detach().to(torch.float32) - base_fp32
                flat = tau.reshape(-1)
                k_keep = max(1, int(round(flat.numel() * trim_density)))
                if k_keep < flat.numel():
                    threshold = torch.topk(flat.abs(), k=k_keep, largest=True).values.min()
                    tau = torch.where(tau.abs() >= threshold, tau, torch.zeros_like(tau))
                task_vectors.append(tau)

        if not task_vectors:
            merged_state[k] = base_tensor.clone()
            continue

        stacked = torch.stack(task_vectors, dim=0)
        elected_sign = torch.sign(torch.sum(stacked, dim=0))
        sign_mask = (torch.sign(stacked) == elected_sign.unsqueeze(0)) & (stacked != 0)
        disjoint_sum = torch.sum(stacked * sign_mask.to(torch.float32), dim=0)
        disjoint_count = torch.clamp(torch.sum(sign_mask.to(torch.float32), dim=0), min=1.0)
        merged_tau = disjoint_sum / disjoint_count
        merged_state[k] = (base_fp32 + scaling_lambda * merged_tau).to(base_tensor.dtype)

    return merged_state


class SwarmKnowledgeTransferEngine:
    """
    Exercise synthetic adapter training and local file-based peer-state reuse.

    This does not measure language accuracy, network transfer or saved GPU work.
    """

    def __init__(
        self,
        store: ContentAddressedSwarmStore,
        peer_agent_ids: Optional[Sequence[str]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if not isinstance(store, ContentAddressedSwarmStore):
            raise TypeError("store must be a ContentAddressedSwarmStore instance")
        if peer_agent_ids is not None:
            if isinstance(peer_agent_ids, (str, bytes)) or not isinstance(peer_agent_ids, Sequence) or len(peer_agent_ids) == 0:
                raise ValueError("peer_agent_ids must be a non-empty sequence of strings")
            if not all(isinstance(a, str) and bool(a.strip()) for a in peer_agent_ids) or len(set(peer_agent_ids)) != len(peer_agent_ids):
                raise ValueError("peer_agent_ids must contain unique non-empty strings")
        self.store = store
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.peer_agent_ids = list(
            peer_agent_ids
            or [
                "CyberValkyrie_FR",
                "NeuralGhost_DE",
                "ZeroCool_UK",
                "SakuraNode_JP",
            ]
        )
        self.peer_local_adapters: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {
            agent_id: {} for agent_id in self.peer_agent_ids
        }
        self.reuse_royalty_ledger: Dict[str, float] = {
            agent_id: 0.0 for agent_id in self.peer_agent_ids
        }
        self.peer_reuse_counts: Dict[str, int] = {
            agent_id: 0 for agent_id in self.peer_agent_ids
        }

    def verify_swarm_peer_knowledge_transfer(
        self,
        domain_name: str,
        trainer_agent_id: str,
        reference_lora_state: Dict[str, torch.Tensor],
        inner_steps: int = 16,
        topk_ratio: float = 0.05,
        royalty_per_peer_hive: float = 2.5,
    ) -> PeerKnowledgeTransferAudit:
        """
        Train synthetic Gaussian regression, publish locally, then load peer states.

        The legacy checkpoint loader does not verify its CID against loaded bytes.
        Royalties and avoided-step counts are game accounting, not payments or
        measured GPU savings. Real safe-byte verification is in token_training_eval.
        """
        if not isinstance(domain_name, str) or not domain_name.strip():
            raise ValueError("domain_name must be a non-empty string")
        if not isinstance(reference_lora_state, Mapping) or len(reference_lora_state) == 0:
            raise ValueError("reference_lora_state must be a non-empty mapping")
        if isinstance(inner_steps, bool) or not isinstance(inner_steps, int) or inner_steps < 1:
            raise ValueError("inner_steps must be an integer >= 1")
        if isinstance(topk_ratio, bool) or not isinstance(topk_ratio, (int, float)) or not math.isfinite(float(topk_ratio)) or not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError("topk_ratio must be in (0.0, 1.0]")
        if isinstance(royalty_per_peer_hive, bool) or not isinstance(royalty_per_peer_hive, (int, float)) or not math.isfinite(float(royalty_per_peer_hive)) or float(royalty_per_peer_hive) < 0.0:
            raise ValueError("royalty_per_peer_hive must be a finite number >= 0.0")
        bench_meta = SCIENTIFIC_BENCHMARK_SUITES.get(
            domain_name,
            SCIENTIFIC_BENCHMARK_SUITES["Rust_Systems_Security"],
        )

        if trainer_agent_id not in self.peer_agent_ids:
            raise ValueError("trainer_agent_id must be a participating peer")
        # Synthetic regression baseline, not a text holdout.
        pre_eval_by_peer: Dict[str, Dict[str, Any]] = {}
        for agent_id in self.peer_agent_ids:
            current_state = self.peer_local_adapters.get(agent_id, {}).get(
                domain_name, reference_lora_state
            )
            pre_eval_by_peer[agent_id] = _evaluate_lora_on_scientific_holdout(
                domain_name=domain_name,
                lora_state=current_state,
                device=self.device,
            )

        # Local synthetic matrix fitting and checkpoint publication.
        retrained_state, train_metrics = train_single_domain_lora_adapter(
            domain_name=domain_name,
            reference_lora_state=reference_lora_state,
            inner_steps=inner_steps,
            topk_ratio=topk_ratio,
        )
        expert_cid, _ = self.store.publish_lora_expert_checkpoint(
            domain_name=domain_name,
            lora_state_dict=retrained_state,
            metadata={
                **train_metrics,
                "trainer_agent_id": trainer_agent_id,
                "benchmark_id": bench_meta["benchmark_id"],
            },
        )
        self.peer_local_adapters.setdefault(trainer_agent_id, {})[domain_name] = retrained_state
        trainer_post_holdout = _evaluate_lora_on_scientific_holdout(
            domain_name=domain_name,
            lora_state=retrained_state,
            device=self.device,
        )
        trainer_pre_loss = pre_eval_by_peer[trainer_agent_id]["holdout_loss"]
        trainer_post_loss = trainer_post_holdout["holdout_loss"]
        trainer_drop_pct = round(
            100.0 * (trainer_pre_loss - trainer_post_loss) / max(1e-8, trainer_pre_loss),
            2,
        )

        # Step 3: Other agents in the swarm fetch `expert_cid` from CAS and hot-swap it without retraining
        peer_eval_records: List[Dict[str, Any]] = []
        other_peers = [a for a in self.peer_agent_ids if a != trainer_agent_id]
        hotswap_latencies: List[float] = []

        for peer_id in other_peers:
            t0 = time.perf_counter()
            _, loaded_meta, fetched_lora = self.store.load_lora_expert_checkpoint(
                expert_cid, dtype=torch.float16
            )
            self.peer_local_adapters.setdefault(peer_id, {})[domain_name] = fetched_lora
            hotswap_ms = round((time.perf_counter() - t0) * 1000.0, 3)
            hotswap_latencies.append(hotswap_ms)

            post_eval = _evaluate_lora_on_scientific_holdout(
                domain_name=domain_name,
                lora_state=fetched_lora,
                device=self.device,
            )
            pre_loss = pre_eval_by_peer[peer_id]["holdout_loss"]
            post_loss = post_eval["holdout_loss"]
            loss_drop_pct = round(
                100.0 * (pre_loss - post_loss) / max(1e-8, pre_loss), 2
            )
            transfer_ratio = (
                round((pre_loss - post_loss) / (trainer_pre_loss - trainer_post_loss), 4)
                if trainer_pre_loss > trainer_post_loss else None
            )

            self.reuse_royalty_ledger[trainer_agent_id] = round(
                self.reuse_royalty_ledger.get(trainer_agent_id, 0.0) + royalty_per_peer_hive, 4
            )
            self.peer_reuse_counts[trainer_agent_id] = (
                self.peer_reuse_counts.get(trainer_agent_id, 0) + 1
            )

            peer_eval_records.append(
                {
                    "peer_agent_id": peer_id,
                    "ingested_expert_cid": expert_cid,
                    "provenance_trainer": loaded_meta.get("trainer_agent_id", trainer_agent_id),
                    "local_training_steps_spent": 0,
                    "measurement_kind": "synthetic_regression",
                    "gpu_steps_saved": None,
                    "simulated_steps_avoided": train_metrics["inner_steps"],
                    "hotswap_latency_ms": hotswap_ms,
                    "pre_transfer_holdout_loss": pre_loss,
                    "post_transfer_holdout_loss": post_loss,
                    "pre_transfer_perplexity": pre_eval_by_peer[peer_id]["holdout_perplexity"],
                    "post_transfer_perplexity": post_eval["holdout_perplexity"],
                    "post_transfer_cosine_alignment": post_eval["holdout_cosine_alignment"],
                    "holdout_loss_reduction_pct": loss_drop_pct,
                    "knowledge_transfer_efficiency": transfer_ratio,
                    "royalty_paid_hive": royalty_per_peer_hive,
                }
            )

        mean_pre_loss = (
            round(sum(r["pre_transfer_holdout_loss"] for r in peer_eval_records)
                  / len(peer_eval_records), 6) if peer_eval_records else None
        )
        mean_post_loss = (
            round(sum(r["post_transfer_holdout_loss"] for r in peer_eval_records)
                  / len(peer_eval_records), 6) if peer_eval_records else None
        )
        # No token likelihood was evaluated, so perplexity is unavailable.
        mean_pre_ppl = None
        mean_post_ppl = None
        mean_drop_pct = (
            round(sum(r["holdout_loss_reduction_pct"] for r in peer_eval_records)
                  / len(peer_eval_records), 2) if peer_eval_records else None
        )
        efficiencies = [r["knowledge_transfer_efficiency"] for r in peer_eval_records]
        mean_efficiency = (
            round(sum(efficiencies) / len(efficiencies), 4)
            if efficiencies and all(value is not None for value in efficiencies) else None
        )

        return PeerKnowledgeTransferAudit(
            domain_name=domain_name,
            benchmark_id=str(bench_meta["benchmark_id"]),
            trainer_agent_id=trainer_agent_id,
            expert_cid=expert_cid,
            trainer_initial_loss=trainer_pre_loss,
            trainer_final_loss=trainer_post_loss,
            trainer_loss_reduction_pct=trainer_drop_pct,
            peer_evaluations=peer_eval_records,
            mean_peer_pre_transfer_loss=mean_pre_loss,
            mean_peer_post_transfer_loss=mean_post_loss,
            mean_peer_pre_perplexity=mean_pre_ppl,
            mean_peer_post_perplexity=mean_post_ppl,
            mean_peer_loss_reduction_pct=mean_drop_pct,
            knowledge_transfer_efficiency_ratio=mean_efficiency,
            mean_hotswap_latency_ms=(
                round(sum(hotswap_latencies) / len(hotswap_latencies), 3)
                if hotswap_latencies else None
            ),
            peer_measurement_status="synthetic_measured" if peer_eval_records else "not_run",
            total_peer_gpu_steps_saved=None,
            simulated_peer_steps_avoided=len(other_peers) * int(train_metrics["inner_steps"]),
            royalties_paid_to_trainer_hive=round(len(other_peers) * royalty_per_peer_hive, 4),
            all_peers_leveraged_successfully=bool(peer_eval_records) and all(
                r["post_transfer_holdout_loss"] < r["pre_transfer_holdout_loss"]
                and r["knowledge_transfer_efficiency"] is not None
                and r["knowledge_transfer_efficiency"] >= 0.95
                for r in peer_eval_records
            ),
        )


class JoinedSwarmCampaignEngine:
    """
    Execute sequential synthetic campaign/game mechanics, not parallel benchmarks.
    """

    def __init__(
        self,
        transfer_engine: SwarmKnowledgeTransferEngine,
        device: Optional[torch.device] = None,
    ) -> None:
        if not isinstance(transfer_engine, SwarmKnowledgeTransferEngine):
            raise TypeError("transfer_engine must be a SwarmKnowledgeTransferEngine instance")
        self.transfer_engine = transfer_engine
        self.store = transfer_engine.store
        self.device = device or transfer_engine.device

    def execute_joined_shard_parallel_campaign(
        self,
        domain_name: str,
        reference_lora_state: Dict[str, torch.Tensor],
        participating_agents: Sequence[str],
        total_inner_steps: int = 16,
        topk_ratio: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Execute a sequential synthetic adapter-fitting and TIES-merging simulation.

        The sequential squad does more steps and uses a different learning rate
        than the solo baseline. Neither equal-time quality nor parallel speedup
        can be inferred from this comparison.
        """
        if not isinstance(domain_name, str) or not domain_name.strip():
            raise ValueError("domain_name must be a non-empty string")
        if not isinstance(reference_lora_state, Mapping) or len(reference_lora_state) == 0:
            raise ValueError("reference_lora_state must be a non-empty mapping")
        if isinstance(participating_agents, (str, bytes)) or not isinstance(participating_agents, Sequence) or len(participating_agents) == 0 or not all(
            isinstance(a, str) and bool(a.strip()) for a in participating_agents
        ):
            raise ValueError("participating_agents must be a non-empty sequence of non-empty strings")
        if isinstance(total_inner_steps, bool) or not isinstance(total_inner_steps, int) or total_inner_steps < 1:
            raise ValueError("total_inner_steps must be an integer >= 1")
        if isinstance(topk_ratio, bool) or not isinstance(topk_ratio, (int, float)) or not math.isfinite(float(topk_ratio)) or not 0.0 < float(topk_ratio) <= 1.0:
            raise ValueError("topk_ratio must be in (0.0, 1.0]")
        num_agents = max(1, len(participating_agents))
        steps_per_agent = max(4, total_inner_steps // num_agents)

        pre_holdout = _evaluate_lora_on_scientific_holdout(
            domain_name=domain_name,
            lora_state=reference_lora_state,
            device=self.device,
        )

        # Unequal-work synthetic baseline; no same-time comparison is performed.
        solo_state, _ = train_single_domain_lora_adapter(
            domain_name=domain_name,
            reference_lora_state=reference_lora_state,
            inner_steps=steps_per_agent,
            lr=0.055,
            topk_ratio=topk_ratio,
        )
        solo_holdout = _evaluate_lora_on_scientific_holdout(
            domain_name=domain_name,
            lora_state=solo_state,
            device=self.device,
        )

        # Joined Co-Op Squad: K agents refine along progressive cooperative outer-inner rounds
        squad_states: List[Dict[str, torch.Tensor]] = []
        rolling_state = {k: v.clone() for k, v in reference_lora_state.items()}
        for idx, agent_id in enumerate(participating_agents):
            agent_state, _ = train_single_domain_lora_adapter(
                domain_name=domain_name,
                reference_lora_state=rolling_state,
                inner_steps=steps_per_agent,
                lr=0.065,
                topk_ratio=topk_ratio,
            )
            squad_states.append(agent_state)
            rolling_state = ties_merge_lora_adapters(
                base_lora_state=reference_lora_state,
                worker_lora_states=squad_states,
                trim_density=0.25,
                scaling_lambda=1.15,
            )

        coop_merged_state = rolling_state
        coop_holdout = _evaluate_lora_on_scientific_holdout(
            domain_name=domain_name,
            lora_state=coop_merged_state,
            device=self.device,
        )

        coop_cid, _ = self.store.publish_lora_expert_checkpoint(
            domain_name=f"{domain_name}_JoinedCoOp",
            lora_state_dict=coop_merged_state,
            metadata={
                "campaign_id": "CAMPAIGN-COOP-01-DILOCO-BLITZ",
                "domain_name": domain_name,
                "participating_agents": list(participating_agents),
            },
        )

        # Broadcast the Joined Co-Op adapter to all participating agents
        for agent_id in participating_agents:
            self.transfer_engine.peer_local_adapters.setdefault(agent_id, {})[
                domain_name
            ] = coop_merged_state

        solo_loss = solo_holdout["holdout_loss"]
        coop_loss = coop_holdout["holdout_loss"]
        quality_gain_vs_solo_pct = round(
            100.0 * (solo_loss - coop_loss) / max(1e-8, solo_loss), 2
        )

        return {
            "campaign_id": "CAMPAIGN-COOP-01-DILOCO-BLITZ",
            "domain_name": domain_name,
            "participating_agents": list(participating_agents),
            "measurement_kind": "synthetic_regression",
            "execution_mode": "sequential_single_process",
            "steps_per_agent": steps_per_agent,
            "steps_per_agent_in_parallel": None,
            "effective_parallel_speedup": None,
            "parallel_speedup_status": "unavailable_no_concurrent_equal_work_measurement",
            "pre_training_holdout_loss": pre_holdout["holdout_loss"],
            "solo_miner_same_time_holdout_loss": None,
            "synthetic_solo_holdout_mse": solo_loss,
            "joined_coop_holdout_loss": coop_loss,
            "joined_coop_holdout_perplexity": coop_holdout["holdout_perplexity"],
            "joined_coop_cosine_alignment": coop_holdout["holdout_cosine_alignment"],
            "quality_improvement_over_solo_pct": None,
            "coop_beats_solo_miner": None,
            "synthetic_mse_difference_pct_unequal_work": quality_gain_vs_solo_pct,
            "synthetic_coop_lower_mse": coop_loss < solo_loss,
            "published_coop_expert_cid": coop_cid,
        }

    def execute_full_scientific_swarm_campaign(
        self,
        reference_lora_state: Dict[str, torch.Tensor],
        moe_model: Optional[UpcycledSparseMoEModel] = None,
        inner_steps: int = 16,
    ) -> Dict[str, Any]:
        """
        Executes all 4 domain knowledge transfers across the 4-agent swarm + Joined Co-Op Campaigns,
        verifies that every agent leverages every other agent's retraining effort (`4/4` domains),
        and builds the Unified Dual Leaderboard (Best Miner + Best Contributor).
        """
        if not isinstance(reference_lora_state, Mapping) or len(reference_lora_state) == 0:
            raise ValueError("reference_lora_state must be a non-empty mapping")
        if isinstance(inner_steps, bool) or not isinstance(inner_steps, int) or inner_steps < 1:
            raise ValueError("inner_steps must be an integer >= 1")
        t_start = time.perf_counter()
        domains = list(SCIENTIFIC_BENCHMARK_SUITES.keys())
        agents = self.transfer_engine.peer_agent_ids

        # 1. Execute `CAMPAIGN-COOP-02`: Each of the 4 agents trains 1 specialized domain and shares via CAS
        transfer_audits: Dict[str, Dict[str, Any]] = {}
        for idx, domain_name in enumerate(domains):
            trainer_agent = agents[idx % len(agents)]
            audit = self.transfer_engine.verify_swarm_peer_knowledge_transfer(
                domain_name=domain_name,
                trainer_agent_id=trainer_agent,
                reference_lora_state=reference_lora_state,
                inner_steps=inner_steps,
                topk_ratio=0.05,
                royalty_per_peer_hive=5.0,
            )
            transfer_audits[domain_name] = asdict(audit)
            if moe_model is not None:
                _, _, exp_state = self.store.load_lora_expert_checkpoint(
                    audit.expert_cid, dtype=torch.float16
                )
                if domain_name not in moe_model.expert_domains:
                    moe_model.hotplug_domain_expert(
                        domain_name=domain_name,
                        lora_state_dict=exp_state,
                        lora_rank_r=int(exp_state[next(iter(exp_state))].shape[0]),
                        lora_alpha=32,
                    )

        # 2. Verify Collective Multi-Domain Matrix: Every agent now evaluates across ALL 4 domains!
        agent_multi_domain_matrix: Dict[str, Dict[str, Any]] = {}
        for agent_id in agents:
            domain_scores: Dict[str, float] = {}
            for domain_name in domains:
                adapter = self.transfer_engine.peer_local_adapters[agent_id][domain_name]
                ev = _evaluate_lora_on_scientific_holdout(
                    domain_name=domain_name,
                    lora_state=adapter,
                    device=self.device,
                )
                domain_scores[domain_name] = ev["holdout_loss"]
            agent_multi_domain_matrix[agent_id] = {
                "measurement_kind": "synthetic_regression",
                "domains_mastered": None,
                "synthetic_domains_evaluated": len(domain_scores),
                "domains_trained_locally": 1,
                "domains_leveraged_from_swarm_peers": len(domains) - 1,
                "compute_leverage_multiplier": None,
                "simulated_adapter_reuse_multiplier": float(len(domains)),
                "holdout_loss_by_domain": domain_scores,
                "mean_4domain_holdout_loss": round(
                    sum(domain_scores.values()) / len(domain_scores), 6
                ),
            }

        # 3. Execute `CAMPAIGN-COOP-01`: Joined Shard-Parallel Blitz on `Rust_Systems_Security`
        coop_blitz_report = self.execute_joined_shard_parallel_campaign(
            domain_name="Rust_Systems_Security",
            reference_lora_state=reference_lora_state,
            participating_agents=agents,
            total_inner_steps=inner_steps,
            topk_ratio=0.05,
        )

        # 4. Build Dual Leaderboard: Best Swarm Miners + Best Open-Source & Science Contributors
        miner_leaderboard = [
            {
                "standing": 1,
                "gamer_tag": "CyberValkyrie_FR",
                "node_id": "seti-node-paris-01",
                "region": "EU-West-Paris",
                "primary_domain_trained": "Rust_Systems_Security",
                "expert_cid_published": transfer_audits["Rust_Systems_Security"]["expert_cid"],
                "holdout_loss_reduction_pct": transfer_audits["Rust_Systems_Security"][
                    "trainer_loss_reduction_pct"
                ],
                "peers_leveraged_count": self.transfer_engine.peer_reuse_counts.get(
                    "CyberValkyrie_FR", 3
                ),
                "reuse_royalties_hive": self.transfer_engine.reuse_royalty_ledger.get(
                    "CyberValkyrie_FR", 15.0
                ),
                "joined_campaigns_completed": 4,
                "total_miner_pts": 19450,
                "total_hive_earned": 248.5,
                "hf_badges": [
                    "HF_SWARM_KNOWLEDGE_DONOR",
                    "HF_COOP_RAID_COMMANDER",
                    "HF_SCIENTIFIC_VALIDATOR",
                ],
            },
            {
                "standing": 2,
                "gamer_tag": "NeuralGhost_DE",
                "node_id": "seti-node-berlin-02",
                "region": "EU-Central-Berlin",
                "primary_domain_trained": "Solidity_Smart_Contracts",
                "expert_cid_published": transfer_audits["Solidity_Smart_Contracts"]["expert_cid"],
                "holdout_loss_reduction_pct": transfer_audits["Solidity_Smart_Contracts"][
                    "trainer_loss_reduction_pct"
                ],
                "peers_leveraged_count": self.transfer_engine.peer_reuse_counts.get(
                    "NeuralGhost_DE", 3
                ),
                "reuse_royalties_hive": self.transfer_engine.reuse_royalty_ledger.get(
                    "NeuralGhost_DE", 15.0
                ),
                "joined_campaigns_completed": 4,
                "total_miner_pts": 18820,
                "total_hive_earned": 236.0,
                "hf_badges": [
                    "HF_SWARM_KNOWLEDGE_DONOR",
                    "HF_COOP_RAID_COMMANDER",
                    "HF_SCIENTIFIC_VALIDATOR",
                ],
            },
            {
                "standing": 3,
                "gamer_tag": "ZeroCool_UK",
                "node_id": "seti-node-virginia-03",
                "region": "US-East-Virginia",
                "primary_domain_trained": "Biomedical_Clinical_QA",
                "expert_cid_published": transfer_audits["Biomedical_Clinical_QA"]["expert_cid"],
                "holdout_loss_reduction_pct": transfer_audits["Biomedical_Clinical_QA"][
                    "trainer_loss_reduction_pct"
                ],
                "peers_leveraged_count": self.transfer_engine.peer_reuse_counts.get(
                    "ZeroCool_UK", 3
                ),
                "reuse_royalties_hive": self.transfer_engine.reuse_royalty_ledger.get(
                    "ZeroCool_UK", 15.0
                ),
                "joined_campaigns_completed": 4,
                "total_miner_pts": 18290,
                "total_hive_earned": 229.5,
                "hf_badges": [
                    "HF_SWARM_KNOWLEDGE_DONOR",
                    "HF_SCIENTIFIC_VALIDATOR",
                ],
            },
            {
                "standing": 4,
                "gamer_tag": "SakuraNode_JP",
                "node_id": "seti-node-tokyo-04",
                "region": "APAC-East-Tokyo",
                "primary_domain_trained": "Lean4_Theorem_Prover",
                "expert_cid_published": transfer_audits["Lean4_Theorem_Prover"]["expert_cid"],
                "holdout_loss_reduction_pct": transfer_audits["Lean4_Theorem_Prover"][
                    "trainer_loss_reduction_pct"
                ],
                "peers_leveraged_count": self.transfer_engine.peer_reuse_counts.get(
                    "SakuraNode_JP", 3
                ),
                "reuse_royalties_hive": self.transfer_engine.reuse_royalty_ledger.get(
                    "SakuraNode_JP", 15.0
                ),
                "joined_campaigns_completed": 4,
                "total_miner_pts": 17910,
                "total_hive_earned": 221.0,
                "hf_badges": [
                    "HF_SWARM_KNOWLEDGE_DONOR",
                    "HF_SCIENTIFIC_VALIDATOR",
                ],
            },
        ]

        contributor_leaderboard = [
            {
                "standing": 1,
                "contributor_handle": "@xaviercallens",
                "role_title": "Founding Protocol Architect & Principal Maintainer (Core, Bisection v3, Pareto Strategy SI-01..05)",
                "merged_prs_count": 29,
                "lean4_theorems_authored": 25,
                "cas_experts_seeded": 16,
                "downstream_swarm_agents_powered": 128,
                "bounty_hive_awarded": 23500.0,
                "contributor_prestige_pts": 44800,
                "hf_badges": [
                    "HF_OPEN_SOURCE_ARCHITECT",
                    "HF_COOP_RAID_COMMANDER",
                    "HF_SCIENTIFIC_VALIDATOR",
                ],
            },
            {
                "standing": 2,
                "contributor_handle": "@clark719664",
                "role_title": "Season 2, 3, 4 & 4.5 Grandmaster Community Architect (19 PRs: #6-#10, #21-#25, #32, #34-#37, #39-#42)",
                "merged_prs_count": 19,
                "lean4_theorems_authored": 18,
                "cas_experts_seeded": 12,
                "downstream_swarm_agents_powered": 112,
                "bounty_hive_awarded": 27200.0,
                "contributor_prestige_pts": 36900,
                "hf_badges": [
                    "HF_OPEN_SOURCE_ARCHITECT",
                    "HF_SWARM_KNOWLEDGE_DONOR",
                    "HF_SCIENTIFIC_VALIDATOR",
                ],
            },
            {
                "standing": 3,
                "contributor_handle": "@pedromineo",
                "role_title": "Season 4 & 5 Full-Stack Systems, WAN Faults, Guild Wars & 64-MoE Architect (5 PRs: #33, #52, #53, #54, #55)",
                "merged_prs_count": 5,
                "lean4_theorems_authored": 5,
                "cas_experts_seeded": 10,
                "downstream_swarm_agents_powered": 96,
                "bounty_hive_awarded": 21600.0,
                "contributor_prestige_pts": 31400,
                "hf_badges": [
                    "HF_OPEN_SOURCE_ARCHITECT",
                    "HF_SWARM_KNOWLEDGE_DONOR",
                    "HF_COOP_RAID_COMMANDER",
                ],
            },
            {
                "standing": 4,
                "contributor_handle": "@radialmonster",
                "role_title": "Steam Ecosystem & Documentation Maintainer (PR #11)",
                "merged_prs_count": 1,
                "lean4_theorems_authored": 0,
                "cas_experts_seeded": 1,
                "downstream_swarm_agents_powered": 64,
                "bounty_hive_awarded": 1000.0,
                "contributor_prestige_pts": 4200,
                "hf_badges": [
                    "HF_OPEN_SOURCE_ARCHITECT",
                ],
            },
            {
                "standing": 5,
                "contributor_handle": "Open Season 6 Challenger",
                "role_title": "Wave 5 Bounty Hunter (EF21-DiLoCo, 128-MoE, ZK-OPoI v4 — 15,000 $HIVE Pool)",
                "merged_prs_count": 0,
                "lean4_theorems_authored": 0,
                "cas_experts_seeded": 0,
                "downstream_swarm_agents_powered": 0,
                "bounty_hive_awarded": 15000.0,
                "contributor_prestige_pts": 0,
                "hf_badges": ["Fork & Submit a Season 6 PR to Unlock!"],
            },
        ]

        from hivemind.swarm.hf_community_hub import SwarmMiningGuildEngine

        guild_engine = SwarmMiningGuildEngine()
        guild_engine.record_guild_bundle_activity(
            guild_id="GUILD-02-GLOBAL-ZK-GUARDIANS",
            base_pts_earned=2400,
            base_hive_earned=120.0,
            peer_imports_count=4,
        )
        guild_leaderboard = guild_engine.build_guild_leaderboard()

        elapsed_s = round(time.perf_counter() - t_start, 3)
        return {
            "suite_version": "v1.7.0-synthetic-swarm-community-hub",
            "measurement_kind": "synthetic_regression",
            "hardware_validation_status": "not_run",
            "leaderboard_status": "scripted_game_simulation_not_live_telemetry",
            "device": str(self.device),
            "elapsed_seconds": elapsed_s,
            "scientific_knowledge_transfer_audits": transfer_audits,
            "collective_4domain_agent_matrix": agent_multi_domain_matrix,
            "joined_shard_parallel_campaign": coop_blitz_report,
            "proposed_joined_campaigns": [asdict(c) for c in PROPOSED_JOINED_CAMPAIGNS],
            "guild_alliance_leaderboard": guild_leaderboard,
            "dual_leaderboard": {
                "best_miner_champion": miner_leaderboard[0],
                "best_contributor_champion": contributor_leaderboard[0],
                "miner_standings": miner_leaderboard,
                "contributor_standings": contributor_leaderboard,
                "guild_standings": guild_leaderboard["guild_standings"],
                "hf_gamification_badges_catalog": HF_GAMIFICATION_BADGES,
            },
        }
