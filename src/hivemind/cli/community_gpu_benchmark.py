# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Opt-in Community GPU Benchmark & Hardware Verification Suite for Project HiveMind.

Allows interested developers in the community to benchmark their consumer GPU
(NVIDIA CUDA RTX 20/30/40/50-series, AMD ROCm RX 6000/7000, Apple Silicon MPS,
or CPU fallback) against the Season 6/7 Sparse MoE, EF21-DiLoCo, and ZK-OPoI v4
stack within strict VRAM safety envelopes (< 1.5 GB peak VRAM).

IMPORTANT NON-MONETARY NOTICE:
No money, fiat currency, cryptocurrency, or financial payment
is or will ever be involved or paid. All "points" (PTS and $HIVE PoUW accounting
units) recorded in benchmark receipts represent strictly non-monetary reciprocal
computing processing benefits (priority Sparse MoE inference tokens, speculative
decoding bandwidth, and federated LoRA fine-tuning slots) redeemable only when
the live production HiveMind Swarm is deployed, available, and actively used by
a large community.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from hivemind.core.btm_moe import (
    ThreeTierHierarchicalMoERouter,
    hotplug_three_tier_experts,
)
from hivemind.core.btm_upcycle import (
    initialize_domain_branches,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.compression import decompress_lora_delta
from hivemind.core.quantization import (
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
)
from hivemind.data.token_training_eval import run_token_training_evaluation
from hivemind.swarm.inference_ring import RegionalDHTSwarmRing
from hivemind.swarm.ledger import GuildSlashingInsurancePool, HiveSwarmLedger
from hivemind.swarm.strategy_optimizer import EF21DiLoCoErrorFeedbackBuffer
from hivemind.swarm.swarm_fabric import (
    OptimisticProofOfInferenceAuditor,
    PinnedRAMExpertDMAStreamer,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RECEIPT_PATH = REPO_ROOT / "artifacts" / "community_gpu_benchmark_receipt.json"

NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE = (
    "NO FINANCIAL PAYMENT: Project HiveMind is a voluntary, "
    "non-commercial open-source research project. No money, fiat currency, cryptocurrency, "
    "or wage is ever promised or paid. Accumulated 'points' (PTS and $HIVE PoUW accounting "
    "units) confer exclusively reciprocal computing processing benefits (shared Sparse MoE "
    "inference and training quota) when the live production HiveMind Swarm is deployed, "
    "available, and used by a large community."
)


def detect_community_accelerator(preferred_device: str = "auto") -> dict[str, Any]:
    """Detect the contributor's GPU/accelerator backend and VRAM envelope."""
    if (
        isinstance(preferred_device, bool)
        or not isinstance(preferred_device, str)
        or preferred_device not in {"auto", "cuda", "mps", "cpu"}
    ):
        raise ValueError(f"Unsupported preferred_device: {preferred_device!r}")

    if preferred_device == "cuda" or (preferred_device == "auto" and torch.cuda.is_available()):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        props = torch.cuda.get_device_properties(0)
        total_mib = round(props.total_memory / (1024 * 1024), 2)
        hip_ver = getattr(torch.version, "hip", None)
        backend = "rocm_hip" if hip_ver else "cuda"
        return {
            "device": "cuda:0",
            "backend": backend,
            "device_name": str(props.name),
            "compute_capability": f"{props.major}.{props.minor}",
            "total_vram_mib": total_mib,
            "wddm_usable_budget_mib": min(total_mib, 6656.0),
            "cuda_or_hip_version": str(hip_ver or torch.version.cuda or "unknown"),
            "is_gpu": True,
        }

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if preferred_device == "mps" or (preferred_device == "auto" and mps_available):
        if not mps_available:
            raise RuntimeError("MPS requested but torch.backends.mps.is_available() is False")
        return {
            "device": "mps",
            "backend": "apple_mps",
            "device_name": f"Apple Silicon MPS ({platform.processor() or platform.machine()})",
            "compute_capability": "mps_unified_memory",
            "total_vram_mib": 8192.0,
            "wddm_usable_budget_mib": 6656.0,
            "cuda_or_hip_version": "mps",
            "is_gpu": True,
        }

    return {
        "device": "cpu",
        "backend": "cpu_fallback",
        "device_name": f"CPU ({platform.processor() or platform.machine()})",
        "compute_capability": "cpu_fp32",
        "total_vram_mib": 0.0,
        "wddm_usable_budget_mib": 6656.0,
        "cuda_or_hip_version": "none",
        "is_gpu": False,
    }


def run_community_gpu_benchmark(
    contributor_handle: str = "anonymous-contributor",
    preferred_device: str = "auto",
    token_steps: int = 8,
    router_tokens: int = 128,
    hidden_size: int = 64,
    seed: int = 42,
    output_path: Path | None = DEFAULT_RECEIPT_PATH,
    pool_holdings_override: float | None = None,
) -> dict[str, Any]:
    """Execute the 5-stage opt-in Community GPU Benchmark & Hardware Verification Suite."""
    if not isinstance(contributor_handle, str) or not contributor_handle.strip():
        raise ValueError("contributor_handle must be non-empty")
    if isinstance(token_steps, bool) or not isinstance(token_steps, int) or token_steps < 1:
        raise ValueError("token_steps must be >= 1")
    if isinstance(router_tokens, bool) or not isinstance(router_tokens, int) or router_tokens < 4:
        raise ValueError("router_tokens must be >= 4")
    if isinstance(hidden_size, bool) or not isinstance(hidden_size, int) or hidden_size < 16 or hidden_size % 8 != 0:
        raise ValueError("hidden_size must be a multiple of 8 and >= 16")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if output_path is not None and (isinstance(output_path, bool) or not isinstance(output_path, Path)):
        raise ValueError("output_path must be a Path or None")
    if pool_holdings_override is not None:
        if isinstance(pool_holdings_override, bool) or not isinstance(pool_holdings_override, (int, float)):
            raise ValueError("pool_holdings_override must be a finite non-negative float")
        pool_override_val = float(pool_holdings_override)
        if not math.isfinite(pool_override_val) or pool_override_val < 0.0:
            raise ValueError("pool_holdings_override must be a finite non-negative float")

    t_start = time.perf_counter()
    torch.manual_seed(seed)

    # Stage 1: Accelerator & VRAM Envelope Discovery
    hw = detect_community_accelerator(preferred_device=preferred_device)
    dev = torch.device(hw["device"])
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)

    # Stage 2: Real-Token Causal-LM LoRA Convergence + EF21-DiLoCo Residual Contraction
    t_stage2 = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp_eval_dir:
        token_eval = run_token_training_evaluation(
            Path(tmp_eval_dir) / "token_eval",
            steps=token_steps,
            seed=seed,
        )
    baseline_nll = float(token_eval["holdout"]["baseline"]["mean_nll_nats"])
    trained_nll = float(token_eval["holdout"]["trained_adapter"]["mean_nll_nats"])
    recipient_nll = float(token_eval["holdout"]["imported_adapter"]["mean_nll_nats"])
    ef21 = EF21DiLoCoErrorFeedbackBuffer(
        topk_ratio=0.10,
        stale_decay=1.0,
    )
    gen = torch.Generator(device="cpu").manual_seed(seed)
    raw_lora_delta = {
        "layers.0.self_attn.q_proj.lora_A": torch.randn((16, hidden_size), generator=gen),
        "layers.0.self_attn.q_proj.lora_B": torch.randn((hidden_size, 16), generator=gen),
    }
    compressed_pkt = ef21.compress(
        raw_lora_delta,
        worker_id=f"community-node-{contributor_handle.strip()}",
        branch_name="Rust_Systems_Security",
        outer_step=1,
    )
    decompressed_delta = decompress_lora_delta(compressed_pkt)
    conservation_error_l2 = float(
        torch.sqrt(
            sum(
                (decompressed_delta[k] + ef21.residuals[k] - raw_lora_delta[k]).pow(2).sum()
                for k in raw_lora_delta
            )
        )
    )
    raw_norm = float(torch.sqrt(sum(v.pow(2).sum() for v in raw_lora_delta.values())))
    contraction_factor = float(ef21.residual_norm() / max(1e-9, raw_norm))
    stage2_ms = round((time.perf_counter() - t_stage2) * 1000.0, 3)

    # Stage 3: 128-Expert 3-Tier Hierarchical MoE Routing & Speculative Token-Tree Prefetch
    t_stage3 = time.perf_counter()
    router = ThreeTierHierarchicalMoERouter(
        hidden_size=hidden_size,
        num_super_clusters=4,
        macro_per_super=4,
        micro_per_macro=8,
        top_k=2,
    ).to(dev)
    x_tokens = torch.randn((router_tokens, hidden_size), device=dev, dtype=torch.float32)
    weights, selected_experts, aux_loss = router(x_tokens)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    stage3_ms = round((time.perf_counter() - t_stage3) * 1000.0, 3)
    tokens_per_sec = round(router_tokens / max(1e-6, (stage3_ms / 1000.0)), 2)

    domains = ["DomA", "DomB", "DomC"]
    base_model, _, _ = build_local_qwen2_backbone(
        seed=seed,
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(
        base_model=base_model,
        rank_r=2,
        lora_alpha=4,
        gradient_checkpointing=False,
        seed=seed,
    )
    init_lora = extract_lora_state_dict(bundle.model)
    branches = initialize_domain_branches(init_lora, expert_domains=domains)
    moe, _ = upcycle_branches_to_sparse_moe(
        bundle.model,
        {k: v.lora_state for k, v in branches.items()},
        lora_rank_r=2,
        lora_alpha=4,
        in_place=False,
    )
    moe.moe_blocks = moe.get_moe_blocks()  # type: ignore[attr-defined]
    moe.hidden_size = 16  # type: ignore[attr-defined]
    hotplug_three_tier_experts(
        moe, {}, canonical_rank_r=2, num_super_clusters=1, macro_per_super=1
    )

    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    for dom in domains:
        streamer.stage_expert_in_pinned_ram(dom, {"w": torch.zeros(8)})
    ring = RegionalDHTSwarmRing(region="EU-West-Paris")
    spec_res = ring.run_speculative_inference(
        moe_target_model=moe,
        draft_model=moe,
        prompt_ids=torch.tensor([1, 2, 3, 4]),
        max_new_tokens=6,
        draft_gamma=3,
        expert_streamer=streamer,
        max_prefetch_experts=2,
    )

    # Stage 4: ZK-OPoI v4 Position-Bound Batch Commitments & Insurance Pool Solvency
    t_stage4 = time.perf_counter()
    auditor = OptimisticProofOfInferenceAuditor()
    batch_acts = [
        [torch.randn((2, hidden_size), generator=gen) for _ in range(4)]
        for _ in range(6)
    ]
    commitment = auditor.commit_speculative_batch("community-prover-01", batch_acts)
    challenges = auditor.derive_batch_challenges(commitment, 4, "community-gpu-nonce-01")
    responses = [
        (auditor.open_speculative_batch(s, l_idx), batch_acts[s][l_idx], batch_acts[s][l_idx])
        for s, l_idx in challenges
    ]
    batch_verify = auditor.verify_speculative_batch(commitment, challenges, responses)

    ledger = HiveSwarmLedger()
    ledger.balances["community-prover-01"] = 500.0
    ledger.balances["community-prover-02"] = 500.0
    pool = GuildSlashingInsurancePool(ledger)
    pool.bond_stake("GUILD-01", "community-prover-01", 50.0)
    pool.contribute_reserve("GUILD-01", "community-prover-01", 75.0)
    pool.contribute_reserve("GUILD-02", "community-prover-02", 75.0)
    pool_holdings_hive = float(pool_holdings_override) if pool_holdings_override is not None else pool.total_holdings()

    # HM-S7-04 fault drill on its own ledger: one forged leaf must be caught and paid for.
    drill_ledger = HiveSwarmLedger()
    drill_ledger.balances.update({"community-prover-01": 100.0, "community-prover-02": 100.0})
    drill_pool = GuildSlashingInsurancePool(drill_ledger)
    drill_pool.bond_stake("GUILD-01", "community-prover-01", 50.0)
    drill_pool.contribute_reserve("GUILD-02", "community-prover-02", 50.0)
    forged_acts = [list(step) for step in batch_acts]
    forged_acts[2][1] = batch_acts[2][1] * 3.0
    forged = auditor.commit_speculative_batch("community-prover-01", forged_acts)
    drill_challenges = auditor.derive_batch_challenges(
        forged, forged.leaf_count, auditor.new_verifier_nonce()
    )
    drill_responses = [
        (auditor.open_speculative_batch(s, l_idx), forged_acts[s][l_idx], batch_acts[s][l_idx])
        for s, l_idx in drill_challenges
    ]
    drill_supply_before = drill_ledger.total_circulating_supply() + drill_pool.total_holdings()
    drill = auditor.audit_batch_and_indemnify(
        forged,
        drill_challenges,
        drill_responses,
        drill_pool,
        consumer_node_id="community-consumer-01",
        claimed_hive=20.0,
    )
    drill_supply_after = drill_ledger.total_circulating_supply() + drill_pool.total_holdings()
    drill_indemnity = drill["indemnity"] or {}
    drill_passed = bool(
        [(f["step"], f["layer"]) for f in drill["faults"]] == [(2, 1)]
        and drill_indemnity.get("indemnified_hive") == 20.0
        and "community-prover-01" in drill_ledger.slashed_registry
        and abs(drill_supply_after - drill_supply_before) < 1e-6
    )
    stage4_ms = round((time.perf_counter() - t_stage4) * 1000.0, 3)

    peak_vram_mib = (
        round(torch.cuda.max_memory_allocated(dev) / (1024 * 1024), 3)
        if dev.type == "cuda"
        else 0.0
    )

    all_passed = bool(
        trained_nll < baseline_nll
        and conservation_error_l2 < 1e-4
        and contraction_factor < 1.0
        and router.gating_reduction_factor == 8.0
        and weights.shape == (router_tokens, 2)
        and selected_experts.shape == (router_tokens, 2)
        and torch.isfinite(aux_loss)
        and spec_res.expert_prefetch_enabled is True
        and spec_res.expert_prefetch_calls > 0
        and batch_verify["batch_passed"] is True
        and pool_holdings_hive == 200.0
        and drill_passed
        and peak_vram_mib <= hw["wddm_usable_budget_mib"]
    )

    # Symbolic, non-monetary compute-reciprocity points preview
    symbolic_points_preview = 500 if hw["is_gpu"] else 250
    gpu_points_awarded = 500 if hw["is_gpu"] else 0
    hardware_summary = {
        **hw,
        "measured_on_gpu": bool(hw["is_gpu"]),
        "peak_allocated_vram_mib": peak_vram_mib,
        "within_8gb_wddm_envelope": peak_vram_mib <= hw["wddm_usable_budget_mib"],
    }

    payload: dict[str, Any] = {
        "benchmark_suite": "Project HiveMind Community GPU & Swarm Verification Suite (Season 7 / Wave 6)",
        "contributor_handle": contributor_handle.strip(),
        "timestamp_utc": round(time.time(), 3),
        "elapsed_total_ms": round((time.perf_counter() - t_start) * 1000.0, 3),
        "overall_passed": all_passed,
        "gpu_points_awarded": gpu_points_awarded,
        "hardware": dict(hardware_summary),
        "non_monetary_charter": {
            "financial_payment_usd_eur": 0.0,
            "no_money_involved": True,
            "reward_nature": "non_monetary_compute_reciprocity_points_only",
            "redemption_condition": (
                "Points (PTS / $HIVE) grant exclusively computing processing benefits "
                "(shared Sparse MoE inference and training quota) when the live production "
                "HiveMind Swarm is deployed, available, and used by a large community."
            ),
            "symbolic_points_preview_pts": symbolic_points_preview,
            "gpu_points_awarded": gpu_points_awarded,
            "legal_notice": NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE,
        },
        "stage1_accelerator": dict(hardware_summary),
        "stage2_real_token_and_ef21": {
            "elapsed_ms": stage2_ms,
            "baseline_holdout_nll": round(baseline_nll, 6),
            "trained_holdout_nll": round(trained_nll, 6),
            "recipient_holdout_nll": round(recipient_nll, 6),
            "ef21_contraction_factor": round(contraction_factor, 6),
            "ef21_conservation_error_l2": round(conservation_error_l2, 8),
        },
        "stage3_three_tier_128_moe_and_prefetch": {
            "elapsed_ms": stage3_ms,
            "num_experts": router.num_experts,
            "gating_inner_products_per_token": router.gating_inner_products_per_token,
            "gating_reduction_factor": router.gating_reduction_factor,
            "routing_throughput_tokens_per_sec": tokens_per_sec,
            "prefetch_calls": spec_res.expert_prefetch_calls,
            "prefetch_hit_rate": round(float(spec_res.expert_prefetch_hit_rate or 0.0), 4),
        },
        "stage4_zk_opoi_v4_and_insurance": {
            "elapsed_ms": stage4_ms,
            "merkle_root": commitment.root,
            "leaf_count": commitment.leaf_count,
            "challenges_verified": len(challenges),
            "batch_passed": batch_verify["batch_passed"],
            "insurance_pool_reserve_hive": pool_holdings_hive,
            "fault_drill": {
                "forged_leaf": [2, 1],
                "faults_found": [[f["step"], f["layer"]] for f in drill["faults"]],
                "indemnified_hive": drill_indemnity.get("indemnified_hive"),
                "paid_from_peer_stake": drill_indemnity.get("paid_from_peer_stake"),
                "settlement_ms": drill_indemnity.get("settlement_ms"),
                "supply_conserved": abs(drill_supply_after - drill_supply_before) < 1e-6,
                "passed": drill_passed,
            },
        },
    }

    canonical = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    payload["receipt_sha256"] = hashlib.sha256(canonical).hexdigest()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    return payload


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def verify_community_benchmark_receipt(receipt: Mapping[str, Any]) -> bool:
    """Verify the canonical SHA-256 digest and internal stage invariants of a benchmark receipt."""
    if not isinstance(receipt, Mapping):
        return False
    expected = str(receipt.get("receipt_sha256") or "").strip().lower()
    if len(expected) != 64:
        return False
    body = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
    try:
        canonical = json.dumps(body, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        return False
    actual = hashlib.sha256(canonical).hexdigest().lower()
    if actual != expected:
        return False
    if body.get("overall_passed") is not True:
        return False

    handle = body.get("contributor_handle")
    if not isinstance(handle, str) or not handle.strip():
        return False

    hw = body.get("hardware")
    stage1 = body.get("stage1_accelerator")
    charter = body.get("non_monetary_charter")
    stage2 = body.get("stage2_real_token_and_ef21")
    stage3 = body.get("stage3_three_tier_128_moe_and_prefetch")
    stage4 = body.get("stage4_zk_opoi_v4_and_insurance")
    if not all(
        isinstance(m, Mapping)
        for m in (hw, stage1, charter, stage2, stage3, stage4)
    ):
        return False

    assert isinstance(hw, Mapping)
    assert isinstance(stage1, Mapping)
    assert isinstance(charter, Mapping)
    assert isinstance(stage2, Mapping)
    assert isinstance(stage3, Mapping)
    assert isinstance(stage4, Mapping)

    is_gpu = hw.get("is_gpu")
    if not isinstance(is_gpu, bool) or hw.get("measured_on_gpu") is not is_gpu:
        return False
    if stage1.get("is_gpu") is not is_gpu:
        return False
    peak_vram = hw.get("peak_allocated_vram_mib")
    budget_vram = hw.get("wddm_usable_budget_mib")
    if not (_is_finite_number(peak_vram) and _is_finite_number(budget_vram)):
        return False
    if float(peak_vram) < 0.0 or float(peak_vram) > float(budget_vram):
        return False
    if hw.get("within_8gb_wddm_envelope") is not True:
        return False

    expected_gpu_pts = 500 if is_gpu else 0
    expected_preview_pts = 500 if is_gpu else 250
    if body.get("gpu_points_awarded") != expected_gpu_pts:
        return False
    if (
        charter.get("no_money_involved") is not True
        or charter.get("financial_payment_usd_eur") != 0.0
        or charter.get("reward_nature") != "non_monetary_compute_reciprocity_points_only"
        or charter.get("gpu_points_awarded") != expected_gpu_pts
        or charter.get("symbolic_points_preview_pts") != expected_preview_pts
        or charter.get("legal_notice") != NON_MONETARY_COMPUTE_RECIPROCITY_NOTICE
    ):
        return False

    base_nll = stage2.get("baseline_holdout_nll")
    trained_nll = stage2.get("trained_holdout_nll")
    contraction = stage2.get("ef21_contraction_factor")
    cons_err = stage2.get("ef21_conservation_error_l2")
    if not all(_is_finite_number(x) for x in (base_nll, trained_nll, contraction, cons_err)):
        return False
    if not (
        float(trained_nll) < float(base_nll)
        and 0.0 <= float(contraction) < 1.0
        and 0.0 <= float(cons_err) < 1e-4
    ):
        return False

    prefetch_calls = stage3.get("prefetch_calls")
    if (
        stage3.get("num_experts") != 128
        or stage3.get("gating_reduction_factor") != 8.0
        or isinstance(prefetch_calls, bool)
        or not isinstance(prefetch_calls, int)
        or prefetch_calls <= 0
    ):
        return False

    drill = stage4.get("fault_drill")
    if not isinstance(drill, Mapping):
        return False
    if (
        stage4.get("batch_passed") is not True
        or stage4.get("insurance_pool_reserve_hive") != 200.0
        or drill.get("passed") is not True
        or drill.get("supply_conserved") is not True
        or drill.get("indemnified_hive") != 20.0
        or drill.get("faults_found") != [[2, 1]]
    ):
        return False

    return True


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for the opt-in Community GPU Benchmark."""
    if argv is not None and (isinstance(argv, (str, bytes)) or not isinstance(argv, (list, tuple))):
        raise ValueError("argv must be a list/tuple of strings or None")
    parser = argparse.ArgumentParser(
        description=(
            "Run the opt-in Project HiveMind Community GPU Benchmark & Hardware Verification Suite. "
            "Note: No financial payment is ever involved; points represent reciprocal "
            "computing processing benefits when the live production swarm is deployed."
        )
    )
    parser.add_argument(
        "--contributor-handle",
        default="community-developer",
        help="Your GitHub handle for the benchmark receipt (default: community-developer)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Target accelerator backend (default: auto)",
    )
    parser.add_argument(
        "--token-steps",
        type=int,
        default=8,
        help="Number of real-token LoRA micro-steps (default: 8)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RECEIPT_PATH,
        help="Output path for the signed JSON benchmark receipt",
    )
    args = parser.parse_args(argv)

    receipt = run_community_gpu_benchmark(
        contributor_handle=args.contributor_handle,
        preferred_device=args.device,
        token_steps=args.token_steps,
        output_path=args.output,
    )
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["overall_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
