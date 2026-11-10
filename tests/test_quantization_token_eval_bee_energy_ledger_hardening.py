# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Strict boundary and negative-control tests for Iteration 143 hardening."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from hivemind.core.quantization import (
    NF4QuantState,
    QuantizedModelBundle,
    apply_nf4_double_quant_to_backbone,
    build_local_qwen2_backbone,
    extract_lora_state_dict,
    measure_vram_footprint,
)
from hivemind.data.token_training_eval import (
    adapter_state,
    build_model,
    evaluate_tokens,
    frozen_state,
    load_fixture,
    tokenize_documents,
)
from hivemind.swarm.ledger import (
    HivePoUWBlock,
    IndemnityReceipt,
    InferenceSettlementReceipt,
    StorageProofReceipt,
    WorkReceipt,
)
from hivemind.verification.bee_energy import (
    AutopoieticPromotionVerdict,
    BeeEnergyEvaluation,
    _has_ast_stub_or_syntax_error,
    _strip_lean4_comments,
)


# ---------------------------------------------------------------------------
# 1. quantization.py
# ---------------------------------------------------------------------------


def test_nf4_quant_state_and_bundle_guards() -> None:
    state = NF4QuantState(
        packed_4bit=torch.zeros(4, dtype=torch.uint8),
        q_absmax_uint8=torch.ones(1, dtype=torch.uint8),
        superblock_scale=0.5,
        original_shape=(2, 4),
        block_size=64,
    )
    assert state.storage_bytes == 9

    with pytest.raises(ValueError, match="packed_4bit"):
        NF4QuantState(
            packed_4bit=[1, 2],  # type: ignore[arg-type]
            q_absmax_uint8=torch.ones(1, dtype=torch.uint8),
            superblock_scale=0.5,
            original_shape=(2, 4),
        )
    with pytest.raises(ValueError, match="superblock_scale"):
        NF4QuantState(
            packed_4bit=torch.zeros(4, dtype=torch.uint8),
            q_absmax_uint8=torch.ones(1, dtype=torch.uint8),
            superblock_scale=0.0,
            original_shape=(2, 4),
        )
    with pytest.raises(ValueError, match="original_shape"):
        NF4QuantState(
            packed_4bit=torch.zeros(4, dtype=torch.uint8),
            q_absmax_uint8=torch.ones(1, dtype=torch.uint8),
            superblock_scale=0.5,
            original_shape=(0, 4),
        )
    with pytest.raises(ValueError, match="block_size"):
        NF4QuantState(
            packed_4bit=torch.zeros(4, dtype=torch.uint8),
            q_absmax_uint8=torch.ones(1, dtype=torch.uint8),
            superblock_scale=0.5,
            original_shape=(2, 4),
            block_size=0,
        )

    bundle = QuantizedModelBundle(
        model=nn.Linear(2, 2),
        base_model_id="qwen2-local",
        base_model_sha256="a" * 64,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        is_4bit_nf4=True,
        nf4_compressed_base_mb=0.1,
        trainable_params=4,
        total_params=8,
        lora_size_mb=0.01,
    )
    assert bundle.total_params == 8

    with pytest.raises(ValueError, match="base_model_sha256"):
        QuantizedModelBundle(
            model=nn.Linear(2, 2),
            base_model_id="qwen2-local",
            base_model_sha256="short",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            is_4bit_nf4=True,
            nf4_compressed_base_mb=0.1,
            trainable_params=4,
            total_params=8,
            lora_size_mb=0.01,
        )
    with pytest.raises(ValueError, match="total_params"):
        QuantizedModelBundle(
            model=nn.Linear(2, 2),
            base_model_id="qwen2-local",
            base_model_sha256="a" * 64,
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            is_4bit_nf4=True,
            nf4_compressed_base_mb=0.1,
            trainable_params=10,
            total_params=8,
            lora_size_mb=0.01,
        )


def test_quantization_helper_guards() -> None:
    with pytest.raises(ValueError, match="double_quant"):
        apply_nf4_double_quant_to_backbone(None, double_quant=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="base_model"):
        apply_nf4_double_quant_to_backbone(None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="vocab_size"):
        build_local_qwen2_backbone(vocab_size=1)
    with pytest.raises(ValueError, match="hidden_size"):
        build_local_qwen2_backbone(hidden_size=2)
    with pytest.raises(ValueError, match="use_4bit_nf4_if_cuda"):
        build_local_qwen2_backbone(use_4bit_nf4_if_cuda=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="named_parameters"):
        extract_lora_state_dict(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="device"):
        measure_vram_footprint("cpu")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. token_training_eval.py
# ---------------------------------------------------------------------------


def test_token_training_eval_guards() -> None:
    with pytest.raises(ValueError, match="path"):
        load_fixture("   ")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="samples"):
        tokenize_documents("raw-string-not-sequence")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="model"):
        evaluate_tokens(object(), {})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="batch"):
        evaluate_tokens(nn.Linear(2, 2), {"input_ids": torch.ones((1, 2), dtype=torch.long)})
    with pytest.raises(ValueError, match="base_bytes"):
        build_model(42, base_bytes=b"")
    with pytest.raises(ValueError, match="named_parameters"):
        adapter_state(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="named_parameters"):
        frozen_state(object())  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. bee_energy.py
# ---------------------------------------------------------------------------


def test_bee_energy_evaluation_and_verdict_guards() -> None:
    ev = BeeEnergyEvaluation(
        candidate_id="cand-1",
        response_text="def f(x):\n    return x + 1\n",
        tests_passed=2,
        tests_total=2,
        latency_ms=5.0,
        memory_mb=8.0,
        invariant_residual=0.0,
        is_stub_or_invalid=False,
        failure_reason=None,
        total_energy=0.1,
        receipt_sha256="b" * 64,
    )
    assert ev.passed_all_tests is True

    with pytest.raises(ValueError, match="receipt_sha256"):
        BeeEnergyEvaluation(
            candidate_id="cand-1",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=2,
            tests_total=2,
            latency_ms=5.0,
            memory_mb=8.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=0.1,
            receipt_sha256="short",
        )
    with pytest.raises(ValueError, match="total_energy"):
        BeeEnergyEvaluation(
            candidate_id="cand-1",
            response_text="def f(x):\n    return x + 1\n",
            tests_passed=2,
            tests_total=2,
            latency_ms=5.0,
            memory_mb=8.0,
            invariant_residual=0.0,
            is_stub_or_invalid=False,
            failure_reason=None,
            total_energy=-1.0,
            receipt_sha256="b" * 64,
        )

    with pytest.raises(ValueError, match="source"):
        _strip_lean4_comments(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="domain"):
        _has_ast_stub_or_syntax_error("def f(): return 1", domain="   ")

    verdict = AutopoieticPromotionVerdict(
        promoted=True,
        median_delta_energy=-5.0,
        parent_regressions_failed=0,
        shuffled_control_gained=False,
        reason="PROMOTED",
    )
    assert verdict.promoted is True

    with pytest.raises(ValueError, match="reason"):
        AutopoieticPromotionVerdict(
            promoted=False,
            median_delta_energy=0.0,
            parent_regressions_failed=0,
            shuffled_control_gained=False,
            reason="   ",
        )


# ---------------------------------------------------------------------------
# 4. ledger.py
# ---------------------------------------------------------------------------


def test_ledger_receipt_and_block_guards() -> None:
    wr = WorkReceipt(
        node_id="node-01",
        steam_id64="76561198000000001",
        branch_name="Rust",
        flops_contributed=1e12,
        cosine_alignment=0.95,
        spot_check_passed=True,
        multikrum_accepted=True,
        delta_sha256="c" * 64,
    )
    assert wr.node_id == "node-01"

    with pytest.raises(ValueError, match="node_id"):
        WorkReceipt(
            node_id="  ",
            steam_id64="76561198000000001",
            branch_name="Rust",
            flops_contributed=1e12,
            cosine_alignment=0.95,
            spot_check_passed=True,
            multikrum_accepted=True,
            delta_sha256="c" * 64,
        )

    with pytest.raises(ValueError, match="challenged_chunk_index"):
        StorageProofReceipt(
            node_id="seeder-01",
            shard_cid="cid://shard",
            challenged_chunk_index=-1,
            merkle_proof_hash="d" * 64,
            bytes_stored=1024,
            verified=True,
        )

    with pytest.raises(ValueError, match="verified_tokens"):
        InferenceSettlementReceipt(
            consumer_node_id="consumer-1",
            expert_provider_nodes=["prov-1"],
            domain_expert="Rust",
            verified_tokens=0,
            rtt_speedup=1.2,
            fee_paid_hive=0.5,
        )

    with pytest.raises(ValueError, match="block_height"):
        HivePoUWBlock(
            block_height=-1,
            prev_block_hash="0" * 64,
            timestamp=1700000000.0,
            outer_step=0,
            expert_cid="cid://exp",
            merkle_root="e" * 64,
            total_block_reward=50.0,
            reward_allocations={},
            slashed_nodes=[],
        )

    with pytest.raises(ValueError, match="claimed_hive"):
        IndemnityReceipt(
            fault_id="f-01",
            commitment_root="f" * 64,
            step=1,
            layer=0,
            fault_reason="MISMATCH",
            faulty_peer_id="peer-bad",
            faulty_guild_id="guild-1",
            consumer_node_id="consumer-1",
            claimed_hive=0.0,
            paid_from_peer_stake=0.0,
            paid_from_reserve=0.0,
            indemnified_hive=0.0,
            shortfall_hive=0.0,
            stake_surplus_to_reserve=0.0,
            settlement_ms=1.0,
        )
