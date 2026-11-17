# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant tests for Iteration 156 hardening across:
- `hivemind.swarm.dynamic_diurnal_moe_engine`
- `hivemind.swarm.crowdgpt_loss_check_service`
- `hivemind.swarm.crowdgpt_bridge`
- `hivemind.swarm.amcp_kafka_campaign_bus`
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.swarm.amcp_kafka_campaign_bus import (
    AMCPKafkaEvent,
    SubjectCampaign,
    build_amcp_kafka_event,
)
from hivemind.swarm.crowdgpt_bridge import (
    CrowdGPTDeltaPayload,
    GpuPrecisionCohort,
    MultiKrumAggregationResult,
    PoolDispatchDecision,
    classify_gpu_precision_cohort,
    multi_krum_select_and_aggregate,
    pack_crowdgpt_delta_stream,
    resolve_forming_pool_starvation,
    unpack_crowdgpt_delta_stream,
)
from hivemind.swarm.crowdgpt_loss_check_service import (
    JOB_SCHEMA,
    LossCheckCoordinator,
    LossCheckJob,
    fastapi_router,
)
from hivemind.swarm.dynamic_diurnal_moe_engine import (
    DiurnalMoETrainingScheduler,
    DiurnalOperationalMode,
    DiurnalTrainingWorkunit,
    LongTailDomainScore,
)


def test_dynamic_diurnal_dataclasses_and_scheduler_init_hardening() -> None:
    score = LongTailDomainScore(
        domain="maths_formal",
        canonical_domain="maths_formal",
        hf_dataset="AI-MO/NuminaMath-CoT",
        request_count=2,
        complexity_weight=2.5,
        priority_score=3.89,
        recommended_base_model="Qwen/Qwen2.5-32B-Instruct",
        cluster="FORMAL",
        reason="High-value long-tail request",
    )
    assert score.priority_score == pytest.approx(3.89)

    with pytest.raises(ValueError, match="domain must be a non-empty string"):
        LongTailDomainScore(
            domain="",
            canonical_domain="maths_formal",
            hf_dataset="AI-MO/NuminaMath-CoT",
            request_count=2,
            complexity_weight=2.5,
            priority_score=3.89,
            recommended_base_model="Qwen/Qwen2.5-32B-Instruct",
            cluster="FORMAL",
            reason="ok",
        )
    with pytest.raises(ValueError, match="request_count must be a non-negative integer"):
        LongTailDomainScore(
            domain="maths_formal",
            canonical_domain="maths_formal",
            hf_dataset="AI-MO/NuminaMath-CoT",
            request_count=-1,
            complexity_weight=2.5,
            priority_score=3.89,
            recommended_base_model="Qwen/Qwen2.5-32B-Instruct",
            cluster="FORMAL",
            reason="ok",
        )
    with pytest.raises(ValueError, match="priority_score must be a finite positive number"):
        LongTailDomainScore(
            domain="maths_formal",
            canonical_domain="maths_formal",
            hf_dataset="AI-MO/NuminaMath-CoT",
            request_count=1,
            complexity_weight=2.5,
            priority_score=0.0,
            recommended_base_model="Qwen/Qwen2.5-32B-Instruct",
            cluster="FORMAL",
            reason="ok",
        )

    wu = DiurnalTrainingWorkunit(
        wu_id="wu-1",
        domain="maths_formal",
        canonical_domain="maths_formal",
        hf_dataset="AI-MO/NuminaMath-CoT",
        base_model_id="Qwen/Qwen2.5-32B-Instruct",
        target_rank=32,
        steps=100,
        operational_mode=DiurnalOperationalMode.INTENSE_NOCTURNAL_TRAINING,
        assigned_producer_id="acc-1",
        producer_handle="node-1",
        producer_timezone_offset=1.0,
        local_clock_hour=23.5,
        pouw_reward_hmt=12.5,
        estimated_tflops=370.0,
    )
    assert wu.to_dict()["target_rank"] == 32

    with pytest.raises(ValueError, match="target_rank must be a positive integer"):
        DiurnalTrainingWorkunit(
            wu_id="wu-1",
            domain="maths_formal",
            canonical_domain="maths_formal",
            hf_dataset="AI-MO/NuminaMath-CoT",
            base_model_id="Qwen/Qwen2.5-32B-Instruct",
            target_rank=0,
            steps=100,
            operational_mode=DiurnalOperationalMode.INTENSE_NOCTURNAL_TRAINING,
            assigned_producer_id="acc-1",
            producer_handle="node-1",
            producer_timezone_offset=1.0,
            local_clock_hour=23.5,
            pouw_reward_hmt=12.5,
            estimated_tflops=370.0,
        )
    with pytest.raises(TypeError, match="operational_mode must be a DiurnalOperationalMode"):
        DiurnalTrainingWorkunit(
            wu_id="wu-1",
            domain="maths_formal",
            canonical_domain="maths_formal",
            hf_dataset="AI-MO/NuminaMath-CoT",
            base_model_id="Qwen/Qwen2.5-32B-Instruct",
            target_rank=32,
            steps=100,
            operational_mode="INVALID_MODE",  # type: ignore[arg-type]
            assigned_producer_id="acc-1",
            producer_handle="node-1",
            producer_timezone_offset=1.0,
            local_clock_hour=23.5,
            pouw_reward_hmt=12.5,
            estimated_tflops=370.0,
        )
    with pytest.raises(ValueError, match="byob_bee_id must be a non-empty string or None"):
        DiurnalTrainingWorkunit(
            wu_id="wu-1",
            domain="maths_formal",
            canonical_domain="maths_formal",
            hf_dataset="AI-MO/NuminaMath-CoT",
            base_model_id="Qwen/Qwen2.5-32B-Instruct",
            target_rank=32,
            steps=100,
            operational_mode=DiurnalOperationalMode.INTENSE_NOCTURNAL_TRAINING,
            assigned_producer_id="acc-1",
            producer_handle="node-1",
            producer_timezone_offset=1.0,
            local_clock_hour=23.5,
            pouw_reward_hmt=12.5,
            estimated_tflops=370.0,
            is_byob=True,
            byob_bee_id="   ",
        )

    with pytest.raises(TypeError, match="router_store must be a RouterStore or None"):
        DiurnalMoETrainingScheduler(router_store="not-a-store")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="scanner must be a LongTailIntentScanner or None"):
        DiurnalMoETrainingScheduler(scanner="not-a-scanner")  # type: ignore[arg-type]


def test_crowdgpt_loss_check_job_and_router_hardening(tmp_path) -> None:
    job = LossCheckJob(
        job_id="job-1",
        base_url="https://example.org/base.bin",
        base_sha256="a" * 64,
        candidate_url="https://example.org/cand.bin",
        candidate_sha256="b" * 64,
        heldout_url="https://example.org/held.bin",
        heldout_sha256="c" * 64,
        window=16,
        n=8,
        seed=42,
        created_at=100.0,
        expires_at=200.0,
    )
    assert job.to_wire()["schema"] == JOB_SCHEMA

    with pytest.raises(ValueError, match="window must be an integer >= 2"):
        LossCheckJob(
            job_id="job-1",
            base_url="https://example.org/base.bin",
            base_sha256="a" * 64,
            candidate_url="https://example.org/cand.bin",
            candidate_sha256="b" * 64,
            heldout_url="https://example.org/held.bin",
            heldout_sha256="c" * 64,
            window=1,
            n=8,
            seed=42,
            created_at=100.0,
            expires_at=200.0,
        )
    with pytest.raises(ValueError, match="expires_at >= created_at"):
        LossCheckJob(
            job_id="job-1",
            base_url="https://example.org/base.bin",
            base_sha256="a" * 64,
            candidate_url="https://example.org/cand.bin",
            candidate_sha256="b" * 64,
            heldout_url="https://example.org/held.bin",
            heldout_sha256="c" * 64,
            window=16,
            n=8,
            seed=42,
            created_at=200.0,
            expires_at=100.0,
        )
    with pytest.raises(ValueError, match="unknown job schema"):
        LossCheckJob(
            job_id="job-1",
            base_url="https://example.org/base.bin",
            base_sha256="a" * 64,
            candidate_url="https://example.org/cand.bin",
            candidate_sha256="b" * 64,
            heldout_url="https://example.org/held.bin",
            heldout_sha256="c" * 64,
            window=16,
            n=8,
            seed=42,
            created_at=100.0,
            expires_at=200.0,
            schema="bad.schema",
        )

    coord = LossCheckCoordinator(
        evaluator_keys={"ev1": b"k" * 32},
        server_secret=b"s" * 32,
        state_path=tmp_path / "state.json",
        loader=lambda _j: (None, None, None),
    )
    with pytest.raises(TypeError, match="coord must be a LossCheckCoordinator"):
        fastapi_router("not-coord")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="prefix must be a non-empty string starting with '/'"):
        fastapi_router(coord, prefix="relative/prefix")


def test_crowdgpt_bridge_dataclasses_hardening() -> None:
    packed = pack_crowdgpt_delta_stream({"tokensProcessed": 128}, torch.zeros(8, dtype=torch.bfloat16))
    unpacked = unpack_crowdgpt_delta_stream(packed)
    assert len(unpacked.payload_sha256) == 64

    with pytest.raises(TypeError, match="base_delta_bf16 must be a bfloat16 torch.Tensor"):
        CrowdGPTDeltaPayload(
            metadata={},
            base_delta_bf16=torch.zeros(4, dtype=torch.float32),
            engram_indices=np.array([], dtype=np.uint32),
            engram_values_uint16=np.array([], dtype=np.uint16),
            payload_sha256="a" * 64,
        )
    with pytest.raises(ValueError, match="payload_sha256 must be a 64-character hex string"):
        CrowdGPTDeltaPayload(
            metadata={},
            base_delta_bf16=torch.zeros(4, dtype=torch.bfloat16),
            engram_indices=np.array([], dtype=np.uint32),
            engram_values_uint16=np.array([], dtype=np.uint16),
            payload_sha256="short",
        )

    res = multi_krum_select_and_aggregate([torch.ones(4, dtype=torch.bfloat16)])
    assert res.selected_indices == [0]

    with pytest.raises(ValueError, match="selected_indices must be a non-empty list"):
        MultiKrumAggregationResult(
            selected_indices=[],
            rejected_indices=[],
            krum_scores=[0.0],
            aggregated_delta_bf16=torch.zeros(4, dtype=torch.bfloat16),
            effective_tokens=1,
        )
    with pytest.raises(TypeError, match="aggregated_delta_bf16 must be a bfloat16 torch.Tensor"):
        MultiKrumAggregationResult(
            selected_indices=[0],
            rejected_indices=[],
            krum_scores=[0.0],
            aggregated_delta_bf16=torch.zeros(4, dtype=torch.float32),
            effective_tokens=1,
        )

    cohort = classify_gpu_precision_cohort("nvidia", (7, 5))
    assert cohort.pre_ampere_fp32_enforced is True

    with pytest.raises(ValueError, match="compute_capability must be a 2-tuple of non-negative integers"):
        GpuPrecisionCohort(
            vendor="nvidia",
            compute_capability=(-1, 5),
            effective_dtype="fp32",
            optimizer_family="adamw",
            pre_ampere_fp32_enforced=True,
            cohort_key="nvidia:sm75:fp32:adamw",
        )

    dispatch = resolve_forming_pool_starvation(current_members=2, waited_seconds=200.0)
    assert dispatch.mode == "SOLO_LOSS_RATCHET_FALLBACK"

    with pytest.raises(ValueError, match="target_pool_size must be a positive integer"):
        PoolDispatchDecision(
            mode="FORMING_WAIT",
            cohort_key="nvidia:sm75:fp32:adamw",
            active_members=1,
            target_pool_size=0,
            requires_multi_krum=True,
            requires_loss_ratchet=True,
        )


def test_amcp_kafka_campaign_bus_dataclasses_hardening() -> None:
    ev = build_amcp_kafka_event(
        topic_key="campaigns",
        event_type="io.hivemind.amcp.test",
        sender_id="bee-1",
        sender_kind="bee_specialist",
        now_utc=1700000000.0,
    )
    assert ev.specversion == "1.0"

    with pytest.raises(TypeError, match="payload must be a dict"):
        AMCPKafkaEvent(
            event_id="ev-1",
            topic="hivemind.amcp.campaigns.v1",
            event_type="io.hivemind.amcp.test",
            sender_id="bee-1",
            sender_kind="bee_specialist",
            recipient_id="broadcast",
            subject_id="SUBJ-01-PYTHON-RUST-CODER",
            correlation_id="corr-1",
            payload="not-a-dict",  # type: ignore[arg-type]
            utc=1700000000.0,
        )
    with pytest.raises(ValueError, match="utc must be a positive finite timestamp"):
        AMCPKafkaEvent(
            event_id="ev-1",
            topic="hivemind.amcp.campaigns.v1",
            event_type="io.hivemind.amcp.test",
            sender_id="bee-1",
            sender_kind="bee_specialist",
            recipient_id="broadcast",
            subject_id="SUBJ-01-PYTHON-RUST-CODER",
            correlation_id="corr-1",
            payload={},
            utc=0.0,
        )

    with pytest.raises(ValueError, match="co_bees must be a non-empty tuple of non-empty strings"):
        SubjectCampaign(
            subject_id="SUBJ-X",
            order_index=1,
            title="Test",
            domain_key="coding_python",
            lead_bee="hivemind/coder",
            co_bees=(),
            dtstart_utc="20261028T180000Z",
            dtend_utc="20261030T220000Z",
            human_window="window",
            objective="obj",
            quantitative_gate="gate",
            dataset_ref="data/ref.jsonl",
            base_nocturnal_weight=0.2,
        )
    with pytest.raises(ValueError, match="base_nocturnal_weight must be a finite number in \\(0.0, 1.0\\]"):
        SubjectCampaign(
            subject_id="SUBJ-X",
            order_index=1,
            title="Test",
            domain_key="coding_python",
            lead_bee="hivemind/coder",
            co_bees=("hivemind/moe-coder",),
            dtstart_utc="20261028T180000Z",
            dtend_utc="20261030T220000Z",
            human_window="window",
            objective="obj",
            quantitative_gate="gate",
            dataset_ref="data/ref.jsonl",
            base_nocturnal_weight=1.5,
        )
