# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Input-validation hardening tests for router opentrainer, producer agent, market, and benchmarks page."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.router.benchmarks_page import (
    load_measured_results,
    render_hivemind_capacity_infographic_svg,
)
from hivemind.router.hivemind_producer_agent import (
    build_heartbeat_payload,
    recommend,
)
from hivemind.router.market import (
    ConsumerBid,
    FeePolicy,
    ProducerOffer,
    certify,
    clear_market,
    split_payment,
)
from hivemind.router.opentrainer import (
    default_active_task_offer,
    validate_trainer_delta_submission,
)


def test_default_active_task_offer_rejects_invalid_inputs() -> None:
    offer = default_active_task_offer(now_utc=1700000000.0, round_index=3, quorum=2)
    assert offer["round"] == 3
    assert offer["quorum"] == 2

    with pytest.raises(ValueError, match="now_utc"):
        default_active_task_offer(now_utc=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="now_utc"):
        default_active_task_offer(now_utc=float("nan"))
    with pytest.raises(ValueError, match="round_index"):
        default_active_task_offer(round_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="round_index"):
        default_active_task_offer(round_index=-1)
    with pytest.raises(ValueError, match="quorum"):
        default_active_task_offer(quorum=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="quorum"):
        default_active_task_offer(quorum=0)


def test_validate_trainer_delta_submission_rejects_bool_and_non_finite() -> None:
    res = validate_trainer_delta_submission(
        run_id="opentrainer-rtx2070-t4-co-train-v1",
        round_index=0,
        delta_l2=0.5,
        eval_loss=1.2,
        tokens_trained=512,
        vram_free_mib=2048,
        vram_total_mib=8192,
    )
    assert res["accepted"] is True

    with pytest.raises(ValueError, match="round_index"):
        validate_trainer_delta_submission(
            run_id="r1",
            round_index=True,  # type: ignore[arg-type]
            delta_l2=0.5,
            eval_loss=1.2,
            tokens_trained=512,
            vram_free_mib=2048,
            vram_total_mib=8192,
        )
    with pytest.raises(ValueError, match="delta_l2"):
        validate_trainer_delta_submission(
            run_id="r1",
            round_index=0,
            delta_l2=False,  # type: ignore[arg-type]
            eval_loss=1.2,
            tokens_trained=512,
            vram_free_mib=2048,
            vram_total_mib=8192,
        )
    with pytest.raises(ValueError, match="max_delta_l2"):
        validate_trainer_delta_submission(
            run_id="r1",
            round_index=0,
            delta_l2=0.5,
            eval_loss=1.2,
            tokens_trained=512,
            vram_free_mib=2048,
            vram_total_mib=8192,
            max_delta_l2=0.0,
        )
    with pytest.raises(ValueError, match="eval_loss"):
        validate_trainer_delta_submission(
            run_id="r1",
            round_index=0,
            delta_l2=0.5,
            eval_loss=True,  # type: ignore[arg-type]
            tokens_trained=512,
            vram_free_mib=2048,
            vram_total_mib=8192,
        )
    with pytest.raises(ValueError, match="tokens_trained"):
        validate_trainer_delta_submission(
            run_id="r1",
            round_index=0,
            delta_l2=0.5,
            eval_loss=1.2,
            tokens_trained=True,  # type: ignore[arg-type]
            vram_free_mib=2048,
            vram_total_mib=8192,
        )
    with pytest.raises(ValueError, match="vram_free_mib"):
        validate_trainer_delta_submission(
            run_id="r1",
            round_index=0,
            delta_l2=0.5,
            eval_loss=1.2,
            tokens_trained=512,
            vram_free_mib=True,  # type: ignore[arg-type]
            vram_total_mib=8192,
        )


def test_producer_agent_heartbeat_and_recommend_reject_invalid_inputs() -> None:
    payload = build_heartbeat_payload(
        endpoint="https://producer.example.com",
        models=[{"model_id": "hivemind/coder", "served_by": "qwen2.5-coder:7b"}],
        cost_per_1k=0.4,
        capacity_1k=60.0,
        execution_class="trusted-community",
    )
    assert payload["endpoint_url"] == "https://producer.example.com"

    with pytest.raises(ValueError, match="endpoint"):
        build_heartbeat_payload(
            endpoint="   ",
            models=[{"model_id": "hivemind/coder", "served_by": "qwen2.5-coder:7b"}],
            cost_per_1k=0.4,
            capacity_1k=60.0,
            execution_class="trusted-community",
        )
    with pytest.raises(ValueError, match="models"):
        build_heartbeat_payload(
            endpoint="https://producer.example.com",
            models=[],
            cost_per_1k=0.4,
            capacity_1k=60.0,
            execution_class="trusted-community",
        )
    with pytest.raises(ValueError, match="cost_per_1k"):
        build_heartbeat_payload(
            endpoint="https://producer.example.com",
            models=[{"model_id": "hivemind/coder", "served_by": "qwen2.5-coder:7b"}],
            cost_per_1k=True,  # type: ignore[arg-type]
            capacity_1k=60.0,
            execution_class="trusted-community",
        )
    with pytest.raises(ValueError, match="capacity_1k"):
        build_heartbeat_payload(
            endpoint="https://producer.example.com",
            models=[{"model_id": "hivemind/coder", "served_by": "qwen2.5-coder:7b"}],
            cost_per_1k=0.4,
            capacity_1k=-1.0,
            execution_class="trusted-community",
        )
    with pytest.raises(ValueError, match="execution_class"):
        build_heartbeat_payload(
            endpoint="https://producer.example.com",
            models=[{"model_id": "hivemind/coder", "served_by": "qwen2.5-coder:7b"}],
            cost_per_1k=0.4,
            capacity_1k=60.0,
            execution_class="",
        )

    tiers = [{"tier_id": "coder", "min_vram_gb": 8.0}]
    assert recommend(tiers, 12.0) == tiers[0]
    with pytest.raises(ValueError, match="vram_gb"):
        recommend(tiers, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vram_gb"):
        recommend(tiers, -1.0)


def test_benchmarks_page_rejects_invalid_path_and_telemetry(tmp_path: Path) -> None:
    assert isinstance(load_measured_results(tmp_path / "missing.json"), list)
    with pytest.raises(ValueError, match="path"):
        load_measured_results("not-a-path")  # type: ignore[arg-type]

    assert "<svg" in render_hivemind_capacity_infographic_svg({})
    with pytest.raises(ValueError, match="live_telemetry"):
        render_hivemind_capacity_infographic_svg("bad")  # type: ignore[arg-type]


def test_market_rejects_bool_inputs_and_invalid_fee_policy() -> None:
    with pytest.raises(ValueError, match="validator_share"):
        FeePolicy(validator_share=False)  # type: ignore[arg-type]

    bids = [ConsumerBid("c1", 2.0), ConsumerBid("c2", 2.0)]
    offers = [ProducerOffer("p1", 0.5, 10.0)]
    eq = clear_market("hivemind/coder", bids, offers)
    assert eq.price_per_1k > 0.0

    with pytest.raises(ValueError, match="model_id"):
        clear_market(True, bids, offers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="measured_load_1k"):
        clear_market("hivemind/coder", bids, offers, measured_load_1k=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="consumer account ids"):
        clear_market("hivemind/coder", [ConsumerBid(True, 2.0)], offers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_price_per_1k"):
        clear_market("hivemind/coder", [ConsumerBid("c1", True)], offers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="producer account ids"):
        clear_market("hivemind/coder", bids, [ProducerOffer(True, 0.5, 10.0)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cost_per_1k"):
        clear_market("hivemind/coder", bids, [ProducerOffer("p1", False, 10.0)])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capacity_1k"):
        clear_market("hivemind/coder", bids, [ProducerOffer("p1", 0.5, True)])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="price_per_1k"):
        split_payment(True, 100, FeePolicy())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="fee must be a FeePolicy"):
        split_payment(1.0, 100, None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="measured_load_1k"):
        certify(eq, bids, offers, measured_load_1k=False)  # type: ignore[arg-type]
