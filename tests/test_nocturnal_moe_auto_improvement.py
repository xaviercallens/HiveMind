"""Tests for Nocturnal MoE Auto-Improvement & Proof-of-Useful-Work (PoUW) Points Retrofit.

Verifies:
1. Night-window detection (European Night 22:00-07:00 CET).
2. Dynamic producer filtering replacing preempted GCP T4 VMs (< 8 GB VRAM envelope).
3. Auto-improvement workunit generation across 5 domains (maths, exoplanet, stars, cyber, coder).
4. Paired McNemar improvement gate verification.
5. PoUW HMT Reflexion Points calculation, idempotent retrofitting, and hash-chain preservation.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from hivemind.core.auto_improvement import (
    AutoImprovementExecutionResult,
    NocturnalAutoImprovementScheduler,
    PointsRetrofitManager,
    SUPPORTED_DOMAINS,
)
from hivemind.router.accounts import RouterStore

TEST_SECRET = b"hivemind-auto-improvement-test-key-32b!"


@pytest.fixture
def store(tmp_path: Path) -> RouterStore:
    return RouterStore(tmp_path / "test_router.db", secret=TEST_SECRET)


def test_night_window_detection() -> None:
    # 23:30 CET (22:30 UTC) should be night
    utc_night = 1730068200.0  # Approx 22:30 UTC
    assert NocturnalAutoImprovementScheduler.is_european_night(utc_night) is True

    # 14:00 CET (13:00 UTC) should be day
    utc_day = 1730034000.0   # Approx 13:00 UTC
    assert NocturnalAutoImprovementScheduler.is_european_night(utc_day) is False


def test_filter_eligible_producers_replaces_preempted_t4() -> None:
    scheduler = NocturnalAutoImprovementScheduler()
    now = time.time()

    producers = [
        # RTX 2070 producer (8 GB, fresh)
        {
            "account_id": "rtx2070_account",
            "execution_class": "rtx2070-8gb-turing",
            "vram_gb": 8.0,
            "last_heartbeat_utc": now - 30.0,
        },
        # RTX 3090 producer (24 GB, fresh)
        {
            "account_id": "rtx3090_account",
            "execution_class": "cuda-sm86-fp16-3090",
            "vram_gb": 24.0,
            "last_heartbeat_utc": now - 10.0,
        },
        # Stale producer (heartbeat 600s ago)
        {
            "account_id": "stale_account",
            "execution_class": "cuda-sm89-4090",
            "vram_gb": 24.0,
            "last_heartbeat_utc": now - 600.0,
        },
        # Under-spec producer (4 GB VRAM)
        {
            "account_id": "low_vram_account",
            "execution_class": "cuda-sm75-gtx1650",
            "vram_gb": 4.0,
            "last_heartbeat_utc": now - 15.0,
        },
    ]

    eligible = scheduler.filter_eligible_producers(producers, min_vram_gb=6.0, now_utc=now)
    assert len(eligible) == 2
    # Ranked by VRAM descending: 3090 first, 2070 second
    assert eligible[0]["account_id"] == "rtx3090_account"
    assert eligible[1]["account_id"] == "rtx2070_account"


def test_domain_workunit_generation() -> None:
    scheduler = NocturnalAutoImprovementScheduler()
    for domain in SUPPORTED_DOMAINS:
        wu = scheduler.create_domain_workunit(
            domain=domain,
            num_samples=20,
            inner_steps=25,
            seed=1234,
        )
        assert len(wu.sample_pairs) == 20
        assert wu.inner_steps == 25
        assert wu.min_vram_gb <= 8.0
        assert len(wu.sha256_digest) == 64
        assert len(wu.hf_dataset) > 0
        assert len(wu.cluster) > 0


def test_multi_base_model_and_24_experts() -> None:
    from hivemind.core.auto_improvement import ALL_DOMAINS, HF_EXPERT_CATALOG, SUPPORTED_BASE_MODELS

    assert len(ALL_DOMAINS) == 24
    for domain, meta in HF_EXPERT_CATALOG.items():
        assert "hf_dataset" in meta
        assert "cluster" in meta

    # Test workunit creation with Gemma and Qwen 3.5
    for model_key in ("gemma-2-9b", "gemma-3-4b", "gemma-3-1b", "qwen-3.5-9b", "qwen-2.5-7b", "qwen-2.5-coder-7b"):
        cfg = SUPPORTED_BASE_MODELS[model_key]
        assert cfg["vram_nf4_gb"] < 8.0  # Must fit under 8 GB ceiling

        scheduler = NocturnalAutoImprovementScheduler(base_model_id=cfg["model_id"])
        wu = scheduler.create_domain_workunit("maths_formal", num_samples=10, inner_steps=20)
        assert wu.base_model_id == cfg["model_id"]
        assert wu.hf_dataset == "hoskinson-center/proofnet"
        assert wu.cluster == "CLUSTER_00_SYSTEMS_FORMAL"



def test_mcnemar_improvement_gate() -> None:
    scheduler = NocturnalAutoImprovementScheduler()

    # Case 1: Substantial improvement with discordant pairs favoring adapted (c >> b)
    # n=200, base=100, adapted=160, b=5 (base only), c=65 (adapted only)
    passed, p_val = scheduler.evaluate_mcnemar_gate(
        base_correct=100,
        adapted_correct=160,
        base_only_correct=5,
        adapted_only_correct=65,
        total_eval_samples=200,
    )
    assert passed is True
    assert p_val < 0.001

    # Case 2: Regression (adapted worse than base: b > c)
    passed_reg, p_val_reg = scheduler.evaluate_mcnemar_gate(
        base_correct=150,
        adapted_correct=120,
        base_only_correct=40,
        adapted_only_correct=10,
        total_eval_samples=200,
    )
    assert passed_reg is False

    # Case 3: No difference (b=0, c=0)
    passed_tie, p_val_tie = scheduler.evaluate_mcnemar_gate(
        base_correct=100,
        adapted_correct=100,
        base_only_correct=0,
        adapted_only_correct=0,
        total_eval_samples=200,
    )
    assert passed_tie is False
    assert p_val_tie == 1.0


def test_points_retrofit_and_strict_idempotency(store: RouterStore) -> None:
    retrofit_mgr = PointsRetrofitManager(base_wu_reward=10.0, step_rate=0.0125, gate_bonus=5.0)

    # 1. Create simulated results for 2 workers
    r1 = AutoImprovementExecutionResult(
        wu_id="wu-test-maths-001",
        domain="maths",
        worker_id="worker-rtx2070",
        account_id="producer_rtx2070_id_hash",
        steps_completed=50,
        loss_initial=2.5,
        loss_final=1.8,
        accuracy_base=0.52,
        accuracy_adapted=0.88,
        mcnemar_p_value=0.0001,
        gate_passed=True,
        contributed_tflops=6.25,
        peak_vram_mb=4100.0,
        duration_seconds=95.0,
        delta_weights_hash="hash123",
    )

    r2 = AutoImprovementExecutionResult(
        wu_id="wu-test-stars-002",
        domain="stars",
        worker_id="worker-rtx3090",
        account_id="producer_rtx3090_id_hash",
        steps_completed=40,
        loss_initial=2.4,
        loss_final=1.9,
        accuracy_base=0.50,
        accuracy_adapted=0.82,
        mcnemar_p_value=0.0005,
        gate_passed=True,
        contributed_tflops=5.0,
        peak_vram_mb=4200.0,
        duration_seconds=75.0,
        delta_weights_hash="hash456",
    )

    # 2. First retrofit pass: both should be newly credited
    report1 = retrofit_mgr.retrofit_work_points(store, [r1, r2])
    assert report1.total_workunits == 2
    assert report1.newly_credited_count == 2
    assert report1.already_credited_count == 0
    assert report1.ledger_verified is True

    # Expected rewards:
    # r1: 10.0 + 50 * 0.0125 (0.625) + 5.0 = 15.625 -> round 15.625
    # r2: 10.0 + 40 * 0.0125 (0.5) + 5.0 = 15.5
    assert report1.total_hmt_granted == pytest.approx(15.625 + 15.5, abs=0.01)

    bal1 = store.balance("producer_rtx2070_id_hash")
    bal2 = store.balance("producer_rtx3090_id_hash")
    assert bal1 == pytest.approx(15.625, abs=0.01)
    assert bal2 == pytest.approx(15.5, abs=0.01)

    # 3. Second retrofit pass (exact same results): should be idempotent and skipped!
    report2 = retrofit_mgr.retrofit_work_points(store, [r1, r2])
    assert report2.total_workunits == 2
    assert report2.newly_credited_count == 0
    assert report2.already_credited_count == 2
    assert report2.total_hmt_granted == 0.0
    assert report2.ledger_verified is True

    # Balances must remain unchanged!
    assert store.balance("producer_rtx2070_id_hash") == bal1
    assert store.balance("producer_rtx3090_id_hash") == bal2

    # Entire cryptographic chain in SQLite must be valid
    assert store.verify_chain() is True
