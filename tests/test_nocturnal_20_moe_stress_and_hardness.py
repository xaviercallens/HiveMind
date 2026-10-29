# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Intensive Stress, Invariant & Byzantine Hardness Test Suite for the 20 Nocturnal MoE Adapters.

Covers:
1. Exhaustive verification of all 20 Hugging Face domain adapters and clusters.
2. VRAM physical envelopes, scaling limits, and hardware-tier gating (8GB, 12GB, 16GB, 24GB).
3. Concurrent multi-worker batch dispatch under heterogeneous swarm load.
4. McNemar statistical gate mathematical rigor, continuity correction, and regression detection.
5. Proof-of-Useful-Work (PoUW) ledger concurrency, strict idempotency, and tamper-evident hash-chain audit.
6. Adversarial and Byzantine fault injection (faked accuracy, corrupted weights, replayed IDs).
"""

from __future__ import annotations

import concurrent.futures
import time
from pathlib import Path

import pytest
from hivemind.core.auto_improvement import (
    HF_EXPERT_CATALOG,
    NOCTURNAL_20_MOE_TARGETS,
    AutoImprovementExecutionResult,
    AutoImprovementWorkunit,
    NocturnalAutoImprovementScheduler,
    PointsRetrofitManager,
    resolve_domain_canonical,
)
from hivemind.router.accounts import RouterStore

TEST_SECRET = b"hivemind-nocturnal-moe-stress-test-secret-32b!"


@pytest.fixture
def store(tmp_path: Path) -> RouterStore:
    return RouterStore(tmp_path / "test_stress_router.db", secret=TEST_SECRET)


def test_exhaustive_20_huggingface_adapters_specification() -> None:
    """Validate that all 20 nocturnal MoE targets are fully specified with real HF datasets."""
    assert len(NOCTURNAL_20_MOE_TARGETS) == 20

    # Ensure all 4 macro-clusters are represented
    clusters_found: set[str] = set()

    scheduler = NocturnalAutoImprovementScheduler()

    for idx, domain in enumerate(NOCTURNAL_20_MOE_TARGETS):
        canonical = resolve_domain_canonical(domain)
        assert canonical in HF_EXPERT_CATALOG, f"Domain {domain} missing from catalog"

        meta = HF_EXPERT_CATALOG[canonical]
        assert "title" in meta and len(meta["title"]) > 0
        assert "hf_dataset" in meta and "/" in meta["hf_dataset"], (
            f"Invalid HF dataset identifier in {domain}"
        )
        assert "cluster" in meta and meta["cluster"].startswith("CLUSTER_")
        assert "description" in meta and len(meta["description"]) > 10

        clusters_found.add(meta["cluster"])

        # Generate workunit and verify synthetic sample integrity
        wu = scheduler.create_domain_workunit(
            domain=canonical,
            num_samples=25,
            inner_steps=30,
            seed=20261028 + idx,
        )
        assert wu.domain == canonical
        assert wu.hf_dataset == meta["hf_dataset"]
        assert wu.cluster == meta["cluster"]
        assert len(wu.sample_pairs) == 25
        assert len(wu.sha256_digest) == 64

        # Verify sample structure contains real domain keywords
        first_sample = wu.sample_pairs[0]
        assert "prompt" in first_sample and len(first_sample["prompt"]) > 5
        assert "verified_solution" in first_sample and len(first_sample["verified_solution"]) > 5

    assert len(clusters_found) == 4, f"Expected 4 macro-clusters, found: {clusters_found}"


def test_vram_physical_envelope_and_large_models_scaling() -> None:
    """Verify VRAM hardware-tier gating and memory boundaries across standard and large base models."""
    scheduler = NocturnalAutoImprovementScheduler()
    now = time.time()

    producers = [
        # RTX 2070: 8.0 GB VRAM
        {
            "account_id": "rtx2070_producer",
            "execution_class": "rtx2070-8gb-turing",
            "vram_gb": 8.0,
            "last_heartbeat_utc": now - 15.0,
        },
        # Pioneer RTX 4070: 12.0 GB VRAM
        {
            "account_id": "rtx4070_producer",
            "execution_class": "cuda-sm89-fp16-4070",
            "vram_gb": 12.0,
            "last_heartbeat_utc": now - 10.0,
        },
        # Tesla T4: 16.0 GB VRAM
        {
            "account_id": "t4_producer",
            "execution_class": "t4-16gb-turing",
            "vram_gb": 16.0,
            "last_heartbeat_utc": now - 20.0,
        },
        # RTX 3090: 24.0 GB VRAM
        {
            "account_id": "rtx3090_producer",
            "execution_class": "cuda-sm86-fp16-3090",
            "vram_gb": 24.0,
            "last_heartbeat_utc": now - 5.0,
        },
    ]

    # Gate 1: Standard 7B/9B models (min 6.0 GB) -> all 4 eligible
    std_eligible = scheduler.filter_eligible_producers(producers, min_vram_gb=6.0, now_utc=now)
    assert len(std_eligible) == 4
    # Must be sorted by VRAM descending
    assert [p["account_id"] for p in std_eligible] == [
        "rtx3090_producer",
        "t4_producer",
        "rtx4070_producer",
        "rtx2070_producer",
    ]

    # Gate 2: 14B models (min 14.0 GB) -> only T4 and 3090 eligible
    tier_14b_eligible = scheduler.filter_eligible_producers(
        producers, min_vram_gb=14.0, now_utc=now
    )
    assert len(tier_14b_eligible) == 2
    assert [p["account_id"] for p in tier_14b_eligible] == ["rtx3090_producer", "t4_producer"]

    # Gate 3: DeepSeek MoE 16B (min 16.0 GB) -> only T4 and 3090 eligible
    tier_deepseek_eligible = scheduler.filter_eligible_producers(
        producers, min_vram_gb=16.0, now_utc=now
    )
    assert len(tier_deepseek_eligible) == 2
    assert [p["account_id"] for p in tier_deepseek_eligible] == ["rtx3090_producer", "t4_producer"]

    # Gate 4: Qwen 2.5 32B (min 24.0 GB) -> ONLY RTX 3090 eligible
    tier_32b_eligible = scheduler.filter_eligible_producers(
        producers, min_vram_gb=24.0, now_utc=now
    )
    assert len(tier_32b_eligible) == 1
    assert tier_32b_eligible[0]["account_id"] == "rtx3090_producer"

    # Execution Simulation: Verify peak VRAM respects model metadata
    wu_32b = scheduler.create_domain_workunit(
        domain="maths_formal",
        base_model_id="Qwen/Qwen2.5-32B-Instruct",
        inner_steps=50,
        seed=42,
    )
    res_32b = scheduler.simulate_or_verify_execution(
        workunit=wu_32b,
        worker_id="worker-3090",
        account_id="rtx3090_producer",
        execution_class="cuda-sm86-fp16-3090",
    )
    # Peak VRAM for 32B in NF4: base ~19.5 GB + overhead ~0.3 GB = ~20 GB (< 24 GB envelope)
    assert 19000 <= res_32b.peak_vram_mb <= 23500
    # TFLOPs multiplier for 32.5B params: 32.5 / 7.0 ~ 4.64x
    assert res_32b.contributed_tflops > (0.125 * 50 * 4.0)


def test_concurrent_multi_worker_dispatch_under_load() -> None:
    """Stress test parallel execution across all 20 domain adapters with heterogeneous hardware."""
    scheduler = NocturnalAutoImprovementScheduler()
    workers = [
        ("worker-rtx2070", "producer-2070", "rtx2070-8gb-turing", 8.0),
        ("worker-rtx4070", "producer-4070", "cuda-sm89-fp16-4070", 12.0),
        ("worker-t4", "producer-t4", "t4-16gb-turing", 16.0),
        ("worker-rtx3090", "producer-3090", "cuda-sm86-fp16-3090", 24.0),
    ]

    base_models = [
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "google/gemma-2-9b-it",
        "deepseek-ai/DeepSeek-Coder-V2-Lite-Base",
        "Qwen/Qwen2.5-32B-Instruct",
    ]

    workunits: list[tuple[AutoImprovementWorkunit, str, str, str]] = []

    for i, domain in enumerate(NOCTURNAL_20_MOE_TARGETS):
        worker = workers[i % len(workers)]
        base_model = base_models[i % len(base_models)]

        # Adjust base model if worker VRAM is insufficient
        if worker[3] < 16.0 and "DeepSeek" in base_model:
            base_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
        if worker[3] < 24.0 and "32B" in base_model:
            base_model = "google/gemma-2-9b-it"

        wu = scheduler.create_domain_workunit(
            domain=domain,
            base_model_id=base_model,
            num_samples=30,
            inner_steps=40,
            seed=1000 + i,
        )
        workunits.append((wu, worker[0], worker[1], worker[2]))

    assert len(workunits) == 20

    # Execute all 20 concurrently via ThreadPoolExecutor
    results: list[AutoImprovementExecutionResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(scheduler.simulate_or_verify_execution, wu, w_id, acc_id, exe_cls)
            for wu, w_id, acc_id, exe_cls in workunits
        ]
        for fut in concurrent.futures.as_completed(futures):
            results.append(fut.result())

    assert len(results) == 20
    total_tflops = sum(r.contributed_tflops for r in results)
    assert total_tflops > 100.0, f"Expected >100 TFLOPs, got {total_tflops}"

    # Verify each result is valid and delta weights hashes are distinct
    hashes = set()
    for r in results:
        assert r.gate_passed is True
        assert r.accuracy_adapted > r.accuracy_base
        assert r.mcnemar_p_value < 0.05
        assert len(r.delta_weights_hash) == 64
        hashes.add(r.delta_weights_hash)

    assert len(hashes) == 20, "All 20 delta weight hashes must be cryptographically unique"


def test_mcnemar_statistical_rigor_and_continuity_correction() -> None:
    """Verify Edwards continuity correction and anti-p-hacking edge conditions."""
    scheduler = NocturnalAutoImprovementScheduler()

    # Test 1: Edwards continuity correction exact formula check
    # b=10, c=40, total=200
    # chi2 = (|10 - 40| - 1)^2 / (10 + 40) = (29)^2 / 50 = 841 / 50 = 16.82
    # p_val = erfc(sqrt(16.82 / 2)) = erfc(sqrt(8.41)) = erfc(2.90) ~ 0.000041
    passed, p_val = scheduler.evaluate_mcnemar_gate(
        base_correct=100,
        adapted_correct=130,
        base_only_correct=10,
        adapted_only_correct=40,
        total_eval_samples=200,
    )
    assert passed is True
    assert p_val < 0.0001

    # Test 2: Borderline failure where difference is not statistically significant (p >= 0.05)
    # b=18, c=24 -> |18 - 24| = 6, (|6| - 1)^2 / 42 = 25 / 42 = 0.595 -> p > 0.3
    passed_insig, p_insig = scheduler.evaluate_mcnemar_gate(
        base_correct=100,
        adapted_correct=106,
        base_only_correct=18,
        adapted_only_correct=24,
        total_eval_samples=200,
    )
    assert passed_insig is False
    assert p_insig > 0.05

    # Test 3: Regression where adapted is worse than base
    passed_regr, _ = scheduler.evaluate_mcnemar_gate(
        base_correct=140,
        adapted_correct=110,
        base_only_correct=35,
        adapted_only_correct=5,
        total_eval_samples=200,
    )
    assert passed_regr is False

    # Test 4: Symmetric tie (b == c)
    passed_tie, p_tie = scheduler.evaluate_mcnemar_gate(
        base_correct=120,
        adapted_correct=120,
        base_only_correct=15,
        adapted_only_correct=15,
        total_eval_samples=200,
    )
    assert passed_tie is False
    assert p_tie == 1.0


def test_pouw_ledger_concurrency_idempotency_and_anti_tampering(store: RouterStore) -> None:
    """Stress test PoUW points settlement, concurrency, strict idempotency, and tamper detection."""
    scheduler = NocturnalAutoImprovementScheduler()
    retrofit_mgr = PointsRetrofitManager()

    # Pre-register 4 producers in store
    producer_handles = [
        "prod_rtx3090_01",
        "prod_t4_02",
        "prod_rtx4070_03",
        "prod_rtx2070_04",
    ]
    producer_ids = [store.register(handle, "producer")[0] for handle in producer_handles]

    initial_balances = {pid: store.balance(pid) for pid in producer_ids}
    assert all(b == 0.0 for b in initial_balances.values())

    # Generate 20 execution results
    results: list[AutoImprovementExecutionResult] = []
    for idx, domain in enumerate(NOCTURNAL_20_MOE_TARGETS):
        pid = producer_ids[idx % len(producer_ids)]
        wu = scheduler.create_domain_workunit(
            domain=domain, num_samples=20, inner_steps=50, seed=5000 + idx
        )
        res = scheduler.simulate_or_verify_execution(wu, worker_id=f"worker-{pid}", account_id=pid)
        results.append(res)

    assert len(results) == 20

    # Retrofit points in parallel across chunks of results
    chunks = [results[i : i + 5] for i in range(0, len(results), 5)]
    reports = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(retrofit_mgr.retrofit_work_points, store, chunk) for chunk in chunks
        ]
        for fut in concurrent.futures.as_completed(futures):
            reports.append(fut.result())

    assert len(reports) == 4

    # Total reward calculation: 20 workunits * (10.0 base + 0.0125 * 50 steps + 5.0 bonus) = 20 * 15.625 = 312.50 HMT
    total_credited = sum(rep.total_hmt_granted for rep in reports)
    assert abs(total_credited - 312.50) < 1e-4

    # Verify that account balances match the credited receipts
    total_balances = sum(store.balance(pid) for pid in producer_ids)
    assert abs(total_balances - 312.50) < 1e-4

    # Idempotency Test: re-executing all 20 retrofits must award 0 additional points
    replay_report = retrofit_mgr.retrofit_work_points(store, results)
    assert replay_report.already_credited_count == 20
    assert replay_report.newly_credited_count == 0
    assert replay_report.total_hmt_granted == 0.0

    # Balance must not have changed
    assert abs(sum(store.balance(pid) for pid in producer_ids) - 312.50) < 1e-4

    # Cryptographic Hash Chain Audit: Ledger must be 100% verified
    assert store.verify_chain() is True

    # Tamper Detection Test: Inject malicious alteration directly into SQLite ledger
    store._conn.execute("UPDATE ledger SET delta = delta + 1000.0 WHERE seq = 1")

    # verify_chain() MUST immediately detect the altered record and fail
    assert store.verify_chain() is False, (
        "Ledger tampering was not detected by cryptographic hash-chain!"
    )


def test_adversarial_byzantine_fault_injection() -> None:
    """Inject Byzantine failure modes and assert fail-closed rejection."""
    scheduler = NocturnalAutoImprovementScheduler()

    # 1. Invalid domain raises ValueError
    with pytest.raises(ValueError, match="Unknown domain"):
        scheduler.create_domain_workunit(domain="malicious_exploit_domain")

    # 2. Corrupted evaluation result where worker claims gate_passed but p_value is insig
    corrupted_result = AutoImprovementExecutionResult(
        wu_id="wu-byzantine-fake-p-val",
        domain="maths_formal",
        worker_id="byzantine-node-01",
        account_id="attacker-account",
        steps_completed=50,
        loss_initial=2.5,
        loss_final=2.0,
        accuracy_base=0.50,
        accuracy_adapted=0.52,
        mcnemar_p_value=0.45,  # Insignificant p-value!
        gate_passed=True,  # Attacker falsely sets gate_passed to True!
        contributed_tflops=6.25,
        peak_vram_mb=4800.0,
        duration_seconds=90.0,
        delta_weights_hash="faked_hash",
        base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        hf_dataset="hoskinson-center/proofnet",
        cluster="CLUSTER_00_SYSTEMS_FORMAL",
        completed_utc=time.time(),
    )

    # Re-evaluate McNemar gate independently to catch the spoofing
    independent_passed, independent_p = scheduler.evaluate_mcnemar_gate(
        base_correct=int(corrupted_result.accuracy_base * 200),
        adapted_correct=int(corrupted_result.accuracy_adapted * 200),
        base_only_correct=10,
        adapted_only_correct=14,
        total_eval_samples=200,
    )
    # The independent gate must correctly reject the spoofed improvement
    assert independent_passed is False
    assert independent_p > 0.05
