# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path unit tests for S36 ZK-OPoI v5, DARL, Hourly Orchestrator, and Circuit."""

from __future__ import annotations

import pytest

from hivemind.swarm.s36_01_zk_opoi_v5_int8_verifier import (
    MERSENNE_61,
    ZKOPoIV5Int8TransitionVerifier,
    ZKOPoIV5VerificationReport,
    _derive_field_challenges,
    _mod_mersenne61,
)
from hivemind.swarm.s36_05_darl_cas_shard_leasing import (
    DARLLedgerSnapshot,
    DARLShardLeaseCoordinator,
    ShardLeaseRecord,
)
from hivemind.swarm.s36_06_hourly_coding_assistant_orchestrator import (
    CodingAssistantTaskReceipt,
    HourlyImprovementCycleReport,
)
from hivemind.swarm.zk_opoi_v5_circuit import (
    MERSENNE_PRIME_61,
    ZKOPoIV5FixedPointCircuit,
    ZKOPoIV5VerificationResult,
    _derive_fiat_shamir_challenge,
    _field_mod,
    _poly_eval_field,
)


class TestS3601ZKOPoIV5Int8VerifierHardening:
    def test_mod_mersenne61_and_derive_challenges_validation(self) -> None:
        assert _mod_mersenne61(MERSENNE_61 + 7) == 7
        assert _mod_mersenne61(-5) == MERSENNE_61 - 5
        with pytest.raises(ValueError, match="val must be an integer"):
            _mod_mersenne61(True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="val must be an integer"):
            _mod_mersenne61(1.5)  # type: ignore[arg-type]

        ch = _derive_field_challenges(b"seed", 5)
        assert len(ch) == 5 and all(1 <= x < MERSENNE_61 for x in ch)
        with pytest.raises(ValueError, match="seed_bytes must be non-empty bytes"):
            _derive_field_challenges(b"", 3)
        with pytest.raises(ValueError, match="seed_bytes must be non-empty bytes"):
            _derive_field_challenges("seed", 3)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="count must be a positive integer"):
            _derive_field_challenges(b"seed", 0)
        with pytest.raises(ValueError, match="count must be a positive integer"):
            _derive_field_challenges(b"seed", True)  # type: ignore[arg-type]

    def test_verification_report_validation(self) -> None:
        rep = ZKOPoIV5VerificationReport(
            workunit_id="wu-1",
            verified=True,
            verify_cpu_ms=0.35,
            proof_bytes=512,
            within_5ms_sla=True,
            within_32kb_bound=True,
            rejection_reason=None,
        )
        assert rep.to_dict()["verified"] is True

        with pytest.raises(ValueError, match="workunit_id must be a non-empty string"):
            ZKOPoIV5VerificationReport(
                workunit_id=" ",
                verified=True,
                verify_cpu_ms=0.35,
                proof_bytes=512,
                within_5ms_sla=True,
                within_32kb_bound=True,
            )
        with pytest.raises(ValueError, match="commitment_scheme must be a non-empty string"):
            ZKOPoIV5VerificationReport(
                workunit_id="wu-1",
                verified=True,
                verify_cpu_ms=0.35,
                proof_bytes=512,
                within_5ms_sla=True,
                within_32kb_bound=True,
                commitment_scheme="",
            )
        with pytest.raises(ValueError, match="verified must be a boolean"):
            ZKOPoIV5VerificationReport(
                workunit_id="wu-1",
                verified=1,  # type: ignore[arg-type]
                verify_cpu_ms=0.35,
                proof_bytes=512,
                within_5ms_sla=True,
                within_32kb_bound=True,
            )
        with pytest.raises(ValueError, match="verify_cpu_ms must be a finite non-negative number"):
            ZKOPoIV5VerificationReport(
                workunit_id="wu-1",
                verified=True,
                verify_cpu_ms=-0.1,
                proof_bytes=512,
                within_5ms_sla=True,
                within_32kb_bound=True,
            )
        with pytest.raises(ValueError, match="proof_bytes must be a non-negative integer"):
            ZKOPoIV5VerificationReport(
                workunit_id="wu-1",
                verified=True,
                verify_cpu_ms=0.35,
                proof_bytes=True,  # type: ignore[arg-type]
                within_5ms_sla=True,
                within_32kb_bound=True,
            )
        with pytest.raises(ValueError, match="rejection_reason must be None or a non-empty string"):
            ZKOPoIV5VerificationReport(
                workunit_id="wu-1",
                verified=False,
                verify_cpu_ms=0.35,
                proof_bytes=512,
                within_5ms_sla=True,
                within_32kb_bound=True,
                rejection_reason="   ",
            )


class TestS3605DARLCASShardLeasingHardening:
    def test_shard_lease_record_and_snapshot_validation(self) -> None:
        rec = ShardLeaseRecord(
            lease_id=1,
            shard_id="shard-01",
            worker_id="worker-01",
            granted_at_s=10.0,
            expires_at_s=18.0,
            status="ACTIVE",
        )
        assert rec.status == "ACTIVE"

        with pytest.raises(ValueError, match="lease_id must be a positive integer"):
            ShardLeaseRecord(
                lease_id=0,
                shard_id="shard-01",
                worker_id="worker-01",
                granted_at_s=10.0,
                expires_at_s=18.0,
            )
        with pytest.raises(ValueError, match="expires_at_s must be >= granted_at_s"):
            ShardLeaseRecord(
                lease_id=1,
                shard_id="shard-01",
                worker_id="worker-01",
                granted_at_s=20.0,
                expires_at_s=18.0,
            )
        with pytest.raises(ValueError, match="status must be"):
            ShardLeaseRecord(
                lease_id=1,
                shard_id="shard-01",
                worker_id="worker-01",
                granted_at_s=10.0,
                expires_at_s=18.0,
                status="INVALID",
            )

        coord = DARLShardLeaseCoordinator(["s1", "s2"])
        snap = coord.snapshot()
        assert snap.conservation_holds is True

        with pytest.raises(ValueError, match="total_shards must be an integer >= 1"):
            DARLLedgerSnapshot(
                total_shards=0,
                granted=0,
                reclaimed=0,
                committed=0,
                active=0,
                unique_committed_shards=0,
                duplicate_commits_blocked=0,
                conservation_holds=True,
            )
        with pytest.raises(ValueError, match="granted must be a non-negative integer"):
            DARLLedgerSnapshot(
                total_shards=2,
                granted=-1,
                reclaimed=0,
                committed=0,
                active=0,
                unique_committed_shards=0,
                duplicate_commits_blocked=0,
                conservation_holds=True,
            )
        with pytest.raises(ValueError, match="conservation_holds must be a boolean"):
            DARLLedgerSnapshot(
                total_shards=2,
                granted=0,
                reclaimed=0,
                committed=0,
                active=0,
                unique_committed_shards=0,
                duplicate_commits_blocked=0,
                conservation_holds=1,  # type: ignore[arg-type]
            )


class TestS3606HourlyCodingAssistantOrchestratorHardening:
    def test_receipt_and_report_validation(self) -> None:
        receipt = CodingAssistantTaskReceipt(
            task_id="HM-S36-01",
            title="ZK Verifier",
            model="hivemind/coder",
            live_completion_id="comp-1",
            live_routed_model="hivemind/coder",
            assistant_excerpt="ok",
            gwaya_zero_stub_passed=True,
            unit_verification_passed=True,
            anse_energy=0.0,
        )
        assert receipt.to_dict()["task_id"] == "HM-S36-01"

        with pytest.raises(ValueError, match="task_id must be a non-empty string"):
            CodingAssistantTaskReceipt(
                task_id="",
                title="ZK Verifier",
                model="hivemind/coder",
                live_completion_id="comp-1",
                live_routed_model="hivemind/coder",
                assistant_excerpt="ok",
                gwaya_zero_stub_passed=True,
                unit_verification_passed=True,
                anse_energy=0.0,
            )
        with pytest.raises(ValueError, match="gwaya_zero_stub_passed must be a boolean"):
            CodingAssistantTaskReceipt(
                task_id="HM-S36-01",
                title="ZK Verifier",
                model="hivemind/coder",
                live_completion_id="comp-1",
                live_routed_model="hivemind/coder",
                assistant_excerpt="ok",
                gwaya_zero_stub_passed=1,  # type: ignore[arg-type]
                unit_verification_passed=True,
                anse_energy=0.0,
            )
        with pytest.raises(ValueError, match="anse_energy must be a finite non-negative number"):
            CodingAssistantTaskReceipt(
                task_id="HM-S36-01",
                title="ZK Verifier",
                model="hivemind/coder",
                live_completion_id="comp-1",
                live_routed_model="hivemind/coder",
                assistant_excerpt="ok",
                gwaya_zero_stub_passed=True,
                unit_verification_passed=True,
                anse_energy=-1.0,
            )

        report = HourlyImprovementCycleReport(
            timestamp_utc=1700000000.0,
            antigravity_sdk_compat=True,
            live_openrouter_called=False,
            tasks_verified=1,
            tasks_total=1,
            s36_01_zk_verify_ms=0.2,
            s36_02_e21_energy_reduction_pct=95.0,
            s36_03_geomesh_p95_reduction_pct=45.0,
            s36_04_subswarm_agreement=0.85,
            s36_05_darl_conservation_holds=True,
            ip21_residual_sampling_verified=True,
            ip67_no_hardcoded_xdev_verified=True,
            task_receipts=(receipt,),
            merkle_root_sha256="a" * 64,
            all_gates_passed=True,
            status="BLOCKED",
            blocked_reason="model not called",
        )
        assert report.to_dict()["tasks_verified"] == 1

        with pytest.raises(ValueError, match="tasks_verified and tasks_total"):
            HourlyImprovementCycleReport(
                timestamp_utc=1700000000.0,
                antigravity_sdk_compat=True,
                live_openrouter_called=False,
                tasks_verified=2,
                tasks_total=1,
                s36_01_zk_verify_ms=0.2,
                s36_02_e21_energy_reduction_pct=95.0,
                s36_03_geomesh_p95_reduction_pct=45.0,
                s36_04_subswarm_agreement=0.85,
                s36_05_darl_conservation_holds=True,
                ip21_residual_sampling_verified=True,
                ip67_no_hardcoded_xdev_verified=True,
                task_receipts=(receipt,),
                merkle_root_sha256="a" * 64,
                all_gates_passed=True,
            )
        with pytest.raises(ValueError, match="merkle_root_sha256 must be a 64-character hex string"):
            HourlyImprovementCycleReport(
                timestamp_utc=1700000000.0,
                antigravity_sdk_compat=True,
                live_openrouter_called=False,
                tasks_verified=1,
                tasks_total=1,
                s36_01_zk_verify_ms=0.2,
                s36_02_e21_energy_reduction_pct=95.0,
                s36_03_geomesh_p95_reduction_pct=45.0,
                s36_04_subswarm_agreement=0.85,
                s36_05_darl_conservation_holds=True,
                ip21_residual_sampling_verified=True,
                ip67_no_hardcoded_xdev_verified=True,
                task_receipts=(receipt,),
                merkle_root_sha256="bad",
                all_gates_passed=True,
            )


class TestZKOPoIV5CircuitHardening:
    def test_field_helpers_and_verification_result_validation(self) -> None:
        assert _field_mod(MERSENNE_PRIME_61 + 3) == 3
        with pytest.raises(ValueError, match="x must be an integer"):
            _field_mod(False)  # type: ignore[arg-type]

        ch = _derive_fiat_shamir_challenge("node:shard:theta0:final:merkle")
        assert 2 <= ch < MERSENNE_PRIME_61
        with pytest.raises(ValueError, match="seed_text must be a non-empty string"):
            _derive_fiat_shamir_challenge("  ")

        assert _poly_eval_field([1, 2, 3], 5) > 0
        with pytest.raises(ValueError, match="challenge_r must be an integer"):
            _poly_eval_field([1, 2], True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="vec_q must be a non-empty sequence of integers"):
            _poly_eval_field([], 5)
        with pytest.raises(ValueError, match="vec_q elements must be integers"):
            _poly_eval_field([1, True], 5)  # type: ignore[list-item]

        res = ZKOPoIV5VerificationResult(
            valid=True,
            reason="zk_opoi_v5_verified_o1_cpu",
            verify_cpu_ms=0.4,
            proof_size_bytes=480,
            shard_commitment_ok=True,
            r1cs_transition_ok=True,
            norm_bounds_ok=True,
        )
        assert res.to_dict()["valid"] is True

        with pytest.raises(ValueError, match="valid must be a boolean"):
            ZKOPoIV5VerificationResult(
                valid=1,  # type: ignore[arg-type]
                reason="ok",
                verify_cpu_ms=0.4,
                proof_size_bytes=480,
                shard_commitment_ok=True,
                r1cs_transition_ok=True,
                norm_bounds_ok=True,
            )
        with pytest.raises(ValueError, match="reason must be a non-empty string"):
            ZKOPoIV5VerificationResult(
                valid=True,
                reason=" ",
                verify_cpu_ms=0.4,
                proof_size_bytes=480,
                shard_commitment_ok=True,
                r1cs_transition_ok=True,
                norm_bounds_ok=True,
            )

        circuit = ZKOPoIV5FixedPointCircuit(coordinator_secret="secret-key")
        receipt, _ = circuit.prove_quantized_diloco_transition(
            node_id="node-1",
            assigned_shard_sha256="a" * 64,
            theta0_float=[0.1, -0.2, 0.3, 0.4],
            step_gradients_float=[[0.05, -0.05, 0.05, -0.05]],
        )
        with pytest.raises(TypeError, match="receipt must be a ZKOPoIV5ProofReceipt"):
            circuit.verify_receipt_cpu_o1(
                "not-a-receipt",  # type: ignore[arg-type]
                expected_shard_sha256="a" * 64,
                expected_theta0_commitment=receipt.theta0_commitment,
            )
        with pytest.raises(ValueError, match="expected_shard_sha256 must be a non-empty string"):
            circuit.verify_receipt_cpu_o1(
                receipt,
                expected_shard_sha256=" ",
                expected_theta0_commitment=receipt.theta0_commitment,
            )
        with pytest.raises(ValueError, match="expected_theta0_commitment must be a non-empty string"):
            circuit.verify_receipt_cpu_o1(
                receipt,
                expected_shard_sha256="a" * 64,
                expected_theta0_commitment="",
            )
