# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Hardening and negative-control tests for Top-1% DCT+EF21, Adaptive Multi-Split WAN, BYOB, and NAT traversal."""

from __future__ import annotations

import math
import socket
import pytest

from hivemind.swarm.adaptive_multi_split_wan import (
    AdaptiveMultiSplitWANController,
    CitizenHardwareProfile,
    WANLinkTelemetry,
)
from hivemind.swarm.byob import (
    BeeManifest,
    HfHttpHubClient,
    StaticReport,
    SubmissionLedger,
    Verdict,
    estimate_resident_gb,
    evaluate,
)
from hivemind.swarm.nat_traversal import (
     BINDING,
    REQUEST,
    StunMessage,
    TurnClient,
    decode_error_code,
    encode_error_code,
    stun_transaction,
)
from hivemind.swarm.top1pct_dct_ef21_compressor import Top1PctQuantizedDCTEF21Compressor


class TestTop1PctDCTEF21CompressorHardening:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"top_k_fraction": True}, "top_k_fraction"),
            ({"top_k_fraction": False}, "top_k_fraction"),
            ({"top_k_fraction": float("nan")}, "top_k_fraction"),
            ({"top_k_fraction": 0.0005}, "top_k_fraction"),
            ({"top_k_fraction": 0.25}, "top_k_fraction"),
            ({"quantization_bits": True}, "quantization_bits"),
            ({"quantization_bits": False}, "quantization_bits"),
            ({"quantization_bits": 8}, "quantization_bits"),
            ({"dampening_gamma": True}, "dampening_gamma"),
            ({"dampening_gamma": False}, "dampening_gamma"),
            ({"dampening_gamma": 0.0}, "dampening_gamma"),
            ({"dampening_gamma": float("nan")}, "dampening_gamma"),
        ],
    )
    def test_init_rejects_invalid_hyperparameters(self, kwargs: dict, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            Top1PctQuantizedDCTEF21Compressor(**kwargs)

    def test_get_residual_and_compress_reject_invalid_inputs_without_corrupting_state(self) -> None:
        comp = Top1PctQuantizedDCTEF21Compressor(top_k_fraction=0.01, quantization_bits=4)
        pkt, recon = comp.compress_and_reconstruct("div1", [0.25, -0.5, 0.75, 0.1] * 4, step_index=0)
        assert pkt.total_elements == 16
        assert len(recon) == 16
        saved_residual = comp.get_residual("div1", 16)

        with pytest.raises(ValueError, match="division_id"):
            comp.get_residual("   ", 16)
        with pytest.raises(ValueError, match="length"):
            comp.get_residual("div1", True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="length"):
            comp.get_residual("div1", 0)

        with pytest.raises(ValueError, match="division_id"):
            comp.compress_and_reconstruct("", [0.1, 0.2])
        with pytest.raises(ValueError, match="step_index"):
            comp.compress_and_reconstruct("div1", [0.1, 0.2], step_index=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="step_index"):
            comp.compress_and_reconstruct("div1", [0.1, 0.2], step_index=-1)
        with pytest.raises(ValueError, match="raw_delta"):
            comp.compress_and_reconstruct("div1", [0.1, float("nan")] * 8, step_index=1)
        with pytest.raises(ValueError, match="raw_delta"):
            comp.compress_and_reconstruct("div1", [True, 0.2] * 8, step_index=1)  # type: ignore[list-item]

        # Residual buffer for div1 must remain intact after rejected calls
        assert comp.get_residual("div1", 16) == saved_residual


class TestAdaptiveMultiSplitWANHardening:
    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"uplink_mbps": True, "downlink_mbps": 50.0, "rtt_ms": 20.0}, "uplink_mbps"),
            ({"uplink_mbps": 20.0, "downlink_mbps": False, "rtt_ms": 20.0}, "downlink_mbps"),
            ({"uplink_mbps": 20.0, "downlink_mbps": 50.0, "rtt_ms": True}, "rtt_ms"),
            ({"uplink_mbps": 20.0, "downlink_mbps": 50.0, "rtt_ms": 20.0, "jitter_ms": True}, "jitter_ms"),
            ({"uplink_mbps": 20.0, "downlink_mbps": 50.0, "rtt_ms": 20.0, "packet_loss_rate": False}, "packet_loss_rate"),
            ({"uplink_mbps": 20.0, "downlink_mbps": 50.0, "rtt_ms": 20.0, "packet_loss_rate": float("nan")}, "packet_loss_rate"),
        ],
    )
    def test_wan_link_telemetry_rejects_boolean_and_nonfinite_fields(
        self, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            WANLinkTelemetry(**kwargs)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"node_id": "  ", "gpu_name": "RTX 3060", "compute_capability_sm": 8.6}, "node_id"),
            ({"node_id": "n1", "gpu_name": "", "compute_capability_sm": 8.6}, "gpu_name"),
            ({"node_id": "n1", "gpu_name": "RTX 3060", "compute_capability_sm": True}, "compute_capability_sm"),
            ({"node_id": "n1", "gpu_name": "RTX 3060", "compute_capability_sm": 0.0}, "compute_capability_sm"),
            ({"node_id": "n1", "gpu_name": "RTX 3060", "compute_capability_sm": 8.6, "vram_budget_mib": False}, "vram_budget_mib"),
            ({"node_id": "n1", "gpu_name": "RTX 3060", "compute_capability_sm": 8.6, "local_tflops": -1.0}, "local_tflops"),
            ({"node_id": "n1", "gpu_name": "RTX 3060", "compute_capability_sm": 8.6, "foreground_game_active": 1}, "foreground_game_active"),
        ],
    )
    def test_citizen_hardware_profile_rejects_invalid_fields(
        self, kwargs: dict, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            CitizenHardwareProfile(**kwargs)

    def test_controller_rejects_invalid_params_cut_layer_and_activations(self) -> None:
        with pytest.raises(ValueError, match="sequence_length"):
            AdaptiveMultiSplitWANController(sequence_length=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="micro_batch_size"):
            AdaptiveMultiSplitWANController(micro_batch_size=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="edge_core_tflops"):
            AdaptiveMultiSplitWANController(edge_core_tflops=0.0)
        with pytest.raises(ValueError, match="monolithic_delta_mb"):
            AdaptiveMultiSplitWANController(monolithic_delta_mb=float("nan"))
        with pytest.raises(ValueError, match="max_client_vram_mib"):
            AdaptiveMultiSplitWANController(max_client_vram_mib=-10.0)

        ctrl = AdaptiveMultiSplitWANController()
        link = WANLinkTelemetry(uplink_mbps=40.0, downlink_mbps=100.0, rtt_ms=25.0)
        hw = CitizenHardwareProfile(node_id="n1", gpu_name="RTX 3060", compute_capability_sm=8.6)
        with pytest.raises(ValueError, match="Unsupported cut_layer"):
            ctrl.estimate_cut_layer_cost(True, link, hw)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="activation_bits"):
            ctrl.estimate_cut_layer_cost(1, link, hw, activation_bits=4)
        with pytest.raises(ValueError, match="activation_bits"):
            ctrl.estimate_cut_layer_cost(1, link, hw, activation_bits=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="activations"):
            ctrl.quantize_cut_activations_int8([0.5, float("inf")])
        with pytest.raises(ValueError, match="activations"):
            ctrl.quantize_cut_activations_int8([True, 0.25])  # type: ignore[list-item]


class TestBYOBHardening:
    def _sample_manifest(self) -> BeeManifest:
        return BeeManifest(
            bee_name="bee-quantum-helper",
            contributor_id="alice_dev",
            hf_repo_id="alice/bee-quantum-helper",
            revision="a" * 40,
            kind="lora_adapter",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            license="apache-2.0",
            domain="quantum-computing",
            vram_tier_gb=8,
            eval_pack="quantum-v1",
        )

    def test_estimate_resident_gb_rejects_boolean_or_negative_weight_bytes(self) -> None:
        m = self._sample_manifest()
        assert estimate_resident_gb(m, 100_000_000) is not None
        with pytest.raises(ValueError, match="weight_bytes"):
            estimate_resident_gb(m, True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="weight_bytes"):
            estimate_resident_gb(m, -1)

    def test_hf_http_hub_client_and_submission_ledger_reject_invalid_init(self) -> None:
        with pytest.raises(ValueError, match="endpoint"):
            HfHttpHubClient(endpoint="   ")
        with pytest.raises(ValueError, match="endpoint"):
            HfHttpHubClient(endpoint="///")
        with pytest.raises(ValueError, match="timeout_s"):
            HfHttpHubClient(timeout_s=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="timeout_s"):
            HfHttpHubClient(timeout_s=0.0)
        with pytest.raises(ValueError, match="timeout_s"):
            HfHttpHubClient(timeout_s=float("nan"))

        with pytest.raises(ValueError, match="max_per_window"):
            SubmissionLedger(max_per_window=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="window_s"):
            SubmissionLedger(window_s=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="window_s"):
            SubmissionLedger(window_s=float("inf"))

    def test_evaluate_rejects_invalid_thresholds_and_timeouts(self) -> None:
        m = self._sample_manifest()
        static = StaticReport(verdict=Verdict.ACCEPT)
        with pytest.raises(ValueError, match="min_pass_rate"):
            evaluate(m, static, pack=None, min_pass_rate=True)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="min_pass_rate"):
            evaluate(m, static, pack=None, min_pass_rate=1.5)
        with pytest.raises(ValueError, match="timeout_s"):
            evaluate(m, static, pack=None, timeout_s=False)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="timeout_s"):
            evaluate(m, static, pack=None, timeout_s=0.0)


class TestNatTraversalHardening:
    def test_encode_error_code_validates_bounds_and_roundtrips(self) -> None:
        raw = encode_error_code(401, "Unauthorized")
        assert decode_error_code(raw) == (401, "Unauthorized")
        for bad_code in (True, False, 299, 700, -401):
            with pytest.raises(ValueError, match="300..699"):
                encode_error_code(bad_code, "Bad")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="reason"):
            encode_error_code(401, 123)  # type: ignore[arg-type]

    def test_stun_transaction_rejects_boolean_and_nonfinite_timers(self) -> None:
        req = StunMessage(BINDING, REQUEST)
        raw = req.encode(fingerprint=True)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            for bad_timeout, bad_rto in [
                (True, 0.1),
                (1.0, False),
                (float("nan"), 0.1),
                (1.0, float("inf")),
            ]:
                with pytest.raises(ValueError, match="STUN timeout and initial RTO must be positive"):
                    stun_transaction(
                        sock,
                        ("127.0.0.1", 3478),
                        raw,
                        req.transaction_id,
                        timeout=bad_timeout,  # type: ignore[arg-type]
                        initial_rto=bad_rto,  # type: ignore[arg-type]
                    )

    def test_turn_client_rejects_invalid_timeout_and_lifetime(self) -> None:
        for bad_timeout in (True, False, 0.0, -1.0, math.nan, math.inf):
            with pytest.raises(ValueError, match="TURN timeout"):
                TurnClient(("127.0.0.1", 3478), "alice", "secret", timeout=bad_timeout)  # type: ignore[arg-type]

        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            client = TurnClient(("127.0.0.1", 3478), "alice", "secret", sock=sock, timeout=0.5)
            for bad_lifetime in (True, False, -1, 0x1_0000_0000):
                with pytest.raises(ValueError, match="TURN lifetime"):
                    client.allocate(lifetime=bad_lifetime)  # type: ignore[arg-type]
                with pytest.raises(ValueError, match="TURN lifetime"):
                    client.refresh(lifetime=bad_lifetime)  # type: ignore[arg-type]
