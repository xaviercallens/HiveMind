# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-26: sequential SPRT shadow-canary traffic router for WSP endpoints."""

from __future__ import annotations

import ast
import importlib.util
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.shadow_canary_wsp_traffic_router import (
    DEFAULT_ANSE_MAXIMUM_PAIN,
    ROUTING_RECEIPT_OBJECT,
    SCHEMA_VERSION,
    STATE_RECEIPT_OBJECT,
    CanaryRoutingDecision,
    ShadowCanaryConfig,
    ShadowCanaryState,
    build_canary_rollout_summary,
    canonical_json_bytes,
    compute_payload_sha256,
    create_shadow_canary_state,
    route_canary_request_deterministic,
    update_sprt_canary_step,
    verify_routing_receipt_provenance,
    verify_shadow_canary_state_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackAuditResult:
    """Minimal AST stub audit result used when optional ``gwaya`` is absent in CI."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = list(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1_000_000.0


class _FallbackZeroStubAudit:
    """CI-safe AST stub verifier implementing both ``check_file`` and ``audit_python_code``."""

    def audit_python_code(self, code: str) -> _FallbackAuditResult:
        """Scan Python source for placeholder ``pass`` / ``...`` / ``NotImplementedError`` stubs."""
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    stmt
                    for stmt in node.body
                    if not (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"empty_or_pass_function:{node.name}")
                elif (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Expr)
                    and isinstance(stmts[0].value, ast.Constant)
                    and stmts[0].value.value is Ellipsis
                ):
                    violations.append(f"ellipsis_function:{node.name}")
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
                    violations.append("not_implemented_raise")
        return _FallbackAuditResult(violations)

    def check_file(self, path: str | Path) -> list[str]:
        """Inspect a Python source file on disk and return detected stub violation strings."""
        return self.audit_python_code(Path(path).read_text(encoding="utf-8")).violations


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "shadow_canary_wsp_traffic_router.py"
TEST_FILE_PATH = Path(__file__).resolve()
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _drive_genuine_s5_promotion(
    *,
    steps: int = 45,
    seed: int = 20261124,
    config: ShadowCanaryConfig | None = None,
) -> tuple[ShadowCanaryState, list[float]]:
    """Drive paired S5 WSP observations (-0.55 nats pooled mean, E0 <= +0.2%) on CPU."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    state = create_shadow_canary_state(
        config=config,
        base_checkpoint_id="crowd_v1_base_pinned_sha256",
        wsp_checkpoint_id="crowd_v1_s5_wsp_alphaedit_sha256",
        canary_salt="dyn26-s1-salt",
    )
    observed_ramps: list[float] = [state.ramp_fraction]
    base_noise = (0.04 * torch.randn(steps, generator=gen, dtype=torch.float64)).tolist()
    spec_noise = (0.04 * torch.randn(steps, generator=gen, dtype=torch.float64)).tolist()
    e0_noise = (0.002 * torch.randn(steps, generator=gen, dtype=torch.float64)).tolist()
    lat_noise = (0.03 * torch.randn(steps, generator=gen, dtype=torch.float64)).tolist()

    for idx in range(steps):
        is_general = idx % 3 == 0
        base_nll = 4.20 + abs(float(base_noise[idx]))
        if is_general:
            # E0 general-prose drift stays strictly within <= +0.2% (<= +0.002).
            e0_rel = min(0.002, max(-0.001, 0.001 + float(e0_noise[idx])))
            wsp_nll = base_nll * (1.0 + e0_rel)
        else:
            # Specialist domain realizes ~ -0.825 nats so pooled mean NLL delta ~ -0.55 nats.
            wsp_nll = max(0.05, base_nll - 0.825 + float(spec_noise[idx]))
        latency_ratio = max(0.75, min(1.12, 1.01 + float(lat_noise[idx])))
        state = update_sprt_canary_step(
            state,
            base_nll,
            wsp_nll,
            is_general_domain=is_general,
            latency_ratio=latency_ratio,
            entropy_collapse=False,
            repetition_rate=0.04,
            token_entropy_nats=2.40,
        )
        observed_ramps.append(state.ramp_fraction)
    return state, observed_ramps


# ---------------------------------------------------------------------------
# Scenario S1: Progressive 0% -> 5% -> 25% -> 100% Live Promotion
# ---------------------------------------------------------------------------


def test_s1_progressive_sprt_promotion_0_5_25_100_on_genuine_s5_gain() -> None:
    """S1: Genuine S5 improvement (-0.55 nats, E0 <= +0.2%) promotes 0% -> 5% -> 25% -> 100%."""
    state, observed_ramps = _drive_genuine_s5_promotion(steps=48, seed=20261124)

    # Every progressive ramp tier must be visited in strictly monotone order.
    unique_stages_in_order: list[float] = []
    for r in observed_ramps:
        if not unique_stages_in_order or unique_stages_in_order[-1] != r:
            unique_stages_in_order.append(r)
    assert unique_stages_in_order == [0.0, 0.05, 0.25, 1.0]

    assert state.ramp_fraction == pytest.approx(1.0)
    assert state.stage_index == 3
    assert state.stage_name == "live_100"
    assert state.active_endpoint == "wsp"
    assert state.promoted_to_live is True
    assert state.rollback_triggered is False
    assert state.rollback_reasons == ()
    assert state.general_guard_violations == 0
    assert state.entropy_collapse_count == 0
    assert state.entropy_collapse_detected is False
    assert state.general_rel_drift <= 0.002 + 1e-9
    assert state.nll_diff_mean == pytest.approx(-0.55, abs=0.08)
    assert state.cs_nll_diff_upper is not None and state.cs_nll_diff_upper < 0.0
    assert state.cs_gain_lower is not None and state.cs_gain_lower > 0.0
    assert state.sprt_log_lr > state.config.sprt_upper_bound
    assert state.anse_energy < 0.0
    assert state.latency_p95_ratio <= state.config.max_latency_p95_ratio
    assert state.min_observed_token_entropy_nats == pytest.approx(2.40)
    assert state.max_observed_repetition_rate == pytest.approx(0.04)
    assert verify_shadow_canary_state_provenance(state) is True
    assert len(state.transition_history) == 3

    # Verify routing behavior at final 100% live stage and soft margin penalty branch.
    live_decision = route_canary_request_deterministic("prod-req-0001", state.ramp_fraction, state=state)
    assert live_decision.routed_to_wsp is True
    assert live_decision.selected_endpoint == "wsp"
    assert live_decision.selected_checkpoint_id == state.wsp_checkpoint_id
    assert verify_routing_receipt_provenance(live_decision) is True

    # Verify soft-margin penalty (+0.05) triggers in [0.85%, 1.0%] without rollback and custom fractional stage label.
    soft_cfg = ShadowCanaryConfig(ramp_stages=(0.0, 0.125, 1.0), min_samples_per_stage=4)
    soft_state = create_shadow_canary_state(soft_cfg)
    for step_idx in range(6):
        is_gen = step_idx == 5
        b_nll = 4.0
        w_nll = 4.0 * 1.009 if is_gen else 3.20
        soft_state = update_sprt_canary_step(soft_state, b_nll, w_nll, is_general_domain=is_gen)
    assert soft_state.rollback_triggered is False
    assert soft_state.ramp_fraction == pytest.approx(0.125)
    assert soft_state.stage_name == "canary_12.5"
    assert soft_state.active_endpoint == "canary_split"
    assert soft_state.anse_energy == pytest.approx(soft_state.nll_diff_mean + soft_cfg.soft_margin_penalty)


# ---------------------------------------------------------------------------
# Scenario S2: Instant Atomic Rollback to 0% on E0 Regression or Entropy Collapse
# ---------------------------------------------------------------------------


def test_s2_instant_atomic_rollback_on_e0_regression_or_entropy_collapse() -> None:
    """S2: +2.5% E0 drift or entropy/repetition collapse triggers instant 0-traffic atomic rollback."""
    # Promote cleanly into live canary first (25% / 100%).
    promoted_state, _ = _drive_genuine_s5_promotion(steps=30, seed=20261124)
    assert promoted_state.ramp_fraction >= 0.25
    assert promoted_state.rollback_triggered is False

    # S2a: Single +2.5% E0 general-prose drift triggers immediate 0% rollback on that exact step.
    e0_regressed = update_sprt_canary_step(
        promoted_state,
        base_nll=4.0,
        wsp_nll=4.0 * 1.025,
        is_general_domain=True,
        latency_ratio=1.0,
        entropy_collapse=False,
    )
    assert e0_regressed.ramp_fraction == 0.0
    assert e0_regressed.stage_index == 0
    assert e0_regressed.stage_name == "rolled_back"
    assert e0_regressed.active_endpoint == "base"
    assert e0_regressed.promoted_to_live is False
    assert e0_regressed.rollback_triggered is True
    assert e0_regressed.general_guard_violations == 1
    assert "general_prose_drift_guard_exceeded" in e0_regressed.rollback_reasons
    assert e0_regressed.anse_energy >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert verify_shadow_canary_state_provenance(e0_regressed) is True

    # Fail-closed routing refuses WSP even if caller supplies stale pre-rollback ramp_fraction.
    routed_after_rollback = route_canary_request_deterministic(
        "req-fail-closed-1",
        promoted_state.ramp_fraction,
        state=e0_regressed,
    )
    assert routed_after_rollback.routed_to_wsp is False
    assert routed_after_rollback.ramp_fraction == 0.0
    assert routed_after_rollback.selected_endpoint == "base"
    assert routed_after_rollback.selected_checkpoint_id == e0_regressed.base_checkpoint_id
    assert verify_routing_receipt_provenance(routed_after_rollback) is True

    # Subsequent clean steps remain permanently locked at 0% after rollback.
    locked_next = update_sprt_canary_step(
        e0_regressed,
        base_nll=4.2,
        wsp_nll=3.4,
        is_general_domain=False,
    )
    assert locked_next.ramp_fraction == 0.0
    assert locked_next.stage_name == "rolled_back"
    assert locked_next.last_transition == "locked_after_rollback"
    assert locked_next.rollback_triggered is True

    # S2b: Explicit entropy_collapse=True triggers instant rollback to 0%.
    ent_collapsed = update_sprt_canary_step(
        promoted_state,
        base_nll=4.2,
        wsp_nll=3.6,
        is_general_domain=False,
        entropy_collapse=True,
    )
    assert ent_collapsed.ramp_fraction == 0.0
    assert ent_collapsed.stage_name == "rolled_back"
    assert ent_collapsed.rollback_triggered is True
    assert ent_collapsed.entropy_collapse_detected is True
    assert ent_collapsed.entropy_collapse_count == 1
    assert "entropy_collapse_or_repetition_loop" in ent_collapsed.rollback_reasons
    assert ent_collapsed.anse_energy >= DEFAULT_ANSE_MAXIMUM_PAIN

    # S2c: Repetition rate / token entropy threshold violations trigger collapse rollback.
    rep_collapsed = update_sprt_canary_step(
        promoted_state,
        base_nll=4.2,
        wsp_nll=3.6,
        is_general_domain=False,
        repetition_rate=0.45,
        token_entropy_nats=0.20,
    )
    assert rep_collapsed.ramp_fraction == 0.0
    assert rep_collapsed.rollback_triggered is True
    assert rep_collapsed.repetition_loop_count == 1
    assert rep_collapsed.min_observed_token_entropy_nats == pytest.approx(0.20)
    assert rep_collapsed.max_observed_repetition_rate == pytest.approx(0.45)

    # S2d: Latency p95 / hard spike breach triggers rollback when enabled.
    lat_spike = update_sprt_canary_step(
        promoted_state,
        base_nll=4.2,
        wsp_nll=3.6,
        is_general_domain=False,
        latency_ratio=3.5,
    )
    assert lat_spike.ramp_fraction == 0.0
    assert lat_spike.rollback_triggered is True
    assert "latency_p95_guard_exceeded" in lat_spike.rollback_reasons
    assert lat_spike.latency_violation_count >= 1

    # S2e: Live SPRT / Page CUSUM regression (stage_index > 0) rolls back to 0%.
    regressed_live = promoted_state
    for _ in range(18):
        regressed_live = update_sprt_canary_step(
            regressed_live,
            base_nll=4.0,
            wsp_nll=4.35,
            is_general_domain=False,
        )
        if regressed_live.rollback_triggered:
            break
    assert regressed_live.rollback_triggered is True
    assert regressed_live.ramp_fraction == 0.0
    assert "sprt_live_regression_below_lower_bound" in regressed_live.rollback_reasons

    # S2f: Rolling p95 latency degradation (>= min_latency_p95_samples) triggers rollback.
    lat_p95_state = promoted_state
    for _ in range(8):
        lat_p95_state = update_sprt_canary_step(
            lat_p95_state,
            base_nll=4.2,
            wsp_nll=3.6,
            is_general_domain=False,
            latency_ratio=1.55,
        )
        if lat_p95_state.rollback_triggered:
            break
    assert lat_p95_state.rollback_triggered is True
    assert "latency_p95_guard_exceeded" in lat_p95_state.rollback_reasons
    assert lat_p95_state.latency_p95_ratio > lat_p95_state.config.max_latency_p95_ratio

    # S2g: Gradual +1.5% E0 drift (between +1.0% and +2.0% hard cap) triggers rollback via sliding window.
    gradual_e0 = promoted_state
    for _ in range(6):
        gradual_e0 = update_sprt_canary_step(
            gradual_e0,
            base_nll=4.0,
            wsp_nll=4.0 * 1.015,
            is_general_domain=True,
        )
        if gradual_e0.rollback_triggered:
            break
    assert gradual_e0.rollback_triggered is True
    assert "general_prose_drift_guard_exceeded" in gradual_e0.rollback_reasons


# ---------------------------------------------------------------------------
# Scenario S3: Neutral / Noisy Candidate Held Safely in Shadow (Type-I Control)
# ---------------------------------------------------------------------------


def test_s3_neutral_noisy_candidate_held_in_shadow_type1_control() -> None:
    """S3: Neutral/noisy candidates across multiple seeds and noise levels never promote prematurely."""
    for noise_std in (0.15, 0.35, 0.65):
        for trial_idx in range(5):
            seed = 7000 + int(noise_std * 100) + trial_idx
            gen = torch.Generator(device="cpu").manual_seed(seed)
            state = create_shadow_canary_state()
            diffs = (noise_std * torch.randn(75, generator=gen, dtype=torch.float64)).tolist()

            for step_idx, delta in enumerate(diffs):
                base_nll = 4.25
                wsp_nll = max(0.05, base_nll + float(delta))
                # Use is_general_domain=False here to isolate pure SPRT/CS Type-I error control under noise.
                state = update_sprt_canary_step(
                    state,
                    base_nll,
                    wsp_nll,
                    is_general_domain=False,
                    latency_ratio=1.02 + (0.01 if step_idx % 4 == 0 else 0.0),
                    entropy_collapse=False,
                )

            assert state.ramp_fraction == 0.0
            assert state.stage_index == 0
            assert state.stage_name == "shadow"
            assert state.active_endpoint == "base"
            assert state.promoted_to_live is False
            assert state.rollback_triggered is False
            assert state.sprt_log_lr < state.config.sprt_upper_bound
            assert state.cs_nll_diff_upper is not None and state.cs_nll_diff_upper >= 0.0
            assert verify_shadow_canary_state_provenance(state) is True

    # Explicit steady-zero candidate triggers hold_shadow_h0 transition once crossing lower bound.
    zero_state = create_shadow_canary_state()
    for _ in range(25):
        zero_state = update_sprt_canary_step(zero_state, 4.2, 4.2, is_general_domain=True)
    assert zero_state.ramp_fraction == 0.0
    assert zero_state.rollback_triggered is False
    assert zero_state.last_transition == "hold_shadow_h0"


# ---------------------------------------------------------------------------
# Scenario S4: Deterministic SHA-256 Request Bucketing & Receipt Provenance
# ---------------------------------------------------------------------------


def test_s4_deterministic_sha256_bucketing_monotonicity_and_provenance() -> None:
    """S4: SHA-256 request bucketing is deterministic, monotone across stages, and tamper-evident."""
    keys = [f"user-session-{idx:04d}" for idx in range(1500)]

    routed_0 = {k for k in keys if route_canary_request_deterministic(k, 0.0, salt="prod-v1").routed_to_wsp}
    routed_5 = {k for k in keys if route_canary_request_deterministic(k, 0.05, salt="prod-v1").routed_to_wsp}
    routed_25 = {k for k in keys if route_canary_request_deterministic(k, 0.25, salt="prod-v1").routed_to_wsp}
    routed_100 = {k for k in keys if route_canary_request_deterministic(k, 1.0, salt="prod-v1").routed_to_wsp}

    assert len(routed_0) == 0
    assert len(routed_100) == len(keys)
    # Strict sticky monotonicity across ramp upgrades: 5% <= 25% <= 100%.
    assert routed_5.issubset(routed_25)
    assert routed_25.issubset(routed_100)
    assert 0.025 <= (len(routed_5) / len(keys)) <= 0.080
    assert 0.200 <= (len(routed_25) / len(keys)) <= 0.300

    # Exact repeatability and salt independence.
    d1 = route_canary_request_deterministic("req-alpha-99", 0.25, salt="salt-a")
    d2 = route_canary_request_deterministic("req-alpha-99", 0.25, salt="salt-a")
    d3 = route_canary_request_deterministic("req-alpha-99", 0.25, salt="salt-b")
    assert d1 == d2
    assert d1.bucket_digest_sha256 != d3.bucket_digest_sha256
    assert 0.0 <= d1.bucket_unit < 1.0
    assert 0 <= d1.bucket_bps < 10_000
    assert verify_routing_receipt_provenance(d1) is True
    assert verify_routing_receipt_provenance(d1.to_dict()) is True

    # Tampering on any routing field or receipt hash must fail provenance verification.
    tampered_dec = replace(d1, routed_to_wsp=not d1.routed_to_wsp)
    assert verify_routing_receipt_provenance(tampered_dec) is False
    tampered_dict = d1.to_dict()
    tampered_dict["ramp_fraction"] = 0.99
    assert verify_routing_receipt_provenance(tampered_dict) is False
    assert verify_routing_receipt_provenance({"receipt_sha256": "short"}) is False
    assert verify_routing_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")}) is False
    assert verify_routing_receipt_provenance({"receipt_sha256": "0" * 64}) is False
    forged_routing = {"object": ROUTING_RECEIPT_OBJECT, "ramp_fraction": 0.25}
    forged_routing["receipt_sha256"] = compute_payload_sha256(forged_routing)
    assert verify_routing_receipt_provenance(forged_routing) is False

    # State provenance tamper detection and summary schema verification.
    live_state, _ = _drive_genuine_s5_promotion(steps=30, seed=99)
    summary = build_canary_rollout_summary(live_state)
    assert summary["object"] == STATE_RECEIPT_OBJECT
    assert summary["schema"] == SCHEMA_VERSION
    assert set(("nll_diff_mean", "sprt_log_lr", "anse_energy")).issubset(summary.keys())
    assert verify_shadow_canary_state_provenance(live_state.to_dict()) is True

    tampered_state = replace(live_state, ramp_fraction=0.05)
    assert verify_shadow_canary_state_provenance(tampered_state) is False
    assert verify_shadow_canary_state_provenance({"receipt_sha256": "bad"}) is False
    assert verify_shadow_canary_state_provenance({"receipt_sha256": "0" * 64, "x": float("inf")}) is False

    forged_obj = {k: v for k, v in live_state.to_dict().items() if k != "receipt_sha256"}
    forged_obj["object"] = "wrong.object"
    forged_obj["receipt_sha256"] = compute_payload_sha256(forged_obj)
    assert verify_shadow_canary_state_provenance(forged_obj) is False

    forged_rb = {k: v for k, v in live_state.to_dict().items() if k != "receipt_sha256"}
    forged_rb["rollback_triggered"] = True
    forged_rb["receipt_sha256"] = compute_payload_sha256(forged_rb)
    assert verify_shadow_canary_state_provenance(forged_rb) is False


# ---------------------------------------------------------------------------
# Scenario S5: Input Validation, Provenance Edge Cases & Security/Stub Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates(tmp_path: Path) -> None:
    """S5: Invalid configurations/inputs fail closed and source/test pass security & stub gates."""
    # 1. Config validation guards
    with pytest.raises(TypeError, match="ramp_stages"):
        ShadowCanaryConfig(ramp_stages="0.0,1.0")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least 2 stages"):
        ShadowCanaryConfig(ramp_stages=(0.0,))
    with pytest.raises(ValueError, match="ramp_stages\\[0\\]"):
        ShadowCanaryConfig(ramp_stages=(0.1, 1.0))
    with pytest.raises(ValueError, match="ramp_stages\\[-1\\]"):
        ShadowCanaryConfig(ramp_stages=(0.0, 0.5))
    with pytest.raises(ValueError, match="strictly increasing"):
        ShadowCanaryConfig(ramp_stages=(0.0, 0.5, 0.5, 1.0))
    with pytest.raises(ValueError, match="alpha"):
        ShadowCanaryConfig(alpha=0.6, beta=0.1)
    with pytest.raises(ValueError, match="alt_gain_nats"):
        ShadowCanaryConfig(null_gain_nats=0.1, alt_gain_nats=0.1)
    with pytest.raises(ValueError, match="soft_general_margin_rel"):
        ShadowCanaryConfig(max_general_drift_rel=0.01, soft_general_margin_rel=0.02)
    with pytest.raises(ValueError, match="single_step_general_hard_cap_rel"):
        ShadowCanaryConfig(max_general_drift_rel=0.02, single_step_general_hard_cap_rel=0.01)
    with pytest.raises(ValueError, match="max_latency_p95_ratio"):
        ShadowCanaryConfig(max_latency_p95_ratio=0.9)
    with pytest.raises(ValueError, match="hard_latency_spike_ratio"):
        ShadowCanaryConfig(max_latency_p95_ratio=1.5, hard_latency_spike_ratio=1.2)
    with pytest.raises(ValueError, match="must be <= 1.0"):
        ShadowCanaryConfig(max_repetition_rate=1.5)
    with pytest.raises(ValueError, match="must be > 0.0"):
        ShadowCanaryConfig(alpha=0.0)
    with pytest.raises(ValueError, match="must be < 1.0"):
        ShadowCanaryConfig(alpha=1.0)
    with pytest.raises(TypeError, match="must be a bool"):
        ShadowCanaryConfig(lock_after_rollback="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="must be an int"):
        ShadowCanaryConfig(min_samples_per_stage=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="must be >= 1"):
        ShadowCanaryConfig(min_samples_per_stage=0)
    with pytest.raises(ValueError, match="must be >= 0"):
        ShadowCanaryConfig(min_general_samples_per_stage=-1)
    with pytest.raises(ValueError, match="must be >= 0.0"):
        ShadowCanaryConfig(soft_margin_penalty=-0.1)

    # 2. State & step & routing validation guards
    with pytest.raises(TypeError, match="ShadowCanaryConfig"):
        ShadowCanaryState(config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="out of bounds"):
        ShadowCanaryState(stage_index=9)
    with pytest.raises(ValueError, match="non-empty printable"):
        ShadowCanaryState(base_checkpoint_id="bad\nkey")
    with pytest.raises(TypeError, match="must be a str"):
        ShadowCanaryState(base_checkpoint_id=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="control characters"):
        ShadowCanaryState(canary_salt="bad\x00salt")
    with pytest.raises(TypeError, match="must be a str"):
        ShadowCanaryState(canary_salt=123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="receipt_sha256"):
        ShadowCanaryState(receipt_sha256=123)  # type: ignore[arg-type]

    clean_state = create_shadow_canary_state()
    with pytest.raises(TypeError, match="ShadowCanaryState"):
        update_sprt_canary_step("bad", 4.0, 3.5, is_general_domain=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ShadowCanaryConfig"):
        update_sprt_canary_step(clean_state, 4.0, 3.5, is_general_domain=True, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        update_sprt_canary_step(clean_state, True, 3.5, is_general_domain=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        update_sprt_canary_step(clean_state, float("nan"), 3.5, is_general_domain=True)
    with pytest.raises(ValueError, match="must be > 0.0"):
        update_sprt_canary_step(clean_state, 0.0, 3.5, is_general_domain=True)
    with pytest.raises(ValueError, match="must be >= 0.0"):
        update_sprt_canary_step(clean_state, 4.0, -0.1, is_general_domain=True)
    with pytest.raises(TypeError, match="bool"):
        update_sprt_canary_step(clean_state, 4.0, 3.5, is_general_domain=1)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="non-empty printable"):
        route_canary_request_deterministic("   ", 0.25)
    with pytest.raises(TypeError, match="ShadowCanaryState"):
        route_canary_request_deterministic("req-1", 0.25, state="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mismatches"):
        route_canary_request_deterministic("req-1", 0.25, state=clean_state)

    with pytest.raises(TypeError, match="ShadowCanaryState"):
        verify_shadow_canary_state_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="CanaryRoutingDecision"):
        verify_routing_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ShadowCanaryState"):
        build_canary_rollout_summary("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})

    # 3. Fallback stub auditor check_file & audit_python_code negative controls
    fallback_auditor = _FallbackZeroStubAudit()
    stub_probe = tmp_path / "stub_probe.py"
    stub_probe.write_text("def incomplete():\n    pass\n", encoding="utf-8")
    assert len(fallback_auditor.check_file(stub_probe)) == 1
    assert fallback_auditor.audit_python_code("def ell():\n    ...\n").is_clean is False
    assert fallback_auditor.audit_python_code("def ni():\n    raise NotImplementedError()\n").is_clean is False

    # 4. AGPL-3.0 header, optional ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn26", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn26", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/shadow_canary_wsp_traffic_router.py", TARGET_MODULE_PATH),
        ("tests/test_shadow_canary_wsp_traffic_router_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"
        assert fallback_auditor.check_file(path_obj) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            stub_res = ZeroStubAudit().audit_python_code(src_text)
            assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
            assert stub_res.penalty_energy == 0.0

        assert scanner.scan_python_ast_security(rel, src_text) == []
        assert scanner.scan_text_for_payment_solicitation(src_text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(src_text, rel) == []
        assert scanner.scan_unicode_trojan_source(src_text, rel) == []
        assert lateral_guard.audit_source_for_lateral_movement(src_text, rel) == []

        synthetic_diff = f"diff --git a/{rel} b/{rel}\n+++ b/{rel}\n" + "\n".join(
            f"+{line}" for line in src_text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []
        changed_files.append(
            {
                "filename": rel,
                "status": "added",
                "additions": len(src_text.splitlines()),
                "patch": "",
                "content": src_text,
            }
        )

    # Verify serialized state, summary, and routing receipts never trigger FORBIDDEN_ADDED.
    promoted_state, _ = _drive_genuine_s5_promotion(steps=30, seed=123)
    sample_decision = route_canary_request_deterministic("gate-req-1", promoted_state.ramp_fraction, state=promoted_state)
    assert isinstance(sample_decision, CanaryRoutingDecision)
    sample_json_lines = [
        json.dumps(promoted_state.to_dict(), sort_keys=True),
        json.dumps(build_canary_rollout_summary(promoted_state), sort_keys=True),
        json.dumps(sample_decision.to_dict(), sort_keys=True),
    ]
    for idx_j, payload_json in enumerate(sample_json_lines):
        assert (
            gatekeeper.forbidden_scan(
                f"diff --git a/r{idx_j}.py b/r{idx_j}.py\n+++ b/r{idx_j}.py\n+RECEIPT = {payload_json!r}"
            )
            == []
        )

    pr_audit = scanner.audit_pull_request(
        pr_number=1126,
        pr_title="feat(swarm): sequential SPRT shadow-canary traffic router and instant rollback for WSP-edited endpoints",
        pr_body="Implements NEURAL-CTRL-DYN-26 sequential SPRT shadow-canary traffic router and 5 deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): sequential SPRT shadow-canary traffic router and instant rollback for WSP-edited endpoints"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
