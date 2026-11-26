# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-42: sequential QLoRA chain TIES & factorized null-space rescuer."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import Any

import pytest
import torch
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.sequential_chain_ties_nullspace_rescuer import (
    ANSE_MAXIMUM_PAIN,
    RESCUER_RECEIPT_OBJECT,
    RESCUER_SCHEMA,
    LoRAChainStep,
    SequentialChainRescueConfig,
    compute_nll_and_ece_from_logits,
    compute_prose_nullspace_projector,
    compute_rescue_state_sha256,
    estimate_factorized_vs_dense_flops,
    execute_verified_chain_rescue,
    extract_incremental_chain_task_vectors,
    factorized_associative_nullspace_delta,
    ties_elect_and_project_increments,
    verify_chain_rescue_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result used when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else 1e6


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, path: str | Path) -> list[str]:
        text = Path(path).read_text(encoding="utf-8")
        return list(self.audit_python_code(text).violations)

    def audit_python_code(self, code: str) -> _FallbackStubResult:
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"stub function: {node.name}")
                if (
                    len(stmts) == 1
                    and isinstance(stmts[0], ast.Raise)
                    and isinstance(stmts[0].exc, ast.Call)
                    and isinstance(stmts[0].exc.func, ast.Name)
                    and stmts[0].exc.func.id == "NotImplementedError"
                ):
                    violations.append(f"NotImplementedError in {node.name}")
        return _FallbackStubResult(violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "sequential_chain_ties_nullspace_rescuer.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Avoid OpenMP thread-pool overhead on small deterministic CPU tensors."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(prev_threads)


def _load_module_from_path(name: str, file_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, file_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_pilot7_synthetic_chain_fixture() -> dict[str, Any]:
    """Construct a deterministic 6-step sequential LoRA chain replicating PILOT7/PILOT8 forgetting.

    - Hidden dim ``d = 24``, vocab ``V = 12``, orthogonal subspaces:
      * ``D0_general_prose`` lives strictly in coordinates ``0..7`` (overconfident base scale)
      * ``D1_code`` (step 2) lives strictly in coordinates ``8..11``
      * ``D2_lean4`` (step 4) lives strictly in coordinates ``12..15``
      * ``D3_physics`` (step 6) lives strictly in coordinates ``16..19``
    - Raw final chain step 6 forgets steps 2 & 4 and injects harmful drift into ``D0`` (``> +4%``).
    """
    gen = torch.Generator().manual_seed(20261126)
    d_model = 24
    vocab = 12

    lm_head = torch.zeros(vocab, d_model, dtype=torch.float32)
    # Assign clean disjoint class prototypes across hidden coordinates.
    for cls_idx in range(vocab):
        lm_head[cls_idx, cls_idx % 8] += 1.4 if cls_idx < 4 else 0.0
        lm_head[cls_idx, 8 + (cls_idx % 4)] += 1.8 if 4 <= cls_idx < 8 else 0.0
        lm_head[cls_idx, 12 + (cls_idx % 4)] += 1.8 if 8 <= cls_idx < 12 else 0.0
    # Add light deterministic cross-class coupling so base NLL/ECE are non-trivial.
    lm_head = lm_head + 0.05 * torch.randn(vocab, d_model, generator=gen)

    # Slightly overconfident final RMSNorm weight (replicating base Granite/Qwen overconfidence).
    final_norm_weight = torch.full((d_model,), 2.4, dtype=torch.float32)
    base_layer_weight = torch.zeros(d_model, d_model, dtype=torch.float32)

    n_per_dom = 64
    # D0 general prose: coordinates 0..7, targets in 0..3 with mild label noise -> ECE > 0.08.
    x_d0 = torch.zeros(n_per_dom, d_model, dtype=torch.float32)
    y_d0 = torch.arange(n_per_dom, dtype=torch.long) % 4
    for i in range(n_per_dom):
        c = int(y_d0[i].item())
        x_d0[i, c] = 1.0
        x_d0[i, (c + 1) % 4] = 0.25
        # Introduce controlled ambiguity on 20% of tokens so temperature folding lowers ECE sharply.
        if i % 5 == 0:
            y_d0[i] = (c + 1) % 4

    # D1 code (forgotten step 2): inputs live in coords 8..11 (shifted by +1 vs head -> needs perm).
    x_d1 = torch.zeros(n_per_dom, d_model, dtype=torch.float32)
    y_d1 = 4 + (torch.arange(n_per_dom, dtype=torch.long) % 4)
    for i in range(n_per_dom):
        sub = int((y_d1[i] - 4).item())
        x_d1[i, 8 + ((sub + 1) % 4)] = 1.0
        x_d1[i, 8 + sub] = 0.20

    # D2 lean4 (forgotten step 4): inputs live in coords 12..15 (shifted by +1 vs head -> needs perm).
    x_d2 = torch.zeros(n_per_dom, d_model, dtype=torch.float32)
    y_d2 = 8 + (torch.arange(n_per_dom, dtype=torch.long) % 4)
    for i in range(n_per_dom):
        sub = int((y_d2[i] - 8).item())
        x_d2[i, 12 + ((sub + 1) % 4)] = 1.0
        x_d2[i, 12 + sub] = 0.20

    # Ideal rank-4 specialist LoRAs mapping shifted features back onto true class coordinates.
    b_code = torch.zeros(d_model, 4, dtype=torch.float32)
    a_code = torch.zeros(4, d_model, dtype=torch.float32)
    b_lean = torch.zeros(d_model, 4, dtype=torch.float32)
    a_lean = torch.zeros(4, d_model, dtype=torch.float32)
    for sub in range(4):
        b_code[8 + sub, sub] = 1.2
        a_code[sub, 8 + ((sub + 1) % 4)] = 1.0
        b_lean[12 + sub, sub] = 1.2
        a_lean[sub, 12 + ((sub + 1) % 4)] = 1.0

    # Prose-corrupting drift rank-4 LoRA acting on coordinates 0..3.
    b_corrupt = torch.zeros(d_model, 4, dtype=torch.float32)
    a_corrupt = torch.zeros(4, d_model, dtype=torch.float32)
    for c in range(4):
        b_corrupt[c, c] = -0.55
        b_corrupt[(c + 2) % 4, c] = 0.55
        a_corrupt[c, c] = 1.0

    def _pack_low_rank(mat: torch.Tensor, rank: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
        u_m, s_m, vh_m = torch.linalg.svd(mat, full_matrices=False)
        r_use = min(rank, int(s_m.shape[0]))
        b_part = u_m[:, :r_use] * s_m[:r_use].unsqueeze(0)
        a_part = vh_m[:r_use, :]
        return b_part, a_part

    w_code = b_code @ a_code
    w_lean = b_lean @ a_lean
    w_corrupt = b_corrupt @ a_corrupt

    # 6-step sequential QLoRA chain:
    # step 1: mild prose drift; step 2: learns code; step 3: drifts prose;
    # step 4: learns lean4 while partially forgetting code; step 5..6: overwrites both + heavy D0 drift.
    cumulative_mats = [
        0.20 * w_corrupt,
        0.30 * w_corrupt + 0.90 * w_code,
        0.55 * w_corrupt + 0.70 * w_code,
        0.70 * w_corrupt + 0.45 * w_code + 0.90 * w_lean,
        0.95 * w_corrupt + 0.25 * w_code + 0.55 * w_lean,
        1.15 * w_corrupt + 0.08 * w_code + 0.10 * w_lean,
    ]
    chain_steps: list[LoRAChainStep] = []
    for idx, w_mat in enumerate(cumulative_mats, start=1):
        b_s, a_s = _pack_low_rank(w_mat, rank=8)
        chain_steps.append(LoRAChainStep(step_name=f"chain_step_{idx}", b_mat=b_s, a_mat=a_s, scale=1.0))

    # Peak donor checkpoint (analogous to step-4 / Lean4 peak adapter before late-chain decay).
    donor_b, donor_a = _pack_low_rank(0.40 * w_corrupt + 0.50 * w_code + 0.50 * w_lean, rank=8)
    donor_step = LoRAChainStep(step_name="peak_donor", b_mat=donor_b, a_mat=donor_a, scale=1.0)

    return {
        "mutable_state": {
            "layer_weight": base_layer_weight.clone(),
            "final_norm_weight": final_norm_weight.clone(),
        },
        "lm_head_weight": lm_head,
        "chain_steps": chain_steps,
        "donor_step": donor_step,
        "domain_hidden_inputs": {
            "D0_general_prose": x_d0,
            "D1_code": x_d1,
            "D2_lean4": x_d2,
        },
        "domain_targets": {
            "D0_general_prose": y_d0,
            "D1_code": y_d1,
            "D2_lean4": y_d2,
        },
    }


def test_s1_sequential_6step_chain_rescue_replicates_pilot7_pilot8() -> None:
    """S1: 6-step sequential LoRA chain (+4%+ D0 drift & forgotten step-2/4) rescued to D0 < 0% & ECE -25%+."""
    fx = _build_pilot7_synthetic_chain_fixture()
    init_sha = compute_rescue_state_sha256(
        fx["mutable_state"]["layer_weight"], fx["mutable_state"]["final_norm_weight"]
    )
    cfg = SequentialChainRescueConfig(
        ties_density=0.25,
        safe_tau_grid=(1.0, 1.08, 1.18, 1.35, 1.55, 1.80),
        ties_alpha_grid=(0.0, 0.5, 1.0, 1.5, 2.0),
        donor_alpha_grid=(0.0, 0.5, 1.0),
    )
    receipt = execute_verified_chain_rescue(
        fx["mutable_state"],
        fx["lm_head_weight"],
        fx["chain_steps"],
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        donor_lora=fx["donor_step"],
        config=cfg,
    )

    assert receipt["schema"] == RESCUER_SCHEMA
    assert receipt["object"] == RESCUER_RECEIPT_OBJECT
    assert verify_chain_rescue_receipt_provenance(receipt)
    assert receipt["chain_length"] == 6
    # Raw final chain checkpoint suffered severe > +4% prose drift and failed guard.
    assert receipt["raw_chain_general_drift_ratio"] > 0.04
    assert receipt["raw_chain_guard_ok"] is False
    # Rescued candidate passed ANSE ratchet with D0 < 0%, recovered D1 & D2, and cut ECE by > 25%.
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["initial_state_sha256"] == init_sha
    assert receipt["final_state_sha256"] != init_sha
    assert receipt["anse_energy"] < 0.0
    assert receipt["general_drift_ratio"] < 0.0
    assert receipt["per_domain_drift_ratio"]["D1_code"] < -0.10
    assert receipt["per_domain_drift_ratio"]["D2_lean4"] < -0.10
    ece_rel_drop = (receipt["ece_value"] - receipt["base_ece_value"]) / receipt["base_ece_value"]
    assert ece_rel_drop < -0.25
    assert receipt["prose_invariance_max_diff"] < 1e-4


def test_s2_factorized_associative_nullspace_equivalence_and_flops() -> None:
    """S2: Exact numerical equivalence (<1e-5) between (s*B)@(A@P_0) and ((s*B)@A)@P_0 + prose invariance."""
    gen = torch.Generator().manual_seed(42)
    d_out, d_in, rank = 64, 48, 4
    b_mat = torch.randn(d_out, rank, generator=gen)
    a_mat = torch.randn(rank, d_in, generator=gen)
    scale = 0.75

    # Build orthonormal prose basis U_0 (rank 12) and P_0 = I - U_0 U_0^T.
    prose_raw = torch.randn(12, d_in, generator=gen)
    q_u, _ = torch.linalg.qr(prose_raw.T, mode="reduced")
    p_null = torch.eye(d_in, dtype=torch.float32) - q_u @ q_u.T

    fast_delta = factorized_associative_nullspace_delta(b_mat * scale, a_mat, p_null)
    dense_delta = ((b_mat * scale) @ a_mat) @ p_null

    max_diff = float((fast_delta - dense_delta).abs().max().item())
    assert max_diff < 1e-5
    # Verify prose invariance ||delta @ X_prose^T||_inf < 1e-5.
    prose_leak = float((fast_delta @ prose_raw.T).abs().max().item())
    assert prose_leak < 1e-5

    # Verify incremental extraction supports both mapping and tuple inputs, with and without p_null.
    inc_dense = extract_incremental_chain_task_vectors(
        [
            {"step_name": "s0", "lora_B": b_mat, "lora_A": a_mat, "scale": scale},
            (b_mat * 1.2, a_mat),
        ]
    )
    inc_proj = extract_incremental_chain_task_vectors(
        [
            (b_mat, a_mat, scale),
            {"step_name": "s1", "b_mat": b_mat * 1.2, "a_mat": a_mat, "scale": 1.0},
        ],
        p_null=p_null,
    )
    assert len(inc_dense) == 2 and len(inc_proj) == 2
    assert float((inc_dense[0].delta_weight @ p_null - inc_proj[0].delta_weight).abs().max().item()) < 1e-5
    assert inc_proj[1].cosine_with_previous > 0.99

    flop_stats = estimate_factorized_vs_dense_flops(d_out=2048, d_in=2048, rank=16)
    assert flop_stats["flop_speedup_ratio"] > 50.0


def test_s3_ties_sign_conflict_cancellation_and_nullspace_projection() -> None:
    """S3: Opposing signs cancel cleanly, agreeing signs reinforce, trimming drops noise, output in P_0."""
    d_in = 8
    # Prose occupies coords 0..1; specialist occupies coords 2..7.
    p_null = torch.eye(d_in, dtype=torch.float32)
    p_null[0, 0] = 0.0
    p_null[1, 1] = 0.0

    # Step 1 and Step 2 oppose on (2, 2), agree on (3, 3), and Step 3 adds tiny noise + prose entry.
    t1 = torch.zeros(d_in, d_in, dtype=torch.float32)
    t2 = torch.zeros(d_in, d_in, dtype=torch.float32)
    t3 = torch.zeros(d_in, d_in, dtype=torch.float32)
    t1[2, 2] = 2.0
    t2[2, 2] = -2.0  # Exact opposite sign & equal mass -> must cancel to 0!
    t1[3, 3] = 1.5
    t2[3, 3] = 2.5   # Agreeing positive signs -> average = 2.0
    t3[3, 3] = -0.2  # Trimmed / opposing minority -> excluded from aligned positive mean
    t1[4, 0] = 3.0   # Pure prose input coordinate -> annihilated by P_0
    t2[4, 0] = 3.0
    t3[5, 5] = 0.01  # Below top-density threshold

    summary = ties_elect_and_project_increments(
        [t1, t2, t3],
        p_null,
        density=0.25,
        step_weights=[1.0, 1.0, 1.0],
    )
    assert summary.n_steps == 3
    assert abs(float(summary.delta_elected[2, 2].item())) < 1e-6
    assert abs(float(summary.delta_elected[3, 3].item()) - 2.0) < 1e-5
    assert abs(float(summary.delta_projected[4, 0].item())) < 1e-6
    assert summary.sign_conflict_cancelled_fraction > 0.25

    # Dense noise tensor with density=0.1 verifies top-k magnitude thresholding drops low noise.
    dense_step = torch.linspace(0.001, 0.05, d_in * d_in, dtype=torch.float32).reshape(d_in, d_in)
    dense_step[3, 3] = 5.0
    dense_summary = ties_elect_and_project_increments([dense_step], p_null, density=0.1)
    assert abs(float(dense_summary.delta_elected[3, 3].item()) - 5.0) < 1e-5
    assert dense_summary.trimmed_nonzero_fraction <= 0.25

    # Full density=1.0 and all-zero tensors edge paths.
    zero_summary = ties_elect_and_project_increments([torch.zeros(d_in, d_in)], p_null, density=1.0)
    assert zero_summary.elected_frobenius_norm == 0.0
    assert zero_summary.sign_conflict_cancelled_fraction == 0.0
    zero_trim = ties_elect_and_project_increments([torch.zeros(d_in, d_in)], p_null, density=0.1)
    assert zero_trim.projected_frobenius_norm == 0.0

    # Zero-activation prose matrix yields identity projector; bad threshold raises ValueError.
    assert torch.allclose(compute_prose_nullspace_projector(torch.zeros(4, 6)), torch.eye(6))
    with pytest.raises(ValueError):
        compute_prose_nullspace_projector(torch.ones(4, 6), variance_threshold=0.0)


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_and_provenance() -> None:
    """S4: Infeasible guard forces ANSE 1e6 Maximum Pain, bitwise rollback, and tamper-evident receipt."""
    fx = _build_pilot7_synthetic_chain_fixture()
    w_before = fx["mutable_state"]["layer_weight"].clone()
    norm_before = fx["mutable_state"]["final_norm_weight"].clone()
    init_sha = compute_rescue_state_sha256(w_before, norm_before)

    # Pass an identity projector (zero prose protection) + uncalibrated grid with only raw corrupting steps
    # and an ultra-tight guard ceiling so no candidate can satisfy both Delta E < 0 and guard_ok.
    d_model = w_before.shape[0]
    anti_proj = torch.zeros(d_model, d_model, dtype=torch.float32)  # Zeroes all specialist updates!
    cfg = SequentialChainRescueConfig(
        ties_density=0.25,
        ties_alpha_grid=(0.5, 1.0),
        donor_alpha_grid=(0.0,),
        safe_tau_grid=(1.0,),  # No temperature scaling -> candidate identical to base -> Delta E == 0!
        max_general_drift_ratio=0.0,
        max_domain_drift_ratio=0.0,
    )
    receipt = execute_verified_chain_rescue(
        fx["mutable_state"],
        fx["lm_head_weight"],
        fx["chain_steps"],
        fx["domain_hidden_inputs"],
        fx["domain_targets"],
        p_null=anti_proj,
        config=cfg,
    )

    assert receipt["accepted"] is False
    assert receipt["rolled_back"] is True
    assert receipt["rollback_verified"] is True
    assert receipt["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert receipt["initial_state_sha256"] == init_sha
    assert receipt["final_state_sha256"] == init_sha
    assert torch.equal(fx["mutable_state"]["layer_weight"], w_before)
    assert torch.equal(fx["mutable_state"]["final_norm_weight"], norm_before)
    assert verify_chain_rescue_receipt_provenance(receipt) is True

    # Tampering must invalidate receipt provenance.
    tampered = dict(receipt)
    tampered["anse_energy"] = -0.5
    assert verify_chain_rescue_receipt_provenance(tampered) is False

    bad_rollback = dict(receipt)
    bad_rollback["final_state_sha256"] = "0" * 64
    assert verify_chain_rescue_receipt_provenance(bad_rollback) is False
    assert verify_chain_rescue_receipt_provenance("not-a-dict") is False  # type: ignore[arg-type]
    assert verify_chain_rescue_receipt_provenance({"schema": "bad"}) is False
    assert verify_chain_rescue_receipt_provenance(
        {"schema": RESCUER_SCHEMA, "object": RESCUER_RECEIPT_OBJECT, "receipt_sha256": "xyz"}
    ) is False
    assert verify_chain_rescue_receipt_provenance(
        {
            "schema": RESCUER_SCHEMA,
            "object": RESCUER_RECEIPT_OBJECT,
            "receipt_sha256": "a" * 64,
            "initial_state_sha256": "bad",
            "final_state_sha256": "a" * 64,
        }
    ) is False
    assert verify_chain_rescue_receipt_provenance(
        {
            "schema": RESCUER_SCHEMA,
            "object": RESCUER_RECEIPT_OBJECT,
            "receipt_sha256": "a" * 64,
            "initial_state_sha256": "a" * 64,
            "final_state_sha256": "bad",
        }
    ) is False


def test_s5_validation_errors_and_repository_compliance_gates() -> None:
    """S5: Thorough input validation + ZeroStubAudit, PRSecurityScanner, OwnerOrderGuardrail, and docs."""
    # Validation checks across helpers and config.
    with pytest.raises(TypeError):
        estimate_factorized_vs_dense_flops(True, 8, 2)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        estimate_factorized_vs_dense_flops(0, 8, 2)
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(ties_density=0.0)
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(safe_tau_grid=(0.95,))
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(safe_tau_grid=())
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(max_general_drift_ratio=-0.1)
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(ece_weight=-1.0)
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(rmsnorm_eps=0.0)
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(prose_invariance_eps=0.0)
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(general_domain_name=" ")
    with pytest.raises(TypeError):
        SequentialChainRescueConfig(ties_density="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SequentialChainRescueConfig(ties_density=float("nan"))

    with pytest.raises(ValueError):
        LoRAChainStep(step_name="", b_mat=torch.ones(4, 2), a_mat=torch.ones(2, 4))
    with pytest.raises(ValueError):
        LoRAChainStep(step_name="bad_rank", b_mat=torch.ones(4, 2), a_mat=torch.ones(3, 4))
    with pytest.raises(TypeError):
        factorized_associative_nullspace_delta("bad", torch.ones(2, 4), torch.eye(4))  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        factorized_associative_nullspace_delta(torch.ones(4), torch.ones(2, 4), torch.eye(4))
    with pytest.raises(ValueError):
        factorized_associative_nullspace_delta(torch.empty(0, 2), torch.ones(2, 4), torch.eye(4))
    with pytest.raises(ValueError):
        factorized_associative_nullspace_delta(
            torch.tensor([[float("nan"), 1.0]]), torch.ones(2, 4), torch.eye(4)
        )
    with pytest.raises(ValueError):
        factorized_associative_nullspace_delta(torch.ones(4, 2), torch.ones(3, 4), torch.eye(4))
    with pytest.raises(ValueError):
        factorized_associative_nullspace_delta(torch.ones(4, 2), torch.ones(2, 4), torch.eye(5))

    with pytest.raises(ValueError):
        extract_incremental_chain_task_vectors([])
    with pytest.raises(TypeError):
        extract_incremental_chain_task_vectors(["invalid"])  # type: ignore[list-item]
    with pytest.raises(ValueError):
        extract_incremental_chain_task_vectors(
            [(torch.ones(4, 2), torch.ones(2, 4)), (torch.ones(5, 2), torch.ones(2, 4))]
        )

    with pytest.raises(ValueError):
        ties_elect_and_project_increments([], torch.eye(4))
    with pytest.raises(ValueError):
        ties_elect_and_project_increments([torch.ones(4, 4)], torch.eye(4), density=1.5)
    with pytest.raises(ValueError):
        ties_elect_and_project_increments([torch.ones(4, 4), torch.ones(5, 4)], torch.eye(4))
    with pytest.raises(ValueError):
        ties_elect_and_project_increments([torch.ones(4, 4)], torch.eye(4), step_weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        ties_elect_and_project_increments([torch.ones(4, 4)], torch.eye(4), step_weights=[0.0])

    fx = _build_pilot7_synthetic_chain_fixture()
    with pytest.raises(TypeError):
        execute_verified_chain_rescue(
            "bad",  # type: ignore[arg-type]
            fx["lm_head_weight"],
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            {},
            fx["lm_head_weight"],
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            {"layer_weight": torch.zeros(4, 5), "final_norm_weight": torch.ones(4)},
            torch.zeros(6, 4),
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            {"layer_weight": torch.zeros(4, 4), "final_norm_weight": torch.ones(5)},
            torch.zeros(6, 4),
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(TypeError):
        execute_verified_chain_rescue(
            {"layer_weight": torch.zeros(4, 4), "final_norm_weight": "bad"},  # type: ignore[dict-item]
            torch.zeros(6, 4),
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            {"layer_weight": torch.zeros(4, 4), "final_norm_weight": torch.ones(4, 1)},
            torch.zeros(6, 4),
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            {"layer_weight": torch.zeros(4, 4), "final_norm_weight": torch.empty(0)},
            torch.zeros(6, 4),
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            {"layer_weight": torch.zeros(4, 4), "final_norm_weight": torch.tensor([float("inf"), 1.0, 1.0, 1.0])},
            torch.zeros(6, 4),
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            fx["domain_targets"],
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            fx["mutable_state"], fx["lm_head_weight"], fx["chain_steps"], {}, {}
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            fx["mutable_state"],
            fx["lm_head_weight"],
            fx["chain_steps"],
            {"D1_code": fx["domain_hidden_inputs"]["D1_code"]},
            {"D1_code": fx["domain_targets"]["D1_code"]},
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            fx["mutable_state"],
            fx["lm_head_weight"],
            fx["chain_steps"],
            fx["domain_hidden_inputs"],
            {"D0_general_prose": fx["domain_targets"]["D0_general_prose"]},
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            fx["mutable_state"],
            fx["lm_head_weight"],
            fx["chain_steps"],
            {"D0_general_prose": torch.zeros(4, 8)},
            {"D0_general_prose": torch.zeros(4, dtype=torch.long)},
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            fx["mutable_state"],
            fx["lm_head_weight"],
            fx["chain_steps"],
            {"D0_general_prose": fx["domain_hidden_inputs"]["D0_general_prose"]},
            {"D0_general_prose": torch.zeros(2, dtype=torch.long)},
        )
    with pytest.raises(ValueError):
        execute_verified_chain_rescue(
            fx["mutable_state"],
            fx["lm_head_weight"],
            fx["chain_steps"],
            {"D0_general_prose": fx["domain_hidden_inputs"]["D0_general_prose"]},
            {"D0_general_prose": torch.full((64,), 99, dtype=torch.long)},
        )

    # Confirm direct helper call works cleanly too.
    nll_v, ece_v = compute_nll_and_ece_from_logits(
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]), torch.tensor([0, 1]), ece_bins=5
    )
    assert nll_v > 0.0 and ece_v >= 0.0

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    # Repository security, AGPL-3.0, stub, forbidden-pattern, and OwnerOrderGuardrail checks.
    for rel, path in (
        ("src/hivemind/swarm/sequential_chain_ties_nullspace_rescuer.py", TARGET_MODULE_PATH),
        ("tests/test_sequential_chain_ties_nullspace_rescuer_scenarios.py", TEST_FILE_PATH),
    ):
        text = path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in text[:300]
        auditor = ZeroStubAudit() if ZeroStubAudit is not None else _FallbackZeroStubAudit()
        audit_res = auditor.audit_python_code(text)
        assert audit_res.is_clean is True
        assert list(audit_res.violations) == []
        changed_files.append(
            {
                "filename": rel,
                "status": "added",
                "additions": len(text.splitlines()),
                "patch": "",
                "content": text,
            }
        )

    # Verify NeuroController roadmap & implementation plan docs reflect Reine-Cortex & DYN-42 cleanly.
    for doc_rel, doc_path in (
        ("docs/NEURO_CONTROLLER_ROADMAP.md", ROADMAP_DOC_PATH),
        ("docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md", PLAN_DOC_PATH),
    ):
        doc_text = doc_path.read_text(encoding="utf-8")
        assert "0.00 EUR / 0.00 USD" in doc_text
        assert "NeuroController" in doc_text
        assert "Reine-Cortex" in doc_text
        assert "NEURAL-CTRL-DYN-42" in doc_text
        assert scanner.scan_text_for_payment_solicitation(doc_text, doc_rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(doc_text, doc_rel) == []
        assert scanner.scan_unicode_trojan_source(doc_text, doc_rel) == []
        intent = classify_request_intent(doc_text, source_channel="pull_request", changed_files=(doc_rel,))
        assert intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
        assert intent.is_directive_order is False
        assert intent.protected_paths_touched == ()
        changed_files.append(
            {
                "filename": doc_rel,
                "status": "modified",
                "additions": len(doc_text.splitlines()),
                "patch": "",
                "content": doc_text,
            }
        )

    gk = _load_module_from_path("pr_gatekeeper", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    rel_files = [item["filename"] for item in changed_files]
    gwaya_res = gk.gwaya_audit(REPO_ROOT, rel_files[:2])
    assert gwaya_res["passed"] is True
    assert gwaya_res["violations"] == {}
    for item in changed_files:
        assert gk.forbidden_scan(item["content"]) == []

    pr_title = (
        "feat(swarm): sequential LoRA chain forgetting rescuer via incremental TIES "
        "task vectors and factorized null-space projection"
    )
    pr_body = (
        "Implements NEURAL-CTRL-DYN-42 training-free sequential QLoRA chain forgetting rescue via "
        "incremental TIES sign election, associative low-RSS factorized P_0 null-space projection, "
        "and safe folded RMSNorm calibration under ANSE Delta E < 0 and D0 <= +1.0% with atomic "
        "bitwise rollback, updating the Reine-Cortex NeuroController roadmap and implementation plan "
        "under the 0.00 EUR / 0.00 USD AGPL-3.0 charter."
    )
    pr_intent = classify_request_intent(
        f"{pr_title}\n{pr_body}",
        source_channel="pull_request",
        changed_files=rel_files,
    )
    assert pr_intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
    assert pr_intent.is_directive_order is False
    assert pr_intent.protected_paths_touched == ()

    scan = scanner.audit_pull_request(
        pr_number=1242,
        pr_title=pr_title,
        pr_body=pr_body,
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[pr_title],
        author_email=None,
    )
    assert scan.passed is True
    assert scan.blocked_merge is False
    assert scan.risk_score == 0
    assert scan.findings == []
