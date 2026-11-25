# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-33: hidden-state isotropy & RankMe representation-collapse guard."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.representation_collapse_wsp_guard import (
    ANSE_MAXIMUM_PAIN,
    GUARD_RECEIPT_OBJECT,
    GUARD_SCHEMA,
    CollapseGuardConfig,
    compute_state_sha256,
    execute_verified_collapse_guarded_wsp_surgery,
    measure_hidden_isotropy_and_rankme,
    synthesize_anisotropy_centering_and_nullspace_repair,
    verify_collapse_guard_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "representation_collapse_wsp_guard.py"
)
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


def _eval_domain_nlls(
    model: CrowdV1WithEngram,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 1,
) -> dict[str, float]:
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    with torch.no_grad():
        h_pre = forward_model_pre_ln(model, ids, batch_size=16)
        nlls = per_chunk_nll_from_pre_ln(
            h_pre,
            model.ln.weight,
            model.ln.bias,
            float(getattr(model.ln, "eps", 1e-5)),
            model.head.weight,
            ids,
            score_from=score_from,
        ).tolist()
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(chunks, nlls, strict=True):
        grouped.setdefault(str(ch["expert_id"]), []).append(float(val))
    return {k: float(sum(v) / float(len(v))) for k, v in sorted(grouped.items())}


def _build_8expert_moe_fixture(*, seed: int = 20261133) -> dict[str, Any]:
    """Build deterministic 8-expert CrowdV1 MoE with orthogonal domain subspaces."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    num_experts = 8
    block_h = 8
    block_i = 16
    hidden_size = num_experts * block_h
    intermediate_size = num_experts * block_i
    vocab_size = num_experts * 8

    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_dirs = torch.zeros((num_experts, hidden_size), dtype=torch.float32)
    for d_idx in range(num_experts):
        vec = torch.tensor(
            [1.0 if i % 2 == 0 else -1.0 for i in range(block_h)],
            dtype=torch.float32,
        )
        u_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h] = vec / vec.norm()

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(num_experts):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            t0, t1 = d_idx * 8, (d_idx + 1) * 8
            noise = 0.20 * torch.randn((8, block_h), generator=gen, dtype=torch.float32)
            noise[:, :2] = 0.0
            wte[t0:t1, h0:h1] = noise
            wte[t0:t1] += 1.5 * u_dirs[d_idx].unsqueeze(0)
            wte[t0 + 2] = 2.6 * u_dirs[d_idx]

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        blk = model.blocks[0]
        blk.ln1.weight.fill_(1.0)
        blk.ln1.bias.zero_()
        blk.ln2.weight.fill_(1.0)
        blk.ln2.bias.zero_()
        blk.attn.v.weight[:block_h, :block_h].copy_(0.08 * torch.eye(block_h))
        blk.attn.o.weight[:block_h, :block_h].copy_(0.08 * torch.eye(block_h))
        for d_idx in range(num_experts):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            i0 = d_idx * block_i
            blk.mlp.w1.weight[i0 : i0 + block_h, h0:h1].copy_(0.90 * torch.eye(block_h))
            blk.mlp.w2.weight[i0 : i0 + block_h, h0:h1].copy_(0.90 * torch.eye(block_h))
            blk.mlp.w3.weight[h0:h1, i0 : i0 + block_h].copy_(0.55 * torch.eye(block_h))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=2,
        lora_rank=4,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 7,
    )

    with torch.no_grad():
        mlp = model.blocks[0].mlp
        mlp.router.weight.zero_()
        for d_idx in range(num_experts):
            mlp.router.weight[d_idx] = 6.0 * u_dirs[d_idx]
            exp = mlp.experts[str(d_idx)]
            exp.w3_A.zero_()
            exp.w3_B.zero_()
            i0 = d_idx * block_i
            exp.w3_A[0, i0 : i0 + block_h] = u_dirs[d_idx, d_idx * block_h : (d_idx + 1) * block_h]

    domain_names = (
        "E0_general_fineweb",
        "E1_python_code",
        "E2_math_reasoning",
        "E3_formal_logic",
        "E4_physics_stem",
        "E5_exoplanet",
        "E6_stellar_spec",
        "E7_defensive_cyber",
    )

    def _make_split(n_per_dom: int, split_gen: torch.Generator) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for d_idx, eid in enumerate(domain_names):
            lo, hi = d_idx * 8, (d_idx + 1) * 8
            tgt = lo + 2
            for _ in range(n_per_dom):
                toks = torch.randint(lo, hi, (12,), generator=split_gen).tolist()
                for pos in range(4, 12):
                    toks[pos] = tgt
                rows.append({"expert_id": eid, "ids": toks})
        return rows

    cal_chunks = _make_split(2, torch.Generator(device="cpu").manual_seed(seed + 101))
    eval_chunks = _make_split(2, torch.Generator(device="cpu").manual_seed(seed + 202))
    return {
        "model": model,
        "cal_chunks": cal_chunks,
        "eval_chunks": eval_chunks,
        "u_dirs": u_dirs,
        "domain_names": domain_names,
        "hidden_size": hidden_size,
        "block_h": block_h,
    }


# ---------------------------------------------------------------------------
# S1: Aggressive MoE Edit Collapsing RankMe >40% Caught & Repaired in Closed Form
# ---------------------------------------------------------------------------


def test_s1_aggressive_moe_edit_collapse_caught_and_repaired_closed_form() -> None:
    """S1: Benign-on-E1 edit collapsing multi-domain RankMe >40% is repaired inside P_0."""
    fix = _build_8expert_moe_fixture(seed=20261133)
    model: CrowdV1WithEngram = fix["model"]
    cal_chunks = fix["cal_chunks"]
    eval_chunks = fix["eval_chunks"]
    u_dirs: torch.Tensor = fix["u_dirs"]

    with torch.no_grad():
        model.ln.weight.fill_(0.25)

    pre_nlls = _eval_domain_nlls(model, eval_chunks, score_from=4)
    pre_mean_nll = sum(pre_nlls.values()) / float(len(pre_nlls))

    def _aggressive_single_domain_edit(m: nn.Module) -> None:
        with torch.no_grad():
            # Genuine specialist boost on E1_python_code (w3_B aligns output with E1 target direction)
            exp1 = m.blocks[0].mlp.experts["1"]
            exp1.w3_B[:, 0] = 0.85 * u_dirs[1]
            # Aggressive collateral anisotropy cone & off-domain channel collapse on final LN
            m.ln.weight.fill_(0.08)
            m.ln.weight[8:16].fill_(1.45)
            m.ln.bias.copy_(1.5 * u_dirs[1])

    # Verify raw unguarded candidate improves E1 while silently collapsing RankMe >40% & breaking E0
    unguarded = copy.deepcopy(model)
    _aggressive_single_domain_edit(unguarded)
    unguarded_nlls = _eval_domain_nlls(unguarded, eval_chunks, score_from=4)
    unguarded_mean_nll = sum(unguarded_nlls.values()) / float(len(unguarded_nlls))
    unguarded_e0_pct = 100.0 * (
        unguarded_nlls["E0_general_fineweb"] / pre_nlls["E0_general_fineweb"] - 1.0
    )
    assert unguarded_nlls["E1_python_code"] < pre_nlls["E1_python_code"] - 0.5
    assert unguarded_mean_nll > pre_mean_nll
    assert unguarded_e0_pct > 2.0

    receipt = execute_verified_collapse_guarded_wsp_surgery(
        model,
        cal_chunks,
        candidate_edit_fn=_aggressive_single_domain_edit,
        eval_chunks=eval_chunks,
        repair_target=model.ln,
        p_null=None,
        min_rankme_ratio=0.85,
        max_anisotropy_delta=0.05,
        general_guard_max_ratio=1.005,
        domain_guard_max_ratio=1.01,
        score_from=4,
        inplace=True,
    )

    assert receipt["object"] == GUARD_RECEIPT_OBJECT
    assert receipt["schema"] == GUARD_SCHEMA
    assert verify_collapse_guard_receipt_provenance(receipt) is True
    assert receipt["collapse_detected"] is True
    assert receipt["repair_attempted"] is True
    assert receipt["repair_applied"] is True
    assert receipt["collapse_cleared"] is True
    assert receipt["accepted"] is True
    assert receipt["committed"] is True
    assert receipt["rolled_back"] is False
    assert receipt["status"] == "ACCEPTED_REPAIRED"
    assert receipt["guard_ok"] is True

    # Post-edit (pre-repair) dropped RankMe by >40%; closed-form repair restored >=85%
    assert receipt["rankme_ratio_post"] < 0.60
    assert receipt["rankme_ratio_final"] >= 0.85
    assert receipt["anisotropy_post"] > receipt["anisotropy_pre"] + 0.25
    assert receipt["anisotropy_delta_final"] <= 0.05
    assert receipt["isotropy_index_final"] > receipt["isotropy_index_post"]

    # Multi-domain NLL strictly lowered vs both parent and collapsed candidate; E0 <= +0.5%
    assert receipt["nll_mean"] < pre_mean_nll
    assert receipt["nll_mean"] < unguarded_mean_nll
    assert receipt["delta_e"] < 0.0
    assert receipt["general_pct_vs_base"] <= 0.5
    assert receipt["repair_reports"][0]["nullspace_drift_norm"] < 1e-6


# ---------------------------------------------------------------------------
# S2: Exact RankMe & Participation-Ratio Monotonicity (Cone -> Isotropic)
# ---------------------------------------------------------------------------


def test_s2_exact_rankme_and_participation_ratio_monotonicity() -> None:
    """S2: RankMe & PR increase monotonically from rank-1 cone to isotropic Gaussian."""
    gen = torch.Generator(device="cpu").manual_seed(202602)
    n_tok, dim = 256, 32

    # 1. Exact rank-1 cone
    v0 = torch.randn((1, dim), generator=gen, dtype=torch.float64)
    v0 = v0 / v0.norm()
    cone = v0.expand(n_tok, dim).clone()
    m_cone = measure_hidden_isotropy_and_rankme(cone, eps=1e-8)
    assert m_cone["rankme"] == pytest.approx(1.0, abs=1e-10)
    assert m_cone["participation_ratio"] == pytest.approx(1.0, abs=1e-10)
    assert m_cone["mean_pairwise_cosine_anisotropy"] == pytest.approx(1.0, abs=1e-10)
    assert m_cone["isotropy_index"] == pytest.approx(0.0, abs=1e-10)
    assert m_cone["collapse_warning"] is True
    assert "high_anisotropy_cone" in m_cone["collapse_warnings"]
    assert "low_effective_rankme" in m_cone["collapse_warnings"]

    # 2. Continuous interpolation from rank-1 cone to isotropic zero-mean Gaussian
    z_iso = torch.randn((n_tok, dim), generator=gen, dtype=torch.float64) / math.sqrt(float(dim))
    alphas = (0.0, 0.05, 0.15, 0.35, 0.65, 1.0)
    series = [
        measure_hidden_isotropy_and_rankme((1.0 - a) * cone + a * z_iso, eps=1e-8)
        for a in alphas
    ]
    for prev, nxt in zip(series[:-1], series[1:], strict=True):
        assert nxt["rankme"] > prev["rankme"]
        assert nxt["participation_ratio"] > prev["participation_ratio"]
        assert nxt["mean_pairwise_cosine_anisotropy"] < prev["mean_pairwise_cosine_anisotropy"]
        assert nxt["isotropy_index"] > prev["isotropy_index"]

    m_iso = series[-1]
    assert m_iso["collapse_warning"] is False
    assert m_iso["isotropy_index"] > 0.90
    assert m_iso["rankme"] > 0.75 * float(dim)

    # 3. Exact discrete rank progression r in {1, 2, 4, 8, 16, 32}
    q_orth, _ = torch.linalg.qr(torch.randn((n_tok, dim), generator=gen, dtype=torch.float64))
    prev_r = 0.0
    for r_val in (1, 2, 4, 8, 16, 32):
        h_r = q_orth[:, :r_val] @ torch.eye(r_val, dim, dtype=torch.float64)
        mr = measure_hidden_isotropy_and_rankme(h_r, eps=1e-8)
        assert mr["rankme"] == pytest.approx(float(r_val), rel=1e-9)
        assert mr["participation_ratio"] == pytest.approx(float(r_val), rel=1e-9)
        assert mr["rankme"] > prev_r
        prev_r = mr["rankme"]

    # 4. 3D tensor equivalence, zero-tensor collapse, 1-token collapse, and reference comparison
    h3d = z_iso.view(8, 32, dim).to(torch.float32)
    m3d = measure_hidden_isotropy_and_rankme(h3d, reference_metrics=m_iso)
    assert m3d["n_tokens"] == n_tok
    assert m3d["collapse_warning"] is False
    assert m3d["rankme_ratio_vs_ref"] == pytest.approx(1.0, abs=1e-5)

    m_zero = measure_hidden_isotropy_and_rankme(torch.zeros((8, dim), dtype=torch.float32))
    assert m_zero["rankme"] == 0.0
    assert m_zero["participation_ratio"] == 0.0
    assert m_zero["collapse_warning"] is True
    assert "zero_or_degenerate_hidden_states" in m_zero["collapse_warnings"]

    m_single = measure_hidden_isotropy_and_rankme(torch.ones((1, dim), dtype=torch.float64))
    assert m_single["anisotropy"] == 1.0
    assert m_single["collapse_warning"] is True

    m_drop = measure_hidden_isotropy_and_rankme(cone, reference_metrics=m_iso)
    assert "rankme_drop_vs_reference" in m_drop["collapse_warnings"]
    assert "anisotropy_spike_vs_reference" in m_drop["collapse_warnings"]


# ---------------------------------------------------------------------------
# S3: Exact Prose Null-Space P_0 Preservation (< 1e-5 Drift)
# ---------------------------------------------------------------------------


def test_s3_exact_prose_nullspace_p0_preservation() -> None:
    """S3: Closed-form centering and singular-floor repair preserve ker(P_0) within <1e-5."""
    gen = torch.Generator(device="cpu").manual_seed(202603)
    n_tok, d_out, d_in = 128, 16, 20

    # Orthonormal protected prose subspace Q_prose (rank 4) -> P_0 = I - Q_prose Q_prose^T
    q_prose, _ = torch.linalg.qr(
        torch.randn((d_out, 4), generator=gen, dtype=torch.float64), mode="reduced"
    )
    p0_dense = torch.eye(d_out, dtype=torch.float64) - q_prose @ q_prose.T

    # 1. 2D Linear layer WITH bias under combined mean-cone + singular collapse
    lin_bias = nn.Linear(d_in, d_out, bias=True)
    with torch.no_grad():
        lin_bias.weight.copy_(torch.randn((d_out, d_in), generator=gen, dtype=torch.float32))
        lin_bias.bias.copy_(torch.randn((d_out,), generator=gen, dtype=torch.float32))

    x_in = torch.randn((n_tok, d_in), generator=gen, dtype=torch.float32)
    h_pre = torch.randn((n_tok, d_out), generator=gen, dtype=torch.float64)
    # Collapse inside P_0: shrink orthogonal subspace by 0.15 and add strong mean cone inside P_0
    cone_dir = p0_dense @ torch.ones((d_out,), dtype=torch.float64)
    cone_dir = cone_dir / cone_dir.norm()
    shrink_op = (torch.eye(d_out, dtype=torch.float64) - p0_dense) + 0.15 * p0_dense
    h_post_lin = h_pre @ shrink_op.T + 6.0 * cone_dir.unsqueeze(0)

    with torch.no_grad():
        y_orig = lin_bias(x_in)
    rep_lin, diag_lin = synthesize_anisotropy_centering_and_nullspace_repair(
        lin_bias, h_pre, h_post_lin, p0_dense, inplace=False
    )
    assert diag_lin["repair_needed"] is True
    assert diag_lin["repair_applied"] is True
    assert diag_lin["singular_floor_applied"] is True
    assert diag_lin["collapse_cleared"] is True
    assert diag_lin["rankme_recovery_ratio"] >= 0.85
    assert diag_lin["nullspace_drift_norm"] < 1e-5

    with torch.no_grad():
        y_rep = rep_lin(x_in)
        prose_out_drift = float(
            ((y_rep - y_orig).to(torch.float64) @ q_prose).abs().max().item()
        )
    assert prose_out_drift < 1e-5

    # Pure mean-cone collapse on linear WITH bias (triggers Step 1 centering-only path)
    h_post_cone_only = h_pre + 5.0 * cone_dir.unsqueeze(0)
    rep_cone, diag_cone = synthesize_anisotropy_centering_and_nullspace_repair(
        lin_bias, h_pre, h_post_cone_only, p0_dense, inplace=False
    )
    assert diag_cone["repair_applied"] is True
    assert diag_cone["singular_floor_applied"] is False
    assert diag_cone["collapse_cleared"] is True
    assert torch.allclose(rep_cone.weight, lin_bias.weight)
    assert diag_cone["nullspace_drift_norm"] < 1e-5

    # 2. Biasless 2D Linear / raw 2D tensor + soft PSD filter P_0
    lin_nobias = nn.Linear(d_in, d_out, bias=False)
    rep_nobias, diag_nobias = synthesize_anisotropy_centering_and_nullspace_repair(
        lin_nobias, h_pre, h_post_lin, p0_dense, inplace=False
    )
    assert diag_nobias["repair_applied"] is True
    assert diag_nobias["collapse_cleared"] is True
    assert diag_nobias["nullspace_drift_norm"] < 1e-5
    with torch.no_grad():
        nb_drift = float(
            ((rep_nobias(x_in) - lin_nobias(x_in)).to(torch.float64) @ q_prose).abs().max().item()
        )
    assert nb_drift < 1e-5

    w_raw = lin_nobias.weight.detach().clone()
    soft_p = 0.7 * p0_dense + 0.3 * torch.eye(d_out, dtype=torch.float64)
    w_soft, diag_soft = synthesize_anisotropy_centering_and_nullspace_repair(
        w_raw, h_pre, h_post_lin, soft_p, inplace=False
    )
    assert diag_soft["repair_applied"] is True
    assert tuple(w_soft.shape) == tuple(w_raw.shape)

    # Zero-rank P_0 leaves target untouched
    _, diag_zero_p = synthesize_anisotropy_centering_and_nullspace_repair(
        w_raw, h_pre, h_post_lin, torch.zeros((d_out, d_out)), inplace=False
    )
    assert diag_zero_p["repair_applied"] is False

    # Healthy post states require no repair
    w_healthy, diag_healthy = synthesize_anisotropy_centering_and_nullspace_repair(
        w_raw, h_pre, h_pre.clone(), p0_dense, inplace=False
    )
    assert diag_healthy["repair_needed"] is False
    assert diag_healthy["repair_applied"] is False
    assert torch.allclose(w_healthy, w_raw)

    # 3. LayerNorm pure cone (dense P_0) + combined variance collapse (coordinate P_0) + RMSNorm/mapping/tuple
    ln_mod = nn.LayerNorm(d_out)
    x_ln_in = torch.randn((n_tok, d_out), generator=gen, dtype=torch.float32)
    rep_ln_cone, diag_ln_cone = synthesize_anisotropy_centering_and_nullspace_repair(
        ln_mod, h_pre, h_post_cone_only, p0_dense, inplace=False
    )
    assert diag_ln_cone["repair_applied"] is True
    assert diag_ln_cone["singular_floor_applied"] is False
    assert diag_ln_cone["nullspace_drift_norm"] < 1e-5
    with torch.no_grad():
        ln_drift = float(
            ((rep_ln_cone(x_ln_in) - ln_mod(x_ln_in)).to(torch.float64) @ q_prose).abs().max().item()
        )
    assert ln_drift < 1e-5

    p0_coord = torch.eye(d_out, dtype=torch.float64)
    p0_coord[:4, :4] = 0.0
    h_post_coord = h_pre.clone()
    h_post_coord[:, 4:] = 0.05 * h_post_coord[:, 4:] + 4.0
    rep_ln_var, diag_ln_var = synthesize_anisotropy_centering_and_nullspace_repair(
        ln_mod, h_pre, h_post_coord, p0_coord, inplace=False
    )
    assert diag_ln_var["repair_applied"] is True
    assert diag_ln_var["singular_floor_applied"] is True
    assert diag_ln_var["collapse_cleared"] is True
    assert diag_ln_var["nullspace_drift_norm"] < 1e-5
    with torch.no_grad():
        assert float((rep_ln_var.weight[:4] - ln_mod.weight[:4]).abs().max().item()) < 1e-6
        assert float((rep_ln_var.bias[:4] - ln_mod.bias[:4]).abs().max().item()) < 1e-6

    # RMSNorm (biasless 1D norm), Mapping, Tuple, and inplace=True tensor targets
    rms_mod = nn.RMSNorm(d_out)
    rep_rms, diag_rms = synthesize_anisotropy_centering_and_nullspace_repair(
        rms_mod, h_pre, h_post_coord, p0_coord, inplace=False
    )
    assert diag_rms["repair_applied"] is True
    assert diag_rms["nullspace_drift_norm"] < 1e-5
    assert float((rep_rms.weight[:4] - rms_mod.weight[:4]).abs().max().item()) < 1e-6

    rep_map, diag_map = synthesize_anisotropy_centering_and_nullspace_repair(
        {"weight": ln_mod.weight.clone(), "bias": ln_mod.bias.clone()},
        h_pre,
        h_post_coord,
        p0_coord,
    )
    assert diag_map["repair_applied"] is True
    assert "weight" in rep_map and "bias" in rep_map

    rep_tup, diag_tup = synthesize_anisotropy_centering_and_nullspace_repair(
        (w_raw.clone(), None), h_pre, h_post_lin, p0_dense
    )
    assert diag_tup["repair_applied"] is True
    assert isinstance(rep_tup, tuple) and rep_tup[1] is None

    w_ip = w_raw.clone()
    w_ip_ret, _ = synthesize_anisotropy_centering_and_nullspace_repair(
        w_ip, h_pre, h_post_lin, p0_dense, inplace=True
    )
    assert w_ip_ret is w_ip

    # Raw 1D tensor, 2D mapping, 1D tuple, and exact float64 Linear (<1e-12 drift)
    t1d_rep, diag_1d = synthesize_anisotropy_centering_and_nullspace_repair(
        rms_mod.weight.clone(), h_pre, h_post_coord, p0_coord
    )
    assert diag_1d["repair_applied"] is True and t1d_rep.ndim == 1

    m2d_rep, diag_m2d = synthesize_anisotropy_centering_and_nullspace_repair(
        {"weight": w_raw.clone(), "bias": None}, h_pre, h_post_lin, p0_dense
    )
    assert diag_m2d["repair_applied"] is True and m2d_rep["weight"].ndim == 2

    tup1d_rep, diag_t1d = synthesize_anisotropy_centering_and_nullspace_repair(
        (ln_mod.weight.clone(), ln_mod.bias.clone()), h_pre, h_post_coord, p0_coord
    )
    assert diag_t1d["repair_applied"] is True and tup1d_rep[1] is not None

    lin64 = nn.Linear(d_in, d_out, bias=True, dtype=torch.float64)
    rep_64, diag_64 = synthesize_anisotropy_centering_and_nullspace_repair(
        lin64, h_pre, h_post_lin, p0_dense
    )
    assert diag_64["nullspace_drift_norm"] < 1e-12
    with torch.no_grad():
        drift_64 = float(((rep_64(x_in.double()) - lin64(x_in.double())) @ q_prose).abs().max().item())
    assert drift_64 < 1e-12


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Rollback on Unrecoverable Collapse
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_and_edge_paths() -> None:
    """S4: Unrecoverable collapse, guard violations, and eval errors trigger ANSE 1e6 atomic rollback."""
    fix = _build_8expert_moe_fixture(seed=202604)
    model: CrowdV1WithEngram = fix["model"]
    cal_chunks = fix["cal_chunks"]
    eval_chunks = fix["eval_chunks"]
    u_dirs: torch.Tensor = fix["u_dirs"]
    parent_sha = compute_state_sha256(model.state_dict())

    # 1. Hard rank-1 projection across all tokens -> unrecoverable rank collapse
    def _hard_rank1_collapse_edit(m: nn.Module) -> None:
        with torch.no_grad():
            # Zero out LN weight everywhere and set bias to a constant rank-1 vector (zero variance)
            m.ln.weight.zero_()
            m.ln.bias.copy_(2.0 * u_dirs[1])

    rec_unrec = execute_verified_collapse_guarded_wsp_surgery(
        model,
        cal_chunks,
        candidate_edit_fn=_hard_rank1_collapse_edit,
        eval_chunks=eval_chunks,
        repair_target=model.ln,
        score_from=1,
        inplace=True,
    )
    assert rec_unrec["accepted"] is False
    assert rec_unrec["committed"] is False
    assert rec_unrec["rolled_back"] is True
    assert rec_unrec["collapse_detected"] is True
    assert rec_unrec["collapse_cleared"] is False
    assert rec_unrec["status"] == "ROLLED_BACK_UNRECOVERABLE_COLLAPSE"
    assert rec_unrec["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert rec_unrec["final_state_sha256"] == parent_sha
    assert compute_state_sha256(model.state_dict()) == parent_sha
    assert verify_collapse_guard_receipt_provenance(rec_unrec) is True

    # 2. Standalone nn.Sequential with explicit hidden_states_fn / eval_fn rollback modes
    seq_mod = nn.Sequential(nn.Linear(8, 8, bias=True), nn.LayerNorm(8))
    gen = torch.Generator(device="cpu").manual_seed(202605)
    with torch.no_grad():
        seq_mod[0].weight.copy_(torch.eye(8))
        seq_mod[0].bias.zero_()
    x_probe = torch.randn((48, 8), generator=gen, dtype=torch.float32)
    seq_sha = compute_state_sha256(seq_mod.state_dict())

    def _hs_seq(m: nn.Module) -> torch.Tensor:
        return m(x_probe)

    # attempt_closed_form_repair=False on collapsing edit
    rec_norepair = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_edit_fn=lambda m: m[1].bias.data.fill_(8.0),
        hidden_states_fn=_hs_seq,
        eval_fn=lambda _m: {"E0_prose": 0.5, "E1_spec": 0.5},
        attempt_closed_form_repair=False,
    )
    assert rec_norepair["status"] == "ROLLED_BACK_UNRECOVERABLE_COLLAPSE"
    assert rec_norepair["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert compute_state_sha256(seq_mod.state_dict()) == seq_sha

    # Guard violation after healthy edit
    rec_guard = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_edit_fn=lambda m: m[0].weight.data.mul_(1.01),
        hidden_states_fn=_hs_seq,
        eval_fn=lambda m: (
            {"E0_prose": 0.5, "E1_spec": 0.5}
            if float(m[0].weight.detach().abs().max().item()) < 1.005
            else {"E0_prose": 0.65, "E1_spec": 0.40}
        ),
        general_guard_max_ratio=1.01,
    )
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert compute_state_sha256(seq_mod.state_dict()) == seq_sha

    # Non-improving energy rollback
    rec_noimp = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_edit_fn=lambda m: m[0].weight.data.mul_(1.01),
        hidden_states_fn=_hs_seq,
        eval_fn=lambda _m: (0.5, {"E0_prose": 0.5, "E1_spec": 0.5}),
        require_energy_improvement=True,
    )
    assert rec_noimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert compute_state_sha256(seq_mod.state_dict()) == seq_sha

    # Eval failure rollback
    eval_calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, Any]:
        eval_calls["n"] += 1
        if eval_calls["n"] >= 2:
            raise RuntimeError("injected eval failure")
        return {"per_domain": {"E0_prose": 0.5, "E1_spec": 0.5}}

    rec_boom = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_edit_fn=lambda m: m[0].weight.data.mul_(1.01),
        hidden_states_fn=_hs_seq,
        eval_fn=_boom_eval,
    )
    assert rec_boom["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert rec_boom["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert compute_state_sha256(seq_mod.state_dict()) == seq_sha

    # Fixed point unchanged
    rec_fp = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_edit_fn=lambda _m: None,
        hidden_states_fn=_hs_seq,
        eval_fn=lambda _m: {"E0_prose": 0.5, "E1_spec": 0.5},
        require_energy_improvement=False,
    )
    assert rec_fp["status"] == "FIXED_POINT_UNCHANGED"
    assert rec_fp["rolled_back"] is False

    # Healthy candidate_state_dict + inplace=False + multi-target sequence + Crowd hook capture
    cand_sd = {k: v.clone() for k, v in seq_mod.state_dict().items()}
    cand_sd["0.weight"] = cand_sd["0.weight"] * 1.02
    rec_sd = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_state_dict=cand_sd,
        hidden_states_fn=lambda m: {"0": m[0](x_probe), "1": m(x_probe)},
        repair_target=["0", "1"],
        p_null={"0": torch.eye(8)},
        eval_fn=lambda m: (
            {"E0_prose": 0.5, "E1_spec": 0.5}
            if float(m[0].weight.detach().abs().max().item()) < 1.01
            else {"E0_prose": 0.49, "E1_spec": 0.40}
        ),
        inplace=False,
    )
    assert rec_sd["status"] == "ACCEPTED"
    assert rec_sd["accepted"] is True
    assert compute_state_sha256(seq_mod.state_dict()) == seq_sha

    # Hook-based hidden state capture on CrowdV1WithEngram 3D ("blocks.0.mlp") & 2D ("blocks.0.mlp.base_mlp.w3")
    rec_hook = execute_verified_collapse_guarded_wsp_surgery(
        model,
        cal_chunks,
        candidate_edit_fn=lambda m: m.blocks[0].mlp.experts["1"].w3_B[:, 0].fill_(0.2),
        repair_target="blocks.0.mlp.base_mlp.w3",
        score_from=1,
        inplace=False,
        require_energy_improvement=False,
    )
    assert rec_hook["collapse_detected"] is False
    assert compute_state_sha256(model.state_dict()) == parent_sha

    rec_hook3d = execute_verified_collapse_guarded_wsp_surgery(
        model,
        cal_chunks,
        candidate_edit_fn=lambda m: m.blocks[0].mlp.experts["1"].w3_B[:, 0].fill_(0.2),
        repair_target="blocks.0.attn.o",
        score_from=1,
        inplace=False,
        require_energy_improvement=False,
    )
    assert rec_hook3d["collapse_detected"] is False

    # Default repair_target=None auto-resolution on CrowdV1 (model.ln) and custom .norm module
    rec_default_crowd = execute_verified_collapse_guarded_wsp_surgery(
        model,
        cal_chunks,
        candidate_edit_fn=lambda m: m.ln.bias.data.fill_(2.5),
        score_from=1,
        inplace=False,
        require_energy_improvement=False,
    )
    assert rec_default_crowd["collapse_detected"] is True

    class _WithNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(x)

    wn = _WithNorm()
    rec_wn = execute_verified_collapse_guarded_wsp_surgery(
        wn,
        candidate_edit_fn=lambda m: m.norm.bias.data.fill_(5.0),
        hidden_states_fn=lambda m: {"any": m(x_probe)},
        p_null=torch.eye(8),
        eval_fn=lambda _m: {"E0_prose": 0.5, "E1_spec": 0.5},
        require_energy_improvement=False,
        inplace=False,
    )
    assert rec_wn["repair_applied"] is True

    rec_seq_hs = execute_verified_collapse_guarded_wsp_surgery(
        seq_mod,
        candidate_edit_fn=lambda m: m[1].bias.data.fill_(5.0),
        hidden_states_fn=lambda m: [m[0](x_probe), m(x_probe)],
        repair_target=[seq_mod[0], seq_mod[1]],
        p_null={"missing_key": torch.eye(8)},
        eval_fn=lambda _m: {"E0_prose": 0.5, "E1_spec": 0.5},
        require_energy_improvement=False,
        inplace=False,
    )
    assert rec_seq_hs["repair_applied"] is True


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance & Security/Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_provenance_and_security_compliance_gates() -> None:
    """S5: Config/tensor validation, SHA-256 provenance, and HiveMind security/stub gates."""
    # 1. CollapseGuardConfig validation
    with pytest.raises(ValueError, match="min_rankme_ratio"):
        CollapseGuardConfig(min_rankme_ratio=0.0)
    with pytest.raises(ValueError, match="min_rankme_ratio"):
        CollapseGuardConfig(min_rankme_ratio=1.2)
    with pytest.raises(ValueError, match="max_anisotropy_delta"):
        CollapseGuardConfig(max_anisotropy_delta=-0.1)
    with pytest.raises(ValueError, match="eps"):
        CollapseGuardConfig(eps=0.0)
    with pytest.raises(ValueError, match="max_repair_gain"):
        CollapseGuardConfig(max_repair_gain=0.5)
    with pytest.raises(ValueError, match="domain_guard_max_ratio"):
        CollapseGuardConfig(domain_guard_max_ratio=0.95)
    with pytest.raises(ValueError, match="general_guard_max_ratio"):
        CollapseGuardConfig(general_guard_max_ratio=0.99)
    with pytest.raises(ValueError, match="anse_safe_e0_ratio"):
        CollapseGuardConfig(anse_safe_e0_ratio=0.9)
    with pytest.raises(TypeError, match="bool"):
        CollapseGuardConfig(enforce_all_domain_guard="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="general_prefix"):
        CollapseGuardConfig(general_prefix=" ")
    with pytest.raises(ValueError, match=">= 1"):
        CollapseGuardConfig(batch_size=0)
    with pytest.raises(TypeError, match="int"):
        CollapseGuardConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        CollapseGuardConfig(score_from=-1)
    with pytest.raises(ValueError, match="finite"):
        CollapseGuardConfig(max_anisotropy_delta=float("nan"))
    with pytest.raises(TypeError, match="real number"):
        CollapseGuardConfig(min_rankme_ratio="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="<= 1.0"):
        CollapseGuardConfig(max_anisotropy_abs=1.5)
    with pytest.raises(ValueError, match="> 0.0"):
        CollapseGuardConfig(singular_floor_frac=0.0)

    # 2. measure_hidden_isotropy_and_rankme & repair validation
    with pytest.raises(TypeError, match="torch.Tensor"):
        measure_hidden_isotropy_and_rankme([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim >= 2"):
        measure_hidden_isotropy_and_rankme(torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        measure_hidden_isotropy_and_rankme(torch.ones((2, 4), dtype=torch.long))
    bad_h = torch.ones((4, 4), dtype=torch.float32)
    bad_h[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        measure_hidden_isotropy_and_rankme(bad_h)
    with pytest.raises(TypeError, match="Mapping"):
        measure_hidden_isotropy_and_rankme(torch.ones((4, 4)), reference_metrics="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rankme"):
        measure_hidden_isotropy_and_rankme(torch.ones((4, 4)), reference_metrics={})
    with pytest.raises(ValueError, match="anisotropy"):
        measure_hidden_isotropy_and_rankme(torch.ones((4, 4)), reference_metrics={"rankme": 1.0})

    h_ok = torch.randn((8, 6), dtype=torch.float32)
    w_ok = torch.ones((6, 6), dtype=torch.float32)
    with pytest.raises(ValueError, match="match hidden_states_post"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, torch.ones((8, 4)))
    with pytest.raises(ValueError, match="max_repair_gain"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, max_repair_gain=0.5)
    with pytest.raises(ValueError, match="output dimension"):
        synthesize_anisotropy_centering_and_nullspace_repair(torch.ones((4, 6)), h_ok, h_ok)
    with pytest.raises(ValueError, match="bias tensor"):
        synthesize_anisotropy_centering_and_nullspace_repair((w_ok, torch.ones(4)), h_ok, h_ok)
    with pytest.raises(ValueError, match="square 2D"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, torch.ones((4, 6)))
    with pytest.raises(ValueError, match="match hidden dimension"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, torch.eye(4))
    with pytest.raises(TypeError, match="torch.Tensor"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="floating-point"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, torch.ones((6, 6), dtype=torch.long))
    bad_p = torch.eye(6)
    bad_p[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, bad_p)
    asym_p = torch.eye(6)
    asym_p[0, 1] = 2.0
    with pytest.raises(ValueError, match="symmetric"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, asym_p)
    with pytest.raises(ValueError, match="eigenvalues"):
        synthesize_anisotropy_centering_and_nullspace_repair(w_ok, h_ok, h_ok, 2.0 * torch.eye(6))
    with pytest.raises(TypeError, match="out_weight_or_ln"):
        synthesize_anisotropy_centering_and_nullspace_repair("bad", h_ok, h_ok)
    with pytest.raises(ValueError, match="Unsupported nn.Module"):
        synthesize_anisotropy_centering_and_nullspace_repair(nn.ReLU(), h_ok, h_ok)
    with pytest.raises(ValueError, match="1D or 2D"):
        synthesize_anisotropy_centering_and_nullspace_repair(torch.ones((2, 3, 6)), h_ok, h_ok)
    with pytest.raises(ValueError, match="Mapping out_weight_or_ln"):
        synthesize_anisotropy_centering_and_nullspace_repair({}, h_ok, h_ok)
    with pytest.raises(TypeError, match="bias"):
        synthesize_anisotropy_centering_and_nullspace_repair({"weight": w_ok, "bias": "bad"}, h_ok, h_ok)
    with pytest.raises(ValueError, match="1D or 2D"):
        synthesize_anisotropy_centering_and_nullspace_repair({"weight": torch.ones((2, 3, 6))}, h_ok, h_ok)
    with pytest.raises(TypeError, match="Tuple"):
        synthesize_anisotropy_centering_and_nullspace_repair(("bad", None), h_ok, h_ok)
    with pytest.raises(TypeError, match="Tuple"):
        synthesize_anisotropy_centering_and_nullspace_repair((w_ok, "bad"), h_ok, h_ok)
    with pytest.raises(ValueError, match="1D or 2D"):
        synthesize_anisotropy_centering_and_nullspace_repair((torch.ones((2, 3, 6)), None), h_ok, h_ok)
    with pytest.raises(ValueError, match="non-empty and floating-point"):
        synthesize_anisotropy_centering_and_nullspace_repair(torch.ones((6, 6), dtype=torch.long), h_ok, h_ok)
    bad_w = w_ok.clone()
    bad_w[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        synthesize_anisotropy_centering_and_nullspace_repair(bad_w, h_ok, h_ok)
    bad_b = torch.ones(6)
    bad_b[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        synthesize_anisotropy_centering_and_nullspace_repair((w_ok, bad_b), h_ok, h_ok)

    # 3. Orchestrator, chunk, eval & provenance validation
    lin = nn.Linear(6, 6, bias=True)
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_collapse_guarded_wsp_surgery("bad", candidate_edit_fn=lambda _m: None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="exactly one"):
        execute_verified_collapse_guarded_wsp_surgery(lin)
    with pytest.raises(TypeError, match="callable"):
        execute_verified_collapse_guarded_wsp_surgery(lin, candidate_edit_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping"):
        execute_verified_collapse_guarded_wsp_surgery(lin, candidate_state_dict="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin, candidate_edit_fn=lambda _m: None, eval_fn="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="callable"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin, candidate_edit_fn=lambda _m: None, hidden_states_fn="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="CollapseGuardConfig"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin, candidate_edit_fn=lambda _m: None, config="bad"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="hidden_states_fn"):
        execute_verified_collapse_guarded_wsp_surgery(lin, candidate_edit_fn=lambda _m: None)
    with pytest.raises(ValueError, match="Either calibration_chunks"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin, candidate_edit_fn=lambda _m: None, hidden_states_fn=lambda _m: h_ok
        )
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin, [{"expert_id": "E0_a", "ids": [0, 1]}], candidate_edit_fn=lambda _m: None
        )
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_chunks=[{"expert_id": "E0_a", "ids": [0, 1]}],
        )
    with pytest.raises(TypeError, match="repair_target items"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            repair_target=[123],  # type: ignore[list-item]
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(ValueError, match="not be empty"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            repair_target=[],
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(TypeError, match="Unsupported repair_target"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            repair_target=123,  # type: ignore[arg-type]
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(ValueError, match="Could not locate"):
        execute_verified_collapse_guarded_wsp_surgery(
            nn.ReLU(),
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(TypeError, match="non-empty mapping"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: 123,  # type: ignore[arg-type,return-value]
        )
    with pytest.raises(ValueError, match="non-empty"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {},
        )
    with pytest.raises(ValueError, match="Non-finite or non-positive"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": -0.5},
        )
    with pytest.raises(ValueError, match="identical keys"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5},
            baseline_domain_nll={"E0_other": 0.5},
        )
    with pytest.raises(ValueError, match="Missing required general-domain"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E1_only": 0.5},
        )
    rec_key_err = execute_verified_collapse_guarded_wsp_surgery(
        lin,
        candidate_edit_fn=lambda m: m.weight.data.zero_(),
        hidden_states_fn=lambda _m: h_ok,
        eval_fn=lambda m: {"E0_a": 0.5} if float(m.weight.sum()) != 0 else {"E0_b": 0.5},
        baseline_domain_nll={"E0_a": 0.5},
    )
    assert rec_key_err["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert "identical keys" in str(rec_key_err["eval_error"])
    with pytest.raises(ValueError, match="mismatch"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_state_dict={"wrong": torch.ones(1)},
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5, "E1_b": 0.5},
        )
    with pytest.raises(ValueError, match="shape mismatch"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_state_dict={"weight": torch.ones((2, 2)), "bias": torch.ones(6)},
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5, "E1_b": 0.5},
        )
    nan_w = torch.ones((6, 6))
    nan_w[0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_state_dict={"weight": nan_w, "bias": torch.ones(6)},
            hidden_states_fn=lambda _m: h_ok,
            eval_fn=lambda _m: {"E0_a": 0.5, "E1_b": 0.5},
        )
    with pytest.raises(ValueError, match="missing target keys"):
        execute_verified_collapse_guarded_wsp_surgery(
            nn.Sequential(nn.Linear(6, 6), nn.LayerNorm(6)),
            candidate_edit_fn=lambda _m: None,
            repair_target=["0", "1"],
            hidden_states_fn=lambda _m: {"0": h_ok, "wrong": h_ok},
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(ValueError, match="sequence length must match"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: [h_ok, h_ok],
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(TypeError, match="hidden_states_fn must return"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda _m: None,
            hidden_states_fn=lambda _m: 42,
            eval_fn=lambda _m: {"E0_a": 0.5},
        )
    with pytest.raises(TypeError, match="p_null must be"):
        execute_verified_collapse_guarded_wsp_surgery(
            lin,
            candidate_edit_fn=lambda m: m.bias.data.fill_(8.0),
            hidden_states_fn=lambda m: m(torch.randn((24, 6))),
            p_null="bad",  # type: ignore[arg-type]
            eval_fn=lambda _m: {"E0_a": 0.5},
        )

    # Chunk validation via CrowdV1WithEngram
    tiny_m = CrowdV1WithEngram(
        CrowdV1RuntimeConfig(
            vocab_size=16,
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=16,
            max_position_embeddings=16,
            engram_buckets=0,
        )
    )
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_collapse_guarded_wsp_surgery(tiny_m, [], candidate_edit_fn=lambda _m: None)
    with pytest.raises(ValueError, match="expert_id"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m, [{"ids": [1, 2]}], candidate_edit_fn=lambda _m: None
        )
    with pytest.raises(ValueError, match="non-empty string"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m, [{"expert_id": " ", "ids": [1, 2]}], candidate_edit_fn=lambda _m: None
        )
    with pytest.raises(TypeError, match="sequence or tensor"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m, [{"expert_id": "E0_a", "ids": "123"}], candidate_edit_fn=lambda _m: None
        )
    with pytest.raises(ValueError, match="non-negative"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m, [{"expert_id": "E0_a", "ids": [1, -1]}], candidate_edit_fn=lambda _m: None
        )
    with pytest.raises(ValueError, match="identical sequence lengths"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m,
            [{"expert_id": "E0_a", "ids": [1, 2]}, {"expert_id": "E1_b", "ids": [1, 2, 3]}],
            candidate_edit_fn=lambda _m: None,
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m,
            [
                {"expert_id": "E0_a", "ids": torch.tensor([1, 2, 3])},
                {"expert_id": "E1_b", "ids": torch.tensor([1, 2, 3])},
            ],
            candidate_edit_fn=lambda _m: None,
            score_from=4,
        )
    with pytest.raises(ValueError, match="both general"):
        execute_verified_collapse_guarded_wsp_surgery(
            tiny_m, [{"expert_id": "E0_a", "ids": [1, 2, 3]}], candidate_edit_fn=lambda _m: None
        )
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("nan")])})
    with pytest.raises(TypeError, match="mapping"):
        verify_collapse_guard_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_collapse_guard_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_collapse_guard_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_collapse_guard_receipt_provenance(
            {"receipt_sha256": "0" * 64, "object": GUARD_RECEIPT_OBJECT, "schema": "wrong"}
        )
        is False
    )
    assert (
        verify_collapse_guard_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": GUARD_RECEIPT_OBJECT,
                "schema": GUARD_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 4. Repository AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement & gatekeeper gates
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn33", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn33", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/representation_collapse_wsp_guard.py", TARGET_MODULE_PATH),
        ("tests/test_representation_collapse_wsp_guard_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(str(path_obj)) == []
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

    sample_receipt_json = json.dumps(
        {
            "nll_mean": 0.42,
            "participation_ratio": 14.2,
            "isotropy_index": 0.94,
            "anse_energy": -0.18,
            "accepted": True,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1133,
        pr_title="feat(swarm): isotropy and effective-dimensionality representation-collapse guard for open-weight WSP surgery",
        pr_body="Implements NEURAL-CTRL-DYN-33 hidden-state isotropy & RankMe collapse guard with deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): isotropy and effective-dimensionality representation-collapse guard for open-weight WSP surgery"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
