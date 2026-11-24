# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-28: activation-free heavy-tail ESD weight surgeon."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import (
    CrowdRuntimeMLP,
    CrowdV1RuntimeConfig,
    CrowdV1WithEngram,
)
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdFineGrainedMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.heavy_tail_esd_weight_surgeon import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_MAX_ALPHA_CAP,
    SURGEON_RECEIPT_OBJECT,
    SURGEON_SCHEMA,
    HeavyTailESDSurgeonConfig,
    compute_payload_sha256,
    compute_state_sha256,
    execute_verified_esd_moe_surgery,
    fit_esd_power_law_metrics,
    regularize_correlation_trap_spectrum_nullspace,
    verify_esd_surgery_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:  # pragma: no cover - optional in lightweight CI runners
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "heavy_tail_esd_weight_surgeon.py"
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
    score_from: int = 4,
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


def _build_8expert_moe_testbed(
    *,
    trapped_experts: set[int] = frozenset({1, 3}),
    num_layers: int = 2,
    rank: int = 8,
    seed: int = 20261128,
) -> tuple[CrowdV1WithEngram, list[dict[str, Any]], dict[str, float]]:
    """Build a deterministic 8-expert Sparse MoE with optional rank-1 correlation traps."""
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
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=64,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)
    domain_dirs = torch.zeros((num_experts, hidden_size), dtype=torch.float32)

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        for d_idx in range(num_experts):
            h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
            t0_tok, t1_tok = d_idx * 8, (d_idx + 1) * 8
            u_dom = torch.tensor(
                [1.0 if i % 2 == 0 else -1.0 for i in range(block_h)],
                dtype=torch.float32,
            )
            u_dom = u_dom / u_dom.norm()
            domain_dirs[d_idx, h0:h1] = u_dom

            raw = torch.randn((8, block_h), generator=gen, dtype=torch.float32)
            raw = raw - raw.mean(dim=1, keepdim=True)
            raw = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-6) * 1.6 + 1.1 * u_dom.unsqueeze(0)
            raw = raw - raw.mean(dim=1, keepdim=True)
            wte[t0_tok:t1_tok, h0:h1] = raw

        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()

        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.attn.q.weight[:block_h, :block_h].copy_(0.05 * torch.eye(block_h))
            blk.attn.k.weight[: block_h // 2, :block_h].copy_(0.05 * torch.eye(block_h // 2, block_h))
            blk.attn.v.weight[: block_h // 2, :block_h].copy_(0.05 * torch.eye(block_h // 2, block_h))
            blk.attn.o.weight[:block_h, :block_h].copy_(0.05 * torch.eye(block_h))
            for d_idx in range(num_experts):
                h0, h1 = d_idx * block_h, (d_idx + 1) * block_h
                i0, i1 = d_idx * block_i, (d_idx + 1) * block_i
                blk.mlp.w1.weight[i0:i1, h0:h1].copy_(
                    torch.randn((block_i, block_h), generator=gen) * 0.12
                )
                blk.mlp.w2.weight[i0:i1, h0:h1].copy_(
                    torch.randn((block_i, block_h), generator=gen) * 0.12
                )
                w3_b = torch.randn((block_h, block_i), generator=gen) * 0.08
                blk.mlp.w3.weight[h0:h1, i0:i1].copy_(w3_b - w3_b.mean(dim=0, keepdim=True))

        dom_names = (
            "E0_general_fineweb",
            "E1_python_systems_code",
            "E2_math_reasoning",
            "E3_formal_logic_arc",
            "E4_physics_stem",
            "E5_exoplanet_transit",
            "E6_stellar_spectroscopy",
            "E7_defensive_cyber",
        )
        chunks: list[dict[str, Any]] = []
        for d_idx, dom_name in enumerate(dom_names):
            t_base = d_idx * 8
            for row_idx in range(4):
                if d_idx == 0:
                    ta = t_base + (row_idx % 8)
                    tb = t_base + ((row_idx + 3) % 8)
                    seq = [ta] * 8 + [tb] * 8
                else:
                    seq = [t_base + ((row_idx + pos) % 8) for pos in range(16)]
                chunks.append({"expert_id": dom_name, "ids": seq})

        base_nll = _eval_domain_nlls(model, chunks, score_from=4)

        upcycle_crowd_v1_to_sparse_moe(
            model,
            num_experts=num_experts,
            top_k=2,
            lora_rank=rank,
            lora_alpha=float(2 * rank),
            routing_mode="sigma_topk",
            seed=seed + 1,
        )

        prose_dir = domain_dirs[0]
        prose_anti = -model.wte.weight[0:8].mean(dim=0)
        prose_anti = prose_anti / prose_anti.norm().clamp_min(1e-6)

        for blk in model.blocks:
            blk.mlp.router.weight.copy_(domain_dirs * 3.0 - 1.5)
            for t_idx in trapped_experts:
                blk.mlp.router.weight[t_idx, 0:8] += 1.6 * prose_dir[0:8]

            for e_idx in range(num_experts):
                exp = blk.mlp.experts[str(e_idx)]
                exp.w1_A.zero_()
                exp.w1_B.zero_()
                exp.w2_A.zero_()
                exp.w2_B.zero_()
                exp.w3_A.zero_()
                exp.w3_B.zero_()

                h0, h1 = e_idx * block_h, (e_idx + 1) * block_h
                i0, i1 = e_idx * block_i, (e_idx + 1) * block_i
                gamma = 0.32 + 0.03 * (e_idx % 3)
                sv_base = torch.arange(1, rank + 1, dtype=torch.float32).pow(-gamma) * 0.05
                s_sqrt = sv_base.sqrt()

                qa_h, _ = torch.linalg.qr(torch.randn((block_h, rank), generator=gen))
                qi_a, _ = torch.linalg.qr(torch.randn((block_i, rank), generator=gen))
                exp.w1_B[i0:i1, :] = qi_a * s_sqrt.unsqueeze(0)
                exp.w1_A[:, h0:h1] = s_sqrt.unsqueeze(1) * qa_h.T

                qa_h2, _ = torch.linalg.qr(torch.randn((block_h, rank), generator=gen))
                qi_2, _ = torch.linalg.qr(torch.randn((block_i, rank), generator=gen))
                exp.w2_B[i0:i1, :] = qi_2 * s_sqrt.unsqueeze(0)
                exp.w2_A[:, h0:h1] = s_sqrt.unsqueeze(1) * qa_h2.T

                qi_b, _ = torch.linalg.qr(torch.randn((block_i, rank), generator=gen))
                if e_idx in trapped_experts:
                    qh_out, _ = torch.linalg.qr(torch.randn((block_h, rank), generator=gen))
                    sv3 = sv_base.clone()
                    sv3[0] = 4.0
                    qi_in = torch.zeros((intermediate_size, rank), dtype=torch.float32)
                    qi_in[0:block_i, 0] = 1.0 / math.sqrt(block_i)
                    qi_in[i0:i1, 1:] = qi_b[:, 1:]
                    qh_full = torch.zeros((hidden_size, rank), dtype=torch.float32)
                    qh_full[0:block_h, 0] = prose_anti[0:block_h]
                    qh_full[h0:h1, 1:] = qh_out[:, 1:]
                    s3_sq = sv3.sqrt()
                    exp.w3_B.copy_(qh_full * s3_sq.unsqueeze(0))
                    exp.w3_A.copy_(s3_sq.unsqueeze(1) * qi_in.T)
                else:
                    if e_idx == 0:
                        col0 = model.wte.weight[0:8, h0:h1].mean(dim=0, keepdim=True).T
                    else:
                        w_blk = model.wte.weight[e_idx * 8 : (e_idx + 1) * 8, h0:h1]
                        col0 = (w_blk.roll(-1, dims=0) - w_blk).mean(dim=0, keepdim=True).T
                    col0 = col0 / col0.norm().clamp_min(1e-6)
                    qh_seed = torch.cat([col0, torch.randn((block_h, rank - 1), generator=gen)], dim=1)
                    qh_out, r_qh = torch.linalg.qr(qh_seed)
                    if float(r_qh[0, 0].item()) < 0.0:
                        qh_out[:, 0] = -qh_out[:, 0]
                    exp.w3_B[h0:h1, :] = qh_out * s_sqrt.unsqueeze(0)
                    exp.w3_A[:, i0:i1] = s_sqrt.unsqueeze(1) * qi_b.T

    return model, chunks, base_nll


# ---------------------------------------------------------------------------
# S1: 8-Expert MoE with 2 Correlation-Trapped Experts Cured Purely via ESD
# ---------------------------------------------------------------------------


def test_s1_eight_expert_moe_correlation_trap_surgery_and_healthy_preservation() -> None:
    """Cure 2 rank-1 trapped experts (alpha < 1.8) via ESD surgery while keeping healthy specialists untouched."""
    model, chunks, base_nll = _build_8expert_moe_testbed(trapped_experts={1, 3})
    healthy_experts = (0, 2, 4, 5, 6, 7)

    # Confirm pre-surgery diagnostics: healthy alpha in [2.2, 3.5], trapped alpha < 1.8
    healthy_snapshots: dict[tuple[int, int, str], torch.Tensor] = {}
    for l_idx, blk in enumerate(model.blocks):
        for e_idx in range(8):
            exp = blk.mlp.experts[str(e_idx)]
            dw3 = exp.scaling * (exp.w3_B @ exp.w3_A)
            metrics = fit_esd_power_law_metrics(dw3)
            if e_idx in {1, 3}:
                assert metrics["pl_alpha"] < 1.8
                assert metrics["correlation_trap"] is True
                assert metrics["needs_surgery"] is True
            else:
                assert 2.2 <= metrics["pl_alpha"] <= 3.5
                assert metrics["correlation_trap"] is False
                assert metrics["is_healthy_heavy_tail"] is True
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    healthy_snapshots[(l_idx, e_idx, attr)] = getattr(exp, attr).detach().clone()

    pre_nll = _eval_domain_nlls(model, chunks, score_from=4)
    assert pre_nll["E0_general_fineweb"] > 1.5 * base_nll["E0_general_fineweb"]

    receipt = execute_verified_esd_moe_surgery(
        model,
        chunks,
        baseline_domain_nll=base_nll,
        score_from=4,
    )

    assert receipt["object"] == SURGEON_RECEIPT_OBJECT
    assert receipt["schema"] == SURGEON_SCHEMA
    assert verify_esd_surgery_receipt_provenance(receipt) is True
    assert receipt["status"] == "ACCEPTED"
    assert receipt["accepted"] is True
    assert receipt["committed"] is True
    assert receipt["rolled_back"] is False
    assert receipt["guard_ok"] is True
    assert receipt["general_pct_vs_base"] <= 0.5
    assert receipt["parent_anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert receipt["anse_energy"] < 0.0
    assert receipt["delta_e"] < 0.0
    assert receipt["trapped_units_repaired"] == 4
    assert receipt["healthy_units_untouched"] == 44

    # Healthy specialists remain 100% bitwise untouched; trapped experts are repaired
    for l_idx, blk in enumerate(model.blocks):
        for e_idx in healthy_experts:
            exp = blk.mlp.experts[str(e_idx)]
            for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                assert torch.equal(getattr(exp, attr), healthy_snapshots[(l_idx, e_idx, attr)])
        for e_idx in (1, 3):
            exp = blk.mlp.experts[str(e_idx)]
            post_m = fit_esd_power_law_metrics(exp.scaling * (exp.w3_B @ exp.w3_A))
            assert post_m["pl_alpha"] >= 2.0
            assert post_m["correlation_trap"] is False
            assert post_m["spike_ratio"] <= 2.5 + 1e-6


# ---------------------------------------------------------------------------
# S2: Hill MLE Alpha & Roy-Vetterli Effective-Rank Monotonicity
# ---------------------------------------------------------------------------


def test_s2_hill_mle_alpha_and_effective_rank_monotonicity() -> None:
    """Verify monotonic ordering of Hill alpha, effective rank, and stable rank across ESD regimes."""
    gen = torch.Generator(device="cpu").manual_seed(20262802)
    d_rows, d_cols = 256, 64

    w_gauss = torch.randn((d_rows, d_cols), generator=gen, dtype=torch.float64)
    u_orth, _ = torch.linalg.qr(torch.randn((d_rows, d_cols), generator=gen, dtype=torch.float64))
    v_orth, _ = torch.linalg.qr(torch.randn((d_cols, d_cols), generator=gen, dtype=torch.float64))

    # Power-law exponent alpha ~ 2.25 -> sigma_i ~ i^(-1 / (2 * (2.25 - 1))) = i^(-0.40)
    idx = torch.arange(1, d_cols + 1, dtype=torch.float64)
    w_power_law = u_orth @ torch.diag(idx.pow(-0.40)) @ v_orth.T

    # Rank-collapsed matrix: 2 dominant modes + tiny noise floor
    sv_rc = torch.cat(
        [
            torch.tensor([12.0, 2.0], dtype=torch.float64),
            torch.full((d_cols - 2,), 0.01, dtype=torch.float64),
        ]
    )
    w_rank_collapsed = u_orth @ torch.diag(sv_rc) @ v_orth.T

    m_gauss = fit_esd_power_law_metrics(w_gauss, tail_quantile=0.25)
    m_pl = fit_esd_power_law_metrics(w_power_law, tail_quantile=0.25)
    m_rc = fit_esd_power_law_metrics(w_rank_collapsed, tail_quantile=0.25)

    assert m_gauss["diagnostic_regime"] == "bulk_random_like"
    assert m_pl["diagnostic_regime"] == "healthy_heavy_tail"
    assert m_rc["diagnostic_regime"] == "rank_collapsed"
    assert m_pl["is_healthy_heavy_tail"] is True
    assert 2.0 <= m_pl["pl_alpha"] <= 3.0
    assert m_rc["pl_alpha"] < 1.8
    assert m_rc["rank_collapsed"] is True
    assert m_rc["correlation_trap"] is True

    # Strict monotonicity across Gaussian > Power-Law > Rank-Collapsed
    assert m_gauss["pl_alpha"] > m_pl["pl_alpha"] > m_rc["pl_alpha"]
    assert m_gauss["effective_rank"] > m_pl["effective_rank"] > m_rc["effective_rank"]
    assert m_gauss["stable_rank"] > m_pl["stable_rank"] > m_rc["stable_rank"]

    # Edge spectral regimes: zero, exact rank-1, constant orthogonal, 1D-shaped vector, steep alpha
    m_zero = fit_esd_power_law_metrics(torch.zeros((16, 16), dtype=torch.float32))
    assert m_zero["diagnostic_regime"] == "zero_matrix"
    assert m_zero["active_rank"] == 0
    assert m_zero["correlation_trap"] is True
    assert m_zero["rank_collapsed"] is True

    w_rank1 = 5.0 * torch.outer(u_orth[:16, 0], v_orth[:16, 0])
    m_r1 = fit_esd_power_law_metrics(w_rank1)
    assert m_r1["active_rank"] == 1
    assert m_r1["rank_collapsed"] is True
    assert m_r1["correlation_trap"] is True

    m_orth = fit_esd_power_law_metrics(v_orth)
    assert m_orth["pl_alpha"] == pytest.approx(DEFAULT_MAX_ALPHA_CAP)
    assert m_orth["effective_rank"] == pytest.approx(float(d_cols), rel=1e-5)

    m_vec = fit_esd_power_law_metrics(torch.ones((1, 16), dtype=torch.float64))
    assert m_vec["min_dim"] == 1
    assert m_vec["rank_collapsed"] is False

    # Spectral plateau at top followed by drop triggers full-spectrum cliff fallback
    u_p32, _ = torch.linalg.qr(torch.randn((32, 16), generator=gen, dtype=torch.float64))
    v_p16, _ = torch.linalg.qr(torch.randn((16, 16), generator=gen, dtype=torch.float64))
    w_plateau = (
        u_p32
        @ torch.diag(torch.tensor([10.0, 10.0, 10.0, 10.0] + [0.1] * 12, dtype=torch.float64))
        @ v_p16.T
    )
    m_plat = fit_esd_power_law_metrics(w_plateau, tail_quantile=0.25)
    assert m_plat["pl_alpha"] < 2.0
    assert m_plat["correlation_trap"] is True

    # Winsorization behavior: preserves healthy matrices bitwise & repairs traps with/without bulk energy
    reg_pl, diag_pl = regularize_correlation_trap_spectrum_nullspace(w_power_law)
    assert diag_pl["repaired"] is False
    assert diag_pl["noop_reason"] == "healthy_spectrum_preserved"
    assert torch.equal(reg_pl, w_power_law)

    reg_zero, diag_zero = regularize_correlation_trap_spectrum_nullspace(torch.zeros((8, 8)))
    assert diag_zero["noop_reason"] == "zero_matrix"
    assert torch.equal(reg_zero, torch.zeros((8, 8)))

    reg_rc_preserve, diag_rc_p = regularize_correlation_trap_spectrum_nullspace(
        w_rank_collapsed, target_max_spike_ratio=2.5, preserve_bulk_energy=True
    )
    assert diag_rc_p["repaired"] is True
    assert diag_rc_p["bulk_energy_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert diag_rc_p["spike_ratio_after"] <= 2.5 + 1e-6
    assert diag_rc_p["effective_rank_after"] > diag_rc_p["effective_rank_before"]
    assert diag_rc_p["stable_rank_after"] > diag_rc_p["stable_rank_before"]
    assert reg_rc_preserve.shape == w_rank_collapsed.shape

    reg_rc_scale, diag_rc_s = regularize_correlation_trap_spectrum_nullspace(
        w_rank_collapsed, target_max_spike_ratio=2.5, preserve_bulk_energy=False
    )
    assert diag_rc_s["repaired"] is True
    assert diag_rc_s["bulk_energy_ratio"] < 0.5
    assert reg_rc_scale.norm() < reg_rc_preserve.norm()

    # Rank-1 regularization & steep alpha under permissive spike cap & within_spike_cap noop
    r1_reg, r1_diag = regularize_correlation_trap_spectrum_nullspace(w_rank1, target_max_spike_ratio=4.0)
    assert r1_diag["repaired"] is True
    assert r1_reg.norm().item() == pytest.approx(w_rank1.norm().item() / 2.0, rel=1e-6)

    _, r1_noop = regularize_correlation_trap_spectrum_nullspace(w_rank1, target_max_spike_ratio=1.0)
    assert r1_noop["noop_reason"] == "within_spike_cap"

    u_16, _ = torch.linalg.qr(torch.randn((16, 8), generator=gen, dtype=torch.float64))
    v_8, _ = torch.linalg.qr(torch.randn((8, 8), generator=gen, dtype=torch.float64))
    w_steep = u_16 @ torch.diag(torch.arange(1, 9, dtype=torch.float64).pow(-1.0)) @ v_8.T
    _, steep_diag = regularize_correlation_trap_spectrum_nullspace(w_steep, target_max_spike_ratio=100.0)
    assert steep_diag["repaired"] is True
    assert steep_diag["clipped_modes"] >= 1


# ---------------------------------------------------------------------------
# S3: Optional Prose Null-Space P_0 Projection & Multi-Arch Checkpoints
# ---------------------------------------------------------------------------


def test_s3_nullspace_projection_preserves_protected_keys_and_multi_arch() -> None:
    """Verify P_0 projection limits protected-key drift to <1e-5 and supports Dense/FineGrained/HF MoE."""
    gen = torch.Generator(device="cpu").manual_seed(20262803)
    d_out, d_in = 32, 48
    q_in, _ = torch.linalg.qr(torch.randn((d_in, d_in), generator=gen, dtype=torch.float64))
    u_prot = q_in[:, :6]
    p_null_in = torch.eye(d_in, dtype=torch.float64) - (u_prot @ u_prot.T)

    # Build trapped matrix whose outlier singular mode mixes protected keys and unprotected directions
    q_out, _ = torch.linalg.qr(torch.randn((d_out, d_out), generator=gen, dtype=torch.float64))
    v_trap = 0.6 * u_prot[:, 0] + 0.8 * q_in[:, 10]
    v_trap = v_trap / v_trap.norm()
    sv_bulk = torch.arange(2, 17, dtype=torch.float64).pow(-0.28) * 0.2
    w_bulk = (q_out[:, 1:16] * sv_bulk.unsqueeze(0)) @ q_in[:, 11:26].T
    w_trapped = w_bulk + 12.0 * torch.outer(q_out[:, 0], v_trap)

    w_unproj, _ = regularize_correlation_trap_spectrum_nullspace(w_trapped, p_null=None)
    raw_drift = float(((w_unproj - w_trapped) @ u_prot).abs().max().item())
    assert raw_drift > 1.0

    w_proj, diag_proj = regularize_correlation_trap_spectrum_nullspace(w_trapped, p_null=p_null_in)
    assert diag_proj["repaired"] is True
    assert diag_proj["nullspace_mode"] == "right_input_nullspace"
    prot_drift = float(((w_proj - w_trapped) @ u_prot).norm(dim=0).max().item())
    assert prot_drift < 1e-5

    # Unprotected orthogonal direction IS regularized
    unprot_key = q_in[:, 10]
    assert float(((w_proj - w_trapped) @ unprot_key).norm().item()) > 1.0

    # Left output nullspace projection when p_null matches axis 0
    u_left = q_out[:, :4]
    p_null_left = torch.eye(d_out, dtype=torch.float64) - (u_left @ u_left.T)
    w_left_trap = w_bulk + 12.0 * torch.outer((0.7 * u_left[:, 0] + 0.7 * q_out[:, 5]), q_in[:, 10])
    w_left_proj, diag_left = regularize_correlation_trap_spectrum_nullspace(w_left_trap, p_null=p_null_left)
    assert diag_left["nullspace_mode"] == "left_output_nullspace"
    left_drift = float((u_left.T @ (w_left_proj - w_left_trap)).abs().max().item())
    assert left_drift < 1e-5

    # Multi-architecture support: CrowdFineGrainedMoEMLP + base_state_dict + mapping p_null
    fg_cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        intermediate_size=128,
        max_position_embeddings=32,
        engram_buckets=0,
    )
    fg_dense = CrowdRuntimeMLP(fg_cfg)
    fg_moe = CrowdFineGrainedMoEMLP(fg_dense, num_experts=4, top_k=2)
    base_sd = {k: v.detach().clone() for k, v in fg_moe.state_dict().items()}
    with torch.no_grad():
        # Corrupt expert 1 w3 delta with a rank-1 spike over base
        fg_moe.experts_w3[1].weight[:, 0] += 8.0

    q_fg, _ = torch.linalg.qr(torch.randn((16, 16), generator=gen, dtype=torch.float64))
    p_fg = torch.eye(16, dtype=torch.float64) - torch.outer(q_fg[:, 0], q_fg[:, 0])
    fg_orig_w3 = fg_moe.experts_w3[1].weight.detach().to(torch.float64).clone()

    fg_receipt = execute_verified_esd_moe_surgery(
        fg_moe,
        p_null={"layers.0.mlp.experts.1.w3": p_fg, "layers.0.act": p_fg, "default": p_fg},
        base_state_dict=base_sd,
        protect_general_expert=True,
        target_factors=("w3",),
        eval_fn=lambda m: {
            "E0_general_fineweb": 1.0 + 0.05 * float(m.experts_w3[1].weight.norm().item()),
            "E1_code": 1.0 + 0.02 * float(m.experts_w3[1].weight.norm().item()),
        },
    )
    assert fg_receipt["accepted"] is True
    fg_new_w3 = fg_moe.experts_w3[1].weight.detach().to(torch.float64)
    assert float((q_fg[:, 0] @ (fg_new_w3 - fg_orig_w3)).abs().max().item()) < 1e-5

    # Direct tensor p_null in execute_verified_esd_moe_surgery (matches matching shape, skips non-matching)
    moe_pnull_model, pn_chunks, pn_base = _build_8expert_moe_testbed(trapped_experts={1, 3}, num_layers=1)
    q_pn, _ = torch.linalg.qr(torch.randn((64, 64), generator=gen, dtype=torch.float64))
    p_pn = torch.eye(64, dtype=torch.float64) - torch.outer(q_pn[:, 0], q_pn[:, 0])
    pn_receipt = execute_verified_esd_moe_surgery(
        moe_pnull_model,
        pn_chunks,
        p_null=p_pn,
        baseline_domain_nll=pn_base,
        protect_general_expert=True,
        target_factors=("w3",),
        score_from=4,
    )
    assert pn_receipt["units_scanned"] == 7

    # Dense CrowdRuntimeMLP and HF-style model.layers / experts + standalone nn.Linear
    dense_mlp = CrowdRuntimeMLP(fg_cfg)
    dense_base_sd = {k: v.detach().clone() for k, v in dense_mlp.state_dict().items()}
    with torch.no_grad():
        dense_mlp.w1.weight[:, 0] += 8.0
    dense_res = execute_verified_esd_moe_surgery(
        dense_mlp,
        base_state_dict=dense_base_sd,
        p_null={"unrelated_key": torch.eye(4)},
        eval_fn=lambda m: {
            "per_domain": {
                "E0_general_fineweb": 1.0 + 0.02 * float(m.w1.weight.norm().item()),
                "E1_code": 0.9 + 0.02 * float(m.w1.weight.norm().item()),
            }
        },
    )
    assert dense_res["accepted"] is True

    class _HFExpert(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gate_proj = nn.Linear(16, 16, bias=False)
            self.up_proj = nn.Linear(16, 16, bias=False)
            self.down_proj = nn.Linear(16, 16, bias=False)

    class _HFMoEMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([_HFExpert(), _HFExpert()])

    class _HFLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _HFMoEMLP()

    class _HFModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([_HFLayer()])

    hf_model = _HFModel()
    with torch.no_grad():
        hf_model.layers[0].mlp.experts[1].gate_proj.weight[:, 0] += 8.0
    hf_res = execute_verified_esd_moe_surgery(
        hf_model,
        protect_general_expert=True,
        target_factors=("gate_proj", "up_proj", "down_proj"),
        eval_fn=lambda m: (
            0.0,
            {
                "E0_general_fineweb": 1.0
                + 0.03 * float(m.layers[0].mlp.experts[1].gate_proj.weight.norm().item())
            },
        ),
    )
    assert hf_res["accepted"] is True

    lin = nn.Linear(16, 16, bias=False)
    with torch.no_grad():
        lin.weight[:, 0] += 8.0
    lin_res = execute_verified_esd_moe_surgery(
        lin,
        eval_fn=lambda m: {"E0_general_fineweb": 1.0 + 0.03 * float(m.weight.norm().item())},
    )
    assert lin_res["accepted"] is True


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback & Fixed-Point Preservation
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_rollback_and_fixed_point() -> None:
    """Verify atomic bitwise rollback on guard violation, non-improving energy, eval error, and fixed point."""
    model, chunks, base_nll = _build_8expert_moe_testbed(trapped_experts=set(), num_layers=1)
    parent_sha = compute_state_sha256(model.state_dict())
    parent_sd_copy = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # 4A. Healthy model with zero traps -> FIXED_POINT_UNCHANGED
    fp_receipt = execute_verified_esd_moe_surgery(
        model, chunks, baseline_domain_nll=base_nll, score_from=4
    )
    assert fp_receipt["status"] == "FIXED_POINT_UNCHANGED"
    assert fp_receipt["accepted"] is False
    assert fp_receipt["committed"] is False
    assert fp_receipt["rolled_back"] is False
    assert fp_receipt["final_state_sha256"] == parent_sha

    # 4B. Forced adverse clipping violating general-domain guard -> ANSE Maximum Pain atomic rollback
    def _adverse_eval(m: nn.Module) -> dict[str, float]:
        if compute_state_sha256(m.state_dict()) == parent_sha:
            return {"E0_general_fineweb": 1.0, "E1_python_systems_code": 0.9}
        return {"E0_general_fineweb": 1.04, "E1_python_systems_code": 0.7}

    pain_receipt = execute_verified_esd_moe_surgery(
        model,
        eval_fn=_adverse_eval,
        only_regularize_traps=False,
        target_max_spike_ratio=0.10,
        preserve_bulk_energy=False,
    )
    assert pain_receipt["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert pain_receipt["accepted"] is False
    assert pain_receipt["committed"] is False
    assert pain_receipt["rolled_back"] is True
    assert pain_receipt["guard_ok"] is False
    assert pain_receipt["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert pain_receipt["delta_e"] > 0.0
    assert pain_receipt["final_state_sha256"] == parent_sha
    for k, orig_t in parent_sd_copy.items():
        assert torch.equal(model.state_dict()[k], orig_t)

    # 4C. Guard-feasible yet energy-non-improving edit -> ROLLED_BACK_ENERGY_NON_IMPROVING
    def _non_improving_eval(m: nn.Module) -> dict[str, float]:
        if compute_state_sha256(m.state_dict()) == parent_sha:
            return {"E0_general_fineweb": 1.0, "E1_python_systems_code": 0.8}
        return {"E0_general_fineweb": 1.002, "E1_python_systems_code": 0.95}

    nonimp_receipt = execute_verified_esd_moe_surgery(
        model,
        eval_fn=_non_improving_eval,
        only_regularize_traps=False,
        target_max_spike_ratio=0.10,
    )
    assert nonimp_receipt["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert nonimp_receipt["rolled_back"] is True
    assert nonimp_receipt["guard_ok"] is True
    assert nonimp_receipt["delta_e"] > 0.0
    assert nonimp_receipt["final_state_sha256"] == parent_sha

    # 4D. Eval exception during candidate scoring -> ROLLED_BACK_EVAL_FAILURE + atomic rollback
    calls = {"n": 0}

    def _raising_eval(_m: nn.Module) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] == 1:
            return {"E0_general_fineweb": 1.0, "E1_python_systems_code": 0.8}
        raise RuntimeError("simulated evaluation failure")

    err_receipt = execute_verified_esd_moe_surgery(
        model,
        eval_fn=_raising_eval,
        only_regularize_traps=False,
        target_max_spike_ratio=0.10,
    )
    assert err_receipt["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert err_receipt["rolled_back"] is True
    assert "RuntimeError" in str(err_receipt["eval_error"])
    assert err_receipt["anse_energy"] == pytest.approx(ANSE_MAXIMUM_PAIN)
    assert err_receipt["final_state_sha256"] == parent_sha

    # 4E. inplace=False leaves caller model untouched even on ACCEPTED surgery
    trapped_model, t_chunks, t_base = _build_8expert_moe_testbed(trapped_experts={1, 3}, num_layers=1)
    trapped_sha = compute_state_sha256(trapped_model.state_dict())
    non_inplace_receipt = execute_verified_esd_moe_surgery(
        trapped_model,
        t_chunks,
        baseline_domain_nll=t_base,
        score_from=4,
        inplace=False,
    )
    assert non_inplace_receipt["accepted"] is True
    assert compute_state_sha256(trapped_model.state_dict()) == trapped_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors & Repository Security / Zero-Stub Compliance Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance() -> None:
    """Validate strict error handling and pass AGPL-3.0, stub, lateral-movement, and PR gates."""
    # 1. Config validation
    with pytest.raises(ValueError, match="alpha_low"):
        HeavyTailESDSurgeonConfig(alpha_low=4.0, alpha_high=2.0)
    with pytest.raises(ValueError, match="target_factors"):
        HeavyTailESDSurgeonConfig(target_factors=())
    with pytest.raises(ValueError, match="Unsupported target_factors"):
        HeavyTailESDSurgeonConfig(target_factors=("bad_proj",))
    with pytest.raises(ValueError, match="general_prefix"):
        HeavyTailESDSurgeonConfig(general_prefix="")
    with pytest.raises(TypeError, match="bool"):
        HeavyTailESDSurgeonConfig(preserve_bulk_energy="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        HeavyTailESDSurgeonConfig(score_from=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        HeavyTailESDSurgeonConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 1"):
        HeavyTailESDSurgeonConfig(batch_size=0)
    with pytest.raises(TypeError, match="real number"):
        HeavyTailESDSurgeonConfig(tail_quantile=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        HeavyTailESDSurgeonConfig(target_max_spike_ratio=float("nan"))
    with pytest.raises(ValueError, match="> 0.0"):
        HeavyTailESDSurgeonConfig(target_max_spike_ratio=0.0)
    with pytest.raises(ValueError, match=">= 0.0"):
        HeavyTailESDSurgeonConfig(soft_margin_penalty=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        HeavyTailESDSurgeonConfig(tail_quantile=0.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        HeavyTailESDSurgeonConfig(tail_quantile=1.5)
    with pytest.raises(ValueError, match="< 1.0"):
        HeavyTailESDSurgeonConfig(rank_collapse_ratio_threshold=1.0)

    # 2. fit_esd_power_law_metrics & regularize_correlation_trap_spectrum_nullspace validation
    with pytest.raises(TypeError, match="torch.Tensor"):
        fit_esd_power_law_metrics([[1.0, 2.0]])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty 2D"):
        fit_esd_power_law_metrics(torch.ones((4,)))
    with pytest.raises(TypeError, match="floating-point"):
        fit_esd_power_law_metrics(torch.ones((4, 4), dtype=torch.int64))
    with pytest.raises(ValueError, match="only finite"):
        fit_esd_power_law_metrics(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))
    with pytest.raises(ValueError, match="alpha_low"):
        fit_esd_power_law_metrics(torch.eye(4), alpha_low=5.0, alpha_high=3.0)
    with pytest.raises(ValueError, match="square 2D"):
        regularize_correlation_trap_spectrum_nullspace(torch.eye(4), p_null=torch.ones((4, 3)))
    with pytest.raises(ValueError, match="incompatible"):
        regularize_correlation_trap_spectrum_nullspace(torch.eye(4), p_null=torch.eye(6))

    # 3. Provenance & hashing validation
    with pytest.raises(ValueError, match="non-empty"):
        compute_state_sha256({})
    with pytest.raises(ValueError, match="non-empty string"):
        compute_state_sha256({"": torch.ones(2)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_state_sha256({"a": [1.0]})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_state_sha256({"a": torch.tensor([float("inf")])})
    with pytest.raises(TypeError, match="mapping"):
        verify_esd_surgery_receipt_provenance("invalid")  # type: ignore[arg-type]
    assert verify_esd_surgery_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_esd_surgery_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_esd_surgery_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": SURGEON_RECEIPT_OBJECT,
                "schema": SURGEON_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )
    valid_min = {"object": SURGEON_RECEIPT_OBJECT, "schema": SURGEON_SCHEMA, "pl_alpha": 2.5}
    valid_min["receipt_sha256"] = compute_payload_sha256(valid_min)
    assert verify_esd_surgery_receipt_provenance(valid_min) is True

    # 4. execute_verified_esd_moe_surgery validation & selective targeting options
    model, chunks, base_nll = _build_8expert_moe_testbed(trapped_experts={1, 3}, num_layers=1)
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_esd_moe_surgery("bad", chunks)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_verified_esd_moe_surgery(model, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Either eval_chunks or eval_fn"):
        execute_verified_esd_moe_surgery(model)
    with pytest.raises(TypeError, match="HeavyTailESDSurgeonConfig"):
        execute_verified_esd_moe_surgery(model, chunks, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_factors"):
        execute_verified_esd_moe_surgery(model, chunks, target_factors=("bad",))
    with pytest.raises(ValueError, match="layer_indices"):
        execute_verified_esd_moe_surgery(model, chunks, layer_indices=[99])
    with pytest.raises(TypeError, match="CrowdV1WithEngram"):
        execute_verified_esd_moe_surgery(nn.Linear(4, 4), chunks)
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_esd_moe_surgery(model, [])
    with pytest.raises(ValueError, match="must be a mapping"):
        execute_verified_esd_moe_surgery(model, [{"ids": [0, 1, 2, 3, 4, 5]}])
    with pytest.raises(ValueError, match="identical sequence lengths"):
        execute_verified_esd_moe_surgery(
            model,
            [
                {"expert_id": "E0_general_fineweb", "ids": [0, 1, 2, 3, 4, 5]},
                {"expert_id": "E1_code", "ids": [8, 9, 10]},
            ],
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_esd_moe_surgery(
            model, [{"expert_id": "E0_general_fineweb", "ids": [0, 1]}], score_from=4
        )
    with pytest.raises(ValueError, match="general chunk"):
        execute_verified_esd_moe_surgery(
            model, [{"expert_id": "E1_code", "ids": [8, 9, 10, 11, 12, 13]}], score_from=2
        )
    with pytest.raises(ValueError, match="Non-finite or non-positive"):
        execute_verified_esd_moe_surgery(model, eval_fn=lambda _m: {"E0_general_fineweb": -1.0})
    with pytest.raises(TypeError, match="mapping of domain"):
        execute_verified_esd_moe_surgery(model, eval_fn=lambda _m: [1.0])  # type: ignore[return-value]
    with pytest.raises(ValueError, match="non-empty"):
        execute_verified_esd_moe_surgery(model, eval_fn=lambda _m: {})
    class _UnsupportedMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = nn.Parameter(torch.ones(2))

    with pytest.raises(ValueError, match="Unsupported MLP"):
        execute_verified_esd_moe_surgery(_UnsupportedMLP(), eval_fn=lambda _m: {"E0_general_fineweb": 1.0})
    with pytest.raises(TypeError, match="p_null"):
        execute_verified_esd_moe_surgery(
            model, chunks, p_null="bad"  # type: ignore[arg-type]
        )

    bad_blocks_mod = nn.Module()
    bad_blocks_mod.blocks = nn.ModuleList()
    with pytest.raises(ValueError, match="blocks"):
        execute_verified_esd_moe_surgery(bad_blocks_mod, eval_fn=lambda _m: {"E0_general_fineweb": 1.0})
    bad_blocks_mod.blocks = nn.ModuleList([nn.Module()])
    with pytest.raises(ValueError, match="mlp"):
        execute_verified_esd_moe_surgery(bad_blocks_mod, eval_fn=lambda _m: {"E0_general_fineweb": 1.0})

    bad_hf_mod = nn.Module()
    bad_hf_mod.layers = nn.ModuleList()
    with pytest.raises(ValueError, match="layers"):
        execute_verified_esd_moe_surgery(bad_hf_mod, eval_fn=lambda _m: {"E0_general_fineweb": 1.0})
    bad_hf_mod.layers = nn.ModuleList([nn.Module()])
    with pytest.raises(ValueError, match="mlp"):
        execute_verified_esd_moe_surgery(bad_hf_mod, eval_fn=lambda _m: {"E0_general_fineweb": 1.0})

    with pytest.raises(ValueError, match="identical keys"):
        execute_verified_esd_moe_surgery(
            model, eval_fn=lambda _m: {"E0_general_fineweb": 1.0}, baseline_domain_nll={"E1_code": 1.0}
        )
    with pytest.raises(ValueError, match="general-domain key"):
        execute_verified_esd_moe_surgery(model, eval_fn=lambda _m: {"E1_code": 1.0})

    # Zero-adapter skip + layer_indices skipping layer 1 on 2-layer model
    m_2layer, chunks_2l, base_2l = _build_8expert_moe_testbed(trapped_experts={1}, num_layers=2)
    with torch.no_grad():
        m_2layer.blocks[0].mlp.experts["2"].w1_B.zero_()
    sel_receipt = execute_verified_esd_moe_surgery(
        m_2layer,
        chunks_2l,
        baseline_domain_nll=base_2l,
        layer_indices=[0],
        expert_indices=[1, 2],
        enforce_all_domain_guard=True,
        score_from=4,
    )
    assert sel_receipt["units_scanned"] == 6
    assert sel_receipt["trapped_units_repaired"] == 1
    assert any(r.get("noop_reason") == "zero_adapter" for r in sel_receipt["unit_reports"])

    # 5. Repository AGPL-3.0, optional ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn28", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn28", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/heavy_tail_esd_weight_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_heavy_tail_esd_weight_surgeon_scenarios.py", TEST_FILE_PATH),
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

    telemetry_json = json.dumps(
        {
            "nll_mean": sel_receipt["nll_mean"],
            "pl_alpha": sel_receipt["pl_alpha"],
            "stable_rank": sel_receipt["stable_rank"],
            "anse_energy": sel_receipt["anse_energy"],
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1128,
        pr_title="feat(swarm): activation-free heavy-tail ESD power-law alpha and effective-rank weight surgeon for open-weight MoEs",
        pr_body="Implements NEURAL-CTRL-DYN-28 activation-free heavy-tail ESD weight surgeon and deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): activation-free heavy-tail ESD power-law alpha and effective-rank weight surgeon for open-weight MoEs"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
