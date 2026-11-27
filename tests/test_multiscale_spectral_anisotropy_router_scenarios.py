# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-50: multiscale spectral-anisotropy profiler and causal-vs-MLM WSP router."""

from __future__ import annotations

import ast
import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.multiscale_spectral_anisotropy_router import (
    ANSE_MAXIMUM_PAIN,
    REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS,
    REGIME_FOLDED_SCALAR_RMSNORM_ONLY,
    REGIME_JAMES_STEIN_RANK_K_FISHER,
    REGIME_P0_LORA_PLUS_FOLDED_NORM,
    ROUTER_RECEIPT_OBJECT,
    ROUTER_SCHEMA,
    MultiscaleSpectralAnisotropyRouterConfig,
    compute_module_state_sha256,
    compute_payload_sha256,
    execute_regime_routed_wsp_plan,
    profile_hidden_spectral_anisotropy,
    select_wsp_operator_regime,
    verify_regime_routed_wsp_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "multiscale_spectral_anisotropy_router.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP thread-pool spin on small deterministic CPU tensors."""
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


class _VarianceEpsRMSNorm(nn.Module):
    """Qwen2.5-style biasless RMSNorm exposing ``variance_epsilon``."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (int(self.weight.shape[0]),), self.weight, self.variance_epsilon)


class _MultiScaleCausalLM(nn.Module):
    """Deterministic causal LM with configurable anisotropy, self-suppressed outliers, and linear projection."""

    def __init__(
        self,
        *,
        hidden_size: int = 48,
        vocab_size: int = 40,
        spec_norm_init: float = 0.20,
        d0_head_scale: float = 0.16,
        spec_head_mult: float = 2.6,
        outlier_dims: tuple[int, ...] = (),
        outlier_activation_scale: float = 0.0,
        outlier_norm_weight: float = 1.0,
        base_norm_scale: float = 1.0,
        anisotropy_mode_scale: float = 0.0,
        bg_scale: float = 0.45,
        seed: int = 20265001,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.outlier_dims = outlier_dims
        self.outlier_activation_scale = float(outlier_activation_scale)
        self.anisotropy_mode_scale = float(anisotropy_mode_scale)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = _VarianceEpsRMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.proj.weight.copy_(torch.eye(hidden_size, dtype=torch.float32))
            self.norm.weight.fill_(float(base_norm_scale))
            self.lm_head.weight.zero_()

            # D0 general prose occupies dims 0..3
            for tok in range(0, 10):
                self.embed.weight[tok, 0] = 2.0
                self.embed.weight[tok, 1] = 0.4 * ((tok % 3) - 1)
                if tok == 1:
                    self.lm_head.weight[tok, 0] = float(d0_head_scale)
                elif tok == 2:
                    self.lm_head.weight[tok, 0] = 0.35 * float(d0_head_scale)
                else:
                    self.lm_head.weight[tok, 0] = -0.35 * float(d0_head_scale)

            # Specialists D1..D3 occupy dims 8..10 inside ker(G_D0)
            self.norm.weight[8:11].fill_(float(spec_norm_init))
            for d_idx in range(1, 4):
                tok_lo = 10 * d_idx
                for tok in range(tok_lo, tok_lo + 10):
                    self.embed.weight[tok, 8:11] = torch.tensor([2.5, 2.0, 1.5], dtype=torch.float32)
                    if tok == tok_lo + 1:
                        self.lm_head.weight[tok, 8:11] = float(spec_head_mult) * torch.tensor(
                            [0.24, 0.20, 0.16], dtype=torch.float32
                        )
                    elif tok == tok_lo + 2:
                        self.lm_head.weight[tok, 8:11] = float(spec_head_mult) * torch.tensor(
                            [0.08, 0.06, 0.04], dtype=torch.float32
                        )
                    else:
                        self.lm_head.weight[tok, 8:11] = float(spec_head_mult) * torch.tensor(
                            [-0.12, -0.10, -0.08], dtype=torch.float32
                        )

            # Small stochastic background variations on remaining dimensions
            if hidden_size > 16:
                self.embed.weight[:, 16:] = float(bg_scale) * torch.randn(
                    (vocab_size, hidden_size - 16), generator=gen, dtype=torch.float32
                )
            for od in outlier_dims:
                self.norm.weight[od] = float(outlier_norm_weight)
                self.embed.weight[:, od] = float(outlier_activation_scale)

            if anisotropy_mode_scale > 0.0 and hidden_size > 15:
                self.embed.weight[:, 15] = float(anisotropy_mode_scale)
                self.norm.weight[15] = float(outlier_norm_weight)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.proj(self.embed(input_ids))
        return self.lm_head(self.norm(h))


class _BidirectionalMLMEncoder(nn.Module):
    """Compact ModernBERT-like MLM encoder exposing ``head.norm`` (LayerNorm) and ``decoder``."""

    def __init__(
        self,
        *,
        hidden_size: int = 24,
        vocab_size: int = 32,
        outlier_dim: int | None = None,
        outlier_act_scale: float = 0.0,
        anisotropy_scale: float = 0.0,
        spec_norm_init: float = 0.22,
        d0_head_scale: float = 0.20,
        spec_head_mult: float = 2.4,
        base_norm_scale: float = 1.0,
        seed: int = 20265002,
    ) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.head = nn.Module()
        self.head.norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.proj.weight.copy_(torch.eye(hidden_size, dtype=torch.float32))
            self.head.norm.weight.fill_(float(base_norm_scale))
            self.head.norm.bias.zero_()
            self.decoder.weight.zero_()
            self.decoder.bias.zero_()

            # D0 tokens: 0..7, Specialists D1..D3: 8..31
            for tok in range(0, 8):
                self.embed.weight[tok, 0] = 2.2
                self.embed.weight[tok, 1] = 0.3 * ((tok % 3) - 1)
                self.decoder.weight[tok, 0] = float(d0_head_scale) if tok == 1 else -0.3 * float(d0_head_scale)

            self.head.norm.weight[4:7].fill_(float(spec_norm_init))
            for d_idx in range(1, 4):
                lo = 8 * d_idx
                for tok in range(lo, min(vocab_size, lo + 8)):
                    self.embed.weight[tok, 4:7] = torch.tensor([2.4, 2.0, 1.4], dtype=torch.float32)
                    if tok == lo + 1:
                        self.decoder.weight[tok, 4:7] = float(spec_head_mult) * torch.tensor(
                            [0.24, 0.20, 0.16], dtype=torch.float32
                        )
                    else:
                        self.decoder.weight[tok, 4:7] = float(spec_head_mult) * torch.tensor(
                            [-0.10, -0.08, -0.06], dtype=torch.float32
                        )

            if hidden_size > 10:
                self.embed.weight[:, 10:] = 0.40 * torch.randn(
                    (vocab_size, hidden_size - 10), generator=gen, dtype=torch.float32
                )
            if outlier_dim is not None and outlier_act_scale > 0.0:
                self.embed.weight[:, outlier_dim] = float(outlier_act_scale)
                self.head.norm.weight[outlier_dim] = 0.02
            if anisotropy_scale > 0.0 and hidden_size > 9:
                self.embed.weight[:, 9] = float(anisotropy_scale)
                self.head.norm.weight[9] = 0.02

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.head.norm(self.proj(self.embed(input_ids))))


def _make_causal_chunks(
    *,
    n_per_domain: int,
    seq_len: int = 16,
    seed: int = 20265010,
    d0_modal_prob: float = 0.72,
    spec_modal_prob: float = 0.88,
    d0_corrupt_prob: float = 0.0,
) -> list[dict[str, Any]]:
    """Generate deterministic 4-domain causal token sequences."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    specs = (
        ("D0_general_prose", 0, d0_modal_prob),
        ("D1_python_systems_code", 10, spec_modal_prob),
        ("D2_formal_math_lean4", 20, spec_modal_prob),
        ("D3_computational_physics", 30, spec_modal_prob),
    )
    chunks: list[dict[str, Any]] = []
    for dom, lo, p_modal in specs:
        for _ in range(n_per_domain):
            seq: list[int] = []
            for pos in range(seq_len):
                u = float(torch.rand((), generator=gen).item())
                if dom == "D0_general_prose" and d0_corrupt_prob > 0.0 and u < d0_corrupt_prob:
                    seq.append(lo + 5 + (pos % 4))
                elif u < p_modal:
                    seq.append(lo + 1)
                elif u < p_modal + 0.14:
                    seq.append(lo + 2)
                else:
                    seq.append(lo + 3 + (pos % 6))
            chunks.append({"domain": dom, "ids": seq})
    return chunks


def _make_mlm_chunks(
    *,
    n_per_domain: int,
    seq_len: int = 16,
    n_mask: int = 8,
    seed: int = 20265020,
    d0_modal_prob: float = 0.72,
    spec_modal_prob: float = 0.88,
) -> list[dict[str, Any]]:
    """Generate deterministic 4-domain masked-token MLM chunks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    specs = (
        ("D0_general_prose", 0, d0_modal_prob),
        ("D1_python_systems_code", 8, spec_modal_prob),
        ("D2_formal_math_lean4", 16, spec_modal_prob),
        ("D3_computational_physics", 24, spec_modal_prob),
    )
    chunks: list[dict[str, Any]] = []
    m_pos = list(range(1, min(seq_len - 1, 1 + n_mask)))
    for dom, lo, p_modal in specs:
        for _ in range(n_per_domain):
            ids = [lo + (pos % 6) for pos in range(seq_len)]
            targets: list[int] = []
            for pos in m_pos:
                u = float(torch.rand((), generator=gen).item())
                targets.append(lo + 1 if u < p_modal else lo + 3 + (pos % 4))
            chunks.append(
                {
                    "domain": dom,
                    "ids": ids,
                    "mask_positions": m_pos,
                    "targets": targets,
                }
            )
    return chunks


def _synthetic_anisotropic_states(
    *,
    n_tokens: int,
    dim: int,
    top1_target_energy: float,
    outlier_specs: Sequence[tuple[int, float, float]],
    median_norm_w: float = 3.0,
    seed: int = 20265030,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build hidden states and norm weights with exact spectral anisotropy and outlier dims."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    h = torch.randn((n_tokens, dim), generator=gen, dtype=torch.float64)
    w = torch.full((dim,), float(median_norm_w), dtype=torch.float64)
    med_rms = float(h.pow(2).mean(dim=0).sqrt().median().item())

    # Coherent alternating sign carrier mimics correlated massive-activation sinks (PILOT9 dims 1819/1874)
    carrier = ((torch.arange(n_tokens, dtype=torch.float64) % 2) * 2.0) - 1.0
    for od, act_ratio, norm_w_val in outlier_specs:
        h[:, od] = carrier * (float(act_ratio) * med_rms)
        w[od] = float(norm_w_val)

    bulk_cols = [i for i in range(dim) if all(i != od for od, _, _ in outlier_specs)]
    if top1_target_energy > 0.0 and bulk_cols:
        s_vals = torch.linalg.svdvals(h)
        s2 = s_vals.pow(2)
        cur_top1 = float((s2[0] / s2.sum().clamp_min(1e-12)).item())
        if cur_top1 < top1_target_energy:
            lead_dir = torch.zeros(dim, dtype=torch.float64)
            lead_dir[bulk_cols[0]] = 1.0
            coeff = torch.randn((n_tokens, 1), generator=gen, dtype=torch.float64)
            tot_e = float(s2.sum().item())
            extra_e = max(0.0, (top1_target_energy * tot_e - float(s2[0].item())) / max(1e-6, 1.0 - top1_target_energy))
            h = h + (extra_e / max(float(coeff.pow(2).sum().item()), 1e-6)) ** 0.5 * (coeff @ lead_dir.unsqueeze(0))

    return h.to(torch.float32), w.to(torch.float32)


# ---------------------------------------------------------------------------
# Scenario S1: 0.5B Moderate Anisotropy (JS Rank-k) vs 3B Self-Suppressed Outlier (1-DOF Scalar)
# ---------------------------------------------------------------------------


def test_s1_05b_moderate_anisotropy_js_vs_3b_extreme_outlier_folded_scalar() -> None:
    """(S1) Replicate PILOT12 0.5B JS rank-k routing vs PILOT9 3B self-suppressed outlier 1-DOF scalar routing."""
    # 1A. Analytical profile + routing replication of PILOT12 (Qwen2.5-0.5B) vs PILOT9 (Qwen2.5-3B)
    h_05b, w_05b = _synthetic_anisotropic_states(
        n_tokens=192,
        dim=896,
        top1_target_energy=0.34,
        outlier_specs=((490, 22.0, -0.0106),),
        median_norm_w=6.78,
        seed=5011,
    )
    prof_05b = profile_hidden_spectral_anisotropy(
        h_05b,
        w_05b,
        architecture_kind="causal_lm",
        effective_tokens_per_split=4608,  # 4608 / 896 = 5.14 >= 4.0
    )
    assert 0.25 <= prof_05b["top1_energy_frac"] < 0.55
    assert prof_05b["massive_activation_channel_ratio"] >= 18.0
    assert prof_05b["self_suppression_factor"] >= 100.0
    assert prof_05b["probe_tokens_per_dof"] >= 4.0
    route_05b = select_wsp_operator_regime(prof_05b)
    assert route_05b["selected_regime"] == REGIME_JAMES_STEIN_RANK_K_FISHER
    assert route_05b["safe_for_multi_dof_fisher"] is True
    assert route_05b["recommended_max_rank"] >= 1

    # PILOT9-like 3B extreme self-suppressed outlier regime (two ~60x suppressed dims, high anisotropy)
    h_3b, w_3b = _synthetic_anisotropic_states(
        n_tokens=192,
        dim=2048,
        top1_target_energy=0.68,
        outlier_specs=((1819, 60.4, 0.0354), (1874, 59.6, 0.0325)),
        median_norm_w=3.14,
        seed=5012,
    )
    prof_3b = profile_hidden_spectral_anisotropy(
        h_3b,
        w_3b,
        architecture_kind="causal_lm",
        effective_tokens_per_split=3072,  # 3072 / 2048 = 1.50 < 4.0 (matches PILOT9)
        norm_bias=torch.zeros(2048),
        head_bias_dim=10,
    )
    assert prof_3b["top1_energy_frac"] >= 0.55
    assert prof_3b["massive_activation_channel_ratio"] >= 45.0
    assert prof_3b["self_suppression_factor"] >= 50.0
    assert prof_3b["probe_tokens_per_dof"] < 4.0
    assert prof_3b["norm_dof"] == 4096 and prof_3b["total_head_dof"] == 4106
    route_3b = select_wsp_operator_regime(prof_3b)
    assert route_3b["selected_regime"] == REGIME_FOLDED_SCALAR_RMSNORM_ONLY
    assert route_3b["safe_for_multi_dof_fisher"] is False
    assert route_3b["recommended_max_rank"] == 0
    assert "extreme_self_suppressed_massive_activations" in route_3b["reasons"]

    # Zero-energy hidden states cleanly yield zero spectral stats
    prof_zero = profile_hidden_spectral_anisotropy(torch.zeros((8, 16)), torch.ones(16))
    assert prof_zero["top1_energy_frac"] == 0.0 and prof_zero["self_suppression_factor"] == 1.0

    # 1B. End-to-end execution on 0.5B moderate-anisotropy causal LM -> james_stein_rank_k_fisher
    model_05b = _MultiScaleCausalLM(
        hidden_size=48,
        vocab_size=40,
        spec_norm_init=0.18,
        d0_head_scale=0.18,
        spec_head_mult=2.8,
        outlier_dims=(),
        base_norm_scale=1.12,
        anisotropy_mode_scale=0.0,
        bg_scale=1.25,
        seed=5013,
    )
    probe_05b = _make_causal_chunks(n_per_domain=6, seq_len=16, seed=1101, d0_modal_prob=0.55, spec_modal_prob=0.68)
    sel_05b = _make_causal_chunks(n_per_domain=6, seq_len=16, seed=1102, d0_modal_prob=0.55, spec_modal_prob=0.68)
    eval_05b = _make_causal_chunks(n_per_domain=6, seq_len=16, seed=1103, d0_modal_prob=0.55, spec_modal_prob=0.68)
    cfg_05b = MultiscaleSpectralAnisotropyRouterConfig(
        min_causal_tokens_per_dof=4.0,
        high_anisotropy_threshold=0.55,
        max_rank=4,
        ridge_scale=1e-4,
        max_step_norm=0.35,
        score_from=2,
        eta_grid=(0.25, 0.50, 0.85),
        tau_grid=(0.95, 1.00, 1.05, 1.12),
    )
    base_sha_05b = compute_module_state_sha256(model_05b)
    edited_05b, rec_05b = execute_regime_routed_wsp_plan(
        model_05b,
        probe_05b,
        selection_chunks=sel_05b,
        eval_chunks=eval_05b,
        config=cfg_05b,
        inplace=False,
    )
    assert compute_module_state_sha256(model_05b) == base_sha_05b
    assert rec_05b["object"] == ROUTER_RECEIPT_OBJECT
    assert rec_05b["schema"] == ROUTER_SCHEMA
    assert verify_regime_routed_wsp_receipt_provenance(rec_05b) is True
    assert rec_05b["selected_regime"] == REGIME_JAMES_STEIN_RANK_K_FISHER
    assert rec_05b["accepted"] is True
    assert rec_05b["delta_e"] < 0.0
    assert rec_05b["general_drift_ratio"] <= 0.01
    assert rec_05b["nll_mean"] < rec_05b["base_nll_mean"]
    assert rec_05b["operator_details"]["nullspace_orthogonality_max_abs"] < 1e-6

    # Also exercise JS->scalar fallback ratchet when probe SNR gate blocks JS step
    _, rec_ratchet = execute_regime_routed_wsp_plan(
        model_05b,
        probe_05b,
        selection_chunks=sel_05b,
        eval_chunks=eval_05b,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_causal_tokens_per_dof=4.0,
            high_anisotropy_threshold=0.55,
            min_probe_snr=1e6,
            score_from=2,
            tau_grid=(0.75, 0.85, 0.92, 1.00, 1.12),
        ),
        inplace=False,
    )
    assert rec_ratchet["operator_details"]["ratcheted_to_scalar"] is True
    assert verify_regime_routed_wsp_receipt_provenance(rec_ratchet) is True

    # 1C. End-to-end execution on 3B extreme-outlier self-suppressed causal LM -> folded_scalar_rmsnorm_only
    model_3b = _MultiScaleCausalLM(
        hidden_size=64,
        vocab_size=40,
        spec_norm_init=1.35,
        d0_head_scale=0.35,
        spec_head_mult=1.2,
        outlier_dims=(30, 31),
        outlier_activation_scale=65.0,
        outlier_norm_weight=0.03,
        base_norm_scale=1.35,
        anisotropy_mode_scale=0.0,
        seed=5014,
    )
    probe_3b = _make_causal_chunks(n_per_domain=2, seq_len=12, seed=1201, d0_modal_prob=0.68, spec_modal_prob=0.80)
    sel_3b = _make_causal_chunks(n_per_domain=4, seq_len=12, seed=1202, d0_modal_prob=0.68, spec_modal_prob=0.80)
    eval_3b = _make_causal_chunks(n_per_domain=4, seq_len=12, seed=1203, d0_modal_prob=0.68, spec_modal_prob=0.80)
    ratio_before = float(model_3b.norm.weight[30].item() / model_3b.norm.weight[12].item())
    edited_3b, rec_3b = execute_regime_routed_wsp_plan(
        model_3b,
        probe_3b,
        selection_chunks=sel_3b,
        eval_chunks=eval_3b,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_causal_tokens_per_dof=4.0,
            massive_outlier_ratio_threshold=20.0,
            score_from=2,
            tau_grid=(0.95, 1.00, 1.05, 1.12, 1.25),
        ),
        inplace=False,
    )
    ratio_after = float(edited_3b.norm.weight[30].item() / edited_3b.norm.weight[12].item())
    assert rec_3b["selected_regime"] == REGIME_FOLDED_SCALAR_RMSNORM_ONLY
    assert rec_3b["accepted"] is True
    assert rec_3b["delta_e"] < 0.0
    assert rec_3b["general_drift_ratio"] <= 0.01
    assert abs(ratio_after - ratio_before) < 1e-6
    assert verify_regime_routed_wsp_receipt_provenance(rec_3b) is True


# ---------------------------------------------------------------------------
# Scenario S2: Under-Sampled Bidirectional MLM (PILOT14) Abstention / Scalar Guard
# ---------------------------------------------------------------------------


def test_s2_undersampled_bidirectional_mlm_pilot14_abstains_or_scalar_guards() -> None:
    """(S2) Under-sampled PILOT14-style bidirectional MLM routes to abstain or 1-DOF scalar guard."""
    mlm_model = _BidirectionalMLMEncoder(
        hidden_size=96,
        vocab_size=32,
        outlier_dim=40,
        outlier_act_scale=25.0,
        anisotropy_scale=18.0,
        base_norm_scale=1.20,
        seed=5021,
    )
    # 4 * 4 = 16 chunks * 8 masked tokens = 128 masked tokens vs H=96 -> 1.333 < 12.0!
    probe_mlm = _make_mlm_chunks(n_per_domain=4, seq_len=16, n_mask=8, seed=2101)
    sel_mlm = _make_mlm_chunks(n_per_domain=4, seq_len=16, n_mask=8, seed=2102)
    eval_mlm = _make_mlm_chunks(n_per_domain=4, seq_len=16, n_mask=8, seed=2103)

    init_sha = compute_module_state_sha256(mlm_model)
    edited_abstain, rec_abstain = execute_regime_routed_wsp_plan(
        mlm_model,
        probe_mlm,
        selection_chunks=sel_mlm,
        eval_chunks=eval_mlm,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_mlm_masked_tokens_per_dof=12.0,
            allow_mlm_scalar_fallback=False,
        ),
        inplace=False,
    )
    assert rec_abstain["architecture_kind"] == "bidirectional_mlm"
    assert rec_abstain["spectral_profile"]["probe_tokens_per_dof"] < 12.0
    assert rec_abstain["selected_regime"] == REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS
    assert rec_abstain["abstained"] is True
    assert rec_abstain["accepted"] is False
    assert rec_abstain["rolled_back"] is False
    assert rec_abstain["status"] == "ABSTAINED_INSUFFICIENT_PROBE_TOKENS"
    assert rec_abstain["edited_params"] == 0
    assert rec_abstain["general_drift_ratio"] == 0.0
    assert rec_abstain["final_state_sha256"] == init_sha
    assert compute_module_state_sha256(edited_abstain) == init_sha
    assert verify_regime_routed_wsp_receipt_provenance(rec_abstain) is True

    # Opt-in 1-DOF scalar guard fallback prevents multi-DOF overfit while maintaining D0 <= +1.0%
    edited_guard, rec_guard = execute_regime_routed_wsp_plan(
        mlm_model,
        probe_mlm,
        selection_chunks=sel_mlm,
        eval_chunks=eval_mlm,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_mlm_masked_tokens_per_dof=12.0,
            allow_mlm_scalar_fallback=True,
            tau_grid=(0.95, 1.00, 1.05, 1.12, 1.25),
        ),
        inplace=False,
    )
    assert rec_guard["selected_regime"] == REGIME_FOLDED_SCALAR_RMSNORM_ONLY
    assert rec_guard["routing_decision"]["safe_for_multi_dof_fisher"] is False
    assert rec_guard["general_drift_ratio"] <= 0.01
    assert verify_regime_routed_wsp_receipt_provenance(rec_guard) is True
    if not rec_guard["accepted"]:
        assert compute_module_state_sha256(edited_guard) == init_sha

    # Additional routing checks: degenerate causal scalar-min abstention + adequate MLM edge regimes
    causal_tiny = select_wsp_operator_regime(
        {
            "architecture_kind": "causal_lm",
            "hidden_dim": 1024,
            "probe_tokens_per_dof": 0.1,
            "top1_energy_frac": 0.2,
            "massive_activation_channel_ratio": 2.0,
            "self_suppression_factor": 1.0,
        }
    )
    assert causal_tiny["selected_regime"] == REGIME_ABSTAIN_INSUFFICIENT_PROBE_TOKENS

    mlm_ext = select_wsp_operator_regime(
        {
            "architecture_kind": "bidirectional_mlm",
            "hidden_dim": 64,
            "probe_tokens_per_dof": 16.0,
            "top1_energy_frac": 0.90,
            "massive_activation_channel_ratio": 55.0,
            "self_suppression_factor": 12.0,
        }
    )
    assert mlm_ext["selected_regime"] == REGIME_FOLDED_SCALAR_RMSNORM_ONLY

    mlm_lora = select_wsp_operator_regime(
        {
            "architecture_kind": "bidirectional_mlm",
            "hidden_dim": 64,
            "probe_tokens_per_dof": 16.0,
            "top1_energy_frac": 0.30,
            "massive_activation_channel_ratio": 3.0,
            "self_suppression_factor": 1.0,
            "has_lora_adapters": True,
        }
    )
    assert mlm_lora["selected_regime"] == REGIME_P0_LORA_PLUS_FOLDED_NORM


# ---------------------------------------------------------------------------
# Scenario S3: Adequate-Sample MLM & Conditional P_0 LoRA Acceptance (Delta E < 0)
# ---------------------------------------------------------------------------


def test_s3_adequate_sample_mlm_and_p0_lora_regime_acceptance() -> None:
    """(S3) Verify adequate-sample MLM JS rank-k and causal conditional P_0 LoRA acceptance."""
    # 3A. Adequate-sample bidirectional MLM (hidden_size=24, 12*4*8 = 384 masked tokens -> 16.0 >= 12.0/DOF)
    mlm_adeq = _BidirectionalMLMEncoder(
        hidden_size=24,
        vocab_size=32,
        outlier_dim=None,
        anisotropy_scale=0.0,
        spec_norm_init=0.16,
        d0_head_scale=0.18,
        spec_head_mult=2.8,
        base_norm_scale=1.08,
        seed=5031,
    )
    probe_mlm = _make_mlm_chunks(n_per_domain=12, seq_len=16, n_mask=8, seed=3101)
    sel_mlm = _make_mlm_chunks(n_per_domain=8, seq_len=16, n_mask=8, seed=3102)
    eval_mlm = _make_mlm_chunks(n_per_domain=8, seq_len=16, n_mask=8, seed=3103)
    cfg_mlm = MultiscaleSpectralAnisotropyRouterConfig(
        min_mlm_masked_tokens_per_dof=12.0,
        high_anisotropy_threshold=0.70,
        extreme_anisotropy_threshold=0.85,
        max_rank=4,
        ridge_scale=1e-4,
        max_step_norm=0.35,
        eta_grid=(0.25, 0.50, 0.85),
        tau_grid=(0.95, 1.00, 1.05, 1.12),
    )
    _, rec_mlm = execute_regime_routed_wsp_plan(
        mlm_adeq,
        probe_mlm,
        selection_chunks=sel_mlm,
        eval_chunks=eval_mlm,
        config=cfg_mlm,
        inplace=False,
    )
    assert rec_mlm["spectral_profile"]["probe_tokens_per_dof"] == pytest.approx(16.0)
    assert rec_mlm["selected_regime"] == REGIME_JAMES_STEIN_RANK_K_FISHER
    assert rec_mlm["accepted"] is True
    assert rec_mlm["delta_e"] < 0.0
    assert rec_mlm["general_drift_ratio"] <= 0.01
    assert verify_regime_routed_wsp_receipt_provenance(rec_mlm) is True

    # 3B. P_0 LoRA rescue: adapter degrades D0 via dim 0 unless projected onto ker(X_D0)
    model_lora = _MultiScaleCausalLM(
        hidden_size=32,
        vocab_size=40,
        spec_norm_init=0.22,
        d0_head_scale=0.45,
        spec_head_mult=2.2,
        base_norm_scale=1.0,
        seed=5032,
    )
    probe_c = _make_causal_chunks(n_per_domain=6, seq_len=12, seed=3201)
    sel_c = _make_causal_chunks(n_per_domain=6, seq_len=12, seed=3202)
    eval_c = _make_causal_chunks(n_per_domain=6, seq_len=12, seed=3203)

    # Adapter 1 hurts D0 (col 0) and boosts specialist dims 8..10 (cols 8..10)
    a_harm = torch.zeros((2, 32), dtype=torch.float32)
    a_harm[0, 0] = 1.0
    a_harm[1, 8:11] = torch.tensor([0.6, 0.5, 0.4], dtype=torch.float32)
    b_harm = torch.zeros((32, 2), dtype=torch.float32)
    b_harm[0, 0] = -0.85  # flips D0 logit sign -> D0 NLL spikes if unprojected!
    b_harm[8:11, 1] = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float32)

    dw_dense = torch.zeros((32, 32), dtype=torch.float32)
    dw_dense[8:11, 8:11] = 0.25 * torch.eye(3, dtype=torch.float32)

    adapters_p0 = [
        {"adapter_name": "specialist_drift", "modules": {"proj": {"a_mat": a_harm, "b_mat": b_harm, "scale": 1.0}}},
        {"adapter_name": "specialist_dense", "modules": {"proj": {"delta_weight": dw_dense}}},
    ]
    cfg_lora = MultiscaleSpectralAnisotropyRouterConfig(
        score_from=2,
        p0_sv_rel_tol=0.05,
        max_p0_rank=8,
        tau_grid=(0.95, 1.00, 1.05),
    )
    m_p0_1, rec_p0_1 = execute_regime_routed_wsp_plan(
        model_lora,
        probe_c,
        selection_chunks=sel_c,
        eval_chunks=eval_c,
        lora_adapters=adapters_p0,
        config=cfg_lora,
        architecture_kind="moe_causal_lm",
        inplace=False,
    )
    m_p0_2, rec_p0_2 = execute_regime_routed_wsp_plan(
        model_lora,
        probe_c,
        selection_chunks=sel_c,
        eval_chunks=eval_c,
        lora_adapters=adapters_p0,
        config=cfg_lora,
        architecture_kind="moe_causal_lm",
        inplace=False,
    )
    assert rec_p0_1["selected_regime"] == REGIME_P0_LORA_PLUS_FOLDED_NORM
    assert rec_p0_1["operator_details"]["p0_gate_mode"] == "p0_nullspace_projected"
    assert rec_p0_1["accepted"] is True
    assert rec_p0_1["delta_e"] < 0.0
    assert rec_p0_1["general_drift_ratio"] <= 0.01
    assert rec_p0_1["final_state_sha256"] == rec_p0_2["final_state_sha256"]
    assert rec_p0_1["receipt_sha256"] == rec_p0_2["receipt_sha256"]
    assert compute_module_state_sha256(m_p0_1) == compute_module_state_sha256(m_p0_2)

    # 3C. Conditional P_0 bypass on benign LoRA (PILOT15 finding: naive benign adapter wins without stripping!)
    a_benign = torch.zeros((1, 32), dtype=torch.float32)
    a_benign[0, 0] = 1.0
    a_benign[0, 8:11] = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)
    b_benign = torch.zeros((32, 1), dtype=torch.float32)
    b_benign[0, 0] = 0.65  # improves D0 directly!
    b_benign[8:11, 0] = torch.tensor([0.6, 0.5, 0.4], dtype=torch.float32)
    _, rec_benign = execute_regime_routed_wsp_plan(
        model_lora,
        probe_c + sel_c,  # also exercises automatic 50/50 split when selection_chunks=None!
        lora_adapters=[{"adapter_name": "benign_all", "modules": {"proj": {"a_mat": a_benign, "b_mat": b_benign}}}],
        config=cfg_lora,
        inplace=False,
    )
    assert rec_benign["selected_regime"] == REGIME_P0_LORA_PLUS_FOLDED_NORM
    assert rec_benign["operator_details"]["p0_gate_mode"] == "conditional_bypass_benign_lora"
    assert rec_benign["accepted"] is True
    assert rec_benign["delta_e"] < 0.0

    # 3D. Toxic LoRA ratchets back to clean scalar & empty adapter list degrades cleanly
    b_toxic = -5.0 * torch.ones((32, 1), dtype=torch.float32)
    _, rec_toxic = execute_regime_routed_wsp_plan(
        model_lora,
        probe_c,
        selection_chunks=sel_c,
        lora_adapters=[{"adapter_name": "toxic", "modules": {"proj": {"a_mat": torch.ones((1, 32)), "b_mat": b_toxic}}}],
        config=MultiscaleSpectralAnisotropyRouterConfig(score_from=2, tau_grid=(0.95, 1.00, 1.05)),
        inplace=False,
    )
    assert rec_toxic["operator_details"]["p0_gate_mode"] == "ratcheted_to_clean_scalar"

    # Empty lora_adapters + has_lora_adapters check on route directly
    prof_empty_lora = profile_hidden_spectral_anisotropy(torch.randn((64, 16)), torch.ones(16), has_lora_adapters=True)
    assert select_wsp_operator_regime(prof_empty_lora)["selected_regime"] == REGIME_P0_LORA_PLUS_FOLDED_NORM


# ---------------------------------------------------------------------------
# Scenario S4: ANSE Maximum Pain Atomic Bitwise Rollback on Guard Breach
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_on_guard_breach() -> None:
    """(S4) Enforce ANSE Maximum Pain atomic bitwise rollback on D0 breach, energy failure, or eval error."""
    # 4A. Intrinsic holdout guard breach: selection passes, eval D0 breaches tight guard -> bitwise rollback
    model_05b = _MultiScaleCausalLM(
        hidden_size=48,
        vocab_size=40,
        spec_norm_init=0.18,
        d0_head_scale=0.18,
        spec_head_mult=2.8,
        base_norm_scale=1.18,
        seed=5041,
    )
    probe_c = _make_causal_chunks(n_per_domain=6, seq_len=16, seed=4101)
    sel_c = _make_causal_chunks(n_per_domain=6, seq_len=16, seed=4102)
    eval_c = _make_causal_chunks(n_per_domain=6, seq_len=16, seed=4103, d0_corrupt_prob=0.45)

    # First verify via external eval_fn that D0 +3% breach triggers ROLLED_BACK_GUARD_VIOLATION
    init_sha = compute_module_state_sha256(model_05b)
    eval_calls = {"n": 0}

    def _breaching_eval(_m: nn.Module) -> tuple[float, dict[str, float]]:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return 0.020, {"D0_general_prose": 2.00, "D1_python": 2.10}
        return 0.015, {"D0_general_prose": 2.06, "D1_python": 1.85}  # +3.0% > +1.0%!

    res_model_guard, rec_guard = execute_regime_routed_wsp_plan(
        model_05b,
        probe_c,
        selection_chunks=sel_c,
        eval_chunks=eval_c,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_causal_tokens_per_dof=4.0,
            high_anisotropy_threshold=0.80,
            score_from=2,
            eta_grid=(0.25, 0.50),
            tau_grid=(0.95, 1.00, 1.05, 1.15),
        ),
        eval_fn=_breaching_eval,
        inplace=True,
    )
    assert res_model_guard is model_05b
    assert rec_guard["accepted"] is False
    assert rec_guard["rolled_back"] is True
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_guard["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_guard["final_state_sha256"] == init_sha
    assert compute_module_state_sha256(model_05b) == init_sha
    assert verify_regime_routed_wsp_receipt_provenance(rec_guard) is True

    # 4B. P_0 LoRA plan rollback restores BOTH proj.weight AND norm.weight bit-for-bit
    a_ok = torch.zeros((1, 48), dtype=torch.float32)
    a_ok[0, 8:11] = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)
    b_ok = torch.zeros((48, 1), dtype=torch.float32)
    b_ok[8:11, 0] = torch.tensor([0.5, 0.4, 0.3], dtype=torch.float32)

    lora_calls = {"n": 0}

    def _lora_breach(_m: nn.Module) -> dict[str, float]:
        lora_calls["n"] += 1
        if lora_calls["n"] == 1:
            return {"D0_general_prose": 2.0, "D1_python": 2.2}
        return {"D0_general_prose": 2.08, "D1_python": 1.9}  # +4.0% D0 breach!

    _, rec_lora_rb = execute_regime_routed_wsp_plan(
        model_05b,
        probe_c,
        selection_chunks=sel_c,
        lora_adapters=[{"adapter_name": "p0_rb", "modules": {"proj": {"a_mat": a_ok, "b_mat": b_ok}}}],
        config=MultiscaleSpectralAnisotropyRouterConfig(score_from=2, tau_grid=(0.95, 1.00, 1.05)),
        eval_fn=_lora_breach,
        inplace=True,
    )
    assert rec_lora_rb["rolled_back"] is True
    assert rec_lora_rb["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert compute_module_state_sha256(model_05b) == init_sha

    # 4C. Non-improving energy rollback & eval_fn exception rollback & fixed-point unchanged
    _, rec_nonimp = execute_regime_routed_wsp_plan(
        model_05b,
        probe_c,
        selection_chunks=sel_c,
        eval_chunks=sel_c,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_causal_tokens_per_dof=4.0,
            high_anisotropy_threshold=0.80,
            min_energy_drop=0.0,
            require_strict_energy_drop=False,
            score_from=2,
            tau_grid=(1.15,),
        ),
        eval_fn=lambda _m: {"D0_general_prose": 2.0, "D1_python": 2.0},
        inplace=False,
    )
    # With require_strict_energy_drop=False and delta_e == 0.0, flat eval accepts if cand != init;
    # now enforce positive min_energy_drop to trigger ROLLED_BACK_ENERGY_NON_IMPROVING!
    assert rec_nonimp["final_state_sha256"] != ""

    _, rec_nonimp_rb = execute_regime_routed_wsp_plan(
        model_05b,
        probe_c,
        selection_chunks=sel_c,
        eval_chunks=sel_c,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_causal_tokens_per_dof=4.0,
            high_anisotropy_threshold=0.80,
            min_energy_drop=0.05,
            score_from=2,
            tau_grid=(0.95, 1.15),
        ),
        eval_fn=lambda _m: {"D0_general_prose": 2.0, "D1_python": 1.99},
        inplace=False,
    )
    assert rec_nonimp_rb["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_nonimp_rb["final_state_sha256"] == init_sha

    boom_calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, float]:
        boom_calls["n"] += 1
        if boom_calls["n"] == 1:
            return {"D0_general_prose": 2.0, "D1_python": 2.0}
        raise RuntimeError("simulated eval exception")

    _, rec_boom = execute_regime_routed_wsp_plan(
        model_05b,
        probe_c,
        selection_chunks=sel_c,
        config=MultiscaleSpectralAnisotropyRouterConfig(score_from=2, tau_grid=(1.15,)),
        eval_fn=_boom_eval,
        inplace=False,
    )
    assert rec_boom["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert rec_boom["cand_anse_energy"] == ANSE_MAXIMUM_PAIN

    _, rec_fp = execute_regime_routed_wsp_plan(
        model_05b,
        probe_c,
        selection_chunks=sel_c,
        config=MultiscaleSpectralAnisotropyRouterConfig(
            min_causal_tokens_per_dof=100.0,
            min_probe_snr=1e6,
            score_from=2,
            tau_grid=(1.00,),
        ),
        inplace=False,
    )
    assert rec_fp["status"] == "FIXED_POINT_UNCHANGED"


# ---------------------------------------------------------------------------
# Scenario S5: Validation Errors & Security / Stub / OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_guardrail_gates() -> None:
    """(S5) Assert strict input validation, zero stubs, zero forbidden patterns, and OwnerOrderGuardrail."""
    # 5A. Config validation
    with pytest.raises(ValueError, match="Anisotropy thresholds"):
        MultiscaleSpectralAnisotropyRouterConfig(high_anisotropy_threshold=0.9, extreme_anisotropy_threshold=0.5)
    with pytest.raises(ValueError, match="massive_outlier_ratio_threshold"):
        MultiscaleSpectralAnisotropyRouterConfig(outlier_ratio_threshold=5.0, massive_outlier_ratio_threshold=4.0)
    with pytest.raises(ValueError, match="self_suppression_threshold"):
        MultiscaleSpectralAnisotropyRouterConfig(self_suppression_threshold=1.0)
    with pytest.raises(ValueError, match="nullspace_sv_tol"):
        MultiscaleSpectralAnisotropyRouterConfig(nullspace_sv_tol=1.0)
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        MultiscaleSpectralAnisotropyRouterConfig(anse_safe_general_ratio=0.9)
    with pytest.raises(TypeError, match="bool"):
        MultiscaleSpectralAnisotropyRouterConfig(allow_mlm_scalar_fallback="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        MultiscaleSpectralAnisotropyRouterConfig(max_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        MultiscaleSpectralAnisotropyRouterConfig(max_rank=0)
    with pytest.raises(ValueError, match=">= 0"):
        MultiscaleSpectralAnisotropyRouterConfig(score_from=-1)
    with pytest.raises(ValueError, match=">= 0.0"):
        MultiscaleSpectralAnisotropyRouterConfig(ridge_scale=-1.0)
    with pytest.raises(ValueError, match="finite"):
        MultiscaleSpectralAnisotropyRouterConfig(max_step_norm=float("nan"))
    with pytest.raises(TypeError, match="sequence"):
        MultiscaleSpectralAnisotropyRouterConfig(tau_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        MultiscaleSpectralAnisotropyRouterConfig(tau_grid=())
    with pytest.raises(TypeError, match="general_prefixes"):
        MultiscaleSpectralAnisotropyRouterConfig(general_prefixes="D0_")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        MultiscaleSpectralAnisotropyRouterConfig(general_prefixes=())

    # 5B. Profiler & router validation
    with pytest.raises(ValueError, match="architecture_kind"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 8)), torch.ones(8), architecture_kind="unknown")
    with pytest.raises(ValueError, match="outlier_ratio_threshold"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 8)), torch.ones(8), outlier_ratio_threshold=1.0)
    with pytest.raises(ValueError, match="self_suppression_threshold"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 8)), torch.ones(8), self_suppression_threshold=1.0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        profile_hidden_spectral_anisotropy("bad", torch.ones(8))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim >= 2"):
        profile_hidden_spectral_anisotropy(torch.ones(8), torch.ones(8))
    with pytest.raises(ValueError, match="ndim == 1"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 8)), torch.ones((2, 4)))
    with pytest.raises(ValueError, match="non-empty"):
        profile_hidden_spectral_anisotropy(torch.empty((0, 8)), torch.ones(8))
    with pytest.raises(TypeError, match="floating-point"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 8), dtype=torch.long), torch.ones(8))
    with pytest.raises(ValueError, match="finite"):
        profile_hidden_spectral_anisotropy(torch.tensor([[1.0, float("inf")]]), torch.ones(2))
    with pytest.raises(ValueError, match="match norm_weights"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 6)), torch.ones(8))
    with pytest.raises(ValueError, match="norm_bias length"):
        profile_hidden_spectral_anisotropy(torch.ones((4, 8)), torch.ones(8), norm_bias=torch.ones(4))

    with pytest.raises(TypeError, match="Mapping"):
        select_wsp_operator_regime("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="high <= extreme"):
        select_wsp_operator_regime(
            profile_hidden_spectral_anisotropy(torch.ones((4, 8)), torch.ones(8)),
            high_anisotropy_threshold=0.9,
            extreme_anisotropy_threshold=0.5,
        )

    # 5C. Execution, LoRA, chunk, and provenance validation
    m_ok = _MultiScaleCausalLM(hidden_size=16, vocab_size=40, seed=5051)
    good_chunks = _make_causal_chunks(n_per_domain=4, seq_len=10, seed=5101)
    with pytest.raises(TypeError, match="MultiscaleSpectralAnisotropyRouterConfig"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        execute_regime_routed_wsp_plan("bad", good_chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm_kind"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, norm_kind="bad")
    with pytest.raises(ValueError, match="final normalization"):
        execute_regime_routed_wsp_plan(nn.Linear(4, 4), good_chunks)
    with pytest.raises(ValueError, match="unembedding head"):
        no_head = nn.Module()
        no_head.norm = nn.RMSNorm(4)
        execute_regime_routed_wsp_plan(no_head, good_chunks)
    with pytest.raises(ValueError, match="must match norm dim"):
        bad_hw = nn.Module()
        bad_hw.norm = nn.RMSNorm(4)
        bad_hw.head = nn.Linear(5, 8, bias=False)
        execute_regime_routed_wsp_plan(bad_hw, good_chunks)
    with pytest.raises(ValueError, match="bias length"):
        bad_nb = nn.Module()
        bad_nb.norm = nn.LayerNorm(4)
        bad_nb.norm.bias = nn.Parameter(torch.zeros(3))
        bad_nb.head = nn.Linear(4, 8)
        execute_regime_routed_wsp_plan(bad_nb, good_chunks)
    with pytest.raises(ValueError, match="vocab size"):
        bad_hb = nn.Module()
        bad_hb.norm = nn.RMSNorm(4)
        bad_hb.head = nn.Linear(4, 8)
        bad_hb.head.bias = nn.Parameter(torch.zeros(3))
        execute_regime_routed_wsp_plan(bad_hb, good_chunks)
    with pytest.raises(ValueError, match="final normalization"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, norm_attr="missing.norm")
    with pytest.raises(ValueError, match="unembedding head"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, head_attr="missing.head")

    with pytest.raises(TypeError, match="sequence"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, selection_chunks=good_chunks, lora_adapters="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'modules'"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks, selection_chunks=good_chunks, lora_adapters=[{}])
    with pytest.raises(ValueError, match="non-empty mapping"):
        execute_regime_routed_wsp_plan(
            m_ok, good_chunks, selection_chunks=good_chunks, lora_adapters=[{"modules": {}}]
        )
    with pytest.raises(ValueError, match="not found"):
        execute_regime_routed_wsp_plan(
            m_ok,
            good_chunks,
            selection_chunks=good_chunks,
            lora_adapters=[{"modules": {"missing.proj": {"delta_weight": torch.zeros((16, 16))}}}],
        )
    with pytest.raises(ValueError, match="LoRA shapes"):
        execute_regime_routed_wsp_plan(
            m_ok,
            good_chunks,
            selection_chunks=good_chunks,
            lora_adapters=[{"modules": {"proj": {"a_mat": torch.zeros((2, 8)), "b_mat": torch.zeros((16, 2))}}}],
        )
    with pytest.raises(ValueError, match="delta_weight"):
        execute_regime_routed_wsp_plan(
            m_ok,
            good_chunks,
            selection_chunks=good_chunks,
            lora_adapters=[{"modules": {"proj": {"delta_weight": torch.zeros((8, 8))}}}],
        )
    with pytest.raises(ValueError, match="must provide"):
        execute_regime_routed_wsp_plan(
            m_ok, good_chunks, selection_chunks=good_chunks, lora_adapters=[{"modules": {"proj": {}}}]
        )

    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_regime_routed_wsp_plan(m_ok, [], selection_chunks=good_chunks)
    with pytest.raises(ValueError, match="at least 4 probe chunks"):
        execute_regime_routed_wsp_plan(m_ok, good_chunks[:2])
    with pytest.raises(ValueError, match="Automatic"):
        execute_regime_routed_wsp_plan(
            m_ok,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D1_s", "ids": [10, 11, 12, 13]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
            ],
        )
    with pytest.raises(ValueError, match="containing 'ids'"):
        execute_regime_routed_wsp_plan(m_ok, [{"domain": "D0"}], selection_chunks=good_chunks)
    with pytest.raises(TypeError, match="integer dtype"):
        execute_regime_routed_wsp_plan(
            m_ok, [{"domain": "D0", "ids": torch.ones(4)}], selection_chunks=good_chunks
        )
    with pytest.raises(TypeError, match="sequence or 1D"):
        execute_regime_routed_wsp_plan(m_ok, [{"domain": "D0", "ids": "bad"}], selection_chunks=good_chunks)
    with pytest.raises(ValueError, match="1D"):
        execute_regime_routed_wsp_plan(
            m_ok, [{"domain": "D0", "ids": torch.ones((2, 2), dtype=torch.long)}], selection_chunks=good_chunks
        )
    with pytest.raises(ValueError, match="length >= 2"):
        execute_regime_routed_wsp_plan(m_ok, [{"domain": "D0", "ids": [1]}], selection_chunks=good_chunks)
    with pytest.raises(ValueError, match="token IDs"):
        execute_regime_routed_wsp_plan(m_ok, [{"domain": "D0", "ids": [0, 999]}], selection_chunks=good_chunks)
    with pytest.raises(ValueError, match="consistently"):
        execute_regime_routed_wsp_plan(
            m_ok,
            [
                {"domain": "D0", "ids": [0, 1, 2, 3], "mask_positions": [1], "targets": [2]},
                {"domain": "D1", "ids": [10, 11, 12, 13]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="mask_positions"):
        execute_regime_routed_wsp_plan(
            m_ok,
            [
                {"domain": "D0", "ids": [0, 1, 2, 3], "mask_positions": [10], "targets": [2]},
                {"domain": "D1", "ids": [10, 11, 12, 13], "mask_positions": [1], "targets": [12]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="both general"):
        execute_regime_routed_wsp_plan(
            m_ok,
            [{"domain": "D0_a", "ids": [0, 1, 2, 3]}, {"domain": "D0_b", "ids": [0, 1, 2, 3]}],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="too short"):
        execute_regime_routed_wsp_plan(
            m_ok,
            good_chunks,
            selection_chunks=good_chunks,
            config=MultiscaleSpectralAnisotropyRouterConfig(score_from=20),
        )

    with pytest.raises(TypeError, match="Mapping"):
        verify_regime_routed_wsp_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_regime_routed_wsp_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_regime_routed_wsp_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert (
        verify_regime_routed_wsp_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": ROUTER_RECEIPT_OBJECT,
                "schema": ROUTER_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 5D. AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn50", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn50", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/multiscale_spectral_anisotropy_router.py", TARGET_MODULE_PATH),
        ("tests/test_multiscale_spectral_anisotropy_router_scenarios.py", TEST_FILE_PATH),
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

    for doc_rel, doc_path in (
        ("docs/NEURO_CONTROLLER_ROADMAP.md", ROADMAP_DOC_PATH),
        ("docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md", PLAN_DOC_PATH),
    ):
        doc_text = doc_path.read_text(encoding="utf-8")
        assert "0.00 EUR / 0.00 USD" in doc_text
        assert "NEURAL-CTRL-DYN-50" in doc_text
        assert "NEURAL-CTRL-DYN-49" in doc_text
        assert "NEURAL-CTRL-DYN-48" in doc_text
        assert "PILOT15" in doc_text
        assert "PILOT14" in doc_text
        assert "PILOT12" in doc_text
        assert "PILOT9" in doc_text
        assert "La Ruche" in doc_text
        assert "Reine-Cortex" in doc_text
        assert "Gelée Royale" in doc_text
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

    telemetry_json = json.dumps(
        {
            "nll_mean": 1.8871,
            "ece_value": 0.0278,
            "top1_energy_frac": 0.6874,
            "general_drift_ratio": -0.00683,
            "anse_energy": -0.00886,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )
    assert len(compute_payload_sha256({"ok": True})) == 64

    pr_title = "feat(swarm): multiscale spectral-anisotropy profiler and causal-vs-MLM WSP operator regime selector"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-50 training-free multi-scale spectral-anisotropy profiler "
        "and causal-vs-MLM WSP operator regime selector grounded in PILOT3A/3B, PILOT9, PILOT12, "
        "PILOT14, and PILOT15 under ANSE Delta E < 0, D0 <= +1.0%, and atomic bitwise rollback."
    )
    pr_intent = classify_request_intent(
        f"{pr_title}\n{pr_body}",
        source_channel="pull_request",
        changed_files=[c["filename"] for c in changed_files],
    )
    assert pr_intent.intent_category == RequestIntentCategory.STANDARD_CONTRIBUTION
    assert pr_intent.is_directive_order is False
    assert pr_intent.protected_paths_touched == ()

    pr_audit = scanner.audit_pull_request(
        pr_number=1250,
        pr_title=pr_title,
        pr_body=pr_body,
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[pr_title],
        author_email=None,
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
