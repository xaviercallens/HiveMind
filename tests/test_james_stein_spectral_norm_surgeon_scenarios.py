# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-48: Empirical-Bayes James-Stein rank-k spectral-shrinkage Fisher-Norm surgeon."""

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
import torch.nn.functional as F
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.james_stein_spectral_norm_surgeon import (
    ANSE_MAXIMUM_PAIN,
    SURGEON_RECEIPT_OBJECT,
    SURGEON_SCHEMA,
    JamesSteinNormSurgeonConfig,
    compute_effective_probe_dof_and_overfit_risk,
    compute_module_state_sha256,
    compute_payload_sha256,
    execute_verified_james_stein_norm_surgery,
    fit_james_stein_rank_k_fisher_norm_step,
    verify_james_stein_norm_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "james_stein_spectral_norm_surgeon.py"
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


class _SmallProbeHighDOFCausalLM(nn.Module):
    """Deterministic causal LM with low-rank specialist signal + high-DOF nuisance dimensions."""

    def __init__(
        self,
        *,
        hidden_size: int = 128,
        vocab_size: int = 40,
        nuisance_head_scale: float = 0.8,
        spec_norm_init: float = 0.25,
        d0_confidence_scale: float = 0.16,
        nuisance_norm_init: float = 0.0,
        sig_head_mult: float = 1.0,
        tok_emb_scale: float = 0.0,
        block_parity: bool = False,
        seed: int = 20264801,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.block_parity = block_parity
        gen = torch.Generator(device="cpu").manual_seed(seed)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm = _VarianceEpsRMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        nuis_dim = max(0, hidden_size - 16)
        if nuis_dim > 0 and not block_parity:
            bank = 0.55 * torch.randn((256, nuis_dim), generator=gen, dtype=torch.float32)
            # Zero-center bank across context keys so population expectation is strictly zero
            bank = bank - bank.mean(dim=0, keepdim=True)
        else:
            bank = torch.empty((256, 0), dtype=torch.float32)
        self.register_buffer("nuisance_bank", bank)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.lm_head.weight.zero_()
            self.norm.weight.fill_(1.0)
            if hidden_size > 16:
                self.norm.weight[16:].fill_(float(nuisance_norm_init))

            # D0 prose uses dim 0 and stays calibrated near tau=1.0 when d0_confidence_scale~0.16
            for tok in range(0, 10):
                self.embed.weight[tok, 0] = 2.0
                if tok == 1:
                    self.lm_head.weight[tok, 0] = float(d0_confidence_scale)
                elif tok == 2:
                    self.lm_head.weight[tok, 0] = 0.35 * float(d0_confidence_scale)
                else:
                    self.lm_head.weight[tok, 0] = -0.35 * float(d0_confidence_scale)

            # Specialists D1, D2, D3 share calibration axes 8..10 inside ker(G_D0)
            # while dims 16..hidden_size carry high-rank zero-mean nuisance features
            self.norm.weight[8:11].fill_(float(spec_norm_init))
            for d_idx in range(1, 4):
                tok_lo = 10 * d_idx
                for tok in range(tok_lo, tok_lo + 10):
                    self.embed.weight[tok, 8:11] = torch.tensor([2.5, 2.0, 1.5], dtype=torch.float32)
                    if tok == tok_lo + 1:
                        self.lm_head.weight[tok, 8:11] = float(sig_head_mult) * torch.tensor(
                            [0.24, 0.20, 0.16], dtype=torch.float32
                        )
                    elif tok == tok_lo + 2:
                        self.lm_head.weight[tok, 8:11] = float(sig_head_mult) * torch.tensor(
                            [0.08, 0.06, 0.04], dtype=torch.float32
                        )
                    else:
                        self.lm_head.weight[tok, 8:11] = float(sig_head_mult) * torch.tensor(
                            [-0.12, -0.10, -0.08], dtype=torch.float32
                        )
                    if nuisance_head_scale > 0.0 and hidden_size > 16:
                        hw = nuisance_head_scale * torch.randn(
                            (hidden_size - 16,), generator=gen, dtype=torch.float32
                        )
                        self.lm_head.weight[tok, 16:] = hw
            if tok_emb_scale > 0.0 and hidden_size > 16:
                gen_tok = torch.Generator(device="cpu").manual_seed(seed + 98)
                self.embed.weight[10:, 16:] = tok_emb_scale * torch.randn(
                    (vocab_size - 10, hidden_size - 16), generator=gen_tok, dtype=torch.float32
                )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        if self.block_parity and h.shape[2] > 16:
            prefix = (input_ids[:, :1] + input_ids[:, 1:2]) % 2
            pos_par = (torch.arange(h.shape[1], device=h.device) // 4) % 2
            par = (((pos_par.unsqueeze(0) + prefix) % 2) * 2 - 1).to(dtype=h.dtype).unsqueeze(-1)
            h = torch.cat([h[..., :16], h[..., 16:] * par], dim=-1)
        elif self.nuisance_bank.shape[1] > 0 and h.shape[2] > 16 and h.shape[1] >= 2:
            key = (input_ids[:, 0] * 17 + input_ids[:, 1] * 31) % 256
            ctx = self.nuisance_bank[key].unsqueeze(1).expand(-1, h.shape[1], -1)
            is_spec = (input_ids[:, :1] >= 10).to(dtype=h.dtype).unsqueeze(-1)
            h = torch.cat([h[..., :16], h[..., 16:] + ctx * is_spec], dim=-1)
        return self.lm_head(self.norm(h))


class _BidirectionalMLMEncoder(nn.Module):
    """Compact ModernBERT-like MLM encoder exposing ``head.norm`` (LayerNorm) and ``decoder``."""

    def __init__(self, *, hidden_size: int = 64, vocab_size: int = 32, seed: int = 20264802) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed)
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.head = nn.Module()
        self.head.norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=True)

        with torch.no_grad():
            self.embed.weight.copy_(0.4 * torch.randn((vocab_size, hidden_size), generator=gen))
            self.head.norm.weight.fill_(1.0)
            self.head.norm.bias.zero_()
            self.decoder.weight.copy_(0.35 * torch.randn((vocab_size, hidden_size), generator=gen))
            self.decoder.bias.copy_(0.05 * torch.randn((vocab_size,), generator=gen))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        return self.decoder(self.head.norm(h))


def _make_causal_chunks(
    *,
    n_per_domain: int,
    seq_len: int = 16,
    seed: int = 20264810,
    d0_modal_prob: float = 0.78,
    spec_modal_prob: float = 0.85,
) -> list[dict[str, Any]]:
    """Generate deterministic 4-domain next-token chunks."""
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
                if pos <= 1:
                    r_ctx = int(torch.randint(0, 10, (), generator=gen).item())
                    seq.append(lo + r_ctx)
                elif u < p_modal:
                    seq.append(lo + 1)
                elif u < p_modal + 0.12:
                    seq.append(lo + 2)
                else:
                    seq.append(lo + 3 + (pos % 6))
            chunks.append({"domain": dom, "ids": seq})
    return chunks


def _make_mlm_chunks(
    *,
    n_per_domain: int,
    seq_len: int = 16,
    n_mask: int = 4,
    vocab_size: int = 32,
    seed: int = 20264820,
) -> list[dict[str, Any]]:
    """Generate deterministic masked-token MLM chunks with noisy masked targets."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    domains = (
        "D0_general_prose",
        "D1_python_systems_code",
        "D2_formal_math_lean4",
        "D3_computational_physics",
    )
    chunks: list[dict[str, Any]] = []
    for dom in domains:
        for _ in range(n_per_domain):
            ids = torch.randint(0, vocab_size, (seq_len,), generator=gen).tolist()
            m_pos = [2, 5, 9, 13][:n_mask]
            # Uniformly random targets create near-zero true SNR across masked tokens (PILOT14 regime)
            targets = torch.randint(0, vocab_size, (len(m_pos),), generator=gen).tolist()
            chunks.append(
                {
                    "domain": dom,
                    "ids": ids,
                    "mask_positions": m_pos,
                    "targets": targets,
                }
            )
    return chunks


# ---------------------------------------------------------------------------
# Scenario 1: PILOT12-Style Small-Probe High-DOF Surgery vs Full-DOF & Scalar
# ---------------------------------------------------------------------------


def test_s1_small_probe_high_dof_pilot12_james_stein_beats_full_dof_and_scalar() -> None:
    """S1: Unregularized full-DOF overfits small probe while James-Stein rank-4 beats scalar & base."""
    model = _SmallProbeHighDOFCausalLM(
        hidden_size=384,
        vocab_size=40,
        nuisance_head_scale=1.80,
        spec_norm_init=0.19,
        d0_confidence_scale=0.18,
        nuisance_norm_init=0.0,
        sig_head_mult=3.0,
        tok_emb_scale=0.12,
        block_parity=True,
        seed=20264801,
    )
    # 4 * 5 = 20 probe chunks << 384 DOF (probe/dof = 0.052 < 4.0)
    probe_chunks = _make_causal_chunks(n_per_domain=5, seq_len=16, seed=101, d0_modal_prob=0.72, spec_modal_prob=0.88)
    selection_chunks = _make_causal_chunks(n_per_domain=12, seq_len=16, seed=202, d0_modal_prob=0.72, spec_modal_prob=0.88)
    eval_chunks = _make_causal_chunks(n_per_domain=12, seq_len=16, seed=303, d0_modal_prob=0.72, spec_modal_prob=0.88)

    # First verify direct Empirical-Bayes James-Stein rank-4 fit on synthetic probe gradients
    gen = torch.Generator(device="cpu").manual_seed(4811)
    dim = 128
    # D0 spans dims 0..11; specialist signal sits on 3 directions inside ker(G_D0) plus 108-dim noise
    g_d0 = torch.randn((8, dim), generator=gen, dtype=torch.float32) * 0.01
    g_d0[:, :12] = torch.randn((8, 12), generator=gen, dtype=torch.float32) * 1.5
    signal_dirs = torch.zeros((3, dim), dtype=torch.float32)
    signal_dirs[0, 12] = 2.6
    signal_dirs[1, 13] = -2.2
    signal_dirs[2, 14] = 2.0
    coeffs = torch.ones((18, 3), dtype=torch.float32) + 0.15 * torch.randn((18, 3), generator=gen)
    g_spec = coeffs @ signal_dirs
    g_spec[:, 20:] = 0.45 * torch.randn((18, dim - 20), generator=gen, dtype=torch.float32)
    g_spec[:, :12] = 0.45 * torch.randn((18, 12), generator=gen, dtype=torch.float32)

    js_direct = fit_james_stein_rank_k_fisher_norm_step(
        g_spec,
        g_d0,
        max_rank=4,
        ridge_scale=1e-2,
        max_step_norm=0.25,
    )
    assert js_direct["effective_rank_k"] <= 4
    assert js_direct["zero_step"] is False
    assert js_direct["shrinkage_factors"][0] > 0.7
    assert js_direct["mean_shrinkage_factor"] > 0.15
    assert js_direct["nullspace_orthogonal"] is True
    assert js_direct["nullspace_orthogonality_max_abs"] < 1e-6
    assert float((g_d0 @ js_direct["delta_w"]).abs().max().item()) < 1e-6
    assert js_direct["delta_norm"] <= 0.25 + 1e-6

    # Execute full end-to-end verified surgery (inplace=False keeps parent pristine)
    cfg = JamesSteinNormSurgeonConfig(
        min_chunks_per_dof=4.0,
        max_rank=4,
        ridge_scale=1e-5,
        max_step_norm=0.30,
        min_probe_snr=1.0,
        score_from=2,
        eta_grid=(0.15, 0.30, 0.60, 1.00),
        tau_grid=(0.98, 1.00, 1.03),
    )
    base_sha = compute_module_state_sha256(model)
    edited_model, receipt = execute_verified_james_stein_norm_surgery(
        model,
        probe_chunks,
        selection_chunks=selection_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
        inplace=False,
    )
    assert compute_module_state_sha256(model) == base_sha
    assert receipt["object"] == SURGEON_RECEIPT_OBJECT
    assert receipt["schema"] == SURGEON_SCHEMA
    assert verify_james_stein_norm_receipt_provenance(receipt) is True

    diag = receipt["probe_overfit_diagnostics"]
    assert diag["full_dof_overfit_risk"] is True
    assert diag["probe_to_full_dof_ratio"] < 4.0

    sel = receipt["selection_comparison"]
    # Unregularized full-DOF Fisher overfits the 20-chunk / 384-DOF probe on held-out selection
    assert sel["full_dof_control"]["nll_mean"] > sel["james_stein_rank_k"]["nll_mean"]
    assert sel["full_dof_control"]["nll_mean"] > sel["base"]["nll_mean"] or not sel["full_dof_control"]["guard_ok"]
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["selected_arm"] == "james_stein_rank_k"
    assert receipt["fallback_triggered"] is False
    assert receipt["status"] == "ACCEPTED_JAMES_STEIN_RANK_K"
    assert receipt["nullspace_orthogonality_max_abs"] < 1e-6
    assert receipt["general_drift_ratio"] <= 0.01
    assert receipt["delta_e"] < 0.0
    assert sel["james_stein_rank_k"]["nll_mean"] < sel["scalar_folded_tau"]["nll_mean"]
    assert sel["james_stein_rank_k"]["nll_mean"] < sel["base"]["nll_mean"]
    assert receipt["nll_mean"] < receipt["base_nll_mean"]
    assert 1 <= receipt["edited_params"] <= 384
    assert compute_module_state_sha256(edited_model) == receipt["final_state_sha256"]


# ---------------------------------------------------------------------------
# Scenario 2: Pure-Noise / Low-SNR MLM Probe (PILOT14 Overfit Guard)
# ---------------------------------------------------------------------------


def test_s2_low_snr_pure_noise_and_pilot14_mlm_overfit_protection() -> None:
    """S2: Pure-noise / low-SNR probes contract James-Stein shrinkage to 0 and fall back or roll back cleanly."""
    # Part A: Pure-noise specialist gradients contract c_j -> 0
    gen = torch.Generator(device="cpu").manual_seed(4821)
    dim = 64
    g_d0 = torch.randn((8, dim), generator=gen, dtype=torch.float64) * 0.5
    # Independent zero-mean Gaussian noise across 12 specialist chunks -> SNR << 1 per mode
    g_noise = torch.randn((12, dim), generator=gen, dtype=torch.float64) * 1.5
    js_noise = fit_james_stein_rank_k_fisher_norm_step(
        g_noise,
        g_d0,
        max_rank=4,
        ridge_scale=1e-2,
        max_step_norm=0.05,
    )
    assert js_noise["mean_shrinkage_factor"] < 0.05
    assert all(c <= 0.1 for c in js_noise["shrinkage_factors"])
    assert js_noise["delta_norm"] <= 0.05

    # Strictly zero mean in every nullspace mode -> exact c_j == 0.0 and zero_step
    g_centered = g_noise - g_noise.mean(dim=0, keepdim=True)
    js_zero = fit_james_stein_rank_k_fisher_norm_step(g_centered, g_d0, max_rank=4)
    assert js_zero["mean_shrinkage_factor"] == 0.0
    assert js_zero["zero_step"] is True
    assert float(js_zero["delta_w"].abs().max().item()) == 0.0

    # Part B: Overconfident causal model with noisy low-SNR specialist probe -> automatic scalar fallback
    overconf_model = _SmallProbeHighDOFCausalLM(
        hidden_size=48,
        vocab_size=40,
        nuisance_head_scale=0.0,
        spec_norm_init=1.0,
        d0_confidence_scale=5.0,
        seed=20264822,
    )
    with torch.no_grad():
        # Conflicting/noisy specialist directions + high overconfidence on D0 so tau > 1 helps
        overconf_model.embed.weight[10:] = 2.5 * torch.randn((30, 48), generator=gen)
        overconf_model.lm_head.weight[10:] = 6.0 * torch.randn((30, 48), generator=gen)
    noisy_probe = _make_causal_chunks(n_per_domain=4, seq_len=14, seed=401, d0_modal_prob=0.62, spec_modal_prob=0.35)
    noisy_sel = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=402, d0_modal_prob=0.62, spec_modal_prob=0.35)
    cfg_fb = JamesSteinNormSurgeonConfig(
        min_probe_snr=50.0,  # Forces low-SNR auto-fallback to 1-DOF scalar
        tau_grid=(0.95, 1.00, 1.15, 1.35, 1.60),
        max_general_drift_rel=0.01,
        max_domain_drift_rel=0.02,
    )
    _, rec_fb = execute_verified_james_stein_norm_surgery(
        overconf_model,
        noisy_probe,
        selection_chunks=noisy_sel,
        config=cfg_fb,
        inplace=True,
    )
    assert rec_fb["fallback_triggered"] is True
    assert rec_fb["general_drift_ratio"] <= 0.01
    assert rec_fb["selected_arm"] in ("scalar_folded_tau", "base_rollback")
    assert verify_james_stein_norm_receipt_provenance(rec_fb) is True

    # Part C: PILOT14-style Bidirectional MLM Encoder with random masked targets -> zero D0 violation
    mlm = _BidirectionalMLMEncoder(hidden_size=64, vocab_size=32, seed=20264823)
    mlm_probe = _make_mlm_chunks(n_per_domain=4, seq_len=16, n_mask=4, vocab_size=32, seed=501)
    mlm_sel = _make_mlm_chunks(n_per_domain=4, seq_len=16, n_mask=4, vocab_size=32, seed=502)
    mlm_eval = _make_mlm_chunks(n_per_domain=4, seq_len=16, n_mask=4, vocab_size=32, seed=503)
    init_mlm_sha = compute_module_state_sha256(mlm)
    _, rec_mlm = execute_verified_james_stein_norm_surgery(
        mlm,
        mlm_probe,
        selection_chunks=mlm_sel,
        eval_chunks=mlm_eval,
        config=JamesSteinNormSurgeonConfig(max_rank=4, tau_grid=(0.95, 1.00, 1.05, 1.12)),
        inplace=True,
    )
    assert verify_james_stein_norm_receipt_provenance(rec_mlm) is True
    assert rec_mlm["norm_kind"] == "layernorm"
    assert rec_mlm["norm_attr"] == "head.norm"
    assert rec_mlm["head_attr"] == "decoder"
    assert rec_mlm["general_drift_ratio"] <= 0.01
    if not rec_mlm["accepted"]:
        assert compute_module_state_sha256(mlm) == init_mlm_sha


# ---------------------------------------------------------------------------
# Scenario 3: Scale-Dependent Spectral Anisotropy & Outlier Channel Detection
# ---------------------------------------------------------------------------


def test_s3_scale_dependent_spectral_anisotropy_and_outlier_channels() -> None:
    """S3: Accurately separates 0.5B moderate anisotropy (~0.35), 3B high anisotropy (~0.69), and rogue channels."""
    gen = torch.Generator(device="cpu").manual_seed(4831)

    def _build_anisotropic_probe(
        n_chunks: int, dim: int, rogue_specs: list[tuple[int, float]], *, spike_amp: float = 0.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        act = torch.randn((n_chunks, dim), generator=gen, dtype=torch.float64)
        if spike_amp > 0.0:
            z = torch.randn((n_chunks, 1), generator=gen, dtype=torch.float64)
            act[:, 0] = act[:, 0] + spike_amp * math.sqrt(float(dim)) * z[:, 0]
        med_est = float(act.pow(2).mean(dim=0).sqrt().median().item())
        shared_sign = torch.sign(torch.randn((n_chunks,), generator=gen, dtype=torch.float64) + 0.05)
        for ch_idx, mult in rogue_specs:
            act[:, ch_idx] = mult * med_est * shared_sign
        grads = 0.05 * torch.randn((n_chunks, dim), generator=gen, dtype=torch.float64)
        grads[:, 1] = grads[:, 1] + 0.35
        return grads, act

    # Mirror PILOT12 (Qwen2.5-0.5B: dim=896, top1~0.35, 1 rogue dim @ 22x)
    g_05b, act_05b = _build_anisotropic_probe(48, 896, [(42, 22.0)])
    diag_05b = compute_effective_probe_dof_and_overfit_risk(
        g_05b, 896, min_chunks_per_dof=4.0, raw_activations=act_05b, outlier_ratio_threshold=5.0
    )
    assert diag_05b["full_dof_overfit_risk"] is True
    assert 0.30 <= diag_05b["top1_sv_energy_frac"] <= 0.48
    assert diag_05b["anisotropy_regime"] == "moderate_anisotropy"
    assert 42 in diag_05b["outlier_channel_indices"]
    assert diag_05b["max_outlier_ratio"] > 15.0
    assert diag_05b["risk_tier"] == "HIGH"
    assert diag_05b["recommended_mode"] == "james_stein_rank_k"

    # Mirror PILOT9 (Qwen2.5-3B: dim=2048, top1~0.69, 2 rogue dims @ 60x)
    g_3b, act_3b = _build_anisotropic_probe(32, 2048, [(19, 48.0), (53, 47.0)])
    diag_3b = compute_effective_probe_dof_and_overfit_risk(
        g_3b, 2048, min_chunks_per_dof=4.0, raw_activations=act_3b, outlier_ratio_threshold=5.0
    )
    assert diag_3b["top1_sv_energy_frac"] > diag_05b["top1_sv_energy_frac"]
    assert 0.60 <= diag_3b["top1_sv_energy_frac"] <= 0.84
    assert diag_3b["anisotropy_regime"] == "high_anisotropy"
    assert 19 in diag_3b["outlier_channel_indices"]
    assert 53 in diag_3b["outlier_channel_indices"]
    assert diag_3b["max_outlier_ratio"] > 30.0

    # Mirror PILOT14 (extreme anisotropy >= 0.85 + low SNR -> CRITICAL / scalar_fallback_or_abort)
    dim_mb = 128
    _, act_mb = _build_anisotropic_probe(32, dim_mb, [(7, 10.5), (33, 9.7), (77, 6.5)], spike_amp=5.0)
    g_mb_noise = torch.randn((32, dim_mb), generator=gen, dtype=torch.float64)
    g_mb_noise = g_mb_noise - g_mb_noise.mean(dim=0, keepdim=True)
    diag_mb = compute_effective_probe_dof_and_overfit_risk(
        g_mb_noise, dim_mb, min_chunks_per_dof=4.0, raw_activations=act_mb, outlier_ratio_threshold=5.0
    )
    assert diag_mb["anisotropy_regime"] == "extreme_anisotropy"
    assert diag_mb["risk_tier"] == "CRITICAL"
    assert diag_mb["recommended_mode"] == "scalar_fallback_or_abort"

    # High-probe safe DOF regime + low anisotropy + single-chunk edge case
    small_dim = 8
    g_safe = (
        0.25 * torch.ones((40, small_dim), dtype=torch.float64)
        + torch.eye(small_dim, dtype=torch.float64).repeat(5, 1)
        + 0.04 * torch.randn((40, small_dim), generator=gen, dtype=torch.float64)
    )
    diag_safe = compute_effective_probe_dof_and_overfit_risk(g_safe, small_dim, min_chunks_per_dof=4.0)
    assert diag_safe["full_dof_overfit_risk"] is False
    assert diag_safe["risk_tier"] == "LOW"
    assert diag_safe["recommended_mode"] == "full_dof_safe"

    diag_one = compute_effective_probe_dof_and_overfit_risk(torch.zeros((1, small_dim)), small_dim, min_chunks_per_dof=4.0)
    assert diag_one["recommended_max_rank"] == 0
    assert diag_one["probe_snr"] == 0.0
    assert diag_one["risk_tier"] == "CRITICAL"


# ---------------------------------------------------------------------------
# Scenario 4: ANSE Maximum Pain Atomic Bitwise Rollback on Infeasible Guard
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_on_infeasible_guard() -> None:
    """S4: Any guard breach or non-improving eval triggers ANSE Maximum Pain and exact bitwise rollback."""
    # Case A1: Candidate passes selection (tau > 1 fixes uncertain D0), then eval_b (modal 0.99) breaches guard
    eval_breach_model = _SmallProbeHighDOFCausalLM(
        hidden_size=48,
        vocab_size=40,
        nuisance_head_scale=0.05,
        spec_norm_init=1.0,
        d0_confidence_scale=0.45,
        seed=20264842,
    )
    breach_sha = compute_module_state_sha256(eval_breach_model)
    breach_tensors = {k: v.detach().clone() for k, v in eval_breach_model.state_dict().items()}
    probe_b = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=701, d0_modal_prob=0.35, spec_modal_prob=0.35)
    sel_b = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=702, d0_modal_prob=0.35, spec_modal_prob=0.35)
    eval_b = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=703, d0_modal_prob=0.99, spec_modal_prob=0.99)
    _, rec_pain = execute_verified_james_stein_norm_surgery(
        eval_breach_model,
        probe_b,
        selection_chunks=sel_b,
        eval_chunks=eval_b,
        config=JamesSteinNormSurgeonConfig(
            max_rank=4,
            max_step_norm=0.40,
            max_general_drift_rel=0.001,
            max_domain_drift_rel=0.05,
            tau_grid=(1.00, 1.50, 2.00),
        ),
        inplace=True,
    )
    assert verify_james_stein_norm_receipt_provenance(rec_pain) is True
    assert rec_pain["accepted"] is False
    assert rec_pain["rolled_back"] is True
    assert rec_pain["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_pain["cand_anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_pain["final_state_sha256"] == breach_sha
    assert compute_module_state_sha256(eval_breach_model) == breach_sha
    for k, orig_t in breach_tensors.items():
        assert torch.equal(eval_breach_model.state_dict()[k], orig_t)

    # Case A2: Eval energy non-improving when guard is relaxed -> ROLLED_BACK_ENERGY_NON_IMPROVING
    eval_worse = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=704, d0_modal_prob=0.99, spec_modal_prob=0.99)
    non_imp_model = _SmallProbeHighDOFCausalLM(
        hidden_size=48,
        vocab_size=40,
        nuisance_head_scale=0.05,
        spec_norm_init=1.0,
        d0_confidence_scale=0.45,
        seed=20264843,
    )
    non_imp_sha = compute_module_state_sha256(non_imp_model)
    probe_n = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=711, d0_modal_prob=0.35, spec_modal_prob=0.35)
    sel_n = _make_causal_chunks(n_per_domain=6, seq_len=14, seed=712, d0_modal_prob=0.35, spec_modal_prob=0.35)
    _, rec_non_imp = execute_verified_james_stein_norm_surgery(
        non_imp_model,
        probe_n,
        selection_chunks=sel_n,
        eval_chunks=eval_worse,
        config=JamesSteinNormSurgeonConfig(
            max_rank=4,
            max_step_norm=0.40,
            max_general_drift_rel=5.0,
            max_domain_drift_rel=5.0,
            tau_grid=(1.00, 1.50, 2.00),
        ),
        inplace=True,
    )
    assert rec_non_imp["accepted"] is False
    assert rec_non_imp["rolled_back"] is True
    assert rec_non_imp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_non_imp["final_state_sha256"] == non_imp_sha

    # Case B: Unchanged fixed point when tau=1.0 only and no rank-k activation allowed
    _, rec_fixed = execute_verified_james_stein_norm_surgery(
        eval_breach_model,
        probe_b,
        selection_chunks=sel_b,
        config=JamesSteinNormSurgeonConfig(
            tau_grid=(1.0,),
            min_probe_snr=1e6,
            max_general_drift_rel=0.0,
            max_domain_drift_rel=0.0,
        ),
        inplace=True,
    )
    assert rec_fixed["accepted"] is False
    assert rec_fixed["final_state_sha256"] == breach_sha
    assert rec_fixed["status"] in ("FIXED_POINT_UNCHANGED", "REJECTED_GUARD_VIOLATION_UNCHANGED")

    # Case C: Non-strict energy drop branch + auto-split probe_chunks (selection_chunks=None)
    mlm = _BidirectionalMLMEncoder(hidden_size=48, vocab_size=32, seed=20264844)
    auto_chunks = _make_causal_chunks(n_per_domain=4, seq_len=12, seed=705)
    # Remap token IDs into [0, 32)
    for ch in auto_chunks:
        ch["ids"] = [tid % 32 for tid in ch["ids"]]
    _, rec_auto = execute_verified_james_stein_norm_surgery(
        mlm,
        auto_chunks,
        config=JamesSteinNormSurgeonConfig(
            require_strict_energy_drop=False,
            max_general_drift_rel=0.02,
            max_domain_drift_rel=0.05,
            tau_grid=(0.95, 1.00, 1.08),
        ),
        inplace=False,
    )
    assert verify_james_stein_norm_receipt_provenance(rec_auto) is True
    assert rec_auto["status"].startswith(("ACCEPTED_", "FIXED_POINT_", "ROLLED_BACK_", "REJECTED_"))

    # Tampered receipt fails provenance verification
    bad_rec = dict(rec_auto)
    bad_rec["nll_mean"] = float(rec_auto["nll_mean"]) + 0.123
    assert verify_james_stein_norm_receipt_provenance(bad_rec) is False


# ---------------------------------------------------------------------------
# Scenario 5: Validation Errors & Security/Stub/OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_guardrail_compliance() -> None:
    """S5: Exhaustive input validation plus AGPL-3.0, ZeroStubAudit, PRSecurityScanner, and OwnerOrderGuardrail."""
    # 1. Config validation
    with pytest.raises(TypeError, match="real number"):
        JamesSteinNormSurgeonConfig(min_chunks_per_dof="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="> 0.0"):
        JamesSteinNormSurgeonConfig(min_chunks_per_dof=0.0)
    with pytest.raises(TypeError, match="int"):
        JamesSteinNormSurgeonConfig(max_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        JamesSteinNormSurgeonConfig(max_rank=0)
    with pytest.raises(ValueError, match=">= 0.0"):
        JamesSteinNormSurgeonConfig(ridge_scale=-1.0)
    with pytest.raises(ValueError, match="finite"):
        JamesSteinNormSurgeonConfig(max_step_norm=float("nan"))
    with pytest.raises(ValueError, match="< 1.0"):
        JamesSteinNormSurgeonConfig(nullspace_sv_tol=1.0)
    with pytest.raises(ValueError, match=">= 1.0"):
        JamesSteinNormSurgeonConfig(anse_safe_general_ratio=0.9)
    with pytest.raises(ValueError, match="> 1.0"):
        JamesSteinNormSurgeonConfig(outlier_ratio_threshold=1.0)
    with pytest.raises(ValueError, match=">= 0"):
        JamesSteinNormSurgeonConfig(score_from=-1)
    with pytest.raises(TypeError, match="bool"):
        JamesSteinNormSurgeonConfig(require_strict_energy_drop="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-empty sequence"):
        JamesSteinNormSurgeonConfig(tau_grid="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        JamesSteinNormSurgeonConfig(tau_grid=())
    with pytest.raises(ValueError, match="<= 1.0"):
        JamesSteinNormSurgeonConfig(eta_grid=(1.5,))
    with pytest.raises(TypeError, match="non-empty sequence"):
        JamesSteinNormSurgeonConfig(general_prefixes="D0_")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        JamesSteinNormSurgeonConfig(general_prefixes=())

    # 2. compute_effective_probe_dof_and_overfit_risk validation
    with pytest.raises(ValueError, match="> 1.0"):
        compute_effective_probe_dof_and_overfit_risk(torch.ones((4, 4)), 4, outlier_ratio_threshold=1.0)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_effective_probe_dof_and_overfit_risk("bad", 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim == 2"):
        compute_effective_probe_dof_and_overfit_risk(torch.ones(4), 4)
    with pytest.raises(ValueError, match="non-empty"):
        compute_effective_probe_dof_and_overfit_risk(torch.empty((0, 4)), 4)
    with pytest.raises(TypeError, match="floating-point"):
        compute_effective_probe_dof_and_overfit_risk(torch.ones((4, 4), dtype=torch.long), 4)
    with pytest.raises(ValueError, match="finite"):
        compute_effective_probe_dof_and_overfit_risk(torch.tensor([[1.0, float("inf")]]), 2)
    with pytest.raises(ValueError, match="match dim"):
        compute_effective_probe_dof_and_overfit_risk(torch.ones((4, 5)), 4)
    with pytest.raises(ValueError, match="last dim"):
        compute_effective_probe_dof_and_overfit_risk(torch.ones((4, 4)), 4, raw_activations=torch.ones((4, 5)))

    # 3. fit_james_stein_rank_k_fisher_norm_step validation & degenerate edges
    with pytest.raises(ValueError, match="must match"):
        fit_james_stein_rank_k_fisher_norm_step(torch.ones((4, 6)), torch.ones((4, 5)))
    with pytest.raises(ValueError, match="< 1.0"):
        fit_james_stein_rank_k_fisher_norm_step(torch.ones((4, 6)), torch.ones((4, 6)), nullspace_sv_tol=1.0)
    with pytest.raises(ValueError, match="length"):
        fit_james_stein_rank_k_fisher_norm_step(
            torch.ones((4, 6)), torch.ones((4, 6)), tok_grads_fisher_diag=torch.ones(5)
        )
    with pytest.raises(ValueError, match="non-negative"):
        fit_james_stein_rank_k_fisher_norm_step(
            torch.ones((4, 6)), torch.ones((4, 6)), tok_grads_fisher_diag=torch.tensor([-1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        )

    res_empty_ker = fit_james_stein_rank_k_fisher_norm_step(torch.ones((3, 4)), torch.eye(4))
    assert res_empty_ker["nullspace_dim"] == 0 and res_empty_ker["zero_step"] is True
    res_zero_g = fit_james_stein_rank_k_fisher_norm_step(torch.zeros((3, 4)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert res_zero_g["zero_step"] is True
    res_one_spec = fit_james_stein_rank_k_fisher_norm_step(
        torch.ones((1, 4)), torch.tensor([[1.0, 0.0, 0.0, 0.0]]), tok_grads_fisher_diag=torch.ones(4)
    )
    assert res_one_spec["zero_step"] is True
    # Zero variance, non-zero mean on 2 specialist chunks -> c_j == 1.0
    res_const = fit_james_stein_rank_k_fisher_norm_step(
        torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    )
    assert res_const["shrinkage_factors"] == [1.0]
    res_zero_ker = fit_james_stein_rank_k_fisher_norm_step(
        torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
        torch.zeros((2, 4)),
    )
    assert res_zero_ker["general_nullspace_rank_removed"] == 0
    assert res_zero_ker["nullspace_dim"] == 4

    # 4. execute_verified_james_stein_norm_surgery & provenance validation
    m_ok = _SmallProbeHighDOFCausalLM(hidden_size=16, vocab_size=40, seed=20264851)
    good_chunks = _make_causal_chunks(n_per_domain=4, seq_len=10, seed=801)
    with pytest.raises(TypeError, match="JamesSteinNormSurgeonConfig"):
        execute_verified_james_stein_norm_surgery(m_ok, good_chunks, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_james_stein_norm_surgery("bad", good_chunks)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="norm_kind"):
        execute_verified_james_stein_norm_surgery(m_ok, good_chunks, norm_kind="bad")
    with pytest.raises(ValueError, match="final normalization"):
        execute_verified_james_stein_norm_surgery(nn.Linear(4, 4), good_chunks)
    with pytest.raises(ValueError, match="unembedding head"):
        no_head = nn.Module()
        no_head.norm = nn.RMSNorm(4)
        execute_verified_james_stein_norm_surgery(no_head, good_chunks)
    with pytest.raises(ValueError, match="must match norm dim"):
        bad_hw = nn.Module()
        bad_hw.norm = nn.RMSNorm(4)
        bad_hw.head = nn.Linear(5, 8, bias=False)
        execute_verified_james_stein_norm_surgery(bad_hw, good_chunks)
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_james_stein_norm_surgery(m_ok, [], selection_chunks=good_chunks)
    with pytest.raises(ValueError, match="at least 4 probe chunks"):
        execute_verified_james_stein_norm_surgery(m_ok, good_chunks[:2])
    with pytest.raises(ValueError, match="Automatic"):
        execute_verified_james_stein_norm_surgery(
            m_ok,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D1_s", "ids": [10, 11, 12, 13]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
                {"domain": "D0_g", "ids": [0, 1, 2, 3]},
            ],
        )
    with pytest.raises(ValueError, match="containing 'ids'"):
        execute_verified_james_stein_norm_surgery(m_ok, [{"domain": "D0"}], selection_chunks=good_chunks)
    with pytest.raises(TypeError, match="integer dtype"):
        execute_verified_james_stein_norm_surgery(
            m_ok, [{"domain": "D0", "ids": torch.ones(4)}], selection_chunks=good_chunks
        )
    with pytest.raises(TypeError, match="sequence or 1D"):
        execute_verified_james_stein_norm_surgery(
            m_ok, [{"domain": "D0", "ids": "bad"}], selection_chunks=good_chunks
        )
    with pytest.raises(ValueError, match="1D"):
        execute_verified_james_stein_norm_surgery(
            m_ok, [{"domain": "D0", "ids": torch.ones((2, 2), dtype=torch.long)}], selection_chunks=good_chunks
        )
    with pytest.raises(ValueError, match="length >= 2"):
        execute_verified_james_stein_norm_surgery(
            m_ok, [{"domain": "D0", "ids": [1]}], selection_chunks=good_chunks
        )
    with pytest.raises(ValueError, match="token IDs"):
        execute_verified_james_stein_norm_surgery(
            m_ok, [{"domain": "D0", "ids": [0, 999]}], selection_chunks=good_chunks
        )
    with pytest.raises(ValueError, match="consistently"):
        execute_verified_james_stein_norm_surgery(
            m_ok,
            [
                {"domain": "D0", "ids": [0, 1, 2, 3], "mask_positions": [1], "targets": [2]},
                {"domain": "D1", "ids": [10, 11, 12, 13]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="mask_positions"):
        execute_verified_james_stein_norm_surgery(
            m_ok,
            [
                {"domain": "D0", "ids": [0, 1, 2, 3], "mask_positions": [10], "targets": [2]},
                {"domain": "D1", "ids": [10, 11, 12, 13], "mask_positions": [1], "targets": [12]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="both general"):
        execute_verified_james_stein_norm_surgery(
            m_ok,
            [{"domain": "D0_a", "ids": [0, 1, 2, 3]}, {"domain": "D0_b", "ids": [0, 1, 2, 3]}],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="too short"):
        execute_verified_james_stein_norm_surgery(m_ok, good_chunks, selection_chunks=good_chunks, score_from=20)

    class _SkipNorm(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(4)
            self.lm_head = nn.Linear(4, 8, bias=False)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros((ids.shape[0], ids.shape[1], 8))

    with pytest.raises(RuntimeError, match="not invoked"):
        execute_verified_james_stein_norm_surgery(
            _SkipNorm(),
            [{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [4, 5, 6, 7]}],
            selection_chunks=[{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [4, 5, 6, 7]}],
        )

    class _BadShapeNorm(nn.Module):
        def __init__(self, *, nonfinite: bool = False) -> None:
            super().__init__()
            self.norm = nn.RMSNorm(4)
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.nonfinite = nonfinite

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            if self.nonfinite:
                x = torch.full((ids.shape[0], ids.shape[1], 4), float("nan"))
            else:
                x = torch.zeros((ids.shape[0] + 1, ids.shape[1], 4))
            return self.lm_head(self.norm(x))

    with pytest.raises(ValueError, match="incompatible"):
        execute_verified_james_stein_norm_surgery(
            _BadShapeNorm(nonfinite=False),
            [{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [4, 5, 6, 7]}],
            selection_chunks=[{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [4, 5, 6, 7]}],
        )
    with pytest.raises(ValueError, match="non-finite"):
        execute_verified_james_stein_norm_surgery(
            _BadShapeNorm(nonfinite=True),
            [{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [4, 5, 6, 7]}],
            selection_chunks=[{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [4, 5, 6, 7]}],
        )

    # Extra edge validation: norm.bias mismatch, head.bias mismatch, explicit norm_attr/head_attr, MLM & bool tensor
    bad_nb = nn.Module()
    bad_nb.norm = nn.LayerNorm(4)
    bad_nb.norm.bias = nn.Parameter(torch.zeros(3))
    bad_nb.head = nn.Linear(4, 8)
    with pytest.raises(ValueError, match="bias length"):
        execute_verified_james_stein_norm_surgery(bad_nb, good_chunks)

    bad_hb = nn.Module()
    bad_hb.norm = nn.RMSNorm(4)
    bad_hb.head = nn.Linear(4, 8)
    bad_hb.head.bias = nn.Parameter(torch.zeros(3))
    with pytest.raises(ValueError, match="vocab size"):
        execute_verified_james_stein_norm_surgery(bad_hb, good_chunks)

    with pytest.raises(ValueError, match="final normalization"):
        execute_verified_james_stein_norm_surgery(m_ok, good_chunks, norm_attr="missing.norm")

    mlm_e = _BidirectionalMLMEncoder(hidden_size=16, vocab_size=32, seed=20264855)
    with pytest.raises(ValueError, match="both 'mask_positions' and 'targets'"):
        execute_verified_james_stein_norm_surgery(
            mlm_e,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3], "mask_positions": [1]},
                {"domain": "D1_s", "ids": [4, 5, 6, 7], "mask_positions": [1], "targets": [2]},
            ],
            selection_chunks=good_chunks,
            norm_attr="head.norm",
            head_attr="decoder",
        )
    with pytest.raises(ValueError, match="non-ignored targets"):
        execute_verified_james_stein_norm_surgery(
            mlm_e,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3], "mask_positions": [1], "targets": [-100]},
                {"domain": "D1_s", "ids": [4, 5, 6, 7], "mask_positions": [1], "targets": [2]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="must lie in"):
        execute_verified_james_stein_norm_surgery(
            mlm_e,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3], "mask_positions": [1], "targets": [999]},
                {"domain": "D1_s", "ids": [4, 5, 6, 7], "mask_positions": [1], "targets": [2]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(ValueError, match="share mask count"):
        execute_verified_james_stein_norm_surgery(
            mlm_e,
            [
                {"domain": "D0_g", "ids": [0, 1, 2, 3], "mask_positions": [1], "targets": [2]},
                {"domain": "D1_s", "ids": [4, 5, 6, 7], "mask_positions": [1, 2], "targets": [2, 3]},
            ],
            selection_chunks=good_chunks,
        )
    with pytest.raises(TypeError, match="integer dtype"):
        execute_verified_james_stein_norm_surgery(
            m_ok, [{"domain": "D0_g", "ids": torch.tensor([True, False])}], selection_chunks=good_chunks
        )
    with pytest.raises(ValueError, match="sequence length"):
        execute_verified_james_stein_norm_surgery(
            m_ok,
            [{"domain": "D0_g", "ids": [0, 1, 2, 3]}, {"domain": "D1_s", "ids": [10, 11, 12]}],
            selection_chunks=good_chunks,
        )

    # Explicit overrides on execute_verified_james_stein_norm_surgery exercise _resolve_config & dotted norm/head
    mlm_chunks_ok = _make_mlm_chunks(n_per_domain=4, seq_len=12, n_mask=3, vocab_size=32, seed=856)
    _, rec_explicit = execute_verified_james_stein_norm_surgery(
        mlm_e,
        mlm_chunks_ok,
        norm_attr="head.norm",
        head_attr="decoder",
        score_from=0,
        max_rank=2,
        ridge_scale=1e-2,
        max_step_norm=0.05,
        max_general_drift_rel=0.02,
        max_domain_drift_rel=0.02,
        inplace=False,
    )
    assert verify_james_stein_norm_receipt_provenance(rec_explicit) is True

    with pytest.raises(TypeError, match="Mapping"):
        verify_james_stein_norm_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_james_stein_norm_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_james_stein_norm_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert (
        verify_james_stein_norm_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": SURGEON_RECEIPT_OBJECT,
                "schema": SURGEON_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 5. AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn48", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn48", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/james_stein_spectral_norm_surgeon.py", TARGET_MODULE_PATH),
        ("tests/test_james_stein_spectral_norm_surgeon_scenarios.py", TEST_FILE_PATH),
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
        assert "NEURAL-CTRL-DYN-48" in doc_text
        assert "PILOT14" in doc_text
        assert "PILOT12" in doc_text
        assert "PILOT9" in doc_text
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
            "nll_mean": 2.2040,
            "ece_value": 0.0291,
            "shrinkage_factor": 0.2582,
            "general_drift_ratio": -0.00189,
            "anse_energy": -0.00373,
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

    pr_title = "feat(swarm): Empirical-Bayes James-Stein rank-k spectral-shrinkage Fisher-Norm surgeon with small-probe overfit guard"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-48 training-free Empirical-Bayes James-Stein rank-k "
        "spectral-shrinkage final-norm surgeon and small-probe overfit guard grounded in PILOT9, "
        "PILOT12, and PILOT14 under ker(G_D0) null-space orthogonality, ANSE Delta E < 0, and D0 <= +1.0%."
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
        pr_number=1248,
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
