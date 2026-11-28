# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-53: NF4 Block64 + D4 Hessian SVD + <1% OFC FP16 + JS-Norm Recovery."""

from __future__ import annotations

import ast
import importlib.util
import json
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
from hivemind.swarm.logit_calibration_temperature_surgeon import compute_module_state_sha256
from hivemind.swarm.nf4_d4_hessian_ofc_quant_recovery import (
    ANSE_MAXIMUM_PAIN,
    ARM0_FP32_REFERENCE,
    ARM1_NAIVE_NF4_BLOCK64,
    ARM2_NF4_RANDOM_1PCT_FP16_CONTROL,
    ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16,
    ARM4_NF4_D4_HESSIAN_SVD_RANK16,
    ARM5_NF4_D4_HESSIAN_OFC_JS_NORM,
    RECOVERY_RECEIPT_OBJECT,
    RECOVERY_SCHEMA,
    NF4D4HessianOFCConfig,
    compute_d4_hessian_weighted_svd_residual,
    execute_verified_nf4_d4_ofc_js_recovery,
    protect_ofc_and_outlier_channels_fp16,
    quantize_dequantize_dettmers_nf4,
    verify_nf4_d4_ofc_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "nf4_d4_hessian_ofc_quant_recovery.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Keep CPU thread usage single-threaded and deterministic across scenario runs."""
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
    """Biasless RMSNorm exposing ``variance_epsilon`` like Qwen2.5."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (int(self.weight.shape[0]),), self.weight, self.variance_epsilon)


class _TransformerBlock(nn.Module):
    """Minimal anisotropic residual MLP block whose NF4 error has rank-8 Hessian + outlier structure."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down_proj(F.silu(self.up_proj(x)))


class _AnisotropicMultiLayerTransformer(nn.Module):
    """Multi-layer transformer exhibiting NF4 block-absmax collapse, massive outliers, OFC neurons, and ECE shift."""

    def __init__(
        self,
        *,
        hidden_size: int = 448,
        intermediate_size: int = 512,
        vocab_size: int = 48,
        num_layers: int = 2,
        ofc_col: int = 8,
        outlier_cols: tuple[int, ...] = (100, 101),
        outlier_amp: float = 30.0,
        seed: int = 20265301,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.ofc_col = ofc_col
        self.outlier_cols = outlier_cols
        gen = torch.Generator(device="cpu").manual_seed(seed)

        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList(
            [_TransformerBlock(hidden_size, intermediate_size) for _ in range(num_layers)]
        )
        self.norm = _VarianceEpsRMSNorm(hidden_size, eps=1e-6)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        with torch.no_grad():
            self.embed.weight.zero_()
            self.lm_head.weight.zero_()
            self.norm.weight.fill_(1.0)
            # Specialists D1..D3 have suppressed final norm scale (0.45), inviting JS-Norm rank-k recovery
            self.norm.weight[12:48].fill_(0.45)
            for oc in outlier_cols:
                self.norm.weight[oc] = 0.002

            # Mild background variance on passive dims, zeroing block-anchor cols (col % 64 == 60)
            self.embed.weight[:, 64:hidden_size] = 0.06 * torch.randn(
                (vocab_size, hidden_size - 64), generator=gen
            )
            for block_start in range(0, hidden_size, 64):
                anchor_col = block_start + 60
                if anchor_col < hidden_size:
                    self.embed.weight[:, anchor_col] = 0.0
            for idx_m, oc in enumerate(outlier_cols):
                self.embed.weight[:, oc] = 30.0 if idx_m == 0 else -15.0

            # Four mutually orthogonal domain subspaces:
            # embed populates [dim0..dim0+3]; layers transport [dim0..dim0+3] -> [dim0+4..dim0+7];
            # lm_head reads predominantly from layer-transformed [dim0+4..dim0+7] & OFC!
            dom_bases = [(0, 0), (12, 12), (24, 24), (36, 36)]
            for tok0, dim0 in dom_bases:
                for tok in range(tok0, tok0 + 12):
                    self.embed.weight[tok, dim0 : dim0 + 4] = torch.tensor(
                        [3.2, 2.4, 1.8, 1.2], dtype=torch.float32
                    )
                    if tok == tok0 + 1:
                        self.lm_head.weight[tok, dim0 + 4 : dim0 + 8] = torch.tensor(
                            [5.5, 4.2, 3.0, 2.0], dtype=torch.float32
                        )
                        self.lm_head.weight[tok, dim0 : dim0 + 4] = torch.tensor(
                            [0.45, 0.30, 0.20, 0.10], dtype=torch.float32
                        )
                    elif tok == tok0 + 2:
                        self.lm_head.weight[tok, dim0 + 4 : dim0 + 8] = torch.tensor(
                            [4.8, 3.6, 2.5, 1.7], dtype=torch.float32
                        )
                        self.lm_head.weight[tok, dim0 : dim0 + 4] = torch.tensor(
                            [0.35, 0.25, 0.15, 0.10], dtype=torch.float32
                        )
                    else:
                        self.lm_head.weight[tok, dim0 + 4 : dim0 + 8] = torch.tensor(
                            [2.0, 1.5, 1.0, 0.7], dtype=torch.float32
                        )
                        self.lm_head.weight[tok, dim0 : dim0 + 4] = torch.tensor(
                            [0.20, 0.15, 0.10, 0.05], dtype=torch.float32
                        )
                self.lm_head.weight[tok0 + 1, ofc_col] = 2.6
                self.lm_head.weight[tok0 + 2, ofc_col] = 2.2

            # Coherent rank-4-per-domain transforms (total rank 16 across D0..D3) with amplitude 0.105
            # (< 0.1273 NF4 zero-threshold when block absmax = 3.2 on zero-input anchor cols!)
            for li, blk in enumerate(self.layers):
                blk.up_proj.weight.zero_()
                blk.down_proj.weight.zero_()
                for _, dim0 in dom_bases:
                    for seg in (0, 64, 128, 192):
                        if seg + dim0 + 4 <= intermediate_size:
                            blk.up_proj.weight[seg + dim0 : seg + dim0 + 4, dim0 : dim0 + 4] = (
                                0.105 * torch.eye(4)
                            )
                            blk.down_proj.weight[dim0 + 4 : dim0 + 8, seg + dim0 : seg + dim0 + 4] = (
                                0.105 * torch.eye(4)
                            )
                blk.up_proj.weight[ofc_col, ofc_col] = 0.55
                blk.down_proj.weight[ofc_col, ofc_col] = 0.55

                # Massive-activation outlier columns have exact cancellation (0.12*30 - 0.24*15 = 0.0) in FP32,
                # but NF4 asymmetric quantization injects huge noise unless protected in FP16!
                for _, dim0 in dom_bases:
                    blk.up_proj.weight[dim0 : dim0 + 4, outlier_cols[0]] = 0.12
                    blk.up_proj.weight[dim0 : dim0 + 4, outlier_cols[1]] = 0.24

                for block_start in range(0, hidden_size, 64):
                    anchor_col = block_start + 60
                    if anchor_col < hidden_size and anchor_col not in outlier_cols:
                        blk.up_proj.weight[:240, anchor_col] = 3.2 if li == 0 else -3.2
                for block_start in range(0, intermediate_size, 64):
                    anchor_col = block_start + 60
                    if anchor_col < intermediate_size:
                        blk.down_proj.weight[:48, anchor_col] = 3.2 if li == 0 else -3.2

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids).clone()
        easy_flag = ((input_ids[:, :1] % 12) == 1).to(dtype=h.dtype) * 2.0 - 1.0
        h[..., self.ofc_col] = 1.65 * easy_flag.expand_as(h[..., self.ofc_col])
        for blk in self.layers:
            h = blk(h)
        return self.lm_head(self.norm(h))


def _make_multidomain_chunks(
    *,
    n_per_domain: int = 6,
    seq_len: int = 12,
    seed: int = 20265302,
    corrupt_d0_tokens: bool = False,
) -> list[dict[str, Any]]:
    """Build deterministic multi-domain chunks across D0..D3 with easy/hard OFC contrast."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    domain_specs = [
        ("D0_general_prose", 0),
        ("D1_python_systems_code", 12),
        ("D2_formal_math_lean4", 24),
        ("D3_computational_physics", 36),
    ]
    out: list[dict[str, Any]] = []
    for idx in range(n_per_domain):
        is_easy = idx % 2 == 0
        ordered = domain_specs if idx % 2 == 0 else [domain_specs[1], domain_specs[0], domain_specs[3], domain_specs[2]]
        for dom_name, tok0 in ordered:
            if corrupt_d0_tokens and dom_name == "D0_general_prose":
                ids = [1 if is_easy else 4] + [4 if i % 2 == 0 else 5 for i in range(seq_len - 1)]
            elif is_easy:
                ids = [tok0 + 1 if i % 2 == 0 else tok0 + 2 for i in range(seq_len)]
            else:
                rand_t = torch.randint(tok0 + 3, tok0 + 12, (seq_len - 1,), generator=gen).tolist()
                ids = [tok0 + 5, *rand_t]
            out.append({"domain": dom_name, "ids": ids})
    return out


# ---------------------------------------------------------------------------
# S1: Synthetic Anisotropic Multi-Layer Transformer Reproducing PILOT17 Ordering
# ---------------------------------------------------------------------------


def test_s1_anisotropic_multilayer_transformer_pilot17_ordering_and_recovery() -> None:
    """S1: Reproduce PILOT17 ordering ARM5 > ARM4 > ARM3 > ARM2 > ARM1 and >65% NF4 loss recovery."""
    model = _AnisotropicMultiLayerTransformer(seed=20265301)
    probe_chunks = _make_multidomain_chunks(n_per_domain=8, seq_len=12, seed=53101)
    selection_chunks = _make_multidomain_chunks(n_per_domain=8, seq_len=12, seed=53102)
    eval_chunks = _make_multidomain_chunks(n_per_domain=8, seq_len=12, seed=53103)

    cfg = NF4D4HessianOFCConfig(
        blocksize=64,
        d4_svd_rank=16,
        ofc_fraction=0.0023,
        outlier_fraction=0.0045,
        max_storage_ratio_vs_fp16=0.35,
        evaluate_ablation_arms=True,
        enable_js_norm_calibration=True,
        seed=2026112717,
    )
    out_model, receipt = execute_verified_nf4_d4_ofc_js_recovery(
        model,
        probe_chunks,
        selection_chunks=selection_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
        inplace=True,
    )
    assert out_model is model
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["zero_gradient_steps"] is True
    assert receipt["selected_arm"] == ARM5_NF4_D4_HESSIAN_OFC_JS_NORM
    assert verify_nf4_d4_ofc_receipt_provenance(receipt) is True

    # Verify strict PILOT17 selection ordering: ARM5 < ARM4 < ARM3 < ARM2 < ARM1 in NLL
    sel = receipt["selection_summary"]
    nll_fp32 = sel[ARM0_FP32_REFERENCE]["nll_mean"]
    nll_arm1 = sel[ARM1_NAIVE_NF4_BLOCK64]["nll_mean"]
    nll_arm2 = sel[ARM2_NF4_RANDOM_1PCT_FP16_CONTROL]["nll_mean"]
    nll_arm3 = sel[ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16]["nll_mean"]
    nll_arm4 = sel[ARM4_NF4_D4_HESSIAN_SVD_RANK16]["nll_mean"]
    nll_arm5 = sel[ARM5_NF4_D4_HESSIAN_OFC_JS_NORM]["nll_mean"]

    assert nll_fp32 < nll_arm1
    assert nll_arm5 < nll_arm4 < nll_arm3 < nll_arm2 < nll_arm1

    # Verify held-out eval metrics, storage ceiling, D0/OFC orthogonality, and >65% recovery
    assert receipt["quant_loss_recovered_pct"] > 65.0
    assert receipt["storage_ratio_vs_fp16"] <= 0.35
    assert receipt["storage_within_budget"] is True
    assert receipt["delta_nll_vs_nf4"] < 0.0
    assert receipt["anse_energy"] < 0.0
    assert receipt["general_drift_ratio"] <= 0.01
    assert receipt["ece_value"] < receipt["nf4_ece_value"]
    assert receipt["domains_improved_count"] == receipt["domains_total_count"] == 4
    assert receipt["d0_mean_grad_ortho_err"] < 1e-6
    assert receipt["ofc_protected_max_delta"] == 0.0
    assert receipt["oof_auroc"] >= 0.85
    assert receipt["ofc_iou"] < 0.15


# ---------------------------------------------------------------------------
# S2: OFC + Massive-Outlier <1% FP16 Protection vs Matched-Bit Random 1%
# ---------------------------------------------------------------------------


def test_s2_ofc_and_outlier_fp16_beats_matched_bit_random_control() -> None:
    """S2: <1% OFC + massive-outlier FP16 slices strictly beat matched-bit random control."""
    gen = torch.Generator(device="cpu").manual_seed(53201)
    d_out, d_in = 512, 512
    w_fp32 = 0.15 * torch.randn((d_out, d_in), generator=gen, dtype=torch.float32)
    # Massive-outlier + OFC columns carry structured weights & high diagonal input Hessian
    prot_cols = [12, 100, 101]  # 3/512 = 0.586% < 1%
    w_fp32[:, prot_cols] = 1.45 * torch.randn((d_out, len(prot_cols)), generator=gen)

    w_nf4, q_stats = quantize_dequantize_dettmers_nf4(w_fp32, blocksize=64)
    assert q_stats["bits_per_param"] == pytest.approx(4.25, abs=1e-9)
    assert q_stats["storage_ratio_vs_fp16"] == pytest.approx(0.265625, abs=1e-9)
    assert q_stats["quant_format"] == "dettmers_nf4"
    assert len(q_stats["packed_nibbles_sha256"]) == 64

    # Anisotropic Hessian peaking on the protected OFC + massive-outlier columns
    diag_h = torch.full((d_in,), 0.25, dtype=torch.float64)
    diag_h[12] = 18.0
    diag_h[100] = 400.0
    diag_h[101] = 360.0

    # Evaluate output prediction loss (NLL proxy over target logits) with protected vs random columns
    x_eval = 0.4 * torch.randn((64, d_in), generator=gen)
    x_eval[:, 12] = 4.2
    x_eval[:, 100] = 20.0
    x_eval[:, 101] = -19.0
    y_targets = torch.randint(0, d_out, (64,), generator=gen)

    w_prot, p_stats = protect_ofc_and_outlier_channels_fp16(
        w_fp32,
        w_nf4,
        prot_cols,
        mode="columns",
        diag_hessian=diag_h,
        random_seed=53202,
    )
    w_rand, _ = protect_ofc_and_outlier_channels_fp16(
        w_fp32,
        w_nf4,
        p_stats["random_control_indices"],
        mode="columns",
        diag_hessian=diag_h,
        random_seed=53203,
    )
    nll_prot = float(F.cross_entropy(x_eval @ w_prot.T, y_targets).item())
    nll_rand = float(F.cross_entropy(x_eval @ w_rand.T, y_targets).item())
    w_prot_certified, p_cert = protect_ofc_and_outlier_channels_fp16(
        w_fp32,
        w_nf4,
        prot_cols,
        mode="columns",
        diag_hessian=diag_h,
        random_seed=53202,
        protected_nll=nll_prot,
        random_control_nll=nll_rand,
    )

    assert torch.allclose(w_prot_certified[:, prot_cols], w_fp32[:, prot_cols].half().float())
    assert p_cert["matched_bit_verified"] is True
    assert p_cert["protected_channel_fraction"] < 0.01
    assert p_cert["beats_random_weighted_err"] is True
    assert p_cert["weighted_err_advantage_vs_random"] > 0.0
    assert p_cert["beats_random_nll"] is True
    assert p_cert["protected_nll_advantage_vs_random"] > 0.0

    # Also verify row, both (with exact non-double-counted intersections), and empty modes
    _, row_stats = protect_ofc_and_outlier_channels_fp16(w_fp32, w_nf4, prot_cols, mode="rows")
    _, both_stats = protect_ofc_and_outlier_channels_fp16(w_fp32, w_nf32 := w_nf4, prot_cols, mode="both")
    w_empty, empty_stats = protect_ofc_and_outlier_channels_fp16(w_fp32, w_nf32, [], mode="columns")
    assert row_stats["restored_element_count"] == len(prot_cols) * d_in
    assert both_stats["restored_element_count"] == (
        len(prot_cols) * d_out + len(prot_cols) * d_in - len(prot_cols) ** 2
    )
    assert torch.equal(w_empty, w_nf4)
    assert empty_stats["restored_element_count"] == 0
    assert empty_stats["bits_per_param"] == pytest.approx(4.25, abs=1e-9)

    # Also check odd-sized 1D tensor NF4 packing and zero-block invariance
    zeros_and_odd = torch.zeros(67, dtype=torch.float32)
    zeros_and_odd[-1] = 1.25
    q_odd, odd_stats = quantize_dequantize_dettmers_nf4(zeros_and_odd, blocksize=64)
    assert torch.all(q_odd[:64] == 0.0)
    assert odd_stats["num_params"] == 67
    assert odd_stats["num_blocks"] == 2


# ---------------------------------------------------------------------------
# S3: D4 Hessian-Weighted SVD vs Unweighted SVD on Anisotropic Curvature
# ---------------------------------------------------------------------------


def test_s3_d4_hessian_svd_beats_unweighted_svd_and_respects_storage_ceiling() -> None:
    """S3: Closed-form D4 Hessian-weighted thin SVD beats unweighted SVD inside <=0.35x FP16."""
    gen = torch.Generator(device="cpu").manual_seed(53301)
    d_out, d_in = 384, 384

    # Construct quantization residual with rank-16 component on high-Hessian dims (0..64)
    # and larger raw Frobenius rank-16 component on near-zero Hessian dims (64..384)
    u_hi = torch.linalg.qr(torch.randn((d_out, 16), generator=gen, dtype=torch.float64)).Q
    v_hi = torch.zeros((d_in, 16), dtype=torch.float64)
    v_hi[:64, :] = torch.linalg.qr(torch.randn((64, 16), generator=gen, dtype=torch.float64)).Q

    u_lo = torch.linalg.qr(torch.randn((d_out, 16), generator=gen, dtype=torch.float64)).Q
    v_lo = torch.zeros((d_in, 16), dtype=torch.float64)
    v_lo[64:, :] = torch.linalg.qr(torch.randn((d_in - 64, 16), generator=gen, dtype=torch.float64)).Q

    residual_true = (1.5 * u_hi @ v_hi.T + 4.0 * u_lo @ v_lo.T).to(torch.float32)
    w_q = torch.randn((d_out, d_in), generator=gen, dtype=torch.float32) * 0.25
    w_fp32 = w_q + residual_true

    diag_h = torch.full((d_in,), 1e-4, dtype=torch.float64)
    diag_h[:64] = 250.0

    u_r, v_r, d4_stats = compute_d4_hessian_weighted_svd_residual(
        w_fp32,
        w_q,
        diag_h,
        rank=16,
        eps=1e-6,
        blocksize=64,
        max_storage_ratio_vs_fp16=0.35,
        clamp_rank_to_budget=True,
        compare_unweighted_svd=True,
    )
    assert u_r.shape == (d_out, 16)
    assert v_r.shape == (d_in, 16)
    assert d4_stats["rank_effective"] == 16
    assert d4_stats["storage_within_budget"] is True
    assert d4_stats["storage_ratio_vs_fp16"] <= 0.35
    assert d4_stats["beats_unweighted_svd"] is True
    assert d4_stats["weighted_err_after"] < 0.05 * d4_stats["unweighted_svd_weighted_err"]
    assert d4_stats["hessian_weighted_err_reduction_pct"] > 95.0

    # Verify rank clamping inside <= 0.35x FP16 on smaller matrices and 0-rank ceiling
    u_sm, v_sm, sm_stats = compute_d4_hessian_weighted_svd_residual(
        w_fp32[:64, :64],
        w_q[:64, :64],
        diag_h[:64],
        rank=16,
        max_storage_ratio_vs_fp16=0.35,
        clamp_rank_to_budget=True,
    )
    assert int(u_sm.shape[1]) == sm_stats["rank_effective"] < 16
    assert sm_stats["storage_within_budget"] is True
    assert sm_stats["storage_ratio_vs_fp16"] <= 0.35

    u_z, v_z, z_stats = compute_d4_hessian_weighted_svd_residual(
        w_fp32[:64, :64],
        w_q[:64, :64],
        diag_h[:64],
        rank=16,
        max_storage_ratio_vs_fp16=0.25,
        clamp_rank_to_budget=True,
        compare_unweighted_svd=True,
    )
    assert u_z.shape == (64, 0) and v_z.shape == (64, 0)
    assert z_stats["storage_within_budget"] is False


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback & Non-Mutation
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback() -> None:
    """S4: Storage-budget breach and D0 guard violation trigger ANSE 1e6 pain & 0.0 bitwise rollback."""
    model = _AnisotropicMultiLayerTransformer(hidden_size=256, intermediate_size=256, seed=20265304)
    init_sha = compute_module_state_sha256(model)
    probe_chunks = _make_multidomain_chunks(n_per_domain=4, seq_len=8, seed=53401)
    selection_chunks = _make_multidomain_chunks(n_per_domain=4, seq_len=8, seed=53402)
    eval_chunks = _make_multidomain_chunks(n_per_domain=4, seq_len=8, seed=53403)

    # 1. Storage-budget breach (below Naive NF4 0.2656x FP16)
    storage_breach_cfg = NF4D4HessianOFCConfig(
        max_storage_ratio_vs_fp16=0.25,
        evaluate_ablation_arms=False,
        clamp_d4_rank_to_budget=False,
        seed=53404,
    )
    _, rep_storage = execute_verified_nf4_d4_ofc_js_recovery(
        model,
        probe_chunks,
        selection_chunks=selection_chunks,
        eval_chunks=eval_chunks,
        forced_arm=ARM5_NF4_D4_HESSIAN_OFC_JS_NORM,
        config=storage_breach_cfg,
        inplace=True,
    )
    assert rep_storage["accepted"] is False
    assert rep_storage["rolled_back"] is True
    assert rep_storage["status"] == "ROLLED_BACK_STORAGE_BUDGET_EXCEEDED"
    assert rep_storage["storage_within_budget"] is False
    assert rep_storage["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rep_storage["exact_bitwise_rollback_max_err"] == 0.0
    assert rep_storage["final_state_sha256"] == init_sha
    assert compute_module_state_sha256(model) == init_sha

    # 2. Tight D0 guard violation on unseen eval chunks -> ANSE Maximum Pain atomic rollback
    corrupt_eval_chunks = _make_multidomain_chunks(
        n_per_domain=4, seq_len=8, seed=53405, corrupt_d0_tokens=True
    )
    d0_tight_cfg = NF4D4HessianOFCConfig(
        max_general_drift_rel=0.0,
        max_domain_drift_rel=0.0,
        evaluate_ablation_arms=False,
        enable_js_norm_calibration=False,
        seed=53406,
    )
    _, rep_d0 = execute_verified_nf4_d4_ofc_js_recovery(
        model,
        probe_chunks,
        selection_chunks=selection_chunks,
        eval_chunks=corrupt_eval_chunks,
        forced_arm=ARM3_NF4_OFC_AND_OUTLIER_1PCT_FP16,
        config=d0_tight_cfg,
        inplace=True,
    )
    assert rep_d0["accepted"] is False
    assert rep_d0["rolled_back"] is True
    assert rep_d0["guard_ok"] is False
    assert rep_d0["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rep_d0["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rep_d0["exact_bitwise_rollback_max_err"] == 0.0
    assert compute_module_state_sha256(model) == init_sha

    # 3. Forced Naive NF4 yields 0.0 recovery gain -> rolled back cleanly when strict energy drop required
    _, rep_nf4_only = execute_verified_nf4_d4_ofc_js_recovery(
        model,
        probe_chunks,
        selection_chunks=None,
        eval_chunks=None,
        forced_arm=ARM1_NAIVE_NF4_BLOCK64,
        config=NF4D4HessianOFCConfig(evaluate_ablation_arms=False, seed=53407),
        inplace=True,
    )
    assert rep_nf4_only["accepted"] is False
    assert rep_nf4_only["rolled_back"] is True
    assert rep_nf4_only["status"] == "ROLLED_BACK_NO_RECOVERY_GAIN_OVER_NF4"
    assert compute_module_state_sha256(model) == init_sha

    # 4. inplace=False preserves original model bitwise even on accepted run
    cloned_out, rep_inplace_false = execute_verified_nf4_d4_ofc_js_recovery(
        model,
        probe_chunks,
        selection_chunks=selection_chunks,
        eval_chunks=eval_chunks,
        target_module_paths=["layers.0.up_proj", "layers.0.down_proj"],
        early_layer_attr="layers.0",
        config=NF4D4HessianOFCConfig(evaluate_ablation_arms=False, ofc_fraction=0.004, outlier_fraction=0.008),
        inplace=False,
    )
    assert cloned_out is not model
    assert rep_inplace_false["accepted"] is True
    assert compute_module_state_sha256(model) == init_sha


# ---------------------------------------------------------------------------
# S5: Validation Errors, Provenance & Repo Security/Stub/Guardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_provenance_and_compliance_gates() -> None:
    """S5: Cover config/tensor/module validation, provenance checks, and all HiveMind security gates."""
    with pytest.raises(ValueError, match="ofc_fraction"):
        NF4D4HessianOFCConfig(ofc_fraction=1.5)
    with pytest.raises(ValueError, match="massive_ratio_threshold"):
        NF4D4HessianOFCConfig(massive_ratio_threshold=1.0)
    with pytest.raises(ValueError, match="max_storage_ratio_vs_fp16"):
        NF4D4HessianOFCConfig(max_storage_ratio_vs_fp16=1.2)
    with pytest.raises(ValueError, match="nullspace_sv_tol"):
        NF4D4HessianOFCConfig(nullspace_sv_tol=1.0)
    with pytest.raises(TypeError, match="ignore_index"):
        NF4D4HessianOFCConfig(ignore_index=True)  # type: ignore[arg-type]

    w = torch.ones((64, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="non-empty"):
        quantize_dequantize_dettmers_nf4(torch.empty((0, 4)))
    with pytest.raises(ValueError, match="finite"):
        bad_w = w.clone()
        bad_w[0, 0] = float("nan")
        quantize_dequantize_dettmers_nf4(bad_w)
    with pytest.raises(TypeError, match="floating-point"):
        quantize_dequantize_dettmers_nf4(torch.ones((4, 4), dtype=torch.int64))

    with pytest.raises(ValueError, match="must match"):
        protect_ofc_and_outlier_channels_fp16(w, torch.ones((32, 64)), [0])
    with pytest.raises(ValueError, match="mode"):
        protect_ofc_and_outlier_channels_fp16(w, w, [0], mode="diag")
    with pytest.raises(ValueError, match="out of bounds"):
        protect_ofc_and_outlier_channels_fp16(w, w, [999], mode="columns")

    with pytest.raises(ValueError, match="must match"):
        compute_d4_hessian_weighted_svd_residual(w, torch.ones((32, 64)), torch.ones(64))
    with pytest.raises(ValueError, match="max_storage_ratio_vs_fp16"):
        compute_d4_hessian_weighted_svd_residual(w, w, torch.ones(64), max_storage_ratio_vs_fp16=1.5)
    with pytest.raises(ValueError, match="diag_hessian length"):
        compute_d4_hessian_weighted_svd_residual(w, w, torch.ones(32))
    with pytest.raises(ValueError, match="non-negative"):
        compute_d4_hessian_weighted_svd_residual(w, w, -torch.ones(64))

    m_small = _AnisotropicMultiLayerTransformer(hidden_size=128, intermediate_size=192, num_layers=1)
    good_ch = _make_multidomain_chunks(n_per_domain=4, seq_len=8, seed=53501)
    with pytest.raises(TypeError, match="NF4D4HessianOFCConfig"):
        execute_verified_nf4_d4_ofc_js_recovery(m_small, good_ch, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="forced_arm"):
        execute_verified_nf4_d4_ofc_js_recovery(m_small, good_ch, forced_arm="ARM_BAD")
    with pytest.raises(TypeError, match="target_module_paths"):
        execute_verified_nf4_d4_ofc_js_recovery(m_small, good_ch, target_module_paths="layers.0.up_proj")
    with pytest.raises(ValueError, match="non-empty"):
        execute_verified_nf4_d4_ofc_js_recovery(m_small, good_ch, target_module_paths=[])
    with pytest.raises(ValueError, match="2D nn.Linear"):
        execute_verified_nf4_d4_ofc_js_recovery(m_small, good_ch, target_module_paths=["norm"])
    with pytest.raises(ValueError, match="early_layer_attr"):
        execute_verified_nf4_d4_ofc_js_recovery(m_small, good_ch, early_layer_attr="missing.layer")

    no_lin = nn.Module()
    no_lin.norm = nn.RMSNorm(8)
    no_lin.lm_head = nn.Linear(8, 48, bias=False)
    with pytest.raises(ValueError, match="non-head 2D"):
        execute_verified_nf4_d4_ofc_js_recovery(no_lin, good_ch)

    with pytest.raises(TypeError, match="Mapping"):
        verify_nf4_d4_ofc_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_nf4_d4_ofc_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_nf4_d4_ofc_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_nf4_d4_ofc_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": RECOVERY_RECEIPT_OBJECT,
                "schema": RECOVERY_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # Compliance gates: AGPL-3.0, ZeroStubAudit (CI-safe), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn53", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn53", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/nf4_d4_hessian_ofc_quant_recovery.py", TARGET_MODULE_PATH),
        ("tests/test_nf4_d4_hessian_ofc_quant_recovery_scenarios.py", TEST_FILE_PATH),
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
        assert "NEURAL-CTRL-DYN-53" in doc_text
        assert "PILOT17" in doc_text
        assert "NEURAL-CTRL-DYN-51" in doc_text
        assert "NEURAL-CTRL-DYN-38" in doc_text
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
            "nll_mean": 3.137610,
            "ece_value": 0.016648,
            "bits_per_param": 4.9104,
            "quant_loss_recovered_pct": 76.32,
            "general_drift_ratio": -0.02843,
            "anse_energy": -0.120909,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): zero-gradient NF4 Dettmers block64 + D4 Hessian-weighted SVD + <1% OFC FP16 + JS-Norm quantization recovery operator (PILOT17)"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-53 zero-gradient NF4 Dettmers block64 quantization recovery "
        "with D4 diagonal-Hessian weighted thin SVD residual, <1% OFC and massive-outlier FP16 "
        "slice restoration, and final-norm JS-Norm calibration grounded in PILOT17."
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
        pr_number=1253,
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
