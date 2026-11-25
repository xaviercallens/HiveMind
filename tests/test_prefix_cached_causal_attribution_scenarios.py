# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-39: prefix-cached MoE causal attribution & co-adaptation guard."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.prefix_cached_causal_attribution import (
    ANSE_MAXIMUM_PAIN,
    COADAPTATION_RECEIPT_OBJECT,
    COADAPTATION_SCHEMA,
    CoAdaptationGuardedSurgeryConfig,
    compute_prefix_cached_layerwise_attribution,
    detect_router_expert_coadaptation_mismatch,
    execute_coadaptation_guarded_surgery,
    verify_coadaptation_receipt_provenance,
)
from hivemind.swarm.self_healing_checkpoint_rescuer import (
    compute_checkpoint_sha256,
    extract_checkpoint_blocks,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "prefix_cached_causal_attribution.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP thread-pool spin overhead on small CPU tensors during tests."""
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


def _make_zero_mean_ortho_rows(n_rows: int, dim: int, seed: int) -> torch.Tensor:
    """Create ``n_rows`` orthonormal vectors orthogonal to all-ones in ``R^dim``."""
    assert n_rows < dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_rows].T.to(torch.float32).contiguous()


def _install_domain_lora_delta(
    exp_mod: Any,
    in_dir: torch.Tensor,
    out_dir: torch.Tensor,
    *,
    strength: float,
    inter_slice: slice,
    seed: int,
) -> None:
    """Configure a ``CrowdLoRAExpertAdapter`` to emit ``strength * out_dir`` along ``in_dir``."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rank = int(exp_mod.w1_A.shape[0])
    scaling = float(exp_mod.scaling)
    b_scale = math.sqrt(abs(strength)) / max(math.sqrt(scaling), 1e-6)
    sign_b = 1.0 if strength >= 0.0 else -1.0

    exp_mod.w1_A.zero_()
    exp_mod.w1_B.zero_()
    exp_mod.w2_A.zero_()
    exp_mod.w2_B.zero_()
    exp_mod.w3_A.zero_()
    exp_mod.w3_B.zero_()

    i_start = inter_slice.start
    i_width = inter_slice.stop - inter_slice.start
    n_active = min(rank, i_width)
    exp_mod.w1_A[0] = in_dir
    exp_mod.w2_A[0] = in_dir
    exp_mod.w1_B[i_start : i_start + n_active, 0] = 2.5
    exp_mod.w2_B[i_start : i_start + n_active, 0] = 2.5
    exp_mod.w3_A[0, i_start : i_start + n_active] = 1.0 / float(n_active)
    exp_mod.w3_B[:, 0] = sign_b * b_scale * out_dir
    if rank > 1:
        exp_mod.w3_A[1] = 1e-4 * torch.randn(exp_mod.w3_A.shape[1], generator=gen)


class _FastMoERouterBlock(nn.Module):
    """Lightweight residual MoE block for 24-layer prefix-cache speedup parity testing."""

    def __init__(self, hidden_size: int, num_experts: int, layer_idx: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator(device="cpu").manual_seed(seed + layer_idx * 13)
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.expert_proj = nn.Parameter(
            0.12 * torch.randn((num_experts, hidden_size, hidden_size), generator=gen)
        )
        with torch.no_grad():
            self.router.weight.copy_(0.4 * torch.randn((num_experts, hidden_size), generator=gen))
        self.harm_bias = 0.08 if layer_idx >= 18 else (-0.06 if layer_idx < 3 else 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.router(x), dim=-1)
        mixed = torch.einsum("...e,ehd->...hd", probs, self.expert_proj)
        delta = torch.einsum("...h,...hd->...d", torch.tanh(x), mixed)
        return x + self.harm_bias * delta


def _build_pilot5_coadaptation_arms(
    *,
    compatible_router: bool = False,
    seed: int = 20261239,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    num_layers: int = 2,
    vocab_size: int = 48,
    num_experts: int = 8,
    lora_rank: int = 4,
    n_per_dom: int = 3,
) -> tuple[
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    CrowdV1WithEngram,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Build deterministic Base, Frozen-Router (Arm D), and Unfrozen (Arm B) MoE checkpoints.

    When ``compatible_router=False`` (PILOT5 C3 mismatch):
      - Arm B rotated its specialist representations and router rows onto an orthogonal
        subspace pair inside each specialist domain while Arm D kept the unaligned frozen
        router, so grafting Arm B's experts onto Arm D's router loses specialist routing
        and increases NLL relative to co-adapted ``WSP_C3_B_rescued``.
    When ``compatible_router=True``:
      - Arm D's router is aligned with Arm B's experts (small routing KL) and cleaner on
        general prose than Arm B's slightly drifted router.
    """
    block_h = hidden_size // 4
    block_v = vocab_size // 4
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    base_model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_fineweb", 0, 0, slice(0, block_h)),
        (1, "E1_python_systems_code", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (6, "E6_stellar_spectroscopy", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    axes_alt: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in base_model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.04 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(4, block_h, seed=seed + 100 + b_idx * 19)
            u_in, u_out, u_alt = basis[0], basis[1], basis[2]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            axes_alt[e_slot] = u_alt
            rows = _make_zero_mean_ortho_rows(min(6, block_h - 1), block_h, seed=seed + 200 + b_idx * 31)
            for t_i in range(block_v):
                vec = 0.20 * rows[t_i % int(rows.shape[0])] + 0.65 * u_in
                wte[v_off + t_i, h_slice] = vec - vec.mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            for tok in (v_off + 2, v_off + 3, v_off + 4):
                head_w[tok, h_slice] = 0.12 * u_in + 0.55 * u_out

        base_model.wte.weight.copy_(wte)
        base_model.head.weight.copy_(head_w)
        base_model.ln.weight.fill_(1.0)
        base_model.ln.bias.zero_()
        for blk in base_model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(0.35 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:hidden_size, :hidden_size].copy_(0.15 * torch.eye(hidden_size))

    upcycle_crowd_v1_to_sparse_moe(
        base_model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 61,
    )
    with torch.no_grad():
        for blk in base_model.blocks:
            blk.mlp.router.weight.zero_()
            for e_slot, _, _, _, h_slice in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 2.6 * axes_in[e_slot]

    frozen_arm = copy.deepcopy(base_model)
    unfrozen_arm = copy.deepcopy(base_model)

    with torch.no_grad():
        for l_idx in range(num_layers):
            # Specialist experts along subspaces disjoint from E0 (dims 0..block_h)
            i_base = block_h
            for e_slot, h_slice, i_slice, str_val in (
                (1, slice(block_h, 2 * block_h), slice(i_base, 2 * i_base), 5.8),
                (2, slice(2 * block_h, 3 * block_h), slice(2 * i_base, 3 * i_base), 6.4),
                (6, slice(3 * block_h, 4 * block_h), slice(3 * i_base, 4 * i_base), 7.2),
            ):
                exp_b = unfrozen_arm.blocks[l_idx].mlp.experts[str(e_slot)]
                f_in = torch.zeros(hidden_size, dtype=torch.float32)
                f_out = torch.zeros(hidden_size, dtype=torch.float32)
                f_in[h_slice] = axes_in[e_slot]
                f_out[h_slice] = axes_out[e_slot]
                _install_domain_lora_delta(
                    exp_b, f_in, f_out, strength=str_val, inter_slice=i_slice, seed=seed + 30 + e_slot + l_idx
                )
                if not compatible_router:
                    # Inject prose leakage into Arm B's specialist adapters (cleaned by P_0 in rescue)
                    exp_b.w1_A[1, :block_h] = 0.8 * axes_in[0]
                    exp_b.w1_B[:4, 1] = 1.4
                    exp_b.w2_A[1, :block_h] = 0.8 * axes_in[0]
                    exp_b.w2_B[:4, 1] = 1.4
                    exp_b.w3_B[:block_h, 0] = -0.35 * axes_out[0]

            if not compatible_router:
                # Arm B also suffers a pure prose-harming ghost sink on expert_3 (no specialist utility)
                sink_exp = unfrozen_arm.blocks[l_idx].mlp.experts["3"]
                sink_in = torch.zeros(hidden_size, dtype=torch.float32)
                sink_out = torch.zeros(hidden_size, dtype=torch.float32)
                sink_in[:block_h] = axes_in[0]
                sink_out[:block_h] = -axes_out[0]
                _install_domain_lora_delta(
                    sink_exp, sink_in, sink_out, strength=4.2, inter_slice=slice(0, block_h), seed=seed + 70 + l_idx
                )
                # Arm D frozen router misroutes specialist tokens away from (1, 2, 6) onto (4, 5, 7)
                r_frz = frozen_arm.blocks[l_idx].mlp.router.weight
                r_frz[1, block_h : 2 * block_h] = -3.0 * axes_in[1]
                r_frz[2, 2 * block_h : 3 * block_h] = -3.0 * axes_in[2]
                r_frz[6, 3 * block_h : 4 * block_h] = -3.0 * axes_in[6]
                r_frz[4, block_h : 2 * block_h] = 4.5 * axes_in[1]
                r_frz[5, 2 * block_h : 3 * block_h] = 4.5 * axes_in[2]
                r_frz[7, 3 * block_h : 4 * block_h] = 4.5 * axes_in[6]

                # Arm B unfrozen router co-adapted to route (1, 2, 6) strongly, plus prose drift onto (2, 3, 6)
                r_unf = unfrozen_arm.blocks[l_idx].mlp.router.weight
                r_unf[1, block_h : 2 * block_h] = 5.5 * axes_in[1]
                r_unf[2, 2 * block_h : 3 * block_h] = 5.5 * axes_in[2]
                r_unf[6, 3 * block_h : 4 * block_h] = 5.5 * axes_in[6]
                r_unf[2, :block_h] = 3.2 * axes_in[0]
                r_unf[3, :block_h] = 4.2 * axes_in[0]
                r_unf[6, :block_h] = 3.0 * axes_in[0]
            else:
                # Compatible routers: Arm D aligns closely with Arm B's specialist routing with stronger
                # confidence while Arm B has mild prose drift, so cross-arm graft achieves lower ANSE energy.
                r_frz = frozen_arm.blocks[l_idx].mlp.router.weight
                r_frz[1, block_h : 2 * block_h] = 4.8 * axes_in[1]
                r_frz[2, 2 * block_h : 3 * block_h] = 4.8 * axes_in[2]
                r_frz[6, 3 * block_h : 4 * block_h] = 4.8 * axes_in[6]

                r_unf = unfrozen_arm.blocks[l_idx].mlp.router.weight
                r_unf[1, block_h : 2 * block_h] = 4.0 * axes_in[1]
                r_unf[2, 2 * block_h : 3 * block_h] = 4.0 * axes_in[2]
                r_unf[6, 3 * block_h : 4 * block_h] = 4.0 * axes_in[6]
                r_unf[2, :block_h] = 0.65 * axes_in[0]
                r_unf[6, :block_h] = 0.65 * axes_in[0]

        if not compatible_router:
            unfrozen_arm.ln.weight.mul_(1.14)
        else:
            unfrozen_arm.ln.weight.mul_(1.04)

    def _make_chunks(seed_off: int, cnt: int) -> list[dict[str, Any]]:
        c_gen = torch.Generator(device="cpu").manual_seed(seed + 500 + seed_off)
        chunks: list[dict[str, Any]] = []
        seq_len = 12
        for e_slot, dom_name, _, v_off, _ in domain_specs:
            target_pool = (v_off + 2, v_off + 3, v_off + 4)
            for r_i in range(cnt):
                ids = torch.randint(v_off, v_off + block_v, (seq_len,), generator=c_gen).tolist()
                for pos in range(4, seq_len):
                    ids[pos] = target_pool[(pos + r_i + e_slot) % len(target_pool)]
                chunks.append({"expert_id": dom_name, "ids": ids})
        return chunks

    return base_model, frozen_arm, unfrozen_arm, _make_chunks(1, n_per_dom), _make_chunks(2, n_per_dom)


# ---------------------------------------------------------------------------
# S1: Exact Prefix-Cached Layer Attribution Parity & ~2x Block-Call Reduction
# ---------------------------------------------------------------------------


def test_s1_prefix_cached_attribution_exact_parity_and_2x_speedup() -> None:
    """S1: Prefix-cached layerwise attribution achieves <1e-6 parity and L + L(L+1)/2 calls vs L^2."""
    gen = torch.Generator(device="cpu").manual_seed(20261301)
    n_layers = 24
    hidden_size = 16
    blocks = nn.ModuleList([_FastMoERouterBlock(hidden_size, 4, idx, seed=20261302) for idx in range(n_layers)])
    embeddings = 0.5 * torch.randn((4, 6, hidden_size), generator=gen, dtype=torch.float32)
    target_dir = torch.ones(hidden_size, dtype=torch.float32) / math.sqrt(hidden_size)

    def _tail_nll(h_final: torch.Tensor) -> torch.Tensor:
        # Per-sample surrogate NLL vector
        return (h_final - target_dir).pow(2).mean(dim=(1, 2))

    # Naive O(L^2) full-forward attribution counting real block forward calls via hooks
    naive_counter = {"calls": 0}
    hooks = [blk.register_forward_hook(lambda _m, _i, _o: naive_counter.__setitem__("calls", naive_counter["calls"] + 1)) for blk in blocks]

    with torch.no_grad():
        xb_base = embeddings.clone()
        for blk in blocks:
            xb_base = blk(xb_base)
        naive_full_nll = float(_tail_nll(xb_base).mean().item())

        naive_ko_nlls: list[float] = []
        for l_idx in range(n_layers):
            x_cur = embeddings.clone()
            for k_idx, blk in enumerate(blocks):
                if k_idx == l_idx:
                    h_r = blk.router.register_forward_hook(lambda _m, _i, out: torch.zeros_like(out))
                    try:
                        x_cur = blk(x_cur)
                    finally:
                        h_r.remove()
                else:
                    x_cur = blk(x_cur)
            naive_ko_nlls.append(float(_tail_nll(x_cur).mean().item()))

    for h in hooks:
        h.remove()
    assert naive_counter["calls"] == n_layers + n_layers * n_layers

    # Prefix-cached attribution with hook call verification
    cached_counter = {"calls": 0}
    hooks_c = [blk.register_forward_hook(lambda _m, _i, _o: cached_counter.__setitem__("calls", cached_counter["calls"] + 1)) for blk in blocks]
    cached_res = compute_prefix_cached_layerwise_attribution(
        blocks,
        embeddings,
        _tail_nll,
        batch_size=24,
        is_general_mask=[True, True, False, False],
    )
    for h in hooks_c:
        h.remove()

    expected_calls = n_layers + (n_layers * (n_layers + 1)) // 2
    assert cached_counter["calls"] == expected_calls
    assert cached_res["total_block_calls"] == expected_calls
    assert cached_res["prefix_block_calls"] == n_layers
    assert cached_res["knockout_block_calls"] == (n_layers * (n_layers + 1)) // 2
    assert cached_res["naive_knockout_block_calls"] == n_layers * n_layers
    assert cached_res["naive_total_block_calls"] == n_layers + n_layers * n_layers
    assert cached_res["knockout_speedup_factor"] == pytest.approx((24 * 24) / 300.0)
    assert cached_res["speedup_factor"] > 1.85
    assert cached_res["block_evals_saved_ratio"] == pytest.approx((600 - 324) / 600.0)

    assert abs(cached_res["nll_mean"] - naive_full_nll) < 1e-6
    for row, naive_val in zip(cached_res["layers"], naive_ko_nlls, strict=True):
        assert abs(row["nll_knockout"] - naive_val) < 1e-6
        assert "general_harm_nats" in row and "spec_benefit_nats" in row

    # Verify CrowdV1WithEngram + CrowdLoRAMoEMLP, CrowdFineGrainedMoEMLP, knockout_blocks/fn, batching
    cfg_rt = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    m_lora = CrowdV1WithEngram(cfg_rt)
    upcycle_crowd_v1_to_sparse_moe(m_lora, num_experts=4, top_k=2, lora_rank=2, seed=17)
    emb_rt = torch.randn((4, 8, 16), generator=gen, dtype=torch.float32)
    res_lora = compute_prefix_cached_layerwise_attribution(
        m_lora.blocks,
        emb_rt,
        lambda h: {"nll_mean": float(h.pow(2).mean().item()), "general_nll": float(h[:2].pow(2).mean().item()), "spec_nll": float(h[2:].pow(2).mean().item())},
        batch_size=2,
        cos=m_lora.cos_cached,
        sin=m_lora.sin_cached,
    )
    assert res_lora["num_layers"] == 3
    assert res_lora["total_block_calls"] == 3 + 6

    m_fg = CrowdV1WithEngram(cfg_rt)
    upcycle_crowd_v1_to_sparse_moe(m_fg, split_mode="fine_grained_width_split", num_experts=4, top_k=2, seed=19)
    res_fg = compute_prefix_cached_layerwise_attribution(
        m_fg.blocks,
        emb_rt,
        lambda h: {"E0": float(h.pow(2).mean().item())},
        batch_size=4,
    )
    assert res_fg["total_block_calls"] == 9

    class _AttrWithKO(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Linear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp(x)

        def forward_knockout(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class _MlpWithRouter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.router = nn.Linear(8, 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.router(x).sum(dim=-1, keepdim=True)

    class _BlkMlpRouter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = _MlpWithRouter()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.mlp(x)

    class _BlkAttnNoHeadDim(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ln1 = nn.LayerNorm(8)
            self.attn = nn.Identity()
            self.mlp = nn.Linear(8, 8, bias=False)

        def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            return x + self.mlp(self.ln1(x)) * cos[..., :8].mean()

    blk_custom = [_AttrWithKO(), _BlkMlpRouter(), _BlkAttnNoHeadDim(), nn.Linear(8, 8)]
    emb_2d = torch.randn((5, 8), generator=gen)
    res_c = compute_prefix_cached_layerwise_attribution(
        blk_custom,
        emb_2d,
        lambda h: float(h.abs().mean().item()),
        batch_size=2,
        is_general_mask=torch.tensor([True, True, False, False, False]),
    )
    assert res_c["total_block_calls"] == 4 + 10
    res_kb = compute_prefix_cached_layerwise_attribution(
        [nn.Linear(8, 8), nn.Linear(8, 8)],
        emb_2d,
        lambda h: float(h.pow(2).mean().item()),
        batch_size=10,
        knockout_blocks=[nn.Identity(), nn.Identity()],
        is_general_mask=np.array([True, False, False, False, False]),
    )
    assert res_kb["num_layers"] == 2
    res_kf = compute_prefix_cached_layerwise_attribution(
        [nn.Linear(8, 8), nn.Linear(8, 8)],
        emb_2d,
        lambda h: torch.tensor(float(h.pow(2).mean().item())),
        batch_size=10,
        knockout_fn=lambda _b, x, _i: (x, None),  # type: ignore[arg-type,return-value]
    )
    assert res_kf["num_layers"] == 2


# ---------------------------------------------------------------------------
# S2: PILOT5 Co-Adaptation Mismatch Auto-Vetoed in Favor of Co-Adapted Rescue
# ---------------------------------------------------------------------------


def test_s2_pilot5_coadaptation_mismatch_vetoes_cross_arm_graft_for_rescued_arm() -> None:
    """S2: PILOT5 mismatch where cross-arm graft degrades NLL is vetoed in favor of co-adapted rescue."""
    # 1. Direct PILOT5 quantitative detector check (WSP_C3_B_rescued -0.2964 vs CrossArm +0.0080)
    gen = torch.Generator(device="cpu").manual_seed(20261303)
    router_d_logits = torch.tensor(
        [[4.0, -3.0, -3.0, 0.2], [3.5, -2.5, -2.5, 0.1], [3.8, -2.8, -2.8, 0.0]], dtype=torch.float32
    )
    router_b_logits = torch.tensor(
        [[-3.0, 4.0, 0.5, -1.0], [-2.5, 0.2, 4.2, -1.0], [-2.8, 3.9, 0.3, -1.0]], dtype=torch.float32
    )
    expert_y_3d = torch.randn((3, 4, 12), generator=gen, dtype=torch.float32)
    expert_y_3d[:, 1, :] *= 4.0

    det_p5 = detect_router_expert_coadaptation_mismatch(
        {"router_arm": router_d_logits, "expert_arm": router_b_logits},
        {"per_expert_outputs": expert_y_3d},
        coadapted_nll=-0.296413,
        grafted_nll=0.008030,
        max_mismatch_nats=0.02,
    )
    assert det_p5["coadaptation_mismatch_nats"] == pytest.approx(0.304443)
    assert det_p5["routing_kl_divergence"] > 1.0
    assert det_p5["nll_mismatch_exceeded"] is True
    assert det_p5["routing_kl_exceeded"] is True
    assert det_p5["mismatch_detected"] is True
    assert det_p5["veto_cross_arm_graft"] is True
    assert det_p5["recommended_action"] == "veto_cross_arm_graft_use_coadapted_rescue"

    # Also exercise 3D tensor auto-detection & tuple router_arm_logits (plus norm-affinity KL without explicit logits)
    det_3d = detect_router_expert_coadaptation_mismatch(
        (router_d_logits, router_b_logits),
        expert_y_3d,
        coadapted_nll=1.5,
        grafted_nll=1.8,
        max_mismatch_nats=0.02,
    )
    assert det_3d["output_l2_mismatch"] > 0.0
    assert det_3d["mismatch_detected"] is True
    det_norm_kl = detect_router_expert_coadaptation_mismatch(
        router_d_logits,
        {"0": expert_y_3d},
        coadapted_nll=1.5,
        grafted_nll=1.6,
    )
    assert det_norm_kl["kl_source"] == "expert_output_norm_affinity"
    det_map_logits = detect_router_expert_coadaptation_mismatch(
        router_d_logits,
        {"0": router_b_logits},
        coadapted_nll=1.5,
        grafted_nll=1.6,
    )
    assert det_map_logits["kl_source"] == "explicit_router_logits"
    det_map_rk = detect_router_expert_coadaptation_mismatch(
        {"router_arm": router_d_logits},
        {"router_logits": router_b_logits, "outputs": expert_y_3d},
        coadapted_nll=1.5,
        grafted_nll=1.6,
    )
    assert det_map_rk["kl_source"] == "explicit_router_logits"

    # 2. End-to-end CrowdV1WithEngram + CrowdLoRAMoEMLP PILOT5 co-adaptation surgery
    base_model, frozen_arm, unfrozen_arm, probe_chunks, eval_chunks = _build_pilot5_coadaptation_arms(
        compatible_router=False
    )
    cfg = CoAdaptationGuardedSurgeryConfig(
        max_mismatch_nats=0.02,
        max_routing_kl=0.25,
        score_from=4,
        batch_size=8,
        inplace=False,
    )
    result_model, receipt = execute_coadaptation_guarded_surgery(
        base_model,
        frozen_arm,
        unfrozen_arm,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )

    assert receipt["object"] == COADAPTATION_RECEIPT_OBJECT
    assert receipt["schema"] == COADAPTATION_SCHEMA
    assert receipt["cross_arm_vetoed"] is True
    assert receipt["veto_reason"] == "router_expert_coadaptation_mismatch"
    assert receipt["selected_strategy"] == "coadapted_rescue"
    assert receipt["status"] == "COADAPTED_RESCUE_COMMITTED_CROSS_ARM_VETOED"
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["coadaptation_mismatch_nats"] > 0.02
    assert receipt["routing_kl_divergence"] > 0.25
    assert receipt["anse_energy"] < 0.0
    assert receipt["delta_e_vs_base"] < 0.0
    assert receipt["final_general_drift_ratio"] <= 1.01
    assert receipt["prefix_attribution_summary"]["executed"] is True
    assert receipt["block_evals_saved_ratio"] > 0.0
    assert verify_coadaptation_receipt_provenance(receipt) is True
    assert compute_checkpoint_sha256(extract_checkpoint_blocks(result_model)) == receipt["final_state_sha256"]


# ---------------------------------------------------------------------------
# S3: Compatible Co-Adapted Graft Acceptance (Small Routing KL & Delta E < 0)
# ---------------------------------------------------------------------------


def test_s3_compatible_coadapted_graft_accepted_when_routing_kl_small_and_delta_e_negative() -> None:
    """S3: Compatible cross-arm graft is accepted when routing KL is small and ANSE Delta E < 0."""
    # 1. Direct detector check for compatible sigma_topk routing and small mismatch
    r_host = torch.tensor([[3.0, -1.0, 0.2, -2.0], [-1.0, 3.0, -2.0, 0.2]], dtype=torch.float32)
    r_exp = torch.tensor([[2.95, -0.95, 0.22, -1.95], [-0.95, 2.95, -1.95, 0.22]], dtype=torch.float32)
    y_coad = torch.tensor([[1.0, 0.5, -0.2, 0.3], [0.4, 1.1, 0.1, -0.3]], dtype=torch.float32)
    y_graft = torch.tensor([[0.99, 0.51, -0.19, 0.30], [0.41, 1.09, 0.10, -0.29]], dtype=torch.float32)

    det_ok = detect_router_expert_coadaptation_mismatch(
        {"0": r_host},
        {"outputs": {"0": y_coad}, "grafted_outputs": {"0": y_graft}},
        coadapted_nll=1.500,
        grafted_nll=1.504,
        max_mismatch_nats=0.02,
        max_routing_kl=0.25,
        routing_mode="sigma_topk",
        expert_arm_router_logits={"0": r_exp},
    )
    assert det_ok["routing_kl_divergence"] < 0.01
    assert det_ok["output_cosine_similarity"] > 0.999
    assert det_ok["mismatch_detected"] is False
    assert det_ok["veto_cross_arm_graft"] is False
    assert det_ok["recommended_action"] == "accept_compatible_coadaptation"

    # 2. End-to-end CrowdV1WithEngram compatible cross-arm graft acceptance
    base_model, frozen_arm, unfrozen_arm, probe_chunks, eval_chunks = _build_pilot5_coadaptation_arms(
        compatible_router=True,
        seed=20261304,
    )
    cfg = CoAdaptationGuardedSurgeryConfig(
        max_mismatch_nats=0.02,
        max_routing_kl=0.25,
        cross_arm_gate_blend_alpha=0.10,
        oof_router_beta=0.0,
        score_from=4,
        batch_size=8,
        inplace=False,
    )
    _, receipt = execute_coadaptation_guarded_surgery(
        base_model,
        frozen_arm,
        unfrozen_arm,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )
    assert receipt["cross_arm_vetoed"] is False
    assert receipt["veto_reason"] is None
    assert receipt["selected_strategy"] == "cross_arm_graft"
    assert receipt["status"] == "CROSS_ARM_GRAFT_COMMITTED"
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["routing_kl_divergence"] <= 0.25
    assert receipt["coadaptation_mismatch_nats"] <= 0.02
    assert receipt["anse_energy"] < 0.0
    assert receipt["final_general_drift_ratio"] <= 1.01
    assert verify_coadaptation_receipt_provenance(receipt) is True


# ---------------------------------------------------------------------------
# S4: ANSE Maximum Pain Atomic Bitwise Rollback on Prose Guard Violation
# ---------------------------------------------------------------------------


def test_s4_anse_maximum_pain_atomic_bitwise_rollback_on_guard_violation() -> None:
    """S4: Prose guard breach (>+1.0%) or non-finite eval enforces ANSE Maximum Pain atomic bitwise rollback."""
    base_model, frozen_arm, unfrozen_arm, probe_chunks, eval_chunks = _build_pilot5_coadaptation_arms(
        compatible_router=False,
        seed=20261305,
    )
    target_model = copy.deepcopy(base_model)
    base_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(base_model))

    def _prose_violating_eval(m: CrowdV1WithEngram) -> dict[str, float]:
        cur_sha = compute_checkpoint_sha256(extract_checkpoint_blocks(m))
        if cur_sha == base_sha:
            return {"E0_general_fineweb": 2.00, "E1_code": 2.40, "E2_math": 2.40}
        # All edited arms violate E0 <= +1.0% (2.06 / 2.00 = +3.0%)
        return {"E0_general_fineweb": 2.06, "E1_code": 1.90, "E2_math": 1.90}

    out_model, rec_rb = execute_coadaptation_guarded_surgery(
        base_model,
        frozen_arm,
        unfrozen_arm,
        target_model=target_model,
        probe_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        eval_fn=_prose_violating_eval,
        config=CoAdaptationGuardedSurgeryConfig(score_from=4, batch_size=8, rollback_target="base", inplace=True),
    )
    assert out_model is target_model
    assert rec_rb["accepted"] is False
    assert rec_rb["rolled_back"] is True
    assert rec_rb["rollback_target"] == "base"
    assert rec_rb["rollback_bitwise_verified"] is True
    assert rec_rb["status"] == "ROLLED_BACK_ANSE_MAXIMUM_PAIN"
    assert rec_rb["candidate_summaries"]["coadapted_rescue"]["anse_energy"] >= ANSE_MAXIMUM_PAIN
    assert rec_rb["final_state_sha256"] == base_sha
    assert compute_checkpoint_sha256(extract_checkpoint_blocks(target_model)) == base_sha
    assert verify_coadaptation_receipt_provenance(rec_rb) is True

    # Dict-block mode (no nn.Module) testing frozen_arm & unfrozen_arm rollback + non-finite + LoRA B!=0 + SwiGLU dense P_0
    gen = torch.Generator(device="cpu").manual_seed(20261306)
    base_blk = {
        "blocks.0.mlp.router": {"weight": torch.eye(4, 6)[:2], "bias": torch.zeros(2)},
        "blocks.0.mlp.experts.0": {
            "w1_A": torch.ones((2, 6)),
            "w1_B": torch.zeros((4, 2)),
            "w2_A": torch.ones((2, 6)),
            "w2_B": torch.zeros((4, 2)),
            "w3_A": torch.ones((2, 4)),
            "w3_B": torch.zeros((6, 2)),
        },
        "blocks.0.mlp.experts.1": {
            "w1_A": torch.ones((2, 6)),
            "w1_B": torch.full((4, 2), 0.1),
            "w2_A": torch.ones((2, 6)),
            "w2_B": torch.full((4, 2), 0.1),
            "w3_A": torch.ones((2, 4)),
            "w3_B": torch.full((6, 2), 0.1),
        },
        "blocks.0.mlp.experts.2": {
            "w1": torch.zeros((4, 6)),
            "w2": torch.zeros((4, 6)),
            "w3": torch.zeros((6, 4)),
        },
        "final_ln": {"weight": torch.ones(6), "bias": torch.zeros(6)},
    }
    frz_blk = copy.deepcopy(base_blk)
    frz_blk["blocks.0.mlp.experts.1"]["w1_B"].fill_(0.05)
    unf_blk = copy.deepcopy(base_blk)
    unf_blk["blocks.0.mlp.router"]["weight"].add_(0.2 * torch.randn((2, 6), generator=gen))
    unf_blk["blocks.0.mlp.experts.0"]["w1_B"].fill_(0.5)
    unf_blk["blocks.0.mlp.experts.1"]["w1_B"].fill_(0.4)
    unf_blk["blocks.0.mlp.experts.1"]["w3_B"].fill_(0.4)
    unf_blk["blocks.0.mlp.experts.2"]["w1"].fill_(0.3)
    unf_blk["blocks.0.mlp.experts.2"]["w3"].fill_(0.2)
    p_x = torch.randn((6, 6), generator=gen)
    s_x = torch.randn((6, 6), generator=gen)
    p_null_x = torch.eye(6, dtype=torch.float64)
    p_null_x[0, 0] = 0.0
    base_blk_sha = compute_checkpoint_sha256(base_blk)

    for rb_target, exp_key, p_arg in (
        ("frozen_arm", "frozen_router_state_sha256", p_null_x),
        ("unfrozen_arm", "unfrozen_arm_state_sha256", {"0": p_null_x}),
    ):
        blocks_out, rec_dict = execute_coadaptation_guarded_surgery(
            base_blk,
            frz_blk,
            unf_blk,
            p_null_by_expert=p_arg,
            prose_hidden_states=p_x,
            spec_hidden_states=s_x,
            sink_experts=(0,),
            anchor_layers=(),
            expert_arm_outputs=torch.ones((12, 2)),
            eval_fn=lambda b: (
                {"E0_gen": 2.0, "E1_spec": 2.0}
                if compute_checkpoint_sha256(b) == base_blk_sha
                else {"non_finite_outputs": True}
            ),
            config=CoAdaptationGuardedSurgeryConfig(
                fold_ln_lambda=0.5,
                fold_tau_temp=1.02,
                rollback_target=rb_target,
                inplace=False,
            ),
            inplace=False,
        )
        assert isinstance(blocks_out, dict)
        assert rec_dict["rolled_back"] is True
        assert rec_dict["rollback_bitwise_verified"] is True
        assert rec_dict["final_state_sha256"] == rec_dict[exp_key]
        assert verify_coadaptation_receipt_provenance(rec_dict) is True

    # Also test biasless final_ln, probe_chunks without eval_chunks, and base/frozen nn.Module branches
    cfg_rt_nb = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    m_nb = CrowdV1WithEngram(cfg_rt_nb)
    upcycle_crowd_v1_to_sparse_moe(m_nb, num_experts=4, top_k=2, lora_rank=2, seed=23)
    m_nb.ln.bias = None  # type: ignore[assignment]
    nb_blocks = extract_checkpoint_blocks(m_nb)
    nb_chunks = [
        {"expert_id": "E0_gen", "ids": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"expert_id": "E0_gen", "ids": [1, 2, 3, 4, 5, 6, 7, 8]},
        {"expert_id": "E1_code", "ids": [8, 9, 10, 11, 12, 13, 14, 15]},
        {"expert_id": "E1_code", "ids": [9, 10, 11, 12, 13, 14, 15, 16]},
    ]
    _, rec_base_m = execute_coadaptation_guarded_surgery(
        copy.deepcopy(m_nb),
        nb_blocks,
        nb_blocks,
        probe_chunks=nb_chunks,
        config=CoAdaptationGuardedSurgeryConfig(score_from=2, batch_size=4, inplace=False),
    )
    assert verify_coadaptation_receipt_provenance(rec_base_m) is True
    _, rec_frz_m = execute_coadaptation_guarded_surgery(
        nb_blocks,
        copy.deepcopy(m_nb),
        nb_blocks,
        probe_chunks=nb_chunks,
        config=CoAdaptationGuardedSurgeryConfig(score_from=2, batch_size=4, inplace=False),
    )
    assert verify_coadaptation_receipt_provenance(rec_frz_m) is True


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / ZeroStub / OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_owner_guardrail_gates() -> None:
    """S5: Cover config/API validation branches and AGPL/ZeroStub/PRSecurity/Lateral/OwnerGuardrail gates."""
    # 1. Config & receipt validation
    with pytest.raises(ValueError, match="anchor_harm_threshold_nats"):
        CoAdaptationGuardedSurgeryConfig(anchor_harm_threshold_nats=0.05)
    with pytest.raises(ValueError, match="routing_mode"):
        CoAdaptationGuardedSurgeryConfig(routing_mode="invalid")
    with pytest.raises(ValueError, match="rollback_target"):
        CoAdaptationGuardedSurgeryConfig(rollback_target="invalid")
    with pytest.raises(ValueError, match="nullspace_eig_threshold"):
        CoAdaptationGuardedSurgeryConfig(nullspace_eig_threshold=1.0)
    with pytest.raises(ValueError, match="Drift ratio bounds"):
        CoAdaptationGuardedSurgeryConfig(max_general_drift_ratio=0.95)
    with pytest.raises(ValueError, match="general_prefix"):
        CoAdaptationGuardedSurgeryConfig(general_prefix=" ")
    with pytest.raises(ValueError, match="lambda_grid"):
        CoAdaptationGuardedSurgeryConfig(lambda_grid=())
    with pytest.raises(ValueError, match="<= 1.0"):
        CoAdaptationGuardedSurgeryConfig(lambda_grid=(1.5,))
    with pytest.raises(ValueError, match="> 0.0"):
        CoAdaptationGuardedSurgeryConfig(nullspace_variance_threshold=0.0)
    with pytest.raises(TypeError, match="bool"):
        CoAdaptationGuardedSurgeryConfig(auto_detect_sinks="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        CoAdaptationGuardedSurgeryConfig(batch_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        CoAdaptationGuardedSurgeryConfig(batch_size=0)
    with pytest.raises(ValueError, match=">= 0"):
        CoAdaptationGuardedSurgeryConfig(score_from=-1)
    with pytest.raises(TypeError, match="real number"):
        CoAdaptationGuardedSurgeryConfig(max_mismatch_nats="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        CoAdaptationGuardedSurgeryConfig(max_routing_kl=float("nan"))
    with pytest.raises(ValueError, match=">= 0.0"):
        CoAdaptationGuardedSurgeryConfig(max_mismatch_nats=-0.1)
    with pytest.raises(ValueError, match="> 0.0"):
        CoAdaptationGuardedSurgeryConfig(fold_tau_temp=0.0)
    assert verify_coadaptation_receipt_provenance({"receipt_sha256": "0" * 64, "object": "bad"}) is False
    assert verify_coadaptation_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert (
        verify_coadaptation_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": COADAPTATION_RECEIPT_OBJECT,
                "schema": COADAPTATION_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )
    with pytest.raises(TypeError, match="mapping"):
        verify_coadaptation_receipt_provenance("bad")  # type: ignore[arg-type]

    # 2. Attribution & detector input validation
    b_lin = [nn.Linear(4, 4)]
    x_ok = torch.ones((4, 4))
    with pytest.raises(ValueError, match="non-empty sequence"):
        compute_prefix_cached_layerwise_attribution([], x_ok, lambda h: 0.0)
    with pytest.raises(TypeError, match="callable"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, "bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="anchor_harm_threshold_nats"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, anchor_harm_threshold_nats=0.1)
    with pytest.raises(ValueError, match="knockout_blocks"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, knockout_blocks=[])
    with pytest.raises(TypeError, match="knockout_fn"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, knockout_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cos and sin"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, cos=x_ok)
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_prefix_cached_layerwise_attribution(b_lin, [[1.0, 2.0]], lambda h: 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ndim >= 2"):
        compute_prefix_cached_layerwise_attribution(b_lin, torch.ones(4), lambda h: 0.0)
    with pytest.raises(ValueError, match="non-empty"):
        compute_prefix_cached_layerwise_attribution(b_lin, torch.empty((0, 4)), lambda h: 0.0)
    with pytest.raises(TypeError, match="floating-point"):
        compute_prefix_cached_layerwise_attribution(b_lin, torch.ones((4, 4), dtype=torch.long), lambda h: 0.0)
    with pytest.raises(ValueError, match="finite"):
        compute_prefix_cached_layerwise_attribution(b_lin, torch.tensor([[float("nan"), 1.0], [1.0, 1.0]]), lambda h: 0.0)
    with pytest.raises(ValueError, match="1D bool"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, is_general_mask=torch.ones(4, dtype=torch.long))
    with pytest.raises(ValueError, match="1D bool"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, is_general_mask=np.ones(4, dtype=np.int64))
    with pytest.raises(ValueError, match="is_general_mask"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, is_general_mask=[False, False, False, False])
    with pytest.raises(TypeError, match="is_general_mask"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: 0.0, is_general_mask="bad")
    with pytest.raises(ValueError, match="empty"):
        compute_prefix_cached_layerwise_attribution(b_lin, x_ok, lambda h: {})
    with pytest.raises(TypeError, match="Block output"):
        compute_prefix_cached_layerwise_attribution([lambda _x: "bad"], x_ok, lambda h: 0.0)

    r_ok = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    y_ok = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
    res_single = detect_router_expert_coadaptation_mismatch(r_ok, y_ok, 1.0, 1.005)
    assert res_single["kl_source"] == "single_router_only" and res_single["mismatch_detected"] is False
    with pytest.raises(ValueError, match="routing_mode"):
        detect_router_expert_coadaptation_mismatch(r_ok, y_ok, 1.0, 1.0, routing_mode="bad")
    with pytest.raises(ValueError, match=">= 2 experts"):
        detect_router_expert_coadaptation_mismatch(torch.ones((2, 1)), y_ok, 1.0, 1.0)
    with pytest.raises(ValueError, match="width mismatch"):
        detect_router_expert_coadaptation_mismatch({"0": r_ok, "1": torch.ones((2, 3))}, y_ok, 1.0, 1.0)
    with pytest.raises(TypeError, match="Tensor or Mapping"):
        detect_router_expert_coadaptation_mismatch("bad", y_ok, 1.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="length 2"):
        detect_router_expert_coadaptation_mismatch((r_ok,), y_ok, 1.0, 1.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shape"):
        detect_router_expert_coadaptation_mismatch(
            r_ok, y_ok, 1.0, 1.0, expert_arm_router_logits=torch.ones((3, 2))
        )
    with pytest.raises(ValueError, match="row count"):
        detect_router_expert_coadaptation_mismatch(r_ok, torch.ones((3, 2, 4)), 1.0, 1.0)
    with pytest.raises(ValueError, match="expert axis"):
        detect_router_expert_coadaptation_mismatch(r_ok, {"per_expert_outputs": torch.ones((2, 5, 4))}, 1.0, 1.0)
    with pytest.raises(TypeError, match="3D\\+ Tensor"):
        detect_router_expert_coadaptation_mismatch(r_ok, {"per_expert_outputs": "bad"}, 1.0, 1.0)
    with pytest.raises(ValueError, match="shape"):
        detect_router_expert_coadaptation_mismatch(r_ok, y_ok, 1.0, 1.0, grafted_outputs=torch.ones((2, 4)))
    with pytest.raises(TypeError, match="expert_arm_outputs"):
        detect_router_expert_coadaptation_mismatch(r_ok, "bad", 1.0, 1.0)  # type: ignore[arg-type]

    # Surgery input & layout validation
    min_blk = {
        "blocks.0.mlp.router": {"weight": torch.ones((2, 4))},
        "blocks.0.mlp.experts.0": {"w1": torch.ones((4, 4))},
    }
    with pytest.raises(ValueError, match="eval_fn"):
        execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk)
    with pytest.raises(TypeError, match="config"):
        execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk, config="bad", eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eval_fn"):
        execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="target_model"):
        execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk, target_model="bad", eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="non-negative integers"):
        execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk, protected_experts="bad", eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="MoE '.router' and '.experts.'"):
        execute_coadaptation_guarded_surgery({"x": {"w": torch.ones(2)}}, min_blk, min_blk, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})
    for bad_blk, exc_t in (
        ("bad", TypeError),
        ({"": {"w": torch.ones(2)}}, ValueError),
        ({"b": {}}, TypeError),
        ({"b": {"w": torch.ones(2, dtype=torch.long)}}, ValueError),
        ({"blocks.0.mlp.router": {"weight": torch.tensor([[float("nan"), 0.0], [0.0, 1.0]])}, "blocks.0.mlp.experts.0": {"w": torch.ones(2)}}, ValueError),
    ):
        with pytest.raises(exc_t):
            execute_coadaptation_guarded_surgery(bad_blk, min_blk, min_blk, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block layout"):
        execute_coadaptation_guarded_surgery(min_blk, {"blocks.0.mlp.router": {"weight": torch.ones((2, 4))}}, min_blk, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})
    with pytest.raises(ValueError, match="Sub-key mismatch"):
        execute_coadaptation_guarded_surgery(min_blk, {"blocks.0.mlp.router": {"bias": torch.ones((2, 4))}, "blocks.0.mlp.experts.0": {"w1": torch.ones((4, 4))}}, min_blk, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})
    with pytest.raises(ValueError, match="shape mismatch"):
        execute_coadaptation_guarded_surgery(min_blk, {"blocks.0.mlp.router": {"weight": torch.ones((2, 5))}, "blocks.0.mlp.experts.0": {"w1": torch.ones((4, 4))}}, min_blk, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})
    for bad_chunks in (
        [],
        [{"ids": [0, 1]}],
        [{"expert_id": " ", "ids": [0, 1]}],
        [{"expert_id": "E0", "ids": [[0, 1]]}],
        [{"expert_id": "E0", "ids": [-1, 1]}],
        [{"expert_id": "E0", "ids": [0, 1, 2]}, {"expert_id": "E1", "ids": [0, 1]}],
        [{"expert_id": "E0", "ids": [0, 1, 2]}, {"expert_id": "E1", "ids": [0, 1, 2]}],
        [{"expert_id": "E0", "ids": [0, 1, 2, 3]}, {"expert_id": "E0", "ids": [0, 1, 2, 3]}],
    ):
        with pytest.raises(ValueError):
            execute_coadaptation_guarded_surgery(
                min_blk, min_blk, min_blk, probe_chunks=bad_chunks, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0}, config=CoAdaptationGuardedSurgeryConfig(score_from=3)
            )
    for bad_p, exc_p in (
        ("bad", TypeError),
        (torch.ones((2, 3)), ValueError),
        (torch.ones((5, 5)), ValueError),
        (torch.tensor([[1.0, 2.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]), ValueError),
    ):
        with pytest.raises(exc_p):
            execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk, p_null_by_expert=bad_p, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0})  # type: ignore[arg-type]
    # Also exercise {"other": ...} and {"0": "bad"} projector fallback, and bad eval_fn return shapes
    execute_coadaptation_guarded_surgery(
        min_blk, min_blk, min_blk, p_null_by_expert={"other": torch.eye(4)}, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0}
    )
    execute_coadaptation_guarded_surgery(
        min_blk, min_blk, min_blk, p_null_by_expert={"0": "bad"}, eval_fn=lambda _: {"E0_gen": 1.0, "E1_code": 1.0}
    )
    for bad_ev, exc_ev in (
        ({}, ValueError),
        ({"E0_gen": "bad", "E1_code": 1.0}, TypeError),
        ({"E0_only": 1.0}, ValueError),
        ({"E0_gen": float("nan"), "E1_code": 1.0}, ValueError),
    ):
        with pytest.raises(exc_ev):
            execute_coadaptation_guarded_surgery(min_blk, min_blk, min_blk, eval_fn=lambda _, r=bad_ev: r)

    # 3. AGPL-3.0 header, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn39", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path(
        "pr_gatekeeper_dyn39", REPO_ROOT / "scripts" / "pr_gatekeeper.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/prefix_cached_causal_attribution.py", TARGET_MODULE_PATH),
        ("tests/test_prefix_cached_causal_attribution_scenarios.py", TEST_FILE_PATH),
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
            "nll_mean": 1.62,
            "coadaptation_mismatch_nats": 0.3044,
            "block_evals_saved_ratio": 0.46,
            "anse_energy": -0.2964,
        },
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): prefix-cached causal MoE attribution and router-expert co-adaptation detector"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-39 prefix-cached layerwise MoE causal attribution "
        "and router-expert co-adaptation mismatch detector with ANSE Delta E < 0 rollback."
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
        pr_number=1239,
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
    assert math.isfinite(float(len(telemetry_json)))
