# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-45: Optimal-Transport Hungarian & Procrustes cross-arm MoE fuser."""

from __future__ import annotations

import ast
import copy
import importlib.util
import itertools
import json
import math
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
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.optimal_transport_moe_arm_fuser import (
    ANSE_MAXIMUM_PAIN,
    OT_MOE_FUSER_SCHEMA,
    OT_MOE_FUSION_RECEIPT_OBJECT,
    OptimalTransportMoEFuserConfig,
    align_lora_rank_basis_procrustes,
    compute_cross_arm_expert_similarity_matrix,
    compute_state_sha256,
    execute_verified_ot_moe_arm_fusion,
    fuse_aligned_moe_layer_weights,
    match_moe_experts_hungarian_procrustes,
    solve_exact_hungarian_assignment,
    verify_ot_moe_fusion_receipt_provenance,
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "optimal_transport_moe_arm_fuser.py"
)
TEST_FILE_PATH = Path(__file__).resolve()
ROADMAP_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_ROADMAP.md"
PLAN_DOC_PATH = REPO_ROOT / "docs" / "NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md"
EXPECTED_AGPL_HEADER = (
    "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
)


@pytest.fixture(autouse=True, scope="module")
def _single_thread_cpu_torch() -> Any:
    """Eliminate OpenMP thread-pool overhead on small deterministic CPU tensors."""
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


def _make_zero_mean_rotation(dim: int, seed: int) -> torch.Tensor:
    """Construct orthogonal matrix ``Q`` preserving all-ones direction (LayerNorm equivariant)."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    v_ones = torch.ones(dim, dtype=torch.float64) / math.sqrt(float(dim))
    b_raw = torch.cat(
        [v_ones.unsqueeze(1), torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    b_orth, _ = torch.linalg.qr(b_raw, mode="reduced")
    if float((b_orth[:, 0] @ v_ones).item()) < 0.0:
        b_orth[:, 0] = -b_orth[:, 0]
    q_sub, _ = torch.linalg.qr(torch.randn((dim - 1, dim - 1), generator=gen, dtype=torch.float64))
    return (b_orth @ torch.block_diag(torch.ones((1, 1), dtype=torch.float64), q_sub) @ b_orth.T).to(
        torch.float32
    )


def _install_domain_lora(
    exp_mod: Any,
    in_dir: torch.Tensor,
    out_dir: torch.Tensor,
    *,
    strength: float,
    inter_slice: slice,
    gate_scale: float = 0.14,
) -> None:
    """Write deterministic rank-1 SwiGLU LoRA mapping ``in_dir -> strength * out_dir``."""
    i_start = inter_slice.start
    i_width = inter_slice.stop - inter_slice.start
    rank = int(exp_mod.w1_A.shape[0])
    n_act = min(rank, i_width)
    scaling = float(exp_mod.scaling)
    b_scale = math.sqrt(abs(strength)) / max(math.sqrt(scaling), 1e-6)
    sign_b = 1.0 if strength >= 0.0 else -1.0

    for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
        getattr(exp_mod, attr).zero_()

    exp_mod.w1_A[0] = in_dir
    exp_mod.w2_A[0] = in_dir
    exp_mod.w1_B[i_start : i_start + n_act, 0] = gate_scale
    exp_mod.w2_B[i_start : i_start + n_act, 0] = gate_scale
    exp_mod.w3_A[0, i_start : i_start + n_act] = 1.0 / float(n_act)
    exp_mod.w3_B[:, 0] = sign_b * b_scale * out_dir


def _permute_and_rotate_crowd_lora_model(
    model: CrowdV1WithEngram,
    perm: Sequence[int],
    rot_q: torch.Tensor | None,
) -> CrowdV1WithEngram:
    """Create a permuted + optionally rotated clone ``Arm_B`` where slot ``perm[i]`` holds ``i``."""
    cloned = copy.deepcopy(model)
    n_e = len(perm)
    inv_perm = [0] * n_e
    for i, b_slot in enumerate(perm):
        inv_perm[b_slot] = i

    q32 = rot_q.to(torch.float32) if rot_q is not None else None
    with torch.no_grad():
        if q32 is not None:
            cloned.wte.weight.copy_(cloned.wte.weight @ q32)
            cloned.head.weight.copy_(cloned.head.weight @ q32)
        for blk in cloned.blocks:
            mlp = blk.mlp
            if q32 is not None:
                blk.attn.q.weight.copy_(blk.attn.q.weight @ q32)
                blk.attn.k.weight.copy_(blk.attn.k.weight @ q32)
                blk.attn.v.weight.copy_(blk.attn.v.weight @ q32)
                blk.attn.o.weight.copy_(q32.T @ blk.attn.o.weight)
                mlp.base_mlp.w1.weight.copy_(mlp.base_mlp.w1.weight @ q32)
                mlp.base_mlp.w2.weight.copy_(mlp.base_mlp.w2.weight @ q32)
                mlp.base_mlp.w3.weight.copy_(q32.T @ mlp.base_mlp.w3.weight)

            # Permute router rows so slot j = perm[i] gets row i (and rotate input dim by Q)
            r_old = mlp.router.weight.detach().clone()
            r_perm = r_old[inv_perm]
            mlp.router.weight.copy_((r_perm @ q32) if q32 is not None else r_perm)

            old_experts = {
                e: {
                    attr: getattr(mlp.experts[str(e)], attr).detach().clone()
                    for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B")
                }
                for e in range(n_e)
            }
            for dst_b in range(n_e):
                src_a = inv_perm[dst_b]
                exp_dst = mlp.experts[str(dst_b)]
                for attr, val in old_experts[src_a].items():
                    t_new = val.clone()
                    if q32 is not None:
                        if attr in ("w1_A", "w2_A"):
                            t_new = t_new @ q32
                        elif attr == "w3_B":
                            t_new = q32.T @ t_new
                    getattr(exp_dst, attr).copy_(t_new)
    return cloned


def _build_pilot5_cross_arm_fixture(
    *,
    seed: int = 20264500,
    hidden_size: int = 48,
    intermediate_size: int = 64,
    num_layers: int = 2,
    vocab_size: int = 64,
    num_experts: int = 8,
    lora_rank: int = 4,
    perm: Sequence[int] = (3, 6, 0, 7, 1, 5, 2, 4),
    rotate_arm_b: bool = True,
) -> dict[str, Any]:
    """Build deterministic base + Arm_A (code) + permuted/rotated Arm_B (math/lean4 + leak)."""
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
        max_position_embeddings=24,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    base_model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_fineweb", 0, 0, slice(0, block_h)),
        (1, "E1_python_systems_code", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (7, "E7_formal_lean4", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
    )

    extra_basis = _make_zero_mean_ortho_rows(4, hidden_size, seed=seed + 701)
    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in base_model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.02 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(3, block_h, seed=seed + 50 + b_idx * 17)
            u_in, u_out = basis[0], basis[1]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            rows = _make_zero_mean_ortho_rows(block_h - 1, block_h, seed=seed + 180 + b_idx * 29)
            cross_mix = 0.08 * _make_zero_mean_ortho_rows(block_v, hidden_size, seed=seed + 280 + b_idx * 31)
            for t_i in range(block_v):
                vec = 0.25 * rows[t_i % int(rows.shape[0])] + 0.65 * u_in
                wte[v_off + t_i] = cross_mix[t_i] + 0.14 * extra_basis[t_i % 4]
                wte[v_off + t_i, h_slice] = wte[v_off + t_i, h_slice] + (vec - vec.mean())
                wte[v_off + t_i] = wte[v_off + t_i] - wte[v_off + t_i].mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            for tok in (v_off + 2, v_off + 3, v_off + 4):
                head_w[tok, h_slice] = 0.05 * u_in + 0.95 * u_out

        base_model.wte.weight.copy_(wte)
        base_model.head.weight.copy_(head_w)
        base_model.ln.weight.fill_(1.45)
        base_model.ln.bias.zero_()
        for blk in base_model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.mlp.w1.weight[:12, :12].copy_(0.20 * torch.eye(12))
            blk.mlp.w2.weight[:12, :12].copy_(0.20 * torch.eye(12))
            blk.mlp.w3.weight[:12, :12].copy_(0.05 * torch.eye(12))

    upcycle_crowd_v1_to_sparse_moe(
        base_model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 77,
    )

    def _full_vec(h_slice: slice, vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(hidden_size, dtype=torch.float32)
        out[h_slice] = vec
        return out

    with torch.no_grad():
        for blk in base_model.blocks:
            blk.mlp.router.weight.zero_()
            for exp in blk.mlp.experts.values():
                for attr in ("w1_B", "w2_B", "w3_B"):
                    getattr(exp, attr).zero_()
            # Base already knows general prose E0 so P_0 preserves E0 without erasing anchor skill
            for e_slot, _, _b_idx, _, h_slice in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 3.0 * axes_in[e_slot]
            _install_domain_lora(
                blk.mlp.experts["0"],
                _full_vec(slice(0, block_h), axes_in[0]),
                _full_vec(slice(0, block_h), axes_out[0]),
                strength=0.008,
                inter_slice=slice(24, 28),
                gate_scale=0.14,
            )
            for e_idx, extra_slot in enumerate((3, 4, 5, 6)):
                blk.mlp.router.weight[extra_slot] = 1.2 * extra_basis[e_idx]
                exp_ex = blk.mlp.experts[str(extra_slot)]
                d_ex = 0.25 * extra_basis[(e_idx + 1) % 4]
                _install_domain_lora(
                    exp_ex,
                    extra_basis[e_idx],
                    d_ex,
                    strength=0.0005 * (e_idx + 1),
                    inter_slice=slice(16 + 2 * e_idx, 18 + 2 * e_idx),
                    gate_scale=0.10,
                )

    # Arm A: strong code specialist on E1
    arm_a = copy.deepcopy(base_model)
    with torch.no_grad():
        for blk in arm_a.blocks:
            mlp = blk.mlp
            in_e1 = _full_vec(slice(block_h, 2 * block_h), axes_in[1])
            out_e1 = _full_vec(slice(block_h, 2 * block_h), axes_out[1])
            _install_domain_lora(
                mlp.experts["1"],
                in_e1,
                out_e1,
                strength=0.015,
                inter_slice=slice(32, 36),
                gate_scale=0.14,
            )
            mlp.router.weight[1] += 2.2 * in_e1

    # Arm B (unpermuted reference first): strong math/Lean4 specialists on E2 & E7 + E0 leak to be pruned by P_0
    arm_b_ref = copy.deepcopy(base_model)
    with torch.no_grad():
        for blk in arm_b_ref.blocks:
            mlp = blk.mlp
            in_e2 = _full_vec(slice(2 * block_h, 3 * block_h), axes_in[2])
            out_e2 = _full_vec(slice(2 * block_h, 3 * block_h), axes_out[2])
            _install_domain_lora(
                mlp.experts["2"],
                in_e2,
                out_e2,
                strength=0.015,
                inter_slice=slice(40, 44),
                gate_scale=0.14,
            )
            mlp.router.weight[2] += 2.2 * in_e2
            # Plant small prose-leak in slot 2 rank-1 component (should be cleaned by P_0!)
            mlp.experts["2"].w3_A[1, 40:44] = 0.08
            mlp.experts["2"].w3_B[:block_h, 1] = 0.15 * axes_in[0]

            in_e7 = _full_vec(slice(3 * block_h, 4 * block_h), axes_in[7])
            out_e7 = _full_vec(slice(3 * block_h, 4 * block_h), axes_out[7])
            _install_domain_lora(
                mlp.experts["7"],
                in_e7,
                out_e7,
                strength=0.015,
                inter_slice=slice(48, 52),
                gate_scale=0.14,
            )
            mlp.router.weight[7] += 2.2 * in_e7

    rot_q = _make_zero_mean_rotation(hidden_size, seed=seed + 909) if rotate_arm_b else None
    arm_b = _permute_and_rotate_crowd_lora_model(arm_b_ref, perm, rot_q)

    # Build disjoint probe and eval chunks where scored positions [8..15] test domain targets
    def _make_chunks(split_seed: int, n_per_dom: int, *, cover_all_vocab: bool = False) -> list[dict[str, Any]]:
        g = torch.Generator(device="cpu").manual_seed(split_seed)
        chunks: list[dict[str, Any]] = []
        for _, dom_id, _, v_off, _ in domain_specs:
            perm_v = torch.randperm(block_v, generator=g)
            for c_i in range(n_per_dom):
                if cover_all_vocab:
                    offset = (c_i * 8) % block_v
                    ctx_local = torch.cat([perm_v, perm_v])[offset : offset + 8]
                    ctx = v_off + ctx_local
                else:
                    ctx = torch.randint(v_off, v_off + block_v, (8,), generator=g)
                tgt = torch.tensor(
                    [v_off + 2 + ((c_i + k) % 3) for k in range(8)],
                    dtype=torch.long,
                )
                ids = torch.cat([ctx, tgt], dim=0).tolist()
                chunks.append({"expert_id": dom_id, "ids": ids})
        return chunks

    return {
        "base_model": base_model,
        "arm_a": arm_a,
        "arm_b": arm_b,
        "arm_b_ref": arm_b_ref,
        "perm": list(perm),
        "rot_q": rot_q,
        "probe_chunks": _make_chunks(seed + 1111, 4, cover_all_vocab=True),
        "eval_chunks": _make_chunks(seed + 2222, 4, cover_all_vocab=False),
    }


def _eval_crowd_mean_nll(model: CrowdV1WithEngram, chunks: list[dict[str, Any]], score_from: int = 8) -> tuple[float, dict[str, float]]:
    ids = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    h_pre = forward_model_pre_ln(model, ids, batch_size=8)
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(model.ln.eps),
        model.head.weight,
        ids,
        score_from=score_from,
    ).tolist()
    grouped: dict[str, list[float]] = {}
    for ch, val in zip(chunks, nlls, strict=True):
        grouped.setdefault(str(ch["expert_id"]), []).append(float(val))
    by_dom = {k: sum(v) / float(len(v)) for k, v in sorted(grouped.items())}
    return sum(nlls) / float(len(nlls)), by_dom


class _HFStyleExpert(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.down_proj = nn.Linear(inter_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _AltStyleExpert(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, inter_dim, bias=False)
        self.w3 = nn.Linear(inter_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class _HFStyleRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.wg = nn.Linear(hidden_dim, num_experts, bias=False)


class _HFStyleMoEMLP(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int, *, use_w123: bool = False) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router = _HFStyleRouter(hidden_dim, num_experts)
        exp_cls = _AltStyleExpert if use_w123 else _HFStyleExpert
        self.experts = nn.ModuleList([exp_cls(hidden_dim, inter_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        xf = x.reshape(-1, self.hidden_dim)
        probs = torch.softmax(self.router.wg(xf), dim=-1)
        out = torch.zeros_like(xf)
        for idx, exp in enumerate(self.experts):
            out = out + probs[:, idx : idx + 1] * exp(xf)
        return out.reshape(orig)


class _HFStyleMoELayer(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int, *, use_w123: bool = False) -> None:
        super().__init__()
        self.mlp = _HFStyleMoEMLP(hidden_dim, inter_dim, num_experts, use_w123=use_w123)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _HFStyleMoEModel(nn.Module):
    def __init__(self, hidden_dim: int = 16, inter_dim: int = 24, num_experts: int = 4, use_w123: bool = False) -> None:
        super().__init__()
        self.embed = nn.Embedding(32, hidden_dim)
        self.layers = nn.ModuleList([_HFStyleMoELayer(hidden_dim, inter_dim, num_experts, use_w123=use_w123)])

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for lyr in self.layers:
            x = lyr(x)
        return x


# ---------------------------------------------------------------------------
# Scenario S1: Hungarian Optimality, Procrustes Rotation & Exact Theorem RCD_D3
# ---------------------------------------------------------------------------


def test_s1_hungarian_and_procrustes_exact_permutation_and_rotation_invariance() -> None:
    """(S1) Prove exact O(E^3) Hungarian optimality, 8/8 recovery under rotation+perm, & RCD_D3."""
    # 1. Exact O(E^3) Hungarian matches brute-force minimum across all 5! = 120 permutations
    gen = torch.Generator(device="cpu").manual_seed(20264501)
    cost = torch.randn((5, 5), generator=gen, dtype=torch.float64)
    perm_h, inv_h, total_cost, ot_diag = solve_exact_hungarian_assignment(cost)
    assert ot_diag["is_bijection"] is True
    assert ot_diag["birkhoff_marginal_error"] == 0.0
    for i, j in enumerate(perm_h):
        assert inv_h[j] == i
    brute_best = min(
        sum(float(cost[i, p[i]].item()) for i in range(5))
        for p in itertools.permutations(range(5))
    )
    assert abs(total_cost - brute_best) < 1e-12

    # Sequence cost_matrix path
    perm_seq, _, cost_seq, _ = solve_exact_hungarian_assignment([[10.0, 1.0], [2.0, 9.0]])
    assert perm_seq == [1, 0]
    assert math.isclose(cost_seq, 3.0)

    # 2. Exact 8/8 permutation recovery + exact forward invariance (Lean 4 RCD_D3) on rotated+permuted MoE
    fix = _build_pilot5_cross_arm_fixture(seed=20264510, rotate_arm_b=True)
    arm_b: CrowdV1WithEngram = fix["arm_b"]
    arm_b_ref: CrowdV1WithEngram = fix["arm_b_ref"]
    true_perm = fix["perm"]
    rot_q = fix["rot_q"]
    assert rot_q is not None

    ids_probe = torch.tensor([c["ids"] for c in fix["probe_chunks"]], dtype=torch.long)
    acts_ref: dict[int, torch.Tensor] = {}
    acts_b: dict[int, torch.Tensor] = {}
    hooks_ref = [
        blk.mlp.register_forward_pre_hook(lambda _m, args, li=i: acts_ref.setdefault(li, args[0].detach().clone()))
        for i, blk in enumerate(arm_b_ref.blocks)
    ]
    hooks_b = [
        blk.mlp.register_forward_pre_hook(lambda _m, args, li=i: acts_b.setdefault(li, args[0].detach().clone()))
        for i, blk in enumerate(arm_b.blocks)
    ]
    with torch.no_grad():
        _ = arm_b_ref(ids_probe)
        _ = arm_b(ids_probe)
    for h in hooks_ref + hooks_b:
        h.remove()

    for li in range(len(arm_b_ref.blocks)):
        # Use Arm_B_ref vs Arm_B to test exact RCD_D3 identity after permutation + Procrustes inversion
        align_res = match_moe_experts_hungarian_procrustes(
            arm_b_ref.blocks[li].mlp.router.weight,
            arm_b.blocks[li].mlp.router.weight,
            acts_ref[li],
            arm_b_ref.blocks[li].mlp,
            arm_b.blocks[li].mlp,
            acts_probe_beta=acts_b[li],
            weight_sim_weight=0.10,
        )
        assert align_res["cka_alignment"] > 0.999
        assert align_res["cka_gate_passed"] is True
        assert align_res["permutation"] == true_perm
        assert align_res["mean_matched_similarity"] > align_res["unaligned_identity_similarity"]
        assert align_res["procrustes_metrics"]["orthogonality_error"] < 1e-6

        # Transport Arm_B completely onto Arm_B_ref (fusion_alpha=1.0) -> exact forward invariance!
        cloned_ref = copy.deepcopy(arm_b_ref)
        fuse_aligned_moe_layer_weights(
            cloned_ref.blocks[li].mlp,
            arm_b.blocks[li].mlp,
            align_res,
            config=OptimalTransportMoEFuserConfig(
                alignment_mode="hungarian_procrustes",
                fusion_alpha=1.0,
                router_fusion_alpha=1.0,
                align_lora_rank_procrustes=False,
            ),
        )
        x_test = acts_ref[li].reshape(-1, acts_ref[li].shape[-1])
        with torch.no_grad():
            y_ref = arm_b_ref.blocks[li].mlp(x_test)
            y_trans = cloned_ref.blocks[li].mlp(x_test)
        assert float((y_ref - y_trans).abs().max().item()) < 1e-4

    # 3. LoRA rank-basis Procrustes (B S, S^T A) removes orthogonal basis mismatch identically
    q_rank, _ = torch.linalg.qr(torch.randn((4, 4), generator=gen, dtype=torch.float32))
    b_a = torch.randn((16, 4), generator=gen, dtype=torch.float32)
    a_a = torch.randn((4, 12), generator=gen, dtype=torch.float32)
    b_b = b_a @ q_rank
    a_b = q_rank.T @ a_a
    b_aligned, a_aligned, r_diag = align_lora_rank_basis_procrustes(b_a, a_a, b_b, a_b)
    assert r_diag["product_invariance_error"] < 1e-6
    assert r_diag["factor_distance_after"] < 1e-5
    assert r_diag["factor_distance_after"] < 0.01 * r_diag["factor_distance_before"]
    assert float(((b_aligned @ a_aligned) - (b_b @ a_b)).abs().max().item()) < 1e-5


# ---------------------------------------------------------------------------
# Scenario S2: PILOT5 Cross-Arm Rescue Beats Naive Graft & Naive 50/50 Merge
# ---------------------------------------------------------------------------


def test_s2_pilot5_cross_arm_rescue_beats_naive_graft_and_naive_avg() -> None:
    """(S2) Prove OT+Procrustes+P0 fusion rescues permuted/rotated PILOT5 arms vs naive grafting."""
    fix = _build_pilot5_cross_arm_fixture(seed=20264520, rotate_arm_b=True)
    base_model: CrowdV1WithEngram = fix["base_model"]
    arm_a: CrowdV1WithEngram = fix["arm_a"]
    arm_b: CrowdV1WithEngram = fix["arm_b"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]

    # Build naive baselines:
    # 1) Naive Cross-Arm Graft (Router_A + Unaligned Experts_B) -> PILOT5 mismatch
    naive_graft = copy.deepcopy(arm_a)
    with torch.no_grad():
        for li in range(len(naive_graft.blocks)):
            for e_s in range(8):
                ea = naive_graft.blocks[li].mlp.experts[str(e_s)]
                eb = arm_b.blocks[li].mlp.experts[str(e_s)]
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(ea, attr).copy_(getattr(eb, attr))

    # 2) Naive Index 50/50 Average (unpermuted, unrotated)
    naive_avg = copy.deepcopy(arm_a)
    with torch.no_grad():
        for li in range(len(naive_avg.blocks)):
            ma = naive_avg.blocks[li].mlp
            mb = arm_b.blocks[li].mlp
            ma.router.weight.copy_(0.5 * ma.router.weight + 0.5 * mb.router.weight)
            for e_s in range(8):
                ea = ma.experts[str(e_s)]
                eb = mb.experts[str(e_s)]
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(ea, attr).copy_(0.5 * getattr(ea, attr) + 0.5 * getattr(eb, attr))

    nll_base, _ = _eval_crowd_mean_nll(base_model, eval_chunks)
    nll_a, _ = _eval_crowd_mean_nll(arm_a, eval_chunks)
    nll_b, _ = _eval_crowd_mean_nll(arm_b, eval_chunks)
    nll_graft, _ = _eval_crowd_mean_nll(naive_graft, eval_chunks)
    nll_avg, _ = _eval_crowd_mean_nll(naive_avg, eval_chunks)

    # Execute OT Hungarian + Procrustes + P_0 Synergy fusion
    cfg = OptimalTransportMoEFuserConfig(
        alignment_mode="hungarian_procrustes_p0_synergy",
        orthogonal_synergy_scale=0.90,
        score_from=8,
        batch_size=8,
        inplace=False,
    )
    pre_a_sha = compute_state_sha256(arm_a.state_dict())
    fused_model, receipt = execute_verified_ot_moe_arm_fusion(
        arm_a,
        arm_b,
        calibration_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        base_model=base_model,
        config=cfg,
    )

    assert compute_state_sha256(arm_a.state_dict()) == pre_a_sha  # inplace=False preserved Arm_A
    assert receipt["object"] == OT_MOE_FUSION_RECEIPT_OBJECT
    assert receipt["schema"] == OT_MOE_FUSER_SCHEMA
    assert receipt["status"] == "ACCEPTED"
    assert receipt["accepted"] is True
    assert receipt["committed"] is True
    assert receipt["rolled_back"] is False
    assert receipt["guard_ok"] is True
    assert receipt["general_ratio"] <= 1.01
    assert receipt["worst_domain_ratio"] <= 1.01
    assert receipt["delta_e_vs_arm_a"] < 0.0
    assert receipt["delta_e_vs_arm_b"] < 0.0
    assert receipt["layernorm_calibration"] is not None
    assert verify_ot_moe_fusion_receipt_provenance(receipt) is True

    nll_fused, _ = _eval_crowd_mean_nll(fused_model, eval_chunks)
    assert math.isclose(nll_fused, float(receipt["nll_mean"]), rel_tol=1e-5, abs_tol=1e-5)
    assert nll_fused < nll_base
    assert nll_fused < nll_a
    assert nll_fused < nll_b
    assert nll_fused < nll_graft
    assert nll_fused < nll_avg

    for li_str, rep in receipt["layer_reports"].items():
        assert rep["permutation"] == fix["perm"], f"Layer {li_str} recovered wrong permutation"
        assert rep["cka_alignment"] > 0.98
        assert rep["max_p0_leak_after"] <= 1e-6


# ---------------------------------------------------------------------------
# Scenario S3: CrowdFineGrainedMoEMLP & HF-Style MoE Modes + Inplace Semantics
# ---------------------------------------------------------------------------


def test_s3_finegrained_and_hf_moe_alignment_modes_and_inplace() -> None:
    """(S3) Prove OT+Procrustes rescues rotated CrowdFineGrainedMoEMLP & HF MoEs vs Hungarian-only."""
    # 1. CrowdFineGrainedMoEMLP in CrowdV1WithEngram (dense SwiGLU sub-experts w1, w2, w3)
    cfg_rt = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    gen = torch.Generator(device="cpu").manual_seed(20264530)
    fg_base = CrowdV1WithEngram(cfg_rt)
    with torch.no_grad():
        fg_base.wte.weight.copy_(0.2 * torch.randn(fg_base.wte.weight.shape, generator=gen))
        fg_base.head.weight.copy_(0.2 * torch.randn(fg_base.head.weight.shape, generator=gen))
    upcycle_crowd_v1_to_sparse_moe(
        fg_base,
        split_mode="fine_grained_width_split",
        num_experts=4,
        top_k=2,
        routing_mode="sigma_topk",
        seed=20264531,
    )
    with torch.no_grad():
        r_axes = _make_zero_mean_ortho_rows(4, 16, seed=20264532)
        fg_base.blocks[0].mlp.router.weight.copy_(2.0 * r_axes)
    fg_a = copy.deepcopy(fg_base)
    fg_b_ref = copy.deepcopy(fg_base)
    with torch.no_grad():
        # Arm_A improves slot 1, Arm_B improves slot 2 in rotated+permuted space
        fg_a.blocks[0].mlp.experts_w3[1].weight.add_(0.08 * torch.ones((16, 8)))
        fg_b_ref.blocks[0].mlp.experts_w3[2].weight.add_(0.08 * torch.ones((16, 8)))
        perm4 = [2, 0, 3, 1]
        inv4 = [0] * 4
        for i, j in enumerate(perm4):
            inv4[j] = i
        q_fg = _make_zero_mean_rotation(16, seed=20264533)
        fg_b = copy.deepcopy(fg_b_ref)
        fg_b.wte.weight.copy_(fg_b.wte.weight @ q_fg)
        fg_b.head.weight.copy_(fg_b.head.weight @ q_fg)
        mlp_b = fg_b.blocks[0].mlp
        mlp_b.router.weight.copy_(mlp_b.router.weight[inv4] @ q_fg)
        w1_old = [m.weight.detach().clone() for m in mlp_b.experts_w1]
        w2_old = [m.weight.detach().clone() for m in mlp_b.experts_w2]
        w3_old = [m.weight.detach().clone() for m in mlp_b.experts_w3]
        for dst in range(4):
            src = inv4[dst]
            mlp_b.experts_w1[dst].weight.copy_(w1_old[src] @ q_fg)
            mlp_b.experts_w2[dst].weight.copy_(w2_old[src] @ q_fg)
            mlp_b.experts_w3[dst].weight.copy_(q_fg.T @ w3_old[src])

    x_fg = torch.randn((32, 16), generator=gen)
    p0_fg = torch.eye(16, dtype=torch.float64) - (r_axes[:1].T.double() @ r_axes[:1].double())
    e_fg_dir = torch.ones((8, 1), dtype=torch.float64) / math.sqrt(8.0)
    p0_act_fg = torch.eye(8, dtype=torch.float64) - (e_fg_dir @ e_fg_dir.T)
    e_hf_dir = torch.ones((24, 1), dtype=torch.float64) / math.sqrt(24.0)
    p0_act_hf = torch.eye(24, dtype=torch.float64) - (e_hf_dir @ e_hf_dir.T)

    # 2. HF-style SwiGLU MoE + custom eval_fn using explicit acts_probe_a/acts_probe_b tensors
    hf_a = _HFStyleMoEModel(hidden_dim=16, inter_dim=24, num_experts=4, use_w123=False)
    hf_alt = _HFStyleMoEModel(hidden_dim=16, inter_dim=24, num_experts=4, use_w123=True)
    with torch.no_grad():
        hf_a.layers[0].mlp.router.wg.weight.copy_(2.0 * r_axes)
        hf_alt.layers[0].mlp.router.wg.weight.copy_(2.0 * r_axes)
        for e_i in range(4):
            e_mod = hf_a.layers[0].mlp.experts[e_i]
            e_mod.gate_proj.weight.copy_(0.6 * torch.randn((24, 16), generator=gen))
            e_mod.up_proj.weight.copy_(0.6 * torch.randn((24, 16), generator=gen))
            e_mod.down_proj.weight.copy_(0.6 * torch.randn((16, 24), generator=gen))
            e_alt = hf_alt.layers[0].mlp.experts[e_i]
            e_alt.w1.weight.copy_(e_mod.gate_proj.weight)
            e_alt.w2.weight.copy_(e_mod.up_proj.weight)
            e_alt.w3.weight.copy_(e_mod.down_proj.weight)

    # Custom domain eval rewards matching the aligned Arm_A + Arm_B synergy mapping
    q_hf = _make_zero_mean_rotation(16, seed=20264534)
    hf_b_ref = copy.deepcopy(hf_a)
    with torch.no_grad():
        for e_i in range(4):
            hf_b_ref.layers[0].mlp.experts[e_i].down_proj.weight.mul_(3.0)
    hf_b = copy.deepcopy(hf_b_ref)
    with torch.no_grad():
        mlp_hb = hf_b.layers[0].mlp
        mlp_hb.router.wg.weight.copy_(mlp_hb.router.wg.weight[inv4] @ q_hf)
        old_g = [e.gate_proj.weight.detach().clone() for e in mlp_hb.experts]
        old_u = [e.up_proj.weight.detach().clone() for e in mlp_hb.experts]
        old_d = [e.down_proj.weight.detach().clone() for e in mlp_hb.experts]
        for dst in range(4):
            src = inv4[dst]
            mlp_hb.experts[dst].gate_proj.weight.copy_(old_g[src] @ q_hf)
            mlp_hb.experts[dst].up_proj.weight.copy_(old_u[src] @ q_hf)
            mlp_hb.experts[dst].down_proj.weight.copy_(q_hf.T @ old_d[src])

    with torch.no_grad():
        target_out = 0.5 * hf_a.layers[0].mlp(x_fg) + 0.5 * hf_b_ref.layers[0].mlp(x_fg)
    acts_b_rot = (x_fg @ q_hf).detach()

    def _hf_eval(mod: nn.Module) -> dict[str, float]:
        err = float((mod.layers[0].mlp(x_fg) - target_out).pow(2).mean().item())
        return {"E0_general": 1.0 + 0.001 * err, "E1_spec": 1.0 + 25.0 * err}

    # Test CrowdFineGrainedMoEMLP direct fusion (including p_null_act) + HF-style inplace=True fusion
    align_fg = match_moe_experts_hungarian_procrustes(
        fg_a.blocks[0].mlp.router.weight,
        fg_b.blocks[0].mlp.router.weight,
        x_fg,
        fg_a.blocks[0].mlp,
        fg_b.blocks[0].mlp,
        acts_probe_beta=(x_fg @ q_fg),
    )
    assert align_fg["permutation"] == perm4
    fg_meta = fuse_aligned_moe_layer_weights(
        fg_a.blocks[0].mlp,
        fg_b.blocks[0].mlp,
        align_fg,
        mlp_base=fg_base.blocks[0].mlp,
        p_null_x=p0_fg,
        p_null_act=p0_act_fg,
        config=OptimalTransportMoEFuserConfig(
            alignment_mode="hungarian_procrustes_p0_synergy",
            project_router_into_nullspace=True,
        ),
    )
    assert fg_meta["max_p0_leak_after"] <= 1e-6

    # Test HF model with both `hungarian` (unrotated fails) vs `hungarian_procrustes` (inplace=True)
    hf_hung_only, rec_hung = execute_verified_ot_moe_arm_fusion(
        hf_a,
        hf_b,
        acts_probe_a=x_fg,
        acts_probe_b=acts_b_rot,
        eval_fn=_hf_eval,
        baseline_domain_nll={"E0_general": 1.05, "E1_spec": 2.50},
        config=OptimalTransportMoEFuserConfig(
            alignment_mode="hungarian",
            align_hidden_space_procrustes=False,
            require_beat_both_arms=False,
            inplace=False,
        ),
    )
    assert rec_hung["rolled_back"] is True  # unrotated Procrustes mismatch degrades & rolls back!

    hf_in = copy.deepcopy(hf_a)
    hf_ret, rec_proc = execute_verified_ot_moe_arm_fusion(
        hf_in,
        hf_b,
        acts_probe_a={0: x_fg},
        acts_probe_b={0: acts_b_rot},
        p_null_by_layer={0: {"x": p0_fg, "act": p0_act_hf}},
        eval_fn=_hf_eval,
        baseline_domain_nll={"E0_general": 1.05, "E1_spec": 2.50},
        config=OptimalTransportMoEFuserConfig(
            alignment_mode="hungarian_procrustes",
            require_beat_both_arms=False,
            min_energy_improvement=0.0,
            inplace=True,
        ),
    )
    assert hf_ret is hf_in
    assert rec_proc["layer_reports"]["0"]["permutation"] == perm4
    assert rec_proc["accepted"] is True
    assert hf_hung_only is not hf_a

    # Verify alt (w1/w2/w3), bare MLP, acts_probe_beta=None, P_0 (no base_mlp), ModuleDict/Linear/Callable
    align_alt = match_moe_experts_hungarian_procrustes(
        hf_alt.layers[0].mlp.router.wg.weight,
        hf_alt.layers[0].mlp.router.wg.weight,
        x_fg,
        hf_alt.layers[0].mlp,
        hf_alt.layers[0].mlp,
        acts_probe_beta=None,
        weight_sim_weight=0.25,
    )
    assert align_alt["permutation"] == [0, 1, 2, 3]
    assert align_alt["similarity_matrix"].shape == (4, 4)

    hf_alt_copy = copy.deepcopy(hf_alt)
    fuse_aligned_moe_layer_weights(
        hf_alt_copy.layers[0].mlp,
        hf_alt.layers[0].mlp,
        align_alt,
        mlp_base=None,
        p_null_x=p0_fg,
        p_null_act=p0_act_hf,
        config=OptimalTransportMoEFuserConfig(
            alignment_mode="hungarian_procrustes_p0_synergy",
            project_into_prose_nullspace=True,
        ),
    )

    bare_ret, bare_rec = execute_verified_ot_moe_arm_fusion(
        hf_alt.layers[0].mlp,
        hf_alt.layers[0].mlp,
        acts_probe_a=x_fg,
        p_null_by_layer=p0_fg,
        eval_fn=lambda _m: {"E0_general": 1.0, "E1_spec": 1.0},
        rollback_target="arm_a",
        config=OptimalTransportMoEFuserConfig(
            alignment_mode="hungarian_procrustes_p0_synergy",
            require_beat_both_arms=False,
            nullspace_max_modes=8,
            inplace=False,
        ),
    )
    assert bare_rec["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert bare_ret is not hf_alt.layers[0].mlp

    dict_exps = nn.ModuleDict({str(idx): hf_alt.layers[0].mlp.experts[idx] for idx in range(4)})
    mixed_exps = [
        nn.Linear(16, 16, bias=False),
        nn.Linear(16, 16, bias=False),
        lambda t_x: t_x * 0.5,
        lambda t_x: t_x * 0.25,
    ]
    sim_dict, _ = compute_cross_arm_expert_similarity_matrix(
        r_axes,
        r_axes,
        x_fg,
        dict_exps,
        dict_exps,
        weight_sim_weight=0.25,
    )
    sim_mixed, _ = compute_cross_arm_expert_similarity_matrix(
        r_axes,
        r_axes,
        x_fg,
        mixed_exps,
        mixed_exps,
        weight_sim_weight=0.25,
    )
    assert sim_dict.shape == (4, 4) and sim_mixed.shape == (4, 4)


# ---------------------------------------------------------------------------
# Scenario S4: CKA Gate & ANSE Maximum Pain Atomic Bitwise Rollback
# ---------------------------------------------------------------------------


def test_s4_cka_gate_and_anse_maximum_pain_atomic_bitwise_rollback() -> None:
    """(S4) Assert CKA gate, D0 guard, energy gate, eval failure, and bitwise rollback."""
    fix = _build_pilot5_cross_arm_fixture(seed=20264540, rotate_arm_b=True)
    base_model: CrowdV1WithEngram = fix["base_model"]
    arm_a: CrowdV1WithEngram = fix["arm_a"]
    arm_b: CrowdV1WithEngram = fix["arm_b"]

    arm_a_sha = compute_state_sha256(arm_a.state_dict())
    arm_b_sha = compute_state_sha256(arm_b.state_dict())
    base_sha = compute_state_sha256(base_model.state_dict())

    # 1. Low-CKA structural mismatch rejects cleanly before fusion
    gen = torch.Generator(device="cpu").manual_seed(20264541)
    x_a = torch.randn((32, 48), generator=gen)
    x_noise = torch.randn((32, 48), generator=gen)
    _, rec_cka = execute_verified_ot_moe_arm_fusion(
        arm_a,
        arm_b,
        acts_probe_a=x_a,
        acts_probe_b=x_noise,
        eval_fn=lambda _m: {"E0_gen": 1.0, "E1_spec": 1.0},
        config=OptimalTransportMoEFuserConfig(min_cka_gate=0.80, inplace=True),
    )
    assert rec_cka["status"] == "ROLLED_BACK_CKA_GATE"
    assert rec_cka["rolled_back"] is True
    assert rec_cka["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_cka["final_state_sha256"] == arm_a_sha
    assert compute_state_sha256(arm_a.state_dict()) == arm_a_sha
    assert verify_ot_moe_fusion_receipt_provenance(rec_cka) is True

    # 2. General prose guard violation (> +1.0%) triggers ANSE Maximum Pain + bitwise rollback to target
    calls = {"n": 0}

    def _guard_fail_eval(_m: nn.Module) -> dict[str, float]:
        calls["n"] += 1
        if calls["n"] <= 3:  # arm_a, arm_b, base
            return {"E0_gen": 1.0, "E1_spec": 1.5}
        return {"E0_gen": 1.05, "E1_spec": 0.9}  # E0 +5.0% -> Maximum Pain!

    for rb_target, want_sha in (("arm_a", arm_a_sha), ("base", base_sha), ("arm_b", arm_b_sha)):
        calls["n"] = 0
        work = copy.deepcopy(arm_a)
        _, rec_rb = execute_verified_ot_moe_arm_fusion(
            work,
            arm_b,
            calibration_chunks=fix["probe_chunks"],
            base_model=base_model,
            eval_fn=_guard_fail_eval,
            config=OptimalTransportMoEFuserConfig(
                score_from=8,
                batch_size=8,
                rollback_target=rb_target,
                inplace=True,
            ),
        )
        assert rec_rb["status"] == "ROLLED_BACK_GUARD_VIOLATION"
        assert rec_rb["rolled_back"] is True
        assert rec_rb["anse_energy"] == ANSE_MAXIMUM_PAIN
        assert rec_rb["final_state_sha256"] == want_sha
        assert compute_state_sha256(work.state_dict()) == want_sha

    # 3. Non-improving energy rollback & eval exception rollback
    non_imp_calls = {"n": 0}

    def _non_improving_eval(_m: nn.Module) -> dict[str, float]:
        non_imp_calls["n"] += 1
        if non_imp_calls["n"] <= 3:
            return {"E0_gen": 1.0, "E1_spec": 1.2}
        return {"E0_gen": 1.0, "E1_spec": 1.205}  # within +1.0% guard, yet higher energy than Arm A/B

    _, rec_nonimp = execute_verified_ot_moe_arm_fusion(
        arm_a,
        arm_b,
        calibration_chunks=fix["probe_chunks"],
        base_model=base_model,
        eval_fn=_non_improving_eval,
        config=OptimalTransportMoEFuserConfig(score_from=8, batch_size=8, inplace=False),
    )
    assert rec_nonimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_nonimp["final_state_sha256"] == arm_a_sha

    exc_calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, float]:
        exc_calls["n"] += 1
        if exc_calls["n"] <= 3:
            return {"E0_gen": 1.0, "E1_spec": 1.2}
        raise RuntimeError("simulated NaN logits")

    _, rec_boom = execute_verified_ot_moe_arm_fusion(
        arm_a,
        arm_b,
        calibration_chunks=fix["probe_chunks"],
        base_model=base_model,
        eval_fn=_boom_eval,
        config=OptimalTransportMoEFuserConfig(score_from=8, batch_size=8, inplace=False),
    )
    assert rec_boom["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert "simulated NaN logits" in str(rec_boom["eval_error"])
    assert rec_boom["final_state_sha256"] == arm_a_sha


# ---------------------------------------------------------------------------
# Scenario S5: Validation Gates & Security / Zero-Stub / OwnerOrder Compliance
# ---------------------------------------------------------------------------


def test_s5_validation_gates_and_security_stub_owner_compliance() -> None:
    """(S5) Assert strict input validation, zero stubs, zero lateral egress, and OwnerOrder."""
    # 1. Config & helper validation errors
    with pytest.raises(ValueError, match="alignment_mode"):
        OptimalTransportMoEFuserConfig(alignment_mode="bad")
    with pytest.raises(ValueError, match="rollback_target"):
        OptimalTransportMoEFuserConfig(rollback_target="bad")
    with pytest.raises(ValueError, match="Sum of similarity weights"):
        OptimalTransportMoEFuserConfig(router_sim_weight=0.0, func_sim_weight=0.0, weight_sim_weight=0.0)
    with pytest.raises(ValueError, match="nullspace_sv_rel_tol"):
        OptimalTransportMoEFuserConfig(nullspace_sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="max_general_drift_ratio"):
        OptimalTransportMoEFuserConfig(max_general_drift_ratio=0.95)
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        OptimalTransportMoEFuserConfig(anse_safe_general_ratio=0.9)
    with pytest.raises(ValueError, match="general_prefixes"):
        OptimalTransportMoEFuserConfig(general_prefixes=())
    with pytest.raises(ValueError, match="lambda_grid"):
        OptimalTransportMoEFuserConfig(lambda_grid=())

    with pytest.raises(TypeError, match="cost_matrix"):
        solve_exact_hungarian_assignment("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="square"):
        solve_exact_hungarian_assignment(torch.ones((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        solve_exact_hungarian_assignment(torch.tensor([[1.0, float("nan")], [0.0, 1.0]]))

    g2 = torch.eye(4, 6)
    e2 = [nn.Linear(6, 6) for _ in range(4)]
    with pytest.raises(ValueError, match="Sum of"):
        compute_cross_arm_expert_similarity_matrix(g2, g2, torch.ones((4, 6)), e2, e2, router_sim_weight=0.0, func_sim_weight=0.0)
    with pytest.raises(TypeError, match="gate_alpha"):
        compute_cross_arm_expert_similarity_matrix("bad", g2, torch.ones((4, 6)), e2, e2)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="matching 2D"):
        compute_cross_arm_expert_similarity_matrix(g2, torch.eye(3, 6), torch.ones((4, 6)), e2, e2)
    with pytest.raises(ValueError, match="acts_probe_alpha width"):
        compute_cross_arm_expert_similarity_matrix(g2, g2, torch.ones((4, 5)), e2, e2)
    with pytest.raises(ValueError, match="acts_probe_beta shape"):
        compute_cross_arm_expert_similarity_matrix(g2, g2, torch.ones((4, 6)), e2, e2, acts_probe_beta=torch.ones((3, 6)))
    with pytest.raises(ValueError, match="experts_alpha length"):
        compute_cross_arm_expert_similarity_matrix(g2, g2, torch.ones((4, 6)), e2[:3], e2)

    with pytest.raises(ValueError, match="shapes must match"):
        align_lora_rank_basis_procrustes(torch.ones((4, 2)), torch.ones((2, 6)), torch.ones((4, 3)), torch.ones((3, 6)))
    with pytest.raises(ValueError, match="Rank mismatch"):
        align_lora_rank_basis_procrustes(torch.ones((4, 3)), torch.ones((2, 6)), torch.ones((4, 3)), torch.ones((2, 6)))
    zb, za, zdiag = align_lora_rank_basis_procrustes(torch.zeros((4, 2)), torch.zeros((2, 6)), torch.zeros((4, 2)), torch.zeros((2, 6)))
    assert float(zb.abs().sum().item()) == 0.0 and float(za.abs().sum().item()) == 0.0 and zdiag["product_invariance_error"] == 0.0

    with pytest.raises(ValueError, match="non-empty"):
        compute_cross_arm_expert_similarity_matrix(
            torch.zeros((0, 4)), torch.zeros((0, 4)), torch.ones((2, 4)), [], []
        )
    with pytest.raises(TypeError, match="ModuleList"):
        compute_cross_arm_expert_similarity_matrix(g2, g2, torch.ones((4, 6)), 123, e2)
    with pytest.raises(TypeError, match="Unsupported expert"):
        compute_cross_arm_expert_similarity_matrix(g2, g2, torch.ones((4, 6)), [1, 2, 3, 4], e2)

    hf1 = _HFStyleMoEModel(hidden_dim=8, inter_dim=12, num_experts=2)
    hf2 = _HFStyleMoEModel(hidden_dim=8, inter_dim=12, num_experts=2)
    hf2_layer2 = nn.Module()
    hf2_layer2.mlp = copy.deepcopy(hf2.layers[0].mlp)
    hf2.layers.append(hf2_layer2)

    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_ot_moe_arm_fusion("bad", hf1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="base_model"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, base_model="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eval_fn"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="OptimalTransportMoEFuserConfig"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, config="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unsupported MoE model"):
        execute_verified_ot_moe_arm_fusion(nn.Linear(4, 4), hf1, acts_probe_a=torch.ones((4, 8)), eval_fn=lambda _m: {"E0_a": 1.0})
    empty_wrap = nn.Module()
    empty_wrap.layers = nn.ModuleList([nn.Linear(4, 4)])
    with pytest.raises(ValueError, match="No MoE MLP layers"):
        execute_verified_ot_moe_arm_fusion(empty_wrap, hf1, acts_probe_a=torch.ones((4, 8)), eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="matching MoE layer"):
        execute_verified_ot_moe_arm_fusion(hf1, hf2, acts_probe_a=torch.ones((4, 8)), eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="Provide calibration_chunks"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1)
    with pytest.raises(ValueError, match="layer_indices"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, acts_probe_a=torch.ones((4, 8)), eval_fn=lambda _m: {"E0_a": 1.0}, layer_indices=[9])
    with pytest.raises(ValueError, match="acts_probe_a missing"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, acts_probe_a={}, eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="acts_probe_b missing"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, acts_probe_a={0: torch.ones((4, 8))}, acts_probe_b={}, eval_fn=lambda _m: {"E0_a": 1.0})
    with pytest.raises(ValueError, match="Missing required general-domain"):
        execute_verified_ot_moe_arm_fusion(hf1, hf1, acts_probe_a=torch.ones((4, 8)), eval_fn=lambda _m: {"E1_only": 1.0})
    with pytest.raises(TypeError, match="p_null_by_layer"):
        execute_verified_ot_moe_arm_fusion(
            hf1, hf1, acts_probe_a={0: torch.ones((4, 8))}, p_null_by_layer={0: "bad"}, eval_fn=lambda _m: {"E0_a": 1.0}
        )
    with pytest.raises(ValueError, match="identical keys"):
        mism_calls = {"n": 0}
        execute_verified_ot_moe_arm_fusion(
            hf1,
            hf1,
            acts_probe_a=torch.ones((4, 8)),
            eval_fn=lambda _m: (mism_calls.__setitem__("n", mism_calls["n"] + 1) or ({"E0_a": 1.0} if mism_calls["n"] == 1 else {"E0_b": 1.0})),
        )
    with pytest.raises(TypeError, match="missing router/gate"):
        bad_mlp = copy.deepcopy(hf1.layers[0].mlp)
        bad_mlp.router = nn.Identity()
        fuse_aligned_moe_layer_weights(bad_mlp, hf1.layers[0].mlp, {"permutation": [0, 1], "r_b2a": torch.eye(8), "cka_alignment": 1.0, "procrustes_metrics": {"orthogonality_error": 0.0}, "mean_matched_similarity": 1.0, "unaligned_identity_similarity": 1.0, "similarity_gain": 0.0})
    with pytest.raises(TypeError, match="Unsupported expert module"):
        bad_exp_mlp = copy.deepcopy(hf1.layers[0].mlp)
        bad_exp_mlp.experts = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        fuse_aligned_moe_layer_weights(bad_exp_mlp, hf1.layers[0].mlp, {"permutation": [0, 1], "r_b2a": torch.eye(8), "cka_alignment": 1.0, "procrustes_metrics": {"orthogonality_error": 0.0}, "mean_matched_similarity": 1.0, "unaligned_identity_similarity": 1.0, "similarity_gain": 0.0})
    with pytest.raises(TypeError, match="experts container"):
        no_exp_mlp = copy.deepcopy(hf1.layers[0].mlp)
        no_exp_mlp.experts = nn.Linear(8, 8)
        fuse_aligned_moe_layer_weights(no_exp_mlp, hf1.layers[0].mlp, {"permutation": [0, 1], "r_b2a": torch.eye(8), "cka_alignment": 1.0, "procrustes_metrics": {"orthogonality_error": 0.0}, "mean_matched_similarity": 1.0, "unaligned_identity_similarity": 1.0, "similarity_gain": 0.0})
    with pytest.raises(ValueError, match="mismatch"):
        fuse_aligned_moe_layer_weights(hf1.layers[0].mlp, hf1.layers[0].mlp, {"permutation": [0], "r_b2a": torch.eye(8), "cka_alignment": 1.0, "procrustes_metrics": {"orthogonality_error": 0.0}, "mean_matched_similarity": 1.0, "unaligned_identity_similarity": 1.0, "similarity_gain": 0.0})

    s5_probe_gen = torch.Generator(device="cpu").manual_seed(20264551)
    x_s5 = torch.randn((8, 8), generator=s5_probe_gen)
    _, rec_p0_miss = execute_verified_ot_moe_arm_fusion(
        hf1,
        hf1,
        acts_probe_a={0: x_s5},
        p_null_by_layer={99: torch.eye(8)},
        eval_fn=lambda _m: {"E0_a": 1.0},
    )
    assert rec_p0_miss["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    _, rec_p0_mat = execute_verified_ot_moe_arm_fusion(
        hf1,
        hf1,
        acts_probe_a={0: x_s5},
        p_null_by_layer={0: torch.eye(8)},
        eval_fn=lambda _m: {"E0_a": 1.0},
    )
    assert rec_p0_mat["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"

    with pytest.raises(TypeError, match="Mapping"):
        verify_ot_moe_fusion_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_ot_moe_fusion_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_ot_moe_fusion_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_ot_moe_fusion_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": OT_MOE_FUSION_RECEIPT_OBJECT,
                "schema": OT_MOE_FUSER_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 2. AGPL-3.0, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn45", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn45", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/optimal_transport_moe_arm_fuser.py", TARGET_MODULE_PATH),
        ("tests/test_optimal_transport_moe_arm_fuser_scenarios.py", TEST_FILE_PATH),
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
        {"nll_mean": 1.38, "cka_alignment": 0.999, "anse_energy": -0.42, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): optimal-transport Hungarian and orthogonal Procrustes cross-arm MoE expert and router fuser"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-45 Gen-3 Direction D3 optimal-transport Hungarian expert "
        "slot matching paired with orthogonal Procrustes hidden/LoRA-rank alignment prior to P_0 "
        "fusion under ANSE Delta E < 0 with atomic bitwise rollback and SHA-256 provenance receipts."
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
        pr_number=1245,
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
