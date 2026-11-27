# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-49: Gen-3 Reine-Cortex D1-D5 unified campaign orchestrator."""

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
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.alveole_mitosis_hessian_nf4_exporter import (
    STD_8GB_WDDM_CEILING_MIB,
    AlveoleMitosisNF4Config,
)
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.kronecker_riemannian_nullspace_transport import (
    KroneckerRiemannianTransportConfig,
)
from hivemind.swarm.lean4_wsp_theorem_registry import audit_lean4_wsp_spec_file
from hivemind.swarm.optimal_transport_moe_arm_fuser import OptimalTransportMoEFuserConfig
from hivemind.swarm.reine_cortex_gen3_orchestrator import (
    ANSE_MAXIMUM_PAIN,
    GEN3_CAMPAIGN_RECEIPT_OBJECT,
    GEN3_ORCHESTRATOR_SCHEMA,
    GENESIS_MERKLE_SHA256,
    STAGE_D1_TROPHALLAXIS,
    STAGE_D2_KRONECKER,
    STAGE_D3_FUSION,
    STAGE_D4_MITOSIS_NF4,
    STAGE_D5_INTAKE,
    STAGE_ORDER,
    Gen3ReineCortexCampaignConfig,
    compute_merkle_root_sha256,
    execute_gen3_five_direction_campaign,
    verify_gen3_campaign_merkle_chain,
    verify_gen3_campaign_receipt_provenance,
)
from hivemind.swarm.self_healing_checkpoint_rescuer import extract_checkpoint_blocks
from hivemind.swarm.zk_wsp_federated_hive_verifier import (
    ZKWSPFederatedHiveConfig,
    build_federated_hive_packet,
    build_prose_subspace_commitment,
    compute_state_sha256,
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
LEAN_SPEC_PATH = REPO_ROOT / "lean4" / "HiveMindSpec.lean"
TARGET_MODULE_PATH = (
    REPO_ROOT / "src" / "hivemind" / "swarm" / "reine_cortex_gen3_orchestrator.py"
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


@pytest.fixture(scope="module")
def cached_lean_audit() -> Any:
    """Parse ``lean4/HiveMindSpec.lean`` once per test module."""
    return audit_lean4_wsp_spec_file(LEAN_SPEC_PATH)


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
    """Construct orthogonal matrix preserving the all-ones vector (LayerNorm equivariant)."""
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
    gate_scale: float = 0.85,
) -> None:
    """Write a deterministic rank-1 SwiGLU LoRA mapping ``in_dir -> strength * out_dir``."""
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


def _permute_and_rotate_crowd_moe(
    model: CrowdV1WithEngram,
    perm: list[int],
    rot_q: torch.Tensor,
) -> CrowdV1WithEngram:
    """Permute expert slots and apply orthogonal hidden rotation ``rot_q`` to ``Arm_B``."""
    cloned = copy.deepcopy(model)
    n_e = len(perm)
    inv_perm = [0] * n_e
    for i, b_slot in enumerate(perm):
        inv_perm[b_slot] = i

    q32 = rot_q.to(torch.float32)
    with torch.no_grad():
        cloned.wte.weight.copy_(cloned.wte.weight @ q32)
        cloned.head.weight.copy_(cloned.head.weight @ q32)
        for blk in cloned.blocks:
            blk.attn.q.weight.copy_(blk.attn.q.weight @ q32)
            blk.attn.k.weight.copy_(blk.attn.k.weight @ q32)
            blk.attn.v.weight.copy_(blk.attn.v.weight @ q32)
            blk.attn.o.weight.copy_(q32.T @ blk.attn.o.weight)
            mlp = blk.mlp
            mlp.base_mlp.w1.weight.copy_(mlp.base_mlp.w1.weight @ q32)
            mlp.base_mlp.w2.weight.copy_(mlp.base_mlp.w2.weight @ q32)
            mlp.base_mlp.w3.weight.copy_(q32.T @ mlp.base_mlp.w3.weight)

            r_old = mlp.router.weight.detach().clone()
            mlp.router.weight.copy_(r_old[inv_perm] @ q32)

            old_exps = {
                e: {
                    attr: getattr(mlp.experts[str(e)], attr).detach().clone()
                    for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B")
                }
                for e in range(n_e)
            }
            for dst_b in range(n_e):
                src_a = inv_perm[dst_b]
                exp_dst = mlp.experts[str(dst_b)]
                for attr, val in old_exps[src_a].items():
                    t_new = val.clone()
                    if attr in ("w1_A", "w2_A"):
                        t_new = t_new @ q32
                    elif attr == "w3_B":
                        t_new = q32.T @ t_new
                    getattr(exp_dst, attr).copy_(t_new)
    return cloned


def _build_gen3_five_direction_fixture(
    lean_audit: Any,
    *,
    seed: int = 20264900,
    round_id: str = "round_1",
) -> dict[str, Any]:
    """Build deterministic 6-domain MoE fixture activating all 5 Gen-3 directions (D1-D5)."""
    hidden_size = 64
    intermediate_size = 64
    vocab_size = 64
    num_experts = 6
    block_h = 8
    block_v = 8

    rt_cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    model = CrowdV1WithEngram(rt_cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_prose", 0, 0, slice(0, 8), slice(0, 8)),
        (1, "E1_code_python", 1, 8, slice(8, 16), slice(8, 16)),
        (2, "E2_math_reasoning", 2, 16, slice(16, 24), slice(16, 24)),
        (3, "E3_physics_sim", 3, 24, slice(24, 32), slice(24, 32)),
        (4, "E4_cyber_sec", 4, 32, slice(32, 40), slice(32, 40)),
        (5, "E5_formal_lean4", 5, 40, slice(40, 48), slice(40, 48)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    axes_alt: dict[int, torch.Tensor] = {}
    full_axes_in: dict[int, torch.Tensor] = {}
    full_axes_out: dict[int, torch.Tensor] = {}

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.015 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice, _ in domain_specs:
            basis = _make_zero_mean_ortho_rows(block_h - 1, block_h, seed=seed + 41 + b_idx * 17)
            u_in, u_out, u_alt = basis[0], basis[1], basis[2]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            axes_alt[e_slot] = u_alt
            f_in = torch.zeros(hidden_size, dtype=torch.float32)
            f_out = torch.zeros(hidden_size, dtype=torch.float32)
            f_in[h_slice] = u_in
            f_out[h_slice] = u_out
            full_axes_in[e_slot] = f_in
            full_axes_out[e_slot] = f_out

            # Tokens v_off+1,2 carry primary eval embeddings; v_off+3..7 span remaining slice axes
            # so Procrustes (D3) identifies full hidden_size rotation across all 48 dims.
            for t_i in range(block_v):
                if t_i in (1, 2):
                    vec = 0.25 * basis[2 + (t_i % 2)] + 0.70 * u_in
                    wte[v_off + t_i, h_slice] = vec - vec.mean()
                elif t_i <= 6:
                    vec = 0.35 * basis[(t_i - 1) % (block_h - 1)] + 0.50 * u_in
                    wte[v_off + t_i, h_slice] = vec - vec.mean()
                else:
                    # Slight asymmetric block offset breaks all-ones block degeneracy for full-rank Procrustes
                    wte[v_off + t_i, h_slice] = 0.30 * u_in + 0.25 * u_out + 0.15 / math.sqrt(float(block_h))
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            if e_slot == 0:
                head_w[v_off + 1, h_slice] = 0.10 * u_in + 0.85 * u_out + 0.25 * u_alt
                head_w[v_off + 2, h_slice] = 0.10 * u_in + 0.85 * u_out - 0.25 * u_alt
            elif e_slot == 1:
                # D5 target: misaligned on E1 head logits so P_0 multi-hive consensus lowers E1 NLL
                head_w[v_off + 1, h_slice] = -0.25 * u_out + 0.20 * u_alt
                head_w[v_off + 2, h_slice] = -0.25 * u_out - 0.20 * u_alt
            elif e_slot in (4, 5):
                # D1/D2 targets: aligned head outputs awaiting w3_B specialist activation
                for off_i in range(1, block_v):
                    sgn = 1.0 if (off_i % 2 == 1) else -1.0
                    head_w[v_off + off_i, h_slice] = 0.65 * u_out + (0.20 * sgn) * u_alt
            else:
                for off_i in range(1, block_v):
                    sgn = 1.0 if (off_i % 2 == 1) else -1.0
                    head_w[v_off + off_i, h_slice] = 0.08 * u_in + 0.72 * u_out + (0.22 * sgn) * u_alt

        model.wte.weight.copy_(wte)
        model.head.weight.copy_(head_w)
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.attn.q.weight.copy_(0.14 * torch.eye(hidden_size))
            blk.attn.k.weight.copy_(0.14 * torch.eye(hidden_size))
            blk.attn.v.weight.copy_(0.14 * torch.eye(hidden_size))
            blk.attn.o.weight.copy_(0.12 * torch.eye(hidden_size))
            blk.mlp.w1.weight[:16, :16].copy_(0.16 * torch.eye(16))
            blk.mlp.w2.weight[:16, :16].copy_(0.16 * torch.eye(16))
            blk.mlp.w3.weight[:16, :16].copy_(0.05 * torch.eye(16))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=4,
        routing_mode="sigma_topk",
        seed=seed + 7,
    )

    with torch.no_grad():
        for blk in model.blocks:
            blk.mlp.router.weight.zero_()
            for e_slot, _, _, _, h_slice, _ in domain_specs:
                blk.mlp.router.weight[e_slot, h_slice] = 3.2 * axes_in[e_slot]
            for exp in blk.mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp, attr).zero_()

            # Slot 3 (E3_physics_sim): Ghost-sink corruptor recycled by D4 Mitosis
            _install_domain_lora(
                blk.mlp.experts["3"],
                full_axes_in[3],
                full_axes_out[3],
                strength=-0.75,
                inter_slice=domain_specs[3][5],
                gate_scale=0.90,
            )
            # Slot 4 (E4_cyber_sec): Seeded gate path (w3_B=0), refined by D1 Forager Bees
            _install_domain_lora(
                blk.mlp.experts["4"],
                full_axes_in[4],
                full_axes_out[4],
                strength=0.0,
                inter_slice=domain_specs[4][5],
                gate_scale=0.55,
            )
            # Slot 5 (E5_formal_lean4): Seeded gate path (w3_B=0), transported by D2 Kronecker surgery
            _install_domain_lora(
                blk.mlp.experts["5"],
                full_axes_in[5],
                full_axes_out[5],
                strength=0.0,
                inter_slice=domain_specs[5][5],
                gate_scale=0.55,
            )

        # Build Cross-Arm Model Arm_B (D3) carrying strong E2_math_reasoning, permuted & rotated
        arm_b_raw = copy.deepcopy(model)
        _install_domain_lora(
            arm_b_raw.blocks[0].mlp.experts["2"],
            full_axes_in[2],
            full_axes_out[2],
            strength=0.85,
            inter_slice=domain_specs[2][5],
            gate_scale=0.90,
        )
        # Keep Arm_B head un-calibrated on E1 so D3 beats both Arm_A (post-D5) and Arm_B
        arm_b_raw.head.weight[8:16, 8:16] = 0.0
        perm_b = [0, 4, 5, 1, 2, 3]
        spec_dim = num_experts * block_h - block_h
        rot_q_spec = _make_zero_mean_rotation(spec_dim, seed=seed + 505)
        rot_q = torch.block_diag(
            torch.eye(block_h, dtype=torch.float32),
            rot_q_spec,
            torch.eye(hidden_size - num_experts * block_h, dtype=torch.float32),
        )
        cross_arm_model = _permute_and_rotate_crowd_moe(arm_b_raw, perm_b, rot_q)

    # Probe and Eval chunks (disjoint sequences of len=10; E0 zero-mean [1, 2], E1..E5 span all 40 specialist dims)
    probe_chunks: list[dict[str, Any]] = []
    eval_chunks: list[dict[str, Any]] = []
    for e_slot, dom_name, _, v_off, _, _ in domain_specs:
        if e_slot == 0:
            t_p1 = [v_off + (1 if i % 2 == 0 else 2) for i in range(10)]
            t_p2 = [v_off + (2 if i % 2 == 0 else 1) for i in range(10)]
            t_e1 = [v_off + (1 if i % 3 == 0 else 2) for i in range(10)]
            t_e2 = [v_off + (2 if i % 3 == 0 else 1) for i in range(10)]
            probe_chunks.append({"expert_id": dom_name, "ids": t_p1})
            probe_chunks.append({"expert_id": dom_name, "ids": t_p2})
            eval_chunks.append({"expert_id": dom_name, "ids": t_e1})
            eval_chunks.append({"expert_id": dom_name, "ids": t_e2})
        else:
            t_p1 = [v_off + 1 + (i % 4) for i in range(10)]
            t_p2 = [v_off + 4 + (i % 4) for i in range(10)]
            t_e1 = [v_off + (1 if i % 2 == 0 else 2) for i in range(10)]
            probe_chunks.append({"expert_id": dom_name, "ids": t_p1})
            probe_chunks.append({"expert_id": dom_name, "ids": t_p2})
            eval_chunks.append({"expert_id": dom_name, "ids": t_e1})

    # D5 commitment & 7 packets (5 honest inside P_0 + 2 Byzantine peers)
    v0_prose = torch.zeros((hidden_size, 3), dtype=torch.float64)
    v0_prose[0:8, 0] = axes_in[0].to(torch.float64)
    v0_prose[0:8, 1] = axes_out[0].to(torch.float64)
    v0_prose[0:8, 2] = axes_alt[0].to(torch.float64)
    subspace_comm = build_prose_subspace_commitment(v0_prose, commitment_id="gen3_d5_v0")
    p0_head = torch.eye(hidden_size, dtype=torch.float64) - v0_prose @ v0_prose.T

    d_honest_head = torch.zeros((vocab_size, hidden_size), dtype=torch.float64)
    d_honest_head[9, 8:16] = 0.25 * axes_in[1].to(torch.float64) + 1.15 * axes_out[1].to(torch.float64)
    d_honest_head[10, 8:16] = 0.25 * axes_in[1].to(torch.float64) + 1.15 * axes_out[1].to(torch.float64)
    d_honest_head = d_honest_head @ p0_head

    hive_packets: list[dict[str, Any]] = []
    for h_i in range(5):
        noise = 0.015 * torch.randn(
            (vocab_size, hidden_size),
            generator=torch.Generator(device="cpu").manual_seed(seed + 700 + h_i),
            dtype=torch.float64,
        )
        noise[:, :8] = 0.0
        noise[9:11, 8:16] = 0.0
        noise[16:48, :] = 0.0
        d_h = ((d_honest_head + noise) @ p0_head).to(torch.float32)
        hive_packets.append(
            build_federated_hive_packet(
                f"hive_honest_{h_i}",
                d_h,
                subspace_comm,
                lean_spec_audit=lean_audit,
                round_id=round_id,
                target_key="head.weight",
            )
        )
    # Byzantine 1: prose-space poisoner violating P_0
    d_byz_poison = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
    d_byz_poison[1, 0:8] = -2.5 * axes_out[0]
    hive_packets.append(
        build_federated_hive_packet(
            "hive_byz_prose_poisoner",
            d_byz_poison,
            subspace_comm,
            lean_spec_audit=lean_audit,
            round_id=round_id,
            target_key="head.weight",
        )
    )
    # Byzantine 2: forged Lean 4 attestation trojan
    d_byz_trojan = 2.0 * torch.randn(
        (vocab_size, hidden_size),
        generator=torch.Generator(device="cpu").manual_seed(seed + 799),
        dtype=torch.float32,
    )
    pkt_forged = build_federated_hive_packet(
        "hive_byz_forged_trojan",
        d_byz_trojan,
        subspace_comm,
        lean_spec_audit=lean_audit,
        round_id=round_id,
        target_key="head.weight",
    )
    pkt_forged["lean4_attestation"]["attestation_sha256"] = "0" * 64
    hive_packets.append(pkt_forged)

    # D1 Forager Bee deltas for slot 4 (E4_cyber_sec): 6 honest peers + 1 norm-clipped outlier bee
    exp4_base = model.blocks[0].mlp.experts["4"]
    dw4_clean = torch.zeros_like(exp4_base.w3_B)
    dw4_clean[:, 0] = 0.75 * full_axes_out[4]
    dw4_prose_noise = torch.zeros_like(exp4_base.w3_B)
    dw4_prose_noise[:, 0] = -0.85 * full_axes_in[0]
    forager_bee_deltas: dict[str, dict[str, Any]] = {}
    for b_i in range(6):
        forager_bee_deltas[f"bee_honest_{b_i}"] = {
            "blocks.0.mlp.experts.4": {"w3_B": dw4_clean + dw4_prose_noise}
        }
    forager_bee_deltas["bee_outlier"] = {
        "blocks.0.mlp.experts.4": {"w3_B": -3.5 * dw4_clean + 4.5 * dw4_prose_noise}
    }

    return {
        "model": model,
        "cross_arm_model": cross_arm_model,
        "probe_chunks": probe_chunks,
        "eval_chunks": eval_chunks,
        "hive_packets": hive_packets,
        "subspace_commitment": subspace_comm,
        "forager_bee_deltas": forager_bee_deltas,
        "full_axes_in": full_axes_in,
        "full_axes_out": full_axes_out,
        "v0_prose": v0_prose,
    }


# ---------------------------------------------------------------------------
# Scenario S1: Five-Direction Synergy Beats Every Partial Ablation
# ---------------------------------------------------------------------------


def test_s1_five_direction_synergy_beats_every_partial_ablation_and_enforces_guards(
    cached_lean_audit: Any,
) -> None:
    """(S1) Prove D1-D5 pipeline beats every partial ablation with D0 <= +1% & WDDM <= 6656 MiB."""
    fix = _build_gen3_five_direction_fixture(cached_lean_audit, seed=20264910)
    d5_cfg = ZKWSPFederatedHiveConfig(
        num_byzantine_f=2,
        score_from=4,
        batch_size=16,
        inplace=False,
    )
    d2_cfg = KroneckerRiemannianTransportConfig(
        target_factors=("w3_B",),
        max_kl_nats=0.05,
        damping_a=0.05,
        damping_g=0.05,
        score_from=4,
        batch_size=16,
        inplace=True,
    )
    d3_cfg = OptimalTransportMoEFuserConfig(
        calibrate_layernorm=False,
        score_from=4,
        batch_size=16,
        inplace=False,
    )
    d4_cfg = AlveoleMitosisNF4Config(
        nf4_block_size=64,
        compensation_rank=4,
        num_alternating_steps=1,
        prose_hessian_weight=16.0,
        quantize_lm_head=False,
        projected_target_param_count=3_300_000_000,
        wddm_vram_ceiling_mib=float(STD_8GB_WDDM_CEILING_MIB),
        calibrate_layernorm=True,
        enforce_all_domain_guard=False,
        score_from=4,
        batch_size=16,
        inplace=False,
    )

    def _run_with_cfg(
        cfg_in: Gen3ReineCortexCampaignConfig,
        *,
        fast_ablation: bool = False,
    ) -> tuple[nn.Module, dict[str, Any]]:
        ko_overrides = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.5, 4: 0.0, 5: 0.0} if fast_ablation else None
        return execute_gen3_five_direction_campaign(
            fix["model"],
            probe_chunks=fix["probe_chunks"],
            eval_chunks=fix["eval_chunks"],
            config=cfg_in,
            hive_packets=fix["hive_packets"],
            subspace_commitment=fix["subspace_commitment"],
            lean_spec_audit=cached_lean_audit,
            d5_config=d5_cfg,
            cross_arm_model=fix["cross_arm_model"],
            d3_config=d3_cfg,
            kronecker_expert_indices=(5,),
            d2_config=d2_cfg,
            forager_bee_deltas=fix["forager_bee_deltas"],
            bee_domains=["E4_cyber_sec"] * len(fix["forager_bee_deltas"]),
            explicit_knockout_deltas=ko_overrides,
            d4_config=d4_cfg,
        )

    full_cfg = Gen3ReineCortexCampaignConfig(
        campaign_id="gen3-s1-full-synergy",
        round_id="round_1",
        score_from=4,
        batch_size=16,
        inplace=False,
    )
    full_model, full_rec = _run_with_cfg(full_cfg)

    assert full_rec["object"] == GEN3_CAMPAIGN_RECEIPT_OBJECT
    assert full_rec["schema"] == GEN3_ORCHESTRATOR_SCHEMA
    assert full_rec["accepted"] is True
    assert full_rec["accepted_stages"] == list(STAGE_ORDER)
    assert full_rec["rejected_stages"] == []
    assert full_rec["skipped_stages"] == []
    assert full_rec["general_drift_pct"] <= 1.0
    assert full_rec["cumulative_nll_delta"] < -0.10
    assert full_rec["final_anse_energy"] < 0.0
    assert full_rec["d1_pheromone_directive"] is not None
    assert full_rec["wddm_footprint"] is not None
    assert full_rec["wddm_footprint"]["fits_wddm_ceiling"] is True
    assert full_rec["wddm_footprint"]["projected_target_wddm_vram_mib"] <= 6656.0
    assert verify_gen3_campaign_receipt_provenance(full_rec) is True
    assert full_model is not fix["model"]

    # D5 stage caught both Byzantine peers with zero honest false positives
    d5_entry = full_rec["stage_ledger"][0]
    assert d5_entry["stage_name"] == STAGE_D5_INTAKE
    assert d5_entry["substage_summary"]["num_rejected_by_verifier"] == 2
    assert set(d5_entry["substage_summary"]["rejected_hive_ids"]) == {
        "hive_byz_prose_poisoner",
        "hive_byz_forged_trojan",
    }

    # Strict monotone energy descent across all 5 accepted stages
    energies = [entry["final_anse_energy"] for entry in full_rec["stage_ledger"]]
    assert energies[0] < 0.0
    assert all(energies[i + 1] < energies[i] for i in range(len(energies) - 1))

    # D1-D5 synergy strictly beats every leave-one-direction-out partial ablation & base
    ablation_flags = (
        ("enable_d5_intake", STAGE_D5_INTAKE),
        ("enable_d3_fusion", STAGE_D3_FUSION),
        ("enable_d2_kronecker", STAGE_D2_KRONECKER),
        ("enable_d1_trophallaxis", STAGE_D1_TROPHALLAXIS),
        ("enable_d4_mitosis_nf4", STAGE_D4_MITOSIS_NF4),
    )
    for flag_name, stage_name in ablation_flags:
        ab_cfg = Gen3ReineCortexCampaignConfig(
            campaign_id=f"gen3-s1-ablate-{stage_name}",
            round_id="round_1",
            score_from=4,
            batch_size=16,
            inplace=False,
            **{flag_name: False},
        )
        _, ab_rec = _run_with_cfg(ab_cfg, fast_ablation=(stage_name != STAGE_D4_MITOSIS_NF4))
        assert stage_name in ab_rec["skipped_stages"]
        assert full_rec["final_nll_mean"] < ab_rec["final_nll_mean"], (
            f"Full D1-D5 NLL {full_rec['final_nll_mean']:.6f} did not beat ablation without {stage_name} "
            f"({ab_rec['final_nll_mean']:.6f})"
        )
        assert full_rec["final_anse_energy"] < ab_rec["final_anse_energy"]
        assert verify_gen3_campaign_receipt_provenance(ab_rec) is True


# ---------------------------------------------------------------------------
# Scenario S2: Per-Stage ANSE Ratchet Rejects Regressive Stage & Retains Gains
# ---------------------------------------------------------------------------


def test_s2_per_stage_anse_ratchet_rejects_regressive_stage_retains_beneficial(
    cached_lean_audit: Any,
) -> None:
    """(S2) Reject a regressive single stage while committing and preserving beneficial stages."""
    fix = _build_gen3_five_direction_fixture(cached_lean_audit, seed=20264920)

    # 1. Corrupt Arm_B hidden space so D3 fails CKA / regression while D5, D2, D1, D4 succeed
    bad_arm_b = copy.deepcopy(fix["cross_arm_model"])
    with torch.no_grad():
        bad_arm_b.wte.weight.normal_(mean=0.0, std=2.5, generator=torch.Generator(device="cpu").manual_seed(99))

    d5_cfg = ZKWSPFederatedHiveConfig(num_byzantine_f=2, score_from=4, batch_size=16, inplace=False)
    d3_bad_cfg = OptimalTransportMoEFuserConfig(
        min_cka_gate=0.95,
        calibrate_layernorm=False,
        score_from=4,
        batch_size=16,
        inplace=False,
    )
    d2_cfg = KroneckerRiemannianTransportConfig(
        target_factors=("w3_B",), max_kl_nats=0.05, score_from=4, batch_size=16, inplace=True
    )
    d4_cfg = AlveoleMitosisNF4Config(
        nf4_block_size=64,
        compensation_rank=4,
        num_alternating_steps=1,
        prose_hessian_weight=16.0,
        quantize_lm_head=False,
        projected_target_param_count=3_300_000_000,
        enforce_all_domain_guard=False,
        score_from=4,
        batch_size=16,
        inplace=False,
    )
    _, rec_a = execute_gen3_five_direction_campaign(
        fix["model"],
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        config=Gen3ReineCortexCampaignConfig(campaign_id="gen3-s2-bad-d3", score_from=4),
        hive_packets=fix["hive_packets"],
        subspace_commitment=fix["subspace_commitment"],
        lean_spec_audit=cached_lean_audit,
        d5_config=d5_cfg,
        cross_arm_model=bad_arm_b,
        d3_config=d3_bad_cfg,
        kronecker_expert_indices=(5,),
        d2_config=d2_cfg,
        forager_bee_deltas=fix["forager_bee_deltas"],
        bee_domains=["E4_cyber_sec"] * len(fix["forager_bee_deltas"]),
        d4_config=d4_cfg,
    )
    assert rec_a["accepted"] is True
    assert STAGE_D3_FUSION in rec_a["rejected_stages"]
    for st in (STAGE_D5_INTAKE, STAGE_D2_KRONECKER, STAGE_D1_TROPHALLAXIS, STAGE_D4_MITOSIS_NF4):
        assert st in rec_a["accepted_stages"]

    d3_entry = rec_a["stage_ledger"][1]
    assert d3_entry["accepted"] is False
    assert d3_entry["rolled_back"] is True
    assert d3_entry["rollback_bitwise_verified"] is True
    assert d3_entry["final_state_sha256"] == d3_entry["pre_state_sha256"]
    assert d3_entry["final_state_sha256"] == rec_a["stage_ledger"][0]["final_state_sha256"]
    assert rec_a["general_drift_pct"] <= 1.0
    assert verify_gen3_campaign_receipt_provenance(rec_a) is True

    # 2. Outer ANSE ratchet catches regressive candidate even when D2 inner guard is relaxed,
    #    and also handles non-improving energy and eval_fn exceptions cleanly.
    guard_calls = {"n": 0}

    def _drift_d0_after_sub(_m: nn.Module) -> dict[str, float]:
        guard_calls["n"] += 1
        if guard_calls["n"] <= 2:
            return {"E0_general_prose": 2.0, "E1_code_python": 2.4}
        return {"E0_general_prose": 2.1, "E1_code_python": 2.0}

    relaxed_d2 = KroneckerRiemannianTransportConfig(
        max_kl_nats=2.0,
        general_guard_max_ratio=5.0,
        domain_guard_max_ratio=5.0,
        enforce_all_domain_guard=False,
        require_energy_improvement=False,
        enable_trust_backtracking=False,
        protect_general_expert=False,
        score_from=4,
        batch_size=16,
        inplace=True,
    )
    _, rec_b = execute_gen3_five_direction_campaign(
        fix["model"],
        probe_chunks=fix["probe_chunks"],
        eval_fn=_drift_d0_after_sub,
        config=Gen3ReineCortexCampaignConfig(
            campaign_id="gen3-s2-outer-ratchet-d2",
            enable_d5_intake=False,
            enable_d3_fusion=False,
            enable_d1_trophallaxis=False,
            enable_d4_mitosis_nf4=False,
            enforce_all_domain_guard=True,
            score_from=4,
        ),
        kronecker_target_deltas={"head.weight": torch.zeros_like(fix["model"].head.weight) + 1e-4},
        d2_config=relaxed_d2,
    )
    assert STAGE_D2_KRONECKER in rec_b["rejected_stages"]
    d2_entry = rec_b["stage_ledger"][2]
    assert d2_entry["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert d2_entry["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert d2_entry["rolled_back"] is True
    assert d2_entry["final_state_sha256"] == d2_entry["pre_state_sha256"]
    assert rec_b["final_state_sha256"] == rec_b["base_state_sha256"]
    assert verify_gen3_campaign_receipt_provenance(rec_b) is True

    # Non-improving D2 candidate using default d2_config=None rejected with ROLLED_BACK_ENERGY_NON_IMPROVING
    _, rec_nonimp = execute_gen3_five_direction_campaign(
        fix["model"],
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        config=Gen3ReineCortexCampaignConfig(
            campaign_id="gen3-s2-nonimp-d2",
            enable_d5_intake=False,
            enable_d3_fusion=False,
            enable_d1_trophallaxis=False,
            enable_d4_mitosis_nf4=False,
            min_energy_improvement=10.0,  # forces outer ratchet non-improving rollback
            score_from=4,
            batch_size=16,
        ),
        kronecker_expert_indices=(5,),
    )
    assert rec_nonimp["stage_ledger"][2]["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"

    # Flat eval_fn where D2 energy does not improve -> ROLLED_BACK_SUBSTAGE_REJECTED (guard intact)
    _, rec_sub_rej = execute_gen3_five_direction_campaign(
        fix["model"],
        probe_chunks=fix["probe_chunks"],
        eval_fn=lambda _m: {"E0_general_prose": 2.0, "E1_code_python": 2.4},
        config=Gen3ReineCortexCampaignConfig(
            campaign_id="gen3-s2-subrej-d2",
            enable_d5_intake=False,
            enable_d3_fusion=False,
            enable_d1_trophallaxis=False,
            enable_d4_mitosis_nf4=False,
            score_from=4,
        ),
        kronecker_target_deltas={"head.weight": torch.zeros_like(fix["model"].head.weight) + 1e-4},
        d2_config=KroneckerRiemannianTransportConfig(
            max_kl_nats=0.05,
            require_energy_improvement=True,
            enable_trust_backtracking=False,
            score_from=4,
            inplace=True,
        ),
    )
    assert rec_sub_rej["stage_ledger"][2]["status"] == "ROLLED_BACK_SUBSTAGE_REJECTED"

    eval_calls = {"n": 0}

    def _boom_on_second_eval(_m: nn.Module) -> Any:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return (None, {"E0_general_prose": 2.0, "E1_code_python": 2.4})
        if eval_calls["n"] == 2:
            return {"E0_general_prose": 2.0, "E1_code_python": 2.4}
        if eval_calls["n"] == 3:
            return {"E0_general_prose": 1.99, "E1_code_python": 2.05}
        raise RuntimeError("simulated eval exception")

    _, rec_boom = execute_gen3_five_direction_campaign(
        fix["model"],
        probe_chunks=fix["probe_chunks"],
        eval_fn=_boom_on_second_eval,
        config=Gen3ReineCortexCampaignConfig(
            campaign_id="gen3-s2-eval-err",
            enable_d5_intake=False,
            enable_d3_fusion=False,
            enable_d1_trophallaxis=False,
            enable_d4_mitosis_nf4=False,
            score_from=4,
        ),
        kronecker_target_deltas={"head.weight": torch.zeros_like(fix["model"].head.weight) + 1e-4},
        d2_config=KroneckerRiemannianTransportConfig(
            max_kl_nats=0.05,
            require_energy_improvement=False,
            enable_trust_backtracking=False,
            score_from=4,
            inplace=True,
        ),
    )
    assert rec_boom["stage_ledger"][2]["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert rec_boom["stage_ledger"][2]["anse_energy"] == ANSE_MAXIMUM_PAIN


# ---------------------------------------------------------------------------
# Scenario S3: Deterministic SHA-256 Merkle Chain Reproducibility & Tamper Audit
# ---------------------------------------------------------------------------


def test_s3_deterministic_merkle_stage_chain_reproducibility_and_tamper_detection(
    cached_lean_audit: Any,
) -> None:
    """(S3) Prove repeated runs yield bit-exact SHA-256 Merkle chains and detect tampering."""
    fix = _build_gen3_five_direction_fixture(cached_lean_audit, seed=20264930)
    troph_rounds = [
        {
            "forager_bee_deltas": fix["forager_bee_deltas"],
            "bee_domains": ["E4_cyber_sec"] * len(fix["forager_bee_deltas"]),
        }
    ]
    cfg = Gen3ReineCortexCampaignConfig(
        campaign_id="gen3-s3-repro",
        round_id="round_1",
        score_from=4,
        batch_size=16,
        inplace=False,
    )
    d5_cfg = ZKWSPFederatedHiveConfig(num_byzantine_f=2, score_from=4, batch_size=16, inplace=False)
    d2_cfg = KroneckerRiemannianTransportConfig(
        target_factors=("w3_B",), max_kl_nats=0.05, score_from=4, batch_size=16, inplace=True
    )
    d3_cfg = OptimalTransportMoEFuserConfig(calibrate_layernorm=False, score_from=4, batch_size=16, inplace=False)
    d4_cfg = AlveoleMitosisNF4Config(
        nf4_block_size=64,
        compensation_rank=4,
        num_alternating_steps=1,
        prose_hessian_weight=16.0,
        quantize_lm_head=False,
        projected_target_param_count=3_300_000_000,
        enforce_all_domain_guard=False,
        score_from=4,
        batch_size=16,
        inplace=False,
    )

    kwargs = {
        "probe_chunks": fix["probe_chunks"],
        "eval_chunks": fix["eval_chunks"],
        "config": cfg,
        "hive_packets": fix["hive_packets"],
        "subspace_commitment": fix["subspace_commitment"],
        "lean_spec_audit": cached_lean_audit,
        "d5_config": d5_cfg,
        "cross_arm_model": fix["cross_arm_model"],
        "d3_config": d3_cfg,
        "kronecker_expert_indices": (5,),
        "d2_config": d2_cfg,
        "trophallaxis_rounds": troph_rounds,
        "d4_config": d4_cfg,
    }

    m1, rec1 = execute_gen3_five_direction_campaign(copy.deepcopy(fix["model"]), **kwargs)
    m2, rec2 = execute_gen3_five_direction_campaign(copy.deepcopy(fix["model"]), **kwargs)

    assert rec1["accepted"] is True
    assert rec1["accepted_stages"] == list(STAGE_ORDER)
    assert rec1["final_state_sha256"] == rec2["final_state_sha256"]
    assert rec1["merkle_root_sha256"] == rec2["merkle_root_sha256"]
    assert rec1["chain_head_sha256"] == rec2["chain_head_sha256"]
    assert rec1["stage_merkle_digests"] == rec2["stage_merkle_digests"]
    assert rec1["receipt_sha256"] == rec2["receipt_sha256"]
    assert compute_state_sha256(m1) == compute_state_sha256(m2)

    audit1 = verify_gen3_campaign_merkle_chain(rec1, raise_on_error=True)
    assert audit1["valid"] is True
    assert audit1["stage_count"] == 5

    # Merkle root check across odd/even digest cardinalities
    d_list = rec1["stage_merkle_digests"]
    assert compute_merkle_root_sha256([d_list[0]]) == d_list[0]
    assert len(compute_merkle_root_sha256(d_list[:3])) == 64

    # Tampering detection across receipt, stage digests, prev link, state SHA, directive, and root
    tamper_cases: list[dict[str, Any]] = []
    t_obj = copy.deepcopy(rec1)
    t_obj["object"] = "bad.object"
    tamper_cases.append(t_obj)

    t_sch = copy.deepcopy(rec1)
    t_sch["schema"] = "bad/schema"
    tamper_cases.append(t_sch)

    t_rsha = copy.deepcopy(rec1)
    t_rsha["receipt_sha256"] = "0" * 64
    tamper_cases.append(t_rsha)

    t_rbad = copy.deepcopy(rec1)
    t_rbad["receipt_sha256"] = "bad"
    tamper_cases.append(t_rbad)

    t_stage = copy.deepcopy(rec1)
    t_stage["stage_ledger"][2]["stage_merkle_sha256"] = "f" * 64
    tamper_cases.append(t_stage)

    t_prev = copy.deepcopy(rec1)
    t_prev["stage_ledger"][1]["prev_stage_merkle_sha256"] = GENESIS_MERKLE_SHA256
    tamper_cases.append(t_prev)

    t_state = copy.deepcopy(rec1)
    t_state["final_state_sha256"] = "a" * 64
    tamper_cases.append(t_state)

    t_dir = copy.deepcopy(rec1)
    t_dir["d1_pheromone_directive"]["round_index"] = 999
    tamper_cases.append(t_dir)

    t_root = copy.deepcopy(rec1)
    t_root["merkle_root_sha256"] = "b" * 64
    tamper_cases.append(t_root)

    t_eobj = copy.deepcopy(rec1)
    t_eobj["stage_ledger"][0]["object"] = "bad.entry"
    tamper_cases.append(t_eobj)

    t_eidx = copy.deepcopy(rec1)
    t_eidx["stage_ledger"][0]["stage_index"] = 99
    tamper_cases.append(t_eidx)

    t_epre = copy.deepcopy(rec1)
    t_epre["stage_ledger"][1]["pre_state_sha256"] = "0" * 64
    tamper_cases.append(t_epre)

    t_esbad = copy.deepcopy(rec1)
    t_esbad["stage_ledger"][0]["stage_merkle_sha256"] = "not-hex"
    tamper_cases.append(t_esbad)

    t_enan = copy.deepcopy(rec1)
    t_enan["stage_ledger"][0]["substage_summary"] = {"bad": float("nan")}
    tamper_cases.append(t_enan)

    t_edrift = copy.deepcopy(rec1)
    t_edrift["stage_ledger"][0]["general_drift_ratio"] = 2.5
    tamper_cases.append(t_edrift)

    t_eup = copy.deepcopy(rec1)
    t_eup["stage_ledger"][0]["final_anse_energy"] = 1.0
    tamper_cases.append(t_eup)

    t_head = copy.deepcopy(rec1)
    t_head["chain_head_sha256"] = "c" * 64
    tamper_cases.append(t_head)

    t_digs = copy.deepcopy(rec1)
    t_digs["stage_merkle_digests"] = []
    tamper_cases.append(t_digs)

    for bad_rec in tamper_cases:
        assert verify_gen3_campaign_receipt_provenance(bad_rec) is False
    with pytest.raises(ValueError, match="Gen-3 Merkle provenance audit failed"):
        verify_gen3_campaign_merkle_chain(t_root, raise_on_error=True)


# ---------------------------------------------------------------------------
# Scenario S4: Full Pipeline Atomic Bitwise Rollback When All Stages Fail Guard
# ---------------------------------------------------------------------------


def test_s4_full_pipeline_atomic_rollback_when_every_stage_fails_guard(
    cached_lean_audit: Any,
) -> None:
    """(S4) Enforce full atomic bitwise rollback when every direction fails guard or budget."""
    fix = _build_gen3_five_direction_fixture(cached_lean_audit, seed=20264940)
    work_model = copy.deepcopy(fix["model"])
    base_snapshot = {k: v.detach().clone() for k, v in work_model.state_dict().items()}
    base_sha = compute_state_sha256(work_model)

    # All-Byzantine packets (n=6 >= 2*1+3) -> D5 all-hives rejected
    byz_only_packets = [
        p for p in fix["hive_packets"] if p["hive_id"].startswith("hive_byz_")
    ] * 3
    byz_packets_unique = []
    for idx, pkt in enumerate(byz_only_packets):
        cloned_pkt = copy.deepcopy(pkt)
        cloned_pkt["hive_id"] = f"byz_peer_{idx}"
        byz_packets_unique.append(cloned_pkt)

    # Incompatible Arm_B (orthogonal attention & MLP corruption -> CKA < 0.65) -> D3 CKA rejection
    bad_arm_b = copy.deepcopy(fix["cross_arm_model"])
    gen_bad = torch.Generator(device="cpu").manual_seed(77)
    with torch.no_grad():
        bad_arm_b.wte.weight.normal_(mean=0.0, std=2.5, generator=gen_bad)
        bad_arm_b.blocks[0].attn.o.weight.normal_(mean=0.0, std=1.5, generator=gen_bad)
        bad_arm_b.blocks[0].mlp.base_mlp.w1.weight.normal_(mean=0.0, std=1.5, generator=gen_bad)
        bad_arm_b.blocks[0].mlp.base_mlp.w2.weight.normal_(mean=0.0, std=1.5, generator=gen_bad)

    # NaN Forager Bee delta -> D1 Maximum Pain rollback
    base_blocks = extract_checkpoint_blocks(work_model)
    nan_bees = [
        {
            "blocks.0.mlp.experts.4": {
                "w3_B": torch.full_like(base_blocks["blocks.0.mlp.experts.4"]["w3_B"], float("nan"))
            }
        }
    ]

    # Adversarial D2 inside P_0 breaching E1 domain guard + unreachable D4 WDDM ceiling (100 MiB)
    adv_d2 = torch.zeros_like(work_model.head.weight)
    adv_d2[9, :] = -3.0 * fix["full_axes_out"][1]
    adv_d2[10, :] = -3.0 * fix["full_axes_out"][1]

    cfg_fail = Gen3ReineCortexCampaignConfig(
        campaign_id="gen3-s4-full-rollback",
        round_id="round_1",
        enforce_all_domain_guard=True,
        wddm_vram_ceiling_mib=100.0,
        enforce_wddm_budget=True,
        projected_target_param_count=3_300_000_000,
        score_from=4,
        batch_size=16,
        inplace=True,
    )
    out_model, rec_fail = execute_gen3_five_direction_campaign(
        work_model,
        probe_chunks=fix["probe_chunks"],
        eval_chunks=fix["eval_chunks"],
        config=cfg_fail,
        hive_packets=byz_packets_unique,
        subspace_commitment=fix["subspace_commitment"],
        lean_spec_audit=cached_lean_audit,
        cross_arm_model=bad_arm_b,
        d3_config=OptimalTransportMoEFuserConfig(
            min_cka_gate=0.95,
            calibrate_layernorm=False,
            score_from=4,
            batch_size=16,
            inplace=False,
        ),
        kronecker_target_deltas={"head.weight": adv_d2},
        d2_config=KroneckerRiemannianTransportConfig(
            max_kl_nats=2.0,
            domain_guard_max_ratio=1.01,
            enforce_all_domain_guard=True,
            require_energy_improvement=False,
            enable_trust_backtracking=False,
            protect_general_expert=False,
            score_from=4,
            inplace=True,
        ),
        forager_bee_deltas=nan_bees,
    )

    assert out_model is work_model
    assert rec_fail["accepted"] is False
    assert rec_fail["all_stages_rolled_back"] is True
    assert rec_fail["full_rollback_verified"] is True
    assert rec_fail["accepted_stages"] == []
    assert len(rec_fail["rejected_stages"]) == 5
    assert rec_fail["base_state_sha256"] == base_sha
    assert rec_fail["final_state_sha256"] == base_sha
    assert rec_fail["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_fail["wddm_footprint"] is None
    assert rec_fail["d1_pheromone_directive"] is None
    assert rec_fail["stage_ledger"][4]["status"] == "ROLLED_BACK_WDDM_VRAM_EXCEEDED"
    for entry in rec_fail["stage_ledger"]:
        assert entry["accepted"] is False
        assert entry["rolled_back"] is True
        assert entry["rollback_bitwise_verified"] is True
        assert entry["final_state_sha256"] == base_sha
    for k, orig_t in base_snapshot.items():
        assert torch.equal(out_model.state_dict()[k], orig_t), f"Bitwise drift on {k}"
    assert verify_gen3_campaign_receipt_provenance(rec_fail) is True
    t_unacc = copy.deepcopy(rec_fail)
    t_unacc["stage_ledger"][0]["final_state_sha256"] = "f" * 64
    assert verify_gen3_campaign_receipt_provenance(t_unacc) is False


# ---------------------------------------------------------------------------
# Scenario S5: Validation Errors & Security / Stub / OwnerOrderGuardrail Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_owner_guardrail_gates(
    cached_lean_audit: Any,
) -> None:
    """(S5) Verify input validation plus AGPL-3.0, zero-stub, security & guardrail gates."""
    # 1. Config & Merkle validation errors
    with pytest.raises(ValueError, match="campaign_id"):
        Gen3ReineCortexCampaignConfig(campaign_id=" ")
    with pytest.raises(TypeError, match="bool"):
        Gen3ReineCortexCampaignConfig(enable_d5_intake="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        Gen3ReineCortexCampaignConfig(max_general_drift_ratio=1.005, anse_safe_general_ratio=1.009)
    with pytest.raises(ValueError, match="general_prefixes"):
        Gen3ReineCortexCampaignConfig(general_prefixes=())
    with pytest.raises(ValueError, match="digests"):
        compute_merkle_root_sha256([])
    with pytest.raises(ValueError, match="64-char"):
        compute_merkle_root_sha256(["not-a-sha"])

    # 2. Orchestrator input & sub-config validation errors
    lin = nn.Linear(8, 4)
    with pytest.raises(TypeError, match="nn.Module"):
        execute_gen3_five_direction_campaign("bad", eval_fn=lambda _m: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        execute_gen3_five_direction_campaign(nn.ReLU(), eval_fn=lambda _m: {"E0": 1.0})
    with pytest.raises(TypeError, match="callable"):
        execute_gen3_five_direction_campaign(lin, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Provide at least one"):
        execute_gen3_five_direction_campaign(lin)
    with pytest.raises(TypeError, match="Gen3ReineCortexCampaignConfig"):
        execute_gen3_five_direction_campaign(lin, eval_fn=lambda _m: {"E0": 1.0}, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="eval_fn or eval_chunks"):
        execute_gen3_five_direction_campaign(lin, probe_chunks=[{"domain": "E0", "ids": [0, 1, 2, 3]}])

    fix = _build_gen3_five_direction_fixture(cached_lean_audit, seed=20264950)
    with pytest.raises(TypeError, match="ZKWSPFederatedHiveConfig"):
        execute_gen3_five_direction_campaign(
            fix["model"],
            probe_chunks=fix["probe_chunks"],
            hive_packets=fix["hive_packets"],
            d5_config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="OptimalTransportMoEFuserConfig"):
        execute_gen3_five_direction_campaign(
            fix["model"],
            probe_chunks=fix["probe_chunks"],
            config=Gen3ReineCortexCampaignConfig(enable_d5_intake=False, score_from=4),
            cross_arm_model=fix["cross_arm_model"],
            d3_config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="KroneckerRiemannianTransportConfig"):
        execute_gen3_five_direction_campaign(
            fix["model"],
            probe_chunks=fix["probe_chunks"],
            config=Gen3ReineCortexCampaignConfig(enable_d5_intake=False, enable_d3_fusion=False, score_from=4),
            d2_config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="InterRoundTrophallaxisConfig"):
        execute_gen3_five_direction_campaign(
            fix["model"],
            probe_chunks=fix["probe_chunks"],
            config=Gen3ReineCortexCampaignConfig(
                enable_d5_intake=False, enable_d3_fusion=False, enable_d2_kronecker=False, score_from=4
            ),
            forager_bee_deltas=fix["forager_bee_deltas"],
            d1_config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="AlveoleMitosisNF4Config"):
        execute_gen3_five_direction_campaign(
            fix["model"],
            probe_chunks=fix["probe_chunks"],
            config=Gen3ReineCortexCampaignConfig(
                enable_d5_intake=False,
                enable_d3_fusion=False,
                enable_d2_kronecker=False,
                enable_d1_trophallaxis=False,
                score_from=4,
            ),
            d4_config="bad",  # type: ignore[arg-type]
        )

    with pytest.raises(TypeError, match="Mapping"):
        verify_gen3_campaign_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Mapping"):
        verify_gen3_campaign_merkle_chain("bad")  # type: ignore[arg-type]
    # Domain-key mismatch inside eval_fn triggers ROLLED_BACK_EVAL_FAILURE cleanly
    dom_calls = {"n": 0}

    def _mismatch_dom_eval(_m: nn.Module) -> dict[str, float]:
        dom_calls["n"] += 1
        if dom_calls["n"] <= 2:
            return {"E0_general_prose": 2.0, "E1_code_python": 2.0}
        return {"E0_general_prose": 2.0, "E2_math_reasoning": 2.0}

    _, rec_dom_mis = execute_gen3_five_direction_campaign(
        fix["model"],
        probe_chunks=fix["probe_chunks"],
        eval_fn=_mismatch_dom_eval,
        config=Gen3ReineCortexCampaignConfig(
            campaign_id="gen3-s5-dom-mismatch",
            enable_d5_intake=False,
            enable_d3_fusion=False,
            enable_d1_trophallaxis=False,
            enable_d4_mitosis_nf4=False,
            score_from=4,
        ),
        kronecker_target_deltas={"head.weight": torch.zeros_like(fix["model"].head.weight) + 1e-4},
        d2_config=KroneckerRiemannianTransportConfig(
            max_kl_nats=0.05,
            require_energy_improvement=False,
            enable_trust_backtracking=False,
            score_from=4,
            inplace=True,
        ),
    )
    assert rec_dom_mis["stage_ledger"][2]["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert verify_gen3_campaign_receipt_provenance({"stage_ledger": []}) is False
    assert (
        verify_gen3_campaign_receipt_provenance(
            {
                "object": GEN3_CAMPAIGN_RECEIPT_OBJECT,
                "schema": GEN3_ORCHESTRATOR_SCHEMA,
                "receipt_sha256": "0" * 64,
                "bad_float": float("nan"),
                "stage_ledger": ["bad"],
            }
        )
        is False
    )

    # 3. AGPL-3.0, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn49", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn49", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/reine_cortex_gen3_orchestrator.py", TARGET_MODULE_PATH),
        ("tests/test_reine_cortex_gen3_orchestrator_scenarios.py", TEST_FILE_PATH),
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
        assert "NEURAL-CTRL-DYN-49" in doc_text
        assert "PILOT14" in doc_text
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
        {"nll_mean": 5.55528, "ece_value": 0.00585, "general_drift_ratio": 0.99752, "anse_energy": -0.01440},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): Gen-3 Reine-Cortex five-directions (D1-D5) end-to-end unified hive campaign orchestrator"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-49 unified Gen-3 D5->D3->D2->D1->D4 stage-ratcheted "
        "Reine-Cortex Multi-Hive Campaign Orchestrator enforcing ANSE Delta E < 0, D0 <= +1.0%, "
        "<= 6656 MiB WDDM ceiling, per-stage atomic bitwise rollback, and chained SHA-256 Merkle provenance."
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
        pr_number=1249,
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
