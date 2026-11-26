# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-46: dynamic Alveole mitosis and prose-Hessian NF4 exporter."""

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
import torch.nn.functional as F
from hivemind.security.owner_order_guardrail import (
    RequestIntentCategory,
    classify_request_intent,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.alveole_mitosis_hessian_nf4_exporter import (
    ALVEOLE_MITOSIS_EXPORTER_SCHEMA,
    ALVEOLE_MITOSIS_RECEIPT_OBJECT,
    ANSE_MAXIMUM_PAIN,
    STD_8GB_WDDM_CEILING_MIB,
    AlveoleMitosisNF4Config,
    VRAMCeilingExceededError,
    compensate_nf4_with_prose_hessian,
    compute_diagonal_prose_hessian,
    compute_projected_wddm_vram_footprint,
    diagnose_alveole_corruptors_and_anchors,
    execute_verified_alveole_mitosis_and_nf4_export,
    export_model_to_compensated_nf4_edge,
    quantize_dequantize_nf4_blockwise,
    recycle_corruptor_alveoles_via_mitosis,
    verify_alveole_mitosis_nf4_receipt_provenance,
)
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import upcycle_crowd_v1_to_sparse_moe
from hivemind.swarm.tangent_task_vector_surgeon import compute_prose_nullspace_projector_scaled
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
    REPO_ROOT / "src" / "hivemind" / "swarm" / "alveole_mitosis_hessian_nf4_exporter.py"
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


def _make_zero_mean_ortho_rows(n_rows: int, dim: int, seed: int) -> torch.Tensor:
    """Create ``n_rows`` orthonormal vectors orthogonal to the all-ones vector in ``R^dim``."""
    assert n_rows < dim
    gen = torch.Generator(device="cpu").manual_seed(seed)
    ones = torch.ones((dim, 1), dtype=torch.float64)
    mat = torch.cat(
        [ones, torch.randn((dim, dim - 1), generator=gen, dtype=torch.float64)],
        dim=1,
    )
    q_mat, _ = torch.linalg.qr(mat)
    return q_mat[:, 1 : 1 + n_rows].T.to(torch.float32).contiguous()


def _install_domain_lora(
    exp_mod: Any,
    in_dir: torch.Tensor,
    out_dir: torch.Tensor,
    *,
    strength: float,
    inter_slice: slice,
    gate_scale: float = 0.85,
) -> None:
    """Install deterministic rank-1 SwiGLU LoRA mapping ``in_dir -> strength * out_dir``."""
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


def _build_corruptor_moe_fixture(
    *,
    seed: int = 20264600,
    hidden_size: int = 64,
    intermediate_size: int = 128,
    num_layers: int = 2,
    vocab_size: int = 64,
    num_experts: int = 8,
    lora_rank: int = 4,
) -> dict[str, Any]:
    """Build deterministic CrowdV1 MoE with helpful anchors and causal-knockout corruptors."""
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
    model = CrowdV1WithEngram(cfg)
    gen = torch.Generator(device="cpu").manual_seed(seed)

    domain_specs = (
        (0, "E0_general_fineweb", 0, 0, slice(0, block_h)),
        (1, "E1_python_systems_code", 1, block_v, slice(block_h, 2 * block_h)),
        (2, "E2_math_reasoning", 2, 2 * block_v, slice(2 * block_h, 3 * block_h)),
        (4, "E4_physics_stem", 3, 3 * block_v, slice(3 * block_h, 4 * block_h)),
    )

    axes_in: dict[int, torch.Tensor] = {}
    axes_out: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        head_w = 0.015 * torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.float32)

        for e_slot, _, b_idx, v_off, h_slice in domain_specs:
            basis = _make_zero_mean_ortho_rows(4, block_h, seed=seed + 50 + b_idx * 19)
            u_in, u_out = basis[0], basis[1]
            axes_in[e_slot] = u_in
            axes_out[e_slot] = u_out
            rows = _make_zero_mean_ortho_rows(block_h - 1, block_h, seed=seed + 170 + b_idx * 23)
            for t_i in range(block_v):
                vec = 0.25 * rows[t_i % int(rows.shape[0])] + 0.70 * u_in
                wte[v_off + t_i, h_slice] = vec - vec.mean()
            head_w[:, h_slice] = head_w[:, h_slice] - head_w[:, h_slice].mean(dim=1, keepdim=True)
            head_w[v_off + 2, h_slice] = 0.05 * u_in + 0.75 * u_out
            head_w[v_off + 3, h_slice] = 0.05 * u_in + 0.45 * basis[2]
            head_w[v_off + 4, h_slice] = 0.05 * u_in + 0.45 * basis[3]

        model.wte.weight.copy_(wte)
        model.head.weight.copy_(head_w)
        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln2.weight.fill_(1.0)
            blk.attn.q.weight.copy_(0.15 * torch.eye(hidden_size) + 0.06 * torch.randn((hidden_size, hidden_size), generator=gen))
            blk.attn.k.weight.copy_(0.06 * torch.randn((hidden_size // 2, hidden_size), generator=gen))
            blk.attn.k.weight[: hidden_size // 2, : hidden_size // 2].add_(0.15 * torch.eye(hidden_size // 2))
            blk.attn.v.weight.copy_(0.06 * torch.randn((hidden_size // 2, hidden_size), generator=gen))
            blk.attn.v.weight[: hidden_size // 2, : hidden_size // 2].add_(0.15 * torch.eye(hidden_size // 2))
            blk.attn.o.weight.copy_(0.12 * torch.eye(hidden_size) + 0.06 * torch.randn((hidden_size, hidden_size), generator=gen))
            blk.mlp.w1.weight.copy_(0.06 * torch.randn((intermediate_size, hidden_size), generator=gen))
            blk.mlp.w1.weight[:16, :16].add_(0.18 * torch.eye(16))
            blk.mlp.w2.weight.copy_(0.06 * torch.randn((intermediate_size, hidden_size), generator=gen))
            blk.mlp.w2.weight[:16, :16].add_(0.18 * torch.eye(16))
            blk.mlp.w3.weight.copy_(0.05 * torch.randn((hidden_size, intermediate_size), generator=gen))
            blk.mlp.w3.weight[:16, :16].add_(0.05 * torch.eye(16))
            # Keep expert LoRA intermediate slices clean in base w3 so base_diff stays aligned
            for lo_i in (0, 16, 32, 48, 64, 80):
                blk.mlp.w3.weight[:, lo_i : lo_i + 8].zero_()
            # Amplify E0 general-prose alignment in base_mlp so NF4 compensation on prose channels shines
            blk.mlp.w3.weight[:block_h, :4] += 0.25 * axes_out[0].unsqueeze(1)
            blk.mlp.w1.weight[:4, :block_h] += 0.30 * axes_in[0].unsqueeze(0)
            blk.mlp.w2.weight[:4, :block_h] += 0.30 * axes_in[0].unsqueeze(0)

    upcycle_crowd_v1_to_sparse_moe(
        model,
        num_experts=num_experts,
        top_k=2,
        lora_rank=lora_rank,
        routing_mode="sigma_topk",
        seed=seed + 77,
    )

    def _full(h_slice: slice, vec: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(hidden_size, dtype=torch.float32)
        out[h_slice] = vec
        return out

    with torch.no_grad():
        for blk in model.blocks:
            mlp = blk.mlp
            mlp.router.weight.zero_()
            for exp in mlp.experts.values():
                for attr in ("w1_B", "w2_B", "w3_B"):
                    getattr(exp, attr).zero_()

            # Helpful general anchor (slot 0) and strong specialist anchors (slots 1, 2)
            for e_slot, _, _, _, h_slice in domain_specs:
                mlp.router.weight[e_slot, h_slice] = 3.2 * axes_in[e_slot]

            _install_domain_lora(
                mlp.experts["0"],
                _full(slice(0, block_h), axes_in[0]),
                _full(slice(0, block_h), axes_out[0]),
                strength=0.45,
                inter_slice=slice(0, 8),
            )
            _install_domain_lora(
                mlp.experts["1"],
                _full(slice(block_h, 2 * block_h), axes_in[1]),
                _full(slice(block_h, 2 * block_h), axes_out[1]),
                strength=0.65,
                inter_slice=slice(16, 24),
            )
            _install_domain_lora(
                mlp.experts["2"],
                _full(slice(2 * block_h, 3 * block_h), axes_in[2]),
                _full(slice(2 * block_h, 3 * block_h), axes_out[2]),
                strength=0.65,
                inter_slice=slice(32, 40),
            )

            # Corruptor / Ghost-Sink slot 4 (physics_stem): wrong inverted direction + prose leakage!
            bad_in_4 = _full(slice(3 * block_h, 4 * block_h), axes_in[4])
            bad_out_4 = -0.95 * _full(slice(3 * block_h, 4 * block_h), axes_out[4]) - 0.65 * _full(
                slice(0, block_h), axes_out[0]
            )
            _install_domain_lora(
                mlp.experts["4"],
                bad_in_4 + 0.45 * _full(slice(0, block_h), axes_in[0]),
                bad_out_4,
                strength=0.60,
                inter_slice=slice(48, 56),
            )
            # Corruptor slot 6: hurts math & prose simultaneously
            mlp.router.weight[6, 2 * block_h : 3 * block_h] = 2.6 * axes_in[2]
            mlp.router.weight[6, 0:block_h] = 2.0 * axes_in[0]
            _install_domain_lora(
                mlp.experts["6"],
                _full(slice(2 * block_h, 3 * block_h), axes_in[2]) + 0.5 * _full(slice(0, block_h), axes_in[0]),
                -0.9 * _full(slice(2 * block_h, 3 * block_h), axes_out[2]) - 0.6 * _full(slice(0, block_h), axes_out[0]),
                strength=0.65,
                inter_slice=slice(64, 72),
            )

            # Anchor slot 7 carries helpful physics weights routed second behind slot 4 on E4_physics_stem
            _install_domain_lora(
                mlp.experts["7"],
                _full(slice(3 * block_h, 4 * block_h), axes_in[4]),
                _full(slice(3 * block_h, 4 * block_h), axes_out[4]),
                strength=1.35,
                inter_slice=slice(80, 88),
            )
            mlp.router.weight[7, 3 * block_h : 4 * block_h] = 2.2 * axes_in[4]

    def _make_chunks(split_seed: int, n_per_dom: int) -> list[dict[str, Any]]:
        g = torch.Generator(device="cpu").manual_seed(split_seed)
        chunks: list[dict[str, Any]] = []
        for _, dom_id, _, v_off, _ in domain_specs:
            for _c_i in range(n_per_dom):
                ctx = torch.randint(v_off, v_off + block_v, (8,), generator=g)
                tgt = torch.full((8,), v_off + 2, dtype=torch.long)
                chunks.append({"expert_id": dom_id, "ids": torch.cat([ctx, tgt], dim=0).tolist()})
        return chunks

    return {
        "model": model,
        "axes_in": axes_in,
        "axes_out": axes_out,
        "probe_chunks": _make_chunks(seed + 1111, 4),
        "eval_chunks": _make_chunks(seed + 2222, 4),
        "block_h": block_h,
    }


def _eval_crowd_mean_and_domains(
    model: CrowdV1WithEngram, chunks: list[dict[str, Any]], score_from: int = 8
) -> tuple[float, dict[str, float]]:
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


class _HFStyleRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.wg = nn.Linear(hidden_dim, num_experts, bias=False)


class _HFStyleMoEMLP(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.router = _HFStyleRouter(hidden_dim, num_experts)
        self.experts = nn.ModuleList([_HFStyleExpert(hidden_dim, inter_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig = x.shape
        xf = x.reshape(-1, self.hidden_dim)
        probs = torch.softmax(self.router.wg(xf), dim=-1)
        out = torch.zeros_like(xf)
        for idx, exp in enumerate(self.experts):
            out = out + probs[:, idx : idx + 1] * exp(xf)
        return out.reshape(orig)


class _HFStyleMoELayer(nn.Module):
    def __init__(self, hidden_dim: int, inter_dim: int, num_experts: int) -> None:
        super().__init__()
        self.mlp = _HFStyleMoEMLP(hidden_dim, inter_dim, num_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class _HFStyleMoELM(nn.Module):
    def __init__(self, vocab_size: int = 32, hidden_dim: int = 32, inter_dim: int = 64, num_experts: int = 4) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([_HFStyleMoELayer(hidden_dim, inter_dim, num_experts)])
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for lyr in self.layers:
            x = lyr(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Scenario S1: Causal-Knockout Corruptor Diagnosis & Dynamic Alveole Mitosis
# ---------------------------------------------------------------------------


def test_s1_causal_knockout_diagnosis_and_alveole_mitosis_recycling_gains() -> None:
    """(S1) Prove LOEO diagnosis isolates corruptors and P_0 Mitosis beats ZeroSink & Base."""
    fix = _build_corruptor_moe_fixture(seed=20264610)
    base_model: CrowdV1WithEngram = fix["model"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]

    base_mean, base_dom = _eval_crowd_mean_and_domains(base_model, eval_chunks)

    # 1. Diagnose corruptors via LOEO causal knockout
    diag = diagnose_alveole_corruptors_and_anchors(
        base_model,
        eval_fn=lambda m: _eval_crowd_mean_and_domains(m, probe_chunks),
        config=AlveoleMitosisNF4Config(
            corruptor_knockout_threshold=1e-4,
            prose_harm_threshold=1e-4,
            score_from=8,
        ),
    )
    assert 4 in diag["corruptor_slots"]
    assert 6 in diag["corruptor_slots"]
    assert 0 not in diag["corruptor_slots"]  # E0 protected
    assert diag["per_expert"]["4"]["knockout_improvement_nats"] > 0.0
    assert diag["per_expert"]["6"]["knockout_improvement_nats"] > 0.0
    assert diag["per_expert"]["7"]["knockout_improvement_nats"] < 0.0
    assert diag["high_deficit_domains"][0] == "E4_physics_stem"

    # 2. Naive ZeroSink baseline (merely zero out corruptors 4 and 6)
    zerosink_model = copy.deepcopy(base_model)
    with torch.no_grad():
        for blk in zerosink_model.blocks:
            for gs in (4, 6):
                exp = blk.mlp.experts[str(gs)]
                exp.w1_B.zero_()
                exp.w2_B.zero_()
                exp.w3_B.zero_()
    zs_mean, zs_dom = _eval_crowd_mean_and_domains(zerosink_model, eval_chunks)
    assert zs_mean < base_mean

    # 3. Dynamic Alveole Mitosis inside prose null-space P_0 (recycling corruptors -> physics_stem & math)
    ids_probe = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)
    gen_flags = torch.tensor([str(c["expert_id"]).startswith("E0_") for c in probe_chunks], dtype=torch.bool)
    acts_by_layer: dict[int, torch.Tensor] = {}
    hooks = [
        blk.mlp.register_forward_pre_hook(lambda _m, args, li=i: acts_by_layer.setdefault(li, args[0].detach().clone()))
        for i, blk in enumerate(base_model.blocks)
    ]
    with torch.no_grad():
        _ = base_model(ids_probe)
    for hk in hooks:
        hk.remove()

    p_null_by_layer: dict[int, torch.Tensor] = {}
    for li, x_l in acts_by_layer.items():
        x_prose = x_l[gen_flags, 7:-1, :].reshape(-1, x_l.shape[-1])
        p0, _ = compute_prose_nullspace_projector_scaled(x_prose, sv_rel_tol=1e-3, variance_threshold=0.999)
        p_null_by_layer[li] = p0

    mitosis_model = copy.deepcopy(base_model)
    plan = [
        {"corruptor_slot": 4, "anchor_slot": 7, "target_domain": "E4_physics_stem"},
        {"corruptor_slot": 6, "anchor_slot": 2, "target_domain": "E2_math_reasoning"},
    ]
    mit_report = recycle_corruptor_alveoles_via_mitosis(
        mitosis_model,
        plan,
        acts_probe_by_layer={li: x_l[:, 7:-1, :] for li, x_l in acts_by_layer.items()},
        domain_labels_probe=[str(c["expert_id"]) for c in probe_chunks],
        p_null_by_layer=p_null_by_layer,
        config=AlveoleMitosisNF4Config(
            project_mitosis_into_p0=True,
            project_router_into_p0=True,
            router_proto_blend=0.35,
            subspace_split_beta=0.15,
        ),
    )
    assert mit_report["num_recycled"] == 2
    assert mit_report["max_p0_prose_leak"] < 1e-10
    assert all(row["router_proto_aligned"] for row in mit_report["recycled_details"])

    # Verify Lean 4 RCD_D4 prose invariance on recycled experts
    for li, blk in enumerate(mitosis_model.blocks):
        p0 = p_null_by_layer[li].to(torch.float64)
        eye_p = torch.eye(p0.shape[0], dtype=torch.float64)
        v_proj = eye_p - p0  # orthogonal projector onto span(V_0)
        for slot in ("4", "6"):
            exp = blk.mlp.experts[slot]
            assert float((exp.w1_A.to(torch.float64) @ v_proj).abs().max().item()) < 1e-6
            assert float((exp.w2_A.to(torch.float64) @ v_proj).abs().max().item()) < 1e-6
            assert float((v_proj @ exp.w3_B.to(torch.float64)).abs().max().item()) < 1e-6

    mit_mean, mit_dom = _eval_crowd_mean_and_domains(mitosis_model, eval_chunks)
    assert mit_mean < base_mean
    assert mit_mean < zs_mean  # Mitosis strictly beats ZeroSink knockout only!
    assert mit_dom["E4_physics_stem"] < zs_dom["E4_physics_stem"]
    d0_ratio = mit_dom["E0_general_fineweb"] / base_dom["E0_general_fineweb"]
    assert d0_ratio <= 1.01


# ---------------------------------------------------------------------------
# Scenario S2: Prose-Hessian Compensated 4-Bit NF4 inside P_0 Error Reduction
# ---------------------------------------------------------------------------


def test_s2_nf4_prose_hessian_compensation_inside_p0_beats_naive_nf4() -> None:
    """(S2) Prove diagonal prose-Hessian NF4 inside P_0 contracts error & beats Naive NF4."""
    gen = torch.Generator(device="cpu").manual_seed(20264620)

    # 1. Exact bit-exact NF4 packing/dequantization with non-divisible cols & Hessian scale search
    w_raw = torch.randn((32, 73), generator=gen, dtype=torch.float32) * 0.25
    x_prose = torch.randn((40, 73), generator=gen, dtype=torch.float32)
    x_prose[:, :8] *= 8.0  # concentrated high-curvature general-prose coordinates
    h_diag = compute_diagonal_prose_hessian(x_prose, prose_weight=4.0, damping=1e-4)

    q_naive, diag_naive = quantize_dequantize_nf4_blockwise(
        w_raw, block_size=64, double_quant_scales=True, hessian_diag=None
    )
    q_hopt, diag_hopt = quantize_dequantize_nf4_blockwise(
        w_raw, block_size=64, double_quant_scales=True, hessian_diag=h_diag
    )
    assert diag_naive["packed_nibbles"].dtype == torch.uint8
    assert diag_hopt["q_scales_uint8"] is not None
    assert diag_hopt["hessian_error"] <= diag_hopt["naive_hessian_error"] + 1e-9
    naive_h_from_q = float(torch.linalg.matrix_norm((w_raw.to(torch.float64) - q_naive.to(torch.float64)) * h_diag.sqrt().unsqueeze(0), ord="fro").item())
    assert diag_hopt["hessian_error"] <= naive_h_from_q + 1e-9
    # Verify FP32 scale mode too
    _, diag_fp32s = quantize_dequantize_nf4_blockwise(
        w_raw, block_size=64, double_quant_scales=False, hessian_diag=h_diag
    )
    assert diag_fp32s["q_scales_uint8"] is None

    # 2. Prose-Hessian low-rank residual inside P_0: exact orthogonality + monotone contraction
    p0_proj, _ = compute_prose_nullspace_projector_scaled(
        x_prose.to(torch.float64), sv_rel_tol=1e-2, variance_threshold=0.95
    )
    q_w, l_b, l_a, comp_p0 = compensate_nf4_with_prose_hessian(
        w_raw,
        h_diag,
        rank=4,
        block_size=64,
        num_alternating_steps=3,
        double_quant_scales=True,
        p_null=p0_proj,
    )
    assert comp_p0["restricted_to_p0"] is True
    assert comp_p0["residual_p0_leak"] < 1e-10
    assert comp_p0["compensated_weighted_err_fro"] < comp_p0["naive_weighted_err_fro"]
    assert comp_p0["weighted_error_reduction_frac"] > 0.05
    # Verify monotone step sequence
    errs = comp_p0["step_weighted_errors"]
    assert all(errs[i + 1] <= errs[i] + 1e-10 for i in range(len(errs) - 1))

    # Full-space (shared trunk) diagonal prose-Hessian compensation achieves deep prose error reduction
    _, _, _, comp_full = compensate_nf4_with_prose_hessian(
        w_raw, h_diag, rank=4, block_size=64, num_alternating_steps=3, p_null=None
    )
    assert comp_full["compensated_weighted_err_fro"] < 0.70 * comp_full["naive_weighted_err_fro"]
    assert comp_full["compression_ratio_vs_fp32"] > 3.0
    w_large = torch.randn((96, 192), generator=gen, dtype=torch.float32) * 0.2
    _, _, _, comp_large = compensate_nf4_with_prose_hessian(
        w_large, torch.ones(192), rank=4, block_size=64, num_alternating_steps=2
    )
    assert comp_large["compression_ratio_vs_fp32"] > 5.0

    # 3. End-to-end model comparison on post-Mitosis clean model: Compensated NF4 beats Naive NF4
    fix = _build_corruptor_moe_fixture(seed=20264625)
    base_m: CrowdV1WithEngram = fix["model"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]
    ids_probe = torch.tensor([c["ids"] for c in probe_chunks], dtype=torch.long)
    gen_mask = [str(c["expert_id"]).startswith("E0_") for c in probe_chunks]

    with torch.no_grad():
        for blk in base_m.blocks:
            blk.attn.o.weight[:, -1] += 1.8
            blk.mlp.base_mlp.w1.weight[:, -1] += 1.8
            blk.mlp.base_mlp.w3.weight[:, -1] += 1.8
        base_m.head.weight[:, -1] += 2.2

    recycle_corruptor_alveoles_via_mitosis(
        base_m,
        [
            {"corruptor_slot": 4, "anchor_slot": 7, "target_domain": "E4_physics_stem"},
            {"corruptor_slot": 6, "anchor_slot": 2, "target_domain": "E2_math_reasoning"},
        ],
    )
    clean_mean, _ = _eval_crowd_mean_and_domains(base_m, eval_chunks)

    # Build Naive NF4 clone (only NF4 RTN, rank=0 proxy via quantize_dequantize_nf4_blockwise)
    naive_nf4_m = copy.deepcopy(base_m)
    with torch.no_grad():
        for name, mod in naive_nf4_m.named_modules():
            if isinstance(mod, nn.Linear) and "router" not in name and mod.weight.numel() >= 128:
                qn, _ = quantize_dequantize_nf4_blockwise(mod.weight, block_size=64)
                mod.weight.copy_(qn)

    comp_nf4_m = copy.deepcopy(base_m)
    exp_report = export_model_to_compensated_nf4_edge(
        comp_nf4_m,
        calibration_ids=ids_probe,
        general_mask=gen_mask,
        config=AlveoleMitosisNF4Config(
            nf4_block_size=64,
            compensation_rank=4,
            num_alternating_steps=1,
            prose_hessian_weight=4.0,
            projected_target_param_count=3_300_000_000,
        ),
    )
    assert exp_report["overall_weighted_error_reduction_frac"] > 0.25
    assert exp_report["wddm_footprint"]["fits_wddm_ceiling"] is True
    assert exp_report["wddm_footprint"]["projected_target_wddm_vram_mib"] <= STD_8GB_WDDM_CEILING_MIB

    naive_mean, naive_dom = _eval_crowd_mean_and_domains(naive_nf4_m, eval_chunks)
    comp_mean, comp_dom = _eval_crowd_mean_and_domains(comp_nf4_m, eval_chunks)
    assert comp_mean < naive_mean
    assert abs(comp_mean - clean_mean) < abs(naive_mean - clean_mean)
    assert comp_dom["E0_general_fineweb"] <= naive_dom["E0_general_fineweb"] + 1e-4


# ---------------------------------------------------------------------------
# Scenario S3: End-to-End Verified Mitosis + NF4 Export & WDDM Envelope
# ---------------------------------------------------------------------------


def test_s3_end_to_end_verified_mitosis_nf4_export_crowd_and_hf_moe() -> None:
    """(S3) Prove ANSE Delta E < 0, D0 <= +1%, WDDM <= 6656 MiB, and valid SHA-256 receipts."""
    fix = _build_corruptor_moe_fixture(seed=20264630)
    base_model: CrowdV1WithEngram = fix["model"]
    probe_chunks = fix["probe_chunks"]
    eval_chunks = fix["eval_chunks"]

    cfg = AlveoleMitosisNF4Config(
        nf4_block_size=64,
        compensation_rank=4,
        num_alternating_steps=1,
        projected_target_param_count=3_300_000_000,
        wddm_vram_ceiling_mib=float(STD_8GB_WDDM_CEILING_MIB),
        score_from=8,
        batch_size=16,
        inplace=False,
    )
    exported_model, receipt = execute_verified_alveole_mitosis_and_nf4_export(
        base_model,
        calibration_chunks=probe_chunks,
        eval_chunks=eval_chunks,
        config=cfg,
    )
    assert receipt["status"] == "ACCEPTED"
    assert receipt["accepted"] is True
    assert receipt["rolled_back"] is False
    assert receipt["object"] == ALVEOLE_MITOSIS_RECEIPT_OBJECT
    assert receipt["schema"] == ALVEOLE_MITOSIS_EXPORTER_SCHEMA
    assert verify_alveole_mitosis_nf4_receipt_provenance(receipt) is True

    # Strictly negative ANSE energy & general prose <= +1.0%
    assert receipt["candidate_eval"]["anse_energy"] < receipt["parent_eval"]["anse_energy"]
    assert receipt["candidate_eval"]["general_pct_vs_ref"] <= 1.0
    assert receipt["candidate_eval"]["worst_domain_pct_vs_ref"] <= 1.0
    assert receipt["mitosis"]["num_recycled"] >= 1
    assert receipt["mitosis"]["max_p0_prose_leak"] < 1e-10
    assert receipt["bos_sink_abs_drift"] <= cfg.max_bos_sink_drift

    wddm = receipt["nf4_export"]["wddm_footprint"]
    assert wddm["fits_wddm_ceiling"] is True
    assert wddm["compression_ratio_vs_fp32"] > 4.0
    assert wddm["projected_target_wddm_vram_mib"] <= 6656.0
    assert receipt["parent_state_sha256"] != receipt["final_state_sha256"]
    # Input base_model untouched when inplace=False
    assert exported_model is not base_model

    # Tamper detection on receipt
    tampered = dict(receipt)
    tampered["status"] = "FORGED"
    assert verify_alveole_mitosis_nf4_receipt_provenance(tampered) is False

    # Also test CrowdFineGrainedMoEMLP + HF-style MoE model execution
    fg_cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    fg_model = CrowdV1WithEngram(fg_cfg)
    upcycle_crowd_v1_to_sparse_moe(
        fg_model, num_experts=4, top_k=2, split_mode="fine_grained_width_split", seed=20264631
    )
    hf_model = _HFStyleMoELM(vocab_size=32, hidden_dim=32, inter_dim=64, num_experts=4)
    g_fg = torch.Generator(device="cpu").manual_seed(20264632)
    fg_chunks = [
        {"domain": "general_chat" if i < 2 else "physics_sim", "ids": torch.randint(0, 32, (12,), generator=g_fg).tolist()}
        for i in range(4)
    ]
    eval_calls = {"n": 0}

    def _fg_eval(_m: nn.Module) -> dict[str, float]:
        eval_calls["n"] += 1
        if eval_calls["n"] == 1:
            return {"general_chat": 2.0, "physics_sim": 2.4}
        if eval_calls["n"] <= 4:
            # Slot 2 knocks out positively (corruptor)
            return {"general_chat": 1.98, "physics_sim": 2.25} if eval_calls["n"] == 4 else {"general_chat": 2.02, "physics_sim": 2.50}
        return {"general_chat": 1.99, "physics_sim": 2.15}

    _, fg_rec = execute_verified_alveole_mitosis_and_nf4_export(
        fg_model,
        calibration_chunks=fg_chunks,
        eval_fn=_fg_eval,
        config=AlveoleMitosisNF4Config(
            nf4_block_size=32,
            compensation_rank=2,
            num_alternating_steps=2,
            score_from=4,
            min_linear_numel_to_quantize=64,
            max_bos_sink_drift=0.25,
        ),
    )
    assert fg_rec["status"] == "ACCEPTED"

    hf_calls = {"n": 0}

    def _hf_eval(_m: nn.Module) -> dict[str, float]:
        hf_calls["n"] += 1
        return {"E0_gen": 1.80, "E1_spec": 2.10} if hf_calls["n"] == 1 else {"E0_gen": 1.79, "E1_spec": 1.95}

    _, hf_rec = execute_verified_alveole_mitosis_and_nf4_export(
        hf_model,
        calibration_chunks=fg_chunks,
        eval_fn=_hf_eval,
        explicit_knockout_deltas={2: {"knockout_improvement_nats": 0.12, "prose_harm_nats": 0.02}},
        config=AlveoleMitosisNF4Config(
            nf4_block_size=32,
            compensation_rank=2,
            num_alternating_steps=2,
            score_from=4,
            min_linear_numel_to_quantize=64,
        ),
    )
    assert hf_rec["status"] == "ACCEPTED"


# ---------------------------------------------------------------------------
# Scenario S4: Atomic Bitwise Rollback on Guard, WDDM, BOS & Eval Failures
# ---------------------------------------------------------------------------


def test_s4_atomic_bitwise_rollback_paths_under_anse_maximum_pain() -> None:
    """(S4) Prove atomic bitwise rollback on D0 drift, WDDM breach, BOS collapse & eval errors."""
    rb_cfg = CrowdV1RuntimeConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=64,
        max_position_embeddings=16,
        engram_buckets=0,
        tie_word_embeddings=False,
    )
    base_model = CrowdV1WithEngram(rb_cfg)
    upcycle_crowd_v1_to_sparse_moe(base_model, num_experts=4, top_k=2, lora_rank=2, seed=20264640)
    g_rb = torch.Generator(device="cpu").manual_seed(20264641)
    probe_chunks = [
        {"domain": "E0_general" if i < 2 else "E1_code", "ids": torch.randint(0, 32, (12,), generator=g_rb).tolist()}
        for i in range(4)
    ]
    base_state_copy = copy.deepcopy(base_model.state_dict())

    def _assert_bitwise_restored(m: nn.Module) -> None:
        for k, v0 in base_state_copy.items():
            assert torch.equal(m.state_dict()[k].cpu(), v0.cpu()), f"Bitwise mismatch on {k}"

    # 1. General-prose guard violation (D0 > +1.0%) -> ROLLED_BACK_GUARD_VIOLATION
    call_g = {"n": 0}

    def _guard_fail_eval(_m: nn.Module) -> dict[str, float]:
        call_g["n"] += 1
        if call_g["n"] == 1:
            return {"E0_general": 2.00, "E1_code": 2.20}
        return {"E0_general": 2.05, "E1_code": 1.60}

    m_guard, rec_guard = execute_verified_alveole_mitosis_and_nf4_export(
        base_model,
        calibration_chunks=probe_chunks,
        eval_fn=_guard_fail_eval,
        explicit_knockout_deltas={2: 0.05},
        config=AlveoleMitosisNF4Config(quantize_model_to_nf4=False, score_from=4, inplace=False),
    )
    assert rec_guard["status"] == "ROLLED_BACK_GUARD_VIOLATION"
    assert rec_guard["rolled_back"] is True
    assert rec_guard["final_eval"]["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_guard["final_state_sha256"] == rec_guard["parent_state_sha256"]
    _assert_bitwise_restored(m_guard)

    # 2. Non-improving ANSE energy -> ROLLED_BACK_ENERGY_NON_IMPROVING
    m_nonimp, rec_nonimp = execute_verified_alveole_mitosis_and_nf4_export(
        base_model,
        calibration_chunks=probe_chunks,
        eval_fn=lambda _m: {"E0_general": 2.00, "E1_code": 2.10},
        explicit_knockout_deltas={2: 0.05},
        config=AlveoleMitosisNF4Config(
            quantize_model_to_nf4=False, calibrate_layernorm=False, score_from=4, inplace=False
        ),
    )
    assert rec_nonimp["status"] == "ROLLED_BACK_ENERGY_NON_IMPROVING"
    assert rec_nonimp["final_state_sha256"] == rec_nonimp["parent_state_sha256"]
    _assert_bitwise_restored(m_nonimp)

    # 3. WDDM 6,656 MiB envelope exceeded -> ROLLED_BACK_WDDM_VRAM_EXCEEDED
    m_wddm, rec_wddm = execute_verified_alveole_mitosis_and_nf4_export(
        base_model,
        calibration_chunks=probe_chunks,
        eval_fn=lambda _m: {"E0_general": 2.00, "E1_code": 2.10},
        explicit_knockout_deltas={2: 0.05},
        config=AlveoleMitosisNF4Config(
            num_alternating_steps=1,
            compensation_rank=2,
            nf4_block_size=32,
            min_linear_numel_to_quantize=64,
            projected_target_param_count=12_000_000_000,
            wddm_vram_ceiling_mib=6656.0,
            score_from=4,
            inplace=False,
        ),
    )
    assert rec_wddm["status"] == "ROLLED_BACK_WDDM_VRAM_EXCEEDED"
    assert rec_wddm["final_eval"]["anse_energy"] == ANSE_MAXIMUM_PAIN
    assert rec_wddm["final_state_sha256"] == rec_wddm["parent_state_sha256"]
    _assert_bitwise_restored(m_wddm)

    # 4. BOS attention-sink collapse -> ROLLED_BACK_BOS_SINK_DRIFT
    bos_calls = {"n": 0}

    def _bos_collapse(_m: nn.Module) -> float:
        bos_calls["n"] += 1
        return 0.42 if bos_calls["n"] == 1 else 0.10

    m_bos, rec_bos = execute_verified_alveole_mitosis_and_nf4_export(
        base_model,
        calibration_chunks=probe_chunks,
        eval_fn=lambda _m: {"E0_general": 2.00, "E1_code": 2.10},
        explicit_knockout_deltas={2: 0.05},
        bos_sink_fn=_bos_collapse,
        config=AlveoleMitosisNF4Config(
            quantize_model_to_nf4=False, calibrate_layernorm=False, max_bos_sink_drift=0.05, score_from=4
        ),
    )
    assert rec_bos["status"] == "ROLLED_BACK_BOS_SINK_DRIFT"
    assert rec_bos["final_state_sha256"] == rec_bos["parent_state_sha256"]
    _assert_bitwise_restored(m_bos)

    # 5. Evaluation failure (e.g., NaN logits) -> ROLLED_BACK_EVAL_FAILURE
    err_calls = {"n": 0}

    def _boom_eval(_m: nn.Module) -> dict[str, float]:
        err_calls["n"] += 1
        if err_calls["n"] == 1:
            return {"E0_general": 2.00, "E1_code": 2.10}
        raise RuntimeError("simulated NaN logits during eval")

    m_err, rec_err = execute_verified_alveole_mitosis_and_nf4_export(
        base_model,
        calibration_chunks=probe_chunks,
        eval_fn=_boom_eval,
        explicit_knockout_deltas={2: 0.05},
        config=AlveoleMitosisNF4Config(quantize_model_to_nf4=False, calibrate_layernorm=False, score_from=4),
    )
    assert rec_err["status"] == "ROLLED_BACK_EVAL_FAILURE"
    assert "simulated NaN logits" in str(rec_err["eval_error"])
    assert rec_err["final_state_sha256"] == rec_err["parent_state_sha256"]
    _assert_bitwise_restored(m_err)


# ---------------------------------------------------------------------------
# Scenario S5: Validation Gates & Security / Zero-Stub / OwnerOrder Compliance
# ---------------------------------------------------------------------------


def test_s5_validation_gates_and_security_stub_owner_compliance() -> None:
    """(S5) Assert input validation, WDDM error bounds, zero stubs, zero egress & OwnerOrder."""
    # 1. Config & helper validation paths
    with pytest.raises(ValueError, match="nullspace_sv_rel_tol"):
        AlveoleMitosisNF4Config(nullspace_sv_rel_tol=1.0)
    with pytest.raises(ValueError, match="max_general_drift_ratio"):
        AlveoleMitosisNF4Config(max_general_drift_ratio=0.9)
    with pytest.raises(ValueError, match="anse_safe_general_ratio"):
        AlveoleMitosisNF4Config(anse_safe_general_ratio=0.95)
    with pytest.raises(ValueError, match="general_prefixes"):
        AlveoleMitosisNF4Config(general_prefixes=())
    with pytest.raises(ValueError, match="max_recycled_experts"):
        AlveoleMitosisNF4Config(max_recycled_experts=0)
    with pytest.raises(ValueError, match="projected_target_param_count"):
        AlveoleMitosisNF4Config(projected_target_param_count=0)

    with pytest.raises(TypeError, match="hessian_diag"):
        quantize_dequantize_nf4_blockwise(torch.ones((4, 4)), hessian_diag="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        quantize_dequantize_nf4_blockwise(
            torch.ones((4, 4)), hessian_diag=torch.tensor([1.0, float("nan"), 1.0, 1.0])
        )
    with pytest.raises(ValueError, match="shape"):
        quantize_dequantize_nf4_blockwise(torch.ones((4, 4)), hessian_diag=torch.ones(5))
    with pytest.raises(ValueError, match="non-negative"):
        quantize_dequantize_nf4_blockwise(torch.ones((4, 4)), hessian_diag=torch.tensor([1.0, -0.1, 1.0, 1.0]))

    with pytest.raises(ValueError, match="width"):
        compute_diagonal_prose_hessian(torch.ones((4, 4)), x_all=torch.ones((4, 5)))
    with pytest.raises(ValueError, match="eigenvalues"):
        compensate_nf4_with_prose_hessian(
            torch.ones((4, 4)), torch.ones(4), p_null=2.0 * torch.eye(4)
        )

    # Check 2D Hessian input acceptance
    _, h2d_diag = quantize_dequantize_nf4_blockwise(
        torch.randn((4, 8), generator=torch.Generator(device="cpu").manual_seed(20264651)),
        block_size=4,
        hessian_diag=torch.eye(8),
    )
    assert h2d_diag["hessian_error"] >= 0.0

    # Check WDDM calculation & ceiling breach
    wddm_ok = compute_projected_wddm_vram_footprint(4096, 1024, projected_target_param_count=3_300_000_000)
    assert wddm_ok["fits_wddm_ceiling"] is True
    assert wddm_ok["projected_target_wddm_vram_mib"] < STD_8GB_WDDM_CEILING_MIB
    wddm_over = compute_projected_wddm_vram_footprint(4096, 2048, projected_target_param_count=6_000_000_000)
    assert wddm_over["fits_wddm_ceiling"] is False
    assert isinstance(VRAMCeilingExceededError("over"), ValueError)

    hf_small = _HFStyleMoELM(vocab_size=16, hidden_dim=8, inter_dim=16, num_experts=2)
    with pytest.raises(TypeError, match="nn.Module"):
        diagnose_alveole_corruptors_and_anchors("bad", eval_fn=lambda _m: {"E0": 1.0})  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="callable"):
        diagnose_alveole_corruptors_and_anchors(hf_small, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layer_indices"):
        diagnose_alveole_corruptors_and_anchors(hf_small, eval_fn=lambda _m: {"E0": 1.0}, layer_indices=[9])
    with pytest.raises(ValueError, match="Missing required general-domain"):
        diagnose_alveole_corruptors_and_anchors(hf_small, eval_fn=lambda _m: {"E1_only": 1.0})

    with pytest.raises(TypeError, match="nn.Module"):
        recycle_corruptor_alveoles_via_mitosis("bad", [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        recycle_corruptor_alveoles_via_mitosis(hf_small, "bad")  # type: ignore[arg-type]
    empty_mit = recycle_corruptor_alveoles_via_mitosis(hf_small, [])
    assert empty_mit["num_recycled"] == 0
    with pytest.raises(ValueError, match="out of"):
        recycle_corruptor_alveoles_via_mitosis(
            hf_small, [{"corruptor_slot": 9, "anchor_slot": 0, "target_domain": "E1"}]
        )
    with pytest.raises(ValueError, match="P_0_x dim"):
        recycle_corruptor_alveoles_via_mitosis(
            hf_small,
            [{"corruptor_slot": 1, "anchor_slot": 0, "target_domain": "E1"}],
            p_null_by_layer={0: torch.eye(4)},
        )

    with pytest.raises(TypeError, match="nn.Module"):
        export_model_to_compensated_nf4_edge("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="No eligible"):
        export_model_to_compensated_nf4_edge(
            hf_small, config=AlveoleMitosisNF4Config(min_linear_numel_to_quantize=10_000)
        )
    with pytest.raises(ValueError, match="general_mask"):
        export_model_to_compensated_nf4_edge(
            hf_small,
            calibration_ids=torch.zeros((2, 8), dtype=torch.long),
            general_mask=[True],
            config=AlveoleMitosisNF4Config(min_linear_numel_to_quantize=16),
        )

    with pytest.raises(TypeError, match="nn.Module"):
        execute_verified_alveole_mitosis_and_nf4_export("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="eval_fn"):
        execute_verified_alveole_mitosis_and_nf4_export(hf_small, eval_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="bos_sink_fn"):
        execute_verified_alveole_mitosis_and_nf4_export(hf_small, eval_fn=lambda _m: {"E0": 1.0}, bos_sink_fn="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AlveoleMitosisNF4Config"):
        execute_verified_alveole_mitosis_and_nf4_export(hf_small, config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Provide calibration_chunks"):
        execute_verified_alveole_mitosis_and_nf4_export(hf_small)
    with pytest.raises(ValueError, match="non-empty sequence"):
        execute_verified_alveole_mitosis_and_nf4_export(hf_small, calibration_chunks=[])
    with pytest.raises(ValueError, match="Sequence length"):
        execute_verified_alveole_mitosis_and_nf4_export(
            hf_small,
            calibration_chunks=[
                {"domain": "E0_gen", "ids": [1, 2, 3]},
                {"domain": "E1_spec", "ids": [4, 5, 6]},
            ],
            config=AlveoleMitosisNF4Config(score_from=8),
        )
    with pytest.raises(ValueError, match="both general and specialist"):
        execute_verified_alveole_mitosis_and_nf4_export(
            hf_small,
            calibration_chunks=[
                {"domain": "E1_a", "ids": list(range(12))},
                {"domain": "E1_b", "ids": list(range(12))},
            ],
            config=AlveoleMitosisNF4Config(score_from=4),
        )
    for bad_chunks in (
        [{"domain": "E0", "ids": [1, 2, 3]}, {"ids": [1, 2, 3]}],
        [{"domain": " ", "ids": [1, 2, 3]}],
        [{"domain": "E0", "ids": "bad"}],
        [{"domain": "E0", "ids": [1]}],
        [{"domain": "E0", "ids": [1, 2, 3, 4]}, {"domain": "E1", "ids": [1, 2, 3]}],
    ):
        with pytest.raises((ValueError, TypeError)):
            execute_verified_alveole_mitosis_and_nf4_export(
                hf_small, calibration_chunks=bad_chunks, config=AlveoleMitosisNF4Config(score_from=1)
            )

    # Extra branch coverage: dense P_act projection, layer filters, & explicit Hessians/activations
    p0_act = torch.eye(16, dtype=torch.float64)
    p0_act[0, 0] = 0.0
    mit_dense = recycle_corruptor_alveoles_via_mitosis(
        copy.deepcopy(hf_small),
        [{"corruptor_slot": 1, "anchor_slot": 0, "target_domain": "E1"}],
        acts_probe_by_layer={0: torch.ones((2, 8), dtype=torch.float32)},
        domain_labels_probe=["E0_a", "E1"],
        p_null_by_layer={0: {"x": torch.eye(8), "act": p0_act}},
    )
    assert mit_dense["num_recycled"] == 1
    hf_exp = copy.deepcopy(hf_small)
    exp_diag = export_model_to_compensated_nf4_edge(
        hf_exp,
        hessian_diag_by_layer={0: torch.ones(16), "head": torch.ones(8)},
        acts_by_linear={"layers.0.mlp.experts.0.gate_proj": torch.ones((4, 8))},
        p_null_by_layer={0: {"x": torch.eye(8), "act": p0_act}},
        config=AlveoleMitosisNF4Config(
            nf4_block_size=8,
            compensation_rank=2,
            num_alternating_steps=1,
            quantize_attention=False,
            quantize_base_mlp=False,
            min_linear_numel_to_quantize=64,
        ),
    )
    assert exp_diag["num_quantized_linears"] >= 1
    exp_head_only = export_model_to_compensated_nf4_edge(
        copy.deepcopy(hf_small),
        config=AlveoleMitosisNF4Config(
            nf4_block_size=8,
            compensation_rank=1,
            num_alternating_steps=1,
            quantize_dense_experts=False,
            quantize_lm_head=True,
            min_linear_numel_to_quantize=64,
        ),
    )
    assert exp_head_only["num_quantized_linears"] == 1

    with pytest.raises(TypeError, match="Mapping"):
        verify_alveole_mitosis_nf4_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_alveole_mitosis_nf4_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_alveole_mitosis_nf4_receipt_provenance({"receipt_sha256": "0" * 64, "object": "wrong"}) is False
    assert (
        verify_alveole_mitosis_nf4_receipt_provenance(
            {
                "receipt_sha256": "0" * 64,
                "object": ALVEOLE_MITOSIS_RECEIPT_OBJECT,
                "schema": ALVEOLE_MITOSIS_EXPORTER_SCHEMA,
                "bad": float("nan"),
            }
        )
        is False
    )

    # 2. AGPL-3.0, ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED, & OwnerOrderGuardrail
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn46", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn46", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/alveole_mitosis_hessian_nf4_exporter.py", TARGET_MODULE_PATH),
        ("tests/test_alveole_mitosis_hessian_nf4_exporter_scenarios.py", TEST_FILE_PATH),
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
        assert "NEURAL-CTRL-DYN-46" in doc_text
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
        {"nll_mean": 1.42, "compression_ratio": 4.458, "anse_energy": -0.35, "accepted": True},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+TELEMETRY = {telemetry_json!r}"
        )
        == []
    )

    pr_title = "feat(swarm): dynamic Alveole mitosis corruptor recycler and prose-Hessian compensated 4-bit NF4 edge exporter"
    pr_body = (
        "Implements NEURAL-CTRL-DYN-46 Gen-3 Direction D4 dynamic Alveole mitosis recycling "
        "causal-knockout corruptor cells into high-deficit specialist domains inside P_0 while "
        "compensating 4-bit NF4 quantization error via diagonal prose-Hessian low-rank residuals "
        "under the 6656 MiB 8 GB consumer GPU WDDM envelope, ANSE Delta E < 0, and atomic rollback."
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
        pr_number=1246,
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
