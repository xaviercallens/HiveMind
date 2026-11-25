# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Five deterministic CPU scenarios for NEURAL-CTRL-DYN-23 Multi-Architecture Open-Weight WSP Adapter."""

from __future__ import annotations

import ast
import importlib.util
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.auto_open_weight_engine import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_GENERAL_DOMAIN,
    AutoOpenWeightConfig,
    compute_anse_objective,
)
from hivemind.swarm.multi_arch_auto_wsp_adapter import (
    ARCH_GRANITE_MOE,
    ARCH_LLAMA_GQA,
    ARCH_QWEN_DENSE,
    ARCH_QWEN_MOE,
    PARAM_FULL_WEIGHT,
    PARAM_LORA,
    canonicalize_state_dict_for_s5,
    clone_native_state_dict,
    count_native_edited_parameters,
    detect_open_weight_architecture,
    hash_native_state_dict,
    materialize_s5_back_to_native_layout,
    run_multi_arch_auto_open_weight_s5,
    verify_multi_arch_receipt_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackAuditResult:
    """CI-safe stub audit result when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = violations
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else float(ANSE_MAXIMUM_PAIN)


class _FallbackZeroStubAudit:
    """CI-safe AST stub verifier implementing both ``check_file`` and ``audit_python_code``."""

    def audit_python_code(self, code: str) -> _FallbackAuditResult:
        tree = ast.parse(code)
        bad: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [
                    s
                    for s in node.body
                    if not (
                        isinstance(s, ast.Expr)
                        and isinstance(s.value, ast.Constant)
                        and isinstance(s.value.value, str)
                    )
                ]
                if not body or (len(body) == 1 and isinstance(body[0], ast.Pass)):
                    bad.append(f"stub function: {node.name}")
            if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
                if isinstance(node.exc.func, ast.Name) and node.exc.func.id == "NotImplementedError":
                    bad.append("NotImplementedError")
        return _FallbackAuditResult(bad)

    def check_file(self, path: str | Path) -> list[str]:
        return self.audit_python_code(Path(path).read_text(encoding="utf-8")).violations


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "multi_arch_auto_wsp_adapter.py"
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


def _sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _orthonormal_subspaces(seed: int, hidden_size: int, intermediate_size: int, n_spec: int) -> dict[str, Any]:
    """Construct orthogonal deterministic domain directions & prose/specialist activations."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q_h, _ = torch.linalg.qr(torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.float64))
    q_i, _ = torch.linalg.qr(
        torch.randn((intermediate_size, intermediate_size), generator=gen, dtype=torch.float64)
    )
    q_h_np = q_h.numpy()
    q_i_np = q_i.numpy()

    spec_domains = [
        "E1_python_systems_code",
        "E2_math_reasoning",
        "E5_exoplanet_transit",
        "E6_stellar_spectroscopy",
        "E7_defensive_cyber",
    ][:n_spec]
    domains = [DEFAULT_GENERAL_DOMAIN, *spec_domains]

    n_tok = 16
    spec_dirs = {dom: q_h_np[:, idx] for idx, dom in enumerate(spec_domains)}
    prose_orth_dir = q_h_np[:, n_spec]
    prose_center = 1.65 * prose_orth_dir + 0.50 * spec_dirs[spec_domains[min(2, n_spec - 1)]]

    prose_mid_subspace = q_i_np[:, :3]
    spec_mid_dirs = {dom: q_i_np[:, 3 + idx] for idx, dom in enumerate(spec_domains)}

    rng = np.random.default_rng(seed)
    domain_x: dict[str, np.ndarray] = {
        DEFAULT_GENERAL_DOMAIN: prose_center[None, :] + 0.06 * rng.standard_normal((n_tok, hidden_size))
    }
    prose_mid_mean = np.sum(prose_mid_subspace, axis=1)
    domain_mid: dict[str, np.ndarray] = {
        DEFAULT_GENERAL_DOMAIN: (
            1.45 * prose_mid_mean[None, :]
            + 0.25 * (rng.standard_normal((n_tok, 3)) @ prose_mid_subspace.T)
            + 0.02 * rng.standard_normal((n_tok, intermediate_size))
        )
    }
    domain_target: dict[str, np.ndarray] = {DEFAULT_GENERAL_DOMAIN: q_h_np[:, n_spec + 1]}

    for idx, dom in enumerate(spec_domains):
        domain_x[dom] = 1.80 * spec_dirs[dom][None, :] + 0.06 * rng.standard_normal((n_tok, hidden_size))
        domain_mid[dom] = (
            1.50 * spec_mid_dirs[dom][None, :]
            + 0.25 * (rng.standard_normal((n_tok, 3)) @ prose_mid_subspace.T)
            + 0.02 * rng.standard_normal((n_tok, intermediate_size))
        )
        domain_target[dom] = q_h_np[:, n_spec + 2 + idx]

    return {
        "domains": domains,
        "spec_domains": spec_domains,
        "spec_dirs": spec_dirs,
        "prose_orth_dir": prose_orth_dir,
        "prose_mid_subspace": prose_mid_subspace,
        "spec_mid_dirs": spec_mid_dirs,
        "domain_x": domain_x,
        "domain_mid": domain_mid,
        "domain_target": domain_target,
    }


def _build_granite_moe_world(seed: int = 20261124) -> dict[str, Any]:
    """Construct a deterministic 4-layer, 6-expert Granite-MoE native state dict & evaluator."""
    n_layers = 4
    n_experts = 6
    hidden_size = 16
    intermediate_size = 24
    rank = 4
    vocab_size = 32
    sub = _orthonormal_subspaces(seed, hidden_size, intermediate_size, n_spec=4)
    domains = sub["domains"]
    spec_domains = sub["spec_domains"]
    prose_orth_dir = sub["prose_orth_dir"]
    prose_mid_subspace = sub["prose_mid_subspace"]
    spec_mid_dirs = sub["spec_mid_dirs"]
    domain_target = sub["domain_target"]
    spec_dirs = sub["spec_dirs"]

    gen = torch.Generator(device="cpu").manual_seed(seed + 11)
    state: dict[str, torch.Tensor] = {
        "wte.weight": torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
        "engram.table.weight": torch.randn((48, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
        "head.weight": torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
    }

    interfering_layers = {1, 3}
    tail_layers = {3}
    anchor_layer = 2
    spec_expert_map = {
        2: spec_domains[0],
        3: spec_domains[1],
        4: spec_domains[2],  # dual-active multigate preserved expert!
        5: spec_domains[3],
    }

    for l_idx in range(n_layers):
        state[f"blocks.{l_idx}.ln1.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.ln1.bias"] = torch.zeros((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.ln2.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.ln2.bias"] = torch.zeros((hidden_size,), dtype=torch.bfloat16)
        # 4:2 GQA -> gqa_kv_head_ratio == 2
        state[f"blocks.{l_idx}.attn.q.weight"] = torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"blocks.{l_idx}.attn.k.weight"] = torch.randn((hidden_size // 2, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"blocks.{l_idx}.attn.v.weight"] = torch.randn((hidden_size // 2, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"blocks.{l_idx}.attn.o.weight"] = torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05

        state[f"blocks.{l_idx}.mlp.base_mlp.w1.weight"] = torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"blocks.{l_idx}.mlp.base_mlp.w2.weight"] = torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"blocks.{l_idx}.mlp.base_mlp.w3.weight"] = torch.randn((hidden_size, intermediate_size), generator=gen, dtype=torch.bfloat16) * 0.05

        rw = np.zeros((n_experts, hidden_size), dtype=np.float32)
        w1_a = np.zeros((n_experts, rank, hidden_size), dtype=np.float32)
        w1_b = np.zeros((n_experts, intermediate_size, rank), dtype=np.float32)
        w2_a = np.zeros((n_experts, rank, hidden_size), dtype=np.float32)
        w2_b = np.zeros((n_experts, intermediate_size, rank), dtype=np.float32)
        w3_a = np.zeros((n_experts, rank, intermediate_size), dtype=np.float32)
        w3_b = np.zeros((n_experts, hidden_size, rank), dtype=np.float32)

        if l_idx == anchor_layer:
            for sink_e in (0, 1):
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w3_b[sink_e, :, 0] = (0.42 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w3_a[sink_e, 0, :] = prose_mid_subspace[:, sink_e].astype(np.float32)
                w1_a[sink_e, 0, :] = prose_orth_dir.astype(np.float32)
                w1_b[sink_e, :, 0] = (0.12 * prose_mid_subspace[:, sink_e]).astype(np.float32)
            for exp_e, dom in spec_expert_map.items():
                rw[exp_e] = (2.2 * spec_dirs[dom]).astype(np.float32)
                w3_b[exp_e, :, 0] = (0.38 * domain_target[dom]).astype(np.float32)
                w3_a[exp_e, 0, :] = spec_mid_dirs[dom].astype(np.float32)
        elif l_idx in interfering_layers:
            for idx_s, sink_e in enumerate((0, 1)):
                scale = 2.1 if sink_e == 1 else 1.2
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w3_b[sink_e, :, 0] = (-0.32 * scale * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                w3_a[sink_e, 0, :] = prose_mid_subspace[:, idx_s].astype(np.float32)
                # Non-zero w1_B/w2_B ensure FoMoE zeroing MUST zero all 3 B factors to stop delta leakage!
                w1_a[sink_e, 0, :] = prose_orth_dir.astype(np.float32)
                w1_b[sink_e, :, 0] = (0.35 * prose_mid_subspace[:, idx_s]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_orth_dir.astype(np.float32)
                w2_b[sink_e, :, 0] = (0.35 * prose_mid_subspace[:, idx_s]).astype(np.float32)

            dom_dual = spec_expert_map[4]
            rw[4] = (2.2 * spec_dirs[dom_dual] + 1.25 * prose_orth_dir).astype(np.float32)
            w3_b[4, :, 0] = (0.55 * domain_target[dom_dual] - 0.16 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
            w3_a[4, 0, :] = (spec_mid_dirs[dom_dual] + 0.35 * prose_mid_subspace[:, 0]).astype(np.float32)

            for exp_e in (2, 3, 5):
                dom = spec_expert_map[exp_e]
                rw[exp_e] = (2.2 * spec_dirs[dom] + (0.25 * prose_orth_dir if l_idx in tail_layers else -0.7 * prose_orth_dir)).astype(np.float32)
                w3_b[exp_e, :, 0] = (0.44 * domain_target[dom] - (0.06 * domain_target[DEFAULT_GENERAL_DOMAIN] if l_idx in tail_layers else 0.0)).astype(np.float32)
                w3_a[exp_e, 0, :] = (spec_mid_dirs[dom] + (0.32 * prose_mid_subspace[:, 1] if l_idx in tail_layers else 0.0)).astype(np.float32)

        state[f"blocks.{l_idx}.mlp.router.weight"] = torch.from_numpy(rw)
        for e_idx in range(n_experts):
            state[f"blocks.{l_idx}.mlp.experts.{e_idx}.w1_A"] = torch.from_numpy(w1_a[e_idx].copy())
            state[f"blocks.{l_idx}.mlp.experts.{e_idx}.w1_B"] = torch.from_numpy(w1_b[e_idx].copy())
            state[f"blocks.{l_idx}.mlp.experts.{e_idx}.w2_A"] = torch.from_numpy(w2_a[e_idx].copy())
            state[f"blocks.{l_idx}.mlp.experts.{e_idx}.w2_B"] = torch.from_numpy(w2_b[e_idx].copy())
            state[f"blocks.{l_idx}.mlp.experts.{e_idx}.w3_A"] = torch.from_numpy(w3_a[e_idx].copy())
            state[f"blocks.{l_idx}.mlp.experts.{e_idx}.w3_B"] = torch.from_numpy(w3_b[e_idx].copy())

    base_ln_w = torch.ones((hidden_size,), dtype=torch.float32)
    base_ln_b = torch.zeros((hidden_size,), dtype=torch.float32)
    state["ln.weight"] = torch.full((hidden_size,), 1.18, dtype=torch.float32)
    state["ln.bias"] = torch.from_numpy((-0.04 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32))
    base_nll = {dom: 4.20 for dom in domains}

    def eval_granite_native(st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        ln_w = st["ln.weight"].detach().to(torch.float64).numpy()
        ln_b = st["ln.bias"].detach().to(torch.float64).numpy()
        cal_pen = 0.22 * ((float(np.mean(ln_w)) - 1.02) ** 2)
        out: dict[str, float] = {}
        for dom in domains:
            x = sub["domain_x"][dom]
            mid = sub["domain_mid"][dom]
            tgt = domain_target[dom]
            residual = np.zeros_like(x, dtype=np.float64)
            for l_idx in range(n_layers):
                rw_l = st[f"blocks.{l_idx}.mlp.router.weight"].detach().to(torch.float64).numpy()
                gates = _sigmoid_np(x @ rw_l.T)
                top_idx = np.argsort(gates, axis=1)[:, -3:]
                mask = np.zeros_like(gates)
                np.put_along_axis(mask, top_idx, 1.0, axis=1)
                active_g = gates * mask
                for e_id in range(n_experts):
                    b3 = st[f"blocks.{l_idx}.mlp.experts.{e_id}.w3_B"].detach().to(torch.float64).numpy()
                    a3 = st[f"blocks.{l_idx}.mlp.experts.{e_id}.w3_A"].detach().to(torch.float64).numpy()
                    b1 = st[f"blocks.{l_idx}.mlp.experts.{e_id}.w1_B"].detach().to(torch.float64).numpy()
                    b2 = st[f"blocks.{l_idx}.mlp.experts.{e_id}.w2_B"].detach().to(torch.float64).numpy()
                    delta_mat = b3 @ a3
                    exp_out = mid @ delta_mat.T
                    # Base-MLP interaction leakage if w1_B or w2_B remains non-zero on prose sinks
                    if dom == DEFAULT_GENERAL_DOMAIN and l_idx in interfering_layers and e_id in (0, 1):
                        base_leak = 0.28 * (float(np.linalg.norm(b1)) + float(np.linalg.norm(b2)))
                        exp_out = exp_out - base_leak * tgt[None, :]
                    residual = residual + active_g[:, e_id : e_id + 1] * exp_out
            normed = residual * ln_w[None, :] + ln_b[None, :]
            align = float(np.mean(normed @ tgt))
            quad_reg = 0.03 * float(np.mean(np.sum(normed * normed, axis=1)))
            out[dom] = float(base_nll[dom] - 0.55 * align + quad_reg + cal_pen)
        return out

    raw_nll = eval_granite_native(state)
    layer_harm: list[float] = []
    for l_idx in range(n_layers):
        byp = clone_native_state_dict(state)
        for e_id in range(n_experts):
            byp[f"blocks.{l_idx}.mlp.experts.{e_id}.w3_B"] = torch.zeros_like(
                byp[f"blocks.{l_idx}.mlp.experts.{e_id}.w3_B"]
            )
        byp_nll = eval_granite_native(byp)
        layer_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - byp_nll[DEFAULT_GENERAL_DOMAIN]))

    exp_harm: list[float] = []
    exp_spec: list[float] = []
    for e_id in range(n_experts):
        z_st = clone_native_state_dict(state)
        for l_idx in interfering_layers:
            for factor_b in ("w1_B", "w2_B", "w3_B"):
                k = f"blocks.{l_idx}.mlp.experts.{e_id}.{factor_b}"
                z_st[k] = torch.zeros_like(z_st[k])
        zn = eval_granite_native(z_st)
        exp_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - zn[DEFAULT_GENERAL_DOMAIN]))
        exp_spec.append(float(np.mean([zn[d] - raw_nll[d] for d in spec_domains])))

    return {
        "state": state,
        "base_nll": base_nll,
        "eval_fn": eval_granite_native,
        "layer_harm": layer_harm,
        "exp_harm": exp_harm,
        "exp_spec": exp_spec,
        "layer_prose": {l_idx: torch.from_numpy(sub["domain_x"][DEFAULT_GENERAL_DOMAIN]) for l_idx in range(n_layers)},
        "layer_spec": {
            l_idx: {d: torch.from_numpy(sub["domain_x"][d]) for d in spec_domains} for l_idx in range(n_layers)
        },
        "tail_mid": {l_idx: torch.from_numpy(sub["domain_mid"][DEFAULT_GENERAL_DOMAIN]) for l_idx in range(n_layers)},
        "base_ln_w": base_ln_w,
        "base_ln_b": base_ln_b,
        "spec_dirs": spec_dirs,
    }


def _build_qwen_moe_world(seed: int = 20261125) -> dict[str, Any]:
    """Construct a deterministic 4-layer, 4-expert HuggingFace Qwen-MoE full-weight world."""
    n_layers = 4
    n_experts = 4
    hidden_size = 12
    intermediate_size = 18
    vocab_size = 32
    sub = _orthonormal_subspaces(seed, hidden_size, intermediate_size, n_spec=3)
    domains = sub["domains"]
    spec_domains = sub["spec_domains"]
    prose_orth_dir = sub["prose_orth_dir"]
    prose_mid_subspace = sub["prose_mid_subspace"]
    spec_mid_dirs = sub["spec_mid_dirs"]
    domain_target = sub["domain_target"]
    spec_dirs = sub["spec_dirs"]

    gen = torch.Generator(device="cpu").manual_seed(seed + 7)
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
        "lm_head.weight": torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
        "model.norm.weight": torch.full((hidden_size,), 1.16, dtype=torch.bfloat16),
    }

    interfering_layers = {1, 3}
    anchor_layer = 2
    spec_map = {1: spec_domains[0], 2: spec_domains[1], 3: spec_domains[2]}

    for l_idx in range(n_layers):
        state[f"model.layers.{l_idx}.input_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"model.layers.{l_idx}.post_attention_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        # 4:1 GQA with QKV biases (Qwen2 style) -> gqa_kv_head_ratio == 4
        q_out = hidden_size
        kv_out = hidden_size // 4
        state[f"model.layers.{l_idx}.self_attn.q_proj.weight"] = torch.randn((q_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.q_proj.bias"] = torch.randn((q_out,), generator=gen, dtype=torch.bfloat16) * 0.01
        state[f"model.layers.{l_idx}.self_attn.k_proj.weight"] = torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.k_proj.bias"] = torch.randn((kv_out,), generator=gen, dtype=torch.bfloat16) * 0.01
        state[f"model.layers.{l_idx}.self_attn.v_proj.weight"] = torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.v_proj.bias"] = torch.randn((kv_out,), generator=gen, dtype=torch.bfloat16) * 0.01
        state[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = torch.randn((hidden_size, q_out), generator=gen, dtype=torch.bfloat16) * 0.05

        state[f"model.layers.{l_idx}.mlp.shared_expert.gate_proj.weight"] = torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
        state[f"model.layers.{l_idx}.mlp.shared_expert.up_proj.weight"] = torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
        state[f"model.layers.{l_idx}.mlp.shared_expert.down_proj.weight"] = torch.randn((hidden_size, intermediate_size), generator=gen, dtype=torch.bfloat16) * 0.03
        state[f"model.layers.{l_idx}.mlp.shared_expert_gate.weight"] = torch.randn((1, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03

        rw = np.zeros((n_experts, hidden_size), dtype=np.float32)
        down_mats = np.zeros((n_experts, hidden_size, intermediate_size), dtype=np.float32)
        if l_idx == anchor_layer:
            rw[0] = (2.0 * prose_orth_dir).astype(np.float32)
            down_mats[0] = np.outer(0.45 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 0]).astype(np.float32)
            for e_id, dom in spec_map.items():
                rw[e_id] = (2.2 * spec_dirs[dom]).astype(np.float32)
                down_mats[e_id] = np.outer(0.40 * domain_target[dom], spec_mid_dirs[dom]).astype(np.float32)
        elif l_idx in interfering_layers:
            rw[0] = (2.0 * prose_orth_dir).astype(np.float32)
            down_mats[0] = np.outer(-0.55 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 0]).astype(np.float32)
            # Dual-active expert 2
            dom_d = spec_map[2]
            rw[2] = (2.2 * spec_dirs[dom_d] + 1.2 * prose_orth_dir).astype(np.float32)
            down_mats[2] = (
                np.outer(0.55 * domain_target[dom_d], spec_mid_dirs[dom_d])
                - np.outer(0.16 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 1])
            ).astype(np.float32)
            for e_id in (1, 3):
                dom = spec_map[e_id]
                rw[e_id] = (2.2 * spec_dirs[dom] + (0.2 * prose_orth_dir if l_idx == 3 else -0.6 * prose_orth_dir)).astype(np.float32)
                down_mats[e_id] = (
                    np.outer(0.44 * domain_target[dom], spec_mid_dirs[dom])
                    - (np.outer(0.06 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 2]) if l_idx == 3 else 0.0)
                ).astype(np.float32)

        state[f"model.layers.{l_idx}.mlp.gate.weight"] = torch.from_numpy(rw)
        for e_id in range(n_experts):
            state[f"model.layers.{l_idx}.mlp.experts.{e_id}.gate_proj.weight"] = (
                torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
            )
            state[f"model.layers.{l_idx}.mlp.experts.{e_id}.up_proj.weight"] = (
                torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
            )
            state[f"model.layers.{l_idx}.mlp.experts.{e_id}.down_proj.weight"] = torch.from_numpy(
                down_mats[e_id].copy()
            ).to(dtype=torch.float32)

    base_nll = {dom: 4.10 for dom in domains}

    def eval_qwen_native(st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        norm_w = st["model.norm.weight"].detach().to(torch.float64).numpy()
        cal_pen = 0.22 * ((float(np.mean(norm_w)) - 1.00) ** 2)
        out: dict[str, float] = {}
        for dom in domains:
            x = sub["domain_x"][dom]
            mid = sub["domain_mid"][dom]
            tgt = domain_target[dom]
            residual = np.zeros_like(x, dtype=np.float64)
            for l_idx in range(n_layers):
                rw_l = st[f"model.layers.{l_idx}.mlp.gate.weight"].detach().to(torch.float64).numpy()
                gates = _sigmoid_np(x @ rw_l.T)
                top_idx = np.argsort(gates, axis=1)[:, -2:]
                mask = np.zeros_like(gates)
                np.put_along_axis(mask, top_idx, 1.0, axis=1)
                active_g = gates * mask
                for e_id in range(n_experts):
                    dw = st[f"model.layers.{l_idx}.mlp.experts.{e_id}.down_proj.weight"].detach().to(torch.float64).numpy()
                    residual = residual + active_g[:, e_id : e_id + 1] * (mid @ dw.T)
            normed = residual * norm_w[None, :]
            align = float(np.mean(normed @ tgt))
            quad_reg = 0.03 * float(np.mean(np.sum(normed * normed, axis=1)))
            out[dom] = float(base_nll[dom] - 0.55 * align + quad_reg + cal_pen)
        return out

    raw_nll = eval_qwen_native(state)
    layer_harm: list[float] = []
    for l_idx in range(n_layers):
        byp = clone_native_state_dict(state)
        for e_id in range(n_experts):
            k = f"model.layers.{l_idx}.mlp.experts.{e_id}.down_proj.weight"
            byp[k] = torch.zeros_like(byp[k])
        layer_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_qwen_native(byp)[DEFAULT_GENERAL_DOMAIN]))

    exp_harm: list[float] = []
    exp_spec: list[float] = []
    for e_id in range(n_experts):
        z_st = clone_native_state_dict(state)
        for l_idx in interfering_layers:
            k = f"model.layers.{l_idx}.mlp.experts.{e_id}.down_proj.weight"
            z_st[k] = torch.zeros_like(z_st[k])
        zn = eval_qwen_native(z_st)
        exp_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - zn[DEFAULT_GENERAL_DOMAIN]))
        exp_spec.append(float(np.mean([zn[d] - raw_nll[d] for d in spec_domains])))

    return {
        "state": state,
        "base_nll": base_nll,
        "eval_fn": eval_qwen_native,
        "layer_harm": layer_harm,
        "exp_harm": exp_harm,
        "exp_spec": exp_spec,
        "layer_prose": {l_idx: sub["domain_x"][DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)},
        "layer_spec": {l_idx: {d: sub["domain_x"][d] for d in spec_domains} for l_idx in range(n_layers)},
        "tail_mid": {l_idx: sub["domain_mid"][DEFAULT_GENERAL_DOMAIN] for l_idx in range(n_layers)},
        "base_norm_w": torch.ones((hidden_size,), dtype=torch.bfloat16),
    }


def _build_llama_gqa_lora_world(seed: int = 20261126) -> dict[str, Any]:
    """Construct a deterministic 4-layer Llama-GQA (4:1) LoRA state dict with biasless RMSNorm."""
    n_layers = 4
    hidden_size = 16
    intermediate_size = 24
    rank = 4
    vocab_size = 32
    sub = _orthonormal_subspaces(seed, hidden_size, intermediate_size, n_spec=3)
    domains = sub["domains"]
    spec_domains = sub["spec_domains"]
    prose_mid_subspace = sub["prose_mid_subspace"]
    spec_mid_dirs = sub["spec_mid_dirs"]
    domain_target = sub["domain_target"]

    gen = torch.Generator(device="cpu").manual_seed(seed + 19)
    state: dict[str, torch.Tensor] = {
        "model.embed_tokens.weight": torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
        "lm_head.weight": torch.randn((vocab_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.02,
        # Over-amplified RMSNorm scale (1.20x) cured by Stage S5.4 1/tau folding
        "model.norm.weight": torch.full((hidden_size,), 1.20, dtype=torch.float32),
    }

    anchor_layer = 1
    tail_layer = 3

    for l_idx in range(n_layers):
        state[f"model.layers.{l_idx}.input_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"model.layers.{l_idx}.post_attention_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        # Llama 4:1 GQA (biasless attention!)
        q_out = hidden_size
        kv_out = hidden_size // 4
        state[f"model.layers.{l_idx}.self_attn.q_proj.weight"] = torch.randn((q_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.k_proj.weight"] = torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.v_proj.weight"] = torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = torch.randn((hidden_size, q_out), generator=gen, dtype=torch.bfloat16) * 0.05
        state[f"model.layers.{l_idx}.self_attn.q_proj.lora_A.weight"] = torch.randn((rank, hidden_size), generator=gen, dtype=torch.float32) * 0.02
        state[f"model.layers.{l_idx}.self_attn.q_proj.lora_B.weight"] = torch.randn((q_out, rank), generator=gen, dtype=torch.float32) * 0.02
        state[f"model.layers.{l_idx}.self_attn.k_proj.lora_A.weight"] = torch.randn((rank, hidden_size), generator=gen, dtype=torch.float32) * 0.02
        state[f"model.layers.{l_idx}.self_attn.k_proj.lora_B.weight"] = torch.randn((kv_out, rank), generator=gen, dtype=torch.float32) * 0.02

        for proj_name, in_d, out_d in (
            ("gate_proj", hidden_size, intermediate_size),
            ("up_proj", hidden_size, intermediate_size),
            ("down_proj", intermediate_size, hidden_size),
        ):
            state[f"model.layers.{l_idx}.mlp.{proj_name}.weight"] = torch.randn((out_d, in_d), generator=gen, dtype=torch.bfloat16) * 0.04
            state[f"model.layers.{l_idx}.mlp.{proj_name}.lora_A.weight"] = torch.randn((rank, in_d), generator=gen, dtype=torch.float32) * 0.02
            state[f"model.layers.{l_idx}.mlp.{proj_name}.lora_B.weight"] = torch.randn((out_d, rank), generator=gen, dtype=torch.float32) * 0.02

        db = np.zeros((hidden_size, rank), dtype=np.float32)
        da = np.zeros((rank, intermediate_size), dtype=np.float32)
        if l_idx == anchor_layer:
            db[:, 0] = (0.18 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
            da[0, :] = prose_mid_subspace[:, 0].astype(np.float32)
            for idx_d, dom in enumerate(spec_domains[:2], start=1):
                db[:, idx_d] = (0.36 * domain_target[dom]).astype(np.float32)
                da[idx_d, :] = spec_mid_dirs[dom].astype(np.float32)
        elif l_idx == tail_layer:
            # Tail layer leaks strong prose interference along prose_mid_subspace while helping specialists
            for idx_d, dom in enumerate(spec_domains):
                db[:, idx_d] = (0.48 * domain_target[dom] - 0.35 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
                da[idx_d, :] = (spec_mid_dirs[dom] + 0.60 * prose_mid_subspace[:, idx_d % 3]).astype(np.float32)
        state[f"model.layers.{l_idx}.mlp.down_proj.lora_B.weight"] = torch.from_numpy(db)
        state[f"model.layers.{l_idx}.mlp.down_proj.lora_A.weight"] = torch.from_numpy(da)

    base_nll = {dom: 4.00 for dom in domains}

    def eval_llama_native(st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        norm_w = st["model.norm.weight"].detach().to(torch.float64).numpy()
        cal_pen = 1.40 * ((float(np.mean(norm_w)) - 1.00) ** 2)
        out: dict[str, float] = {}
        for dom in domains:
            mid = sub["domain_mid"][dom]
            tgt = domain_target[dom]
            residual = np.zeros((mid.shape[0], hidden_size), dtype=np.float64)
            for l_idx in range(n_layers):
                db_l = st[f"model.layers.{l_idx}.mlp.down_proj.lora_B.weight"].detach().to(torch.float64).numpy()
                da_l = st[f"model.layers.{l_idx}.mlp.down_proj.lora_A.weight"].detach().to(torch.float64).numpy()
                residual = residual + mid @ (db_l @ da_l).T
            normed = residual * norm_w[None, :]
            align = float(np.mean(normed @ tgt))
            quad_reg = 0.025 * float(np.mean(np.sum(normed * normed, axis=1)))
            out[dom] = float(base_nll[dom] - 0.55 * align + quad_reg + cal_pen)
        return out

    raw_nll = eval_llama_native(state)
    layer_harm: list[float] = []
    for l_idx in range(n_layers):
        byp = clone_native_state_dict(state)
        k = f"model.layers.{l_idx}.mlp.down_proj.lora_B.weight"
        byp[k] = torch.zeros_like(byp[k])
        layer_harm.append(float(raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_llama_native(byp)[DEFAULT_GENERAL_DOMAIN]))

    return {
        "state": state,
        "base_nll": base_nll,
        "eval_fn": eval_llama_native,
        "layer_harm": layer_harm,
        "tail_mid": {l_idx: torch.from_numpy(sub["domain_mid"][DEFAULT_GENERAL_DOMAIN]) for l_idx in range(n_layers)},
        "base_norm_w": torch.ones((hidden_size,), dtype=torch.float32),
    }


def test_s1_granite_moe_native_detection_s5_surgery_and_roundtrip() -> None:
    """Scenario 1: Granite-MoE native detection, lossless roundtrip, and S5 surgery lowering ANSE energy."""
    world = _build_granite_moe_world()
    state = world["state"]
    spec = detect_open_weight_architecture(state)
    assert spec.arch_family == ARCH_GRANITE_MOE
    assert spec.parameterization == PARAM_LORA
    assert spec.layer_count == 4
    assert spec.expert_count == 6
    assert spec.is_moe is True
    assert spec.gqa_kv_head_ratio == 2
    assert spec.kv_to_q_head_ratio == pytest.approx(0.5)
    assert spec.final_norm_key == "ln.weight"
    assert spec.final_norm_bias_key == "ln.bias"
    assert spec.norm_type == "layernorm"
    assert spec.has_attention is True

    # Lossless round-trip before S5
    canon = canonicalize_state_dict_for_s5(state)
    assert "layer_0.w2_B" in canon and "layer_0.router_w" in canon and "final_ln_w" in canon
    roundtrip = materialize_s5_back_to_native_layout(state, canon)
    assert hash_native_state_dict(roundtrip) == hash_native_state_dict(state)

    # Raw Granite checkpoint violates +1.0% general prose guard -> ANSE Maximum Pain
    raw_e, raw_diff, raw_gen_rel, _ = compute_anse_objective(world["eval_fn"](state), world["base_nll"])
    assert raw_gen_rel > 0.01
    assert raw_e == ANSE_MAXIMUM_PAIN
    assert raw_diff < 0.0

    cfg = AutoOpenWeightConfig(
        router_beta=1.35,
        svd_damp_strength=0.90,
        lambda_ln_grid=(0.25, 0.75, 1.00),
        tau_temp_grid=(0.95, 1.00, 1.15),
    )
    edited_state, res = run_multi_arch_auto_open_weight_s5(
        state,
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_harm"],
        expert_e0_harm_nats=world["exp_harm"],
        expert_spec_loss_when_zeroed=world["exp_spec"],
        layer_prose_inputs=world["layer_prose"],
        layer_specialist_inputs=world["layer_spec"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        config=cfg,
    )

    assert res.committed is True
    assert res.verdict == "PASS"
    assert res.arch_family == ARCH_GRANITE_MOE
    assert res.anse_energy < 0.0
    assert res.general_rel_drift <= 0.01
    assert res.all_domains_negative is True
    assert res.roundtrip_verified is True
    assert res.untouched_keys_bitwise_preserved is True
    assert verify_multi_arch_receipt_provenance(res) is True

    # Beneficial anchor layer 2 must remain 100% bitwise untouched
    for k in state:
        if k.startswith("blocks.2."):
            assert torch.equal(state[k], edited_state[k]) and state[k].dtype == edited_state[k].dtype

    # Pure sink experts {0, 1} on interfering layers {1, 3} must have w1_B = w2_B = w3_B == 0.0
    for l_idx in (1, 3):
        for e_sink in (0, 1):
            for factor_b in ("w1_B", "w2_B", "w3_B"):
                k_b = f"blocks.{l_idx}.mlp.experts.{e_sink}.{factor_b}"
                assert float(torch.linalg.norm(edited_state[k_b]).item()) == 0.0

    # Untouched bfloat16 embeddings & attention remain bitwise identical
    for k in ("wte.weight", "engram.table.weight", "head.weight", "blocks.1.attn.q.weight"):
        assert torch.equal(state[k], edited_state[k]) and state[k].dtype == torch.bfloat16


def test_s2_qwen_moe_hf_detection_and_lossless_roundtrip_with_s5() -> None:
    """Scenario 2: Qwen-MoE HF state-dict auto-detection, lossless roundtrip, and S5 surgery."""
    world = _build_qwen_moe_world()
    state = world["state"]
    spec = detect_open_weight_architecture(state)
    assert spec.arch_family == ARCH_QWEN_MOE
    assert spec.parameterization == PARAM_FULL_WEIGHT
    assert spec.layer_count == 4
    assert spec.expert_count == 4
    assert spec.is_moe is True
    assert spec.gqa_kv_head_ratio == 4
    assert spec.final_norm_key == "model.norm.weight"
    assert spec.final_norm_bias_key is None
    assert spec.norm_type == "rmsnorm"

    # Lossless roundtrip preserves full-weight matrices AND mixed bfloat16/float32 dtypes bitwise
    canon = canonicalize_state_dict_for_s5(state)
    roundtrip = materialize_s5_back_to_native_layout(state, canon)
    assert hash_native_state_dict(roundtrip) == hash_native_state_dict(state)
    for k in state:
        assert roundtrip[k].dtype == state[k].dtype
        assert torch.equal(roundtrip[k], state[k])

    # Also verify truncated full_weight_max_rank canonicalization roundtrip
    canon_trunc = canonicalize_state_dict_for_s5(state, full_weight_max_rank=6)
    assert canon_trunc["layer_0.w2_B"].shape == (4, 12, 6)
    rt_trunc = materialize_s5_back_to_native_layout(state, canon_trunc, full_weight_max_rank=6)
    assert hash_native_state_dict(rt_trunc) == hash_native_state_dict(state)

    # Verify Qwen-Dense auto-detection (no experts + QKV biases / QK norms)
    qwen_dense_state = {
        "model.embed_tokens.weight": state["model.embed_tokens.weight"],
        "model.layers.0.self_attn.q_proj.weight": state["model.layers.0.self_attn.q_proj.weight"],
        "model.layers.0.self_attn.q_proj.bias": state["model.layers.0.self_attn.q_proj.bias"],
        "model.layers.0.self_attn.k_proj.weight": state["model.layers.0.self_attn.k_proj.weight"],
        "model.layers.0.self_attn.q_norm.weight": torch.ones((4,), dtype=torch.float32),
        "model.layers.0.mlp.gate_proj.weight": state["model.layers.0.mlp.experts.0.gate_proj.weight"],
        "model.layers.0.mlp.up_proj.weight": state["model.layers.0.mlp.experts.0.up_proj.weight"],
        "model.layers.0.mlp.down_proj.weight": state["model.layers.0.mlp.experts.0.down_proj.weight"],
        "model.norm.weight": state["model.norm.weight"],
    }
    qd_spec = detect_open_weight_architecture(qwen_dense_state)
    assert qd_spec.arch_family == ARCH_QWEN_DENSE
    assert qd_spec.expert_count == 1
    assert qd_spec.is_moe is False

    # End-to-end S5 surgery on Qwen-MoE
    edited_qwen, res = run_multi_arch_auto_open_weight_s5(
        state,
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_harm"],
        expert_e0_harm_nats=world["exp_harm"],
        expert_spec_loss_when_zeroed=world["exp_spec"],
        layer_prose_inputs=world["layer_prose"],
        layer_specialist_inputs=world["layer_spec"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        base_final_ln_w=world["base_norm_w"],
        config=AutoOpenWeightConfig(
            router_beta=1.20,
            svd_damp_strength=0.90,
            lambda_ln_grid=(0.25, 0.75, 1.00),
            tau_temp_grid=(0.95, 1.00, 1.15),
        ),
    )
    assert res.committed is True
    assert res.verdict == "PASS"
    assert res.arch_family == ARCH_QWEN_MOE
    assert res.anse_energy < 0.0
    assert res.general_rel_drift <= 0.01
    assert res.untouched_keys_bitwise_preserved is True
    assert res.final_norm_key == "model.norm.weight"
    assert edited_qwen["model.norm.weight"].dtype == torch.bfloat16
    assert torch.equal(state["model.embed_tokens.weight"], edited_qwen["model.embed_tokens.weight"])
    assert torch.equal(state["model.layers.2.mlp.experts.0.down_proj.weight"], edited_qwen["model.layers.2.mlp.experts.0.down_proj.weight"])
    assert float(torch.linalg.norm(edited_qwen["model.layers.1.mlp.experts.0.down_proj.weight"]).item()) == 0.0


def test_s3_llama_gqa_lora_detection_tail_nullspace_and_rmsnorm_temp_folding() -> None:
    """Scenario 3: Llama-GQA LoRA detection, tail-layer QR-SVD nullspace, and folded RMSNorm calibration."""
    world = _build_llama_gqa_lora_world()
    state = world["state"]
    spec = detect_open_weight_architecture(state)
    assert spec.arch_family == ARCH_LLAMA_GQA
    assert spec.parameterization == PARAM_LORA
    assert spec.layer_count == 4
    assert spec.expert_count == 1
    assert spec.is_moe is False
    assert spec.gqa_kv_head_ratio == 4
    assert spec.kv_to_q_head_ratio == pytest.approx(0.25)
    assert spec.final_norm_key == "model.norm.weight"
    assert spec.final_norm_bias_key is None
    assert spec.norm_type == "rmsnorm"

    raw_e, raw_diff, raw_gen_rel, _ = compute_anse_objective(world["eval_fn"](state), world["base_nll"])
    assert raw_gen_rel > 0.01
    assert raw_e == ANSE_MAXIMUM_PAIN
    assert raw_diff < 0.0

    # Pure temperature calibration (auto_temperature_calibration=True without base_final_ln_w) + tail QR-SVD
    edited_llama, res = run_multi_arch_auto_open_weight_s5(
        state,
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_harm"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        config=AutoOpenWeightConfig(
            svd_damp_strength=0.95,
            tau_temp_grid=(0.95, 1.00, 1.12, 1.20),
        ),
    )

    assert res.committed is True
    assert res.verdict == "PASS"
    assert res.arch_family == ARCH_LLAMA_GQA
    assert res.expert_count == 1
    assert res.gqa_kv_head_ratio == 4
    assert res.anse_energy < 0.0
    assert res.general_rel_drift <= 0.01
    assert res.all_domains_negative is True
    assert res.untouched_keys_bitwise_preserved is True
    assert res.selected_tau_temp > 1.0
    assert float(edited_llama["model.norm.weight"].mean().item()) < float(state["model.norm.weight"].mean().item())

    # Anchor layer 1 and all self_attn / gate_proj / up_proj / embed_tokens remain bitwise identical
    for k in (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.1.mlp.down_proj.lora_A.weight",
        "model.layers.1.mlp.down_proj.lora_B.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.lora_B.weight",
        "model.layers.3.mlp.gate_proj.lora_B.weight",
    ):
        assert torch.equal(state[k], edited_llama[k]) and state[k].dtype == edited_llama[k].dtype

    # Also test base_state norm extraction branch
    base_state = {"model.norm.weight": world["base_norm_w"]}
    _, res_base = run_multi_arch_auto_open_weight_s5(
        state,
        world["base_nll"],
        world["eval_fn"],
        layer_e0_harm_nats=world["layer_harm"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        base_state=base_state,
        config=AutoOpenWeightConfig(svd_damp_strength=0.95),
    )
    assert res_base.committed is True and res_base.anse_energy < 0.0


def _tiny_granite_state() -> dict[str, torch.Tensor]:
    """Minimal deterministic 1-layer, 2-expert Granite-MoE state dict for fast rollback/validation tests."""
    gen = torch.Generator(device="cpu").manual_seed(9001)
    return {
        "blocks.0.attn.q.weight": torch.randn((8, 8), generator=gen, dtype=torch.float32),
        "blocks.0.attn.k.weight": torch.randn((4, 8), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.router.weight": torch.randn((2, 8), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w1_A": torch.randn((2, 8), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w1_B": torch.randn((12, 2), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w2_A": torch.randn((2, 8), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w2_B": torch.randn((12, 2), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w3_A": torch.randn((2, 12), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.0.w3_B": torch.randn((8, 2), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.1.w1_A": torch.randn((2, 8), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.1.w1_B": torch.randn((12, 2), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.1.w2_A": torch.randn((2, 8), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.1.w2_B": torch.randn((12, 2), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.1.w3_A": torch.randn((2, 12), generator=gen, dtype=torch.float32),
        "blocks.0.mlp.experts.1.w3_B": torch.randn((8, 2), generator=gen, dtype=torch.float32),
        "ln.weight": torch.ones((8,), dtype=torch.float32),
        "ln.bias": torch.zeros((8,), dtype=torch.float32),
    }


def _tiny_qwen_state() -> dict[str, torch.Tensor]:
    """Minimal deterministic 1-layer, 2-expert Qwen-MoE state dict for fast validation tests."""
    gen = torch.Generator(device="cpu").manual_seed(9002)
    return {
        "model.embed_tokens.weight": torch.randn((16, 8), generator=gen, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.weight": torch.randn((8, 8), generator=gen, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn((2, 8), generator=gen, dtype=torch.bfloat16),
        "model.layers.0.self_attn.q_proj.bias": torch.zeros((8,), dtype=torch.bfloat16),
        "model.layers.0.mlp.gate.weight": torch.randn((2, 8), generator=gen, dtype=torch.float32),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.randn((8, 12), generator=gen, dtype=torch.float32),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.randn((8, 12), generator=gen, dtype=torch.float32),
        "model.norm.weight": torch.ones((8,), dtype=torch.float32),
        "guard_bool": torch.tensor([True, False], dtype=torch.bool),
    }


def _tiny_llama_state() -> dict[str, torch.Tensor]:
    """Minimal deterministic 1-layer Llama-GQA LoRA state dict for fast rollback/validation tests."""
    gen = torch.Generator(device="cpu").manual_seed(9003)
    return {
        "model.layers.0.self_attn.q_proj.weight": torch.randn((8, 8), generator=gen, dtype=torch.bfloat16),
        "model.layers.0.self_attn.k_proj.weight": torch.randn((2, 8), generator=gen, dtype=torch.bfloat16),
        "model.layers.0.mlp.down_proj.lora_A.weight": torch.randn((2, 12), generator=gen, dtype=torch.float32),
        "model.layers.0.mlp.down_proj.lora_B.weight": torch.randn((8, 2), generator=gen, dtype=torch.float32),
        "model.norm.weight": torch.ones((8,), dtype=torch.float32),
    }


def test_s4_anse_maximum_pain_rollback_restores_native_bitwise() -> None:
    """Scenario 4: Guard violation triggers ANSE Maximum Pain rollback restoring native state bitwise."""
    granite_state = _tiny_granite_state()
    llama_state = _tiny_llama_state()
    base_nll = {DEFAULT_GENERAL_DOMAIN: 4.1, "E1_code": 4.2}

    for label, state, l_harm, extra in (
        (
            "granite",
            granite_state,
            [0.05],
            {
                "expert_e0_harm_nats": [0.05, -0.01],
                "expert_spec_loss_when_zeroed": [0.0, 0.04],
            },
        ),
        (
            "llama",
            llama_state,
            [0.05],
            {"auto_temperature_calibration": False},
        ),
    ):
        def _infeasible_eval(_st: Mapping[str, torch.Tensor]) -> dict[str, float]:
            return {DEFAULT_GENERAL_DOMAIN: base_nll[DEFAULT_GENERAL_DOMAIN] * 1.25, "E1_code": 3.8}

        pre_hash = hash_native_state_dict(state)
        rolled_state, fail_res = run_multi_arch_auto_open_weight_s5(
            state,
            base_nll,
            _infeasible_eval,
            layer_e0_harm_nats=l_harm,
            **extra,
        )
        assert fail_res.committed is False, label
        assert fail_res.verdict == "ROLLED_BACK_MAXIMUM_PAIN", label
        assert fail_res.anse_energy == ANSE_MAXIMUM_PAIN, label
        assert fail_res.native_edited_params == 0, label
        assert fail_res.post_native_sha256 == pre_hash, label
        assert hash_native_state_dict(rolled_state) == pre_hash, label
        assert fail_res.untouched_keys_bitwise_preserved is True, label
        assert verify_multi_arch_receipt_provenance(fail_res) is True, label

        tampered = fail_res.to_dict()
        tampered["anse_energy"] = -0.5
        assert verify_multi_arch_receipt_provenance(tampered) is False
        assert verify_multi_arch_receipt_provenance({"receipt_sha256": "bad"}) is False
        assert verify_multi_arch_receipt_provenance({"receipt_sha256": "0" * 64}) is False
        assert verify_multi_arch_receipt_provenance(
            {
                "schema": "hivemind.multi_arch_auto_wsp_adapter/1",
                "object": "hivemind.multi_arch_auto_wsp_receipt",
                "receipt_sha256": "0" * 64,
                "bad_nan": float("nan"),
            }
        ) is False


def test_s5_validation_unsupported_layouts_and_security_stub_gates() -> None:
    """Scenario 5: Validation/unsupported-layout error paths and full security/stub compliance gates."""
    granite_state = _tiny_granite_state()
    qwen_state = _tiny_qwen_state()
    llama_state = _tiny_llama_state()
    base_nll = {DEFAULT_GENERAL_DOMAIN: 4.1, "E1_code": 4.2}

    def _ok_eval(_st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        return dict(base_nll)

    # 1. State-dict validation & unsupported-layout rejections
    with pytest.raises(ValueError, match="non-empty mapping"):
        detect_open_weight_architecture({})
    with pytest.raises(ValueError, match="non-empty strings"):
        detect_open_weight_architecture({"": torch.ones((2,))})
    with pytest.raises(TypeError, match="torch.Tensor"):
        detect_open_weight_architecture({"a": np.ones((2,))})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-empty"):
        detect_open_weight_architecture({"a": torch.empty((0, 4))})
    with pytest.raises(ValueError, match="non-finite"):
        detect_open_weight_architecture({"a": torch.tensor([float("nan")])})
    with pytest.raises(ValueError, match="Unsupported arch_hint"):
        detect_open_weight_architecture(granite_state, arch_hint="unknown_arch")
    with pytest.raises(ValueError, match="Unsupported open-weight architecture layout"):
        detect_open_weight_architecture({"vit.encoder.layer.0.weight": torch.ones((4, 4))})
    with pytest.raises(ValueError, match="Ambiguous mixed architecture layouts"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 4)),
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 4)),
            }
        )
    with pytest.raises(ValueError, match="conflicts with Granite-MoE"):
        detect_open_weight_architecture(granite_state, arch_hint=ARCH_LLAMA_GQA)
    with pytest.raises(ValueError, match="conflicts with Qwen-MoE"):
        detect_open_weight_architecture(qwen_state, arch_hint=ARCH_GRANITE_MOE)
    with pytest.raises(ValueError, match="conflicts with dense"):
        detect_open_weight_architecture(llama_state, arch_hint=ARCH_QWEN_MOE)
    with pytest.raises(ValueError, match="no MLP projections"):
        detect_open_weight_architecture({"model.layers.0.self_attn.q_proj.weight": torch.ones((4, 4))})
    with pytest.raises(ValueError, match="Router/gate weights found without MoE"):
        detect_open_weight_architecture({"model.layers.0.mlp.gate.weight": torch.ones((2, 4))})

    # Non-contiguous layers / experts & incomplete pairs & shape mismatches
    with pytest.raises(ValueError, match="Non-contiguous"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 4)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.2.mlp.experts.0.w3_A": torch.ones((2, 4)),
                "blocks.2.mlp.experts.0.w3_B": torch.ones((4, 2)),
            }
        )
    with pytest.raises(ValueError, match="Missing w3_A/w3_B"):
        detect_open_weight_architecture({"blocks.0.mlp.experts.0.w1_A": torch.ones((2, 4))})
    with pytest.raises(ValueError, match="Incomplete w1 LoRA pair"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.0.mlp.experts.0.w1_A": torch.ones((2, 4)),
            }
        )
    with pytest.raises(ValueError, match="Mismatched LoRA shapes"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((3, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
            }
        )
    with pytest.raises(ValueError, match="Invalid GQA dimensions"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((8, 12)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((8, 8)),
                "model.layers.0.self_attn.k_proj.weight": torch.ones((6, 8)),
            }
        )
    with pytest.raises(ValueError, match="Router key"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.0.mlp.router.weight": torch.ones((3, 4)),
            }
        )
    with pytest.raises(ValueError, match="Final norm key"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "ln.weight": torch.ones((99,)),
            }
        )
    with pytest.raises(ValueError, match="Final norm bias"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "ln.weight": torch.ones((4,)),
                "ln.bias": torch.ones((99,)),
            }
        )
    with pytest.raises(ValueError, match="Incomplete Q/K"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((4, 4)),
            }
        )
    with pytest.raises(ValueError, match="must be 2D"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((4,)),
                "model.layers.0.self_attn.k_proj.weight": torch.ones((4, 4)),
            }
        )
    with pytest.raises(ValueError, match="input width"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((4, 9)),
                "model.layers.0.self_attn.k_proj.weight": torch.ones((4, 4)),
            }
        )
    with pytest.raises(ValueError, match="V projection"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((4, 4)),
                "model.layers.0.self_attn.k_proj.weight": torch.ones((2, 4)),
                "model.layers.0.self_attn.v_proj.weight": torch.ones((3, 4)),
            }
        )
    with pytest.raises(ValueError, match="O projection"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((4, 4)),
                "model.layers.0.self_attn.k_proj.weight": torch.ones((2, 4)),
                "model.layers.0.self_attn.o_proj.weight": torch.ones((9, 4)),
            }
        )
    with pytest.raises(ValueError, match="Inconsistent attention"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.1.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.self_attn.q_proj.weight": torch.ones((4, 4)),
                "model.layers.0.self_attn.k_proj.weight": torch.ones((2, 4)),
                "model.layers.1.self_attn.q_proj.weight": torch.ones((8, 4)),
                "model.layers.1.self_attn.k_proj.weight": torch.ones((2, 4)),
            }
        )
    with pytest.raises(ValueError, match="Inconsistent dims"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.0.mlp.experts.1.w3_A": torch.ones((2, 8)),
                "blocks.0.mlp.experts.1.w3_B": torch.ones((4, 2)),
            }
        )
    with pytest.raises(ValueError, match="Inconsistent LoRA rank"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.0.mlp.experts.1.w3_A": torch.ones((3, 6)),
                "blocks.0.mlp.experts.1.w3_B": torch.ones((4, 3)),
            }
        )
    with pytest.raises(ValueError, match="Invalid w1 LoRA shapes"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.0.mlp.experts.0.w1_A": torch.ones((2, 99)),
                "blocks.0.mlp.experts.0.w1_B": torch.ones((6, 2)),
            }
        )
    with pytest.raises(ValueError, match="has 1 experts; expected 2"):
        detect_open_weight_architecture(
            {
                "blocks.0.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.0.w3_B": torch.ones((4, 2)),
                "blocks.0.mlp.experts.1.w3_A": torch.ones((2, 6)),
                "blocks.0.mlp.experts.1.w3_B": torch.ones((4, 2)),
                "blocks.1.mlp.experts.0.w3_A": torch.ones((2, 6)),
                "blocks.1.mlp.experts.0.w3_B": torch.ones((4, 2)),
            }
        )
    with pytest.raises(ValueError, match="Incomplete LoRA pair"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.lora_A": torch.ones((2, 6)),
                "model.layers.0.mlp.down_proj.lora_B": torch.ones((4, 2)),
                "model.layers.0.mlp.gate_proj.lora_A": torch.ones((2, 4)),
            }
        )
    with pytest.raises(ValueError, match="Invalid gate_proj LoRA shapes"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.lora_A": torch.ones((2, 6)),
                "model.layers.0.mlp.down_proj.lora_B": torch.ones((4, 2)),
                "model.layers.0.mlp.gate_proj.lora_A": torch.ones((2, 99)),
                "model.layers.0.mlp.gate_proj.lora_B": torch.ones((6, 2)),
            }
        )
    with pytest.raises(ValueError, match="Invalid gate_proj.weight shape"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.mlp.gate_proj.weight": torch.ones((99, 4)),
            }
        )
    with pytest.raises(ValueError, match="mixed LoRA and full-weight"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.down_proj.lora_A": torch.ones((2, 6)),
                "model.layers.0.mlp.down_proj.lora_B": torch.ones((4, 2)),
                "model.layers.1.mlp.down_proj.weight": torch.ones((4, 6)),
            }
        )
    with pytest.raises(ValueError, match="Duplicate router/gate"):
        detect_open_weight_architecture(
            {
                "model.layers.0.mlp.experts.0.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.mlp.experts.1.down_proj.weight": torch.ones((4, 6)),
                "model.layers.0.mlp.gate.weight": torch.ones((2, 4)),
                "model.layers.0.mlp.router.weight": torch.ones((2, 4)),
            }
        )
    # Qwen-MoE LoRA variant auto-detection & lossless roundtrip
    qwen_moe_lora = {
        "model.layers.0.mlp.gate.weight": torch.ones((2, 4)),
        "model.layers.0.mlp.experts.0.down_proj.lora_A.weight": torch.ones((2, 6)),
        "model.layers.0.mlp.experts.0.down_proj.lora_B.weight": torch.ones((4, 2)),
        "model.layers.0.mlp.experts.1.down_proj.lora_A.weight": torch.ones((2, 6)),
        "model.layers.0.mlp.experts.1.down_proj.lora_B.weight": torch.ones((4, 2)),
    }
    qml_spec = detect_open_weight_architecture(qwen_moe_lora)
    assert qml_spec.arch_family == ARCH_QWEN_MOE and qml_spec.parameterization == PARAM_LORA and qml_spec.norm_type == "none"
    assert hash_native_state_dict(
        materialize_s5_back_to_native_layout(qwen_moe_lora, canonicalize_state_dict_for_s5(qwen_moe_lora))
    ) == hash_native_state_dict(qwen_moe_lora)

    # 2. Canonicalize / Materialize / Runner validation errors
    with pytest.raises(ValueError, match="full_weight_max_rank"):
        canonicalize_state_dict_for_s5(qwen_state, full_weight_max_rank=0)
    canon_ok = canonicalize_state_dict_for_s5(llama_state)
    with pytest.raises(ValueError, match="non-empty mapping"):
        materialize_s5_back_to_native_layout(llama_state, {})
    with pytest.raises(ValueError, match="keys mismatch"):
        materialize_s5_back_to_native_layout(llama_state, {"wrong_key": np.ones((2,))})
    bad_shape = dict(canon_ok)
    bad_shape["layer_0.w2_B"] = np.ones((1, 2, 2))
    with pytest.raises(ValueError, match="Shape mismatch"):
        materialize_s5_back_to_native_layout(llama_state, bad_shape)
    bad_nan = dict(canon_ok)
    bad_nan["layer_0.w2_B"] = np.full_like(canon_ok["layer_0.w2_B"], np.nan)
    with pytest.raises(ValueError, match="NaN or Inf"):
        materialize_s5_back_to_native_layout(llama_state, bad_nan)
    with pytest.raises(ValueError, match="identical keys"):
        count_native_edited_parameters({"a": torch.ones((2,))}, {"b": torch.ones((2,))})
    with pytest.raises(ValueError, match="Shape mismatch"):
        count_native_edited_parameters({"a": torch.ones((2,))}, {"a": torch.ones((3,))})
    with pytest.raises(TypeError, match="callable"):
        run_multi_arch_auto_open_weight_s5(llama_state, base_nll, "not_callable")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="Unexpected keyword"):
        run_multi_arch_auto_open_weight_s5(llama_state, base_nll, _ok_eval, bad_kw=1)
    with pytest.raises(ValueError, match="layer_e0_harm_nats is required"):
        run_multi_arch_auto_open_weight_s5(llama_state, base_nll, _ok_eval)
    with pytest.raises(ValueError, match="layer_e0_harm_nats length"):
        run_multi_arch_auto_open_weight_s5(
            llama_state, base_nll, _ok_eval, layer_e0_harm_nats=[0.0, 0.0]
        )
    with pytest.raises(ValueError, match="required for MoE"):
        run_multi_arch_auto_open_weight_s5(
            granite_state, base_nll, _ok_eval, layer_e0_harm_nats=[0.0]
        )
    with pytest.raises(ValueError, match="base_final_ln_b requires base_final_ln_w"):
        run_multi_arch_auto_open_weight_s5(
            granite_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats=[0.0, 0.0],
            expert_spec_loss_when_zeroed=[0.0, 0.0],
            base_final_ln_b=torch.zeros((8,)),
        )
    with pytest.raises(ValueError, match="biasless norm"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            base_final_ln_w=torch.ones((8,)),
            base_final_ln_b=torch.zeros((8,)),
        )
    with pytest.raises(ValueError, match="out of range"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            layer_prose_inputs={5: torch.ones((4, 8))},
        )
    with pytest.raises(ValueError, match="must have shape"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            tail_layer_prose_ffn_mid={0: torch.ones((4, 99))},
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            layer_specialist_inputs={},
        )
    with pytest.raises(ValueError, match="out of range"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            layer_specialist_inputs={5: {"E1": torch.ones((4, 8))}},
        )
    with pytest.raises(ValueError, match="must be a non-empty mapping"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            layer_specialist_inputs={0: {}},
        )
    with pytest.raises(ValueError, match="must be a non-empty string"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            layer_specialist_inputs={0: {"": torch.ones((4, 8))}},
        )
    with pytest.raises(ValueError, match="must have shape"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            layer_specialist_inputs={0: {"E1": torch.ones((4, 99))}},
        )
    with pytest.raises(TypeError, match="AutoOpenWeightConfig"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            config="bad",
        )
    with pytest.raises(ValueError, match="full_weight_max_rank"):
        run_multi_arch_auto_open_weight_s5(
            qwen_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats=[0.0, 0.0],
            expert_spec_loss_when_zeroed=[0.0, 0.0],
            full_weight_max_rank=0,
        )
    with pytest.raises(ValueError, match="base_final_ln_w shape"):
        run_multi_arch_auto_open_weight_s5(
            llama_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            base_final_ln_w=torch.ones((99,)),
        )
    with pytest.raises(ValueError, match="base_final_ln_b shape"):
        run_multi_arch_auto_open_weight_s5(
            granite_state,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats=[0.0, 0.0],
            expert_spec_loss_when_zeroed=[0.0, 0.0],
            base_final_ln_w=torch.ones((8,)),
            base_final_ln_b=torch.zeros((99,)),
        )
    # Verify Granite base_state norm+bias extraction
    _, granite_base_res = run_multi_arch_auto_open_weight_s5(
        granite_state,
        base_nll,
        _ok_eval,
        layer_e0_harm_nats=[0.0],
        expert_e0_harm_nats=[0.0, 0.0],
        expert_spec_loss_when_zeroed=[0.0, 0.0],
        base_state={"ln.weight": torch.ones((8,)), "ln.bias": torch.zeros((8,))},
    )
    assert granite_base_res.arch_family == ARCH_GRANITE_MOE
    with pytest.raises(TypeError, match="MultiArchAutoWSPResult"):
        verify_multi_arch_receipt_provenance("invalid")  # type: ignore[arg-type]

    # Verify fallback stub detector catches stub patterns cleanly
    fallback_auditor_check = _FallbackZeroStubAudit()
    assert fallback_auditor_check.audit_python_code("def f():\n    raise NotImplementedError()\n").is_clean is False

    # Verify receipt JSON emits zero forbidden patterns
    _, clean_res = run_multi_arch_auto_open_weight_s5(
        qwen_state,
        base_nll,
        _ok_eval,
        layer_e0_harm_nats=[0.0],
        expert_e0_harm_nats=[0.0, 0.0],
        expert_spec_loss_when_zeroed=[0.0, 0.0],
    )
    receipt_json = json.dumps(clean_res.to_dict(), sort_keys=True)
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn23", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    assert gatekeeper.forbidden_scan(f"diff --git a/x.py b/x.py\n+++ b/x.py\n+J = {receipt_json!r}") == []

    # 3. AGPL-3.0 header, CI-safe ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn23", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/multi_arch_auto_wsp_adapter.py", TARGET_MODULE_PATH),
        ("tests/test_multi_arch_auto_wsp_adapter_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"
        assert fallback_auditor.check_file(str(path_obj)) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True
        if ZeroStubAudit is not None:
            stub_res = ZeroStubAudit.audit_python_code(src_text)
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

    pr_audit = scanner.audit_pull_request(
        pr_number=1123,
        pr_title="feat(swarm): multi-architecture open-weight WSP adapter for Granite, Qwen, and Llama state dicts",
        pr_body="Implements NEURAL-CTRL-DYN-23 multi-arch WSP adapter and 5 deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): multi-architecture open-weight WSP adapter for Granite, Qwen, and Llama state dicts"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
