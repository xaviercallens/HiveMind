# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Five deterministic CPU scenarios for NEURAL-CTRL-DYN-27 AutoOpenWeightOrchestrator."""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
from collections.abc import Mapping, MutableMapping
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from hivemind.core.expert_registry import (
    CANDIDATE as EXPERT_CANDIDATE,
)
from hivemind.core.expert_registry import (
    LIVE as EXPERT_LIVE,
)
from hivemind.core.expert_registry import (
    SHADOW as EXPERT_SHADOW,
)
from hivemind.core.expert_registry import (
    ExpertRecord,
    ExpertRegistry,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.auto_open_weight_engine import (
    ANSE_MAXIMUM_PAIN,
    DEFAULT_GENERAL_DOMAIN,
    AutoOpenWeightConfig,
    SingleShotVaultLock,
)
from hivemind.swarm.auto_open_weight_orchestrator import (
    PIPELINE_MANIFEST_OBJECT,
    PREFLIGHT_DIAGNOSIS_OBJECT,
    PROMOTION_BUNDLE_OBJECT,
    SCHEMA_VERSION,
    AutoOpenWeightOrchestrator,
    AutoOpenWeightOrchestratorConfig,
    run_full_auto_open_weight_pipeline,
    verify_pipeline_manifest_provenance,
)
from hivemind.swarm.multi_arch_auto_wsp_adapter import (
    ARCH_GRANITE_MOE,
    ARCH_LLAMA_GQA,
    ARCH_QWEN_MOE,
    PARAM_FULL_WEIGHT,
    PARAM_LORA,
    clone_native_state_dict,
    hash_native_state_dict,
)
from hivemind.swarm.opentrainer_auto_wsp_hook import (
    PROMOTION_STAGE_PROMOTED_LIVE,
    PROMOTION_STAGE_PROMOTED_SHADOW,
    PROMOTION_STAGE_REJECTED_PARENT,
    PROMOTION_STAGE_REJECTED_RAW,
    verify_lineage_receipt_provenance,
)
from hivemind.swarm.volunteer_wsp_workunit_exporter import (
    MAX_NF4_NLL_GAP_NATS,
    VRAMCeilingExceededError,
    verify_workunit_receipt_provenance,
)
from hivemind.swarm.wsp_formal_invariant_verifier import (
    compute_payload_sha256,
    verify_wsp_formal_certificate_provenance,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """CI-safe AST stub audit result when optional ``gwaya`` is absent."""

    def __init__(self, violations: list[str]) -> None:
        self.violations = tuple(violations)
        self.is_clean = len(violations) == 0
        self.penalty_energy = 0.0 if self.is_clean else float(ANSE_MAXIMUM_PAIN)


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
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "auto_open_weight_orchestrator.py"
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


def _orthonormal_subspaces(
    seed: int,
    hidden_size: int,
    intermediate_size: int,
    n_spec: int,
) -> dict[str, Any]:
    """Construct deterministic orthogonal domain subspaces and prose/specialist activations."""
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
    """Build deterministic 4-layer, 6-expert Granite-MoE native state dict & evaluator."""
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
        4: spec_domains[2],  # dual-active multigate preserved expert
        5: spec_domains[3],
    }

    for l_idx in range(n_layers):
        state[f"blocks.{l_idx}.ln1.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.ln1.bias"] = torch.zeros((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.ln2.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.ln2.bias"] = torch.zeros((hidden_size,), dtype=torch.bfloat16)
        state[f"blocks.{l_idx}.attn.q.weight"] = (
            torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"blocks.{l_idx}.attn.k.weight"] = (
            torch.randn((hidden_size // 2, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"blocks.{l_idx}.attn.v.weight"] = (
            torch.randn((hidden_size // 2, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"blocks.{l_idx}.attn.o.weight"] = (
            torch.randn((hidden_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"blocks.{l_idx}.mlp.base_mlp.w1.weight"] = (
            torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"blocks.{l_idx}.mlp.base_mlp.w2.weight"] = (
            torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"blocks.{l_idx}.mlp.base_mlp.w3.weight"] = (
            torch.randn((hidden_size, intermediate_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )

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
                # Rogue expert 1 has huge norm spike (>= 2.2x median) detected by preflight
                scale = 3.8 if sink_e == 1 else 1.2
                rw[sink_e] = (2.0 * prose_orth_dir).astype(np.float32)
                w3_b[sink_e, :, 0] = (-0.32 * scale * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(
                    np.float32
                )
                w3_a[sink_e, 0, :] = prose_mid_subspace[:, idx_s].astype(np.float32)
                w1_a[sink_e, 0, :] = prose_orth_dir.astype(np.float32)
                w1_b[sink_e, :, 0] = (0.35 * prose_mid_subspace[:, idx_s]).astype(np.float32)
                w2_a[sink_e, 0, :] = prose_orth_dir.astype(np.float32)
                w2_b[sink_e, :, 0] = (0.35 * prose_mid_subspace[:, idx_s]).astype(np.float32)

            dom_dual = spec_expert_map[4]
            rw[4] = (2.2 * spec_dirs[dom_dual] + 1.25 * prose_orth_dir).astype(np.float32)
            w3_b[4, :, 0] = (
                0.55 * domain_target[dom_dual] - 0.16 * domain_target[DEFAULT_GENERAL_DOMAIN]
            ).astype(np.float32)
            w3_a[4, 0, :] = (spec_mid_dirs[dom_dual] + 0.35 * prose_mid_subspace[:, 0]).astype(
                np.float32
            )

            for exp_e in (2, 3, 5):
                dom = spec_expert_map[exp_e]
                rw[exp_e] = (
                    2.2 * spec_dirs[dom]
                    + (0.25 * prose_orth_dir if l_idx in tail_layers else -0.7 * prose_orth_dir)
                ).astype(np.float32)
                w3_b[exp_e, :, 0] = (
                    0.44 * domain_target[dom]
                    - (0.06 * domain_target[DEFAULT_GENERAL_DOMAIN] if l_idx in tail_layers else 0.0)
                ).astype(np.float32)
                w3_a[exp_e, 0, :] = (
                    spec_mid_dirs[dom]
                    + (0.32 * prose_mid_subspace[:, 1] if l_idx in tail_layers else 0.0)
                ).astype(np.float32)

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
    state["ln.weight"] = torch.full((hidden_size,), 1.24, dtype=torch.float32)
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
                    exp_out = mid @ (b3 @ a3).T
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
        layer_harm.append(
            float(raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_granite_native(byp)[DEFAULT_GENERAL_DOMAIN])
        )

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
    }


def _build_qwen_moe_world(seed: int = 20261125) -> dict[str, Any]:
    """Build deterministic 4-layer, 4-expert Qwen-MoE full-weight state dict & evaluator."""
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
        q_out = hidden_size
        kv_out = hidden_size // 4
        state[f"model.layers.{l_idx}.self_attn.q_proj.weight"] = (
            torch.randn((q_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.q_proj.bias"] = (
            torch.randn((q_out,), generator=gen, dtype=torch.bfloat16) * 0.01
        )
        state[f"model.layers.{l_idx}.self_attn.k_proj.weight"] = (
            torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.k_proj.bias"] = (
            torch.randn((kv_out,), generator=gen, dtype=torch.bfloat16) * 0.01
        )
        state[f"model.layers.{l_idx}.self_attn.v_proj.weight"] = (
            torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.v_proj.bias"] = (
            torch.randn((kv_out,), generator=gen, dtype=torch.bfloat16) * 0.01
        )
        state[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = (
            torch.randn((hidden_size, q_out), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.mlp.shared_expert.gate_proj.weight"] = (
            torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
        )
        state[f"model.layers.{l_idx}.mlp.shared_expert.up_proj.weight"] = (
            torch.randn((intermediate_size, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
        )
        state[f"model.layers.{l_idx}.mlp.shared_expert.down_proj.weight"] = (
            torch.randn((hidden_size, intermediate_size), generator=gen, dtype=torch.bfloat16) * 0.03
        )
        state[f"model.layers.{l_idx}.mlp.shared_expert_gate.weight"] = (
            torch.randn((1, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.03
        )

        rw = np.zeros((n_experts, hidden_size), dtype=np.float32)
        down_mats = np.zeros((n_experts, hidden_size, intermediate_size), dtype=np.float32)
        if l_idx == anchor_layer:
            rw[0] = (2.0 * prose_orth_dir).astype(np.float32)
            down_mats[0] = np.outer(
                0.45 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 0]
            ).astype(np.float32)
            for e_id, dom in spec_map.items():
                rw[e_id] = (2.2 * spec_dirs[dom]).astype(np.float32)
                down_mats[e_id] = np.outer(0.40 * domain_target[dom], spec_mid_dirs[dom]).astype(
                    np.float32
                )
        elif l_idx in interfering_layers:
            rw[0] = (2.0 * prose_orth_dir).astype(np.float32)
            down_mats[0] = np.outer(
                -0.55 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 0]
            ).astype(np.float32)
            dom_d = spec_map[2]
            rw[2] = (2.2 * spec_dirs[dom_d] + 1.2 * prose_orth_dir).astype(np.float32)
            down_mats[2] = (
                np.outer(0.55 * domain_target[dom_d], spec_mid_dirs[dom_d])
                - np.outer(0.16 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 1])
            ).astype(np.float32)
            for e_id in (1, 3):
                dom = spec_map[e_id]
                rw[e_id] = (
                    2.2 * spec_dirs[dom]
                    + (0.2 * prose_orth_dir if l_idx == 3 else -0.6 * prose_orth_dir)
                ).astype(np.float32)
                down_mats[e_id] = (
                    np.outer(0.44 * domain_target[dom], spec_mid_dirs[dom])
                    - (
                        np.outer(0.06 * domain_target[DEFAULT_GENERAL_DOMAIN], prose_mid_subspace[:, 2])
                        if l_idx == 3
                        else 0.0
                    )
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
                    dw = (
                        st[f"model.layers.{l_idx}.mlp.experts.{e_id}.down_proj.weight"]
                        .detach()
                        .to(torch.float64)
                        .numpy()
                    )
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
        layer_harm.append(
            float(raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_qwen_native(byp)[DEFAULT_GENERAL_DOMAIN])
        )

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
    """Build deterministic 4-layer Llama-GQA (4:1) LoRA state dict with biasless RMSNorm."""
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
        "model.norm.weight": torch.full((hidden_size,), 1.20, dtype=torch.float32),
    }

    anchor_layer = 1
    tail_layer = 3

    for l_idx in range(n_layers):
        state[f"model.layers.{l_idx}.input_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        state[f"model.layers.{l_idx}.post_attention_layernorm.weight"] = torch.ones((hidden_size,), dtype=torch.bfloat16)
        q_out = hidden_size
        kv_out = hidden_size // 4
        state[f"model.layers.{l_idx}.self_attn.q_proj.weight"] = (
            torch.randn((q_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.k_proj.weight"] = (
            torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.v_proj.weight"] = (
            torch.randn((kv_out, hidden_size), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.o_proj.weight"] = (
            torch.randn((hidden_size, q_out), generator=gen, dtype=torch.bfloat16) * 0.05
        )
        state[f"model.layers.{l_idx}.self_attn.q_proj.lora_A.weight"] = (
            torch.randn((rank, hidden_size), generator=gen, dtype=torch.float32) * 0.02
        )
        state[f"model.layers.{l_idx}.self_attn.q_proj.lora_B.weight"] = (
            torch.randn((q_out, rank), generator=gen, dtype=torch.float32) * 0.02
        )
        state[f"model.layers.{l_idx}.self_attn.k_proj.lora_A.weight"] = (
            torch.randn((rank, hidden_size), generator=gen, dtype=torch.float32) * 0.02
        )
        state[f"model.layers.{l_idx}.self_attn.k_proj.lora_B.weight"] = (
            torch.randn((kv_out, rank), generator=gen, dtype=torch.float32) * 0.02
        )

        for proj_name, in_d, out_d in (
            ("gate_proj", hidden_size, intermediate_size),
            ("up_proj", hidden_size, intermediate_size),
            ("down_proj", intermediate_size, hidden_size),
        ):
            state[f"model.layers.{l_idx}.mlp.{proj_name}.weight"] = (
                torch.randn((out_d, in_d), generator=gen, dtype=torch.bfloat16) * 0.04
            )
            state[f"model.layers.{l_idx}.mlp.{proj_name}.lora_A.weight"] = (
                torch.randn((rank, in_d), generator=gen, dtype=torch.float32) * 0.02
            )
            state[f"model.layers.{l_idx}.mlp.{proj_name}.lora_B.weight"] = (
                torch.randn((out_d, rank), generator=gen, dtype=torch.float32) * 0.02
            )

        db = np.zeros((hidden_size, rank), dtype=np.float32)
        da = np.zeros((rank, intermediate_size), dtype=np.float32)
        if l_idx == anchor_layer:
            db[:, 0] = (0.18 * domain_target[DEFAULT_GENERAL_DOMAIN]).astype(np.float32)
            da[0, :] = prose_mid_subspace[:, 0].astype(np.float32)
            for idx_d, dom in enumerate(spec_domains[:2], start=1):
                db[:, idx_d] = (0.36 * domain_target[dom]).astype(np.float32)
                da[idx_d, :] = spec_mid_dirs[dom].astype(np.float32)
        elif l_idx == tail_layer:
            for idx_d, dom in enumerate(spec_domains):
                db[:, idx_d] = (
                    0.48 * domain_target[dom] - 0.35 * domain_target[DEFAULT_GENERAL_DOMAIN]
                ).astype(np.float32)
                da[idx_d, :] = (
                    spec_mid_dirs[dom] + 0.60 * prose_mid_subspace[:, idx_d % 3]
                ).astype(np.float32)
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
        layer_harm.append(
            float(raw_nll[DEFAULT_GENERAL_DOMAIN] - eval_llama_native(byp)[DEFAULT_GENERAL_DOMAIN])
        )

    return {
        "state": state,
        "base_nll": base_nll,
        "eval_fn": eval_llama_native,
        "layer_harm": layer_harm,
        "tail_mid": {l_idx: torch.from_numpy(sub["domain_mid"][DEFAULT_GENERAL_DOMAIN]) for l_idx in range(n_layers)},
        "base_norm_w": torch.ones((hidden_size,), dtype=torch.float32),
    }


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


def test_s1_granite_moe_full_pipeline_nf4_certificate_and_shadow_canary() -> None:
    """S1: Granite-MoE end-to-end S5 checkpoint + PASS WSPFormalCertificate + <0.02 nat NF4 + shadow canary."""
    world = _build_granite_moe_world()
    state = world["state"]
    cfg = AutoOpenWeightOrchestratorConfig(
        s5_config=AutoOpenWeightConfig(
            router_beta=1.35,
            svd_damp_strength=0.90,
            lambda_ln_grid=(0.25, 0.75, 1.00),
            tau_temp_grid=(0.95, 1.00, 1.15),
        ),
        vram_budget_mib=6144.0,
        base_model_id="ibm-granite/granite-3.1-1b-a400m-base",
        cluster_id="granite_moe_swarm",
        round_id="s35_round_1",
    )

    shadow_reg = ExpertRegistry(strict=True)
    edited_state, res = run_full_auto_open_weight_pipeline(
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
        registry=shadow_reg,
        expert_id="granite_s5_bee",
        promote_to_live=False,
        run_id="granite_s1",
        config=cfg,
    )

    # 1. Preflight detected rogue expert norm spike and final LayerNorm drift
    assert res.preflight_diagnosis.arch_spec.arch_family == ARCH_GRANITE_MOE
    assert res.preflight_diagnosis.arch_spec.parameterization == PARAM_LORA
    assert res.preflight_diagnosis.healthy is False
    assert res.preflight_diagnosis.rogue_alert_count >= 1
    assert res.preflight_diagnosis.norm_drift_detected is True

    # 2. S5 surgery committed with monotone ANSE improvement and bitwise anchor protection
    assert res.committed is True
    assert res.verdict == "PASS"
    assert res.halted_before_nf4_export is False
    assert res.rolled_back_bitwise is False
    assert res.anse_energy < 0.0
    assert res.general_rel_drift <= 0.01
    assert res.all_domains_negative is True
    assert res.untouched_keys_bitwise_preserved is True

    for key in state:
        if key.startswith("blocks.2."):
            assert torch.equal(state[key], edited_state[key])
            assert state[key].dtype == edited_state[key].dtype
    for l_idx in (1, 3):
        for sink_e in (0, 1):
            for factor_b in ("w1_B", "w2_B", "w3_B"):
                k_b = f"blocks.{l_idx}.mlp.experts.{sink_e}.{factor_b}"
                assert float(torch.linalg.norm(edited_state[k_b]).item()) == 0.0

    # 3. Runtime algebraic & bitwise WSPFormalCertificate passed all 5 obligations
    assert res.formal_certificate is not None
    assert res.formal_certificate.certified is True
    assert res.formal_certificate.passed_obligations == 5
    assert res.formal_certificate.anchor_mutated_bits == 0
    assert verify_wsp_formal_certificate_provenance(res.formal_certificate) is True

    # 4. 4-bit NF4 Hessian-compensated edge bundle < 0.02 nat NLL gap within 8GB WDDM ceiling
    assert res.nf4_export_executed is True
    assert res.nf4_edge_bundle is not None
    assert res.nf4_gap_nats < MAX_NF4_NLL_GAP_NATS
    assert res.nf4_edge_bundle.nf4_gap_nats < 0.02
    assert res.nf4_edge_bundle.edge_vram_mib <= 6144.0
    assert res.nf4_edge_bundle.quantized_tensor_count > 0
    assert verify_workunit_receipt_provenance(res.nf4_edge_bundle.receipt) is True

    # 5. Shadow canary gate readiness & subsequent strict ExpertRegistry candidate -> shadow -> live
    assert res.promotion_bundle.shadow_canary_ready is True
    assert res.promotion_bundle.promoted_to_live is False
    assert res.promotion_bundle.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
    assert res.promotion_bundle.gate_payload["servable"] is True
    assert res.promotion_bundle.registry_state == EXPERT_SHADOW
    assert shadow_reg.get("granite_s5_bee").state == EXPERT_SHADOW
    assert verify_lineage_receipt_provenance(res.promotion_bundle.lineage_receipt) is True
    assert verify_pipeline_manifest_provenance(res) is True

    # Also test candidate -> shadow transition when expert already registered as CANDIDATE
    cfg_fast_no_nf4 = AutoOpenWeightOrchestratorConfig(
        s5_config=cfg.s5_config,
        enable_nf4_export=False,
        base_model_id=cfg.base_model_id,
        cluster_id=cfg.cluster_id,
        round_id=cfg.round_id,
    )
    cand_reg = ExpertRegistry(strict=True)
    cand_reg.register(
        ExpertRecord(
            expert_id="granite_cand_bee",
            kind="lora_domain",
            cluster_id="granite_moe_swarm",
            base_model_id="ibm-granite/granite-3.1-1b-a400m-base",
            adapter_sha256=res.post_native_sha256,
            adapter_size_mib=48,
            keywords=("wsp_s5",),
        )
    )
    assert cand_reg.get("granite_cand_bee").state == EXPERT_CANDIDATE
    _, res_cand = run_full_auto_open_weight_pipeline(
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
        registry=cand_reg,
        expert_id="granite_cand_bee",
        promote_to_live=False,
        config=cfg_fast_no_nf4,
    )
    assert cand_reg.get("granite_cand_bee").state == EXPERT_SHADOW
    assert res_cand.promotion_bundle.registry_state == EXPERT_SHADOW

    # Live promotion via canary_probe_fn (15/15 PASS)
    live_reg = ExpertRegistry(strict=True)
    _, res_live = run_full_auto_open_weight_pipeline(
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
        registry=live_reg,
        expert_id="granite_live_bee",
        canary_probe_fn=lambda st: (15, 15) if "ln.weight" in st else (0, 15),
        promote_to_live=True,
        run_id="granite_live",
        config=cfg_fast_no_nf4,
    )
    assert res_live.promotion_bundle.promoted_to_live is True
    assert res_live.promotion_bundle.promotion_stage == PROMOTION_STAGE_PROMOTED_LIVE
    assert live_reg.get("granite_live_bee").state == EXPERT_LIVE
    assert verify_pipeline_manifest_provenance(res_live) is True

    # Failing live canary probe (14/15 < 1.0) stays blocked in SHADOW
    blocked_reg = ExpertRegistry(strict=True)
    _, res_blocked = run_full_auto_open_weight_pipeline(
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
        registry=blocked_reg,
        expert_id="granite_blocked_bee",
        canary_probe_pass_count=14,
        canary_probe_total=15,
        shadow_general_drift_rel=0.001,
        promote_to_live=True,
        config=cfg_fast_no_nf4,
    )
    assert res_blocked.promotion_bundle.promoted_to_live is False
    assert res_blocked.promotion_bundle.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
    assert blocked_reg.get("granite_blocked_bee").state == EXPERT_SHADOW


def test_s2_qwen_moe_and_llama_gqa_end_to_end_bitwise_untouched() -> None:
    """S2: Qwen-MoE and Llama-GQA end-to-end pipelines preserving untouched weights bitwise."""
    # 1. Qwen-MoE (full weights + bfloat16 untouched embeddings/attention/shared experts + RMSNorm)
    qwen = _build_qwen_moe_world()
    q_state = qwen["state"]
    q_cfg = AutoOpenWeightOrchestratorConfig(
        s5_config=AutoOpenWeightConfig(
            router_beta=1.20,
            svd_damp_strength=0.90,
            lambda_ln_grid=(0.25, 0.75, 1.00),
            tau_temp_grid=(0.95, 1.00, 1.15),
        ),
        full_weight_max_rank=6,
        quantize_attention_nf4=True,
        base_model_id="Qwen/Qwen3-4B-Instruct-2507",
        round_id="qwen_s2",
    )
    edited_qwen, q_res = run_full_auto_open_weight_pipeline(
        q_state,
        qwen["base_nll"],
        qwen["eval_fn"],
        layer_e0_harm_nats=qwen["layer_harm"],
        expert_e0_harm_nats=qwen["exp_harm"],
        expert_spec_loss_when_zeroed=qwen["exp_spec"],
        layer_prose_inputs=qwen["layer_prose"],
        layer_specialist_inputs=qwen["layer_spec"],
        tail_layer_prose_ffn_mid=qwen["tail_mid"],
        base_final_ln_w=qwen["base_norm_w"],
        config=q_cfg,
    )
    assert q_res.committed is True
    assert q_res.verdict == "PASS"
    assert q_res.arch_family == ARCH_QWEN_MOE
    assert q_res.parameterization == PARAM_FULL_WEIGHT
    assert q_res.anse_energy < 0.0
    assert q_res.general_rel_drift <= 0.01
    assert q_res.untouched_keys_bitwise_preserved is True
    assert q_res.formal_certificate is not None and q_res.formal_certificate.certified is True
    assert q_res.nf4_edge_bundle is not None and q_res.nf4_gap_nats < 0.02
    assert verify_pipeline_manifest_provenance(q_res) is True

    for key in q_res.preflight_diagnosis.arch_spec.untouched_keys:
        assert torch.equal(q_state[key], edited_qwen[key])
        assert q_state[key].dtype == edited_qwen[key].dtype
    for key in q_state:
        if key.startswith("model.layers.2."):
            assert torch.equal(q_state[key], edited_qwen[key])
    assert float(torch.linalg.norm(edited_qwen["model.layers.1.mlp.experts.0.down_proj.weight"]).item()) == 0.0

    # 2. Llama-GQA (4:1 GQA LoRA + biasless RMSNorm temperature folding)
    llama = _build_llama_gqa_lora_world()
    l_state = llama["state"]
    l_cfg = AutoOpenWeightOrchestratorConfig(
        s5_config=AutoOpenWeightConfig(
            svd_damp_strength=0.95,
            tau_temp_grid=(0.95, 1.00, 1.12, 1.20),
        ),
        enable_nf4_export=True,
        base_model_id="meta-llama/Llama-3.2-3B",
        round_id="llama_s2",
    )
    edited_llama, l_res = run_full_auto_open_weight_pipeline(
        l_state,
        llama["base_nll"],
        llama["eval_fn"],
        layer_e0_harm_nats=llama["layer_harm"],
        tail_layer_prose_ffn_mid=llama["tail_mid"],
        config=l_cfg,
    )
    assert l_res.committed is True
    assert l_res.verdict == "PASS"
    assert l_res.arch_family == ARCH_LLAMA_GQA
    assert l_res.parameterization == PARAM_LORA
    assert l_res.preflight_diagnosis.arch_spec.gqa_kv_head_ratio == 4
    assert l_res.anse_energy < 0.0
    assert l_res.general_rel_drift <= 0.01
    assert l_res.untouched_keys_bitwise_preserved is True
    assert l_res.formal_certificate is not None and l_res.formal_certificate.certified is True
    assert l_res.formal_certificate.metadata["multigate_mode"] == "not_applicable_dense_or_unmodified"
    assert l_res.nf4_edge_bundle is not None and l_res.nf4_gap_nats < 0.02
    assert verify_pipeline_manifest_provenance(l_res) is True

    for k_untouched in (
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.layers.1.mlp.down_proj.lora_A.weight",
        "model.layers.1.mlp.down_proj.lora_B.weight",
        "model.layers.3.self_attn.q_proj.weight",
        "model.layers.3.self_attn.k_proj.lora_B.weight",
        "model.layers.3.mlp.gate_proj.lora_B.weight",
    ):
        assert torch.equal(l_state[k_untouched], edited_llama[k_untouched])
        assert l_state[k_untouched].dtype == edited_llama[k_untouched].dtype

    # Also verify optional explicit hessian_diags, no-tail-SVD fallback, OOF direction fallback,
    # explicit anchor_layer_prefixes_override, and enable_nf4_export=False path
    custom_hess = {
        k: torch.ones((int(v.shape[1]),), dtype=torch.float64)
        for k, v in edited_llama.items()
        if v.ndim == 2 and "down_proj.lora_" in k
    }
    sub_l = _orthonormal_subspaces(20261126, 16, 24, n_spec=3)
    _, l_res_no_nf4 = run_full_auto_open_weight_pipeline(
        l_state,
        llama["base_nll"],
        llama["eval_fn"],
        layer_e0_harm_nats=llama["layer_harm"],
        layer_prose_inputs={0: sub_l["domain_x"][DEFAULT_GENERAL_DOMAIN]},
        layer_specialist_inputs={0: {d: sub_l["domain_x"][d] for d in sub_l["spec_domains"]}},
        tail_layer_prose_ffn_mid=llama["tail_mid"],
        anchor_layer_prefixes_override=("model.layers.1.",),
        delta_w_override=np.zeros((16, 24), dtype=np.float64),
        u_gate_override=np.eye(16, dtype=np.float64)[0],
        specialist_centroids_override={"s": np.eye(16, dtype=np.float64)[1]},
        hessian_diags=custom_hess,
        config=AutoOpenWeightOrchestratorConfig(
            s5_config=l_cfg.s5_config,
            enable_nf4_export=False,
        ),
    )
    assert l_res_no_nf4.nf4_export_executed is False
    assert l_res_no_nf4.nf4_edge_bundle is None
    assert verify_pipeline_manifest_provenance(l_res_no_nf4) is True

    # Verify fallback projector/untouched-key anchor when tail_layer_prose_ffn_mid=None and no anchor layer
    l_no_norm = {k: v for k, v in l_state.items() if k != "model.norm.weight"}
    l_no_norm_diag = AutoOpenWeightOrchestrator().diagnose_preflight(l_no_norm)
    assert l_no_norm_diag.final_norm_max_abs_drift_from_unit == 0.0
    assert l_no_norm_diag.final_norm_mean_scale == 1.0

    l_sanitized_perturbed = clone_native_state_dict(edited_llama)
    l_sanitized_perturbed["model.norm.weight"] = torch.full((16,), 1.45, dtype=torch.float32)
    _, l_res_no_tail = run_full_auto_open_weight_pipeline(
        l_sanitized_perturbed,
        llama["base_nll"],
        llama["eval_fn"],
        layer_e0_harm_nats=[0.002, 0.002, 0.002, 0.002],
        layer_prose_inputs={0: sub_l["domain_x"][DEFAULT_GENERAL_DOMAIN]},
        layer_specialist_inputs={0: {d: sub_l["domain_x"][d] for d in sub_l["spec_domains"]}},
        config=AutoOpenWeightOrchestratorConfig(
            s5_config=l_cfg.s5_config,
            enable_nf4_export=False,
        ),
    )
    assert l_res_no_tail.committed is True
    assert l_res_no_tail.formal_certificate is not None and l_res_no_tail.formal_certificate.certified is True
    assert l_res_no_tail.formal_certificate.metadata["multigate_mode"] == "verified_from_oof_direction"


def test_s3_tampered_or_infeasible_guard_halts_before_nf4_with_maximum_pain() -> None:
    """S3: Tampered or infeasible guard halts before NF4 export and rolls back bitwise with ANSE Maximum Pain."""
    world = _build_granite_moe_world()
    state = world["state"]
    pre_sha = hash_native_state_dict(state)

    # 1. Infeasible general guard at Step 2 -> halts at s5_synergy before NF4 export
    def _infeasible_eval(_st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        return {
            dom: (4.90 if dom == DEFAULT_GENERAL_DOMAIN else 3.80)
            for dom in world["base_nll"]
        }

    out_inf, res_inf = run_full_auto_open_weight_pipeline(
        state,
        world["base_nll"],
        _infeasible_eval,
        layer_e0_harm_nats=world["layer_harm"],
        expert_e0_harm_nats=world["exp_harm"],
        expert_spec_loss_when_zeroed=world["exp_spec"],
        layer_prose_inputs=world["layer_prose"],
        layer_specialist_inputs=world["layer_spec"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
    )
    assert res_inf.committed is False
    assert res_inf.verdict == "ROLLED_BACK_MAXIMUM_PAIN"
    assert res_inf.halted_before_nf4_export is True
    assert res_inf.halted_stage == "s5_synergy"
    assert res_inf.rolled_back_bitwise is True
    assert res_inf.nf4_export_executed is False
    assert res_inf.nf4_edge_bundle is None
    assert res_inf.anse_energy == ANSE_MAXIMUM_PAIN
    assert res_inf.post_native_sha256 == pre_sha
    assert hash_native_state_dict(out_inf) == pre_sha
    assert res_inf.promotion_bundle.promotion_stage == PROMOTION_STAGE_REJECTED_RAW
    assert res_inf.promotion_bundle.shadow_canary_ready is False
    assert verify_pipeline_manifest_provenance(res_inf) is True

    # 2. Tampered anchor layer via extra_custom_stages -> Step 3 formal verifier catches anchor mutation & halts
    def _tamper_anchor_stage(canon: MutableMapping[str, np.ndarray]) -> int:
        w2_b = np.array(canon["layer_2.w2_B"], copy=True)
        w2_b[0, :, 0] += 0.25
        canon["layer_2.w2_B"] = w2_b
        return int(w2_b[0].size)

    out_anc, res_anc = run_full_auto_open_weight_pipeline(
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
        extra_custom_stages=(("tamper_anchor_layer_2", _tamper_anchor_stage),),
        config=AutoOpenWeightOrchestratorConfig(
            s5_config=AutoOpenWeightConfig(
                router_beta=1.35,
                svd_damp_strength=0.90,
                lambda_ln_grid=(0.25, 0.75, 1.00),
                tau_temp_grid=(0.95, 1.00, 1.15),
            ),
            enable_nf4_export=False,
        ),
    )
    assert res_anc.committed is False
    assert res_anc.verdict == "ROLLED_BACK_FORMAL_INVARIANT_FAILURE"
    assert res_anc.halted_before_nf4_export is True
    assert res_anc.halted_stage == "formal_verification"
    assert res_anc.rolled_back_bitwise is True
    assert res_anc.nf4_export_executed is False
    assert res_anc.nf4_edge_bundle is None
    assert res_anc.anse_energy == ANSE_MAXIMUM_PAIN
    assert res_anc.formal_certificate is not None
    assert res_anc.formal_certificate.certified is False
    assert "anchor_layer_bitwise_immutability" in res_anc.formal_certificate.failed_obligations
    assert res_anc.formal_certificate.anchor_mutated_bits > 0
    assert hash_native_state_dict(out_anc) == pre_sha
    assert verify_pipeline_manifest_provenance(res_anc) is True

    # 3. Tampered non-idempotent projector override -> Step 3 halts before NF4 export
    bad_p = np.eye(24, dtype=np.float64) * 2.5
    bad_p[0, 1] = 0.75
    out_proj, res_proj = run_full_auto_open_weight_pipeline(
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
        p_null_override=bad_p,
        delta_w_override=np.zeros((16, 24), dtype=np.float64),
        k_preserved_override=np.eye(24, dtype=np.float64),
        u_gate_override=np.eye(16, dtype=np.float64)[0],
        specialist_centroids_override={"s": np.eye(16, dtype=np.float64)[1]},
        prose_centroid_override=np.eye(16, dtype=np.float64)[0],
    )
    assert res_proj.halted_before_nf4_export is True
    assert res_proj.halted_stage == "formal_verification"
    assert res_proj.nf4_export_executed is False
    assert res_proj.anse_energy == ANSE_MAXIMUM_PAIN
    assert "projector_idempotence_and_symmetry" in res_proj.formal_certificate.failed_obligations  # type: ignore[union-attr]
    assert hash_native_state_dict(out_proj) == pre_sha

    # 4. Corrupted non-finite native_state with clean parent_state -> preflight bitwise rollback
    corrupted = clone_native_state_dict(state)
    corrupted["blocks.1.mlp.experts.0.w3_B"] = torch.full_like(
        corrupted["blocks.1.mlp.experts.0.w3_B"], float("nan")
    )
    out_nan, res_nan = run_full_auto_open_weight_pipeline(
        corrupted,
        world["base_nll"],
        world["eval_fn"],
        parent_state=state,
        layer_e0_harm_nats=world["layer_harm"],
        expert_e0_harm_nats=world["exp_harm"],
        expert_spec_loss_when_zeroed=world["exp_spec"],
    )
    assert res_nan.committed is False
    assert res_nan.halted_before_nf4_export is True
    assert res_nan.halted_stage == "preflight"
    assert res_nan.anse_energy == ANSE_MAXIMUM_PAIN
    assert res_nan.promotion_bundle.promotion_stage == PROMOTION_STAGE_REJECTED_PARENT
    assert hash_native_state_dict(out_nan) == pre_sha
    assert verify_pipeline_manifest_provenance(res_nan) is True

    # 5. Virgin vault guard breach at Step 4 -> halts before NF4 export & rolls back bitwise
    failing_vault_lock = SingleShotVaultLock("failing_vault_s3")
    out_vfail, res_vfail = run_full_auto_open_weight_pipeline(
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
        vault_lock=failing_vault_lock,
        vault_eval_fn=_infeasible_eval,
        base_vault_nll=world["base_nll"],
    )
    assert failing_vault_lock.consumed is True
    assert res_vfail.committed is False
    assert res_vfail.verdict == "ROLLED_BACK_VAULT_GUARD_FAILURE"
    assert res_vfail.halted_before_nf4_export is True
    assert res_vfail.halted_stage == "virgin_vault"
    assert res_vfail.nf4_export_executed is False
    assert res_vfail.anse_energy == ANSE_MAXIMUM_PAIN
    assert hash_native_state_dict(out_vfail) == pre_sha
    assert verify_pipeline_manifest_provenance(res_vfail) is True


def test_s4_single_shot_virgin_vault_lock_and_unified_manifest_verification() -> None:
    """S4: SingleShotVaultLock enforcement and unified SHA-256 manifest tamper detection."""
    world = _build_granite_moe_world()
    state = world["state"]
    vault_lock = SingleShotVaultLock("pilot2_vault_skip320")
    cfg = AutoOpenWeightOrchestratorConfig(
        s5_config=AutoOpenWeightConfig(
            router_beta=1.35,
            svd_damp_strength=0.90,
            lambda_ln_grid=(0.25, 0.75, 1.00),
            tau_temp_grid=(0.95, 1.00, 1.15),
        ),
        nf4_block_size=16,
        round_id="vault_s4",
    )

    _, res = run_full_auto_open_weight_pipeline(
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
        vault_lock=vault_lock,
        vault_eval_fn=world["eval_fn"],
        base_vault_nll=world["base_nll"],
        config=cfg,
    )

    assert res.committed is True
    assert vault_lock.consumed is True
    assert res.vault_receipt is not None
    assert res.vault_receipt["verdict_ip300"] == "PASS"
    assert res.vault_receipt["guard_ok"] is True
    assert res.vault_receipt["vault_receipt_sha256"] == vault_lock.receipt_sha256
    assert res.manifest["vault_receipt_sha256"] == vault_lock.receipt_sha256
    assert verify_pipeline_manifest_provenance(res) is True
    assert verify_pipeline_manifest_provenance(res.to_dict()) is True
    assert verify_pipeline_manifest_provenance(dict(res.manifest)) is True

    # Attempting a second run with the already-consumed virgin vault lock MUST fail immediately
    with pytest.raises(RuntimeError, match="already evaluated"):
        run_full_auto_open_weight_pipeline(
            state,
            world["base_nll"],
            world["eval_fn"],
            layer_e0_harm_nats=world["layer_harm"],
            expert_e0_harm_nats=world["exp_harm"],
            expert_spec_loss_when_zeroed=world["exp_spec"],
            vault_lock=vault_lock,
            vault_eval_fn=world["eval_fn"],
            base_vault_nll=world["base_nll"],
            config=cfg,
        )

    # Also test auto-instantiated vault lock when vault_eval_fn is passed without vault_lock
    _, res_auto_lock = run_full_auto_open_weight_pipeline(
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
        vault_eval_fn=world["eval_fn"],
        base_vault_nll=world["base_nll"],
        config=AutoOpenWeightOrchestratorConfig(
            s5_config=cfg.s5_config,
            enable_nf4_export=False,
            round_id="vault_s4",
        ),
    )
    assert res_auto_lock.vault_receipt is not None
    assert res_auto_lock.vault_receipt["vault_id"] == "vault_s4_virgin_vault"
    assert verify_pipeline_manifest_provenance(res_auto_lock) is True

    # Tamper detection across every link in the unified SHA-256 manifest
    base_doc = res.to_dict()
    assert verify_pipeline_manifest_provenance(base_doc) is True

    tampered_manifest_val = copy.deepcopy(base_doc)
    tampered_manifest_val["manifest"]["anse_energy"] = -99.0
    assert verify_pipeline_manifest_provenance(tampered_manifest_val) is False

    tampered_top_field = copy.deepcopy(base_doc)
    tampered_top_field["anse_energy"] = -99.0
    assert verify_pipeline_manifest_provenance(tampered_top_field) is False

    tampered_schema = copy.deepcopy(base_doc)
    tampered_schema["manifest"]["schema"] = "tampered/0"
    assert verify_pipeline_manifest_provenance(tampered_schema) is False

    tampered_preflight = copy.deepcopy(base_doc)
    tampered_preflight["preflight_diagnosis"]["rogue_alert_count"] = 99
    assert verify_pipeline_manifest_provenance(tampered_preflight) is False

    tampered_multi = copy.deepcopy(base_doc)
    tampered_multi["multi_arch_result"]["anse_energy"] = -9.9
    assert verify_pipeline_manifest_provenance(tampered_multi) is False

    tampered_cert = copy.deepcopy(base_doc)
    tampered_cert["formal_certificate"]["anchor_mutated_bits"] = 1
    assert verify_pipeline_manifest_provenance(tampered_cert) is False

    tampered_vault = copy.deepcopy(base_doc)
    tampered_vault["vault_receipt"]["anse_energy"] = -9.9
    assert verify_pipeline_manifest_provenance(tampered_vault) is False

    tampered_nf4 = copy.deepcopy(base_doc)
    tampered_nf4["nf4_edge_bundle"]["receipt"]["nll_gap_nats"] = 0.0
    assert verify_pipeline_manifest_provenance(tampered_nf4) is False

    tampered_promo = copy.deepcopy(base_doc)
    tampered_promo["promotion_bundle"]["shadow_canary_ready"] = False
    assert verify_pipeline_manifest_provenance(tampered_promo) is False

    tampered_lineage = copy.deepcopy(base_doc)
    tampered_lineage["promotion_bundle"]["lineage_receipt"]["s5_anse_energy"] = -8.0
    tampered_promo_core = {
        k: v for k, v in tampered_lineage["promotion_bundle"].items() if k != "receipt_sha256"
    }
    tampered_lineage["promotion_bundle"]["receipt_sha256"] = compute_payload_sha256(tampered_promo_core)
    assert verify_pipeline_manifest_provenance(tampered_lineage) is False

    # Additional structural/hash-mismatch checks across manifest verifier branches
    bad_manifest_type = copy.deepcopy(base_doc)
    bad_manifest_type["manifest"] = "not_a_mapping"
    assert verify_pipeline_manifest_provenance(bad_manifest_type) is False

    bad_manifest_sha_fmt = copy.deepcopy(base_doc)
    bad_manifest_sha_fmt["manifest"]["manifest_sha256"] = "not_hex"
    assert verify_pipeline_manifest_provenance(bad_manifest_sha_fmt) is False

    bad_manifest_top_sha = copy.deepcopy(base_doc)
    bad_manifest_top_sha["manifest_sha256"] = "0" * 64
    assert verify_pipeline_manifest_provenance(bad_manifest_top_sha) is False

    bad_manifest_nan = copy.deepcopy(base_doc)
    bad_manifest_nan["manifest"]["anse_energy"] = float("nan")
    assert verify_pipeline_manifest_provenance(bad_manifest_nan) is False

    bad_pre_meta = copy.deepcopy(base_doc)
    bad_pre_meta["preflight_diagnosis"]["schema"] = "bad"
    assert verify_pipeline_manifest_provenance(bad_pre_meta) is False

    bad_pre_hex = copy.deepcopy(base_doc)
    bad_pre_hex["preflight_diagnosis"]["receipt_sha256"] = "bad_hex"
    assert verify_pipeline_manifest_provenance(bad_pre_hex) is False

    bad_pre_nan = copy.deepcopy(base_doc)
    bad_pre_nan["preflight_diagnosis"]["final_norm_mean_scale"] = float("nan")
    assert verify_pipeline_manifest_provenance(bad_pre_nan) is False

    bad_pre_link = copy.deepcopy(base_doc)
    pre_core = {k: v for k, v in bad_pre_link["preflight_diagnosis"].items() if k != "receipt_sha256"}
    pre_core["rogue_alert_count"] = 77
    bad_pre_link["preflight_diagnosis"]["rogue_alert_count"] = 77
    bad_pre_link["preflight_diagnosis"]["receipt_sha256"] = compute_payload_sha256(pre_core)
    assert verify_pipeline_manifest_provenance(bad_pre_link) is False

    bad_multi_link = copy.deepcopy(base_doc)
    m_core = {k: v for k, v in bad_multi_link["multi_arch_result"].items() if k != "receipt_sha256"}
    m_core["nll_mean"] = 1.234
    bad_multi_link["multi_arch_result"]["nll_mean"] = 1.234
    bad_multi_link["multi_arch_result"]["receipt_sha256"] = compute_payload_sha256(m_core)
    assert verify_pipeline_manifest_provenance(bad_multi_link) is False

    bad_cert_link = copy.deepcopy(base_doc)
    c_core = {k: v for k, v in bad_cert_link["formal_certificate"].items() if k != "receipt_sha256"}
    c_core["surgery_id"] = "other_id"
    bad_cert_link["formal_certificate"]["surgery_id"] = "other_id"
    bad_cert_link["formal_certificate"]["receipt_sha256"] = compute_payload_sha256(c_core)
    assert verify_pipeline_manifest_provenance(bad_cert_link) is False

    bad_vault_nan = copy.deepcopy(base_doc)
    bad_vault_nan["vault_receipt"]["anse_energy"] = float("inf")
    assert verify_pipeline_manifest_provenance(bad_vault_nan) is False

    bad_vault_sha = copy.deepcopy(base_doc)
    bad_vault_sha["vault_receipt"]["vault_receipt_sha256"] = "bad"
    assert verify_pipeline_manifest_provenance(bad_vault_sha) is False

    bad_vault_link = copy.deepcopy(base_doc)
    v_core = {k: v for k, v in bad_vault_link["vault_receipt"].items() if k != "vault_receipt_sha256"}
    v_core["vault_id"] = "other_vault"
    bad_vault_link["vault_receipt"]["vault_id"] = "other_vault"
    bad_vault_link["vault_receipt"]["vault_receipt_sha256"] = compute_payload_sha256(v_core)
    assert verify_pipeline_manifest_provenance(bad_vault_link) is False

    bad_nf4_missing = copy.deepcopy(base_doc)
    bad_nf4_missing["nf4_edge_bundle"] = {"receipt": "not_a_mapping"}
    assert verify_pipeline_manifest_provenance(bad_nf4_missing) is False

    bad_nf4_link = copy.deepcopy(base_doc)
    n_core = {k: v for k, v in bad_nf4_link["nf4_edge_bundle"]["receipt"].items() if k != "receipt_sha256"}
    n_core["block_size"] = 32
    bad_nf4_link["nf4_edge_bundle"]["receipt"]["block_size"] = 32
    bad_nf4_link["nf4_edge_bundle"]["receipt"]["receipt_sha256"] = compute_payload_sha256(n_core)
    assert verify_pipeline_manifest_provenance(bad_nf4_link) is False

    bad_nf4_weights_link = copy.deepcopy(base_doc)
    bad_nf4_weights_link["manifest"]["nf4_weights_sha256"] = "0" * 64
    m_w_core = {k: v for k, v in bad_nf4_weights_link["manifest"].items() if k != "manifest_sha256"}
    bad_nf4_weights_link["manifest"]["manifest_sha256"] = compute_payload_sha256(m_w_core)
    bad_nf4_weights_link["manifest_sha256"] = bad_nf4_weights_link["manifest"]["manifest_sha256"]
    assert verify_pipeline_manifest_provenance(bad_nf4_weights_link) is False

    bad_promo_type = copy.deepcopy(base_doc)
    bad_promo_type["promotion_bundle"] = "not_a_mapping"
    assert verify_pipeline_manifest_provenance(bad_promo_type) is False

    bad_promo_nan = copy.deepcopy(base_doc)
    bad_promo_nan["promotion_bundle"]["extra_bad"] = float("nan")
    assert verify_pipeline_manifest_provenance(bad_promo_nan) is False

    bad_lineage_link = copy.deepcopy(base_doc)
    bad_lineage_link["manifest"]["lineage_receipt_sha256"] = "0" * 64
    m_l_core = {k: v for k, v in bad_lineage_link["manifest"].items() if k != "manifest_sha256"}
    bad_lineage_link["manifest"]["manifest_sha256"] = compute_payload_sha256(m_l_core)
    bad_lineage_link["manifest_sha256"] = bad_lineage_link["manifest"]["manifest_sha256"]
    assert verify_pipeline_manifest_provenance(bad_lineage_link) is False

    bad_lineage_obj = copy.deepcopy(base_doc)
    bad_lineage_obj["promotion_bundle"]["lineage_receipt"] = "not_a_mapping"
    p_l_core = {k: v for k, v in bad_lineage_obj["promotion_bundle"].items() if k != "receipt_sha256"}
    p_l_sha = compute_payload_sha256(p_l_core)
    bad_lineage_obj["promotion_bundle"]["receipt_sha256"] = p_l_sha
    bad_lineage_obj["manifest"]["promotion_bundle_sha256"] = p_l_sha
    m_lo_core = {k: v for k, v in bad_lineage_obj["manifest"].items() if k != "manifest_sha256"}
    bad_lineage_obj["manifest"]["manifest_sha256"] = compute_payload_sha256(m_lo_core)
    bad_lineage_obj["manifest_sha256"] = bad_lineage_obj["manifest"]["manifest_sha256"]
    assert verify_pipeline_manifest_provenance(bad_lineage_obj) is False


def test_s5_validation_errors_barrier_normalization_and_compliance_gates() -> None:
    """S5: Config/input validation errors, pain-barrier stage normalization, and security/stub gates."""
    tiny = _tiny_granite_state()
    base_nll = {DEFAULT_GENERAL_DOMAIN: 4.0, "E1_code": 4.0}

    def _ok_eval(_st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        return {DEFAULT_GENERAL_DOMAIN: 3.99, "E1_code": 3.50}

    # 1. Config validation checks
    with pytest.raises(TypeError, match="AutoOpenWeightConfig"):
        AutoOpenWeightOrchestratorConfig(s5_config="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unsupported arch_hint"):
        AutoOpenWeightOrchestratorConfig(arch_hint="unknown_arch")
    with pytest.raises(ValueError, match="full_weight_max_rank"):
        AutoOpenWeightOrchestratorConfig(full_weight_max_rank=0)
    with pytest.raises(TypeError, match="full_weight_max_rank"):
        AutoOpenWeightOrchestratorConfig(full_weight_max_rank=True)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="auto_temperature_calibration"):
        AutoOpenWeightOrchestratorConfig(auto_temperature_calibration="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="rogue_norm_ratio_threshold"):
        AutoOpenWeightOrchestratorConfig(rogue_norm_ratio_threshold=1.0)
    with pytest.raises(ValueError, match="rogue_norm_ratio_threshold"):
        AutoOpenWeightOrchestratorConfig(rogue_norm_ratio_threshold=0.0)
    with pytest.raises(ValueError, match="ln_drift_threshold"):
        AutoOpenWeightOrchestratorConfig(ln_drift_threshold=-0.1)
    with pytest.raises(TypeError, match="projector_tol"):
        AutoOpenWeightOrchestratorConfig(projector_tol="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="projector_tol"):
        AutoOpenWeightOrchestratorConfig(projector_tol=float("nan"))
    with pytest.raises(ValueError, match="canary_min_pass_ratio"):
        AutoOpenWeightOrchestratorConfig(canary_min_pass_ratio=0.0)
    with pytest.raises(ValueError, match="canary_min_pass_ratio"):
        AutoOpenWeightOrchestratorConfig(canary_min_pass_ratio=1.5)
    with pytest.raises(ValueError, match="base_model_id"):
        AutoOpenWeightOrchestratorConfig(base_model_id="  ")
    with pytest.raises(TypeError, match="base_model_id"):
        AutoOpenWeightOrchestratorConfig(base_model_id=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="canary_default_probe_pass_count"):
        AutoOpenWeightOrchestratorConfig(canary_default_probe_pass_count=-1)
    with pytest.raises(VRAMCeilingExceededError):
        AutoOpenWeightOrchestratorConfig(vram_budget_mib=8192.0)
    with pytest.raises(ValueError, match="canary default probe"):
        AutoOpenWeightOrchestratorConfig(canary_default_probe_pass_count=20, canary_default_probe_total=15)
    with pytest.raises(TypeError, match="keywords"):
        AutoOpenWeightOrchestratorConfig(keywords="bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AutoOpenWeightOrchestratorConfig"):
        AutoOpenWeightOrchestrator(config="bad")  # type: ignore[arg-type]

    # 2. Pipeline input validation checks
    with pytest.raises(TypeError, match="callable"):
        run_full_auto_open_weight_pipeline(
            tiny, base_nll, "not_callable", layer_e0_harm_nats=[0.1]  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        run_full_auto_open_weight_pipeline(
            tiny, {}, _ok_eval, layer_e0_harm_nats=[0.1]
        )
    with pytest.raises(ValueError, match="general_domain"):
        run_full_auto_open_weight_pipeline(
            tiny, {"E1_code": 4.0}, _ok_eval, layer_e0_harm_nats=[0.1]
        )
    with pytest.raises(ValueError, match="non-empty mapping"):
        run_full_auto_open_weight_pipeline(
            {}, base_nll, _ok_eval, layer_e0_harm_nats=[0.1]
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        run_full_auto_open_weight_pipeline(
            {"x": np.ones((2, 2))}, base_nll, _ok_eval, layer_e0_harm_nats=[0.1]  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="non-empty"):
        run_full_auto_open_weight_pipeline(
            {"x": torch.empty((0,))}, base_nll, _ok_eval, layer_e0_harm_nats=[0.1]
        )
    with pytest.raises(ValueError, match="must reside on CPU"):
        run_full_auto_open_weight_pipeline(
            {"x": torch.empty((2, 2), device="meta")}, base_nll, _ok_eval, layer_e0_harm_nats=[0.1]
        )
    with pytest.raises(ValueError, match="vault_eval_fn and base_vault_nll"):
        run_full_auto_open_weight_pipeline(
            tiny, base_nll, _ok_eval, layer_e0_harm_nats=[0.1], vault_eval_fn=_ok_eval
        )
    with pytest.raises(ValueError, match="vault_lock requires"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            vault_lock=SingleShotVaultLock("unused"),
        )
    with pytest.raises(TypeError, match="SingleShotVaultLock"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            vault_lock="bad",  # type: ignore[arg-type]
            vault_eval_fn=_ok_eval,
            base_vault_nll=base_nll,
        )
    with pytest.raises(TypeError, match="vault_eval_fn must be callable"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            vault_eval_fn="bad",  # type: ignore[arg-type]
            base_vault_nll=base_nll,
        )
    with pytest.raises(TypeError, match="ExpertRegistry"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            registry="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="canary_probe_fn"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            canary_probe_fn="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="canary_probe_pass_count and canary_probe_total"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            canary_probe_pass_count=5,
        )
    with pytest.raises(ValueError, match="cannot exceed"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            canary_probe_pass_count=6,
            canary_probe_total=5,
        )
    with pytest.raises(ValueError, match="promote_to_live=True requires"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            promote_to_live=True,
            expert_id="standalone_expert",
        )
    with pytest.raises(ValueError, match="parent_state keys"):
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            parent_state={"wrong_key": torch.ones((2,))},
        )
    with pytest.raises(ValueError, match="shape"):
        bad_shape_parent = clone_native_state_dict(tiny)
        bad_shape_parent["ln.weight"] = torch.ones((16,), dtype=torch.float32)
        run_full_auto_open_weight_pipeline(
            tiny,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.1],
            parent_state=bad_shape_parent,
        )
    with pytest.raises(ValueError, match="contains non-finite"):
        bad_tiny = clone_native_state_dict(tiny)
        bad_tiny["ln.weight"] = torch.full((8,), float("inf"))
        run_full_auto_open_weight_pipeline(
            bad_tiny, base_nll, _ok_eval, layer_e0_harm_nats=[0.1]
        )
    with pytest.raises(TypeError, match="AutoOpenWeightPipelineResult"):
        verify_pipeline_manifest_provenance("invalid")  # type: ignore[arg-type]
    assert verify_pipeline_manifest_provenance({"schema": "bad", "object": PIPELINE_MANIFEST_OBJECT}) is False

    # 3. Exercise multi-stage pain-barrier normalization (S5.1 reduces infeasible drift, S5.2 rescues guard)
    world = _build_granite_moe_world()
    state = world["state"]

    def _two_step_rescue_eval(st: Mapping[str, torch.Tensor]) -> dict[str, float]:
        out = world["eval_fn"](st)
        # Keep E0 slightly above +1.0% while both sinks not zeroed, or before router & LN finish rescue
        b1_l1 = float(torch.linalg.norm(st["blocks.1.mlp.experts.1.w3_B"]).item())
        rw_delta = float(
            torch.linalg.norm(
                st["blocks.1.mlp.router.weight"][4] - state["blocks.1.mlp.router.weight"][4]
            ).item()
        )
        if b1_l1 > 1e-6:
            out[DEFAULT_GENERAL_DOMAIN] = world["base_nll"][DEFAULT_GENERAL_DOMAIN] * 1.030
        elif rw_delta < 1e-6:
            out[DEFAULT_GENERAL_DOMAIN] = world["base_nll"][DEFAULT_GENERAL_DOMAIN] * 1.015
        return out

    _, res_rescue = run_full_auto_open_weight_pipeline(
        state,
        world["base_nll"],
        _two_step_rescue_eval,
        layer_e0_harm_nats=world["layer_harm"],
        expert_e0_harm_nats=world["exp_harm"],
        expert_spec_loss_when_zeroed=world["exp_spec"],
        layer_prose_inputs=world["layer_prose"],
        layer_specialist_inputs=world["layer_spec"],
        tail_layer_prose_ffn_mid=world["tail_mid"],
        base_final_ln_w=world["base_ln_w"],
        base_final_ln_b=world["base_ln_b"],
        config=AutoOpenWeightOrchestratorConfig(
            s5_config=AutoOpenWeightConfig(
                router_beta=1.35,
                svd_damp_strength=0.90,
                lambda_ln_grid=(0.25, 0.75, 1.00),
                tau_temp_grid=(0.95, 1.00, 1.15),
            ),
            enable_nf4_export=False,
        ),
    )
    assert res_rescue.committed is True
    assert res_rescue.formal_certificate is not None and res_rescue.formal_certificate.certified is True
    assert verify_pipeline_manifest_provenance(res_rescue) is True

    # 4. AGPL-3.0 header, CI-safe ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn27", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn27", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    assert fallback_auditor.audit_python_code("def f():\n    raise NotImplementedError()\n").is_clean is False

    changed_files: list[dict[str, Any]] = []
    for rel, path_obj in (
        ("src/hivemind/swarm/auto_open_weight_orchestrator.py", TARGET_MODULE_PATH),
        ("tests/test_auto_open_weight_orchestrator_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"
        assert fallback_auditor.check_file(path_obj) == []
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

    serialized_json = json.dumps(res_rescue.to_dict(), sort_keys=True)
    assert SCHEMA_VERSION in serialized_json
    assert PREFLIGHT_DIAGNOSIS_OBJECT in serialized_json
    assert PROMOTION_BUNDLE_OBJECT in serialized_json
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {serialized_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1127,
        pr_title="feat(swarm): end-to-end AutoOpenWeightOrchestrator wiring multi-arch S5 surgery, formal certificate, and NF4 edge export",
        pr_body="Implements NEURAL-CTRL-DYN-27 unified AutoOpenWeightOrchestrator and 5 deterministic CPU scenarios.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): end-to-end AutoOpenWeightOrchestrator wiring multi-arch S5 surgery, formal certificate, and NF4 edge export"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
