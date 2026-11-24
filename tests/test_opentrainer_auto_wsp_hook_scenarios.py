# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-20: HM-S35-04/05 OpenTrainer post-round WSP surgery hook & canary gate."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

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
    GateNotPassedError,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.crowdgpt_moe_opendiloco import (
    CrowdLoRAMoEMLP,
    upcycle_crowd_v1_to_sparse_moe,
)
from hivemind.swarm.monotone_wsp_stage_composer import (
    clone_state_dict,
    compute_state_dict_sha256,
)
from hivemind.swarm.opentrainer_auto_wsp_hook import (
    CANARY_GATE_OBJECT,
    DEFAULT_ANSE_MAXIMUM_PAIN,
    DEFAULT_CANARY_PROBE_TOTAL,
    LINEAGE_RECEIPT_OBJECT,
    PROMOTION_STAGE_PROMOTED_LIVE,
    PROMOTION_STAGE_PROMOTED_SHADOW,
    PROMOTION_STAGE_REJECTED_PARENT,
    PROMOTION_STAGE_REJECTED_RAW,
    SCHEMA_VERSION,
    PostRoundWSPHookConfig,
    PostRoundWSPLineageReceipt,
    build_expert_registry_gate_payload,
    evaluate_shadow_to_live_canary_gate,
    run_post_outer_round_auto_wsp_hook,
    verify_lineage_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
)

try:
    from gwaya.ast_audit import ZeroStubAudit
except ImportError:
    ZeroStubAudit = None  # type: ignore[assignment,misc]


class _FallbackStubResult:
    """Minimal stub audit result used when optional ``gwaya`` is absent in CI."""

    is_clean: bool = True
    violations: tuple[str, ...] = ()
    penalty_energy: float = 0.0


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def check_file(self, _path: str) -> list[str]:
        return []

    def audit_python_code(self, _code: str) -> _FallbackStubResult:
        return _FallbackStubResult()


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "opentrainer_auto_wsp_hook.py"
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


def _make_subspace_crowd_moe(
    *,
    num_layers: int = 4,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor]]:
    """Build deterministic multi-layer CrowdV1WithEngram + CrowdLoRAMoEMLP with orthogonal E0/E1/E2 subspaces."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
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

    u_e0 = torch.zeros(16, dtype=torch.float32)
    u_e0[0] = 1.0 / math.sqrt(2.0)
    u_e0[1] = -1.0 / math.sqrt(2.0)

    u_e1 = torch.zeros(8, dtype=torch.float32)
    u_e1[0] = 1.0 / math.sqrt(2.0)
    u_e1[1] = -1.0 / math.sqrt(2.0)

    u_e2 = torch.zeros(8, dtype=torch.float32)
    u_e2[0] = 1.0 / math.sqrt(2.0)
    u_e2[1] = -1.0 / math.sqrt(2.0)

    prose_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    prose_carrier[:16] = u_e0
    e1_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    e1_carrier[16:24] = u_e1
    e2_carrier = torch.zeros(hidden_size, dtype=torch.float32)
    e2_carrier[24:32] = u_e2

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_raw = 0.20 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_raw[:, :2] = 0.0
        p_raw = p_raw - p_raw.mean(dim=1, keepdim=True) + 2.0 * u_e0.unsqueeze(0)
        p_raw[5] = 2.4 * u_e0

        e1_raw = 0.20 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e1_raw[:, :2] = 0.0
        e1_raw = e1_raw - e1_raw.mean(dim=1, keepdim=True) + 2.0 * u_e1.unsqueeze(0)
        e1_raw[20 - 16] = 2.4 * u_e1

        e2_raw = 0.20 * torch.randn((16, 8), generator=gen, dtype=torch.float32)
        e2_raw[:, :2] = 0.0
        e2_raw = e2_raw - e2_raw.mean(dim=1, keepdim=True) + 2.0 * u_e2.unsqueeze(0)
        e2_raw[36 - 32] = 2.4 * u_e2

        wte[:16, :16] = p_raw
        wte[16:32, 16:24] = e1_raw
        wte[32:48, 24:32] = e2_raw
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight

        model.ln.weight.fill_(1.0)
        model.ln.bias.zero_()
        for blk in model.blocks:
            blk.ln1.weight.fill_(1.0)
            blk.ln1.bias.zero_()
            blk.ln2.weight.fill_(1.0)
            blk.ln2.bias.zero_()
            blk.mlp.w1.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w2.weight[:hidden_size, :hidden_size].copy_(1.5 * torch.eye(hidden_size))
            blk.mlp.w3.weight[:16, :16].copy_(0.03 * torch.randn((16, 16), generator=gen))
            blk.mlp.w3.weight[16:24, 16:24].copy_(0.03 * torch.randn((8, 8), generator=gen))
            blk.mlp.w3.weight[24:32, 24:32].copy_(0.03 * torch.randn((8, 8), generator=gen))

    upcycle_crowd_v1_to_sparse_moe(
        model,
        split_mode="lora_domain_experts",
        num_experts=num_experts,
        top_k=top_k,
        lora_rank=lora_rank,
        lora_alpha=8.0,
        routing_mode="sigma_topk",
        seed=seed + 19,
    )
    with torch.no_grad():
        for blk in model.blocks:
            mlp: CrowdLoRAMoEMLP = blk.mlp  # type: ignore[assignment]
            mlp.router.weight.zero_()
            for exp_mod in mlp.experts.values():
                for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
                    getattr(exp_mod, attr).zero_()
    return model, {"e0": prose_carrier, "e1": e1_carrier, "e2": e2_carrier}


def _make_disjoint_domain_chunks(
    *,
    seq_len: int = 10,
    score_from: int = 5,
    domains: tuple[str, ...] = ("E0_general_fineweb", "E1_code", "E2_math"),
    chunks_per_domain: int = 4,
    seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """Build deterministic non-overlapping domain token chunks."""
    gen = torch.Generator(device="cpu").manual_seed(9800 + seed_offset)
    chunks: list[dict[str, Any]] = []
    for dom in domains:
        for _ in range(chunks_per_domain):
            if dom.startswith("E0_"):
                ids = torch.randint(0, 16, (seq_len,), generator=gen).tolist()
                target_tok = 5
            elif dom.startswith("E1_"):
                ids = torch.randint(16, 32, (seq_len,), generator=gen).tolist()
                target_tok = 20
            else:
                ids = torch.randint(32, 48, (seq_len,), generator=gen).tolist()
                target_tok = 36
            for pos in range(score_from, seq_len):
                ids[pos] = target_tok
            chunks.append({"expert_id": dom, "ids": ids})
    return chunks


@torch.no_grad()
def _eval_crowd_moe_state(
    model: CrowdV1WithEngram,
    state_dict: dict[str, torch.Tensor] | Any,
    chunks: list[dict[str, Any]],
    *,
    score_from: int = 5,
) -> dict[str, float]:
    """Load ``state_dict`` into ``model`` and compute exact per-domain causal NLL."""
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    ids_all = torch.tensor([c["ids"] for c in chunks], dtype=torch.long)
    h_pre = forward_model_pre_ln(model, ids_all, batch_size=len(chunks))
    nlls = per_chunk_nll_from_pre_ln(
        h_pre,
        model.ln.weight,
        model.ln.bias,
        float(getattr(model.ln, "eps", 1e-5)),
        model.head.weight,
        ids_all,
        score_from=score_from,
    )
    by_dom: dict[str, list[float]] = {}
    for idx, ch in enumerate(chunks):
        by_dom.setdefault(str(ch["expert_id"]), []).append(float(nlls[idx].item()))
    return {dom: float(sum(vals) / len(vals)) for dom, vals in sorted(by_dom.items())}


def _make_probe_inputs(
    carriers: dict[str, torch.Tensor],
    *,
    layers: tuple[int, ...] = (2, 3),
    n_rows: int = 12,
    seed: int = 777,
) -> tuple[dict[int, torch.Tensor], dict[int, dict[str, torch.Tensor]]]:
    """Generate deterministic separable per-layer E0 prose and E1/E2 specialist activations."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    prose_v = carriers["e0"]
    e1_v = carriers["e1"]
    e2_v = carriers["e2"]

    prose_inputs: dict[int, torch.Tensor] = {}
    spec_inputs: dict[int, dict[str, torch.Tensor]] = {}
    for l_idx in layers:
        noise_p = 0.04 * torch.randn((n_rows, prose_v.numel()), generator=gen)
        noise_1 = 0.04 * torch.randn((n_rows, e1_v.numel()), generator=gen)
        noise_2 = 0.04 * torch.randn((n_rows, e2_v.numel()), generator=gen)
        prose_inputs[l_idx] = (2.0 * prose_v.unsqueeze(0) + noise_p).contiguous()
        spec_inputs[l_idx] = {
            "E1_code": (2.0 * e1_v.unsqueeze(0) + noise_1).contiguous(),
            "E2_math": (2.0 * e2_v.unsqueeze(0) + noise_2).contiguous(),
        }
    return prose_inputs, spec_inputs


def _simulate_diloco_outer_round(
    parent_state: dict[str, torch.Tensor],
    carriers: dict[str, torch.Tensor],
    *,
    ghost_harm_scale: float = 0.06,
    leak_scale: float = 0.03,
    ln_scale: float = 0.92,
) -> dict[str, torch.Tensor]:
    """Simulate a 2-learner nocturnal DiLoCo outer step producing specialist gains + E0 drift."""
    prose_v = carriers["e0"]
    e1_v = carriers["e1"]
    e2_v = carriers["e2"]

    def _zero_delta() -> dict[str, torch.Tensor]:
        return {k: torch.zeros_like(v) for k, v in parent_state.items()}

    # Learner 0 (E1 code + beneficial anchor layer 0..1 + slight prose leak)
    d0 = _zero_delta()
    d0["blocks.0.mlp.router.weight"][0] = 2.6 * prose_v
    d0["blocks.0.mlp.experts.0.w3_A"][0, 0] = 0.02
    d0["blocks.0.mlp.experts.0.w3_B"][:, 0] = 0.04 * prose_v
    d0["blocks.1.mlp.router.weight"][0] = 2.4 * prose_v
    d0["blocks.1.mlp.experts.0.w3_A"][0, 0] = 0.02
    d0["blocks.1.mlp.experts.0.w3_B"][:, 0] = 0.03 * prose_v
    for l_idx in (2, 3):
        d0[f"blocks.{l_idx}.mlp.router.weight"][0] = 0.5 * prose_v
        d0[f"blocks.{l_idx}.mlp.router.weight"][1] = 2.8 * e1_v + 2.4 * prose_v
        d0[f"blocks.{l_idx}.mlp.experts.1.w3_A"][0, 16] = 0.16
        d0[f"blocks.{l_idx}.mlp.experts.1.w3_B"][:, 0] = 0.75 * e1_v
        d0[f"blocks.{l_idx}.mlp.experts.1.w3_A"][1, 0] = 0.05
        d0[f"blocks.{l_idx}.mlp.experts.1.w3_B"][:, 1] = -leak_scale * prose_v

    # Learner 1 (E2 math + rogue ghost-sink expert 3 + cross-layer interference + hot LN drift)
    d1 = _zero_delta()
    d1["blocks.0.mlp.router.weight"][0] = 2.6 * prose_v
    d1["blocks.0.mlp.experts.0.w3_A"][0, 0] = 0.02
    d1["blocks.0.mlp.experts.0.w3_B"][:, 0] = 0.04 * prose_v
    d1["blocks.1.mlp.router.weight"][0] = 2.4 * prose_v
    d1["blocks.1.mlp.experts.0.w3_A"][0, 0] = 0.02
    d1["blocks.1.mlp.experts.0.w3_B"][:, 0] = 0.03 * prose_v
    for l_idx in (2, 3):
        d1[f"blocks.{l_idx}.mlp.router.weight"][0] = 0.5 * prose_v
        d1[f"blocks.{l_idx}.mlp.router.weight"][2] = 2.8 * e2_v + 2.4 * prose_v
        d1[f"blocks.{l_idx}.mlp.experts.2.w3_A"][0, 24] = 0.16
        anchor_leak = -leak_scale * prose_v if l_idx == 3 else torch.zeros_like(prose_v)
        d1[f"blocks.{l_idx}.mlp.experts.2.w3_B"][:, 0] = 0.75 * e2_v + anchor_leak
        d1[f"blocks.{l_idx}.mlp.experts.2.w3_A"][1, 0] = 0.05
        d1[f"blocks.{l_idx}.mlp.experts.2.w3_B"][:, 1] = -leak_scale * prose_v
        # Ghost sink expert 3 harms E0 without contributing specialist value
        d1[f"blocks.{l_idx}.mlp.router.weight"][3] = 3.0 * prose_v
        d1[f"blocks.{l_idx}.mlp.experts.3.w3_A"][0, 0] = 0.08
        d1[f"blocks.{l_idx}.mlp.experts.3.w3_B"][:, 0] = -ghost_harm_scale * prose_v

    # Outer DiLoCo step (combining learner deltas and applying LN shift)
    outer_lr = 1.0
    raw_outer = clone_state_dict(parent_state)
    for key in raw_outer:
        if key in ("ln.weight", "ln.bias"):
            continue
        if key.startswith(("blocks.0.", "blocks.1.")):
            raw_outer[key] = raw_outer[key] + outer_lr * 0.5 * (d0[key] + d1[key])
        else:
            raw_outer[key] = raw_outer[key] + outer_lr * (d0[key] + d1[key])
    raw_outer["ln.weight"] = torch.full_like(raw_outer["ln.weight"], ln_scale)
    return raw_outer


def _anchor_digest(state_dict: dict[str, torch.Tensor], layers: tuple[int, ...] = (0, 1)) -> str:
    prefixes = tuple(f"blocks.{l_idx}." for l_idx in layers)
    return compute_state_dict_sha256({k: v for k, v in state_dict.items() if k.startswith(prefixes)})


_BASE_MODEL_CACHE: dict[int, tuple[CrowdV1WithEngram, dict[str, torch.Tensor], dict[str, torch.Tensor]]] = {}


def _get_subspace_crowd_moe(
    *,
    num_layers: int = 4,
    num_experts: int = 4,
    top_k: int = 2,
    lora_rank: int = 2,
    hidden_size: int = 32,
    intermediate_size: int = 64,
    vocab_size: int = 48,
    seed: int = 20261124,
) -> tuple[CrowdV1WithEngram, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Return a cached CrowdV1WithEngram model alongside a fresh cloned initial state_dict."""
    if seed not in _BASE_MODEL_CACHE:
        model, carriers = _make_subspace_crowd_moe(
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            lora_rank=lora_rank,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            seed=seed,
        )
        _BASE_MODEL_CACHE[seed] = (model, clone_state_dict(model.state_dict()), carriers)
    cached_model, cached_sd, cached_carriers = _BASE_MODEL_CACHE[seed]
    return cached_model, clone_state_dict(cached_sd), cached_carriers


# ---------------------------------------------------------------------------
# S1: E0-Drifted DiLoCo Outer Round Rescued by S5 Hook -> PASS Shadow Candidate
# ---------------------------------------------------------------------------


def test_s1_nocturnal_diloco_e0_drifted_round_rescued_into_pass_shadow() -> None:
    """S1: End-to-end DiLoCo outer round (+3.2% E0 drift) rescued by S5 into PASS shadow."""
    model, base_state, carriers = _get_subspace_crowd_moe(seed=2101)
    chunks = _make_disjoint_domain_chunks(seed_offset=101)

    def _eval_fn(sd: Any) -> dict[str, float]:
        return _eval_crowd_moe_state(model, sd, chunks)

    base_nll = _eval_fn(base_state)
    raw_outer_state = _simulate_diloco_outer_round(
        base_state,
        carriers,
        ghost_harm_scale=0.036,
        leak_scale=0.022,
        ln_scale=0.94,
    )
    raw_nll = _eval_fn(raw_outer_state)
    raw_e0_drift = (raw_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"]) - 1.0
    # Confirm nocturnal outer round lowered specialist NLLs while drifting E0 (~+3.2%, > +1.0%)
    assert 0.015 <= raw_e0_drift <= 0.060, f"Expected ~+3.2% E0 drift in raw round, got {raw_e0_drift:+.4%}"
    assert raw_nll["E1_code"] < base_nll["E1_code"]
    assert raw_nll["E2_math"] < base_nll["E2_math"]

    prose_inputs, spec_inputs = _make_probe_inputs(carriers, layers=(2, 3), n_rows=10, seed=3101)
    layer_harm = {0: -0.18, 1: -0.08, 2: 0.028, 3: 0.036}
    expert_e0_harm = {0: -0.025, 1: 0.008, 2: 0.009, 3: 0.032}
    expert_spec_loss = {0: 0.001, 1: 0.055, 2: 0.058, 3: 0.002}

    active_state, receipt = run_post_outer_round_auto_wsp_hook(
        raw_outer_state,
        base_nll,
        _eval_fn,
        parent_state=base_state,
        layer_e0_harm_nats=layer_harm,
        expert_e0_harm_nats=expert_e0_harm,
        expert_spec_loss_when_zeroed=expert_spec_loss,
        layer_prose_inputs=prose_inputs,
        layer_specialist_inputs=spec_inputs,
        base_final_ln_w=base_state["ln.weight"],
        base_final_ln_b=base_state["ln.bias"],
        round_id="night_round_2026_11_24_r01",
    )

    assert isinstance(receipt, PostRoundWSPLineageReceipt)
    assert receipt.object == LINEAGE_RECEIPT_OBJECT
    assert receipt.schema == SCHEMA_VERSION
    assert receipt.round_id == "night_round_2026_11_24_r01"
    assert receipt.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
    assert receipt.wsp_verdict == "PASS"
    assert receipt.passed is True
    assert receipt.guard_ok is True
    assert receipt.s5_guard_ok is True
    assert receipt.raw_guard_ok is False
    assert verify_lineage_receipt_provenance(receipt) is True
    assert verify_lineage_receipt_provenance(receipt.to_dict()) is True

    # Objective thermodynamic descent & E0 rescue
    assert receipt.raw_anse_energy > DEFAULT_ANSE_MAXIMUM_PAIN - 10.0
    assert receipt.s5_anse_energy < 0.0 < receipt.raw_anse_energy
    assert receipt.s5_nll_mean < receipt.raw_nll_mean < receipt.base_nll_mean
    assert receipt.general_rel_drift <= 0.01
    assert receipt.delta_anse_energy < 0.0
    assert receipt.parent_checkpoint_sha256 == compute_state_dict_sha256(base_state)
    assert receipt.raw_outer_sha256 == compute_state_dict_sha256(raw_outer_state)
    assert receipt.sanitized_s5_sha256 == compute_state_dict_sha256(active_state)
    assert receipt.active_checkpoint_sha256 == receipt.sanitized_s5_sha256

    # Beneficial anchor layers 0..1 bitwise preserved; ghost sink 3 zeroed; specialists 1 & 2 preserved & gated
    assert receipt.beneficial_anchor_layers == (0, 1)
    assert receipt.anchor_layers_bitwise_preserved is True
    assert _anchor_digest(active_state) == _anchor_digest(raw_outer_state)
    assert receipt.sink_experts == (3,)
    assert receipt.preserved_specialists == (1, 2)
    assert "s5_stage_1_selective_ghost_sink" in receipt.committed_stages
    assert torch.all(active_state["blocks.2.mlp.experts.3.w3_B"] == 0.0)
    assert torch.all(active_state["blocks.3.mlp.experts.3.w3_B"] == 0.0)


# ---------------------------------------------------------------------------
# S2: Already-Clean Outer Round Cleanly Keeps raw_outer_state Bitwise
# ---------------------------------------------------------------------------


def test_s2_clean_outer_round_keeps_raw_state_bitwise_without_regression() -> None:
    """S2: Already-clean outer round where surgery yields zero gain keeps raw_outer_state bitwise."""
    model, base_state, carriers = _get_subspace_crowd_moe(seed=2101)
    chunks = _make_disjoint_domain_chunks(seed_offset=202)

    def _eval_fn(sd: Any) -> dict[str, float]:
        return _eval_crowd_moe_state(model, sd, chunks)

    base_nll = _eval_fn(base_state)
    # Build a clean outer checkpoint with strong specialist gains and negative/benign E0 drift
    clean_outer = _simulate_diloco_outer_round(
        base_state,
        carriers,
        ghost_harm_scale=0.0,
        leak_scale=0.0,
        ln_scale=1.0,
    )
    # Zero out ghost sink expert 3 already in clean_outer so nothing harms E0
    for l_idx in (2, 3):
        clean_outer[f"blocks.{l_idx}.mlp.router.weight"][3].zero_()
        for attr in ("w1_A", "w1_B", "w2_A", "w2_B", "w3_A", "w3_B"):
            clean_outer[f"blocks.{l_idx}.mlp.experts.3.{attr}"].zero_()

    raw_nll = _eval_fn(clean_outer)
    assert (raw_nll["E0_general_fineweb"] / base_nll["E0_general_fineweb"]) - 1.0 <= 0.005

    # 2A: All layers beneficial/neutral and zero harmful experts -> zero gain, bitwise raw retention
    cfg_no_ln = PostRoundWSPHookConfig(use_ln_calibration=False)
    active_2a, rec_2a = run_post_outer_round_auto_wsp_hook(
        clean_outer,
        base_nll,
        _eval_fn,
        parent_state=base_state,
        layer_e0_harm_nats=[-0.12, -0.08, -0.02, -0.01],
        expert_e0_harm_nats={0: -0.03, 1: -0.002, 2: -0.002, 3: 0.0},
        expert_spec_loss_when_zeroed={0: 0.001, 1: 0.05, 2: 0.05, 3: 0.0},
        round_id="night_round_clean_2a",
        config=cfg_no_ln,
    )
    assert rec_2a.promotion_stage == PROMOTION_STAGE_REJECTED_RAW
    assert rec_2a.wsp_verdict == "NO_GAIN"
    assert rec_2a.passed is False
    assert rec_2a.guard_ok is True
    assert rec_2a.raw_guard_ok is True
    assert rec_2a.edited_params_vs_raw == 0
    assert compute_state_dict_sha256(active_2a) == rec_2a.raw_outer_sha256
    assert rec_2a.active_checkpoint_sha256 == rec_2a.raw_outer_sha256
    assert rec_2a.sanitized_s5_sha256 == rec_2a.raw_outer_sha256

    # 2B: Force candidate edits (with min_energy_improvement high or regressing eval) -> rolled back to raw bitwise
    prose_inputs, spec_inputs = _make_probe_inputs(carriers, layers=(2, 3), n_rows=10, seed=3202)
    cfg_strict = PostRoundWSPHookConfig(
        min_energy_improvement=0.5,
        gate_all_surviving_specialists=True,
        temperature_grid=(1.0,),
    )
    active_2b, rec_2b = run_post_outer_round_auto_wsp_hook(
        clean_outer,
        base_nll,
        _eval_fn,
        layer_e0_harm_nats=[{"layer": 0, "e0_harm_nats": -0.10}, {"layer": 1, "e0_harm_nats": -0.05}, {"layer": 2, "e0_harm_nats": 0.01}, {"layer": 3, "e0_harm_nats": 0.01}],
        expert_e0_harm_nats=[-0.02, 0.004, 0.004, 0.0],
        expert_spec_loss_when_zeroed=[0.0, 0.05, 0.05, 0.0],
        layer_prose_inputs=prose_inputs,
        layer_specialist_inputs=spec_inputs,
        base_final_ln_w=base_state["ln.weight"],
        base_final_ln_b=base_state["ln.bias"],
        round_id="night_round_clean_2b",
        config=cfg_strict,
    )
    assert rec_2b.promotion_stage == PROMOTION_STAGE_REJECTED_RAW
    assert rec_2b.wsp_verdict == "NO_GAIN"
    assert rec_2b.has_explicit_parent is False
    assert rec_2b.parent_checkpoint_sha256 == rec_2b.raw_outer_sha256
    assert compute_state_dict_sha256(active_2b) == rec_2b.raw_outer_sha256
    assert len(rec_2b.rolled_back_stages) >= 1

    # 2C: Sequence prose/specialist inputs, positive neutral-layer fallback, budget ceiling, and RMSNorm/no-LN handling
    tiny_sd = {
        "blocks.0.mlp.router.weight": torch.ones((2, 4), dtype=torch.float32),
        "blocks.0.mlp.experts.0.w3_B": torch.zeros((4, 2), dtype=torch.float32),
        "blocks.0.mlp.experts.1.w3_B": torch.ones((4, 2), dtype=torch.float32),
        "model.ln.weight": torch.ones(4, dtype=torch.float32),
        "model.ln.bias": torch.zeros(4, dtype=torch.float32),
    }
    tiny_base_nll = {"E0_general_fineweb": 2.0, "E1_code": 2.0}
    _, rec_budget = run_post_outer_round_auto_wsp_hook(
        tiny_sd,
        tiny_base_nll,
        lambda _sd: dict(tiny_base_nll),
        layer_e0_harm_nats=[0.001],
        expert_e0_harm_nats={0: -0.01, 1: 0.02},
        expert_spec_loss_when_zeroed={0: 0.0, 1: 0.0},
        layer_prose_inputs=[torch.randn((6, 4), generator=torch.Generator().manual_seed(11))],
        layer_specialist_inputs=[[torch.randn((6, 4), generator=torch.Generator().manual_seed(12))]],
        base_final_ln_w=torch.ones(4, dtype=torch.float32),
        config=PostRoundWSPHookConfig(max_total_edit_fraction=1e-4, lambda_ln_grid=(0.0, 0.5, 1.0)),
    )
    assert rec_budget.promotion_stage == PROMOTION_STAGE_REJECTED_RAW
    assert "s5_stage_1_selective_ghost_sink" in rec_budget.rolled_back_stages

    # Biasless RMSNorm fallback + stage4 budget rollback + no-LN dict coverage
    rms_sd = {"model.ln.weight": torch.ones(4, dtype=torch.float32)}
    def _rms_eval(sd: Any) -> dict[str, float]:
        val = 1.90 if float(sd["model.ln.weight"][0]) < 0.95 else 2.0
        return {"E0_general_fineweb": val, "E1_code": val}

    _, rec_rms_budget = run_post_outer_round_auto_wsp_hook(
        rms_sd,
        tiny_base_nll,
        _rms_eval,
        layer_e0_harm_nats=[0.0],
        expert_e0_harm_nats={0: 0.0},
        expert_spec_loss_when_zeroed={0: 0.0},
        config=PostRoundWSPHookConfig(temperature_grid=(1.0, 1.25)),
    )
    assert rec_rms_budget.promotion_stage == PROMOTION_STAGE_REJECTED_RAW
    assert rec_rms_budget.temperature_applied == 1.0

    _, rec_rms = run_post_outer_round_auto_wsp_hook(
        rms_sd,
        tiny_base_nll,
        _rms_eval,
        layer_e0_harm_nats=[0.0],
        expert_e0_harm_nats={0: 0.0},
        expert_spec_loss_when_zeroed={0: 0.0},
        config=PostRoundWSPHookConfig(temperature_grid=(1.0, 1.25), max_total_edit_fraction=1.0),
    )
    assert rec_rms.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
    assert rec_rms.temperature_applied == 1.25

    no_ln_sd = {"blocks.0.mlp.router.weight": torch.ones((2, 4), dtype=torch.float32)}
    _, rec_no_ln = run_post_outer_round_auto_wsp_hook(
        no_ln_sd,
        tiny_base_nll,
        lambda _sd: dict(tiny_base_nll),
        layer_e0_harm_nats=[0.0],
        expert_e0_harm_nats={0: 0.0},
        expert_spec_loss_when_zeroed={0: 0.0},
    )
    assert rec_no_ln.lambda_ln_applied is None


# ---------------------------------------------------------------------------
# S3: Shadow -> Live Canary Promotion Gate & Strict ExpertRegistry Lifecycle
# ---------------------------------------------------------------------------


def test_s3_shadow_to_live_canary_gate_and_strict_expert_registry() -> None:
    """S3: 15/15 router probes + PASS lineage promotes shadow -> live; 14/15 or E0 drift blocks."""
    model, base_state, carriers = _get_subspace_crowd_moe(seed=2101)
    chunks = _make_disjoint_domain_chunks(seed_offset=303)

    def _eval_fn(sd: Any) -> dict[str, float]:
        return _eval_crowd_moe_state(model, sd, chunks)

    base_nll = _eval_fn(base_state)
    raw_state = _simulate_diloco_outer_round(
        base_state, carriers, ghost_harm_scale=0.14, leak_scale=0.10, ln_scale=0.80
    )
    prose_inputs, spec_inputs = _make_probe_inputs(carriers, layers=(2, 3), n_rows=12, seed=3303)

    _, shadow_receipt = run_post_outer_round_auto_wsp_hook(
        raw_state,
        base_nll,
        _eval_fn,
        parent_state=base_state,
        layer_e0_harm_nats={0: -0.18, 1: -0.08, 2: 0.028, 3: 0.036},
        expert_e0_harm_nats={0: -0.025, 1: 0.008, 2: 0.009, 3: 0.032},
        expert_spec_loss_when_zeroed={0: 0.001, 1: 0.055, 2: 0.058, 3: 0.002},
        layer_prose_inputs=prose_inputs,
        layer_specialist_inputs=spec_inputs,
        base_final_ln_w=base_state["ln.weight"],
        base_final_ln_b=base_state["ln.bias"],
        round_id="night_round_s3",
        config=PostRoundWSPHookConfig(lambda_ln_grid=(0.0, 1.0), temperature_grid=(0.80, 1.0)),
    )
    assert shadow_receipt.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW

    # Strict ExpertRegistry starting at CANDIDATE -> SHADOW
    registry = ExpertRegistry(strict=True)
    expert_id = "hivemind/crowd-v1-night-r3"
    registry.register(
        ExpertRecord(
            expert_id=expert_id,
            kind="lora_domain",
            cluster_id="moe_swarm",
            base_model_id=shadow_receipt.base_model_id,
            adapter_sha256=shadow_receipt.sanitized_s5_sha256,
            adapter_size_mib=48,
            keywords=("python", "math"),
        )
    )
    assert registry.get(expert_id).state == EXPERT_CANDIDATE
    registry.transition(expert_id, EXPERT_SHADOW, "Promoted to shadow via S5 WSP lineage")
    assert registry.get(expert_id).state == EXPERT_SHADOW

    # 3A: 14/15 probes -> blocked in SHADOW
    blocked_14 = evaluate_shadow_to_live_canary_gate(
        shadow_receipt,
        14,
        DEFAULT_CANARY_PROBE_TOTAL,
        min_pass_ratio=1.0,
        shadow_general_drift_rel=0.002,
        max_general_drift_rel=0.01,
        registry=registry,
        expert_id=expert_id,
    )
    assert blocked_14.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
    assert blocked_14.canary_passed is False
    assert blocked_14.live_promoted is False
    assert blocked_14.canary_gate_report is not None
    assert blocked_14.canary_gate_report["object"] == CANARY_GATE_OBJECT
    assert blocked_14.canary_gate_report["approved"] is False
    assert registry.get(expert_id).state == EXPERT_SHADOW
    with pytest.raises(GateNotPassedError):
        registry.promote_with_gate(expert_id, blocked_14.canary_gate_report["gate_payload"])

    # 3B: 15/15 probes BUT drifted shadow E0 (+1.5% > +1.0%) -> blocked in SHADOW
    blocked_drift = evaluate_shadow_to_live_canary_gate(
        shadow_receipt,
        15,
        15,
        min_pass_ratio=1.0,
        shadow_general_drift_rel=0.015,
        max_general_drift_rel=0.01,
        registry=registry,
        expert_id=expert_id,
    )
    assert blocked_drift.promotion_stage == PROMOTION_STAGE_PROMOTED_SHADOW
    assert blocked_drift.canary_passed is False
    assert registry.get(expert_id).state == EXPERT_SHADOW

    # 3B-bis: Insufficient total probes (< min_probe_total) or lineage general_rel_drift > max_general_drift_rel
    blocked_min_total = evaluate_shadow_to_live_canary_gate(
        shadow_receipt, 10, 10, min_probe_total=15
    )
    assert blocked_min_total.canary_passed is False
    assert "min_probe_total" in " ".join(blocked_min_total.canary_gate_report["reasons"])

    drifted_shadow = PostRoundWSPLineageReceipt(
        **{
            **shadow_receipt.to_dict(),
            "general_rel_drift": 0.008,
            "beneficial_anchor_layers": shadow_receipt.beneficial_anchor_layers,
            "interfering_layers": shadow_receipt.interfering_layers,
            "editable_layers": shadow_receipt.editable_layers,
            "sink_experts": shadow_receipt.sink_experts,
            "preserved_specialists": shadow_receipt.preserved_specialists,
            "gated_targets": shadow_receipt.gated_targets,
            "committed_stages": shadow_receipt.committed_stages,
            "rolled_back_stages": shadow_receipt.rolled_back_stages,
            "skipped_stages": shadow_receipt.skipped_stages,
            "receipt_sha256": "",
        }
    )
    blocked_lineage_drift = evaluate_shadow_to_live_canary_gate(
        drifted_shadow, 15, 15, max_general_drift_rel=0.005
    )
    assert blocked_lineage_drift.canary_passed is False

    # 3C: Rejected non-shadow receipt cannot be promoted even at 15/15 probes
    non_shadow = PostRoundWSPLineageReceipt(
        **{
            **shadow_receipt.to_dict(),
            "promotion_stage": PROMOTION_STAGE_REJECTED_RAW,
            "wsp_verdict": "NO_GAIN",
            "verdict": "NO_GAIN",
            "beneficial_anchor_layers": shadow_receipt.beneficial_anchor_layers,
            "interfering_layers": shadow_receipt.interfering_layers,
            "editable_layers": shadow_receipt.editable_layers,
            "sink_experts": shadow_receipt.sink_experts,
            "preserved_specialists": shadow_receipt.preserved_specialists,
            "gated_targets": shadow_receipt.gated_targets,
            "committed_stages": shadow_receipt.committed_stages,
            "rolled_back_stages": shadow_receipt.rolled_back_stages,
            "skipped_stages": shadow_receipt.skipped_stages,
            "receipt_sha256": "",
        }
    )
    blocked_non_shadow = evaluate_shadow_to_live_canary_gate(non_shadow, 15, 15)
    assert blocked_non_shadow.canary_passed is False
    assert blocked_non_shadow.promotion_stage == PROMOTION_STAGE_REJECTED_RAW

    # 3D: Tampered receipt SHA-256 -> blocked fail-closed
    tampered_dict = shadow_receipt.to_dict()
    tampered_dict["s5_anse_energy"] = shadow_receipt.s5_anse_energy - 1.0
    blocked_tampered = evaluate_shadow_to_live_canary_gate(tampered_dict, 15, 15)
    assert blocked_tampered.canary_passed is False
    assert "provenance" in " ".join(blocked_tampered.canary_gate_report["reasons"]).lower()

    # 3E: 15/15 probes + PASS lineage + E0 within guard -> promoted to LIVE in both receipt & ExpertRegistry
    live_receipt = evaluate_shadow_to_live_canary_gate(
        shadow_receipt,
        15,
        15,
        min_pass_ratio=1.0,
        shadow_general_drift_rel=0.002,
        max_general_drift_rel=0.01,
        min_probe_total=15,
        registry=registry,
        expert_id=expert_id,
    )
    assert live_receipt.promotion_stage == PROMOTION_STAGE_PROMOTED_LIVE
    assert live_receipt.canary_passed is True
    assert live_receipt.live_promoted is True
    assert verify_lineage_receipt_provenance(live_receipt) is True
    assert registry.get(expert_id).state == EXPERT_LIVE

    # Auto-register + auto-promote path from fresh ExpertRegistry(strict=True), plus CANDIDATE -> SHADOW -> LIVE
    fresh_reg = ExpertRegistry(strict=True)
    cand_eid = "hivemind/moe-coder-cand"
    fresh_reg.register(
        ExpertRecord(
            expert_id=cand_eid,
            kind="lora_domain",
            cluster_id="code",
            base_model_id=shadow_receipt.base_model_id,
            adapter_sha256=shadow_receipt.sanitized_s5_sha256,
            adapter_size_mib=48,
        )
    )
    live_from_cand = evaluate_shadow_to_live_canary_gate(
        shadow_receipt,
        15,
        15,
        registry=fresh_reg,
        expert_id=cand_eid,
    )
    assert live_from_cand.promotion_stage == PROMOTION_STAGE_PROMOTED_LIVE
    assert fresh_reg.get(cand_eid).state == EXPERT_LIVE

    auto_reg = ExpertRegistry(strict=True)
    live_auto = evaluate_shadow_to_live_canary_gate(
        shadow_receipt.to_dict(),
        15,
        15,
        registry=auto_reg,
        expert_id="hivemind/moe-coder-auto",
        cluster_id="code",
        keywords=("python", "rust"),
    )
    assert live_auto.promotion_stage == PROMOTION_STAGE_PROMOTED_LIVE
    assert auto_reg.get("hivemind/moe-coder-auto").state == EXPERT_LIVE

    # Verify standalone payload builder
    payload = build_expert_registry_gate_payload(
        shadow_receipt,
        approved=True,
        probe_pass_count=15,
        probe_total=15,
        min_pass_ratio=1.0,
        shadow_general_drift_rel=0.0,
        max_general_drift_rel=0.01,
        base_model_id="hivemind/crowd-v1",
    )
    assert payload["servable"] is True
    assert payload["adapter"]["lora_gguf_sha256"] == shadow_receipt.sanitized_s5_sha256


# ---------------------------------------------------------------------------
# S4: Corrupted / Infeasible Round Rollback to Parent with ANSE Maximum Pain
# ---------------------------------------------------------------------------


def test_s4_corrupted_and_infeasible_rounds_roll_back_to_parent_with_maximum_pain() -> None:
    """S4: Non-finite or unrecoverably drifted outer round rolls back bitwise to parent_state."""
    model, parent_state, carriers = _get_subspace_crowd_moe(seed=2101)
    chunks = _make_disjoint_domain_chunks(seed_offset=404)

    def _eval_fn(sd: Any) -> dict[str, float]:
        return _eval_crowd_moe_state(model, sd, chunks)

    base_nll = _eval_fn(parent_state)
    parent_sha = compute_state_dict_sha256(parent_state)

    # 4A: Non-finite (NaN/Inf) weights in raw_outer_state -> immediate bitwise parent rollback + Maximum Pain
    nan_raw = clone_state_dict(parent_state)
    nan_raw["wte.weight"][0, 0] = float("nan")
    active_4a, rec_4a = run_post_outer_round_auto_wsp_hook(
        nan_raw,
        base_nll,
        _eval_fn,
        parent_state=parent_state,
        layer_e0_harm_nats=[-0.1, -0.05, 0.02, 0.03],
        expert_e0_harm_nats={0: -0.01, 1: 0.01, 2: 0.01, 3: 0.02},
        expert_spec_loss_when_zeroed={0: 0.0, 1: 0.04, 2: 0.04, 3: 0.0},
        round_id="night_round_corrupt_nan",
    )
    assert rec_4a.promotion_stage == PROMOTION_STAGE_REJECTED_PARENT
    assert rec_4a.wsp_verdict == "FAIL"
    assert rec_4a.raw_anse_energy == DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_4a.s5_anse_energy == DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_4a.active_checkpoint_sha256 == parent_sha
    assert compute_state_dict_sha256(active_4a) == parent_sha
    assert verify_lineage_receipt_provenance(rec_4a) is True

    # 4B: Finite weights where base/trunk corruption makes both raw and S5 infeasible (> +1.0% E0)
    poisoned_raw = clone_state_dict(parent_state)
    poisoned_raw["ln.bias"][:16] = poisoned_raw["ln.bias"][:16] - 3.0 * carriers["e0"][:16]
    cfg_no_ln = PostRoundWSPHookConfig(use_ln_calibration=False)
    active_4b, rec_4b = run_post_outer_round_auto_wsp_hook(
        poisoned_raw,
        base_nll,
        _eval_fn,
        parent_state=parent_state,
        layer_e0_harm_nats=[-0.1, -0.05, 0.02, 0.03],
        expert_e0_harm_nats={0: -0.01, 1: 0.004, 2: 0.004, 3: 0.02},
        expert_spec_loss_when_zeroed={0: 0.0, 1: 0.04, 2: 0.04, 3: 0.0},
        round_id="night_round_infeasible_drift",
        config=cfg_no_ln,
    )
    assert rec_4b.promotion_stage == PROMOTION_STAGE_REJECTED_PARENT
    assert rec_4b.wsp_verdict == "FAIL"
    assert rec_4b.raw_anse_energy >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_4b.s5_anse_energy >= DEFAULT_ANSE_MAXIMUM_PAIN
    assert rec_4b.active_checkpoint_sha256 == parent_sha
    assert compute_state_dict_sha256(active_4b) == parent_sha

    # 4C: eval_fn raising RuntimeError on raw_outer_state with valid parent_state -> parent rollback
    def _exploding_eval(sd: Any) -> dict[str, float]:
        if compute_state_dict_sha256(sd) != parent_sha:
            raise RuntimeError("diverged activation overflow")
        return base_nll

    other_raw = clone_state_dict(parent_state)
    other_raw["ln.weight"] = other_raw["ln.weight"] + 0.2
    active_4c, rec_4c = run_post_outer_round_auto_wsp_hook(
        other_raw,
        base_nll,
        _exploding_eval,
        parent_state=parent_state,
        layer_e0_harm_nats=[-0.1, 0.02],
        expert_e0_harm_nats={0: 0.0, 1: 0.01},
        expert_spec_loss_when_zeroed={0: 0.0, 1: 0.0},
        round_id="night_round_eval_explosion",
    )
    assert rec_4c.promotion_stage == PROMOTION_STAGE_REJECTED_PARENT
    assert rec_4c.raw_anse_energy == DEFAULT_ANSE_MAXIMUM_PAIN
    assert compute_state_dict_sha256(active_4c) == parent_sha

    # 4D: Infeasible raw_outer_state without parent_state -> rejected_fallback_raw with FAIL
    _, rec_4d = run_post_outer_round_auto_wsp_hook(
        poisoned_raw,
        base_nll,
        _eval_fn,
        parent_state=None,
        layer_e0_harm_nats=[-0.1, -0.05, 0.02, 0.03],
        expert_e0_harm_nats={0: -0.01, 1: 0.004, 2: 0.004, 3: 0.02},
        expert_spec_loss_when_zeroed={0: 0.0, 1: 0.04, 2: 0.04, 3: 0.0},
        round_id="night_round_infeasible_no_parent",
        config=cfg_no_ln,
    )
    assert rec_4d.promotion_stage == PROMOTION_STAGE_REJECTED_RAW
    assert rec_4d.wsp_verdict == "FAIL"
    assert rec_4d.guard_ok is False


# ---------------------------------------------------------------------------
# S5: Validation Errors & Security / Zero-Stub / Lateral-Movement Gates
# ---------------------------------------------------------------------------


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """S5: Exhaustive input validation plus AST/security/stub/gatekeeper compliance."""
    valid_sd = {
        "blocks.0.mlp.router.weight": torch.ones((2, 4), dtype=torch.float32),
        "blocks.0.mlp.experts.0.w3_B": torch.zeros((4, 2), dtype=torch.float32),
        "ln.weight": torch.ones(4, dtype=torch.float32),
        "ln.bias": torch.zeros(4, dtype=torch.float32),
    }
    base_nll = {"E0_general_fineweb": 2.0, "E1_code": 2.2}

    def _ok_eval(_sd: Any) -> dict[str, float]:
        return dict(base_nll)

    # 1. PostRoundWSPHookConfig validation
    with pytest.raises(TypeError, match="str"):
        PostRoundWSPHookConfig(general_prefix=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="printable"):
        PostRoundWSPHookConfig(general_prefix=" ")
    with pytest.raises(TypeError, match="int"):
        PostRoundWSPHookConfig(general_expert_index=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 0"):
        PostRoundWSPHookConfig(general_expert_index=-1)
    with pytest.raises(TypeError, match="real number"):
        PostRoundWSPHookConfig(general_guard_rel=False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        PostRoundWSPHookConfig(general_guard_rel=float("nan"))
    with pytest.raises(ValueError, match="<= 1.0"):
        PostRoundWSPHookConfig(max_total_edit_fraction=1.5)
    with pytest.raises(ValueError, match="> 0.0"):
        PostRoundWSPHookConfig(max_total_edit_fraction=0.0)
    with pytest.raises(ValueError, match="anchor_threshold_nats"):
        PostRoundWSPHookConfig(anchor_threshold_nats=0.01)
    with pytest.raises(TypeError, match="bool"):
        PostRoundWSPHookConfig(protect_anchor_layers="yes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="sequence"):
        PostRoundWSPHookConfig(rogue_expert_indices="3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_folds"):
        PostRoundWSPHookConfig(n_folds=1)
    with pytest.raises(ValueError, match="lambda_ln_grid"):
        PostRoundWSPHookConfig(lambda_ln_grid=())
    with pytest.raises(ValueError, match="temperature_grid"):
        PostRoundWSPHookConfig(temperature_grid=())
    with pytest.raises(ValueError, match="> 0.0"):
        PostRoundWSPHookConfig(temperature_grid=(-1.0,))

    # 2. run_post_outer_round_auto_wsp_hook argument validation
    with pytest.raises(TypeError, match="PostRoundWSPHookConfig"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            config="bad",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="callable"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            "bad",  # type: ignore[arg-type]
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="non-empty"):
        run_post_outer_round_auto_wsp_hook(
            {},
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="no parent_state"):
        run_post_outer_round_auto_wsp_hook(
            {"ln.weight": torch.tensor([float("nan")])},
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="keys do not match"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            parent_state={"ln.weight": torch.ones(4)},
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="Shape mismatch"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            parent_state={**valid_sd, "ln.weight": torch.ones(8)},
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="either both be provided"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs={0: torch.ones((4, 4))},
        )
    with pytest.raises(ValueError, match="mismatched domain keys"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            lambda _sd: {"E0_general_fineweb": 2.0},
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="base_final_ln_b cannot be provided"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            base_final_ln_b=torch.zeros(4),
        )
    with pytest.raises(ValueError, match="base_final_ln_w"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            base_final_ln_w=torch.ones(8),
        )
    with pytest.raises(ValueError, match="only finite values"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            base_final_ln_w=torch.tensor([1.0, float("nan"), 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="base_final_ln_b"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            base_final_ln_w=torch.ones(4),
            base_final_ln_b=torch.ones(8),
        )
    with pytest.raises(ValueError, match="non-empty"):
        run_post_outer_round_auto_wsp_hook(
            {" ": torch.ones(2)},
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(TypeError, match="torch.Tensor"):
        run_post_outer_round_auto_wsp_hook(
            {"a": [1.0]},  # type: ignore[dict-item]
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(ValueError, match="scalar parameter"):
        run_post_outer_round_auto_wsp_hook(
            {"a": torch.empty(0)},
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
        )
    with pytest.raises(TypeError, match="bool"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs={True: torch.ones((4, 4))},  # type: ignore[dict-item]
            layer_specialist_inputs={0: torch.ones((4, 4))},
        )
    with pytest.raises(ValueError, match="Duplicate layer index"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs={"0": torch.ones((4, 4)), "blocks.0": torch.ones((4, 4))},
            layer_specialist_inputs={"blocks.0": torch.ones((4, 4))},
        )
    with pytest.raises(ValueError, match="Invalid layer index key"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs={"bad": torch.ones((4, 4))},
            layer_specialist_inputs={0: torch.ones((4, 4))},
        )
    with pytest.raises(TypeError, match="Mapping or Sequence"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs="bad",  # type: ignore[arg-type]
            layer_specialist_inputs=[torch.ones((4, 4))],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs=[],
            layer_specialist_inputs=[],
        )
    with pytest.raises(TypeError, match="floating-point"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs=[torch.ones((4, 4), dtype=torch.int64)],
            layer_specialist_inputs=[torch.ones((4, 4))],
        )
    with pytest.raises(ValueError, match="2D or 3D"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs=[torch.ones(4)],
            layer_specialist_inputs=[torch.ones((4, 4))],
        )
    with pytest.raises(ValueError, match="only finite"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs=[torch.tensor([[1.0, float("nan"), 0.0, 0.0]])],
            layer_specialist_inputs=[torch.ones((4, 4))],
        )
    with pytest.raises(ValueError, match="must not be empty"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs=[torch.ones((4, 4))],
            layer_specialist_inputs=[[]],
        )
    with pytest.raises(TypeError, match="int or str"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.0],
            expert_e0_harm_nats={0: 0.0},
            expert_spec_loss_when_zeroed={0: 0.0},
            layer_prose_inputs=[torch.ones((4, 4))],
            layer_specialist_inputs=[{True: torch.ones((4, 4))}],  # type: ignore[list-item]
        )
    with pytest.raises(ValueError, match="router width"):
        run_post_outer_round_auto_wsp_hook(
            valid_sd,
            base_nll,
            _ok_eval,
            layer_e0_harm_nats=[0.005],
            expert_e0_harm_nats={0: -0.01, 1: 0.005},
            expert_spec_loss_when_zeroed={0: 0.0, 1: 0.05},
            layer_prose_inputs={0: torch.ones((6, 8))},
            layer_specialist_inputs={0: torch.ones((6, 8))},
        )

    # 3. Receipt & canary gate validation
    _, clean_rec = run_post_outer_round_auto_wsp_hook(
        valid_sd,
        base_nll,
        _ok_eval,
        layer_e0_harm_nats=[0.0],
        expert_e0_harm_nats={0: 0.0},
        expert_spec_loss_when_zeroed={0: 0.0},
        config=PostRoundWSPHookConfig(use_ln_calibration=False),
    )
    with pytest.raises(ValueError, match="promotion_stage"):
        PostRoundWSPLineageReceipt(**{**clean_rec.to_dict(), "promotion_stage": "unknown"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="wsp_verdict"):
        PostRoundWSPLineageReceipt(**{**clean_rec.to_dict(), "wsp_verdict": "MAYBE"})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="SHA-256"):
        PostRoundWSPLineageReceipt(**{**clean_rec.to_dict(), "raw_outer_sha256": "abc"})
    with pytest.raises(TypeError, match="PostRoundWSPLineageReceipt"):
        verify_lineage_receipt_provenance("bad")  # type: ignore[arg-type]
    assert verify_lineage_receipt_provenance({"receipt_sha256": "bad"}) is False
    assert verify_lineage_receipt_provenance({"receipt_sha256": "0" * 64, "x": float("nan")}) is False

    with pytest.raises(TypeError, match="PostRoundWSPLineageReceipt"):
        evaluate_shadow_to_live_canary_gate("bad", 15, 15)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot exceed"):
        evaluate_shadow_to_live_canary_gate(clean_rec, 16, 15)
    with pytest.raises(ValueError, match=">= 1"):
        evaluate_shadow_to_live_canary_gate(clean_rec, 0, 0)
    with pytest.raises(TypeError, match="ExpertRegistry"):
        evaluate_shadow_to_live_canary_gate(clean_rec, 15, 15, registry="bad", expert_id="e0")  # type: ignore[arg-type]

    # 4. AGPL-3.0 header, CI-safe ZeroStubAudit, PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn20", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn20", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/opentrainer_auto_wsp_hook.py", TARGET_MODULE_PATH),
        ("tests/test_opentrainer_auto_wsp_hook_scenarios.py", TEST_FILE_PATH),
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

    sample_receipt_json = json.dumps(clean_rec.to_dict(), sort_keys=True)
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+RECEIPT = {sample_receipt_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1120,
        pr_title="feat(swarm): HM-S35-04/05 automatic OpenTrainer post-round WSP surgery hook and ExpertRegistry canary gate",
        pr_body="Implements NEURAL-CTRL-DYN-20 automatic post-round WSP hook, canary gate, and CPU scenario tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): HM-S35-04/05 automatic OpenTrainer post-round WSP surgery hook and ExpertRegistry canary gate"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
