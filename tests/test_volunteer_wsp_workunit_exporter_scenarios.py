# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Scenario tests for NEURAL-CTRL-DYN-21: volunteer 8 GB WSP workunit & 4-bit NF4 edge exporter."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from hivemind.security.pr_security_scanner import PRSecurityScanner
from hivemind.swarm.byzantine_wsp_delta_aggregator import compute_alphaedit_prose_nullspace
from hivemind.swarm.crowdgpt_live_runtime import CrowdV1RuntimeConfig, CrowdV1WithEngram
from hivemind.swarm.volunteer_wsp_workunit_exporter import (
    ANSE_MAXIMUM_PAIN,
    CANONICAL_NF4_LEVELS,
    EDGE_EXPORT_RECEIPT_OBJECT,
    MAX_GAME_PREEMPTION_LATENCY_MS,
    MAX_NF4_NLL_GAP_NATS,
    MAX_OUTBOUND_MB_PER_HOUR,
    MODEL_EDGE_EXPORT_RECEIPT_OBJECT,
    PREEMPTION_RECEIPT_OBJECT,
    SCHEMA_VERSION,
    STD_8GB_TIER,
    STD_8GB_WDDM_CEILING_MIB,
    WORKUNIT_RECEIPT_OBJECT,
    CorporateHostRefusedError,
    PayloadBudgetExceededError,
    PreemptionBudgetExceededError,
    VolunteerWSPDeltaSubmission,
    VolunteerWSPWorkunitSpec,
    VRAMCeilingExceededError,
    aggregate_volunteer_wsp_workunits_nullspace,
    compress_rank_bounded_wsp_delta,
    compute_tensor_map_sha256,
    compute_tensor_sha256,
    dequantize_nf4_compensated_edge_blocks,
    estimate_hourly_payload_mb,
    estimate_rank_bounded_wsp_payload_bytes,
    evaluate_volunteer_workunit_admission,
    export_nf4_compensated_edge_blocks,
    export_s5_model_to_nf4_edge,
    preempt_volunteer_workunit_for_game,
    verify_edge_export_receipt_provenance,
    verify_workunit_receipt_provenance,
)
from hivemind.swarm.wsp_alphaedit_control import (
    compute_embeddings_batch,
    forward_model_pre_ln,
    per_chunk_nll_from_pre_ln,
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
        self.penalty_energy = 1_000_000.0 if violations else 0.0


class _FallbackZeroStubAudit:
    """CI-safe fallback implementing both ``check_file`` and ``audit_python_code``."""

    def audit_python_code(self, code: str) -> _FallbackStubResult:
        tree = ast.parse(code)
        violations: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                stmts = [
                    stmt
                    for stmt in node.body
                    if not (
                        isinstance(stmt, ast.Expr)
                        and isinstance(stmt.value, ast.Constant)
                        and isinstance(stmt.value.value, str)
                    )
                ]
                if not stmts or (len(stmts) == 1 and isinstance(stmts[0], ast.Pass)):
                    violations.append(f"stub function: {node.name}")
                elif len(stmts) == 1 and isinstance(stmts[0], ast.Raise):
                    exc = stmts[0].exc
                    if isinstance(exc, ast.Call) and isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                        violations.append(f"NotImplementedError in {node.name}")
        return _FallbackStubResult(violations)

    def check_file(self, path: str | Path) -> list[str]:
        return list(self.audit_python_code(Path(path).read_text(encoding="utf-8")).violations)


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_MODULE_PATH = REPO_ROOT / "src" / "hivemind" / "swarm" / "volunteer_wsp_workunit_exporter.py"
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


def _naive_rtn_nf4(weight: torch.Tensor, block_size: int = 64) -> torch.Tensor:
    """Deterministic round-to-nearest canonical 16-level NF4 baseline for comparison."""
    w64 = weight.detach().to(device="cpu", dtype=torch.float64)
    rows, cols = int(w64.shape[0]), int(w64.shape[1])
    pad_cols = (block_size - (cols % block_size)) % block_size
    w_pad = F.pad(w64, (0, pad_cols)) if pad_cols > 0 else w64
    blk = w_pad.reshape(rows, -1, block_size)
    scales = blk.abs().amax(dim=-1, keepdim=True)
    safe_s = torch.where(scales > 0.0, scales, torch.ones_like(scales))
    codebook = torch.tensor(CANONICAL_NF4_LEVELS, dtype=torch.float64)
    idx = ((blk / safe_s).clamp(-1.0, 1.0).unsqueeze(-1) - codebook).abs().argmin(dim=-1)
    deq = torch.where(scales > 0.0, codebook[idx] * scales, torch.zeros_like(blk))
    return deq.reshape(rows, -1)[:, :cols].to(dtype=weight.dtype)


def _build_s5_crowd_edge_fixture(seed: int = 20262103) -> dict[str, Any]:
    """Build deterministic base vs S5-edited CrowdV1WithEngram model and held-out calibration/eval chunks."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden_size = 64
    intermediate_size = 128
    vocab_size = 32
    cfg = CrowdV1RuntimeConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=hidden_size // 4,
        intermediate_size=intermediate_size,
        max_position_embeddings=32,
        engram_buckets=0,
        tie_word_embeddings=True,
    )
    model = CrowdV1WithEngram(cfg)

    u_prose = torch.zeros(hidden_size, dtype=torch.float32)
    u_prose[0] = 1.0 / math.sqrt(2.0)
    u_prose[1] = -1.0 / math.sqrt(2.0)

    u_spec = torch.zeros(hidden_size, dtype=torch.float32)
    u_spec[32] = 1.0 / math.sqrt(2.0)
    u_spec[33] = -1.0 / math.sqrt(2.0)

    with torch.no_grad():
        for param in model.parameters():
            param.zero_()

        wte = torch.zeros((vocab_size, hidden_size), dtype=torch.float32)
        p_emb = 0.18 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        p_emb = p_emb - p_emb.mean(dim=1, keepdim=True) + 1.4 * u_prose[:16].unsqueeze(0)
        p_emb[4] = 2.0 * u_prose[:16]

        s_emb = 0.18 * torch.randn((16, 16), generator=gen, dtype=torch.float32)
        s_emb = s_emb - s_emb.mean(dim=1, keepdim=True) + 1.4 * u_spec[32:48].unsqueeze(0)
        s_emb[6] = 2.1 * u_spec[32:48]

        wte[:16, :16] = p_emb
        wte[16:32, 32:48] = s_emb
        model.wte.weight.copy_(wte)
        model.head.weight = model.wte.weight
        model.ln.weight.fill_(1.0)

        blk = model.blocks[0]
        blk.ln1.weight.fill_(1.0)
        blk.ln2.weight.fill_(1.0)
        blk.attn.q.weight[:16, :16].copy_(0.05 * torch.eye(16))
        blk.attn.k.weight[:8, :16].copy_(0.05 * torch.eye(8, 16))
        blk.attn.v.weight[:8, :16].copy_(0.05 * torch.eye(8, 16))
        blk.attn.o.weight[:16, :16].copy_(0.05 * torch.eye(16))

        # Base MLP weights: realistic Gaussian initialization + uncalibrated specialist degradation (cured by S5)
        blk.mlp.w1.weight.copy_(0.12 * torch.randn((intermediate_size, hidden_size), generator=gen))
        blk.mlp.w2.weight.copy_(0.12 * torch.randn((intermediate_size, hidden_size), generator=gen))
        blk.mlp.w3.weight.copy_(0.08 * torch.randn((hidden_size, intermediate_size), generator=gen))
        # Preserve general prose subspace gently on block 0
        blk.mlp.w1.weight[:16, :16].copy_(0.8 * torch.eye(16))
        blk.mlp.w2.weight[:16, :16].copy_(0.8 * torch.eye(16))
        blk.mlp.w3.weight[:16, :16].copy_(0.35 * torch.outer(u_prose[:16], u_prose[:16]))
        # Uncalibrated anti-specialist perturbation in base checkpoint before S5 surgery
        blk.mlp.w1.weight[32, 32:48].copy_(1.2 * u_spec[32:48])
        blk.mlp.w2.weight[32, 32:48].copy_(1.2 * u_spec[32:48])
        blk.mlp.w3.weight[32:48, 32].copy_(-1.6 * u_spec[32:48])

    base_sd = {k: v.detach().clone() for k, v in model.state_dict().items()}

    # Deterministic prose and specialist chunks
    prose_tokens = torch.randint(0, 16, (8, 12), generator=gen, dtype=torch.long)
    prose_tokens[:, 1::2] = 4
    spec_tokens = torch.randint(16, 32, (8, 12), generator=gen, dtype=torch.long)
    spec_tokens[:, 1::2] = 22

    def _eval_nll(weights_map: dict[str, torch.Tensor]) -> tuple[float, float]:
        saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(weights_map, strict=True)
        model.eval()
        with torch.no_grad():
            h_p = forward_model_pre_ln(model, prose_tokens)
            nll_p = float(
                per_chunk_nll_from_pre_ln(
                    h_p, model.ln.weight, model.ln.bias, float(model.ln.eps), model.head.weight, prose_tokens, score_from=4
                ).mean().item()
            )
            h_s = forward_model_pre_ln(model, spec_tokens)
            nll_s = float(
                per_chunk_nll_from_pre_ln(
                    h_s, model.ln.weight, model.ln.bias, float(model.ln.eps), model.head.weight, spec_tokens, score_from=4
                ).mean().item()
            )
        model.load_state_dict(saved, strict=True)
        return nll_p, nll_s

    base_prose_nll, base_spec_nll = _eval_nll(base_sd)

    # Apply S5 specialist continuous WSP edit orthogonal to prose u_prose (on specialist dims 32..63)
    s5_sd = {k: v.detach().clone() for k, v in base_sd.items()}
    with torch.no_grad():
        s5_sd["blocks.0.mlp.w1.weight"][32, 32:48].copy_(0.9 * u_spec[32:48])
        s5_sd["blocks.0.mlp.w2.weight"][32, 32:48].copy_(0.9 * u_spec[32:48])
        s5_sd["blocks.0.mlp.w3.weight"][32:48, 32].copy_(0.7 * u_spec[32:48])

    s5_prose_nll, s5_spec_nll = _eval_nll(s5_sd)

    # Calibration Hessian diagonals E[x^2] at layer 0 pre-MLP & post-SwiGLU
    model.load_state_dict(s5_sd, strict=True)
    with torch.no_grad():
        all_tokens = torch.cat([prose_tokens, spec_tokens], dim=0)
        xb = compute_embeddings_batch(model, all_tokens)
        ln_x = model.blocks[0].ln2(xb).reshape(-1, hidden_size)
        h_in = ln_x.to(torch.float64).pow(2).mean(dim=0) + 1e-4
        gate = F.silu(F.linear(ln_x, s5_sd["blocks.0.mlp.w1.weight"]))
        up = F.linear(ln_x, s5_sd["blocks.0.mlp.w2.weight"])
        swiglu_act = gate * up
        h_mid = swiglu_act.to(torch.float64).pow(2).mean(dim=0) + 1e-4

    hessian_diags = {
        "blocks.0.mlp.w1.weight": h_in,
        "blocks.0.mlp.w2.weight": h_in,
        "blocks.0.mlp.w3.weight": h_mid,
    }
    return {
        "model": model,
        "base_sd": base_sd,
        "s5_sd": s5_sd,
        "hessian_diags": hessian_diags,
        "base_prose_nll": base_prose_nll,
        "base_spec_nll": base_spec_nll,
        "s5_prose_nll": s5_prose_nll,
        "s5_spec_nll": s5_spec_nll,
        "eval_nll_fn": _eval_nll,
        "x_in": ln_x,
    }


def test_s1_seven_volunteers_two_byzantine_nullspace_aggregation() -> None:
    """S1: 7 home-PC volunteers with 2 Byzantine peers aggregate cleanly in P_0 with <1e-6 prose drift."""
    gen = torch.Generator(device="cpu").manual_seed(20262101)
    d_in = 48
    rank = 4

    # Construct general prose keys K_0 occupying a 6D subspace
    prose_basis = torch.linalg.qr(torch.randn((d_in, 6), generator=gen, dtype=torch.float64))[0]
    prose_keys = (torch.randn((24, 6), generator=gen, dtype=torch.float64) @ prose_basis.T).to(torch.float32)
    p_null, null_meta = compute_alphaedit_prose_nullspace(prose_keys, sv_threshold=1e-2)
    assert null_meta["active_modes"] == 6

    # True specialist direction (orthogonal to prose_keys) + benign local prose noise before projection
    spec_dir = torch.randn((rank, d_in), generator=gen, dtype=torch.float64)
    spec_dir = spec_dir / torch.linalg.matrix_norm(spec_dir, ord="fro")
    spec_dir_null = (spec_dir @ p_null).to(torch.float32)

    honest_specs: list[VolunteerWSPWorkunitSpec] = []
    submissions_case_a: list[VolunteerWSPDeltaSubmission] = []
    for idx in range(5):
        spec = VolunteerWSPWorkunitSpec(
            workunit_id=f"wu_s35_honest_{idx}",
            peer_id=f"home_rtx2070_{idx}",
            hardware_tier=STD_8GB_TIER,
            vram_budget_mib=6144.0 + idx * 64.0,
            preemption_latency_ms=350.0 + idx * 25.0,
            is_corporate_managed=False,
            max_rank=rank,
            num_layers=24,
            hidden_size=1536,
            compression_bits=4,
            workunits_per_hour=4.0,
        )
        admission = evaluate_volunteer_workunit_admission(spec)
        assert admission["admitted"] is True
        assert admission["paused"] is False
        assert admission["vram_budget_mib"] <= STD_8GB_WDDM_CEILING_MIB
        assert spec.estimated_mb_per_hour < MAX_OUTBOUND_MB_PER_HOUR
        honest_specs.append(spec)

        # Compress a rank-bounded delta via SVD + 8-bit group quantization
        raw_delta = spec_dir + 0.01 * torch.randn((rank, d_in), generator=gen, dtype=torch.float64)
        comp_delta, comp_meta = compress_rank_bounded_wsp_delta(raw_delta.to(torch.float32), max_rank=rank, compression_bits=8)
        assert comp_meta["effective_rank"] == rank
        submissions_case_a.append(
            VolunteerWSPDeltaSubmission(
                peer_id=spec.peer_id,
                workunit_id=spec.workunit_id,
                delta_tensors={"blocks.0.mlp.experts.1.w1_A": comp_delta},
                rank=rank,
                vram_budget_mib=spec.vram_budget_mib,
                preemption_latency_ms=spec.preemption_latency_ms,
                compression_bits=4,
                workunits_per_hour=4.0,
            )
        )

    # Test 16-bit compression path as well
    comp_fp16, fp16_meta = compress_rank_bounded_wsp_delta(spec_dir.to(torch.float32), max_rank=rank, compression_bits=16)
    assert fp16_meta["compression_bits"] == 16 and comp_fp16.shape == (rank, d_in)

    # Case A: 5 honest + 1 norm-matched opposite Trojan peer + 1 NaN corrupted peer (7 total, max_byzantine_peers=2)
    nan_tensor = torch.full((rank, d_in), float("nan"), dtype=torch.float32)
    poison_trojan = -3.0 * spec_dir.to(torch.float32) + 0.5 * torch.randn((rank, d_in), generator=gen, dtype=torch.float32)
    submissions_case_a.append(
        VolunteerWSPDeltaSubmission(
            peer_id="byzantine_trojan_0",
            workunit_id="wu_s35_byz_0",
            delta_tensors={"blocks.0.mlp.experts.1.w1_A": poison_trojan},
            rank=rank,
            vram_budget_mib=5120.0,
            preemption_latency_ms=400.0,
        )
    )
    submissions_case_a.append(
        VolunteerWSPDeltaSubmission(
            peer_id="byzantine_nan_1",
            workunit_id="wu_s35_byz_1",
            delta_tensors={"blocks.0.mlp.experts.1.w1_A": nan_tensor},
            rank=rank,
            vram_budget_mib=5120.0,
            preemption_latency_ms=400.0,
        )
    )

    agg_a, rec_a = aggregate_volunteer_wsp_workunits_nullspace(
        submissions_case_a,
        p_null,
        max_byzantine_peers=2,
        max_allowed_rank=rank,
        prose_keys=prose_keys,
    )
    assert rec_a["object"] == WORKUNIT_RECEIPT_OBJECT
    assert rec_a["schema"] == SCHEMA_VERSION
    assert rec_a["num_submissions"] == 7
    assert rec_a["num_accepted"] == 5
    assert rec_a["num_rejected"] == 2
    assert set(rec_a["rejected_peer_ids"]) == {"byzantine_nan_1", "byzantine_trojan_0"}
    assert "non_finite_values" in rec_a["rejection_reasons"]["byzantine_nan_1"]
    assert rec_a["max_prose_key_drift"] < 1e-6
    assert rec_a["max_peer_mb_per_hour"] < MAX_OUTBOUND_MB_PER_HOUR
    assert rec_a["vram_budget_mib"] <= STD_8GB_WDDM_CEILING_MIB
    assert rec_a["max_preemption_latency_ms"] < MAX_GAME_PREEMPTION_LATENCY_MS
    assert rec_a["anse_energy"] < 0.0

    # Verify prose keys suffer <1e-6 drift and consensus direction matches honest specialist subspace
    w1_a_agg = agg_a["blocks.0.mlp.experts.1.w1_A"]
    prose_drift_direct = float((prose_keys.to(torch.float64) @ w1_a_agg.to(torch.float64).T).abs().max().item())
    assert prose_drift_direct < 1e-6
    cos_sim = float(
        F.cosine_similarity(
            w1_a_agg.reshape(1, -1).to(torch.float64),
            spec_dir_null.reshape(1, -1).to(torch.float64),
        ).item()
    )
    assert cos_sim > 0.99

    # Case B: 5 honest + 1 norm-spiked peer (50x norm) + 1 direction-flipped peer
    submissions_case_b = list(submissions_case_a[:5])
    submissions_case_b.append(
        VolunteerWSPDeltaSubmission(
            peer_id="byzantine_spike_0",
            workunit_id="wu_s35_spike_0",
            delta_tensors={"blocks.0.mlp.experts.1.w1_A": 50.0 * spec_dir.to(torch.float32)},
            rank=rank,
        )
    )
    submissions_case_b.append(
        VolunteerWSPDeltaSubmission(
            peer_id="byzantine_dir_1",
            workunit_id="wu_s35_dir_1",
            delta_tensors={"blocks.0.mlp.experts.1.w1_A": -1.5 * spec_dir.to(torch.float32)},
            rank=rank,
        )
    )
    _, rec_b = aggregate_volunteer_wsp_workunits_nullspace(
        submissions_case_b, p_null, max_byzantine_peers=2, prose_keys=prose_keys
    )
    assert set(rec_b["rejected_peer_ids"]) == {"byzantine_dir_1", "byzantine_spike_0"}
    assert "norm_spike_exceeded" in rec_b["rejection_reasons"]["byzantine_spike_0"]


def test_s2_strict_refusal_corporate_vram_payload_and_game_preemption() -> None:
    """S2: Enforce corporate/managed refusal, >6656 MiB VRAM block, <50 MB/h cap, and <1.5s game pause."""
    # 1. Immediate refusal of corporate/managed hosts and corporate host markers
    with pytest.raises(CorporateHostRefusedError, match="Corporate or managed"):
        VolunteerWSPWorkunitSpec(
            workunit_id="wu_corp",
            peer_id="pc_work",
            is_corporate_managed=True,
        )
    with pytest.raises(CorporateHostRefusedError, match="Corporate/managed host markers"):
        VolunteerWSPWorkunitSpec(
            workunit_id="wu_amadeus",
            peer_id="pc_work2",
            host_markers=("amadeus-corp-laptop",),
        )
    with pytest.raises(CorporateHostRefusedError):
        VolunteerWSPDeltaSubmission(
            peer_id="sub_corp",
            workunit_id="wu_corp_sub",
            delta_tensors=torch.ones((2, 8)),
            is_corporate_managed=True,
        )

    # 2. Strict >6656 MiB WDDM VRAM ceiling & tier enforcement
    with pytest.raises(VRAMCeilingExceededError, match="6656 MiB"):
        VolunteerWSPWorkunitSpec(
            workunit_id="wu_vram_over",
            peer_id="home_pc",
            vram_budget_mib=6657.0,
        )
    with pytest.raises(VRAMCeilingExceededError, match="6656 MiB"):
        VolunteerWSPDeltaSubmission(
            peer_id="sub_vram",
            workunit_id="wu_vram_sub",
            delta_tensors=torch.ones((2, 8)),
            vram_budget_mib=7168.0,
        )
    with pytest.raises(ValueError, match="std_8gb"):
        VolunteerWSPWorkunitSpec(
            workunit_id="wu_tier",
            peer_id="home_pc",
            hardware_tier="big_12gb",
        )

    # 3. Strict <50 MB/h outbound payload budget enforcement
    with pytest.raises(PayloadBudgetExceededError, match="50.00 MB/h"):
        VolunteerWSPWorkunitSpec(
            workunit_id="wu_heavy",
            peer_id="home_pc",
            max_rank=64,
            num_layers=48,
            hidden_size=4096,
            compression_bits=16,
            workunits_per_hour=10.0,
        )
    with pytest.raises(PayloadBudgetExceededError, match="50.0 MB/h"):
        VolunteerWSPDeltaSubmission(
            peer_id="sub_heavy",
            workunit_id="wu_heavy_sub",
            delta_tensors=torch.ones((2, 8)),
            reported_payload_bytes=15_000_000,
            workunits_per_hour=4.0,
        )

    # 4. Strict <1.5s game preemption budget & active game pause
    with pytest.raises(PreemptionBudgetExceededError, match="1500.0 ms"):
        VolunteerWSPWorkunitSpec(
            workunit_id="wu_slow_preempt",
            peer_id="home_pc",
            preemption_latency_ms=1500.0,
        )

    # Spec with game_active=True refuses GPU dispatch via admission pause
    active_game_spec = VolunteerWSPWorkunitSpec(
        workunit_id="wu_game_active",
        peer_id="gamer_rtx2070",
        vram_budget_mib=6400.0,
        preemption_latency_ms=320.0,
        game_active=True,
        sparse_keep_ratio=0.5,
    )
    paused_admission = evaluate_volunteer_workunit_admission(active_game_spec)
    assert paused_admission["admitted"] is False
    assert paused_admission["paused"] is True
    assert paused_admission["game_active"] is True
    assert paused_admission["vram_budget_mib"] == 0.0
    assert paused_admission["reason"] == "game_active_preemption_pause"
    assert paused_admission["preemption_latency_ms"] < MAX_GAME_PREEMPTION_LATENCY_MS

    # Live foreground game preemption flushes rank-r deltas to Host RAM < 1.5s with 0 MiB residual VRAM
    idle_spec = VolunteerWSPWorkunitSpec(
        workunit_id="wu_gamer_idle",
        peer_id="gamer_rtx2070",
        vram_budget_mib=6656.0,
        preemption_latency_ms=350.0,
    )
    inflight = {"blocks.0.mlp.experts.1.w1_A": torch.randn((4, 32))}
    preempt_rec = preempt_volunteer_workunit_for_game(
        idle_spec,
        trigger_process="cyberpunk2077.exe",
        inflight_deltas=inflight,
    )
    assert preempt_rec["object"] == PREEMPTION_RECEIPT_OBJECT
    assert preempt_rec["paused"] is True
    assert preempt_rec["admitted"] is False
    assert preempt_rec["vram_before_mib"] == 6656.0
    assert preempt_rec["vram_budget_mib"] == 0.0
    assert preempt_rec["eviction_latency_ms"] < MAX_GAME_PREEMPTION_LATENCY_MS
    assert preempt_rec["preemption_satisfied"] is True
    assert preempt_rec["checkpoint_tensor_count"] == 1

    empty_preempt = preempt_volunteer_workunit_for_game(idle_spec, trigger_process="steam.exe")
    assert empty_preempt["checkpoint_tensor_count"] == 0

    with pytest.raises(PreemptionBudgetExceededError, match="1500.0 ms"):
        preempt_volunteer_workunit_for_game(idle_spec, kernel_drain_ms=1450.0)


def test_s3_nf4_hessian_compensated_edge_export_and_nll_gap() -> None:
    """S3: 4-bit NF4 Hessian-compensated export beats RTN NF4 and keeps <0.02 nat NLL gap vs FP32."""
    # 1. Verify heterogeneous Hessian + outlier weight block & non-divisible cols=70 (block_size=64)
    gen = torch.Generator(device="cpu").manual_seed(20262103)
    w_rect = torch.randn((24, 70), generator=gen, dtype=torch.float32) * 0.18
    # Inject a low-Hessian outlier in block 0 so naive RTN NF4 stretches scale and loses high-Hessian precision
    w_rect[:, 63] = 4.2
    h_rect = torch.ones(70, dtype=torch.float32) * 0.05
    h_rect[:32] = 24.0

    w_rect_nf4, rect_rec = export_nf4_compensated_edge_blocks(w_rect, h_rect, block_size=64)
    w_rect_rtn = _naive_rtn_nf4(w_rect, block_size=64)
    assert rect_rec["object"] == EDGE_EXPORT_RECEIPT_OBJECT
    assert rect_rec["quant_format"] == "nf4"
    assert rect_rec["shape"] == [24, 70]
    assert rect_rec["padded_cols"] == 128
    assert len(rect_rec["nf4_levels"]) == 16
    assert rect_rec["hessian_weighted_error"] < rect_rec["naive_hessian_weighted_error"]
    assert rect_rec["hessian_error_reduction_ratio"] > 0.25
    assert rect_rec["edge_payload_bytes"] < w_rect.numel() * w_rect.element_size()

    # Check exact naive RTN computation matches receipt
    h64 = h_rect.to(torch.float64) + 1e-6 * float(h_rect.mean().item())
    rtn_h_err_direct = float((h64 * (w_rect.to(torch.float64) - w_rect_rtn.to(torch.float64)).pow(2)).sum().item())
    assert math.isclose(rect_rec["naive_hessian_weighted_error"], rtn_h_err_direct, rel_tol=1e-5)

    # Bitwise round-trip dequantization from packed nibbles + block scales
    w_unpacked = dequantize_nf4_compensated_edge_blocks(
        rect_rec["packed_nibbles"],
        rect_rec["block_scales"],
        shape=(24, 70),
        block_size=64,
        dtype=torch.float32,
    )
    assert torch.equal(w_unpacked, w_rect_nf4)

    # Also test full Cholesky GPTQ intra-block compensation path with positive-definite H matrix
    x_cal = torch.randn((96, 64), generator=gen, dtype=torch.float64) * 0.4
    x_cal[:, :16] *= 2.5
    h_full = (x_cal.T @ x_cal) / 96.0 + 1e-3 * torch.eye(64, dtype=torch.float64)
    w_sq = torch.randn((16, 64), generator=gen, dtype=torch.float32) * 0.2
    w_chol_nf4, chol_rec = export_nf4_compensated_edge_blocks(
        w_sq, torch.diagonal(h_full).to(torch.float32), block_size=32, hessian_matrix=h_full
    )
    w_sq_rtn = _naive_rtn_nf4(w_sq, block_size=32)
    out_err_chol = float(((x_cal @ w_sq.to(torch.float64).T) - (x_cal @ w_chol_nf4.to(torch.float64).T)).pow(2).mean().item())
    out_err_rtn = float(((x_cal @ w_sq.to(torch.float64).T) - (x_cal @ w_sq_rtn.to(torch.float64).T)).pow(2).mean().item())
    assert out_err_chol < out_err_rtn
    assert chol_rec["bits_per_weight"] < 5.5
    assert verify_edge_export_receipt_provenance(chol_rec) is True

    # 2. Full S5 CrowdV1WithEngram model export: <0.02 nat NLL gap vs FP32 and specialist gain preserved
    fx = _build_s5_crowd_edge_fixture()
    eval_nll_fn = fx["eval_nll_fn"]
    # Confirm S5 achieved substantial held-out specialist gain over base
    assert fx["s5_spec_nll"] < fx["base_spec_nll"] - 0.15

    exported_sd, model_rec = export_s5_model_to_nf4_edge(
        fx["s5_sd"],
        fx["hessian_diags"],
        block_size=64,
        eval_fn=lambda sd: eval_nll_fn(dict(sd))[1],
        vram_budget_mib=6144.0,
        max_nll_gap_nats=MAX_NF4_NLL_GAP_NATS,
    )
    nf4_prose_nll, nf4_spec_nll = eval_nll_fn(exported_sd)

    assert model_rec["object"] == MODEL_EDGE_EXPORT_RECEIPT_OBJECT
    assert model_rec["nll_gap_nats"] < MAX_NF4_NLL_GAP_NATS
    assert abs(nf4_spec_nll - fx["s5_spec_nll"]) < MAX_NF4_NLL_GAP_NATS
    assert abs(nf4_prose_nll - fx["s5_prose_nll"]) < MAX_NF4_NLL_GAP_NATS
    assert nf4_spec_nll < fx["base_spec_nll"] - 0.15
    assert model_rec["hessian_weighted_error"] < model_rec["naive_hessian_weighted_error"]
    assert model_rec["edge_vram_mib"] <= model_rec["vram_budget_mib"] <= STD_8GB_WDDM_CEILING_MIB


def test_s4_deterministic_sha256_workunit_and_edge_export_receipts() -> None:
    """S4: Verify deterministic SHA-256 receipts across independent runs and fail on tampering."""
    gen1 = torch.Generator(device="cpu").manual_seed(20262104)
    gen2 = torch.Generator(device="cpu").manual_seed(20262104)

    def _run_pipeline(gen: torch.Generator) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        p_null = torch.eye(16, dtype=torch.float64)
        subs = [
            VolunteerWSPDeltaSubmission(
                peer_id=f"peer_{i}",
                workunit_id=f"wu_{i}",
                delta_tensors=torch.randn((2, 16), generator=gen, dtype=torch.float32) * 0.05 + 0.3,
                rank=2,
            )
            for i in range(5)
        ]
        _, wu_rec = aggregate_volunteer_wsp_workunits_nullspace(subs, p_null, max_byzantine_peers=1)
        w = torch.randn((8, 64), generator=gen, dtype=torch.float32)
        h = torch.rand(64, generator=gen, dtype=torch.float32) + 0.1
        _, edge_rec = export_nf4_compensated_edge_blocks(w, h, block_size=64)
        spec = VolunteerWSPWorkunitSpec(workunit_id="wu_det", peer_id="peer_det")
        adm_rec = evaluate_volunteer_workunit_admission(spec)
        return wu_rec, edge_rec, adm_rec

    wu1, edge1, adm1 = _run_pipeline(gen1)
    wu2, edge2, adm2 = _run_pipeline(gen2)

    assert wu1["delta_sha256"] == wu2["delta_sha256"]
    assert wu1["receipt_sha256"] == wu2["receipt_sha256"]
    assert edge1["packed_nibbles_sha256"] == edge2["packed_nibbles_sha256"]
    assert edge1["receipt_sha256"] == edge2["receipt_sha256"]
    assert adm1["receipt_sha256"] == adm2["receipt_sha256"]

    assert verify_workunit_receipt_provenance(wu1) is True
    assert verify_workunit_receipt_provenance(adm1) is True
    assert verify_edge_export_receipt_provenance(edge1) is True

    # Tampering scalar fields or tensor blobs must immediately invalidate provenance
    bad_wu = dict(wu1)
    bad_wu["vram_budget_mib"] = 4096.0
    assert verify_workunit_receipt_provenance(bad_wu) is False
    assert verify_workunit_receipt_provenance({"receipt_sha256": "invalid"}) is False
    assert verify_workunit_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")}) is False

    bad_edge_scalar = dict(edge1)
    bad_edge_scalar["hessian_weighted_error"] = 0.0
    assert verify_edge_export_receipt_provenance(bad_edge_scalar) is False

    bad_edge_blob = dict(edge1)
    bad_edge_blob["packed_nibbles"] = edge1["packed_nibbles"].clone() ^ 0x01
    assert verify_edge_export_receipt_provenance(bad_edge_blob) is False

    bad_edge_type = dict(edge1)
    bad_edge_type["packed_nibbles"] = "not_a_tensor"
    assert verify_edge_export_receipt_provenance(bad_edge_type) is False
    assert verify_edge_export_receipt_provenance({"receipt_sha256": "short"}) is False
    assert verify_edge_export_receipt_provenance({"receipt_sha256": "0" * 64, "bad": float("nan")}) is False

    bad_edge_nonfinite_blob = dict(edge1)
    bad_edge_nonfinite_blob["block_scales"] = torch.full_like(edge1["block_scales"], float("nan"))
    assert verify_edge_export_receipt_provenance(bad_edge_nonfinite_blob) is False


def test_s5_validation_errors_and_security_stub_compliance_gates() -> None:
    """S5: Cover validation/error guards, quorum edge cases, and all security/stub/lateral gates."""
    # 1. Scalar & spec validation guards
    with pytest.raises(TypeError, match="bool"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", is_corporate_managed="no")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", max_rank=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=">= 1"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", max_rank=0)
    with pytest.raises(ValueError, match="non-empty string"):
        VolunteerWSPWorkunitSpec(workunit_id="", peer_id="p")
    with pytest.raises(TypeError, match="sequence"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", host_markers="amadeus")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="real number"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", vram_budget_mib=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", vram_budget_mib=float("nan"))
    with pytest.raises(ValueError, match="> 0.0"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", vram_budget_mib=0.0)
    with pytest.raises(ValueError, match=">= 0.0"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", preemption_latency_ms=-1.0)
    with pytest.raises(ValueError, match="<= 1.0"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", sparse_keep_ratio=1.5)
    with pytest.raises(ValueError, match="> 0.0"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", sparse_keep_ratio=0.0)
    with pytest.raises(ValueError, match="compression_bits"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", compression_bits=3)
    with pytest.raises(PayloadBudgetExceededError, match="50.0 MB/h"):
        VolunteerWSPWorkunitSpec(workunit_id="wu", peer_id="p", max_mb_per_hour=60.0)
    with pytest.raises(ValueError, match=">= 0"):
        estimate_rank_bounded_wsp_payload_bytes(num_layers=1, hidden_size=8, rank=2, header_bytes=-1)
    assert estimate_hourly_payload_mb(1_000_000, 2.0) == 2.0
    spec_dict = VolunteerWSPWorkunitSpec(workunit_id="wu_ok", peer_id="p_ok").to_dict()
    assert spec_dict["vram_budget_mib"] == 6144.0

    # 2. Admission, preemption, hashing & submission errors
    with pytest.raises(TypeError, match="VolunteerWSPWorkunitSpec"):
        evaluate_volunteer_workunit_admission("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="VolunteerWSPWorkunitSpec"):
        preempt_volunteer_workunit_for_game("bad")  # type: ignore[arg-type]
    ok_spec = VolunteerWSPWorkunitSpec(workunit_id="wu_ok", peer_id="p_ok")
    with pytest.raises(ValueError, match="inflight_deltas"):
        preempt_volunteer_workunit_for_game(ok_spec, inflight_deltas={})
    with pytest.raises(ValueError, match="inflight_deltas"):
        preempt_volunteer_workunit_for_game(ok_spec, inflight_deltas={"k": torch.empty(0)})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_tensor_sha256("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-finite"):
        compute_tensor_sha256(torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="non-empty mapping"):
        compute_tensor_map_sha256({})
    with pytest.raises(TypeError, match="torch.Tensor"):
        compute_tensor_map_sha256({"k": 1.0})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="non-finite"):
        compute_tensor_map_sha256({"k": torch.tensor([float("inf")])})
    with pytest.raises(TypeError, match="mapping"):
        verify_workunit_receipt_provenance("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping"):
        verify_edge_export_receipt_provenance("bad")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="delta_tensors"):
        VolunteerWSPDeltaSubmission(peer_id="p", workunit_id="w", delta_tensors={})
    with pytest.raises(ValueError, match="non-empty"):
        VolunteerWSPDeltaSubmission(peer_id="p", workunit_id="w", delta_tensors=torch.empty(0))
    with pytest.raises(TypeError, match="floating-point"):
        VolunteerWSPDeltaSubmission(peer_id="p", workunit_id="w", delta_tensors=torch.ones(4, dtype=torch.int64))
    with pytest.raises(ValueError, match="2D"):
        compress_rank_bounded_wsp_delta(torch.ones(4), max_rank=2)
    with pytest.raises(TypeError, match="floating-point"):
        compress_rank_bounded_wsp_delta(torch.ones((2, 4), dtype=torch.int32), max_rank=2)
    with pytest.raises(ValueError, match="finite"):
        compress_rank_bounded_wsp_delta(torch.tensor([[float("nan"), 1.0]]), max_rank=1)

    # 3. Aggregator validation & mapping-based peer rejections
    p_null = torch.eye(8, dtype=torch.float64)
    with pytest.raises(TypeError, match="submissions"):
        aggregate_volunteer_wsp_workunits_nullspace([], p_null)
    good_subs = [
        VolunteerWSPDeltaSubmission(peer_id=f"p{i}", workunit_id=f"w{i}", delta_tensors=torch.ones((2, 8)) * 0.2, rank=2)
        for i in range(5)
    ]
    with pytest.raises(ValueError, match="quorum"):
        aggregate_volunteer_wsp_workunits_nullspace(good_subs[:4], p_null, max_byzantine_peers=1)
    with pytest.raises(TypeError, match="Mapping"):
        aggregate_volunteer_wsp_workunits_nullspace(["bad", *good_subs], p_null)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="Duplicate"):
        aggregate_volunteer_wsp_workunits_nullspace([good_subs[0], *good_subs], p_null)

    # Mapping submissions testing corporate rejection, shape mismatch, rank exceedance, l2_norm_exceeded, and >max_byzantine_peers
    bad_dict_subs: list[Any] = list(good_subs[:4])
    bad_dict_subs.append({"peer_id": "corp_dict", "workunit_id": "wc", "delta_tensors": torch.ones((2, 8)), "is_corporate_managed": True})
    _, corp_dict_rec = aggregate_volunteer_wsp_workunits_nullspace(bad_dict_subs, p_null, max_byzantine_peers=1)
    assert corp_dict_rec["rejected_peer_ids"] == ["corp_dict"]
    assert "corporate_managed_host" in corp_dict_rec["rejection_reasons"]["corp_dict"]

    with pytest.raises(ValueError, match="exceeding max_byzantine_peers"):
        aggregate_volunteer_wsp_workunits_nullspace(
            [
                good_subs[0],
                good_subs[1],
                good_subs[2],
                {"peer_id": "c1", "delta_tensors": torch.ones((2, 8)), "is_corporate_managed": True},
                {"peer_id": "s1", "delta_tensors": torch.ones((3, 8))},
            ],
            p_null,
            max_byzantine_peers=1,
        )

    with pytest.raises(ValueError, match="delta_tensors"):
        aggregate_volunteer_wsp_workunits_nullspace([*good_subs, {"peer_id": "e1", "delta_tensors": {}}], p_null)
    with pytest.raises(ValueError, match="floating-point"):
        aggregate_volunteer_wsp_workunits_nullspace(
            [*good_subs, {"peer_id": "e2", "delta_tensors": {"k": torch.ones(2, dtype=torch.int64)}}], p_null
        )

    # Full-rank rejection & absolute L2 norm rejection
    gen = torch.Generator(device="cpu").manual_seed(99)
    rank3_mat = torch.linalg.qr(torch.randn((8, 8), generator=gen))[0][:3]
    agg_r, rec_r = aggregate_volunteer_wsp_workunits_nullspace(
        [
            *good_subs[:4],
            VolunteerWSPDeltaSubmission(peer_id="p4_ok", workunit_id="w4", delta_tensors=torch.ones((2, 8)) * 0.2, rank=2),
            {"peer_id": "p_over_norm", "delta_tensors": torch.ones((2, 8)) * 2.0, "workunits_per_hour": 4.0},
            {"peer_id": "p_over_budget", "delta_tensors": torch.ones((2, 8)) * 0.2, "payload_bytes": 20_000_000, "workunits_per_hour": 3.0},
        ],
        p_null,
        max_byzantine_peers=2,
        max_l2_norm=2.0,
        max_allowed_rank=2,
        prose_keys=torch.zeros((4, 8)),
    )
    assert "delta" in agg_r and rec_r["num_rejected"] == 2
    assert rank3_mat.shape == (3, 8)

    # High-rank rejection + prose drift ANSE_MAXIMUM_PAIN trigger
    rank3_subs = [
        VolunteerWSPDeltaSubmission(peer_id=f"r{i}", workunit_id=f"wr{i}", delta_tensors=torch.ones((3, 8)) * 0.1, rank=1)
        for i in range(4)
    ]
    rank3_subs.append(
        VolunteerWSPDeltaSubmission(peer_id="r_full", workunit_id="wr_f", delta_tensors=rank3_mat, rank=3)
    )
    _, rec_hr = aggregate_volunteer_wsp_workunits_nullspace(
        rank3_subs,
        torch.eye(8, dtype=torch.float64),
        max_byzantine_peers=1,
        max_allowed_rank=1,
        prose_keys=torch.ones((2, 8)) * 5.0,
    )
    assert "rank_bound_exceeded" in rec_hr["rejection_reasons"]["r_full"]
    assert rec_hr["anse_energy"] == ANSE_MAXIMUM_PAIN

    # 4. NF4 exporter & dequantizer validation errors
    with pytest.raises(ValueError, match="non-empty 2D"):
        export_nf4_compensated_edge_blocks(torch.ones(4), torch.ones(4))
    with pytest.raises(TypeError, match="floating-point"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4), dtype=torch.int32), torch.ones(4))
    with pytest.raises(ValueError, match="finite"):
        export_nf4_compensated_edge_blocks(torch.tensor([[float("nan"), 1.0]]), torch.ones(2))
    with pytest.raises(VRAMCeilingExceededError, match="6656 MiB"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4)), torch.ones(4), vram_budget_mib=7000.0)
    with pytest.raises(TypeError, match="floating-point"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4)), torch.ones(4, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4)), torch.tensor([1.0, float("nan"), 1.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4)), torch.ones(3))
    with pytest.raises(ValueError, match="non-negative"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4)), torch.tensor([1.0, -0.1, 1.0, 1.0]))
    with pytest.raises(ValueError, match="hessian_matrix"):
        export_nf4_compensated_edge_blocks(torch.ones((2, 4)), torch.ones(4), hessian_matrix=torch.ones((3, 3)))
    with pytest.raises(ValueError, match="finite"):
        export_nf4_compensated_edge_blocks(
            torch.ones((2, 4)), torch.ones(4), hessian_matrix=torch.full((4, 4), float("nan"))
        )

    # Zero-weight block + 2D hessian_diag + odd block_size edge case
    w_zero, zero_rec = export_nf4_compensated_edge_blocks(torch.zeros((2, 5)), torch.ones((2, 5)), block_size=3)
    assert torch.equal(w_zero, torch.zeros((2, 5))) and zero_rec["hessian_error_reduction_ratio"] == 0.0

    with pytest.raises(TypeError, match="uint8"):
        dequantize_nf4_compensated_edge_blocks(torch.zeros(4), torch.ones((2, 1)), shape=(2, 4))
    with pytest.raises(TypeError, match="floating-point"):
        dequantize_nf4_compensated_edge_blocks(torch.zeros(4, dtype=torch.uint8), torch.ones((2, 1), dtype=torch.int32), shape=(2, 4))
    with pytest.raises(ValueError, match="2-tuple"):
        dequantize_nf4_compensated_edge_blocks(torch.zeros(4, dtype=torch.uint8), torch.ones((2, 1)), shape=(2,))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="block_scales"):
        dequantize_nf4_compensated_edge_blocks(torch.zeros(4, dtype=torch.uint8), torch.ones((3, 1)), shape=(2, 4))
    with pytest.raises(ValueError, match="packed_nibbles"):
        dequantize_nf4_compensated_edge_blocks(torch.zeros(1, dtype=torch.uint8), torch.ones((2, 1)), shape=(2, 4))

    with pytest.raises(ValueError, match="non-empty mapping"):
        export_s5_model_to_nf4_edge({}, {})
    with pytest.raises(TypeError, match="hessian_diags"):
        export_s5_model_to_nf4_edge({"w": torch.ones((2, 4))}, [])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="torch.Tensor"):
        export_s5_model_to_nf4_edge({"w": "bad"}, {})  # type: ignore[dict-item]
    with pytest.raises(VRAMCeilingExceededError, match="6656 MiB"):
        export_s5_model_to_nf4_edge({"w": torch.ones((2, 4))}, {}, vram_budget_mib=7000.0)
    with pytest.raises(ValueError, match="NLL gap"):
        export_s5_model_to_nf4_edge(
            {"w": torch.ones((2, 8))},
            {"w": torch.ones(8)},
            eval_fn=lambda _sd: {"nll_mean": 1.5},
            fp32_baseline_nll=1.0,
            max_nll_gap_nats=0.02,
        )
    with pytest.raises(ValueError, match="must not be empty"):
        export_s5_model_to_nf4_edge({"w": torch.ones((2, 8))}, {"w": torch.ones(8)}, eval_fn=lambda _sd: {})
    with pytest.raises(VRAMCeilingExceededError, match="footprint"):
        export_s5_model_to_nf4_edge({"big": torch.ones((1024, 1024))}, {}, vram_budget_mib=1.0)
    _, map_eval_rec = export_s5_model_to_nf4_edge(
        {"w": torch.ones((2, 8))},
        {"w": torch.ones(8)},
        eval_fn=lambda _sd: {"E0_a": 1.0},
    )
    assert map_eval_rec["nll_gap_nats"] == 0.0

    # 5. AGPL-3.0 header, optional ZeroStubAudit (with CI fallback), PRSecurityScanner, lateral movement, FORBIDDEN_ADDED
    lateral_guard = _load_module_from_path(
        "test_no_lateral_movement_dyn21", REPO_ROOT / "tests" / "test_no_lateral_movement.py"
    )
    gatekeeper = _load_module_from_path("pr_gatekeeper_dyn21", REPO_ROOT / "scripts" / "pr_gatekeeper.py")
    scanner = PRSecurityScanner()
    fallback_auditor = _FallbackZeroStubAudit()
    changed_files: list[dict[str, Any]] = []

    for rel, path_obj in (
        ("src/hivemind/swarm/volunteer_wsp_workunit_exporter.py", TARGET_MODULE_PATH),
        ("tests/test_volunteer_wsp_workunit_exporter_scenarios.py", TEST_FILE_PATH),
    ):
        src_text = path_obj.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in src_text[:300], f"Missing AGPL-3.0 header in {rel}"

        assert fallback_auditor.check_file(str(path_obj)) == []
        assert fallback_auditor.audit_python_code(src_text).is_clean is True

        if ZeroStubAudit is not None:
            auditor = ZeroStubAudit()
            if hasattr(auditor, "audit_python_code"):
                stub_res = auditor.audit_python_code(src_text)
                assert stub_res.is_clean, f"ZeroStubAudit violations in {rel}: {stub_res.violations}"
                assert stub_res.penalty_energy == 0.0
            if hasattr(auditor, "check_file"):
                assert auditor.check_file(str(path_obj)) == []

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

    telemetry_json = json.dumps(
        {"nll_gap_nats": 0.008, "vram_budget_mib": 6144.0, "anse_energy": -0.142},
        sort_keys=True,
    )
    assert (
        gatekeeper.forbidden_scan(
            f"diff --git a/x.py b/x.py\n+++ b/x.py\n+PAYLOAD = {telemetry_json!r}"
        )
        == []
    )

    pr_audit = scanner.audit_pull_request(
        pr_number=1121,
        pr_title="feat(swarm): HM-S35-06/07 volunteer 8GB Gamer-Whisper WSP calibration workunit and 4-bit NF4 edge exporter",
        pr_body="Implements NEURAL-CTRL-DYN-21 volunteer 8 GB WSP calibration workunit policy and 4-bit NF4 edge exporter.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=[
            "feat(swarm): HM-S35-06/07 volunteer 8GB Gamer-Whisper WSP calibration workunit and 4-bit NF4 edge exporter"
        ],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.passed is True
    assert pr_audit.blocked_merge is False
    assert pr_audit.risk_score == 0
    assert pr_audit.findings == []
