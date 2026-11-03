"""Comprehensive unit and integration coverage tests for hivemind.core and hivemind.swarm."""

from __future__ import annotations

import asyncio
import json
import socket
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from hivemind.config import HardwareProfile
from hivemind.core.btm_moe import (
    FP8PagedKVCache,
    HierarchicalMoERouter,
    ProjectedLoRAStateDict,
    RegisteredDomainExpert,
    Season3DomainExpertRegistry,
    hotplug_hierarchical_experts,
    project_lora_rank_svd,
    project_lora_rank_svd_with_report,
)
from hivemind.core.btm_upcycle import (
    UpcycledFactoredSwiGLUExpert,
    UpcycledSparseMoEModel,
    _dequantize_linear_weight,
    _extract_ab_pair,
    _orient_lora_pair,
    detect_lora_rank,
    hotplug_rank_adaptive_experts,
    initialize_domain_branches,
    measure_moe_vram_footprint_gb,
    project_lora_state_dict_to_rank,
    svd_project_lora_pair,
    train_moe_router_gating,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.compression import (
    CompressedDeltaPackage,
    compress_lora_delta,
    compress_tensor_topk_fp8,
    compute_pseudo_gradient_delta,
    decompress_lora_delta,
    decompress_tensor_topk_fp8,
)
from hivemind.core.diloco_outer import (
    DiLoCoOuterOptimizer,
    filter_byzantine_deltas_multikrum,
    flatten_delta_dict,
)
from hivemind.core.quantization import (
    NF4QuantState,
    attach_qlora_adapters,
    build_local_qwen2_backbone,
    compute_state_dict_sha256,
    extract_lora_state_dict,
    load_hf_qlora_model,
    load_lora_state_dict,
    measure_vram_footprint,
    quantize_tensor_nf4_double_quant,
    resolve_compute_dtype,
    stream_load_hf_causal_lm,
)
from hivemind.swarm.dht_storage import (
    PARTIAL_DIR_NAME,
    ChunkIntegrityError,
    ContentAddressedSwarmStore,
)
from hivemind.swarm.hf_community_hub import (
    CommunityAdapterBundleManager,
    SwarmMiningGuildEngine,
)
from hivemind.swarm.inference_ring import RegionalDHTSwarmRing, SwarmPeerNode
from hivemind.swarm.ledger import HiveSwarmLedger
from hivemind.swarm.nat_traversal import (
    FaultProfile,
    NatMapping,
    StunError,
    TurnRelayServer,
    udp_socket,
)
from hivemind.swarm.quic_link import LinkEndpoint, LinkRun
from hivemind.swarm.quic_transport import (
    AdaptiveCongestionController,
    AsyncQUICGeoRingTransport,
    ContinentalSuperRingRouter,
    RegionalBridgePeer,
    SuperRingPeerNode,
    _run_coroutine,
    classify_continent,
)
from hivemind.swarm.strategy_optimizer import (
    AdaptiveDiLoCoCadenceController,
    DynamicFisherSVDSelector,
    ParetoGuildCampaignPlanner,
    UCB1CurriculumShardScheduler,
    reputation_weighted_ties_merge_lora,
)
from hivemind.swarm.swarm_fabric import (
    GeoRingPeer,
    HiveLayer2StateChannel,
    OptimisticProofOfInferenceAuditor,
    PinnedRAMExpertDMAStreamer,
    RegionalGeoRingTransport,
    TokenTreeSpeculativeVerifier,
    TwoLevelMacroMicroRouter,
    benchmark_fused_int4_fp8_moe_kernel,
    compress_hidden_states_fp8,
    decompress_hidden_states_fp8,
    fused_top2_moe_fp8_dispatch,
    pack_int4_nf4_weights,
    unpack_int4_nf4_weights,
)
from hivemind.swarm.wan_fault_benchmark import (
    ExternalEndpoints,
    FaultScenario,
    StrikeRegister,
    _bbr_pacing,
    _decode_latency,
    _environment,
    _pacing_adaptation,
    _run_quic_leg,
    _run_turn_leg,
    _run_websocket_leg,
    _throughput_mbps,
    default_scenarios,
    latency_summary,
    main as wan_bench_main,
    percentile,
    report_path,
    run_fault_scenario,
    run_wan_quic_turn_fault_benchmark,
)


def _write_safetensors_file(path: Path, tensors: Dict[str, torch.Tensor]) -> Path:
    """Serialize a dict of CPU tensors into the binary .safetensors format."""
    dtype_to_name = {
        torch.float32: "F32",
        torch.float16: "F16",
        torch.bfloat16: "BF16",
        torch.int64: "I64",
        torch.int32: "I32",
    }
    header: Dict[str, Any] = {"__metadata__": {"format": "pt"}}
    blobs = []
    offset = 0
    for name, tensor in tensors.items():
        t_cpu = tensor.detach().cpu().contiguous()
        raw = t_cpu.view(torch.uint8).numpy().tobytes()
        end = offset + len(raw)
        header[name] = {
            "dtype": dtype_to_name[t_cpu.dtype],
            "shape": list(t_cpu.shape),
            "data_offsets": [offset, end],
        }
        blobs.append(raw)
        offset = end

    header_bytes = json.dumps(header).encode("utf-8")
    with path.open("wb") as fp:
        fp.write(struct.pack("<Q", len(header_bytes)))
        fp.write(header_bytes)
        for blob in blobs:
            fp.write(blob)
    return path


# ============================================================================
# 1. Core Quantization Tests
# ============================================================================


def test_quantization_nf4_edge_cases_and_dtypes(monkeypatch: pytest.MonkeyPatch) -> None:
    # NF4QuantState without _computed_storage_bytes attribute (line 68)
    raw_state = NF4QuantState(
        packed_4bit=torch.zeros(8, dtype=torch.uint8),
        q_absmax_uint8=torch.zeros(2, dtype=torch.uint8),
        superblock_scale=0.5,
        original_shape=(4, 4),
    )
    assert raw_state.storage_bytes == 8 + 2 + 4

    # quantize_tensor_nf4_double_quant with non-multiple of block_size -> pad > 0 (line 104)
    w_unaligned = torch.randn(3, 5, dtype=torch.float32)
    q_state, recon = quantize_tensor_nf4_double_quant(w_unaligned, block_size=64)
    assert recon.shape == (3, 5)
    assert q_state.storage_bytes > 0

    # super_scale <= 0.0 branch (line 112): bypass min=1e-12 clamp on all-zero tensor
    orig_clamp_ = torch.Tensor.clamp_

    def _clamp_noop_min(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if kwargs.get("min") == 1e-12:
            return self
        return orig_clamp_(self, *args, **kwargs)

    with patch.object(torch.Tensor, "clamp_", _clamp_noop_min):
        zero_state, zero_recon = quantize_tensor_nf4_double_quant(torch.zeros(4, 4), block_size=16)
        assert zero_state.superblock_scale == 1.0
        assert torch.all(zero_recon == 0)

    # resolve_compute_dtype branches (lines 147-154)
    assert resolve_compute_dtype("bfloat16", torch.device("cpu")) == torch.float32
    with patch("torch.cuda.get_device_capability", return_value=(8, 6)):
        assert resolve_compute_dtype("bfloat16", torch.device("cuda:0")) == torch.bfloat16
    with patch("torch.cuda.get_device_capability", return_value=(7, 5)):
        assert resolve_compute_dtype("bf16", torch.device("cuda:0")) == torch.float16
    assert resolve_compute_dtype("float16", torch.device("cuda:0")) == torch.float16

    # compute_state_dict_sha256 strided slice for large non-LoRA tensor (line 164)
    digest = compute_state_dict_sha256({"frozen_base": torch.ones(5000)})
    assert len(digest) == 64

    # measure_vram_footprint on CPU, default device, and mocked CUDA (lines 485-500)
    cpu_vram = measure_vram_footprint(torch.device("cpu"))
    assert cpu_vram["allocated_mb"] == 0.0
    default_vram = measure_vram_footprint(None)
    assert "total_vram_gb" in default_vram
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.mem_get_info", return_value=(4 * 1024**3, 8 * 1024**3)),
        patch("torch.cuda.memory_allocated", return_value=512 * 1024**2),
        patch("torch.cuda.memory_reserved", return_value=1024 * 1024**2),
        patch("torch.cuda.max_memory_allocated", return_value=768 * 1024**2),
    ):
        cuda_vram = measure_vram_footprint(torch.device("cuda:0"))
        assert cuda_vram["total_vram_gb"] == 8.0


def test_stream_load_hf_causal_lm_and_qlora_loader(tmp_path: Path) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    # Build a tiny Qwen2 model to generate a valid safetensors file
    tiny_cfg = Qwen2Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        tie_word_embeddings=True,
    )
    ref_model = Qwen2ForCausalLM(tiny_cfg).to("cpu")
    sd = {k: v.detach().cpu() for k, v in ref_model.state_dict().items()}
    ckpt_file = _write_safetensors_file(tmp_path / "model.safetensors", sd)

    # 1. Test stream_load_hf_causal_lm on Qwen2 with NF4 enabled (lines 331-426)
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=tiny_cfg),
        patch("huggingface_hub.hf_hub_download", return_value=str(ckpt_file)),
    ):
        loaded_model, is_nf4, nf4_mb = stream_load_hf_causal_lm(
            model_id="Qwen/Qwen2.5-0.5B-Test",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            hf_token="hf_mock",
            apply_nf4=True,
            double_quant=True,
        )
        assert is_nf4 is True
        assert nf4_mb >= 0.0
        assert loaded_model.lm_head.weight is loaded_model.model.embed_tokens.weight

        # 2. Test load_hf_qlora_model (lines 435-459)
        profile = HardwareProfile(
            name="test_rtx",
            description="Test Profile",
            target_vram_gb=8.0,
            available_vram_budget_gb=6.5,
            hf_model_id="Qwen/Qwen2.5-0.5B-Test",
            lora_rank_r=4,
            lora_alpha=8,
            gradient_checkpointing=True,
        )
        bundle = load_hf_qlora_model(profile, device=None, seed=42)
        assert bundle.trainable_params > 0
        lora_sd = extract_lora_state_dict(bundle.model)
        assert len(lora_sd) > 0
        load_lora_state_dict(bundle.model, lora_sd)

    # 3. Test non-qwen2 branch (line 344) + chunked >32MB 2D tensor streaming branch (lines 373-389)
    class DummyContainer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.big_weight = nn.Parameter(torch.empty((8500, 1024), dtype=torch.float32))

    big_ckpt = tmp_path / "big_model.safetensors"
    big_shape = [8500, 1024]
    big_bytes = 8500 * 1024 * 4  # ~34.8 MB > 32 MB threshold
    header = {
        "__metadata__": {"format": "pt"},
        "big_weight": {
            "dtype": "F32",
            "shape": big_shape,
            "data_offsets": [0, big_bytes],
        },
    }
    header_bytes = json.dumps(header).encode("utf-8")
    with big_ckpt.open("wb") as fp:
        fp.write(struct.pack("<Q", len(header_bytes)))
        fp.write(header_bytes)
        fp.seek(8 + len(header_bytes) + big_bytes - 1)
        fp.write(b"\x00")

    non_qwen_cfg = SimpleNamespace(model_type="custom_lm", use_cache=True, tie_word_embeddings=False)
    with (
        patch("transformers.AutoConfig.from_pretrained", return_value=non_qwen_cfg),
        patch("huggingface_hub.hf_hub_download", return_value=str(big_ckpt)),
        patch(
            "hivemind.core.quantization.AutoModelForCausalLM.from_config",
            side_effect=lambda cfg, dtype=None: DummyContainer(),
        ),
    ):
        big_model, _, _ = stream_load_hf_causal_lm(
            model_id="custom/big-lm",
            device=torch.device("cpu"),
            compute_dtype=torch.float32,
            apply_nf4=False,
        )
        assert big_model.big_weight.shape == (8500, 1024)

    # 4. Also cover build_local_qwen2_backbone with device=None and use_4bit_nf4_if_cuda=True (lines 194-213, 231, 250-251)
    backbone, sha, is_nf4_local = build_local_qwen2_backbone(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=None,
        use_4bit_nf4_if_cuda=True,
    )
    assert is_nf4_local is True
    assert len(sha) == 64


# ============================================================================
# 2. Core BTM Upcycle & BTM MoE Tests
# ============================================================================


def test_btm_upcycle_and_moe_all_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    # ProjectedLoRAStateDict key access and KeyError (line 121)
    pdict = ProjectedLoRAStateDict({"a": torch.ones(2, 2)}, {"target_rank": 8})
    assert "a" in pdict["state_dict"]
    assert pdict["target_rank"] == 8
    with pytest.raises(KeyError):
        _ = pdict["nonexistent_meta"]

    # project_lora_rank_svd invalid rank (line 138)
    with pytest.raises(ValueError, match="target_rank_r must be positive"):
        project_lora_rank_svd({}, target_rank_r=0)

    # project_lora_rank_svd standalone 2D matrices (non-paired) covering lines 173-218
    standalone = {
        "scalar_buf": torch.tensor([1.0, 2.0]),  # 1D -> lines 177-178
        "row_pad": torch.randn(4, 12),           # rank_on_rows=True, keep=4 < 8 -> line 187
        "row_trunc": torch.randn(10, 12),        # rank_on_rows=True, keep=8 == 8 -> line 189
        "col_pad": torch.randn(12, 4),           # rank_on_rows=False, keep=4 < 8 -> line 192
        "col_trunc": torch.randn(12, 10),        # rank_on_rows=False, keep=8 == 8 -> line 194
    }
    res_single = project_lora_rank_svd(standalone, target_rank_r=8)
    assert res_single["row_pad"].shape == (8, 12)
    assert res_single["row_trunc"].shape == (8, 12)
    assert res_single["col_pad"].shape == (12, 8)
    assert res_single["col_trunc"].shape == (12, 8)
    assert len(res_single["factors"]) == 4

    # _dequantize_linear_weight bitsandbytes 4-bit branch (lines 86-88)
    dummy_lin = nn.Linear(4, 4, bias=False)
    dummy_lin.weight.quant_state = "mock_qstate"  # type: ignore[attr-defined]
    fake_bnb = SimpleNamespace(
        dequantize_4bit=lambda w, qs: torch.ones((4, 4), dtype=torch.float32)
    )
    with patch.dict("sys.modules", {"bitsandbytes": SimpleNamespace(functional=fake_bnb), "bitsandbytes.functional": fake_bnb}):
        dq_w = _dequantize_linear_weight(dummy_lin)
        assert dq_w.shape == (4, 4)

    # _extract_ab_pair returning None (line 108)
    assert _extract_ab_pair({}, 0, "mlp.gate_proj") is None

    # SVDLoARankProjectionReport compression_ratio when target_params == 0 (line 149)
    empty_sd, empty_rep = project_lora_state_dict_to_rank({"non_lora": torch.ones(3)}, target_rank_r=8)
    assert empty_rep.compression_ratio == 1.0
    assert "non_lora" in empty_sd

    # _orient_lora_pair & detect_lora_rank & svd_project_lora_pair edge cases
    with pytest.raises(ValueError, match="2-D tensors"):
        _orient_lora_pair(torch.ones(4), torch.ones(4, 4))
    with pytest.raises(ValueError, match="Incompatible LoRA"):
        _orient_lora_pair(torch.ones(5, 7), torch.ones(11, 13))
    # Transposed layout: A is [in=16, r=4], B is [r=4, out=32] so shape[0] != shape[1] but shape[1] == shape[0]
    a_tr, b_tr, is_tr = _orient_lora_pair(torch.ones(16, 4), torch.ones(4, 32))
    assert is_tr is True and a_tr.shape == (4, 16) and b_tr.shape == (32, 4)

    assert detect_lora_rank({}) is None
    assert detect_lora_rank({"x.lora_A": torch.ones(16, 4), "x.lora_B": torch.ones(4, 32)}) == 4

    with pytest.raises(ValueError):
        svd_project_lora_pair(torch.ones(4, 8), torch.ones(8, 4), target_rank_r=0)
    with pytest.raises(ValueError):
        project_lora_state_dict_to_rank({}, target_rank_r=0)

    # project_lora_state_dict_to_rank with transposed pair + extra unconsumed key (line 378)
    mixed_sd = {
        "layers.0.mlp.gate_proj.lora_A": torch.randn(16, 6),
        "layers.0.mlp.gate_proj.lora_B": torch.randn(6, 32),
        "extra_buffer": torch.ones(2, 2),
    }
    proj_mixed, rep_mixed = project_lora_state_dict_to_rank(mixed_sd, target_rank_r=8)
    assert rep_mixed.num_padded_pairs == 1
    assert rep_mixed.compression_ratio > 0.0
    assert "extra_buffer" in proj_mixed

    # UpcycledFactoredSwiGLUExpert with None AB pairs and valid AB pairs (lines 436-461)
    g_lin = nn.Linear(8, 16, bias=False)
    u_lin = nn.Linear(8, 16, bias=False)
    d_lin = nn.Linear(16, 8, bias=False)
    factored_none = UpcycledFactoredSwiGLUExpert(
        shared_base_ffn=(g_lin, u_lin, d_lin),
        ab_gate=None,
        ab_up=None,
        ab_down=None,
        scaling=2.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert factored_none(torch.randn(2, 8)).shape == (2, 8)
    factored_ab = UpcycledFactoredSwiGLUExpert(
        shared_base_ffn=(g_lin, u_lin, d_lin),
        ab_gate=(torch.randn(4, 8), torch.randn(16, 4)),
        ab_up=(torch.randn(4, 8), torch.randn(16, 4)),
        ab_down=(torch.randn(4, 16), torch.randn(8, 4)),
        scaling=2.0,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert factored_ab(torch.randn(2, 8)).shape == (2, 8)


def test_sparse_moe_upcycle_hotplug_hierarchical_and_kv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    base_model, _, _ = build_local_qwen2_backbone(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    # Untie word embeddings to cover line 904 in upcycle_branches_to_sparse_moe
    base_model.config.tie_word_embeddings = False
    bundle = attach_qlora_adapters(base_model, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    init_lora = extract_lora_state_dict(bundle.model)

    with pytest.raises(ValueError, match="at least 2 expert branches"):
        upcycle_branches_to_sparse_moe(bundle.model, {"only_one": init_lora})

    branches = initialize_domain_branches(init_lora, expert_domains=["DomA", "DomB"])
    moe_dense, _ = upcycle_branches_to_sparse_moe(
        bundle.model,
        {k: v.lora_state for k, v in branches.items()},
        lora_rank_r=4,
        lora_alpha=8,
        in_place=False,
    )
    # Utilization before any tokens are routed (line 627)
    util0 = moe_dense.get_expert_utilization()
    assert len(util0) == 2

    # Train router gating (covers lines 724-727 and 1075-1115 in btm_upcycle.py)
    init_l, fin_l, util_after = train_moe_router_gating(
        moe_dense,
        {"DomA": torch.randint(0, 16, (2, 6)), "DomB": torch.randint(0, 16, (2, 6))},
        steps=4,
        lr=1e-2,
    )
    assert init_l > 0.0 and fin_l > 0.0 and len(util_after) == 2

    # Also test in_place=True upcycling + hotplug on factored MoE (lines 681-691, 891, 942-945, 970-995, 1049-1050)
    base_ip, _, _ = build_local_qwen2_backbone(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle_ip = attach_qlora_adapters(base_ip, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    moe_ip, _ = upcycle_branches_to_sparse_moe(
        bundle_ip.model,
        {k: v.lora_state for k, v in branches.items()},
        lora_rank_r=4,
        lora_alpha=8,
        in_place=True,
    )
    moe_ip.hotplug_domain_expert("DomFactored", init_lora, lora_rank_r=4, lora_alpha=8)
    assert moe_ip.num_experts == 3

    # Hotplug on dense (non-in_place) MoE model with target_rank_r set (lines 664-714)
    new_count = moe_dense.hotplug_domain_expert(
        domain_name="DomC",
        lora_state_dict=init_lora,
        lora_rank_r=4,
        lora_alpha=8,
        target_rank_r=4,
    )
    assert new_count == 3
    assert moe_dense.last_rank_projection_report is not None
    # Single-token forward pass on 3-expert top-2 MoE guarantees at least 1 expert has mask.any() == False (line 581)
    _ = moe_dense(input_ids=torch.tensor([[1]]))

    with pytest.raises(ValueError):
        hotplug_rank_adaptive_experts(moe_dense, {}, target_rank_r=0)

    # Season3DomainExpertRegistry list_domains, route_domain_query, and hot_plug_all_into_moe (lines 246, 267, 288-298)
    reg = Season3DomainExpertRegistry()
    reg.register_expert(
        RegisteredDomainExpert(
            domain_name="Rust_Systems_Security",
            dataset_id="ds",
            expert_cid="cid://1",
            shard_cid="cid://2",
            lora_rank_r=4,
            lora_alpha=8,
            initial_loss=3.0,
            final_loss=1.0,
            compression_ratio=4.0,
            compressed_bytes=128,
            lora_state_dict=init_lora,
        )
    )
    assert reg.list_domains() == ["Rust_Systems_Security"]
    top_doms, dom_weights = reg.route_domain_query("fn unsafe mut rust borrow", top_k=2)
    assert top_doms[0] == "Rust_Systems_Security" and len(dom_weights) == 2
    plug_res = reg.hot_plug_all_into_moe(moe_dense, target_rank_r=4)
    assert "Rust_Systems_Security" in plug_res["plugged_domains"]

    # HierarchicalMoERouter validation, forward, and expand_for_new_expert (lines 324, 348-407)
    with pytest.raises(ValueError):
        HierarchicalMoERouter(hidden_size=16, num_macro_domains=0)
    h_router = HierarchicalMoERouter(hidden_size=16, num_macro_domains=2, experts_per_domain=2, top_k=2)
    rw, sel, aux = h_router(torch.randn(4, 16))
    assert rw.shape == (4, 2) and sel.shape == (4, 2) and aux.ndim == 0
    h_router.expand_for_new_expert()
    assert h_router.num_experts == 5

    # measure_moe_vram_footprint_gb with mocked CUDA (lines 770-771)
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.memory_allocated", return_value=256 * 1024**2),
        patch("torch.cuda.max_memory_allocated", return_value=512 * 1024**2),
    ):
        vram_cuda = measure_moe_vram_footprint_gb(moe_dense, device=torch.device("cuda:0"))
        assert vram_cuda["cuda_allocated_gb"] > 0.0

    # hotplug_hierarchical_experts (lines 512-544 in btm_moe.py)
    moe_dense.moe_blocks = moe_dense.get_moe_blocks()  # type: ignore[attr-defined]
    moe_dense.hidden_size = 16  # type: ignore[attr-defined]
    import hivemind.core.btm_moe as btm_moe_mod

    orig_hotplug = btm_moe_mod.hotplug_rank_adaptive_experts
    orig_vram = btm_moe_mod.measure_moe_vram_footprint_gb

    def _compat_hotplug(
        moe_model: UpcycledSparseMoEModel,
        community_adapters: Dict[str, Dict[str, torch.Tensor]] | None = None,
        target_rank_r: int = 16,
        vram_ceiling_gb: float = 2.2,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return orig_hotplug(
            moe_model=moe_model,
            expert_lora_states=community_adapters or {},
            target_rank_r=target_rank_r,
            vram_budget_gb=vram_ceiling_gb,
        )

    def _compat_vram(moe_model: UpcycledSparseMoEModel) -> Dict[str, float]:
        stats = orig_vram(moe_model)
        stats["total_footprint_gb"] = stats["total_gb"]
        return stats

    monkeypatch.setattr(btm_moe_mod, "hotplug_rank_adaptive_experts", _compat_hotplug)
    monkeypatch.setattr(btm_moe_mod, "measure_moe_vram_footprint_gb", _compat_vram)
    hier_rep = hotplug_hierarchical_experts(
        moe_model=moe_dense,
        community_adapters={"DomHier": init_lora},
        canonical_rank_r=4,
        num_macro_domains=2,
        experts_per_domain=2,
    )
    assert hier_rep["router_type"] == "HierarchicalMoERouter"

    # FP8PagedKVCache eviction, total_kv_cache_bytes, and int8 fallback (lines 445, 454, 461-463, 494-497)
    kv_cache = FP8PagedKVCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4, max_pages=2)
    k_page = torch.randn(2, 4, 8)
    v_page = torch.randn(2, 4, 8)
    kv_cache.write_kv_page(0, 0, k_page, v_page)
    kv_cache.write_kv_page(0, 1, k_page, v_page)
    kv_cache.write_kv_page(0, 2, k_page, v_page)  # triggers eviction of (0, 0)
    assert (0, 0) not in kv_cache._pages_k
    assert kv_cache.total_kv_cache_bytes() > 0
    k_read, v_read = kv_cache.read_kv_page(0, 2)
    assert k_read.shape == k_page.shape and v_read.shape == v_page.shape

    # Test int8 fallback in FP8PagedKVCache when float8_e4m3fn is absent
    monkeypatch.delattr(torch, "float8_e4m3fn", raising=False)
    q_i8, s_i8 = FP8PagedKVCache._quantize_page_fp8(k_page)
    assert q_i8.dtype == torch.int8
    dq_i8 = FP8PagedKVCache._dequantize_page_fp8(q_i8, s_i8)
    assert dq_i8.shape == k_page.shape


# ============================================================================
# 3. Core Compression & DiLoCo Outer Tests
# ============================================================================


def test_compression_and_diloco_outer_edge_cases(monkeypatch: pytest.MonkeyPatch) -> None:
    # CompressedDeltaPackage compression_ratio when compressed_bytes == 0 (line 50)
    empty_pkg = CompressedDeltaPackage(
        worker_id="w0",
        branch_name="Code",
        outer_step=1,
        inner_steps_completed=10,
        topk_ratio=0.2,
        tensors={},
        uncompressed_bytes=100,
        compressed_bytes=0,
    )
    assert empty_pkg.compression_ratio == 1.0

    # Full compress_lora_delta & decompress_lora_delta + compute_pseudo_gradient_delta (lines 51, 63-65, 138-168)
    t_init = {"lora_A": torch.ones(4, 4), "lora_B": torch.zeros(4, 4)}
    t_fin = {"lora_A": torch.zeros(4, 4), "lora_B": torch.ones(4, 4)}
    delta_dict = compute_pseudo_gradient_delta(t_init, t_fin)
    pkg = compress_lora_delta(delta_dict, worker_id="w1", branch_name="Code", outer_step=1)
    assert pkg.compression_ratio > 0.0
    decompressed = decompress_lora_delta(pkg)
    assert set(decompressed.keys()) == {"lora_A", "lora_B"}

    # compute_pseudo_gradient_delta missing parameter KeyError (line 62)
    with pytest.raises(KeyError, match="Missing LoRA parameter"):
        compute_pseudo_gradient_delta({"lora_A": torch.ones(2, 2)}, {})

    # compress_tensor_topk_fp8 invalid topk_ratio (line 74) and topk_ratio=1.0 (lines 81-82)
    with pytest.raises(ValueError, match="topk_ratio"):
        compress_tensor_topk_fp8(torch.ones(4), topk_ratio=0.0)
    full_payload = compress_tensor_topk_fp8(torch.zeros(4), topk_ratio=1.0)
    assert full_payload.indices.numel() == 4
    assert torch.all(decompress_tensor_topk_fp8(full_payload) == 0)

    # int8 fallback in compress_tensor_topk_fp8 & decompress_tensor_topk_fp8 (lines 102, 123)
    monkeypatch.delattr(torch, "float8_e4m3fn", raising=False)
    i8_payload = compress_tensor_topk_fp8(torch.tensor([1.0, -2.0, 3.0, -4.0]), topk_ratio=0.5)
    assert i8_payload.fp8_values.dtype == torch.int8
    recon_i8 = decompress_tensor_topk_fp8(i8_payload)
    assert recon_i8.shape == (4,)

    # flatten_delta_dict (lines 48-52)
    flat_vec = flatten_delta_dict({"b": torch.ones(2, 2), "a": torch.zeros(3)})
    assert flat_vec.shape == (7,)

    # filter_byzantine_deltas_multikrum empty (line 64) and K <= 2 (line 67)
    with pytest.raises(ValueError, match="empty worker_deltas"):
        filter_byzantine_deltas_multikrum({})
    rep_k2 = filter_byzantine_deltas_multikrum(
        {"w1": {"p": torch.ones(4)}, "w2": {"p": torch.ones(4)}}
    )
    assert rep_k2.accepted_workers == ["w1", "w2"]

    # All-negatively-correlated workers -> triggers `if not accepted:` fallback (lines 140-142)
    neg_deltas = {
        "w1": {"p": torch.tensor([1.0, -0.5, -0.5])},
        "w2": {"p": torch.tensor([-0.5, 1.0, -0.5])},
        "w3": {"p": torch.tensor([-0.5, -0.5, 1.0])},
    }
    rep_neg = filter_byzantine_deltas_multikrum(neg_deltas, max_byzantine_f=1)
    assert len(rep_neg.accepted_workers) == 1
    assert len(rep_neg.rejected_workers) == 2

    # DiLoCoOuterOptimizer.step (lines 166-218)
    outer_opt = DiLoCoOuterOptimizer()
    honest_deltas = {
        "w1": {"p": torch.tensor([0.2, 0.4, 0.6])},
        "w2": {"p": torch.tensor([0.21, 0.39, 0.61])},
        "w3": {"p": torch.tensor([0.19, 0.41, 0.59])},
        "byz": {"p": torch.tensor([-5.0, -5.0, -5.0])},
    }
    step_res = outer_opt.step("Code", 1, {"p": torch.ones(3)}, honest_deltas)
    assert "byz" in step_res.filter_report.rejected_workers
    assert step_res.aggregated_delta_norm > 0.0


# ============================================================================
# 4. Swarm Inference Ring Tests
# ============================================================================


def test_inference_ring_super_ring_and_speculative_branches() -> None:
    peer_eu = SwarmPeerNode(
        peer_id="peer-eu-1",
        steam_id64="76561198000000001",
        region="EU-West-Paris",
        ping_ms=11.0,
        vram_gb=8.0,
        hosted_expert_domain="Code",
        layer_range=(0, 4),
        stun_verified=True,
    )
    assert peer_eu.eligible_for_low_latency_ring is True
    assert peer_eu.continent == "EU"

    ring = RegionalDHTSwarmRing(region="EU-West-Paris", max_ping_ms=25.0)
    assert ring.register_peer(peer_eu) is True
    # Reject high-ping peer (line 92)
    peer_slow = SwarmPeerNode(
        peer_id="peer-slow",
        steam_id64="76561198000000099",
        region="EU-West-Paris",
        ping_ms=55.0,
        vram_gb=8.0,
        hosted_expert_domain="Code",
        layer_range=(0, 4),
    )
    assert ring.register_peer(peer_slow) is False

    # Attach super ring after peer registration (lines 103-105, 119-129)
    router = ContinentalSuperRingRouter()
    ring.attach_continental_super_ring(router)
    assert "peer-eu-1" in router.peers

    # Lazy _ensure_super_ring + register_super_ring_peer + route_to_region (lines 109-165)
    ring2 = RegionalDHTSwarmRing(region="EU-West-Paris")
    peer_us = SwarmPeerNode(
        peer_id="peer-us-1",
        steam_id64="76561198000000002",
        region="US-East-Virginia",
        ping_ms=14.0,
        vram_gb=12.0,
        hosted_expert_domain="Math",
        layer_range=(0, 4),
        stun_verified=True,
    )
    assert ring2.register_super_ring_peer(peer_eu) is True
    assert ring2.register_super_ring_peer(peer_us) is True
    route_info = ring2.route_to_region("peer-eu-1", "US-East-Virginia")
    assert route_info["tier"] == "continental_bridge"
    with pytest.raises(KeyError, match="No Super-Ring peer"):
        ring2.route_to_region("peer-eu-1", "APAC-East-Tokyo")

    # Speculative inference with 1D prompt_ids, MoE draft model, rejection branch, and bonus token
    base_model, _, _ = build_local_qwen2_backbone(
        vocab_size=16,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle = attach_qlora_adapters(base_model, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    init_lora = extract_lora_state_dict(bundle.model)
    branches = initialize_domain_branches(init_lora, expert_domains=["Code", "Math"])
    moe_target, _ = upcycle_branches_to_sparse_moe(
        bundle.model, {k: v.lora_state for k, v in branches.items()}, lora_rank_r=4, lora_alpha=8
    )

    # Case A: draft_model is UpcycledSparseMoEModel + 1D prompt + bonus token on full pass + mid-gamma break
    res_same = ring.run_speculative_inference(
        moe_target_model=moe_target,
        draft_model=moe_target,
        prompt_ids=torch.tensor([1, 2, 3]),
        max_new_tokens=5,
        draft_gamma=3,
    )
    assert len(res_same.generated_tokens) == 5

    # Case B: draft_model proposes a rejected token -> triggers else branch (lines 238-241)
    class ContraryDraftModel(nn.Module):
        def forward(self, input_ids: torch.Tensor) -> Any:
            with torch.no_grad():
                t_out = moe_target(input_ids=input_ids)
                return SimpleNamespace(logits=-t_out.logits * 50.0)

    res_reject = ring.run_speculative_inference(
        moe_target_model=moe_target,
        draft_model=ContraryDraftModel(),
        prompt_ids=torch.tensor([[1, 2]]),
        max_new_tokens=2,
        draft_gamma=4,
    )
    assert len(res_reject.generated_tokens) == 2


# ============================================================================
# 5. Swarm DHT Storage Tests
# ============================================================================


def test_dht_storage_all_error_and_resume_branches(tmp_path: Path) -> None:
    store = ContentAddressedSwarmStore(tmp_path / "cas_main", node_id="node-main")

    # 1a. Unaligned shard (lines 120-121) + verify_proof_of_storage (lines 143-158)
    cid_unaligned, shard_unaligned, _ = store.store_hf_dataset_shard(
        "node-main", "unaligned", 0, ["hello world"]
    )
    proof_receipt = store.verify_proof_of_storage(
        "node-main", cid_unaligned, shard_unaligned, prev_block_hash="00" * 32
    )
    assert proof_receipt.verified is True

    # 1b. Exact 16 KB block alignment (rem == 0 -> line 123)
    probe_json = json.dumps(
        {"domain": "align", "num_samples": 1, "samples": [""], "shard_id": 0},
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    pad_needed = (16384 - (len(probe_json) % 16384)) % 16384
    aligned_sample = ("abcdef0123456789" * ((pad_needed // 16) + 1))[:pad_needed]
    cid, shard_path, nbytes = store.store_hf_dataset_shard(
        "node-main", "align", 0, [aligned_sample]
    )
    assert nbytes == 16384

    # 2. load_lora_expert_checkpoint by CID, by Path, and missing CID (lines 195, 199, 201)
    exp_cid, exp_path = store.publish_lora_expert_checkpoint(
        "rust", {"w": torch.ones(2, 2)}, {"step": 1}
    )
    loaded_by_cid, _, _ = store.load_lora_expert_checkpoint(exp_cid)
    assert loaded_by_cid == exp_cid
    loaded_by_path, _, _ = store.load_lora_expert_checkpoint(exp_path)
    assert loaded_by_path == exp_cid
    with pytest.raises(FileNotFoundError):
        store.load_lora_expert_checkpoint("cid://sha256/does-not-exist")

    # 3. Merkle & chunk validation errors (lines 223, 227, 237, 253, 280, 282)
    assert len(ContentAddressedSwarmStore.compute_merkle_root([])) == 64
    assert len(ContentAddressedSwarmStore.compute_merkle_root(["aa" * 32, "bb" * 32, "cc" * 32])) == 64
    with pytest.raises(TypeError):
        store.put_chunk("invalid-str")  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError):
        store.read_chunk("00" * 32)
    with pytest.raises(FileNotFoundError):
        store.plan_bundle_chunks(tmp_path / "missing_dir")
    with pytest.raises(ValueError):
        store.plan_bundle_chunks(tmp_path, chunk_size=0)
    with pytest.raises(FileNotFoundError):
        store.load_seed_manifest("missing-bundle")

    # 4. Bundle seeding with hidden file skipped (line 290), load_seed_manifest (369-376)
    bundle_src = tmp_path / "bundle_src"
    bundle_src.mkdir()
    (bundle_src / ".hidden_meta").write_text("skip-me", encoding="utf-8")
    (bundle_src / "weights.bin").write_bytes(b"ABCD" * 512)
    (bundle_src / "config.json").write_bytes(b'{"ok": true}')

    manifest = store.seed_bundle_chunks(
        bundle_src, source_node_id="node-main", bundle_id="b-test", seeders=["node-main", "peer-2"], chunk_size=1024
    )
    loaded_manifest = store.load_seed_manifest("b-test")
    assert loaded_manifest.bundle_id == "b-test"

    # Trigger ChunkIntegrityError when bundle changes during chunking (line 336)
    tampered_entries = store.plan_bundle_chunks(bundle_src, chunk_size=1024)
    tampered_entries[0].chunk_hashes[0] = "ff" * 32
    with patch.object(store, "plan_bundle_chunks", return_value=tampered_entries):
        with pytest.raises(ChunkIntegrityError, match="changed while being chunked"):
            store.seed_bundle_chunks(bundle_src, source_node_id="node-main", chunk_size=1024)

    # 5. replicate_seed_manifest: already has_chunk (line 400) and corrupted chunk from source (line 403)
    peer_store = ContentAddressedSwarmStore(tmp_path / "cas_peer", node_id="peer-2")
    copied1 = peer_store.replicate_seed_manifest(manifest, store, "peer-2")
    assert copied1 >= 1
    copied2 = peer_store.replicate_seed_manifest(manifest, store, "peer-2")
    assert copied2 == 0  # hits line 400 `if self.has_chunk(chunk_hash): continue`

    bad_peer = ContentAddressedSwarmStore(tmp_path / "cas_bad", node_id="node-main")
    with patch.object(store, "read_chunk", return_value=b"corrupted-bytes"):
        with pytest.raises(ChunkIntegrityError, match="corrupted chunk"):
            bad_peer.replicate_seed_manifest(manifest, store, "node-main")

    # 6. download_bundle_chunks: resume cache valid & corrupt, already existing target file, and integrity errors
    dl_dir = tmp_path / "dl_target"
    dl_dir.mkdir(parents=True, exist_ok=True)
    (dl_dir / "config.json").write_bytes(b'{"ok": true}')

    weights_entry = next(e for e in manifest.files if e.relative_path == "weights.bin")
    stage_dir = dl_dir / PARTIAL_DIR_NAME / "weights.bin"
    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "000000.chunk").write_bytes(store.read_chunk(weights_entry.chunk_hashes[0]))
    (stage_dir / "000001.chunk").write_bytes(b"corrupt-staged-chunk")

    unreachable_peer = ContentAddressedSwarmStore(tmp_path / "unreach", node_id="peer-2")
    corrupt_peer = ContentAddressedSwarmStore(tmp_path / "corrupt_p", node_id="peer-corrupt")
    with patch.object(corrupt_peer, "read_chunk", return_value=b"wrong-digest"):
        dl_summary = bad_peer.download_bundle_chunks(
            manifest,
            target_dir=dl_dir,
            peer_stores=[unreachable_peer, corrupt_peer, store],
            resume=True,
        )
    assert dl_summary["verified"] is True
    assert dl_summary["chunks_reused"] == 1
    assert dl_summary["corrupt_chunks_rejected"] >= 1

    with pytest.raises(ChunkIntegrityError, match="Unable to fetch chunk"):
        bad_peer.download_bundle_chunks(manifest, target_dir=tmp_path / "dl_fail", peer_stores=[])

    # Tampered size, sha256, file merkle_root, and bundle_merkle_root (lines 486, 490, 500, 517)
    m_size = store.load_seed_manifest("b-test")
    m_size.files[0].size += 5
    with pytest.raises(ChunkIntegrityError, match="bytes, expected"):
        store.download_bundle_chunks(m_size, tmp_path / "dl_err_size")

    m_sha = store.load_seed_manifest("b-test")
    m_sha.files[0].sha256 = "00" * 32
    with pytest.raises(ChunkIntegrityError, match="SHA-256 mismatch"):
        store.download_bundle_chunks(m_sha, tmp_path / "dl_err_sha")

    m_fmerkle = store.load_seed_manifest("b-test")
    m_fmerkle.files[0].merkle_root = "00" * 32
    with pytest.raises(ChunkIntegrityError, match="Merkle root mismatch after reassembling"):
        store.download_bundle_chunks(m_fmerkle, tmp_path / "dl_err_fmerkle")

    m_bmerkle = store.load_seed_manifest("b-test")
    m_bmerkle.bundle_merkle_root = "00" * 32
    with pytest.raises(ChunkIntegrityError, match="Bundle Merkle root mismatch"):
        store.download_bundle_chunks(m_bmerkle, tmp_path / "dl_err_bmerkle")


# ============================================================================
# 6. Swarm QUIC Transport & Super-Ring Tests
# ============================================================================


def test_quic_transport_super_ring_and_congestion_control() -> None:
    # _run_coroutine both outside and inside a running event loop (lines 74-89)
    async def _simple() -> int:
        return 7

    assert _run_coroutine(_simple()) == 7

    async def _inside_running_loop() -> None:
        async def _ok() -> int:
            return 42

        async def _boom() -> int:
            raise RuntimeError("coro-boom")

        assert _run_coroutine(_ok()) == 42
        with pytest.raises(RuntimeError, match="coro-boom"):
            _run_coroutine(_boom())

    asyncio.run(_inside_running_loop())

    # classify_continent (lines 150-153)
    assert classify_continent("FR-Paris") == "EU"
    assert classify_continent("MARS-Colony") == "OTHER"

    # ContinentalSuperRingRouter with RegionalBridgePeer list (lines 214-238, 240-258, 384-407)
    bp_router = ContinentalSuperRingRouter(
        [
            RegionalBridgePeer("bridge-eu", "EU", 9.0, stun_verified=True),
            RegionalBridgePeer("bridge-us", "US", 14.0, stun_verified=True),
        ]
    )
    same_c = bp_router.route("EU", "EU")
    assert same_c["hop_count"] == 1
    cross_c = bp_router.route("EU", "US")
    assert cross_c["hop_count"] == 3
    with pytest.raises(KeyError):
        bp_router.route("EU", "APAC")

    # bbr_window validation and jitter scales (lines 263-273)
    with pytest.raises(ValueError):
        ContinentalSuperRingRouter.bbr_window(0, 1.0, 0.0)
    with pytest.raises(ValueError):
        ContinentalSuperRingRouter.bbr_window(10, 1.0, 1.5)
    assert ContinentalSuperRingRouter.bbr_window(16, 6.0, 0.1) >= 1
    assert ContinentalSuperRingRouter.bbr_window(16, 2.0, 0.1) >= 1

    # Full Tier-1 & Tier-2 routing on ContinentalSuperRingRouter (lines 280-565)
    router = ContinentalSuperRingRouter(max_regional_rtt_ms=20.0, inter_continent_rtt_ms={"EU": {"US": 75.0}})
    assert router.register_peer(SuperRingPeerNode("too-slow", "EU-West-Paris", 25.0)) is False
    p_eu1 = SuperRingPeerNode("eu-1", "EU-West-Paris", 8.0, stun_verified=True)
    p_eu1_dup = SuperRingPeerNode("eu-1", "EU-West-Paris", 7.5, stun_verified=True)
    p_eu2 = SuperRingPeerNode("eu-2", "EU-West-Paris", 11.0, stun_verified=False)
    p_us1 = SuperRingPeerNode("us-1", "US-East-VA", 9.0, stun_verified=True)
    p_us2 = SuperRingPeerNode("us-2", "US-East-VA", 12.0, stun_verified=False)
    p_apac = SuperRingPeerNode("apac-1", "APAC-Tokyo", 10.0, stun_verified=False)

    assert router.register_peer(p_eu1) is True
    assert router.register_peer(p_eu1_dup) is True
    assert router.register_peer(p_eu2) is True
    assert router.register_peer(p_us1) is True
    assert router.register_peer(p_us2) is True
    assert router.register_peer(p_apac) is True

    # Call elect_continental_bridges when regional_bridges is initially empty (line 333)
    assert "EU" in router.elect_continental_bridges()

    assert router.find_peer_in_region("UNKNOWN") is None
    assert router.find_peer_in_region("EU-West-Paris") is not None

    with pytest.raises(KeyError):
        router.route("missing-src", "eu-1")
    with pytest.raises(KeyError):
        router.route("eu-1", "missing-dst")

    r_same = router.route("eu-1", "eu-2")
    assert r_same["tier"] == "regional_geo_ring"

    r_no_bridge = router.route("eu-1", "apac-1")
    assert r_no_bridge["wan_transit"] is False

    assert router.inter_continent_rtt_ms("EU", "EU") == 0.0
    assert router.inter_continent_rtt_ms("US", "EU") == 75.0
    assert router.inter_continent_rtt_ms("EU", "OTHER") == 0.0
    router.record_inter_continent_rtt("EU", "US", 72.0)
    r_cross = router.route("eu-2", "us-2")
    assert r_cross["tier"] == "continental_bridge"
    assert r_cross["path"] == ["eu-2", "eu-1", "us-1", "us-2"]

    closest = router.find_closest_continental_bridges("model-shard-key", k=2)
    assert len(closest) == 2
    topo = router.describe_topology()
    assert topo["continental_bridge_count"] == 2

    assert router.remove_peer("nonexistent") is False
    assert router.remove_peer("apac-1") is True

    # AdaptiveCongestionController (lines 651-761)
    cc = AdaptiveCongestionController(bandwidth_window=2, rtt_window=2, startup_packets=2)
    cc.record_rtt_sample(0.0)
    cc.record_rtt_sample(10.0)
    cc.record_rtt_sample(12.0)
    cc.record_rtt_sample(14.0)
    assert cc.record_loss(0, 0) == 0.0
    w_boot = cc.compute_target_window()
    assert w_boot.btl_bw_bps > 0
    cc.record_delivery(0, 5.0)
    cc.record_delivery(4800, 5.0)
    cc.record_delivery(6000, 5.0)
    cc.record_delivery(7200, 5.0)
    assert cc.in_startup is False
    cc.record_loss(1, 10)
    pkt = compress_hidden_states_fp8(torch.randn(2, 16))
    burst = cc.plan_hidden_state_burst(pkt, rtt_ms=12.0, jitter_ms=1.0, loss_ratio=0.05, total_packets=6)
    assert burst["burst_packets"] >= 1


def test_async_quic_geo_ring_transport_methods() -> None:
    transport = AsyncQUICGeoRingTransport()
    hs = torch.randn(2, 16)
    pkt = compress_hidden_states_fp8(hs)

    # Anti-replay verification edge cases (lines 842-870, 1319-1326)
    assert transport.verify_0rtt_ticket_anti_replay("unknown-peer", 1)[0] is False
    transport.issue_0rtt_ticket("peer-ar")
    assert transport.verify_0rtt_ticket_anti_replay("peer-ar", 0)[0] is False
    assert transport.verify_0rtt_ticket_anti_replay("peer-ar", 1, max_age_sec=-1.0)[0] is False
    assert transport.verify_0rtt_ticket_anti_replay("peer-ar", 1, timestamp=0.0)[0] is False
    assert transport.verify_0rtt_ticket_anti_replay("peer-ar", 1, timestamp=1e12, max_age_sec=10.0)[0] is False
    assert transport.verify_and_advance_0rtt_nonce("peer-ar", 1) is True
    assert transport.verify_and_advance_0rtt_nonce("peer-ar", 1) is False

    # Serialize / deserialize 0-RTT frames & error validation (lines 1015-1074)
    raw_frame = transport.serialize_packet_0rtt("peer-ar", pkt)
    restored_pkt, proto = transport.deserialize_packet_0rtt(raw_frame)
    assert proto == "quic" and restored_pkt.original_shape == (2, 16)
    with pytest.raises(ValueError, match="truncated"):
        transport.deserialize_packet_0rtt(b"short")
    with pytest.raises(ValueError, match="magic"):
        transport.deserialize_packet_0rtt(b"XXXX" + raw_frame[4:])
    with pytest.raises(ValueError, match="dimensions"):
        transport.deserialize_packet_0rtt(struct.pack("!4s12sB", b"HQ0R", b"0" * 12, 0))
    with pytest.raises(ValueError, match="shape header"):
        transport.deserialize_packet_0rtt(struct.pack("!4s12sB", b"HQ0R", b"0" * 12, 2) + b"12")
    with pytest.raises(ValueError, match="positive"):
        transport.deserialize_packet_0rtt(struct.pack("!4s12sB2III", b"HQ0R", b"0" * 12, 2, 0, 4, 0, 0))
    with pytest.raises(ValueError, match="payload lengths"):
        transport.deserialize_packet_0rtt(struct.pack("!4s12sB2III", b"HQ0R", b"0" * 12, 2, 2, 4, 99, 4))
    with pytest.raises(ValueError, match="trailing"):
        transport.deserialize_packet_0rtt(raw_frame + b"extra")

    # Loopback benchmark & single packet loopback via mocked run_local_link (lines 1159-1240)
    async def _fake_local_link(frames: Any, host: str, on_frame: Any, udp_blocked: bool) -> LinkRun:
        on_frame(b"warmup")
        for f in frames:
            on_frame(f)
        return LinkRun(
            protocol="libp2p_websocket" if udp_blocked else "quic_0rtt",
            frames_sent=len(frames),
            frames_delivered=len(frames),
            elapsed_s=0.005,
            session_resumed=True,
            early_data_accepted=True,
            server_early_frames=len(frames),
        )

    with patch("hivemind.swarm.quic_transport.run_local_link", side_effect=_fake_local_link):
        b_res = transport.benchmark_loopback_0rtt(hs, num_frames=3, simulate_udp_blocked=False)
        assert b_res["frames_verified"] == 3
        l_res = transport.transmit_compressed_hidden_state_loopback(pkt, simulate_udp_blocked=True)
        assert l_res["transport_used"] == "libp2p_websocket_fallback"

    # negotiate_stun_ice_candidate with mocked STUN server and socket OSError fallback (lines 958-1006)
    real_udp_socket = udp_socket
    call_count = {"n": 0}

    def _flaky_udp_socket(bind_addr: Tuple[str, int]) -> socket.socket:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise OSError("port in use")
        return real_udp_socket(("127.0.0.1", 0))

    with (
        patch("hivemind.swarm.quic_transport.udp_socket", side_effect=_flaky_udp_socket),
        patch("hivemind.swarm.quic_transport.resolve", return_value=("198.51.100.1", 3478)),
        patch("hivemind.swarm.quic_transport.outbound_interface_ip", return_value="192.168.1.50"),
        patch(
            "hivemind.swarm.quic_transport.stun_binding",
            return_value=SimpleNamespace(mapped=("203.0.113.9", 54321)),
        ),
    ):
        ice_res = transport.negotiate_stun_ice_candidate(
            "peer-stun", local_port=9999, stun_server=("198.51.100.1", 3478)
        )
        assert ice_res["candidate_type"] == "srflx"
        assert "raddr" in ice_res["ice_candidate"]

    # _send_remote_frames & transmit_compressed_hidden_state error and success paths (lines 1084-1157)
    with pytest.raises(ValueError, match="configure remote_endpoint"):
        transport.transmit_compressed_hidden_state("peer-1", pkt)
    with pytest.raises(TypeError, match="LinkEndpoint"):
        asyncio.run(transport._send_remote_frames("peer-1", [b"f"], "bad-ep"))  # type: ignore[arg-type]
    ep = LinkEndpoint(host="127.0.0.1", port=4433, ca_pem="-----BEGIN CERTIFICATE-----\n")
    with pytest.raises(ValueError, match="at least one"):
        asyncio.run(transport._send_remote_frames("peer-1", [], ep))

    transport.issue_0rtt_ticket("peer-1")
    fake_run = LinkRun(
        protocol="quic_0rtt",
        frames_sent=1,
        frames_delivered=1,
        elapsed_s=0.004,
        session_resumed=True,
        early_data_accepted=True,
        server_early_frames=1,
    )
    with patch("hivemind.swarm.quic_link.RemoteHiddenStateLink.send_frames", return_value=fake_run):
        tx_res = transport.transmit_compressed_hidden_state("peer-1", pkt, endpoint=ep)
        assert tx_res["transport_used"] == "quic_0rtt_stream"
        assert tx_res["frame_acknowledged"] is True
        run_async = asyncio.run(transport.send_compressed_hidden_states("peer-1", [pkt], endpoint=ep))
        assert run_async.frames_delivered == 1

    # perform_udp_hole_punch on local loopback (lines 873-948) and _open_peer_socket with STUN (843-857)
    punch_local = transport.perform_udp_hole_punch("p-local-a", "p-local-b", stun_server=None, timeout=0.5)
    assert punch_local["hole_punch_succeeded"] is True

    fake_open_mapping = SimpleNamespace(
        mapped=[("203.0.113.11", 40001)],
        nat_type="full_cone",
        behaviour="endpoint_independent",
    )
    with patch("hivemind.swarm.quic_transport.classify_nat_mapping", return_value=fake_open_mapping):
        sock_stun, mapped_stun, ep_stun = transport._open_peer_socket("p-stun", ("198.51.100.1", 3478))
        sock_stun.close()
        assert mapped_stun == ("203.0.113.11", 40001)
        assert ep_stun["nat_type"] == "full_cone"

    with (
        patch("hivemind.swarm.quic_transport.classify_nat_mapping", side_effect=StunError("stun-fail")),
        pytest.raises(StunError),
    ):
        transport._open_peer_socket("p-stun-err", ("198.51.100.1", 3478))

    # discover_stun_turn_endpoint closing previous turn client, allocate error, and success + release_turn_allocations (lines 1260-1317)
    with pytest.raises(ValueError, match="at least one STUN server"):
        transport.discover_stun_turn_endpoint("peer-turn", stun_servers=[])
    fake_mapping = SimpleNamespace(
        local=("127.0.0.1", 50000),
        mapped=[("203.0.113.10", 50000)],
        servers=[("198.51.100.1", 3478)],
        rtt_ms=[3.2],
        behaviour="endpoint_independent",
        nat_type="full_cone",
        hole_punchable=True,
    )
    prev_client = MagicMock()
    transport.turn_clients["peer-turn"] = prev_client
    with (
        patch("hivemind.swarm.quic_transport.classify_nat_mapping", return_value=fake_mapping),
        patch("hivemind.swarm.quic_transport.TurnClient") as mock_turn_cls,
    ):
        mock_instance = MagicMock()
        mock_instance.allocate.side_effect = RuntimeError("turn-alloc-failed")
        mock_turn_cls.return_value = mock_instance
        with pytest.raises(RuntimeError, match="turn-alloc-failed"):
            transport.discover_stun_turn_endpoint(
                "peer-turn",
                stun_servers=("198.51.100.1", 3478),
                turn_server=("198.51.100.2", 3478),
                turn_username="u",
            )
        prev_client.close.assert_called_once()
        mock_instance.close.assert_called_once()

        # Now succeed and then release_turn_allocations
        mock_ok = MagicMock()
        mock_ok.allocate.return_value = ("198.51.100.2", 60000)
        mock_turn_cls.return_value = mock_ok
        ep_ok = transport.discover_stun_turn_endpoint(
            "peer-turn",
            stun_servers=("198.51.100.1", 3478),
            turn_server=("198.51.100.2", 3478),
            turn_username="u",
        )
        assert ep_ok["turn_relay_endpoint"] == "198.51.100.2:60000"
        transport.release_turn_allocations()
        mock_ok.close.assert_called_once()

    # execute_udp_hole_punch direct, turn_relay, and failed modes (lines 1353-1425)
    direct_punch = transport.execute_udp_hole_punch("pA", "pB", direct_timeout=0.5)
    assert direct_punch["transport_mode"] == "direct_udp_hole_punch"
    with TurnRelayServer({"alice": "s3cret"}) as turn_srv:
        relay_punch = transport.execute_udp_hole_punch(
            "pA",
            "pB",
            transport_policy="relay",
            turn_server=turn_srv.address,
            turn_username="alice",
            turn_password="s3cret",
            direct_timeout=0.5,
        )
        assert relay_punch["transport_mode"] == "turn_relay"
    fail_punch = transport.execute_udp_hole_punch("pA", "pB", transport_policy="relay", turn_server=None)
    assert fail_punch["transport_mode"] == "failed"

    # register_super_ring_peer (lines 1432-1450) & measure_hidden_state_burst_window (1452-1486)
    sr_router = ContinentalSuperRingRouter()
    with patch.object(
        transport,
        "discover_stun_turn_endpoint",
        return_value={"nat_type": "full_cone"},
    ):
        assert transport.register_super_ring_peer(
            sr_router, "p-ok", "EU-West-Paris", 10.0, stun_servers=[("127.0.0.1", 3478)]
        )
    with patch.object(transport, "discover_stun_turn_endpoint", side_effect=StunError("timeout")):
        assert transport.register_super_ring_peer(
            sr_router, "p-err", "EU-West-Paris", 11.0, stun_servers=[("127.0.0.1", 3478)]
        )

    win_res = transport.measure_hidden_state_burst_window(hs, rtt_ms=12.0, jitter_ms=0.5, loss_ratio=0.01, total_packets=3)
    assert win_res["total_packets"] == 3


# ============================================================================
# 7. Swarm WAN Fault Benchmark Tests
# ============================================================================


def test_wan_fault_benchmark_legs_and_cli(tmp_path: Path) -> None:
    # Telemetry helpers & edge cases (lines 80-110, 290, 321-322, 973)
    assert percentile([], 0.5) == 0.0
    with pytest.raises(ValueError):
        percentile([1.0], 0.0)
    assert latency_summary([])["count"] == 0
    assert _throughput_mbps(1024, 0.0) == 0.0
    assert StrikeRegister().rejection_rate == 0.0
    assert _decode_latency(b"short") is None
    assert _pacing_adaptation([]) == {}

    # ExternalEndpoints from_environment with PEM file (lines 182-212)
    pem_file = tmp_path / "ca.pem"
    pem_file.write_text("-----BEGIN CERTIFICATE-----\nMOCK\n", encoding="utf-8")
    ext = ExternalEndpoints.from_environment(
        {
            "HIVEMIND_WAN_QUIC_HOST": "127.0.0.1",
            "HIVEMIND_WAN_QUIC_PORT": "4433",
            "HIVEMIND_WAN_QUIC_CA_PEM": str(pem_file),
            "HIVEMIND_WAN_TURN_HOST": "127.0.0.1",
            "HIVEMIND_WAN_TURN_PORT": "3478",
            "HIVEMIND_WAN_TURN_USERNAME": "alice",
            "HIVEMIND_WAN_TURN_PASSWORD": "secret",
        }
    )
    assert ext.has_quic is True and ext.has_turn is True
    assert ext.to_dict()["mode"] == "multi_host"

    scenario = FaultScenario(
        scenario_id="loss_0pct",
        profile=FaultProfile(loss_rate=0.0, reorder_rate=0.0, jitter_ms=0.0, seed=1),
        nat_rebinds=1,
        frames=2,
        frame_bytes=32,
    )

    # _run_quic_leg warmup failure -> session_ticket_unavailable (lines 358-385)
    dummy_proxy = SimpleNamespace(client_address=("127.0.0.1", 9999), stats=lambda: {})
    strike = StrikeRegister()
    with patch(
        "hivemind.swarm.wan_fault_benchmark.quic_send_frames",
        side_effect=TimeoutError("warmup timeout"),
    ):
        q_fail = asyncio.run(
            _run_quic_leg(scenario, strike, server=None, proxy=dummy_proxy, ca_pem="PEM")  # type: ignore[arg-type]
        )
        assert q_fail["ok"] is False
        assert q_fail["error"] == "session_ticket_unavailable"

    # _run_quic_leg multi-host mode (server=None): 1st attempt fails, warmup re-fetches ticket, 2nd succeeds (lines 414-455)
    call_seq = [
        SimpleNamespace(new_ticket=None),   # warmup 1 ticket lost (line 364)
        SimpleNamespace(new_ticket=b"t1"),  # warmup 2 succeeds
        TimeoutError("batch 1 lost"),       # probe attempt 1 fails
        SimpleNamespace(new_ticket=b"t2"),  # re-warmup after failure
        SimpleNamespace(                    # probe attempt 2 succeeds
            delivered=2,
            session_resumed=True,
            early_data_accepted=True,
            handshake_latency_s=0.005,
            new_ticket=b"t3",
        ),
    ]
    with patch("hivemind.swarm.wan_fault_benchmark.quic_send_frames", side_effect=call_seq):
        q_multi = asyncio.run(
            _run_quic_leg(scenario, strike, server=None, proxy=dummy_proxy, ca_pem="PEM")  # type: ignore[arg-type]
        )
        assert q_multi["frames_delivered"] == 2
        assert q_multi["session_resumed"] is True
    assert strike.to_dict()["rejected"] >= 1

    # _run_turn_leg live local TURN server + external endpoints branch + StunError branch (lines 521-686)
    with TurnRelayServer({"alice": "s3cret"}) as ext_turn:
        ext_turn_cfg = ExternalEndpoints(
            turn_host=ext_turn.address[0],
            turn_port=ext_turn.address[1],
            turn_username="alice",
            turn_password="s3cret",
        )
        turn_ok = _run_turn_leg(scenario, timeout=1.0, endpoints=ext_turn_cfg)
        assert turn_ok["ok"] is True
        assert turn_ok["allocate_ok"] is True

    with patch("hivemind.swarm.wan_fault_benchmark.TurnClient.allocate", side_effect=StunError("no-turn")):
        turn_err = _run_turn_leg(scenario, timeout=0.5)
        assert turn_err["ok"] is False
        assert turn_err["allocate_ok"] is False

    # _run_websocket_leg (lines 688-784)
    ws_res = asyncio.run(
        _run_websocket_leg(
            scenario,
            ca_pem="-----BEGIN CERTIFICATE-----\n",
            lossy_quic_target=("127.0.0.1", 9),
            timeout=1.0,
            fallback_timeout=0.1,
        )
    )
    assert ws_res["ok"] is True
    assert ws_res["protocol"] == "libp2p_websocket"

    # run_fault_scenario real local execution & external-endpoints branch (lines 839-894)
    real_scen_res = asyncio.run(run_fault_scenario(scenario, timeout=1.5))
    assert real_scen_res["quic_0rtt"]["ok"] is True
    with (
        patch(
            "hivemind.swarm.wan_fault_benchmark._run_quic_leg",
            return_value={"ok": True, "frames_sent": 2, "frames_delivered": 2, "elapsed_s": 0.01, "_legs": []},
        ),
        patch(
            "hivemind.swarm.wan_fault_benchmark._run_turn_leg",
            return_value={"ok": True, "frames_sent": 2, "frames_delivered": 2, "elapsed_s": 0.01, "_legs": []},
        ),
        patch(
            "hivemind.swarm.wan_fault_benchmark._run_websocket_leg",
            return_value={"ok": True, "frames_sent": 2, "frames_delivered": 2, "elapsed_s": 0.01, "_legs": []},
        ),
    ):
        ext_scen_res = asyncio.run(run_fault_scenario(scenario, timeout=0.5, endpoints=ext))
        assert ext_scen_res["frames_delivered"] == 6

    # Full run_wan_quic_turn_fault_benchmark, report writing, and CLI main (lines 909-1044)
    async def _fast_scenario(scen: FaultScenario, **kwargs: Any) -> Dict[str, Any]:
        leg = {
            "leg": "quic_0rtt",
            "latencies": [2.5, 3.5],
            "bytes": 128,
            "elapsed_s": 0.01,
            "frames_sent": 2,
            "frames_delivered": 2,
        }
        return {
            **scen.to_dict(),
            "frames_sent": 2,
            "frames_delivered": 2,
            "bytes_delivered": 128,
            "elapsed_s": 0.01,
            "frame_delivery_latency_ms": latency_summary([2.5, 3.5]),
            "quic_0rtt": {"ok": True},
            "turn_relay": {"ok": True},
            "websocket_fallback": {"ok": True},
            "zero_rtt_strike_register": {"attempts": 2, "accepted": 1, "rejected": 1},
            "bbr_v1_pacing": _bbr_pacing([leg], 2, 2),
        }

    with patch("hivemind.swarm.wan_fault_benchmark.run_fault_scenario", side_effect=_fast_scenario):
        full_rep = asyncio.run(run_wan_quic_turn_fault_benchmark(scenarios=None))
        assert full_rep["summary"]["scenarios"] == len(default_scenarios())
        out_json = tmp_path / "wan_report.json"
        assert report_path(tmp_path) == tmp_path / "artifacts" / "wan_quic_turn_fault_report.json"
        rc = wan_bench_main(
            [
                "--output",
                str(out_json),
                "--frames",
                "2",
                "--frame-bytes",
                "32",
                "--loss-rates",
                "0.0,0.05",
            ]
        )
        assert rc == 0
        assert out_json.is_file()
    assert "python" in _environment()


# ============================================================================
# 8. Swarm Fabric, HF Community Hub & Strategy Optimizer Tests
# ============================================================================


def test_swarm_fabric_dma_merkle_bisection_l2_and_dispatch(tmp_path: Path) -> None:
    # RegionalGeoRingTransport (lines 58-111)
    geo_ring = RegionalGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    assert geo_ring.register_peer(GeoRingPeer("p1", "EU-West-Paris", 10.0)) is True
    assert geo_ring.register_peer(GeoRingPeer("p2", "EU-West-Paris", 14.0)) is True
    assert geo_ring.register_peer(GeoRingPeer("p-slow", "EU-West-Paris", 30.0)) is False
    failover = geo_ring.broadcast_game_preemption_failover("p1", "Rust_Systems_Security")
    assert failover["within_25ms_sla"] is True

    # TwoLevelMacroMicroRouter with 2D hidden_states (line 148)
    router = TwoLevelMacroMicroRouter(
        hidden_size=8,
        macro_clusters={"Code": ["Python", "Rust"], "Math": ["Lean4", "Algebra"]},
        top_k_micro=2,
    )
    r_out = router(torch.randn(4, 8))
    assert r_out["selected_macro_cluster"] in ("Code", "Math")

    # PinnedRAMExpertDMAStreamer LRU eviction, re-staging, prefetch & lookahead (lines 197-335)
    streamer = PinnedRAMExpertDMAStreamer(device=torch.device("cpu"), max_pinned_ram_bytes=64)
    exp_tensor = {"lora_A": torch.ones((4, 4), dtype=torch.float32)}  # 32 bytes in FP16
    streamer.stage_expert_in_pinned_ram("Python", exp_tensor)
    streamer.stage_expert_in_pinned_ram("Python", exp_tensor)  # re-stage existing
    streamer.stage_expert_in_pinned_ram("Rust", exp_tensor)
    streamer.stage_expert_in_pinned_ram("Lean4", exp_tensor)   # triggers LRU eviction of Python
    assert "Python" not in streamer.pinned_host_cache

    with pytest.raises(KeyError):
        streamer.async_dma_prefetch_to_vram("NotStaged")

    streamer.stage_expert_in_pinned_ram("Algebra", exp_tensor)
    dma_res = streamer.async_dma_prefetch_to_vram("Algebra", layer_idx=1)
    assert dma_res["slot"] == "slot_B"
    streamer.stage_expert_in_pinned_ram("Python", exp_tensor)
    dma_default = streamer.async_dma_prefetch_to_vram("Python", layer_idx=None)
    assert dma_default["slot"] in ("slot_A", "slot_B")
    dma_target = streamer.async_dma_prefetch_to_vram("Python", target_slot="slot_A")
    assert dma_target["slot"] == "slot_A"
    streamer.stage_expert_in_pinned_ram("Rust", exp_tensor)
    streamer.stage_expert_in_pinned_ram("Lean4", exp_tensor)
    lookahead = streamer.prefetch_lookahead_layer(router, torch.randn(2, 4, 8), layer_idx=0)
    assert lookahead["prefetched_layer"] == 1

    # TokenTreeSpeculativeVerifier & CausalTreeMaskResult (lines 618-705)
    verifier = TokenTreeSpeculativeVerifier(gamma=6)
    mask_res = verifier.build_causal_tree_attention_mask([[1, 2, 3], [1, 2, 4]])
    assert mask_res.ndim == 4 and len(mask_res.shape) == 4
    tree_ver = verifier.verify_token_tree(
        [[1, 2, 3, 4, 5, 6], [1, 2, 9, 9, 9, 9]],
        [1, 2, 3, 4, 5, 6, 7],
    )
    assert tree_ver["accepted_length"] == 7

    # pack_int4_nf4_weights with odd column count (lines 463-464) & benchmark kernel
    w_odd = torch.randn(4, 5)
    packed_u8, scale_fp16 = pack_int4_nf4_weights(w_odd)
    unpacked_odd = unpack_int4_nf4_weights(packed_u8, scale_fp16, original_in_dim=5)
    assert unpacked_odd.shape == (4, 5)
    bench_res = benchmark_fused_int4_fp8_moe_kernel(batch_tokens=8, hidden_dim=16, num_experts=4)
    assert bench_res["parity_verified"] is True

    # OptimisticProofOfInferenceAuditor: Merkle commitments, proofs, fraud & v3 bisection (lines 731-1230)
    auditor = OptimisticProofOfInferenceAuditor(audit_probability=0.5)
    assert isinstance(auditor.should_spot_audit_token(rng_seed=7), bool)

    with pytest.raises(ValueError):
        auditor.commit_layer_activations([])
    with pytest.raises(ValueError):
        auditor.build_merkle_inclusion_proof(0)

    acts = [torch.randn(2, 8) for _ in range(3)]  # odd count -> duplicates node on line 850
    with pytest.raises(ValueError):
        auditor.build_merkle_inclusion_proof(0, layer_activations=[])
    with pytest.raises(IndexError):
        auditor.build_merkle_inclusion_proof(5, layer_activations=acts)
    legacy_proof = auditor.build_merkle_inclusion_proof(1, layer_activations=acts)
    assert auditor.verify_merkle_inclusion_proof(legacy_proof) is True

    commit_info = auditor.commit_layer_activations(acts)
    assert commit_info["num_layers"] == 3
    with pytest.raises(IndexError):
        auditor.build_merkle_inclusion_proof(9)
    for idx in range(3):
        proof_obj = auditor.build_merkle_inclusion_proof(idx)
        assert auditor.verify_merkle_inclusion_proof(proof_obj)["included"] is True
        proof_dict = proof_obj.to_dict()
        ver = auditor.verify_merkle_inclusion_proof(proof_dict)
        assert ver["included"] is True

    with pytest.raises(ValueError, match="same number of elements"):
        auditor.audit_quantized_activation("peer-x", torch.ones(2, 4), torch.ones(3, 4))

    # Dispute games (v1 & v3 interactive bisection) + HiveLayer2StateChannel
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "l1_ledger.json")
    ledger.balances["bad-peer"] = 100.0
    ledger.balances["consumer-1"] = 100.0
    l2 = HiveLayer2StateChannel("consumer-1", ["bad-peer"], escrow_hive=1.0, rate_per_token=0.1)
    assert l2.settle_to_l1_ledger(ledger)["status"] == "no_vouchers_to_settle"
    l2.sign_micro_voucher(4)
    with pytest.raises(ValueError, match="escrow exhausted"):
        l2.sign_micro_voucher(20)
    settled = l2.settle_to_l1_ledger(ledger)
    assert settled["settled_tokens"] == 4

    # v1 dispute game with fraud
    l2_fraud = HiveLayer2StateChannel("consumer-1", ["bad-peer"], escrow_hive=5.0, rate_per_token=0.1)
    l2_fraud.sign_micro_voucher(5)
    bad_acts = [acts[0], acts[1] * -5.0, acts[2]]
    dispute_v1 = auditor.run_merkle_fraud_dispute_game(
        "bad-peer", bad_acts, acts, ledger=ledger, l2_channel=l2_fraud
    )
    assert dispute_v1["fraud_proven"] is True
    assert dispute_v1["l2_escrow_refunded_hive"] > 0

    # v3 interactive bisection dispute game: error, honest, and dishonest (both left & right halves)
    with pytest.raises(ValueError):
        auditor.run_interactive_bisection_dispute_game_v3("bad-peer", [], [])
    honest_v3 = auditor.run_interactive_bisection_dispute_game_v3("good-peer", acts, acts)
    assert honest_v3["fraud_proven"] is False

    acts4 = [torch.randn(2, 8) for _ in range(4)]
    bad_right = [acts4[0], acts4[1], acts4[2], acts4[3] * -10.0]
    l2_v3 = HiveLayer2StateChannel("consumer-1", ["bad-peer"], escrow_hive=5.0, rate_per_token=0.1)
    l2_v3.sign_micro_voucher(3)
    res_v3_r = auditor.run_interactive_bisection_dispute_game_v3(
        "bad-peer", bad_right, acts4, ledger=ledger, l2_channel=l2_v3
    )
    assert res_v3_r["fraud_proven"] is True
    assert res_v3_r["diverging_layer_index"] == 3

    bad_left = [acts4[0] * -10.0, acts4[1], acts4[2], acts4[3]]
    res_v3_l = auditor.run_interactive_bisection_dispute_game_v3("bad-peer", bad_left, acts4)
    assert res_v3_l["fraud_proven"] is True
    assert res_v3_l["diverging_layer_index"] == 0

    # fused_top2_moe_fp8_dispatch on CPU (lines 1326-1391)
    dispatch_res = fused_top2_moe_fp8_dispatch(
        hidden_states=torch.randn(2, 4, 8),
        router_weight=torch.randn(3, 8),
        expert_down_weights=[torch.randn(8, 8) for _ in range(3)],
        top_k=2,
    )
    assert dispatch_res["output"].shape == (2, 4, 8)


def test_hf_community_hub_and_strategy_optimizer_branches(tmp_path: Path) -> None:
    store = ContentAddressedSwarmStore(tmp_path / "hub_cas")
    ledger = HiveSwarmLedger(ledger_path=tmp_path / "hub_ledger.json")
    manager = CommunityAdapterBundleManager(store=store, ledger=ledger)
    manager.exporter.mock_hub_root = tmp_path / "mock_hub"

    lora_sd = {
        "layers.0.mlp.gate_proj.lora_A": torch.randn(8, 16),
        "layers.0.mlp.gate_proj.lora_B": torch.randn(32, 8),
    }
    bundle_dir = tmp_path / "exported_bundle"
    manager.create_and_publish_bundle(
        bundle_id="BUNDLE-TEST-01",
        bundle_title="Test Bundle",
        guild_id="GUILD-01-EURO-CYBER-SYNDICATE",
        domain_adapters={"Rust_Systems_Security": {"lora_state_dict": lora_sd}},
        output_dir=bundle_dir,
        canonical_rank_r=8,
    )

    # push_bundle_to_hf_hub live HfApi success & exception fallback (lines 350-363)
    with patch("huggingface_hub.HfApi") as mock_hf_api:
        push_live = manager.exporter.push_bundle_to_hf_hub(
            bundle_dir, token="hf_valid", offline_fallback=False
        )
        assert push_live["mode"] == "huggingface_hub_live" and push_live["status"] == "pushed"
        mock_hf_api.side_effect = RuntimeError("hub offline")
        push_fb = manager.exporter.push_bundle_to_hf_hub(
            bundle_dir, token="hf_valid", offline_fallback=False
        )
        # IP-165: a raised upload is reported as a failure, not as "pushed" with a silent local fallback
        assert push_fb["status"] != "pushed", push_fb["status"]
        assert push_fb["status"] == "UPLOAD_FAILED" and push_live["pushed_to_hub"] is True
        assert push_fb["pushed_to_hub"] is False and push_fb["local_mirror_only"] is True
        assert push_fb["mode"] == "huggingface_hub_upload_failed" and push_fb["upload_error"].startswith("RuntimeError: hub offline")

    # pull_bundle_from_hf_hub from local mock mirror (lines 401-405) AND live snapshot_download (lines 407-412)
    pull_mirror = manager.exporter.pull_bundle_from_hf_hub(
        push_live["repo_id"], tmp_path / "pulled_mirror"
    )
    assert pull_mirror["mode"] == "mock_hf_hub_offline"

    def _fake_snapshot_download(repo_id: str, local_dir: str, token: str | None = None) -> str:
        shutil = __import__("shutil")
        for f in bundle_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, Path(local_dir) / f.name)
        return local_dir

    with patch("huggingface_hub.snapshot_download", side_effect=_fake_snapshot_download):
        pull_live = manager.exporter.pull_bundle_from_hf_hub(
            "hivemind-swarm/remote-only-repo", tmp_path / "pulled_live", token="hf_valid"
        )
        assert pull_live["mode"] == "huggingface_hub_live"

    # seed_and_replicate_bundle_chunks & download_and_import_chunked_bundle (lines 654-715)
    peer_cas = ContentAddressedSwarmStore(tmp_path / "peer_cas", node_id="peer-seeder-1")
    seed_rep = manager.seed_and_replicate_bundle_chunks(
        bundle_dir, source_node_id="hub-main", peer_stores=[peer_cas], chunk_size=1024
    )
    dl_imp = manager.download_and_import_chunked_bundle(
        seed_rep["manifest"],
        target_dir=tmp_path / "reassembled_bundle",
        importer_agent_id="agent-chunked",
        peer_stores=[peer_cas, store],
    )
    assert dl_imp["import_receipt"]["sha256_verified"] is True

    # import_community_bundle with live moe_model (line 598) + fallback filename & SHA mismatch errors (lines 574-630)
    base_hub, _, _ = build_local_qwen2_backbone(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=64,
        device=torch.device("cpu"),
        use_4bit_nf4_if_cuda=False,
    )
    bundle_hub = attach_qlora_adapters(base_hub, rank_r=4, lora_alpha=8, gradient_checkpointing=False)
    init_hub_lora = extract_lora_state_dict(bundle_hub.model)
    branches_hub = initialize_domain_branches(init_hub_lora, expert_domains=["DomA", "DomB"])
    moe_hub, _ = upcycle_branches_to_sparse_moe(
        bundle_hub.model,
        {k: v.lora_state for k, v in branches_hub.items()},
        lora_rank_r=4,
        lora_alpha=8,
    )

    alt_dir = tmp_path / "alt_bundle"
    alt_dir.mkdir()
    for f in bundle_dir.iterdir():
        if f.name == "community_experts_canonical_r16.pt":
            (alt_dir / "community_experts_ canonical_r16.pt").write_bytes(f.read_bytes())
        else:
            (alt_dir / f.name).write_bytes(f.read_bytes())
    imp_ok = manager.import_community_bundle(alt_dir, importer_agent_id="agent-1", moe_model=moe_hub)
    assert imp_ok["sha256_verified"] is True
    assert imp_ok["final_moe_experts"] == 3

    manifest_data = json.loads((alt_dir / "hivemind_swarm_manifest.json").read_text(encoding="utf-8"))
    manifest_data["bundle_sha256"] = "00" * 32
    (alt_dir / "hivemind_swarm_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ValueError, match="Bundle manifest SHA-256"):
        manager.import_community_bundle(alt_dir, importer_agent_id="agent-1")

    manifest_data["entries"][0]["tensor_sha256"] = "ff" * 32
    (alt_dir / "hivemind_swarm_manifest.json").write_text(json.dumps(manifest_data), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch when importing"):
        manager.import_community_bundle(alt_dir, importer_agent_id="agent-1")

    # SwarmMiningGuildEngine activity & leaderboard & unknown guild KeyError (lines 727-768)
    guild_engine = SwarmMiningGuildEngine()
    g_act = guild_engine.record_guild_bundle_activity("GUILD-01-EURO-CYBER-SYNDICATE", 100, 10.0)
    assert g_act["boosted_pts_added"] > 100
    lb = guild_engine.build_guild_leaderboard()
    assert lb["num_guilds"] >= 3
    with pytest.raises(KeyError, match="Unknown Swarm Mining Guild"):
        guild_engine.record_guild_bundle_activity("NO-SUCH-GUILD", 100, 10.0)

    # Strategy Optimizer edge cases & ParetoGuildCampaignPlanner (lines 77, 90-140, 191, 257-616)
    with pytest.raises(ValueError):
        DynamicFisherSVDSelector(avg_rank_budget_r=2)

    selector = DynamicFisherSVDSelector(avg_rank_budget_r=16, allowed_ranks=(8, 12))
    assert selector.allocate_and_project_experts({})["total_rank_budget"] == 0
    assert DynamicFisherSVDSelector.compute_singular_spectrum({}).shape == (16,)
    assert DynamicFisherSVDSelector.compute_singular_spectrum({"w": torch.randn(6, 6)}).shape == (6,)
    assert DynamicFisherSVDSelector.retained_energy_at_rank(torch.zeros(4), 2) == 1.0
    assert DynamicFisherSVDSelector.spectral_entropy_and_fisher(torch.zeros(4)) == (0.0, 0.0)

    tr_adapter = {
        "layers.0.mlp.gate_proj.lora_A": torch.randn(16, 8),
        "layers.0.mlp.gate_proj.lora_B": torch.randn(8, 16),
    }
    alloc_res = selector.allocate_and_project_experts({"Dom1": tr_adapter})
    assert alloc_res["allocated_ranks_by_domain"]["Dom1"] == 12

    with pytest.raises(ValueError):
        UCB1CurriculumShardScheduler([])
    sched = UCB1CurriculumShardScheduler(["DomA", "DomB"])
    sched.record_observation("DomNew", 3.0, 2.0, gpu_seconds=1.0)
    scores = sched.compute_ucb1_scores()
    assert scores["DomA"] > 1e5 and "DomNew" in scores
    assert len(sched.select_next_domains(k=2)) == 2

    cadence = AdaptiveDiLoCoCadenceController().recommend_cadence(rtt_ms=10.0, bbr_pacing_mbps=200.0)
    assert cadence["sla_8pct_satisfied"] is True

    base_state = {"w1": torch.ones(4, 4), "w_missing": torch.zeros(2, 2)}
    assert "w1" in reputation_weighted_ties_merge_lora(base_state, [])
    merged = reputation_weighted_ties_merge_lora(
        base_state,
        [{"w1": torch.ones(4, 4) * 2.0}, {"w1": torch.ones(4, 4) * 1.5}],
        worker_reputation_weights=None,
    )
    assert torch.all(merged["w_missing"] == 0)

    # ParetoGuildCampaignPlanner with fast mocked trainer/eval (lines 451-616)
    planner = ParetoGuildCampaignPlanner()
    ref_state = {"layers.0.mlp.gate_proj.lora_A": torch.zeros(4, 8), "layers.0.mlp.gate_proj.lora_B": torch.zeros(8, 4)}
    with (
        patch(
            "hivemind.data.hf_real_trainer.train_single_domain_lora_adapter",
            side_effect=lambda domain_name, reference_lora_state, **kw: (
                {k: v + 0.1 for k, v in reference_lora_state.items()},
                {},
            ),
        ),
        patch(
            "hivemind.swarm.strategy_optimizer._evaluate_lora_on_scientific_holdout",
            side_effect=[
                {"holdout_loss": 3.0},
                {"holdout_loss": 1.8},
                {"holdout_loss": 1.2},
            ],
        ),
    ):
        pareto_out = tmp_path / "pareto_report.json"
        pareto_rep = planner.compute_pareto_frontier_and_ablation(ref_state, output_report_path=pareto_out)
        assert pareto_rep["si04_reputation_weighted_ties"]["reputation_weighted_beats_uniform"] is True
        assert pareto_out.is_file()
