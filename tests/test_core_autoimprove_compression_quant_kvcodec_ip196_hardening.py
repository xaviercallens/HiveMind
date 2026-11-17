# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Iteration 196 negative-control and invariant hardening tests for:
- `hivemind.core.auto_improvement`
- `hivemind.core.compression`
- `hivemind.core.quantization`
- `hivemind.core.kv_page_codecs`
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn as nn

from hivemind.core.auto_improvement import (
    AutoImprovementExecutionResult,
    AutoImprovementWorkunit,
    NocturnalAutoImprovementScheduler,
    RetrofitReport,
    resolve_domain_canonical,
    select_dream_phase_teacher_for_vram,
)
from hivemind.core.compression import (
    CompressedDeltaPackage,
    CompressedTensorPayload,
    DCTSparse2BitTensorPayload,
    compress_lora_delta,
    compress_lora_delta_dct_sparse_2bit,
    compress_tensor_dct_sparse_2bit,
    compress_tensor_topk_fp8,
)
from hivemind.core.kv_page_codecs import (
    FP8E4M3PageCodec,
    IntMinMaxPageCodec,
    LatticePageCodec,
    PluggableKVPageCache,
)
from hivemind.core.quantization import (
    QuantizedModelBundle,
    build_local_qwen2_backbone,
    create_lora_config,
)


def test_auto_improvement_control_chars_hex_digest_and_mcnemar_bounds() -> None:
    with pytest.raises(ValueError, match="domain_or_cluster"):
        select_dream_phase_teacher_for_vram(8.0, "coding\x7f")
    with pytest.raises(ValueError, match="domain"):
        resolve_domain_canonical("maths\x7f")

    wu = AutoImprovementWorkunit(
        wu_id="wu-196",
        domain="maths_formal",
        base_model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        target_adapter_name="adapter-196",
        rank=16,
        num_samples=8,
        inner_steps=10,
        min_vram_gb=6.0,
        time_budget_minutes=10.0,
        seed=42,
        created_utc=1700000000.0,
    )
    assert len(wu.sha256_digest) == 64

    with pytest.raises(ValueError, match="wu_id"):
        replace(wu, wu_id="wu\x7f196")
    with pytest.raises(ValueError, match="hf_dataset"):
        replace(wu, hf_dataset="bad\nrepo")
    with pytest.raises(ValueError, match="cluster"):
        replace(wu, cluster="bad\x7fcluster")
    with pytest.raises(ValueError, match="sample_pairs"):
        replace(wu, sample_pairs=["not-a-dict"])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="sha256_digest"):
        replace(wu, sha256_digest="z" * 64)
    with pytest.raises(ValueError, match="sha256_digest"):
        replace(wu, sha256_digest="abc123")

    res = AutoImprovementExecutionResult(
        wu_id="wu-196",
        domain="maths_formal",
        worker_id="w-1",
        account_id="acct-1",
        steps_completed=10,
        loss_initial=2.5,
        loss_final=1.8,
        accuracy_base=0.5,
        accuracy_adapted=0.8,
        mcnemar_p_value=0.001,
        gate_passed=True,
        contributed_tflops=1.25,
        peak_vram_mb=4096.0,
        duration_seconds=30.0,
        delta_weights_hash="a" * 64,
    )
    with pytest.raises(ValueError, match="worker_id"):
        replace(res, worker_id="w\x7f1")
    with pytest.raises(ValueError, match="hf_dataset"):
        replace(res, hf_dataset="bad\x00dataset")
    with pytest.raises(ValueError, match="cluster"):
        replace(res, cluster="bad\x7fcluster")

    with pytest.raises(ValueError, match="credited_accounts"):
        RetrofitReport(
            total_workunits=1,
            newly_credited_count=1,
            already_credited_count=0,
            total_hmt_granted=10.0,
            total_tflops_attested=1.0,
            credited_accounts={"acct\x7f1": 10.0},
            ledger_verified=True,
            timestamp_utc=1700000000.0,
        )

    with pytest.raises(ValueError, match="base_model_id"):
        NocturnalAutoImprovementScheduler(base_model_id="qwen\x7f")

    sched = NocturnalAutoImprovementScheduler()
    with pytest.raises(ValueError, match="account_id"):
        sched.filter_eligible_producers([{"account_id": "acct\x7f1", "vram_gb": 8.0, "last_heartbeat_utc": 100.0}], now_utc=105.0)
    with pytest.raises(ValueError, match="base_correct cannot exceed total_eval_samples"):
        sched.evaluate_mcnemar_gate(
            base_correct=201,
            adapted_correct=150,
            base_only_correct=5,
            adapted_only_correct=25,
            total_eval_samples=200,
        )
    with pytest.raises(ValueError, match="adapted_correct cannot exceed total_eval_samples"):
        sched.evaluate_mcnemar_gate(
            base_correct=100,
            adapted_correct=201,
            base_only_correct=5,
            adapted_only_correct=25,
            total_eval_samples=200,
        )
    with pytest.raises(ValueError, match="base_only_correct \\+ adapted_only_correct cannot exceed total_eval_samples"):
        sched.evaluate_mcnemar_gate(
            base_correct=100,
            adapted_correct=150,
            base_only_correct=120,
            adapted_only_correct=90,
            total_eval_samples=200,
        )
    with pytest.raises(ValueError, match="worker_id"):
        sched.verify_workunit_execution(wu, worker_id="w\x7f1", account_id="acct-1")


def test_compression_payload_and_package_post_init_invariants() -> None:
    sample = torch.linspace(-1.0, 1.0, 64, dtype=torch.float32).reshape(8, 8)
    fp8_payload = compress_tensor_topk_fp8(sample, topk_ratio=0.25)
    assert fp8_payload.compressed_bytes > 0

    with pytest.raises(ValueError, match="original_shape"):
        replace(fp8_payload, original_shape=(8, 0))
    with pytest.raises(ValueError, match="original_numel"):
        replace(fp8_payload, original_numel=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="math.prod\\(original_shape\\) must equal original_numel"):
        replace(fp8_payload, original_numel=32)
    with pytest.raises(ValueError, match="indices and fp8_values must be torch.Tensors"):
        replace(fp8_payload, indices=[1, 2])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="indices and fp8_values must have equal numel"):
        replace(fp8_payload, indices=fp8_payload.indices[:1])
    with pytest.raises(ValueError, match="scale must be a number"):
        replace(fp8_payload, scale=True)  # type: ignore[arg-type]

    dct_payload = compress_tensor_dct_sparse_2bit(sample, topk_ratio=0.25)
    assert dct_payload.compressed_bytes > 0

    with pytest.raises(ValueError, match="original_shape"):
        replace(dct_payload, original_shape=())
    with pytest.raises(ValueError, match="math.prod\\(original_shape\\) must equal original_numel"):
        replace(dct_payload, original_numel=16)
    with pytest.raises(ValueError, match="padded_shape"):
        replace(dct_payload, padded_shape=(4, 4))
    with pytest.raises(ValueError, match="indices, packed_2bit, and lookup must be torch.Tensors"):
        replace(dct_payload, lookup=[1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="shift and scale must be numbers"):
        replace(dct_payload, scale=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_coefficients"):
        replace(dct_payload, n_coefficients=-1)
    with pytest.raises(ValueError, match="chunk_size"):
        replace(dct_payload, chunk_size=1)
    with pytest.raises(ValueError, match="index_width"):
        replace(dct_payload, index_width=3)
    with pytest.raises(ValueError, match="dct_energy_retained_fraction"):
        replace(dct_payload, dct_energy_retained_fraction=1.5)

    pkg = compress_lora_delta({"lora_A": sample}, worker_id="w-1", branch_name="Code", outer_step=1)
    assert pkg.compression_ratio > 0.0

    with pytest.raises(ValueError, match="worker_id"):
        replace(pkg, worker_id="w\n1")
    with pytest.raises(ValueError, match="branch_name"):
        replace(pkg, branch_name="Code\x7f")
    with pytest.raises(ValueError, match="compression_mode"):
        replace(pkg, compression_mode="")
    with pytest.raises(ValueError, match="outer_step"):
        replace(pkg, outer_step=-1)
    with pytest.raises(ValueError, match="topk_ratio"):
        replace(pkg, topk_ratio=1.5)
    with pytest.raises(ValueError, match="tensors must be a dict"):
        replace(pkg, tensors="not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tensors keys"):
        replace(pkg, tensors={"bad\nkey": fp8_payload})
    with pytest.raises(ValueError, match="tensors values"):
        replace(pkg, tensors={"lora_A": sample})  # type: ignore[dict-item]

    for fn in (compress_lora_delta, compress_lora_delta_dct_sparse_2bit):
        with pytest.raises(ValueError, match="worker_id"):
            fn({"lora_A": sample}, "w\x7f1", "Code", 1)
        with pytest.raises(ValueError, match="branch_name"):
            fn({"lora_A": sample}, "w-1", "Code\n1", 1)


def test_quantization_bundle_hex_digest_lora_targets_and_head_divisibility() -> None:
    bundle = QuantizedModelBundle(
        model=nn.Linear(4, 4),
        base_model_id="Qwen/Qwen2.5-0.5B",
        base_model_sha256="a" * 64,
        device=torch.device("cpu"),
        compute_dtype=torch.float32,
        is_4bit_nf4=True,
        nf4_compressed_base_mb=1.0,
        trainable_params=4,
        total_params=20,
        lora_size_mb=0.05,
    )
    with pytest.raises(ValueError, match="base_model_id"):
        replace(bundle, base_model_id="Qwen\x7f")
    with pytest.raises(ValueError, match="base_model_sha256"):
        replace(bundle, base_model_sha256="g" * 64)

    with pytest.raises(ValueError, match="target_modules"):
        create_lora_config(target_modules=[])
    with pytest.raises(ValueError, match="target_modules"):
        create_lora_config(target_modules=["q_proj", "bad\nproj"])

    with pytest.raises(ValueError, match="hidden_size must be divisible by num_attention_heads"):
        build_local_qwen2_backbone(hidden_size=250, num_attention_heads=4, num_key_value_heads=2)
    with pytest.raises(ValueError, match="num_attention_heads must be divisible by num_key_value_heads"):
        build_local_qwen2_backbone(hidden_size=256, num_attention_heads=4, num_key_value_heads=3)
    with pytest.raises(ValueError, match="device must be a torch.device"):
        build_local_qwen2_backbone(device="cpu")  # type: ignore[arg-type]


def test_kv_page_codecs_decode_dtype_and_cache_layer_bounds() -> None:
    page = torch.linspace(-1.0, 1.0, 32, dtype=torch.float32).reshape(4, 8)
    fp8 = FP8E4M3PageCodec()
    int4 = IntMinMaxPageCodec(bits=4)
    lat = LatticePageCodec(kind="E8", scale=0.25)

    for codec in (fp8, int4, lat):
        encoded = codec.encode(page)
        with pytest.raises(ValueError, match="dtype must be a torch.dtype"):
            codec.decode(encoded, dtype="float16")  # type: ignore[arg-type]

    cache = PluggableKVPageCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4)
    cache.write_kv_page(1, 0, page, page)
    k_out, v_out = cache.read_kv_page(1, 0, target_dtype=torch.float32)
    assert k_out.shape == page.shape and v_out.shape == page.shape

    with pytest.raises(ValueError, match="layer_idx must be an integer in 0..1"):
        cache.write_kv_page(2, 0, page, page)
    with pytest.raises(ValueError, match="layer_idx must be an integer in 0..1"):
        cache.read_kv_page(2, 0)
    with pytest.raises(ValueError, match="target_dtype must be a torch.dtype"):
        cache.read_kv_page(1, 0, target_dtype="float16")  # type: ignore[arg-type]
