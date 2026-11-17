# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control tests for core hybrid distributor, KV page codecs, nocturnal coding, and vision V-MoE."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from hivemind.core.hybrid_inference_and_large_gpu_image_moe_distributor import (
    PRODUCER_AND_HYBRID_FLEET,
    execute_hybrid_node_inference_and_micro_training,
    execute_large_gpu_producer_training,
)
from hivemind.core.kv_page_codecs import (
    FP8E4M3PageCodec,
    IntMinMaxPageCodec,
    PluggableKVPageCache,
)
from hivemind.core.nocturnal_coding_moe import (
    decompose_engineering_epic_to_dag,
    multi_krum_select,
)
from hivemind.core.reddit_vision_vlm_moe import (
    QLoRAVisionAdapter,
    dequantize_sparseloco_dct_2bit,
    merge_btm_attention_weights,
    quantize_adamw_state_8bit,
    quantize_sparseloco_dct_2bit,
)


def test_execute_large_gpu_and_hybrid_node_validation() -> None:
    large_node = next(n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "LARGE_GPU_DEDICATED_TRAINER")
    hybrid_node = next(n for n in PRODUCER_AND_HYBRID_FLEET if n.pool_role == "HYBRID_INFERENCE_AND_MICRO_TRAINER")

    with pytest.raises(ValueError, match="node must be a GPUProducerNodeSpec"):
        execute_large_gpu_producer_training(
            node="not-a-node",  # type: ignore[arg-type]
            assigned_shards=2,
            assigned_divisions=["div_1_aesthetics_digital_art"],
            inner_steps=2,
        )
    with pytest.raises(ValueError, match="assigned_divisions must be a sequence"):
        execute_large_gpu_producer_training(
            node=large_node,
            assigned_shards=2,
            assigned_divisions="div_1_aesthetics_digital_art",  # type: ignore[arg-type]
            inner_steps=2,
        )
    with pytest.raises(ValueError, match="Unknown division ID"):
        execute_large_gpu_producer_training(
            node=large_node,
            assigned_shards=2,
            assigned_divisions=["nonexistent_division"],
            inner_steps=2,
        )
    with pytest.raises(ValueError, match="node must be a GPUProducerNodeSpec"):
        execute_hybrid_node_inference_and_micro_training(
            node=None,  # type: ignore[arg-type]
        )
    report = execute_hybrid_node_inference_and_micro_training(
        node=hybrid_node, incoming_inference_requests=4, hot_swapped_adapters_count=2
    )
    assert report.node_id == hybrid_node.node_id


def test_kv_page_codecs_payload_and_codec_validation() -> None:
    fp8_codec = FP8E4M3PageCodec()
    with pytest.raises(ValueError, match="payload must be a 2-tuple of torch.Tensor"):
        fp8_codec.decode(("bad",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload must be a 2-tuple of torch.Tensor"):
        fp8_codec.payload_bits((torch.ones(4), "not-a-tensor"))  # type: ignore[arg-type]

    int4_codec = IntMinMaxPageCodec(4)
    with pytest.raises(ValueError, match="payload must be a 3-tuple of torch.Tensor"):
        int4_codec.decode((torch.ones(4), torch.ones(1)))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="payload must be a 3-tuple of torch.Tensor"):
        int4_codec.payload_bits((torch.ones(4), torch.ones(1), 123))  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="codec must implement KVPageCodec"):
        PluggableKVPageCache(codec=object())  # type: ignore[arg-type]

    cache = PluggableKVPageCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4)
    page = torch.randn(2, 4, 8)
    meta = cache.write_kv_page(0, 0, page, page)
    assert meta["codec"] == "fp8_e4m3"


def test_nocturnal_coding_multi_krum_and_dag_validation() -> None:
    with pytest.raises(ValueError, match="must be a non-empty 1D numpy ndarray"):
        multi_krum_select([np.zeros((2, 2))], num_byzantine=0, num_select=1)
    with pytest.raises(ValueError, match="does not match"):
        multi_krum_select(
            [np.ones(4, dtype=np.float32), np.ones(6, dtype=np.float32)],
            num_byzantine=0,
            num_select=1,
        )
    idx = multi_krum_select(
        [np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32) * 1.1, np.ones(4, dtype=np.float32) * 50.0],
        num_byzantine=1,
        num_select=2,
    )
    assert set(idx) == {0, 1}

    with pytest.raises(ValueError, match="target_languages must be a sequence"):
        decompose_engineering_epic_to_dag("Valid Epic", target_languages="Python")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_languages must contain non-empty printable strings"):
        decompose_engineering_epic_to_dag("Valid Epic", target_languages=("Python", ""))
    dag = decompose_engineering_epic_to_dag("Valid Epic", target_languages=("Rust", "Python"))
    assert len(dag) == 8


def test_reddit_vision_vlm_moe_validation() -> None:
    with pytest.raises(ValueError, match="packet must be a SparseLoCoDCT2BitPacket"):
        dequantize_sparseloco_dct_2bit("not-a-packet")  # type: ignore[arg-type]

    mat = np.ones((8, 8), dtype=np.float32) * 0.25
    pkt = quantize_sparseloco_dct_2bit(mat, bee_id="bee_1", cluster_id="c_1", backbone_id="m_1")
    recon = dequantize_sparseloco_dct_2bit(pkt)
    assert recon.shape == (8, 8)

    adapter = QLoRAVisionAdapter(hidden_dim=16, rank=4, alpha=8.0)
    with pytest.raises(ValueError, match="delta_matrix must be a finite"):
        adapter.load_effective_delta(np.zeros((8, 8), dtype=np.float32))
    with pytest.raises(ValueError, match="delta_matrix must be a finite"):
        bad = np.zeros((16, 16), dtype=np.float32)
        bad[0, 0] = np.nan
        adapter.load_effective_delta(bad)

    with pytest.raises(ValueError, match="tensor must be a non-empty torch.Tensor"):
        quantize_adamw_state_8bit(torch.empty(0))
    with pytest.raises(ValueError, match="tensor must contain only finite values"):
        quantize_adamw_state_8bit(torch.tensor([1.0, float("inf")]))

    with pytest.raises(ValueError, match="attention_state_dicts must contain at least one"):
        merge_btm_attention_weights("bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keys do not match branch 0"):
        merge_btm_attention_weights([{"a": torch.ones(4)}, {"b": torch.ones(4)}])
    with pytest.raises(ValueError, match="shape"):
        merge_btm_attention_weights([{"a": torch.ones(4)}, {"a": torch.ones(6)}])
    merged = merge_btm_attention_weights([{"a": torch.ones(4)}, {"a": torch.ones(4) * 3.0}])
    assert torch.allclose(merged["a"], torch.ones(4) * 2.0)
