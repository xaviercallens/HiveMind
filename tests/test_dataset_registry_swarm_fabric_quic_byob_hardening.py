# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control boundary hardening tests for Iteration 137 (`dataset_registry`, `swarm_fabric`, `quic_transport`, `byob`)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from hivemind.data import dataset_registry as dr
from hivemind.swarm import byob
from hivemind.swarm import quic_transport as qt
from hivemind.swarm import swarm_fabric as sf


def test_dataset_registry_boundary_validation() -> None:
    for bad_path in ("", "  ", 123, None):
        with pytest.raises(ValueError, match="path must be a non-empty Path or string"):
            dr.load_candidates(bad_path)  # type: ignore[arg-type]

    for bad_url in ("", "  ", 123):
        with pytest.raises(ValueError, match="url must be a non-empty string"):
            dr.default_fetch(bad_url)  # type: ignore[arg-type]
    for bad_to in (0.0, -5.0, True, math.nan, math.inf):
        with pytest.raises(ValueError, match="timeout_s must be a finite positive number"):
            dr.default_fetch("https://huggingface.co", timeout_s=bad_to)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="exc must be a BaseException"):
        dr._err("not-an-exception")  # type: ignore[arg-type]
    assert "ValueError" in dr._err(ValueError("bad"))

    spec = dr.SourceSpec(
        dataset_id="openai/gsm8k",
        domain="math",
        config="main",
        split="train",
        mapping={"kind": "fields", "prompt_fields": ["question"], "completion_field": "answer"},
        verified_signal=True,
        signal_basis="benchmark",
    )
    with pytest.raises(TypeError, match="spec must be a SourceSpec"):
        dr.registry_entry("not-spec", lambda u: b"{}", "2026-01-01T00:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fetch must be callable"):
        dr.registry_entry(spec, "not-callable", "2026-01-01T00:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="retrieved_utc must be a non-empty string"):
        dr.registry_entry(spec, lambda u: b"{}", "  ")

    with pytest.raises(TypeError, match="spec must be a SourceSpec"):
        dr.split_size("not-spec", lambda u: b"{}")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fetch must be callable"):
        dr.split_size(spec, 123)  # type: ignore[arg-type]

    for bad_paths in ("not-seq", [], [""], [123]):
        with pytest.raises(ValueError, match="paths must be a non-empty sequence of Paths"):
            dr.read_jsonl_shards(bad_paths)  # type: ignore[arg-type]

    for bad_v in (True, False, math.nan, math.inf, "1.0"):
        with pytest.raises(ValueError, match="v must be a finite number"):
            dr.fmt_number(bad_v)  # type: ignore[arg-type]
    assert dr.fmt_number(0.0) == "0"


def test_swarm_fabric_router_prefetch_and_batch_digest_validation() -> None:
    ring = sf.RegionalGeoRingTransport(region="EU-West-Paris", max_rtt_ms=20.0)
    with pytest.raises(TypeError, match="peer must be a GeoRingPeer"):
        ring.register_peer("not-a-peer")  # type: ignore[arg-type]

    router = sf.TwoLevelMacroMicroRouter(
        hidden_size=8,
        macro_clusters={"code": ["py", "rust"], "math": ["lean", "alg"]},
        top_k_micro=1,
    )
    for bad_hs in (
        "not-tensor",
        torch.zeros((8,)),
        torch.zeros((0, 8)),
        torch.zeros((2, 6)),
        torch.tensor([[1.0, math.nan, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
    ):
        with pytest.raises(ValueError, match="hidden_states must be a finite 2D or 3D tensor"):
            router(bad_hs)  # type: ignore[arg-type]
    out = router(torch.randn(2, 8))
    assert out["selected_macro_cluster"] in ("code", "math")

    streamer = sf.PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    with pytest.raises(TypeError, match="router must be a TwoLevelMacroMicroRouter"):
        streamer.prefetch_lookahead_layer("not-router", torch.randn(2, 8))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="hidden_states must be a torch.Tensor"):
        streamer.prefetch_lookahead_layer(router, "not-tensor")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layer_idx must be a non-negative integer"):
        streamer.prefetch_lookahead_layer(router, torch.randn(2, 8), layer_idx=-1)
    with pytest.raises(ValueError, match="simulated_attention_compute_ms must be a finite non-negative number"):
        streamer.prefetch_lookahead_layer(router, torch.randn(2, 8), simulated_attention_compute_ms=-1.0)

    valid_hex = "a" * 64
    for bad_step_layer in ((-1, 0), (0, -1), (True, 0), (0, False)):
        with pytest.raises(ValueError, match="step and layer must be non-negative integers"):
            sf._batch_leaf_digest(bad_step_layer[0], bad_step_layer[1], valid_hex)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="layer_leaf_hex must be a 64-character hex string"):
        sf._batch_leaf_digest(0, 0, "abc")
    leaf = sf._batch_leaf_digest(0, 0, valid_hex)
    assert len(leaf) == 32

    with pytest.raises(ValueError, match="left and right must be 32-byte digests"):
        sf._batch_node_digest(b"short", leaf)
    node = sf._batch_node_digest(leaf, leaf)
    assert len(node) == 32

    with pytest.raises(ValueError, match="peer_id must be a non-empty string"):
        sf._batch_root_hex("", 1, 1, node)
    for bad_sl in ((0, 1), (1, 0), (True, 1), (1, False)):
        with pytest.raises(ValueError, match="num_steps and num_layers must be positive integers"):
            sf._batch_root_hex("peer-1", bad_sl[0], bad_sl[1], node)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="top must be a 32-byte digest"):
        sf._batch_root_hex("peer-1", 1, 1, b"short")
    assert len(sf._batch_root_hex("peer-1", 1, 1, node)) == 64

    for bad_lc in (0, -1, True):
        with pytest.raises(ValueError, match="leaf_count must be a positive integer"):
            sf._batch_level_sizes(bad_lc)  # type: ignore[arg-type]
    assert sf._batch_level_sizes(5) == [5, 3, 2, 1]


def test_quic_transport_super_ring_and_congestion_window_validation() -> None:
    with pytest.raises(TypeError, match="region must be a string"):
        qt.classify_continent(123)  # type: ignore[arg-type]
    assert qt.classify_continent("EU-West-Paris") == "EU"

    sr = qt.ContinentalSuperRingRouter()
    with pytest.raises(TypeError, match="node must be a SuperRingPeerNode"):
        sr.register_peer("not-a-node")  # type: ignore[arg-type]

    for bad_pair in (("", "US"), ("EU", "  "), (123, "US")):
        with pytest.raises(ValueError, match="source_continent and target_continent must be non-empty strings"):
            sr.inter_continent_rtt_ms(bad_pair[0], bad_pair[1])  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="source_peer_id and target_peer_id must be non-empty strings"):
            sr.route(bad_pair[0], bad_pair[1])  # type: ignore[arg-type]

    cc = qt.AdaptiveCongestionController()
    for bad_rtt in (0.0, -5.0, True, math.nan):
        with pytest.raises(ValueError, match="rtt_ms must be a finite positive number"):
            cc.compute_target_window(rtt_ms=bad_rtt)  # type: ignore[arg-type]
    for bad_jit in (-0.1, False, math.inf):
        with pytest.raises(ValueError, match="jitter_ms must be a finite non-negative number"):
            cc.compute_target_window(jitter_ms=bad_jit)  # type: ignore[arg-type]
    for bad_loss in (-0.1, 1.1, True, math.nan):
        with pytest.raises(ValueError, match="loss_ratio must be in"):
            cc.compute_target_window(loss_ratio=bad_loss)  # type: ignore[arg-type]
    w = cc.compute_target_window(rtt_ms=15.0, jitter_ms=1.0, loss_ratio=0.01)
    assert w.cwnd_packets >= cc.min_cwnd_packets


def test_byob_manifest_collision_and_tree_validation() -> None:
    with pytest.raises(ValueError, match="field_name and rule must be non-empty strings"):
        byob._bad("", "v", "rule")
    with pytest.raises(ValueError, match="field_name and rule must be non-empty strings"):
        byob._bad("field", "v", "  ")

    with pytest.raises(TypeError, match="manifest must be a BeeManifest"):
        byob.check_manifest_fields("not-manifest")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="name must be a string"):
        byob.normalised_name_key(123)  # type: ignore[arg-type]
    assert byob.normalised_name_key("bee-coder-v1") == "coderv1"

    with pytest.raises(ValueError, match="bee_name must be a non-empty string"):
        byob.check_name_collision("", ["bee-other"])
    with pytest.raises(TypeError, match="taken must be an iterable of strings"):
        byob.check_name_collision("bee-coder", "bee-coder")

    with pytest.raises(TypeError, match="base_model must be a string"):
        byob.served_base(123)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="path must be a string"):
        byob._extension(123)  # type: ignore[arg-type]
    assert byob._extension("weights/model.safetensors") == ".safetensors"

    for bad_entries in ("not-seq", b"bytes", ["not-mapping"]):
        with pytest.raises(TypeError, match="entries must be a sequence of mappings"):
            byob.parse_tree(bad_entries)  # type: ignore[arg-type]
    assert byob.parse_tree([{"type": "file", "path": "adapter_model.safetensors", "size": 128}]) == [
        byob.RepoFile(path="adapter_model.safetensors", size=128, sha256="")
    ]
