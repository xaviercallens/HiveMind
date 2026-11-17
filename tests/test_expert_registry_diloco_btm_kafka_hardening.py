# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and positive-path regression suite for Iteration 141 hardening:
- `hivemind.core.expert_registry`
- `hivemind.swarm.diloco_nextgen`
- `hivemind.swarm.amcp_kafka_transport`
- `hivemind.core.btm_moe`
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from hivemind.core.btm_moe import (
    FP8PagedKVCache,
    ProjectedLoRAStateDict,
    Season3DomainExpertRegistry,
    ThreeTierHierarchicalMoERouter,
    UpcycledSparseMoEModel,
    _footprint_report,
    _install_router_on_blocks,
    _stack_ragged_gates,
    initialize_domain_branches,
    upcycle_branches_to_sparse_moe,
)
from hivemind.core.expert_registry import (
    CANDIDATE,
    LIVE,
    LORA_DOMAIN,
    SERVING,
    SHADOW,
    AdapterFootprint,
    ExpertRecord,
    ExpertRegistry,
    ServingNode,
    _boundary_pattern,
    _greedy_place,
    keyword_hits,
    replay_history,
)
from hivemind.swarm.amcp_kafka_transport import (
    AMCPConsumer,
    AMCPKafkaEvent,
    AMCPPublisher,
    EnvelopeVerifier,
    HMACKeyring,
    InMemoryTransport,
    KAFKA_TOPICS,
    KafkaBrokerTransport,
    KafkaSettings,
    TransportRecord,
    kafka_transport_readiness,
)
from hivemind.swarm.diloco_nextgen import (
    ChunkedOrthonormalDCTTransform,
    ComputeVisibleSparseLoCoCompressor,
    CPUOffloadedFlatBucketOuterState,
    FactoredGossipMix1Mix2Coordinator,
    MuonNewtonSchulzLoRAPreconditioner,
    SkipAttentionSpeculativeRouter,
)


# ---------------------------------------------------------------------------
# 1. Expert Registry Hardening
# ---------------------------------------------------------------------------


def test_expert_record_from_dict_and_replay_history_validation() -> None:
    with pytest.raises(ValueError, match="data must be a Mapping"):
        ExpertRecord.from_dict("not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="history must be an iterable"):
        replay_history("bad-history", CANDIDATE)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="final_state must be one of"):
        replay_history([], "unknown_state")

    replay_history([(CANDIDATE, SHADOW, "ok")], SHADOW)


def test_expert_registry_methods_reject_malformed_arguments(tmp_path) -> None:
    reg = ExpertRegistry()
    with pytest.raises(ValueError, match="rec must be an ExpertRecord"):
        reg._index("not-a-record")  # type: ignore[arg-type]

    sha = "a" * 64
    rec = reg.register(
        ExpertRecord(
            expert_id="exp-1",
            kind=LORA_DOMAIN,
            cluster_id="code",
            base_model_id="qwen-1.5b",
            adapter_sha256=sha,
            adapter_size_mib=16,
            keywords=("python", "pytest"),
        )
    )
    assert rec.state == CANDIDATE
    reg.transition("exp-1", SHADOW, "validated")

    with pytest.raises(ValueError, match="expert_id must be a non-empty string"):
        reg.promote_with_gate("", {})
    with pytest.raises(ValueError, match="gate must be a Mapping"):
        reg.promote_with_gate("exp-1", "not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="cluster_id must be a non-empty string"):
        reg.experts_in_cluster("")
    with pytest.raises(ValueError, match="unknown lifecycle state"):
        reg.experts_in_cluster("code", state="invalid")
    assert len(reg.experts_in_cluster("code", state=SHADOW)) == 1

    with pytest.raises(ValueError, match="states must be a Collection"):
        reg.residency_inputs("live")  # type: ignore[arg-type]
    assert "exp-1" in reg.residency_inputs((SHADOW,))

    with pytest.raises(ValueError, match="path must be a non-empty Path or string"):
        reg.save("   ")
    saved = reg.save(tmp_path / "registry.json")
    assert saved.exists()

    with pytest.raises(ValueError, match="data must be a Mapping"):
        ExpertRegistry.from_dict(["bad"])  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="base_model_id must be a non-empty string"):
        reg.import_serving_experts("")
    with pytest.raises(ValueError, match="cluster_id must be a non-empty string"):
        reg.import_serving_experts("qwen-1.5b", cluster_id="  ")


def test_greedy_place_and_keyword_helpers_validation() -> None:
    node = ServingNode("n1", 8192, "qwen-1.5b", 4096, 4)
    fp = AdapterFootprint(16, "qwen-1.5b")
    with pytest.raises(ValueError, match="order must be a Sequence"):
        _greedy_place("e1", [node], {"e1": fp}, {"e1": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nodes must be a Sequence"):
        _greedy_place(["e1"], "n1", {"e1": fp}, {"e1": 1.0})  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="adapters and demand must be Mappings"):
        _greedy_place(["e1"], [node], [], {"e1": 1.0})  # type: ignore[arg-type]

    hot, cold, _, _ = _greedy_place(["e1"], [node], {"e1": fp}, {"e1": 1.0})
    assert hot == {"e1": "n1"}
    assert cold == {}

    with pytest.raises(ValueError, match="kw must be a non-empty string"):
        _boundary_pattern("   ")
    with pytest.raises(ValueError, match="prompt_lower must be a string"):
        keyword_hits(123, ["rust"])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="keywords must be an iterable"):
        keyword_hits("write rust", "rust")  # type: ignore[arg-type]
    assert "rust" in keyword_hits("write rust code", ["rust", "c++"])


# ---------------------------------------------------------------------------
# 2. DiLoCo NextGen Hardening
# ---------------------------------------------------------------------------


def test_sparse_loco_compressor_and_dct_validation() -> None:
    comp = ComputeVisibleSparseLoCoCompressor(codec="E8")
    vec = torch.randn(16)
    with pytest.raises(ValueError, match="kind must be 'E8' or 'Z'"):
        comp._resolve_round_scale("A2", vec)
    with pytest.raises(ValueError, match="flat_vec must be a non-empty torch.Tensor"):
        comp._resolve_round_scale("E8", torch.empty(0))

    with pytest.raises(ValueError, match="delta must be a non-empty torch.Tensor"):
        comp.compress_and_reconstruct(torch.empty(0))
    with pytest.raises(ValueError, match="reference_param must be a torch.Tensor"):
        comp.compress_and_reconstruct(vec, reference_param=torch.randn(8))
    with pytest.raises(ValueError, match="fragment_name must be a non-empty string"):
        comp.compress_and_reconstruct(vec, fragment_name="  ")

    recon, meta = comp.compress_and_reconstruct(vec, fragment_name="lora_a")
    assert recon.shape == vec.shape
    assert meta["codec"] == "E8"

    with pytest.raises(ValueError, match="n must be an integer >= 2"):
        ChunkedOrthonormalDCTTransform._build_orthonormal_dct_matrix(1)
    with pytest.raises(ValueError, match="n must be an integer >= 2"):
        ChunkedOrthonormalDCTTransform._build_orthonormal_dct_matrix(True)
    c_mat = ChunkedOrthonormalDCTTransform._build_orthonormal_dct_matrix(4)
    assert torch.allclose(c_mat @ c_mat.t(), torch.eye(4), atol=1e-5)


def test_flat_bucket_skip_router_muon_and_gossip_validation() -> None:
    state = CPUOffloadedFlatBucketOuterState({"w": torch.ones(4, 4)})
    with pytest.raises(ValueError, match="inner_params must match"):
        state.compute_pseudo_gradients({})
    with pytest.raises(ValueError, match="must be a tensor of shape"):
        state.compute_pseudo_gradients({"w": torch.ones(2, 2)})
    pg = state.compute_pseudo_gradients({"w": torch.zeros(4, 4)})
    assert torch.allclose(pg["w"], torch.ones(4, 4))

    with pytest.raises(ValueError, match="updated_master_params must match"):
        state.apply_outer_update_and_sync_inner({})
    with pytest.raises(ValueError, match="must be a tensor of shape"):
        state.apply_outer_update_and_sync_inner({"w": torch.ones(2, 2)})
    synced = state.apply_outer_update_and_sync_inner({"w": torch.full((4, 4), 2.0)})
    assert torch.allclose(synced["w"], torch.full((4, 4), 2.0))

    router = SkipAttentionSpeculativeRouter(num_experts=8, top_k=2)
    x = torch.randn(3, 8)
    nw = torch.ones(8)
    rw = torch.randn(8, 8)
    with pytest.raises(ValueError, match="x and norm_weight must be torch.Tensors"):
        router._rms_norm("bad", nw)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="match on their last dimension"):
        router._rms_norm(x, torch.ones(4))
    with pytest.raises(ValueError, match="next_layer_router_weight must be a 2D"):
        router.predict_next_layer_experts(x, nw, torch.ones(8))
    with pytest.raises(ValueError, match="post_attn_residual must be a torch.Tensor matching"):
        router.route_with_cache_aware_fallback(x, torch.randn(2, 8), nw, rw, [0, 1])
    with pytest.raises(ValueError, match="baseline_cached_experts must be a Sequence"):
        router.route_with_cache_aware_fallback(x, x, nw, rw, "01")  # type: ignore[arg-type]
    res = router.route_with_cache_aware_fallback(x, x, nw, rw, [0, 1])
    assert res["routed_experts"].shape == (3, 2)

    pre = MuonNewtonSchulzLoRAPreconditioner(steps=3)
    with pytest.raises(ValueError, match="g must be a non-empty 2D torch.Tensor"):
        pre._orthogonalize_float32(torch.ones(4))
    orth = pre._orthogonalize_float32(torch.randn(4, 4))
    assert orth.shape == (4, 4)

    coord = FactoredGossipMix1Mix2Coordinator()
    with pytest.raises(ValueError, match="fragment_name must be a non-empty string"):
        coord.is_sensitive_fragment("   ")
    assert coord.is_sensitive_fragment("layers.0.router.weight") is True


# ---------------------------------------------------------------------------
# 3. AMCP Kafka Transport Hardening
# ---------------------------------------------------------------------------


def _make_fake_kafka_module():
    class FakeKafkaException(Exception):
        pass

    class FakeProducer:
        def __init__(self, cfg):
            self.cfg = cfg

        def produce(self, topic, key=None, value=None, on_delivery=None):
            if on_delivery:
                msg = SimpleNamespace(
                    topic=lambda: topic, partition=lambda: 0, offset=lambda: 1
                )
                on_delivery(None, msg)

        def flush(self, timeout):
            return 0

    class FakeConsumer:
        def __init__(self, cfg):
            self.cfg = cfg
            self.subscribed = []

        def subscribe(self, topics):
            self.subscribed = list(topics)

        def consume(self, num_messages=1, timeout=1.0):
            return []

        def commit(self, offsets=None, asynchronous=False):
            return None

        def seek(self, tp):
            return None

        def close(self):
            return None

    return SimpleNamespace(
        KafkaException=FakeKafkaException,
        KafkaError=SimpleNamespace(_PARTITION_EOF=-191),
        Producer=FakeProducer,
        Consumer=FakeConsumer,
        TopicPartition=lambda t, p, o: (t, p, o),
    )


def test_amcp_kafka_transport_methods_validation() -> None:
    with pytest.raises(ValueError, match="env must be a Mapping or None"):
        kafka_transport_readiness("invalid")  # type: ignore[arg-type]

    settings = KafkaSettings(bootstrap="localhost:9092", security_protocol="PLAINTEXT")
    broker = KafkaBrokerTransport(settings, client_module=_make_fake_kafka_module())

    with pytest.raises(ValueError, match="topic must be a non-empty string"):
        broker.publish("  ", b"k", b"v")
    with pytest.raises(TypeError, match="key and value must be bytes"):
        broker.publish("hivemind.amcp.v1.human_agent", "k", b"v")  # type: ignore[arg-type]
    ack = broker.publish("hivemind.amcp.v1.human_agent", b"k", b"v")
    assert ack["offset"] == 1

    with pytest.raises(ValueError, match="topics must be an iterable"):
        broker.subscribe("hivemind.amcp.v1.human_agent")
    with pytest.raises(ValueError, match="non-empty iterable"):
        broker.subscribe([])
    broker.subscribe(["hivemind.amcp.v1.human_agent"])

    with pytest.raises(ValueError, match="max_records must be an integer >= 1"):
        broker.poll(0, 0.1)
    with pytest.raises(ValueError, match="timeout_s must be a finite number >= 0"):
        broker.poll(1, -0.1)
    assert broker.poll(1, 0.0) == []

    with pytest.raises(TypeError, match="record must be a TransportRecord"):
        broker.commit("bad")  # type: ignore[arg-type]
    rec = TransportRecord("hivemind.amcp.v1.human_agent", 0, 0, b"k", b"{}")
    broker.commit(rec)

    with pytest.raises(ValueError, match="topic must be a non-empty string"):
        broker.seek("", 0, 0)
    with pytest.raises(ValueError, match="partition must be an integer >= 0"):
        broker.seek("hivemind.amcp.v1.human_agent", -1, 0)
    with pytest.raises(ValueError, match="offset must be an integer >= 0"):
        broker.seek("hivemind.amcp.v1.human_agent", 0, True)
    broker.seek("hivemind.amcp.v1.human_agent", 0, 0)


def test_amcp_publisher_and_consumer_internal_validation() -> None:
    keyring = HMACKeyring(keys={"k1": b"0" * 32}, active_kid="k1")
    mem = InMemoryTransport()
    pub = AMCPPublisher(mem, keyring)
    with pytest.raises(TypeError, match="event must be an AMCPKafkaEvent"):
        pub.publish("bad")  # type: ignore[arg-type]

    ev = AMCPKafkaEvent(
        event_id="e1",
        topic=KAFKA_TOPICS["human_agent"],
        event_type="io.hivemind.amcp.human.ping",
        sender_id="s1",
        sender_kind="human_researcher",
        recipient_id="r1",
        subject_id="sub1",
        correlation_id="c1",
        payload={"msg": "hi"},
        utc=1_700_000_000.0,
    )
    res = pub.publish(ev)
    assert res["event_id"] == "e1"

    verifier = EnvelopeVerifier(keyring, clock=lambda: 1_700_000_000.0)
    consumer = AMCPConsumer(mem, verifier, clock=lambda: 1_700_000_000.0)
    rec = TransportRecord(KAFKA_TOPICS["human_agent"], 0, 0, b"k", b"{}")

    with pytest.raises(TypeError, match="record must be a TransportRecord"):
        consumer._dead_letter("bad", "malformed", "detail")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        consumer._dead_letter(rec, "  ", "detail")
    with pytest.raises(TypeError, match="record must be a TransportRecord"):
        consumer._commit("bad")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="record must be a TransportRecord"):
        consumer._rewind("bad", set())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="held must be a set"):
        consumer._rewind(rec, [])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 4. BTM MoE Hardening
# ---------------------------------------------------------------------------


def test_btm_moe_projected_dict_registry_and_router_validation() -> None:
    with pytest.raises(ValueError, match="tensors and meta must be Mappings"):
        ProjectedLoRAStateDict("bad", {})  # type: ignore[arg-type]
    psd = ProjectedLoRAStateDict({"w": torch.ones(2, 2)}, {"target_rank": 4})
    with pytest.raises(KeyError):
        _ = psd[""]
    assert psd["target_rank"] == 4

    s3 = Season3DomainExpertRegistry()
    with pytest.raises(ValueError, match="moe_model must be an UpcycledSparseMoEModel"):
        s3.hot_plug_all_into_moe("not-a-model")  # type: ignore[arg-type]

    moe = UpcycledSparseMoEModel(upcycled_backbone=nn.Linear(8, 8), expert_domains=["code", "math"])
    assert isinstance(moe, UpcycledSparseMoEModel)
    with pytest.raises(ValueError, match="target_rank_r must be a positive integer"):
        s3.hot_plug_all_into_moe(moe, target_rank_r=0)

    with pytest.raises(ValueError, match="gates must be a non-empty nn.ModuleList"):
        _stack_ragged_gates(nn.ModuleList(), 8)
    with pytest.raises(ValueError, match="hidden_size must be a positive integer"):
        _stack_ragged_gates(nn.ModuleList([nn.Linear(8, 2, bias=False)]), 0)

    router = ThreeTierHierarchicalMoERouter(
        hidden_size=8, num_super_clusters=2, macro_per_super=2, micro_per_macro=2, top_k=2
    )
    with pytest.raises(ValueError, match="hs_fp32 must be a 2D torch.Tensor"):
        router._route_path(torch.randn(4))
    with pytest.raises(ValueError, match="hidden_states must be a non-empty torch.Tensor"):
        router.predict_candidate_experts(torch.randn(2, 4))
    with pytest.raises(ValueError, match="top_n must be a positive integer"):
        router.predict_candidate_experts(torch.randn(2, 8), top_n=0)
    idx, probs = router.predict_candidate_experts(torch.randn(2, 8), top_n=2)
    assert idx.shape == (2, 2)
    assert probs.shape == (2, 2)

    with pytest.raises(ValueError, match="moe_model must be an UpcycledSparseMoEModel"):
        _install_router_on_blocks("bad", lambda: router)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="vram_ceiling_gb must be a positive finite float"):
        _footprint_report(moe, 0.0)


def test_fp8_paged_kv_cache_internal_and_benchmark_validation() -> None:
    with pytest.raises(ValueError, match="tensor must be a non-empty torch.Tensor"):
        FP8PagedKVCache._quantize_page_fp8(torch.empty(0))
    with pytest.raises(ValueError, match="q and scale must be torch.Tensors"):
        FP8PagedKVCache._dequantize_page_fp8("bad", torch.ones(1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="codec must be one of"):
        FP8PagedKVCache._encode_decode_with_codec(torch.randn(8), "bad_codec")
    with pytest.raises(ValueError, match="target_bits_per_dim must be > 0"):
        FP8PagedKVCache._encode_decode_with_codec(torch.randn(8), "fp8_e4m3", target_bits_per_dim=0.0)

    cache = FP8PagedKVCache(num_layers=2, num_kv_heads=2, head_dim=8, page_size_tokens=4)
    k = torch.randn(2, 4, 8)
    v = torch.randn(2, 4, 8)
    cache.write_kv_page(0, 0, k, v)
    with pytest.raises(ValueError, match="layer_idx must be an integer >= 0"):
        cache.read_kv_page(-1, 0)
    with pytest.raises(ValueError, match="page_idx must be an integer >= 0"):
        cache.read_kv_page(0, True)
    rk, rv = cache.read_kv_page(0, 0)
    assert rk.shape == k.shape and rv.shape == v.shape

    with pytest.raises(ValueError, match="k_tensor and v_tensor must be non-empty torch.Tensors"):
        FP8PagedKVCache.benchmark_kv_page_codecs(torch.empty(0), v)
    with pytest.raises(ValueError, match="target_bits_per_dim must be > 0"):
        FP8PagedKVCache.benchmark_kv_page_codecs(k, v, target_bits_per_dim=-1.0)
