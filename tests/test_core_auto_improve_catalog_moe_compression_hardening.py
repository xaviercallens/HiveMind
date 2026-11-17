from __future__ import annotations

import pytest
import torch

from hivemind.core.auto_improvement import NocturnalAutoImprovementScheduler, PointsRetrofitManager
from hivemind.core.base_model_catalog import (
    DEFAULT_PROMOTION_POLICY,
    ModelGeometry,
    ServeConfig,
    TrainRecipe,
    estimate_gguf_serving,
    estimate_qlora_training,
    promotion_gate,
)
from hivemind.core.btm_moe import (
    HierarchicalMoERouter,
    SEASON_3_DOMAIN_EXPERT_SPECS,
    ThreeTierHierarchicalMoERouter,
    keyword_route,
    project_lora_rank_svd,
)
from hivemind.core.compression import _pack_2bit_codes, _unpack_2bit_codes, decompress_tensor_topk_fp8


def test_auto_improvement_verify_and_retrofit_validation() -> None:
    orch = NocturnalAutoImprovementScheduler()
    wu = orch.create_domain_workunit("maths_formal", num_samples=2, inner_steps=4, seed=42)

    with pytest.raises(ValueError, match="observed_telemetry must be a Mapping"):
        orch.verify_workunit_execution(wu, "worker-1", "acct-1", observed_telemetry="bad")  # type: ignore[arg-type]

    res = orch.verify_workunit_execution(wu, "worker-1", "acct-1")
    mgr = PointsRetrofitManager()
    assert mgr.calculate_work_reward(res) == 0.0

    with pytest.raises(ValueError, match="result must be an AutoImprovementExecutionResult"):
        mgr.calculate_work_reward("not-a-result")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="results must be a Sequence"):
        mgr.retrofit_work_points(object(), "bad-seq")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="results must contain only AutoImprovementExecutionResult"):
        mgr.retrofit_work_points(object(), ["bad-item"])  # type: ignore[list-item]


def test_base_model_catalog_estimates_and_promotion_gate_validation() -> None:
    geom = ModelGeometry(
        model_id="Qwen/Qwen2.5-Coder-7B-Instruct",
        params_text_linear=7_000_000_000,
        params_embedding=500_000_000,
        params_text_other=100_000_000,
        params_vision=0,
        params_mtp=0,
        lora_in_plus_out_all_linear=1_000_000,
        lora_in_plus_out_mlp=600_000,
        max_linear_elements=50_000_000,
        hidden=3584,
        layers=28,
        intermediate=18944,
        vocab=152064,
        n_kv_heads=4,
        head_dim=128,
        n_full_attention_layers=28,
        n_linear_attention_layers=0,
        linear_num_key_heads=0,
        linear_num_value_heads=0,
        linear_key_head_dim=0,
        linear_value_head_dim=0,
        linear_conv_kernel_dim=0,
    )
    recipe = TrainRecipe()
    serve = ServeConfig()

    est_t = estimate_qlora_training(geom, recipe)
    assert est_t.total_mib > 0.0
    est_s = estimate_gguf_serving(geom, serve)
    assert est_s.total_mib > 0.0

    with pytest.raises(ValueError, match="g must be a ModelGeometry"):
        estimate_qlora_training("not-geom", recipe)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="r must be a TrainRecipe"):
        estimate_qlora_training(geom, "not-recipe")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="g must be a ModelGeometry"):
        estimate_gguf_serving("not-geom", serve)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="s must be a ServeConfig"):
        estimate_gguf_serving(geom, "not-serve")  # type: ignore[arg-type]

    cand = {"i1": True, "i2": False}
    inc = {"i1": False, "i2": False}
    with pytest.raises(ValueError, match="candidate must be a Mapping"):
        promotion_gate("bad", inc, cand, inc)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="policy must be a PromotionPolicy"):
        promotion_gate(cand, inc, cand, inc, policy="bad")  # type: ignore[arg-type]
    receipt = promotion_gate(cand, inc, cand, inc, policy=DEFAULT_PROMOTION_POLICY)
    assert receipt.decision == "INSUFFICIENT"


def test_btm_moe_keyword_route_svd_and_hierarchical_routers_validation() -> None:
    domains, weights = keyword_route("Prove Lean theorem with omega tactic", [SEASON_3_DOMAIN_EXPERT_SPECS])
    assert len(domains) >= 1 and sum(weights.values()) > 0.0

    with pytest.raises(ValueError, match="prompt must be a string"):
        keyword_route(123, [SEASON_3_DOMAIN_EXPERT_SPECS])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spec_tables must be a Sequence"):
        keyword_route("test", "not-a-sequence")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="lora_state_dict must be a dict"):
        project_lora_rank_svd("not-a-dict", target_rank_r=8)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="target_rank_r must be positive"):
        project_lora_rank_svd({}, target_rank_r=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_macro_domains and experts_per_domain"):
        HierarchicalMoERouter(hidden_size=16, num_macro_domains=True, experts_per_domain=4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="every macro cluster needs at least one expert"):
        HierarchicalMoERouter(hidden_size=16, cluster_sizes=[4, True])  # type: ignore[list-item]

    with pytest.raises(ValueError, match="num_super_clusters, macro_per_super and micro_per_macro"):
        ThreeTierHierarchicalMoERouter(hidden_size=16, num_super_clusters=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="micro_cluster_sizes"):
        ThreeTierHierarchicalMoERouter(
            hidden_size=16, num_super_clusters=2, macro_per_super=2, micro_cluster_sizes=[4, 4, 4, True]  # type: ignore[list-item]
        )


def test_compression_decompress_and_unpack_2bit_validation() -> None:
    with pytest.raises(ValueError, match="payload must be a CompressedTensorPayload"):
        decompress_tensor_topk_fp8("not-a-payload")  # type: ignore[arg-type]

    codes = torch.tensor([0, 1, 2, 3, 1, 2], dtype=torch.uint8)
    packed = _pack_2bit_codes(codes)
    unpacked = _unpack_2bit_codes(packed, 6)
    assert torch.equal(unpacked, codes.to(torch.long))

    with pytest.raises(ValueError, match="packed must be a torch.Tensor"):
        _unpack_2bit_codes([1, 2], 4)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="count must be a non-negative integer"):
        _unpack_2bit_codes(packed, True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="count exceeds capacity"):
        _unpack_2bit_codes(packed, 99)
