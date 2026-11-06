"""Invalid optimizer configurations and analytic small-matrix merger cases."""

import pytest
import torch

from hivemind.swarm import diloco_nextgen as d


@pytest.mark.parametrize("factory,kwargs,message", [
    (d.CosineGatedAdamDecayOuterOptimizer, {"weight_decay": float("nan")}, "finite"),
    (d.ReputationWeightedIsoCTSMerger, {"common_space_fraction": 0.09}, "common_space_fraction"),
    (d.ReputationWeightedIsoCTSMerger, {"trim_density": 0}, "trim_density"),
    (d.HeLoCoDirectionCorrector, {"conflict_damping": 1.1}, "conflict_damping"),
    (d.AutoLoCoIntervalCorrector, {"reference_inner_steps_h": 0}, "reference_inner_steps_h"),
    (d.AutoLoCoIntervalCorrector, {"reference_inner_lr": 0}, "reference_inner_lr"),
    (d.FoMoEPartialReplicaPlanner, {"total_experts": 1}, "total_experts"),
    (d.FoMoEPartialReplicaPlanner, {"active_top_k": 0}, "active_top_k"),
    (d.ComputeVisibleSparseLoCoCompressor, {"topk_ratio": 0}, "topk_ratio"),
    (d.ComputeVisibleSparseLoCoCompressor, {"n_bins": 1}, "n_bins"),
    (d.ComputeVisibleSparseLoCoCompressor, {"n_bins": 257}, "n_bins"),
    (d.ChunkedOrthonormalDCTTransform, {"chunk_size": 1}, "chunk_size"),
    (d.CPUOffloadedFlatBucketOuterState, {"params": {}}, "empty"),
    (d.SkipAttentionSpeculativeRouter, {"num_experts": 1}, "Require"),
    (d.SkipAttentionSpeculativeRouter, {"car_threshold": -0.1}, "car_threshold"),
    (d.MuonNewtonSchulzLoRAPreconditioner, {"steps": 0}, "steps"),
])
def test_invalid_optimizer_configuration_is_rejected(factory, kwargs, message):
    with pytest.raises(ValueError, match=message):
        factory(**kwargs)


@pytest.mark.parametrize("matrix", [torch.tensor([1., 2.]), torch.empty(0, 2), torch.zeros(2, 2)])
def test_isoc_degenerate_input_returns_an_independent_unchanged_tensor(matrix):
    result, before, after = d.ReputationWeightedIsoCTSMerger.apply_isoc_2d(matrix)
    assert result is not matrix
    torch.testing.assert_close(result, matrix)
    assert (before, after) == (1., 1.)


def test_isoc_equalizes_known_singular_values_and_preserves_input_dtype():
    original = torch.diag(torch.tensor([8., 2.], dtype=torch.float64))
    snapshot = original.clone()
    result, before, after = d.ReputationWeightedIsoCTSMerger.apply_isoc_2d(original)
    torch.testing.assert_close(result, torch.eye(2, dtype=torch.float64) * 5)
    torch.testing.assert_close(original, snapshot)
    assert result.dtype == original.dtype
    assert before == 4.
    assert after == 1.


@pytest.mark.parametrize("workers", [[], [{}, {}]])
def test_isoc_merge_with_no_updates_preserves_independent_base_tensors(workers):
    base = {"weight": torch.tensor([3., 5.])}
    result, report = d.ReputationWeightedIsoCTSMerger().merge_worker_deltas(base, workers)
    torch.testing.assert_close(result["weight"], base["weight"])
    assert result["weight"] is not base["weight"]
    assert report["num_workers"] == len(workers)


def test_isoc_merge_skips_missing_fragments_and_retains_unupdated_parameters():
    base = {"weight": torch.tensor([1., 2.]), "unused": torch.tensor([8.])}
    workers = [{}, {"weight": torch.tensor([3., 6.])}]
    result, _ = d.ReputationWeightedIsoCTSMerger(scaling_lambda=1, trim_density=1).merge_worker_deltas(base, workers)
    torch.testing.assert_close(result["weight"], torch.tensor([3., 6.]))
    torch.testing.assert_close(result["unused"], torch.tensor([8.]))
    assert result["unused"] is not base["unused"]
    torch.testing.assert_close(base["weight"], torch.tensor([1., 2.]))


def test_incomplete_reputation_list_raises_value_error():
    """IP-235: reputation-weight length mismatch raises ValueError instead of silently falling back to uniform."""
    merger = d.ReputationWeightedIsoCTSMerger(scaling_lambda=1, trim_density=1)
    with pytest.raises(ValueError, match="worker_reputation_weights"):
        merger.merge_worker_deltas(
            {"weight": torch.zeros(2)},
            [{"weight": torch.ones(2)}, {"weight": torch.full((2,), 3.)}, {"weight": torch.full((2,), 5.)}],
            [1.0, 2.0],
        )
    with pytest.raises(ValueError, match="finite"):
        merger.merge_worker_deltas(
            {"weight": torch.zeros(2)},
            [{"weight": torch.ones(2)}, {"weight": torch.full((2,), 3.)}],
            [1.0, float("nan")],
        )


@pytest.mark.parametrize("steps,lr", [(0, 0.001), (1, 0)])
def test_autoloco_rejects_invalid_realized_interval(steps, lr):
    with pytest.raises(ValueError, match="actual_inner"):
        d.AutoLoCoIntervalCorrector().correct_outer_hyperparams(steps, lr)


@pytest.mark.parametrize("workers,overlap,gamma,message", [
    (0, 1, 0, "num_workers"), (2, 0, 0, "overlap_factor"),
    (2, 3, 0, "overlap_factor"), (2, 1, -0.1, "reshuffle_frequency"),
    (2, 1, 1.1, "reshuffle_frequency"),
])
def test_fomoe_rejects_invalid_overlap_schedules(workers, overlap, gamma, message):
    with pytest.raises(ValueError, match=message):
        d.FoMoEPartialReplicaPlanner().evaluate_overlap_configuration(workers, overlap, gamma)


@pytest.mark.parametrize("matrix,topk,message", [
    (torch.zeros(4), 1, "2D"), (torch.zeros(2, 3), 1, "divisible"),
    (torch.zeros(2, 2), 0, "topk_per_chunk"), (torch.zeros(2, 2), 5, "topk_per_chunk"),
])
def test_dct_rejects_invalid_shapes_and_frequency_budgets(matrix, topk, message):
    snapshot = matrix.clone()
    with pytest.raises(ValueError, match=message):
        d.ChunkedOrthonormalDCTTransform(chunk_size=2).encode_and_decode_topk(matrix, topk)
    torch.testing.assert_close(matrix, snapshot)


@pytest.mark.parametrize("deltas,message", [([], "non-empty"), ([torch.ones(3)], "2D")])
def test_tsv_requires_nonempty_matrix_deltas(deltas, message):
    with pytest.raises(ValueError, match=message):
        d.TSVDecorrelatedMerger.merge_2d_deltas(deltas)


def test_muon_rejects_vector_for_spectral_telemetry():
    with pytest.raises(ValueError, match="2D tensor"):
        d.MuonNewtonSchulzLoRAPreconditioner().orthogonalize_2d(torch.ones(3))


@pytest.mark.parametrize("gradient,momentum,cosine", [
    (torch.tensor([0., 0.]), torch.tensor([1., 0.]), 0.),
    (torch.tensor([1., 0.]), torch.tensor([0., 0.]), 0.),
    (torch.tensor([1., 0.]), torch.tensor([2., 0.]), 1.),
])
def test_heloco_leaves_zero_or_aligned_directions_unchanged(gradient, momentum, cosine):
    result, report = d.HeLoCoDirectionCorrector().correct_pseudo_gradient(gradient, momentum)
    torch.testing.assert_close(result, gradient)
    assert result is not gradient
    assert report["conflict_detected"] == 0
    assert report["cosine_before"] == report["cosine_after"] == cosine


@pytest.mark.parametrize("fraction", [0.5, 1.])
def test_isoc_handles_zero_task_residual_and_full_common_subspace(fraction):
    matrix = torch.diag(torch.tensor([8., 0.]))
    result, report = d.ReputationWeightedIsoCTSMerger(
        common_space_fraction=fraction, scaling_lambda=1, trim_density=1,
    ).merge_worker_deltas({"w": torch.zeros(2, 2)}, [{"w": matrix}])
    torch.testing.assert_close(result["w"], torch.eye(2) * 4)
    torch.testing.assert_close(matrix, torch.diag(torch.tensor([8., 0.])))
    assert report["mean_condition_number_after"] == 1


def test_sparse_compression_without_reference_keeps_the_only_nonzero_coordinate():
    delta = torch.tensor([0., 0., 0., 4.])
    reconstructed, _ = d.ComputeVisibleSparseLoCoCompressor(topk_ratio=0.25).compress_and_reconstruct(delta)
    torch.testing.assert_close(reconstructed, delta)


def test_tsv_single_worker_without_reputation_preserves_its_singular_components():
    delta = torch.diag(torch.tensor([4., 2.], dtype=torch.float64))
    merged, report = d.TSVDecorrelatedMerger.merge_2d_deltas([delta])
    torch.testing.assert_close(merged, delta)
    assert report["num_workers"] == 1
    assert report["rank_per_worker"] == 2


def test_flat_bucket_rejects_wrong_reference_shape_without_changing_master_state():
    bucket = d.CPUOffloadedFlatBucketOuterState({"w": torch.ones(2)})
    snapshot = bucket.flat_master_cpu.clone()
    with pytest.raises(ValueError, match="theta_prev"):
        bucket.outer_nesterov_step({"w": torch.ones(2)}, theta_prev={"w": torch.ones(1, 2)})
    torch.testing.assert_close(bucket.flat_master_cpu, snapshot)


def test_cache_aware_router_keeps_true_expert_when_cached_alternative_is_too_costly():
    router = d.SkipAttentionSpeculativeRouter(num_experts=4, top_k=1, car_threshold=0.01)
    report = router.route_with_cache_aware_fallback(
        post_ffn_residual=torch.tensor([[1., 0.]]),
        post_attn_residual=torch.tensor([[0., 1.]]),
        next_layer_norm_weight=torch.ones(2),
        next_layer_router_weight=torch.tensor([[10., 0.], [0., 10.], [-10., 0.], [0., -10.]]),
        baseline_cached_experts=[],
    )
    assert report["synchronous_dma_misses"] == 1
    assert report["effective_vram_cache_hit_rate"] == 0
    torch.testing.assert_close(report["routed_experts"], torch.tensor([[1]]))
