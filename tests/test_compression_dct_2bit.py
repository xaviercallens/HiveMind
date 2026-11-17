"""Unit tests for `HM-S8-05`: Chunked Orthonormal DCT + 2-Bit SparseLoCo wire compression."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from hivemind.core.compression import (
    COMPRESSION_MODES,
    DCT_SPARSE_2BIT_MIN_COMPRESSION_RATIO,
    CompressedDeltaPackage,
    DCTSparse2BitTensorPayload,
    _index_width,
    _orthonormal_dct_basis,
    _pack_2bit_codes,
    _quantize_2bit_centroids,
    _unpack_2bit_codes,
    compress_lora_delta,
    compress_lora_delta_dct_sparse_2bit,
    compress_tensor_dct_sparse_2bit,
    compress_tensor_topk_fp8,
    decompress_lora_delta,
    decompress_lora_delta_dct_sparse_2bit,
    decompress_tensor_dct_sparse_2bit,
)


def _smooth_delta(size: int = 32) -> torch.Tensor:
    grid = torch.linspace(0.0, 1.0, size)
    return torch.outer(torch.cos(grid * 3.0), torch.sin(grid * 2.0))


def test_2bit_code_bit_packing_uses_four_codes_per_uint8() -> None:
    codes = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0, 2], dtype=torch.uint8)
    packed = _pack_2bit_codes(codes)

    # 0 | 1<<2 | 2<<4 | 3<<6 = 228; 3 | 2<<2 | 1<<4 | 0<<6 = 27; 2 | 0 | 0 | 0 = 2.
    assert packed.dtype == torch.uint8
    assert packed.tolist() == [228, 27, 2]
    assert packed.numel() == math.ceil(codes.numel() / 4)
    assert packed.numel() * 4 == 12  # four 2-bit codes per byte, padded with zeros
    assert _unpack_2bit_codes(packed, codes.numel()).tolist() == codes.tolist()

    # No padding path: exactly two full bytes.
    full_quads = _pack_2bit_codes(torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.uint8))
    assert full_quads.tolist() == [228, 228]
    assert _unpack_2bit_codes(full_quads, 8).tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_2bit_centroid_lut_handles_empty_bins_and_constant_values() -> None:
    codes, lookup, shift, scale = _quantize_2bit_centroids(
        torch.tensor([0.0, 0.0, 0.0, 1.0]), n_bins=4, range_in_sigmas=4.0
    )
    assert codes.dtype == torch.uint8
    assert lookup.dtype == torch.float16
    assert shift == pytest.approx(0.25)
    assert scale > 0.0
    # Centered values are [-0.25, -0.25, -0.25, 0.75]: bins 1 and 3 are populated; empty bins 0 and 2 get symmetric centers.
    assert lookup[0].item() == pytest.approx(-0.75)
    assert lookup[1].item() == pytest.approx(-0.25)
    assert lookup[2].item() == pytest.approx(0.25)
    assert lookup[3].item() == pytest.approx(0.75)

    constant_codes, constant_lookup, constant_shift, constant_scale = _quantize_2bit_centroids(
        torch.full((5,), 3.5), n_bins=4, range_in_sigmas=4.0
    )
    assert constant_codes.tolist() == [0, 0, 0, 0, 0]
    assert torch.count_nonzero(constant_lookup).item() == 0
    assert constant_shift == pytest.approx(3.5)
    assert constant_scale == 1.0


def test_chunked_dct_payload_roundtrip_and_telemetry() -> None:
    torch.manual_seed(7)
    smooth = _smooth_delta()
    payload = compress_tensor_dct_sparse_2bit(smooth, topk_ratio=0.03, chunk_size=8)

    assert isinstance(payload, DCTSparse2BitTensorPayload)
    assert payload.original_shape == (32, 32)
    assert payload.original_numel == 1024
    assert payload.padded_shape == (32, 32)
    assert payload.chunk_size == 8
    assert payload.index_width == 2
    assert payload.packed_2bit.dtype == torch.uint8
    assert payload.lookup.dtype == torch.float16
    assert payload.packed_2bit.numel() == math.ceil(payload.n_coefficients / 4)
    assert payload.dct_energy_retained_fraction > 0.99
    assert payload.compressed_bytes == (
        payload.packed_2bit.numel()
        + payload.indices.numel() * payload.index_width
        + payload.lookup.numel() * 2
        + 12
    )

    restored = decompress_tensor_dct_sparse_2bit(payload)
    assert restored.shape == smooth.shape
    relative_l2 = float((restored - smooth).norm().item() / smooth.norm().item())
    cosine = float(F.cosine_similarity(restored.reshape(1, -1), smooth.reshape(1, -1)).item())
    assert relative_l2 < 0.40, f"relative L2 error too high: {relative_l2}"
    assert cosine > 0.90, f"cosine similarity too low: {cosine}"
    assert restored.dtype == torch.float32

    # Deterministic replay: identical input produces a bitwise-identical wire payload.
    replay = compress_tensor_dct_sparse_2bit(smooth, topk_ratio=0.03, chunk_size=8)
    assert torch.equal(replay.packed_2bit, payload.packed_2bit)
    assert torch.equal(replay.indices, payload.indices)
    assert torch.equal(replay.lookup, payload.lookup)
    assert replay.shift == payload.shift
    assert replay.scale == payload.scale

    # 1-D tensors are reshaped to a chunk-padded row and cropped back to their original shape.
    vector = torch.linspace(-1.0, 1.0, 37)
    vector_payload = compress_tensor_dct_sparse_2bit(vector, topk_ratio=0.10, chunk_size=8)
    assert vector_payload.original_shape == (37,)
    assert vector_payload.padded_shape == (8, 40)
    vector_restored = decompress_tensor_dct_sparse_2bit(
        vector_payload, dtype=torch.float64
    )
    assert vector_restored.shape == (37,)
    assert vector_restored.dtype == torch.float64

    # Rectangular tensors with padded rows and columns must crop on both axes.
    torch.manual_seed(13)
    rectangular = torch.randn(30, 50)
    rectangular_payload = compress_tensor_dct_sparse_2bit(
        rectangular, topk_ratio=1.0, chunk_size=8
    )
    assert rectangular_payload.padded_shape == (32, 56)
    rectangular_restored = decompress_tensor_dct_sparse_2bit(rectangular_payload)
    assert rectangular_restored.shape == rectangular.shape
    rectangular_error = float(
        (rectangular_restored - rectangular).norm().item() / rectangular.norm().item()
    )
    assert rectangular_error < 0.50

    assert _orthonormal_dct_basis(8) is _orthonormal_dct_basis(8)


def test_dct_package_exceeds_16x_wire_reduction_and_roundtrips() -> None:
    torch.manual_seed(123)
    delta_dict = {
        "layers.0.self_attn.q_proj.lora_A.weight": torch.randn((64, 512)),
        "layers.0.self_attn.q_proj.lora_B.weight": torch.randn((512, 64)),
        "layers.0.mlp.gate_proj.lora_A.weight": torch.randn((64, 512)),
        "layers.0.mlp.gate_proj.lora_B.weight": torch.randn((1024, 64)),
    }

    package = compress_lora_delta(
        delta_dict=delta_dict,
        worker_id="worker-code-01",
        branch_name="Code",
        outer_step=1,
        inner_steps_completed=500,
        compression_mode="dct_sparse_2bit",
    )
    assert package.compression_mode == "dct_sparse_2bit"
    assert package.topk_ratio == 0.02
    assert package.uncompressed_bytes == sum(tensor.numel() * 2 for tensor in delta_dict.values())
    assert package.compressed_bytes == sum(
        payload.compressed_bytes for payload in package.tensors.values()
    )
    assert package.compression_ratio >= DCT_SPARSE_2BIT_MIN_COMPRESSION_RATIO
    assert all(
        isinstance(payload, DCTSparse2BitTensorPayload) for payload in package.tensors.values()
    )

    restored = decompress_lora_delta(package)
    assert set(restored) == set(delta_dict)
    for name, original in delta_dict.items():
        assert restored[name].shape == original.shape
        assert torch.isfinite(restored[name]).all()

    small_package = compress_lora_delta(
        delta_dict={"lora_A": torch.randn(16, 16)},
        worker_id="worker-code-01",
        branch_name="Code",
        outer_step=2,
        compression_mode="dct_sparse_2bit",
    )
    # Even the smallest realistic 2-D LoRA shard clears the target at the 2% default.
    assert small_package.compression_ratio >= DCT_SPARSE_2BIT_MIN_COMPRESSION_RATIO

    direct = compress_lora_delta_dct_sparse_2bit(
        delta_dict=delta_dict,
        worker_id="worker-code-01",
        branch_name="Code",
        outer_step=3,
        inner_steps_completed=500,
        topk_ratio=0.03,
    )
    assert direct.compression_mode == "dct_sparse_2bit"
    assert direct.compression_ratio >= DCT_SPARSE_2BIT_MIN_COMPRESSION_RATIO
    direct_restored = decompress_lora_delta_dct_sparse_2bit(direct)
    assert {name: tensor.shape for name, tensor in direct_restored.items()} == {
        name: tensor.shape for name, tensor in delta_dict.items()
    }


def test_legacy_topk_fp8_mode_remains_the_default() -> None:
    torch.manual_seed(321)
    delta_dict = {"lora_A": torch.randn(64, 512), "lora_B": torch.randn(512, 64)}

    package = compress_lora_delta(
        delta_dict=delta_dict,
        worker_id="worker-fp8",
        branch_name="Code",
        outer_step=1,
    )
    assert package.compression_mode == "topk_fp8"
    assert package.topk_ratio == 0.20
    assert package.compression_ratio >= 3.0
    restored = decompress_lora_delta(package)
    assert set(restored) == set(delta_dict)
    assert all(restored[name].shape == original.shape for name, original in delta_dict.items())

    explicit = compress_lora_delta(
        delta_dict=delta_dict,
        worker_id="worker-fp8",
        branch_name="Code",
        outer_step=2,
        topk_ratio=0.10,
        compression_mode="topk_fp8",
    )
    assert explicit.compression_mode == "topk_fp8"
    assert explicit.topk_ratio == 0.10


def test_index_width_boundaries_and_large_tensor_uses_four_byte_indices() -> None:
    assert _index_width(0) == 1
    assert _index_width(255) == 1
    assert _index_width(256) == 1
    assert _index_width(257) == 2
    assert _index_width(65535) == 2
    assert _index_width(65536) == 2
    assert _index_width(65537) == 4

    torch.manual_seed(99)
    large = torch.randn(256, 260)  # 66,560 elements > uint16 address space
    payload = compress_tensor_dct_sparse_2bit(large, topk_ratio=0.005)
    assert payload.original_numel == 66560
    assert payload.index_width == 4
    restored = decompress_tensor_dct_sparse_2bit(payload)
    assert restored.shape == large.shape


def test_dct_compression_rejects_invalid_configuration() -> None:
    tensor = torch.randn(16, 16)

    for kwargs in (
        {"topk_ratio": 0.0},
        {"topk_ratio": 1.5},
        {"chunk_size": 1},
        {"n_bins": 8},
        {"range_in_sigmas": 0.0},
    ):
        with pytest.raises(ValueError):
            compress_tensor_dct_sparse_2bit(tensor, **kwargs)

    with pytest.raises(ValueError):
        compress_tensor_dct_sparse_2bit(torch.empty(0, 4))

    with pytest.raises(ValueError):
        _quantize_2bit_centroids(torch.randn(8), n_bins=8, range_in_sigmas=4.0)
    with pytest.raises(ValueError):
        _quantize_2bit_centroids(torch.randn(8), n_bins=4, range_in_sigmas=0.0)

    delta_dict = {"lora_A": tensor}
    with pytest.raises(ValueError):
        compress_lora_delta(
            delta_dict,
            worker_id="w",
            branch_name="Code",
            outer_step=1,
            compression_mode="unsupported_mode",
        )
    with pytest.raises(ValueError):
        compress_lora_delta(
            delta_dict,
            worker_id="w",
            branch_name="Code",
            outer_step=1,
            topk_ratio=0.0,
        )
    with pytest.raises(ValueError):
        compress_lora_delta_dct_sparse_2bit(
            delta_dict,
            worker_id="w",
            branch_name="Code",
            outer_step=1,
            topk_ratio=2.0,
        )


def test_dct_decompression_rejects_mode_and_payload_mismatches() -> None:
    torch.manual_seed(5)
    dct_payload = compress_tensor_dct_sparse_2bit(torch.randn(16, 16))
    with pytest.raises(TypeError):
        decompress_tensor_dct_sparse_2bit(compress_tensor_topk_fp8(torch.randn(16, 16)))
    with pytest.raises(ValueError):
        decompress_tensor_dct_sparse_2bit(
            replace(dct_payload, n_coefficients=dct_payload.n_coefficients + 1)
        )

    package = CompressedDeltaPackage(
        worker_id="w",
        branch_name="Code",
        outer_step=1,
        inner_steps_completed=1,
        topk_ratio=0.02,
        tensors={"lora_A": dct_payload},
        uncompressed_bytes=512,
        compressed_bytes=dct_payload.compressed_bytes,
        compression_mode="dct_sparse_2bit",
    )
    assert decompress_lora_delta(package)["lora_A"].shape == (16, 16)
    with pytest.raises(ValueError):
        decompress_lora_delta(replace(package, compression_mode="unsupported_mode"))
    with pytest.raises(ValueError):
        decompress_lora_delta_dct_sparse_2bit(replace(package, compression_mode="topk_fp8"))
    with pytest.raises(TypeError):
        decompress_lora_delta_dct_sparse_2bit(
            replace(
                package,
                tensors={"lora_A": compress_tensor_topk_fp8(torch.randn(16, 16))},
            )
        )

    assert "topk_fp8" in COMPRESSION_MODES
    assert "dct_sparse_2bit" in COMPRESSION_MODES
