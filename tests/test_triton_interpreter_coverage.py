"""Numerical CPU-interpreter tests; these do not validate GPU execution.

Run only through the approved remote quarantine launcher. Triton's optional
interpreter executes the real kernel bodies using NumPy operations:
https://triton-lang.org/main/programming-guide/chapter-3/debugging.html
"""

import importlib.util
import sys

import pytest
import torch


@pytest.fixture
def interpreter_fabric(monkeypatch):
    """Decorate an isolated module's kernels without reloading the original."""
    import os

    if os.environ.get("TRITON_INTERPRET") != "1":
        pytest.skip(
            "CPU interpreter tests require TRITON_INTERPRET=1 before starting "
            "coverage/pytest so Triton language helpers are interpreted at import time"
        )
    pytest.importorskip("triton", reason="optional Triton CPU interpreter")
    from triton.runtime.interpreter import InterpretedFunction

    from hivemind.swarm import swarm_fabric as original

    name = "hivemind.swarm._fabric_interpreter_test"
    spec = importlib.util.spec_from_file_location(name, original.__file__)
    assert spec is not None and spec.loader is not None
    clone = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, clone)
    spec.loader.exec_module(clone)

    assert clone._HAS_TRITON, "the isolated module must load real Triton kernels"
    assert isinstance(clone._triton_fp8_quant_kernel, InterpretedFunction)
    assert isinstance(
        clone._triton_fused_int4_fp8_moe_router_kernel, InterpretedFunction
    )
    return clone


@pytest.mark.parametrize("n_cols", [3, 5, 9])
def test_fp8_kernel_matches_torch_for_padded_rows(interpreter_fabric, n_cols):
    """Catch incorrect row addressing, masked stores, scale floors, or clipping."""
    fabric = interpreter_fabric
    x = torch.stack(
        [
            torch.zeros(n_cols, dtype=torch.float32),
            torch.linspace(-3.5, 7.0, n_cols, dtype=torch.float32),
            torch.linspace(1e-8, -3e-7, n_cols, dtype=torch.float32),
            torch.linspace(448.0, -448.0, n_cols, dtype=torch.float32),
        ]
    )
    n_rows = x.shape[0]
    block_size = fabric.triton.next_power_of_2(n_cols)
    assert block_size > n_cols
    storage = torch.full(
        (x.numel() + block_size,), float("nan"), dtype=torch.float32
    )
    quantized = storage[: x.numel()].reshape_as(x)
    scales = torch.full((n_rows,), float("nan"), dtype=torch.float32)

    fabric._triton_fp8_quant_kernel[(n_rows,)](
        x, quantized, scales, n_cols, BLOCK_SIZE=block_size
    )

    expected_scales = x.abs().amax(dim=1).clamp_min(1e-6) / 448.0
    expected_quantized = torch.clamp(x / expected_scales[:, None], -448.0, 448.0)
    torch.testing.assert_close(scales, expected_scales, rtol=1e-6, atol=0)
    torch.testing.assert_close(
        quantized, expected_quantized, rtol=1e-5, atol=1e-5
    )
    assert torch.isfinite(quantized).all()
    assert (quantized >= -448.0).all() and (quantized <= 448.0).all()
    assert torch.equal(quantized[0], torch.zeros_like(quantized[0]))
    assert torch.isnan(storage[x.numel() :]).all(), "padded lanes must not store"


def test_int4_router_kernel_matches_independent_nibble_expansion(interpreter_fabric):
    """Catch swapped nibbles, unsigned subtraction, or incorrect row scales."""
    x = torch.tensor(
        [
            [1.0, -2.0, 3.0, 4.0, -5.0, 6.0],
            [-3.0, 0.5, 2.0, -1.0, 4.0, -2.0],
            [0.0, 2.0, -4.0, 1.0, 3.0, -1.0],
        ],
        dtype=torch.float32,
    )
    packed = torch.tensor(
        [[0xF0, 0x87, 0x1E], [0x2D, 0x40, 0xB9], [0x88, 0xFF, 0x00]],
        dtype=torch.uint8,
    )
    scales = torch.tensor([0.5, 0.25, 2.0], dtype=torch.float16)
    output = torch.full((3,), float("nan"), dtype=torch.float32)

    interpreter_fabric._triton_fused_int4_fp8_moe_router_kernel[(3,)](
        x, packed, scales, output, 6, 3, BLOCK_SIZE=4
    )

    # Arithmetic expansion is independent of the production bit operations and
    # deliberately does not use the project's packing/unpacking helpers.
    unsigned = packed.to(torch.int16)
    low = unsigned.remainder(16) - 8
    high = unsigned.div(16, rounding_mode="floor") - 8
    weights = torch.stack((low, high), dim=-1).reshape_as(x).to(torch.float32)
    expected = (x * weights * scales.to(torch.float32)[:, None]).sum(dim=1)
    torch.testing.assert_close(
        expected,
        torch.tensor([-48.5, -8.0, -74.0], dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


class _CPUInterpreterInput:
    """Enter the device boundary while every tensor operation remains on CPU."""

    is_cuda = True

    def __init__(self, tensor):
        assert tensor.device.type == "cpu"
        self.tensor = tensor

    @property
    def shape(self):
        return self.tensor.shape

    def numel(self):
        return self.tensor.numel()

    def element_size(self):
        return self.tensor.element_size()

    def detach(self):
        return self.tensor.detach()


class _RecordingInterpreterKernel:
    """Record completion only after the real interpreted kernel has returned."""

    def __init__(self, kernel):
        from triton.runtime.interpreter import InterpretedFunction

        assert isinstance(kernel, InterpretedFunction)
        self.kernel = kernel
        self.completed = False
        self.grid = None

    def __getitem__(self, grid):
        launch = self.kernel[grid]
        self.grid = grid

        def run(*args, **kwargs):
            result = launch(*args, **kwargs)
            self.completed = True
            return result

        return run


def test_compress_hidden_states_uses_completed_interpreter_launch(
    interpreter_fabric, monkeypatch
):
    """A failed kernel hidden by the fallback must not satisfy this test."""
    fabric = interpreter_fabric
    x = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0], [-2.0, -1.0, 0.0, 1.0, 2.0]],
            [[4.0, -3.0, 2.0, -1.0, 0.0], [-0.5, 0.125, 1.0, -2.0, 8.0]],
        ],
        dtype=torch.float32,
    )
    recording = _RecordingInterpreterKernel(fabric._triton_fp8_quant_kernel)
    monkeypatch.setattr(fabric, "_triton_fp8_quant_kernel", recording)

    packet = fabric.compress_hidden_states_fp8(_CPUInterpreterInput(x))

    assert recording.completed, "the real interpreted launch must return successfully"
    assert recording.grid == (4,)
    expected_scales = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448.0
    expected_fp8 = (x / expected_scales).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    wire_scales = expected_scales.to(torch.float16)
    assert packet.fp8_data.device.type == "cpu"
    assert packet.absmax_scale.device.type == "cpu"
    assert packet.fp8_data.dtype == torch.float8_e4m3fn
    assert packet.absmax_scale.dtype == torch.float16
    assert packet.original_shape == (2, 2, 5)
    assert packet.fp8_data.shape == x.shape
    assert packet.absmax_scale.shape == (2, 2, 1)
    assert packet.original_bytes == 80
    assert packet.compressed_bytes == 28
    torch.testing.assert_close(packet.absmax_scale, wire_scales, rtol=0, atol=0)
    torch.testing.assert_close(
        packet.fp8_data.to(torch.float32), expected_fp8.to(torch.float32), rtol=0, atol=0
    )

    restored = fabric.decompress_hidden_states_fp8(packet, target_dtype=torch.float32)
    expected_restored = expected_fp8.to(torch.float32) * wire_scales.to(torch.float32)
    assert restored.shape == x.shape
    assert restored.dtype == torch.float32
    assert restored.device.type == "cpu"
    torch.testing.assert_close(restored, expected_restored, rtol=0, atol=0)
