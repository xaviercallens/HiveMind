"""Device-adapter unit contracts; mocks do not claim to verify GPU kernel execution."""

from contextlib import nullcontext
import importlib.util
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, sentinel

import pytest
import torch

from hivemind.swarm import swarm_fabric as fabric


@pytest.mark.parametrize(
    "device_name, hip, mode",
    [("cpu", None, "cpu_copy"), ("mps", None, "mps_unified"), ("cuda:0", None, "cuda_stream"), ("cuda:0", "6.2", "hip_stream")],
)
def test_streamer_selects_staging_backend_without_allocating_on_accelerator(monkeypatch, device_name, hip, mode):
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "Stream", stream)
    monkeypatch.setattr(torch.version, "hip", hip, raising=False)
    device = torch.device(device_name)
    streamer = fabric.PinnedRAMExpertDMAStreamer(device=device)
    assert streamer.staging_mode == mode
    assert streamer.current_pinned_bytes == 0
    if device.type == "cuda":
        stream.assert_called_once_with(device=device)
        assert streamer._dma_stream is stream.return_value
    else:
        stream.assert_not_called()


def test_dma_adapter_requests_nonblocking_transfer_and_waits_on_stream(monkeypatch):
    stream = Mock(name="dma_stream")
    current_stream = Mock()
    start, end = Mock(), Mock()
    start.elapsed_time.return_value = 3.25
    event = Mock(side_effect=[start, end])
    context = Mock(return_value=nullcontext())
    monkeypatch.setattr(torch.cuda, "Stream", Mock(return_value=stream))
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "stream", context)
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=current_stream))
    device = torch.device("cuda:0")
    streamer = fabric.PinnedRAMExpertDMAStreamer(device=device)
    weight = Mock(name="staged_weight")
    weight.to.return_value = sentinel.resident_weight
    streamer.pinned_host_cache["expert"] = {"weight": weight}
    streamer.lru_order.append("expert")

    result = streamer.async_dma_prefetch_to_vram("expert", layer_idx=2)

    weight.to.assert_called_once_with(device=device, non_blocking=True)
    context.assert_called_once_with(stream)
    start.record.assert_called_once_with(stream)
    end.record.assert_called_once_with(stream)
    end.synchronize.assert_called_once_with()
    current_stream.wait_stream.assert_called_once_with(stream)
    assert result["raw_dma_transfer_ms"] == 3.25
    assert result["exposed_swap_latency_ms"] == 0.75
    assert result["tensors_streamed"] == 1
    assert result["slot"] == "slot_A"
    assert streamer.active_vram_cache["expert"]["weight"] is sentinel.resident_weight
    assert streamer.double_buffer_slots["slot_A"]["expert"]["weight"] is sentinel.resident_weight


def test_fp8_adapter_falls_back_to_cpu_math_after_kernel_dispatch_failure(monkeypatch):
    values = torch.tensor([[1.0, -1.0]])

    class DeviceInput:
        """Only the device boundary is simulated; fallback uses real CPU tensors."""

        shape = values.shape
        is_cuda = True

        def numel(self):
            return values.numel()

        def element_size(self):
            return values.element_size()

        def detach(self):
            return values

    kernel = MagicMock()
    kernel.__getitem__.return_value.side_effect = RuntimeError("fixture accelerator dispatch failure")
    monkeypatch.setattr(fabric, "_HAS_TRITON", True)
    monkeypatch.setattr(fabric, "triton", SimpleNamespace(next_power_of_2=lambda n: 2), raising=False)
    monkeypatch.setattr(fabric, "_triton_fp8_quant_kernel", kernel, raising=False)

    packet = fabric.compress_hidden_states_fp8(DeviceInput())

    kernel.__getitem__.assert_called_once_with((1,))
    assert kernel.__getitem__.return_value.call_count == 1
    restored = packet.fp8_data.to(torch.float32) * packet.absmax_scale.to(torch.float32)
    torch.testing.assert_close(restored, values, atol=0.01, rtol=0)
    assert packet.original_shape == (1, 2)
    assert packet.original_bytes == 8
    assert packet.compressed_bytes == 4


def test_skip_attention_prefetch_requires_a_name_for_every_expert():
    streamer = fabric.PinnedRAMExpertDMAStreamer(device=torch.device("cpu"))
    with pytest.raises(ValueError, match="expert_names length"):
        streamer.prefetch_and_route_skip_attention_car(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.ones(2), torch.zeros(2, 2), ["only-one"]
        )


def test_fused_dispatch_cuda_event_adapter_preserves_cpu_reference(monkeypatch):
    """Verify event/copy orchestration with real CPU math, without GPU allocation."""
    hidden = torch.tensor([[1.0, 0.0]])
    router = torch.eye(2)
    experts = [torch.eye(2), 2 * torch.eye(2)]
    reference = fabric.fused_top2_moe_fp8_dispatch(hidden, router, experts)
    packet = fabric.compress_hidden_states_fp8(hidden)
    decoded = fabric.decompress_hidden_states_fp8(packet)
    device = torch.device("cuda:0")
    copy_requests = []

    class DeviceCopyAdapter:
        """Record device requests while retaining actual tensors on the CPU."""

        def __init__(self, tensor, name):
            self.tensor = tensor
            self.name = name
            self.device = device
            self.dtype = tensor.dtype

        def to(self, *args, **kwargs):
            copy_requests.append((self.name, args, dict(kwargs)))
            cpu_kwargs = {key: value for key, value in kwargs.items() if key != "device"}
            return self.tensor.to(*args, **cpu_kwargs)

    order = []
    start, end = Mock(), Mock()
    start.record.side_effect = lambda: order.append("start")
    end.record.side_effect = lambda: order.append("end")
    start.elapsed_time.side_effect = lambda other: order.append("elapsed") or 2.75
    event = Mock(side_effect=[start, end])
    synchronize = Mock(side_effect=lambda target: order.append("synchronize"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "Event", event)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(fabric, "compress_hidden_states_fp8", lambda value: packet)
    monkeypatch.setattr(fabric, "decompress_hidden_states_fp8", lambda value: DeviceCopyAdapter(decoded, "decoded"))

    result = fabric.fused_top2_moe_fp8_dispatch(
        DeviceCopyAdapter(hidden, "hidden"),
        DeviceCopyAdapter(router, "router"),
        [DeviceCopyAdapter(weight, f"expert-{i}") for i, weight in enumerate(experts)],
    )

    torch.testing.assert_close(result["output"], reference["output"])
    torch.testing.assert_close(result["top_k_weights"], reference["top_k_weights"])
    assert result["kernel_latency_ms"] == 2.75
    assert order == ["start", "end", "synchronize", "elapsed"]
    assert event.call_args_list[0].kwargs == event.call_args_list[1].kwargs == {"enable_timing": True}
    synchronize.assert_called_once_with(device)
    start.elapsed_time.assert_called_once_with(end)
    for name in ("router", "decoded", "expert-0", "expert-1"):
        requests = [kwargs for item, _, kwargs in copy_requests if item == name]
        assert requests == [{"device": device, "dtype": torch.float32}]


def test_fabric_remains_usable_when_optional_triton_is_absent(monkeypatch):
    name = "hivemind.swarm._fabric_without_optional_triton_test"
    monkeypatch.setitem(sys.modules, "triton", None)
    spec = importlib.util.spec_from_file_location(name, fabric.__file__)
    assert spec is not None and spec.loader is not None
    isolated = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, isolated)
    spec.loader.exec_module(isolated)
    assert isolated._HAS_TRITON is False
    values = torch.tensor([[1.0, -1.0, 0.0]])
    packet = isolated.compress_hidden_states_fp8(values)
    restored = isolated.decompress_hidden_states_fp8(packet, target_dtype=torch.float32)
    torch.testing.assert_close(restored, values, atol=0.01, rtol=0)
    assert packet.original_shape == (1, 3)
    assert packet.fp8_data.device.type == "cpu"
