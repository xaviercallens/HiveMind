"""Telemetry boundary failures and hardware-protection decisions without a GPU."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hivemind.security import gpu_execution_circuit_breaker as gpu


@pytest.fixture(autouse=True)
def _nominal_host_telemetry():
    """Replace the suite's host overrides with this module's explicit probe fakes."""


@pytest.fixture
def hardware(monkeypatch):
    monkeypatch.setattr(gpu.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(gpu.torch.cuda, "get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr(gpu.torch.cuda, "memory_allocated", lambda index: 128.5 * 1024**2)
    monkeypatch.setattr(gpu.torch.cuda, "memory_reserved", lambda index: 256.25 * 1024**2)
    monkeypatch.setattr(gpu.torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=8192 * 1024**2))
    monkeypatch.setattr(gpu.psutil, "sensors_battery", lambda: SimpleNamespace(power_plugged=False, percent=42.125))
    monkeypatch.setattr(gpu.psutil, "virtual_memory", lambda: SimpleNamespace(percent=64.5))
    query = Mock(return_value="72.25, 145.5, 200, 87\n99, 999, 999, 99\n")
    monkeypatch.setattr(gpu.subprocess, "check_output", query)
    return query


def test_first_gpu_csv_units_and_battery_are_preserved(hardware):
    result = gpu.query_live_gpu_hardware_telemetry()
    assert result.gpu_name == "Test GPU"
    assert result.temperature_c == 72.25
    assert result.power_draw_watts == 145.5
    assert result.power_limit_watts == 200
    assert result.power_fraction == pytest.approx(0.7275)
    assert result.vram_allocated_mb == 128.5
    assert result.vram_reserved_mb == 256.25
    assert result.total_vram_mb == 8192
    assert result.gpu_utilization_pct == 87
    assert result.on_battery_power is True
    assert result.battery_percent == 42.12
    assert result.system_ram_used_pct == 64.5
    hardware.assert_called_once_with(
        ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.limit,utilization.gpu", "--format=csv,noheader,nounits"],
        text=True, timeout=0.4, stderr=subprocess.DEVNULL,
    )


@pytest.mark.parametrize("output", ["", "71, 123", "N/A, 123, 200, 50"])
def test_unavailable_csv_fields_keep_defaults_without_losing_vram(hardware, output):
    hardware.return_value = output
    result = gpu.query_live_gpu_hardware_telemetry()
    assert result.temperature_c == 48
    assert result.power_draw_watts == 35
    assert result.power_fraction == 0.2
    assert result.vram_allocated_mb == 128.5
    assert result.on_battery_power is True


def test_failed_optional_probes_keep_fallback_snapshot(monkeypatch, hardware):
    def unavailable(*args, **kwargs):
        raise OSError("device probe unavailable")

    monkeypatch.setattr(gpu.torch.cuda, "get_device_name", unavailable)
    monkeypatch.setattr(gpu.psutil, "sensors_battery", unavailable)
    monkeypatch.setattr(gpu.psutil, "virtual_memory", unavailable)
    hardware.side_effect = subprocess.TimeoutExpired("nvidia-smi", 0.4)
    result = gpu.query_live_gpu_hardware_telemetry()
    assert result.gpu_name == "CPU-Fallback-Device"
    assert result.vram_allocated_mb == result.vram_reserved_mb == 0
    assert result.temperature_c == 48
    assert result.battery_percent == 100
    assert result.on_battery_power is False
    assert result.system_ram_used_pct == 50
    hardware.assert_called_once()


def test_explicit_overrides_skip_smi_and_clamp_power_limit(hardware):
    result = gpu.query_live_gpu_hardware_telemetry({"temperature_c": 80.75, "power_limit_watts": 0, "power_draw_watts": 30})
    hardware.assert_not_called()
    assert result.temperature_c == 80.75
    assert result.power_limit_watts == 1
    assert result.power_fraction == 30
    assert result.vram_allocated_mb == 128.5


def test_zero_reported_power_limit_is_clamped(hardware):
    hardware.return_value = "61, 20, 0, 15"
    result = gpu.query_live_gpu_hardware_telemetry()
    assert result.power_limit_watts == 1
    assert result.power_fraction == 20


@pytest.mark.parametrize("cooldown", [90, 91])
def test_cooldown_must_be_below_trip_temperature(cooldown):
    with pytest.raises(ValueError, match="strictly below"):
        gpu.GPUExecutionCircuitBreaker(cooldown_resume_temp_c=cooldown, hard_trip_temp_c=90)


@pytest.mark.parametrize("batch,sequence,latency", [(0, 8, 0), (1, 9, 0), (1, 8, 1.6)])
def test_tdr_violations_can_return_false_without_raising(batch, sequence, latency):
    breaker = gpu.GPUExecutionCircuitBreaker(max_tokens_per_microstep=8, wddm_tdr_step_ceiling_sec=1.5)
    assert not breaker.verify_microbatch_tdr_safety(batch, sequence, latency, raise_on_violation=False)
    with pytest.raises(gpu.SwarmSecurityViolationError, match="HW-03"):
        breaker.verify_microbatch_tdr_safety(batch, sequence, latency)
    assert breaker.verify_microbatch_tdr_safety(1, 8, 1.5)


def test_temperature_trip_is_visible_in_status_and_latch(hardware):
    breaker = gpu.GPUExecutionCircuitBreaker(simulated_telemetry={"temperature_c": 100})
    assert not breaker.thermal_latch_tripped
    status = breaker.evaluate()
    assert status.should_preempt
    assert breaker.thermal_latch_tripped
    assert status.thermal_latch_tripped
    assert status.recommended_sleep_sec == status.recommended_cooling_sleep_ms / 1000
