# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit tests for HM-S33-03 Eco-Compute Domestic Thermal Valorization & Carbon-Offset Ledger."""

from __future__ import annotations

from hivemind.swarm.s33_03_eco_thermal_carbon_ledger import (
    EcoComputeTelemetrySample,
    EcoThermalCarbonLedger,
)


def test_eco_thermal_ledger_winter_domestic_heating_valorization() -> None:
    ledger = EcoThermalCarbonLedger()
    sample = EcoComputeTelemetrySample(
        node_id="rtx2070-paris-winter",
        gpu_power_watts=160.0,
        gpu_temp_celsius=68.5,
        outdoor_temp_celsius=7.0,
        grid_carbon_intensity_g_per_kwh=55.0,
        duration_seconds=3600.0,
        tokens_generated=120_000,
    )
    receipt = ledger.evaluate_sample(sample)
    assert receipt.safe_thermal_envelope is True
    assert receipt.thermal_valorization_efficiency == 0.92
    assert receipt.useful_domestic_heat_wh > 140.0
    assert receipt.net_avoided_co2_grams > 70.0
    assert receipt.co2_grams_per_1k_tokens_hivemind < receipt.co2_grams_per_1k_tokens_datacenter


def test_eco_thermal_ledger_flags_unsafe_gpu_temperature() -> None:
    ledger = EcoThermalCarbonLedger()
    sample = EcoComputeTelemetrySample(
        node_id="rtx3080-summer",
        gpu_power_watts=220.0,
        gpu_temp_celsius=81.0,
        outdoor_temp_celsius=26.0,
        grid_carbon_intensity_g_per_kwh=200.0,
        duration_seconds=1800.0,
        tokens_generated=60_000,
    )
    receipt = ledger.evaluate_sample(sample)
    assert receipt.safe_thermal_envelope is False
    assert receipt.thermal_valorization_efficiency == 0.0
