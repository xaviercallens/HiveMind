# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S33-03 — Eco-Compute Domestic Thermal Valorization & Carbon-Offset Telemetry Ledger.

Models the thermodynamic & ecological advantage of decentralized consumer GPUs (Project HiveMind)
compared to hyperscale cooling-tower datacenters:
1. Tracks GPU thermal dissipation within the safe domestic comfort band (``60–72 °C``).
2. Computes useful domestic heating displacement during cool ambient conditions (``T_outdoor < 16 °C``).
3. Calculates net avoided ``gCO2eq`` per 1,000 tokens vs hyperscale datacenters with ``PUE = 1.25``
   and cooling overhead.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class EcoComputeTelemetrySample:
    """Telemetry from a domestic HiveMind prosumer GPU node."""

    node_id: str
    gpu_power_watts: float
    gpu_temp_celsius: float
    outdoor_temp_celsius: float
    grid_carbon_intensity_g_per_kwh: float
    duration_seconds: float
    tokens_generated: int


@dataclass(frozen=True)
class EcoComputeValorizationReceipt:
    """Thermodynamic and carbon-offset receipt for a domestic HiveMind inference/training window."""

    node_id: str
    energy_consumed_wh: float
    useful_domestic_heat_wh: float
    thermal_valorization_efficiency: float
    safe_thermal_envelope: bool
    datacenter_equivalent_wh: float
    net_avoided_co2_grams: float
    co2_grams_per_1k_tokens_hivemind: float
    co2_grams_per_1k_tokens_datacenter: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EcoThermalCarbonLedger:
    """Thermodynamic valorization and carbon-displacement calculator for domestic Bee nodes."""

    HYPERSCALE_PUE = 1.25
    HYPERSCALE_GRID_G_PER_KWH = 380.0
    SAFE_TEMP_CEILING_C = 76.0

    def evaluate_sample(self, sample: EcoComputeTelemetrySample) -> EcoComputeValorizationReceipt:
        if sample.gpu_power_watts <= 0.0 or sample.duration_seconds <= 0.0 or sample.tokens_generated <= 0:
            raise ValueError("power, duration, and tokens_generated must be positive")

        energy_wh = sample.gpu_power_watts * (sample.duration_seconds / 3600.0)
        safe_temp = sample.gpu_temp_celsius <= self.SAFE_TEMP_CEILING_C

        # Domestic heating displacement: when outdoor temp < 18 C, 100% of Joule heat replaces resistive heating
        if sample.outdoor_temp_celsius <= 12.0:
            heat_eff = 0.92
        elif sample.outdoor_temp_celsius < 18.0:
            heat_eff = 0.92 * (18.0 - sample.outdoor_temp_celsius) / 6.0
        else:
            heat_eff = 0.0

        useful_heat_wh = energy_wh * heat_eff
        dc_wh = energy_wh * self.HYPERSCALE_PUE

        dc_co2_g = (dc_wh / 1000.0) * self.HYPERSCALE_GRID_G_PER_KWH
        gross_hm_co2_g = (energy_wh / 1000.0) * sample.grid_carbon_intensity_g_per_kwh
        # Displaced domestic heating credit
        heating_credit_co2_g = (useful_heat_wh / 1000.0) * sample.grid_carbon_intensity_g_per_kwh
        net_hm_co2_g = max(0.0, gross_hm_co2_g - heating_credit_co2_g)
        avoided_co2_g = max(0.0, dc_co2_g - net_hm_co2_g)

        k_tokens = sample.tokens_generated / 1000.0
        return EcoComputeValorizationReceipt(
            node_id=sample.node_id,
            energy_consumed_wh=round(energy_wh, 4),
            useful_domestic_heat_wh=round(useful_heat_wh, 4),
            thermal_valorization_efficiency=round(heat_eff, 4),
            safe_thermal_envelope=safe_temp,
            datacenter_equivalent_wh=round(dc_wh, 4),
            net_avoided_co2_grams=round(avoided_co2_g, 4),
            co2_grams_per_1k_tokens_hivemind=round(net_hm_co2_g / k_tokens, 6),
            co2_grams_per_1k_tokens_datacenter=round(dc_co2_g / k_tokens, 6),
        )
