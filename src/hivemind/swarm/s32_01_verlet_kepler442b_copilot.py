# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HM-S32-01 — Symplectic Velocity-Verlet N-Body Orbital Integrator for KEPLER-HM-442b & Google Antigravity SDK Co-Pilot Bridge.

Starts the 10-PR self-evolving roadmap with a foundational Python scientific & SDK integration module:
1. Ensures ``google.antigravity.LocalOpenAIAgentConfig(...).lightweight()`` works seamlessly across SDK versions.
2. Implements the symplectic Velocity-Verlet (Kick-Drift-Kick) orbital integrator for exoplanet ``KEPLER-HM-442b``
   (semi-major axis ``a = 0.409 AU``, eccentricity ``e = 0.04``, host star ``M = 0.61 M_sun``), proving
   bounded Hamiltonian drift ``|dH / H_0| < 1e-6`` and angular momentum conservation ``|dL_z / L_{z,0}| < 1e-12``.
3. Provides :class:`AntigravityHiveMindCoPilot` connecting the Google Antigravity SDK and ``hivemind/bee-stars``
   with the caller's own HiveMind key (``api_key=`` or ``HIVEMIND_API_KEY``); no key is built into this module.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import httpx

DEFAULT_HIVEMIND_BASE_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1"
HIVEMIND_API_KEY_ENV = "HIVEMIND_API_KEY"
DEFAULT_BEE_STARS_MODEL = "hivemind/bee-stars"

# Canonical gravitational parameter in AU^3 / year^2: G * M_sun = 4 * pi^2
FOUR_PI_SQ = 4.0 * math.pi * math.pi
KEPLER_442_STAR_MASS_SOLAR = 0.61
KEPLER_442B_SEMI_MAJOR_AXIS_AU = 0.409
KEPLER_442B_ECCENTRICITY = 0.04


@dataclass
class FallbackLocalOpenAIAgentConfig:
    """Lightweight compatibility config when ``google-antigravity`` is not installed in an isolated venv/CI."""

    model: str
    base_url: str
    is_lightweight: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.base_url, str) or not self.base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(self.is_lightweight, bool):
            raise ValueError("is_lightweight must be a boolean")

    def lightweight(self) -> FallbackLocalOpenAIAgentConfig:
        self.is_lightweight = True
        return self


def ensure_antigravity_sdk_lightweight_compat() -> bool:
    """Ensure ``LocalOpenAIAgentConfig.lightweight()`` is available in ``google.antigravity`` (or fallback shim)."""
    import site
    import sys

    user_site = site.getusersitepackages()
    if user_site and user_site not in sys.path:
        sys.path.append(user_site)
    try:
        from google.antigravity import CapabilitiesConfig, LocalOpenAIAgentConfig
        from google.antigravity.connections.local.local_connection_config import BaseLocalAgentConfig
    except ImportError:
        return hasattr(FallbackLocalOpenAIAgentConfig, "lightweight")

    if not hasattr(BaseLocalAgentConfig, "lightweight"):
        def _lightweight(self: Any) -> Any:
            self.capabilities = CapabilitiesConfig(
                enabled_tools=[],
                subagents=False,
                mcp=False,
            )
            return self

        BaseLocalAgentConfig.lightweight = _lightweight  # type: ignore[attr-defined]
    return hasattr(LocalOpenAIAgentConfig, "lightweight")


@dataclass(frozen=True)
class Kepler442bOrbitalState:
    """2D barycentric orbital state in (AU, AU/year)."""

    x_au: float
    y_au: float
    vx_au_yr: float
    vy_au_yr: float
    t_yr: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("x_au", "y_au", "vx_au_yr", "vy_au_yr", "t_yr"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{field_name} must be a finite number")
        if math.hypot(self.x_au, self.y_au) <= 0.0:
            raise ValueError("orbital radius must be positive")

    @property
    def radius_au(self) -> float:
        return math.hypot(self.x_au, self.y_au)

    @property
    def speed_au_yr(self) -> float:
        return math.hypot(self.vx_au_yr, self.vy_au_yr)


@dataclass(frozen=True)
class SymplecticBenchmarkReport:
    """Comparison of Symplectic Velocity-Verlet vs Forward Euler on KEPLER-HM-442b."""

    target_body: str
    steps: int
    dt_yr: float
    total_simulated_yr: float
    initial_energy: float
    final_energy_verlet: float
    max_relative_energy_drift_verlet: float
    max_relative_energy_drift_euler: float
    verlet_over_euler_improvement_ratio: float
    angular_momentum_relative_drift: float
    periapsis_au: float
    apoapsis_au: float
    symplectic_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SymplecticVelocityVerletIntegrator:
    """Second-order symplectic Velocity-Verlet (Kick-Drift-Kick) orbital integrator for KEPLER-HM-442b."""

    def __init__(
        self,
        star_mass_solar: float = KEPLER_442_STAR_MASS_SOLAR,
        semi_major_axis_au: float = KEPLER_442B_SEMI_MAJOR_AXIS_AU,
        eccentricity: float = KEPLER_442B_ECCENTRICITY,
    ) -> None:
        if (
            isinstance(star_mass_solar, bool)
            or not isinstance(star_mass_solar, (int, float))
            or not math.isfinite(star_mass_solar)
            or star_mass_solar <= 0.0
        ):
            raise ValueError("star_mass_solar must be a positive finite number")
        if (
            isinstance(semi_major_axis_au, bool)
            or not isinstance(semi_major_axis_au, (int, float))
            or not math.isfinite(semi_major_axis_au)
            or semi_major_axis_au <= 0.0
        ):
            raise ValueError("semi_major_axis_au must be a positive finite number")
        if (
            isinstance(eccentricity, bool)
            or not isinstance(eccentricity, (int, float))
            or not math.isfinite(eccentricity)
            or not (0.0 <= eccentricity < 1.0)
        ):
            raise ValueError("eccentricity must be a finite number in [0, 1)")
        self.star_mass_solar = float(star_mass_solar)
        self.semi_major_axis_au = float(semi_major_axis_au)
        self.eccentricity = float(eccentricity)
        self.mu = FOUR_PI_SQ * self.star_mass_solar

    def periapsis_initial_state(self) -> Kepler442bOrbitalState:
        """Return exact periapsis state for KEPLER-HM-442b."""
        r_p = self.semi_major_axis_au * (1.0 - self.eccentricity)
        v_p = math.sqrt(
            self.mu * (1.0 + self.eccentricity) / (self.semi_major_axis_au * (1.0 - self.eccentricity))
        )
        return Kepler442bOrbitalState(x_au=r_p, y_au=0.0, vx_au_yr=0.0, vy_au_yr=v_p, t_yr=0.0)

    def acceleration(self, x_au: float, y_au: float) -> tuple[float, float]:
        if (
            isinstance(x_au, bool)
            or not isinstance(x_au, (int, float))
            or not math.isfinite(float(x_au))
            or isinstance(y_au, bool)
            or not isinstance(y_au, (int, float))
            or not math.isfinite(float(y_au))
        ):
            raise ValueError("x_au and y_au must be finite numbers")
        r2 = float(x_au) * float(x_au) + float(y_au) * float(y_au)
        if r2 <= 0.0:
            raise ValueError("orbital radius must be positive")
        r = math.sqrt(r2)
        inv_r3 = -self.mu / (r2 * r)
        return float(x_au) * inv_r3, float(y_au) * inv_r3

    def specific_energy(self, state: Kepler442bOrbitalState) -> float:
        kinetic = 0.5 * (state.vx_au_yr * state.vx_au_yr + state.vy_au_yr * state.vy_au_yr)
        potential = -self.mu / state.radius_au
        return kinetic + potential

    @staticmethod
    def specific_angular_momentum(state: Kepler442bOrbitalState) -> float:
        return state.x_au * state.vy_au_yr - state.y_au * state.vx_au_yr

    def step_velocity_verlet(self, state: Kepler442bOrbitalState, dt_yr: float) -> Kepler442bOrbitalState:
        """Execute one symplectic Kick-Drift-Kick Velocity-Verlet orbital step."""
        if not isinstance(state, Kepler442bOrbitalState):
            raise ValueError("state must be a Kepler442bOrbitalState")
        if isinstance(dt_yr, bool) or not isinstance(dt_yr, (int, float)) or not math.isfinite(dt_yr) or dt_yr <= 0.0:
            raise ValueError("dt_yr must be a positive finite number")
        ax0, ay0 = self.acceleration(state.x_au, state.y_au)
        half_dt = 0.5 * dt_yr
        vx_half = state.vx_au_yr + half_dt * ax0
        vy_half = state.vy_au_yr + half_dt * ay0

        x_next = state.x_au + dt_yr * vx_half
        y_next = state.y_au + dt_yr * vy_half

        ax1, ay1 = self.acceleration(x_next, y_next)
        vx_next = vx_half + half_dt * ax1
        vy_next = vy_half + half_dt * ay1
        return Kepler442bOrbitalState(
            x_au=x_next,
            y_au=y_next,
            vx_au_yr=vx_next,
            vy_au_yr=vy_next,
            t_yr=state.t_yr + dt_yr,
        )

    def step_forward_euler(self, state: Kepler442bOrbitalState, dt_yr: float) -> Kepler442bOrbitalState:
        """Non-symplectic 1st-order Forward Euler step used as baseline comparison."""
        if not isinstance(state, Kepler442bOrbitalState):
            raise ValueError("state must be a Kepler442bOrbitalState")
        if isinstance(dt_yr, bool) or not isinstance(dt_yr, (int, float)) or not math.isfinite(dt_yr) or dt_yr <= 0.0:
            raise ValueError("dt_yr must be a positive finite number")
        ax0, ay0 = self.acceleration(state.x_au, state.y_au)
        return Kepler442bOrbitalState(
            x_au=state.x_au + dt_yr * state.vx_au_yr,
            y_au=state.y_au + dt_yr * state.vy_au_yr,
            vx_au_yr=state.vx_au_yr + dt_yr * ax0,
            vy_au_yr=state.vy_au_yr + dt_yr * ay0,
            t_yr=state.t_yr + dt_yr,
        )

    def benchmark_kepler442b_orbit(
        self,
        steps: int = 2000,
        dt_yr: float = 0.00015,
    ) -> SymplecticBenchmarkReport:
        """Integrate KEPLER-HM-442b and verify symplectic energy & angular momentum invariants."""
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps < 10
            or isinstance(dt_yr, bool)
            or not isinstance(dt_yr, (int, float))
            or not math.isfinite(dt_yr)
            or dt_yr <= 0.0
        ):
            raise ValueError("steps must be >= 10 and dt_yr > 0")
        s_verlet = self.periapsis_initial_state()
        s_euler = s_verlet
        h0 = self.specific_energy(s_verlet)
        lz0 = self.specific_angular_momentum(s_verlet)

        max_drift_verlet = 0.0
        max_drift_euler = 0.0
        max_lz_drift = 0.0
        min_r = s_verlet.radius_au
        max_r = s_verlet.radius_au

        for _ in range(steps):
            s_verlet = self.step_velocity_verlet(s_verlet, dt_yr)
            s_euler = self.step_forward_euler(s_euler, dt_yr)

            hv = self.specific_energy(s_verlet)
            he = self.specific_energy(s_euler)
            lzv = self.specific_angular_momentum(s_verlet)

            max_drift_verlet = max(max_drift_verlet, abs((hv - h0) / h0))
            max_drift_euler = max(max_drift_euler, abs((he - h0) / h0))
            max_lz_drift = max(max_lz_drift, abs((lzv - lz0) / lz0))
            r = s_verlet.radius_au
            min_r = min(min_r, r)
            max_r = max(max_r, r)

        ratio = max_drift_euler / max(max_drift_verlet, 1e-16)
        gate_ok = max_drift_verlet < 1e-6 and max_lz_drift < 1e-12 and ratio > 100.0
        return SymplecticBenchmarkReport(
            target_body="KEPLER-HM-442b",
            steps=steps,
            dt_yr=dt_yr,
            total_simulated_yr=round(steps * dt_yr, 6),
            initial_energy=h0,
            final_energy_verlet=self.specific_energy(s_verlet),
            max_relative_energy_drift_verlet=max_drift_verlet,
            max_relative_energy_drift_euler=max_drift_euler,
            verlet_over_euler_improvement_ratio=ratio,
            angular_momentum_relative_drift=max_lz_drift,
            periapsis_au=min_r,
            apoapsis_au=max_r,
            symplectic_gate_passed=gate_ok,
        )


class AntigravityHiveMindCoPilot:
    """Co-Pilot bridge between Google Antigravity SDK (``LocalOpenAIAgentConfig``) and HiveMind OpenRouter."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_HIVEMIND_BASE_URL,
        model: str = DEFAULT_BEE_STARS_MODEL,
    ) -> None:
        # The key is the caller's: an explicit argument or the HIVEMIND_API_KEY variable. There is no built-in key.
        if api_key is not None and not isinstance(api_key, str):
            raise ValueError("api_key must be None or a string")
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base_url must be a non-empty string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty string")
        api_key = (api_key or os.environ.get(HIVEMIND_API_KEY_ENV, "")).strip()
        if not api_key:
            raise ValueError(f"api_key is required: pass api_key= or set {HIVEMIND_API_KEY_ENV}; no key is built in")
        if not api_key.startswith("hm-"):
            raise ValueError("api_key must be a valid HiveMind 'hm-...' key")
        ensure_antigravity_sdk_lightweight_compat()
        self.api_key = api_key
        cleaned_url = re.sub(r"/k/[^/]+", "", base_url.rstrip("/"))
        self.base_url = cleaned_url
        self.model = model

    @property
    def authenticated_base_url(self) -> str:
        """Header-only authentication base URL (never embeds ``api_key`` in the URL path)."""
        return self.base_url

    def build_antigravity_config(self, *, use_key_path: bool = False) -> Any:
        """Construct ``LocalOpenAIAgentConfig(model=..., base_url=...).lightweight()``."""
        ensure_antigravity_sdk_lightweight_compat()
        url = self.authenticated_base_url if use_key_path else self.base_url
        try:
            from google.antigravity import LocalOpenAIAgentConfig

            return LocalOpenAIAgentConfig(model=self.model, base_url=url).lightweight()
        except ImportError:
            return FallbackLocalOpenAIAgentConfig(model=self.model, base_url=url).lightweight()

    async def chat_via_antigravity_agent(
        self,
        prompt: str = "Optimize our Symplectic Velocity-Verlet N-Body orbital step for KEPLER-HM-442b.",
    ) -> dict[str, Any]:
        """Execute a live chat turn through ``google.antigravity.Agent`` with ``LocalOpenAIAgentConfig``."""
        from google.antigravity import Agent

        config = self.build_antigravity_config(use_key_path=False)
        async with Agent(config=config) as agent:
            reply = await agent.chat(prompt)
            text = (await reply.text()).strip()
        integrator = SymplecticVelocityVerletIntegrator()
        bench = integrator.benchmark_kepler442b_orbit()
        return {
            "sdk": "google-antigravity",
            "model": self.model,
            "base_url": self.base_url,
            "prompt": prompt,
            "reply_text": text,
            "symplectic_verification": bench.to_dict(),
        }

    def query_openrouter_direct(
        self,
        prompt: str,
        *,
        model: str | None = None,
        max_tokens: int = 240,
        timeout_s: float = 45.0,
    ) -> dict[str, Any]:
        """Send an authenticated OpenRouter request using ``self.api_key`` and return completion telemetry."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        if model is not None and (not isinstance(model, str) or not model.strip()):
            raise ValueError("model must be a non-empty string when provided")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer >= 1")
        if isinstance(timeout_s, bool) or not isinstance(timeout_s, (int, float)) or not math.isfinite(float(timeout_s)) or float(timeout_s) <= 0.0:
            raise ValueError("timeout_s must be a positive finite number")
        target_model = model or self.model
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "model": target_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        hm = data.get("hivemind") or {}
        return {
            "completion_id": data.get("id", ""),
            "request_id": hm.get("request_id", ""),
            "routed_model": data.get("model", target_model),
            "charged_hmt": hm.get("charged_hmt", 0.0),
            "producer_id": hm.get("producer_id", ""),
            "content": str(msg.get("content") or "").strip(),
        }
