# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Windows Zero-Intrusion Daemon (GetLastInputInfo, Active Game Process & GPU Hardware Safety Monitor).

Monitors OS input hooks (GetLastInputInfo on Windows via ctypes), active process tables,
and optional Phase S-4 GPU Hardware Safety Circuit-Breaker telemetry.
If another Steam game launches, mouse/keyboard activity occurs within the 30-second idle
window, or the GPU hardware safety circuit-breaker trips (thermal/power/VRAM/battery),
signals the DiLoCo inner loop to dump its optimizer state to NVMe SSD (<1.5s) and
yield 100% of GPU VRAM.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes
import math
import sys
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import psutil

if TYPE_CHECKING:
    from hivemind.security.gpu_execution_circuit_breaker import GPUExecutionCircuitBreaker


class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", ctypes.wintypes.UINT),
        ("dwTime", ctypes.wintypes.DWORD),
    ]


KNOWN_GAME_EXECUTABLES = {
    "cs2.exe",
    "dota2.exe",
    "cyberpunk2077.exe",
    "eldenring.exe",
    "bg3.exe",
    "helldivers2.exe",
    "unrealengine.exe",
    "unity.exe",
}


class SETIAgentState(str, Enum):
    """Autonomous 3-State SETI-Swarm Agent Lifecycle."""

    STATE_1_GAMING_SEED_ONLY = "STATE_1_GAMING_SEED_ONLY"
    STATE_2_IDLE_SETI_MINER = "STATE_2_IDLE_SETI_MINER"
    STATE_3_SWARM_CONSUMER = "STATE_3_SWARM_CONSUMER"


@dataclass
class DaemonTelemetry:
    """Live snapshot of system idle time, active game process detection, and SETI-Swarm state."""

    os_idle_seconds: float
    idle_threshold_seconds: float
    active_game_detected: str | None
    should_yield_vram: bool
    reason: str
    seti_agent_state: str = SETIAgentState.STATE_2_IDLE_SETI_MINER.value
    hardware_circuit_breaker_tripped: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.os_idle_seconds, bool)
            or not isinstance(self.os_idle_seconds, (int, float))
            or not math.isfinite(float(self.os_idle_seconds))
            or float(self.os_idle_seconds) < 0.0
        ):
            raise ValueError("os_idle_seconds must be finite and >= 0")
        if (
            isinstance(self.idle_threshold_seconds, bool)
            or not isinstance(self.idle_threshold_seconds, (int, float))
            or not math.isfinite(float(self.idle_threshold_seconds))
            or float(self.idle_threshold_seconds) <= 0.0
        ):
            raise ValueError("idle_threshold_seconds must be finite and > 0")
        if not isinstance(self.should_yield_vram, bool):
            raise ValueError("should_yield_vram must be a boolean")
        if not isinstance(self.hardware_circuit_breaker_tripped, bool):
            raise ValueError("hardware_circuit_breaker_tripped must be a boolean")
        if isinstance(self.reason, bool) or not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")


def get_windows_idle_seconds() -> float:
    """Query Win32 user32.GetLastInputInfo to return seconds since last mouse/keyboard event."""
    if sys.platform != "win32":
        return 999.0
    try:
        lii = LASTINPUTINFO()
        lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
            tick_count = ctypes.windll.kernel32.GetTickCount()
            elapsed_ms = (tick_count - lii.dwTime) & 0xFFFFFFFF
            return round(elapsed_ms / 1000.0, 3)
    except Exception:
        pass
    return 999.0


def detect_running_game_process(extra_watchlist: set[str] | None = None) -> str | None:
    """Scan process table for active Steam games that require 100% of GPU VRAM."""
    if extra_watchlist is not None and (
        isinstance(extra_watchlist, (str, bytes))
        or not isinstance(extra_watchlist, (set, list, tuple, frozenset))
        or any(not isinstance(item, str) or not item.strip() for item in extra_watchlist)
    ):
        raise ValueError("extra_watchlist must be a collection of non-empty strings or None")
    watchlist = set(KNOWN_GAME_EXECUTABLES)
    if extra_watchlist:
        watchlist.update(n.lower() for n in extra_watchlist)

    for proc in psutil.process_iter(["name"]):
        try:
            pname = (proc.info.get("name") or "").lower()
            if pname in watchlist:
                return pname
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class ZeroIntrusionDaemon:
    """Monitors user input, game launches, and GPU hardware safety to protect gaming framerate and hardware."""

    def __init__(
        self,
        idle_threshold_seconds: float = 30.0,
        enforce_os_idle_hook: bool = False,
        simulated_idle_seconds: float | None = None,
        simulated_active_game: str | None = None,
        consumer_query_active: bool = False,
        hardware_circuit_breaker: GPUExecutionCircuitBreaker | None = None,
    ) -> None:
        if isinstance(idle_threshold_seconds, bool) or not isinstance(idle_threshold_seconds, (int, float)) or not math.isfinite(float(idle_threshold_seconds)) or float(idle_threshold_seconds) <= 0.0:
            raise ValueError("idle_threshold_seconds must be finite and > 0")
        if not isinstance(enforce_os_idle_hook, bool):
            raise ValueError("enforce_os_idle_hook must be a boolean")
        if not isinstance(consumer_query_active, bool):
            raise ValueError("consumer_query_active must be a boolean")
        if simulated_idle_seconds is not None and (
            isinstance(simulated_idle_seconds, bool)
            or not isinstance(simulated_idle_seconds, (int, float))
            or not math.isfinite(float(simulated_idle_seconds))
            or float(simulated_idle_seconds) < 0.0
        ):
            raise ValueError("simulated_idle_seconds must be finite and >= 0")
        if simulated_active_game is not None and (not isinstance(simulated_active_game, str) or not simulated_active_game.strip()):
            raise ValueError("simulated_active_game must be a non-empty string when provided")
        self.idle_threshold_seconds = float(idle_threshold_seconds)
        self.enforce_os_idle_hook = enforce_os_idle_hook
        self.simulated_idle_seconds = (
            float(simulated_idle_seconds) if simulated_idle_seconds is not None else None
        )
        self.simulated_active_game = simulated_active_game
        self.consumer_query_active = consumer_query_active
        self.hardware_circuit_breaker = hardware_circuit_breaker

    def poll(self) -> DaemonTelemetry:
        idle_sec = (
            self.simulated_idle_seconds
            if self.simulated_idle_seconds is not None
            else get_windows_idle_seconds()
        )
        active_game = (
            self.simulated_active_game
            if self.simulated_active_game is not None
            else detect_running_game_process()
        )

        if self.hardware_circuit_breaker is not None:
            cb_status = self.hardware_circuit_breaker.evaluate()
            if cb_status.should_preempt:
                return DaemonTelemetry(
                    os_idle_seconds=idle_sec,
                    idle_threshold_seconds=self.idle_threshold_seconds,
                    active_game_detected=active_game,
                    should_yield_vram=True,
                    reason=f"Hardware Circuit-Breaker [{cb_status.rule_id}]: {cb_status.reason}",
                    seti_agent_state=SETIAgentState.STATE_1_GAMING_SEED_ONLY.value,
                    hardware_circuit_breaker_tripped=True,
                )

        if active_game is not None:
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=active_game,
                should_yield_vram=True,
                reason=f"Steam game launched ({active_game}) — yielding 100% VRAM (background CAS seeding only).",
                seti_agent_state=SETIAgentState.STATE_1_GAMING_SEED_ONLY.value,
            )

        if self.consumer_query_active:
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=None,
                should_yield_vram=False,
                reason="Swarm Consumer mode active — querying P2P Sparse MoE ring with $HIVE credits.",
                seti_agent_state=SETIAgentState.STATE_3_SWARM_CONSUMER.value,
            )

        if self.enforce_os_idle_hook and (
            not math.isfinite(float(idle_sec))
            or float(idle_sec) < 0.0
            or float(idle_sec) < self.idle_threshold_seconds
        ):
            return DaemonTelemetry(
                os_idle_seconds=idle_sec,
                idle_threshold_seconds=self.idle_threshold_seconds,
                active_game_detected=None,
                should_yield_vram=True,
                reason=(
                    f"User input activity or invalid idle reading detected (idle {idle_sec}s < {self.idle_threshold_seconds:.1f}s)."
                    if not math.isfinite(float(idle_sec)) or float(idle_sec) < 0.0
                    else f"User input activity detected (idle {idle_sec:.1f}s < {self.idle_threshold_seconds:.1f}s)."
                ),
                seti_agent_state=SETIAgentState.STATE_1_GAMING_SEED_ONLY.value,
            )

        return DaemonTelemetry(
            os_idle_seconds=idle_sec,
            idle_threshold_seconds=self.idle_threshold_seconds,
            active_game_detected=None,
            should_yield_vram=False,
            reason="System idle / clear for background DiLoCo training & MoE expert hosting.",
            seti_agent_state=SETIAgentState.STATE_2_IDLE_SETI_MINER.value,
        )

    def resolve_seti_agent_state(self, consumer_query_active: bool | None = None) -> SETIAgentState:
        """Resolve the current 3-state SETI-Swarm lifecycle state."""
        if consumer_query_active is not None:
            if not isinstance(consumer_query_active, bool):
                raise ValueError("consumer_query_active must be a boolean")
            self.consumer_query_active = consumer_query_active
        telemetry = self.poll()
        return SETIAgentState(telemetry.seti_agent_state)

    def should_preempt(self) -> bool:
        return self.poll().should_yield_vram
