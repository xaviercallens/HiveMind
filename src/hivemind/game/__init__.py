# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind: Neural Breach — Gamified Hugging Face Hack-Miner, Free-to-Play Steam Edition & WebGPU Galaxy Cockpit.

Uses PEP 562 lazy attribute loading (`__getattr__`) so lightweight submodules like
`hivemind.game.webgpu_galaxy_cockpit` can be imported inside the zero-dependency
OpenRouter Cloud Run container without eagerly importing `torch` or `transformers`.
"""

from __future__ import annotations

from typing import Any

_HF_HACK_MINER_EXPORTS = {
    "ALL_HF_BOUNTY_CONTRACTS",
    "SEASON_1_HF_BOUNTY_CONTRACTS",
    "SEASON_2_HF_BOUNTY_CONTRACTS",
    "GamerMinerProfile",
    "HackMiningGameEngine",
    "HFBountyChallenge",
    "HFDataReconScanner",
    "MinerScoreBreakdown",
    "execute_interactive_bounty_probe",
    "execute_interactive_hf_shard_scan",
    "execute_interactive_lora_micro_forge",
    "run_hf_hack_mining_game",
}

_RTX_STEAM_EXPORTS = {
    "deploy_free_steam_rtx_tool_v1",
    "run_intense_rtx_stress_suite",
}

_STEAM_EDITION_EXPORTS = {
    "STEAM_APP_ID",
    "build_free_steam_edition_package",
    "create_hack_miner_fastapi_app",
}

__all__ = [
    "ALL_HF_BOUNTY_CONTRACTS",
    "SEASON_1_HF_BOUNTY_CONTRACTS",
    "SEASON_2_HF_BOUNTY_CONTRACTS",
    "STEAM_APP_ID",
    "GamerMinerProfile",
    "HFBountyChallenge",
    "HFDataReconScanner",
    "HackMiningGameEngine",
    "MinerScoreBreakdown",
    "build_free_steam_edition_package",
    "create_hack_miner_fastapi_app",
    "deploy_free_steam_rtx_tool_v1",
    "execute_interactive_bounty_probe",
    "execute_interactive_hf_shard_scan",
    "execute_interactive_lora_micro_forge",
    "run_hf_hack_mining_game",
    "run_intense_rtx_stress_suite",
]


def __getattr__(name: str) -> Any:
    if name in _HF_HACK_MINER_EXPORTS:
        from hivemind.game import hf_hack_miner

        return getattr(hf_hack_miner, name)
    if name in _RTX_STEAM_EXPORTS:
        from hivemind.game import rtx_steam_tool_deployer

        return getattr(rtx_steam_tool_deployer, name)
    if name in _STEAM_EDITION_EXPORTS:
        from hivemind.game import steam_edition

        return getattr(steam_edition, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
