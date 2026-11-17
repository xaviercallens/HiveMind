# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind: Neural Breach — Gamified Hugging Face Hack-Miner & Free-to-Play Steam Edition."""

try:
    from hivemind.game.hf_hack_miner import (
        ALL_HF_BOUNTY_CONTRACTS,
        SEASON_1_HF_BOUNTY_CONTRACTS,
        SEASON_2_HF_BOUNTY_CONTRACTS,
        GamerMinerProfile,
        HackMiningGameEngine,
        HFBountyChallenge,
        HFDataReconScanner,
        MinerScoreBreakdown,
        execute_interactive_bounty_probe,
        execute_interactive_hf_shard_scan,
        execute_interactive_lora_micro_forge,
        run_hf_hack_mining_game,
    )
    from hivemind.game.rtx_steam_tool_deployer import (
        deploy_free_steam_rtx_tool_v1,
        run_intense_rtx_stress_suite,
    )
    from hivemind.game.steam_edition import (
        STEAM_APP_ID,
        build_free_steam_edition_package,
        create_hack_miner_fastapi_app,
    )
except (ImportError, ValueError):  # Slim router image without torch/numpy or incompatible system sklearn
    ALL_HF_BOUNTY_CONTRACTS = ()  # type: ignore[assignment]
    SEASON_1_HF_BOUNTY_CONTRACTS = ()  # type: ignore[assignment]
    SEASON_2_HF_BOUNTY_CONTRACTS = ()  # type: ignore[assignment]
    STEAM_APP_ID = 0  # type: ignore[assignment]
    GamerMinerProfile = None  # type: ignore[assignment,misc]
    HackMiningGameEngine = None  # type: ignore[assignment,misc]
    HFBountyChallenge = None  # type: ignore[assignment,misc]
    HFDataReconScanner = None  # type: ignore[assignment,misc]
    MinerScoreBreakdown = None  # type: ignore[assignment,misc]
    build_free_steam_edition_package = None  # type: ignore[assignment]
    create_hack_miner_fastapi_app = None  # type: ignore[assignment]
    deploy_free_steam_rtx_tool_v1 = None  # type: ignore[assignment]
    execute_interactive_bounty_probe = None  # type: ignore[assignment]
    execute_interactive_hf_shard_scan = None  # type: ignore[assignment]
    execute_interactive_lora_micro_forge = None  # type: ignore[assignment]
    run_hf_hack_mining_game = None  # type: ignore[assignment]
    run_intense_rtx_stress_suite = None  # type: ignore[assignment]

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



