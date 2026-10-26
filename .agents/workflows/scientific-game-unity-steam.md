---
description: Workflow to download, configure, and verify the Unity + Steamworks.NET + Unity-MCP + Google Antigravity Scientific Discovery Game on Project HiveMind
---

# HiveMind Scientific Game Workflow (Unity + Steamworks.NET + Google Antigravity SDK)

This workflow orchestrates the integration of:
1. **`SebLague/Solar-System` (MIT)** — N-Body Orbital Physics (`Universe.cs`, `GravityObject.cs`, `CelestialBody.cs`, `NBodySimulation.cs`, `OrbitDebugDisplay.cs`, `ShipHUD.cs`) upgraded with 2nd-order Symplectic Velocity-Verlet (`HiveMindScientificGameBridge.cs`) to enforce $|\Delta H / H_0| \le 10^{-4}$.
2. **`rlabrecque/Steamworks.NET` (`com.rlabrecque.steamworks.net` v2025.165.0, MIT)** — C# Steamworks wrapper coupled with `src/hivemind/client/steamworks.py` (`SteamworksSDKBridge`) for Non-Limited Sybil resistance, Rich Presence, and Achievements.
3. **`CoplayDev/unity-mcp` (MIT) + Google Antigravity SDK (`google-antigravity`)** — `LocalOpenAIAgentConfig` / `LiteRTAgentConfig` connecting directly to HiveMind's OpenRouter (`https://hivemind-openrouter-1003063861791.us-central1.run.app/v1`) and Specialist Bees (`hivemind/bee-stars`, `hivemind/coder`, `hivemind/crowd-v1`).
4. **Python & WebGL Scientific Discovery Game Engine (`src/hivemind/game/scientific_discovery_game.py`)** — Playable in-browser at `/scientific-game` or via the Steam HUD (`hivemind serve-hack-hud`).

## Step 1: Verify Downloaded Unity + Steamworks.NET + Solar-System Assets
```bash
ls -la unity_steam_client/Assets/Scripts/SolarSystem/
ls -la unity_steam_client/Assets/Scripts/HiveMind/
ls -la unity_steam_client/Packages/com.rlabrecque.steamworks.net/
```

## Step 2: Run the 3-Mission Scientific Discovery Odyssey Engine
```bash
PYTHONPATH=src uv run python -m hivemind.game.scientific_discovery_game
```

## Step 3: Run Unit & Integration Tests + Anti-Stub Gate
```bash
PYTHONPATH=src uv run pytest tests/test_scientific_discovery_game.py -q
uv run python scripts/gwaya_stub_audit.py
```
