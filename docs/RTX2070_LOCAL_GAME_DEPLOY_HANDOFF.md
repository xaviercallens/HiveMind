# Local RTX 2070 (8 GB VRAM) Scientific Game Deployment Handoff

> **Target Session**: Local RTX 2070 Agent / Terminal Session (`Claude Code` / `Antigravity CLI`)
> **Repository**: `https://github.com/xaviercallens/HiveMind`
> **Branch**: `feat/s19-scientific-game-unity-steam-antigravity` (**[PR #216](https://github.com/xaviercallens/HiveMind/pull/216)** -> `main`)
> **Live La Ruche Cloud Run**: `https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app`
> **Local Playable Game URL**: `http://127.0.0.1:8790/scientific-game`

---

## 1. One-Command Deployment on the Local RTX 2070 Machine

Execute the following commands in the local RTX 2070 session:

```bash
cd /home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/HiveMind || git clone https://github.com/xaviercallens/HiveMind.git && cd HiveMind
git fetch origin
git checkout feat/s19-scientific-game-unity-steam-antigravity
git pull origin feat/s19-scientific-game-unity-steam-antigravity

# Launch the RTX 2070 (8 GB) + Live HiveMind La Ruche Scientific Game Server
./scripts/play_scientific_game_rtx2070.sh
```

Or run the sync + server directly via Python:

```bash
PYTHONPATH=src uv run python -m hivemind.game.scientific_discovery_game --sync-la-ruche
PYTHONPATH=src uv run python -m hivemind.game.scientific_discovery_game --serve --host 127.0.0.1 --port 8790
```

---

## 2. Verified Hardware, Campaign 01 & 5-Player Cohort Endpoints

Once running on `http://127.0.0.1:8790/scientific-game`, the local RTX 2070 session exposes:

| Endpoint / Action | Method | Purpose |
| :--- | :---: | :--- |
| `/scientific-game` | `GET` | Playable **Google Stitch MCP** (`Cyber-Honeycomb Supercluster`) + Unity/WebGL Scientific Discovery Game HUD |
| `/api/scientific-game/rtx2070-la-ruche` | `GET` | Probes local RTX 2070 (`8192 MB` ceiling, `6645.8 MB` worker budget, `1546.2 MB` WDDM game headroom) + sends live producer heartbeat to La Ruche (`https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app`) |
| `/api/scientific-game/solve` | `POST` | Runs **Mission 1** (`MISSION-SCI-01-SYMPLECTIC-EXOPLANET`, $P_{\text{trial}}=11.23\text{ d}$, `+1200 PTS`), **Mission 2** (`MISSION-SCI-02-E8-LATTICE-QUORUM`, $f=0..3$, `+1350 PTS`), or **Mission 3** (`MISSION-SCI-03-ANSE-ENERGY-REACTOR`, `+1500 PTS`) |
| `/api/scientific-game/playtest-cohort` | `POST` | Runs the **5-Player End-User Gameplay Cohort Playtest** (`Xavier_RTX2070_FR`, `MatrixMagician_ROCm_Fedora`, `Vxtzq_CrowdGPT_Trunk`, `Blaich499_DirectML_Win11`, `SybilBot_ZeroSpend_VAC` -> `13,650 COHORT PTS`) |
| `/api/scientific-game/la-ruche-copilot` | `POST` | Queries live Specialist Bees (`hivemind/bee-stars`, `hivemind/coder`, `qwen2.5-coder:3b`) and evaluates ANSE Bee Energy $E_{\text{Bee}}(x,y)$ |
| `/api/scientific-game/preempt` | `POST` | Verifies `<1.5s` (`14.24 ms`) WDDM foreground game GPU preemption SLA |
