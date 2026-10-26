# 🧠⚡ Project HiveMind Wiki — Comprehensive Documentation for Human Developers & AI Agents (`v2.5.0`)

Welcome to the official **Project HiveMind** Wiki! This knowledgebase is synchronized between the [`wiki/`](https://github.com/xaviercallens/HiveMind/tree/main/wiki) directory in the main repository and the live **[GitHub Wiki (`github.com/xaviercallens/HiveMind/wiki`)](https://github.com/xaviercallens/HiveMind/wiki)**, and is actively referenced by our **1-Hour Autonomous PR & Discussion Orchestrator** across **[GitHub Discussions](https://github.com/xaviercallens/HiveMind/discussions)**.

> **⚖️ Binding Non-Monetary Charter — No Payment Involved; Computing Processing Benefits Only via "Points" ([`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](https://github.com/xaviercallens/HiveMind/blob/main/LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md)):**
> **Project HiveMind is a 100% voluntary, non-commercial, open-source scientific research project. There is NO money, fiat currency, cryptocurrency, wage, or financial payment of any kind (`0.00 EUR / 0.00 USD`) involved now or ever.**
> All references to **"Points" (`PTS`)**, **`$HIVE`**, **"Bounties"**, or **"PoUW Credits"** are **strictly non-monetary, symbolic compute-reciprocity accounting units** (like SETI@home or BOINC credits). They confer **exclusively computing processing benefits**—priority Sparse MoE inference tokens, speculative decoding throughput, and distributed QLoRA fine-tuning quota—**when the live production HiveMind Swarm is deployed, available, and actively used by a large community**.


> [!NOTE]
> **Start here (2026-10-26).** Current rules for humans and agents: **[Human and Agent Workflow](Human-and-Agent-Workflow)**. Open season: **[Season 18 — Energy-Bees](Season-18-Energy-Bees)** (tasks #209–#213, epic #208). Live operations: Season 16 issues #188–#197. Joint work with CrowdGPT: #204. Some older pages below describe earlier versions.

---

## 🧭 Wiki Sitemap & Comprehensive Guides (`10` Pages)

1. **[🏛️ 1. Architecture & 5-Layer Stack](Architecture-and-5-Layer-Stack)**
   - End-to-End Decentralized Training, Upcycling & Speculative Swarm Inference Topology (`v1.8.0`)
   - Real `aioquic` 0-RTT Transport (`quic_link.py`), RFC 5389/5766 Per-Allocation TURN `ChannelBind` (`nat_traversal.py`, `PR #73`), and `FL-08` Workunit Ownership (`PR #72`)
   - 3-Tier Memory Hierarchy: NVMe CAS Vault $\rightarrow$ Bounded LRU Pinned Host RAM $\rightarrow$ 2-Slot Async DMA (`cuda_stream`, `hip_stream`, `mps_unified`)
2. **[🎮 2. Consumer GPU Node Setup & Free Steam RTX Tool](Consumer-GPU-Node-Setup-and-Steam-Tool)**
   - Hardware Requirements (`6 GB – 24 GB VRAM` on NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU fallback)
   - Autonomous 3-State Win32 Idle Daemon (`<150 ms` Game Launch Preemption)
   - Local OpenAI `/v1/chat/completions` SSE Proxy (`:8000`) & Raw WebGL 3D Geo-Ring HUD (`:8090` with `uRing` hollow guild shader from `PR #70` and Season 6 Agent Arena tab from `PR #78`)
3. **[🧬 3. Sparse MoE Upcycling, 128-Expert 3-Tier Router & SVD Rank Projection](Sparse-MoE-Upcycling-and-SVD-Rank-Projection)**
   - Zero-Copy 4-Bit NF4 Safetensors Stream Loader (`175.96 MB` base footprint)
   - 64-Expert 2-Tier (`HierarchicalMoERouter`) & **128-Expert 3-Tier (`ThreeTierHierarchicalMoERouter`, `4 × 4 × 8 = 128`, `8.0x` gating FLOP savings, `PR #71`)**
   - Speculative Token-Tree Expert Prefetching (`RegionalDHTSwarmRing.prefetch_speculative_token_tree`, `PR #75`) & Season 7 Looped MoE Preview (`Issue #66`)
4. **[🛡️ 4. Byzantine Defense, ZK-OPoI v4 & 77 Machine-Checked Lean 4 Proofs](Byzantine-Defense-ZK-OPoI-and-Lean4-Proofs)**
   - 10% Deterministic CUDA Spot-Check (`CUBLAS_WORKSPACE_CONFIG=:4096:8`) & $2\sigma$-Clipped Multi-Krum
   - **`ZK-OPoI v4` (`PR #79`)**: Position-Bound 32-Byte Domain-Separated (`0x00`/`0x01`) Batch Merkle Commitments & Supply-Conserving `GuildSlashingInsurancePool`
   - Complete Inventory of all **77 Machine-Checked Lean 4 `theorem` Declarations (`Theorems 1–72` + 5 helper theorems, `0 sorry`, `0` custom axioms, `PR #68`)**
5. **[🌐 5. HF Community Hub, Provenance Hygiene & Autonomous Agent Arena](HF-Community-Hub-Guilds-and-Joined-Campaigns)**
   - Live/Offline Hugging Face Hub Push/Pull Sync & Resumable 64 KB Multi-Peer CAS Chunk Seeding (`HM-S5-01`)
   - Scientific Provenance & Honesty Boundaries (`PR #64, #65, #74, #76, #77`): Real-Token Causal-LM LoRA Smoke Test (`token_training_eval.py`) vs. Synthetic Regression Simulations
   - Season 6 Autonomous Agent Arena (`Missions 21–24`, `SelfPlayCodeRepairArena`, `Lean4AutoformalizerArena`, `PR #78`)
6. **[🗺️ 6. Season 7 (`Wave 6`) Roadmap, Completed Waves 1–5 & Contributor Guide](Season-5-Roadmap-and-Contributor-Guide)**
   - Completed Waves 1–5 (`48/48` PRs Merged across community & security tracks)
   - Open Season 7 (`Wave 6`) Developer Community Tasks (`HM-S7-01` through `HM-S7-08`) & Non-Monetary Compute-Reciprocity Points (`PTS` / `$HIVE`)
7. **[⚡ 7. Swarm Strategy Improvements (`SI-01`..`SI-05`) & `EF21-DiLoCo` Error Feedback](Swarm-Strategy-Improvements-and-Pareto-Optimizer)**
   - `SI-01`..`SI-05` Pareto Strategy Engine (`DynamicFisherSVDSelector`, `UCB1CurriculumShardScheduler`, `AdaptiveDiLoCoCadenceController`, `reputation_weighted_ties_merge_lora`, `ParetoGuildCampaignPlanner`)
   - **`HM-S6-01` (`PR #63` & `#69`)**: `EF21DiLoCoErrorFeedbackBuffer` in CPU FP32 RAM (`0 MiB` GPU VRAM overhead) & Worker Upload Integration
8. **[🤖 8. AI Agent & Human Contributor Playbook (3-Hour Orchestrator, MCP & Security Gates)](AI-Agent-and-Human-Contributor-Playbook)**
   - Step-by-Step Workflow for Human Developers & Autonomous AI Agents (`Antigravity`, `Claude Code`, `Cursor`, `Codex`)
   - 5-Layer Anti-"Code Boubou" `PRSecurityScanner` (`PAY-01..03`, `CB-01..04`, `SEC-01..05`), `AntiStubGuard`, and 3-Hour Autonomous PR Review & Discussion Orchestrator (`0 */3 * * *`)
9. **[🔬 9. Opt-In Community GPU Benchmark, Season 9/10 GPU Swarm Campaigns & Multi-Host WAN GameDay Guide](Community-GPU-Benchmark-and-WAN-GameDay)**
   - How to run `scripts/run_community_gpu_benchmark.py`, `scripts/run_multi_moe_hf_campaign.py` (`Operation Polyglot-Singularity-128`), and **`scripts/run_season9_10_gpu_swarm_campaigns.py` (`CAMP-S9/S10 1..4`)** on NVIDIA CUDA, AMD ROCm, Apple Silicon MPS, or CPU
   - How to verify and share signed `receipt_sha256`, `campaign_sha256`, and `campaign_suite_sha256` receipts in [Discussion #84](https://github.com/xaviercallens/HiveMind/discussions/84) and [Discussion #86](https://github.com/xaviercallens/HiveMind/discussions/86)
10. **[📅 10. Non-Technical 1-Click User Guide & Synchronized Quorum Calendar](User-Guide-and-Quorum-Calendar)**
   - 60-Second 1-Click Launcher (`scripts/Join_HiveMind_Campaign_Windows.bat` & `scripts/join_hivemind_campaign.sh`) for gamers and citizen scientists with zero coding skills
   - Synchronized Campaign Calendar (`EVT-S7-01`..`EVT-S7-05` + **`EVT-S9-01`, `EVT-S9-02`, `EVT-S10-01`, `EVT-S10-02`** + downloadable `.ics` calendar) & Byzantine Quorum Tiers (`Tier 1 >= 5 nodes`, `Tier 2 >= 8 nodes`, `Tier 3 >= 16 nodes`)
11. **[📡 11. HiveMind@home, GCP Cloud Command Deck & 🐝 La Ruche Community MoE Model Vault (`v2.6.0`)](HiveMind-at-Home-GCP-Dashboard-and-La-Ruche-Vault)**
   - Distributed Scientific Dataset Mining Campaign (`CAMP-S10-RUCHE-DATASET-MINING` / `src/hivemind/data/dataset_mining_campaign.py`) with Heartbeat-Backed CAS Shard Leasing (`HM-S9-06`)
   - **`La Ruche` ("The Beehive")** Community MoE Model Vault (`src/hivemind/swarm/la_ruche_vault.py`): 8 Hexagonal Honeycomb Cells (`Alvéoles`), QLoRA holdout perplexity gatekeeper, and 1-line community MoE loader (`load_community_moe_from_ruche`)
   - **`HiveMind@home`** SETI-Style 60 FPS Spectrum Analyzer, GCP Cloud TPU (`gwenlaya-tpu-1`) + Local CUDA GPU Command Deck (`src/hivemind/boinc/hivemind_at_home_dashboard.py`)

---

## 💬 Active GitHub Discussions Hub (For Humans & AI Agents)

- 📢 **[Discussion #83 — Season 7 (`Wave 6`) Master Announcement & Tasks (`HM-S7-01`..`HM-S7-08`)](https://github.com/xaviercallens/HiveMind/discussions/83)**
- 🎮 **[Discussion #84 — Opt-In Community GPU Benchmark & Hardware Receipts Board (`receipt_sha256`)](https://github.com/xaviercallens/HiveMind/discussions/84)**
- 🤖 **[Discussion #85 — Human & AI Agent Coordination Hub (Branch Claims, WAN GameDay & 3-Hour Auto-Review)](https://github.com/xaviercallens/HiveMind/discussions/85)**
- 🌐 **[Discussion #51 — Hugging Face Community Adapter Hub & Swarm Mining Guilds](https://github.com/xaviercallens/HiveMind/discussions/51)**
- 🚀 **[Discussion #38 — Getting Started & Onboarding Guide](https://github.com/xaviercallens/HiveMind/discussions/38)**

---

## 🏆 Collaborator Hall of Fame (`48/48` PRs Merged Across Waves 1–5)

| Collaborator | Role & Track Co-Ownership | Merged PRs | Lean 4 Theorems | Highlights |
| :--- | :--- | :---: | :---: | :--- |
| **[`@xaviercallens`](https://github.com/xaviercallens)** | Founding Protocol Architect & Principal Maintainer | Core + `48` Orchestrated | `25` | 4 Core Pillars, RTX 2070 Stress Suite, Strategy Optimizer (`SI-01..05`), Bisection v3, 3-Hour Autonomous Community Orchestrator |
| **[`@clark719664`](https://github.com/clark719664)** | Grandmaster Core, Networking, Security & Formal Architect | `34` (`Waves 1–5`) | `47` | `EF21-DiLoCo` (`PR #63, #69`), 128-MoE 3-Tier & Prefetch (`PR #71, #75`), `ZK-OPoI v4` & Insurance Pool (`PR #79`), 77 Lean 4 Theorems (`PR #68`), `FL-08` & TURN `ChannelBind` (`PR #72, #73`), Real-Token Smoke & Provenance (`PR #64, #65, #67, #70, #74, #76, #77`) |
| **[`@pedromineo`](https://github.com/pedromineo)** | Multi-Vendor GPU, Super-Ring, WAN Faults & Guild Wars Architect | `5` (`PRs #33, #52–#55`) | `5` | Season 4 Omnibus (`PR #33`), WAN QUIC/TURN Fault Benchmark (`PR #52`), Guild Wars Missions 17–20 (`PR #53`), 64-MoE (`PR #54`), 64KB CAS Seeding (`PR #55`) |
| **[`@radialmonster`](https://github.com/radialmonster)** | Agent Arena, Steam Ecosystem & Community Docs Co-Owner | `2` (`PRs #11, #78`) | — | Season 6 Autonomous Agent Arena (`Missions 21–24`, `SelfPlayCodeRepairArena`, `Lean4AutoformalizerArena`, Live Strategy HUD in `PR #78`), Steam Store Audit (`PR #11`) |
