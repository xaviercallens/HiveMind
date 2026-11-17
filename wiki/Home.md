# 🧠⚡ Project HiveMind Wiki — Comprehensive Documentation for Human Developers & AI Agents (`v3.8.0`)

> [!WARNING]
> **Proof of concept (PoC): not a production service yet.** Until HiveMind is fully secured, send only **public, non-sensitive data**: no secrets, passwords, API keys, personal information, or private or enterprise data. See [SECURITY.md](https://github.com/xaviercallens/HiveMind/blob/main/SECURITY.md).

Welcome to the official **Project HiveMind** Wiki! This knowledgebase is synchronized between the [`wiki/`](https://github.com/xaviercallens/HiveMind/tree/main/wiki) directory in the main repository and the live **[GitHub Wiki (`github.com/xaviercallens/HiveMind/wiki`)](https://github.com/xaviercallens/HiveMind/wiki)**, and is actively referenced by our **Autonomous PR & Discussion Orchestrators** across **[GitHub Discussions](https://github.com/xaviercallens/HiveMind/discussions)**.

> **⚖️ Binding Non-Monetary Charter — No Payment Involved; Computing Processing Benefits Only via "Points" ([`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](https://github.com/xaviercallens/HiveMind/blob/main/LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md)):**
> **Project HiveMind is a 100% voluntary, non-commercial, open-source scientific research project. There is NO money, fiat currency, cryptocurrency, wage, or financial payment of any kind (`0.00 EUR / 0.00 USD`) involved now or ever.**
> All references to **"Points" (`PTS`)**, **`$HIVE`**, **`HMT`**, or **"PoUW Credits"** are **strictly non-monetary, symbolic compute-reciprocity accounting units** (like SETI@home or BOINC credits). They confer **exclusively computing processing benefits**—priority Sparse MoE inference tokens, speculative decoding throughput, and distributed QLoRA fine-tuning quota—**when the live production HiveMind Swarm is deployed, available, and actively used by a large community**.

> [!IMPORTANT]
> **🌟 LATEST MILESTONE & VISION (2026-10-26 — `v3.8.0`):**
> - **Must Read:** **[🌟 Live Results, Engineering Lessons Learned & Community Gratitude](Live-Results-Lessons-Learned-and-Roadmap)** — Complete report on our 100 assimilated BOINC MoE workunits, Dual-GPU live serving (Tesla T4 + RTX 2070), real user telemetry (47 accounts, 1,427 completions), unvarnished lessons learned, and heartfelt thanks to all contributors!
> - Current Rules for Humans and Agents: **[Human and Agent Workflow](Human-and-Agent-Workflow)**.
> - Active Roadmap: **[Season 18 — Energy-Bees](Season-18-Energy-Bees)** (tasks #209–#213, epic #208) & **[Season 17 — CrowdSwarm Federation](https://github.com/xaviercallens/HiveMind/issues/204)**.
> - Live Portals: **[Web Portal (GitHub Pages)](https://xaviercallens.github.io/HiveMindatHome.github.io/)** · **[La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/)** · **[OpenRouter API](https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app)** · **[Zenodo v3.8.0 DOI: 10.5281/zenodo.23649757](https://doi.org/10.5281/zenodo.23649757)**.

---

## 🧭 Wiki Sitemap & Comprehensive Guides (`13` Master Pages)

0. **[🌟 0. Live Results, Lessons Learned & Gratitude](Live-Results-Lessons-Learned-and-Roadmap)** — Measured 100 BOINC MoE workunits, Dual-GPU operations, 47 accounts, lessons learned, and contributor acknowledgments.
1. **[🏛️ 1. Architecture & 5-Layer Stack](Architecture-and-5-Layer-Stack)**
   - End-to-End Decentralized Training, Upcycling & Speculative Swarm Inference Topology (`v1.8.0`)
   - Real `aioquic` 0-RTT Transport (`quic_link.py`), RFC 5389/5766 Per-Allocation TURN `ChannelBind` (`nat_traversal.py`, `PR #73`), and `FL-08` Workunit Ownership (`PR #72`)
   - 3-Tier Memory Hierarchy: NVMe CAS Vault $\rightarrow$ Bounded LRU Pinned Host RAM $\rightarrow$ 2-Slot Async DMA (`cuda_stream`, `hip_stream`, `mps_unified`)
2. **[🎮 2. Consumer GPU Node Setup & Free Steam RTX Tool](Consumer-GPU-Node-Setup-and-Steam-Tool)**
   - Hardware Requirements (`6 GB – 24 GB VRAM` on NVIDIA CUDA, AMD ROCm, Apple MPS, or CPU fallback)
   - Autonomous Win32/Linux Idle Daemon (`<1.13s` Game & Interactive Preemption SLA)
   - Local OpenAI `/v1/chat/completions` SSE Proxy (`:8000`) & WebGL 3D Geo-Ring HUD
3. **[🧬 3. Sparse MoE Upcycling, 128-Expert 3-Tier Router & SVD Rank Projection](Sparse-MoE-Upcycling-and-SVD-Rank-Projection)**
   - Zero-Copy 4-Bit NF4 Safetensors Stream Loader (`175.96 MB` base footprint)
   - 64-Expert 2-Tier & 128-Expert 3-Tier Routers (`8.0x` gating FLOP savings)
4. **[🛡️ 4. Byzantine Defense, ZK-OPoI v4 & 128 Machine-Checked Lean 4 Proofs](Byzantine-Defense-ZK-OPoI-and-Lean4-Proofs)**
   - 10% Deterministic CUDA Spot-Check (`CUBLAS_WORKSPACE_CONFIG=:4096:8`) & $2\sigma$-Clipped Multi-Krum
   - **`ZK-OPoI v4`**: Position-Bound 32-Byte Domain-Separated Batch Merkle Commitments
   - Complete Inventory of all **128 Machine-Checked Lean 4 `theorem` Declarations (`0 sorry`, `0` custom axioms)**
5. **[🌐 5. HF Community Hub, Provenance Hygiene & Autonomous Agent Arena](HF-Community-Hub-Guilds-and-Joined-Campaigns)**
   - Live/Offline Hugging Face Hub Push/Pull Sync & Resumable 64 KB Multi-Peer CAS Chunk Seeding
   - Scientific Provenance & Honesty Boundaries: Real-Token Causal-LM LoRA Smoke Test vs. Synthetic Simulations
6. **[🗺️ 6. Season Roadmap & Contributor Guide](Season-5-Roadmap-and-Contributor-Guide)**
   - Roadmap progression across Seasons 1–19, completed waves, and active milestones.
7. **[⚡ 7. Swarm Strategy Improvements & Next-Gen DiLoCo](Swarm-Strategy-Improvements-and-Pareto-Optimizer)**
   - Pareto Strategy Engine & `EF21DiLoCoErrorFeedbackBuffer` in CPU FP32 RAM.
8. **[🤖 8. AI Agent & Human Contributor Playbook](AI-Agent-and-Human-Contributor-Playbook)**
   - Collaboration guidelines for Antigravity, Claude Code, Cursor, and human developers.
9. **[🔬 9. Community GPU Benchmark & WAN GameDay Guide](Community-GPU-Benchmark-and-WAN-GameDay)**
   - Running community benchmarks and generating verified hardware receipts.
10. **[📅 10. Non-Technical 1-Click User Guide & Quorum Calendar](User-Guide-and-Quorum-Calendar)**
    - 60-Second 1-Click Launcher for gamers and citizen scientists.
11. **[📡 11. HiveMind@home, GCP Cloud Command Deck & La Ruche Model Vault](HiveMind-at-Home-GCP-Dashboard-and-La-Ruche-Vault)**
    - Live command deck, Kelly market clearing, Litestream SQLite ledger, and honeycomb cell vault.
12. **[🧬 12. Season 18 — Operation Energy-Bees](Season-18-Energy-Bees)**
    - Physical energy function ($E$), execution sandbox, Banach fixed-point hot swapping, and warm verifiers.
13. **[🐝 13. Community MoE Training, Custom Bees & Open-Source Datasets](Community-MoE-Training-and-Custom-Bee-Guide)**
    - "Bring Your Own Bee & Dataset" decentralized training under 8 GB VRAM (NF4 Gemma & Qwen 3.5), prorated fair-share leaderboard, and public open-source charter.

---

## 🏆 Collaborator Hall of Honor (`217` Merged Pull Requests)

| Collaborator | Role & Contribution Scope | Highlights |
| :--- | :--- | :--- |
| **[`@xaviercallens`](https://github.com/xaviercallens)** | Founding Protocol Architect & Principal Maintainer | Conception of HiveMind, 4 Core Pillars, ANSE autopoietic physics, RTX 2070 stress suites, strategy optimizer (`SI-01..05`), and project stewardship. |
| **Claude Code Agent** | Autonomous AI Co-Developer | Co-developer on over 100 Pull Requests (`PR #180`–`#217`), implementing router multi-endpoint dispatch, Litestream single-writer resilience, and Season 18 Energy-Bee specifications. |
| **Antigravity CLI Agent** | Autonomous Neuro-Symbolic Agent | Zero-trust execution attestation, 25-vector red-team gate enforcement, 128 Lean 4 proof validations, and autonomous 30m/2h live monitoring. |
| **`Pioneer #1` (`user.913f8b6732605708`)** | Pioneer Citizen Prosumer | First public prosumer node, serving **205 live completions** on dedicated hardware and earning 844+ HMT in compute reciprocity. |
| **[`@clark719664`](https://github.com/clark719664)** | Grandmaster Core, Networking & Formal Verification | `44` merged PRs, `47` Lean 4 theorems, `EF21-DiLoCo`, 128-MoE 3-tier router, and TURN ChannelBind. |
| **[`@pedromineo`](https://github.com/pedromineo)** | Multi-Vendor GPU & Fault Resiliency | `5` merged PRs (`#33`, `#52`–`#55`), WAN QUIC fault-injection, and 64KB CAS seeding. |
| **[`@radialmonster`](https://github.com/radialmonster)** | Agent Arena & Steam Store Tooling | `2` merged PRs (`#11`, `#78`), Season 6 Agent Arena HUD, and Steam tooling audits. |
| **[`@lyrasol99-dev`](https://github.com/lyrasol99-dev)** | Telemetry & Infrastructure | Infrastructure stability reviews and dashboard metrics. |
| **`cybervalkyrie-fr-5ecb`** | Community Prosumer | Rapid onboarding, producer node contribution, and validation of `hivemind/bee-stars`. |
| **`u/artsnumeriques`** | Community Champion (`r/artificielle`) | Essential French community outreach guidance, ethical citizen AI advocacy, and forum moderation feedback. |
