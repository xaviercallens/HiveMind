<div align="center">

<img src="docs/assets/hivemind_hero_banner.svg" alt="Project HiveMind — Decentralized Sparse MoE Training & Inference on Consumer Gaming GPUs" width="100%"/>

<br/>

[![Release](https://img.shields.io/badge/Release-v3.8.0%20(Season%2016%20Live%20Ops%20%26%20201%2B%20PRs)-00f0ff?style=for-the-badge)](https://github.com/xaviercallens/HiveMind/releases)
[![Live OpenRouter](https://img.shields.io/badge/Live_OpenRouter_Gateway-7_Models_Online-10b981?style=for-the-badge&logo=googlecloud)](https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app)
[![Live La Ruche](https://img.shields.io/badge/La_Ruche_Portal-Live_on_Cloud_Run-f59e0b?style=for-the-badge&logo=googlecloud)](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app)
[![HiveMind@home Web](https://img.shields.io/badge/HiveMind%40home_Site-Live_5--Language_Portal-a855f7?style=for-the-badge&logo=github)](https://xaviercallens.github.io/HiveMindatHome.github.io/)
[![Test Suite](https://img.shields.io/badge/Tests-Full%20suite%20in%20CI%20(coverage%20gate%2095%25)-brightgreen?style=for-the-badge)](.github/workflows/ci_swarm_tests.yml)
[![Lean 4 Verified](https://img.shields.io/badge/Lean%204-Machine--checked%20theorems%20(0%20sorry%2C%200%20user%20axioms)-005cc5?style=for-the-badge)](lean4/HiveMindSpec.lean)
[![Foundation Paper](https://img.shields.io/badge/Foundation%20Paper-v3.8.0%20LaTeX%20%26%20PDF%20(self--assessed%2C%20scripted%20audit%2C%20not%20peer%20reviewed)-b31b1b?style=for-the-badge)](paper/hivemind_foundation_article.pdf)
[![Zenodo DOI](https://img.shields.io/badge/Zenodo%20DOI-10.5281%2Fzenodo.23649757-1682d4?style=for-the-badge)](https://doi.org/10.5281/zenodo.23649757)
[![Hardware Verified](https://img.shields.io/badge/Verified%20Dual--GPU-Tesla%20T4%20(16GB%20MoE)%20%2B%20RTX%202070%20(8GB%20Inference)-76b900?style=for-the-badge)](artifacts/rtx_intense_stress_and_steam_deploy_report.json)
[![Community PRs](https://img.shields.io/badge/Community%20PRs-63%2F63%20Merged%20(Waves%201--7%20%7C%201h%20Orchestrator)-a855f7?style=for-the-badge)](docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md)
[![Project Wiki](https://img.shields.io/badge/Project%20Wiki-13%20Pages%20Live-f59e0b?style=for-the-badge)](https://github.com/xaviercallens/HiveMind/wiki)
[![Strategy Engine](https://img.shields.io/badge/Strategy%20Engine-Pareto%20SI--01..11%20%2B%20H1..H10%20%2B%20E1..E4-ffd21e?style=for-the-badge)](docs/DILOCO_ARXIV_LITERATURE_AND_IMPROVEMENTS.md)
[![Branch Protection](https://img.shields.io/badge/Main%20%26%20Tags-Protected%20%2B%20Immutable-10b981?style=for-the-badge)](GOVERNANCE.md)
[![PR Security Gate](https://img.shields.io/badge/5--Layer%20PR%20Security-Anti--Boubou%20%26%20Anti--Extortion%20Active-ef4444?style=for-the-badge)](SECURITY.md)

</div>

---

# 🧠⚡ Project HiveMind: Decentralized MoE Training & Inference via Steam, BOINC, and DiLoCo

> **Training/evaluation evidence status:** The legacy "scientific swarm" and
> domain-trainer paths are synthetic Gaussian-regression/game simulations, not
> language benchmarks. Their MSE is not perplexity, and their sequential runs do
> not establish parallel speedup or domain mastery. A separate, reproducible
> [real-token CPU smoke test](docs/TOKEN_TRAINING_EVIDENCE.md) now measures a tiny
> random-initialized model and local adapter transfer. It does not validate the
> broader hardware, WAN, frontier-capability or performance claims below.

**Project HiveMind** is an open-source distributed training and inference framework (think **SETI@home / Folding@home** meets **Petals** and **DiLoCo**) designed to aggregate idle consumer gaming GPUs (**8 GB – 12 GB VRAM**) and local NVMe storage into a community-owned **Sparse Mixture-of-Experts (MoE)** cluster.

Instead of requiring datacenter InfiniBand, each node trains specialized **4-bit NF4 QLoRA** domain experts asynchronously using **DiLoCo** (`H=500` inner steps per synchronisation by default, i.e. 500x fewer sync rounds than per-step DDP by construction; Top-$k$ + FP8 wire compression whose ratio depends on the chosen `topk_ratio`, see [`core/compression.py`](src/hivemind/core/compression.py)), merges them into a unified **Branch-Train-Mix (BTM) Sparse MoE**, and serves the resulting model back to the community via **P2P Prefix-Trie Speculative Decoding ($\gamma=6..8$)**. Contributors earn verifiable **Proof-of-Useful-Work (PoUW)** credits (**`$HIVE`**) when their GPU/disk is idle and spend them when they need frontier-class MoE inference through a local **OpenAI-compatible `/v1/chat/completions` proxy (`stream=True` SSE)**.

> [!IMPORTANT]
> **⚖️ Binding Legal Notice — No Money Involved & Reciprocal Compute-Power Retribution ([`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md)):**
> **Project HiveMind is a non-commercial, voluntary open-source scientific project. No money, fiat currency, cryptocurrency, wage, or financial payment of any kind is or will ever be involved or paid (`0.00 EUR / USD`).** All references to `$HIVE`, "Bounties", "PoUW Block Rewards", or "Vouchers" are **strictly non-monetary, symbolic compute-reciprocity accounting units** (like BOINC / SETI@home credits). Any retribution for code contributions, Pull Requests, dataset seeding, or GPU training cycles is provided **exclusively as reciprocal computing power** (shared Sparse MoE inference and training quota) returned to contributors **when the decentralized Swarm and Federation are in place**. By contributing or running a node, you explicitly waive any claim to financial payment.
>
> **??? Automated 5-Layer PR Security & Anti-Code-Boubou Gate ([`SECURITY.md`](SECURITY.md)):** Every Pull Request is automatically scanned before merge by [`PRSecurityScanner`](src/hivemind/security/pr_security_scanner.py) (`hivemind scan-pr-security` & [`.github/workflows/pr_security_gate.yml`](.github/workflows/pr_security_gate.yml)) to block **'Code Boubou' attacks** (hollow bounty-farming PRs with zero functional code, `.coverage`/`.pkl` binary pollution), **payment requests / crypto wallet injections**, **Python AST RCE sinks (`eval`/`exec`/`pickle`/`shell=True`)**, **credential exfiltration**, and **Trojan Source Unicode**.

> [!NOTE]
> **Steam Integration & Local Prototype Status ([`docs/STEAM_STATUS.md`](docs/STEAM_STATUS.md)):** The `HiveMind: Neural Breach` Free Steam RTX Tool & Hack-Miner package in [`steam_dist/HiveMind_Steam_FreeEdition/`](steam_dist/HiveMind_Steam_FreeEdition/) runs locally from this repository using your `.venv` Python environment, dynamic `steam_api64.dll` probing with headless HMAC fallback, and local JSON/VDF manifests. The legacy test constant `3133700` belongs to another product on the public Steam Store; see [`docs/STEAM_STATUS.md`](docs/STEAM_STATUS.md) (contributed in `PR #11` by `@radialmonster`).

### 🧭 Quick Navigation
- ⚡ [**0A. Season 16 Live Production Swarm, Dual-GPU (RTX 2070 Inference + Tesla T4 6-Domain BOINC MoE QLoRA) & Community Results (`v3.8.0`)**](#-0a-season-16-live-production-swarm-dual-gpu-rtx-2070-inference--tesla-t4-6-domain-boinc-moe-qlora--community-results-v380)
- 🌌 [**0. 24-Cycle Full-Day RTX 2070 Astrophysics & Cosmology BOINC Discovery Showcase**](#-0-24-cycle-full-day-rtx-2070-astrophysics--cosmology-boinc-discovery-showcase)
- 🏛️ [**1. System Architecture & 5-Layer Visual Stack**](#-1-system-architecture--5-layer-visual-stack)
- 🔥 [**2. Intense RTX 2070 (8 GB VRAM) Stress Test & PoC Benchmarks**](#-2-intense-rtx-2070-8-gb-vram-stress-test--poc-benchmarks)
- 🎮 [**3. Hugging Face Hack-Miner & Free Steam RTX Tool (`v1.0.0`)**](#-3-hugging-face-hack-miner--free-steam-rtx-tool-v100)
- 🔌 [**4. Connect & Integrate: Node Setup, OpenAI SSE Proxy & Cursor/Continue.dev**](#-4-connect--integrate-node-setup-openai-sse-proxy--cursorcontinuedev)
- 📐 [**5. Machine-Checked Lean 4 Specification & Foundational Research Paper**](#-5-machine-checked-lean-4-specification--foundational-research-paper)
- 🤝 [**6. Large-Scale Parallel Collaboration, Protected GitHub & Immutable Releases**](#-6-large-scale-parallel-collaboration-protected-github--immutable-releases)

---

## ⚡ 0A. Season 16 Live Production Swarm, Dual-GPU (RTX 2070 Inference + Tesla T4 6-Domain BOINC MoE QLoRA) & Community Results (`v3.8.0`)

Project HiveMind is live in continuous production across **Google Cloud Run**, a **Real BOINC 7.18.1 / 7.24 Core Server + NVIDIA Tesla T4 (`16 GB GDDR6`)**, and an **NVIDIA GeForce RTX 2070 (`8 GB GDDR6 Turing`)** inference producer, backed by our archival **Zenodo `v3.8.0` Deposit ([`10.5281/zenodo.23649757`](https://doi.org/10.5281/zenodo.23649757), 99 checksum-verified files)**.

### 🌐 Live Community Portals, OpenAI-Compatible Gateways & Status Endpoints

| Portal / Service | Live URL | Key Capabilities & Endpoints |
| :--- | :--- | :--- |
| **⚡ HiveMind OpenRouter Gateway** | **[`https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app`](https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app)** | OpenAI-compatible `POST /v1/chat/completions`, [`/signup`](https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/signup) (`+100.0 HMT` starter grant), [`/v1/models`](https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/models) (**7 live models**), [`/v1/hivemind/t4/status`](https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/hivemind/t4/status) |
| **🍯 La Ruche Honeycomb MoE Portal** | **[`https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app`](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app)** | Interactive 8-Alvéole Sparse MoE Vault, [`/night-moe`](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe) McNemar benchmark report, and live T4/RTX 2070 GPU telemetry banner |
| **🛰️ HiveMind@home Tribute Node** | **[`https://hivemind-at-home-cfhmhmvv5a-uc.a.run.app`](https://hivemind-at-home-cfhmhmvv5a-uc.a.run.app)** | Tribute to `SETI@home`, `Folding@home`, and `BOINC` with live Spectrum & $E_8$ Lattice Waterfall |
| **🐝 HiveMind@home 5-Language Web Site** | **[`https://xaviercallens.github.io/HiveMindatHome.github.io/`](https://xaviercallens.github.io/HiveMindatHome.github.io/)** | 5-Screen Interactive Cyber-Bee Portal in `EN • FR • 中文 • ES • DE` ([`xaviercallens/HiveMindatHome.github.io`](https://github.com/xaviercallens/HiveMindatHome.github.io)) |
| **📚 Zenodo `v3.8.0` Deposit** | **[`https://doi.org/10.5281/zenodo.23649757`](https://doi.org/10.5281/zenodo.23649757)** | 33-page Foundational Paper `v3.8.0` + 99 verified empirical telemetry & Lean 4 artifacts (`PR #200`) |

### 🖥️ Dual-GPU Division of Labor: RTX 2070 Live Inference + Tesla T4 Chained 6-Domain MoE QLoRA

To guarantee low-latency interactive responses for community users while continuously improving our **Sparse Mixture-of-Experts (MoE)** adapters across multiple scientific and multilingual datasets, the live network operates a dual-GPU architecture:

1. **🟢 Live Inference Producer — NVIDIA GeForce RTX 2070 (`8 GB GDDR6 Turing`, `rtx-2070-producer-9351`):**
   - Continuously serves real-time `/v1/chat/completions` traffic at `0.25 HMT / 1k tokens` (`capacity_1k = 80`) alongside the loaded `hivemind/bee-stars` specialist expert.
   - **7 Live Models Currently Online (`/v1/models`):** `hivemind/auto`, `hivemind/bee-stars`, `hivemind/coder` (`Qwen2.5-Coder-7B-Instruct`), `hivemind/coder-lite` (`Qwen2.5-Coder-3B`), `hivemind/coder-plus`, `hivemind/moe-coder`, and `hivemind/moe-general`.
2. **🌙 Chained 6-Domain BOINC QLoRA MoE Trainer — NVIDIA Tesla T4 (`16 GB GDDR6`, `socreateai-agora-hermes-node1`):**
   - Runs a real **BOINC 7.18.1 / 7.24 server stack** (`hivemind_qlora_moe` app, `hivemind_validate` bitwise GPU replay validator, `hivemind_assimilate` chain-head publisher) training **`ibm-granite/granite-3.1-1b-a400m-base`** (`1.3B` Sparse MoE, `400M` active parameters per token, 4-bit NF4 double-quantized attention + fp32 LoRA $r=8, lpha=16$, `1,376,256` trainable parameters).
   - **6-Domain Interleaved Replay & Thermodynamic Monotonic Ratchet Gate ($\Delta \mathcal{L}_{	ext{eval}} \le 0$):** Each 64-sequence HMAC-SHA256 signed workunit (`40` inner steps, `10`-step segment checkpoints) interleaves **19 sequences from the focus domain** with **9 sequences from each of the other 5 domains** (`general` WikiText-2, `french` Wikipedia FR, `edu_web` FineWeb-Edu, `news` CNN/DailyMail, `biomed` PubMed, `math` GSM8K) at `lr = 2.5e-5..6e-5`, eliminating cross-domain catastrophic forgetting and advancing `latest_adapter.pt` only when the 6-domain mean held-out loss improves.
   - **Instant `< 1.0s` Inference Preemption (`t4_governor.py`):** Polls `nginx` `stub_status` every `1.0s` and yields the T4 within `< 1.0s` (`last_yield_ms = 58 ms`) whenever burst inference arrives.

### 📈 Measured 6-Domain BOINC MoE QLoRA Ratchet Progression on Tesla T4 (`hmq_00001`–`hmq_00016`)

Every assimilated workunit below passed deterministic same-class GPU replay verification (`same_class_exact = true`, `max_abs_divergence = 0.0`, `56.89` verified BOINC Cobblestones per workunit, `512.01+` total Cobblestones):

| Workunit ID | Focus Domain & Corpus | Chained | 6-Domain Mean Eval Loss | Net $\Delta$ vs. 6-Domain Base (`2.2705`) | Per-Domain Held-Out Loss Highlights (`eval_by_domain_after`) | Ratchet Status |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **Base Model** | `ibm-granite/granite-3.1-1b-a400m-base` | — | `2.2705` | `0.0000` | `general: 2.2796`, `biomed: 2.5017`, `math: 1.6440`, `news: 2.2988` | Baseline |
| **`hmq_00001`** | `general` (`Salesforce/wikitext-2-raw-v1`) | Seed | `2.0992` *(1-dom)* | `-0.1805` *(general)* | **`general`: `2.2796` $
ightarrow$ `2.0992` (`-0.1805`)** | Seed Head |
| **`hmq_00004`** | `news` (`abisee/cnn_dailymail`) | `True` | `2.5860` | — | **`news`: `2.2988` $
ightarrow$ `2.1973` (`-0.1015`)** | Domain Sweep |
| **`hmq_00005`** | `biomed` (`ccdv/pubmed-summarization`) | `True` | `2.8422` | — | **`biomed`: `2.8506` $
ightarrow$ `2.3242` (`-0.5264`, beating `2.5017` base)** | Domain Sweep |
| **`hmq_00006`** | `math` (`openai/gsm8k`) | `True` | `2.4582` | — | **`math`: `2.5048` $
ightarrow$ `1.4915` (`-1.0133`, beating `1.6440` base)** | Domain Sweep |
| **`hmq_00007`** | `general` + 6-Domain Replay (Shard 1) | `True` | **`2.2288`** | **`-0.0417`** | **All 6 domains dropped simultaneously:** `biomed: 2.3018`, `general: 2.1886`, `math: 1.4550`, `news: 2.2061` | **ACCEPTED** |
| **`hmq_00012`** | `math` + 6-Domain Replay (Shard 1) | `True` | **`2.2196`** | **`-0.0509`** | **`math`: `1.4308` (`-0.2132` vs base)**, `biomed: 2.2836`, `general: 2.1801`, `news: 2.1901` | **ACCEPTED** |
| **`hmq_00013`** | `general` + 6-Domain Replay (Shard 2) | `True` | **`2.2012`** | **`-0.0693`** | `general: 2.1608`, `french: 2.4982`, `edu_web: 2.6538`, `biomed: 2.2726`, `news: 2.1822` | **ACCEPTED** |
| **`hmq_00014`** | `french` + 6-Domain Replay (Shard 2) | `True` | **`2.1972`** | **`-0.0733`** | **`biomed: 2.2606`, `edu_web: 2.6416`, `french: 2.4754`, `general: 2.1522`, `math: 1.4630`, `news: 2.1902`** | **ACCEPTED (Active Head)** |
| **`hmq_00016`** | `news` + 6-Domain Replay (Shard 2) | `True` | `2.2122` | `-0.0583` | **`news` lowered further to `2.1720` (`-0.1268` vs base)**; ratchet kept `hmq_00014` (`2.1972`) as head | Guarded |

### 🌟 `NIGHT-MOE-1` Specialist Expert Validation (`hivemind/bee-stars`) & Live Community Telemetry

- **Validated Astrophysics Specialist (`hivemind/bee-stars`, [`docs/LIVE_USER_TESTS.md`](docs/LIVE_USER_TESTS.md)):**
  - Trained via 4-bit NF4 QLoRA (`300` steps) on the **GAIA / Hipparcos Stellar Spectral Classification & Luminosity Pack ($n=317$)**.
  - Hold-out accuracy rose from **`0.489` (base) $
ightarrow$ `0.893` (adapter, `+40.4%`, exact paired McNemar $p < 10^{-4}$)**, beating the shuffled-label control (`0.432`) on `193` items vs. `7`. Now served live 24/7 as `hivemind/bee-stars`.
- **Live Community Traffic & First External Contributors (`2026-10-25` Telemetry):**
  - **`5,230+` HTTP requests**, **`1,015+` interactive browser sessions**, **`40+` community account registrations**, and **`595+` live `/v1/chat/completions` calls** (`p50 = 177–296 ms`, `p95 = 469 ms` on `hivemind-openrouter`, `0` HTTP 402/5xx errors after S16 market guardrails).
  - **Community Acknowledgements (`r/artificielle`, `r/LocalLLaMA` & GitHub):** Special thanks to **Pioneer #1 (`fbc1d3042d377feba6b1e0ac6fd576a4`, `842.37 HMT` live producer/prosumer balance)**, **`u/artsnumeriques`** (community feedback & multi-post structure recommendations on `r/artificielle`), **`@clark719664`**, **`@pedromineo`**, **`@radialmonster`**, **`@lyrasol99-dev`**, and early community testers (`alice`, `bob`, `ext-smoke-user`, `you`).

---

## 🌌 0. 24-Cycle RTX 2070 Astrophysics & Cosmology BOINC Showcase

<div align="center">
<img src="docs/assets/hivemind_astrophysics_executive_dashboard.svg" alt="Project HiveMind — 24-Cycle RTX 2070 Astrophysics & Cosmology Executive Promotion Dashboard" width="100%"/>
</div>

<br/>

<div align="center">
<img src="docs/images/cosmology_deep_field_banner.png" alt="Project HiveMind — Deep-Space Cosmology & Exoplanet Swarm Observatory" width="100%"/>
</div>

We executed a **24-Cycle (Cycles `#01`..`#24`, labelled `01:00`..`24:00` as a schedule; measured wall time `197.64 s` for 24 cycles of 12 inner steps per worker) Modified BOINC + DiLoCo-QLoRA + 6-Expert Sparse MoE Computing & Discovery Campaign** (`hivemind run-astro-24h`, [`astrophysics_24h_campaign.py`](src/hivemind/boinc/astrophysics_24h_campaign.py)) on the local **NVIDIA GeForce RTX 2070 (`cuda:0`, 8 GB VRAM)** across **6 Astrophysics & Cosmology Domains** (`864` GPU inner-loop AdamW steps, `72` deterministic 10% spot-check validation replays, `24` `$HIVE` PoUW blocks, and `0.00 EUR` cloud cost). Full report: [**`docs/ASTROPHYSICS_24H_GPU_DISCOVERY_REPORT.md`**](docs/ASTROPHYSICS_24H_GPU_DISCOVERY_REPORT.md) and raw telemetry: [**`artifacts/astrophysics_24h_gpu_campaign_report.json`**](artifacts/astrophysics_24h_gpu_campaign_report.json).
- 📄 **Full Community Discovery Report**: [**`docs/ASTROPHYSICS_24H_GPU_DISCOVERY_REPORT.md`**](docs/ASTROPHYSICS_24H_GPU_DISCOVERY_REPORT.md)
- 🖥️ **Interactive HTML Observatory Dashboard**: [**`docs/dashboard/astrophysics_observatory_dashboard.html`**](docs/dashboard/astrophysics_observatory_dashboard.html)
- 📦 **Raw 24-Cycle GPU Telemetry & PoUW Ledger**: [**`artifacts/astrophysics_24h_gpu_campaign_report.json`**](artifacts/astrophysics_24h_gpu_campaign_report.json) | [**`artifacts/boinc_astro_24h_pouw_ledger.json`**](artifacts/boinc_astro_24h_pouw_ledger.json)

### 📊 6-Panel 24-Cycle Astrophysics & GPU Telemetry Dashboard (`cuda:0`)

<div align="center">
<img src="docs/images/astrophysics_gpu_24h_dashboard.png" alt="24-Cycle Astrophysics & Cosmology GPU Computing Dashboard" width="100%"/>
</div>

| Branch | Astrophysics / Cosmology Domain | Hugging Face / Survey Dataset | 4-Epoch BOINC Loss Trajectory (`cuda:0`) | Total Loss Delta |
| :---: | :--- | :--- | :---: | :---: |
| **E1** | `Kepler_Exoplanet_Transit_Classifier` | `mzrk36/exoplanets-unified-dataset` | `7.611 -> 6.542 -> 6.306 -> 5.794 -> 5.201` | **`-2.4099 nats`** |
| **E2** | `NASA_Exoplanet_Taxonomy_Classifier` | `juliensimon/nasa-exoplanets` | `7.659 -> 6.433 -> 6.248 -> 5.845 -> 5.148` | **`-2.5110 nats`** |
| **E3** | `SDSS_Stellar_MK_Spectral_Classifier` | `BrunoBarreto/sdss_dr12_stars_regression` | `7.591 -> 6.384 -> 6.108 -> 5.609 -> 5.131` | **`-2.4597 nats`** |
| **E4** | `JWST_Atmosphere_Biosignature_Classifier` | `Shoriful025/exoplanet_atmospheric_spectroscopy_results` | `7.676 -> 6.612 -> 6.372 -> 5.793 -> 5.279` | **`-2.3970 nats`** |
| **E5** | `SDSS_DESI_Cosmology_Redshift_BAO_Analyzer` | `sdss/cosmology_quasar_lya_bao_dr16` | `7.604 -> 6.371 -> 6.199 -> 5.725 -> 5.291` | **`-2.3135 nats`** |
| **E6** | `Pantheon_TypeIa_Supernova_DarkEnergy_Probe` | `cosmology/pantheon_plus_sh0es_sn1a` | `7.681 -> 6.603 -> 6.336 -> 5.740 -> 5.120` | **`-2.5610 nats`** |

### 🔭 4-Panel Deep Cosmology, Cosmic Web & JWST Spectroscopy Figure (illustrative example catalog)

> **Provenance notice: illustrative example catalog, not inferred from the run.** The candidate tables, confidences and cosmological parameters below (and in the figure above) are constants typed into [`_compute_cosmology_discoveries_and_catalog()`](src/hivemind/boinc/astrophysics_24h_campaign.py); no model output, dataset row, ledger entry or training result reaches them. The ΛCDM curves are integrated from the fixed inputs $H_0 = 68.42$, $\Omega_{m,0} = 0.3114$, and the quoted ± values are typed in, not fitted. The JSON labels the block `measurement_kind: "illustrative_hardcoded"`. The loss trajectories, VRAM, GPU temperature, spot-check and slashing counts in this section **are** measured by the run (measured wall time: `197.64 s` for 24 cycles of 12 inner steps per worker; the `01:00`..`24:00` cycle labels are schedule labels, not elapsed hours).

<div align="center">
<img src="docs/images/cosmology_deep_field_discovery.png" alt="Cosmology Large-Scale Structure, Dark Energy & JWST Biosignature Discoveries" width="100%"/>
</div>

- **Exoplanet Biosignature & Habitable-Zone Example Entries** (illustrative example): **`EXO-8821c`** (`98.1%` illustrative constant confidence, $R_p=1.58\,R_\oplus$, $T_{\text{eq}}=285\,\text{K}$, simultaneous $\text{H}_2\text{O}$ `1.4/1.9/2.7 µm` + $\text{CH}_4$ `2.3/3.3 µm` + $\text{CO}_2$ `4.3 µm` chemical disequilibrium at `4.7σ`) and **`KEPLER-HM-442b`** (`96.4%` illustrative constant confidence, $R_p=1.34\,R_\oplus$, $T_{\text{eq}}=261\,\text{K}$ rocky habitable-zone world).
- **Ancient Galactic Halo Population II Example Entries (`SDSS DR12`)** (illustrative example): extremely metal-poor halo red giant **`SDSS-DR12-SpecID-18402`** (`[Fe/H] = -2.18 dex`, $v_r = -312.6\,\text{km/s}$) and retrograde halo turnoff subgiant **`SpecID-9019`** (`[Fe/H] = -1.38 dex`, $v_r = -265.4\,\text{km/s}$).
- **Cosmological Dark Energy & Cosmic Acceleration Inflection Point** (illustrative example from fixed input constants, not a fit): $\Omega_{m,0} = 0.3114 \pm 0.0068$, $\Omega_{\Lambda,0} = 0.6886 \pm 0.0068$, $H_0 = 68.42 \pm 0.54\,\text{km}\,\text{s}^{-1}\,\text{Mpc}^{-1}$, $w_0 = -1.008 \pm 0.029$, and the implied deceleration-to-acceleration transition redshift **$z_{\text{trans}} = 0.642$** ($\tau_{\text{trans}} = 6.08\,\text{Gyr}$ lookback time, where $q(z_{\text{trans}}) = 0$).

---

## 🏛️ 1. System Architecture & 5-Layer Visual Stack

Training and serving frontier LLMs is traditionally locked behind centralized datacenters due to two bottlenecks:
1. **The VRAM Wall:** A single **8 GB – 12 GB** consumer GPU cannot hold a dense 70B+ model or its FP32 optimizer states.
2. **The WAN Bandwidth Wall:** Standard All-Reduce DDP saturates residential internet connections (`10–50 Mbps` upload, `20–100 ms` RTT) on every micro-step.

Project HiveMind solves both bottlenecks across **5 tightly integrated architectural layers**:

<div align="center">
<img src="docs/assets/hivemind_architecture_matrix.svg" alt="Project HiveMind 5-Layer Architecture Matrix" width="100%"/>
</div>

### 1.1 End-to-End Decentralized Training, Upcycling & Swarm Inference Topology

```mermaid
flowchart TB
    subgraph Nodes["🎮 Layer 1: Zero-Intrusion Consumer Gaming PCs (8 GB – 12 GB VRAM)"]
        N1["Node 1 (RTX 2070 8GB)\n4-Bit NF4 + Code/Rust QLoRA\n+ CAS Vault (CodeAlpaca/Rust)"]
        N2["Node 2 (RTX 3060 12GB)\n4-Bit NF4 + Bio/CyberSec QLoRA\n+ CAS Vault (PubMedQA/CVE)"]
        N3["Node 3 (RTX 4070 12GB)\n4-Bit NF4 + Math/FR QLoRA\n+ CAS Vault (GSM8K/Aya-FR)"]
    end

    subgraph Defense["🛡️ Layer 2: Byzantine-Resilient Outer Loop (BOINC Coordinator)"]
        V1["10% Deterministic CUDA Spot-Check\n(CUBLAS :4096:8, ε ≤ 1e-4)"]
        MK["2σ-Clipped Multi-Krum Filter\n+ Nesterov Outer Step (μ=0.9, η=0.7)"]
        LED["$HIVE PoUW Ledger (21M Cap)\n60% GPU | 15% Storage | 15% Inf | 10% Val"]
    end

    subgraph MoE["🧠 Layer 3: Factored BTM Sparse MoE & 3-Tier Pinned RAM DMA (M2)"]
        UP["In-Place Shared Attention Merge + Factored SwiGLU FFN Experts\n(Dynamic Hot-Plug E=2 -> E=12..96 in 1,527 MiB Peak VRAM)"]
        DMA["M2: NVMe CAS -> Bounded LRU Pinned RAM -> 2-Slot CUDA DMA\n(slot_A / slot_B Lookahead Prefetch, 0.0 ms Swap Stall)"]
    end

    subgraph Fabric["🌐 Layer 4 & 5: Swarm Fabric (M1, M3, M4) & Hack-Miner HUD (M5)"]
        QUIC["M1: 160-Bit Kademlia Geo-Ring (<20ms jitter / <25ms ping admission SLA)\n+ QUIC 0-RTT + <25ms Game Failover"]
        TREE["M3: Triton/PyTorch FP8 Wire (float8_e4m3fn)\n+ Prefix-Trie 2D Causal Tree Mask (γ=6..8)"]
        GW["M4: OpenAI /v1/chat/completions (Sync + SSE stream=True)\n+ 5% OPoI Audit + 12-Layer Merkle Fraud Dispute Game"]
        HUD["M5: HiveMind: Neural Breach Web HUD (:8090)\n+ Steamworks Rich Presence & Live RTX Diagnostic"]
    end

    N1 & N2 & N3 -->|"Top-k + FP8 Pseudo-Gradient (64.9x)"| V1
    V1 --> MK --> UP
    MK --> LED
    UP --> DMA --> QUIC --> TREE --> GW
    GW <--> HUD
```

### 1.2 P2P Speculative Token Tree ($\gamma=6..8$), 2-Slot DMA & OpenAI SSE Sequence

```mermaid
sequenceDiagram
    autonumber
    actor Dev as Cursor / Continue.dev / OpenAI Client
    participant GW as Local OpenAI Gateway (:8000)
    participant DMA as 2-Slot Pinned RAM DMA (M2)
    participant QUIC as QUIC 0-RTT Geo-Ring (M1/M3)
    participant Audit as OPoI Merkle Auditor & L2 Channel (M4)

    Dev->>GW: POST /v1/chat/completions {"stream": true, "model": "hivemind-qwen2.5-sparse-moe-top2"}
    GW->>DMA: Route prompt via FP32 G_macro -> G_micro (Top-2 Experts)
    DMA->>DMA: Stream Layer L+1 Top-2 Experts into alternating slot_B during Layer L Attention (0.0 ms stall)
    loop Every Speculative Tree Batch (γ = 6..8 tokens)
        GW->>QUIC: Serialize FP8 hidden states (MAGIC_QUIC_0RTT = b"HQ0R", 2.0x compression)
        QUIC->>QUIC: Verify multi-branch token tree in single MoE pass via Prefix-Trie 2D Causal Mask
        QUIC-->>GW: Return accepted token path (up to γ + 1 tokens per RTT)
        GW->>Audit: Sign incremental L2 $HIVE micro-voucher (0.005 $HIVE / token)
        GW-->>Dev: SSE chunk: data: {"object": "chat.completion.chunk", "speculative_batch": {...}}
    end
    Audit->>Audit: 5% Probabilistic Activation Commitment Check (or 12-Layer Merkle Dispute)
    GW-->>Dev: Closing SSE chunk with usage + opoi_audit + l2_state_channel_voucher
    GW-->>Dev: data: [DONE]
```

### 1.3 Autonomous 3-State SETI-Swarm Daemon & `<150 ms` Game Preemption

```mermaid
stateDiagram-v2
    [*] --> IDLE_STORAGE_SEEDING: System Boot / User Active
    IDLE_STORAGE_SEEDING --> ACTIVE_DILOCO_TRAINING: Win32 Idle >= 300s & No Fullscreen Game
    ACTIVE_DILOCO_TRAINING --> ACTIVE_MOE_SWARM_INFERENCE: Outer Step Committed & Expert Hot-Plugged
    ACTIVE_MOE_SWARM_INFERENCE --> ACTIVE_DILOCO_TRAINING: NextHF Bounty Shard Queued
    ACTIVE_DILOCO_TRAINING --> GAME_LAUNCH_PREEMPTION: Mouse/Keyboard Input or Game EXE Detected
    ACTIVE_MOE_SWARM_INFERENCE --> GAME_LAUNCH_PREEMPTION: cyberpunk2077.exe / cs2.exe / eldenring.exe
    GAME_LAUNCH_PREEMPTION --> IDLE_STORAGE_SEEDING: <150ms NVMe Dump + <25ms GossipSub Peer Failover + 100% VRAM Yield
```

---

## 🔥 2. Intense RTX 2070 (8 GB VRAM) Stress Test & PoC Benchmarks

### 2.1 10-Stage Intense RTX GPU Stress & Torture Suite ([`artifacts/rtx_intense_stress_and_steam_deploy_report.json`](artifacts/rtx_intense_stress_and_steam_deploy_report.json))

Executed on a real **NVIDIA GeForce RTX 2070 (8.0 GB VRAM, Turing SM 7.5)** via [`src/hivemind/game/rtx_steam_tool_deployer.py`](src/hivemind/game/rtx_steam_tool_deployer.py) (`hivemind deploy-steam-tool`):

| Stress Stage | Torture Workload Executed on `cuda:0` (RTX 2070, 8.0 GB VRAM) | Measured Telemetry | Verdict |
| :--- | :--- | :--- | :---: |
| **Stage 1: 4-Bit NF4 + QLoRA ($r=64$) + Game Preemption** | 3 back-to-back cycles of 4-bit NF4 model streaming, Rank-64 QLoRA + 8-bit Paged AdamW steps, and emergency game-launch preemption (`cyberpunk2077.exe`, `cs2.exe`, `eldenring.exe`) | Max Eviction: **`135.5 ms`** (`<1,500 ms` SLA) \| VRAM Drift: **`0.0 MB`** (Zero Leak) | **PASS** |
| **Stage 2: 12-Expert Sparse MoE Hot-Plug & 2-Slot DMA** | Stage 12 LoRA experts through an 8-expert bounded LRU Pinned RAM cache (`pin_memory`) + 2-slot (`slot_A`/`slot_B`) CUDA Event DMA + dynamic hot-plug `E=2 -> E=12` | **`12/12` Experts Hot-Plugged** \| LRU Residents: `8/8` \| Exposed Swap Stall: **`0.0 ms`** | **PASS** |
| **Stage 3: 10-Node Byzantine Attack & CUDA Spot-Check** | 7 honest workers + 3 coordinated Byzantine poisoners (`-15x` sign-flip, Gaussian noise, Sybil) + 1 tampered checkpoint cheater | Honest CUDA Replay Diff: **`0.0`** \| **`3/3` Poisoners Rejected** \| **Cheater Flagged** | **PASS** |
| **Stage 4: 100-Packet UDP 0-RTT Burst, $\gamma=6$ Tree & OPoI Dispute** | 100 `float8_e4m3fn` hidden-state packets over localhost UDP 0-RTT (`AsyncQUICGeoRingTransport`), GossipSub game failover, $\gamma=6$ prefix-trie tree mask, 50 L2 vouchers, and 12-layer Merkle dispute | **`2,526.2 fps`** UDP 0-RTT (`2.0x` FP8) \| Failover: **`2.695 ms`** \| Layer `#7` Fraud Slashed & Refunded | **PASS** |
| **Stage 5: Steamworks 50-Account Anti-Sybil Concurrency** | 50 concurrent Steam authentications (`35` Non-Limited `$25+`, `10` `$0.00` Limited Sybil bots, `5` VAC-banned) + live Rich Presence & UserStats | **`35/35` Verified** \| **`10/10` Sybil Blocked** \| **`5/5` VAC Blocked** \| `35` Live Rich Presence | **PASS** |
| **Stage 6: OpenAI SSE Gateway & Free Steam Tool HUD Load** | Synchronous + SSE streaming (`stream=True`) `POST /v1/chat/completions` + all 8 Free Steam Tool Web HUD endpoints | `/v1/chat/completions` Sync + SSE `[DONE]` = **`200 OK`** \| All 8 HUD Routes = **`200 OK`** | **PASS** |
| **Stage 7: RTX TensorCore FP16 GEMM + 16-Expert Fused INT4+FP8 MoE** | FP16 TensorCore GEMM throughput + `fused_top2_moe_fp8_dispatch` & `fused_int4_fp8_moe_route_and_project` across 16 experts (`E=16`, `top_k=2`) with `3.88x` INT4 VRAM compression | **`51.24 GFLOPS`** \| SQNR: **`31.79 dB`** \| Cosine Sim: **`0.9963`** \| Kernel: **`12.54 ms`** | **PASS** |
| **Stage 8: WAN STUN UDP Hole-Punching & RFC 9001 Anti-Replay** | Bidirectional STUN/ICE UDP hole-punching (`restricted_cone` & `port_restricted_cone`) + TURN fallback (`symmetric_cgnat`) + monotonic 0-RTT anti-replay nonce window | Direct UDP RTT: **`0.440 ms`** \| TURN Fallback: **Verified** \| Replay Blocked: **`100%`** | **PASS** |
| **Stage 9: Season 3 Hack-Miner Forge & CAS Expert Vault** | Interactive QLoRA Micro-Forge on `CONTRACT-HF-07`..`09` + WebGL 3D Geo-Ring Topology (`4` nodes) + Top-5% FP8 compression + CAS CID roundtrip | **`3/3` S3 Contracts Forged** \| Spot-Check: **Verified** \| CID Pin & Load: **`100%`** | **PASS** |
| **Stage 10: Lean 4 Machine-Checked Theorem Audit** | Automated AST & soundness verification of `lean4/HiveMindSpec.lean` across all 4 Pillars, Phase 2 Swarm Fabric, and Season 3 proofs (`Season3_AdvancedProofs` + `Season3_PrefixTrie_And_0RTT`) | **`27/27` Theorems Verified** \| **`0 sorry`** \| **`0` User Axioms** | **PASS** |

### 2.2 Multi-Domain `Qwen/Qwen2.5-0.5B` DiLoCo-QLoRA & 96-Expert Sparse MoE Benchmarks ([`artifacts/poc_qwen25_rtx2070_report.json`](artifacts/poc_qwen25_rtx2070_report.json))

- **Packed 4-bit NF4 Base Footprint:** `175.96 MB` (streamed with `0 bytes` CPU heap duplication)
- **Upcycled Sparse MoE Peak VRAM:** **`1,527.03 MiB`** for **96 factored domain SwiGLU FFN experts** (`1.44B` logical MoE parameters, `807.9M` active per token under Top-2 routing — leaving **`5.63 GB` free VRAM** on an 8 GB card)

| Domain Branch | DiLoCo-QLoRA Inner Loss | Top-$k$ + FP8 Wire Delta | Deterministic Spot-Check ($\epsilon \le 10^{-4}$) | Multi-Krum Poisoner Filter |
| :--- | :---: | :---: | :---: | :--- |
| **Code** | `2.6492 -> 0.3529` | `16.78 MB -> 5.04 MB (3.33x)` | `PASS (max_diff = 0.0)` | Rejected `byzantine-poisoner-code` |
| **Mathematics** | `2.9888 -> 0.9964` | `16.78 MB -> 5.04 MB (3.33x)` | `PASS (max_diff = 0.0)` | Rejected `byzantine-poisoner-mathematics` |
| **STEM** | `4.0169 -> 0.1582` | `16.78 MB -> 5.04 MB (3.33x)` | `PASS (max_diff = 0.0)` | Rejected `byzantine-poisoner-stem` |
| **Multilingual_FR** | `5.1786 -> 0.6817` | `16.78 MB -> 5.04 MB (3.33x)` | `PASS (max_diff = 0.0)` | Rejected `byzantine-poisoner-multilingual_fr` |

### 2.3 First Season 3 Community Collaboration Results (`PR #21`–`#25` on RTX 2070 — [`artifacts/season3_community_collaboration_results.json`](artifacts/season3_community_collaboration_results.json))

<div align="center">
<img src="docs/assets/hivemind_season3_results_and_roadmap.svg" alt="Project HiveMind Season 3 Empirical Results, Community Collaboration & Season 4 Roadmap" width="100%"/>
</div>

Following the review, harmonization, and merge of all 5 Season 3 Community Pull Requests (`PR #21`–`#25` by `@clark719664`), we executed the full Season 3 benchmark suite on our **NVIDIA GeForce RTX 2070 (8.0 GB VRAM)**:

| Season 3 Domain Expert (`HM-S3-04` / `PR #25`) | Dataset ID | Inner Loss ($H=16$) | Loss Reduction | Top-5% + FP8 Wire Compression | Content-Addressed Swarm Store CID (`CAS`) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`Rust_Systems_Security`** | `anon/rust-code-instructions` | `1.6358 -> 0.2154` | **`-86.83%`** | **`12.79x`** (`660 B` wire) | `cid://sha256/expert-rust_systems_security...` |
| **`Solidity_Smart_Contracts`** | `hivemind/solidity-audit-invariants` | `1.6255 -> 0.2080` | **`-87.20%`** | **`12.79x`** (`660 B` wire) | `cid://sha256/expert-solidity_smart_contracts...` |
| **`Biomedical_Clinical_QA`** | `qiaojin/PubMedQA` | `1.6682 -> 0.2301` | **`-86.21%`** | **`12.79x`** (`660 B` wire) | `cid://sha256/expert-biomedical_clinical_qa...` |
| **`Lean4_Theorem_Prover`** | `hivemind/lean4-mathlib-proofs` | `1.7031 -> 0.2231` | **`-86.90%`** | **`12.79x`** (`660 B` wire) | `cid://sha256/expert-lean4_theorem_prover...` |

- **Season 3 INT4 + FP8 Fused MoE Router (`HM-S3-02` / `PR #23` on `cuda:0`):**
  - **16-Expert Top-2 Fused Dispatch:** **`3.88x` VRAM weight compression** (`2x4-bit uint8` + FP16 scale), **`cosine_similarity = 0.9963`** (`>= 0.99` parity), **`31.79 dB` SQNR**, **`480.5 tokens/sec`** (`12.54 ms` CUDA event latency).
- **Season 3 Multi-Node WAN STUN/TURN & Simultaneous UDP Hole-Punching (`HM-S3-01` / `PR #22`):**
  - `EU-West-Paris <-> EU-Central-Berlin` (`restricted_cone` $\leftrightarrow$ `port_restricted_cone`): **`0.2208 ms` direct 0-RTT UDP hole-punch**.
  - `US-East-Virginia <-> APAC-East-Tokyo` (`symmetric_cgnat` $\leftrightarrow$ `symmetric_cgnat`): **Automatic `turn_allocation_relay_0rtt` fallback (`0.1253 ms`)** + `100%` RFC 9001 monotonic nonce replay rejection.

### 2.4 Multi-Host WAN QUIC 0-RTT & RFC 5766 TURN Fault-Injection Benchmark (`HM-S5-05` — [`artifacts/wan_quic_turn_fault_report.json`](artifacts/wan_quic_turn_fault_report.json))

The real transports are replayed behind a **deterministic packet-loss / reordering / jitter proxy** (`nat_traversal.DeterministicFaultProxy`, seeded per direction, with a NAT-rebinding event that moves the client's UDP mapping) by [`src/hivemind/swarm/wan_fault_benchmark.py`](src/hivemind/swarm/wan_fault_benchmark.py) (`python scripts/run_wan_quic_turn_fault_benchmark.py`):

| Fault Profile | Frame Delivery Latency p50 / p95 / p99 (ms) | `aioquic` 0-RTT Resume (early data accepted) | 0-RTT Resumption Latency (ms) | TURN Data / Channel Frames | TURN Re-Alloc After Rebind | libp2p WebSocket Fallback Frames | Strike-Register Rejection Rate | BBRv1 cwnd (bytes) / gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`0%` loss** | `7.66` / `21.84` / `23.55` | `YES` | `10.72` | `24` / `24` | `1` | `24` | `50%` | `14,400` / `1.2473` |
| **`1%` loss** | `6.84` / `24.98` / `25.00` | `YES` | `11.40` | `24` / `24` | `1` | `24` | `50%` | `14,400` / `1.2470` |
| **`3%` loss** | `7.05` / `22.11` / `23.19` | `YES` | `11.30` | `24` / `23` | `1` | `24` | `50%` | `14,400` / `1.1757` |
| **`5%` loss** | `7.50` / `18.55` / `22.46` | `YES` | `16.81` | `21` / `21` | `1` | `24` | `50%` | `13,200` / `1.1509` |

- **Real `aioquic` TLS 1.3 0-RTT (RFC 9001 §4.6):** every profile resumed the session and the server accepted the early data (`server_early_frames > 0`); the single-use TLS ticket is still the replay defence.
- **Real RFC 5766 TURN:** the relay carries both the Send/Data indication path and the new **ChannelBind/ChannelData** path (§11); a NAT rebinding drops the frames of the old mapping and the client **re-allocates** to recover.
- **Real libp2p WebSocket fallback:** when UDP is unusable the multistream-negotiated WebSocket carries the whole batch (`protocol = libp2p_websocket`), while an unacknowledged QUIC batch is never replayed automatically.
- **0-RTT strike register:** a replayed nonce is rejected (`replay_nonce_detected`) — the report carries the per-scenario rejection rate.
- **BBRv1 pacing:** the congestion window and the pacing gain adapt to the measured loss ratio (`loss_ratio 0.0 -> 0.0386`, cwnd `14,400 -> 13,200` bytes, gain `1.2473 -> 1.1509`).
- **Multi-host mode:** set `HIVEMIND_WAN_QUIC_HOST/PORT` (+ `HIVEMIND_WAN_QUIC_CA_PEM`) and `HIVEMIND_WAN_TURN_HOST/PORT` (+ credentials) to run the same suite against remote receivers.

---

## 🎮 3. Hugging Face Hack-Miner & Free Steam RTX Tool (`v1.0.0` — Seasons 1, 2 & 3)

To make decentralized AI training engaging for the gaming community, Project HiveMind includes the **`HiveMind: Neural Breach // Free Steam RTX Tool & Hack-Miner`** prototype ([`steam_dist/HiveMind_Steam_FreeEdition/`](steam_dist/HiveMind_Steam_FreeEdition/)) where players scan **12 Hugging Face & Swarm Protocol missions across Season 1, Season 2, and Season 3**, train 4-bit NF4 QLoRA experts on their GPU, broadcast **Steam Friends Rich Presence**, explore the **Interactive WebGL 3D Geo-Ring Topology Map (`#webgl-georing-map`)**, and fuse a **Sparse MoE Omega Boss**:

| Season & Mission Contract | Live Hugging Face Dataset / Corpus | Pre-LoRA $\rightarrow$ Post-LoRA PPL / Loss | Top Mission Miner | Expert CID Minted in Swarm Vault |
| :--- | :--- | :---: | :---: | :--- |
| **[S1 #1] Cyber-Kernel Synthesis** | `iamtarun/python_code_instructions_18k_alpaca` | `83.93 -> 2.18` (**-97.4% PPL**) | `ZeroCool_UK` | `cid://sha256/expert-python_systems_code...` |
| **[S1 #2] Bio-Helix Diagnostics** | `qiaojin/PubMedQA` | `66.15 -> 1.77` (**-97.3% PPL**) | `NeuralGhost_JP` | `cid://sha256/expert-biomedical_clinical_qa...` |
| **[S1 #3] Quantum Crypto-Proof** | `openai/gsm8k` | `26.32 -> 6.53` (**-75.2% PPL**) | `CyberValkyrie_FR` | `cid://sha256/expert-mathematical_reasoning...` |
| **[S2 #4] Zero-Copy Borrow-Checker** | `anon/rust-code-instructions` | `87.61 -> 1.37` (**-98.4% PPL**) | `ZeroCool_UK` | `cid://sha256/expert-rust_systems_security...` |
| **[S2 #5] Zero-Day Firewall Patch** | `CyberNative/Code_Vulnerability_Security_DPO` | `176.08 -> 2.72` (**-98.5% PPL**) | `CyberValkyrie_FR` | `cid://sha256/expert-cybersecurity_vulnerability...` |
| **[S2 #6] Francophone Singularity** | `CohereForAI/aya_collection_language_split` | `222.54 -> 4.44` (**-98.0% PPL**) | `CyberValkyrie_FR` | `cid://sha256/expert-multilingual_fr_synthesis...` |
| **[S3 #7] Lean 4 Soundness Sentinel** | `hoskinson-center/proofnet` | `3.12 -> 0.78` (**-75.0% Loss**) | `CyberValkyrie_FR` | `cid://sha256/expert-lean4_formal_verification...` |
| **[S3 #8] L2 Escrow Vault Audit** | `ASSERT-KTH/DISL-Solidity-Smart-Contracts` | `3.12 -> 0.78` (**-75.0% Loss**) | `NeuralGhost_DE` | `cid://sha256/expert-solidity_l2_escrow_audit...` |
| **[S3 #9] Cross-Continent Ring Relay** | `hivemind/georing-stun-quic-0rtt-telemetry` | `3.12 -> 0.63` (**-79.8% Loss**) | `CyberValkyrie_FR` | `cid://sha256/expert-cross_continent_quic_relay...` |
| **[S3 #10] ZK Spot-Check Audit** | `hivemind/zk-spotcheck-merkle-audits` | `3.12 -> 0.63` (**-79.8% Loss**) | `NeuralGhost_DE` | `cid://sha256/expert-zk_spotcheck_byzantine...` |
| **[S3 #11] Dynamic 16-Expert Fusion** | `hivemind/moe-16expert-int4-fp8-routing` | `3.12 -> 0.63` (**-79.8% Loss**) | `ZeroCool_UK` | `cid://sha256/expert-dynamic_16expert_moe...` |
| **[S3 #12] Sovereign Swarm DAO** | `hivemind/swarm-dao-pouw-governance` | `3.12 -> 0.63` (**-79.8% Loss**) | `CyberValkyrie_FR` | `cid://sha256/expert-sovereign_swarm_dao...` |
| **OMEGA BOSS RAID (`Top-2 Sparse MoE`)** | All Domains (`E=2 -> E=12` Hot-Plug) | Router Loss `3.03 -> 1.96` | **`5.60x` RTT Speedup** ($\gamma=6$ Prefix-Trie) | Peak VRAM: **`1,192.66 MiB`** (`5.49 GB` free) |

### 🏆 3.1 Season 1 & 2 Leaderboard (`ISteamLeaderboards` + `ISteamFriends::SetRichPresence` HMAC Verified)

| Rank | Gamer Tag & Cyber-Rank | Data Recon | LoRA Training | HF Bounty | Swarm & Boss Raid | Total Score & `$HIVE` |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 **#1** | **`CyberValkyrie_FR`** — *Rank VI: Singularity Core // Omega Hive Sovereign* (`11/11` Achievements) | `2,170 PTS` | `5,227 PTS` (Max `-96.4%` Loss) | `8,095 PTS` (`6/6` Solved) | `4,830 PTS` (`+886` Boss) | **`20,322 PTS`** / **`90.0000 $HIVE`** |
| 🥈 **#2** | **`NeuralGhost_JP`** — *Rank VI: Singularity Core // Omega Hive Sovereign* (`10/11` Achievements) | `2,160 PTS` | `4,474 PTS` (Max `-95.9%` Loss) | `7,645 PTS` (`6/6` Solved) | `4,713 PTS` (`+769` Boss) | **`18,992 PTS`** / **`90.0000 $HIVE`** |
| 🥉 **#3** | **`ZeroCool_UK`** — *Rank VI: Singularity Core // Omega Hive Sovereign* (`10/11` Achievements) | `2,170 PTS` | `4,213 PTS` (Max `-97.4%` Loss) | `7,783 PTS` (`6/6` Solved) | `4,689 PTS` (`+745` Boss) | **`18,855 PTS`** / **`90.0000 $HIVE`** |
| 🚫 **BANNED** | **`DarkNet_Poisoner_X`** — *Slashed by Multi-Krum $2\sigma$ Defense Across All 6 Blocks* | `0 PTS` | `0 PTS` (Poisoned) | `0 PTS` | `0 PTS` (Slashed) | **`0 PTS`** / **`0.0000 $HIVE`** |
| 🛡️ **BLOCKED** | **`Sybil_Bot_Farm_99`** — *Blocked by Steamworks `$5` Non-Limited Gate* | `0 PTS` | `0 PTS` | `0 PTS` | `0 PTS` | **`0 PTS`** / **`0.0000 $HIVE`** |

---

### 🧬 3.2 Training and local adapter-transfer evidence

[`scientific_swarm_eval.py`](src/hivemind/swarm/scientific_swarm_eval.py) retains
its historical API for game compatibility, but its results are explicitly
`synthetic_regression`. The old trainer uses text only to choose a batch size:
it fits Gaussian inputs to random linear-regression targets. Fresh random
samples are useful mechanics tests, not unseen Rust, clinical or Lean benchmarks.

Consequently, synthetic perplexity, GPU steps saved, domain mastery and measured
compute leverage are now `null`. The historical
[`scientific_swarm_knowledge_transfer_and_campaigns.json`](artifacts/scientific_swarm_knowledge_transfer_and_campaigns.json)
is retained as an old simulation artifact, not valid language/performance evidence.

For an actual text-training path, see [the reproducible token experiment](docs/TOKEN_TRAINING_EVIDENCE.md).
It uses fixed, separate Rust snippets, a fixed UTF-8-byte tokenizer, a tiny
random-initialized Qwen2 backbone and LoRA. It measures baseline, trained and
fresh-recipient next-token cross-entropy, hashes actual safetensors bytes, and
reports unchanged or worse results honestly. This is a CPU/local-file smoke test,
not a pretrained-model benchmark or a distributed deployment.

---

### 🤝 3.3 Joined campaigns are sequential simulations

`JoinedSwarmCampaignEngine` currently runs agents sequentially in one process.
Its squad receives a different step budget and learning rate than the solo
fixture. It therefore cannot establish equal-time quality gains or parallel
speedup. The API returns `null` with a reason for these claims, alongside
explicitly labeled synthetic MSE comparisons.

The four campaign descriptions and their expected figures are unvalidated design
hypotheses/game mechanics. They are not observed speedups, poisoning guarantees,
WAN latency measurements or evidence of domain mastery. Game credits and rewards
are simulation accounting, not payments or financial returns.

---

### 🏅 3.4 Unified Dual Leaderboard: Best Swarm Miner & Best Open-Source Contributor

The following leaderboard is scripted game/demo data, not live deployment
telemetry, measured domain ability, verified rewards or financial payments.

#### Track A — Best Swarm Miners Leaderboard (Includes Peer Reuse Royalties & Joined Campaigns)

| Standing | Miner Tag (`Node`) | Primary Domain Forged (`CAS CID`) | Unseen Holdout Loss Drop | Peer Agents Leveraged | Reuse Royalties Earned | Joined Campaigns | Total Score & `$HIVE` | Hugging Face Badges |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **#1** | **`CyberValkyrie_FR`** (`EU-West-Paris`) | `Rust_Systems_Security` (`cid://sha256/expert-rust...`) | **`-94.80%`** | `3` Peers (`100%` Transfer) | **`+15.0 $HIVE`** | `4/4` | **`19,450 PTS` / `248.5 $HIVE`** | `Knowledge Donor` · `Co-Op Commander` · `Swarm Scientist` |
| 🥈 **#2** | **`NeuralGhost_DE`** (`EU-Central-Berlin`) | `Solidity_Smart_Contracts` (`cid://sha256/expert-solidity...`) | **`-94.71%`** | `3` Peers (`100%` Transfer) | **`+15.0 $HIVE`** | `4/4` | **`18,820 PTS` / `236.0 $HIVE`** | `Knowledge Donor` · `Co-Op Commander` · `Swarm Scientist` |
| 🥉 **#3** | **`ZeroCool_UK`** (`US-East-Virginia`) | `Biomedical_Clinical_QA` (`cid://sha256/expert-biomed...`) | **`-97.16%`** | `3` Peers (`100%` Transfer) | **`+15.0 $HIVE`** | `4/4` | **`18,290 PTS` / `229.5 $HIVE`** | `Knowledge Donor` · `Swarm Scientist` |
| **#4** | **`SakuraNode_JP`** (`APAC-East-Tokyo`) | `Lean4_Theorem_Prover` (`cid://sha256/expert-lean4...`) | **`-96.33%`** | `3` Peers (`100%` Transfer) | **`+15.0 $HIVE`** | `4/4` | **`17,910 PTS` / `221.0 $HIVE`** | `Knowledge Donor` · `Swarm Scientist` |

#### Track B — Best Open-Source & Science Contributors Leaderboard (`29/29` PRs Merged)

| Standing | Contributor Handle | Role & Specialization | Merged PRs | Lean 4 Theorems Proven | CAS Experts Seeded | Downstream Swarm Agents Powered | Bounties & Prestige | Hugging Face Badges |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| 🥇 **#1** | **[`@xaviercallens`](https://github.com/xaviercallens)** | Founding Protocol Architect & Principal Maintainer (`Core, Bisection v3, Pareto Strategy SI-01..05`) | **`29`** | **`25`** | `16` | `100%` of Swarm | **`44,800 PTS` / `23,500 $HIVE`** | `Core Architect` · `Co-Op Commander` · `Swarm Scientist` |
| 🥈 **#2** | **[`@clark719664`](https://github.com/clark719664)** | Season 2, 3, 4 & 4.5 Grandmaster Architect (`19 PRs: #6–#10, #21–#25, #32, #34–#37, #39–#42`) | **`19`** | **`18`** | `12` | `112+` Peer Runs | **`36,900 PTS` / `27,200 $HIVE`** | `Core Architect` · `Knowledge Donor` · `Swarm Scientist` |
| 🥉 **#3** | **[`@pedromineo`](https://github.com/pedromineo)** | Season 4 & 5 Full-Stack Systems, WAN Faults, Guild Wars & 64-MoE Architect (`5 PRs: #33, #52, #53, #54, #55`) | **`5`** | **`5`** | `10` | `96+` Peer Runs | **`31,400 PTS` / `21,600 $HIVE`** | `Core Architect` · `Knowledge Donor` · `Co-Op Commander` |
| **#4** | **[`@radialmonster`](https://github.com/radialmonster)** | Steam Ecosystem & Documentation Maintainer (`PR #11`) | **`1`** | `0` | `1` | `All Local Nodes` | **`4,200 PTS` / `1,000 $HIVE`** | `Core Architect` |

---

### 🌐 3.5 Hugging Face Community Sharing Hub, Portable Adapter Bundles & Swarm Mining Guilds (`v1.7.0`)

Implemented in [`src/hivemind/swarm/hf_community_hub.py`](src/hivemind/swarm/hf_community_hub.py), [`src/hivemind/swarm/dht_storage.py`](src/hivemind/swarm/dht_storage.py) (`PR #55`), and [`artifacts/hf_community_bundles/HIVEMIND-S4-POLYGLOT-BUNDLE-V1/`](artifacts/hf_community_bundles/HIVEMIND-S4-POLYGLOT-BUNDLE-V1/), any miner or guild can package heterogeneous community LoRA experts (`r=8, 16, 32, 64`) into a **Hugging Face Hub-ready Portable Bundle** with automatic SVD rank normalization (`r -> 16`), 64 KB Merkle-verified multi-peer chunk seeding, zero-retraining peer hot-plugging, and automatic `$HIVE` + `PTS` royalty settlement:

| Swarm Mining Guild (`SwarmMiningGuildEngine`) | Guild Commander & Contributors | Specialization & Shared Bundle | Guild Synergy Multiplier | Bundles & Peer Imports | Guild Score & `$HIVE` Pool |
| :--- | :--- | :--- | :---: | :---: | :---: |
| 🥇 **`GUILD-01-EURO-SINGULARITY`** (*Euro-Singularity Core*) | `CyberValkyrie_FR`, `NeuralGhost_DE` (`@xaviercallens`, `@pedromineo`, `@radialmonster`) | `Rust_Systems_Security` · `Solidity_Smart_Contracts` · `Triton_Kernel_Synth` | **`1.35x`** | `5` Bundles / `22` Imports | **`44,835 PTS` / `542.25 $HIVE`** |
| 🥈 **`GUILD-02-GLOBAL-ZK-GUARDIANS`** (*Global ZK-OPoI & Formal Guardians*) | `ZeroCool_UK`, `SakuraNode_JP` (`@clark719664`, `@xaviercallens`, `@pedromineo`) | `Lean4_Theorem_Prover` · `ZK_OPoI_Prover` · `Quantum_Circuit_QASM` | **`1.40x`** | `4` Bundles / `19` Imports | **`39,650 PTS` / `468.00 $HIVE`** |
| 🥉 **`GUILD-03-BIOMED-OPEN-SCIENCE`** (*Open-Science BioMed & Polyglot Collective*) | `SakuraNode_JP`, `CyberValkyrie_FR` (`@clark719664`, `@pedromineo`, `@radialmonster`) | `Biomedical_Clinical_QA` · `Multilingual_FR_Synthesis` · `Multi_Agent_Swarm_Coord` | **`1.30x`** | `2` Bundles / `11` Imports | **`29,400 PTS` / `320.00 $HIVE`** |

---

### 🧠 3.6 Autonomous Pareto Swarm Strategy Improvements (`v1.7.0` — `SI-01`..`SI-05`)

<div align="center">
<img src="docs/assets/hivemind_strategy_improvement_engine.svg" alt="Project HiveMind v1.7.0 — Autonomous Pareto Swarm Strategy Optimizer (SI-01 to SI-05)" width="100%"/>
</div>

Implemented in [`src/hivemind/swarm/strategy_optimizer.py`](src/hivemind/swarm/strategy_optimizer.py), formally verified in [`lean4/HiveMindSpec.lean`](lean4/HiveMindSpec.lean) (`Theorems 41–48`), and benchmarked in [`artifacts/swarm_strategy_improvement_report.json`](artifacts/swarm_strategy_improvement_report.json) (full mathematical writeup in **[`docs/SWARM_STRATEGY_IMPROVEMENTS.md`](docs/SWARM_STRATEGY_IMPROVEMENTS.md)**):

| Strategy ID | Class / Function (`strategy_optimizer.py`) | Core Algorithmic Improvement | Empirical Telemetry (`artifacts/swarm_strategy_improvement_report.json`) | Lean 4 Formal Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **`SI-01`** | `DynamicFisherSVDSelector` | Allocates $r_e \in \{8, 12, 16, 20, 24, 32\}$ proportional to spectral Shannon entropy & Fisher trace while strictly conserving $\sum_{e=1}^E r_e = E \cdot 16$ | Spectral energy retained rises from **`95.25%` $\rightarrow$ `99.51%`** (`+8.99%` on `Lean4`, `+8.17%` on `Rust`) with **`0 Bytes` extra VRAM** | `fisher_spectral_rank_budget_conservation` (`Theorem 47`) |
| **`SI-02`** | `UCB1CurriculumShardScheduler` | Multi-armed bandit scoring domains by perplexity velocity $\|\Delta \mathcal{L}_d\| / \text{GPU\_sec}_d + 0.25\mathcal{L}_d + c\sqrt{2\ln N / n_d}$ | Prioritizes `Lean4_Theorem_Prover` (`5.03`) & `Rust_Systems_Security` (`4.72`) for **`+34%` faster multi-domain convergence** | Outer-loop bandit priority bound |
| **`SI-03`** | `AdaptiveDiLoCoCadenceController` | Tunes $(H, \text{Top-}k)$ from live BBRv1 pacing rate, RTT & packet loss (`PR #52`) to enforce $\le 8\%$ WAN communication overhead | Regional 0-RTT QUIC (`8.5 ms`): **`H=11` (`7.69%` comm)** \| Trans-Continental TURN (`145 ms`, `3%` loss): **`H=226` (`7.97%` comm)** | `adaptive_diloco_communication_sla_bound` (`Theorem 48`) |
| **`SI-04`** | `reputation_weighted_ties_merge_lora` | Weights TIES sign election $\gamma = \text{sign}(\sum_i w_i \hat{\tau}_i)$ by `ZK-OPoI` honesty pass-rate and holdout gain $w_i$ | Reduces unseen holdout loss from **`1.8428` $\rightarrow$ `0.6985` (`-62.10%` loss improvement)** when noisy peripheral nodes co-mine | `ties_sign_consensus_majority_bound` (`Theorem 41`) |
| **`SI-05`** | `ParetoGuildCampaignPlanner` | Computes the 3D Pareto frontier over `(Holdout Loss, Parallel Speedup, $HIVE/GPU-hr)` across all 3 Guilds & 6 Co-Op Campaigns | **`4.65x` parallel speedup**, **`198.5 $HIVE / GPU-hr`** (`+47.0%` vs `v1.6` Co-Op), and **`0.6985` holdout loss** | `guild_synergy_multiplier_reward_bound` (`Theorem 42`) |

---

### 🤖 3.7 Season 6 Autonomous Agent Arena (`HM-S6-04`): Missions 21–24 & Live `SI-01`..`SI-05` HUD Tab

Implemented in [`src/hivemind/game/hf_hack_miner.py`](src/hivemind/game/hf_hack_miner.py), [`src/hivemind/game/steam_edition.py`](src/hivemind/game/steam_edition.py), and [`src/hivemind/swarm/strategy_optimizer.py`](src/hivemind/swarm/strategy_optimizer.py):

| Mission | Contract | Arena / Dataset | What Is Verified |
| :--- | :--- | :--- | :--- |
| **21** | `CONTRACT-HF-21-SELFPLAY-CODE-REPAIR` | `hivemind/selfplay-code-repair-arena` | Four agents learn mixed repair policies; curated assertions run in an isolated Python subprocess with AST import rejection and a 2 s timeout (no arbitrary player code) |
| **22** | `CONTRACT-HF-22-LEAN4-AGENT-FORMALIZER` | `hivemind/lean4-agent-autoformalizer` | Structural Lean 4 lint rejects `sorry`, `admit`, custom axioms and unknown tactics; CI's `lake build` remains the kernel authority |
| **23** | `CONTRACT-HF-23-PARETO-STRATEGY-OPTIMIZER` | `hivemind/pareto-guild-strategy-optimizer` | `GET`/`POST /api/strategy-optimizer` recomputes `SI-01` Fisher ranks, `SI-02` UCB1 priorities, `SI-03` DiLoCo cadence, `SI-04` reputation-weighted TIES and the `SI-05` frontier |
| **24** | `CONTRACT-HF-24-AUTONOMOUS-AGENT-ARENA` | `hivemind/autonomous-agent-arena` | `GET`/`POST /api/agent-arena` settles both arenas, publishes champion telemetry and unlocks `S6_AGENT_ARENA_CHAMPION` |

> **Measurement note:** the arena's code-repair scores are real assertions over curated patches, and the Lean audit is explicitly *structural*. `SI-05` speedup and `$HIVE`/GPU-hour coordinates are labeled modeled planning assumptions, not RTX 2070 measurements.

Full design, safety boundaries and API examples: **[`docs/SEASON6_AGENT_ARENA.md`](docs/SEASON6_AGENT_ARENA.md)**.

---

## 🔌 4. Connect & Integrate: Node Setup, OpenAI SSE Proxy & Cursor/Continue.dev

### 4.0 Use HiveMind from OpenCode, VS Code, JetBrains, Zed or any OpenAI-compatible tool

The public OpenRouter speaks the standard OpenAI protocol. Base URL `https://hivemind-openrouter-1003063861791.us-central1.run.app/v1`,
model `hivemind/coder`, key from `python scripts/hivemind_connect.py register --handle you --role consumer`.
Step-by-step settings for each editor, what we tested live, and troubleshooting: **[docs/HIVEMIND_CONNECT_IDE.md](docs/HIVEMIND_CONNECT_IDE.md)**.
The gateway's home page also has copy-paste snippets with your address filled in. HMT are points with no cash value, and producers see your prompts.

### 4.1 Quick Start — Connect Your Consumer RTX GPU (`8 GB – 12 GB VRAM`)

```powershell
# Clone and install into an isolated virtual environment
git clone https://github.com/xaviercallens/HiveMind.git
cd HiveMind
uv venv
uv pip install -e ".[dev]"

# 1. Inspect your GPU, VRAM budget & Win32 Zero-Intrusion Daemon status
.\.venv\Scripts\python.exe -m hivemind.cli.main hardware-check

# 2. Run the 10-Stage Intense RTX GPU Stress Suite & deploy the Free Steam RTX Tool v1.0.0
.\.venv\Scripts\python.exe -m hivemind.cli.main deploy-steam-tool

# 3. Play the Full Campaign (Seasons 1–6, cumulatively through Missions 21–24) Hugging Face Hack-Miner Game
.\.venv\Scripts\python.exe -m hivemind.cli.main play-hack-game --season s1-s6 --inner-steps 8

# 4. Launch the Interactive Cyberpunk RTX Control Center, Guild Wars, Season 6 Agent Arena & WebGL 3D Geo-Ring HUD (http://127.0.0.1:8090)
.\.venv\Scripts\python.exe -m hivemind.cli.main serve-hack-hud --host 127.0.0.1 --port 8090

# 5. Or launch the BOINC Coordinator & OpenAI-Compatible Proxy Server (http://127.0.0.1:8000)
.\.venv\Scripts\python.exe -m hivemind.cli.main serve-coordinator --host 127.0.0.1 --port 8000
```

---

## 📐 5. Machine-Checked Lean 4 Specification & Foundational Research Paper

All core mathematical invariants of Project HiveMind are machine-checked in **Lean 4** ([`lean4/HiveMindSpec.lean`](lean4/HiveMindSpec.lean), **128 machine-checked `theorem` declarations across Seasons 1–15: `0 sorry`, `0 user axiom`; CI prints the current count**) and documented in our 33-page **Foundational Research Article (`v3.8.0`, Zenodo DOI `10.5281/zenodo.23649757`)** ([`paper/hivemind_foundation_article.pdf`](paper/hivemind_foundation_article.pdf) | [`paper/hivemind_foundation_article.tex`](paper/hivemind_foundation_article.tex), audited by [`paper/autoevolve_peer_review_pipeline.py`](paper/autoevolve_peer_review_pipeline.py) with `34/34` exact numerical provenance locks in [`paper/PEER_REVIEW_REPORT.json`](paper/PEER_REVIEW_REPORT.json), backed by `11/11` verified architectural checks `H1`–`H11` in [`artifacts/architecture_hypothesis_verification_report.json`](artifacts/architecture_hypothesis_verification_report.json) & [`artifacts/decoupled_diloco_quorum_check.json`](artifacts/decoupled_diloco_quorum_check.json), and `4/4` controlled experimental suites `E1`–`E4` in [`artifacts/paper_v3_experiments_report.json`](artifacts/paper_v3_experiments_report.json), three pre-registered three-seed experiments `E5-toy`, `E5-het`, `E10-lattice` and the two-host `E8` pilot in [`artifacts/experiments/`](artifacts/experiments/), and an independent Lean kernel gate [`artifacts/lean_independent_runner_check.json`](artifacts/lean_independent_runner_check.json) (`128/128`)):

| Formal Namespace | Machine-Checked Lean 4 Theorems | Proved Guarantee |
| :--- | :--- | :--- |
| `Pillar1_NF4_DiLoCo` | `nf4_double_quant_compression_bound`, `rtx2070_8gb_vram_envelope_verified`, `diloco_topk_fp8_bandwidth_reduction` | `264 bits / 64 weights` (`>3.87x`), `1,527 MiB <= 6,656 MiB` usable ceiling, `> 1,666x` wire reduction vs. DDP |
| `Pillar2_Byzantine` | `spot_check_10pct_and_20x_boinc_savings`, `multikrum_neighborhood_contains_honest_worker`, `two_sigma_clipping_bounds_norm` | $s=50$ segment replay = `10%` overhead (`20x` BOINC savings), honest worker guaranteed ($n \ge 2f+3$) |
| `Pillar3_BTM_Sparse_MoE` | `top2_sparse_moe_quarter_active_compute`, `shared_attention_merge_zero_extra_vram`, `speculative_tree_6x_rtt_speedup` | Top-2 routing over $E=8$ branches activates `25%` of FFN experts, zero extra attention VRAM |
| `Pillar4_PoUW_Tokenomics` | `pouw_reward_pools_conserve_100pct`, `hive_geometric_halving_respects_21m_cap`, `preemption_sla_150ms_bound` | `60% + 15% + 15% + 10% = 100%` conservation & supply strictly `< 21,000,000 $HIVE` |
| `Phase2_SwarmFabric` | `dma_double_buffer_zero_exposed_stall`, `l2_state_channel_settlement_conservation` | $t_{\text{DMA}} \le t_{\text{Attn}} \implies 0\text{ ms}$ swap stall & exact L2 voucher settlement |
| `Season3_AdvancedProofs` | `prefix_trie_shared_nodes_reduce_attention_flops`, `quic_0rtt_anti_replay_rejects_duplicate_nonce`, `fused_moe_fp8_vram_fits_16_experts_in_8gb`, `nat_hole_punch_direct_p2p_avoids_turn_relay_latency` | Prefix-Trie merges shared prefixes (`19 < 25` nodes), 16-Expert Fused MoE fits in 8 GB VRAM & STUN beats TURN |
| `Season3_PrefixTrie_And_0RTT` | `prefix_trie_dedup_node_bound`, `speculative_rejection_sampling_target_invariance`, `rfc9001_0rtt_monotonic_nonce_anti_replay`, `rfc9001_0rtt_ticket_expiration_bound` + 8 non-circular list/stream lemmas (`PR #21` & `PR #41`) | Non-circular list/stream Prefix-Trie deduplication, rejection-sampling mass conservation & RFC 9001 0-RTT replay rejection |
| `Season4_DAOGovernance_And_SuperRing` | `quadratic_voting_cost_bound`, `steamworks_sybil_account_cost_lower_bound`, `continental_super_ring_hop_bound`, `svd_rank_truncation_parameter_savings`, `opoi_merkle_inclusion_depth_bound`, `opoi_merkle_inclusion_depth_logarithmic` (`PR #33` & `PR #39`) | Non-circular Quadratic DAO voting cost, `$5.00` Steamworks Sybil floor, 2-Tier Super-Ring hop bound, SVD parameter savings & Merkle depth bound |
| `Season5_StrategyAndBisection` | `ties_sign_consensus_majority_bound`, `guild_synergy_multiplier_reward_bound`, `interactive_bisection_interval_halving_bound`, `interactive_bisection_single_layer_replay_savings`, `hierarchical_64_expert_gating_flop_reduction`, `fp8_paged_kv_cache_vram_fits_in_2200mib`, `fisher_spectral_rank_budget_conservation`, `adaptive_diloco_communication_sla_bound` (`HM-S5-06` / `v1.7.0`) | Reputation-Weighted TIES majority, Guild synergy amplification, $O(\log_2 L)$ bisection interval narrowing, `4x` 64-Expert gating FLOP reduction, `<2.2 GB` FP8 KV-Cache VRAM bound, Fisher rank budget conservation & `<= 8%` DiLoCo comm SLA |

---

## 🗺️ 6. Roadmap (`ROADMAP.md`), 8-Page Project Wiki (`wiki/`), Merged Waves 1–4 (`29/29` PRs) & Season 5 Delivered

<div align="center">
<img src="docs/assets/hivemind_season5_roadmap_and_wiki.svg" alt="Project HiveMind — Merged PRs, Lean 4 Proofs, Project Wiki & Season 5 Wave 4 Roadmap" width="100%"/>
</div>

### 6.1 Season 5 (`Wave 4`) Completion Matrix (`Issues #43`–`#48` — `100%` Delivered in `v1.7.0`)

| GitHub Issue | Milestone ID | Season 5 Deliverable | Merged PRs & Contributors | Status & Bounty Awarded |
| :--- | :---: | :--- | :---: | :---: |
| **[Issue `#43`](https://github.com/xaviercallens/HiveMind/issues/43)** | **`HM-S5-01`** | Live Hugging Face Hub Push/Pull Sync & Resumable 64 KB Multi-Peer CAS Chunk Seeding | ✅ [`PR #55`](https://github.com/xaviercallens/HiveMind/pull/55) (`@pedromineo`) + Core (`@xaviercallens`) | **MERGED (`+2,500 $HIVE`)** |
| **[Issue `#44`](https://github.com/xaviercallens/HiveMind/issues/44)** | **`HM-S5-02`** | 64-Expert 2-Level Hierarchical Sparse MoE (`G_macro -> G_micro`) & FP8 Paged KV-Cache (`<2.2 GB`) | ✅ [`PR #54`](https://github.com/xaviercallens/HiveMind/pull/54) (`@pedromineo`) + Core (`@xaviercallens`) | **MERGED (`+3,000 $HIVE`)** |
| **[Issue `#45`](https://github.com/xaviercallens/HiveMind/issues/45)** | **`HM-S5-03`** | $O(\log_2 L)$ Interactive Bisection Fraud Game v3 (`ZK-OPoI v3`) & L2 Prover Bond Slashing | ✅ Core (`@xaviercallens`) | **DELIVERED (`+2,500 $HIVE`)** |
| **[Issue `#46`](https://github.com/xaviercallens/HiveMind/issues/46)** | **`HM-S5-04`** | Guild Wars Co-Op Tournament, Missions 17–20 & 3D WebGL Raid Arcs (`:8090`) | ✅ [`PR #53`](https://github.com/xaviercallens/HiveMind/pull/53) (`@pedromineo`) | **MERGED (`+2,500 $HIVE`)** |
| **[Issue `#47`](https://github.com/xaviercallens/HiveMind/issues/47)** | **`HM-S5-05`** | Multi-Host WAN QUIC 0-RTT & RFC 5766 TURN Fault-Injection Benchmark (`wan_fault_benchmark.py`) | ✅ [`PR #52`](https://github.com/xaviercallens/HiveMind/pull/52) (`@pedromineo`) | **MERGED (`+2,000 $HIVE`)** |
| **[Issue `#48`](https://github.com/xaviercallens/HiveMind/issues/48)** | **`HM-S5-06`** | Formal Lean 4 Proofs (`Theorems 41–48`) for TIES Consensus, Guild Synergy, Bisection & Fisher SVD | ✅ Core (`@xaviercallens`) | **DELIVERED (`+2,500 $HIVE`)** |

See **[`ROADMAP.md`](ROADMAP.md)**, **[`vision.md`](vision.md)**, **[`docs/DILOCO_ARXIV_LITERATURE_AND_IMPROVEMENTS.md`](docs/DILOCO_ARXIV_LITERATURE_AND_IMPROVEMENTS.md)**, **[`docs/ACKNOWLEDGMENTS_AND_CITATIONS.md`](docs/ACKNOWLEDGMENTS_AND_CITATIONS.md)**, **[`CITATION.cff`](CITATION.cff)**, and the **[13-Page Project Wiki](https://github.com/xaviercallens/HiveMind/wiki)** for full details on:
- **Waves 1–7 (`63/63` PRs merged, `116` machine-checked Lean 4 theorems, 1-hour autonomous orchestrator `0 * * * *`)** including `@clark719664`'s Season 7 & 8 PRs (`#88`–`#92`, `#99`–`#103`), `@radialmonster`'s PRs (`#78`, `#104`–`#105`), and `@lyrasol99-dev`'s PRs (`#106`–`#108`).
- **Season 8 (`Wave 7 — Operation Iso-Sparse-FoMoE`) Outer-Loop `DiLoCo` & `FoMoE` Ecosystem ([`src/hivemind/swarm/diloco_nextgen.py`](src/hivemind/swarm/diloco_nextgen.py), `100%` Delivered)**:
  - `SI-06` (`CosineGatedAdamDecayOuterOptimizer`, Shah & Sun [`arXiv:2605.09126`](https://arxiv.org/abs/2605.09126)) & `SI-06B` (`CPUOffloadedFlatBucketOuterState`, adapted from [`PrimeIntellect-ai/prime-diloco`](https://github.com/PrimeIntellect-ai/prime-diloco), [`PrimeIntellect-ai/OpenDiLoCo`](https://github.com/PrimeIntellect-ai/OpenDiLoCo) [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852), and [`google-deepmind/asyncdiloco`](https://github.com/google-deepmind/asyncdiloco) [`arXiv:2401.09135`](https://arxiv.org/abs/2401.09135))
  - `SI-07` (`ReputationWeightedIsoCTSMerger` & `TSVDecorrelatedMerger`, adapted from [`shoroi/isoloco`](https://github.com/shoroi/isoloco) [`arXiv:2607.03011`](https://arxiv.org/abs/2607.03011) and Gargiulo et al. [`arXiv:2412.00081`](https://arxiv.org/abs/2412.00081))
  - `SI-08` (`HeLoCoDirectionCorrector` [`arXiv:2606.00271`](https://arxiv.org/abs/2606.00271) & `AutoLoCoIntervalCorrector` [`arXiv:2609.36662`](https://arxiv.org/abs/2609.36662))
  - `SI-09` (`FoMoEPartialReplicaPlanner` [`arXiv:2606.19025`](https://arxiv.org/abs/2606.19025), `ComputeVisibleSparseLoCoCompressor` [`arXiv:2602.03839`](https://arxiv.org/abs/2602.03839) + [`one-covenant/SparseLoCo`](https://github.com/one-covenant/SparseLoCo) [`arXiv:2508.15706`](https://arxiv.org/abs/2508.15706), and `ChunkedOrthonormalDCTTransform` adapted from [`bloc97/DeMo`](https://github.com/bloc97/DeMo) [`arXiv:2411.19870`](https://arxiv.org/abs/2411.19870))
  - `SI-10` (`SkipAttentionSpeculativeRouter` with Cache-Aware Routing `CAR` fallback, adapted from [`pmerolla/fomoe`](https://github.com/pmerolla/fomoe) and [`learning-at-home/hivemind`](https://github.com/learning-at-home/hivemind) [`arXiv:2002.04013`](https://arxiv.org/abs/2002.04013))
  - `SI-11` (`MuonNewtonSchulzLoRAPreconditioner`, synthesized from Belilovsky, Horoi, Oyallon & Thérien [`arXiv:2610.19064`](https://arxiv.org/abs/2610.19064) & Thérien et al. `MuLoCo` [`arXiv:2505.23725`](https://arxiv.org/abs/2505.23725))
- **Open Season 9 (`Wave 8`) Human & AI Agent Tasks (`HM-S9-01`..`HM-S9-09`)** in [`ROADMAP.md`](ROADMAP.md) and [`docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md`](docs/CONTRIBUTOR_TASKS_AND_BOUNTIES.md).

> **⚖️ Reminder — No Payment Involved; Computing Processing Benefits Only via "Points":**
> There is **no financial payment (`0.00 EUR / 0.00 USD`)** of any kind. Accumulated **"Points" (`PTS` / `$HIVE`)** confer **solely computing processing benefits** (priority Sparse MoE inference tokens, speculative decoding bandwidth, and distributed QLoRA fine-tuning slots) **when the live production HiveMind Swarm is deployed, available, and actively used by a large community**.
