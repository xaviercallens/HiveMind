# Massive Weekend Swarm Training Raid (14–15 November 2026) — `CrowdGPT-MoE` x `HiveMind` Decoupled OpenDiLoCo

> [!IMPORTANT]
> **Status, Saturday 14 November 2026 (supersedes the plan below where they differ).**
> - **Crowd-v1 is a dense model and stays whole.** Its author (Vxtzq) asked that it not be treated as a MoE or split into parts. HiveMind does **not** run the sparse-MoE upcycle of Crowd-v1 described below; that design is parked.
> - **What runs this weekend:** `hivemind/crowd-v1` is served live by the HiveMind gateway (OpenAI-compatible API). The Tesla T4 (Spot, 14:00–22:00 Europe/Paris) is available to CrowdGPT as a GPU evaluator through `scripts/crowdgpt_loss_check_agent.py` once CrowdGPT issues an evaluator key: it takes loss-check jobs from CrowdGPT's server, verifies every file's SHA-256 and works only while the governor reports the GPU idle.
> - **How others can help:** producers serve models (outbound HTTPS only, no open port); prosumers test `hivemind/crowd-v1` in their IDE and report bad answers; GPU owners can run the CrowdGPT loss-check agent if CrowdGPT gives them a key.
> - **Train Crowd-v1 with your GPU:** run CrowdGPT's own client (https://github.com/Vxtzq/CrowdGPT: one-command install or Docker, then `crowdgpt --headless`, `--anon` without an account, `--vram-gb N` to cap memory, `--doctor` to check the setup). It trains the shared dense model and submits each round to CrowdGPT's server; HiveMind's T4 does exactly this from 14 November (round 1 submitted).
> - **Not available this weekend:** HiveMind's own OpenTrainer weight-sharing path. The OpenTrainer agent currently reports numbers only, not weights, and the gateway records them as `CLIENT_REPORTED_UNVERIFIED`. **No HMT is credited per training round**, and the gateway's sign-up rewards are off; the "+250 to +1,000 HMT" milestones below are not active.
> - Machine-readable version for agents: [`docs/announcements/2026-11-14-crowdgpt-weekend.json`](announcements/2026-11-14-crowdgpt-weekend.json).

> **Campaign ID**: `CROWDGPT-WE-NOV14-15-2026-MOE-OPENDILOCO` · **Public Model Target**: `hivemind/crowd-moe-v1` (`Vxtzq/Crowd-v1` `1.01B` Sparse-MoE Upcycle)  
> **Dates**: Friday **13 Nov 2026** (Preflight) → Saturday **14 Nov** & Sunday **15 Nov 2026** (6 Global Quorum Waves)  
> **Calendar Export (`.ics`)**: [`docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics`](calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics)  
> **Live Raid Manifest API**: `GET https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/opentrainer/crowdgpt-weekend-raid`  
> **Discord Raid Hub**: https://discord.gg/ZeFtZpKHf  
> **Non-Monetary Charter**: All `HMT` Reflexion Points (`+250` to `+1,000 HMT` raid milestones) are strictly **non-monetary reciprocal compute credits** (`zero financial payment`, `zero cash value`), redeemable solely for shared OpenAI-compatible inference and training priority across the HiveMind swarm.

---

## 1. Sparse MoE Split + Decoupled OpenDiLoCo Architecture (`src/hivemind/swarm/crowdgpt_moe_opendiloco.py`)

### 1.1 Function-Preserving `8`-Expert Sparse MoE Upcycling of `Vxtzq/Crowd-v1`
`Vxtzq/Crowd-v1` comprises `20` causal Transformer blocks (`hidden_size = 1280`, SwiGLU `intermediate_size = 2560`, `657.8M` trunk parameters) paired with a `350.0M`-parameter bigram [`CrowdEngram`](../src/hivemind/swarm/crowdgpt_live_runtime.py) table (`1.01B` total parameters). We provide **two exact step-0 function-preserving Sparse MoE splits** in [`crowdgpt_moe_opendiloco.py`](../src/hivemind/swarm/crowdgpt_moe_opendiloco.py):

| Split Mode | Class | Algebraic Step-0 Equivalence | Active Compute / Token | Target Hardware Tier |
| :--- | :--- | :--- | :--- | :--- |
| **`fine_grained_width_split`** | `CrowdFineGrainedMoEMLP` | Splits `w1, w3 ∈ R^{2560×1280}` row-wise and `w2 ∈ R^{1280×2560}` column-wise into `E = 8` exact `320`-channel SwiGLU sub-experts (`8 × 320 = 2560`, **`0` extra trunk params**). Exact dense equality under unit sum-routing. | `Top-k = 2` of `8` (`25%` FFN FLOPs) or `Top-k = 4` (`50%` FFN FLOPs) | `16 GB` GCP Spot Tesla T4 (`>= 4,300 MiB` free) & `16–24 GB` Trainer GPUs |
| **`lora_domain_experts`** | `CrowdLoRAMoEMLP` | Keeps shared `CrowdRuntimeMLP` frozen in `fp16`/`bf16` and attaches `E = 8` zero-initialized `B` SwiGLU LoRA experts (`r = 8, α = 16`). At step `0`, `y_moe == y_dense` identically (`max |Δy| = 0`). | Shared trunk + `Top-k = 2` domain LoRA branches + **FoMoE Ghost-Expert Skip-Token** mask | `8 GB` Consumer GPUs (`RTX 2070` `>= 1,536 MiB` free, `2` resident experts) + `16 GB` Tesla T4 (`8` experts) |

**FoMoE `σ`-MoE Non-Competitive Routing (`arXiv:2603.05485`)**:
- Uses independent Sigmoid Top-$k$ gating (`sigmoid_independent_topk_routing` in [`fomoe_sigma_routing.py`](../src/hivemind/core/fomoe_sigma_routing.py)) instead of competitive Softmax so specialist experts (`E1_python_systems_code`, `E2_math_reasoning`, `E7_defensive_cyber`) never cannibalize general bilingual FR/EN fluency (`E0_general_fineweb`).
- **Ghost-Expert Skip-Token Support for `8 GB` Home GPUs**: When an `8 GB` Prosumer/Trainer hosts `2` local experts (e.g., `resident_expert_indices=(1, 2)`), tokens routed to remote experts pass cleanly through the frozen shared `CrowdRuntimeMLP` trunk (`0` NaNs, `0` OOMs) while local experts accumulate clean pseudo-gradients.

### 1.2 PrimeIntellect OpenDiLoCo + Google DeepMind Decoupled DiLoCo (`CrowdOpenDiLoCoCoordinator`)
`CrowdOpenDiLoCoCoordinator` unifies three peer-reviewed communication-efficient training mechanisms:
1. **PrimeIntellect OpenDiLoCo (`arXiv:2407.07852`)**:
   - **CPU-Offloaded FP32 Master Weights & Nesterov Momentum (`CPUOffloadedFlatBucketOuterState`)**: Keeps FP32 master weights and outer optimizer state in pinned host RAM (`0 MiB` GPU VRAM overhead!).
   - **Partitioned Streaming Fragments (`P = 4` router fragments + `E = 8` expert buckets)**: Streams per-expert and per-router bucket pseudo-gradients independently so slow home uplinks (`10–50 Mbps`) never stall on monolithic `1B`-parameter barriers.
   - Exports reproducible `PrimeIntellect-ai/OpenDiLoCo` Hivemind DHT / `torchrun` launch manifests (`export_opendiloco_launch_spec()`) alongside firewall-free HTTPS `TCP/443` HiveMind OpenTrainer relays.
2. **Google DeepMind Decoupled DiLoCo (`arXiv:2604.21428`)**:
   - **Quorum-of-$K$ (`select_quorum`) + `AdaptiveGraceWindowController`**: Advances each bucket as soon as $K$ heterogeneous GPUs report plus a bounded grace window $\delta_g$, eliminating straggler blocking between the Spot T4 and home RTX cards.
   - **Radial-Directional Averaging (`RDA`)**: Decouples step radial norm (geometric mean) from unit direction (weighted vector mean), preventing high-variance home-GPU step norms from destabilizing the shared MoE router.
3. **Staleness-Gated `PA-CGAD` (`arXiv:2605.09126`) + Monotone Validation-Loss Ratchet**:
   - Nocturnal and high-latency cohorts can switch `--outer-optimizer pa_cgad` (`CosineGatedAdamDecayOuterOptimizer`) to gate Adam momentum by directional cosine alignment $\gamma_t = \max(0, \cos(\bar{\Delta}_t, m_{t-1}))$ and staleness $1/(1 + \alpha \tau)$.
   - Every outer step is gated by a held-out **monotone validation-loss ratchet** (`eval_loss_after <= best_eval_loss + ε`) before advancing `theta_master`.

### 1.3 Curated 8-Expert S21 Dataset Plan (`configs/datasets_s21.json`)
`load_crowdgpt_moe_dataset_plan()` dynamically filters [`configs/datasets_s21.json`](../configs/datasets_s21.json):
- **`18` `ACCEPTED` Datasets Only** (`odc-by`, `cc-by-4.0`, `mit`, `apache-2.0`, `cc-by-sa-4.0` — strictly compatible with AGPL-3.0).
- **`6` Non-Commercial / Custom Candidates Automatically Blocked** (`allenai/sciq` `cc-by-nc-3.0`, `camel-ai/physics` `cc-by-nc-4.0`, `camel-ai/biology` `cc-by-nc-4.0`, `TheFinAI/finben-fomc` `custom`, etc.).
- **Verification & Decontamination**: `unverified_cap <= 0.30` and mandatory 13-gram decontamination against `HumanEval`, `MBPP`, and `ARC-Challenge`.

| Expert | Domain (`datasets_s21.json`) | Mix | Primary Allowlisted Datasets | Target Weekend Cohort |
| :--- | :--- | :---: | :--- | :--- |
| **`E0_general_fineweb`** | `general_pretrain` | `15%` | `HuggingFaceFW/fineweb-edu` | GCP Spot Tesla T4 + Prosumer Bilingual Anchor |
| **`E1_python_systems_code`** | `python_coding` | `22%` | `bigcode/self-oss-instruct-sc2-exec-filter-50k`, `nvidia/OpenCodeInstruct`, `ise-uiuc/Magicoder-OSS-Instruct-75K` | `XAVGAMSRV` RTX 2070 + Coding Trainers |
| **`E2_math_reasoning`** | `maths` | `16%` | `nvidia/OpenMathInstruct-2`, `AI-MO/NuminaMath-CoT`, `meta-math/MetaMathQA` | GCP Spot Tesla T4 + Math Trainers |
| **`E3_formal_logic_arc`** | `physics` / logic | `11%` | `allenai/ai2_arc` | Formal Logic Trainers & Prosumers |
| **`E4_physics_stem`** | `physics` | `10%` | `allenai/ai2_arc`, `maximus-21/PhysicsQA` | STEM Trainers & BOINC Nocturnal Nodes |
| **`E5_exoplanet_transit`** | `exoplanet` | `8%` | `bingbangboom/exoplanet-transit-detection`, `juliensimon/nasa-exoplanets` | Citizen-Science BOINC & Prosumer Cohort |
| **`E6_stellar_spectroscopy`** | `stars` | `8%` | `OneAstronomy/sspp-dr9-stellar-spectra`, `jeff-vincent/astronomy-textbook-qa-context` | GCP Spot Tesla T4 (`bee-stars`) + Astro Cohort |
| **`E7_defensive_cyber`** | `cyber` | `10%` | `CyberNative/Code_Vulnerability_Security_DPO`, `nuhmanpk/cybersecurity-controls-instructions`, `Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset` | Producer Shield & Security Auditors |

---

## 2. Massive Weekend Raid Calendar (13–15 November 2026) & Role Matrix

Import [`docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics`](calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics) directly into Google Calendar, Apple Calendar, Thunderbird, or Proton Calendar:

| Wave | Date | UTC | CET (`Europe/Paris`) | EST (`New York`) | PST (`San Francisco`) | JST (`Tokyo`) | Quorum Focus & Milestone |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Wave 0** | **Fri 13 Nov** | `18:00–20:00` | `19:00–21:00` | `13:00–15:00` | `10:00–12:00` | `03:00–05:00 (+1)` | **Preflight Dry-Run**: Verify `Crowd-v1` cache, S21 shard hashes, VRAM floors (`>= 4,300 MiB` T4 / `>= 1,536 MiB` 8 GB), and OpenTrainer auth (`+250 HMT`). |
| **Wave 1** | **Sat 14 Nov** | `09:00–13:00` | `10:00–14:00` | `04:00–08:00` | `01:00–05:00` | `18:00–22:00` | **EU + APAC Kickoff**: Step-0 exact `8`-expert MoE split (`E0..E7`) + FoMoE `σ`-MoE router warm-start (`+500 HMT`). |
| **Wave 2** | **Sat 14 Nov** | `13:00–21:00` | `14:00–22:00` | `08:00–16:00` | `05:00–13:00` | `22:00–06:00 (+1)` | **Peak Global Sprint #1**: GCP Spot T4 extended window (`14:00–22:00 CET`) + all EU/Americas Producers, Trainers & Prosumers (`RDA` Quorum $K \ge 12$, `+1,000 HMT`). |
| **Wave 3** | **Sat 14 → Sun 15** | `22:00–08:00` | `23:00–09:00` | `17:00–03:00` | `14:00–00:00` | `07:00–17:00` | **Nocturnal Prosumer & APAC Relay**: Async `PA-CGAD` staleness-gated outer steps while Spot T4 sleeps (`\$0` overnight cloud burn, `+600 HMT`). |
| **Wave 4** | **Sun 15 Nov** | `13:00–18:00` | `14:00–19:00` | `08:00–13:00` | `05:00–10:00` | `22:00–03:00 (+1)` | **Peak Global Sprint #2**: Cross-domain expert consolidation (`E0..E7`) + Spot T4 outer `RDA` checkpointing (`+1,000 HMT`). |
| **Wave 5** | **Sun 15 Nov** | `18:00–21:00` | `19:00–22:00` | `13:00–16:00` | `10:00–13:00` | `03:00–06:00 (+1)` | **Blind A/B Arena & Release Freeze**: Live `hivemind/crowd-v1` vs `hivemind/crowd-moe-v1` evaluation + `safetensors` SHA-256 export for `@Vxtzq` (`+750 HMT`). |

### Role Commands for Every HiveMinder

1. **Producers (Inference GPUs — keep live `hivemind/crowd-v1` & Sunday `hivemind/crowd-moe-v1` low-latency)**:
   ```bash
   python scripts/hivemind_crowd_v1_producer.py --host 127.0.0.1 --port 8791 --threads 6
   ```
2. **Trainers (`8 GB–24 GB` Dedicated GPUs — run Sparse-MoE Decoupled OpenDiLoCo inner rounds)**:
   ```bash
   # 8 GB GPU (trains 2 resident experts in < 1.80 GiB VRAM, >= 1,536 MiB free):
   HIVEMIND_API_KEY=hm_xxx python scripts/gcp_t4_crowd_diloco_worker.py \
     --mode moe-opendiloco-round --moe-split-mode lora_domain_experts \
     --resident-experts "1,2" --rounds 4 --inner-steps 8 \
     --submit-opentrainer --campaign-id CROWDGPT-WE-NOV14-15-2026-MOE-OPENDILOCO
   ```
3. **Prosumers (Dual-Mode GPUs — serve inference by day, auto-train idle windows with VRAM Yield Guard)**:
   ```bash
   python scripts/hivemind_connect.py crowdgpt-weekend-raid --handle <your_handle> --role prosumer
   # Or query personalized expert assignment & CLI directly:
   curl -s "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/opentrainer/crowdgpt-weekend-raid?handle=<your_handle>&role=prosumer&vram_mib=8192"
   ```
4. **GCP Spot Tesla T4 Co-Trainer (`socreateai-agora-hermes-node1`, `us-east4-b` Spot)**:
   ```bash
   CONFIRM=yes bash scripts/t4_training_window.sh weekend-raid       # 14:00-22:00 Paris, Sat 14 & Sun 15 Nov only (ends 16 Nov 00:00); +7h Spot ~ +$1.30 USD
   CONFIRM=yes bash scripts/t4_training_window.sh weekend-raid-end   # Monday: delete the weekend schedule and restore the daily training window
   ```

---

## 3. Ready-to-Open GitHub Issues (4 Track Issues)

> Run `python scripts/publish_crowdgpt_weekend_raid_kit.py --print-urls` for 1-click pre-filled GitHub Issue URLs, or `python scripts/publish_crowdgpt_weekend_raid_kit.py --publish-github` once `gh auth login` is active.

### Issue 1 — `[Weekend Raid · Track 1] CrowdGPT 8-Expert Sparse MoE Split & FoMoE sigma-Router Calibration (14–15 Nov 2026)`
- **Labels**: `crowdgpt-weekend-raid`, `moe-opendiloco`, `domain-expert`
- **Checklist**:
  - [x] Implement step-0 function-preserving `CrowdFineGrainedMoEMLP` (`8 × 320 = 2560`) and `CrowdLoRAMoEMLP` (`8` SwiGLU LoRA experts) in `src/hivemind/swarm/crowdgpt_moe_opendiloco.py`.
  - [x] Wire FoMoE non-competitive `sigmoid_independent_topk_routing` + Ghost-Expert Skip-Token mask for `8 GB` GPUs.
  - [ ] Execute Wave 1 (`Sat 14 Nov 09:00–13:00 UTC`) router warm-start across `E0..E7` and verify expert load-balance CV `< 0.35`.

### Issue 2 — `[Weekend Raid · Track 2] Decoupled OpenDiLoCo + RDA Quorum Syncer on GCP Spot T4 & Home Swarm`
- **Labels**: `crowdgpt-weekend-raid`, `moe-opendiloco`, `gcp-cost-governance`
- **Checklist**:
  - [x] Integrate `CPUOffloadedFlatBucketOuterState` (`0 MiB` GPU overhead, `P=4` router + `E=8` expert fragments) with `DecoupledDiLoCoQuorumSyncer` (`RDA`) and `PA-CGAD`.
  - [x] Configure `scripts/t4_training_window.sh weekend-raid` (`14:00–22:00 Europe/Paris` Sat 14 & Sun 15 Nov, `+\$1.30` Spot, `>= 4,300 MiB` free VRAM guard, zero writes to `/opt/hivemind/t4/`).
  - [ ] Complete Wave 2 & Wave 4 Peak Quorum outer checkpoints with monotone held-out loss decrease.

### Issue 3 — `[Weekend Raid · Track 3] Producer, Trainer & Prosumer Onboarding + S21 License-Clean Shards`
- **Labels**: `crowdgpt-weekend-raid`, `community`, `datasets`
- **Checklist**:
  - [x] Enforce `configs/datasets_s21.json` `ACCEPTED`-only allowlist (`18` datasets, blocking `cc-by-nc-*`) + deterministic `assign_hiveminder_raid_shard()`.
  - [x] Expose `GET /v1/hivemind/opentrainer/crowdgpt-weekend-raid` and `.github/ISSUE_TEMPLATE/04_crowdgpt_weekend_moe_raid.yml`.
  - [ ] Onboard `>= 25` active weekend Producer/Trainer/Prosumer GPUs across `8 GB`, `12 GB`, `16 GB`, and `24 GB` tiers.

### Issue 4 — `[Weekend Raid · Track 4] Sunday Wave-5 Blind A/B Arena & Vxtzq/Crowd-MoE-v1 Checkpoint Export`
- **Labels**: `crowdgpt-weekend-raid`, `evaluation`, `release`
- **Checklist**:
  - [ ] Serve `hivemind/crowd-v1` (baseline) vs `hivemind/crowd-moe-v1` (weekend raid checkpoint) side-by-side on Sunday `15 Nov 18:00–21:00 UTC`.
  - [ ] Run decontamination & bilingual FR/EN + Python/Math eval suite and publish signed `safetensors` SHA-256 receipt for `@Vxtzq`.

---

## 4. GitHub Discussion Post (`Announcements` / `Swarm Campaigns`)

**Title**: `🐝 [Weekend Raid · 14–15 Nov 2026] Massive CrowdGPT-MoE x HiveMind Decoupled OpenDiLoCo Training — Calling All Producers, Trainers & Prosumers!`

```markdown
## 🐝 Massive Weekend Swarm Raid — Saturday 14 & Sunday 15 November 2026

Together with **@Vxtzq** (creator of **`Vxtzq/Crowd-v1`**, `1.01B` bilingual FR/EN + `CrowdEngram` LM), **Project HiveMind** is running a 48-hour decentralized **Sparse-MoE + Decoupled OpenDiLoCo** training raid this weekend (**14–15 November 2026**) to train **`hivemind/crowd-moe-v1`** across our community GPUs and GCP Spot Tesla T4 co-trainer!

### 🧠 What We Are Training
- **Step-0 Exact 8-Expert Sparse MoE Upcycle**: Splits `Crowd-v1`'s SwiGLU FFN (`2560 = 8 × 320` fine-grained experts + `8` domain LoRA experts) with **FoMoE σ-MoE** non-competitive sigmoid routing and **Ghost-Expert Skip-Token** support so even **8 GB GPUs (`RTX 2070 / 3060 / 4060`)** train 2 local experts in `< 1.8 GiB` VRAM!
- **PrimeIntellect OpenDiLoCo + DeepMind Decoupled DiLoCo (`RDA`)**: CPU-offloaded FP32 Nesterov/PA-CGAD outer state (`0 MiB` GPU overhead), `P=4` router + `E=8` expert streaming buckets, Quorum-of-$K$ straggler-free sync, and a strict **monotone held-out validation-loss ratchet**.
- **18 License-Clean S21 Datasets (AGPL-3.0 Allowlisted)**: Python (`OpenCodeInstruct`, `Magicoder`), Math (`OpenMathInstruct-2`, `NuminaMath-CoT`), Logic/Physics (`AI2 ARC`, `PhysicsQA`), Exoplanets & Stellar Spectra (`NASA Kepler`, `SDSS DR9`), Defensive Cyber, and `FineWeb-Edu` bilingual anchor.

### 🗓️ 6 Global Quorum Waves (Import `.ics`: `docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics`)
- **Wave 0 (Preflight)**: Fri 13 Nov `18:00–20:00 UTC` (`19:00–21:00 CET` / `13:00–15:00 EST`)
- **Wave 1 (EU+APAC Kickoff)**: Sat 14 Nov `09:00–13:00 UTC` (`10:00–14:00 CET` / `18:00–22:00 JST`)
- **Wave 2 (Peak Global Sprint #1 + Spot T4)**: Sat 14 Nov `13:00–21:00 UTC` (`14:00–22:00 CET` / `08:00–16:00 EST`)
- **Wave 3 (Nocturnal Prosumer & APAC PA-CGAD Relay)**: Sat 14 Nov `22:00 UTC` → Sun 15 Nov `08:00 UTC`
- **Wave 4 (Peak Global Sprint #2 + RDA Consolidation)**: Sun 15 Nov `13:00–18:00 UTC` (`14:00–19:00 CET`)
- **Wave 5 (Blind A/B Arena & Release Freeze)**: Sun 15 Nov `18:00–21:00 UTC` (`19:00–22:00 CET`)

### ⚡ Join in 60 Seconds (Producers, Trainers & Prosumers)
1. Grab or check your HiveMinder key at https://hivemind-openrouter-1003063861791.us-central1.run.app/signup (`+5,000 HMT` welcome compute credits + peer-to-peer HMT transfers live!).
2. Fetch your personalized GPU shard & command:
   ```bash
   curl -s "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/opentrainer/crowdgpt-weekend-raid?handle=<your_handle>&role=prosumer&vram_mib=8192"
   ```
3. Reply below with your **Handle + GPU + Role (`Producer` / `Trainer` / `Prosumer`) + Preferred Expert (`E0`–`E7`)**, or open a Raid Registration Issue!

*(Non-Monetary Charter: `HMT` Reflexion Points carry zero financial value and represent reciprocal shared swarm inference & training quota only.)*
```

---

## 5. Discord Raid Announcements (`#announcements` & `#crowdgpt-raid`)

### 5.1 Compact Pin / `@everyone` Blast (`< 950` chars)
```text
🐝 **MASSIVE WEEKEND SWARM TRAINING RAID — SAT 14 & SUN 15 NOV 2026 (`CrowdGPT-MoE` x `HiveMind`)** 🚀
All **Producers, Trainers & Prosumers** (`8 GB–24 GB` GPUs) are invited to co-train **`hivemind/crowd-moe-v1`** with **@Vxtzq**!
• **What**: Step-0 exact `8`-expert Sparse MoE split of `Vxtzq/Crowd-v1` (`1.01B`) + **OpenDiLoCo** + **Decoupled DiLoCo (`RDA`)** on 18 license-clean S21 datasets.
• **8 GB Friendly**: FoMoE Ghost-Expert routing trains 2 local experts in `< 1.8 GiB` VRAM (`>= 1.5 GiB` free); GCP Spot T4 anchors `14:00–22:00 CET`!
• **Waves**: Fri 13 Nov `19:00 CET` Preflight · Sat 14 Nov `10:00` & `14:00–22:00 CET` Peak · Overnight PA-CGAD · Sun 15 Nov `14:00–22:00 CET` Consolidation + Blind A/B Arena!
• **Join**: https://hivemind-openrouter-1003063861791.us-central1.run.app/signup | `.ics` calendar & commands in GitHub PR #871 | Raid Hub: https://discord.gg/ZeFtZpKHf
*(Non-monetary reciprocal HMT compute credits only)*
```

### 5.2 Detailed `#crowdgpt-raid` Role Briefing (`< 1,900` chars)
```text
🛠️ **HOW TO JOIN THIS WEEKEND'S `CrowdGPT-MoE` RAID (14–15 NOV 2026) BY ROLE**

1️⃣ **PRODUCERS (Inference Nodes — `RTX 2070+` / `8 GB+`)**
Keep `hivemind/crowd-v1` fast during training waves and host Sunday's Wave-5 Blind A/B Arena (`crowd-v1` vs `crowd-moe-v1`):
`python scripts/hivemind_crowd_v1_producer.py --host 127.0.0.1 --port 8791 --threads 6`

2️⃣ **TRAINERS (Dedicated Inner-Loop GPUs — `8 GB` to `24 GB`)**
Run Sparse-MoE OpenDiLoCo inner steps over HTTPS `TCP/443` (zero port-forwarding):
`HIVEMIND_API_KEY=hm_xxx python scripts/gcp_t4_crowd_diloco_worker.py --mode moe-opendiloco-round --moe-split-mode lora_domain_experts --resident-experts "1,2" --rounds 4 --inner-steps 8 --submit-opentrainer --campaign-id CROWDGPT-WE-NOV14-15-2026-MOE-OPENDILOCO`

3️⃣ **PROSUMERS (Serve by Day + Train When Idle)**
Check your deterministic non-IID expert shard (`E0` FineWeb FR/EN, `E1` Python, `E2` Math, `E3` Logic, `E4` Physics, `E5` Exoplanets, `E6` Stars, `E7` Defensive Cyber):
`curl -s "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/opentrainer/crowdgpt-weekend-raid?handle=YOUR_HANDLE&role=prosumer&vram_mib=8192"`

📅 **Import `.ics` Calendar**: `docs/calendar/hivemind_crowdgpt_weekend_nov14_15_2026.ics`
💬 Drop your **GPU + Role + Timezone** in `#crowdgpt-raid` (https://discord.gg/ZeFtZpKHf) so we balance all 8 experts before Friday `19:00 CET` preflight!
```

---

## 6. Reddit Collaboration Announcement (`HiveMind x @Vxtzq CrowdGPT`)

> Target subreddits: **`r/LocalLLaMA`** (flair: `Resources / Open-Source`), **`r/MachineLearning`** (`[P]`), **`r/BOINC`**, and **`r/SelfHostedAI`**.

**Suggested Title**:  
`[Collab / Open-Source] Weekend Swarm Training Raid (Nov 14–15): Upcycling CrowdGPT-1B (w/ Engram memory) into an 8-Expert Sparse MoE across 8 GB Home GPUs using Decoupled OpenDiLoCo`

**Post Body (Markdown)**:
```markdown
Hey everyone! 👋

We’re teaming up across two open-source community projects — **CrowdGPT** (by **@Vxtzq**, creator of [`Vxtzq/Crowd-v1`](https://huggingface.co/Vxtzq/Crowd-v1), a from-scratch `1.01B` bilingual EN/FR causal LM with a `350M` bigram **Engram** memory table trained collaboratively by volunteer GPUs) and **Project HiveMind** (AGPL-3.0 volunteer GPU inference + DiLoCo swarm fabric) — for a **48-hour Weekend Swarm Training Raid this Saturday Nov 14 & Sunday Nov 15, 2026**!

### Why we built this collab
Training a `1B+` model across volunteers over residential internet usually hits three walls:
1. **VRAM limits on 8 GB cards** (`RTX 2070 / 3060 / 4060`) once Adam states and activations load.
2. **All-Reduce bandwidth & stragglers** when a home GPU shares a round with a cloud GPU.
3. **Catastrophic forgetting** when fine-tuning on code/math hurts general bilingual chat.

### How the Weekend Raid solves all three (`src/hivemind/swarm/crowdgpt_moe_opendiloco.py`)
1. **Step-0 Function-Preserving 8-Expert Sparse MoE Split + FoMoE `σ`-MoE Routing**:
   - We split `Crowd-v1`'s `2560`-wide SwiGLU FFN into **8 specialized experts** (`E0` FineWeb-Edu EN/FR anchor, `E1` unit-test verified Python, `E2` GSM8K/MATH CoT, `E3` ARC formal logic, `E4` Physics, `E5` NASA Exoplanet transits, `E6` SDSS Stellar spectra, `E7` Defensive Cyber — all 18 datasets strictly license-audited `ODC-By / CC-BY / MIT / Apache-2.0` + 13-gram decontaminated).
   - Using **FoMoE non-competitive Sigmoid Top-2 routing** + **Ghost-Expert Skip-Token** masking, an **8 GB home GPU** only loads **2 resident LoRA experts** on top of the `fp16` trunk + CPU-offloaded Engram table — training comfortably in **`< 1.8 GiB` VRAM** while keeping `>= 1.5 GiB` free!
2. **PrimeIntellect OpenDiLoCo + DeepMind Decoupled DiLoCo (`RDA`)**:
   - FP32 master weights & outer Nesterov / staleness-gated `PA-CGAD` optimizer states live in **CPU RAM (`0 MiB` GPU overhead)**, partitioned into `4` router fragments + `8` expert buckets.
   - Instead of synchronous barriers, **Quorum-of-$K$ + Radial-Directional Averaging (`RDA`)** merges pseudo-gradients over standard HTTPS (`TCP/443`, zero port-forwarding) and gates every outer checkpoint with a **monotone validation-loss ratchet**.
   - Our cloud footprint stays tiny: a single **GCP Spot Tesla T4** joins for `14:00–22:00 CET` on Saturday & Sunday (`+\$1.30` total Spot cost!).

### Join as a Producer, Trainer, or Prosumer this Weekend (Nov 14–15, 2026)
- **Wave 0 (Dry-Run)**: Fri Nov 13 `18:00–20:00 UTC` (`19:00 CET` / `13:00 EST`)
- **Wave 1 & 2 (Saturday Kickoff + Peak Global Sprint)**: Sat Nov 14 `09:00–21:00 UTC` (`10:00–22:00 CET`)
- **Wave 3 (Overnight Prosumer & APAC Relay)**: Sat `22:00 UTC` → Sun `08:00 UTC`
- **Wave 4 & 5 (Sunday Consolidation + Blind A/B Arena `crowd-v1` vs `crowd-moe-v1`)**: Sun Nov 15 `13:00–21:00 UTC`

**Links to jump in (100% open-source AGPL-3.0, zero crypto / zero financial tokens — reciprocal compute quota only)**:
- **GitHub Repo & Raid Kit (`.ics` calendar + code)**: https://github.com/xaviercallens/HiveMind
- **Base Model (`Vxtzq/Crowd-v1`)**: https://huggingface.co/Vxtzq/Crowd-v1
- **Signup & Live OpenAI-compatible Endpoint**: https://hivemind-openrouter-1003063861791.us-central1.run.app/signup
- **Discord Raid Coordination**: https://discord.gg/ZeFtZpKHf

Whether you have an `8 GB` RTX 2070 running overnight as a **Prosumer**, a `24 GB` 3090/4090 **Trainer**, or just want to test `hivemind/crowd-v1` vs `hivemind/crowd-moe-v1` in Sunday's **Blind A/B Arena**, we’d love to have you in the quorum! Happy to answer any questions about the MoE split, Engram CPU offload, or Decoupled OpenDiLoCo math below.
```
