# Claude Code Session Handoff & Owner Decisions: Season 20 Solver Bee (`HM-S20-SOLVER`) + RTX 2070 & GCP T4 Agent Fleet

> **Date:** `2026-10-27` · **Owner:** Xavier Callens (`@xaviercallens`)
> **Target Sessions:** Claude Code Scientific Lead (`session_01JWcvy9P3NMPXiV6BycKYyY`, [`PR #260`](https://github.com/xaviercallens/HiveMind/pull/260)) & Antigravity Engineering Coordinator ([`PR #259`](https://github.com/xaviercallens/HiveMind/pull/259))
> **Master Roadmap Section:** [`ROADMAP.md` §3.11 (`HM-S20-SOLVER`)](../ROADMAP.md) · **GitHub Project:** [Project #7 (`Season 19–20 Launch & Engineering`)](https://github.com/users/xaviercallens/projects/7)

---

## 1. Owner Decisions on `docs/BEE_MOE_NUMERIC_SOLVERS_PROPOSAL.md` (§7 of `PR #260`)

All six questions raised by the Claude Code session in [`docs/BEE_MOE_NUMERIC_SOLVERS_PROPOSAL.md`](BEE_MOE_NUMERIC_SOLVERS_PROPOSAL.md) §7 are **decided and approved** by the project owner (`@xaviercallens`):

| # | Question in `BEE_MOE_NUMERIC_SOLVERS_PROPOSAL.md` §7 | Owner Decision (`2026-10-27`) |
| :---: | :--- | :--- |
| **1** | Approve Wave 1 (`W1-1`..`W1-9`) as scoped, or narrow it? | **APPROVED IN FULL (`Wave 1` + `Wave 2` + `Wave 3` roadmap recorded as Season 20 `HM-S20-SOLVER`).** Start immediately with `W1-1`..`W1-9` in parallel with the already-merged Python reference & gateway rescue endpoint [`src/hivemind/swarm/rusty_sundials_scientific_bee.py`](../src/hivemind/swarm/rusty_sundials_scientific_bee.py) (`PR #259`). |
| **2** | `rusty-SUNDIALS`: confirm it is yours to license; check `NOTICE` for retained LLNL terms before vendoring? | **CONFIRMED:** [`xaviercallens/rusty-SUNDIALS`](https://github.com/xaviercallens/rusty-SUNDIALS) (`/home/xavkal/xdev/rusty-SUNDIALS`) is authored and owned by Xavier Callens. Keep it as a **pinned external dependency** in Wave 1 with its `LICENSE` (BSD-3-Clause) and LLNL `NOTICE` preserved verbatim. |
| **3** | Missing licence files (`SocrateAI-Scientific-Genetic`, `LeanMaster`) for later waves? | **RECORDED & TRACKED (`#271`):** Keep Wave 1 100% self-contained (zero external datasets). Add explicit `LICENSE` files to `SocrateAI-Scientific-Genetic` and `LeanMaster` before Wave 2/3 dataset ingestion. |
| **4** | Is a CPU tier in the router catalog acceptable in principle? | **YES — APPROVED:** `hivemind/bee-rusty-sundials` is already live at `min_vram_gb=6.0` (`PR #259`) for hybrid neural+numeric rescue; adding an explicit CPU/hybrid `solver` tier in Wave 2 (`#271`) is approved. |
| **5** | Price Solver Bee answers in `HMT`, or free during the pilot? | **FREE / ZERO-SURCHARGE DURING WAVE 1 PILOT** on `/v1/hivemind/bees/rusty-sundials/rescue`, transitioning to standard low-cost market settlement (`base_price_per_1k_hmt = 0.06`) on `/v1/chat/completions` (`model="hivemind/bee-rusty-sundials"`). |
| **6** | Stretch goal `W1b` (small coder model mapping natural-language prompts to JSON solver specs) needs a GPU slot on the T4 or RTX 2070: your call? | **APPROVED (`#269` & `#272`):** Allocated a dedicated lightweight slot on both **Xavier's local RTX 2070 (`8 GB`)** and the **GCP Tesla T4 (`16 GB` / multi-T4)** using the schedule below so live OpenRouter serving is never interrupted. |

---

## 2. How `PR #259` (`hivemind/bee-rusty-sundials`) and `PR #260` (`Solver Bee W1–W3`) Fit Together

1. **Already Merged on `main` (`PR #259`, `d8428c5`):**
   - `src/hivemind/swarm/rusty_sundials_scientific_bee.py`: Python-level structural diagnosis (`analyze_physical_mathematical_problem`), 4-domain reference rescue (`astronomy`, `fluid_mechanics`, `quantum_fluid`, `genetics`), 12-record SFT+DPO dataset (`data/rusty_sundials_scientific_bee/rusty_sundials_numeric_rescue_dataset.jsonl`), and `BeeRetentionStore`.
   - Router catalog tier `bee-rusty-sundials` (`hivemind/bee-rusty-sundials`) + gateway endpoints `GET /v1/hivemind/bees/rusty-sundials` and `POST /v1/hivemind/bees/rusty-sundials/rescue`.
2. **Next Engineering & Scientific Deliverables (`PR #260` $\to$ Issues `#261`–`#272`):**
   - **`W1-1` (`#261`):** Compile the pinned Rust binary `hivemind-solver` from `/home/xavkal/xdev/rusty-SUNDIALS` (`--locked`, `libm`, no fast-math, JSON spec in $\to$ JSON trajectory + `input_sha256` + `binary_sha256` out).
   - **`W1-2` (`#262`) & `W1-3` (`#263`):** 12-IVP reference library (Robertson, Van der Pol, Brusselator, Lotka-Volterra, damped oscillator, stiff chain, linear $x'=Ax+b$, mass-action stoichiometry) + stdlib-only 4-check verifier (`VERIFIED` / `AUDITED` / `FAILED` / `UNVERIFIED`).
   - **`W1-4` (`#264`) & `W1-5` (`#265`):** Extend `scripts/e15_gate.py` to run `hivemind-solver` under strict resource limits and expose `solver_bee` loopback `/v1/chat/completions` behind `hivemind_producer_shield.py`.
   - **`W1-6` (`#266`) & `W1-7` (`#267`):** Pre-registered experiments **S-1** (same-host bitwise determinism + cross-CPU reproducibility between RTX 2070 host CPU and GCP T4 host CPU) and **S-2** (5-class fault injection against the verifier, 100% detection, 0% false rejection).
   - **`W1-8` (`#268`) & `W1-9`/`W1b` (`#269`):** Wilson lower-bound bee scoring (`queenbee/dance.py`) + newcomer quickstart + `W1b` NL-to-JSON-Spec translator on 40 held-out prompts.
   - **`Wave 2` (`#270`) & `Wave 3` (`#271`):** BOINC CPU sweep app + solver validator branch, Solver-Taught Neural Surrogate Experts (`hivemind/bee-rusty-sundials`), AMCP v1.5 `solver` mesh, method-of-lines PDEs, and Lean 4 soundness proof.

---

## 3. Execution Plan for Xavier's Own GPU/CPU Agents (`RTX 2070` + `GCP Tesla T4` Fleet, `#272`)

### 3.1 Local Agent #1: NVIDIA GeForce RTX 2070 (`8 GB VRAM`, `std_8gb` Tier, `< 6,656 MiB` WDDM Cap)
- **Live OpenRouter Serving (24/7 Foreground Priority):**
  - Runs `scripts/hivemind_producer_shield.py --control-state-file ~/.hivemind/control.json` and `scripts/hivemind_producer_agent.py` advertising `hivemind/coder` (`Qwen2.5-Coder-7B-Instruct` Q4_K_M, `~4,850 MiB`) and `hivemind/bee-rusty-sundials`.
  - Runs the `W1-1`/`W1-5` `solver_bee` single-core CPU worker in the E15 sandbox when CPU load is low (`1 vCPU`, `0 MiB` GPU VRAM impact).
- **Daytime Micro-Idle (`07:00–22:00 CET`, `idle >= 30s`) & Nocturnal Deep Evolution (`22:00–07:00 CET`):**
  - Preempts in `< 1.0s` (`p99 < 450 ms`) whenever a game, local UI, or live OpenRouter inference request arrives, flushing gradients to host RAM via `EF21-DiLoCo`.
  - Retrains `hivemind/bee-rusty-sundials` and `W1b` NL-to-Spec LoRA adapters ($r=16$, `seq_len=512`) on solver-certified trajectories and retains checkpoints via `BeeRetentionStore` (`\Delta E < 0`).
- **Experiment `S-1(b)` Consumer CPU Anchor:**
  - Executes the 10-run `hivemind-solver` reproducibility suite on the local consumer CPU and exports the signed JSON receipt for comparison against the GCP T4 Xeon host.

### 3.2 Cloud Agent #2 (and Multi-T4 Scale-Out): GCP Tesla T4 (`16 GB VRAM` / `15,360 MiB`, `xl_16gb_plus` Tier)
- **Live OpenRouter Serving & BOINC Resident Validation:**
  - Serves `hivemind/bee-rusty-sundials`, `hivemind/bee-stars`, `hivemind/coder`, and `hivemind/coder-plus` (`14B` Q4_K_M on secondary T4 or scheduled window).
  - Runs the optional BOINC `resident_validator.py` (`PR #256`) over a local UNIX socket (`HIVEMIND_RESIDENT_VALIDATOR_SOCKET`) when GPU headroom permits.
- **Wave 1 `W1b` Stretch Slot & Wave 2 Teacher-to-Student Distillation:**
  - Runs the `W1b` 40-prompt pre-registered NL-to-JSON-Spec evaluation during low-traffic windows (`< 2 min` total GPU time).
  - Trains rank-32 Teacher LoRA adapters on `hivemind-solver` verified trajectories (`Wave 2` Surrogate Expert) and projects them down to rank-16 (`project_lora_rank_svd`, `>94%` spectral energy) for 1-click hot-swap onto the local RTX 2070 and community `8 GB` nodes.
- **Multi-T4 Elastic Scale-Out (for Sunday Nov 1 Launch & Show HN):**
  - Additional GCP T4 instances attach via the same `hivemind_producer_agent.py` + AMCP v1.5 `replicate()` warmup (`PR #249` + `PR #250`), scaling swarm capacity from `2,200 tok/s` (Sun Nov 1) to `13,000+ tok/s` (Nov 6–8).
