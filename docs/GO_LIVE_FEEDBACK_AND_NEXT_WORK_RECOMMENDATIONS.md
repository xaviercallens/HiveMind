# Go-Live 48-Hour Feedbacks & Recommendations for Next Work (`2026-10-24` – `2026-10-26`)

> **Scope:** Synthesizes all empirical observations, user/community feedback (`r/artificielle`, `r/LocalLLaMA`, GitHub PRs/Issues), live Cloud Run & Tesla T4 telemetry (`socreateai-agora-hermes-node1`), and experimental findings (`NIGHT-MOE-1`, BOINC 6-Domain QLoRA, `E13`–`E16`) into a prioritized execution sequence for **Season 14 (`Night-Hive`)**, **Season 15 (`QueenBee`)**, and **Season 16 (`Operation First-Flight`)**.

---

## 1. Recorded Feedbacks from the 48-Hour Go-Live (`2026-10-24` – `2026-10-26`)

### A. Live End-User & Console Feedback (`docs/LIVE_USER_TESTS.md`)
1. **First Real Console Request (`hivemind/auto` — Served & Settled, Answer Quality Gap):**
   - **What worked:** First byte in `273 ms`, `294` tokens streamed at `36 tok/s`, settled cleanly in the shared HMT ledger.
   - **What failed:** The returned Python code on an open-ended coding prompt was functionally wrong (`HM-S16-04`, Issue `#191`). Liveness probes (`def add(a, b): return a + b`) pass `15/15`, but open-prompt coding quality is not yet gated or routed by executable tests.
2. **New-User First-Request `HTTP 402` Pricing Spike (Fixed in PR `#182`, Structural Fix in `HM-S16-03` / `#190`):**
   - **Observed:** Idle accounts that had previously raised their max bid (`max_price_per_1k = 10.0 HMT`) kept the Kelly clearing price at `7.5 HMT/1k`, rejecting brand-new accounts (`default max = 2.0 HMT`) with `HTTP 402` even when the T4 GPU was 100% idle.
   - **Immediate fix applied:** Restricted Kelly price clearing to consumers active within the last `900s` (`HIVEMIND_BID_WINDOW_S`, PR `#182`).
   - **Remaining edge case (`#190`):** Even within a `900s` window, 4 active high-bidding test accounts can still price out a new user when total demand is well below T4 capacity (`S = 100,000k`).
3. **Cold-Model `HTTP 504` on Unpinned Tier (`HM-S16-06` / `#193`):**
   - **Observed:** On the first 18-call E2E matrix (`3 services × 3 tiers × JSON/SSE`), `17/18` passed; the single failure was a `504` after `120s` waiting for a cold 7B coder reload (not charged).
   - **Immediate fix applied:** Pinned `Qwen2.5-Coder-7B-Instruct` in VRAM (`keep_alive = -1`) in `deploy/t4_bee_expert/start.sh` (PR `#183`).
4. **Source Redeploy Regression Incident (`HM-S16-01` / `#188` & `HM-S16-02` / `#189`):**
   - **Observed:** Deploying from an uncommitted working tree overwrote live router fixes twice (`1-minute` wrong-revision window; zero lost requests or ledger writes, fixed by rebasing revision `00014-w7f`).
5. **Synthetic Probe & Smoke-Test Account Distortion (`HM-S16-07` / `#194`):**
   - **Observed:** `probe.live-monitor` and E2E `scientist-*` accounts appear in raw registration counts (`81` registrations) and Kelly market bids unless flagged as `test`/`probe`.

### B. T4 Hardware Governor & Dual-GPU BOINC Training Feedback (`PR #181`, `#199`, `#201`, `#203`)
1. **BOINC-to-Inference Hand-Back Latency:**
   - **Observed:** Initial real BOINC GPU yields under `nginx` burst traffic took up to `7,129 ms`–`23,406 ms` when waiting for BOINC client polling alone, and the monitor mixed validator GPU replays (`trigger="gpu"`, no BOINC task running) with real user yields.
   - **Fix applied & verified:** `t4_governor.py` now terminates/suspends the active GPU task directly after a short grace period (`ended_by_governor = true`, PR `#203`), bringing real inference preemption down to **`1,133–1,135 ms`**, while `scripts/live_user_monitor.py` separates real `nginx` yields from validator replay noise (PR `#201`).
2. **6-Domain BOINC MoE QLoRA Ratchet (`hmq_00001`–`hmq_00091`):**
   - **Observed:** Single-domain QLoRA sweeps alone improved their target domain (`math` `-1.013`, `biomed` `-0.526`) but caused cross-domain drift. Switching to **80% focus + 20% 6-domain replay (`general`, `french`, `edu_web`, `news`, `biomed`, `math`)** eliminated catastrophic forgetting and lowered 6-domain mean eval loss from **`2.2705` → `2.1380`** across **`93` assimilated workunits**.
3. **Inactive Systemd Units on T4:**
   - **Observed:** Four legacy units (`hivemind-live-agent-bee`, `hivemind-live-bee-llama`, `hivemind-live-shield-bee`, `hivemind-live-shield-coder`) remain enabled-but-inactive on the T4 VM and should be disabled or consolidated so `units_inactive` is `[]`.

### C. `NIGHT-MOE-1` Specialist Training & Scientific Control Feedback (`docs/NIGHT_MOE_RESULTS.md`)
1. **`hivemind/bee-stars` Validated (`+40.4%` Gain):**
   - Accuracy rose from `48.9%` to **`89.3%`** ($p < 10^{-4}$) while its shuffled-label control fell to `30.6%`.
2. **Control Design & Generative Fine-Tuning Lessons (`HM-S14-01`..`04`, `#165`–`#168`):**
   - **Shuffled-label control preserves class prior (`exoplanet` `VOID`):** Even though the `exoplanet` adapter beat its shuffled control `309` to `17` (`87.3%` vs `38.7%`), the shuffled control gained slightly over the base model (`38.7%` vs `36.2%`, $p=0.0237$) by learning the majority label prior, triggering the `VOID` rule (`#165`).
   - **Gold-text SFT hurt `maths` (`78.5%` → `71.5%`, `#166`):** Training on static gold text degraded chain-of-thought formatting; `maths` and `coder` experts must train on **rejection-sampled, verified self-solutions (`E15` GWAYA RFT recipe)**.
3. **Community & External Review Feedback (`r/artificielle`, `r/LocalLLaMA`, `@clark719664`):**
   - **`u/artsnumeriques` (`r/artificielle`):** Recommended splitting dense multi-topic announcements into focused, single-result technical posts with direct reproducible colab/CLI snippets.
   - **`@clark719664` (PR `#186`):** Identified that duplicate or future-step async DiLoCo submissions should preserve work rather than dropping valid gradients.

---

## 2. Prioritized Recommendations for Next Work (Execution Order)

### Priority 1 — Gate `G-L1`: Safe Live Operations & Clean Single-Source Deployment (`Season 16`, Immediate)
1. **`HM-S16-02` (`#189`) + `HM-S16-10` (`#197`) — Canary Contract Gate & Single Litestream Writer Lock:**
   - Gate every Cloud Run traffic switch on `scripts/live_user_monitor.py` passing `15/15` checked probes against the canary revision (running with an ephemeral ledger), and enforce a GCS generation-precondition startup lock so two revisions never write to the SQLite Litestream replica simultaneously.
2. **`HM-S16-01` (`#188`) + `HM-S16-09` (`#196`) — Deploy Strictly from Tagged `main` & Retire Overlays:**
   - Merge the remaining `deploy/router_overlay` and `deploy/dashboard_overlay` fixes into `deploy/gcp_cloud_run/` and `deploy/gcp_openrouter/`, remove template fallback completions and browser-side point credits from `La Ruche`/`Tribute`, restore `/night-moe`, and deploy all 3 services from a single tagged `main` commit.
3. **`HM-S16-07` (`#194`) + `HM-S16-08` (`#195`) — Isolate `test`/`probe` Accounts & Publish `/status` Alerting:**
   - Add an `is_probe_or_test: bool` flag in `src/hivemind/router/accounts.py` so `probe.live-monitor` and `scientist-*` smoke accounts never move Kelly market prices or inflate organic registration counters, and expose a public `/status` badge fed by `live_user_monitor.py`.

### Priority 2 — Gate `G-L2`: User Experience, Dual-Regime Market & Verified Routing (`Season 16`)
4. **`HM-S16-03` (`#190`) — Two-Regime Market Pricing (Posted Price Below Capacity, Kelly Under Congestion):**
   - When total active token demand is below producer capacity (`D < S`), clear at the producer's posted reserve price (`0.21–0.25 HMT/1k`) so new users are never priced out by high bidders during idle periods; switch to Kelly proportional allocation only when `D >= S` (backed by a Lean 4 theorem).
5. **`HM-S16-06` (`#193`) + `HM-S16-05` (`#192`) — Loaded-Model Heartbeats, T4 Unit Cleanup & Enforced Remote Control:**
   - Report resident Ollama/llama-server models in producer heartbeats so the router only dispatches to warm models, clean up the 4 inactive systemd units on the T4, and wire producer agents to actively enforce `pause` and `vram_cap_gb` commands from the dashboard.
6. **`HM-S16-04` (`#191`) — Open-Prompt Coding Evaluation & Verified-Score Routing:**
   - Run held-out coding prompts with hidden unit tests inside an isolated sandbox (on the TPU VM or gVisor host), measure `pass@1` per served tier, and route `hivemind/auto` coding queries using `verified_mode` (`PR #153`) + GWAYA-Lite escalation (`1.5B -> 3B -> 7B`).

### Priority 3 — `Season 14` (`NIGHT-MOE-2`) & `Season 15` (`QueenBee`) Scientific Upgrades
7. **Review & Merge Open Community/Core PRs (`#186`, `#181`, `#163`):**
   - Audit and merge `@clark719664`'s async submission preservation fix (`PR #186`), finalize the BOINC 7.24 core-node PR (`#181`), and land the `ALIXQueenBee` planner & dance-record spec (`#163`).
8. **`HM-S14-01`..`04` (`#165`–`#168`) — `NIGHT-MOE-2` Preregistered Run:**
   - Replace the prior-sensitive shuffled control with a paired `adapter-vs-control` McNemar test (`#165`), train `maths` and `coder` experts on rejection-sampled verified self-solutions (`#166`), and evaluate the routed Bee MoE end-to-end across the full interference grid (`#167`, `#168`).
9. **`HM-S15-01`..`03` (`#171`–`#173`) — `QueenBee` Bandit Routing on Live Served Experts:**
   - Bound waggle-dance influence by per-signer pseudo-trial evidence volume (`#171`), require two independent rotating scouts per cell (`#172`), and deploy the `QueenBee` contextual bandit to route live traffic across `hivemind/bee-stars`, `hivemind/coder`, and the 6-domain BOINC MoE adapter (`#173`).
