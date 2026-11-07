# Changelog

## v3.1.2 — 2026-11-07 — Legacy PBKDF2-SHA256 Login Restoration, HTTPS Cloud Run Portal, OpenTrainer Campaign Gate, E8 Lattice Parity & Streaming TTFT Telemetry (PRs #692–#703)

- **Legacy Email + Password Login Restoration & HTTPS Cloud Run Fixes (PR #702, deployed as `v3.1.2-https` across `hivemind-at-home`, `hivemind-at-home-ruche`, and `hivemind-openrouter`):**
  - Preserved `LEGACY_KNOWN_USERS` (200,000-iteration PBKDF2-SHA256) and added automatic fallback verification against `credentials` table rows (`pbkdf2_sha256$120000$...` / `pbkdf2_sha256$200000$...`) keyed by `SHA256("hivemind-email-v1:" + email)` in `/api/v1/auth/login`, `/v1/hivemind/accounts/login`, and `/api/auth/credentials/login`.
  - Fixed mixed-content browser fetch blocking behind the Cloud Run TLS proxy by deriving `https://` `base_url` from `X-Forwarded-Proto: https` (or `K_SERVICE`) and upgrading `http://` `API_BASE` references to `https://` on `https:` pages.
  - Seeded verified legacy credential records into `/data/hivemind_router.sqlite3` on container startup in `deploy/gcp_cloud_run/Dockerfile` and `deploy/gcp_openrouter/Dockerfile`.
- **OpenTrainer Campaign Validation, Gosset $E_8$ Lattice Parity, Producer Agent/Shield Unification & Streaming TTFT Telemetry (PR #703 — Issue #618, IP-259, IP-30, IP-38, Audit C4/C9/B10):**
  - Enforced `is_allowed_opentrainer_campaign_id` in `/api/opentrainer/heartbeat` and `/api/opentrainer/submit_delta` (`HTTP 400 unknown_campaign` on unregistered campaigns).
  - Unified `GossetE8LatticeQuantizer` with canonical `nearest_e8` / `nearest_dn` (`hivemind.compress.lattice_quantizer`) for bit-for-bit tie-breaking parity and grounded $16\times$ dense / $30.89\times$ sparse+E8 compression figures against `artifacts/experiments/E10_lattice_diloco.json`.
  - Synced `scripts/hivemind_producer_{agent,shield}.py` with `src/hivemind/router/hivemind_producer_{agent,shield}.py` byte-for-byte and exposed `GET /hivemind_producer_agent.py` and `GET /hivemind_producer_shield.py`.
  - Recorded streaming `ttft_ms` (`p50`/`p95`/`mean`) and `tokens_per_s` in `usage_telemetry` and `AccountStore.usage_capture_summary`.
  - Marked deterministic game telemetry with `"simulated": True` (`HiveMindScientificGameEngine`) and removed fabricated OpenRouter completion IDs from `CANONICAL_5_ROADMAP_PRS`.
- **Core, Swarm, Security, Economy & Data Hardening (PRs #692–#701 — 45+ Improvement Plan Audits):**
  - Multi-Krum collinear outlier rejection ($>5\times$ median norm when $f$ is clamped to $0$, IP-34), NF4/DCT quantizer fidelity, eval-answer substring decontamination (`IP-40`), Lean 4 `QueenBee.lean` section SHA-256 verification (`IP-41`), pure-AST candidate energy evaluator (`SEC-01`), and T4 governor validator awareness (`IP-24`/`IP-50`).

## v3.1.1 — 2026-11-04 — Security and honesty fixes to the v3.1 web interface

Fixes the review of `v3.1.0` (PR #595). `v3.1.0` should not be deployed.

- **Writes need a caller.** On the router and the dashboard, Kafka publish, training proposals and GPU pledges need a signed-in account, and the handle comes from that account rather than the request body. T4 schedule, maintenance toggle and portal review are owner-only. Router writes share the AMCP per-account rate limit; malformed bodies return 400.
- **No more made-up success.** Kafka publish reports `RECORDED_IN_MEMORY_NOT_KAFKA` with no invented broker or acknowledgement; the schedule reports `SCHEDULE_SAVED_NOT_APPLIED`; the globe reports `DEMO_DATA_NOT_LIVE`; the overview reports `RELEASE_V3_MVP_IN_PROGRESS`; PR auto-merge reports `STATIC_LIST_NO_ACTION_TAKEN`; the maintenance acknowledgement and PR auto-merge credit 0 HMT; the red-team route reports `SELF_REPORTED_NOT_INDEPENDENTLY_VERIFIED`.
- **Accounts.** No default password in the page; login no longer creates an account for an unknown handle; the page no longer shows a locally invented key or demo token when a call fails; the failed-password limit is per account and client (8 per minute), with a per-account ceiling of 200, so a stranger cannot lock an owner out.
- **Script injection.** Fields from the server or the user are HTML-escaped before they reach `innerHTML`; `/v3.1` on the router now sends the portal's CSP and HSTS. That CSP still allows inline scripts because the page's handlers need them.
- **Secret scan.** Removed the placeholder `hm_live_` value that failed SEC-05 (it never authenticated).
- **Tests.** The Playwright module skips when Playwright is not installed; the Steam status test no longer assumes an RTX GPU.

Known limits: portal `hm_live_` keys do not work on the gateway, so router writes from the portal are refused until the two are joined; portal accounts and the calendar live in memory and vanish on a restart or scale to zero; the globe's numbers are still example data.

## v3.1.0 — 2026-11-04 — Unified GCP Web Interface (v3.1) on HiveMind v3.0.0 Core Engine, Real-Claim Scorecard Binding & Red Team GHSA-gjc6-74f9-wmrr Verification

- **Deployable on HiveMind `v3.0.0` Core Engine (`https://github.com/xaviercallens/HiveMind/releases/tag/v3.0.0`):**
  - Mounted the Stitch V3.1 GCP Unified Web Interface (`/stitch-portal`, `/v3-portal`, `/v3.1`) and all V3.1 API endpoints (`/api/v1/v3-mvp/overview`, `/api/v1/v3.1/overview`, `/api/v1/amcp/kafka/*`, `/api/v1/training/*`, `/api/v1/dashboard/*`, `/api/v1/security/*`, `/api/v1/portal/gcp-staged-release`) onto both `create_router_app` (`src/hivemind/router/gateway.py` / `deploy/gcp_openrouter/deploy.sh`) and `create_hivemind_at_home_app` (`src/hivemind/boinc/hivemind_at_home_dashboard.py` / `deploy/gcp_cloud_run/`).
  - Made `numpy` and `safetensors` optional imports in `hivemind_at_home_dashboard.py` and staged `artifacts/mvp/*.json` + `artifacts/experiments/GRANITE_ADAPTER_hmq_00147_gate_f16_independent_chunks.json` + `docs/MVP_SCORECARD.md` in `deploy/gcp_openrouter/deploy.sh` & `Dockerfile`.
- **Real-Claim Binding to `v3.0.0` Core Engine & Scorecard (`2/11 PASS • 1 WAIVED • B3 Gate Enforced`):**
  - Bound `ReleaseV3MVPManager` directly to `src/hivemind/router/measured_results.json` (`hivemind.measured.v1` `results` array: `hivemind/bee-stars` 283/317 = 89.27%, `hivemind/coder-lite` 15/15 smoke checks, `hivemind/granite-1b-base` CPU tier 68 tok/s), `artifacts/mvp/A2_split_moe.json` (20/20 token-identical 2-host split MoE at 41.75 tok/s vs 68.95 tok/s single host), `artifacts/mvp/B1_custom_shards.json` (1,044 docs -> 2,082 train + 304 held-out signed chunks), and `artifacts/experiments/GRANITE_ADAPTER_hmq_00147_gate_f16_independent_chunks.json` (`hmq_00147` held back by `scripts/adapter_to_served.py` with `servable=False`, `mean_nll_gain_nats=-0.00403`).
  - Wired `/api/v1/dashboard/nash-equilibrium` to execute live checks against `v3.0.0` core economy & ledger functions: `route_supply_equilibrium` (`ff394be`), `erlang_c_delay_probability` (`176a32d`), and `settle_swarm_inference` (`37f5ede` integer micro-unit fee split conservation).
- **Red Team Security Verification (`GHSA-gjc6-74f9-wmrr` — 6/6 Controls Active):**
  - Enforced `Authorization: Bearer` / `PBKDF2-HMAC-SHA256` (200k iterations + 8/60s rate limit) across all auth & Steam routes, persistent unverified-volunteer prompt privacy notice (`#poc-notice`), mandatory `hivemind_producer_shield.py` + Ollama `>= 0.5.0` (`CVE-2024-37032`) preflight validator, zero public node IP/URL exposure, and honest `served_by` off-the-shelf model catalog disclosure.

## v3.0.0 — 2026-11-04 — Honest MVP baseline: measured results, a scorecard, and the train-to-serve gate (proof of concept)

**Proof of concept, not a production service: send only public, non-sensitive data.** This is not the finished MVP: the scorecard (`docs/MVP_SCORECARD.md`) reads **2 of 11 criteria passing** (B1 custom shards, A2 split inference), 1 waived by the owner (F1), and the rest failing or not run, each with its reason.

- **Target and measure:** `docs/MVP_TARGET.md` (11 machine-checkable criteria), `scripts/mvp_scorecard.py`, `docs/MVP_OWNER_DECISIONS.md`.
- **Reality audit:** `docs/MVP_REALITY_AUDIT.md` classifies all 191 source files; hard-coded benchmark claims, fabricated browser credit, invented training and hardware records are removed; the benchmarks page shows measured results only.
- **Inference:** real-model smoke test (`scripts/mvp_e2e_smoke.py`); failed endpoints cool down for 10 s; one MoE split across two hosts over llama.cpp RPC is token-identical to a single host on 20/20 prompts with `--no-repack` (CPU on both hosts; 8/20 with the default repacked kernels); `granite-1b-base` CPU tier; catalog tiers state base, adapter or alias.
- **Training:** `scripts/adapter_to_served.py` gates a trained adapter on the served files. **The BOINC adapter `hmq_00147` fails it**, so no HiveMind-trained adapter from the chain is served. `make_custom_shards.py` builds signed shards from your own JSONL; stratified rounds (`t4_workgen.py --mix`) are implemented, not yet run.
- **Security:** live BOINC verdicts verify an honest workunit and reject a lazy worker; Multi-Krum rejects poisoned deltas offline only; billing inflation is bounded loosely (about 4x). Round-4 hardening. Gated expert promotion.
- **Operations:** maintenance mode (`HIVEMIND_MAINTENANCE_NOTICE`); budget target 150 USD a month, `hivemind-openrouter` scale-to-zero (AGENTS.md changed); `scripts/d1_cost_estimate.py` (currently over budget), `scripts/kafka_voters.sh`, `scripts/t4_training_window.sh` (not applied).

## v2.15.0 — 2026-10-29 — Season 29 International Launch Weekend Scaler (`HM-S29-01`, `HM-S29-02`, `HM-S31-02`), Human & Agent Calendar, Nocturnal Image-MoE Evolution Monitor & 161 Lean 4 Theorems

- **Pre-Registered Experiment `E20` & Large-Audience Scalability (`src/hivemind/swarm/international_launch_scaler.py`):**
  - Pre-registered `artifacts/experiments/E20_international_launch_split_dct_prereg.json` and verified `artifacts/experiments/E20_international_launch_split_dct_report.json`.
  - **`HM-S29-01` Adaptive Multi-Split ML (`select_adaptive_multisplit_cut`)**: Dynamic cut-layer controller selecting between `patch_stem_int4_bottleneck` (`680 MiB`), `early_vit_stage2_token_merger` (`1,120 MiB`), and `full_frozen_vit_plus_prelude_adapter` (`1,480 MiB`) based on real-time QUIC uplink bandwidth (`10–1,000 Mbps`) and RTT, keeping client Prelude VRAM strictly `<= 1,536 MiB` and cutting WAN step latency by `88–99%`.
  - **`HM-S29-02` Extreme Top-1% 2D-DCT + EF21 Error-Feedback (`compress_top1pct_dct_ef21`, `decompress_top1pct_dct`)**: Orthonormal 2D-DCT Top-1% sparsification + 4-level (`2-bit`) quantization achieving **`225.4x` wire compression** (`>= 100x` target) with exact host-RAM `EF21` residual conservation (`max drift = 3e-08`, `0 MiB` extra GPU VRAM).
  - **`HM-S31-02` 6-Region Super-Peer Hierarchy, 15–20 Node Sub-Swarms & SQLite Batch Coalescer (`compute_hierarchical_fanout_plan`, `coalesce_ledger_settlement_batch`)**: Groups `10,000` international leaf nodes into `558` localized sub-swarms (`18` nodes) and `318` regional Super-Peers (`B=32`), reducing central `hivemind-openrouter` ingress by **`31.45x`** and coalescing single-writer Litestream SQLite HMT settlements with `62.5–90%` write IOPS reduction.
- **Human & Autonomous Agent International Launch Calendar & 4 Grand Challenges (`Oct 30 – Nov 1, 2026`):**
  - Published `docs/community/INTERNATIONAL_WEEKEND_LAUNCH_CALENDAR.md`, `artifacts/international_weekend_launch_calendar.json`, and FastAPI routes `GET /api/v1/launch/calendar`, `GET /.well-known/hivemind-launch-calendar.json`, `POST /api/v1/launch/enroll`, and `POST /api/v1/launch/scalability-plan` with 6-language human announcements (`EN`, `FR`, `ES`, `ZH`, `PT`, `JA`) and A2A/MCP agent discovery manifests.
- **Tonight's Nocturnal Image-MoE Evolution Monitor & 161 Lean 4 Theorems:**
  - Executed 3-generation Pareto evolution (`monitor_nocturnal_image_moe_evolution`, `GET /api/v1/night-moe/reddit-vision/evolution`, `POST /api/v1/night-moe/reddit-vision/evolve`) across `Reddit-Vision` Culturally-Grounded VLM MoE (`0.00213 -> 0.00122` upcycled MoE loss), `<1.5 GB` Exoplanet & Astrophysics V-MoE PoC (`0.00001` focal loss, `1,410 MiB` peak VRAM), and `La Ruche` 7-Division 42.87M-Sample 104-Bee Massive Image-MoE Swarm (`0.00000` focal loss, `100%` routing accuracy) while maintaining `10,073 MiB` free VRAM (`>= 4,300 MiB` floor) on the local Tesla T4.
  - Proved Theorems 159–161 (`super_peer_fanout_reduces_coordinator_load`, `sqlite_write_coalescing_conserves_hmt`, `nocturnal_vlm_moe_pareto_ratchet_monotone`) in `lean4/HiveMindSpec.lean` (**161 machine-checked Lean 4 theorems, 0 sorry, 0 custom axioms**).

## v2.14.0 — 2026-10-29 — Season 28 CrowdSwarm v2 Resilience (`HM-S17-CW-02/03`), Scout Scientific Parsers (`#319`), Multi-Node Nectar Sync & 158 Lean 4 Theorems

- **CrowdSwarm v2 Resilience (`src/hivemind/swarm/crowdgpt_bridge.py`, `#204`):**
  - Added `check_weight_rms_collapse_guard` to prevent the Step-282 zero-participant weight-halving failure (`0.5 * old_weights`, `rms 0.0387 -> 0.0195`) diagnosed by `@jpmlima` in `Vxtzq/CrowdGPT#48`.
  - Added `classify_gpu_precision_cohort` enforcing `fp32` on pre-Ampere NVIDIA GPUs (`compute_capability < (8, 0)`, e.g. Pascal `sm_61`, Turing RTX 2070 / Tesla T4 `sm_75`) and isolating `bf16` vs `fp32` cohorts (`Vxtzq/CrowdGPT#44`, `#51`).
  - Added `resolve_forming_pool_starvation` (falling back from stalled `1/3` or `2/3` pools to solo Multi-Krum + Held-Out Loss-Ratchet verification after timeout) and `evaluate_client_resilience_policy` (OOM-at-`batch_size=1` abort circuit breaker `#52` and HTTP `502/503` backoff `#50`).
- **Scout Bee Scientific Domain Parsers & Multi-Node Nectar Sync (`#319` Tasks 1 & 2):**
  - Added `parse_scientific_scout_record` in `src/hivemind/swarm/abc_moe_agent_mesh.py` for structured `arXiv`, `PubMed`, `ChEMBL`, and `materials_science` extraction (`arxiv_id`, `pmid`, `ic50_nm`, `chembl_id`, `pchembl_value`, `canonical_smiles`, `doi`).
  - Added `MultiNodeNectarSyncBackend` and `InMemoryClusterNectarHub` in `src/hivemind/swarm/swarm_nectar_cache.py` for `< 50 ms` cross-node SHA-256 deduplication and HMT discovery royalty propagation.
- **Formal Verification & Roadmap Sync (`lean4/HiveMindSpec.lean`, `ROADMAP.md`, `#292`):**
  - Proved Theorems 156–158 (`crowdgpt_rms_halving_rejected`, `pre_ampere_fp32_cohort_isolation`, `forming_pool_timeout_liveness` — **158 machine-checked Lean 4 theorems, 0 sorry**) and integrated the Bee-Mesh Season 1 (`HM-S1-01..09`) and Seasons 26–28 tables into `ROADMAP.md`.

## v2.11.0 — 2026-10-29 — Bi-Layer Honeybee Mesh, 24 MoE Bees, Low-Bit (1.58/2/3-Bit) Dream Teachers, RTX 2070 BOINC + Steam v5.1 Fat Client & 5-Language `/rejoindre` Onboarding

- **Low-Bit (1.58-Bit / 2-Bit / 3-Bit) Nocturnal Dream Phase Teachers (`#323`):**
  - Integrated `peculiar-ragdoll/Tiel-Coder-35B-A3B-GGUF` (`UD-IQ3_XXS` `12.30 GiB`, `UD-IQ3_S` `12.74 GiB`, `48.0%` SWE-bench-Live) and `prism-ml/Ternary-Bonsai-2-27B-gguf` (`PTQ1_0` `1.75` bpw `5.54 GiB` fitting 100% inside the `6.5 GB` RTX 2070 Golden Invariant I1; `PQ2_0` `2.13` bpw `6.71 GiB` enabling simultaneous 27B Teacher + QLoRA Student co-residency on a 16 GB Tesla T4), `Ternary-Bonsai-2-9B-gguf` (`2.25 GiB`), and `Bonsai-27B-gguf` (`3.54 GiB` 1-bit) into `src/hivemind/core/auto_improvement.py` (`LOW_BIT_DREAM_TEACHER_MODELS`, `select_dream_phase_teacher_for_vram`) and `src/hivemind/swarm/campaign_orchestrator.py` (`select_dream_teacher_for_job`).
- **Live 5-Language Onboarding (`/rejoindre` & `/join`) & BOINC Community Hub (`/community`, `#314`, `#323`):**
  - Full 5-language onboarding guide (`FR`, `EN`, `ES`, `DE`, `ZH`) across `hivemind-openrouter` and `hivemind-at-home-ruche`.
  - Mounted `/community`, `/boinc`, `/api/v1/community/hub`, `POST /api/v1/community/boinc-sync`, `POST /api/v1/community/post-note`, and direct `/scripts/{script_name}` downloads on `hivemind-at-home-ruche`.
- **Two-Layer Honeybee Agent Mesh (AMCP Kafka + ABC MoE) & Swarm Nectar Cache (`#318`, `#320`, `#321`, `#322`):**
  - Layer 1 AMCP v1.5 + CloudEvents / Kafka Quorum Bus (`src/hivemind/swarm/honeybee_mesh_fabric.py`, `src/hivemind/swarm/nectar_cache.py`) and Layer 2 Artificial Bee Colony (Employed / Onlooker / Scout Bees) across 24 specialized MoE micro-experts (`/life-as-a-bee`).
  - Season 25 LayerStoRm x HiveMind PCIe expert streaming bridge (`src/hivemind/core/layerstorm_bridge.py`) and Foundation Paper v4.0.0 (`DOI: 10.5281/zenodo.23756909`).
- **RTX 2070 CUDA PTX BLS Worker, Free Steam Fat Client v5.1 (AppID `3133700`) & Stitch MCP v4.6 Cockpit (`#314`):**
  - Real `cupy.RawKernel` CUDA PTX Box-Least-Squares (BLS) transit periodogram worker (`scripts/hivemind_boinc_idle_worker.py`) enforcing the `<= 6.5 GB` VRAM ceiling (`I1`) and `< 1.5s` game preemption (`I4`).
  - Zero-install WebGPU Spiral Galaxy & WGSL `@compute` citizen science cockpit (`/webgpu-galaxy`) and Stitch v4.6 12-screen Tactical Bridge (`src/hivemind/game/exoplanet_hunter_cockpit.py`, `tests/test_game_and_agent_coverage_gate.py`).

## v2.16.1 — 2026-11-02 — security hardening (GHSA-gjc6-74f9-wmrr) and a truthful model catalog

- **Security:** account ownership is now verified before any key or token is returned on the OpenRouter gateway and the La Ruche dashboard; a previously published gateway key is revoked; contribution routes no longer award HMT for client-supplied receipts; sign-in tokens are single-use in every spelling; dashboard admin routes need an admin key; several public routes got input caps and rate limits. Details are in the advisory once it is published; upgrading is recommended.
- **Behaviour changes:** BOINC/Steam/WebGPU contributions are recorded but earn 0 HMT until results are verified server-side; handle-only accounts sign in with their API key; the dashboard admin needs `HIVEMIND_DASHBOARD_ADMIN_KEY`; `/api/v1/community/boinc-sync` needs `HIVEMIND_BOINC_SYNC_KEY`; `POST /v1/hivemind/admin/accounts/rotate-key` (admin) re-issues an account's key.
- **Catalog:** 26 partner entries that named Hugging Face repositories which do not exist are now shown as "Planned, not trained yet" (no link, no parameter or token claims) with the off-the-shelf model that actually serves each tier; new real small-MoE tier `granite-moe` (IBM Granite 3.1 MoE 3B, 800M active, Apache-2.0).
- **Notice:** HiveMind is a proof of concept: send only public, non-sensitive data (README, SECURITY.md and the sign-up, join, portal and dashboard pages).

## Unreleased

- docs(roadmap): Season 21 "Operation Kafka-Hive" section in `ROADMAP.md` (§3.18, tasks `HM-S21-01`..`HM-S21-06`): AMCP Kafka transport ([#296](https://github.com/xaviercallens/HiveMind/pull/296)), expert registry and residency planner ([#299](https://github.com/xaviercallens/HiveMind/pull/299)), campaign joins and night dream-phase planner ([#298](https://github.com/xaviercallens/HiveMind/pull/298), open), base-model catalog and promotion gate ([#295](https://github.com/xaviercallens/HiveMind/pull/295), [#302](https://github.com/xaviercallens/HiveMind/pull/302)), dataset registry ([#297](https://github.com/xaviercallens/HiveMind/pull/297)) and bring-your-own-bee ([#300](https://github.com/xaviercallens/HiveMind/pull/300), open), each with its gate, what is not yet done, and the wiring order (BYOB -> registry; orchestrator -> Kafka transport; dream phase -> datasets + base models + registry). Docs only. No training was run and no broker or service was deployed by these PRs; the Qwen 3.5 night training on the T4 still awaits the owner's approval to stop serving at night.

- docs(night): `docs/NIGHT_INTENSE_PLAN.md` and pre-registration `artifacts/experiments/NIGHT_INTENSE_1_prereg.json` (NIGHT-INTENSE-1: does a larger base or higher LoRA rank reduce held-out loss per night-hour versus Granite-1B, without degrading morning coder latency). Three VRAM tiers, candidate bases, acceptance rules, probe protocol and governor options for the owner. New `scripts/qlora_vram_estimate.py` (stdlib-only analytic NF4 QLoRA peak-VRAM estimator; leave-one-out calibration error 17.6 % on three points, so a guide, not a measurement) with `tests/test_qlora_vram_estimate.py`. Nothing applied: no governor change, no probe, no GPU use, no model loaded.

- HM-S19-CAMPAIGNS: Subject-by-Subject Joint Improvement Campaigns (`SUBJ-01`..`SUBJ-06`), Community Interest & VRAM Pledge Poll (`POST /v1/hivemind/campaigns/poll`, `GET /v1/hivemind/campaigns`), RFC 5545 Joint Campaign Calendar (`GET /v1/hivemind/campaigns/calendar.ics`, `docs/calendar/hivemind_joint_subject_campaigns.ics`), and AMCP v1.5 + CloudEvents / Kafka Cross-Human, Cross-Agent & Inter-Bee Bus (`src/hivemind/swarm/amcp_kafka_campaign_bus.py`, `POST /v1/hivemind/amcp/publish`, `GET /v1/hivemind/amcp/events`, `POST /v1/hivemind/bees/coordinate`, `POST /v1/hivemind/bees/trade`, `GET /v1/hivemind/bees/market`). Covered by `tests/test_amcp_kafka_campaign_bus.py`.
- HM-S19-USAGE: privacy-preserving usage capture (`src/hivemind/router/usage_capture.py`, `usage_telemetry` and `usage_feedback` tables in `RouterStore`), smart intent routing on `hivemind/auto` (`bee-rusty-sundials` for stiff ODE/PDE/SUNDIALS, `bee-stars` for astrophysics, `coder` for coding when live, falling back to largest supply), client SDK/IDE detection (`X-Title`, `HTTP-Referer`, `User-Agent`), public aggregate `GET /v1/hivemind/usage-capture` (domain share, `fr` vs `en` language share, IDE breakdown, latency `p50`/`p95`, nocturnal MoE workunit weights), authenticated `POST /v1/hivemind/feedback` (`+1`/`-1`), and 1-click Rusty-SUNDIALS + French presets in `/playground`. Covered by `tests/test_usage_capture.py`.
- docs(boinc): `docs/BOINC_RTX_GPU_INSTALL.md`, an installation and validation guide for volunteers with NVIDIA RTX cards, with an explicit verified / not-tested table (only a Linux Tesla T4 has run), the E12 cross-device replay-tolerance risk, and a checklist for the first RTX volunteer. Docs only; nothing was run.
- Proposal: Bee MoE with numeric solver experts, three waves (docs/BEE_MOE_NUMERIC_SOLVERS_PROPOSAL.md); nothing implemented

- Optional resident validator (off by default): `src/hivemind/boinc/real_client/resident_validator.py` keeps the validator's base model loaded across results (thread-safe LRU `ModelCache`, UNIX-socket server/client with a length-prefixed JSON protocol, CLI `serve|ping|stats`). `hivemind_validate` forwards to it when `HIVEMIND_RESIDENT_VALIDATOR_SOCKET` (or `--resident`) is set and falls back to the one-shot path with one warning line when it is unreachable. `boinc_server_scripts.validate` now delegates to the shared `run_validation`, which both paths call. Tests in `tests/test_resident_validator.py` (CPU, fake model). Not done: no real model or GPU run, not enabled on the live node, GPU sharing with volunteer tasks unsolved, no measured speed-up. See `docs/BOINC_PRODUCTION_SERVER.md`.
- Volunteer preflight (`scripts/hivemind_volunteer_preflight.py`, docs/community/VOLUNTEER_PREFLIGHT.md): stdlib-only check of OS, NVIDIA GPU and VRAM tier, free disk against the ~4.4 GB first download, RAM/CPU, existing BOINC client and battery, with a ready / ready_with_warnings / not_ready verdict (exit 0/1/2) and `--json`. No network connection unless `--project-url` is given (https and public addresses only); no hostname, user name or GPU UUID is output.

- HM-S16-07 (refs #194): account kinds `user` (default), `test` and `probe`, set only with the admin key (`POST /v1/hivemind/register` with `kind`, or `POST /v1/hivemind/admin/test-flag` with `kind`). Test and probe accounts never bid, are priced at the equilibrium computed without them, and their producers are not supply for pricing; `/v1/hivemind/admin/users` and `/v1/hivemind/me` report the kind and count `real`, `test` and `probe` apart. New gateway options `starter_grant_hmt` (default 0 = off), `starter_grants_per_ip_24h` (default 3) and `trusted_proxies` (default empty = `X-Forwarded-For` never trusted): over the per-address cap an account is registered without the starter grant; only a keyed hash of the address is kept, for 24 h. Old databases gain the `kind` column in place. See `docs/HIVEMIND_OPENROUTER.md`.

- HM-S16-05 (refs #192): remote control now acts on producers. `POST /v1/hivemind/producer/control` (own API key, or the admin key plus `account_id`/`handle`) stores `paused` and an optional `vram_cap_mib`; the gateway excludes a paused producer from routing, `/v1/models` and the market at once, scales capacity for a cap below the loaded models' `vram_mib` with the same clamp as the card check (`fit_loaded_models`), and returns a `control` object in the heartbeat ack (only once a state exists). `scripts/hivemind_producer_agent.py` obeys it (zero capacity and `paused` marker, over-cap models left out, cap reported back, malformed `control` ignored and logged once) and publishes it to `--control-state-file` (mode 0600); `scripts/hivemind_producer_shield.py --control-state-file` answers chat with 503 `producer_paused` while paused and keeps `/v1/models`. New table `producer_control` (created in place). `tests/test_router_control.py` (34 tests, one end-to-end). Not done: no deploy, live T4 agents not restarted, the cap cannot unload a model from Ollama.

- HM-S16-06 (refs #193): producers may report model residency in heartbeats (`loaded`, `vram_mib`, `cold_start_s` per model, `gpu_vram_mib` per card; the producer agent probes the local Ollama `/api/ps`, `--no-residency` disables). The router prefers loaded models (unknown after loaded, known-not-loaded last), answers 503 `model_loading` with `Retry-After` instead of forwarding to a model that must cold-load longer than `cold_start_wait_budget_s` (default 20 s) with no charge, reports upstream timeouts as `model_loading` or 504 `upstream_timeout` instead of an opaque error, and scales declared capacity down when the loaded models exceed the card. Old producers are unaffected. See `docs/HIVEMIND_OPENROUTER.md`.

## Zenodo deposit of manuscript v3.8.0 — 2026-10-25

- Published as 10.5281/zenodo.23649757 (new version of 10.5281/zenodo.23607597, concept DOI 10.5281/zenodo.23597697; 99 files; checksums verified against the local tree and, for the eight carried-over E12 replay tensors, against the v3.7.1 receipt). Not peer reviewed; the record bundles the E10b-E16 preregistrations, amendments, raw data and reports, the GWAYA-Lite live check and the T4 campaign report.
- `scripts/publish_hivemind_paper_zenodo.py`: default version 3.8.0, extended bundle, an honest description (128 theorems, invalid rows named), carried-over file verification, and a filter that keeps run logs containing the machine username out of the public record. `paper/zenodo_carried_over_v3.7.1.json` lists the carried-over md5 values.
- Known differences from main (also in `paper/ZENODO_DEPOSIT_RECEIPT.json`): the Lean file in the record is the 128-theorem version the paper describes (main has since added 13 theorems); the T4 queue run log is not in the record.

## Unreleased — 2026-10-24 — paper v3.8.0 corrections and T4 replication kit (no version tag)

- Manuscript v3.8.0 (not deposited; the deposited record stays v3.7.1, 10.5281/zenodo.23607597): withdraws the INVALID E11 Qwen2.5-Coder-7B row (negative control failed by -0.47 nats), states P11a on Qwen2.5-0.5B as refuted on one rate-matched cell, adds a labelled-untested likely-cause note to E10b, and adds provenance checks PROV-72 and PROV-73; fig. 8 is regenerated without the 7B panel; the ledger gains a `withdrawn` block for P11a/P11b.
- T4 replication kit: `docs/community/T4_REPLICATION_KIT.md` and `scripts/compare_replication.py` (compares a replication run against the campaign receipts; 4 tests in `tests/test_compare_replication.py`).
- Test: `tests/test_e15_gate.py` parses with `ast.literal_eval` instead of `eval`.
## Unreleased — General OpenID Connect sign-in (after Google)

- `src/hivemind/router/oidc_auth.py`: provider-agnostic ID-token verifier (`OidcProvider`, `OidcIdTokenVerifier`, `providers_from_json`); `google_auth.py` is now this engine with Google's settings (all 43 Google tests unchanged and passing).
- `POST /v1/hivemind/register/oidc/{provider}` and public `GET /v1/hivemind/signin/providers`; `RouterStore.allow_identity_provider` reserves each provider's handle prefix (refused if a handle already uses it); `HIVEMIND_OIDC_PROVIDERS` in the Cloud Run server.
- `tests/test_router_oidc.py` (13 tests). Mutation checks: turning off the issuer check, the verified-email rule, the per-store prefix reservation, or the provider lookup each makes a test fail.
- Not done: Google itself still needs the owner's OAuth client id; `deploy.sh` does not forward `HIVEMIND_OIDC_PROVIDERS`; no live provider was exercised.
## Unreleased — moe-general (gpt-oss 20B) measured on a T4

- `catalog.py`: `moe-general` minimum memory 16 GB to 15 GB, from a measurement (12,331 MiB resident on a T4, card otherwise empty); the catalog test now covers 14.9, 15.0 and 15.36 GB.
- `gateway.py`: `reasoning_effort` is forwarded when it is exactly `low`, `medium` or `high` (any other value is dropped), so consumers can limit a reasoning model's thinking. A test caught an unhashable-value crash in the first version; fixed and covered.
- `docs/HIVEMIND_OPENROUTER.md`: how to serve the tier on a 15 GB card, with the measured memory, speed and the six-task result (n = 6). Not yet served live: it needs the T4, which is being used for E17 training.

## Unreleased — Prompt-level routing: whole-word keywords and abstention

- `src/hivemind/core/btm_moe.py`: new `keyword_route` (whole-word keyword matching, abstain with `([], {})` when no keyword hits, ties by domain name, optional `min_weight` default 0.0), used by `Season3DomainExpertRegistry.route_domain_query` and `route_with_serving_experts`. Before: with no hit every domain scored 1.0 and the dict order decided, so five of seven probe prompts (python, cardiology, geography, a proof, a pandas traceback) went to Rust + Solidity; keywords matched inside longer words ("arc", "nat", "pub", "by").
- **Contract change:** a prompt with no keyword hit returns `([], {})`; callers must use the base model. The existing property test now allows the empty result. No code in `src/` calls these functions outside the two modules.
- Not fixed here: some registry keywords are ordinary English whole words (`by`, `require`, `contract`, `trial`, `proof`), so they can still fire; the labelled routing set must carry them as hard negatives. Patch written by the GWAYA session, read in full and applied; full suite 100% with no failures.
## Unreleased — GWAYA Level B: sandbox-gated acceptance of LoRA / DiLoCo updates

- `src/hivemind/swarm/verified_update_gate.py`: `decide` (exact paired McNemar gate on held-out sandbox outcomes: same tasks, minimum n, UNVERIFIED never a pass and too much of it blocks, net-loss and significant-regression rejection, optional required improvement, sealed receipt), `evaluate_tasks` (generator plus injected sandbox gate such as `e15_gate.gate`) and `gate_update`.
- `docs/community/GWAYA_LEVEL_B_UPDATE_GATE.md`: rules, limits and use.
- **No LoRA was trained or evaluated and no pass-rate number is claimed.** The gate is not yet wired into the DiLoCo merge or the E14 admission policies.
- `tests/test_verified_update_gate.py`: 17 tests (exact-p closed forms, better/regressed/identical controls, UNVERIFIED-heavy refusal, Hypothesis properties, evaluator with a mocked sandbox, and one real `e15_gate` sandbox run when the host allows it). Mutation checks: scoring an unverified baseline as a fail, an unverified candidate as a pass, or switching off the net-loss rule each fail a test.
## Unreleased — Improvements recorded from the live T4 runs

- `scripts/hivemind_connect.py`: the OpenCode snippet now carries a model output limit (`limit.output` = the chosen max tokens) and a context hint, and explains how to turn tools off when a 7B model writes tool calls as text. The docs already claimed snippets set 2,048; the OpenCode one did not.
- `docs/HIVEMIND_OPENROUTER.md`: new section "Lessons from the first live runs" (producer slots, reserve, tool calls, context, measured quality, determinism, adapters, disk, sandbox boundary).
- `ROADMAP.md`: HM-S13-16 to HM-S13-20 (isolated verifier service, busy-producer handling, reserve sizing, routing defects reported by the GWAYA session, llama.cpp adapter serving).
- New `docs/HIVEMIND_CONNECT_IDE.md`: base URL, key, model ids, per-editor settings (OpenCode, VS Code Continue/Cline, JetBrains, Zed, Aider, env), a table of what was tested live versus format-checked only, verified mode, and a troubleshooting table. Linked from the README, the OpenRouter doc and a new wiki page.
- Portal (`/`): snippet tabs for OpenCode, Zed, JetBrains and env variables; Continue shows the current YAML format; OpenCode carries the output limit; no tab points at a tier that has no producer. A test checks every button has a snippet and the limit is present.
- Nothing here changes the gateway's behaviour.

## Unreleased — Verified mode for coding requests (client side)

- `scripts/hivemind_verified.py`: generate through the router, judge with the caller's own tests in the local fail-closed sandbox, repair with the failure text (<= 3 repairs), return the first passing candidate; otherwise the first attempt flagged `UNVERIFIED_FALLBACK`; no sandbox means `BLOCKED`. Library functions `verified_solve`, `sandbox_judge`, `router_ask`.
- Measured on the live router and the T4 (MBPP-sanitized, n=257, 7B `coder`): 76.3% first attempt, 82.9% with the loop, 17 rescued / 0 lost, exact McNemar p = 1.5e-5; ~192 HMT. Sampling uncontrolled (Ollama ignores temperature/seed on /v1). One benchmark, one run.
- Deliberately NOT in the gateway or the producers: they would have to execute code supplied by strangers. A server-side version needs a separate isolated verifier service; not built.
- `tests/test_hivemind_verified.py` (9 tests). Mutation checks: first-attempt fallback, blocked sandbox, always-accept and withheld feedback each fail tests.

## Unreleased — GWAYA-Lite Python code expert as a serving-layer expert

- `src/hivemind/core/serving_experts.py`: `SERVING_EXPERT_SPECS` (first entry `Python_Code_GWAYA`), `route_with_serving_experts` (same `1.0 + 2.5 x keyword hits` scale as `route_domain_query`, over LoRA domains and serving experts) and `load_gwaya_code_expert` (lazy import of the optional `gwaya` package; 7B-only by default).
- Kept out of `SEASON_3_DOMAIN_EXPERT_SPECS` on purpose: a serving expert has no adapter or dataset, and adding it there would give `train_and_publish_season3_domain_experts` a fifth domain and break the 4-domain Season 3 test.
- Source and numbers: GWAYA repo PR #21 (T4, MBPP-sanitized, n=257, all tests required): 7B Q4 82.1% raw / 82.9% with the GWAYA loop (+2/0, p=0.5, not significant), 3B Q4 70.8% / 71.6%; a 3B-first cascade did not pay off (83.3% vs 82.9%). Measured by the GWAYA session, not re-run here. "VERIFIED" means the visible tests passed.
- `tests/test_serving_experts.py`: 5 tests run anywhere; 2 need a GWAYA release with the Code expert and are skipped otherwise (run with `PYTHONPATH=<GWAYA repo>`: 7 passed).

## v2.10.2 — 2026-10-24 — OpenRouter Interactive Portal, In-Browser Playground & GPU Point Calculator

- `src/hivemind/router/portal_page.py`: interactive OpenRouter portal and AI playground at `/` and `/playground`. Features in-browser chat completion console with streaming token speed (tok/s) and latency metrics, 5 capability tiers matrix, live account inspector (`/v1/hivemind/me`), 1-click editor integration snippets (Python, Continue, Cline, Aider, OpenCode, cURL), and gamer/producer onboarding. Strict CSP with per-request nonces and textContent-only rendering.
- `src/hivemind/router/gateway.py`: mounted `GET /` and `HEAD /` (and alias `/playground`), plus explicit HEAD support for `/signup`, `/health`, `/healthz` ensuring RFC 9110 compliant 200 responses to health probes and HEAD curl requests.
- `src/hivemind/boinc/hivemind_at_home_dashboard.py` & `HiveMindatHome.github.io`: added interactive GPU compute points calculator (8 consumer GPU presets, idle hours slider, monthly HMT points & free token projection) and interactive live completion console in Screen 02C, fully localized across 5 languages (`en`, `fr`, `zh`, `es`, `de`).
- `tests/test_router_portal.py`: comprehensive unit test suite covering root portal and playground routes, HEAD methods, CSP headers, and client ID validation. 227 passed, 1 skipped. GWAYA zero-stub audit passing 111/111 files.

## 10-hour T4 benchmark campaign (E10b, E11, E11b, E12, E13, E14, E15, E16) — 2026-10-24

- Pre-registered experiments run on one Tesla T4 with Hugging Face data and models (Qwen2.5-0.5B, Qwen2.5-Coder-7B in fp16 and Gemma-4-E4B in bf16 on a 16 GB card): E11 refutes P11a on Qwen2.5-0.5B (one rate-matched cell, R = 4) and Gemma-4-E4B and P11b on Gemma, while the Qwen2.5-Coder-7B row is INVALID (its negative control failed by -0.47 nats) and its verdicts are withdrawn; E12 supports P12-same and P12-cross; E10b at 0.5B is invalid by its positive control (descriptive 0.869 distortion ratio only). Report: `docs/benchmarks/T4_10h_campaign_report.md`.
- New code: `hivemind.compress.kv_lattice` (GPU ℤ⁸ / A₂ / E₈ codecs, bit-identical to the reference decoders), `hivemind.core.kv_page_codecs` (pluggable KV page codec, FP8 default unchanged), `hivemind.boinc.validator_assignment` (architecture-matched spot-check validators), `ComputeVisibleSparseLoCoCompressor(value_codec=...)`, experiment runners E10b / E11 / E12, the E11b aggregator and a resumable GPU job queue with NVML sampling.
- Fix: `hivemind.core.quantization` imports `peft` lazily, so the spot-check replay path and its tests run without peft.
- Extension queue (same day, one T4, all pre-registered; no peer review of record, PTS / $HIVE non-monetary): E13 (localhost swarm over real HTTP: S1 equals S0, wire cost 0.09 of raw; viability P13b refuted on Qwen2.5-0.5B and Granite-1B), E13d (six domains, six workers: P13b inconclusive), E14 (fair quorum / deferred carry: H14a and H14b supported on dense Qwen2.5-0.5B, inconclusive on Granite-3.1-1B; exploratory), E15 (GWAYA-gated RFT of Qwen2.5-Coder-1.5B: inference loop +7.0 points supported, RFT +0.8 points inconclusive, memory cap supported) and E16 (GWAYA-Lite tiers on the T4: PR #21 P1/P2 measured). Statuses are in `artifacts/predictions_ledger.json` (P13-P16, computed for E14/E15 by `scripts/analyze_E14_E15_hypotheses.py`) and in the regenerated `docs/benchmarks/T4_10h_campaign_report.md`.
- New code: `scripts/gwaya_lite.py` (GWAYA-Lite assistant, small-first ladder behind the fail-closed gate, localhost OpenAI-compatible gateway; deployed loopback-only on the T4, guide `docs/community/GWAYA_LITE_ASSISTANT.md`), the Swarm Lab (`scripts/swarm_lab.py`, static site `docs/site/` with `data.js` generated by `scripts/build_site_data.py`), `scripts/checkpoint_state.py` (Spot-VM checkpointing and resume), `scripts/publish_campaign_to_gcs.py` (local staging of the campaign package; the public upload is pending the owner's permission and has not been done), E13 swarm library / runners, E14 admission policies (opt-in), E15 / E16 runners and gate.
- Limits: peers are threads sharing one model where memory required it; the E15 / E16 gate is the vendored GWAYA auditor plus our namespace sandbox, not GWAYA's own oracle; E11b is partial (only the fineweb-edu configurations ran); no WAN claim anywhere.

## v2.10.1 — 2026-10-24 — OpenRouter Google Flow: CLI integration, UI showcase & e2e verification

- `scripts/hivemind_connect.py`: added `register-google` and `login-google` subcommands supporting direct Google ID token (`--token`), automatic `gcloud auth print-identity-token --include-email` token generation (`--gcloud`), and browser-directed registration (`--browser`).
- `src/hivemind/boinc/hivemind_at_home_dashboard.py` & `HiveMindatHome.github.io`: added Screen 02C showcase card for the OpenRouter and Google Sign-in gateway, featuring 5 capability tiers (`coder-lite`, `coder`, `coder-plus`, `moe-general`, `moe-coder`) and direct link to `/signup`, localized across 5 languages (`en`, `fr`, `zh`, `es`, `de`).
- `scripts/test_router_google_flow.py`: end-to-end verification suite covering `/signup` CSP, RS256 token verification, anti-replay, key rotation, negative controls, anti-squatting handle protection, and CLI connector. 10/10 checks passing with cryptographic receipt SHA-256 `78631759c841adfa1c847af172cc47229e276a0bcf1b2fa0c75bdba926b6baad`.
- GWAYA zero-stub audit passing 110/110 files.

## v2.10.0 — 2026-10-24 — Easy registration: Sign in with Google

- `hivemind.router.google_auth`: ID-token verification with the standard library plus `cryptography` only. RS256 is pinned (no `none`, no HS256-with-public-key confusion), keys come only from Google's published set (cached, refetched at most once a minute on an unknown key id), audience must be our client id, issuer Google, token fresh and unexpired, e-mail verified, and each token works once. 33 tests, one negative control per rule; three reintroduced bugs (no audience check, any algorithm, no signature check) each made a test fail.
- `POST /v1/hivemind/register/google` and the `/signup` page (strict content-security policy with a per-request nonce, text written with `textContent` only). The store keeps a keyed hash of Google's `sub`, never the e-mail; a returning user keeps account, balance and role and gets a fresh key (old key retired). `google.` handles are reserved against squatting. Optional `HIVEMIND_OPEN_REGISTRATION=0` closes handle-only registration.
- Off until the operator creates an OAuth web client in the Google Cloud console and sets `HIVEMIND_GOOGLE_CLIENT_ID` (steps in `docs/HIVEMIND_OPENROUTER.md`). Not exercised against real Google yet: tests use a fake key set.
- Also: `uv.lock` refreshed for the dev extra (PR #143).

## v2.9.0 — 2026-10-24 — Open-Router for Python and game coders: tiers, GPU presets, producer shield, editor connector

- Capability tiers (`hivemind.router.catalog`, served publicly at `/v1/hivemind/catalog`): `coder-lite` (3B), `coder` (7B), `coder-plus` (14B) and two mixture-of-experts tiers, `moe-general` (gpt-oss 20B, 16 GB) and `moe-coder` (Qwen3-Coder 30B, 24 GB). One market per tier keeps liquidity deep; the tier id is a claim until `HM-S13-13`.
- Producer agent: `--suggest --vram-gb/--detect-vram`, `--preset TIER`, preflight that refuses to advertise a model the endpoint does not list, optional endpoint token. The gateway stores the token and presents it as a bearer secret to the producer's own server (additive column; databases created earlier are upgraded in place, tested on an old-schema file).
- Producer shield (`scripts/hivemind_producer_shield.py`): only `GET /v1/models` and `POST /v1/chat/completions`, constant-time token check, body and concurrency caps, optional model pin, and a fallback that lists models for older Ollama releases. The live run found that fallback was needed: this host's Ollama has no `/v1/models`.
- Connector (`scripts/hivemind_connect.py`): register (key printed once or saved owner-only), `check`, and editor snippets for Continue, Cline, Aider, OpenCode, Zed, JetBrains, Python and environment variables.
- Guide `docs/HIVEMIND_PYTHON_GAME_DEVS.md` with game-development recipes and an honest list of what does not exist yet. ROADMAP §3.11 gains `HM-S13-09..14`.
- Live end-to-end on this host (`scripts/router_gamer_e2e.py`): connector, gateway, shield and real Ollama; negative controls refuse a request without the token and Ollama's management API.

## v2.8.0 — 2026-10-24 — Operation Open-Router: an OpenAI-compatible MoE gateway with a producer/consumer equilibrium, deployed on GCP

- `hivemind.router`: `market` (Kelly consumer game with a price independent of supply, producer dominant-strategy supply, posted-price fallback, a win-win certificate), `accounts` (HMAC-hashed API keys, hash-chained HMT ledger, escrow holds, conservation invariant), `gateway` (`/v1/chat/completions` incl. streaming, `/v1/models`, registration, producer heartbeat with SSRF guard, public market certificate). Works with any OpenAI client; IDE setup for OpenCode, VS Code (Continue) and generic clients in `docs/HIVEMIND_OPENROUTER.md`.
- Producer daemon `scripts/hivemind_producer_agent.py` (refuses a non-https gateway so the key is never sent in clear).
- Adversarial security review found 5 high and 6 medium issues (sybil bids setting the price, double-spend of one balance, conflicting token limits, unbounded billing and response size, `n`/`best_of` multiplication, capacity weighting, unauthenticated costly market endpoint, body limit after parse, rate limit behind a proxy, empty-choice billing); all fixed with regression tests. Low findings fixed: NAT64/6to4/Teredo/IPv4-compatible SSRF bypass, unmetered request fields (allow-list), producer-agent scheme check, single instance.
- 144 tests; live end-to-end against local Ollama through the official `openai` client. Public Cloud Run service with a Litestream-replicated ledger in a private versioned GCS bucket; account and balance verified to survive a revision rollout.
- ROADMAP §3.11 Season 13 (`HM-S13-01..08`). HMT are points with no cash value; producers can read the prompts they serve; output correctness is not yet verified (`HM-S13-05`).


## v2.7.0 — 2026-10-24 — Operation Fair-Market: BOINC computing coin and Nash-equilibrium compute market (proposal + code + simulation)

- `hivemind.economy.compute_coin`: BOINC cobblestone credit granted only for verified work (spot-check + Multi-Krum; class-peak and canonical-replica caps); work-backed $HIVE issuance with geometric decay under the 21 M cap and an 80/15/5 fee split with burn; Kelly consumer equilibrium (efficiency ≥ 0.75 by the 4/3 price-of-anarchy bound), Tullock worker equilibrium (closed form, ε-Nash certified), damped best-response routing across queues with a deviation certificate. 12 tests, 5 property-based.
- Pre-registered simulation (seed 2026): Kelly efficiency min 0.800; 300/300 Tullock certified; routing 55/60 undamped (hypothesis refuted, run kept) → 60/60 damped; controls passed. Status: proposed, not adopted (`HM-S12-02`).
- ROADMAP §3.10 Season 12 (`HM-S12-01..06`); proposal `docs/COMPUTE_COIN_AND_MARKET.md`.


## v2.6.0 — 2026-10-24 — Operation Hardened-Hive: HiveMindFirewall, chain of command, scripted red team

- `hivemind.security.hivemind_firewall`: keyed-hash identities + Ed25519 delegation certificates rooted at the owner; signed, sequenced, hash-chained orders with nonce burning; deterministic prompt-injection scan (PI-01..07, decoded payloads re-scanned) and data envelopes; pure state machine with QUARANTINED as the default for anything undefined; allow-listed argv sandbox (bubblewrap or BLOCKED); hash-chained audit log. 17 tests (two Hypothesis properties).
- Scripted red team `scripts/run_firewall_pentest_campaign.py` over 59 pre-registered vectors: first run 47/49 attacks blocked (two real defects: percent-encoding threshold, owner-delegation rule order), second run 49/49, 0 false positives, controls passed.
- Gatekeeper records a report-only injection scan of PR text and the GitHub signature-verification status of every commit; policy switch to require signed owner commits. ROADMAP §3.9 Season 11 (`HM-S11-01..08`, campaigns `HM-S11-PT-1..3`); SECURITY.md §5; design in `docs/SECURITY_HARDENING_PHASE.md`.


## v2.5.2 — 2026-10-24 — preprint v3.7.2 (not yet deposited): Lean gate at 128; T4 branch review; main locked to PRs

- Independent Lean gate re-run on the 128-theorem `HiveMindSpec.lean` that reached `main` by direct push (4474802): 128/128, 0 sorry, 0 untrusted axioms; Theorems 125 and 127 flagged as not constraining their mechanism (`HM-S10-09`). Paper inventory table, limitations and README updated to 128.
- Scientific lead's review of the T4 experiment branch before its PR: `docs/review/t4_branch_review_2026-10-24.md`.
- Governance: `main` requires a pull request and enforces admins (PR #125 adds the gatekeeper's merge retry); the shared checkout's git e-mail reset to the coordinator's address.


## v2.5.1 — 2026-10-24 — preprint v3.7.1: E12 pilot complete (bf16 cell)

- E12-pilot P12d supported: bf16 autocast replayed by an fp32 validator diverges to 3.3e-3 with 96/96 tensors above ε, 2.7× the tampered control; dtype is a class boundary (validators assigned within dtype class; bf16 ε to be measured on GPU, `HM-S10-08`). PROV-34; ledger P12d; paper §4.1, §E12 and limitations updated. No other numbers changed.


## v2.5.0 — 2026-10-23 — preprint v3.7.0: E12 replay-divergence pilot; hourly PR gatekeeper

- E12-pilot (pre-registered): the Byzantine spot-check's ε = 1e-4 replay rule fails on CPU for a thread-count change alone (2.6e-4) and for a second host (1.5e-4); bit-identical execution reproduces exactly; tampered control 1.25e-3 with 96/96 tensors above ε. Rule re-specified as execution-class matched with a per-class ε and a tensor-count term. bf16 cell pending at build time.
- `scripts/pr_gatekeeper.py` (hourly, owner-only merge authority; PR text is data); branch protection on `main`.



## v2.4.0 — 2026-10-23 — preprint v3.6.0 after an external review of v3.5.0

- Review recorded verbatim (`paper/review/external_review_v3.5.0_2026-10-23.md`, provenance unstated, no score) with a point-by-point response (`response_to_external_review_v3.5.0.md`).
- Paper v3.6.0: architecture figure (worker inner step; outer WAN round), hardware-heterogeneity caveat and architecture-matched rule for the ε = 1e-4 spot-check (new E12 / RQ11), compounding-approximations and PoC-scale limitations, Skip-Token-as-dropout note, PA-CGAD-under-quorum note. Numbers unchanged.
- Kept in the manuscript, on purpose: the correction that versions 2.9–3.1 labelled a scripted audit a "peer review" (the error is on public records; so is its correction).
- Zenodo: new version of the concept record; Lean: `lean4/mathlib/RDAMultiplier.lean` (3 Mathlib theorems, independent gate).

