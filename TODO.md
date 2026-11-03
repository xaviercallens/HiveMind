<!-- Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0. -->
# TODO: what is still open

Generated 2026-11-03 from the repository's open GitHub issues and pull requests, not from memory. Nothing here claims that something works; it lists what is not finished. Re-generate it instead of hand-editing the counts.

Open issues: 116. Open pull requests: 24 (2 drafts).

Rules for anyone taking an item: AGPL header on every file under src/hivemind; no stub bodies and no mock_, simulate_ or dummy_ identifiers outside tests/; a new test must fail on unfixed main before it passes; no fabricated numbers; merges are owner-only through the hourly gatekeeper.

What is real and what is simulated across the codebase is audited in [docs/MVP_REALITY_AUDIT.md](docs/MVP_REALITY_AUDIT.md); the smoke path is in [docs/MVP_E2E_SMOKE.md](docs/MVP_E2E_SMOKE.md). Per-row status lives in the dated docs/IMPROVEMENT_PLAN_*.md files.

## P0: security or safety, fix first (2)

- [ ] issue 493: [P0 Security] Cockpit steam/register-login rotates an existing handle's key without auth; contribute-boinc mints 250 HMT per anonymous call (IP-141, IP-142)
- [ ] issue 492: [P0 Security] Router API key committed as DEFAULT_FOUNDER_API_KEY in the swarm co-pilot (IP-139)

## P1: a claim the code does not back, or a verifier that cannot fail (13)

- [ ] issue 522: [P1 Security] /api/astro/gemini-tts is an unauthenticated open proxy with an unbounded on-disk cache and the API key in the query string (IP-169)
- [ ] issue 521: [P1 Privacy] /api/v1/monitor/telemetry is unauthenticated and returns every end-user's IP address, user agent and latest paths (IP-168)
- [ ] issue 520: [P1 Swarm] push_bundle_to_hf_hub reports status: pushed after upload_folder raised (IP-165)
- [ ] issue 519: [P1 Swarm] deposit_qlora_expert quality gate is NaN-blind and verified=True relabels caller-supplied NLLs (IP-163)
- [ ] issue 512: [P1 Security] Scanner manufactures 'email verified' and the owner e-mail when author_email is omitted, exempting the security-rule files from the scans (IP-172)
- [ ] issue 511: [P1 Security/CI] The PR security gate runs the scanner from the PR's own checkout, so a PR editing pr_security_scanner.py gates itself (IP-162)
- [ ] issue 510: [P1 Lean/CI] PROV-31 'Mathlib RDA theorems verified' is never compiled: lake has no Mathlib dependency and the pipeline trusts the committed receipt (IP-161)
- [ ] issue 501: [P1 Swarm] Verifiers that cannot fail: VRAM guard fails open, ZK-OPoI norms outside the HMAC and NaN-blind, surrogate 'VERIFIED' from the teacher's drift, scripted supervisors (IP-147, IP-149 to IP-152)
- [ ] issue 500: [P1 BOINC] Credit cap disabled by a NaN wall_seconds and set by the volunteer's own accelerator class (IP-159, IP-160)
- [ ] issue 499: [P1 Game] Points and HIVE minted from self-reports: guild-tournament overrides, forge-lora, solve-bounty, webgpu contribute, Sybil gate (IP-153 to IP-156)
- [ ] issue 496: [P1 Router] /v1/hivemind/live-telemetry serves literal T4 figures under LIVE_GCP_PRODUCTION and any producer bearer can crash every page with one telemetry push (IP-143, IP-145)
- [ ] issue 495: [P1 Privacy] scripts/checkpoint_state.py pushes the owner's private Claude memory notes to the public repository (IP-140)
- [ ] issue 494: [P1 Router] Reconnect keepalive manufactures producer liveness and overrides real heartbeats (IP-144; root cause of the 2026-11-02 incident)

## P2: correctness and CI (2)

- [ ] issue 523: [P2 CI/tests] Collection-time CUDA_VISIBLE_DEVICES='' in one test module hides the GPU from every test collected afterwards (IP-179)
- [ ] issue 513: [P2 Router] Service-state literals: OpenTrainer ONLINE with zero nodes, /boinc/status reports the remote Ruche ONLINE without a probe, manifesto defaults to ONLINE (IP-177)

## Incidents (1)

- [ ] issue 487: [Incident] RTX 2070 failover orchestrator re-pointed all 42 producers to a dead tunnel while the T4 was reachable (2026-11-02 06:28Z onward)

## Other open issues (honesty, experiments, UX, federation) (98)

- [ ] issue 509: [T4 owner] Daytime QLoRA starved by scripted chat traffic: 161 assimilated since 06:22Z while ~1 scripted chat/min keeps the serving-first governor from ever idling 1800 s
- [ ] issue 483: [Security/Honesty] admin/producers/reconnect: committed fallback endpoint token literal and fabricated 'loaded' residency for 14 models
- [ ] issue 481: [Honesty] Foundation-paper portal: E20 block reports a literal VERIFIED_PASS and the portal HTML ignores its payload (IP-94, part 2)
- [ ] issue 473: [Safety] GPU circuit breaker fails open: unreadable sensor reads 48 C and any override skips the real temperature poll (IP-86)
- [ ] issue 467: [Honesty] Ruche community feed and cockpit carry fabricated swarm telemetry (48,920 nodes, 542.8 PFLOPS, Epoch #412, invented discoveries)
- [ ] issue 464: [Honesty/Security] Ruche dashboard: invented BOINC daemon counters and a hard-coded account registry with token-like literals
- [ ] issue 453: [Join UX] /join and /rejoindre: show the trainer path and the live status table
- [ ] issue 452: [Paper] Foundation paper: add the H-Net assessment and the HN1 boundary-signal result (after #446 is fixed)
- [ ] issue 451: [Night-003] Prepare the first federated night: overlay at boot, UDP 51820, learner keys, VRAM floor
- [ ] issue 448: [Experiment] H-Net probe on the T4 for CrowdGPT: schedule with the core-node session
- [ ] issue 440: [Honesty] Life-as-a-Bee page shows a 1.13 s yield and fixed speeds as live results, with no source
- [ ] issue 438: [Federation] Point the RTX 2070 plan's cross-host step at the AMCP learner once #410 lands
- [ ] issue 437: [Docs] /join, /signup and README disagree on starting points (0, 100, +100)
- [ ] issue 436: [Honesty] La Ruche copy, wallet and leaderboard still show fixed +250 points and hard-coded totals
- [ ] issue 431: Announcement: co-train over the AMCP bus (Federation v2) — signed task agents, cross-machine validated; testers welcome
- [ ] issue 425: Personal contact data committed in the outreach script (IP-125)
- [ ] issue 424: L2 state channel settle is not idempotent and never verifies vouchers (IP-109)
- [ ] issue 423: TURN relay forwards to private/loopback peers and has no allocation quota (IP-110)
- [ ] issue 422: Community orchestrator auto-merges after scanning a synthetic patch (IP-72)
- [ ] issue 421: Gatekeeper executes PR code before the owner check and merges without pinning the head (IP-73, IP-119)
- [ ] issue 415: [Quality] Clear the 7 stub-audit failures on main (they make gwaya_audit refuse #410, #362, #344)
- [ ] issue 412: [Security] SEC-02 flags base64 weight payloads read with weights_only=True (blocks #401 and #410)
- [ ] issue 405: [DiLoCo] Stage 2: real two-host LoRA delta round (after #401; no Multi-Krum control at K=2)
- [ ] issue 404: [Perf] Router chat p95 of 28-57 s at night versus the 2026-10-25 idle baseline: find the cause
- [ ] issue 403: [Honesty] Launch-model baseline of 59 accounts is a constant labelled REAL_LIVE telemetry
- [ ] issue 394: [Core] Load-balancing auxiliary loss is detached, so the router gets no balancing gradient
- [ ] issue 392: [Security] Remove default HMAC and state-channel secrets from source
- [ ] issue 389: [T4 owner] Run the approved Qwen3.5 smoke test; reconcile the QLoRA and VRAM counters
- [ ] issue 387: [Security] Security scanner: run from the base ref; fix CB-02 false positives and evasions
- [ ] issue 386: [Economy] Community bundle import: verify before crediting; forbid self-import; pay each bundle once
- [ ] issue 385: [Core] Multi-Krum: derive the reference schema from the round, not from the first worker
- [ ] issue 382: [Ledger] Reload persisted state on open; write atomically; reject a corrupt file
- [ ] issue 381: [Security] BOINC validator: bind uploads to their workunit; credit from server-side time
- [ ] issue 380: [Security] Dashboard admin routes need authentication; overlays use an allow-list
- [ ] issue 379: [Security] Router: credit only server-verified work; contribute endpoints require auth
- [ ] issue 378: [Security] Router: remove the unauthenticated owner-key fallback and the account takeover paths
- [ ] issue 367: [Collaboration Hub] Unified HiveMind × AMCP × SocrateAI × OptimalEnergyAI × GWAYA Ecosystem & x* Fork Bridges
- [ ] issue 339: CrowdGPT collaboration follow-up
- [ ] issue 338: [International Launch Weekend · Oct 30 – Nov 1, 2026] Global 10,000-Node Scalability Plan, 4 Grand Challenges & Nocturnal Image-MoE Evolution Monitor (Humans & A2A/MCP Agents)
- [ ] issue 319: [HiveMinders Guide] Season 20 Roadmap Progress & Community Contribution Opportunities (S22-S23)
- [ ] issue 273: [Epic] Solver Bee: certified numeric experts and the first natural-language science bee (Wave 1)
- [ ] issue 272: [-AGENTS-RTX2070-T4] Owner Agent Fleet Plan: Local RTX 2070 (8 GB) + GCP Tesla T4 (16 GB / Multi-T4) Co-Serving & Nocturnal Training
- [ ] issue 269: [-W1-9] Newcomer quickstart for Solver Bee + Stretch W1b NL-to-Spec Coder Translator
- [ ] issue 268: [-W1-8] Bee scoring for solver bees (Wilson lower bound & bee_energy.py)
- [ ] issue 267: [-W1-7] Pre-registered S-2: fault injection against the numeric verifier
- [ ] issue 266: [-W1-6] Pre-registered S-1: cross-CPU reproducibility of solver results (RTX 2070 host vs GCP T4 host)
- [ ] issue 265: [-W1-5] solver_bee loopback service with an OpenAI-compatible endpoint
- [ ] issue 264: [-W1-4] E15 sandbox: run a solver binary under resource limits
- [ ] issue 263: [-W1-3] Independent verifier and the VERIFIED / AUDITED / FAILED / UNVERIFIED vocabulary for numeric results
- [ ] issue 262: [-W1-2] Problem library with independent references and invariants (12 IVPs + 4-domain rescue pack)
- [ ] issue 261: [-W1-1] hivemind-solver: pinned Rust CLI around rusty-SUNDIALS (JSON in, certificate out)
- [ ] issue 251: Help wanted: idle GPUs as HiveMind producers, and testers for multi-producer routing
- [ ] issue 247: [-INTL] Mathematical, Social & Economic Model & 4-Stage Rollout for the International English Launch
- [ ] issue 244: [-Q35-06] Day 6-7: Developer Adoption & BYOB Onboarding Campaign (VS Code / Cursor Preset, Portal & CrowdGPT Bridge)
- [ ] issue 243: [-Q35-05] Day 5: Speculative Decoding (Qwen3.5-Coder-3B Draft -> 14B/30B Target) & Execution-Verified auto-coder Routing
- [ ] issue 242: [-Q35-04] Day 4: Big-Producer Teacher-to-Student Distillation (Qwen3.5-30B-A3B / 14B -> 9B/7B SVD Rank-16 Projection)
- [ ] issue 241: [-Q35-03] Day 3: Daytime Micro-Idle Harvester (>=30s Idle, <1.0s Preemption) & Nocturnal Deep Evolution (22:00-07:00 CET)
- [ ] issue 240: [-Q35-02] Day 2: Usage-Driven 13-Bee MoE Workunit Scheduler (65% Coding / 25% Science / 10% BYOB)
- [ ] issue 239: [-Q35-01] Day 1: Hardware-Stratified Producer Registry (std_8gb, big_12gb, xl_16gb_plus) & Qwen 3.5 Profiles
- [ ] issue 238: [-Q35] Epic: Operation Qwen 3.5 Coding Mastery, Big-GPU Producer Tiers & Usage-Driven Idle/Nocturnal MoE
- [ ] issue 213: [-AE-05] Warm verifier process; its energies become the bandit reward for #173
- [ ] issue 212: [-AE-04] Promotion gate and versioned Bee registry with rollback (Phase 3 port)
- [ ] issue 211: [-AE-03] Energy-gap pairs (D6) as training data for #166, opt-in users only
- [ ] issue 210: [-AE-02] BOINC chain: disjoint reported split and a pre-registered plateau rule
- [ ] issue 209: [-AE-01] Bee energy function in the #191 sandbox: hidden tests first, bounded cost second
- [ ] issue 208: [Epic] Season 18 Operation Energy-Bees (-AE-01..05): measured Bee energy and a self-evolution gate
- [ ] issue 204: [-CW] Operation CrowdSwarm: Joint Federation with CrowdGPT (Vxtzq/Crowd-v1 1B Trunk, Multi-Krum Coordinator & Multi-Vendor ROCm/DirectML Installer)
- [ ] issue 197: [] Ledger: one Litestream writer, enforced
- [ ] issue 196: [] Dashboard source: remove template completions and browser-side point credits; restore /night-moe
- [ ] issue 195: [] Public status page and alerting from the live monitor
- [ ] issue 194: [] Test and probe accounts must not move the market or the user counts
- [ ] issue 193: [] T4 residency: route to loaded models and report them in heartbeats
- [ ] issue 192: [] Remote control must act on producers, not only be stored
- [ ] issue 191: [] Measure answer quality on open prompts, not only liveness
- [ ] issue 190: [] Market: do not price new users out when capacity is idle
- [ ] issue 189: [] Post-deploy contract check gates every traffic switch
- [ ] issue 188: [] Deploy the three public services from a tagged main commit only
- [ ] issue 185: Ruche and Tribute: show HiveMind Reflexion Points only (no EUR/USD), and carry the router fixes onto the deploy branch
- [ ] issue 180: Ruche dashboard: /v1/chat/completions must not return template text
- [ ] issue 176: [] QueenBee: exact LoRA aggregation and a privacy accountant for shared deltas
- [ ] issue 175: [] QueenBee: verified correction pairs from developers, local adapter
- [ ] issue 174: [] QueenBee: split a task into verified subtasks and route each
- [ ] issue 173: [] QueenBee: bandit routing among the served experts on real tasks
- [ ] issue 172: [] QueenBee: scouts that a sybil cannot corrupt alone
- [ ] issue 171: [] QueenBee: bound dance influence by evidence volume, not claimed level
- [ ] issue 170: [] Run the verified update gate on the stars adapter
- [ ] issue 169: [] Cyber expert on verified targets, or a code expert with executable tests
- [ ] issue 168: [] Full interference grid for the night adapters
- [ ] issue 167: [] Test the routed Bee MoE as one system
- [ ] issue 166: [] Train experts on verified self-solutions instead of gold text
- [ ] issue 165: [] Night-MoE adapter decisions: adapter-vs-control paired test as the primary rule
- [ ] issue 130: [Season 10 Community Model Hub & Dashboard: La Ruche + HiveMind@home] Beehive MoE Vault, SETI-Style Spectrum Analyzer & GCP Cloud Command Deck
- [ ] issue 129: [Season 10 Dataset Mining Campaign: CAMP-S10-RUCHE-DATASET-MINING] 8-Domain Scientific Corpus Curation, Heartbeat CAS Leases & QLoRA-to-La-Ruche Training
- [ ] issue 123: [Season 9/10 GPU/TPU WAN GameDay: ..12, ] Multi-Host WAN Pilot, Live HF Multi-MoE & Heterogeneous GPU/TPU Swarm Campaign
- [ ] issue 122: [Season 10 GPU Campaign 4: CAMP-S10-E12-HETERO-SPOTCHECK-WAN-GPU] Cross-GPU Two-Statistic Spot-Check & Q16 Digest (), CAS Leasing () & Factored Gossip () (+3,500 PTS Symbolic Compute Credits)
- [ ] issue 121: [Season 10 GPU Campaign 3: CAMP-S10-E11-LATTICE-KV-WIRE-GPU] Conway-Sloane E8/A2 Hexagonal Lattice Wire () & 5-Codec Paged KV-Cache () (+3,500 PTS Symbolic Compute Credits)
- [ ] issue 120: [Season 9 GPU Campaign 2: CAMP-S9-E7-SIGMA-LOOPED-MOE-GPU] FoMoE Sigma-MoE + Peri-LN () & Looped OrthoInj Terminal-KV () (+3,500 PTS Symbolic Compute Credits)
- [ ] issue 119: [Season 9 GPU Campaign 1: CAMP-S9-E5-QUORUM-RDA-GPU] Decoupled Quorum (K < M), Async Stragglers () & RDA Delta-Cosine () (+3,500 PTS Symbolic Compute Credits)

## Open pull requests

Owner pull requests are merged by the hourly gatekeeper when every check passes; the rest are held.

- PR 549 by xaviercallens: docs(roadmap): 18:30 CET addendum — eight plan groups implemented and verified locally, fourteen branches queu
- PR 541 by xaviercallens: cockpit: IP-200/202/204 feedback, discovery award, benchmark and TAP cache
- PR 539 by xaviercallens: Merge remote-tracking branch 'origin/main' into fix/ip179-no-collection-time-cuda-visible-devices
- PR 538 by xaviercallens: fix(router): the Ruche probe reports ONLINE only for a 2xx answer, never for a 404 or a redirect (IP-213)
- PR 536 by xaviercallens: fix(router): a producer that reports the model unloaded does not take part in the equilibrium while a warm one
- PR 533 by pedromineo: fix: [P1 Game] Points and HIVE minted from self-reports: guild-tournament overrides, forge-lora, solve-bounty,
- PR 530 by app/dependabot: build(deps): bump transformers from 5.20.2 to 5.22.1
- PR 529 by app/dependabot: build(deps): bump torch from 2.14.1 to 2.15.0
- PR 528 by app/dependabot: build(deps): bump leanprover/lean-action from 1.6.2 to 1.7.0
- PR 485 by xaviercallens: docs: Gen 6 verified-swarm design (router v2, expert registry, receipts, staged preregistered plan)
- PR 446 by xaviercallens: docs(paper,todo): update foundation paper to v4.3.0 (Wave 3 15-MoE, Two-Host E8 Stage 1, Audit Reconciliations
- PR 442 by xaviercallens: docs: roadmap round 2026-11-01 afternoon
- PR 434 by xaviercallens: fix(swarm): harden L2 state channel settlement, TURN relay SSRF/quotas, and atomic ledger (#424, #423, #382)
- PR 410 by xaviercallens: feat(federation): signed task agents over AMCP and Kafka (Aglet life cycle, no mobile code), co-training round
- PR 397 (draft) by xaviercallens: T4 audit: governor telemetry and report, validator lockout finding, two owner-review changes (nothing deployed
- PR 396 by xaviercallens: docs(night): measured NIGHT-INTENSE-1 feasibility results (draft, unreviewed)
- PR 369 by xaviercallens: chore(lean4): resolve unused hypothesis warning in Theorem 120 and silence pynvml FutureWarning
- PR 366 by xaviercallens: fix(boinc): port thermodynamic ratchet, mixed-domain 80/20 replay, and cosine LR to BOINC QLoRA MoE
- PR 362 by xaviercallens: feat(s29): modularize international launch forecast views & Foundation Paper v4.2.0
- PR 344 by xaviercallens: feat(router,webgpu,s26): fix /webgpu 500 without torch, 5-lang OpenRouter i18n & TensorSharp (@zhongkaifu) Co-
- PR 219 by app/dependabot: chore(deps): bump actions/upload-artifact from 4 to 7
- PR 218 by app/dependabot: build(deps): bump actions/setup-python from 5 to 7
- PR 163 (draft) by xaviercallens: Propose ALIXQueenBee: a planner queen, dance records and local learning from developer feedback
- PR 141 by app/dependabot: chore(deps): bump setuptools from 84.0.0 to 84.1.0
