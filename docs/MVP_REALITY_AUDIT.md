# MVP reality audit: what is real, what is simulated, and the path to a real MVP

> **Proof of concept (PoC):** HiveMind is not a production service yet. Send only public, non-sensitive data.

**Date:** 2026-11-03. **Method:** every `src/hivemind` file (191 files, about 136k lines) was classified by a low-tier model agent per chunk of about
6,000 lines (real / partial / simulated / scaffold / dead); a stronger agent then tried to refute each flagged item by reading the code (160 items
confirmed); a separate agent traced the real model-serving and training path end to end. Nothing was run against production. The numbers below are
agent findings: re-check a line reference before acting on it.

**Verdicts (191 files):** real 114, partial 61, simulated 13, scaffold 1, dead 2.

## Status of this plan on branch `hardening/readiness-round4`

| Plan item | Status |
| :-- | :-- |
| A. End-to-end proof with a real model | **Done**: `scripts/mvp_e2e_smoke.py`, 15 checks, `docs/MVP_E2E_SMOKE.md` |
| B1 / C1. Honest benchmarks page | **Done**: measured results only (`router/measured_results.json`), no vendor comparison, unmeasured tiers listed |
| B3. Dashboard `train-expert` returned invented metrics | **Done**: 202 `QUEUED_NOT_STARTED`, no metrics; model select records a selection only; seeded fake active model removed |
| B6. Browser pages minting keys / posting a fake RTX 2070 contribution on load | **Done** (WebGPU and Stitch pages); guard tests |
| B14. OpenTrainer stored 6400/8192 MiB and an RTX 2070 as readings | **Done**: stored as "not reported", token count flagged as an estimate |
| C3. Catalog `kind` (base / adapter / alias) | **Done**: computed from the tags, exposed in `/v1/hivemind/catalog` |
| Honest catalog entries (26 unpublished Hugging Face repos) | **Done** earlier (PR #532 / v2.16.1) |
| B2, B4, B5, B7-B13, B15 and the "lower priority" list | Open (see below) |
| C4. One source for agent and shield (`scripts/` vs `src/` copies differ) | Open |
| C5. Gateway canary prompt and tokenizer cross-check of reported usage | Open |
| C8. Training-round table and rollback ratchet | Open |
| C9. Time-to-first-token and tokens/s in settlement | Open |
| C10. Close the train-to-serve loop (BOINC adapter -> GGUF -> evaluated -> served tier) | **Pipeline done** (`scripts/adapter_to_served.py`, `docs/TRAIN_TO_SERVE_GATE.md`); the current adapter **fails the gate** (no measurable gain on the served files), so nothing is served from it |
| C11. Run the smoke test nightly in CI | Open: needs a tiny GGUF in CI |

## The plan, as produced by the audit

Source: the supplied audit data only. Verdict counts are real 114, partial 61, simulated 13, dead 2, scaffold 1.

## (A) What is real today (honest claims)

- **Routing and metering.** A request goes through a real gateway to a volunteer producer running an open model (Granite MoE 3B, Qwen, Gemma, Mistral). The producer sits behind an authenticating shield (`router/gateway.py:1084-1294`, `router/hivemind_producer_shield.py:53-164`). The gateway escrows the consumer's points, takes token usage from the producer's reply, caps it by bytes and `max_tokens` (`:149-163`), and settles into a hash-chained points ledger. The ledger has no cash value. Failures release the hold (`:1248-1254`).
- **End-to-end smoke test.** `scripts/mvp_e2e_smoke.py` runs a GGUF model on CPU through shield, gateway and ledger, with a negative control (commit 78fcc80). It is not in CI.
- **Latency telemetry.** The gateway records per-request latency and computes p50/p95 (`gateway.py:1208-1230`, `accounts.py:1612-1638`). It is end-to-end time, not time-to-first-token.
- **Small-scale training.** BOINC QLoRA workunits train `granite-3.1-1b-a400m-base` with PEFT LoRA for 80 inner steps. They use cached real datasets and measure per-domain held-out loss before and after (`boinc/real_client/moe_app.py:41-155`, `moe_work.py`). Workunits chain from `latest_adapter.pt` (`scripts/t4_workgen.py`).
- **One measured expert.** `bee-stars` was trained by `e17_train`, converted to a GGUF LoRA and served with llama.cpp.
  - It scored 283/317 = 0.893 (Wilson 0.854-0.922).
  - It got 148 items right that the base model missed, against 20 only the base got right.
  - Through the router it scored 282/317, against 283 direct.
  - A shuffled-label control agreed on only 104/317, so the check can fail (`artifacts/experiments/BEE_STARS_serving_check.json`).
  - It is a narrow classification pack, not general ability.
- **Protocol demo.** The E8 pilot is a real two-host DiLoCo run on a toy LM (`scripts/run_experiment_E8_pilot.py`).
- **Genuine algorithms.** Multi-Krum, lattice and EF21 compression, and `core/diloco_inner.py` are real.
- **Honesty fixes already in.** `catalog._truthful()` marks unpublished repos as not trained (`router/catalog.py:1229-1245`). `telemetry_figures.py:21-30` returns "n/a".

## (B) Simulated or misleading: top 15

1. `router/benchmarks_page.py:124-135, 1347, 1357, 1369`. Hard-coded latency, quality scores, "beats frontier 100% vs 64%", and a fake campaign id and timestamp. These are served publicly at `gateway.py:1560-1610`.
   - Action: load only results from `scripts/night_eval.py` artifacts, delete the frontier and coverage claims, and show "not measured" otherwise.
2. `core/dev_image_video_moe_integration_benchmark.py:757, 898-940`. Literal gains (99.07, 19.4, FVD 41.8), a "We trained" Reddit post, and a sha256-jitter stress test reported as 100% HTTP 200.
   - Action: remove the literals, then relabel or archive the module. `router/life_as_a_bee.py:1808-1829` republishes them.
3. `hivemind_at_home_dashboard.py:8025`. `api_admin_train_moe_expert` returns canned "TRAINING_COMPLETED_VERIFIED" metrics.
   - Action: return QUEUED, or NOT_IMPLEMENTED.
4. `hivemind_at_home_dashboard.py:7181, 7132, 6370, 1585, 2792, 2107`. Canned Lean/exoplanet tool results, canned bee replies, invented seed messages, self-test checks that are always `passed: True`, and a literal 128/128 Lean count.
   - Action: return NOT_MEASURED, and count theorems and `sorry` from `lean4/HiveMindSpec.lean`.
5. `game/exoplanet_hunter_cockpit.py:3333, 3631, 3787, 3283`. Synthetic BLS receipts, canned GPU-fold fallback, HMT granted for unverified or self-reported work, and a seeded "verified" feed.
   - Action: set `fold_res = None`, grant only after a real fold, and start the feed empty.
6. `game/webgpu_galaxy_cockpit.py:902, 1005` and `game/stitch_v46_cockpit_template.py:2717`. The client mints `hm-*` keys and a 500 HMT balance, shows "+250 HMT" without checking `gcpRes.ok`, and posts a fake RTX 2070 contribution on page load for every visitor.
   - Action: delete the fabricated paths and the load-time call.
7. `core/auto_improvement.py:1015, 1157` and `scripts/run_nocturnal_moe_auto_improvement.py:280`. RNG-generated McNemar accuracies feed real HMT point grants.
   - Action: require telemetry, add a `simulated` flag, and skip grants for simulated results.
8. `game/exoplanet_hunter_engine.py:2048, 2158, 2133`. Seeded fake users and traffic written to `artifacts/hivemind_traffic_tflops_monitor.json`, constant "measured" CPU TFLOPS, and a fixed 1.85x multiplier.
   - Action: delete the seeding, time a real CPU matmul, and regenerate or delete the artifact.
9. `client/steamworks.py:352-441`, `client/worker_node.py:115`, and `game/rtx_steam_tool_deployer.py:574, 855`. A self-signed HMAC Steam "Sybil check", fake leaderboard receipts, a committed HMAC key, and a tautological gate.
   - Action: relabel as simulated and remove the committed key. Real Sybil resistance needs the Steam Web API.
10. `swarm/pr_first_openrouter_orchestrator.py:59, 149` and `swarm/s34_04_self_evolving_roadmap_supervisor.py:86`. Fabricated OpenRouter request ids and a vacuous `ready_to_merge`.
    - Action: remove the id fields.
11. `economy/international_launch_model.py:1301-1393`, `economy/international_launch_views.py:1205`, and `swarm/international_launch_scaler.py:958-1050`. "LIVE PRODUCTION TELEMETRY" literals, E20 "VERIFIED_PASS", and `gate_passed: True` for waves that never ran.
    - Action: fill from the telemetry endpoint, otherwise show "n/a" or PLANNED. Drop `gate_passed` for future waves.
12. `swarm/abc_moe_agent_mesh.py:358, 436, 445`. A mock web search with real-looking DOIs and IC50 values, placeholder field extraction, and a claimed MoE expert that never ran.
    - Action: make `web_search_fn` required and move the canned data into tests.
13. `swarm/exemplar_byob_bees.py:209` and `swarm/hf_community_hub.py:274, 497`. Hard-coded McNemar values, "APPROVED_LIVE" with no evaluation, "32/32 theorems", and default losses that give a fake 78% reduction.
    - Action: return SIMULATED_NOT_EVALUATED and require real loss inputs.
14. `router/gateway.py:2963-2987`. OpenTrainer stores `inner_steps*512` tokens and a fake 6400/8192 MiB VRAM as if measured.
    - Action: store client-reported values flagged as such, or NULL.
15. The zk-OPoI modules (`swarm/zk_opoi_v5_circuit.py`, `swarm/s36_01_zk_opoi_v5_int8_verifier.py`, `core/reddit_vision_vlm_moe.py:1260`) are shared-secret HMAC checks named "ZK", and the circuit module has a hard-coded default secret.
    - Action: rename them as tamper-evident commitments and drop the default secret.

Lower priority, same relabel pattern:
- `boinc/astrophysics_24h_campaign.py:1146-1250, 717, 1365` has literal cosine, Multi-Krum and storage-proof receipts and fixed dashboard cards.
- `boinc/local_astrophysics_boinc.py:225, 785` pads data silently and prints "3 Honest PASS" as a literal.
- Random-input metrics in `cli/multi_moe_hf_campaign.py`, `cli/season9_10_gpu_swarm_campaigns.py`, `swarm/strategy_optimizer.py:793`, and `swarm/top1pct_dct_ef21_compressor.py:241` are unlabelled.

## (C) MVP critical path

1. **Honest benchmarks page.** Rewrite `router/benchmarks_page.py` to read `artifacts/` results from `scripts/night_eval.py`, with n, date and model tag. Delete frontier scores. Make the metric fields Optional and render "not measured". Update `life_as_a_bee.py:135-243` and the test.
2. **Static-number guard.** Add a CI test that fails on numeric claims in `router/*page*.py` with no artifact reference. Replace `portal_page.py:1521-1523` and `life_as_a_bee.py:2153` with `shown_figure()` over `build_live_telemetry_snapshot()`.
3. **Catalog `kind` field.** Add `kind` (base, adapter, alias, planned) in `router/catalog.py` and expose it in `/v1/models`. Mark `bee-rusty-sundials` and `coder` as aliases of the same tag. Add a pinned tag-list test.
4. **Single source for agent and shield.** The `scripts/` and `src/` copies differ. Make the `scripts/` files re-export `src/hivemind/router/...` and add a diff test.
5. **Gateway canary and token cross-check.** Send a fixed prompt to each producer on first heartbeat and periodically, record real latency and tokens/s, and flag producers whose usage deviates by more than 20% from a tokenizer count. Files: `gateway.py`, `hivemind_producer_agent.py`.
6. **Remove fabricated credit paths.** Items B3, B5, B6, B7 and B14, in `hivemind_at_home_dashboard.py`, `exoplanet_hunter_cockpit.py`, `webgpu_galaxy_cockpit.py`, `stitch_v46_cockpit_template.py`, `auto_improvement.py` and `gateway.py`. Each is small, with a test asserting there is no grant without real input.
7. **Delete seeded data.** Remove `_init_baseline_end_users` and the fake feed. Regenerate or delete `artifacts/hivemind_traffic_tflops_monitor.json`.
8. **Training-round table and ratchet.** Publish (round, domain, `eval_by_domain` before/after) from the ledger. Add a rollback in `scripts/t4_workgen.py` if mean held-out loss rises. Replace the static OpenTrainer loss quotes in `router/opentrainer.py:25-35` with artifact references.
9. **Streaming TTFT and tokens/s.** Record both in settle (`gateway.py`, `accounts.py`).
10. **Close the train-to-serve loop.** Write one script that takes `latest_adapter.pt`, converts it with `convert_lora_to_gguf.py`, runs `night_eval.py` base against adapter, and serves it as a new `kind=adapter` catalog tier only if McNemar p<0.05.
    - The trained base is granite-1b-a400m but the served one is granite3.1-moe:3b, so this is the largest step. Do it last.
11. **Run `mvp_e2e_smoke.py` nightly** in CI with a tiny GGUF.

## (D) Remove or archive

- **Dead.**
  - `boinc/validator_assignment.py`, used by tests only.
  - `router/welcome_email.py`, whose send function has no callers and reports a spool that is never written.
  - `swarm/pr_first_openrouter_orchestrator.py`, tests only.
- **Scaffold.** `swarm/dynamic_diurnal_moe_engine.py`, tests only.
- **Archive as simulation or demo.** Line counts were not supplied except where noted.
  - `core/dev_image_video_moe_integration_benchmark.py`
  - `core/image_moe_inference_gallery_benchmark.py`
  - `game/fat_steam_astro_builder.py`, tests only
  - `swarm/exemplar_byob_bees.py`
  - `swarm/s34_04_*`, `s35_03_*`, `s35_05_*`
  - `swarm/abc_moe_agent_mesh.py`, tests only
  - `boinc/real_client/app.py`, a toy LoRA that is a validator fixture
- **Largest single file.** `hivemind_at_home_dashboard.py` is 9,596 lines. Split off the canned endpoints and examples gallery before relabelling.
- **Nonexistent script.** `scripts/e8_qlora_moe_round.py` is referenced somewhere but does not exist. Fix the references.

## (E) Owner decisions and hardware

- **Public claims.** Decide whether to drop the frontier-comparison and "daily coverage" claims, or fund a real eval against named baselines. That eval needs API keys and cost.
- **Steam.** A real Sybil gate needs a Steam publisher key. Decide whether to keep Steam at all or label it simulated.
- **Credit and points policy.**
  - Should points be earned only from validated work?
  - What should the HMT, cobblestone and TFLOPS formulas be?
  - Should the demo auto-contribution to production be removed?
- **Base model.** Choose one base for the train-serve loop. Either serve granite-1b-a400m, or retrain on granite3.1-moe:3b.
- **Hardware.**
  - A GPU to run `night_eval.py` and the adapter-to-GGUF conversion. The T4 has a VRAM floor of 4,300 MiB per AGENTS.md.
  - At least two independent volunteer hosts, since the evidence so far is a single host.
  - A real NF4/QLoRA or pretrained base for `data/dataset_mining_campaign.py`, which currently uses a random-init toy.
- **Unverified third-party repo.** Confirm the `hf.co/Vxtzq/Crowd-v1` repo, which the audit could not verify.
- **Public T4 exposure.** Memory notes list this as an open owner decision.
