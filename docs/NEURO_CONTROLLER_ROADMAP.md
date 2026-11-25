# HiveMind NeuroController — Roadmap & Empirical Weight-Improvement Ledger

> **Non-Monetary Charter (`AGPL-3.0`):** All Project HiveMind NeuroController modules, CPU weight-space calibration pipelines, and open-weight experiments operate strictly under the **`0.00 EUR / 0.00 USD`** Personal-PC & Volunteer Compute-Reciprocity Charter (`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`), with zero monetary payouts, zero cryptocurrency wallets, and zero external payment links. Root `ROADMAP.md` remains reserved; NeuroController tracking is maintained exclusively here and in [`docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`](./NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md).

---

## 1. Mission & Thermodynamic Invariants

The **HiveMind NeuroController** program builds deterministic, training-free, CPU-only (`CUDA_VISIBLE_DEVICES=""`, `0 MiB` GPU VRAM, `0` gradient steps) Weight-Space Programming (`WSP`) surgeons for open-weight Sparse Mixture-of-Experts (`CrowdV1WithEngram` / `CrowdLoRAMoEMLP`) and dense causal language models (`Qwen2.5`, `Qwen3.5`, `Granite`, `Llama`).

Every candidate edit is governed by non-negotiable physical invariants:
- **Monotone ANSE Descent (`Delta E < 0`):** Candidate edits must strictly reduce objective Autopoietic Neuro-Symbolic Energy (`ANSE = mean_diff + 0.25 * ece_diff + margin_penalty`), else atomic bitwise state restoration (`final_state_sha256 == initial_state_sha256`).
- **General-Prose & Worst-Domain Guards (`E0 / D0 <= +1.0%`):** No edit may degrade general-prose NLL (`E0_general_fineweb` / `D0_general_prose`) or any individual domain NLL by more than `+1.0%` relative to the pristine reference checkpoint; any breach incurs `ANSE_MAXIMUM_PAIN = 1e6`.
- **Single-Shot Virgin Holdout Vaults (`SingleShotVaultLock`):** Pairwise SHA-256 chunk disjointness is certified across all `probe`, `selection`, `confirm`, and `virgin_vault` splits (`13`-way and `17`-way verified), and each holdout vault is evaluated at most once.
- **Zero-Overhead Edge Deployment:** Final calibration edits (`1 / tau_temp` folded into `RMSNorm`/`LayerNorm`, orthogonalized LoRA/down-projection updates, NF4 Hessian compensation) incur zero runtime hooks and `0` extra FLOPs during inference.

---

## 2. Empirical Multi-Model Results (T4 CPU & Local CPU)

### 2.1 GCP T4 CPU Ledger & Active Ratchet Snapshot

```text
T4 (socreateai-agora-hermes-node1, us-east4-b) checked at 2026-11-25T05:32:29Z (MemAvailable=16512568 kB / ~15.75 GiB >= 11.0 GiB). Locked/certified vaults preserved untouched: skip=0; skip=160 (PILOT1 WSP_S4_multigate_only: mean_diff_vs_base=-0.554969 nats [-0.5550], E0_general_fineweb=+0.119%, PASS, MOSTLY_CONTENT); skip=320 (PILOT2 WSP_S4_multigate_only: locked_utc=2026-11-24T20:49:06Z, mean_diff_vs_base=-0.493307 nats [-0.4933], E0_general_fineweb=+0.526%, PASS, MOSTLY_CONTENT); pilot4b_qwen35_vault_lock.json (PILOT4B WSP_AlphaEdit_Nullspace_LoRA_Qwen35: locked_utc=2026-11-24T22:57:09Z, base_nll=2.424393 -> cand_nll=2.410910, mean_diff=-0.013483 nats [-0.0135], CI95=[-0.018053, -0.009293], base_ece=0.040123 -> cand_ece=0.023257 [-42.0%], selection mean_4gram_repeat_ratio 0.175676 -> 0.148649 [-15.4%], PASS). Active CPU-only PILOT4A run in progress: PID 1181000 (pilot4_t4_qwen35_and_t3_rescue.py --part 4a --threads 4 --min-avail-gb 11.0) synthesized 3 CPU arms (WSP_T3_self_healed_rescue, WSP_T6_calibrated, WSP_Tangent_Fusion_C2_plus_T6) on probe_skip360 and spawned PID 1258026 (crowdgpt_moe_multi_eval.py --skip-eligible 400); pilot4a.log shows selection_skip400 base-equivalence gate passed (4.219194 vs 4.219194) and started scoring RAW_T3_discarded_baseline at 2026-11-25T05:31:46. pilot4a_summary.json and vault_skip480_single_shot.lock.json do not exist yet. pilot5_c3_auto_wsp/ contains pre-staged pilot5_c3_auto_wsp.py, pilot5_preregistration.json (probe_skip=520, selection_skip=560, confirm_skip=600, virgin_vault_skip=640), and seventeen_way_disjoint_receipt.json, with no pilot5.log, pilot5_summary.json, or vault_skip640_single_shot.lock.json yet. Auto-advance ratchet condition NOT met (pilot4a_summary.json absent and pilot4_t4_qwen35 + crowdgpt_moe_multi_eval actively running), so pilot5_c3_auto_wsp.py was NOT launched.
```

### 2.2 Certified Single-Shot Holdout Vaults Across Models

| Experiment | Model / Checkpoint | Winning Arm | Vault Split | Mean NLL Delta | General Guard (`E0` / `D0`) | Calibration / Repetition Gain | Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **`PILOT1`** | `C2_final` (`CROWDGPT-MOE-8X-UPCYCLE-V1`, 3B MoE) | `WSP_S4_multigate_only` | `vault_skip160` (`n=240`) | **`-0.5550`** nats (`95% CI [-0.6270, -0.4837]`) | `E0 = +0.119%` (`<= +1.0%`) | `E6 = -46.694%`, `E7 = -14.651%`, `E5 = -11.789%` | **`PASS`** (`MOSTLY_CONTENT`) |
| **`PILOT2`** | `C2_final` (`CROWDGPT-MOE-8X-UPCYCLE-V1`, 3B MoE) | `WSP_S4_multigate_only` | `vault_skip320` (`n=240`) | **`-0.4933`** nats (`95% CI [-0.5662, -0.4223]`) | `E0 = +0.526%` (`<= +1.0%`) | Locked `2026-11-24T20:49:06Z` | **`PASS`** (`MOSTLY_CONTENT`) |
| **`PILOT3A`** | `Qwen/Qwen2.5-0.5B-Instruct` + 3 real LoRA adapters | `WSP_S5_Full_Synergy_Qwen05B` (`sel`) / `NAIVE` (`vault`) | 4-way disjoint (`n=40`) | `-0.0165` nats (`S5` sel) / `-0.0083` (`vault`) | `D0 = -0.732%` (`S5`) / `-1.245%` | `S5` ECE `0.0332 -> 0.0212` (`-36.3%`) at `0.0139%` DOF | **`PASS`** |
| **`PILOT3B`** | `Qwen/Qwen2.5-1.5B` (`1,543,714,304` params base) | `WSP_15B_Folded_RMSNorm_Temp` (`tau_temp=1.05`) | 4-way disjoint (`n=40`) | **`-0.0059`** nats (`95% CI [-0.0110, -0.0013]`) | `D0 = -0.725%` (`D1=-0.322%`) | Virgin-vault ECE `0.0260 -> 0.0138` (**`-46.8%`**), `0.0000995%` edited (`1,536` params); tail ESD rolled back by ratchet | **`PASS`** |
| **`PILOT4B`** | `Qwen/Qwen3.5-0.8B` (`752,393,024` params) + `qwen35_08b_smoke` LoRA | `WSP_AlphaEdit_Nullspace_LoRA_Qwen35` | 4-way disjoint (`vault`) | **`-0.0135`** nats (`95% CI [-0.0181, -0.0093]`) | `D0 = -0.699%` (`sel`) | Virgin-vault ECE `0.0401 -> 0.0233` (**`-42.0%`**); 4-gram loop `0.1757 -> 0.1486` (**`-15.4%`**) | **`PASS`** |

### 2.3 Active & Pre-Staged T4 Campaigns (`PILOT4A`, `E-MOE-C3`, `PILOT5`)

- **`E-MOE-C3` Baseline Diagnosis:** Pure gradient OpenDiLoCo across 4 arms finished `30/30` rounds with `kept = [], winner = null` — router-frozen Arm `D` kept `E0 = +0.098%` at `mean_diff = -0.0943` nats vs base, while router-unfrozen Arm `B` unlocked `E6 = -43.162%` (`mean_diff = -0.2824`) at `E0 = +13.773%` prose drift (and Arm `A` hit `E0 = +1.208%`).
- **`PILOT4A` (In Progress on T4 CPU):** 13-way SHA-256 split disjointness certified (`skip=0..480`). Arms `WSP_T3_self_healed_rescue`, `WSP_T6_calibrated`, and `WSP_Tangent_Fusion_C2_plus_T6` synthesized on `probe_skip360` and actively scoring on `selection_skip400`.
- **`PILOT5` (Pre-Staged, Not Yet Launched):** 17-way SHA-256 split disjointness certified (`skip=0..640`) in `seventeen_way_disjoint_receipt.json` to rescue `C3` Arms `A`/`B` and cross-compose `D`-router + `B`-experts once `PILOT4A` finishes.

---

## 3. NeuroController Delivery Ledger (`43` Merged PRs `#978`–`#1038` + Current `NEURAL-CTRL-DYN-37`)

| # | Slice ID | PR | Status | Target Module(s) | Scenario Test File | Capability Summary |
| :-: | :--- | :-: | :--- | :--- | :--- | :--- |
| 1 | `NEURAL-CTRL-01` | `#978` | Merged | `src/hivemind/swarm/wsp_alphaedit_control.py` | `tests/test_wsp_alphaedit_control_scenarios.py` | Discriminative AlphaEdit filter, FoMoE ghost sink, LayerNorm calibration |
| 2 | `NEURAL-CTRL-02` | `#979` | Merged | `src/hivemind/swarm/sigma_router_gate_control.py` | `tests/test_sigma_router_gate_scenarios.py` | Contrastive `sigma_topk` multi-gate controller with 2-fold OOF shutoff |
| 3 | `NEURAL-CTRL-03` | `#981` | Merged | `src/hivemind/swarm/layer_localized_attribution.py` | `tests/test_layer_localized_attribution_scenarios.py` | Causal logit-lens layer-localized & LOEO expert attribution |
| 4 | `NEURAL-CTRL-04` | `#982` | Merged | `src/hivemind/swarm/lora_svd_mode_surgery.py` | `tests/test_lora_svd_mode_surgery_scenarios.py` | QR-SVD rank-selective LoRA singular-mode surgery (`<0.05%` DOF) |
| 5 | `NEURAL-CTRL-05` | `#983` | Merged | `scripts/hivemind_neural_control_harness.py` | `tests/test_hivemind_neural_control_harness.py` | 4-split anti-leakage ANSE benchmark harness & receipt verifier |
| 6 | `NEURAL-CTRL-06` | `#984` | Merged | `src/hivemind/swarm/adaptive_steering_guard.py` | `tests/test_adaptive_steering_guard_scenarios.py` | Runtime thermodynamic dense-fallback hot-swap guard |
| 7 | `NEURAL-CTRL-DYN-01` | `#985` | Merged | `src/hivemind/swarm/self_healing_weight_repair.py` | `tests/test_self_healing_weight_repair_scenarios.py` | Self-rewriting weight-block corruption scanner & closed-loop repair |
| 8 | `NEURAL-CTRL-DYN-02` | `#986` | Merged | `src/hivemind/swarm/entropy_loop_steering.py` | `tests/test_entropy_loop_steering_scenarios.py` | Sub-0.02% Astra-style thinking-loop neuron suppression |
| 9 | `NEURAL-CTRL-DYN-03` | `#988` | Merged | `src/hivemind/swarm/marchenko_pastur_covariance_guard.py` | `tests/test_marchenko_pastur_covariance_guard_scenarios.py` | Ledoit-Wolf Marchenko-Pastur shrinkage & norm-bounded AlphaEdit |
| 10 | `NEURAL-CTRL-DYN-04` | `#989` | Merged | `src/hivemind/swarm/unified_wsp_neural_controller.py` | `tests/test_unified_wsp_neural_controller_scenarios.py` | Unified 5-stage Weight-Space Programming controller |
| 11 | `NEURAL-CTRL-DYN-05` | `#990` | Merged | `src/hivemind/swarm/decision_skill_weight_steering.py` | `tests/test_decision_skill_weight_steering_scenarios.py` | Decision-energy rank-1 skill steering in preserve null-space |
| 12 | `NEURAL-CTRL-DYN-06` | `#991` | Merged | `src/hivemind/swarm/ties_dare_moe_merger.py` | `tests/test_ties_dare_moe_merger_scenarios.py` | TIES/DARE sign-consensus & prose-nullspace distributed MoE merger |
| 13 | `NEURAL-CTRL-DYN-07` | `#993` | Merged | `src/hivemind/swarm/quantization_compensated_wsp.py` | `tests/test_quantization_compensated_wsp_scenarios.py` | Quantization-compensated WSP & LoftQ prose-weighted LoRA residual |
| 14 | `NEURAL-CTRL-DYN-08` | `#994` | Merged | `src/hivemind/swarm/representation_circuit_breaker.py` | `tests/test_representation_circuit_breaker_scenarios.py` | Weight-space RepE circuit breaker & false-refusal abliteration |
| 15 | `NEURAL-CTRL-DYN-09` | `#995` | Merged | `scripts/hivemind_wsp_checkpoint_doctor.py` | `tests/test_hivemind_wsp_checkpoint_doctor.py` | Offline safetensors MoE surgery doctor & ANSE preflight CLI |
| 16 | `NEURAL-CTRL-DYN-10` | `#996` | Merged | `src/hivemind/swarm/cross_layer_residual_nullspace.py` | `tests/test_cross_layer_residual_nullspace_scenarios.py` | Cross-layer residual interference cancellation & anchor protection |
| 17 | `NEURAL-CTRL-DYN-11` | `#997` | Merged | `src/hivemind/swarm/fisher_kfac_diagonal_surgeon.py` | `tests/test_fisher_kfac_diagonal_surgeon_scenarios.py` | Wanda/OBS activation-weighted LoRA surgical sparsification |
| 18 | `NEURAL-CTRL-DYN-12` | `#999` | Merged | `src/hivemind/swarm/rome_memit_knowledge_editor.py` | `tests/test_rome_memit_knowledge_editor_scenarios.py` | AlphaEdit-MEMIT multi-layer SwiGLU knowledge & skill editor |
| 19 | `NEURAL-CTRL-DYN-13` | `#1000` | Merged | `src/hivemind/swarm/moe_load_balance_bias_controller.py` | `tests/test_moe_load_balance_bias_controller_scenarios.py` | Auxiliary-loss-free router bias load-balance calibration |
| 20 | `NEURAL-CTRL-DYN-14` | `#1001` | Merged | `src/hivemind/swarm/logit_calibration_temperature_surgeon.py` | `tests/test_logit_calibration_temperature_surgeon_scenarios.py` | Post-surgery final LayerNorm temperature & channelwise calibration |
| 21 | `NEURAL-CTRL-DYN-15` | `#1002` | Merged | `src/hivemind/swarm/orthogonal_multigate_router_surgeon.py` | `tests/test_orthogonal_multigate_router_surgeon_scenarios.py` | Orthogonal multi-gate specialist preserver (`S4` pipeline) |
| 22 | `NEURAL-CTRL-DYN-16` | `#1003` | Merged | `src/hivemind/swarm/monotone_wsp_stage_composer.py` | `tests/test_monotone_wsp_stage_composer_scenarios.py` | Monotone multi-stage WSP composer & single-shot vault lock |
| 23 | `NEURAL-CTRL-DYN-17` | `#1005` | Merged | `src/hivemind/swarm/attention_sink_kv_surgeon.py` | `tests/test_attention_sink_kv_surgeon_scenarios.py` | StreamingLLM attention-sink & GQA `o_proj` null-space stabilizer |
| 24 | `NEURAL-CTRL-DYN-18` | `#1007` | Merged | `src/hivemind/swarm/byzantine_wsp_delta_aggregator.py` | `tests/test_byzantine_wsp_delta_aggregator_scenarios.py` | Byzantine-resilient Multi-Krum WSP delta aggregator in `P_0` |
| 25 | `NEURAL-CTRL-DYN-19` | `#1009` | Merged | `src/hivemind/swarm/sequential_nullspace_memory.py` | `tests/test_sequential_nullspace_memory_scenarios.py` | Lifelong sequential null-space basis tracker across rounds |
| 26 | Season 35 Core | `#1008` | Merged | `src/hivemind/swarm/auto_open_weight_engine.py`, `scripts/hivemind_auto_open_weight.py`, `docs/AUTO_OPEN_WEIGHT_CAPABILITY_PLAN.md` | `tests/test_auto_open_weight_engine_scenarios.py` | Gen-2 `S5` PASS-Synergy autonomous open-weight engine & CLI |
| 27 | `NEURAL-CTRL-DYN-20` | `#1012` | Merged | `src/hivemind/swarm/opentrainer_auto_wsp_hook.py` | `tests/test_opentrainer_auto_wsp_hook_scenarios.py` | Automatic OpenTrainer post-round WSP hook & canary gate |
| 28 | `NEURAL-CTRL-DYN-21` | `#1013` | Merged | `src/hivemind/swarm/volunteer_wsp_workunit_exporter.py` | `tests/test_volunteer_wsp_workunit_exporter_scenarios.py` | Volunteer 8 GB Gamer-Whisper WSP workunit & 4-bit NF4 exporter |
| 29 | `NEURAL-CTRL-DYN-22` | `#1014` | Merged | `src/hivemind/swarm/wsp_formal_invariant_verifier.py` | `tests/test_wsp_formal_invariant_verifier_scenarios.py` | Algebraic soundness verifier for `S5` WSP invariants |
| 30 | `NEURAL-CTRL-DYN-23` | `#1016` | Merged | `src/hivemind/swarm/multi_arch_auto_wsp_adapter.py` | `tests/test_multi_arch_auto_wsp_adapter_scenarios.py` | Multi-arch open-weight WSP adapter (Granite, Qwen, Llama) |
| 31 | `NEURAL-CTRL-DYN-24` | `#1017` | Merged | `src/hivemind/swarm/lean4_wsp_theorem_registry.py`, `lean4/HiveMindSpec.lean` | `tests/test_lean4_wsp_theorem_registry_scenarios.py` | Machine-checked Lean 4 Season 35 WSP theorems (`0 sorry`) |
| 32 | `NEURAL-CTRL-DYN-25` | `#1018` | Merged | `src/hivemind/swarm/auto_wsp_campaign_scheduler.py` | `tests/test_auto_wsp_campaign_scheduler_scenarios.py` | Autonomous WSP campaign ledger & drift-triggered scheduler |
| 33 | `NEURAL-CTRL-DYN-26` | `#1020` | Merged | `src/hivemind/swarm/shadow_canary_wsp_traffic_router.py` | `tests/test_shadow_canary_wsp_traffic_router_scenarios.py` | Sequential SPRT shadow-canary traffic router & rollback |
| 34 | `NEURAL-CTRL-DYN-27` | `#1021` | Merged | `src/hivemind/swarm/auto_open_weight_orchestrator.py` | `tests/test_auto_open_weight_orchestrator_scenarios.py` | End-to-end `AutoOpenWeightOrchestrator` bundle pipeline |
| 35 | `NEURAL-CTRL-DYN-28` | `#1022` | Merged | `src/hivemind/swarm/heavy_tail_esd_weight_surgeon.py` | `tests/test_heavy_tail_esd_weight_surgeon_scenarios.py` | Activation-free heavy-tail ESD power-law alpha weight surgeon |
| 36 | `NEURAL-CTRL-DYN-29` | `#1023` | Merged | `src/hivemind/swarm/procrustes_wsp_skill_transfer.py` | `tests/test_procrustes_wsp_skill_transfer_scenarios.py` | Orthogonal Procrustes & CKA cross-checkpoint skill transfer |
| 37 | `NEURAL-CTRL-DYN-30` | `#1025` | Merged | `src/hivemind/swarm/natural_gradient_trust_wsp_surgeon.py` | `tests/test_natural_gradient_trust_wsp_surgeon_scenarios.py` | Fisher-Rao Kronecker natural-gradient KL trust-region surgeon |
| 38 | `NEURAL-CTRL-DYN-31` | `#1026` | Merged | `src/hivemind/swarm/contrastive_steering_weight_compiler.py` | `tests/test_contrastive_steering_weight_compiler_scenarios.py` | Zero-overhead contrastive steering vector to `P_0` weight compiler |
| 39 | `NEURAL-CTRL-DYN-32` | `#1029` | Merged | `src/hivemind/swarm/tangent_task_vector_surgeon.py` | `tests/test_tangent_task_vector_surgeon_scenarios.py` | Tangent-space task-vector polar orthogonalization surgeon |
| 40 | `NEURAL-CTRL-DYN-33` | `#1031` | Merged | `src/hivemind/swarm/representation_collapse_wsp_guard.py` | `tests/test_representation_collapse_wsp_guard_scenarios.py` | Isotropy & RankMe representation-collapse guard |
| 41 | `NEURAL-CTRL-DYN-34` | `#1033` | Merged | `src/hivemind/swarm/thinking_loop_wsp_suppressor.py` | `tests/test_thinking_loop_wsp_suppressor_scenarios.py` | Sub-0.02% thinking-loop & repetition-attractor suppressor |
| 42 | `NEURAL-CTRL-DYN-35` | `#1035` | Merged | `src/hivemind/swarm/self_healing_checkpoint_rescuer.py` | `tests/test_self_healing_checkpoint_rescuer_scenarios.py` | Self-healing discarded checkpoint rescuer (`T3` rescue pattern) |
| 43 | `NEURAL-CTRL-DYN-36` | `#1038` | Merged | `src/hivemind/swarm/router_freeze_cross_arm_surgeon.py` | `tests/test_router_freeze_cross_arm_surgeon_scenarios.py` | Post-DiLoCo router-frozen anchor + router-unfrozen cross-arm surgeon |
| 44 | **`NEURAL-CTRL-DYN-37`** | **Current** | **Implemented** | `src/hivemind/swarm/folded_norm_ece_calibrator.py` | `tests/test_folded_norm_ece_calibrator_scenarios.py` | Sub-0.001% zero-overhead folded final RMSNorm/LayerNorm temperature & ECE calibration surgeon with ANSE stage ratchet (`PILOT3B`) |
| 45 | `NEURAL-CTRL-DYN-38` | Queued | Pending | `src/hivemind/swarm/domain_matched_moe_arm_composer.py` | `tests/test_domain_matched_moe_arm_composer_scenarios.py` | Domain-matched multi-arm MoE specialist composer & 2-matmul affine LayerNorm calibrator (`PILOT4A` / `PILOT5`) |
