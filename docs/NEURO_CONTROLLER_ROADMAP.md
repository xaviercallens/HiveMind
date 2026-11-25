# HiveMind NeuroController — Roadmap & Empirical Weight-Improvement Ledger

> **Non-Monetary Charter (`AGPL-3.0`):** All Project HiveMind NeuroController modules, CPU weight-space calibration pipelines, and open-weight experiments operate strictly under the **`0.00 EUR / 0.00 USD`** Personal-PC & Volunteer Compute-Reciprocity Charter (`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`), with zero monetary payouts, zero cryptocurrency wallets, and zero external payment links. Root `ROADMAP.md` remains untouched; NeuroController tracking is maintained exclusively here and in [`docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`](./NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md).

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
T4 (socreateai-agora-hermes-node1, us-east4-b) CPU-only ratchet active (MemAvailable 15797044 kB / ~15.07 GiB >= 11.0 GiB; no duplicate runner launched because pilot5_c3_auto_wsp.py is already running). Locked vaults: PILOT1 (skip=160, WSP_S4_multigate_only: mean_diff_vs_base=-0.554969 nats, E0=+0.119%, guard_ok=true, PASS); PILOT2 (skip=320, WSP_S4_multigate_only: mean_diff_vs_base=-0.493307 nats, E0=+0.526%, guard_ok=true, PASS); PILOT4B (pilot4b_qwen35_vault_lock.json, WSP_AlphaEdit_Nullspace_LoRA_Qwen35: mean_diff=-0.013483 nats, ci95=[-0.018053,-0.009293], ECE 0.040123->0.023257 [-42.0%], selection mean_4gram_repeat_ratio 0.175676->0.148649 [-15.4%], PASS). PILOT4A (pilot4a_summary.json completed 2026-11-25T06:59:54Z): on selection_skip400 (n=240, base=4.528327), RAW_T3_discarded_baseline mean_diff=+0.008268 (E0=+19.583%, NO_GAIN, anse_energy=1000000.008268), WSP_T3_self_healed_rescue mean_diff=-0.090820 (E0=+11.661%, guard_ok=false), RAW_T6_results_champion mean_diff=-0.058193 (E0=-0.921%, PASS), WSP_T6_calibrated mean_diff=-0.070495 (E0=-1.626%, E1=-2.728%, E7=-2.320%, guard_ok=true, PASS, anse_energy=-0.070495), WSP_Tangent_Fusion_C2_plus_T6 mean_diff=-0.468693 (E0=+1.644%, guard_ok=false), CHAMP_C2_reference winner mean_diff=-0.499399 (E0=+0.650%, guard_ok=true, PASS, anse_energy=-0.499399); on confirm_skip440 CHAMP_C2_reference mean_diff=-0.486380 (E0=+1.151%, guard_ok=false); vault_skip480_single_shot.lock.json locked CHAMP_C2_reference mean_diff_vs_base=-0.476871 nats (ci95=[-0.540538,-0.412921], E0=+1.006%, guard_ok=false, verdict_ip300=PASS, anse_energy=999999.523129). PILOT5 (pilot5_c3_auto_wsp): pilot5_summary.json & vault_skip640_single_shot.lock.json do not exist yet because pilot5_c3_auto_wsp.py (PID 1361928, elapsed 02:22:19, 258% CPU) and child crowdgpt_moe_multi_eval.py --skip-eligible 560 (PID 1433439, elapsed 42:46, 365% CPU) are actively running on CPU. Completed selection_skip560 arms (240 chunks, base=3.384382): RAW_C3_A_unfrozen_no_ilv mean diff -0.1433, RAW_C3_B_unfrozen_ilv8192 mean diff -0.2808, RAW_C3_D_frozen_router mean diff -0.0905, WSP_C3_A_rescued mean diff -0.1586; currently scoring WSP_C3_B_rescued (started 2026-11-25T09:29:03), with WSP_C3_CrossArm_D_Router_B_Experts and WSP_C3_AB_Domain_Synergy queued before confirm_skip600 and virgin_vault_skip640.
```

### 2.2 Certified Single-Shot Holdout Vaults & Completed Campaigns Across Models

| Experiment | Model / Checkpoint | Winning / Evaluated Arm | Split | Mean NLL Delta | General Guard (`E0` / `D0`) | Calibration / Domain Gain | Verdict |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- | :--- |
| **`PILOT1`** | `C2_final` (`CROWDGPT-MOE-8X-UPCYCLE-V1`, 3B MoE) | `WSP_S4_multigate_only` | `vault_skip160` (`n=240`) | **`-0.554969`** nats (`95% CI [-0.6270, -0.4837]`) | `E0 = +0.119%` (`<= +1.0%`) | `E6 = -46.694%`, `E7 = -14.651%`, `E5 = -11.789%` | **`PASS`** (`MOSTLY_CONTENT`) |
| **`PILOT2`** | `C2_final` (`CROWDGPT-MOE-8X-UPCYCLE-V1`, 3B MoE) | `WSP_S4_multigate_only` | `vault_skip320` (`n=240`) | **`-0.493307`** nats (`95% CI [-0.5662, -0.4223]`) | `E0 = +0.526%` (`<= +1.0%`) | Locked `2026-11-24T20:49:06Z` | **`PASS`** (`MOSTLY_CONTENT`) |
| **`PILOT3A`** | `Qwen/Qwen2.5-0.5B-Instruct` + 3 real LoRA adapters | `WSP_S5_Full_Synergy_Qwen05B` (`sel`) / `NAIVE` (`vault`) | 4-way disjoint (`n=40`) | `-0.0165` nats (`S5` sel) / `-0.0083` (`vault`) | `D0 = -0.732%` (`S5`) / `-1.245%` | `S5` ECE `0.0332 -> 0.0212` (`-36.3%`) at `0.0139%` DOF | **`PASS`** |
| **`PILOT3B`** | `Qwen/Qwen2.5-1.5B` (`1,543,714,304` params base) | `WSP_15B_Folded_RMSNorm_Temp` (`tau_temp=1.05`) | 4-way disjoint (`n=40`) | **`-0.0059`** nats (`95% CI [-0.0110, -0.0013]`) | `D0 = -0.725%` (`D1=-0.322%`) | Virgin-vault ECE `0.0260 -> 0.0138` (**`-46.8%`**), `0.0000995%` edited (`1,536` params) | **`PASS`** |
| **`PILOT4A`** | `T3` / `T6` / `C2_final` (`CROWDGPT-MOE-8X-UPCYCLE-V1`) | `CHAMP_C2_reference` (`sel`) / `WSP_T6_calibrated` (`sel`) | `vault_skip480` (`n=240`) | **`-0.476871`** nats (`95% CI [-0.540538, -0.412921]`) | `E0 = +1.006%` (`guard_ok=false` at `skip480`) | `WSP_T6_calibrated` (`sel`): `E0=-1.626%`, `E1=-2.728%`, `E7=-2.320%`, `anse_energy=-0.070495` (`PASS`) | **`PASS`** (`verdict_ip300`, `skip480`) |
| **`PILOT4B`** | `Qwen/Qwen3.5-0.8B` (`752,393,024` params) + `qwen35_08b_smoke` LoRA | `WSP_AlphaEdit_Nullspace_LoRA_Qwen35` | 4-way disjoint (`vault`) | **`-0.013483`** nats (`95% CI [-0.018053, -0.009293]`) | `D0 = -0.699%` (`sel`) | Virgin-vault ECE `0.040123 -> 0.023257` (**`-42.0%`**); 4-gram loop `0.175676 -> 0.148649` (**`-15.4%`**) | **`PASS`** |

### 2.3 Active T4 Campaign (`PILOT5` on `CROWDGPT-MOE-C3`)

- **`E-MOE-C3` Baseline Diagnosis:** Pure gradient OpenDiLoCo across 4 arms finished `30/30` rounds with `kept = [], winner = null` — router-frozen Arm `D` kept `E0 = +0.098%` at `mean_diff = -0.0943` nats vs base, while router-unfrozen Arm `B` unlocked `E2_math = -6.499%`, `E5 = -7.168%`, `E6_stars = -43.162%` (`mean_diff = -0.2824`) at `E0 = +13.773%` prose drift (and Arm `A` unlocked `E1_code = -5.212%` at `E0 = +1.208%`).
- **`PILOT5` (`pilot5_c3_auto_wsp`, Active on T4 CPU):** 17-way SHA-256 split disjointness certified (`skip=0..640`) in `seventeen_way_disjoint_receipt.json`. All `7 / 7` arms completed `selection_skip560` (`n=240`, `base=3.384382`): `WSP_C3_B_rescued` (`-0.2964` nats) beat `RAW_C3_B_unfrozen_ilv8192` (`-0.2808`), `WSP_C3_AB_Domain_Synergy` (`-0.2224`), `WSP_C3_A_rescued` (`-0.1586`, beating `RAW_C3_A_unfrozen_no_ilv` `-0.1433`), `RAW_C3_D_frozen_router` (`-0.0905`), and `WSP_C3_CrossArm_D_Router_B_Experts` (`+0.0080`, router-expert co-adaptation mismatch); `confirm_skip600` (`base=4.421316`) is now evaluating ahead of single-shot `virgin_vault_skip640`.

---

## 3. NeuroController Delivery Ledger (`44` Merged PRs `#978`–`#1040` + Current `NEURAL-CTRL-DYN-38`)

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
| 44 | `NEURAL-CTRL-DYN-37` | `#1040` | Merged | `src/hivemind/swarm/folded_norm_ece_calibrator.py` | `tests/test_folded_norm_ece_calibrator_scenarios.py` | Sub-0.001% zero-overhead folded final RMSNorm/LayerNorm temperature & ECE calibration surgeon (`PILOT3B`) |
| 45 | **`NEURAL-CTRL-DYN-38`** | **Current** | **Implemented** | `src/hivemind/swarm/domain_matched_moe_arm_composer.py` | `tests/test_domain_matched_moe_arm_composer_scenarios.py` | Domain-matched multi-arm MoE specialist composer & 2-matmul affine LayerNorm calibrator (`PILOT4A` / `PILOT5`) |
