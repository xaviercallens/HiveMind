# HiveMind NeuroController — Architecture & Implementation Plan

> **Non-Monetary Charter (`AGPL-3.0`):** The HiveMind NeuroController subsystem executes under the **`0.00 EUR / 0.00 USD`** Personal-PC & Volunteer Compute-Reciprocity Charter (`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`), strictly on CPU (`0 MiB` GPU VRAM, `0` gradient steps), with zero cryptocurrency wallets, zero external payment links, and zero unauthorized lateral egress. Root `ROADMAP.md` is untouched; NeuroController status and milestones live in this document and [`docs/NEURO_CONTROLLER_ROADMAP.md`](./NEURO_CONTROLLER_ROADMAP.md).

---

## 1. End-to-End NeuroController Operator Pipeline

```text
  Preflight MP Guard
   (Ledoit-Wolf Marchenko-Pastur shrinkage, spectral outlier scan, NaN/Inf quarantine)
         |
         v
  Byzantine Null-Space Filter
   (Multi-Krum peer filtering, spectral spike rejection, P_0 = I - U_0 U_0^T projection)
         |
         v
  Causal Anchor Protection + Selective Sinks + OOF Multi-Gate + QR-SVD/ESD + Folded Norm Calibration
   (Prefix-cached L+L(L+1)/2 anchor protection; router-expert co-adaptation mismatch veto;
    FoMoE w_B=0 ghost sinks; 2-fold OOF SNR gating; rank-r QR-SVD & ESD; folded 1/tau_temp)
         |
         v
  Monotone ANSE Ratchet
   (Per-stage Delta E < 0 & general-prose E0/D0 <= +1.0% guard; atomic bitwise rollback + 1e6 pain)
         |
         v
  Single-Shot Holdout Vault
   (Pairwise SHA-256 split disjointness across probe/selection/confirm/vault + immutable lock)
         |
         v
  4-Bit NF4 Export & SPRT Canary
   (Diagonal-Hessian NF4 edge blocks under 6,656 MiB 8 GB VRAM ceiling + sequential SPRT routing)
```

### Operator Mapping to Cleanroom Modules

| Pipeline Stage | Primary Module(s) | Algebraic / Thermodynamic Invariant |
| :--- | :--- | :--- |
| **Preflight MP Guard** | `marchenko_pastur_covariance_guard.py`, `self_healing_weight_repair.py`, `representation_collapse_wsp_guard.py` | Oracle shrinkage $\Sigma_{\text{shrunk}} = (1-\rho)M + \rho \frac{\text{tr}(M)}{d}I$, $\|G^*\|_2 \le 1.05$, RankMe preservation |
| **Byzantine Null-Space Filter** | `byzantine_wsp_delta_aggregator.py`, `sequential_nullspace_memory.py`, `volunteer_wsp_workunit_exporter.py` | Multi-Krum filters $f < (n-2)/2$ poisoned peers inside orthonormal prose null-space $P_0$ |
| **Causal Attribution & Selective Sinks** | `layer_localized_attribution.py`, `cross_layer_residual_nullspace.py`, `self_healing_checkpoint_rescuer.py`, **`prefix_cached_causal_attribution.py`** | Prefix-cached $L + L(L+1)/2$ knockouts; beneficial anchor layers ($h_\ell \le -0.010$ nats) bitwise immutable; rogue `w_B = 0` yields $\Delta y = 0$ |
| **OOF Multi-Gate & Co-Adaptation Guard** | `sigma_router_gate_control.py`, `orthogonal_multigate_router_surgeon.py`, `router_freeze_cross_arm_surgeon.py`, `domain_matched_moe_arm_composer.py`, **`prefix_cached_causal_attribution.py`** | 2-fold OOF SNR gating & automatic veto of cross-arm router grafts with $\Delta\text{NLL}_{\text{graft}} > \max$ or large routing KL |
| **Spectral & RepE Weight Surgery** | `lora_svd_mode_surgery.py`, `heavy_tail_esd_weight_surgeon.py`, `thinking_loop_wsp_suppressor.py`, `contrastive_steering_weight_compiler.py` | Thin QR-SVD & Hill-MLE ESD (`pl_alpha in [2, 4]`) + sub-0.02% anti-loop updates inside $P_0$ |
| **Folded Final-Norm Calibration** | `logit_calibration_temperature_surgeon.py`, `folded_norm_ece_calibrator.py`, `domain_matched_moe_arm_composer.py` | Exact affine 2-matmul search over $(\lambda_{\text{ln}}, \tau_{\text{temp}})$ folding $1/\tau_{\text{temp}}$ into final `RMSNorm`/`LayerNorm` ($25\times$ matmul speedup) |
| **Monotone ANSE Ratchet & Vault** | `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `wsp_formal_invariant_verifier.py`, `lean4_wsp_theorem_registry.py` | Stage commit iff $\Delta E < 0$ and $E_0/D_0 \le +1.0\%$; Lean 4 machine-checked invariants (`0 sorry`) |
| **4-Bit NF4 Export & SPRT Canary** | `quantization_compensated_wsp.py`, `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | NF4 error absorbed into $P_0$-weighted LoRA/Hessian blocks; Wald SPRT progressive traffic ramp |

---

## 2. Periodic 4-Hour NeuroController Workflow Stages

The 4-hour periodic NeuroController workflow executes five gated stages in deterministic sequence:

1. **Stage 1 — CPU Weight-Improvement Ratchet Audit & Slice Triage:**
   - Audits CPU-only (`CUDA_VISIBLE_DEVICES=""`, `MemAvailable >= 11.0 GiB`) experiment directories without re-evaluating certified holdout vaults (`skip=0, 160, 320, 480, 640`, `pilot4b`).
   - Checks active process locks (avoiding duplicate CPU runners while `pilot6_multicampaign_synergy_and_t7.py` is running) before triggering downstream ratchets.
   - Syncs merged PR states in `neural_control_backlog.json` and prepares the next pending `NEURAL-CTRL-*` slice on a clean branch off `origin/main`.
2. **Stage 2 — Cleanroom Module, 5-Scenario Test Suite & Repository Docs Authoring:**
   - Implements the target module in `src/hivemind/swarm/` with AGPL-3.0 copyright header, zero stubs, zero forbidden patterns, and zero network/subprocess egress.
   - Authors 5 deterministic CPU scenarios (`tests/test_*_scenarios.py`, CI-safe `try/except ImportError` guard around optional `gwaya`) targeting `>= 95.0%` standalone line coverage, plus updates both `docs/NEURO_CONTROLLER_ROADMAP.md` and `docs/NEURO_CONTROLLER_IMPLEMENTATION_PLAN.md`.
3. **Stage 3 — Independent Scenario Verification, Standalone Coverage, Ruff & 5-Layer Security Gate:**
   - Runs scenario tests alongside repository compliance suites (`test_agpl3_ip_protection.py`, `test_static_numbers_have_artifacts.py`, `test_no_lateral_movement.py`).
   - Measures standalone line coverage (`python -m coverage run` without `COVERAGE_CORE=ctrace`), runs `ruff check`, runs `ZeroStubAudit`, verifies `pr_gatekeeper.forbidden_scan`, checks `OwnerOrderGuardrail`, and validates `PRSecurityScanner` with `0` findings.
4. **Stage 4 — Review-Ready Slice Packaging & Telemetry Receipt Logging:**
   - Stages verified slice files cleanly with owner identity and logs JSONL telemetry receipts (leaving final publication and release actions exclusively to the Phase 4 owner workflow).
5. **Stage 5 — Plan & Roadmap Progress Sync:**
   - Records latest empirical T4 / CPU verdicts, standalone coverage, and slice completion status in tracking documents.

---

## 3. Implementation Status of `NEURAL-CTRL-DYN-39`

- **Target Module:** [`src/hivemind/swarm/prefix_cached_causal_attribution.py`](../src/hivemind/swarm/prefix_cached_causal_attribution.py) — **Implemented**
- **Scenario Suite:** [`tests/test_prefix_cached_causal_attribution_scenarios.py`](../tests/test_prefix_cached_causal_attribution_scenarios.py) — **Implemented (5 deterministic CPU scenarios: S1..S5)**
- **Prior Slice (`NEURAL-CTRL-DYN-38`):** [`src/hivemind/swarm/domain_matched_moe_arm_composer.py`](../src/hivemind/swarm/domain_matched_moe_arm_composer.py) — **Merged (`#1043`)**

### Implemented Capabilities & Mathematical Guarantees

1. **`compute_prefix_cached_layerwise_attribution(blocks, embeddings, eval_tail_fn, *, batch_size=24)`:**
   - Caches running prefix hidden states $h_0, h_1, \dots, h_{L-1}$ from a single unablated $L$-layer pass so all $L$ single-layer router/MoE knockouts require $L + L(L+1)/2$ block evaluations instead of $L^2$ (approaching $2.0\times$ speedup) with bitwise/float32 numerical parity (`< 1e-6` max diff) and exact anchor detection (`harm_nats <= -0.010`).
2. **`detect_router_expert_coadaptation_mismatch(router_arm_logits, expert_arm_outputs, coadapted_nll, grafted_nll, *, max_mismatch_nats=0.02)`:**
   - Quantifies forward/reverse/symmetric routing KL divergence, JSD, top-$k$ routing overlap, output alignment, and NLL co-adaptation loss (`coadaptation_mismatch_nats = grafted_nll - coadapted_nll`), directly capturing the `PILOT5` phenomenon where cross-arm grafting Arm B experts onto frozen-router Arm D degraded `selection_skip560` to `+0.0080` nats while co-adapted `WSP_C3_B_rescued` achieved `-0.2964` nats.
3. **`execute_coadaptation_guarded_surgery(...)`:**
   - Synthesizes and evaluates both `cross_arm_graft` and `coadapted_rescue` (selective FoMoE ghost sinks + $P_0$ null-space projection + 2-fold OOF multi-gate contrastive router suppression + 2-matmul folded LayerNorm calibration), automatically vetoing mismatched cross-arm router grafts in favor of co-adapted rescue under `ANSE Delta E < 0` and `E0 / D0 <= +1.0%` guards, with atomic bitwise rollback (`final_state_sha256 == expected_rollback_sha256`) and canonical SHA-256 provenance receipt (`verify_coadaptation_receipt_provenance`).
