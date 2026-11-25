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
   (Prefix-cached L+L(L+1)/2 anchor protection; rank-preserving input-side P_0 LoRA injection;
    FoMoE w_B=0 ghost sinks; 2-fold OOF SNR gating; rank-r QR-SVD & ESD; unfolded safe-tau >= 1.0)
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
| **Causal Attribution & Selective Sinks** | `layer_localized_attribution.py`, `cross_layer_residual_nullspace.py`, `self_healing_checkpoint_rescuer.py`, `prefix_cached_causal_attribution.py` | Prefix-cached $L + L(L+1)/2$ knockouts; beneficial anchor layers ($h_\ell \le -0.010$ nats) bitwise immutable; rogue `w_B = 0` yields $\Delta y = 0$ |
| **Rank-Preserving Multi-Campaign Injection & OOF Gate** | `sigma_router_gate_control.py`, `orthogonal_multigate_router_surgeon.py`, `domain_matched_moe_arm_composer.py`, `prefix_cached_causal_attribution.py`, **`rank_preserving_nullspace_input_injector.py`** | Preserves co-adapted `(router, B_expert)` bitwise and injects multi-campaign donor `A` factors via $A_{\text{new}} = A_{\text{anchor}} + \sum_k \alpha_k (A_{\text{donor}, k} - A_{\text{anchor}}) P_0$ with exact rank-$r$ preservation (`0` SVD truncation error) |
| **Spectral & RepE Weight Surgery** | `lora_svd_mode_surgery.py`, `heavy_tail_esd_weight_surgeon.py`, `thinking_loop_wsp_suppressor.py`, `contrastive_steering_weight_compiler.py` | Thin QR-SVD & Hill-MLE ESD (`pl_alpha in [2, 4]`) + sub-0.02% anti-loop updates inside $P_0$ |
| **Unfolded Safe-Tau Final-Norm Calibration** | `logit_calibration_temperature_surgeon.py`, `folded_norm_ece_calibrator.py`, `domain_matched_moe_arm_composer.py`, **`rank_preserving_nullspace_input_injector.py`** | Un-multiplies prior sharpening $\tau_{\text{old}} < 1.0$ and runs exact 2-matmul affine search strictly over safe $\tau \ge 1.0$ under strict general-prose ceiling (`1.002` default) |
| **Monotone ANSE Ratchet & Vault** | `monotone_wsp_stage_composer.py`, `auto_wsp_campaign_scheduler.py`, `wsp_formal_invariant_verifier.py`, `lean4_wsp_theorem_registry.py` | Stage commit iff $\Delta E < 0$ and $E_0/D_0 \le +1.0\%$; Lean 4 machine-checked invariants (`0 sorry`) |
| **4-Bit NF4 Export & SPRT Canary** | `quantization_compensated_wsp.py`, `auto_open_weight_orchestrator.py`, `shadow_canary_wsp_traffic_router.py` | NF4 error absorbed into $P_0$-weighted LoRA/Hessian blocks; Wald SPRT progressive traffic ramp |

---

## 2. Periodic 4-Hour NeuroController Workflow Stages

The 4-hour periodic NeuroController workflow executes five gated stages in deterministic sequence:

1. **Stage 1 — CPU Weight-Improvement Ratchet Audit & Slice Triage:**
   - Audits CPU-only (`CUDA_VISIBLE_DEVICES=""`, `MemAvailable >= 12.0 GiB`) experiment directories without re-evaluating certified holdout vaults (`skip=0, 160, 320, 480, 640, 800`, `pilot4b`, `pilot7`).
   - Checks active process locks (avoiding duplicate CPU runners while `pilot8_granite31_3b_moe_wsp.py` is running) before triggering downstream ratchets.
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

## 3. Implementation Status of `NEURAL-CTRL-DYN-40`

- **Target Module:** [`src/hivemind/swarm/rank_preserving_nullspace_input_injector.py`](../src/hivemind/swarm/rank_preserving_nullspace_input_injector.py) — **Implemented**
- **Scenario Suite:** [`tests/test_rank_preserving_nullspace_input_injector_scenarios.py`](../tests/test_rank_preserving_nullspace_input_injector_scenarios.py) — **Implemented (5 deterministic CPU scenarios: S1..S5)**
- **Prior Slice (`NEURAL-CTRL-DYN-39`):** [`src/hivemind/swarm/prefix_cached_causal_attribution.py`](../src/hivemind/swarm/prefix_cached_causal_attribution.py) — **Merged (`#1046`)**

### Implemented Capabilities & Mathematical Guarantees (`PILOT6` Grounding)

1. **`inject_donor_lora_via_input_nullspace(a_anchor, b_anchor, donor_a_list, alphas, p_null_input)`:**
   - Preserves co-adapted `(router, B_expert)` bitwise from the anchor (`WSP_S4_multigate_only`) and computes $A_{\text{new}} = A_{\text{anchor}} + \sum_k \alpha_k (A_{\text{donor}, k} - A_{\text{anchor}}) P_0$ exclusively on the input factor, preserving exact rank $r$ with zero SVD truncation error (unlike full-delta $W P_0$ SVD re-factorization) while verifying $\|B_{\text{anchor}} (A_{\text{new}} - A_{\text{anchor}}) x_{\text{prose}}^T\|_\infty \le \varepsilon$ (`2.86e-6` float32 prose invariance in `PILOT6`).
2. **`unfold_and_recalibrate_safe_layernorm_temperature(folded_ln_w, folded_ln_b, prior_tau, base_ln_w, base_ln_b, h_pre_ln, head_weight, targets, is_general_mask, *, tau_grid=(1.0, 1.02, 1.05), max_general_ratio=1.002)`:**
   - Un-multiplies prior sharpening $\tau_{\text{old}} < 1.0$ (`prior_tau = 0.98`) from folded final LayerNorm weights/biases and recalibrates strictly over safe $\tau \ge 1.0$ using exact 2-matmul affine logit linearity under a strict general-prose ceiling (`max_general_ratio = 1.002`), rescuing `E0 > +1.0%` boundary drift.
3. **`execute_verified_multicampaign_nullspace_injection(...)`:**
   - Sequentially builds/resolves per-layer prose null-space projectors $P_0^{(\ell)}$ on `CrowdV1WithEngram` / `CrowdLoRAMoEMLP` or structured block dicts, injects multi-campaign donor input factors (`T7` code/math + `C3_A` science), applies unfolded safe-tau LayerNorm recalibration, and enforces `ANSE Delta E < 0` and `E0 / D0 <= +1.0%` with atomic bitwise rollback and SHA-256 provenance receipts (`verify_nullspace_injection_receipt_provenance`).
