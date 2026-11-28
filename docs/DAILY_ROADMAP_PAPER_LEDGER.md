# Project HiveMind — Daily Roadmap & Foundation Paper Improvement Ledger

> **Non-Monetary AGPL-3.0 Charter:** Project HiveMind operates strictly under the **`0.00 EUR / 0.00 USD`** non-commercial volunteer compute-reciprocity charter (`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`) and **`AGPL-3.0`** license, with zero monetary payouts, zero cryptocurrency wallets, zero external payment links, and zero unauthorized network egress. Root `ROADMAP.md` remains unmodified; daily roadmap and paper-improvement progress is tracked exclusively in this ledger.

---

## Foundation Paper Anchor

- **Foundation Manuscript:** Project HiveMind Foundation Paper **v4.4.0** — DOI [**`10.5281/zenodo.24553269`**](https://doi.org/10.5281/zenodo.24553269) (`paper/hivemind_foundation_article.tex`).
- **Companion Extension:** `Reine-Cortex` Zero-Gradient NeuroController Extension — DOI [**`10.5281/zenodo.24542441`**](https://doi.org/10.5281/zenodo.24542441).
- **Target Paper Section:** Section `15.4` — *Opportunity `J1` (Theorem `15.1(2)`)* (Null-Space Protected Post-JEPA *Gelee Royale* Rescue), *Opportunity `J2` (`E15-J` & `P2`)* (Pre-Registered Paired-View Benchmark Builder on Natural Code & Lean 4 Proof Views), & *Opportunity `J3` (`JEPA-WSP` Critic)* (Frozen Latent JEPA World-Model Critic for `Reine-Cortex` *Operculation* and *Pheromones Royales*).

---

## Daily Roadmap Ledger (`C1`)

| Step ID | Track | Status | Paper Anchor | Target Module(s) | Test Suite |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **`DAILY-C1-S1`** | **`C1`** | **Implemented** | v4.4.0 §15.4 Opportunity `J1` (Theorem `15.1(2)`, `10.5281/zenodo.24553269`) | `src/hivemind/jepa/nullspace_jepa_rescuer.py` | `tests/test_nullspace_jepa_rescuer.py` |
| **`DAILY-C1-S2`** | **`C1`** | **Implemented** | v4.4.0 §15.4 Opportunity `J2` (`E15-J` & `P2`, `10.5281/zenodo.24553269`) | `src/hivemind/jepa/paired_view_benchmark_harness.py` | `tests/test_paired_view_benchmark_harness.py` |
| **`DAILY-C1-S3`** | **`C1`** | **Implemented** | v4.4.0 §15.4 Opportunity `J3` (`JEPA-WSP` Critic, `10.5281/zenodo.24553269`) | `src/hivemind/jepa/latent_jepa_wsp_critic.py` | `tests/test_latent_jepa_wsp_critic.py` |

---

## Step `DAILY-C1-S1` Measurable Acceptance Criteria

1. **Pure CPU Cleanroom Execution (`0 MiB` GPU VRAM):** Pure PyTorch/NumPy/stdlib CPU implementation with zero external network calls, zero subprocess invocations, zero stubs (`ZeroStubAudit` clean), and AGPL-3.0 header within the first 300 characters.
2. **Exact Algebraic Prose-Nullspace Annihilation (`Theorem 15.1(2)`):** Given `P_0 = I - V_prose^T V_prose`, associative rank-preserving projection `(s * B)(A @ P_0)` achieves `||((s * B)(A @ P_0)) V_prose^T||_F < 1e-6` while preserving specialist JEPA alignment in the orthogonal complement (`S1`).
3. **Post-JEPA *Gelee Royale* Rescue (`Opportunity J1`):** Turns a guard-failing post-round JEPA or STP LoRA delta (`D0 > +1.0%`, `+0.90` nat / `+1.37` nat regression) into an ANSE-admissible candidate (`ANSE Delta E < 0`, `D0 <= +1.0%`) (`S2`).
4. **RankMe Isotropy & Cosine-Collapse Guard:** Computes spectral RankMe effective dimensionality and pairwise cosine distance (`1 - cos > threshold`), rejecting cosine-collapsed or low-RankMe representations with atomic rollback (`S3`).
5. **Monotone ANSE Gate & SHA-256 Provenance:** Enforces `ANSE Delta E < 0` and `D0 / worst_domain <= +1.0%`, performing verified atomic bitwise rollback (`ANSE_MAXIMUM_PAIN = 1e6`) on any guard or energy failure (`S4`), emitting deterministic SHA-256 provenance receipts (`S5`), and achieving `>= 95.0%` standalone line coverage on `src/hivemind/jepa/nullspace_jepa_rescuer.py` across 5 deterministic CPU scenarios.

---

## Step `DAILY-C1-S2` Measurable Acceptance Criteria

1. **Pure CPU Cleanroom Execution (`0 MiB` GPU VRAM):** Deterministic Python stdlib and NumPy CPU implementation in `src/hivemind/jepa/paired_view_benchmark_harness.py` with zero external network calls, zero subprocess invocations, zero stubs (`ZeroStubAudit` clean), and AGPL-3.0 header within the first 300 characters.
2. **Natural Paired-View Builders & Zero-Overlap Split Integrity (`S1`):** Extracts deterministic paired views for Python (`docstring/specification <-> function body` via AST) and Lean 4 (`theorem signature <-> proof term` with zero `sorry`/`admit`/`sorryAx` holes), enforcing zero overlap across `item_id`, normalized `view_a_spec`, normalized `view_b_target`, and `pair_sha256`.
3. **Hash-Deterministic Dropout & Shuffled-Pair Control (`Arm S`, `S2`):** Evaluates `sha256(seed, step, row, k)` loss dropout via `hivemind.jepa.spec`, constructs strict Sattolo derangement negative controls (`Arm S`) alongside `Arm A` (SFT) and `Arm J` (JEPA) with zero fixed points or identical-target collisions, and quantifies matched vs shuffled CPU alignment margins.
4. **Exact Paired McNemar & `>= +3.0` pts Promotion Gate (`S3`–`S4`):** Evaluates `E15-J` (`n = 257` MBPP-sanitized) and `P2` (Lean 4) three-arm outcomes with exact two-sided McNemar `p < 0.05` (`hivemind.verification.bee_energy.exact_mcnemar_p`), `>= +3.0` percentage-point gain over `Arm A`, negative-control non-significance (`Arm S` cannot gain significantly), general-NLL guard (`<= +0.05` nat), and cosine non-collapse (`1 - cos >= 0.05`).
5. **Canonical Provenance & Standalone Coverage (`S5`):** Emits deterministic SHA-256 gate receipts verified by `verify_e15j_receipt_provenance`, passes `PRSecurityScanner`, `OwnerOrderGuardrail`, and `ZeroStubAudit`, and achieves `>= 95.0%` standalone line coverage on `src/hivemind/jepa/paired_view_benchmark_harness.py` across 5 deterministic CPU scenarios under 10 s runtime.

---

## Step `DAILY-C1-S3` Measurable Acceptance Criteria

1. **Pure CPU Cleanroom Execution (`0 MiB` GPU VRAM, `0` Backprop):** Deterministic PyTorch/stdlib CPU implementation in `src/hivemind/jepa/latent_jepa_wsp_critic.py` with zero gradient steps, zero external network calls, zero subprocess invocations, zero stubs (`ZeroStubAudit` clean), and AGPL-3.0 header within the first 300 characters.
2. **Frozen Latent JEPA Predictor & RankMe Isotropy (`S1`):** Implements frozen linear and two-layer MLP predictors `f_phi(z_ctx) -> z_hat_target` (including closed-form ridge calibration via `torch.linalg.solve`), computing stop-gradient semantic surprise `E_JEPA = ||f_phi(z_ctx) - sg(z_target)||_2^2` alongside RankMe effective-rank and cosine-anisotropy checks.
3. **Opening-Template CE Gaming Rejection (Paper §14, `S2`):** Reproduces and rejects superficial opening-template CE gaming (Section 14 `81%` full-chunk vs mid-row format artifact, stagnant or increasing `E_JEPA`, or RankMe collapse) inside CPU `Operculation d'Alveole` with `ANSE_MAXIMUM_PAIN` (`1e6`) atomic bitwise rollback.
4. **Semantic-Deficit *Phéromones Royales* & Simplex Conservation (`S3`):** Converts per-domain semantic surprise `E_JEPA` and mid-row conceptual deficits into strictly order-preserving, simplex-normalized foraging weights `p_d` (`sum(p_d) = 1.0`, `p_d > 0` on foraging domains, `0.0` on guarded general prose) packaged inside a SHA-256-signed `QueenPheromoneDirective`.
5. **Monotone ANSE Gate, Atomic Bitwise Rollback & Standalone Coverage (`S4`–`S5`):** Enforces `ANSE Delta E < 0` and `D0 <= +1.0%` with verified atomic bitwise rollback across `dict` and `nn.Module` state, verifies canonical SHA-256 provenance receipts, passes `PRSecurityScanner`, `OwnerOrderGuardrail`, and `ZeroStubAudit`, and achieves `>= 95.0%` standalone line coverage on `src/hivemind/jepa/latent_jepa_wsp_critic.py` across 5 deterministic CPU scenarios under 10 s runtime.
