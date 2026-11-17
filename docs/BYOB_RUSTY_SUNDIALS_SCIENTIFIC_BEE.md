# Bring Your Own Bee (`BYOB`): Flagship Scientific & Numeric Rescue Bee (`hivemind/bee-rusty-sundials`)

> **Season:** `HM-S19-SUNDIALS` · **Lead-by-Example Producer:** Xavier Callens (`@xaviercallens`)
> **Public Model ID:** `hivemind/bee-rusty-sundials` (Aliases: `rusty-sundials/numeric-7b`, `hivemind/bee-scientific-sundials`, `hivemind/sundials`)
> **Gateway Endpoints:** `GET /v1/hivemind/bees/rusty-sundials` · `POST /v1/hivemind/bees/rusty-sundials/rescue`

---

## 1. Why `hivemind/bee-rusty-sundials` Exists ("Lead by Example")

General-purpose LLMs frequently fail on scientific and numerical computing tasks because they suggest **naive explicit Euler or unprojected RK4 schemes** that blow up on stiff systems, violate physical conservation laws (energy, mass, enstrophy, quantized circulation, topological linking number), or fail on coordinate singularities.

To lead the **Bring Your Own Bee (`BYOB`)** movement by example, Xavier Callens's scientific repositories on GitHub and local producer machines have been unified into a live, open-source specialized Bee (`hivemind/bee-rusty-sundials`):

| Domain | GitHub Repository & Local Workspace | Core Mathematical & Physical Engine |
|---|---|---|
| **Core Numerical Integrators** | [`xaviercallens/rusty-SUNDIALS`](https://github.com/xaviercallens/rusty-SUNDIALS) (`/home/xavkal/xdev/rusty-SUNDIALS`) | Pure-Rust `CVODE` (`BDF`/`Adams`), `ARKode` (`ARS(2,2,2) L-stable IMEX` & `ERKStep + Dual-RRK`), `IDA` (`DAE`), `KINSOL`, Hyper-Dual exact AD `Dual::new(y,v)`, and `SchurSpectralRouter` |
| **Astronomy & Relativistic Astrophysics** | [`xaviercallens/SocrateAI-Scientific-AutoEvolve-K3xT2`](https://github.com/xaviercallens/SocrateAI-Scientific-AutoEvolve-K3xT2) & `SocrateAI-Scientific-DualScaleSimulator` | Kepler & Post-Newtonian binary orbits with `Dual-RRK` relaxation conserving Hamiltonian $H$ and angular momentum $L_z$ ($|\Delta H/H_0| < 10^{-14}$), plus desingularized stellar Lane-Emden / TOV structure |
| **Fluid Mechanics & Turbulence** | [`xaviercallens/SocrateAI-Scientific-MechanicaFluidorum`](https://github.com/xaviercallens/SocrateAI-Scientific-MechanicaFluidorum) (`/home/xavkal/xdev/SocrateAI-Scientific-MechanicaFluidorum`) | 3D/Dyadic Navier-Stokes Hypothesis U, Lean 4 `EnstrophyProductionBound` ($S_N^2 \le 2\Omega_N^3$, 52 theorems), and `ARKode ARS(2,2,2)` L-stable IMEX rescuing `Obstruction D4` explicit viscosity blowup |
| **Quantum Fluids & Superfluidity** | [`xaviercallens/SocrateAI-Scientific-QuantumFluids`](https://github.com/xaviercallens/SocrateAI-Scientific-QuantumFluids) (`/home/xavkal/xdev/SocrateAI-Scientific-QuantumFluids` & `crates/qf-pgpe`) | Projected Gross-Pitaevskii (`PGPE`) `IF-RK4` with sharp circular de-aliasing cutoff $k_{\text{cut}} = k_{\max}/2$ (`K1`–`K4`) and Lean 4 `VortexWinding` / `QuantizedCirculation` ($\Gamma = q \cdot h/m$, 257 theorems) |
| **Genetics & Biophysical Kinetics** | [`xaviercallens/SocrateAI-Scientific-Genetic`](https://github.com/xaviercallens/SocrateAI-Scientific-Genetic) | Stiff Hopfield-Ninio kinetic proofreading & ART reverse-transcription (`CVODE BDF`, $\kappa \sim 10^4$), Călugăreanu-White-Fuller DNA supercoiling conservation ($Lk = Tw + Wr$), and 1D KJMA genome replication kinetics |

---

## 2. Three Core Capabilities

### 2.1 Underlying Physical & Mathematical Structural Analysis
`analyze_physical_mathematical_problem(prompt_or_equation, domain_hint)` inspects any scientific prompt or differential equation and diagnoses:
- **Mathematical Structure:** Stiffness ratio $\kappa = |\lambda_{\max}|/|\lambda_{\min}|$, symplectic/Hamiltonian manifold structure, parabolic vs. dispersive operator splitting, or coordinate singularities ($\xi \to 0^+$).
- **Physical Conservation Laws:** Hamiltonian $H$, angular momentum $L_z$, inviscid triad energy telescoping $\sum u_n \text{NL}_n = 0$, local enstrophy production bound $S_N^2 \le 2\Omega_N^3$, unitary $L^2$ condensate norm $N = \int |\psi|^2 dx$, quantized vortex circulation $\Gamma = q(h/m)$, stoichiometric simplex $\sum y_i = M_0$, and DNA linking number $Lk = Tw + Wr$.
- **Why Naive Schemes Fail:** Explicit Euler amplification $|1 + \Delta t \lambda| > 1$, parabolic CFL violation (`Obstruction D4`), UV cubic aliasing, or negative biochemical concentration overshoot.

### 2.2 Deterministic Numeric Calculation Rescue (`POST /v1/hivemind/bees/rusty-sundials/rescue`)
`rescue_numeric_calculation(prompt_or_equation, domain, steps, dt)` runs both the **naive scheme** (recording its exact failure or drift) and the **rescued `rusty-SUNDIALS` solver**, returning a verified trajectory and invariant receipt:

| Scientific Domain | Naive Scheme Failure | Rescued `rusty-SUNDIALS` Solver | Rescued Invariant Drift | Improvement Factor |
|---|---|---|---|---|
| **Astronomy** (`astronomy`) | Explicit Euler spirals outward ($\|\Delta H/H_0\| = 2.09 \times 10^{-1}$) | `arkode::ERKStep` + `Dual-RRK` Hamiltonian Relaxation | **$1.78 \times 10^{-15}$** | **$1.18 \times 10^{14}\times$** |
| **Fluid Mechanics** (`fluid_mechanics`) | Explicit Euler overflows to `Inf` (`Obstruction D4`, $\Delta t \nu k_7^2 = 8.192 > 2$) | `arkode::ARKStep ARS(2,2,2)` L-stable IMEX + `EnstrophyProductionBound` ($S_N^2 \le 2\Omega_N^3$) | **$3.80 \times 10^{-17}$** | **$1.00 \times 10^{22}\times$** |
| **Quantum Fluids** (`quantum_fluid`) | Unprojected Explicit Euler violates unitarity ($\|\Delta N/N_0\| = 5.36 \times 10^{-2}$) | `qf-pgpe::ComplexField2D` `IF-RK4` ($k_{\text{cut}} = k_{\max}/2$) | **$3.51 \times 10^{-14}$** | **$1.53 \times 10^{12}\times$** |
| **Genetics** (`genetics`) | Explicit Euler overshoots to negative concentrations ($\Delta t k_{\text{on}} = 50 > 2$) | `cvode::BdfSolver` + `Dual-RRK` DNA Linking Projection ($Lk = Tw + Wr$) | **$1.94 \times 10^{-13}$** | **$5.16 \times 10^{18}\times$** |

### 2.3 Dataset & Local/Swarm Model Retention (`BeeRetentionStore`)
1. **Verified SFT + DPO Dataset (`data/rusty_sundials_scientific_bee/rusty_sundials_numeric_rescue_dataset.jsonl`):**
   - 12 multi-step scientific numeric rescue problems across `astronomy`, `fluid_mechanics`, `quantum_fluid`, and `genetics`.
   - Each record includes the full physical/mathematical diagnosis, chosen `rusty-SUNDIALS` implementation, rejected naive implementation ($E = 10^6$), thermodynamic energy gap $\Delta E$, and verified conservation receipts.
2. **Checkpoint Retention Gate (`BeeRetentionStore`):**
   - Producer machines retrain `hivemind/bee-rusty-sundials` during nocturnal (`22:00–07:00`) or daytime micro-idle windows (`idle >= 30s`, `<1.0s` preemption).
   - A candidate LoRA adapter is promoted and retained on disk (`latest_promoted.json` + SHA-256 manifest) **if and only if** all 4 domain rescue gates pass and $\Delta E = E_{\text{child}} - E_{\text{parent}} < 0$.

---

## 3. Quickstart for Xavier's Machines & Community Producers

```bash
# 1. Build the 4-domain SFT+DPO dataset, run all 4 numeric rescue solvers, and retain the verified checkpoint:
uv run python scripts/train_and_serve_rusty_sundials_bee.py --mode all

# 2. Advertise and serve hivemind/bee-rusty-sundials on the live OpenRouter gateway while serving coder traffic:
uv run python scripts/hivemind_producer_agent.py \
  --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app \
  --endpoint https://YOUR_SHIELD_ENDPOINT \
  --model bee-rusty-sundials=qwen2.5-coder:7b \
  --model coder=qwen2.5-coder:7b
```
