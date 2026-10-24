# The HiveMind computing coin: BOINC-style verified credit, work-backed issuance and a Nash-equilibrium compute market

Owner's brief (2026-10-24): *propose the BOINC computing-coin model for HiveMind and create a Nash-equilibrium model
to distribute the demand and the consumers of HiveMind computing well.* This document is the proposal of record;
the code is `src/hivemind/economy/compute_coin.py` (pure functions, torch-free), tests in
`tests/test_compute_coin_market.py` (12, five property-based), the simulation in `scripts/run_compute_market_sim.py`
with its pre-registration `artifacts/economy/compute_market_prereg.json` and report `compute_market_report.json`.
Status: **proposed**, not adopted. It becomes the ledger's rule only after `HM-S12-02` reproduces the results on real
receipts and the owner signs the change as a `change_security_policy` order through the firewall.

## 1. What exists and what is missing

`hivemind.swarm.ledger.HiveSwarmLedger` already has a coin, **$HIVE**: 21,000,000 cap, Bitcoin-style block reward
(50 $HIVE halving every 10,000 blocks) split among training miners, storage provers and inference, with Byzantine
slashing and a guild insurance pool. Two things are missing. First, issuance is **per block, not per work**: a block
mined for a round with ten workers mints the same as one with a thousand, so the coin does not measure compute.
Second, there is **no price**: consumers who want training or inference have no way to express how much they value
a unit of compute, and workers have no signal telling them where their GPU is worth most. The model below adds both
and keeps everything else (cap, slashing, insurance) unchanged.

## 2. Credit: the BOINC unit, granted only for verified work

BOINC's unit is the *cobblestone*: 1/200 of a day of one GFLOPS, i.e. **432 GFLOP**. HiveMind keeps the unit and
changes the verification: BOINC validates by replication (quorum = 2, wasting half the compute); HiveMind replays a
10 % sample of segments with the deterministic spot-check validator (`DeterministicSpotCheckValidator`, execution-
class matched after E12) and admits deltas through Multi-Krum. `verified_credit(claim, replicas)` then applies
three rules, in order:

1. **No verification, no credit.** A failed spot-check or a delta rejected by Multi-Krum earns exactly zero; there
   is no partial credit for "almost right" work, because partial credit is what a free-rider farms (`FL-02`).
2. **Class peak cap.** A worker cannot claim more FLOP than its execution class can do in the wall time it reports:
   `credit <= class_peak_gflops x wall_seconds`. The peaks come from the T4/RTX 2070 benchmarks already in the
   repository; an over-claim is capped, never rewarded.
3. **Canonical replica.** When a workunit *was* replicated, every verified replica is capped at the smallest
   verified claim (BOINC's canonical-result rule). An unverified replica cannot drag the canonical value down.

## 3. Issuance: coins per verified cobblestone, under the existing cap

`EmissionSchedule` mints `e(t) = e0 x d^t` $HIVE per verified cobblestone in epoch `t` (defaults `e0 = 0.01`,
`d = 0.999`), so supply tracks verified work and decays with time; the ledger's 21,000,000 cap is a hard ceiling
applied pro rata inside an epoch (`settle_epoch` reports `cap_binding`). Consumers pay for compute in $HIVE; each
payment is split **80 % to the queue's workers pro rata to verified credit, 15 % to validators, 5 % burned**. The
burn is the demand-side sink: when demand is high the coin becomes scarcer, which is the only monetary policy this
proposal needs. `apply_settlement` pushes a settlement into the ledger through its existing `credit_reward` so that
slashing and reputation stay exactly where they are.

## 4. The market: two games with closed-form Nash equilibria

The design goal is that *nobody can gain by moving*: a worker cannot earn more by shifting GPU-hours to another
queue, a consumer cannot buy more value by re-bidding. That is the definition of a Nash equilibrium, and both sides
of the market are chosen so that the equilibrium exists, is unique and can be computed in closed form and
*certified* rather than hoped for.

### 4.1 Consumers: proportional allocation (Kelly mechanism)

Each queue (a training campaign, an inference ring) has a capacity in cobblestones per epoch. Consumers submit bids
`w_j` in $HIVE and receive the share `x_j = w_j / sum w`. With price-anticipating consumers whose value for the whole
capacity is `v_j`, the unique Nash equilibrium (Johari & Tsitsiklis 2004, *Efficiency loss in a network resource
allocation game*, Math. Oper. Res. 29(3)) is

    x_j = max(0, 1 - W / v_j),   sum_j x_j = 1,   w_j = W x_j,

where `W`, the total payment, is the market price of the queue. `kelly_equilibrium` solves it by bisection. Their
theorem bounds the price of anarchy by 4/3: the equilibrium captures **at least 75 %** of the welfare an omniscient
planner would reach. Consumers who value the queue little are priced out (their share is zero), which is what
distributes *demand*: a campaign nobody values empties itself.

### 4.2 Workers: proportional-share contest (Tullock)

A queue's pool `P` (consumer payments routed to it plus the epoch's emission) is paid out in proportion to verified
credit. A worker with marginal cost `c_i` $HIVE per cobblestone (its energy price divided by its measured
throughput) choosing supply `s_i` earns `P s_i / S - c_i s_i`. This is a Tullock contest with linear costs; its pure
Nash equilibrium is unique (Szidarovszky & Okuguchi 1997, Games Econ. Behav. 18) and closed-form over the active set:

    S = (n - 1) P / sum c,   s_i = S (1 - (n - 1) c_i / sum c),

workers whose implied supply is not positive stay out. `tullock_equilibrium` computes it and `is_epsilon_nash`
certifies it by trying every unilateral deviation on a grid. Two consequences matter for the design: the
equilibrium price per cobblestone `P / S` exceeds the cheapest cost (workers earn rents, which is what pays for
GPUs), and expensive workers are excluded by competition rather than by rule.

### 4.3 Several queues: routing and the certificate

With several queues the worker's problem is where to put each cobblestone of capacity. `route_supply_equilibrium`
runs damped best-response dynamics: each worker's exact best response is a water-filling that equalises marginal
returns `P_q B_q / (B_q + a_q)^2 - c_i` across the queues it uses (`B_q` is the others' supply plus a small reserve
`kappa_q > 0`, which keeps the share continuous at zero supply; without it an epsilon of lone supply would capture a
whole pool, a defect the first smoke test exposed). The result carries its own certificate: the largest profitable
unilateral deviation found, by the exact best response and by a log-grid of single-queue moves. A caller reads
`is_epsilon_nash`, never trusts convergence. At equilibrium marginal returns are equal across queues, so **supply
follows the pools**: a queue with more consumer money attracts more GPUs until its price per cobblestone matches the
others'. That is the operational meaning of "well distributed".

## 5. Simulation results (pre-registered, seed 2026)

| Block | Instances | Result | Hypothesis |
| :-- | --: | :-- | :-- |
| Kelly consumers (2–12, values log-uniform 0.1–100) | 300 | efficiency min **0.800**, median 0.869; all feasible | H-M1 supported (bound 0.75 respected) |
| Tullock workers (2–8, costs log-uniform 0.1–10) | 300 | **300/300** certified ε-Nash; price / cheapest cost median 2.65, min 1.55; 2.7 workers inactive on average | H-M2 supported |
| Routing (2–6 workers, 1–4 queues), undamped (run 1) | 60 | 55/60 converged and certified | H-M3 (≥ 95 %) **refuted** for the undamped iteration |
| Routing, damped (relaxation 0.5, run 2) | 60 | **60/60** converged and certified | H-M3 supported for the damped solver |

Controls passed in both runs: the Johari–Tsitsiklis near-worst instance (one consumer at 1 against fifty at 1/2)
gives efficiency 0.758, inside the registered [0.75, 0.77]; symmetric workers and equal consumers split exactly.
The undamped run is kept as `artifacts/economy/compute_market_report_run1_undamped.json`: the damping was added
*after* seeing it cycle on 5 instances with binding capacities, and the solver change is stated as such. The median
price spread across queues (0.157) is not zero because 53 of 60 instances have a capacity-bound worker; a capped
worker cannot arbitrage, so its queue keeps a rent. That is a property of scarce GPUs, not a defect of the solver.

## 6. What this does and does not protect against

- **FLOP inflation**: handled by credit (class peak, canonical replica, spot-check), not by the market.
- **Sybil workers**: splitting one GPU into many identities does not change `sum c` or total capacity; the Tullock
  payoff is per unit of verified credit, so Sybils gain nothing. Identity is the firewall's keyed hash (`HM-S11`).
- **Collusion among workers** to withhold supply raises `P / S`: not modelled; the reserve `kappa` and the emission
  floor limit the damage, and a cartel is visible as a price jump in the audit. Open.
- **Dynamic entry and exit** across epochs, and **real energy prices** feeding `c_i`: not modelled; `HM-S12-03`.
- The simulation is a simulation. Nothing above is a measurement of real volunteers; `HM-S12-02` is.

## 7. Season 12 tasks (ROADMAP §3.10, "Operation Fair-Market")

1. `HM-S12-01` — this proposal, code, tests and simulation (delivered).
2. `HM-S12-02` — replay the E13/E8 receipts through `verified_credit` and `settle_epoch`; compare minted $HIVE per
   verified cobblestone with the block-reward schedule; report both. Adoption decision by owner order.
3. `HM-S12-03` — cost model: `c_i` from measured W per GFLOPS on the T4 and RTX 2070 and a configurable energy
   price; entry/exit dynamics over 100 epochs with the emission decay.
4. `HM-S12-04` — queue API: consumers bid through the BOINC coordinator; bids are signed orders (`HM-S11-02`),
   the firewall's `DELEGATED` receipt is the consumer's invoice.
5. `HM-S12-05` — collusion and withholding red team on the market (vectors added to the pen-test corpus).
6. `HM-S12-06` — Lean 4: the two closed forms (Kelly FOC, Tullock active-set) as theorems over ℚ, through the
   independent gate.
