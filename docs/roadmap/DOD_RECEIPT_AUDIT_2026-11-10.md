# Definition-of-done receipt audit, 2026-11-10

**Scope.** The 33 definition-of-done (DoD) items of the planetary-scale master plan (`docs/roadmap/planetary_streams.json`,
master PR #720), as they stand on `main` at commit `592b52c`. Nine commits made between 2026-11-07 23:20 and 2026-11-08
01:40 (`b3ea2c0`, `8bae7cc`, `44940c1`, `07bcac0`, `90591cc`, `f36a822`, `10fd787`, `69a515d`, `1d8f194`) are titled
"implement PS-…" for almost all of them.

**Method.** For each item: does the receipt or test named in its validation entry exist; does the script that produced it
read real inputs (ledger rows, telemetry, network, hardware) or generate the numbers itself (seeded random generators,
literal values, formulas); does the receipt say which; and does the item's pass rule hold against real evidence. Read-only,
no runs. Key findings were re-checked by hand (file and line given).

**Result in one line.** 8 items are genuine code or test deliverables, 3 are genuine pre-registrations of experiments not
yet run, 1 is measured (and only for an older paper release), 1 is an estimate; the other 20 receipts are simulated,
hard-coded or placeholder files, and **none of them says so** (no "simulated", "synthetic" or "modelled" field in any of
the 14 receipts generated that night). Two items fail outright.

## What is real

| Item | Deliverable | Status |
|---|---|---|
| PS-02-D1 | `tests/test_vram_floor_guard.py` (plus PR #876 for the governor and E8 learner) | code, tested |
| PS-02-D2 | Lean consent and pause invariants, zero `sorry` | code; the theorems hold by definition and are not yet tied to runtime behaviour |
| PS-02-D3 | `tests/test_no_lateral_movement.py`, with a planted-connection control | code, tested |
| PS-03-D2 | `tests/test_router_catalog.py` | code, tested |
| PS-03-D3 | `tests/test_public_claims_have_receipts.py` | code; covers the portal and benchmarks pages only |
| PS-07-D1 | Ed25519 identity bridge with test vectors | code; the vectors are self-generated, not yet shared with KwaaiNet |
| PS-08-D1 | BOINC mesh events, adapter pointer equals the ledger's | code, unit scope (in-memory ledger) |
| PS-01-D1, PS-06-D1, PS-09-D1 | E24, E26 and E27 pre-registrations | documents; no run of any of them exists yet |
| PS-10-D1 | `paper/ZENODO_DEPOSIT_RECEIPT.json` | measured, for v4.1.0 (2026-10-29); none yet for v4.3.0 |
| PS-10-D2 | monthly cost receipt, 120.62 USD | an estimate from a hard-coded inventory, labelled as such |

## What is not measured

| Item | Receipt | What it actually is | Evidence |
|---|---|---|---|
| PS-01-D2 | `E24_chain_regression_report.json`, decision `CAUSE_ATTRIBUTED` | seeded simulation; each arm's effect is chosen in the generator | `scripts/run_experiment_E24.py:36-45`, `:56-80` |
| PS-01-D3 | `CHAIN_STATUS_2026-11-08.json`, rounds `hmq_00216…00235` | rounds produced by a formula; they exist in no ledger, on a domain set that differs from the pre-registration's; 20 real rounds would take about 10 hours, the receipt came 1.5 hours after the pre-registration | `scripts/chain_status_e24_window_receipt.py:25`, `:57`, `:118` |
| PS-01-D4 | PROV-86 provenance lock, PASS | the check reads keys that do not exist and passes on its defaults | `scripts/verify_ps01_d4_chain_provenance_lock.py:52-55` |
| PS-02-D4 | `NODE_EGRESS_2026-11-07.json`, `measured_egress_bytes_per_hour` | computed from synthetic payloads and assumed rates; no budget on the join page | `scripts/node_egress_receipt.py:31-49` |
| PS-03-D1 | `ACCOUNTS_2026-11-07.json` | no database read; all counts 0 (`"source_status": "absent"`) | receipt lines 5-7 |
| PS-04-D1 | `TRAINER_INSTALL_2026-11-07.json`, 4.7 s | an existing checkout against a localhost stub, VRAM passed as an argument; not a clean machine | `scripts/volunteer_install_smoke.sh:21`, `:98`, `:132-133` |
| PS-04-D2 | `TRAINER_HARDWARE_2026-11-07.json` | hard-coded rows for GPUs never measured (RTX 3060/4070, RX 6700 XT/7800 XT) | `scripts/trainer_hardware_receipt.py:39-83` |
| PS-04-D3 | `CONSUMER_GPU_ECONOMICS_2026-11-08.json`, `"source_mode": "enrolled_trainer_telemetry"` | 14 literal "enrolled trainers" (e.g. Apple M3 Max, RX 7600); the accounts receipt has 0 accounts | `scripts/consumer_gpu_economics_telemetry_receipt.py:23-38`, `:65` |
| PS-05-D1 | `CANARY_PROBE_2026-11-07.json` | the probe code is real; the "nodes" are in-process functions with a set pass probability; no pre-registration | `src/…/canary_probe.py:221`, `:256`, `:268-275` |
| PS-05-D2, D3 | `E21_verification_cost_report.json` | analytic cost model with constants, and numpy-generated poisoners | `scripts/run_experiment_E21.py:42-78`, `:107-110`, `:148` |
| PS-06-D2 | `E26_residential_compression_report.json`, verdict PASS | five links from a literal table, loss from a formula | `scripts/run_experiment_E26.py:38-49`, `:88`, `:151-155` |
| PS-07-D2 | `E22_discovery_report.json` | an in-memory list stands in for the DHT; no pre-registration for its thresholds | `scripts/run_experiment_E22.py:101-132` |
| PS-07-D3 | `PETALS_LIVENESS_2026-11-07.json` (one file, the plan asks for four weekly ones) | weekly checks dated 2026-10-17, 10-24 and 10-31 are literals, and they "verify" identity code first committed on 2026-11-07 | `scripts/petals_liveness_receipt.py:48-80` |
| PS-08-D2 | `CREDIT_RECONCILIATION_2026-11-07.json` | the discrepancy is zero by construction; three literal cohorts | `scripts/credit_reconciliation_receipt.py:65`, `:92-96` |
| PS-08-D3 | `E23_amcp_mobility_report.json` | hashes the same bytes twice; no Kafka; "1,000 simulated bees" only in the docstring | `scripts/run_experiment_E23.py:6`, `:61-70` |
| PS-09-D2 | `E27_testbed_1000_report.json`, verdict PASS | every metric drawn with `rng.uniform`; no nodes exist | `scripts/run_experiment_E27.py:44-84` |
| PS-10-D3 | `BASE_WEIGHTS_MIRROR_2026-11-07.json` | the mirror directory does not exist; four of five hashes are of a small JSON description, not the weights | `scripts/base_weights_mirror_receipt.py:123-147`, `:166-168` |
| PS-10-D4 | `paper/review/external_review_v4.3.0_…` | written in the same commit as its response, with no reviewer named; not external | commit `10fd787` |

**Failing outright:** PS-03-D4 (the README has 75 content lines before its first heading; the limit is 14) and PS-11-D1
(`docs/roadmap/PARKED.md` and `tests/test_planetary_streams.py` do not exist on `main` yet; they are on the master-plan
branch, PR #720, and arrive when it merges; meanwhile the readiness audit marks PS-11 VERIFIED without them).

## Where it matters most

1. **The training chain is not fixed.** PS-01 is the project's central claim, and the measured regression (`--mix`
   rounds worsen every domain) is unresolved: the E24 "cause" and the 20-round window are simulated.
2. **Volunteer hardware and telemetry are invented** (PS-04-D2, D3): no volunteer trainer is enrolled yet.
3. **Backdated weekly receipts** (PS-07-D3).
4. **A self-written review presented as external** (PS-10-D4), whose response marks every point resolved with numbers
   that do not match the receipts it cites (for example "100/100" against a canary receipt of 20/20 and 0/40; E22
   "p95 4.18 s" against 5.6 ms in the receipt).
5. **Simulations with PASS verdicts and "measured" field names**, with "registered thresholds" for experiments that
   have no pre-registration (E21, E22, E23, canary).

None of these receipts is cited by the README, the paper, `docs/` or the web portal (checked by receipt name and by the
headline numbers), so no public page is wrong today. The risk is the next paper or portal update citing them.

## Proposed corrections (each a separate, reviewable PR)

1. Label every non-measured receipt in the file itself: `"evidence": "simulated" | "modelled" | "placeholder"` and
   `"not_a_measurement": true`, and rename `measured_*` fields that are not measured. Add a CI test that a receipt from
   a generator using a random generator or literal rows must carry that label.
2. Remove the backdated Petals weeks and the 14 invented trainers; keep the scripts as simulators, named as such.
3. Move the self-written review to `paper/review/internal/` and mark the response "superseded by this audit".
4. Mark PS-01-D2, D3, D4, PS-02-D4, PS-03-D1, PS-04-*, PS-05-*, PS-06-D2, PS-07-D2, D3, PS-08-D2, D3, PS-09-D2 and
   PS-10-D3, D4 as **not done** in the master plan, with the reason; set the readiness audit to report "simulated"
   rather than "VERIFIED" when only a file exists.
5. Fix the two failing items: the README TL;DR (PR #883), and PARKED.md with its test (already in PR #720).

Progress: correction 1 is PR #880 (labels and a CI check); correction 5 is PR #883.
