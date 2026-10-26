# Implementation plan — Season 16 "Operation First-Flight" (live operations)

Starting point (2026-10-25): three public services serve real inference from one T4 under one ledger. The first day's record
is `artifacts/live_tests/FIRST_REAL_USER_TEST_20261025.json`; what it exposed is listed in ROADMAP §3.14.

## Gates
- **G-L1 (safe to change the live system):** a deploy cannot take traffic without passing the contract on a canary; it is
  built from main; the ledger has one writer; test accounts cannot move prices or metrics; failures alert someone.
- **G-L2 (useful to users):** new users are served at idle capacity; answer quality is measured and drives routing; the
  T4 serves resident models without cold-start failures; producers obey remote control.

## Order
1. **HM-S16-02** contract check and **HM-S16-10** ledger lock. Both protect every later change.
2. **HM-S16-01** deploy from main, using 1 as its gate; then retire `deploy/router_overlay` and `deploy/dashboard_overlay`.
3. **HM-S16-07** account flags and **HM-S16-08** alerting.
4. **HM-S16-09** dashboard source clean-up (closes #180 and the client-side credit in #185).
5. **HM-S16-03** market regimes, **HM-S16-06** residency, **HM-S16-05** remote control.
6. **HM-S16-04** answer-quality evaluation. It needs an isolated sandbox: bubblewrap is unavailable on the T4 host, so use
   the TPU VM or a gVisor/Firecracker host.

## Rules carried over
- Every reported number comes from the monitor log, the request logs or a preregistered evaluation with positive and negative
  controls; a checker that cannot fail makes the run INVALID.
- No traffic switch by `latestReadyRevisionName`; switch by the revision name the deploy printed, after checking image
  digest and ledger configuration.
- Canaries of the router run with an empty replica URL (ephemeral ledger).
