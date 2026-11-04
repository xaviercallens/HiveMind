# MVP owner decisions

> **Proof of concept (PoC):** not a production service. Send only public, non-sensitive data.

Decisions given by the owner (xaviercallens) in the Claude Code session of 2026-11-04, recorded here so the scorecard and the next sessions can rely on them.

| Topic | Decision | Consequence in the repository |
| :-- | :-- | :-- |
| Two physical GPUs for A1 | T4 (`socreateai-agora-hermes-node1`) and the RTX 2070 (`XAVGAMSRV`) both run producers for the same tier; one producer may be killed during the test | `scripts/a1_*` run it; the owner (or Antigravity) must start the 2070 producer, no Claude session has a shell on that host |
| Monthly budget | 150 USD (raised from 100 later the same day). HiveMinders contribute as trainers, producers and consumers; the RTX 2070 serves inference all the time for free; the T4 VM runs as Spot a few hours a day or night for training only; the 1 TB and 2 TB disks shrink; Kafka voters stay Spot and stop when idle | `scripts/d1_cost_estimate.py` compares the bill with it (`artifacts/mvp/D1_gcp_cost.json`); `scripts/kafka_voters.sh` |
| `hivemind-openrouter` scaling | `--min-instances=0`, autoscale when used; AGENTS.md changed accordingly | `deploy/gcp_openrouter/deploy.sh` now matches AGENTS.md and the live service |
| A2 second host | A Spot VM may be created on GCP if needed | `hm-fed-master` (already running) is used first; any new VM is deleted after the run |
| **F1 page figures** | **Keep the 48 typed-in figures** | **WAIVED, not passed**: they stay unverified, the ratchet test (`tests/test_static_numbers_have_artifacts.py`) stops new ones, and a verified replacement lowers the count. The scorecard shows `WAIVED` and does not count it as a pass |
| `--mix` stratified training | May be deployed to the protected `/opt/hivemind/t4/` copy and given GPU time | only through the reviewed installer `deploy/t4_node/install_node.sh` (it clears and restores the immutable flag and refreshes `expected_code_hashes.txt`); the governor policy (1800 s night / 7200 s day idle, 4300 MiB free VRAM) is unchanged |
| Outages | Allowed for MVP work if end users are told | maintenance mode: `HIVEMIND_MAINTENANCE_NOTICE` (header on every response, banner on HTML pages) |
| Router fallback fix and redeploy | Allowed | fix is #536 (merged); deployed from PR #566's branch as `hivemind-openrouter-00065-ner` (previous good revision `00054-kxf`) |
| RTX 2070 role | Serves `coder-lite` and `granite-moe` all the time; MoE training on hold, later only in the European night (22:00-07:00 Paris), yielding to inference | other tiers need volunteers or the T4 window |
| T4 VM | 17:00-22:00 Paris, training | `scripts/t4_training_window.sh`, applied only after the 2070 is serving and BOINC has moved |
| BOINC server | Moves to always-on infrastructure; HiveMinders train and serve as volunteers | `docs/BOINC_ALWAYS_ON_PLAN.md`; the GPU replay validator stays on a GPU host |
| GitHub token | Rotate later | the token pasted in the chat is visible in the transcript |

