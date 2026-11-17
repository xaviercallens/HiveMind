# E8 two-host cell, T4 side (2026-10-31)

Protocol-validation cell of the E8 pilot (docs/E8_TWO_HOST_REAL_TEST.md) between two separate machines: learner 0 on the T4 VM (Linux, `socreateai-agora-hermes-node1`) and learner 1 on `XAVGAMSRV` (Windows 11, RTX 2070 host), the syncer on the T4 at `10.77.0.4:8765`.
Settings: `K=2`, `--grace off`, `--merge rda`, 12 rounds, `HM_INNER_STEPS=400`, seeds 1, 2, 3, E1 toy model, **CPU only** (neither GPU was used).
Files here: `wan_seed{1,2,3}_rda.json` (syncer reports, each names both hosts), `syncer_seed*.log`, `learner0_seed*.log`. The single-host baselines and loopback controls were produced and committed by the 2070 host (branch `feat/openrouter-5lang-i18n`).

| seed | two-host final loss | single-host baseline | two-host minus baseline | loopback minus baseline |
|---|---|---|---|---|
| 1 | 2.3397736549 | 2.3397736549 | +0.000e+00 | +0.000e+00 |
| 2 | 2.2846515179 | 2.2846515179 | +0.000e+00 | +0.000e+00 |
| 3 | 2.7570157051 | 2.7570142746 | +1.431e-06 | +0.000e+00 |

Paired two-host minus baseline, n=3: mean +4.8e-07, sd 8.3e-07. The paper's pilot reported `-0.00000 +/- 0.00000`; seed 3 differs at float32 scale. The cause was **not** investigated (different CPU and operating system is a hypothesis, not a finding).

Round latency (36 rounds): p50 1.21 s, p95 8.52 s, max 146.57 s. Excluding round 0 (which includes the second learner joining: 146.6 s, 8.5 s, 5.9 s in seeds 1 to 3): n=33, p50 1.20 s, p95 3.79 s, max 3.89 s.

Limits, stated so nobody overreads this:
* The 2070 learner reached the syncer through an **SSH port-forward over the public internet**, not the WireGuard tunnel (no WireGuard handshake from the 2070 occurred); latencies include that tunnel.
* Toy model on CPU: this validates the shipping of shards and parameters, the remote inner loop and the RDA merge across two real machines. It says nothing about GPU, LoRA, VRAM or the 2070's card, and it is not a Byzantine-filtering result (Multi-Krum keeps everyone at n=2).
* Three seeds, `K=2` only. The `K=1` cells of the paper's pilot were not repeated.
* Both reports state `completed: true`, 12 rounds, both hosts admitted in the rounds sampled; the full per-round admission list is in the JSON (`rounds`).
