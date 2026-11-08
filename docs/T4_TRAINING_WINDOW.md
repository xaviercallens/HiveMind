# T4 training window and the 150 USD budget

> **Proof of concept (PoC):** not a production service. Send only public, non-sensitive data.

Owner decision (2026-11-04): the RTX 2070 serves inference all the time (free), the T4 VM runs as Spot a few hours a day or night for training only. `scripts/t4_training_window.sh` applies this with a GCE instance schedule. **Applied 2026-11-04 at 18:00 Paris on the owner's word (17:00-22:00 Europe/Paris), before the BOINC move and the 2070 advertisement were finished**: until they are, BOINC and every T4-served tier are off outside the window. The first scheduled stop and start show whether GCE's service agent has the permission it needs (`roles/compute.instanceAdmin.v1` if not).

## Check before applying

While the VM is off, everything on it is off.

1. **Inference:** the RTX 2070 serves the tiers users need. Today T4 producers carry `bee-stars`, and `coder`, `moe-general` and others are served from the VM; check `/v1/models` and each tier's residency with the T4 producers stopped. `bee-stars` needs its adapter on the 2070 (`deploy/rtx2070/`).
2. **BOINC:** the project server (Apache, MariaDB, daemons, work generator) runs on this VM. Volunteer trainers get no work and cannot report outside the window. To let other HiveMinders train around the clock it must move to an always-on small VM, which adds about 15-25 USD a month.
3. **Other work on the VM:** Claude Code sessions, Redis, the local Kafka and other projects on the data disk stop with it. Long jobs must fit the window or resume.
4. **Boot:** units must start by themselves (`deploy/t4_node/install_node.sh` enables them). `hivemind-kafka-t4-node` stays off unless the voters are started.
5. **Training time:** the owner chose 17:00-22:00 (Europe/Paris). The governor's training window matches it (`--night-start-h 17 --night-end-h 22`), so training starts after 1800 s without a request and with 4,300 MiB free: about 4.5 hours a day; `coder` must not be pinned in VRAM.

TPU is not an option for this chain today: the QLoRA workunits use 4-bit bitsandbytes on CUDA.

## Estimated bill

Approximate list prices (see `scripts/d1_cost_estimate.py`), Cloud Run from measured usage, after the 1 TB -> 100 GB and 2 TB -> 800 GB shrinks, with the Kafka voters stopped and no temporary VM. Recompute with `--vm-hours` and `--disk-gb`.

| T4 hours per day | Estimate (USD/month) |
| :-- | :-- |
| 24 | about 272 |
| 8 | about 179 |
| 6 | about 167 |
| 4 | about 156 |

To get under 150 at 4 hours a day, prune snapshots (691 GB, about 18 USD a month; the old 2 TB disk's chain goes once that disk is deleted). The largest fixed line is `hivemind-openrouter` (about 35 USD): it is billed about 67% of the time because producer heartbeats keep waking it, so scale-to-zero saves little until heartbeats are less frequent.
