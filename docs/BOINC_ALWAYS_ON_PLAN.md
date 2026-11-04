# Moving the BOINC server to always-on infrastructure

> **Proof of concept (PoC):** not a production service. Send only public, non-sensitive data.

Owner decision (2026-11-04): the BOINC server moves off the T4 VM, which will run only 17:00-22:00 Paris, and work is delegated to HiveMinder trainers and producers, and the T4 when it is on.

## What needs a GPU and what does not

| Part | GPU? | Where |
| :-- | :-- | :-- |
| MariaDB, Apache + scheduler, upload handler, feeder, transitioner, file deleter, assimilator, work generator | no | always-on host (`hm-fed-master` plus a 30 GB data disk) |
| **Replay validator** (`hivemind_validate` -> `MoEValidator`: re-runs a training segment on `cuda:0`) | **yes** | T4 during its window, over WireGuard; results wait as "pending validation" meanwhile |
| Training and inference workunits | yes | HiveMinders' BOINC clients, the T4 when on |

## Open points, to measure before cutover

1. **Cross-class validation.** The replay check is exact only within one execution class (`docs/BOINC_PRODUCTION_SERVER.md`). A T4 may not be able to validate RTX 2070 or other volunteers' results within `1e-4`. Measure it on a real 2070 result. If it fails, volunteers' work cannot be validated by the T4 alone: each class needs a validator of its own class or a quorum.
2. **Timeouts:** nothing may expire or be reissued during about 19 hours a day without validation.
3. **Host resilience:** `hm-fed-master` is Spot. One preemption stops the WireGuard hub, the 2070 relay and BOINC together, and its IP is ephemeral, while BOINC clients need a stable project URL. A static IP in use costs about the same as an ephemeral one; a standard (non-Spot) e2-medium costs about 15 USD a month more.

## Rules for going public (today `master_url` is `127.0.0.1:8081`)

- The code-signing private key **never** goes on the internet-facing host: a compromised server could otherwise push code to volunteers' machines. Versions are signed elsewhere and copied.
- HTTPS through Caddy; invite code for account creation; upload size limit; database on the WireGuard address only, never public.
- Install from a committed snapshot. Today's `hmproj/bin/hivemind_validate` points at a Claude worktree (`feat-boinc-real-client/src`), so validation does not run the installed code and breaks if that worktree is deleted.
- Database moved with `mysqldump` (Debian 12 target, Ubuntu 22.04 source); BOINC server rebuilt from source there.
- Cutover with no workunit in flight: a new `master_url` means the T4 client detaches and re-attaches.

## Order

A. Prepare the always-on host privately (data disk, packages with services off, BOINC build). B. Copy, run side by side on loopback, split the validator, test end to end. C. Measure cross-class validation. D. Public URL, cutover. E. Only then, with the 2070 confirmed serving, apply `scripts/t4_training_window.sh` (`START=17 STOP=22`) and finish the 2 TB disk cutover (the new disk has a different UUID, so `/etc/fstab` changes with it).
