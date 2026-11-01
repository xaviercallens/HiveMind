# Join HiveMind as a trainer (what works today, what does not)

Written 2026-11-01 from a read-only dry run of the join path on a CPU-only host and from the T4 operator's statement for
the night of 2026-11-01. Every status below is what the tools printed, not a promise.

## Who this is for

- You have an NVIDIA card with at least 6 GB of VRAM (consumer RTX 20xx to 50xx), or an Apple Silicon or ROCm machine, and
  you want it to train during the night window (22:00-07:00 Europe/Paris) instead of idling.
- You already serve as a **producer**: no migration is needed. The router treats `producer`, `prosumer` and `trainer` as
  the same role family for training endpoints (`TRAINER_ROLES` in `src/hivemind/router/accounts.py`). Your existing key
  works; only register a new handle if you want a separate trainer identity.

## Step 1: preflight (read-only, installs nothing)

```bash
git clone https://github.com/xaviercallens/HiveMind && cd HiveMind
python3 scripts/hivemind_volunteer_preflight.py --project-url https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app
```

## Step 2: register (skip if you already have a producer key)

```bash
curl -s https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/hivemind/register \
  -H 'Content-Type: application/json' -d '{"handle":"<pseudonym>","role":"trainer"}'
```

The API key is shown once; the handle is public, so pick a pseudonym, not your hostname. Store the key in
`~/.config/hivemind/` with mode 600 and never paste it into an issue.

## Step 3: print your federation plan

```bash
PYTHONPATH=src python3 scripts/hivemind_federate_node.py --handle <pseudonym> \
  --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app
```

It reads your GPU with `nvidia-smi`, picks the workunit class (`cuda_hivemind` at 3,500 MiB or more, `cuda_hivemind_big`
at 12,000 MiB or more, `cpu_hivemind_solver` otherwise) and the bee that fits, and lists five steps with their real status.
On the CPU-only host used to write this page the output was:

| Step | Status printed | Meaning |
|------|----------------|---------|
| preflight | READY | the command above |
| register | READY | the curl above |
| idle_gate_dry_run | BLOCKED | no GPU could be read on that host; with a card it runs `scripts/hivemind_boinc_idle_worker.py --dry-run` |
| qlora_training | BLOCKED (`no_trainer_wired`) | the idle worker does not yet lease or train a workunit; no points are earned by the worker today |
| cross_host_pilot | MANUAL | needs the operator to open UDP 51820 on the T4 and add your WireGuard key (`docs/E8_TWO_HOST_REAL_TEST.md`) |

Exit code 2 means no GPU was read. Add `--json` for a machine-readable plan.

## Step 4: the night run, honestly

- **Tonight (2026-11-01):** the T4 operator runs `night-002` at 22:02 Paris on the T4 alone. It is not federated: no
  second trainer GPU is connected. External learners cannot join it, because the learner loop runs over a signed AMCP bus
  on a WireGuard overlay whose port (UDP 51820) is not yet open and whose keys are issued by the operator by hand.
- **What you can do tonight:** run steps 1-3 and keep the plan output; if step 3 shows `cuda_hivemind` or
  `cuda_hivemind_big`, you are eligible for the first federated night. Open an issue titled
  `[Trainer] <pseudonym>: ready for the overlay` with the plan's JSON (no key inside) so the operator can add your
  WireGuard key.
- **Points:** the ruche accepts federated workunit submissions but marks them `RECEIVED_CLIENT_CLAIMS_UNVERIFIED` and
  credits 0 until server-side verification exists. Nothing on this page promises a reward.

## Operator checklist before the first federated night (owner / T4 session)

1. Tag the T4 so UDP 51820 is reachable and enable the overlay and the Kafka node at boot (one-line command held by the owner).
2. Issue WireGuard keys for the trainers that opened a `[Trainer]` issue; the 2070 is reserved at 10.77.0.5.
3. Publish the code pin and base hash the learners must match, and the VRAM floor (4,300 MiB on the T4 tonight).
4. Record the round in `artifacts/experiments/e8_real/` so the first cross-host number is measured, not announced.
