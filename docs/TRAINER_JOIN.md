# Join HiveMind as a trainer

Updated 2026-11-01 (night) after the T4 operator session published the co-training calendar and per-learner signing keys. This page is the short
entry point; the operator's runbooks are the reference and win on any disagreement. They live on the branch of PR #410
until it merges:

- `docs/community/COTRAINING_NIGHTS.md`: calendar and join steps (ICS feed: `docs/community/cotraining_nights_2026-11.ics`).
- `docs/FEDERATION_V2_AMCP_AGLETS.md`, section 8 (procedure for a new participant) and section 9 (web trainer-registration contract).
- `docs/OVERLAY_OPEN_SOURCE_TUNNELS.md`, section 2 (WireGuard configuration through the T4 hub).

Invitation and slot replies: GitHub issue #431. H-Net probe schedule: issue #448.

## The next federated night

**Night 3: Monday 2 Nov 22:00 to Tuesday 3 Nov 07:00 Paris (21:00 to 06:00 UTC), run `night-003`.** Coordinator on the T4
CPU; learners are registered trainers only. Tonight (night 2) runs on the T4 alone and cannot be joined.

## Who this is for

- You have an NVIDIA card with at least 6 GB of VRAM (or an Apple Silicon / ROCm machine) and want it to train during the
  night window instead of idling.
- You already serve as a **producer**: no account migration is needed. The router treats `producer`, `prosumer` and
  `trainer` as one role family for training endpoints (`TRAINER_ROLES` in `src/hivemind/router/accounts.py`).

## What you need (from the operator's statement, 2026-11-01)

1. **A slot.** Reply on #431 **by 18:00 Paris on the day of the night** with your WireGuard **public** key, GPU model, free
   VRAM, and OS. Key pair: Linux/macOS `umask 077; wg genkey | tee hivemind-wg.key | wg pubkey` (the private key stays in
   `hivemind-wg.key`); Windows: WireGuard app, "Add empty tunnel" shows the public key. The operator answers with a learner id (`t01`-`t08`), a shard index, an overlay address `10.77.0.N`, and
   the commit to check out. Never post a private key.
2. **The overlay.** UDP 51820 on the T4 is open (tag `hm-kafka` is applied; the Kafka node and the WireGuard interface start
   at boot). The operator adds your peer (`wg set … peer <pub> allowed-ips 10.77.0.N/32`, `hm-wg-hub add-member 10.77.0.N/32`);
   you configure your side per `docs/OVERLAY_OPEN_SOURCE_TUNNELS.md` section 2.
3. **Your signing keys**, given out of band with the slot: a per-learner key `lk-<learner id>` and the coordinator's
   verification key `coord` (commit 7ad7988 on the #410 branch). Set `HIVEMIND_AMCP_HMAC_KEYS=coord:<hex>,lk-<id>:<hex>` and
   `HIVEMIND_AMCP_HMAC_ACTIVE_KID=lk-<id>`; run the learner with `--coordinator-key coord` (the coordinator runs with
   `--learner-key-prefix lk-`). Effect: nobody can submit a claim or result in another learner's name, and learners obey
   only coordinator-signed offers. **Limit, stated by the operator:** HMAC is symmetric, so whoever holds `coord` can still
   sign fake offers or a fake `run_complete` (time-wasting, never an accepted result); Ed25519 signatures would close that
   and are not built. Slots are therefore given one by one and a key is revoked by removing it from the coordinator keyring.
   Store keys with mode 600; never paste them anywhere.
4. **The repository at the pinned commit.** The code pin and base hash are checked automatically by the learner; no manual
   hash handling.
5. **Python 3.11** with `torch`, `transformers` 5.17, `peft` 0.21, `confluent-kafka`, and `bitsandbytes` (NF4).

## Before you ask for a slot: read-only checks on your machine

```bash
git clone https://github.com/xaviercallens/HiveMind && cd HiveMind
python3 scripts/hivemind_volunteer_preflight.py --project-url https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app
PYTHONPATH=src python3 scripts/hivemind_federate_node.py --handle <pseudonym> \
  --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app --json
```

`hivemind_federate_node.py` reads your GPU with `nvidia-smi`, names the workunit class (`cuda_hivemind` at 3,500 MiB or
more, `cuda_hivemind_big` at 12,000 MiB or more, `cpu_hivemind_solver` otherwise) and prints each step with its real status.
Exit code 2 means no GPU was read. On the CPU-only host used to write this page it printed steps 1-2 READY, 3-4 BLOCKED,
5 MANUAL; with a card, step 5 is the slot request above. Attach the JSON (it contains no key) to your #431 reply.

If you do not have a key yet, register one (shown once; the handle is public, so pick a pseudonym):

```bash
curl -s https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1/hivemind/register \
  -H 'Content-Type: application/json' -d '{"handle":"<pseudonym>","role":"trainer"}'
```

## Points, honestly

The ruche accepts federated workunit submissions but marks them `RECEIVED_CLIENT_CLAIMS_UNVERIFIED` and credits 0 until
server-side verification exists. Nothing on this page promises a reward.
