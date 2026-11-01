# Federation v2: outbound-only, signed task agents over AMCP and Kafka

An approach to make joining a co-training run simple, using the AMCP bus and the Kafka cluster that already exist, and the life cycle of IBM's old **Aglets** (mobile agents) **without their mobile code**. Written 2026-11-01. What is built and measured is separated from what is only designed in section 7.

## 1. The problem, measured

Joining the first two-host run (PR #401, #407) took, for the newcomer: a WireGuard key exchange, a firewall change on the T4, an SSH key added by a person with project access (refused at first by the permission system, rightly), the same git commit on both machines, exact commands on both ends, and a person on the T4 side starting the coordinator. The 2070 host never completed the WireGuard handshake and fell back to an SSH port-forward. Each step needs a human on the other side, and each is an attack surface.

## 2. What an Aglet was, and what we keep

An Aglet (IBM Tokyo Research, 1996; open-sourced 2002) was a Java object that **moved with its code and state** between hosts running an Aglet server, and ran there. Its life cycle: *create, dispatch* (go to a URL), *arrival* (`onArrival`), *clone*, *retract* (call it back), *dispose*.
Moving executable code to machines that did not write it is exactly what a public volunteer network must not do. Deserialising an object from an untrusted peer is a classic route to remote code execution. So we keep the **life cycle, carried by signed data**, and refuse the mobility of code:

| Aglet verb | Here | Message |
|---|---|---|
| dispatch | the coordinator offers a task to whoever listens | `task_offer`: round, algorithm, the **hash of the model state**, the **hash of the code version** (a pin), parameters, deadline, quorum, the global LoRA state |
| arrival / `onArrival` | the host verifies, then decides | `check_offer`: pin equals the hash of the code **already on this host** (from the repository), base model hash equal, deadline not passed, global state hashes to what the offer says. Any failure is a refusal that says why; nothing is executed |
| clone | several learners run the same task independently | each answers with `task_claim` |
| result | the delta comes back, never code | `task_result`: delta plus the hashes of the global state it started from and the code it ran |
| retract / dispose | a round that misses its quorum is withdrawn and offered again | `task_abort` (also `run_complete`) |

Nothing is pulled in by a host except data it checks: tensors are loaded weights-only, the model is rebuilt locally from a seed and compared by hash, and the code that runs is the code the operator installed.

## 3. Why AMCP and Kafka

* **Outbound only.** A learner connects out to the bus and pulls tasks. It opens no port, needs no tunnel to the coordinator and no address of anybody else's machine.
* **Already signed, time-limited and size-capped.** AMCP envelopes carry an HMAC, a TTL of at most one hour and a 64 KiB cap, and bad messages go to a dead-letter topic.
* **Already durable and tested.** The Kafka cluster (three Spot voters plus the T4 broker over WireGuard) survived the failure tests of PR #398. A new topic, `hivemind.amcp.federation.v1`, carries the four event types above.
* **Decoupled.** The coordinator does not need to know who is listening; learners can come and go.

## 4. Design

Participant tiers (the bus is the same; only the way in differs):

| Tier | Who | Way in | State |
|---|---|---|---|
| A | machines on the WireGuard overlay (T4, the Kafka VMs, a 2070 once its tunnel is up) | direct to Kafka | **built and measured** (section 6) |
| B | an ordinary end user, no tunnel | HTTPS to the router's `/v1/hivemind/amcp/publish` and `/events`; a small **bridge inside the overlay** mirrors the router feed to and from Kafka (outbound HTTPS from the bridge, so Cloud Run never needs a route into the overlay) | **designed, not built** |
| C | BOINC volunteers | unchanged: the BOINC client and project | exists, separate |

Round protocol (`scripts/e8_amcp_round.py`, `src/hivemind/swarm/federation_agents.py`): the coordinator publishes an offer carrying the current global LoRA state (about 38 KB for the toy model); allowed learners verify it, claim it, train `inner_steps` on their own shard, and publish only the **delta**; the coordinator checks each result (run id, round, learner on the allowed list, the global-state hash the learner started from, the code pin, the delta's names, shapes, finiteness and norm) and takes the token-weighted mean with an outer step with momentum. A round closes when every allowed learner answered, or at its deadline with at least `quorum` results; below quorum it is aborted and re-offered (up to `--max-retries`).

## 5. Security model, and where it stops

* **One trust domain.** The signing key is a shared symmetric secret (`HIVEMIND_AMCP_HMAC_KEYS`, never committed). Whoever holds it can sign anything. Fine for a small group; a public tier needs per-participant keys or asymmetric signatures (not built).
* **Checks that exist:** allowed learner list, code pin, base and global hashes, delta shape, finiteness and norm bound, size cap, TTL, weights-only decoding, the learner-side VRAM floor (4,300 MiB on the T4) and gate (night window, governor busy, VRAM).
* **What they do not stop:** a keyholder who sends a plausible but harmful delta. With fewer than five learners no Byzantine filter can work (Multi-Krum keeps everyone at n < 5). Treat results from untrusted participants as unverified until at least five independent learners and a filter are in place.
* A refused or silent learner costs nothing but its data: the **missing learner's data is not learned** and there is no carry of late work (the requirement the paper's own E8 pilot found, HM-S9-01). The tests record this cost instead of hiding it.

## 6. What was validated

All on the T4 unless stated. **Both learners ran on the same machine in every run below**, so this validates the protocol, the checks and the failure handling over the real bus; it is not yet a multi-machine result (the 2070 host is not on the overlay).

| What | Result |
|---|---|
| Tests | 28 new tests (protocol library, a signed in-memory bus, end-to-end rounds); with the existing AMCP suites 179 pass. Includes: a foreign-key message is ignored, every refusal path on both sides, a CRLF checkout pins the same as an LF one |
| Two learners over the signed bus equal the single-process baseline | CPU, in-process: final loss within 1e-4 |
| A learner missing from the run | rounds close at the quorum; the run finishes (tested) |
| A poisoned delta (NaN) from a participant that holds the key | rejected with its reason, never reaches the global state, the honest learner carries the round (tested) |
| A learner whose code differs | refuses the task, sends nothing (tested) |
| **Live, real Kafka cluster** (three Spot VM brokers plus the T4, over WireGuard), NF4 on the T4 GPU, 6 rounds | the learners' start-of-round losses equal the single-process baseline's in all six rounds (difference 0 to the printed precision); coordinator and both learners exit 0 |
| **Live churn soak**, 70 rounds, 28.6 minutes, learner `l1` killed with SIGKILL and restarted, then `l0` the same | 62 rounds complete, 8 closed by quorum (5 with only `l0`, 3 with only `l1`: exactly the outage windows), 0 retries, 0 rejections; both restarted learners rejoined on their own; held-out loss 4.171 to **1.448** (best 1.407); rounds took p50 2.9 s, quorum rounds about 60 s (the deadline) |
| **The night run** | started 2026-11-01 01:09 UTC as run `night-001`: 540 rounds at a 30 s cadence (about 4.5 hours), learners gated to 22:00-07:00 Paris, the governor and the VRAM floor. Results: `artifacts/experiments/e8_real/night-001.json` (rewritten every 10 rounds) and `night-001_status.txt`. **Not known when this was written** |

Caveats: the coordinator evaluates on a CPU full-precision model while learners evaluate on the NF4 GPU model, so compare like with like (learner-side losses with learner-side losses); the held-out set mixes both learners' data, so when a learner is absent its half is not learned and the mixture loss can rise (the soak shows it recovering once the learner returns); the settings (`--inner-lr 1e-2 --outer-lr 0.5 --outer-mu 0.5`) were not tuned, and the harsher ones (2e-2, 0.7, 0.65) overshoot after round 2 in the baseline too.

Found on the way (and fixed): two learners in one process built **different** base models from torch's global RNG, which the base-hash check refused (so the check works); the base model is now built from a platform-independent generator. The coordinator sent a list where learners expect a count. The shuffled-label control indexed the wrong dimension.


## 7. Not built (honest list)

1. **Tier B** (the router relay bridge). The router already has AMCP publish and events endpoints, but the bus is not configured there and the Cloud Run to Kafka route is not solved; the bridge above is the proposed way and is untested.
2. **Adapters larger than one envelope.** Granite-scale LoRA is megabytes; the 64 KiB cap needs chunking with a hash tree or an object store with signed URLs. The library refuses an oversize payload with an explicit message rather than truncating.
3. **Per-participant identity** (asymmetric signatures, SASL per participant, ACLs) and rate limits.
4. **Late-work carry**, and a Byzantine filter with at least five learners.
5. **The 2070 host over the bus.** It needs the overlay tunnel (UDP 51820 is open on the T4; the handshake never happened) so that it can reach every broker address.

## 8. Procedure for a new participant (tier A today)

You need: a machine with an NVIDIA GPU (or CPU for a smoke test), Python with `torch`, `transformers`, `peft`, `confluent-kafka` (and `bitsandbytes` for NF4), the repository **at the commit the coordinator names**, a route to the Kafka brokers, and the shared signing key from the project operator (out of band; never paste it into a chat, an issue or a log).

1. **Get the code at the pinned commit.** Your learner refuses every task if its files differ from the coordinator's; the message says so. (Line endings are normalised, so Windows and Linux checkouts of the same commit agree.)
2. **Set two environment variables.** `HIVEMIND_KAFKA_BOOTSTRAP` (the broker list, given by the operator) and `HIVEMIND_AMCP_HMAC_KEYS` (`kid:hex`).
3. **Check the machine without joining.** `python scripts/hivemind_amcp.py --dev-keyring check` (broker reachable, topics present) and, on a GPU machine, `nvidia-smi` showing at least 4,300 MiB free beyond what you want to keep for your own use.
4. **Join.** `python scripts/e8_amcp_round.py learner --run-id <run> --learner-id <the id the operator gave you> --index <shard> --quant nf4 --device cuda --night-only`. It waits for offers, and prints each decision.
5. **Understand what it will not do.** It does not claim a task outside 22:00 to 07:00 Paris time (`--night-only`), while the T4 governor reports inference in flight, or if the free VRAM would fall below the floor. It never opens a port and never runs code from a message.
6. **Stop any time** with Ctrl+C. The coordinator closes the round with whoever answered; your data for that round is simply not used.
7. **If it says it refused:** "different code version" means update your checkout; "base model hashes differently" means your Python or numpy produces different bytes (report it with the versions); "deadline has passed" means the machine was too slow for this round (raise the coordinator's `--round-deadline-s`).

Operator side: `python scripts/hivemind_amcp.py provision` once (creates the topics), then `python scripts/e8_amcp_round.py coordinator --run-id <run> --learners <ids> --rounds <n> --quorum 1 --checkpoint <file> --out <report.json>`. The report records, per round, who was accepted and from which host, who was rejected and why, the held-out loss, delta sizes, peak VRAM and latency.
