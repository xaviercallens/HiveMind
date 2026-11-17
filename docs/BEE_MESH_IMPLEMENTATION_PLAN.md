# Bee mesh: implementation plan

Scope: the bee mesh (`mesh/`) that runs HiveMind inference as many small workers ("bees") behind a host and an
OpenAI-compatible gateway, on the AMCP agent framework over Kafka. Decision of record: option B in
[`community/BEE_AGENT_MESH_DESIGN.md`](community/BEE_AGENT_MESH_DESIGN.md) (owner, 2026-10-27). Protocol:
[`mesh/WIRE_PROTOCOL.md`](../mesh/WIRE_PROTOCOL.md). Newcomer guide: [`community/BEE_MESH_QUICKSTART.md`](community/BEE_MESH_QUICKSTART.md).
Roadmap ids: `HM-S1-01` to `HM-S1-09`, section 3.21 of [`../ROADMAP.md`](../ROADMAP.md) (the mesh line numbers its own seasons from 1, at the owner's instruction of 2026-10-28; the ids are unrelated to `HM-S10` to `HM-S28`).
Hardened-Hive IDs).

**Status legend.** *Done in this branch* = code and a recorded run exist in the working tree. *Planned* = no code.
Nothing here is deployed on the production router. Of the phases after P1, only P2 has been measured (a pre-registered run, MESH-SCALE-1; see its section).

**Ground rules for every phase.** Each phase has a pre-registered acceptance test written before the run, the owner
signs off on the owner decisions, and results are reported as measured, including failures. No scale claim beyond one
machine until P5 is done and tested over a network.

---

## Phase summary

| Phase | Name | Status | Roadmap |
|---|---|---|---|
| P0 | AMCP Kafka repair | Proposed upstream, not merged: agentmeshcommunicationprotocol/amcpcore.github.io#1 (hardened; 9 + 23 tests) | `HM-S1-01` |
| P1 | Wire protocol v1 and mesh v0.1 | Done (`mesh/`, merged in #279) | `HM-S1-02` to `HM-S1-05` |
| P2 | 1000-bee control-plane test on one VM | Measured, pre-registered (MESH-SCALE-1): PASS at N=1000 on A1 to A4; owner decisions on VM and window still open | `HM-S1-06` |
| P3 | Security on the bus | Planned | `HM-S1-07` |
| P4 | Bridge to the OpenRouter gateway | Planned | `HM-S1-08` |
| P5 | Multi-broker, multi-region Kafka | Planned | `HM-S1-09` |
| P6 | Lightweight Rust bee | Planned | `HM-S1-09` |
| P7 | GPU bees with the GWAYA verification gate | Planned (not in the roadmap rows yet: to add once its owner decisions are made) | to add |

---

## P0. AMCP Kafka repair

**Problem.** The AMCP Kafka broker in `amcp-broker-kafka` (1.6.0) did not deliver as the mesh needs (queue consumers and
committed offsets). The mesh cannot start on upstream `main` without this fix.

**Deliverables.**
- The fix on branch `fix/kafka-broker-delivery` of the AMCP clone (its own test coverage is not reviewed here).
- `mesh/quickstart.sh` builds and installs `amcp-core` and `amcp-broker-kafka` from `AMCP_DIR`, `AMCP_REPO` and
  `AMCP_REF`, so a newcomer can run the fix before it is merged upstream.

**Acceptance.**
- `amcp-broker-kafka` builds and installs with the Java 21 Maven command in `mesh/README.md`.
- The mesh integration tests (`HIVEMIND_MESH_KAFKA` set) run and pass. The README records 21 of 21 (17 unit and 4
  integration) on the development machine. Re-run before any claim.
- A fresh `mesh/quickstart.sh` start answers one chat request, and `stop` leaves no process of ours running.

**Status.** Quickstart verified on this machine on 2026-10-27: two full start and stop cycles on a Kafka broker at
`127.0.0.1:19092` answered `sim[coder]: Hello from the bee mesh quickstart`. The mesh unit tests passed in the build
(17 run, 0 failures, the 4 integration tests skipped because `HIVEMIND_MESH_KAFKA` was not set). Whether the upstream
PR is merged is not confirmed in this branch.

**Risks.** The fix may diverge from upstream. Consumers use AMCP's auto-commit, so a crash between poll and handling
loses a message (documented; a 504 for the caller).

**Owner decision.** Which AMCP repository and branch is the reference for the mesh (upstream `main` only once merged, or
a fork that the owner controls).

---

## P1. Wire protocol v1 and mesh v0.1

**Deliverables (present in `mesh/`).**
- `WIRE_PROTOCOL.md`: topics, payloads (`"v": 1`), routing, attempts and reassignment, handoff, finalisation, gateway
  behaviour, delivery semantics.
- Host (routing by least load among live bees, attempt timers, at most two reassignments, the only `TaskReply` writer).
- Bee runtime (virtual-thread backends, heartbeats, bee-to-bee help and offer, bounded local queue).
- Gateway (`POST /v1/chat/completions`, `GET /mesh/health`, loopback only, Host check, 64 KiB body cap).
- CLI (`host`, `bees`, `gateway`, `ask`, `status`), a backend interface with a simulator and an OpenAI-compatible client.
- `quickstart.sh` (start and stop, with tool installation and SHA checks).

**Acceptance (v0.1).**
- Unit tests pass: routing policy, wire round trip, shard function. Re-run on 2026-10-27 during the quickstart build:
  17 run, 0 failures (9 registry and 8 wire tests), 4 integration tests skipped without a broker.
- A single-broker smoke test answers one request through the gateway (done 2026-10-27 by the quickstart).
- Integration tests (run only with `HIVEMIND_MESH_KAFKA` set, each on its own topic root): one host, five bees and a
  gateway answer twenty requests; a busy bee hands its task to a peer; a hung backend is failed over to another bee; two
  hosts sharing `task.submit` answer every task exactly once. These are the tests that need a broker. The README records
  21 of 21 passing. Re-run on 2026-10-27 in the review of the bee-mesh work: 21 run, 0 failures, 0 errors, 0 skipped
  (`HIVEMIND_MESH_KAFKA=127.0.0.1:9092`, surefire XML in `mesh/target/surefire-reports`).

**Known deviations.** Hosts write every `TaskReply` (bees never do), for exactly-once finalisation with two hosts on the
queue. Heartbeat and attempt-timeout values are extensions to the first brief and are documented in the wire protocol.

**Risks.** Auto-commit gives at-most-once behaviour in crashes. Fan-out heartbeats are retained for 24 h, so every new
JVM re-reads recent history (the README suggests a fresh topic root per run).

**Status.** Done in this branch. `mesh/` is untracked in git at the time of writing; the lead commits it.

---

## P2. 1000-bee control-plane test on one VM

**Question.** Does the control plane (heartbeats, registry, routing, reassignment) stay correct with 1000 simulated bees
in four JVMs on one VM, with a fixed request load?

**Pre-registration (written before the runs; see Status).** Bee count and JVM split, heap per JVM, request count and concurrency,
the success criteria (for example: every request answered with `ok=true` or a stated error, `liveBees` reaches 1000 within
a fixed time, no bee dropped without a reason, median and p95 latency bounds fixed in advance), the stop rules, and what
is reported when a criterion fails.

**Deliverables.** `mesh/` scale runbook in `docs/community/` (commands only, no results), a pre-registration file under
`artifacts/experiments/`, and the receipt of the run.

**Acceptance.** The pre-registered criteria, decided by the scientific lead, not by the engineering side.

**Status.** Measured. The pre-registration is `artifacts/experiments/MESH_SCALE_prereg.json` (registered 2026-10-27
14:48 UTC) with amendments 1 to 3, which change harness mechanics only. Runs at N=100, 300 and 1000 (200 tasks/s for 120 s,
10% of bees killed at load start + 60 s) are reported in `docs/benchmarks/BEE_MESH_SCALE_report.md`, receipt
`artifacts/experiments/MESH_SCALE_report.json`. Verdict at N=1000: **PASS** on A1, A2, A3 and A4. One run per N, simulated
bees, one VM. The earlier unregistered 1000-bee run in `mesh/README.md` (one of 50 scale-up requests got no HTTP status,
cause undiagnosed) is context only.

**Risks.** Kafka topic creation (32 inbox shards) and consumer-group joins during scale-up; the 30 s gateway timeout equal
to the host's worst case (three attempts of 10 s); memory on a shared host (the T4 machine serves live users, so only a
dedicated VM or a window the owner approves).

**Owner decisions.** The VM and the window (the shared production host must not be used); the pass criteria; whether the
504 path is tested separately first.

---

## P3. Security on the bus

**Problem.** Anyone who reaches the Kafka port can publish tasks or impersonate a bee. The gateway's Host check is not
access control.

**Deliverables.**
- Kafka listeners with SASL/SCRAM or mTLS, with the broker bound to the private interface only.
- Per-bee credentials (issued and revocable), and an ACL per topic (bees may publish only their own heartbeats and
  `TaskDone`, hosts may write `TaskReply` only).
- Signed results: `TaskDone` carries a signature from the bee's key, checked by the host.
- Gateway authentication (API key per end user) and TLS if it leaves loopback.

**Acceptance.** A test that an unauthenticated producer is refused; a test that a bee cannot publish another bee's
`TaskDone`; a test that a forged `TaskDone` with a good `taskId` is rejected; all three run in CI on every change.

**Risks.** Key management on volunteer machines (see `HM-S11-01` and `HM-S11-08` for the owner's key work); latency cost of
signatures (measured, not assumed).

**Owner decisions.** Who issues bee credentials; whether results are signed by the bee or by the coordinator; whether
the gateway is reachable beyond loopback at all.

---

## P4. Bridge to the OpenRouter gateway

**Question.** Can the existing HiveMind OpenRouter gateway (Cloud Run, `hivemind-openrouter`, see
`docs/HIVEMIND_OPENROUTER.md`) send requests into the mesh, with producers appearing as bees, and with HMT accounting
unchanged?

**Deliverables.** A bridge that maps a producer to one or more bees (its tier and capacity from its heartbeat), a
gateway-side adapter that submits tasks to the mesh, and the accounting hook that stays as it is today.

**Acceptance.** Same request, same answer shape and the same HMT charge with the mesh on and off (paired test); a producer
that disconnects is removed from routing within three heartbeat intervals; no change to the HMT ledger rules.

**Risks.** The gateway is the payment and audit point; a mesh-side failure must not create an unaccounted request.
Latency added by the bus.

**Owner decisions.** Whether consumer traffic may reach the mesh at all (the design doc records that the owner has not
decided); the merge of PR #206 that the design depends on.

---

## P5. Multi-broker and multi-region Kafka

**Deliverables.** Kafka cluster with replication factor of at least three, rack or region awareness, a documented failover
drill, and the mesh configuration for more than one bootstrap address. Region tags on bees already exist (`--region`).

**Acceptance.** A broker stop during load loses no acknowledged task beyond the documented delivery rule; a region
partition is tested with a network fault injector; measured failover time.

**Risks.** Cost and operations; cross-region latency; the AMCP broker's support for replication settings.

**Owner decisions.** Which provider and regions; whether a managed Kafka is acceptable.

---

## P6. Lightweight Rust bee for low-memory volunteers

**Goal.** A bee binary that uses little memory, speaks the same wire protocol (`hivemind-mesh/1`) and runs on
volunteer machines that cannot run a JVM.

**Deliverables.** A Rust bee that subscribes to its inbox shard, publishes heartbeats and `TaskDone`, and calls a local
backend over HTTP. A conformance test against the Java host using the same wire fixtures (`WireTest` payloads).

**Acceptance.** Conformance: the Rust bee and the Java bee produce the same event bodies for the same inputs; memory
footprint measured on a named machine; one 1000-bee run is not required for this phase.

**Risks.** Kafka client libraries in Rust for AMCP's transport (the AMCP framing is Java-side); a second implementation
drifting from the protocol.

**Owner decisions.** Whether a non-Java bee joins the AMCP transport or talks to a bridge instead.

---

## P7. GPU bees with the GWAYA verification gate

**Goal.** A bee whose backend is a GPU model, with answers passed through the GWAYA verification gate (see
`docs/community/GWAYA_LITE_ASSISTANT.md`) before `TaskDone ok=true`.

**Deliverables.** A gated backend: generate, verify, and report `ok=false` with the gate's reason when the gate refuses.
Measured gate precision and the cost of the gate per request.

**Acceptance.** Refused answers never reach the user as `ok=true`; precision of verified answers reported with its
interval, not as a point claim; GPU use only on a machine the owner has approved (the T4 serves live users).

**Risks.** The gate covers code tests, not every kind of answer; GPU memory per bee; the production T4 must not be
loaded by the mesh without a window.

**Owner decisions.** Which GPU hosts may run bees; the gate's scope for non-code answers.

---

## What is not decided

- Whether the mesh replaces, or only sits beside, the production router (owner, design doc decision 4).
- Whether AMCP stays a dependency for the long term (design doc decision 4).
- Any public launch of a bee network.
