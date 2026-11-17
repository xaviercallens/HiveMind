# hivemind-bee-mesh 0.1.0

A bee mesh for HiveMind inference: many small model workers ("bees") behind a router ("host") and an
OpenAI-compatible HTTP gateway. The bees are AMCP agents; the bus is AMCP's Kafka broker. Wire protocol:
[`WIRE_PROTOCOL.md`](WIRE_PROTOCOL.md). Design history and the owner's decision:
[`../docs/community/BEE_AGENT_MESH_DESIGN.md`](../docs/community/BEE_AGENT_MESH_DESIGN.md).

Status: v0.1, single local Kafka broker, loopback only, no authentication, not WAN-tested.

## Build

Java 21 and Maven, with the AMCP 1.6.0 artifacts (`io.amcp:amcp-core`, `io.amcp:amcp-broker-kafka`) in the
local repository. On the development machine, Maven runs as:

```sh
cd mesh
JAVA_HOME=/mnt/disks/disk-socrateai-local-1/amcp/tools/jdk21 \
  /mnt/disks/disk-socrateai-local-1/amcp/tools/maven/bin/mvn -B \
  -Dmaven.repo.local=/mnt/disks/disk-socrateai-local-1/amcp/m2 package
```

This produces `target/hivemind-bee-mesh-0.1.0-all.jar` (fat jar, `Main-Class org.hivemind.mesh.cli.Main`).

Tests:

- Unit tests (routing policy, wire round trip, shard function) always run.
- Integration tests need a broker and run only when `HIVEMIND_MESH_KAFKA` is set to its bootstrap
  address. Without it they are skipped:

```sh
HIVEMIND_MESH_KAFKA=127.0.0.1:9092 JAVA_HOME=... mvn -B -Dmaven.repo.local=... package
```

Each integration test uses its own random topic root, so tests never collide with each other or with a
running mesh.

## Run

Kafka must be up on `127.0.0.1:9092`. Each command takes `--bootstrap` (default `127.0.0.1:9092`) and
`--root` (topic root, default `hivemind`).

```sh
java -Xmx256m -jar target/hivemind-bee-mesh-0.1.0-all.jar host                  # router; --host-id, --attempt-timeout-ms
java -Xmx256m -jar target/hivemind-bee-mesh-0.1.0-all.jar bees --count 3       # 3 simulated bees, coder+general
java -Xmx256m -jar target/hivemind-bee-mesh-0.1.0-all.jar gateway --port 8097   # HTTP front door on 127.0.0.1
java -jar target/hivemind-bee-mesh-0.1.0-all.jar ask --tier coder "hello"      # one task, prints the reply
java -jar target/hivemind-bee-mesh-0.1.0-all.jar status --wait-bees 3 --timeout 60   # exit 0 when liveBees >= 3
```

`bees` options: `--tiers coder,general` (default), `--max-concurrent 4`, `--sim-latency-ms 50`,
`--sim-jitter-ms 10`, `--fail-rate 0`, `--backend-url URL --backend-model M` (OpenAI-compatible; there is
no default URL), `--region local`, `--help-wait-ms 500`, `--queue-bound 16`, `--heartbeat-ms 5000`.

Gateway call:

```sh
curl -s http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"coder","messages":[{"role":"user","content":"hello"}]}'
curl -s http://127.0.0.1:8097/mesh/health
```

The reply is OpenAI-compatible, with an extra `hivemind` object `{beeId, attempts, e2eMs}`. Errors:
403 for a Host header other than `127.0.0.1` or `localhost`, 413 above 64 KiB, 504 on timeout (default
30 s, `--timeout-s`), 503 when the mesh answers with an error (for example `no capacity`).

## Architecture

```
  client ──HTTP──► gateway ──R.task.submit (queue "hosts")──► host(s) ──R.bee.inbox.sNN──► bees
                      ▲                                          ▲   │
                      │ R.user.reply.<gw>                        │   │ R.bee.mesh.help / offer
                      └──────────── host finalises TaskReply ◄───┤   │  (bee-to-bee handoff)
                                                                 │   ▼
                        R.bee.heartbeat, R.bee.lifecycle ────────┘  bees
                        R.task.done (every host, the matching one finalises)
                        R.host.announce ─► gateway /mesh/health, status
```

- **One bus per JVM.** `MeshBus` holds one `KafkaEventBroker` and one AMCP `AgentContext`. Every bee,
  host and gateway in the JVM shares them.
- **Bees are AMCP agents, not Kafka consumers.** `BeeAgent extends AbstractMobileAgent` and is registered in
  the context for identity and lifecycle. The JVM subscribes once per inbox shard that holds one of its bees,
  plus `help` and `offer` once. Each inbox event is decoded once, routed by `targetBeeId` through a map
  keyed by beeId, and handed to that bee.
- **One heartbeat sweep per JVM.** A single scheduled task publishes the heartbeats of all bees in the JVM.
- **Backends run on virtual threads.** Each assignment runs on a virtual thread, bounded by the bee's
  `maxConcurrent`. Backend calls never run inside a broker callback.
- **Host.** Keeps a `Registry` of live bees and routes least-loaded. Timers handle attempt timeouts and
  reassignment. It publishes the only TaskReply for each task.
- **Gateway.** `com.sun.net.httpserver` bound to 127.0.0.1. Each request waits on a future that the reply
  subscription completes.

### Event flow of one request

1. The gateway publishes `TaskSubmit` to `R.task.submit`. One host of the `hosts` queue receives it.
2. The host picks the least-loaded live bee for the tier and publishes `TaskAssignment` to that bee's
   inbox shard. It arms an attempt timer.
3. The bee runs its backend on a virtual thread and publishes `TaskDone` to `R.task.done`.
4. The host whose `hostId` matches accepts the `TaskDone` and publishes `TaskReply` to the gateway's reply
   topic. The gateway completes the HTTP request.

Failures and saturation:

- **Bee full or tier not served:** the bee publishes `HelpRequest` and waits up to `helpWaitMs`. The first
  `HelpOffer` wins, and the task is handed to the helper as attempt + 1.
- **No offer in time:** the task is queued locally, or it fails with `TaskDone ok=false`.
- **No `TaskDone` within `attemptTimeoutMs`:** the host reassigns to another bee, at most twice, then replies
  `no bee answered`. A `TaskDone ok=false` triggers the same reassignment.
- **Bee silent for three heartbeat intervals, or LEAVING:** the host drops it from routing.

## Limits (v0.1)

- **Single broker, loopback only.** One Kafka node on `127.0.0.1:9092`. No replication, no WAN testing, and
  every server binds `127.0.0.1`. The gateway is HTTP without TLS and is reachable only from the same host.
- **No authentication.** Anyone who can reach the Kafka port can publish tasks or impersonate bees. The
  gateway's Host check is not an access control.
- **Delivery is not guaranteed across crashes.** AMCP's Kafka broker auto-commits. A message can be lost if
  its JVM crashes after polling. The gateway's timeout turns a lost task into a 504.
- **Startup latency.** A new Kafka consumer group waits about 3 s before it gets partitions. A request sent
  right after startup can take a few seconds. Warm up before measuring.
- **Fan-out history.** A new JVM reads retained history from the earliest offset and then filters by time, so
  its consumer decodes every retained record. Use a fresh `--root` for each deployment or scale run, or
  shorten the retention of `R.bee.heartbeat`. Otherwise every restart re-reads the heartbeat backlog.
- **AMCP agent registry key.** `AgentContext.registerAgent` keys agents by class name and clock milliseconds,
  so bees registered in the same millisecond overwrite each other. `BeeRuntime` waits for the clock to tick
  between registrations (about 1 ms per bee). This is a workaround for an AMCP limitation.
- **TaskReply is published only by hosts** (see WIRE_PROTOCOL.md, "Deviation"). Bees never publish a reply.
- **Simulated backends by default.** `sim` returns deterministic text and sleeps. Real inference needs
  `--backend-url`, and that path is not exercised by the tests.
- **No persistence.** Host and gateway keep their state in memory. A restart forgets in-flight tasks, and the
  gateway returns 504 for them.
- **Not a production router.** This mesh is not wired into the HiveMind production router (`src/hivemind/router`).

## Measured (v0.1, one machine, local Kafka broker)

Measured by the developer on the dev machine with the commands in this README. Numbers are only what was
observed in that run.

The pre-registered 1000-bee control-plane test (MESH-SCALE-1) is the test of record and is reported in
`../docs/benchmarks/BEE_MESH_SCALE_report.md`: PASS on A1 to A4 at N=1000. The bullets below that come from the
unregistered by-hand run are context only.

- **Tests:** unit tests 17 (routing 9, wire 8), integration 4. With `HIVEMIND_MESH_KAFKA` set: 21 run, 21 pass.
  Without it: 17 run, 4 skipped.
- **Smoke test:** one host, three bees (`--count 3`), one gateway at `-Xmx256m` each. A `coder` request
  returned `sim[coder]: hello` from a bee with `attempts: 1`.
- **Scale run:** one host and one gateway at `-Xmx256m` and `-Xmx768m`, plus four bee JVMs at `-Xmx768m`
  with 250 bees each (1000 bees in 4 JVMs), on one unique topic root. The host reported `liveBees: 1000`
  (`status --wait-bees 1000 --timeout 90` exited 0). Observed RSS: bee JVMs 497 to 570 MB each, host 302 MB,
  gateway 177 MB.
- **Scale load:** 200 requests at concurrency 20 after the scale-up: 200 HTTP 200, all on the first attempt,
  median latency 111 ms, p95 351 ms, max 1.29 s.
- **Scale failure:** during the first minutes of the scale run, while Kafka created the 32 inbox topics and
  joined groups, one of 50 sequential requests got no HTTP status: curl reported code 000. The exit code and
  duration were not recorded, so it is not known whether curl timed out or the connection failed early. The
  host's counters show one failed task over the whole run, which is probably this request, but that was not
  confirmed. Bee JVMs logged `OutOfOrderSequenceException` and `NOT_LEADER_OR_FOLLOWER` during the same
  period. The last `OutOfOrderSequenceException` was logged at 13:25:55, and later checks showed 1000 live bees.
  The cause is not diagnosed. Before any scale claim, the gateway's 504 path needs a separate check (see the
  timeout note below).
- **Timeout collision:** the gateway's default timeout (30 s) equals the host's worst case: three attempts of
  10 s each. A task that uses up its reassignments therefore races the gateway timeout, and the caller usually
  sees 504 rather than the mesh's `no bee answered`.


## Known limitations (from the independent review; not exercised by the scale test)

- **Handoff versus host timer.** A bee-to-bee handoff sends attempt 2 but does not re-arm the host's attempt timer, so a helper that takes longer than the attempt timeout (10 s by default) can trigger a needless reassignment.
- **Duplicate submit after finish.** The host deduplicates only tasks that are still in flight. A task submit redelivered after the task finished would run again and could produce a second reply. "Exactly once" holds for the tested paths only.
- **Gateway coverage.** The gateway's 504 path is not covered by any test; 403 and 413 were checked only by hand.
- **Help requests scan all bees in a JVM.** `BeeRuntime.onHelp` loops over every bee for each help request; at 1000 bees this is a possible hot spot at higher rates than were tested.
- **Largest end-to-end test uses 8 bees.** The 1000-bee evidence is the pre-registered scale experiment, which uses simulated bees; the integration tests with real Kafka are small.
- **Security.** Single local broker, loopback only, no authentication or encryption on the bus yet.
