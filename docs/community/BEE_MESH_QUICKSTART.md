# Bee mesh quickstart (for newcomers)

The bee mesh is HiveMind's experiment in running many small model workers ("bees") behind one router
("host") and one OpenAI-compatible HTTP door ("gateway"). The bees talk over a message bus (AMCP on Kafka).
This page is the guided tour. The reference is [`mesh/README.md`](../../mesh/README.md) (commands, limits,
measurements) and the wire format is [`mesh/WIRE_PROTOCOL.md`](../../mesh/WIRE_PROTOCOL.md). The design
history and the owner's decision are in [`BEE_AGENT_MESH_DESIGN.md`](BEE_AGENT_MESH_DESIGN.md). The plan
is [`docs/BEE_MESH_IMPLEMENTATION_PLAN.md`](../BEE_MESH_IMPLEMENTATION_PLAN.md).

**Status: v0.1, one machine, one Kafka broker on loopback, no authentication, not tested over a WAN.**
It is a control-plane prototype. It is not wired into the production router.

## 1. The three roles

| Role | What it is | Where it runs |
|---|---|---|
| **Bee** | A worker that answers tasks for one or more *tiers* (for example `coder`, `general`). It has a *backend*: the function that actually produces the text. It advertises its load with a heartbeat every 5 s. | One JVM can host many bees. A bee can have a real model behind it or the built-in simulator. |
| **Host** | The router. It keeps a live list of bees (from heartbeats), sends each task to the least-loaded bee for the tier, watches the clock, reassigns a task that a bee does not finish, and writes the one reply for each task. | One or more JVMs. Every host sees every bee. |
| **End user** | Anyone who sends a chat request. Users never see the bus. They call the gateway over HTTP, the same way they would call any OpenAI-compatible API. | Your terminal, a script, or an app on the same machine. |

The **gateway** is the HTTP front door for end users. It turns `POST /v1/chat/completions` into a task on the
bus and waits for the reply. It binds to `127.0.0.1` only.

## 2. The event flow

Every message is an AMCP event (CloudEvents, JSON on the wire). The topics are named `hivemind.*` by default.

```
 end user (curl, a script)
    |  POST /v1/chat/completions  {"model":"coder", "messages":[...]}
    v
 gateway  ----------------------------------------------------------------+
    |  hivemind.task.submit   (one of the hosts takes it: queue "hosts")  |
    v                                                                     |
 host   picks the least-loaded LIVE bee for the tier                      |
    |  hivemind.bee.inbox.sNN  (shard = hash(beeId) mod 32)               |
    v                                                                     |
 bee    runs its backend on a virtual thread                              |
    |  hivemind.task.done      (fan-out to every host)                    |
    v                                                                     |
 host   the host whose hostId matches writes the ONE reply                |
    |  hivemind.user.reply.<gatewayId>                                    |
    +----------------------------------------------------------------------+
    v
 gateway  returns HTTP 200 with the answer (or 503 / 504)
```

Bee-to-bee help, when a bee is full or does not serve the tier:

```
 bee A (full)  --- hivemind.bee.mesh.help  -->  every bee with a free slot
 bee B         --- hivemind.bee.mesh.offer -->  bee A   (the first offer that arrives wins)
 bee A         --- hivemind.bee.inbox.sNN  -->  bee B   (the same task, attempt + 1)
```

Bees also send `heartbeat` and `lifecycle` (JOINING, LEAVING) events, and hosts send `host.announce` every 2 s.
A bee that stays silent for three heartbeat intervals is dropped from routing. If a bee does not finish within
`attemptTimeoutMs` (default 10 s), the host tries another bee, at most twice.

## 3. Five-minute quickstart

You need Linux with `bash`, `curl`, `python3`, `git`, `tar`, and about 2 GB of free RAM for the mesh (Kafka 512 MB heap, four JVMs at 256 MB each, plus the build). The script installs Java 21,
Maven 3.9.9 and Kafka 3.8.0 under `~/.hivemind-mesh/tools` (checksums are verified), so it does not touch your system.

```sh
git clone https://github.com/xaviercallens/HiveMind.git && cd HiveMind
cd mesh && ./quickstart.sh          # start: installs, builds, runs everything, sends one chat request
```

The first run takes a few minutes (downloads and the AMCP build). It ends with something like:

```
Answer:
  sim[coder]: Hello from the bee mesh quickstart
  bee=bee-... attempts=1 e2eMs=...
```

Then try it yourself:

```sh
curl -s http://127.0.0.1:8097/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"general","messages":[{"role":"user","content":"hi"}]}'
curl -s http://127.0.0.1:8097/mesh/health        # the latest host announcement
```

When you are done:

```sh
cd mesh && ./quickstart.sh stop     # stops exactly the processes the script started, and its Kafka broker
```

Useful settings (environment variables):

| Variable | Default | What it does |
|---|---|---|
| `HIVEMIND_MESH_HOME` | `~/.hivemind-mesh` | tools, AMCP clone, Maven repo, Kafka data, logs, pid file |
| `QUICKSTART_KAFKA_PORT` | `9092` | Kafka client port (loopback). Change it if you already run a broker |
| `QUICKSTART_KAFKA_CONTROLLER_PORT` | port + 1 | KRaft controller port (loopback) |
| `QUICKSTART_GATEWAY_PORT` | `8097` | gateway port (loopback) |
| `AMCP_REPO`, `AMCP_REF` | upstream AMCP, `main` | where the AMCP sources come from (see the note below) |
| `AMCP_DIR` | unset | use an AMCP clone you already have (built as it is) |
| `JAVA_HOME_21`, `MVN_BIN`, `KAFKA_HOME` | unset | reuse tools you already have instead of downloading them |

**Note on AMCP.** The Kafka delivery fix for AMCP is not merged upstream yet. Until it is, point the script at the
fix branch (`AMCP_REPO=<your fork> AMCP_REF=fix/kafka-broker-delivery`, or `AMCP_DIR=<your clone of it>`). The script
prints this note at every start.

If the script says a port is in use, another program already listens there. Pick other ports with the variables
above. The script never stops anything it did not start.

## 4. Your first bee backend in Java

A bee's behaviour is one interface:

```java
package org.hivemind.mesh.bee;

public interface Backend {
    String complete(String tier, List<Wire.Message> messages) throws Exception;
}
```

`messages` is the chat history (`role`, `content`). Return the answer text. A backend may block: bees call it on
virtual threads, never inside a broker callback. Throwing an exception makes the bee report `ok=false`, and the host
then tries another bee.

A minimal bee that answers with the reversed last message (not a model, just a demonstration). It joins the same
root as the quickstart (`hivemind` by default, so start the host with the same `--root`):

```java
import org.hivemind.mesh.bee.Backend;
import org.hivemind.mesh.bee.BeeRuntime;
import org.hivemind.mesh.wire.MeshBus;
import org.hivemind.mesh.wire.Topics;

import java.util.List;
import java.util.UUID;

public final class ReverseBee {
    public static void main(String[] args) throws Exception {
        Backend reverse = (tier, messages) -> {
            String last = messages.get(messages.size() - 1).content();
            return new StringBuilder(last).reverse().toString();
        };
        MeshBus bus = new MeshBus("127.0.0.1:9092", new Topics("hivemind"), UUID.randomUUID().toString());
        // tiers, maxConcurrent, region, heartbeatMs, helpWaitMs, queueBound
        var config = new BeeRuntime.Config(List.of("general"), 4, "local", 5_000, 500, 16);
        BeeRuntime runtime = new BeeRuntime(bus, config, 2, "reverse", i -> reverse);
        runtime.start();
        Thread.currentThread().join();   // run until you stop the JVM
    }
}
```

Build it against the mesh jar (`mesh/target/hivemind-bee-mesh-0.1.0-all.jar`, produced by the quickstart):

```sh
javac -cp mesh/target/hivemind-bee-mesh-0.1.0-all.jar ReverseBee.java
java -Xmx256m -cp mesh/target/hivemind-bee-mesh-0.1.0-all.jar:. ReverseBee
```

The example compiles against v0.1 (checked with `javac`). It has **not** been run as written: the expected answer to
`ask --tier general "hello"` (`olleh`) is what the code should do, not a recorded result. The example is not part of the
repository; it uses the signatures of `Backend`, `BeeRuntime.Config` and `MeshBus` in `mesh/src/main/java`.

The simulated bees you start with `bees --count 3` use `SimBackend`. It sleeps for `--sim-latency-ms` plus jitter and
returns `sim[<tier>]: <last message>`. It is useful for testing the routing and failure paths (`--fail-rate 0.2`).

## 5. Plugging in a real local model

The `bees` command can call any OpenAI-compatible `/v1/chat/completions` endpoint:

```sh
java -Xmx256m -jar mesh/target/hivemind-bee-mesh-0.1.0-all.jar bees --count 1 --tiers coder \
  --backend-url http://127.0.0.1:<your-port> --backend-model my-model --bootstrap 127.0.0.1:9092 --root hivemind
```

> **Warning: only point `--backend-url` at a server that you run yourself, on this machine or on a network you
> control.** Every request a bee receives is sent to that URL, with the prompt inside. There is no default URL and no
> authentication. Do not point a bee at someone else's API or at a shared production service without the owner's
> approval.

The bee passes the tier as the model name unless `--backend-model` is given. The bee's `maxConcurrent` should not be
higher than your server can run at once. One slow model makes every bee on that server slow.

## 6. The wire protocol

Topic names, payload fields, the routing rules, the attempt and reassignment rules and the delivery limits are in
[`mesh/WIRE_PROTOCOL.md`](../../mesh/WIRE_PROTOCOL.md) (protocol `hivemind-mesh/1`, field `"v": 1`). If you write a bee
in another language, speak that protocol: subscribe to the inbox shard of your bee, publish heartbeats, and publish a
`TaskDone` for every assignment.

## 7. FAQ

**Is this the HiveMind production router?** No. The production router is `src/hivemind/router` (OpenRouter-compatible,
on Cloud Run). The mesh is a separate experiment. Bridging the two is planned as phase P4 of the implementation plan.

**Why Kafka?** It is the transport that AMCP's broker uses. It gives topics with fan-out and queues with competing
consumers, which is what the mesh needs. A single broker is enough for the current scope.

**How many bees can it run?** The pre-registered test (MESH-SCALE-1, roadmap item `HM-S1-06`) ran 100, 300 and 1000 simulated
bees on one VM with one Kafka broker. At 1000 bees, all four acceptance criteria passed: all bees live within 60 s, no lost
or duplicated task at 200 tasks/s, every task on a killed bee answered after reassignment, and no OutOfMemoryError. The
figures and the N=100 and N=300 rows are in [`docs/benchmarks/BEE_MESH_SCALE_report.md`](../benchmarks/BEE_MESH_SCALE_report.md).
An earlier run of 1000 bees by hand, not pre-registered, is described in the "Measured" section of
[`mesh/README.md`](../../mesh/README.md); it is context only. Treat any number as one-machine, simulated, and not a
production claim.

**Why does the first request sometimes take several seconds?** A new Kafka consumer group waits for its partitions, and
each bee JVM subscribes to the inbox shards that hold its bees. The quickstart sets `group.initial.rebalance.delay.ms=0` and waits until the host
sees three bees before it sends its request. Wait a few seconds after starting a mesh before you measure anything.

**What happens if a bee dies mid-task?** The host notices after the attempt timeout (default 10 s) and sends the task to
another bee. If none answers after two reassignments, the reply is `no bee answered`. The gateway returns 503 for mesh
errors and 504 when its own timeout (default 30 s) expires first.

**Can I run the whole thing on two machines?** Not yet. Kafka runs on loopback, and the gateway refuses any `Host` header
other than `127.0.0.1` or `localhost`. Multi-machine use needs phases P3 (security) and P5 (multi-broker) of the plan.

**Where are the logs?** `~/.hivemind-mesh/logs/` (one file per JVM and `kafka.out`). The last answer is saved in
`~/.hivemind-mesh/run/last-answer.json`.

**Does it use my GPU?** No. The quickstart uses simulated bees unless you pass `--backend-url`.

## 8. Honest limits (v0.1)

- **One broker, one machine, loopback only.** No replication, no WAN testing, no multi-region.
- **No authentication.** Anyone who can reach the Kafka port can publish tasks or impersonate a bee. The gateway's
  `Host` check is not access control.
- **Simulated scale only.** The 1000-bee figures are one-machine control-plane numbers with simulated bees. They say
  nothing about real model throughput.
- **No delivery guarantee across crashes.** The AMCP Kafka consumers auto-commit. A task lost in a crash becomes a 504.
- **No persistence.** A restart forgets in-flight tasks.
- **Replies are written by hosts only.** This deviates from the first brief; see the wire protocol.
- **The AMCP fix is not in upstream AMCP yet.** Use the fix branch as described above.
- **The mesh is not connected to the production router or to any user's data.**
