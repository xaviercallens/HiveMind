# hivemind-mesh/1 wire protocol

Version 1. Every payload is a JSON object with field `"v": 1`. Unknown fields are ignored; a payload
with any other `v` is rejected. Every event is an AMCP event (CloudEvents, structured JSON on the
wire) whose CloudEvents extension `partitionkey` is set to the key shown below.

Topic names use a configurable root (default `hivemind`). Tests and scale runs use a random root
such as `hmitab12cd34ef`, so they never see each other's events. Below, `R` is the root.

## Topics

| Topic | Delivery | Publisher | Subscriber | partitionkey |
|---|---|---|---|---|
| `R.bee.heartbeat` | fan-out (every JVM) | every bee, every `heartbeatMs` (default 5000) | hosts | beeId |
| `R.bee.lifecycle` | fan-out | bees (JOINING, LEAVING) | hosts | beeId |
| `R.bee.inbox.sNN` | fan-out; each JVM subscribes only to the shards of its own bees and dispatches by `targetBeeId` | hosts, bees (handoff) | bees | targetBeeId |
| `R.task.submit` | competing consumers, queue `hosts` | gateways, `ask` | hosts | taskId |
| `R.task.done` | fan-out to every host; only the host whose `hostId` matches finalises | bees | hosts | taskId |
| `R.user.reply.<gatewayId>` | fan-out | hosts | the gateway that owns the id | taskId |
| `R.bee.mesh.help` | fan-out among bees | bees | bees | taskId |
| `R.bee.mesh.offer` | fan-out among bees | bees | bees | taskId |
| `R.host.announce` | fan-out, every 2 s | hosts | gateways, `status` | hostId |

The inbox shard of a bee is `Math.floorMod(beeId.hashCode(), 32)`, written as two digits
(`R.bee.inbox.s00` to `R.bee.inbox.s31`). Java `String.hashCode()` makes this stable across JVMs.

## Payloads

Field types: `string`, `int`, `long` (epoch milliseconds), `bool`, `[T]` a JSON array, `{K:V}` an object.

### hivemind.bee.heartbeat (fan-out, every heartbeatMs)
```json
{"v":1,"beeId":"bee-3f2a1b-00007","tiers":["coder","general"],"inFlight":2,"maxConcurrent":4,
 "queueDepth":0,"region":"local","startedAtMs":1793100000000,"sentAtMs":1793100005000,"heartbeatMs":5000}
```
`heartbeatMs` is an extension to the brief: the bee's own interval. Hosts use it for the dead-bee
rule and fall back to their own configured interval when it is absent or zero.

### hivemind.bee.lifecycle (fan-out)
```json
{"v":1,"beeId":"bee-3f2a1b-00007","state":"JOINING","atMs":1793100000000}
```
`state` is `JOINING` or `LEAVING`.

### hivemind.bee.inbox.sNN: TaskAssignment (fan-out, dispatched by targetBeeId)
```json
{"v":1,"taskId":"0b5c...","attempt":2,"targetBeeId":"bee-3f2a1b-00009","tier":"coder",
 "messages":[{"role":"user","content":"hello"}],"replyTopic":"R.user.reply.gw-1a2b3c",
 "deadlineMs":1793100030000,"hostId":"host-9f8e7d"}
```

### hivemind.task.submit: TaskSubmit (queue "hosts")
```json
{"v":1,"taskId":"0b5c...","tier":"coder","messages":[{"role":"user","content":"hello"}],
 "replyTopic":"R.user.reply.gw-1a2b3c","deadlineMs":1793100030000,"userId":null}
```

### hivemind.task.done: TaskDone (fan-out to all hosts)
```json
{"v":1,"taskId":"0b5c...","attempt":2,"beeId":"bee-3f2a1b-00009","ok":true,"output":"...",
 "error":null,"serviceMs":412,"hostId":"host-9f8e7d"}
```

### reply topic: TaskReply
```json
{"v":1,"taskId":"0b5c...","ok":true,"output":"...","error":null,"beeId":"bee-3f2a1b-00009",
 "attempts":2,"e2eMs":1530}
```

### hivemind.bee.mesh.help: HelpRequest
```json
{"v":1,"fromBeeId":"bee-3f2a1b-00007","taskId":"0b5c...","tier":"coder","reason":"full"}
```
`reason` is `full` (no free slot) or `tier` (the tier is not served here).

### hivemind.bee.mesh.offer: HelpOffer
```json
{"v":1,"fromBeeId":"bee-3f2a1b-00002","toBeeId":"bee-3f2a1b-00007","taskId":"0b5c..."}
```

### hivemind.host.announce: HostAnnounce (every 2 s)
```json
{"v":1,"hostId":"host-9f8e7d","liveBees":1000,"liveBeesByTier":{"coder":1000,"general":1000},
 "inFlight":3,"assigned":120,"completed":115,"failed":1,"reassigned":4,"atMs":1793100000000}
```

## Behaviour

### Routing (host)
- A host keeps a registry from heartbeats and lifecycle events.
- A bee is **dead** when it has been silent for three heartbeat intervals (silence >= 3 x interval), or
  after LEAVING.
- Routing picks, among live bees that serve the tier, the one with the lowest score
  `(inFlight from its latest heartbeat + this host's assignments since that heartbeat) / maxConcurrent`.
  Ties are broken at random.
- No live bee serves the tier: `TaskReply ok=false, error="no capacity"`, immediately.

### Attempts and reassignment (host)
- The first assignment is attempt 1. If no `TaskDone` arrives within `attemptTimeoutMs` (default 10000),
  the host reassigns to a different live bee (attempt + 1), excluding every bee already tried.
- At most 2 reassignments. After that, or when no other bee is live: `TaskReply ok=false, error="no bee answered"`.
- A `TaskDone` with `ok=false` also triggers a reassignment, under the same limit. This is an extension to the
  brief: the bee has reported a failure, so another bee is tried. When no attempts remain, the error
  is the bee's error text.
- The host accepts a `TaskDone` only when its `attempt` is at least the host's current attempt. Lower
  attempts are stale and ignored. A higher attempt is a bee-to-bee handoff and is accepted.

### Handoff (bee to bee)
- A bee that receives an assignment while full (`inFlight >= maxConcurrent`), or for a tier it does not
  serve, publishes a `HelpRequest` and keeps the task for up to `helpWaitMs` (default 500).
- Every JVM with an eligible bee (serves the tier, spare slot, empty local queue, not the requester)
  answers with one `HelpOffer`. The first offer the requester receives wins: it publishes the same
  assignment with `attempt + 1`, `targetBeeId` set to the helper, and the same `hostId` and `replyTopic`.
- If no offer arrives in time, the task is queued locally (bounded queue, default 16). When the queue
  is full or the tier is not served, the bee publishes `TaskDone ok=false`.
- A handed-off assignment that finds its helper busy is queued locally and is not handed on again.

### Finalisation (host)
- Only the host whose `hostId` matches a `TaskDone` finalises that task. Each task is therefore answered
  exactly once, and only the host publishes the `TaskReply`.

### Deviation: TaskReply ownership
The brief says that a bee publishes the TaskReply for the final attempt outcome. v0.1 instead has the
**host** publish every TaskReply, and bees never publish one. Reasons: the exactly-once requirement
(two hosts share the queue), and a single writer that knows the attempt budget. The reply contents
are the same as the brief describes.

### Gateway behaviour (HTTP, not on the bus)
- `POST /v1/chat/completions` returns an OpenAI-compatible body with an extra `hivemind` object
  `{beeId, attempts, e2eMs}`. Mesh failures (`ok=false`) return 503 with `error.type = "mesh_unavailable"`.
  Timeout returns 504 (`gateway_timeout`).
- `GET /mesh/health` returns the latest HostAnnounce of every host.
- Requests whose `Host` header is not `127.0.0.1` or `localhost` get 403. Bodies above 64 KiB get 413.

## Delivery semantics (AMCP on Kafka)
- Fan-out subscriptions start from the earliest retained offset and skip events older than the
  subscription time minus 5 s. A newly started JVM therefore reads the recent past, but not the 24 h of
  retained heartbeats.
- Queue subscriptions (`hosts`) use a shared consumer group with committed offsets. Tasks queued while
  no host runs are processed when a host starts.
- Consumers use auto-commit (AMCP's KafkaEventBroker default), so a message can be lost if its JVM crashes
  between poll and handling. v0.1 gives no delivery guarantee across crashes. A lost task is not silent: the
  gateway's timeout returns 504. Duplicate deliveries are harmless: the host finalises a task once, and stale
  attempts are ignored.
