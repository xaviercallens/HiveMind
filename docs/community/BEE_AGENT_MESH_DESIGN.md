# Bee agent mesh: scaling HiveMind inference across producers (design, not built)

Status: proposal for the owner's decision, 2026-10-27. Nothing here is deployed or measured. Numbers below are either from the repository or are marked as the project's own claims.

## What exists today (verified in the repository)

- **Routing is by declared capacity.** On `main`, `src/hivemind/router/gateway.py` picks producers by the `capacity_1k` each producer declares in its heartbeat, clamped by an operator maximum (`max_producer_capacity_1k`). When one account serves several URLs, load is spread by those declared shares with the rest kept as fallbacks. The gateway has no live signal of queue depth, in-flight requests or GPU state.
- **One endpoint per account** until PR #206 (multi-endpoint, branch `feat/router-multi-endpoint`) is merged and the gateway redeployed.
- **Measured on the owner's T4 (memory note, 25 Oct):** a coder-lite request made the next 7B request about 8 s slower, because Ollama's `MAX_LOADED_MODELS=1` swaps models. A single GPU serving two tiers is a scaling limit independent of routing.
- **Single router.** All consumer traffic enters one Cloud Run service (`hivemind-openrouter`).

## What AMCP is (checked, not verified in use)

- The core is a Java framework (`agentmeshcommunicationprotocol/amcpcore.github.io`, last pushed 2025-11-10, no stars). It provides agents, event routing, optional Kafka transport, and agent mobility.
- The landing page quotes performance figures ("< 100 ms startup", "1M+ events/sec"). These are the project's own claims. I have not reproduced them.
- Nothing in HiveMind imports or runs AMCP. Adopting it means a JVM runtime on each producer, a Kafka cluster for the bus, and a Java port of the producer path. That is a large change.

## Options

| Option | What changes | Cost | Fixes which limit |
|---|---|---|---|
| A. Load-aware HTTP mesh (recommended first) | Producers report in-flight count, queue depth and loaded models in each heartbeat; gateway routes to least-loaded healthy producer and fails over on 429/503. Builds on PR #206. | Small: Python only, same HTTP protocol, no new runtime. | Self-declared capacity, failover, swap-induced latency (route by loaded model). |
| B. AMCP agents on Kafka | Each producer runs an AMCP agent; heartbeats and work go through Kafka topics; agents can migrate. | Large: JVM, Kafka, a Java producer path, new ops burden. | Only worthwhile with many producers across regions. Not justified by the current single T4. |
| C. Peer gossip between producers | Producers route to each other directly. | Large, and it weakens the gateway's payment and audit role. | Not recommended now. |

**Recommendation:** A now, with two concrete tests before any B work: (1) a load test with two producers (the T4 plus one more) measuring p50/p95 latency and failover time under the gateway; (2) only if the router becomes the bottleneck after A, consider moving the transport (not the agent logic) to a message bus.

## Bee agent mesh, as the design term

"Bee" here names the producer role (a single-GPU expert or tier). A bee agent is one producer's control loop: it advertises loaded models, measured capacity and health; it accepts or refuses work by a local budget; and it reports completions back to the gateway. The mesh is the set of such agents plus the gateway's routing rules. Option A implements this with plain HTTP heartbeats. The AMCP framing would only change the transport.

## What was NOT verified, and why

- **The 8095/8092 "RTX 2070 producer" status pasted into the session on 27 Oct:** nothing listens on those ports on this machine, and the machine has one GPU (the Tesla T4). That status is not evidence and nothing here depends on it.
- **The night-training probe (Granite 1B, micro-batch 1):** not run. The T4 serves live traffic (about 1.5 requests/s on nginx, never idle), and the memory note says probes run one at a time while idle. Running it now would slow user requests. It needs a window the owner approves.

## Decisions for the owner

1. Approve option A as the next scaling step, and its load test with a second producer.
2. Merge PR #206 and redeploy the gateway (the session does not merge).
3. Choose the probe window (for example a Paris night slot with the governor's HOLD set), or accept probing under live traffic at reduced batch.
4. Confirm whether AMCP should remain a candidate for later (option B) or be dropped.
