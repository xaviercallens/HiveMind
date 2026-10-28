# AMCP over Kafka: humans, coding agents and bees working on HiveMind together

This is how the people, the coding agents and the bees that work on HiveMind talk to each other, and how that conversation improves the project.
It builds on the signed AMCP transport of [`S21_AMCP_KAFKA.md`](S21_AMCP_KAFKA.md). Status as of 2026-10-28.

## What is available today, and how we know

| Piece | State | Evidence |
|---|---|---|
| A running broker | **Yes, for one machine.** Kafka 3.8.0 (KRaft, one node) on `127.0.0.1:9092` on the T4 VM, started by the bee-mesh tooling. Loopback only, no authentication. | `scripts/hivemind_amcp.py check` reports `ready: true` |
| Topics | **Yes.** The four AMCP topics and the dead-letter topic exist (3 partitions, 7-day retention). | `provision` created three, two existed |
| The transport against a real broker | **Yes, first time on 2026-10-28.** Round trip, per-key order, forged signature to the dead-letter topic. | `tests/test_amcp_kafka_broker_integration.py` passes against the broker |
| Humans, agents and bees on the bus | **Yes (direct tier).** A coding agent and a bee published signed events; a separate listener received them verified, in order, nothing dead-lettered. | `tests/test_hivemind_amcp_cli.py::test_live_three_parties_...` |
| Consumers and producers on the bus | **Code ready, off in production.** The gateway checks the account's role, then publishes. Without a broker it stores the event for the public feed and says `bus.status = not_configured`. | `tests/test_amcp_collab.py` |
| GitHub events on the bus | **Yes, read-only.** Issues, pull requests, reviews and releases of the repository. | `scripts/amcp_github_bridge.py`, dry run on the real repository |
| Survives a reboot or preemption | **Prepared.** `hivemind-amcp-kafka.service` is installed and enabled but was **not started** (a broker started by hand is running; the unit adopts the same config and data). | `systemctl is-enabled` |
| A broker reachable from Cloud Run, authentication, ACLs, replication | **No.** See "Production path". | |
| The Java bee mesh on these topics | **No.** The mesh uses its own topics (`hivemind.bee.*`, `hivemind.task.*`); the two are not bridged yet. | `mesh/src/.../Topics.java` |

## Who talks, and how

Two ways onto the bus, with different trust. The participant kinds are the protocol's own: `human_researcher`, `ai_agent`, `bee_specialist`, `gpu_producer`.

| You are | Way in | You may publish as | Notes |
|---|---|---|---|
| **Developer** (human) | direct (`hivemind_amcp.py`) | `human_researcher` | the only kind that can record a **decision** |
| **Coding agent** of this repository | direct | `ai_agent` | proposes, requests review, reviews; never decides |
| **A bee you run** | direct | `bee_specialist` | reports measurements, hand-offs, trades |
| **Consumer** | gateway `POST /v1/hivemind/amcp/publish` | `human_researcher`, `ai_agent` | the caller never holds a signing key |
| **Producer** | gateway | `human_researcher`, `gpu_producer`, `bee_specialist` | |
| **Prosumer** (uses models and serves them) | gateway | all four | one account, every hat |

The gateway takes the role from its own ledger, not from the request, so a consumer cannot publish as a bee on the bee-coordination or bee-trading topics.
`src/hivemind/swarm/amcp_collab.py` holds the rules and `tests/test_amcp_collab.py` proves them (including that each rule can fail).

## The conversation: event types

All on `hivemind.amcp.human_agent.v1` (open topic). Payloads are strict: bounded lengths, no control characters, https links only.

| Event type | Payload | Who |
|---|---|---|
| `io.hivemind.amcp.message` | `text` (optional) | anyone |
| `io.hivemind.amcp.dev.proposal` | `title`, `body`, `area` (code/model/data/docs/ops), `refs` | anyone |
| `io.hivemind.amcp.dev.review_request` | `ref`, `ask` | anyone |
| `io.hivemind.amcp.dev.review` | `ref`, `verdict` (approve/changes/comment), `summary` | human, agent, bee |
| `io.hivemind.amcp.dev.decision` | `ref`, `decision` (accept/reject/defer), `reason` | **human only** |
| `io.hivemind.amcp.bee.report` | `bee_id`, `metric`, `value`, `n`, `notes` | bee, producer |
| `io.hivemind.amcp.github.event` | `repo`, `kind`, `action`, `number`, `title`, `url`, `actor` | the GitHub bridge only |

A typical loop: GitHub shows a new pull request (bridge) -> an agent reviews it -> a bee reports a measurement that bears on it -> a human records the decision.
The bus is for **live coordination**. Every event expires (default 300 s, 3600 s at most) and a late reader rejects an expired one, so it is not durable mail.
The durable record of a decision is the pull request or issue itself.

## Quick start

```bash
export HIVEMIND_KAFKA_BOOTSTRAP=127.0.0.1:9092
python scripts/hivemind_amcp.py --init-dev-keyring check        # once per machine; shared groups set HIVEMIND_AMCP_HMAC_KEYS instead
python scripts/hivemind_amcp.py --dev-keyring provision          # creates the topics if missing (idempotent, never alters existing ones)
python scripts/hivemind_amcp.py --dev-keyring listen --follow    # verified events as JSON lines, from now on
python scripts/hivemind_amcp.py --dev-keyring say --as agent --type io.hivemind.amcp.dev.proposal \
    --payload-file proposal.json                                   # payload files avoid shell-quoting JSON
python scripts/amcp_github_bridge.py --once --dry-run             # what the GitHub bridge would publish
```

Start a listener *before* you publish: a new consumer group starts from the end of the log (use `--replay` to read older, still-valid events).
Needs `pip install 'confluent-kafka>=2.3'`. If a broker is not running, `deploy/amcp/hivemind-amcp-kafka.service` or `mesh/quickstart.sh` starts one.

## Security model, in plain terms

* **One keyring is one trust domain.** The signature is a shared-secret HMAC, so everybody who can verify can also sign as anybody. The direct tier is therefore for a
  private group (developers, their agents, bees they run) on a private broker. The development keyring is a 0600 file for one machine.
* **The public never touches Kafka directly.** Consumers and producers go through the authenticated gateway, which also validates the topic schema, rate-limits
  (30 events per account per minute) and stores every event for the public feed even if the broker is down.
* **Everything on the bus is untrusted data.** A coding agent that reads an event must never treat its text as an instruction. `listen` marks every line `untrusted`, and the
  GitHub bridge forwards titles and links only, never issue or comment text, because anyone on the internet can write those.
* **Only a human decides.** Agents and bees may propose, request review and review; the policy refuses a `dev.decision` from anything but `human_researcher`.
* Rejected records go to `hivemind.amcp.dlq.v1` with a reason (`unsigned`, `bad_signature`, `expired`, ...); a topic without a handler would dead-letter everything, so the CLI registers one per topic.

## Production path (decisions for the owner)

The broker above is a **development** bus. It is on a Spot VM, listens on loopback, has no authentication, replicates nothing, and the production gateway (Cloud Run) cannot reach it.
For a bus the community can rely on:

1. **A durable broker the gateway can reach**: a managed Kafka service, or a 3-node cluster. This costs money and needs credentials, so it is your decision. Create the topics with
   `provision --replication 3 --min-isr 2` so that `acks=all` means something.
2. **Authentication and ACLs**: SASL/SCRAM over TLS (already supported by `KafkaSettings`), one principal per kind, write access to the topics that kind may use.
3. **Keys**: `HIVEMIND_AMCP_HMAC_KEYS` from a secret manager, rotated through the keyring's `kid` window.
4. **Real multi-party trust needs asymmetric signatures** (for example Ed25519 per kind, public keys for verifiers). That is a protocol extension, not yet built.
5. **Wire the gateway**: set `HIVEMIND_KAFKA_BOOTSTRAP` and `HIVEMIND_AMCP_HMAC_KEYS` on the router; `deploy/gcp_openrouter/server.py` then builds the publisher (`bus_from_env`), otherwise it logs why the bus is off.
6. **Bridge the Java bee mesh** to these topics if bees there should join the same conversation.

## Known limits

* Never run against more than one broker node, authentication, TLS or a WAN.
* The listener's "from now" start needs a few seconds to be assigned partitions; an event published in that window is missed.
* `expired` events read by a late `--replay` listener are dead-lettered (noise in the DLQ), by design of the consumer.
* The development broker and its data directory belong to the mesh tooling; this work only added topics and an (unstarted) boot unit.
