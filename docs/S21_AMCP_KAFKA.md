# S21 — AMCP Kafka transport

Module: `src/hivemind/swarm/amcp_kafka_transport.py`
Tests: `tests/test_amcp_kafka_transport.py` (unit), `tests/test_amcp_kafka_broker_integration.py` (live broker, skipped unless configured)

## Problem

`amcp_kafka_campaign_bus.py` defines the AMCP v1.5 event (`AMCPKafkaEvent`), the four topics and
`InMemoryKafkaBridge`, a per-topic deque that the gateway's public `/v1/hivemind/amcp/events` feed reads.
Nothing in it reaches a broker, nothing is signed, and a consumer has no delivery guarantee. This PR adds
the transport layer underneath. The existing module and its tests are unchanged.

## Design

### Transport

`Transport` is a structural protocol with six operations: `publish(topic, key, value)`, `subscribe(topics)`,
`poll(max_records, timeout_s)`, `commit(record)`, `seek(topic, partition, offset)` and `close()`. `seek`
is needed on top of the four the brief named: at-least-once redelivery after a handler failure means moving
the fetch position back.

| Implementation | Use | Notes |
|---|---|---|
| `InMemoryTransport` | tests, single process | `crc32(key) % partitions`, per-partition logs with a retention cap, one consumer group's committed offsets |
| `KafkaBrokerTransport` | real broker | `confluent-kafka` (optional extra `kafka`, imported lazily); `enable.idempotence=true`, `acks=all`, `max.in.flight=5`; consumer with auto-commit and auto offset-store disabled, `read_committed` |

`InMemoryKafkaBridge` stays the gateway's read model. `AMCPPublisher(mirror=bridge)` copies each event
it publishes into the bridge, so `recent()` keeps working when the publisher is backed by either
transport.

Configuration (`KafkaSettings.from_env`):

| Variable | Default | Rule |
|---|---|---|
| `HIVEMIND_KAFKA_BOOTSTRAP` | — | required; if it is missing, readiness is BLOCKED |
| `HIVEMIND_KAFKA_SECURITY_PROTOCOL` | `SASL_SSL` if a SASL username is set, else `PLAINTEXT` | one of PLAINTEXT, SSL, SASL_SSL, SASL_PLAINTEXT |
| `HIVEMIND_KAFKA_SASL_MECHANISM` | `SCRAM-SHA-512` | PLAIN, SCRAM-SHA-256 or SCRAM-SHA-512; PLAIN over SASL_PLAINTEXT is refused (it would send the password in clear text) |
| `HIVEMIND_KAFKA_SASL_USERNAME` / `_PASSWORD` | — | both are required with SASL; credentials given without SASL are refused instead of being silently ignored |
| `HIVEMIND_KAFKA_SSL_CA_LOCATION` | — | optional |
| `HIVEMIND_KAFKA_GROUP_ID`, `HIVEMIND_KAFKA_CLIENT_ID` | `hivemind-amcp` | |

The password field is excluded from `repr`. `kafka_transport_readiness()` returns `READY`, or `BLOCKED`
with the reason (a missing variable or a missing client library). It never connects and never falls back to
the in-memory transport on its own.

### Wire format and signing

Each record is a CloudEvents 1.0 structured-mode JSON envelope (`id`, `source`, `type`, `subject`, `time`,
`datacontenttype`, and `data` = the full AMCP event dict) plus three extension attributes:

* `hmkeyid`: the key id in the keyring (`HIVEMIND_AMCP_HMAC_KEYS="kid:hex,kid2:hex"`, with
  `HIVEMIND_AMCP_HMAC_ACTIVE_KID` naming the signing key; each key must be at least 32 bytes). Every key in
  the ring can verify, which gives a rotation window.
* `hmexpiry`: event time plus TTL (default 300 s). The verifier refuses lifetimes above 3600 s, so a key
  holder cannot mint an event that never expires.
* `hmsig`: HMAC-SHA256 over the canonical JSON of the envelope without `hmsig` (sorted keys, no whitespace,
  UTF-8, NaN/Infinity refused). It is compared in constant time.

Canonical JSON here means Python `json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False)`.
It is not RFC 8785 (JCS). A producer written in another language has to reproduce the same bytes, which is
straightforward for strings and integers. For floats it needs care.

### Consume pipeline

The checks run in this order, and the first failure decides the dead-letter reason:

`oversize` (> 64 KiB) → `malformed` (not UTF-8 JSON object, or not canonicalisable) → `not_cloudevent` →
`unsigned` → `unknown_key_id` → `bad_signature` → `topic_mismatch` (an event signed for one topic and
replayed onto another) → `schema_invalid` → `envelope_mismatch` (CloudEvents id/type/subject differ from the
signed data) → `ttl_too_long` → `future_dated` (more than 30 s clock skew) → `expired`.

Rejected records go to `hivemind.amcp.dlq.v1`. The dead-letter record carries the reason, a detail string,
the source topic/partition/offset, and the raw key and value in base64 (the value is capped at 64 KiB, with
a `value_truncated` flag). The source offset is committed only after the DLQ write succeeds. If the DLQ write
fails, the record is rewound and retried. Rejected records never reach a handler.

### Per-topic schema

| Topic | Event types | Payload rules |
|---|---|---|
| `hivemind.amcp.campaigns.v1` | closed: `campaign.poll_registered` | `pledged_vram_gb` 0–1024, `availability_slot` in the four slots, `comment` ≤ 256; `subject_id` must be a known campaign |
| `hivemind.amcp.human_agent.v1` | open: any `io.hivemind.amcp.*` | payload must be a JSON object |
| `hivemind.amcp.bee_coord.v1` | closed: `bee.handoff_request`, `bee.handoff_certified` | the fields `coordinate_inter_bee_inference` emits; `verdict` ∈ {PASS, FAIL} |
| `hivemind.amcp.bee_trading.v1` | closed: `trade.offer`, `trade.bid` | `order_id`, `side` matching the type, `capacity_1k` in (0, 100000], `price_per_1k` in (0, 1000] (same bounds as the gateway's `BeeTradeOrderRequest`) |

All topics also require the exact AMCP event field set, string length bounds, a finite `utc`, and
`sender_kind` from the four participant kinds. `AMCPPublisher` validates before it signs, so an invalid
event never leaves the producer.

### Delivery, replay and order

* **At-least-once.** The offset is committed only after the handler returns (or after the record was
  dead-lettered). When a handler raises, the consumer seeks back to that offset and holds the rest of that
  partition's batch. Later records with the same key therefore cannot overtake the failed one. After
  `max_attempts` (default 3), the record is dead-lettered as `handler_failed` and the partition moves on.
* **Replay protection.** `ReplayWindow` records event ids whose handler succeeded, bounded by capacity
  (default 100,000) and by TTL. A redelivered or replayed id is committed and dropped without calling the
  handler again; this is counted, not dead-lettered, because duplicates are normal under at-least-once
  delivery. The consumer refuses a window TTL shorter than max lifetime + skew. The intent is that an event
  old enough to have left the window is already rejected as `expired`. Limitation: when more than `capacity`
  distinct events arrive within one TTL, the oldest ids are evicted early (counted in `evictions`).
* **Partition key.** Campaigns and human-agent traffic use `subject_id` (the campaign). Bee coordination uses
  `correlation_id`, so one hand-off conversation keeps its request ahead of its reply. Bee trading uses
  `sender_id`, so one bee's order sequence stays ordered.
* **Backpressure.** Each iteration fetches at most `max_batch` records and fetches nothing new until they are
  handled or rewound. The backlog stays in the broker, not in process memory.
* **Stop.** `stop()` sets a flag. `run()` exits after the current record, and records already fetched but
  not yet handled are rewound on every partition of the batch, so a resumed consumer gets them again.

## Verification run for this PR

All commands ran in the PR worktree with `/home/xavkal/xdev/AutoevolveAI/.venv/bin/python` on 2026-10-28.
The PR description has the exact output.

* Unit tests, plus the neighbouring AMCP tests, under `coverage run --source=src/hivemind/swarm`. The new
  module reaches 100% statement coverage.
* Mutation check. Six deliberate defects were applied to a copy of the module one at a time: skip the HMAC
  compare, never mark the replay window, commit on handler failure, don't hold the partition after a
  failure, skip the expiry check, key trading by event id. Each made at least one test fail.
* Live-broker test: **not run.** This host has no docker or podman and no installed Kafka. Downloading the
  Apache Kafka 4.3.2 binary worked (SHA-512 matched), but starting it locally was refused by this session's
  permission policy, so no broker was ever reachable. The test skips with
  `live Kafka not configured: HIVEMIND_KAFKA_BOOTSTRAP is not set`.

## Follow-up wiring (not in this PR)

* Gateway: the `/v1/hivemind/amcp/*` endpoints still write to the SQLite store only. To put them on Kafka,
  build one `AMCPPublisher(KafkaBrokerTransport.from_env(), HMACKeyring.from_env(), mirror=bridge)` when
  `kafka_transport_readiness()` reports READY. When it reports BLOCKED, surface the reason. The generic
  `/amcp/publish` endpoint accepts any `event_type` on any topic, while three of the four topic schemas are
  closed. Before wiring, that endpoint must either restrict free-form events to `human_agent` or map them to
  declared types; otherwise the publisher will refuse them with `SchemaError`.
* S21 orchestrator (`campaign_orchestrator.py`), BYOB (`byob.py`) and the expert registry can consume through
  `AMCPConsumer.register(event_type, handler)`. Any new event types they introduce need a `TOPIC_SCHEMAS`
  entry; the alternative is the open `human_agent` topic.
* Topic provisioning (partitions, replication factor, `min.insync.replicas=2` so that `acks=all` has meaning)
  is a broker-side operation and is not done here.
