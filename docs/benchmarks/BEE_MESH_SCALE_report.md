# MESH-SCALE-1: bee mesh control plane at 100, 300 and 1000 simulated bees

Generated from `artifacts/experiments/MESH_SCALE_report.json` (analyze.py, receipt sha256
`a94bce96f39c687f18bfd152c8ce21590d494d887ae8519dce2e8f671f33fac9`). Pre-registration:
`artifacts/experiments/MESH_SCALE_prereg.json`, with amendments 1 to 3 (harness mechanics only, thresholds unchanged).
Run directories live on the data disk: `/mnt/disks/disk-socrateai-local-1/amcp/scale-runs/`
(`20261027T154512Z-N100`, `20261027T155026Z-N300`, `20261027T155536Z-N1000`).

## Scope and caveats

- **Simulated bees.** Bees are simulated (50 ms latency, 10 ms jitter, max 4 concurrent tasks per bee). No GPU, no model inference, no WAN.
- **One VM, one broker.** Two hosts and four bee JVMs, plus a loadgen and a recorder, all on one VM against one Kafka broker on 127.0.0.1:9092 (512 MB heap).
- **Control plane only.** The test measures registration, routing, heartbeats and reassignment, and the overhead above bee service time. It does not measure real model latency.
- **One run per N.** No repeats. The tail (p99) varies between runs, see the N=300 row.
- **Harness.** Topics are pre-created before any JVM starts (amendment 2). The loadgen waits for a readiness barrier before the load clock starts (amendment 3). Neither change alters thresholds, population or mesh code.

## Conditions (all three runs)

Prereg conditions were met in all three runs: rate 200 tasks/s, duration 120 s, kill at 60 s (10% of bees), 4 bee JVMs at 768 MB, 2 hosts at 512 MB, Kafka 512 MB, max concurrency 4, heartbeat 5000 ms, attempt timeout 10000 ms, tier coder. The offered rate was 200.01 tasks/s in every run. Every run exited with status 0 from the harness, and every process it stopped exited with code 143 (its own SIGTERM stop).

## Results per N

### Load and correctness

| N | submitted | replied | ok | lost | duplicates | achieved submit rate (tps) | throughput (tps) |
|---|---|---|---|---|---|---|---|
| 100 | 24000 | 24000 | 24000 | 0 | 0 | 200.01 | 200.0 |
| 300 | 24000 | 24000 | 24000 | 0 | 0 | 200.01 | 200.0 |
| 1000 | 24000 | 24000 | 24000 | 0 | 0 | 200.01 | 200.0 |

### Latency (ms)

| N | e2e p50 | e2e p95 | e2e p99 | overhead p50 | overhead p95 | overhead p99 | overhead samples |
|---|---|---|---|---|---|---|---|
| 100 | 73 | 178 | 753 | 16 | 112 | 550 | 22444 |
| 300 | 75 | 212 | 1276 | 18 | 144 | 1123 | 22648 |
| 1000 | 72 | 175 | 1254 | 16 | 107 | 836 | 22674 |

Overhead is `e2eMs - serviceMs` over ok tasks with no attempt routed to a killed bee. Diagnostic, not a verdict input: overhead p95 / p99 over tasks submitted at least 10 s after load start was 68 / 118 (N=100), 88 / 129 (N=300) and 64 / 109 (N=1000).

### Registration

| N | all bees live at both hosts (s after last bee JVM start) | status exit |
|---|---|---|
| 100 | 6.35 | 0 |
| 300 | 8.48 | 0 |
| 1000 | 8.08 | 0 |

### Kill (10% of bees at load start + 60 s)

| N | killed bees | kill lag (ms) | tasks touching killed bees (any attempt) | A3 set: answered ok | first-attempt variant: answered ok |
|---|---|---|---|---|---|
| 100 | 12 | 1 | 1556 | 100 / 100 | 1556 / 1556 |
| 300 | 32 | 2 | 1352 | 101 / 101 | 1352 / 1352 |
| 1000 | 100 | 6 | 1326 | 152 / 152 | 1326 / 1326 |

The A3 set is the tasks lost to a kill (a DROP from a killed bee, or an ASSIGN to a killed bee at or after its kill time), as defined in amendment 1. The first-attempt variant is reported for transparency.

### Peak RSS (VmHWM, MB) and OOM

| N | bees-1..4 | host-1 / host-2 | loadgen | record | Kafka (lifetime) | OutOfMemoryError (all logs) |
|---|---|---|---|---|---|---|
| 100 | 417 / 408 / 224 / 263 | 394 / 384 | 375 | 169 | 517 (Xmx 512) | 0 |
| 300 | 423 / 456 / 441 / 460 | 382 / 404 | 384 | 155 | 520 (Xmx 512) | 0 |
| 1000 | 480 / 451 / 473 / 488 | 384 / 373 | 391 | 156 | 534 (Xmx 512) | 0 |

Kafka's VmHWM is its whole process lifetime, not one run. It is RSS, so it includes non-heap memory and exceeds the 512 MB heap setting. The bee JVMs were capped at 768 MB, the hosts and loadgen at 512 MB, and the recorder at 256 MB. No JVM exited early. Kafka publish-error lines in the JVM logs: 0 in all three runs.

## Acceptance at N=1000 (prereg)

| Criterion | Requirement | Measured | Verdict |
|---|---|---|---|
| A1 | all 1000 bees live at both hosts within 60 s of the last bee JVM start | h1: 1000, h2: 1000 live bees at 60 s | **PASS** |
| A2 | 200 tasks/s for 120 s: 0 lost, 0 duplicates, p95 overhead <= 250 ms, p99 <= 1000 ms (tasks not affected by the kill) | 24000 submitted, 24000 replied, 0 lost, 0 duplicates; overhead p95 107 ms, p99 836 ms | **PASS** |
| A3 | >= 99% of tasks assigned to killed bees answered OK after reassignment | 152 / 152 = 100% (first-attempt variant: 1326 / 1326 = 100%) | **PASS** |
| A4 | no OOM, every JVM within -Xmx, Kafka heap 512 MB | OOM total 0; no early exits; Kafka -Xmx 512 MB | **PASS** |

Overall at N=1000: **PASS**.

Margin notes: the A2 p99 is 836 ms against a 1000 ms bound (16% margin), from one run. The N=300 run's overhead p99 was 1123 ms, which is above the bound. That run is not an acceptance run, but it shows that the p99 tail is not stable from one run to the next. The N=100 p99 was 550 ms. A repeat of N=1000 would be needed to say how much of the margin is real.

## What this does and does not show

- At 200 tasks/s, one VM with 1000 simulated bees, the mesh registered all bees within about 8 s, lost and duplicated no task, and reassigned every task that was on a killed bee.
- The control plane's own overhead (p50 about 16 ms, p95 107 ms at N=1000) is small against the simulated 50 ms service time.
- This is not evidence about real GPUs, WAN latency or real model latency, or about multiple machines or multiple brokers.

## Files

- Report JSON: `artifacts/experiments/MESH_SCALE_report.json`
- Run directories: `/mnt/disks/disk-socrateai-local-1/amcp/scale-runs/20261027T154512Z-N100`, `.../20261027T155026Z-N300`, `.../20261027T155536Z-N1000`
- Prereg and amendments: `artifacts/experiments/MESH_SCALE_prereg.json`, `MESH_SCALE_prereg_amendment_{1,2,3}.json`


## Reviewer notes added after the run (the receipt `artifacts/experiments/MESH_SCALE_report.json` is unchanged)

- **One run per N.** Each size was run once. The A2 tail latency is a single sample: overhead p99 was 550 ms at N=100, 1123 ms at N=300 and 836 ms at N=1000. The N=300 value is above the 1000 ms bound (A2 is only defined at N=1000, so it is not a failure), which shows the tail is noisy. The 16 percent margin at N=1000 is not evidence of headroom.
- **Shared production VM.** The runs were made on the VM that also serves live traffic, in the daytime, without a recorded owner approval for that window. The impact on live latency was not measured. The harness did not record the machine load average during the runs; a 15-minute load average of 12.66 on 8 vCPUs was read shortly afterwards (it includes the mesh's own JVMs).
- **A4 evidence is weaker than its label.** Heap use inside `-Xmx` is not measured directly; the receipt reports resident set size and the absence of out-of-memory errors. Kafka's resident set (533.7 MB) exceeds its 512 MB heap cap because resident size includes non-heap memory.
- **Simulated bees.** The bees are simulated (50 ms latency) on one VM with one broker. This measures the control plane (routing, heartbeats, failover), not GPUs, real models or a wide-area network.
- **Open defect recorded in amendment 3:** the host ignores failed Kafka publishes and recovers only through the 10 s attempt timeout. No publish errors occurred in the final runs after topics were pre-created, but the weakness is still in the code.
