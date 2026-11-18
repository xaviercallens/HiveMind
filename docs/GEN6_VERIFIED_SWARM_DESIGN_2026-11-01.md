# Gen 6 design: a verified swarm, built on what is measured to be true (2026-11-01)

Status: design proposal, not implemented. Nothing here is deployed, and no live configuration is changed by this document.
Base: `main` at `f651a35`. Continues [`VISION_AND_NEXTGEN_SWARM_ROADMAP.md`](VISION_AND_NEXTGEN_SWARM_ROADMAP.md) (Seasons 29-33) and the Gen 1-5 table in [`../README.md`](../README.md) (L119-125).

Source of every measurement below: the independent verification report `docs/studies/HIVEMIND_ROUTER_VERIFICATION_2026-11-01.md` in the AutoevolveAI repository (worktree `openai-math-discovery`), report time 2026-11-01 about 09:45 UTC, called "the report" below. I did not re-measure anything; numbers marked "(report)" are quoted, not independently reproduced. Code references (`file:line`) were read on `main` at `f651a35`. Anything not measured or read is marked **Assumption**. The report's own limits apply: it sent 0 chat completions (no safe credential), so "served by model X" is inferred from agent command lines and ledger telemetry, not confirmed per response.

Legal frame: HMT ("HiveMind Reflexion Points", see `src/hivemind/router/points_wording.py:2`) and `$HIVE` are non-monetary, non-transferable accounting units. The project is non-commercial and nothing in this design pays, prices in currency, or promises money ([`../LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](../LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md) sections 1 and 2). Reciprocity is compute quota only, "AS IS / AS AVAILABLE".

---

## A. Honest baseline: claimed versus measured

### A.1 Table

| # | Claim (where) | Measured (report) | Verdict |
|---|---|---|---|
| 1 | "24 specialized MoE micro-experts (Bees)" live (README L98, L123, L166) | 24 alias ids with status `LIVE_NOCTURNAL_MOE` normalise to 7 of 14 tiers (`catalog.py:1375` `normalize_model_alias`); none of the 24 appears in `/v1/models` | Not verified as hosted experts. They are names. |
| 2 | Four tiers are MoE (`moe-coder`, `moe-general`, `moe-qwen35-bee`, `bee-ruche-moe`); `moe-coder` catalogued at 24 GB min VRAM | Agent C advertises 14 models and forwards 13 of them to `qwen2.5-coder:7b-instruct` (agent args; the gateway forwards the producer's `local_name`, `gateway.py:1017`). The host is one 16 GB T4 | Not verified, contrary evidence (inference; no response-level confirmation) |
| 3 | "15 live models" (README L135) vs "7 live models, 4 producers" (README L228, L240, `portal_page.py:1154`) | Live `/v1/models`: 15 entries = `hivemind/auto` + 14 models; 3 producer accounts | 15 matches only if `auto` is counted; 7 and 4 are stale |
| 4 | Fleet of independent producers | 3 producer accounts, 3 endpoints, **1 distinct physical GPU**, `shared_host_endpoints` 3 (`scale-100-status`) | Hardware redundancy is 1 |
| 5 | Per-model redundancy / failover | Failover is coded and tested (`gateway.py:1090-1128`); 12 of 14 models have exactly one live producer, on that one GPU | Partially verified |
| 6 | Load split in proportion to capacity (docstring `gateway.py:5`; smooth WRR at `gateway.py:405-428`, applied at `:983-994`) | Last 8 h, 3 live producers, N=93: top producer 94.6% (Jain 0.371; capacity-normalised 0.426). Per model: P2 got 4 of 62 `coder` requests (expected 1/3, binomial p about 5e-7), P3 got 1 of 31 `bee-stars` (expected 1/5, p about 9e-3). All-time, 6 producers: Jain 0.549, max share 0.465 | Falsified in recorded data |
| 7 | Likely cause of row 6 | Active producers are only those with `cost < keep*price` (`market.py:184`, `keep = 0.85`); the posted price is `cheapest/keep*(1+margin)` (`market.py:229-230`, `posted_margin = 0.05` at `market.py:52`). With cheapest 0.35 the threshold is 0.3675, so cost-0.4 producers are excluded | **Inference by the report**, not confirmed (`/v1/hivemind/market` needs a key). The S-B simulator tests it first. |
| 8 | Residency-aware routing HM-S16-06 (README L123) | `residency_reporting_endpoints = 0`, no `loaded` field in `/v1/models`; the code path exists (`gateway.py:608-636`, sort at `:994`) and ranks everyone "unknown" | Not in production |
| 9 | "100-producer P2C residency-aware dispatcher" | Referenced only by `/v1/hivemind/producers/scale-100-simulate` on synthetic candidates (`gateway.py:1300`, `:1334`) | Falsified as live routing |
| 10 | Prosumers participate | 32 prosumer accounts; 2 ever served a settled request (239 requests, 8.2%); none since 2026-10-26; prosumers issued 9.8% of requests as consumers; 22 self-served requests | Historic only |
| 11 | Model-level balance | 24 h, N=565 over 14 models: `coder` 64.6%, `bee-stars` 28.3%, 12 others 7.1% together; Jain 0.143. Top consumer is 61.8% of settled requests | Demand is concentrated; most small models look like probes |
| 12 | Reliability | Cloud Run log, 4,008 `POST /v1/chat/completions` in 7 days: 5xx 69 (1.7%), of which 502 = 45; 200-latency p50 0.31 s, p95 14.9 s | Baseline for error budgets |
| 13 | Registered fleet | 39 endpoint rows share one heartbeat timestamp to the microsecond (bulk-written, not real heartbeats); 44 of 47 endpoint rows expired or bulk-written | Registry hygiene problem |
| 14 | Single gateway | Cloud Run `--max-instances 1` (`deploy/gcp_openrouter/deploy.sh:57`), `--min-instances 0` (`:56`), concurrency 200 (`:61`); ledger is SQLite replicated by Litestream from one process (`entrypoint.sh:16-17`, `--workers 1`); WRR state in process memory | Single point of failure by design (cost control) |

### A.2 What Gen 5 actually delivers

Delivered and verified in code and tests (report: 180 router tests pass): an OpenAI-compatible gateway that authenticates, escrows, forwards to a producer chosen by capacity-weighted smooth round-robin, fails over before the first byte, settles in HMT, and is reachable on both URLs. Delivered as research artifacts (not re-examined here): the DiLoCo, optimizer and codec work and Lean specs of Gen 1-2.

Not delivered: 24 distinct served experts; any MoE-backed tier distinct from a dense 7B; measured load balance; residency routing; multi-host decentralisation; a router that is more than WRR plus failover.

One-line summary: **Gen 5 is a single-host, single-GPU, three-account gateway serving two real models (`coder`, `bee-stars`) under 14 labels, with a price filter that probably skews load.**

---

## B. Design principles

**B1. Truth in labelling.**
1. An id in `/v1/models` means a distinct served expert with (a) a manifest (section C.1), (b) a weights SHA-256, (c) a measured eval card on a frozen eval set. Anything else is listed in a separate `aliases` block, never in `data`, with `"kind": "alias"`, `"resolves_to"` and `"backed_by"`.
2. No tier may carry `moe` in its name (or be described as MoE) unless its manifest points to a checkpoint whose config declares more than one expert (check mechanised in S-D). A tier whose weights hash equals a dense model's hash is by definition not distinct.
3. Counts in README, portal and `/life-as-a-bee` are generated from `/v1/models` plus `scale-100-status`, never typed (the 7 / 15 / 24 inconsistency of A.1 row 3 is the failure this prevents).
4. Every claim about the swarm carries a tier (section G).

**B2. Measurable balance as an invariant, with SLOs.** Definitions are fixed before measuring.
- Eligible producer for model m in window W: heartbeat within TTL (`PRODUCER_TTL_S = 120`, `gateway.py:80`), offers m, circuit closed, and on a distinct **host** (attested host id, not account id; the report shows 3 accounts on 1 host).
- Load x_i = settled requests of producer i for m in W divided by its declared capacity c_i. Jain J = (sum x_i)^2 / (n * sum x_i^2).
- SLO-1: J >= 0.8 per model over a rolling 24 h, evaluated only when n >= 2 eligible hosts and at least 200 requests in W. With fewer, the status is `INSUFFICIENT` (never `PASS`). A model with 1 eligible host is shown as `SINGLE_SUPPLIER`.
- SLO-2: no eligible producer exceeds its capacity-proportional share plus 0.15 over W, except while another producer is circuit-open.
- SLO-3 (fairness floor): every eligible producer receives at least 50% of its capacity-proportional entitlement over W (prevents the 6.5% and 3.2% outcome of A.1 row 6).
- SLO-4 failover: after a producer stops answering, zero new requests are routed to it within 10 s (circuit open), and non-streaming client-visible errors caused by it are <= 1 per incident.
- SLO-5 availability: 5xx ratio over a rolling 7 days <= 1.0% (baseline 1.7%); 502 "all candidates failed" <= 0.5%.
- All thresholds are proposals (**Assumption**); the owner freezes them before S-B starts. Probe and test accounts, and self-served requests (consumer == producer), are excluded from demand-side metrics and reported separately.

**B3. Verifiable serving.** Each completion carries a signed receipt naming producer node, model/weights hash and residency epoch (C.2, S-E). Scheduled canary probes check the receipt's claims against behaviour. A receipt is an attestation (Tier C), not a proof; it gets stronger only through independent probes (S-E) and, later, zk (C.9).

**B4. Participation incentives, non-monetary only.** Producers and prosumers earn HMT accounting units and compute-quota priority as reciprocity under the charter; recognition is by anonymous ordinal by default, named only on opt-in. Forbidden wording and mechanisms: payout, revenue, withdrawal, fiat or crypto conversion, "profit" in user-facing text (note that `market.py:177-179` still uses `profit` and `surplus` as internal names; user-facing text goes through `points_wording.py`). Incentive rules: (a) honest residency reporting is the condition for being in the preferred set; (b) a producer at its fairness floor is not penalised for declaring a higher cost; (c) misreporting (receipt/probe mismatch) suspends eligibility and forfeits only the unsettled HMT of the offending requests.

---

## C. Architecture

### C.1 Expert registry (builds on S21, does not replace it)
`src/hivemind/core/expert_registry.py` already exists with kinds `lora_domain | serving | byob`, lifecycle `candidate -> shadow -> live -> retired`, and adapter/base mismatch checks (see [`S21_EXPERT_REGISTRY.md`](S21_EXPERT_REGISTRY.md)). Gen 6 makes it the **source of `/v1/models`**:
- Manifest per expert: `expert_id`, `kind` (`dense | moe_checkpoint | base_plus_adapter | alias`), `base_model` and its SHA-256, `adapter` SHA-256 (if any), `n_experts`/`top_k` read from the checkpoint config (not typed), `hardware_profile` (min VRAM, GPU it was tested on), `eval_card` (frozen eval set id, metric, value, n, date, log path), `license`, `state`.
- Only state `live` is listed in `data`; `shadow` receives probe traffic only; `alias` entries come from `PREWIRED_PARTNER_MODELS` (`catalog.py`) and are listed under `aliases`.
- A producer's advertisement must match a registered `expert_id`; the producer reports the local weights digest (for example the Ollama or llama.cpp model digest) at heartbeat. A mismatch with the manifest makes it ineligible for that id.

### C.2 Mandatory residency heartbeats, and reported
Today `loaded`, `vram_mib` and `cold_start_s` are optional (`gateway.py:485`, `:608-613`) and 0 endpoints send them. Gen 6: the heartbeat is signed (producer key registered at onboarding) and carries `host_id`, `weights_digest` per model, `loaded`, `queue_depth`, `in_flight`, `cold_start_s`, a monotone `epoch` and a nonce. Unsigned or replayed heartbeats are dropped; identical timestamps across accounts are flagged (A.1 row 13). A producer that does not report residency stays routable but ranks below reporting ones and shows `residency: unreported` in `/status`. Public `/v1/models` keeps the aggregate `loaded` field already implemented (`gateway.py:1391-1402`).

### C.3 Router v2 (replaces the selection step only, `gateway.py:983-994`)
Keep: escrow, 402 checks, the failover loop (`:1090-1128`), the residency-rank idea. Change:
1. **Price becomes a soft score, not an exclusion.** Remove the hard `c < keep*price` condition from *eligibility* (`market.py:184`); keep it only for the HMT accounting of the clearing price. Declared cost enters routing as a bounded multiplier: `score_i = (in_flight_i + 1) / c_i * (1 + lambda * (cost_i - cost_min) / cost_min)`, with `lambda` capped so cost can shift share by at most a factor 1.5 (**Assumption**: value fitted in the S-B simulator).
2. **Bounded-load power-of-two-choices:** sample two eligible producers with probability proportional to capacity, take the lower score, and reject any producer whose load exceeds `(1 + eps)` times the average (P2C: Mitzenmacher 2001; bounded loads: Mirrokni, Thorup, Zadimoghaddam 2016; Tier L). Inputs: per-model queue depth from heartbeats plus the gateway's local in-flight counter.
3. **Fairness floor:** a deficit counter per (model, producer) tracks entitlement minus served; a producer below 50% of entitlement is forced into the sampled pair. This enforces SLO-3 and needs no cross-instance state if the counter is refreshed from the shared ledger every few seconds (staleness tolerated).
4. **Sticky only for streaming sessions:** a stream stays on its producer once the first byte is sent; blocking calls are never sticky.
5. **Hedged requests** (Dean and Barroso, "The Tail at Scale", 2013; Tier L) for non-streaming calls only: after a delay equal to the model's rolling p95, send to a second producer, cancel the loser, hedge at most 5% of requests, settle once. Off for a model with fewer than 2 hosts.
6. **Circuit breakers** per (producer, model): open after 3 consecutive failures or an error rate above 50% over 20 requests; half-open probe after 30 s (**Assumption**: tuned in the simulator). The closed/open state is what SLO-4 measures.
7. The existing smooth WRR stays as the fallback ordering and as the baseline policy in the simulator.

### C.4 Horizontal scaling of the gateway
Today: one instance, in-process WRR and rate-limit state, SQLite+Litestream with a single writer (A.1 row 14). Target: stateless gateway instances, state in a replicated ledger. Constraint from the owner: **min instances 0** (scale to zero) stays.
- Routing state (WRR counters, in-flight, circuit state) becomes per-instance and advisory; correctness comes from the stochastic policy (C.3), not from a global counter.
- Ledger: settlement is already keyed by `request_id` (`gateway.py:1042-1045`), so settlement can be made idempotent (**Assumption**: needs a check that a retried settle is a no-op). Options for the owner (D2): (a) keep SQLite+Litestream as a separate single-writer ledger service that gateways call, so only the small ledger is a single point (cheapest, still a SPOF, narrower); (b) a managed database with multi-writer transactions (removes the SPOF, adds a standing cost). **Assumption**: try (a) first, with a settlement queue that tolerates ledger restarts; move to (b) only if (a) misses SLO-5.
- Raise `--max-instances` only after the ledger is out of process; start at 2. Cost bound in section D.

### C.5 Producer onboarding and WAN producers
- Gate for any "decentralised" wording: at least 3 independent physical GPU hosts, at least 2 per listed model (S-C). Today: 1.
- Registration binds an ed25519 public key to a `host_id`, with a hardware attestation line (GPU name, VRAM, runtime version as reported; **Assumption**: self-reported, so Tier C until probes agree).
- WAN/NAT producers connect **outbound-only**: the producer opens a long-lived outbound connection (long-poll or WebSocket) and receives work over it; no inbound port. Present code POSTs to a producer URL (`gateway.py:1015-1019`, including an SSRF re-check), so this is new work. Tension to resolve (D2/D4): a held connection can keep a Cloud Run instance billed and alive, which conflicts with min-instances-0 cost control, and with `max-instances > 1` a producer's connection and a client's request can land on different instances, which needs a rendezvous queue. **Assumption**: a small always-on relay (one low-cost VM, owner-approved) is simpler than making Cloud Run hold sockets; the owner prices it before S-C.
- The owner's other known hardware (RTX 2070, README L222-224; report: the P1 node of class `rtx2070-8gb-turing` last heartbeat 2026-10-31 23:42) is the fastest second host.

### C.6 Prosumer mode
A prosumer is a consumer that also serves when idle (role exists; only `producer|prosumer` may heartbeat, `gateway.py:1571-1573`). Finding: 30 of 32 never served. Gen 6: (a) prosumer eligibility needs the same signed, residency-reporting heartbeat; (b) idle-only service honouring invariants I1 (<= 6.5 GB VRAM) and I4 (< 1.5 s game preemption), README L121, stays in the agent and is not relaxed for routing convenience; (c) the router weights a prosumer by declared and probed availability, and the fairness floor applies, so an up prosumer gets its entitlement; (d) self-served requests (consumer == producer) earn no HMT.

### C.7 Observability
Server-computed from the ledger and probe results only, never from producer self-report:
- per (model, producer ordinal): request share, capacity-normalised load, error rate;
- per model: Jain (B2), max share, eligible host count, `SINGLE_SUPPLIER` flag, queue depth, p50/p95/p99, error-budget burn against SLO-5;
- global: distinct attested hosts, residency-reporting fraction, receipt-verification rate, probe pass rate.
Add `model`, `routed_model` and a producer ordinal to the Cloud Run request log (the report found none).

### C.8 Public read-only `/status` that cannot be gamed
Unauthenticated, cacheable, aggregates only, producer ordinals instead of identities, cells with fewer than 5 requests suppressed. Gaming resistance: (1) every figure is computed server-side from settled ledger rows and gateway-run probes; (2) test, probe and self-served traffic is excluded and counted separately; (3) host count uses attested `host_id`s that passed a probe in the window, not registered rows; (4) a producer cannot raise its share by self-requests; (5) the page states window, N and the definition of each metric, and shows `INSUFFICIENT` instead of a number when N is small; (6) it never prints a user count not computed from the ledger. **Assumption**: a determined actor can still register sybil hosts; the multi-operator rule (section E) limits but does not eliminate this.

### C.9 Where Seasons 29-33 slot in (only after the baseline holds)
| Season item | Vision reference | Entry gate (all measured on `/status` for 14 days) | Why gated |
|---|---|---|---|
| S29 Adaptive Multi-Split ML, HM-S29-01 | VISION L127 | S-C done (>= 3 hosts), S-E receipts live | A split across clients assumes more than one real client host; WAN claims need real WAN producers |
| S29 Top-1% quantised DCT + EF21, HM-S29-02 | VISION L128 | Same, plus its own preregistered E20 result | Training-side and independent of routing, but its "100x" claims need real multi-host links |
| S30 zk-SNARK validation (ZK-Verifiable-Swarm), HM-S30-01/02 | VISION L97-105, L129-130 | S-E done (signed receipts, probes) and 30 days of receipt-vs-probe agreement data | zk strengthens receipts; it adds nothing over unsigned, unprobed serving. Training workunits first (VISION target verify < 10 ms); inference ZKML (Kang et al., arXiv:2210.08674, cited at VISION L147) is a later research item |
| S31 GeoMesh sub-swarm MoE routing (GeoMesh-SubSwarm) | VISION L107-115 | S-D done (>= 1 real second MoE), router v2 SLOs held 30 days, >= 3 hosts across >= 2 regions | Geo routing and 15-20 node consensus presuppose many independent nodes and distinct experts; today: 1 GPU, 2 real models |
| S32 lifelong ratchet and open expert marketplace, HM-S32-01 | VISION L134 | Expert registry (C.1) with the `byob` lifecycle in production | A marketplace of experts needs the manifest and eval-card gate first |
| S33 100k-node federation, multi-region sovereignty, 5 launch waves | VISION L135, L139-146 | All of the above | The wave node targets (VISION L144-146: 150, 1,200, 4,500 nodes) are not reconciled with the measured 3 accounts on 1 GPU. **Treat them as targets, never as facts**, until `/status` shows them |

---

## D. Staged plan with preregistered acceptance tests

Common rules. Each stage freezes its metric, threshold, controls and kill criterion in a dated file **before** any number is produced; reports quote tool output only. A positive control is a case that must pass (the test can detect success); a negative control is a case that must fail (the test is not vacuous). Thresholds below are proposals (**Assumption**) that the owner confirms at the decision point. Producer != verifier: whoever builds a stage does not run its acceptance check.

Cloud cost frame. Cloud Run is billed on resources while requests are active and scales to zero otherwise (`--min-instances 0 --cpu-throttling`, `deploy.sh:56-58`). Measured demand (report): 4,008 chat POSTs in 7 days with mean 200-latency 3.58 s, about 14,300 request-seconds per week (about 2,050 per day; arithmetic on the report's figures; billed instance time is lower because concurrency is up to 200). Cost therefore scales with request-seconds times instance count; a cap of N instances bounds worst-case spend at N times the single-instance worst case. I do not have the current price list, so no currency figure is stated: the owner enters the rate at D2 and each stage gives its bound as a multiple of today's. S-A, the S-B simulator and S-D add 0 cloud cost.

### S-A. Truth in labelling and README repair (days 0-7)
- Work: `/v1/models` split into `data` (distinct experts) and `aliases`; README, portal pill and `/life-as-a-bee` counts generated; MoE-named tiers marked `dense-backed (provisional)` until S-D; a `claims_lint` check comparing README/portal numbers with the live endpoints.
- Metric: number of README/portal numeric claims that differ from the endpoint values. Threshold: 0.
- Positive control: the lint run on the current README must report >= 3 known mismatches (7 vs 15 vs 24; 4 vs 3 producers). Negative control: the lint on a synthetic consistent fixture must report 0.
- Kill criterion: if the lint's false-positive rate on 20 hand-labelled lines exceeds 10%, stop and simplify the lint rather than ship it.
- Owner decision D1. Cost: 0 cloud.

### S-B. Router v2: simulator, shadow, then canary (days 5-25)
- Simulator (offline): scenarios from the report: three producers, capacities 60/30/120, costs 0.4/0.4/0.35 (agents A/B/C), 10,000 requests, 20 seeds; a second scenario with 3 hosts of different speeds; a failure scenario (kill one producer at request 5,000); a heavy-tail latency scenario for hedging.
- Simulator validity (positive control): the *v1* policy model, given the report's parameters and the posted-price rule, must reproduce the observed pattern (A and B at <= 10% of their entitlement). If v1 in the simulator instead splits in proportion to capacity, the report's price-filter inference is **falsified** and the root cause must be re-investigated before v2 is tuned (a reportable outcome, not a failure). Second positive control: the existing unit test `test_weighted_round_robin_serves_each_candidate_in_proportion` (`tests/test_router_gateway.py:789`) still passes.
- Acceptance (simulator): v2 capacity-normalised Jain >= 0.9 in 20 of 20 seeds; every producer >= 50% of its entitlement; max share within +0.15 of capacity share; SLO-4 met in the kill scenario; hedging cuts p99 by >= 20% with <= 5% extra requests. Negative control: v2 with fairness floor and price softening disabled must fail at least one of these.
- Shadow stage (live, no routing effect): compute v2's choice beside v1's for each request and log the divergence. Needs owner approval to deploy the shadow code (D3).
- Canary (live): route 5% of requests from flagged probe/test accounts for 72 h, then 10% of general traffic if clean. Metrics: 5xx <= 1.7% (baseline), p95 <= 1.1 x baseline, Jain >= 0.8 on any model with >= 2 eligible hosts. **On the present single host, Jain is computed over accounts and says nothing about hardware; report it as `SINGLE_HOST` and do not claim a balance success until S-C.**
- Kill criterion: any 5xx ratio above 2x baseline for 1 h, or no improvement in simulator Jain over v1, or hedging load > 10%: revert the flag and publish the data.
- Cost: 0 cloud delta (same instance, `max-instances` unchanged).

### S-C. Multi-host (days 15-60)
- Work: second and third independent physical GPU hosts (first candidate: the owner's RTX 2070), signed heartbeats (C.2), host attestation, the outbound-only WAN producer path (C.5), registry hygiene for the 39 bulk-written rows.
- Metric: distinct attested physical hosts reporting in the last 7 days, each with >= 95% heartbeat uptime; hosts per listed model. Threshold: >= 3 hosts and >= 2 hosts per listed model (a model that fails this drops out of `data`).
- Positive control: a host added by the owner passes attestation and appears in `/status`. Negative control: today's fleet (`distinct_physical_gpus: 1`) must FAIL the gate, and a duplicate registration of the same host under a second account must be counted once.
- Kill criterion: if after 60 days fewer than 3 hosts exist, the project keeps the label "single-host" and re-plans; it does not weaken the gate.
- Owner decisions D2 (relay/ledger), D4 (hosts). Cost bound: ledger service plus optional relay (owner supplies the standing cost); `max-instances` raised to 2 only after the ledger is out of process, so worst-case Cloud Run request cost is 2x today's (**Assumption** of linear billing).

### S-D. Expert registry and one real second MoE (days 20-70, parallel to S-C)
- Work: the registry as source of `/v1/models` (C.1); serve **one** real MoE checkpoint distinct from `qwen2.5-coder`. Candidate named in the repo: `ibm-granite/granite-3.1-1b-a400m-base`, a 1.3B sparse MoE with 400M active parameters (README L242), small enough for one T4 (**Assumption** about serving support in the chosen runtime; the large `Tiel-Coder-35B-A3B` teacher of README L125 is not a candidate for a shared T4). The owner may choose another (D5).
- Metric (all required): (1) weights SHA-256 differs from every other listed model; (2) the checkpoint config declares more than one expert (read mechanically); (3) an eval card on the frozen set with n >= 200 and a paired comparison against the dense baseline using the repo's exact paired McNemar gate (GWAYA Level B, README L122); (4) a behavioural fingerprint (fixed prompts, greedy decoding) that differs from `qwen2.5-coder` outputs on >= 90% of 30 prompts (**Assumption**: threshold calibrated on two known-different models first).
- Positive control: the real MoE passes (1)-(4). Negative control: a deliberately created alias (same weights, new name) must FAIL (1) and (4); a dense model named `moe-test` must FAIL (2).
- The result need not favour the MoE: the claim is "distinct and measured", and a worse eval is reported as such.
- Kill criterion: if no real MoE can be served with p95 within 2x of the dense tier on owner hardware by day 70, the MoE-named tiers stay renamed `dense-backed` and the README says so. No new MoE claim.
- Cost: owner GPU time only; 0 cloud.

### S-E. Verifiable receipts and probes (days 30-90)
- Work: a signed receipt on every response (producer signature plus gateway countersignature) with `request_id`, public expert id, manifest hash, producer node/host ordinal, weights digest as reported, residency epoch, token counts, timestamp and response body hash; scheduled canary probes (1 tiny request per listed model per hour, 14 x 24 = 336 requests/day, using a documented probe credential, which the report found missing); an independent verifier tool in a separate code path. Today the response already carries `producer_id` but unsigned (`gateway.py:1126`).
- Metric: (a) the verifier accepts 100% of 1,000 honest receipts; (b) it rejects 100% of 1,000 tampered receipts (field flips, wrong model hash, replay, wrong signer); (c) probe/fingerprint agreement for each listed model >= 99% over 14 days; (d) a mislabelled producer is detected within 10 probes.
- Positive control: honest receipts verify. Negative control: tampered and replayed receipts, and a test producer that answers with `qwen2.5-coder` while claiming another model, must be flagged.
- Honest limit: a receipt proves what a key holder said, not that the weights ran; only probes and later zk narrow that gap. The wording stays "attested", not "verified", until a producer != verifier check has passed for that producer.
- Kill criterion: if the fingerprint probe cannot separate two known-different models (false-negative rate > 10%), do not advertise probe-verified status; keep receipts as signed logs only.
- Cost: negligible cloud (336 probe requests/day against 573 chat requests/day measured on average: 4,008 / 7); producer GPU seconds.

### Order and dependencies
S-A -> S-B (simulator can start day 5) -> S-C and S-D in parallel -> S-E (needs S-C hosts for meaningful independence) -> Seasons 29-33 gates (C.9).

---

## E. Risks, and what NOT to claim

Do not claim:
- "decentralised" while there is one host. Vocabulary ladder: `single-host` (today), `multi-host` (>= 3 attested hosts), `multi-operator` (>= 3 distinct operators and no operator above 50% of a model's traffic over 14 days); only then `decentralised`.
- "MoE" for an alias, or for any tier whose weights hash equals a dense model's.
- any user, node or PFLOPS count not computed from the ledger and `/status` (the wave node figures at VISION L144-146 and the README telemetry that the report could not reproduce are not facts).
- "balanced" or "fair" without the SLO-1/2/3 numbers, window and N beside it.
- "verified" for receipts alone.

Risks:
- The price-filter hypothesis may be wrong (A.1 row 7); S-B tests it first.
- Router v2 must not make one bad producer worse: bounded load and circuit breakers mitigate it, and the S-B kill criterion is concrete.
- Hedging increases producer load; it is capped at 5% and producers can opt out (voluntary compute).
- Sybil hosts: one actor could register several hosts; the multi-operator rule and probes limit but cannot prove independence (**Assumption**).
- Fingerprint probes can be evaded by a producer that detects them; probe prompts are drawn from a randomised pool and sent under ordinary-looking accounts (**Assumption**; the scheme needs review).
- Cost creep (relay VM, ledger DB): each stage states its bound and the owner can stop at any stage.
- Privacy: receipts name producer ordinals, never identities; `/status` never shows prompts.

Ethics and safety. Compute is voluntary; a producer can leave at any time; no money is paid or promised (charter). Game-preemption invariants stay: **I1** (<= 6.5 GB VRAM) and **I4** (< 1.5 s preemption), README L121. Router v2 must not push work onto a prosumer faster than its declared capacity and idle policy allow, and the remote pause / VRAM cap (HM-S16-05) remains honoured. No stage collects prompts beyond what settling a request needs.

---

## F. Next 14 days: one-page checklist

Day 1-2
- [ ] Owner freezes the SLO values (B2) and signs off the label policy (D1).
- [ ] Create the `claims_lint` skeleton and run it on the current README/portal; keep the output as the S-A positive control.
- [ ] Decide the probe credential (documented, test-flagged account); the report found none.

Day 3-5
- [ ] Repair README: one source for model and producer counts; mark the 24 as aliases; mark `moe-*` tiers `dense-backed (provisional)`.
- [ ] Specify `model` / `routed_model` / producer ordinal in the request-log schema (design only until deploy approval).
- [ ] Write the S-B simulator with the report's three-producer scenario; run the v1 validity control (does v1 reproduce A/B <= 10%?).

Day 6-9
- [ ] Freeze S-B acceptance thresholds in a dated file before running v2.
- [ ] Implement the v2 policy (C.3) in the simulator only; run 20 seeds with positive and negative controls; commit the logs.
- [ ] Inventory the 39 bulk-written endpoint rows; propose flagging them `kind=test` (owner decision on touching live data).

Day 10-12
- [ ] Draft the signed-heartbeat and receipt schemas (C.2, S-E); no deployment.
- [ ] Decide the ledger path (D2) and relay (D4) within the owner's cost ceiling.
- [ ] Check whether the RTX 2070 host can return as the second physical host (D4).

Day 13-14
- [ ] Publish the simulator report with raw outputs; state whether the price-filter hypothesis held.
- [ ] Owner decision D3: approve shadow-mode deployment of v2 (no routing effect).
- [ ] Update this document's status line with what ran and what did not.

Owner decisions: **D1** relabel or retire MoE aliases and freeze SLO values; **D2** ledger/relay path under min-instances-0 and a cost ceiling; **D3** approve shadow, then canary deployment; **D4** which additional physical hosts to bring online; **D5** which real second MoE to serve.

---

## G. Relation to the Elenchus method

Claims about the swarm get tiers, as the owner's other formal work does (my reading, **Assumption**: Tier A kernel-checked in Lean, Tier L literature with a pinned source, Tier C conjecture or design hypothesis, Tier X experimental measurement with its log):
- Tier A: only properties actually kernel-checked, for example a Lean lemma that Jain's index lies in [1/n, 1] or that the fairness-floor rule bounds a share. None exists yet for the router; nothing is Tier A until it passes the Lean gates.
- Tier L: P2C and bounded loads (Mitzenmacher 2001; Mirrokni, Thorup, Zadimoghaddam 2016), hedging (Dean and Barroso 2013), Jain's index (Jain, Chiu, Hawe 1984), ZKML (Kang et al., arXiv:2210.08674, already cited at VISION L147). Cite the pinned source when used.
- Tier X: every measured number in section A (Jain 0.549 and 0.371, 1.7% 5xx, 1 GPU, ...) is Tier X **with the report and its logs as the artifact**, and carries the report's caveats (no live chat test; ledger snapshot of 07:44 UTC). The price-filter explanation stays Tier C until the simulator, then a canary, tests it.
- Tier C: this design, the SLO thresholds, the expected benefit of v2 and hedging.
- Nothing is called "verified" without a producer != verifier check: the stage author does not run the acceptance test, the probe harness is separate code from the router, and receipt verification is a separate implementation. Positive and negative controls are reported with every result, and a failed gate is reported, not hidden.

---

## Résumé en français

Cette note prolonge la vision des Saisons 29 à 33 pour une « Gen 6 », mais elle part de ce qui est mesuré. Selon le rapport de vérification indépendant du 2026-11-01 : les 24 « experts MoE » ne sont que des alias qui se ramènent à 7 paliers et n'apparaissent pas dans `/v1/models` ; la flotte réelle se compose de 3 agents producteurs sur un seul GPU T4 ; 13 des 14 modèles listés renvoient vers le même `qwen2.5-coder:7b-instruct` ; la charge n'est pas équitable (Jain 0,55 sur l'historique, 0,37 récemment, un producteur à 95 % des 93 dernières requêtes, cause probable le filtre `cost < 0.85*price`, à confirmer) ; les prosumers (2 sur 32 ont déjà servi) n'ont rien servi depuis le 26 octobre ; aucun point d'extrémité ne rapporte sa résidence ; le dispatcheur P2C n'existe que dans une simulation ; Cloud Run est limité à une instance.

Principes : étiquetage véridique (un id listé = un expert distinct avec manifeste, hash des poids et fiche d'évaluation ; les alias sont séparés), équilibre mesurable avec des SLO (Jain >= 0,8 par modèle, part maximale, temps de bascule), service vérifiable (reçus signés, sondes), incitations uniquement non monétaires (unités HMT, charte de réciprocité). Architecture : registre d'experts, battements de résidence obligatoires et signés, routeur v2 (P2C à charge bornée, plancher d'équité, prix comme score souple, requêtes dupliquées pour la latence de queue, disjoncteurs), passerelle sans état avec grand livre répliqué en gardant min-instances 0, producteurs multi-hôtes (au moins 3 GPU physiques indépendants avant de parler de décentralisation) avec connexions sortantes, observabilité et `/status` public non manipulable. Les éléments des Saisons 29 à 33 (Multi-Split, DCT Top-1 %, zk-SNARK, GeoMesh) ne viennent qu'après le respect de ces invariants.

Plan en cinq étapes avec tests préenregistrés (métrique, seuil, contrôles positif et négatif, critère d'arrêt, coût) : S-A étiquetage et README ; S-B routeur v2 en simulateur puis canari ; S-C multi-hôtes ; S-D registre et un vrai second MoE ; S-E reçus vérifiables et sondes. Ne pas affirmer « décentralisé » avec un seul hôte, ni « MoE » pour un alias, ni de nombre d'utilisateurs non vérifié. Décisions du propriétaire : relabelliser les alias et figer les SLO, choisir le chemin grand livre/relais dans la contrainte min-instances 0, et décider quels hôtes supplémentaires mettre en ligne.
