# S21 — Expert registry, adapter residency and hierarchical routing for many experts

Module: `src/hivemind/core/expert_registry.py` (`HM-S21-MOESCALE`). Tests: `tests/test_expert_registry.py`.
Timing run: `tests/bench_expert_registry_scale.py` (not collected by pytest).

## Problem

Before this change, the experts HiveMind can route to were two hand-written dicts with five entries
(`SEASON_3_DOMAIN_EXPERT_SPECS` in `btm_moe.py`, `SERVING_EXPERT_SPECS` in `serving_experts.py`). `keyword_route` scores
every keyword of every domain on every prompt. That works for five experts. It cannot answer the questions a swarm with
hundreds of community adapters has to answer:

- Which experts may receive traffic? A BYOB candidate that has not been evaluated should not.
- Which base model does an adapter belong to? A LoRA trained on Qwen3-8B is meaningless on Qwen2.5-Coder-7B.
- Which adapters fit on which GPU at once, given llama-server / vLLM multi-LoRA slot limits?
- What happens when the best expert for a request is not loaded anywhere?

## Design

### 1. `ExpertRegistry`

- **Kinds**: `lora_domain` (trained domain adapter), `serving` (a verified generator behind a model server, no adapter,
  e.g. `Python_Code_GWAYA`), `byob` (bring-your-own-bee adapter submitted by a contributor).
- **Lifecycle**: `candidate -> shadow -> live -> retired`. Legal moves are exactly:

  | from | to |
  |---|---|
  | candidate | shadow, retired |
  | shadow | live, candidate (another evaluation round), retired |
  | live | shadow (demotion), retired |
  | retired | none (terminal) |

  Every transition needs a reason and is appended to the expert's history. `register()` always admits an expert as a
  `candidate` with empty history, so nothing enters the registry already live.
- **Content addressing**: adapter kinds must carry the lowercase hex sha256 of the adapter bytes and a size in MiB. The
  registry indexes the hash: the same bytes cannot be registered as two experts, and the same bytes cannot be registered
  under a different base model (`AdapterBaseMismatchError`). `check_attachable(expert_id, base)` refuses any base but
  the training base; `verify_adapter_bytes` re-hashes downloaded bytes.
- **Persistence**: `save()` writes JSON to a temp file in the same directory, fsyncs it and `os.replace`s it over the
  target, so a crash leaves the old file or the new one. `load()` re-validates everything: kinds, hashes, duplicate ids,
  duplicate adapters, and that each expert's history replays legally from `candidate` to its stored state. A file where a
  retired expert was edited back to `live` is rejected (tested).
- `import_serving_experts()` registers the existing `SERVING_EXPERT_SPECS` entries as `serving` candidates.

### 2. `plan_adapter_residency(nodes, adapters, demand)`

Input: `ServingNode(node_id, vram_mib, base_model_id, base_resident_mib, max_lora_slots)` per GPU, where
`base_resident_mib` is everything that is not an adapter (base weights, KV reserve, runtime), an `AdapterFootprint`
(size, base) per adapter (`ExpertRegistry.residency_inputs()` builds these for live experts), and a non-negative demand
per adapter.

Algorithm: greedy placement, each adapter at most once, onto the same-base node with the most free adapter VRAM that still
has a slot (worst fit, ties by node id). It runs under two deterministic orders, by demand and by demand per MiB, and
keeps the one that serves more demand (ties keep the demand order). Zero-demand adapters stay cold. A node whose
`base_resident_mib` exceeds its VRAM is reported in `blocked_nodes` and gets nothing. Cold adapters carry a reason
(`zero_demand`, `no_node_with_base`, `no_capacity`).

This is a heuristic for a multiple-knapsack problem with a cardinality constraint, not an optimum. The hard constraints
are checked separately by `validate_residency_plan` (VRAM, slots, base match, each adapter at most once, not both hot and
cold, nothing on blocked or unknown nodes). A Hypothesis property runs the planner on random nodes and adapters and
requires zero violations; a hand-made bad plan shows the validator reports each kind of violation.

Not modelled: replicating a very hot adapter on several nodes, per-node max LoRA rank, adapter load latency, and KV
cache growth with traffic. They are follow-ups.

### 3. `HierarchicalExpertRouter`

A snapshot over the registry's `live` experts bound to one base model (serving experts are always eligible and always
resident, since they are model servers). Routing is two-level:

1. **Cluster**: the cluster whose keywords the prompt hits most (ties by name). Keyword semantics are those of
   `btm_moe.keyword_route` (whole words / phrases; a Hypothesis property checks equality against `btm_moe._keyword_hit`).
   An inverted index keeps a request at O(prompt tokens + phrase keywords) instead of O(all keywords of all experts).
   No hit at all routes to the base model. A caller can also pass `cluster_id` directly.
2. **Expert**: live experts of that cluster ranked by QueenBee recruitment weight (`RecruitmentReport.weights`, optional),
   then keyword hits, then id.
3. **Fallback**: a cold winner falls back to the best-ranked resident expert of the same cluster, then to the base model.
   With `allow_cold_load=True` the cold winner is returned with `load_on_demand=True` instead.

`candidate`, `shadow` and `retired` experts are never routed; a Hypothesis property over random registries, states,
resident sets and weights checks that every decision is the base model or a live expert that is resident (or explicitly
`load_on_demand`).

## Measured timing

Command: `PYTHONPATH=src .venv/bin/python tests/bench_expert_registry_scale.py`, run 2026-10-28 on the development host:
Intel Core i7-4930MX @ 3.00 GHz, 8 logical CPUs, 31 GiB RAM, Linux 6.8.0-147, Python 3.12.3, single thread, other jobs
running (load average about 3). Experts are synthetic (generated ids, keywords and hashes; seeded sizes 30-160 MiB);
80% live, 10% shadow, 10% retired; 16 nodes of 15360 MiB with a 9000 MiB base and 32 LoRA slots; demand Zipf(1) over a
random order; 2000 routed requests each naming one cluster keyword and one expert keyword.

| experts | clusters | register | lifecycle (2-3 transitions each) | plan | router build | route / request | save | load | JSON |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | 8 | 8.3 ms | 4.8 ms | 2.8 ms | 0.6 ms | 35.7 µs | 13.7 ms | 3.6 ms | 46 KB |
| 1000 | 20 | 43.4 ms | 45.1 ms | 19.4 ms | 5.4 ms | 121.8 µs | 89.1 ms | 59.8 ms | 468 KB |
| 5000 | 100 | 253.1 ms | 257.2 ms | 66.4 ms | 49.3 ms | 145.8 µs | 417.3 ms | 200.2 ms | 2.36 MB |

Plans had 0 validator violations at every size. With 16 x 32 = 512 slots, the hot set was 80 / 512 / 512 adapters and
served 100% / 93.9% / 76.8% of the synthetic demand. Per-request routing cost grows with cluster size (about 10 live
experts per cluster at 100, about 40 at 1000 and 5000), not with the total number of experts. These are timings of
bookkeeping code on synthetic inputs; they say nothing about the quality of any real expert.

## Interfaces with the other S21 PRs

- **byob** (`swarm/byob.py`): a validated submission becomes `registry.register(ExpertRecord(..., kind="byob",
  adapter_sha256=..., base_model_id=...))` and starts as a candidate; its validation verdict drives
  `transition(..., "shadow", reason)`.
- **orch** (`swarm/campaign_orchestrator.py`): the night dream phase produces new adapters that enter as candidates;
  promotion to live stays a gated transition.
- **basemodels** (`core/base_model_catalog.py`): a base upgrade means new adapters on the new base id; the content
  binding prevents attaching old adapters to it. `ServingNode.base_resident_mib` should come from the catalog's VRAM fit.
- **moescale <-> queenbee**: pass `RecruitmentReport.weights` as `recruitment`.

None of this wiring is in this PR.
