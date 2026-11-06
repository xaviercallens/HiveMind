# ALIXQueenBee: a queen that splits work, routes it to federated experts, and learns from the dance

Status: **proposal, nothing here is built or measured.** Owner's brief (2026-10-25): a Queen of the HiveMinders splits a request into smaller tasks and routes them to distributed, federated MoE experts running on several GPUs; bee-inspired signalling (the waggle dance) reports what each expert learned so the others improve; reinforcement learning from end-user feedback runs locally; developers who use the OpenRouter contribute their own corrections to the shared knowledge.
Grounding: what exists today is listed in section 2; what is new is section 3 onward. Citations in section 9 are from memory and must be checked before they go in a paper.

## 1. Goals and non-goals

Goals: (G1) route sub-tasks to the expert most likely to solve them, using evidence and not keywords alone; (G2) turn verified outcomes (a test passed, a developer accepted a patch) into a shared, hard-to-game signal; (G3) let a developer's own corrections improve a local model without sending their code anywhere by default; (G4) share improvements across producers only through gates that can reject them.
Non-goals: no new token, no cash value for HMT, no claim that routing beats a single large model until measured, no training on raw prompts of other users.

## 2. What already exists to build on

| Piece | Where | Role in ALIXQueenBee |
|---|---|---|
| OpenAI-compatible gateway, accounts, HMT ledger, producer agent, tiers | `hivemind.router` | transport and payment for worker calls |
| Kelly proportional-allocation market with a certificate | `router/market.py` | prices; the dance adds quality, not price |
| Verified mode (generated Python kept only if the caller's own tests pass) | `scripts/hivemind_verified.py` | the reward source for coding tasks |
| Keyword and learned routers, whole-word routing with abstention | `core/btm_moe.py`, `core/serving_experts.py` | baseline the dance must beat |
| Decoupled DiLoCo quorum, Multi-Krum | `swarm/decoupled_diloco.py` | federated aggregation of adapter updates |
| La Ruche vault of domain experts | `swarm/la_ruche_vault.py` | where accepted experts are deposited |
| Chain-of-command firewall, PR/scan gates | `security/` | untrusted text and updates are data, never instructions |
| NIGHT-MOE-1 harness: paired McNemar, shuffled and cross-domain controls | `docs/NIGHT_MOE_RESULTS.md` | the acceptance test for any new expert |

## 3. Roles

- **Queen (planner).** Receives a request, builds a small task graph (a DAG of sub-tasks, each with a verifier when one exists: unit tests, a numeric check, a schema), assigns each node to an expert, and merges results. The Queen holds no weights worth stealing and no ledger power; it is replaceable and several can run. A Queen is itself an expert-tier model behind the gateway, so it can be evaluated like any other.
- **Workers.** Producer-hosted experts (tiers, adapters) on different GPUs, reached through the existing gateway.
- **Scouts.** Probe producers with fixed, held-out verified tasks to measure what an expert can really do (extends the existing tier-probe idea). Scouts are how a new adapter earns its first dance.
- **Foragers (end users).** Developers whose IDE calls the router. Their accepted or corrected outputs are the richest reward signal, and stay on their machine unless they opt in.

## 4. The dance: a compact, verifiable report

A waggle dance encodes a direction, a distance and a quality, and followers are recruited in proportion to quality, while dances fade when the source is exhausted. The mapping:

| Bee | HiveMind |
|---|---|
| direction | the task cluster: a prototype id in embedding space (not free text) |
| distance | cost: tokens, latency, HMT per 1k tokens |
| vigour | verified success rate with a Wilson interval and the count n, from outcomes with a verifier |
| recruitment | routers pick an expert with probability proportional to a lower-confidence-bound score, with exploration kept by Thompson sampling |
| fading | a dance record expires; no new verified outcomes means no recruitment (stigmergy) |

A **dance record** is a signed, append-only entry: (expert id, task-cluster id, n, successes, cost, epoch, verifier kind). Only verified outcomes count. Records are aggregate statistics: no prompts, no code. Influence per identity is capped, and a record from an expert that is not the one that served the task is rejected. The record is the thing producers and routers share, so improvement is coordinated by evidence rather than by trust.

## 5. Local reinforcement learning from developer feedback

Three layers, cheapest first, each opt-in and local:
1. **Contextual bandit over experts** (client side): context = task cluster, action = which expert, reward = verified accept. No weights change; it only learns who to ask. Cost: negligible.
2. **Preference pairs** (client side): a developer's correction turns a rejected completion and the corrected one into a pair, kept only when a verifier confirms the corrected one (tests pass). Pairs train a small local LoRA adapter with a preference loss on the developer's own machine or a rented GPU.
3. **Federated sharing** (optional, gated): a clipped, noised adapter delta, or just the dance records, goes to the quorum aggregator (Multi-Krum plus the DiLoCo quorum). Raw pairs never leave the machine by default. The effect of noise on quality is unmeasured.
Dilution rule inherited from the repo: unverified signal is capped at 30% of any training mix.

## 6. Acceptance gate for anything shared

A shared update (adapter, router weights) is accepted only if, on a frozen held-out set it never trained on: (a) it beats its own matched control by a paired exact test (the NIGHT-MOE-1 lesson: compare against the control, not only against base); (b) it does not lose on the interference grid, run on the full packs and with unparsable counts reported; (c) the routed system, not just the isolated adapter, is no worse than before. Rollout: scouts, then a canary share of traffic, then the tier. Rollback is one switch.

## 7. Threats and the defence for each

| Threat | Defence |
|---|---|
| Sybil or inflated dances | influence caps, verified outcomes only, identity cost (sign-in), cross-check by scouts |
| Poisoned or backdoored adapter deltas | Multi-Krum, clipping, the section 6 gate, canary traffic |
| Prompt injection through task text | firewall: task text is data; the Queen's plan is validated against a schema before execution |
| Privacy leak through gradients | opt-in only, clipping and noise, no raw pairs shared; producers already see prompts, which the docs must keep saying |
| Free riding | HMT only for served or verified work |
| A Queen that mis-splits | a fall-back to single-expert routing, plus plan-level verification where a verifier exists |

## 8. Experiments (to preregister before running)

- **QB-1 routing.** Dance-weighted routing against keyword, learned and static routing on a mixed pool (BigCodeBench-style code, GSM8K, and held-out domains). Metric: end-to-end verified success; paired tests; start with simulated producers on CPU, then real ones. Known weakness to beat: the keyword route abstains on 65% of keyword-blind Python (GWAYA measurement, dev set).
- **QB-2 decomposition.** Queen against single-shot on multi-step tasks that have a verifier per step. Metric: success, tokens, latency. The null (no gain) is a live possibility.
- **QB-3 local learning.** Bandit regret on replayed, anonymised feedback logs first; synthetic users second; real consenting users last.
- **QB-4 robustness.** Fraction of dishonest dancers from 0 to 40%: routing quality and detection rate.
Each experiment needs a positive and a negative control, full-size sets, and a stated Bonferroni family.

## 9. References (from memory; verify before citing)

- T. D. Seeley, *Honeybee Democracy*, Princeton University Press, 2010 (dance-based decision making).
- D. Karaboga, *An idea based on honey bee swarm for numerical optimization*, Erciyes University technical report TR06, 2005 (artificial bee colony).
- P. K. Visscher, *Group decision making in nest-site selection among social insects*, Annual Review of Entomology 52, 2007.
- McMahan et al., *Communication-efficient learning of deep networks from decentralized data*, AISTATS 2017 (federated averaging).
- Blanchard et al., *Machine learning with adversaries: Byzantine tolerant gradient descent*, NeurIPS 2017 (Krum).
- Rafailov et al., *Direct preference optimization*, NeurIPS 2023.

## 10. Phases and decisions for the owner

1. Dance record format, signing and aggregation, with QB-1 on simulated producers (CPU, no new infrastructure).
2. Bandit router in the IDE connector, opt-in, with QB-3 on replayed logs.
3. Queen planner as an expert tier, QB-2.
4. Preference-pair local training and the federated gate (needs a GPU and the privacy review).
Decisions: opt-in and consent wording for feedback sharing; whether dance records are public; which tier hosts the Queen; whether to adopt the name ALIXQueenBee in the public docs. Nothing is claimed as built.
