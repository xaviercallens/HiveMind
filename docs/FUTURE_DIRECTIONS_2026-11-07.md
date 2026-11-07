# Future directions after Season 32: interoperability, replication safety, and what the community actually asked for

Status: **proposal, 2026-11-07; decisions 1 and 2 of section 7 taken by the owner the same day** (cite but do not adopt the learning-at-home library; a shared Kwaai/Petals-style swarm is wanted). Nothing here is implemented. It extends, and does not replace, `ROADMAP.md` sections 3.23 (Seasons 29–32), 3.26 (Improvement Plan) and 6 (`HM-REDDIT-01..08`). Every external claim below is labelled by where it was read and when; claims I could not verify are marked as such.

## 1. What was checked, and what could not be

| Source | Read on 2026-11-07? | What it establishes |
|---|---|---|
| `Kwaai-AI-Lab/KwaaiNet` (GitHub page) | Yes | v0.6.8, MIT, 1,264 commits, 26 stars; inference only; "Petals/Hivemind-compatible" libp2p + Kademlia swarm; block sharding "under active development"; no settlement mechanism shipped; "carbon-negative" accounting is planned, no code. |
| `learning-at-home/hivemind` (GitHub page) | Yes | 2.5k stars, MIT, README says "active development stage"; latest commit date not shown on the page. HiveMind **cites** it (the DMoE paper, arXiv:2002.04013) but does **not** import it: our `hivemind.*` package is our own code. |
| `bigscience-workshop/petals` (GitHub page) | Yes | 10.7k stars, 94 open issues, **no archive banner, no deprecation notice**, README still says "Petals is a community-run system". The page does not show the last commit date, so **"seems abandoned" is neither confirmed nor refuted here**; it needs a dated check of the commit log and the public swarm's health page. |
| Les Numériques, 14 Oct 2026, on the Anthropic Frontier Red Team post of 12 Oct 2026 | Yes | Fifteen 55-dollar Android TV boxes (4 GB RAM each) ran Gemma4-26B-A4B at 0.39 tokens/s with no server; sub-30B open models do the isolated steps of self-replication but fail to chain them; GLM-5.2 infected 8/8 emulated nodes in 4 of 5 trials; Anthropic extrapolates small models reach that level by end 2026; the test bed had an unauthenticated ADB debug port, which Anthropic itself calls optimistic; Mozilla 0Din's lead calls it "marketing alarmiste". The article recommends hardening devices and preparing for persistence; the EU Cyber Resilience Act applies from 11 Dec 2027. |
| `PrimeIntellect-ai/OpenDiloco` (GitHub page) and arXiv:2407.07852 | Yes | The paper (Jaghouar, Ong, Hagemann; submitted 10 July 2024, one version) replicates DiLoCo on the learning-at-home hivemind library, reports training "across two continents and three countries" at "90-95% compute utilization" and scaling to "3x the size of the original work" (billion-parameter models); demonstrations are 150M and 1B models. The repository (Apache-2.0, 604 stars, 43 commits) states: **"OpenDiLoCo is no longer maintained"** and points to `prime`. |
| `PrimeIntellect-ai/prime` (GitHub page) | Yes | A CLI and SDK for Prime Intellect's *hosted* training, evaluations and GPU resources (MIT, 346 stars). It is not an open volunteer-training framework and does not mention the hivemind library. So the OpenDiLoCo line has **no maintained open successor for volunteer hardware** on these pages. |
| Google DeepMind, Decoupled DiLoCo (blog 23 April 2026; arXiv:2604.21428) | Yes (search and blog) | Divides a run into decoupled compute "islands" that exchange data asynchronously so a local failure does not stall the run; built on Pathways and DiLoCo; reports a 12-billion-parameter model trained across four separate US sites. Lineage: DiLoCo (arXiv:2311.08105, about 500x less communication than synchronous training on C4) and Streaming DiLoCo (arXiv:2501.18512, overlapped communication but still lock-step learners). HiveMind's design already names Decoupled DiLoCo; this confirms the current reference and its scale, which is datacenter sites, not home PCs. |
| AMCP whitepaper (agentmeshcommunicationprotocol.github.io/whitepaper) | Yes | Asynchronous publish/subscribe over hierarchical topics; discovery by capability announcements on system topics; "Strong Mobility" operations (`dispatch`, `clone`, `migrate`, `replicate`, `retract`, `federateWith`); transports Solace, Kafka, NATS; security listed as SASL, mTLS, JWT, ACLs, TLS; CloudEvents 1.0, an A2A bridge and an MCP connector. **The only quantitative claim is "97.3% test coverage"; scalability, latency and "battle-tested" are asserted without measurements, and the title says v1.5 while the footer says v1.6.** The evidence for AMCP in HiveMind is therefore HiveMind's own: the Kafka transport we repaired (upstream PR #1) and MESH-SCALE-1. |
| BOINC (boinc.berkeley.edu) | Yes | Run at UC Berkeley with NSF support; about 30 science projects; volunteers join through Science United or by picking projects; scientists can run Docker and Autodock jobs through BOINC Central. The home page states no software version, no GPU policy and no validation rules (the validation wiki page returned 404 on 2026-11-07), so those must be cited from the BOINC source we already build (`docs/BOINC_PRODUCTION_SERVER.md`). News: a site outage 3 to 10 September 2026; a self-hosted BOINC dashboard for Linux (23 August 2026). |
| The three Reddit threads (r/artificielle `1xctxbf`, r/SelfHostedAI `1xjc76k`, r/TechnologieFrance `1xjdy8q`) | **No.** Reddit returned HTTP 403 to every fetch from this host (www, old, JSON API). | I could not read them. Section 6 of `ROADMAP.md` already digests the earlier threads (`HM-REDDIT-01..08`), and `docs/MANIFESTE_IA_CITOYENNE_SETI_AT_HOME.md` links `1xctxbf` as the founding manifesto thread. **The "50 pioneers" claim and the SelfHostedAI comment are therefore not reflected below from their own text**; the owner should paste them, or I read them from a browser session. |

## 2. Where HiveMind stands today, honestly (inputs to any direction)

- **The chain is degrading.** From the assimilated ledger (receipt `artifacts/experiments/CHAIN_STATUS_2026-11-07.json`): median per-domain held-out change +0.0046 nats over rounds 1–98, +0.0051 over 99–170, **+0.1266 over 171 onward (99 % of domain changes worse)**. The first diagnostic round with `--mix` off and the learning rate halved (`hmq_00187`) was also worse on all six evaluated domains. One round cannot attribute the cause. Any direction that assumes the chain improves the MoE is built on sand until this is explained.
- **Serving has one GPU.** The only GPU in the GCP project is the Spot T4; the bee tier is stopped as of this evening; a permanent GPU VM does not fit the 150 USD/month target in `AGENTS.md`.
- **Scale evidence is control-plane only.** MESH-SCALE-1 passed its four pre-registered rules with 1,000 *simulated* bees on one VM, one run per size, with a thin p99 margin (`docs/benchmarks/BEE_MESH_SCALE_report.md`).
- **The VRAM guard was breached once** by probe arm A2 (3,557 MiB free against the 4,300 MiB floor); the probe queue's cap does not enforce the floor yet.

## 3. Direction A: interoperate, do not re-implement (KwaaiNet, Petals, hivemind)

**What is attractive in KwaaiNet.** It solves, in a shipped form, three things HiveMind lists as planned: node identity without a registry (Ed25519 keypairs plus W3C Verifiable Credentials), local time-decayed trust scores, and a libp2p/Kademlia swarm it describes as compatible with Petals and hivemind. It is MIT, nonprofit-run, and its README states its limits plainly (no settlement, no training, Apple Silicon serves no shards). That honesty is the same posture as `HM-REDDIT-08`.

**What it does not give us.** No training, no volunteer-credit accounting, no replay validation, no governor for shared GPUs. Those are precisely HiveMind's contributions. So the useful relationship is **HiveMind as a training and accounting layer that can expose its bees to a Kwaai/Petals-style swarm for inference**, not a merger.

Proposed items (pre-registration required before any run; ids placeholder until the owner picks a season):

| Id | Item | Acceptance gate | Owner decision needed |
|---|---|---|---|
| `DIR-A-01` | **Identity bridge:** a HiveMind producer can present a KwaaiNet-style Ed25519 identity and Verifiable Credential; the OpenRouter gateway verifies it in addition to the HMT API key. | Signature verification test vectors shared with KwaaiNet; no change to HMT accounting. | Whether to adopt VC-based identity at all (it replaces nothing; it adds a second credential). |
| `DIR-A-02` | **Swarm discovery experiment (`E22`):** advertise one HiveMind bee on a Kademlia DHT using KwaaiNet's wire conventions; measure discovery time, false peers, and whether the trust score behaves as documented. | Pre-registered thresholds; the result is reported whichever way it falls. | Which bee (must be one whose prompts may be seen by any peer: a producer sees full prompts). |
| `DIR-A-03` | **Petals liveness check:** a dated, scripted health check of the public Petals swarm and the repo's commit log, run weekly, written to a receipt. Decides whether Petals is a partner, a reference, or history. | The receipt exists and is quoted; no narrative claims about Petals without it. | None. |
| `DIR-A-04` | **Do not adopt the learning-at-home library.** We cite its DMoE paper; its DHT/averaging stack would duplicate our mesh (`HM-S1-*`). Record this as a decision so it is not re-asked. | A one-paragraph decision record in `docs/`. | Confirm. |

**Decision taken (owner, 2026-11-07): a shared Kwaai/Petals-style swarm is wanted.** `DIR-A-01` and `DIR-A-02` move from "if wanted" to planned, in that order, each still behind its pre-registration.

**Where this sits in the DiLoCo lineage.** OpenDiLoCo (July 2024) showed DiLoCo working on the hivemind library across continents at 1B scale, but its authors no longer maintain it and their successor is a hosted service. Google DeepMind's Decoupled DiLoCo (April 2026) is the current reference and runs at 12B across four datacenter sites. HiveMind occupies the space neither covers today: **decoupled, low-communication training on consumer GPUs with replay validation and non-monetary credit.** The honest statement of our position is "the method is DeepMind's; the volunteer-hardware adaptation, the validation and the accounting are ours; the open OpenDiLoCo implementation is unmaintained and we do not depend on it."

| `DIR-A-05` | **Decoupled DiLoCo alignment note:** a one-page comparison of HiveMind's outer loop with Decoupled DiLoCo's learner/synchronizer split (arXiv:2604.21428), stating which properties we inherit, which we cannot (datacenter interconnect), and which we add (replay validation, HMT). | Reviewed by the owner; cited in the paper's related-work section with the arXiv id. | None. |

**Risk to state up front:** a shared swarm means our bees serve prompts from strangers and strangers' nodes may serve our users. The producer-sees-prompts rule and the "VERIFIED is not correct" rule must appear in any joint documentation.

## 4. Direction B: the self-replication finding is a design input, not a marketing hook

The Anthropic red-team result cuts two ways for a project that *wants* small models on many cheap devices.

- **The uncomfortable overlap.** HiveMind's pitch (small experts on 8 GB consumer GPUs, outbound-only agents, nodes that join by themselves) is architecturally close to what the red team built to show self-propagation. The difference is consent and control, so consent and control must be verifiable, not asserted.
- **What is real in the finding, per the article's own caveats:** isolated steps work on sub-30B models; the full chain only worked for GLM-5.2 on an emulated network with an open debug port; the extrapolation to end-2026 is Anthropic's, and Mozilla's 0Din lead disputes the framing. We should quote the finding with those caveats, never as "AI will spread on its own".

Proposed items:

| Id | Item | Acceptance gate |
|---|---|---|
| `DIR-B-01` | **Consent and kill-switch invariants, machine-checked.** Every HiveMind node process starts only from an owner-initiated install, carries a signed enrolment, and honours a remote pause (`HM-S16-05`, PR #252 merged) and a local kill file. Add Lean statements for "no node starts without a stored owner enrolment" and "pause implies no new work within one heartbeat", in `lean4/HiveMindSpec.lean`, zero `sorry`. | Lean compiles; integration test shows a paused producer serves nothing within one heartbeat interval. |
| `DIR-B-02` | **No lateral movement by construction.** The producer agent, shield and BOINC client must have no code path that opens a connection to another volunteer's device or installs software elsewhere. Add a repository test that fails if any of those modules gains an outbound connection to a non-gateway, non-loopback host. | Test in CI; the PR security scanner already flags exfiltration patterns and should be extended with this rule. |
| `DIR-B-03` | **Egress budget per node.** Multi-gigabyte model transfers between devices are exactly what the article says operators can detect. HiveMind should cap per-node egress to the gateway and bucket only, and publish the figure, so a HiveMind node is distinguishable from a replicator by its traffic shape. | Measured egress receipt per node type; documented in the volunteer guide. |
| `DIR-B-04` | **Cyber Resilience Act readiness note** (applies from 11 Dec 2027 per the article): list which obligations touch a volunteer node and which touch the project as a distributor of software. | A dated note in `docs/`, marked "not legal advice". |

## 4b. Direction D: AMCP mesh and BOINC as one volunteer fabric

HiveMind runs two volunteer channels that do not yet talk to each other: the AMCP/Kafka bee mesh (inference and routing, `HM-S1-*`) and the BOINC stack (training work units, replay validation, credit). The AMCP whitepaper's "Strong Mobility" operations and its MCP connector suggest a bridge, but the whitepaper offers no measurements, so the bridge must be measured by us.

| Id | Item | Acceptance gate | Owner decision needed |
|---|---|---|---|
| `DIR-D-01` | **BOINC work-unit events on the mesh:** publish `assimilated` and `validated` events from the BOINC validator onto AMCP topics so hosts can route to bees serving the newest validated adapter. Read-only on the ledger; the ledger stays the source of truth. | A paired test shows the adapter pointer a host sees equals the ledger's pointer; no event changes credit. | None. |
| `DIR-D-02` | **Volunteer registry unification:** one identity for a volunteer who both trains (BOINC) and serves (mesh), reusing `DIR-A-01`'s credential; BOINC credit and HMT remain separate ledgers until the credit mismatch recorded in `project_boinc_core_node` is resolved. | The two ledgers reconcile per volunteer in a receipt. | The BOINC-credit versus `verified_credit` mismatch (order of magnitude) is still unresolved. |
| `DIR-D-03` | **AMCP mobility, measured:** before using `migrate()`/`replicate()` for bees, a pre-registered experiment (`E23`) on the repaired Kafka transport measures migration time, state loss and duplicate delivery at 100 and 1,000 simulated bees. | Thresholds registered first; result reported either way. | None. |
| `DIR-D-04` | **BOINC-side honesty:** the BOINC site states no validation rules on its front page; our volunteer documentation must cite the validator we actually run (quorum, replay tolerance, class matching) from our own docs, not from BOINC's. | Every validation claim in `docs/BOINC_*.md` links a file in this repo. | None. |

## 5. Direction C: what the community asked for, and what we can honestly answer

`ROADMAP.md` section 6 already maps eight community objections to engineering answers with live proofs. Two of your three new threads could not be read (section 1), so I list only what the earlier synthesis and the manifesto establish, and what remains open:

- **Plain-French explanation and zero-install trial** (`HM-REDDIT-01`): done (`/join`, `/rejoindre`, WebGPU page).
- **"Are the numbers real?"** (`HM-REDDIT-02`, `-08`): the honest answer today includes the degradation in section 2 and the one-GPU serving reality. **Proposal `DIR-C-01`:** publish the chain-status receipt on the portal's `/benchmarks` page with the same wording as the paper, and never show a "trained rounds" counter without the held-out trend beside it.
- **"Opening a port is a dealbreaker"** (`HM-REDDIT-03`): outbound-only is done; `DIR-B-02` makes it a tested invariant rather than a design note.
- **"Will it hurt my games or my card?"** (`HM-REDDIT-04`): the 6,656 MiB WDDM ceiling and the 1.5 s preemption exist; the T4's own 4,300 MiB floor was breached once by a probe (section 2), so **`DIR-C-02`:** the probe queue and every trainer must enforce the floor in code, with a test, before any further night run.
- **"La Ruche passe les 50 pionniers"** (`1xjdy8q`): not read. If the thread claims 50 pioneers, the roadmap should state what a pioneer is (an account, a producer, a trainer?) and link the receipt that counts them. **`DIR-C-03`:** a `pioneers` receipt generated from the accounts store, with kinds (`user`/`test`/`probe`, PR #253) excluded as appropriate.

## 6. Order and effort (estimates, unvalidated)

1. `DIR-C-02` (floor enforcement) and `DIR-B-02` (no-lateral-movement test): days; safety first.
2. `DIR-A-03` (Petals liveness receipt) and `DIR-C-01` (chain status on the portal): days; honesty of the public face.
3. `DIR-B-01` (Lean invariants): one to two weeks; needs the Lean gate.
4. `DIR-A-02` (`E22` swarm discovery experiment): two to three weeks after pre-registration; needs the owner's choice of bee.
5. `DIR-A-01`, `DIR-B-03`, `DIR-B-04`, `DIR-C-03`: after the above.

## 7. Decisions for the owner

1. Paste or relay the three Reddit threads, or allow a browser-session read, so section 5 can reflect their actual text.
2. ~~Confirm `DIR-A-04`~~ **Confirmed 2026-11-07:** cite, do not adopt, the learning-at-home library.
3. Choose the season number for `DIR-*` items (the roadmap is at Season 32 plus Mesh Season 1).
4. ~~Decide whether a shared swarm is wanted~~ **Decided 2026-11-07: wanted.** The producer-sees-prompts rule stays in every joint document.
5. Approve `DIR-C-02` as a blocker on further night training (it is a safety rule the project already has in `AGENTS.md`).
6. Resolve the BOINC-credit versus `verified_credit` mismatch before `DIR-D-02`.
7. Choose whether the paper's related-work section cites Decoupled DiLoCo (arXiv:2604.21428) and OpenDiLoCo (arXiv:2407.07852, unmaintained) in v4.3.
