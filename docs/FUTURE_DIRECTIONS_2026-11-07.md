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
| r/TechnologieFrance `1xjdy8q` ("La Ruche passe les 50 pionniers") and r/SelfHostedAI `1xjc76k` ("Turned my idle 8 GB GPU into a P2P AI node"), **pasted by the owner on 2026-11-07** (the server gets HTTP 403 from Reddit) | Yes, from the pasted text | r/TechnologieFrance: 18 upvotes, 30K views, "55 active accounts today between GPU producers and testers". r/SelfHostedAI: 31 upvotes, 33K views. Both threads carry the project's own claims and the replies quoted in section 5. The r/artificielle manifesto thread `1xctxbf` was still not provided. |

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

## 5. Direction C: what the community actually asked, from the two threads the owner pasted

The replies fall into five asks. Each is paired with what the project answered in-thread, and with what the repository can prove today. Where the in-thread answer went beyond the evidence, that is said.

**C-1. Privacy of prompts on strangers' machines** (u/StevenVincentOne: "What can a node operator actually see, and what stops a Bee from logging requests or returning junk to farm points?"; u/anonymousart3: "this is processing people's private data"; u/Alyx_695: "les requêtes arrivent chez un random"; u/lcpjj_: "Anything you send through `/v1/chat/completions` goes to whichever volunteer node picks it up. Nothing checks who runs that node or what they keep."). The in-thread answer described split-layer execution where "remote Bees never see raw text" and a prompt scrubber. **What is proven:** the producer sees the full prompt today (`docs/HIVEMIND_OPENROUTER.md`); the scrubber strips accidental secrets, it does not hide the prompt. **Proposal `DIR-C-01`:** the public pages and the paper state the producer-sees-prompts rule in those words until a split-layer path is measured end to end under a pre-registered experiment; the Trainer role (no user prompts, public datasets only) is the honest answer for people who will not accept that, and it exists (`role=trainer`).

**C-2. Point farming and free-riders** (u/C-c-c-ccomboBreaker: "Il y a un système de crédits ou de limite comme pour le p2p? Sinon y'aura des profiteurs"; u/StevenVincentOne on junk answers). The owner answered in-thread: "pas encore de système de limite. Bon idée à mettre en place." The later v3.1.4 note claims canary prompts with a Wilson lower bound, Multi-Krum, a step-time floor and a 10% holdout replay before credit. **What is proven:** Kelly proportional-share clearing and the per-IP starter-grant cap are merged (PR #253); Wilson scoring exists in `queenbee/dance.py`; blinded canary probing of inference nodes is **not** in the repository I can see. **Proposal `DIR-C-02`:** implement and pre-register the canary probe before the claim is repeated; publish the farming rate it catches.

**C-3. One company in the middle** (u/StevenVincentOne: "a shared pool with a single front door still has one company in the middle"; u/Alyx_695 on centralisation). The in-thread answer: Cloud Run is "bootstrap training wheels", a Kademlia DHT and QUIC/STUN exist, and a coordinator can be self-hosted. **What is proven:** `src/hivemind/swarm/dht.py` and the fabric exist and are described in the paper (line 283); **no measurement of nodes finding each other without the gateway exists.** This is exactly `DIR-A-02` (`E22`), now with a community question behind it. The owner's decision that a shared Kwaai/Petals-style swarm is wanted makes this the first experiment to run.

**C-4. Honesty about models and benchmarks** (u/lcpjj_: the "trained HiveMind MoE" entries "point to a Hugging Face account that doesn't exist" and "the models the gateway actually serves are standard off-the-shelf ones"; u/ValuableDapper9415: "vous avez quelques stats de benchmark à partager?"; u/LyreEtLettres: "Explique moi vraiment l'intérêt"). The owner conceded in-thread ("You were right on all counts") and the catalog on `main` no longer references those repos. **The paper must match this concession.** Its header still advertises a ">200-Node Launch Forecast" and the text claims "47 WAN nodes and 35 models"; the chain receipt shows the only HiveMind-trained artefacts are the BOINC QLoRA adapters, which are currently getting worse (section 2). **Measured on 2026-11-07** (receipt `artifacts/experiments/CATALOG_SERVED_2026-11-07.json`, `scripts/catalog_served_receipt.py`, source text and runtime after import): 16 serving tiers over 13 Ollama tags, one HiveMind-built tag (`hivemind-bee-stars`); 49 public ids. The source text still carries 15 `hf_repo` literals under the unpublished account and 35 `LIVE` statuses on off-the-shelf tiers, **but `_truthful()` rewrites at import**: the gateway exposes 26 of 49 ids as "Planned, not trained yet" with no repository and no training figure, and no unpublished repository remains. What is still wrong at runtime: **13 ids carry a `LIVE` status while served by an off-the-shelf tag, 8 of them with a trained-MoE status (`LIVE_NOCTURNAL_MOE`), 13 with a `tokens_trained_b` figure.** So the in-thread statement was closer to true than my first static count suggested; the first version of this paragraph overstated it and is corrected here. **`DIR-C-03` (done in this PR):** the paper's v4.3 draft carries the runtime-measured paragraph, locked by `PROV-88`. **`DIR-C-03b` (owner decided 2026-11-07: relabel "planned"):** extend `_truthful()` so a trained-MoE status on an off-the-shelf tier is also relabelled; branch `fix/catalog-planned-labels`.

**C-5. Onboarding and the README** (u/ValuableDapper9415: "c'est très bordélique", asks for a TL;DR; u/Cold-Ad5815: "Comment on participe ?"; u/Rgoplay_: AMD integrated GPUs on Linux, untested; u/Alyx_695: "abliterated" models). The owner promised a 10-line TL;DR and a separate session history. **Proposal `DIR-C-04`:** keep that promise in the README, record AMD iGPU and "abliterated" models as open questions with their constraint (the catalog serves what volunteers pull; a model policy is an owner decision), and link the volunteer preflight tool (PR #255) from the join page. **`DIR-C-05`:** the thread claims "55 active accounts today between GPU producers and testers"; publish an accounts receipt generated from the accounts store, by kind (`user`/`test`/`probe`, PR #253) and by role (producer, trainer, consumer), so the next such number links to a file.

**Security disclosure, for the record.** A community reviewer reported authentication problems privately through GitHub's vulnerability reporting (code read at `main` `7fe8f66`) and publicly warned that some setup guides exposed Ollama without the shield (CVE-2024-37032 context). The project's public fix list (2 November 2026, and release v3.1.4) is to be verified against the private advisory before the paper states any of it as fixed.

## 5c. Direction E: the economics of millions of idle consumer GPUs, with sourced parameters

Model: `scripts/consumer_gpu_economics.py`, receipt `artifacts/experiments/CONSUMER_GPU_ECONOMICS_2026-11-07.json`. It is arithmetic on parameters that each carry a source and date; it is not a measurement and not a forecast. Cloud prices are indicative because the trackers disagree.

**Supply.** Discrete GPUs ship at about 11.8 M boards a quarter (Jon Peddie Research, Q1 2026); the Steam survey's 8, 12 and 16 GB tiers together are about 67% of surveyed GPUs (July–September 2026), and Steam's record of 43,505,724 logged-in accounts (11 October 2026) times that share is about 29 M machines in the project's VRAM class. That is the sense in which "millions of consumer GPUs" is right. The project has 55 claimed accounts. The gap is recruitment and trust, not silicon. At the project's own 18 h/day idle assumption, 100,000 volunteers are about 1,500 H100-equivalents at a 1:50 throughput ratio (750 to 3,000 across 1:100 to 1:25), before validation overhead and churn.

**Cost to the volunteer.** At the EU average household price of 0.28 EUR/kWh (Eurostat, H1 2026), an RTX 2070 at its 175 W board power costs about 0.049 EUR per GPU-hour, 0.39 EUR per 8-hour night, and about 0.19 EUR of electricity per million generated tokens at the project's own 72 tok/s figure (0.50 EUR a night in Germany, 0.15 in Hungary). Depreciation, the volunteer's time and the project's validation cost are not in that number. Under the non-monetary reciprocity charter the project never pays it, so the model's conclusion is a design rule: **a night of donated GPU must be worth about 0.4 EUR to the volunteer in API access, locality or purpose, or they stop.**

**Against cloud.** A T4 lists at about 0.35 USD/h on GCP and an H100 at roughly 1.8–3.81 USD/h on demand (2026 trackers, which disagree). Per hour the home GPU's electricity-only cost is far below both; per token, an H100 wins on throughput. So the honest pitch is sovereignty, locality and reciprocity, not price. Items: `HM-S33-19` publish this receipt on the portal and remove price-based claims from the join pages; `HM-S33-20` measure the real idle fraction and draw of enrolled producers (replacing the 18 h/day and 175 W parameters with telemetry).

## 5d. Direction F: if the AI capex cycle turns

Hyperscaler 2026 capex guidance was reported in mid-October at about 732 USD billion; H100 rents are well below their 2024 peak and partly recovered in 2026; one analyst puts a severe credit unwind at 20%, which is an opinion. **The project's financial exposure is close to nil**: no GPUs owned (0 USD capex), no debt, cloud capped at 150 USD a month. Its real exposures are volunteer attention, the supply of open base weights, and dependence on Ollama and Hugging Face as distribution. Three scenarios, each with an action:

- **S1, cloud price collapse.** For inference, cheap cloud narrows the gap; electricity-only cost at home (~0.05 EUR/h for a 2070) still undercuts any rented H100 per hour, but per token the H100 wins on throughput. For training small adapters, the home GPU stays viable only where the data must stay local (BYOM) or where purpose, not price, drives participation. Action: Keep the pitch on sovereignty, locality and reciprocity, not on cost; stop citing cloud prices as the advantage.
- **S2, capex retrenchment.** Demand for non-commercial compute rises; second-hand datacenter GPUs (A100, H100) reach the secondary market at low prices, and a 'consumer' swarm becomes a prosumer swarm. The binding risk is model supply: a QLoRA swarm needs open base weights. Action: Pin and mirror the base weights the chain depends on (Granite, Qwen, Gemma) under their licences; support 24-40 GB prosumer cards in the trainer; keep the catalog honest so the project is credible when attention arrives. (`HM-S33-21` pin and mirror base weights under their licences; `HM-S33-22` 24–40 GB prosumer cards in the trainer.)
- **S3, no burst.** Status quo: volunteer compute competes on locality and purpose; the project's bottleneck remains the chain's quality (currently regressing) and recruitment, not the macro cycle. Action: Fix the chain (HM-S33-06) before scaling recruitment.

## 5e. Direction G: chunk-level routing with H-Net dynamic chunking (candidate, not adopted)

The owner forwarded an analysis of H-Net (Hwang, Wang, Gu, "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling", arXiv:2507.07955, v2 15 July 2025) arguing that its dynamic chunking could cut the per-token all-to-all traffic of a distributed MoE. The paper was read on 2026-11-07 (HTML v2, first 100,000 characters) to separate what it claims from what the analysis infers.

| Claim | In the paper? | Where |
|---|---|---|
| Learned, differentiable chunk boundaries from cosine similarity of adjacent vectors, with a straight-through estimator and a smoothing module | Yes | Section 2.2 |
| Ratio loss "inspired by load balancing mechanisms in Mixture-of-Experts", $L_{ratio}=\frac{N}{N-1}((N-1)FG+(1-F)(1-G))$, $\alpha=0.03$ | Yes | Section 2.2.3, eq. 10 |
| Target ratios: one stage $N=6$; two stages $N^0=N^1=3$; measured two-stage compression about 7 bytes per chunk | Yes | Section 3, Table 1 |
| Byte-level one-stage H-Net outperforms a compute- and data-matched BPE Transformer; two stages match a token Transformer of twice the size | Yes (680 M–1.6 B models, 100 B-token FineWeb-Edu subset) | Section 3, Table 1 |
| DNA: same perplexity with 3.6× less data (abstract says "nearly 4×") | Yes | Section 3.2, Figure 6 |
| Appendix on distilling a pretrained token-level model into a byte-level H-Net | Yes, Appendix F exists; its results were not read (page truncated) | Appendix F |
| "4.5 to 6× per stage", "5 to 9× fewer inter-node messages" | **No**; the analyst's extrapolation from $N$ | — |
| Acts as a Mixture-of-Depths; applies to distributed or decentralised training | **No**; neither term appears in the portion read | — |

**Why it is interesting for HiveMind anyway.** The paper does not address networks, but the mechanism is exactly a learned decision of *when* a higher stage needs to be consulted. In a mesh where the heavy expert lives on another machine, that decision is a message-rate control: the encoder, boundary router and decoder (hundreds of millions of parameters, linear memory, no large KV cache) could run on the 8 GB node, and only chunk vectors cross the WAN. That is a hypothesis, with three things that could break it: the boundary router's compression on our domains (code, Lean, French) may be far from 6; the smoothing and de-chunking step needs the expert's output back before the next byte, so WAN latency is paid per chunk, not removed; and the chain's experts are QLoRA adapters on a token-level base, which H-Net would replace, not wrap, unless Appendix F's distillation works on our bases.

| Id | Item | Acceptance gate |
|---|---|---|
| `HM-S33-23` | **Measure, do not adopt (`E25`):** run the authors' released 1-stage H-Net on our six BOINC domains and record chunks per byte, encoder+decoder VRAM, and bytes-on-the-wire per 1,000 generated bytes if chunk vectors were the only traffic; compare with the token-level all-to-all count. | Pre-registered thresholds; a ratio below 3 on code or Lean refutes the WAN argument for those domains. |
| `HM-S33-24` | **Read Appendix F and decide:** whether distilling an existing base (Granite-1B or Qwen) into an H-Net wrapper is feasible on a T4; if the appendix reports a quality loss above our ratchet tolerance, stop here. | A dated reading note in `docs/`; no code. |
| `HM-S33-25` | **Chunk-level routing only after `HM-S33-06`:** no architecture change to the chain until the current regression is explained; H-Net would add a confound. | Ordering rule, no gate. |

## 5b. Season assignment (owner decision, 2026-11-07: Mesh Season 2 and Season 33)

Mesh-line items take `HM-S2-*` (the mesh line numbers its own seasons from 1, `ROADMAP.md` 3.21). Everything else takes `HM-S33-*`, continuing after the `HM-S33-01..05` ids already reserved in `docs/VISION_AND_NEXTGEN_SWARM_ROADMAP.md` (`HM-S33-01` is "Decoupled DiLoCo sans retardataires" and stays as it is).

| Roadmap id | Item | Pre-registration |
|---|---|---|
| `HM-S2-01` | `DIR-A-01` identity bridge (Ed25519 + Verifiable Credential, beside the HMT key) | none (no measurement) |
| `HM-S2-02` | `DIR-A-02` swarm discovery without the gateway (`E22`) | yes, thresholds first |
| `HM-S2-03` | `DIR-D-01` BOINC work-unit events on the AMCP mesh | paired test |
| `HM-S2-04` | `DIR-D-02` one volunteer identity across BOINC and mesh | blocked on the credit mismatch |
| `HM-S2-05` | `DIR-D-03` AMCP mobility measured (`E23`) | yes |
| `HM-S2-06` | `DIR-D-04` BOINC-side validation claims cite our own files | none |
| `HM-S33-06` | **Decoupled DiLoCo outer loop for the QLoRA-MoE chain, measured (`E24`):** the chain's learner/synchroniser split with the ratchet as acceptance rule; separate three candidate causes of the post-`hmq_00171` regression (acceptance rule, single-domain work units, stratified mixing) with held-out per-domain losses as the only criterion. Cited from the paper's chain paragraph. | yes, before any further chain run |
| `HM-S33-07` | `DIR-A-03` Petals liveness receipt, weekly | none |
| `HM-S33-08` | `DIR-A-04` decision record: cite, do not adopt, learning-at-home | none |
| `HM-S33-09` | `DIR-A-05` Decoupled DiLoCo alignment note (related work) | none |
| `HM-S33-10` | `DIR-B-01` consent and pause invariants in Lean | Lean gate |
| `HM-S33-11` | `DIR-B-02` no-lateral-movement test in CI | none |
| `HM-S33-12` | `DIR-B-03` per-node egress budget and receipt | measured |
| `HM-S33-13` | `DIR-B-04` Cyber Resilience Act readiness note | none |
| `HM-S33-14` | `DIR-C-01` producer-sees-prompts rule on every public page until split-layer is measured | none |
| `HM-S33-15` | `DIR-C-02` floor enforcement in the probe queue and trainers; blinded canary probe implemented before it is claimed | test + pre-registration |
| `HM-S33-16` | `DIR-C-03b` catalog relabel or removal in code | owner decision |
| `HM-S33-17` | `DIR-C-04` README TL;DR, AMD iGPU and model-policy open questions | none |
| `HM-S33-18` | `DIR-C-05` accounts receipt by kind and role | none |

## 6. Order and effort (estimates, unvalidated)

1. `DIR-C-02` (floor enforcement) and `DIR-B-02` (no-lateral-movement test): days; safety first.
2. `DIR-A-03` (Petals liveness receipt) and `DIR-C-01` (chain status on the portal): days; honesty of the public face.
3. `DIR-B-01` (Lean invariants): one to two weeks; needs the Lean gate.
4. `DIR-A-02` (`E22` swarm discovery experiment): two to three weeks after pre-registration; needs the owner's choice of bee.
5. `DIR-A-01`, `DIR-B-03`, `DIR-B-04`, `DIR-C-03`: after the above.

## 7. Decisions for the owner

1. Paste or relay the three Reddit threads, or allow a browser-session read, so section 5 can reflect their actual text.
2. ~~Confirm `DIR-A-04`~~ **Confirmed 2026-11-07:** cite, do not adopt, the learning-at-home library.
3. ~~Choose the season number~~ **Decided 2026-11-07: Mesh Season 2 and Season 33** (section 5b).
4. ~~Decide whether a shared swarm is wanted~~ **Decided 2026-11-07: wanted.** The producer-sees-prompts rule stays in every joint document.
5. Approve `DIR-C-02` as a blocker on further night training (it is a safety rule the project already has in `AGENTS.md`).
6. Resolve the BOINC-credit versus `verified_credit` mismatch before `DIR-D-02`.
7. ~~Choose whether the paper cites Decoupled DiLoCo and OpenDiLoCo~~ **Decided 2026-11-07: yes**, done in the v4.3 draft (Decoupled DiLoCo tied to the QLoRA-MoE chain paragraph; OpenDiLoCo kept with its maintenance caveat; Soma et al. 2025, arXiv:2410.17517, cited as a conceptual reference at QueenBee).
8. Decide `HM-S33-16`: which of the catalog entries to relabel or remove in code.
