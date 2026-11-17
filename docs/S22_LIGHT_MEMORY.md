# S22 — Light-Memory: retrieval for the Bees with a low-storage vector index (proposal)

Status: **proposal, 2026-10-28.** Nothing in this document is implemented. No code, index, model or dependency was added to the repository, and nothing was measured on HiveMind hardware.
The roadmap entry is [`ROADMAP.md` § 3.19](../ROADMAP.md). Roles follow the operating model: the scientific lead owns the predictions and acceptance criteria below (the thresholds here are
the proposer's starting points, not pre-registered), the engineering coordinator splits the tasks, and the owner decides the open questions in section 7.

## 1. Why this is on the roadmap

HiveMind has no retrieval layer. Its Bees answer from model weights and the prompt. A Bee that could look things up in a domain corpus (astrophysics papers, Mathlib, Rust crate docs, the project's own
documentation) is an obvious next step, but a normal vector index is expensive exactly where HiveMind lives: producers on consumer cards and small disks, volunteers who already download several gigabytes,
and a Spot VM. LEANN, a paper accepted at MLSys 2026, claims to cut that storage cost by one to two orders of magnitude. That claim is worth testing on our hardware before anyone builds on it.

## 2. Sources, and what was and was not read

| Source | Read? | Notes |
|---|---|---|
| Paper, [arXiv:2506.08276](https://arxiv.org/abs/2506.08276v2) (v1 9 Jun 2025, v2 25 Nov 2025), HTML body | yes, most of it | Table 4 (Mac) and parts of the appendices were only available as a summary and are marked below |
| MLSys 2026 proceedings page ([link](https://proceedings.mlsys.org/paper_files/paper/2026/hash/e27ea0cd50b798ff8942caf9203f0992-Abstract-Conference.html)) | abstract only | the PDF was not readable by our tooling. The page shows **no award**; a search result mentions a "Best Research Paper" certificate on the conference site, which we have **not** verified and do not claim |
| Code README (MIT licence) | yes | its own storage figures are not in the paper (see 4) |
| The MarktechPost article of 2026-10-27 | **no** | the site refused our fetch (HTTP 403). Nothing here comes from it |

Titles differ: arXiv "LEANN: A Low-Storage Vector Index", MLSys "LEANN: A Low-Storage Overhead Vector Index". The arXiv v2 HTML carries an "Under Review" tag. Authors (13): Yichuan Wang, Zhifei Li, Shu Liu,
Yongji Wu, Ziming Mao, Yilong Zhao, Xiao Yan, Zhiying Xu, Yang Zhou, Ion Stoica, Sewon Min, Matei Zaharia, Joseph E. Gonzalez (UC Berkeley, CUHK, AWS, UC Davis).
**Which repository is canonical is unclear:** the paper footnote points at `github.com/yichuan-w/LEANN`, the README's clone command at `github.com/StarTrail-org/LEANN`. Verify before depending on either.

## 3. How LEANN works (as the paper describes it)

* **Build:** embed the chunks (the paper uses Contriever, 768 dimensions), build a proximity graph (HNSW-based) and **prune it keeping high-degree nodes** (the top beta% by degree keep up to `M` neighbours, the
  rest get `M/5`), build a product-quantisation (PQ) table, then **throw the dense embeddings away**. What is stored is the pruned graph plus the PQ-compressed table.
* **Query:** best-first traversal of the graph. Approximate distances come from the PQ codes; only the most promising candidates (the top alpha% of an approximate queue) get their **exact embedding recomputed
  with the embedding model at query time**. Candidates are **batched dynamically** (for example 64) to keep the GPU busy, at the price of a slightly stale exploration order.
* **Updates:** inserts are buffered and merged at query time; deletes mark nodes inactive; the rebuild policy is "left for future work".

## 4. What the paper reports

| Claim | Figure | Where, and what it is measured against |
|---|---|---|
| Index size | "up to 50x" smaller than conventional indexes; about 5% of the raw data | abstract and section 1 |
| 76 GB text corpus (60M passages, Qwen3-4B, RTX 4090) | storage BM25 59 GB / HNSW 188 GB / PQ 20 GB / **LEANN 4 GB** | Table 1 |
| Downstream accuracy at that point | BM25 18.3% / HNSW 25.5% / PQ 17.9% / **LEANN 25.5%** | Table 1 (matches HNSW when 90% recall is enforced; BM25 and PQ cannot reach it) |
| Standalone search time | HNSW 0.05 s, PQ 4.53 s, **LEANN 2.48 s** | Table 1 (NQ) |
| End-to-end latency (retrieval plus generation) | "about 10% end-to-end latency overhead" | section 1; Table 2 per dataset, RTX 4090, 90% recall: NQ 10.6%, GPQA 1.6%, TriviaQA 14.7%, HotpotQA 23.4% |
| Recall | "over 90% top-3 recall within one second while using less than 5% of the raw data storage" | section 1 (this "within one second" does not match Table 1's 2.48 s for NQ; the datasets and settings differ) |
| Graph pruning | average degree 18 to 9 (half the edges); random pruning needs up to 1.8x, a smaller `M` up to 5.8x, more recomputations | section 7.3 |
| Two-level search, then dynamic batching | 1.4x average speed-up (up to 1.6x), then 1.8x (up to 2.0x) | section 7.3 |
| Versus recomputing everything (IVF-Recompute) | up to about 200x faster | section 7.2 |
| Personal-style datasets | storage savings versus HNSW 97%, 98%, 97% (FinanceBench, Enron, LAION); retrieval 1.5 to 1.9 s | Table 3 |

Hardware and data: an NVIDIA RTX 4090 workstation (32 GB RAM, 1 TB disk, WSL2) and an AWS M1 Ultra instance; datasets RPJ-Wiki (76 GB), NQ, TriviaQA, GPQA, HotpotQA, FinanceBench, Enron, LAION.
The README has separate numbers that are **not** in the paper (for example "97% less storage", Wikipedia 60M: 201 GB with FAISS versus 6 GB); treat them as different claims.

## 5. What the sources do not say, and where they disagree with themselves

* No **build/indexing time**, no **insert latency**, no **throughput**, and **no CPU-only or small-GPU numbers**. Everything was measured on a 24 GB RTX 4090 or a large server.
* **Recomputation is the cost.** About 76% of the latency is the embedding model running at query time (appendix summary). On a producer that is also serving a chat model, or a gamer's PC, that is the same GPU.
* Standalone search is **slower than HNSW or DiskANN** (2.48 s versus 0.05 s on NQ); the end-to-end overhead is small only because answer generation takes about 20 s.
* The text says overhead is "consistently under 20%", while the paper's own Table 2 shows 23.4% for HotpotQA; the text says GPQA "under 3%" and the table shows 1.6%. Small, but we cite the table.
* Gains are smaller on GPQA and HotpotQA (out-of-distribution datastore, multi-hop reasoning); the evaluation is single-hop retrieval.
* The 5% claim is 4 GB of 76 GB (about 5.3%) for a very large corpus. Fixed overheads could matter much more on a small corpus such as our documentation. That is a thing to measure.

## 6. Hypotheses for HiveMind (to be tested, not claims)

1. **Storage-limited producers and gamers** could hold a domain index that would otherwise not fit.
2. **Bee knowledge packs:** a content-hashed, signed index per Bee domain, shipped through the same standard download channel as the BOINC application archives (the portable app is about 4.4 GB per volunteer today),
   so a Bee can retrieve from a corpus the producer never stored in full precision.
3. **Developer and coding-agent memory:** a local semantic index over the repository, docs and issues for contributors who connect through GitHub and their editor (the library ships an MCP server). Text that comes from
   the internet stays untrusted data, as on the AMCP bus.
4. **Privacy:** private documents of a producer or consumer are indexed on their own machine and never leave it.

The counter-hypothesis is real: query-time recomputation competes with inference and with a game for the same GPU, and on small cards it may cost more than the disk it saves. Task `HM-S22-02` exists to find out.

## 7. Tasks

| Task | Problem | Deliverable | Proposed prediction (refutable) | Gate |
|---|---|---|---|---|
| `HM-S22-01` measure on our hardware | every figure above comes from a 4090 or a server | pre-registered benchmark: LEANN versus exact flat search and HNSW on (a) the repository docs, (b) one Bee-domain corpus, on T4 (idle), RTX 2070 and CPU only; report size, build time, recall@3, latency p50/p95, recompute share, GPU memory | recall@3 against exact search at least 0.90 with an index at most 10% of the raw text size; **refuted** otherwise, and reported as refuted | scientific lead pre-registers, then measure |
| `HM-S22-02` contention rule | recompute shares a GPU with inference and games | a retrieval mode that runs the embedding model on CPU or on the GPU only when the governor allows it; measured effect on chat latency and on the idle-yield time | p95 chat latency rises by at most 10% with retrieval active; **refuted** otherwise (the 10% is a placeholder for the scientific lead to set) | no regression of the governor's yield time |
| `HM-S22-03` knowledge packs | no way to ship or verify a retrieval corpus | a pack format: index files, manifest with source revisions and licences (reuse the S21 dataset licence gate), hash and signature; download through the standard channel | a pack verifies end to end on a fresh machine; a tampered pack is rejected | licence gate passes; hash and signature verified |
| `HM-S22-04` does retrieval help a Bee | the value of retrieval for our Bees is unknown | pre-registered A/B on a Bee domain with graded or executable checks: no retrieval versus LEANN versus exact retrieval, at least 3 seeds | lift over no retrieval whose interval excludes zero at equal latency budget; **refuted** otherwise, in which case this season stops after `-01` | adopt only if the decision rule holds |
| `HM-S22-05` developer memory | contributors and coding agents cannot search the project semantically | a local, reproducible index of docs and issues with an editor/MCP entry point; secret scan before indexing; indexed text marked untrusted | a fresh clone builds the index; no secret-pattern match in it | secret scan clean |
| `HM-S22-06` privacy and threat model | an index is derived from the content and may leak it | written threat model; a test that shipped packs contain no user data; local-only default for private corpora | none (a document and a test) | owner review |
| `HM-S22-07` dependency decision | two repositories claim to be upstream; the paper is marked "under review" | pin an exact commit and licence (MIT is stated), optional extra only, decide fork or vendor | none | owner decision |

## 8. Not goals

No retrieval is added to the live gateway by this proposal. No claim is made that retrieval improves answers until `HM-S22-04` says so. Nothing here changes pricing, points or the non-monetary charter.

## 9. Decisions for the owner

1. Is retrieval for the Bees wanted at all before the current season's work lands? (`HM-S22-01` is cheap and decides the rest.)
2. Which repository is the upstream to track, and may it be an optional dependency?
3. Which Bee domain is the first corpus (the dataset registry's licence gate must pass)?
4. May query-time recompute ever use a gamer's or a producer's GPU, or only CPU and idle windows?
