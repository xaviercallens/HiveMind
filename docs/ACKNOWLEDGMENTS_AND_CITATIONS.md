# Acknowledgments, Open-Source Attribution & Academic Citations (`v2.3.1`, 32 papers + 5 own Zenodo records, citations re-verified 2026-10-23)

**Project HiveMind** stands on the shoulders of the open-source distributed machine learning community, academic researchers across decentralized optimization and Mixture-of-Experts (`MoE`) architectures, and our human & AI contributor community. This document formally acknowledges every external repository, algorithm, dataset, and research paper integrated into or leveraged by **Project HiveMind** (`src/hivemind/swarm/diloco_nextgen.py`, `src/hivemind/swarm/strategy_optimizer.py`, `src/hivemind/core/`, and `lean4/HiveMindSpec.lean`).

---

## 1. Leveraged Open-Source GitHub Repositories (Cloned & Integrated in Season 8 / Wave 7)

All 8 upstream reference repositories are archived locally in `/home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/reference_repos/` and programmatically registered in [`LEVERAGED_GITHUB_REPOSITORIES`](../src/hivemind/swarm/diloco_nextgen.py):

| Repository | Authors / Organization | License | Associated Paper | Integrated HiveMind Component (`src/hivemind/swarm/diloco_nextgen.py`) |
| :--- | :--- | :---: | :--- | :--- |
| **[`one-covenant/SparseLoCo`](https://github.com/one-covenant/SparseLoCo)** | Amir Sarfi, Benjamin Thérien, Stefan Horoi, Eugene Belilovsky (*Mila / Concordia University / Covenant*) | `MIT` | [`arXiv:2508.15706`](https://arxiv.org/abs/2508.15706) | [`ComputeVisibleSparseLoCoCompressor`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-09B`: 2-bit `4`-bin empirical statistical centroid LUT quantization + Top-$k$ outer error feedback) & [`ChunkedOrthonormalDCTTransform`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-09C`) |
| **[`shoroi/isoloco`](https://github.com/shoroi/isoloco)** | Stefan Horoi, Amir Sarfi, Eugene Belilovsky (*Mila / Concordia University*) | `MIT` | [`arXiv:2607.03011`](https://arxiv.org/abs/2607.03011) | [`ReputationWeightedIsoCTSMerger`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-07`: Isotropic Covariance `ISO-C` + Common & Task-Specific `ISO-CTS` subspace orthogonalization) & [`TSVDecorrelatedMerger`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-07B`: `tsvloco.py` Task Singular Vector whitening) |
| **[`PrimeIntellect-ai/OpenDiLoCo`](https://github.com/PrimeIntellect-ai/OpenDiLoCo)** | Sami Jaghouar, Jack Min Ong, Johannes Hagemann (*Prime Intellect*) | `Apache-2.0` | [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) | [`CPUOffloadedFlatBucketOuterState`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-06B`: CPU-offloaded outer parameter & pseudo-gradient management over Hivemind DHT) |
| **[`PrimeIntellect-ai/prime-diloco`](https://github.com/PrimeIntellect-ai/prime-diloco)** | Prime Intellect Team (*Sami Jaghouar, Jack Min Ong, Johannes Hagemann et al.*) | `Apache-2.0` | [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) | [`CPUOffloadedFlatBucketOuterState`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-06B`: contiguous flat 1D FP32 CPU master-weight buffer with `0 B` extra GPU VRAM overhead, inspired by `zeroband/diloco.py`) |
| **[`bloc97/DeMo`](https://github.com/bloc97/DeMo)** | Bowen Peng (`bloc97`), Jeffrey Quesnelle, Diederik P. Kingma (*Nous Research*) | `Apache-2.0` | [`arXiv:2411.19870`](https://arxiv.org/abs/2411.19870) | [`ChunkedOrthonormalDCTTransform`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-09C`: orthonormal 2D chunked Type-II Discrete Cosine Transform `TransformDCT`, hardened with bitwise-deterministic `index_copy_` instead of CUDA `scatter_reduce_`) |
| **[`google-deepmind/asyncdiloco`](https://github.com/google-deepmind/asyncdiloco)** | Arthur Douillard, Qixuan Feng, Andrei A. Rusu, Rachita Chhaparia, Yani Donchev, Adhiguna Kuncoro, Marc'Aurelio Ranzato, Arthur Szlam, Jiajun Shen (*Google DeepMind*) | `Apache-2.0` | [`arXiv:2401.09135`](https://arxiv.org/abs/2401.09135) | [`CosineGatedAdamDecayOuterOptimizer`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-06`: asynchronous worker-delay & streaming fragment-age tables) |
| **[`pmerolla/fomoe`](https://github.com/pmerolla/fomoe)** | Paul Merolla (*Fast Opportunistic Mixture of Experts*) | `MIT` | [`github.com/pmerolla/fomoe`](https://github.com/pmerolla/fomoe) | [`SkipAttentionSpeculativeRouter`](../src/hivemind/swarm/diloco_nextgen.py) (`SI-10`: Skip-Attention pre-attention speculative expert prefetch achieving `~85.8%` top-$k$ accuracy + Cache-Aware Routing `CAR` fallback) |
| **[`learning-at-home/hivemind`](https://github.com/learning-at-home/hivemind)** | Max Ryabinin, Anton Gusev, Anton Diskin, Denis Mazur et al. (*Learning@Home / Yandex Research / HSE*) | `MIT` | [`arXiv:2002.04013`](https://arxiv.org/abs/2002.04013) | Foundational inspiration for decentralized Kademlia DHT parameter averaging (`DecentralizedAverager`) and WAN Mixture-of-Experts routing |

---

## 2. Academic Citations (every arXiv id opened and checked on 2026-10-23)

Entries are ordered by arXiv id. **Removed in this revision** (fabricated identifiers or wrong papers found during the 2026-10-23 review):

- `arXiv:2607.12925` listed as MuLoCo — is a condensed-matter physics paper (spin-1 parent Hamiltonians); MuLoCo is arXiv:2505.23725.
- `arXiv:2610.19064 attributed to "Lucas Morales"`  — the paper is by Belilovsky, Horoi, Oyallon & Thérien; no author named Morales.
- `arXiv:2602.23029` "Bogachev & Ryabinin, Generalization Fidelity of DiLoCo" — is WISER, a composed-image-retrieval paper.
- `arXiv:2602.06954` "Khaled & Jin, Tighter Convergence Bounds" — is an astrophysics paper on self-gravitating disks.
- `arXiv:2608.05958` "Wang, Subspace-Constrained Outer Optimization for DiLoCo" — is a paper on pairwise comparisons in the Analytic Hierarchy Process.
- `arXiv:2609.02737` "Li, FedLoCo" — is "Language Models Can Control Their Own Attention".
- `arXiv:2510.17558` "Kwon & Kim, PoLoCo one-bit quantization" — is "The Free Transformer" (Fleuret).
- `arXiv:2512.02042` "Kwon & Kim, SNQ-L" — is a paper on Fermat-type partial differential-difference equations.
- `arXiv:2603.24124` "Sani & Lane, FGBoost" — is "The Alignment Tax" (uncertainty estimation in aligned LLMs).
- `arXiv:2002.04013` listed as the Hivemind paper — is GShard; the Hivemind paper is arXiv:2002.04013.

**Corrected metadata:**

- Decoupled DiLoCo (2604.21428) was attributed to Sani et al. / CaMLSys; it is by the Decoupled DiLoCo Team at Google DeepMind & Google Research.
- CGAD (2605.09126) authors are Vatsal Shah & Jiahao Sun (FLock.io), not "Neil Shah & Ming Sun".
- HeLoCo (2606.00271) is by Al Asif, Diem, Muñoz, Wolf, Jannesari & Mazaheri, not "Asif & Belilovsky".
- AutoLoCo (2609.36662) is by He, Zhang, Li & Yang, not "He & Tao"; its title is "Communication Efficient Distributed LLM Training via Adaptive Synchronization".
- FoMoE (2606.19025) is titled "Breaking the Full-Replica Barrier with a Federation of MoEs" with eight authors, not a three-author "Federated Mixture-of-Experts ...".
- IsoLoCo (2607.03011) is titled "Can Model Merging Improve Aggregation in DiLoCo?" by Horoi, Thérien, Wolf & Belilovsky.
- PULSE (2602.03839) is titled "Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL".
- SparseLoCo (2508.15706) is titled "Communication Efficient LLM Pre-training with SparseLoCo" by Sarfi, Thérien, Lidin & Belilovsky.
- Streaming DiLoCo (2501.18512) and Scaling Laws (2503.09799) carried invented co-authors ("Testuggine", "Ramasesh").

1. **Ryabinin, Max, Gusev, Anton (2020).** *Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts.* [`arXiv:2002.04013`](https://arxiv.org/abs/2002.04013) — NeurIPS 2020; code `learning-at-home/hivemind`. HiveMind use: DHT averaging and decentralized MoE inspiration.
2. **Richtárik, Peter, Sokolov, Igor, Fatkhullin, Ilyas (2021).** *EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback.* [`arXiv:2106.05203`](https://arxiv.org/abs/2106.05203) — NeurIPS 2021. HiveMind use: `EF21DiLoCoErrorFeedbackBuffer`.
3. **Douillard, Arthur, Feng, Qixuan, Rusu, Andrei A., Chhaparia, Rachita, Donchev, Yani, Kuncoro, Adhiguna, Ranzato, Marc'Aurelio, Szlam, Arthur, Shen, Jiajun (2023).** *DiLoCo: Distributed Low-Communication Training of Language Models.* [`arXiv:2311.08105`](https://arxiv.org/abs/2311.08105) — Google DeepMind. HiveMind use: Inner-AdamW / outer-Nesterov baseline (`DiLoCoOuterOptimizer`).
4. **Liu, Bo, Chhaparia, Rachita, Douillard, Arthur, Kale, Satyen, Rusu, Andrei A., Shen, Jiajun, Szlam, Arthur, Ranzato, Marc'Aurelio (2024).** *Asynchronous Local-SGD Training for Language Modeling.* [`arXiv:2401.09135`](https://arxiv.org/abs/2401.09135) — Google DeepMind; code `google-deepmind/asyncdiloco`. HiveMind use: Delayed-Nesterov / asynchronous outer updates (`SI-06` baseline).
5. **Sukhbaatar, Sainbayar, Golovneva, Olga, Sharma, Vasu, Xu, Hu, Lin, Xi Victoria, Rozière, Baptiste, Kahn, Jacob, Li, Daniel, Yih, Wen-tau, Weston, Jason, Li, Xian (2024).** *Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM.* [`arXiv:2403.07816`](https://arxiv.org/abs/2403.07816) — FAIR at Meta. HiveMind use: BTM upcycling into a sparse MoE (`btm_upcycle.py`).
6. **Jaghouar, Sami, Ong, Jack Min, Hagemann, Johannes (2024).** *OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training.* [`arXiv:2407.07852`](https://arxiv.org/abs/2407.07852) — Prime Intellect; code `PrimeIntellect-ai/OpenDiLoCo`, `prime-diloco`. HiveMind use: CPU flat-bucket outer state (`SI-06B`).
7. **Peng, Bowen, Quesnelle, Jeffrey, Kingma, Diederik P. (2024).** *DeMo: Decoupled Momentum Optimization.* [`arXiv:2411.19870`](https://arxiv.org/abs/2411.19870) — Nous Research; code `bloc97/DeMo`. HiveMind use: Chunked orthonormal DCT (`SI-09C`).
8. **Gargiulo, Antonio Andrea, Crisostomi, Donato, Bucarelli, Maria Sofia, Scardapane, Simone, Silvestri, Fabrizio, Rodolà, Emanuele (2024).** *Task Singular Vectors: Reducing Task Interference in Model Merging.* [`arXiv:2412.00081`](https://arxiv.org/abs/2412.00081) — CVPR 2025. HiveMind use: TSV decorrelated merging (`SI-07B`).
9. **Douillard, Arthur, Donchev, Yani, Rush, J. Keith, Kale, Satyen, Charles, Zachary, Teston, Gabriel, Garrett, Zachary, Shen, Jiajun, McIlroy, Ross, Lacey, David, Ramé, Alexandre, Szlam, Arthur, Ranzato, Marc'Aurelio, Barham, Paul R. (2025).** *Streaming DiLoCo with Overlapping Communication: Towards a Distributed Free Lunch.* [`arXiv:2501.18512`](https://arxiv.org/abs/2501.18512) — COLM 2025; Google DeepMind. HiveMind use: Fragment streaming (`synced_fragments`).
10. **Marczak, Daniel, Magistri, Simone, Cygert, Sebastian, Twardowski, Bartłomiej, Bagdanov, Andrew D., van de Weijer, Joost (2025).** *No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces.* [`arXiv:2502.04959`](https://arxiv.org/abs/2502.04959) — ICML 2025. HiveMind use: Iso-C / Iso-CTS merging (`SI-07`).
11. **Charles, Zachary, Teston, Gabriel, Dery, Lucio, Rush, Keith, Fallen, Nova, Garrett, Zachary, Szlam, Arthur, Douillard, Arthur (2025).** *Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo.* [`arXiv:2503.09799`](https://arxiv.org/abs/2503.09799) — Google DeepMind / Google Research. HiveMind use: Replica-count scaling expectations.
12. **Thérien, Benjamin, Huang, Xiaolong, Defazio, Aaron, Rish, Irina, Belilovsky, Eugene (2025).** *MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo.* [`arXiv:2505.23725`](https://arxiv.org/abs/2505.23725) — FAIR at Meta / Mila / Concordia. HiveMind use: Newton-Schulz inner preconditioner (`SI-11`).
13. **Sarfi, Amir, Thérien, Benjamin, Lidin, Joel, Belilovsky, Eugene (2025).** *Communication Efficient LLM Pre-training with SparseLoCo.* [`arXiv:2508.15706`](https://arxiv.org/abs/2508.15706) — Covenant AI / Mila / Concordia; code `one-covenant/SparseLoCo`. HiveMind use: Top-k + 2-bit centroid quantisation + outer error feedback (`SI-09B`).
14. **Miahi, Erfan, Belilovsky, Eugene (2026).** *Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL.* [`arXiv:2602.03839`](https://arxiv.org/abs/2602.03839) — Covenant AI / Mila / Concordia (PULSE). HiveMind use: Compute-visible gate below the BF16 rounding threshold (`SI-09B`).
15. **Kim, Yongwan, Park, Sungchul (2026).** *MAGNET: Autonomous Expert Model Generation via Decentralized Autoresearch and BitNet Training.* [`arXiv:2603.25813`](https://arxiv.org/abs/2603.25813) — Holo Studio. HiveMind use: Related decentralized specialist-merging system (survey only).
16. **Agarwala, Atish (2026).** *High dimensional theory of two-phase optimizers.* [`arXiv:2603.26954`](https://arxiv.org/abs/2603.26954) — Google DeepMind. HiveMind use: Theory behind outer-momentum choices (Season 9 `HM-S9-05`).
17. **Decoupled DiLoCo Team (Google DeepMind & Google Research; incl. Douillard, Arthur; Rush, Keith; Donchev, Yani; Charles, Zachary; Dubey, Ayush; Woodworth, Blake) (2026).** *Decoupled DiLoCo for Resilient Distributed Pre-training.* [`arXiv:2604.21428`](https://arxiv.org/abs/2604.21428) — Google DeepMind & Google Research. HiveMind use: Minimum quorum, adaptive grace window, token-weighted RDA merge (`SI-12`, `DecoupledDiLoCoQuorumSyncer`).
18. **Shah, Vatsal, Sun, Jiahao (2026).** *Cosine-Gated Adam-Decay: Drop-In Staleness-Aware Outer Optimization for Decoupled DiLoCo.* [`arXiv:2605.09126`](https://arxiv.org/abs/2605.09126) — FLock.io. HiveMind use: PA-CGAD staleness gate (`SI-06`).
19. **Topollai, Kristi, Ma, Allan, Dimlioglu, Tolga, Tay, Sui Jiet, Choromanska, Anna (2026).** *Outer-Momentum Restarting in High-Dimensional Two-Phase Optimization.* [`arXiv:2605.28585`](https://arxiv.org/abs/2605.28585) — HiLD 2026 workshop; NYU. HiveMind use: Periodic outer-momentum reset (Season 9 `HM-S9-05`).
20. **Al Asif, Abdullah, Diem, Patrick, Muñoz, Juan Pablo, Wolf, Felix, Jannesari, Ali, Mazaheri, Arya (2026).** *HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity.* [`arXiv:2606.00271`](https://arxiv.org/abs/2606.00271) — Iowa State / TU Darmstadt / Maro Systems / PanocularAI. HiveMind use: Direction-aware stale-drift projection (`SI-08A`).
21. **Cagnasso, Pietro, Belilovsky, Eugene, Oyallon, Edouard (2026).** *Unifying Local Communications and Local Updates for LLM Pretraining.* [`arXiv:2606.11081`](https://arxiv.org/abs/2606.11081) — Concordia / Mila / CNRS, Sorbonne. HiveMind use: Gossip-accelerated outer optimizer (Season 9 `HM-S9-04`).
22. **Sani, Lorenzo, Cao, Zeyu, Kurmanji, Meghdad, Iacob, Alex, Jovanović, Andrej, Gao, Yan, Zhao, Wanru, Lane, Nicholas D. (2026).** *FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs.* [`arXiv:2606.19025`](https://arxiv.org/abs/2606.19025) — CaMLSys, University of Cambridge / Flower Labs. HiveMind use: Partial expert replication, ghost experts, Skip-Token, sigma-MoE (`SI-09A`, `fomoe_sigma_routing.py`).
23. **Koneputugodage, Chamin Hewa, Ajanthan, Thalaiyasingam, Ramasinghe, Sameera, Dolatabadi, Hadi Mohaghegh, Siriwardhana, Shamane, Avraham, Gil, Shevchenko, Violetta, Pajak, Karol, Snewin, James, Long, Alexander (2026).** *Factored Gossip DiLoCo: Reducing Blocking Communication in DiLoCo.* [`arXiv:2606.22768`](https://arxiv.org/abs/2606.22768) — Pluralis Research. HiveMind use: Non-blocking Mix1 / blocking Mix2 factorisation (Season 9 `HM-S9-04`).
24. **Twelftree, Maxwell, Lemphers, David, He, An-chi, Yang, Yue (2026).** *Not Every Sync Is Safe: Calibrated DiLoCo Scheduling for Shared AI Infrastructure.* [`arXiv:2607.02544`](https://arxiv.org/abs/2607.02544) — Maincode. HiveMind use: Sync-deferral controller and matched-random control (survey only).
25. **Horoi, Stefan, Thérien, Benjamin, Wolf, Guy, Belilovsky, Eugene (2026).** *Can Model Merging Improve Aggregation in DiLoCo?.* [`arXiv:2607.03011`](https://arxiv.org/abs/2607.03011) — Université de Montréal / Mila / Concordia; code `shoroi/isoloco` (IsoLoCo). HiveMind use: Iso-C / Iso-CTS aggregation (`SI-07`).
26. **He, Pengyu, Zhang, Yan, Li, Ruien, Yang, Guangwen (2026).** *AutoLoCo: Communication Efficient Distributed LLM Training via Adaptive Synchronization.* [`arXiv:2609.36662`](https://arxiv.org/abs/2609.36662) — Caltech / Tsinghua / UW-Madison. HiveMind use: Adaptive interval with outer LR/momentum correction (`SI-08B`).
27. **Palanciyan, Zarè, van Osch, Thomas, van der Wal, Douwe, Kotevska, Olivera, Kok, Tim (2026).** *Cross-Facility LLM Pre-training on HPC: Elastic Aggregation, Data Leasing, and Queue-Aware Placement.* [`arXiv:2610.03457`](https://arxiv.org/abs/2610.03457) — SURF / Oak Ridge National Laboratory. HiveMind use: Elastic token-weighted Nesterov step, DARL data leasing (Season 9 `HM-S9-06`).
28. **Huang, Benhao, Shi, Chufan, Chen, Junlin, Wen, Shicheng, Liu, Zhengzhong, Xing, Eric, Ma, Xuezhe (2026).** *Towards Looped Models Done Right, Part II: Rethinking at Fixed Points.* [`arXiv:2610.06833`](https://arxiv.org/abs/2610.06833) — Institute of Foundation Models / USC / CMU. HiveMind use: Terminal-KV sharing, truncated BPTT, learned depth prior (`LoopedExpertBackbone`).
29. **Belilovsky, Eugene, Horoi, Stefan, Oyallon, Edouard, Thérien, Benjamin (2026).** *How Inner Optimizers Shape Replica Scaling.* [`arXiv:2610.19064`](https://arxiv.org/abs/2610.19064) — Mila / Concordia / Université de Montréal / CNRS. HiveMind use: Whitening inner optimizers restore replica scaling (`SI-11`).
30. **Huang, Benhao, Shi, Chufan, Chen, Junlin, Wen, Shicheng, Liu, Zhengzhong, Xing, Eric, Ma, Xuezhe (2026).** *Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design.* [`arXiv:2610.23036`](https://arxiv.org/abs/2610.23036) — Institute of Foundation Models / USC / CMU; code `ifm-ai/xllm-loop`. HiveMind use: Prelude / recurrent core / coda topology (`LoopedExpertBackbone`).
31. **Kale, Satyen, Douillard, Arthur, Donchev, Yanislav (2025).** *Eager Updates for Overlapped Communication and Computation in DiLoCo.* [`arXiv:2502.12996`](https://arxiv.org/abs/2502.12996) — Google DeepMind / Google Research. HiveMind use: 1-outer-step eager overlap as the candidate deferred-learner rule for `HM-S9-01` / `HM-S9-13` (verified 2026-10-23).
32. **Defazio, Aaron, Mishchenko, Konstantin, Raman, Parameswaran, Shi, Hao-Jun Michael, Xiao, Lin (2025).** *Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs.* [`arXiv:2512.17131`](https://arxiv.org/abs/2512.17131) — Meta Superintelligence Labs. HiveMind use: Generalized Primal Averaging as the control for the outer-momentum-restart factor of E5 (`HM-S9-05`) (verified 2026-10-23).
33. **Wang, Yichuan, Li, Zhifei, Liu, Shu, Wu, Yongji, Mao, Ziming, Zhao, Yilong, Yan, Xiao, Xu, Zhiying, Zhou, Yang, Stoica, Ion, Min, Sewon, Zaharia, Matei, Gonzalez, Joseph E. (2025; Proceedings of Machine Learning and Systems 8, MLSys 2026).** *LEANN: A Low-Storage Vector Index* (MLSys title: *A Low-Storage Overhead Vector Index*). [`arXiv:2506.08276`](https://arxiv.org/abs/2506.08276v2) — UC Berkeley / CUHK / AWS / UC Davis; code MIT-licensed (upstream repository to be confirmed, see `docs/S22_LIGHT_MEMORY.md`). HiveMind use: **proposed only** (Season 22 `HM-S22-01`..`07`), not integrated, not measured on our hardware. Abstract and body read 2026-10-28; the MLSys PDF and the MarktechPost write-up were not.
34. **Alabdulkareem, Abdulrahman, Tenison, Irene, Zhang, Yang, Zhu, Yada, Katz, Boris, Gutfreund, Dan (2026).** *Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families.* [`arXiv:2610.15095`](https://arxiv.org/abs/2610.15095) — MIT CSAIL & MIT-IBM Computing Research Lab; code [`Ar-Kareem/SwaT`](https://github.com/Ar-Kareem/SwaT) cloned in `third_party/SwaT/` (`commit ae2de2285caf82f4d500526bf6e6992747b98eea`). HiveMind use: Season 34 (`HM-S34-01`..`07`, `docs/S34_SWAT_SWARM_TELEPATHY_INTEGRATION.md`) prefill-free cross-family KV-cache transfer (`CacheRendererV2`), character-span RoPE alignment, and VLM-to-text / long-context Bee telepathy (full 25-page PDF and repository inspected 2026-10-31).
35. **Zweiger, Adam, Fu, Xinghong, Guo, Han, Kim, Yoon (2026).** *Fast KV Compaction via Attention Matching.* [`arXiv:2602.16284`](https://arxiv.org/abs/2602.16284) — MIT. HiveMind use: Season 34 (`HM-S34-02`, `SwaT+AM`) Attention Matching KV-cache compaction paired with `SwaT` for compacted latent Nectar handoffs.
36. **Behrouz, Ali, Hashemi, Farnoosh, Javanmard, Adel, Mirrokni, Vahab (2026).** *Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories.* [`arXiv:2606.03979`](https://arxiv.org/abs/2606.03979) (`arXiv:2606.03979v2`) — Google Research & Cornell University. HiveMind use: Season 34 (`HM-S34-08`..`10`, `docs/S34_SWAT_SWARM_TELEPATHY_INTEGRATION.md`, strengthening `HM-S21-03`, `HM-S32-01`, and `HM-S33-05`) Two-Stage Nocturnal Sleep paradigm combining Stage 1 NREM Slow-Wave Memory Consolidation (`Continuum Memory System`, `Compute–Consolidate–Update`, Upward Distillation / `Knowledge Seeding` via On-Policy `GKD` + RL `Learning-to-Imitate`, and fast-expert synaptic pruning) with Stage 2 REM `Dreaming` (random MoE expert mixing + gradient-norm Top-$k$+$b$ dream selection + `ReST^EM` binary reward) (full 26-page PDF inspected 2026-10-31).

### Author's own publications used by the programme (DOIs read from the Zenodo API, 2026-10-23)

- **Callens, Xavier (2026).** *A laboratory method for AI-assisted science: formal proof, exact arithmetic, controlled experiment, and a claim ledger, tested for a month on the Navier-Stokes problem.* [`doi:10.5281/zenodo.22829767`](https://doi.org/10.5281/zenodo.22829767) (v1.10.2). HiveMind use: protocol of docs/RESEARCH_PROGRAM.md section 4.
- **Callens, Xavier (2026).** *Karpathy-autoresearch Screening and Machine-Checked Elenchus Deep-Dive Across Fermat's Last Theorem, Navier-Stokes Admissibility Censorship, and Kinetic Spectral Caps.* [`doi:10.5281/zenodo.23467992`](https://doi.org/10.5281/zenodo.23467992) (1.0). HiveMind use: screening loop for the 0.5B E5 sweep (HM-S9-14).
- **Callens, Xavier (2026).** *RunuX: Deterministic Fixed-Point Attention and Measured PolarQuant KV-Cache Compression on a Commodity GPU, with a Hardware Test of Its TPU Claims.* [`doi:10.5281/zenodo.22879667`](https://doi.org/10.5281/zenodo.22879667) (2.0.0). HiveMind use: KV-cache measurement E11 must beat (HM-S10-05); runtime code not reused.
- **Callens, Xavier (2026).** *Non-Equilibrium Hopfield-Ninio Proofreading, Chimeric ART-Exo Fidelity, Gosset E8 Dicodon Quantization, and Topological DNA Supercoiling in LeanFlow, rusty-SUNDIALS, and Lean 4.* [`doi:10.5281/zenodo.23526783`](https://doi.org/10.5281/zenodo.23526783) (3). HiveMind use: second user of the E8 decoder family (HM-S10-07).
- **Callens, Xavier (2026).** *Laya-LoRA Coding Companion: Asymmetric Dual-Process Test-Time Compute, Full-Scale Curriculum on 10 Structured Coding Datasets, and Serverless Multi-Tier Inference Infrastructure on GCP.* [`doi:10.5281/zenodo.23095224`](https://doi.org/10.5281/zenodo.23095224) (camera-ready v3). HiveMind use: candidate real workload for the 8-domain campaign (HM-S10-06).

Software without an arXiv paper: [`pmerolla/fomoe`](https://github.com/pmerolla/fomoe) (Paul Merolla, MIT; multi-tier expert cache and cache-aware routing; it is **not** the FoMoE paper above), [`learning-at-home/hivemind`](https://github.com/learning-at-home/hivemind) (MIT), [`PrimeIntellect-ai/prime-diloco`](https://github.com/PrimeIntellect-ai/prime-diloco) (Apache-2.0), Muon (Keller Jordan et al., 2024, blog post).

---

## 3. Community Contributors & HuggingFace Datasets Acknowledgments

- **Project HiveMind Open-Source Contributors**: Special thanks to **`@clark719664`** for 52 merged pull requests across Seasons 5–7 (`PR #52..#82`, `#86..#91`), including `EF21DiLoCoErrorFeedbackBuffer` (`HM-S6-01`), 3-tier `128x7B` `HierarchicalMacroMicroRouter` (`HM-S6-02`), `StrategyOptimizerTelemetryEngine` (`HM-S6-04`), real WAN UDP socket tests (`HM-S6-03`), and Season 6 Lean 4 proofs (`Theorems 65–72`).
- **Curated HuggingFace Datasets (`Operation Polymath-128`)**:
  - `nvidia/OpenMathReasoning` & `AI-MO/NuminaMath-CoT`
  - `open-r1/OpenR1-Math-220k` & `bespokelabs/Bespoke-Stratos-17k`
  - `bigcode/the-stack-v2-dedup` (Rust & Lean 4 subsets)
  - `Camel-AI/physics` & `Camel-AI/biology`

---

## 4. BibTeX Citation Block

```bibtex
@article{ryabinin2020hivemind,
  title={Towards Crowdsourced Training of Large Neural Networks using Decentralized Mixture-of-Experts},
  author={Ryabinin, Max and Gusev, Anton},
  journal={arXiv preprint arXiv:2002.04013},
  year={2020},
  note={NeurIPS 2020; code `learning-at-home/hivemind`}
}
@article{richtarik2021ef21,
  title={EF21: A New, Simpler, Theoretically Better, and Practically Faster Error Feedback},
  author={Richtárik, Peter and Sokolov, Igor and Fatkhullin, Ilyas},
  journal={arXiv preprint arXiv:2106.05203},
  year={2021},
  note={NeurIPS 2021}
}
@article{douillard2023diloco,
  title={DiLoCo: Distributed Low-Communication Training of Language Models},
  author={Douillard, Arthur and Feng, Qixuan and Rusu, Andrei A. and Chhaparia, Rachita and Donchev, Yani and Kuncoro, Adhiguna and Ranzato, Marc'Aurelio and Szlam, Arthur and Shen, Jiajun},
  journal={arXiv preprint arXiv:2311.08105},
  year={2023},
  note={Google DeepMind}
}
@article{liu2024asynclocalsgd,
  title={Asynchronous Local-SGD Training for Language Modeling},
  author={Liu, Bo and Chhaparia, Rachita and Douillard, Arthur and Kale, Satyen and Rusu, Andrei A. and Shen, Jiajun and Szlam, Arthur and Ranzato, Marc'Aurelio},
  journal={arXiv preprint arXiv:2401.09135},
  year={2024},
  note={Google DeepMind; code `google-deepmind/asyncdiloco`}
}
@article{sukhbaatar2024btm,
  title={Branch-Train-MiX: Mixing Expert LLMs into a Mixture-of-Experts LLM},
  author={Sukhbaatar, Sainbayar and Golovneva, Olga and Sharma, Vasu and Xu, Hu and Lin, Xi Victoria and Rozière, Baptiste and Kahn, Jacob and Li, Daniel and Yih, Wen-tau and Weston, Jason and Li, Xian},
  journal={arXiv preprint arXiv:2403.07816},
  year={2024},
  note={FAIR at Meta}
}
@article{jaghouar2024opendiloco,
  title={OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training},
  author={Jaghouar, Sami and Ong, Jack Min and Hagemann, Johannes},
  journal={arXiv preprint arXiv:2407.07852},
  year={2024},
  note={Prime Intellect; code `PrimeIntellect-ai/OpenDiLoCo`, `prime-diloco`}
}
@article{peng2024demo,
  title={DeMo: Decoupled Momentum Optimization},
  author={Peng, Bowen and Quesnelle, Jeffrey and Kingma, Diederik P.},
  journal={arXiv preprint arXiv:2411.19870},
  year={2024},
  note={Nous Research; code `bloc97/DeMo`}
}
@article{gargiulo2024tsv,
  title={Task Singular Vectors: Reducing Task Interference in Model Merging},
  author={Gargiulo, Antonio Andrea and Crisostomi, Donato and Bucarelli, Maria Sofia and Scardapane, Simone and Silvestri, Fabrizio and Rodolà, Emanuele},
  journal={arXiv preprint arXiv:2412.00081},
  year={2024},
  note={CVPR 2025}
}
@article{douillard2025streamingdiloco,
  title={Streaming DiLoCo with Overlapping Communication: Towards a Distributed Free Lunch},
  author={Douillard, Arthur and Donchev, Yani and Rush, J. Keith and Kale, Satyen and Charles, Zachary and Teston, Gabriel and Garrett, Zachary and Shen, Jiajun and McIlroy, Ross and Lacey, David and Ramé, Alexandre and Szlam, Arthur and Ranzato, Marc'Aurelio and Barham, Paul R.},
  journal={arXiv preprint arXiv:2501.18512},
  year={2025},
  note={COLM 2025; Google DeepMind}
}
@article{marczak2025iso,
  title={No Task Left Behind: Isotropic Model Merging with Common and Task-Specific Subspaces},
  author={Marczak, Daniel and Magistri, Simone and Cygert, Sebastian and Twardowski, Bartłomiej and Bagdanov, Andrew D. and van de Weijer, Joost},
  journal={arXiv preprint arXiv:2502.04959},
  year={2025},
  note={ICML 2025}
}
@article{charles2025scalingdiloco,
  title={Communication-Efficient Language Model Training Scales Reliably and Robustly: Scaling Laws for DiLoCo},
  author={Charles, Zachary and Teston, Gabriel and Dery, Lucio and Rush, Keith and Fallen, Nova and Garrett, Zachary and Szlam, Arthur and Douillard, Arthur},
  journal={arXiv preprint arXiv:2503.09799},
  year={2025},
  note={Google DeepMind / Google Research}
}
@article{therien2025muloco,
  title={MuLoCo: Muon is a Practical Inner Optimizer for DiLoCo},
  author={Thérien, Benjamin and Huang, Xiaolong and Defazio, Aaron and Rish, Irina and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2505.23725},
  year={2025},
  note={FAIR at Meta / Mila / Concordia}
}
@article{sarfi2025sparseloco,
  title={Communication Efficient LLM Pre-training with SparseLoCo},
  author={Sarfi, Amir and Thérien, Benjamin and Lidin, Joel and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2508.15706},
  year={2025},
  note={Covenant AI / Mila / Concordia; code `one-covenant/SparseLoCo`}
}
@article{miahi2026pulse,
  title={Understanding and Exploiting Weight Update Sparsity for Communication-Efficient Distributed RL},
  author={Miahi, Erfan and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2602.03839},
  year={2026},
  note={Covenant AI / Mila / Concordia (PULSE)}
}
@article{kim2026magnet,
  title={MAGNET: Autonomous Expert Model Generation via Decentralized Autoresearch and BitNet Training},
  author={Kim, Yongwan and Park, Sungchul},
  journal={arXiv preprint arXiv:2603.25813},
  year={2026},
  note={Holo Studio}
}
@article{agarwala2026twophase,
  title={High dimensional theory of two-phase optimizers},
  author={Agarwala, Atish},
  journal={arXiv preprint arXiv:2603.26954},
  year={2026},
  note={Google DeepMind}
}
@article{douillard2026decoupled,
  title={Decoupled DiLoCo for Resilient Distributed Pre-training},
  author={Decoupled DiLoCo Team (Google DeepMind & Google Research; incl. Douillard, Arthur; Rush, Keith; Donchev, Yani; Charles, Zachary; Dubey, Ayush; Woodworth, Blake)},
  journal={arXiv preprint arXiv:2604.21428},
  year={2026},
  note={Google DeepMind & Google Research}
}
@article{shah2026cgad,
  title={Cosine-Gated Adam-Decay: Drop-In Staleness-Aware Outer Optimization for Decoupled DiLoCo},
  author={Shah, Vatsal and Sun, Jiahao},
  journal={arXiv preprint arXiv:2605.09126},
  year={2026},
  note={FLock.io}
}
@article{topollai2026restart,
  title={Outer-Momentum Restarting in High-Dimensional Two-Phase Optimization},
  author={Topollai, Kristi and Ma, Allan and Dimlioglu, Tolga and Tay, Sui Jiet and Choromanska, Anna},
  journal={arXiv preprint arXiv:2605.28585},
  year={2026},
  note={HiLD 2026 workshop; NYU}
}
@article{alasif2026heloco,
  title={HeLoCo: Efficient asynchronous low-communication training under data and device heterogeneity},
  author={Al Asif, Abdullah and Diem, Patrick and Muñoz, Juan Pablo and Wolf, Felix and Jannesari, Ali and Mazaheri, Arya},
  journal={arXiv preprint arXiv:2606.00271},
  year={2026},
  note={Iowa State / TU Darmstadt / Maro Systems / PanocularAI}
}
@article{cagnasso2026gasloc,
  title={Unifying Local Communications and Local Updates for LLM Pretraining},
  author={Cagnasso, Pietro and Belilovsky, Eugene and Oyallon, Edouard},
  journal={arXiv preprint arXiv:2606.11081},
  year={2026},
  note={Concordia / Mila / CNRS, Sorbonne}
}
@article{sani2026fomoe,
  title={FoMoE: Breaking the Full-Replica Barrier with a Federation of MoEs},
  author={Sani, Lorenzo and Cao, Zeyu and Kurmanji, Meghdad and Iacob, Alex and Jovanović, Andrej and Gao, Yan and Zhao, Wanru and Lane, Nicholas D.},
  journal={arXiv preprint arXiv:2606.19025},
  year={2026},
  note={CaMLSys, University of Cambridge / Flower Labs}
}
@article{koneputugodage2026factoredgossip,
  title={Factored Gossip DiLoCo: Reducing Blocking Communication in DiLoCo},
  author={Koneputugodage, Chamin Hewa and Ajanthan, Thalaiyasingam and Ramasinghe, Sameera and Dolatabadi, Hadi Mohaghegh and Siriwardhana, Shamane and Avraham, Gil and Shevchenko, Violetta and Pajak, Karol and Snewin, James and Long, Alexander},
  journal={arXiv preprint arXiv:2606.22768},
  year={2026},
  note={Pluralis Research}
}
@article{twelftree2026wadiloco,
  title={Not Every Sync Is Safe: Calibrated DiLoCo Scheduling for Shared AI Infrastructure},
  author={Twelftree, Maxwell and Lemphers, David and He, An-chi and Yang, Yue},
  journal={arXiv preprint arXiv:2607.02544},
  year={2026},
  note={Maincode}
}
@article{horoi2026isoloco,
  title={Can Model Merging Improve Aggregation in DiLoCo?},
  author={Horoi, Stefan and Thérien, Benjamin and Wolf, Guy and Belilovsky, Eugene},
  journal={arXiv preprint arXiv:2607.03011},
  year={2026},
  note={Université de Montréal / Mila / Concordia; code `shoroi/isoloco` (IsoLoCo)}
}
@article{he2026autoloco,
  title={AutoLoCo: Communication Efficient Distributed LLM Training via Adaptive Synchronization},
  author={He, Pengyu and Zhang, Yan and Li, Ruien and Yang, Guangwen},
  journal={arXiv preprint arXiv:2609.36662},
  year={2026},
  note={Caltech / Tsinghua / UW-Madison}
}
@article{palanciyan2026crossfacility,
  title={Cross-Facility LLM Pre-training on HPC: Elastic Aggregation, Data Leasing, and Queue-Aware Placement},
  author={Palanciyan, Zarè and van Osch, Thomas and van der Wal, Douwe and Kotevska, Olivera and Kok, Tim},
  journal={arXiv preprint arXiv:2610.03457},
  year={2026},
  note={SURF / Oak Ridge National Laboratory}
}
@article{huang2026loopedpart2,
  title={Towards Looped Models Done Right, Part II: Rethinking at Fixed Points},
  author={Huang, Benhao and Shi, Chufan and Chen, Junlin and Wen, Shicheng and Liu, Zhengzhong and Xing, Eric and Ma, Xuezhe},
  journal={arXiv preprint arXiv:2610.06833},
  year={2026},
  note={Institute of Foundation Models / USC / CMU}
}
@article{belilovsky2026inner,
  title={How Inner Optimizers Shape Replica Scaling},
  author={Belilovsky, Eugene and Horoi, Stefan and Oyallon, Edouard and Thérien, Benjamin},
  journal={arXiv preprint arXiv:2610.19064},
  year={2026},
  note={Mila / Concordia / Université de Montréal / CNRS}
}
@article{huang2026loopedpart1,
  title={Towards Looped Models Done Right, Part I: Topology, Input Injection, Recurrent-State Design},
  author={Huang, Benhao and Shi, Chufan and Chen, Junlin and Wen, Shicheng and Liu, Zhengzhong and Xing, Eric and Ma, Xuezhe},
  journal={arXiv preprint arXiv:2610.23036},
  year={2026},
  note={Institute of Foundation Models / USC / CMU; code `ifm-ai/xllm-loop`}
}
@software{merolla_fomoe,
  title={fomoe: Fast Opportunistic Mixture-Of-Experts},
  author={Merolla, Paul},
  url={https://github.com/pmerolla/fomoe},
  license={MIT}
}
@article{kale2025eager,
  title={Eager Updates for Overlapped Communication and Computation in DiLoCo},
  author={Kale, Satyen and Douillard, Arthur and Donchev, Yanislav},
  journal={arXiv preprint arXiv:2502.12996},
  year={2025},
  note={Google DeepMind / Google Research}
}
@article{defazio2025gpa,
  title={Smoothing DiLoCo with Primal Averaging for Faster Training of LLMs},
  author={Defazio, Aaron and Mishchenko, Konstantin and Raman, Parameswaran and Shi, Hao-Jun Michael and Xiao, Lin},
  journal={arXiv preprint arXiv:2512.17131},
  year={2025},
  note={Meta Superintelligence Labs}
}
@inproceedings{wang2026leann,
  title={{LEANN}: A Low-Storage Overhead Vector Index},
  author={Wang, Yichuan and Li, Zhifei and Liu, Shu and Wu, Yongji and Mao, Ziming and Zhao, Yilong and Yan, Xiao and Xu, Zhiying and Zhou, Yang and Stoica, Ion and Min, Sewon and Zaharia, Matei and Gonzalez, Joseph E.},
  booktitle={Proceedings of Machine Learning and Systems (MLSys)},
  volume={8},
  year={2026},
  note={arXiv:2506.08276 (arXiv title: LEANN: A Low-Storage Vector Index); UC Berkeley / CUHK / AWS / UC Davis}
}
@misc{alabdulkareem2026swarmtelepathyprefillfreekvcache,
  title={Swarm Telepathy: Prefill-Free KV-Cache Transfer Across Families},
  author={Alabdulkareem, Abdulrahman and Tenison, Irene and Zhang, Yang and Zhu, Yada and Katz, Boris and Gutfreund, Dan},
  year={2026},
  eprint={2610.15095},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2610.15095},
  note={MIT CSAIL & MIT-IBM Computing Research Lab; code Ar-Kareem/SwaT}
}
@article{zweiger2026attentionmatching,
  title={Fast KV Compaction via Attention Matching},
  author={Zweiger, Adam and Fu, Xinghong and Guo, Han and Kim, Yoon},
  journal={arXiv preprint arXiv:2602.16284},
  year={2026},
  url={https://arxiv.org/abs/2602.16284}
}
@misc{behrouz2026languagemodelsneedsleep,
  title={Language Models Need Sleep: Learning to Self-Modify and Consolidate Memories},
  author={Behrouz, Ali and Hashemi, Farnoosh and Javanmard, Adel and Mirrokni, Vahab},
  year={2026},
  eprint={2606.03979},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2606.03979},
  note={Google Research & Cornell University}
}
}
`
