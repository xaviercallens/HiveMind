# Third-party notices

This file lists software that the Solver Bee and the mesh depend on and that carries its own notice or licence conditions. It is a good-faith record kept by the project, not legal advice. Vendored files live under `third_party/`.

## rusty-SUNDIALS (the solver of the Solver Bee)

**What and how it is used.** `solvers/hivemind-solver` depends on the `cvode` and `nvector` crates (and, transitively, `sundials-core`) of https://github.com/xaviercallens/rusty-SUNDIALS (a Rust port of the CVODE integrator of LLNL's SUNDIALS) as a **git dependency pinned to commit `4e3e32de9ff188bace35c9e80e433b4e795a9d76`** (`solvers/hivemind-solver/Cargo.toml` and `Cargo.lock`). The source is not copied into this repository, but a `hivemind-solver` binary built from it **links code from those crates**. Of the "original innovations" listed in the rusty-SUNDIALS NOTICE, the Solver Bee uses none; it uses only the CVODE integrator and the vector crates.

**Permission recorded at the owner's request (2026-10-28).** rusty-SUNDIALS and HiveMind have the same owner and author. At the owner's instruction (2026-10-27, confirmed 2026-10-28) rusty-SUNDIALS is the solver of the Solver Bee, and the owner asked for this notice so that its use is explicit: **HiveMind and the Solver Bee may build with, run and distribute binaries linked against the pinned rusty-SUNDIALS revision, on the conditions below.** This notice records that permission; it does not change the licence files of either project.

**Conditions kept (these are the stricter reading of the dependency's own files):**
1. **The upstream NOTICE is preserved verbatim** in `third_party/rusty-sundials/NOTICE` (it says it "must be preserved in all copies and redistributions"). Any redistribution of a `hivemind-solver` binary must include this file, `third_party/rusty-sundials/LICENSE` and `third_party/rusty-sundials/LLNL-SUNDIALS-LICENSE`.
2. **The citation the NOTICE requires** for outputs that use the software is reproduced here, in `CITATION.cff` (references), and in `solvers/hivemind-solver/README.md`:

   > Callens, X. (2026). rusty-SUNDIALS: Formally Verified Scientific Machine Learning Engine for Exascale Physics and Planetary Carbon Capture. SocrateAI Lab | SymbioticFactory Research. https://github.com/xaviercallens/rusty-SUNDIALS. Version: 9.0.1 | License: Apache-2.0 AND CC-BY-4.0

3. **Upstream attribution:** the integrator is derived from SUNDIALS, copyright Lawrence Livermore National Security and contributors, BSD-3-Clause; its licence text is kept in `third_party/rusty-sundials/LLNL-SUNDIALS-LICENSE`.
4. **No claim of formal verification is made by HiveMind for the solver.** The title of the cited work mentions Lean 4 verification of parts of that project; the Solver Bee's own checks are the ones described in `docs/solver_bee/VERIFIER.md` (agreement with an independent reference, conserved quantities, and agreement when the tolerance is tightened), and the verifier states what a VERIFIED verdict does not mean.

**Open question for the owner: the dependency's own files disagree.** Reading the pinned revision:
- its `LICENSE` and the `license` field of its workspace `Cargo.toml` say **BSD-3-Clause**;
- its `NOTICE` and the citation above say the code is distributed under **Apache-2.0 AND CC-BY-4.0**;
- its workspace `Cargo.toml` declares version **6.4.0**, while the NOTICE and the citation say **9.0.1**.

This repository does not decide which applies; that is for the owner to reconcile **in the rusty-SUNDIALS repository** (one authoritative licence, a matching version, and a single citation). Both readings are permissive licences, but **no legal review of their compatibility with HiveMind's AGPL-3.0 licence has been done**. Until it is reconciled, the conditions above follow the stricter reading.

## GWAYA zero-stub auditor

`third_party/gwaya/` holds one vendored, unmodified file (MIT, same owner); see `third_party/gwaya/PROVENANCE.md`.

## AMCP (Agent Mesh Communication Protocol) and the bee mesh

The mesh in `mesh/` builds against AMCP (https://github.com/agentmeshcommunicationprotocol/amcpcore.github.io), MIT-licensed, same owner. The Kafka transport changes it needs are proposed upstream in https://github.com/agentmeshcommunicationprotocol/amcpcore.github.io/pull/1. The other Java and Rust dependencies keep their own licences; see `mesh/pom.xml`, `solvers/hivemind-solver/Cargo.toml` and `Cargo.lock`.
