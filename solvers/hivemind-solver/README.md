# hivemind-solver

Solver Bee wave 1: a deterministic, data-only ODE initial-value solver command line.
JSON spec on stdin, JSON result on stdout. Nothing in a spec is ever evaluated as code.

Full documentation: [`docs/solver_bee/SOLVER_CLI.md`](../../docs/solver_bee/SOLVER_CLI.md).

```
CARGO_TARGET_DIR=/some/dir/outside/the/repo cargo build --release --locked -j2
hivemind-solver run < spec.json > result.json
hivemind-solver version
hivemind-solver families
```

Licence of this crate: AGPL-3.0 (Project HiveMind).

## Dependency licences

| Dependency | Use | Licence |
|---|---|---|
| rusty-SUNDIALS `cvode`, `nvector`, `sundials-core` (git, pinned `rev = 4e3e32de9ff188bace35c9e80e433b4e795a9d76`) | CVODE BDF/Adams solver; **not vendored**, fetched by cargo | BSD-3-Clause (its LICENSE file and Cargo.toml). Copyright (c) 2026 Xavier Callens / SocrateAI Lab. Its NOTICE file additionally asks for a citation and, inconsistently, names Apache-2.0 AND CC-BY-4.0; see the open point in SOLVER_CLI.md |
| `serde`, `serde_json` | spec parsing, result writing | MIT OR Apache-2.0 |
| `sha2` | input hash | MIT OR Apache-2.0 |
| `rayon` (transitive, via `nvector`) | not used by this crate (serial vectors only) | MIT OR Apache-2.0 |

Citation requested by the rusty-SUNDIALS NOTICE:
Callens, X. (2026). rusty-SUNDIALS: Formally Verified Scientific Machine Learning Engine for Exascale Physics and
Planetary Carbon Capture. SocrateAI Lab | SymbioticFactory Research. https://github.com/xaviercallens/rusty-SUNDIALS
Version: 9.0.1 | License: Apache-2.0 AND CC-BY-4.0

The dependency's NOTICE and LICENSE, and LLNL's SUNDIALS licence, are kept verbatim in `third_party/rusty-sundials/` (see its
`PROVENANCE.md`) and must accompany any redistributed binary. The source itself is not copied. What the notice requires, the
owner's recorded permission and the open licence question are in `THIRD_PARTY_NOTICES.md` at the repository root.
