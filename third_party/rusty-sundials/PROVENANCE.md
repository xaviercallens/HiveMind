# rusty-SUNDIALS notices (verbatim copies)

This directory holds **unmodified copies of the notice and licence files** that must travel with the Solver Bee. The rusty-SUNDIALS source code itself is **not** copied into this repository: `solvers/hivemind-solver` fetches the `cvode` and `nvector` crates as a pinned git dependency at build time. See `THIRD_PARTY_NOTICES.md` at the repository root for what this means and for the open licence question.

| File (here) | Upstream | Pinned at | sha256 |
|---|---|---|---|
| `NOTICE` | https://github.com/xaviercallens/rusty-SUNDIALS, path `NOTICE` | commit `4e3e32de9ff188bace35c9e80e433b4e795a9d76` (the revision in `solvers/hivemind-solver/Cargo.toml`) | `6d90a16d9b6fafd1cd95a3ac438f39e1b1bf595bbfcc9d5bca3c85b90ba8ae33` |
| `LICENSE` | same repository, path `LICENSE` | same commit | `1372cb1f16e2bf7a0f40574dac4cd792654cf15cccc7e7223589fade3551c116` |
| `LLNL-SUNDIALS-LICENSE` | https://github.com/LLNL/sundials, path `LICENSE` | commit `312fc0f3684f27209ca9dc9249d194436eb41a7a` (tip of `main`, committed 2026-09-03) | `d83f244e1bef3d00a88d043d37d9cd2493c7aa19c9fbf35a837b310e867897a6` |

Retrieved 2026-10-28 through the GitHub contents API. The upstream NOTICE states that it "must be preserved in all copies and redistributions"; the LLNL licence is the BSD-3-Clause text of the SUNDIALS library from which the rusty-SUNDIALS CVODE port is translated, and the BSD-3-Clause conditions require its copyright notice to accompany binary redistributions.

## Modifications

None. All three files are byte-identical to upstream at the pinned commits (verify with `sha256sum`).

## Note on the pinned revision of the LLNL file

The LLNL licence was fetched at the tip of its default branch on the retrieval date because rusty-SUNDIALS does not record which LLNL release its port follows. The copyright years in that file may therefore be newer than the ones in the code the port was translated from. This is recorded rather than guessed.
