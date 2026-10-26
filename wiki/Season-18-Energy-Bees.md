# 🧬 Season 18 — Operation Energy-Bees

**One sentence:** give every Bee (specialist model) a measured **energy** — how wrong and how costly its answer is, checked by running it — and let a child Bee replace its parent **only** when that energy goes down by more than noise, with a rollback.

Roadmap of record: [`ROADMAP.md` §3.16](https://github.com/xaviercallens/HiveMind/blob/main/ROADMAP.md) · Proposal: [`docs/AUTOPOIETIC_ENERGY_BEES_PROPOSAL.md`](https://github.com/xaviercallens/HiveMind/blob/main/docs/AUTOPOIETIC_ENERGY_BEES_PROPOSAL.md) · Epic: [#208](https://github.com/xaviercallens/HiveMind/issues/208) · How to contribute: [Human and Agent Workflow](Human-and-Agent-Workflow)

## Why now

Measured on 2026-10-25/26:
- The BOINC night chain (98 rounds on the T4) lowered held-out loss from 2.2705 to 2.1380 (−5.8 %), then **plateaued**, and its head was chosen best-of-97 on the same small sets it is reported on.
- Loss is not answer quality: the first real coding answer was fast and wrong; adapters trained on terse gold text interfered with other skills.

## What comes from AutoevolveAI, and what does not

[AutoevolveAI](https://github.com/xaviercallens/AutoevolveAI) tested these ideas with gates that could fail (`docs/EVOLUTION_LAB.md`):

| Imported (its gate passed) | Not imported (its gate failed) |
| :-- | :-- |
| Hidden-test energy (Phase 1: the self-graded loop claimed 11/20, hidden tests verified 7) | A learned energy predictor that skips execution (Phase 2: 40.0 % vs 44.3 % random) |
| Promotion gate with equivalence check, A/A noise margin, rollback (Phase 3: 5/5, toy components) | Full-solution lesson memory and verbose retries on small models (Phase 1 UC2/UC4) |
| Best-so-far and short error prompts for small models (Phase 4 `D1`/`D3`/`D6`) | New crossover operators (8 tested, all hit the same cliff) |
| One warm verifier process (one Mathlib import was the whole cost of a Lean check) | |

## The energy, in plain words

For an answer `y` to prompt `x`:
- **Crash, timeout, empty function, `sorry`:** energy `1 000 000` (rejected).
- **Otherwise:** `100 × (share of hidden tests failed)` + conservation/consistency residual + a **cost** for time and memory that is **always smaller than one failed test**. So a correct slow answer always beats a wrong fast one; cost only breaks ties between equally correct answers.

Code: `src/hivemind/verification/bee_energy.py` (PR [#207](https://github.com/xaviercallens/HiveMind/pull/207)), tests: `tests/test_bee_energy_autopoiesis.py`.

## Tasks (claim one)

| Task | Issue | Unblocks | Labels | Depends on |
| :-- | :-- | :-- | :-- | :-- |
| `HM-S18-AE-01` energy function in the sandbox | [#209](https://github.com/xaviercallens/HiveMind/issues/209) | #191 | agent-ready, human-review | — |
| `HM-S18-AE-02` disjoint reported split + plateau rule | [#210](https://github.com/xaviercallens/HiveMind/issues/210) | BOINC chain | agent-ready, needs-prereg | — |
| `HM-S18-AE-03` energy-gap training pairs | [#211](https://github.com/xaviercallens/HiveMind/issues/211) | #166 | agent-ready, needs-prereg | AE-01; owner approval for data |
| `HM-S18-AE-04` promotion gate + registry with rollback | [#212](https://github.com/xaviercallens/HiveMind/issues/212) | #165, #170 | agent-ready, human-review | AE-01 |
| `HM-S18-AE-05` warm verifier → bandit reward | [#213](https://github.com/xaviercallens/HiveMind/issues/213) | #173 | agent-ready | — |

Suggested order: AE-01 → AE-04 → AE-03; AE-02 and AE-05 in parallel.

## Same-day review of the first draft

| # | Finding | Fix |
| :-- | :-- | :-- |
| R1 | Four tasks duplicated open issues (#191, #166, #165/#170, #173) | S18 builds the shared pieces; each task names the issue it unblocks |
| R2 | Correct 6 s answer scored 60.5, half-wrong 5 ms answer 50.1 | cost bounded below one failed test; regression test (now 20.4 vs 51.7) |
| R3 | "Plateau rule fires at round ~32" | replayed on the real ledger: first fires at round 47; labelled descriptive |
| R4 | Fixed noise margin 0.5, McNemar unspecified | margin from ≥ 3 measured A/A runs; exact McNemar primary |
| R5 | Acceptance written by the engineering coordinator | all gates are drafts until the scientific lead pre-registers them |
| R6 | 10/10 on 10 hand-picked cases proposed as a pre-ranker | dropped |

## Run it locally

```bash
uv sync --all-extras
uv run pytest tests/test_bee_energy_autopoiesis.py -q
uv run python scripts/gwaya_stub_audit.py
```

No GPU is needed for AE-01, AE-03 and AE-04. AE-02 runs on the T4 night window and needs the owner's go-ahead.
