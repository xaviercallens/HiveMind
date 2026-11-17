# S21: retrain on more datasets (dataset registry, decontamination, mixture caps)

Scope: give every bee domain 2-4 more real training sources than the single source each NIGHT-MOE-1 pack uses, without letting
in unlicensed data, eval contamination, or a flood of unverified LLM-generated text. Nothing is trained here.

## Files
| File | Role |
|---|---|
| `src/hivemind/data/dataset_registry.py` | Library: licence gate, Hub lookup, slice streaming, mapping to the night-pack format, decontamination, mixture planner, provenance manifest |
| `configs/datasets_s21_candidates.json` | Hand-curated input: dataset ids, where to read them, how to map them, and the curator's verified-signal judgement. No licence, size or revision is typed here |
| `configs/datasets_s21.json` | Generated output of the script below (licence, revision sha, sizes, retrieval time, slice checks, decontamination counts). Do not edit by hand |
| `scripts/refresh_dataset_registry.py` | Queries the Hub API, streams a slice of each accepted source, runs decontamination and the mixture planner on the slices |
| `tests/test_dataset_registry.py` | 80 tests (Hypothesis for the pure functions); every check has a negative control |

## Design
1. **Licence gate.** `https://huggingface.co/api/datasets/<id>` gives `cardData.license`, `sha`, `lastModified`, `usedStorage`,
   `gated`. The licence is checked against an allowlist of licences that let derived artefacts ship under the AGPL-3.0
   (permissive licences, CC-BY, ODC-BY, CDLA-Permissive, CC-BY-SA, GPL family). Missing licence, `other`/`unknown`, NC and ND
   licences, gated and disabled datasets are recorded as `REJECTED` with the reason, never dropped. A list of licences is accepted
   only if every entry is allowed. The allowlist is printed into the generated config (`policy.license_allowlist`).
2. **Slice check.** For accepted sources the script reads the split size from datasets-server `/size` and streams the first
   200-500 rows through `/rows` (100 per page, retries on 429/5xx with backoff; 404 is not retried). Rows are mapped to the
   night-pack train format `{key, messages:[system,user], completion}` with one of four mappers: `fields`, `messages`, `mcq`,
   `label` (tabular features + label map, missing-value sentinels, leakage columns left out). The check separates
   *filtered by design* (language filter, label outside the map, excluded dual-use text) from *unmappable* (schema problem):
   `PASSED` needs at least one mapped row and at most 10% unmappable among the unfiltered rows.
3. **Decontamination** (`EvalIndex`, `decontaminate`). Held-out items come from the night packs' `eval.jsonl` (user turns and
   keys, as written by `scripts/night_pack_build.py`) plus reference eval sets listed in the candidates file (HumanEval, MBPP test,
   ARC-Challenge test). Text is NFKC-normalised, case-folded and split into alphanumeric tokens. A training row is removed, first
   hit wins, when (a) its entity key (KOI name normalised to `K#####.##`, or `star-<hostname>`) is an eval key, or the KOI
   shares its host star with an eval KOI (`K00752.02` vs eval `K00752.01`: the night pack splits whole stars), (b) its prompt
   hash equals an eval prompt hash, (c) it shares any word 13-gram with an eval item, or (d) its prompt+completion hash equals a
   row already kept (verified sources are processed first, so a cross-source duplicate survives in its verified copy).
   13-grams present in more than 10 eval items are treated as shared scaffolding and ignored; the count is reported.
   A key column that is not unique within a slice (nuhmanpk's `prompt_id` is a task type: 255 rows, 6 values) gets the row's
   split position appended (`rows_rekeyed_duplicate_key`), so keys stay unique and sampling stays order-independent.
4. **Mixture** (`plan_mixture`, `build_mixture`). Per domain: target = weight share x total; each source capped at
   `per_source_cap` x target; verified rows first, then unverified rows up to `floor(cap/(1-cap) x verified)` with
   `unverified_cap = 0.30`. A domain with no verified rows gets none and is `BLOCKED`; a domain that cannot reach its target is
   `SHORT` with the reason. Sampling ranks rows by `sha256(seed:source:key)`, so it is deterministic and independent of read
   order. The manifest holds the plan, per-source removal counts, the sha256 of every shard read and the sha256 of the mixture.

## Real run (2026-10-28)
Command: `python scripts/night_pack_build.py --out $S/night_packs` (all six packs BUILT: maths 200, exoplanet 600, stars 317,
cyber 500, physics_transfer 488, astro_transfer 152 eval rows), then
`python scripts/refresh_dataset_registry.py --work $S/s21_work2 --eval-packs $S/night_packs` with `$S` a scratch directory
outside the repo. After review fixes it was re-run (`--work $S/s21_work3`, same cached `/rows` pages, Hub metadata re-queried,
generated_utc 2026-10-28T08:20:58Z): licences, revision shas, slice counts, decontamination counts, plan and mixture sha256
unchanged; `eval_index_complete` is true. Hub metadata retrieved 2026-10-28 (UTC timestamps per entry in `configs/datasets_s21.json`).

Result: 24 candidates, 18 ACCEPTED, 6 REJECTED, 17 slices PASSED, 1 slice BLOCKED.

| Domain | Dataset (config) | Licence (card) | Status | Split rows | Slice read / mapped / filtered / unmappable | Removed: entity / eval-exact / 13-gram / dup |
|---|---|---|---|---|---|---|
| python_coding | bigcode/self-oss-instruct-sc2-exec-filter-50k | odc-by | ACCEPTED | 50661 | 300 / 300 / 0 / 0 | 0 / 0 / 2 / 0 |
| python_coding | nvidia/OpenCodeInstruct (train) | cc-by-4.0 | ACCEPTED | 1400000 | 300 / 136 / 164 / 0 | 0 / 0 / 3 / 0 |
| python_coding | ise-uiuc/Magicoder-OSS-Instruct-75K | mit | ACCEPTED | 75197 | 300 / 153 / 147 / 0 | 0 / 0 / 0 / 0 |
| python_coding | m-a-p/CodeFeedback-Filtered-Instruction | apache-2.0 | ACCEPTED | 156526 | 300 / 205 / 95 / 0 | 0 / 0 / 12 / 0 |
| python_coding | iamtarun/python_code_instructions_18k_alpaca | none | REJECTED: no licence in the card | - | - | - |
| maths | nvidia/OpenMathInstruct-2 | cc-by-4.0 | ACCEPTED | 13972791 | 500 / 105 / 395 / 0 | 0 / 0 / 0 / 0 |
| maths | meta-math/MetaMathQA | mit | ACCEPTED | 395000 | 300 / 300 / 0 / 0 | 0 / 0 / 0 / 0 |
| maths | TIGER-Lab/MathInstruct | mit | ACCEPTED | 262039 | 300 / 300 / 0 / 0 | 0 / 0 / 0 / 1 |
| maths | AI-MO/NuminaMath-CoT | apache-2.0 | ACCEPTED, slice BLOCKED (datasets-server /rows 404) | - | - | - |
| physics | allenai/ai2_arc (ARC-Challenge) | cc-by-sa-4.0 | ACCEPTED | 1119 | 300 / 300 / 0 / 0 | 0 / 0 / 1 / 0 |
| physics | allenai/ai2_arc (ARC-Easy) | cc-by-sa-4.0 | ACCEPTED | 2251 | 300 / 300 / 0 / 0 | 0 / 0 / 0 / 0 |
| physics | maximus-21/PhysicsQA | apache-2.0 | ACCEPTED | 370 | 300 / 280 / 0 / 20 | 0 / 0 / 0 / 12 |
| physics | camel-ai/physics | cc-by-nc-4.0 | REJECTED: non-commercial | - | - | - |
| physics | allenai/sciq | cc-by-nc-3.0 | REJECTED: non-commercial | - | - | - |
| exoplanet | bingbangboom/exoplanet-transit-detection | cc-by-4.0 | ACCEPTED | 18853 | 200 / 59 / 139 / 2 | 6 / 0 / 2 / 0 |
| exoplanet | juliensimon/nasa-exoplanets | cc-by-4.0 | ACCEPTED | 6451 | 300 / 299 / 1 / 0 | 15 / 0 / 0 / 0 |
| exoplanet | hieuvu21390/exoplanet | none | REJECTED: no licence in the card | - | - | - |
| stars | OneAstronomy/sspp-dr9-stellar-spectra | cc-by-4.0 | ACCEPTED | 186409 | 200 / 166 / 34 / 0 | 0 / 0 / 0 / 0 |
| stars | jeff-vincent/astronomy-textbook-qa-context | mit | ACCEPTED | 678 | 300 / 300 / 0 / 0 | 0 / 0 / 0 / 0 |
| stars | christianschwarz/deep-multimodal-...-stellar-spectra | cc-by-nc-sa-4.0 | REJECTED: non-commercial | - | - | - |
| cyber | CyberNative/Code_Vulnerability_Security_DPO | apache-2.0 | ACCEPTED | 4656 | 300 / 297 / 3 / 0 | 0 / 0 / 0 / 1 |
| cyber | nuhmanpk/cybersecurity-controls-instructions | cc-by-4.0 | ACCEPTED | 13106 | 300 / 255 / 45 / 0 | 0 / 0 / 0 / 0 |
| cyber | Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset | apache-2.0 | ACCEPTED | 53201 | 300 / 114 / 186 / 0 | 0 / 0 / 0 / 0 |
| cyber | Vanessasml/cybersecurity_32k_instruction_input_output | none | REJECTED: no licence in the card | - | - | - |

Eval index: 4076 eval items hashed (six night packs + HumanEval 164 + MBPP test 500 + ARC-Challenge test 1172), 2830 entity
ids (2257 eval keys plus 573 KOI host-star aliases; the host rule removed no further slice row), 0 n-grams ignored as scaffolding. Mixture planner on the slices (600-row target, weights 0.30/0.20/0.15/0.10/0.10/0.15):
457 rows, unverified fraction 0.081. python_coding, physics and exoplanet BUILT with verified rows only; maths SHORT (60
verified + 25 unverified of 120: the verified OpenMathInstruct-2 slice had 105 rows, capped at 60); stars SHORT (30 + 12 of 60);
**cyber BLOCKED**: all three accepted cyber sources are LLM-generated, so the 30% cap admits none of their rows.

## What the numbers do and do not show
- They are slice counts (the first 200-500 rows of each split), not full-dataset rates. Contamination rates on a full split can
  differ; the full run needs the same script with larger slices or the parquet shards.
- The 13-gram rule on code flags shared algorithm idioms. Spot check of the 12 CodeFeedback removals: the matched 13-grams are
  bubble-sort and prime-test loops found in one MBPP item each, e.g. `for i in range 3 int math sqrt n 1 2 while n`. These are
  conservative removals (false positives cost rows, not leakage).
- The entity-key rule did real work: 6 bingbangboom KOIs and 15 juliensimon host stars are eval items of the night packs, which
  the text rules alone would not have caught (different prompt wording).
- `verified_signal` is a curator's judgement recorded with its basis (`signal_basis`). Execution-filtered code counts as verified
  even though its tests are model-written; SSPP spectral types are pipeline labels. Change the judgement in the candidates file,
  not in code.
- The licence is the one the dataset card declares. It does not settle upstream terms (several sources contain model outputs
  whose generator terms may restrict use); those need the owner's review before a training run. CC-BY-SA sources (ARC) keep
  share-alike for redistributed data.
- datasets-server serves the current parquet conversion, so a slice is not pinned to the recorded `revision_sha`; the manifest
  says so per source. Shard sha256s identify the bytes actually read.

## Interface for the other S21 PRs
- Orchestrator dream phase (`campaign_orchestrator.py`): consume `build_mixture(...)` output: JSONL rows in the night-pack train
  format plus the manifest; refuse a domain whose plan status is `BLOCKED`. Wiring is a follow-up.
- Base-model catalogue / promotion gate: the decontamination index (`EvalIndex`) can be reused to prove a promotion eval set is
  disjoint from the training mixture.
- BYOB: a submitted bee's declared training data can be put through `registry_entry` + `classify_license`.
