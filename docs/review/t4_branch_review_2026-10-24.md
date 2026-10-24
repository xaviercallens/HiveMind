# Scientific lead's review of the T4 experiment branch (`worktree-feat-s10-e11-e12-t4`), 2026-10-24

Reviewed at branch head `9655124` (04:28 UTC), 38 commits ahead of `main`, 14 behind, 78 files, 56,766 added lines
(40 experiment artifacts). Written before any pull request exists for the branch, from the committed artifacts only;
no number below was recomputed. The branch is owner-authored (git author `callensxavier@gmail.com`), so the hourly
gatekeeper would ACCEPT it on checks alone once it is mergeable. This memo lists what must change first.

## 1. Blocking before a PR

1. **Rebase on `main`.** `git merge-tree` reports 11 content conflicts: `CHANGELOG.md`, `ROADMAP.md`,
   `artifacts/predictions_ledger.json`, `docs/RESEARCH_PROGRAM.md`, `paper/AUTOEVOLVE_SESSION_MANIFEST.json`,
   `paper/PEER_REVIEW_REPORT.json`, `paper/autoevolve_peer_review_pipeline.py`, `paper/hivemind_foundation_article.{tex,pdf}`,
   `scripts/update_predictions_ledger.py`, `src/hivemind/swarm/diloco_nextgen.py`. The gatekeeper holds a CONFLICTING PR.
2. **The paper draft on the branch is labelled v3.7.0 and forks from before v3.7.1/v3.7.2.** `main` already carries E12-CPU
   (P12a–d, PROV-32..34), the 128-theorem inventory and the v3.7.1 Zenodo DOI. The branch's three new subsections (E10b,
   E11, E12-GPU) and its "Status of the research program" section must be re-based onto the `main` text as **v3.8.0**;
   the branch must not replace the `main` manuscript. Numbers enter the text through `PROV-*` checks only (the branch
   adds 17; keep them, renumber after PROV-34).
3. **E10b is INVALID and must be reported as such.** `E10b_lattice_05b_report.json` has `controls_all_passed: false`:
   the positive control (sign codec without error feedback must hurt) failed in seed 3 (2.9449 vs 2.9467 for the
   unquantised run). The `analysis` block still carries statuses (P10a "supported" 0.869 ± 0.003, P10b inconclusive,
   P10c supported). Per the lab method a run whose control failed decides nothing: the ledger entries for P10a/b/c at
   0.5B stay as E10-toy decided them, and the paper may cite the 0.869 ratio only as a *descriptive* value with the
   failed control stated in the same sentence. The likely cause is worth one sentence: at an outer LR where twelve
   rounds move the loss by 1e-3, the sign codec's damage is below the seed noise, so the control has no power.
   A control that cannot fail is a bug; a control that cannot *detect* is the same bug from the other side.
4. **E11 Qwen2.5-Coder-7B: the negative control failed** (`negative_A2at8_equals_fp8`: −0.47 ± 0.20 nats, `passed: false`).
   An 8-bit hexagonal lattice cannot be half a nat *better* than FP8 on a float16 model unless the FP8 arm is broken
   on that model (likely: FP8 scaling of float16 keys with a 128-dim head). The 7B verdicts (P11a refuted, P11b
   "supported on one matched cell") therefore do not count; report the model as *invalid (negative control failed)*,
   and investigate the FP8 arm before any 7B number is quoted. The 0.5B and Gemma rows stand (controls passed).
5. **P11a's 0.5B verdict rests on a single rate-matched cell** (R = 4; R = 2 and 3 not rate-matched). The committed
   wording "refuted on qwen0.5b" must carry the cell count in the same sentence, as commit `e3b64ef` already does
   in the docs; carry it into the paper and the ledger `status_detail`.

## 2. Findings that are sound and should enter v3.8.0

- **E12 on the T4** (prereg `E12_replay_divergence_prereg.json`): same T4, separate OS process, fp32 with the
  deterministic-algorithm settings: ℓ∞ = 0.0 in 9/9 segments (P12-same supported). T4 vs x86 CPU fp32: 2.0e-4 to
  6.3e-4, above ε in 9/9 (P12-cross supported). The fp64 + 2^-20 snapping proxy for fixed-point replay does **not**
  remove the cross-class divergence (cell D ≈ cell C): the fixed-point fallback of paper §4.1 is not rescued by
  precision alone; it needs an integer kernel. Tamper control 2.6–3.0 (9/9). This completes the execution-class
  picture started on CPU: *GPU same-class replay is exact; every class crossing fails ε.*
- **E13 single-GPU swarm over real localhost HTTP** (Qwen2.5-0.5B, M = 4 non-IID domains, K = 2, three seeds):
  S1 == S0 bit-identical (P13a); the all-mechanisms-on swarm retains −0.05 ± 0.35 of the single-learner improvement
  (P13b **refuted**: the full stack as configured loses the learning); upload 9.1 % of raw (P13c supported); the
  each-off cells attribute the loss to the **K < M quorum with a persistent straggler** (no-quorum retention 0.94,
  no-codec 0.11, no-RDA 0.26; 25 late discards). This is the compounding-approximations threat the external review
  raised, measured, and it is consistent with E5-het P4 (PA-CGAD under quorum) and the E8 pilot (K = 1 drops the
  slow learner): the quorum admits the fast workers and the straggled domain never enters the merge. The eager /
  deferred learner rule (`HM-S9-13`) is now the most important engineering task of the season; E13c (rotating
  straggler, registered before running) will say whether the mechanism is domain starvation or K < M itself.
- **E11 Qwen2.5-0.5B and Gemma-4-E4B**: A2 per RoPE plane does not beat entropy-coded scalar at matched measured
  bits (P11a refuted; one matched cell at 0.5B); on Gemma E8 beats Z without any transform (P11b refuted there).
  Decision applied by the branch and endorsed: the inference ring stays on FP8; lattice KV codecs are not adopted.
- **Thread-hosted peers are a validity caveat, not a flaw**: the memory preflight (4 × 4.56 GiB > 13.8 GiB free)
  forced four peer threads over the real HTTP protocol around one frozen model; only the M = 2 identity run used
  separate OS processes. State it in the E13 scope line; it does not touch P13b/P13d, which compare swarm
  configurations under the same hosting.

## 3. Method

Six E13 amendments were registered, each time-stamped before the run it affects, each stating that no registered
rule changes; amendment 5 waives the learning gate for an identity-only run (acceptable: that run asserts only
reproducibility). Granite-1B was stopped by the registered time fail-fast (2.69 s/step → 3.09 h > 3 h) and a step
microbenchmark was filed before any rerun. This is the protocol working as written.

## 4. What the scientific lead will do once the PR is open and mergeable

Gate it (checks only; authority is the owner's e-mail on every commit), then write v3.8.0: E10b as invalid, E11 with
the 7B row withdrawn, E12-GPU and E13 as the headline results, the ledger regenerated from the artifacts, and a new
Zenodo version. Until then the paper of record is v3.7.1 (10.5281/zenodo.23607597).
