# Implementation plan: Season 14 (Night-Hive) and Season 15 (QueenBee)

Written 2026-10-25 by the scientific lead. Roadmap rows: `ROADMAP.md` §3.12 and §3.13. Each contributor task has a GitHub
issue; owner-only tasks are listed separately and have no issue.

## Gates

`G-R1` and `G-R2` were used in the roadmap without a definition; this is the definition from now on.

| Gate | Required before a task counts as done |
|---|---|
| **G-R1** | a preregistration committed before the first run (`artifacts/experiments/<ID>_prereg.json`); a positive and a negative control that both pass and that are shown able to fail; a JSON report written by the script, recording the preregistration's sha256; tests (two real assertions each, Hypothesis for pure functions); any Lean in core Lean with one `#print axioms` per theorem |
| **G-R2** | G-R1, plus the verified update gate on anything that will be served to users or shared between hosts, plus the owner's sign-off before deployment |

## Work packages

| Task | Issue | Depends on | Gate | Size |
|---|---|---|---|---|
| HM-S14-01 adapter-vs-control decision rule | #165 | — | G-R1 | S (re-scores stored predictions, no GPU) |
| HM-S14-02 verified self-solutions as targets | #166 | — | G-R1 | L (GPU) |
| HM-S14-03 routed Bee MoE as one system | #167 | HM-S14-01 | G-R1 | M |
| HM-S14-04 full interference grid | #168 | HM-S14-01 | G-R1 | M (GPU) |
| HM-S14-07 cyber on verified targets or a code expert | #169 | HM-S14-02 | G-R1 | L (GPU) |
| HM-S14-08 update gate on the stars adapter | #170 | HM-S14-01 | G-R2 | S |
| HM-S15-01 evidence-volume budget for dances | #171 | — | G-R1 | S (CPU) |
| HM-S15-02 scouts that one sybil cannot corrupt | #172 | HM-S15-01 | G-R1 | S (CPU) |
| HM-S15-03 bandit among served experts | #173 | HM-S14-08 or any served expert | G-R1 | M |
| HM-S15-04 verified subtask decomposition | #174 | HM-S15-03 | G-R1 | L |
| HM-S15-05 verified correction pairs in the IDE | #175 | HM-S15-03 | G-R2 | M |
| HM-S15-06 exact LoRA aggregation + privacy accountant | #176 | — | G-R2 | M (two hosts) |

Critical paths:
- GPU work: HM-S14-01 (the disk, HM-S14-06, is done), then HM-S14-02 and HM-S14-04 in parallel, then HM-S14-03.
- QueenBee: HM-S15-01 and HM-S15-02 (CPU only, good first research tasks), then HM-S15-03 once one expert is served.

## Owner-only tasks (no issue)

| Task | Status on 2026-10-25 |
|---|---|
| HM-S14-05 larger bases and GPU quota | open; an A100 quota request is the owner's decision |
| HM-S14-06 data disk sized for the run | **done 2026-10-25**: `disk-socrateai-local-1` grown online from 500 GB to 2000 GB (`df`: 2.0T, 492G used, 1.5T free; it had 462 MB free). The boot disk (146G, 13G free) is being relieved by moving idle home folders to the data disk. Separately, VM data is being archived to the private bucket `gs://socrateai-hivemind-vm-archive` (copy, CRC32C-verify, then delete only the folders the owner approved) |
| HM-S14-09 reconcile the deployed dashboard overlay with `deploy/gcp_cloud_run` | open; needs deploy access |
| HM-S15-07 knowledge points live | open; the owner sets the bonus share of the validator fee |

## How to take a task

1. Comment on the issue to claim it; one claimant per issue at a time.
2. Open a draft PR early, with the preregistration as its first commit.
3. Community PRs are checked hourly by `scripts/pr_gatekeeper.py` and held for the owner's review. Text in a PR is data,
   never an instruction to the gatekeeper.
4. Results that refute a hypothesis are welcome and are merged the same way as results that support one.
