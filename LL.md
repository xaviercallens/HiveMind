# Lessons learned (LL)

Short, measured lessons from HiveMind operations. Each entry says what happened, the evidence, and the rule we now follow.
Newest first. Add an entry when something costs time, breaks a rule, or misleads a reader.

## LL-2026-11 — CrowdGPT × HiveMind weekend (14–15 November 2026)

Context: a joint training weekend with CrowdGPT (Vxtzq) on the dense 1 B model Crowd-v1. HiveMind offered its Tesla T4
(GCP Spot), its gateway (`hivemind/crowd-v1` served) and its volunteers. Release v3.4.2 carries the fixes below.

### 1. GPU memory: a "budget" is not a cap, and pausing does not free memory

- **What happened.** The T4 must keep at least 4,300 MiB free for inference (AGENTS.md). Running CrowdGPT's client next to
  `llama-server` (4.8 GB, always on) and a producer that loads ~3.6–3.9 GB on demand broke that floor four times:
  - 16:02 UTC: 1,061 MiB free. A paused client still held ~4.9 GB when the producer loaded a model.
  - 17:15 UTC: 3,421 MiB free. At round start the server-assigned fp32 cohort used ~4 GB despite `--vram-gb 1`, and the
    finished round's 948 MB upload was lost when the watchdog stopped the client.
  - 10:38 UTC (MoE sweep): 3,925 and 4,061 MiB free, in two runs.
- **What worked.** A watchdog that reads free memory every 5 s and stops (not pauses) the client after 5 s below the floor.
  No inference service was OOM-killed.
- **Rules.**
  - Treat any client's memory flag as a hint. Only a measured free-memory watchdog, or a hard per-process cap
    (`torch.cuda.set_per_process_memory_fraction`) in our own workers, protects the floor.
  - A paused process keeps its VRAM: when memory is short, **stop** the process, then relaunch with backoff.
  - Do not co-train on a GPU that also serves on-demand model loads. Parked as `PARK_T4_COTRAINING_WITH_INFERENCE`
    (`docs/roadmap/PARKED.md`, receipt `artifacts/experiments/CROWDGPT_T4_COTRAIN_2026-11-14.json`).
  - Before a breach, budget it: `free − inference peak − 4,300 MiB` must exceed the trainer's measured peak.

### 2. Ask the model's author before changing its architecture

- **What happened.** We prepared an 8-expert Sparse-MoE upcycle of Crowd-v1. Its author said Crowd-v1 is dense and should
  not be split. Our runs also showed no benefit: 1 of 7 rounds accepted, for a 0.0012 eval-loss gain.
- **Rule.** For a partner's model, confirm the architecture plan with its author first, then measure on real data before
  announcing it. Parked as `PARK_CROWD_V1_MOE_UPCYCLE`.

### 3. Public pages and agent-readable JSON drift away from the truth

- **What happened.** On the weekend, the live `/raid` page promised "500 HMT welcome grants" while the gateway's starter
  grant was 0 (and `/signup` said 0.00 HMT). It still offered the stopped MoE split and its command. The public
  `/v1/hivemind/community/v34-outreach` JSON, read by agents, did the same.
- **Rules.** Render amounts from the live configuration (`starter_grant_hmt`), never from hard-coded text. When a plan is
  stopped, update every page and JSON that announces it the same day, and add a test that the stopped claim is gone.
  Fixed in PR #917.

### 4. "Measured" must mean measured

- **What happened.** Three receipts overstated or mislabelled reality:
  - the egress receipt called modeled numbers (payload size × assumed cadence) "measured" (PR #918);
  - the accounts receipt was all zeros because it ran where no database exists (PR #919);
  - on the real database, "88 accounts / 64 active nodes" were 8 accounts active in 7 days and about 3 machines. Endpoint
    rows multiply behind rotating tunnels, and seeded accounts look like users (PR #919).
- **Rules.** Every receipt row states its basis (`measured`, `modeled`, `client_log_reported`). Count machines by hardware
  class and accounts by evidence of use (requests, password, heartbeat). Quote these numbers in posts and the paper.

### 5. Documentation can promise a safety check the code does not run

- **What happened.** The OpenTrainer orchestrator's docstring promised Multi-Krum, but the round close averaged every
  accepted delta, so one learner under the L2 cap could steer the model (PR #923). The delta L2 cap also had three values
  (25, 50, 100); it is being unified at 25 (PR #924).
- **Rule.** Each safety promise in a docstring or doc needs a test that fails without it (for example, a planted poisoned
  delta must be excluded).

### 6. A decision to stop must be enforced, not only announced

- **What happened.** Three minutes after the owner's decision to stop T4 contributions, the launcher was run again by an
  unknown caller (no SSH session, no cron; probably another agent session). That run submitted one more round.
- **Rules.** Encode stop decisions in a file the launcher refuses on (`crowdgpt_client/T4_CONTRIBUTIONS_STOPPED`, exit 3).
  Agent sessions sharing a machine announce what they start and stop, claim shared work before doing it, and match other
  processes by command line **and** working directory.

### 7. Getting volunteers: the link and the first step matter more than the post

- **What happened.** Reddit brought about three quarters of identified referrals; each post's traffic was mostly gone within
  a week. About 1,000 unique clones in 14 days produced no outside comment, issue or PR. Links pasted with a trailing ")"
  or "]" returned 404 on `/signup`.
- **Rules.** Put each URL on its own line. The gateway now redirects these typos (PR #917). Lead with a three-command
  start, state the cost honestly (no rewards, ~0.9 GB upload per round, ~1.5 GB/hour), and follow up in the same
  thread rather than relying on one post. The join page now lists each role's upload budget (PR #918).

### 8. Cost: check that scheduled machines really stop

- **What happened.** The T4 VM rebooted at 22:18 Paris on 14 November, just after its 22:00 schedule stop, and ran idle all
  night.
- **Rule.** After a scheduled stop, confirm the instance state; a schedule does not stop an instance that something
  restarted. Monthly target: 150 USD.

### 10. Other projects' jobs share the GPU (night of 15–16 November)

- **What happened.** An owner-authorized expert (MoE) night run had a hard memory cap of 5,091 MiB and stayed at
  ≥ 5,161 MiB free for 250 rounds. At 03:02 Paris free VRAM fell to **365 MiB**. The cause was a cron job from another of
  the owner's projects on the same VM (`0 2 * * * AutoevolveAI/scripts/run_cron_dream.sh`): it loads an 842 MB model and
  trains a LoRA on the T4, peaking at about 6.6 GB. Next to llama-server alone it would leave about 3.5 GB free, below the
  floor. The watchdog stopped the HiveMind worker; inference survived.
- **Rules.** Before a night run, list every GPU job on the machine (`crontab -l`, systemd timers, other agent sessions),
  not only HiveMind's. Every GPU job on the T4, from any project, must check free VRAM against the floor before it
  starts.

### 11. Write results as you go, and compare with the starting point

- **What happened.** That night's worker printed its report only at the end, so the stop lost every round's result
  (fixed: `<work_dir>/rounds.jsonl`, PR #926). Reconstructing from checkpoints, I first said no round changed the model
  because all 250 checkpoints were identical. That was wrong. Compared with the **starting** state, round 1 had changed
  it, reproducing byte for byte the one accepted round of 14 November.
- **Rules.** Flush one result line per round. When judging whether training did anything, compare against the start
  state (or a rejected run), not only consecutive checkpoints.

### 12. Small bugs in the data path cost more than bad hyper-parameters (16 November)

- **What happened.** The expert (MoE) training was stuck for two runs. The cause wasn't the learning rate but three
  quiet bugs:
  - every round trained on the same first `steps × 64` tokens (the window never moved with the round);
  - the evaluation measured only the first of 7 experts;
  - "more data" was never read, because of a 600-row cap per source.
- **What changed once fixed (PR #926).** Held-out loss over all 7 experts fell from 5.385 to 5.271 (−2.12 %), with 26 of
  75 rounds accepted. Before, only 1 round in ~257 had been accepted, always to the same state.
- **Rules.**
  - Log what each round actually trained on (`data_offsets`) and check that it moves.
  - Make the evaluation cover every part of the model you claim to improve.
  - When adding data, check how many tokens the run actually consumed. Here it was ~154 k of 1.66 M available.

### 13. One bad step must not end a run; a recipe does not transfer by default

- **What happened.** Round 50 raised `router_logits must be finite` (fp16 on a T4, where bf16 isn't available) and ended
  a 49-round run. Separately, the JEPA study's recipe (lr 2e-5, taken from a paper on fully trained models) left
  Crowd-v1, an early checkpoint, at 0 % exact match: the answer loss went 8.17 → ~3.0, and the predictions start right
  but never stop.
- **Rules.**
  - A non-finite step rejects that round, restores the model and moves past the data window (PR #926). Only other
    errors stop a run.
  - Before a pre-registered comparison on a new model, check that the baseline can clear the floor. Calibrate the
    learning rate on the baseline arm only, by training loss, never by test accuracy.
  - If a frozen study turns out uninformative, don't re-score it: pre-register a separate follow-up before seeing more
    data (E-STP-1b, committed before any E-STP-2 result).

### 14. Read the failing step, not the first log you are shown

- **What happened.** For PR #927, `gh run view --log-failed` showed the Lean audit step. I called the failure flaky and
  re-ran it. It failed again. The job API showed the failing step was the test suite: 1 real failure in 6,005
  (`test_no_lateral_movement`: the new `--train` mode fetches data from `datasets-server.huggingface.co`).
- **Rules.**
  - Use `gh api …/actions/runs/<id>/jobs` to find the failed **step**, then read that job's full log.
  - Don't re-run CI before the failure is understood.
  - A safety test that flags new egress is a policy question for the owner. Don't hide the URL from the scanner.

### 15. A red `main` makes every PR red; a CI re-run tests the old merge (17 November)

- **What happened.** After #937 made criterion E1 pass, nobody regenerated `docs/MVP_SCORECARD.md`. The scorecard
  test then failed on `main` and on every open PR (#927, #935). Re-running #935's failed job failed again, because a
  re-run tests the merge commit built at the first run, against the old `main`.
- **Rules.**
  - When every open PR fails the same test, check `main` first: fix it there once (PR #939), then update each
    branch.
  - After `main` changes, use `gh pr update-branch <n>` (or merge `main` in). A plain re-run won't pick up the fix.
  - Any PR that changes an MVP receipt runs `python scripts/mvp_scorecard.py --write` in the same PR.
  - Dependabot PRs touching workflows or protected dependency files are blocked by the owner-only gate
    (OWN-01/CB-04) on purpose. Leave them to the owner.

### 16. Re-measure a stop rule after you change what a round does (18 November)

- **What happened.** The IP-293 brief asked to stop a MoE run after 3 rejected rounds by default. It was written
  when a rejected round replayed the same tokens and could never succeed. After the data-cursor fix, every round
  reads new data, and rejected streaks often end in an improvement: in run `104052Z`, 3 of 23 accepted rounds came
  after 3 or more rejections, and the longest streak before an acceptance was 6. On 18 November, round 8 (a new best)
  came right after 4 rejections. A default of 3 would have stopped run 2 near round 19 and lost 12 of its 23
  accepted improvements.
- **Rules.**
  - Set stop thresholds from measured streak lengths of the current recipe, with headroom (default 8 suggested to
    PR #941).
  - When a brief's premise is fixed by a later PR, update the brief, so contributors don't implement the old
    assumption.

### 17. Check which data a run actually reads, not only how much (18 November)

- **What happened.** Each learner's MoE shard lays its experts' data out one after another, and the data cursor
  reads it in order. For three days, about 262 k tokens per learner, learner 0 trained only on general web text and
  learner 1 only on code. Maths and physics data were first read at run 4 round 107; the all-expert eval then fell
  5.19 → 5.09 in five rounds. Two evaluations that day (#944 at the row start, then mid-row) had already shown the
  pattern: code −22 %, maths and exoplanets slightly worse.
- **Rules.**
  - A run's receipt must say which experts' tokens were read so far: the manifest now records `shard_layout` and
    `expert_first_offset` (PR #946).
  - Prefer `--interleave-block-tokens 1024` for new MoE campaigns. Don't change the layout in the middle of a
    campaign, because data offsets would change meaning.
  - When a loss falls suddenly, check the data cursor before you credit the recipe.
  - Evaluate a model on content as well as format. Scoring from the start of a row mostly measures the template;
    score mid-row too.

### 9. Tooling notes for agents

- In worktree-isolated sessions, compound shell commands that touch git are refused: run plain, separate commands.
- `rm -rf` is refused; delete files by name and `rmdir` the folder.
- The PR security scanner reads any roadmap ID of the form `HM-S…`/`HM-M…`, or "fixes #N", in a PR's title or body
  as a claim (CB-02). A docs-only PR that cites one is blocked, so describe such PRs without those IDs.
- A check that failed and later passed on the same commit stays in `statusCheckRollup`. Judge each check by its
  latest run.
- `docs/roadmap/planetary_streams.json` keeps one entry per line: edit it as text. `json.dump` rewrites ~700 lines.
- Restoring the gateway database for counts: `litestream restore` into a private scratch folder, read aggregates only,
  delete the copy at once.

## Weekend numbers (T4, 14 November 2026)

| | Value |
|---|---|
| CrowdGPT rounds submitted by the T4 | 5 (895–949 MB each, ~36 min apart) |
| Floor breaches stopped by the watchdog | 2 during CrowdGPT runs, 2 during the MoE sweep |
| MoE upcycle rounds accepted | 1 of 7 (−0.0012 eval loss) |
| Outside contributors who posted for help | 0 |
| T4 contributions stopped | 2026-11-14 19:31 UTC (owner decision); all T4 training stopped 2026-11-15 ~19:45 Paris |
