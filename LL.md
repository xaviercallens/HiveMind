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

### 9. Tooling notes for agents

- In worktree-isolated sessions, compound shell commands that touch git are refused: run plain, separate commands.
- `rm -rf` is refused; delete files by name and `rmdir` the folder.
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
