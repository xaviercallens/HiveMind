# T4 audit: MoE inference by day, QLoRA by night, GPU federation (2026-10-31)

*Read-only audit of the live node and of `origin/main` (214b8d4), plus a policy-neutral change set on branch `audit/t4-governor-telemetry`.
Nothing under `/opt/hivemind/t4` was touched and the `chattr +i` flags were not removed. Numbers are from `hivemind_telemetry/governor.jsonl`
and `t4_node.jsonl` unless stated; "per subagent read" marks statements taken from a delegated read of the repo that were not re-checked line by line.*

## 1. Summary

| Roadmap goal | State | Evidence |
|---|---|---|
| Inference has priority by day, QLoRA trains at night | **Built and running** (`t4_governor.py`, owner policy 2026-10-26) | 183 yields, 191 resumes, policy args match `expected_governor_args.txt` |
| Preemption < 1.0 s, p99 < 450 ms (HM-S19-Q35-03) | **Missed**: 1.12-1.15 s (n = 4); no p99 possible | §3.2; `src/hivemind/core/preemption.py` does not exist |
| Nocturnal QLoRA actually training | **Marginal, and capped by the governor itself**: 96 % of bursts are empty because the project's own validator replay is read as a foreign GPU user and locks out the next round for 30 min (observed live, §3.1); training-enabled time fell from 164-286 min/day (25-28 Oct) to 20-35 min/day (29-31 Oct) | §3.1 |
| MoE served on the T4 by day | One MoE measured (gpt-oss:20b, 12.3 GiB, 19-45 tok/s); the live mix is dense (Bee 4.8 GB + 7B coder) | per subagent read: `deploy/t4_node/README.md` |
| Daytime micro-batch harvesting (30 s idle, R:386) | **Contradicts the owner's policy** (7200 s by day); not built | `POLICY_READ_ME.txt` |
| Large-GPU federation (T4 + RTX 2070 + volunteers) | **Not built**: no sharding or pooling code, no `tensorsharp`, no `swarm_fabric.py`; Kafka/AMCP and expert registry are in-memory or simulated | per subagent read; `grep` found nothing |

## 2. What I did not change (owner decisions)

The governor's thresholds (1800 s quiet at night, 7200 s by day, 4300 MiB free, night window 22:00-07:00 `Europe/Paris`) are the owner's decision of
2026-10-26 and change only through a reviewed repo change plus `sudo bash deploy/t4_node/install_node.sh`. The "intense-moe" 5-10 s mode is not wanted.
Stopping T4 serving at night, the A100 quota request, HMT grant policies and automatic merges are also owner decisions (ROADMAP.md:425, 442, 285, 251, 258, 36;
per subagent read). This audit changes none of them.

## 3. Findings

### 3.1 The training duty cycle is set by the quiet rule, and most bursts end empty

`scripts/t4_governor_report.py` over the whole log (191 resumes, 183 yields):

- **Yield triggers**: gpu 171, nginx 11, unknown 1. The nginx in-flight signal explains 6 % of yields; almost every yield is another process using the card.
- **Empty bursts**: 175 of 183 bursts ended with no BOINC GPU task running (`had_boinc_gpu_task` false). Median burst 134 s.
- **Time with GPU mode `always` per UTC day**: 25 Oct 164 min, 26 Oct 186, 27 Oct 213, 28 Oct 286, **29 Oct 35, 30 Oct 32, 31 Oct 20 (so far)**. The 25-28 Oct values include the
  superseded experimental thresholds (5 s, 3800 MiB) and the first days of the campaign, so the drop is not a clean before/after of the owner policy; it is a fact to explain, not a conclusion.
- **Root cause of most empty bursts, observed live (one cycle, 2026-10-31, a 1 Hz sampler of GPU processes and the governor state)**: the governor resumed at about 21:56Z; the QLoRA task
  (`python -W ignore -m hivemind.boinc.real_client.moe_app`, PID 774811, 3.0-3.7 GiB, 37-61 % SM) ran 21:56:21-21:58:41Z. At 21:59:17Z, 36 s later, **the BOINC server script**
  (`... -m hivemind.boinc.real_client.boinc_server_scripts`, PID 779153, 2.5-3.7 GiB, 5-37 % SM, the validator's replay of the finished round) used the GPU for about 20 s. At 21:59:19Z the governor yielded with trigger `gpu`
  and restarted its quiet period (`quiet_s` back to 0). The governor only recognises `moe_app` as its own GPU process (`boinc_gpu_pids`), so the project's validator counts as "another user". Each round therefore locks out the next:
  one round per about 32 min at night (the work generator's cadence), 175 of 183 bursts with no task running when they ended. This is one observed cycle plus the consistent history, not a proof for every one of the 183 yields:
  171 of the 183 yields were `gpu`-triggered, 11 nginx-triggered (all 8 yields that stopped a real task were nginx-triggered, i.e. a real request) and 1 has no trigger field. I have not identified the process behind each of the 171; the live sample identifies it for one.
- **Structural ceiling at the owner's policy as the governor behaves today**: the night window has 9 h; with a 30 min quiet period before each burst and rounds of 3-4 min (`t4_workgen.py --inner-steps 80`), at most about 18 bursts x 3-4 min = 1 h of
  training per night, 6-12 % of the window. With the validator no longer counted as foreign (§5, item 5) the quiet period would be required only once per night-time idle stretch, and rounds could follow each other until a real user touches the GPU.
- **Also seen in the same sample**: at 22:03:20-22:03:22Z an Ollama `llama-server` loaded a model (a user or producer request), so the next night-time quiet period would have restarted anyway.

### 3.2 Preemption latency

Governor-ended yields that stopped a real task (`had_boinc_gpu_task` and `ended_by_governor`): **1122, 1133, 1135, 1148 ms** (n = 4). Before the governor ended the task itself: 766, 7129, 15929, 23406 ms (n = 4).
The new latency is `--yield-grace-s` (default 1.0 s) plus about 0.12 s. With n = 4 no p99 can be stated; the roadmap's p99 < 450 ms is unverifiable until at least 20 real yields exist.
`terminate_tree` is SIGTERM then SIGKILL; the QLoRA app checkpoints per segment with an atomic `os.replace` (per subagent read, `moe_app.py:105-133`), so a kill loses at most the current segment whatever the grace. Suspected, not verified: `torch.save` is not fsynced, so a power loss could lose the last segment.

### 3.3 Observations not acted on

- **(Updated 2026-11-08: `main` now fails closed here, with a `governor_step` pure function and `should_resume_training`; the deployed node does not have them yet.) `other_gpu_busy` fails open**: if `nvidia-smi pmon` raises while a BOINC task is running it returns not-busy (`t4_governor.py`, `except Exception: return False, 0`). I found no evidence it has ever failed here, and failing closed would silently stop training whenever the tool misbehaves, so I left it. A counter of pmon failures in the snapshot would settle it.
- **Short requests are invisible** between two 1 s polls when they also finish between GPU-utilisation samples (known, README).
- **Telemetry schema drift**: events before schema 2 lack `trigger`, `ended_by_governor` and `other_procs`. Consumers (`live_user_monitor.py`, `t4_inference_report.py`, `campaign_orchestrator.py`, ...) read `governor.jsonl`; schema 2 only adds fields.
- **Docs disagree**: `deploy/t4_node/node/README.md` said "after 30 s of quiet" in one row and 30 min in the policy section (fixed here); ROADMAP.md:386 (daytime harvesting after 30 s) conflicts with the owner's 7200 s by-day rule and should be marked superseded by the owner.
- **Roadmap items with no code**: `src/hivemind/core/preemption.py`, EF21 host-RAM checkpointing ("zero dropped gradients"), tensorsharp, Big GPU core, GPU pooling (per subagent read).

### 3.4 Added 2026-11-07: the QLoRA rounds have stopped improving the model (and the lockout reproduced)

Read-only analysis of `project/hmproj/hivemind_ledger/verdict_hmq_*.json` (186 rounds, eval loss on the fixed held-out set, `eval_loss_before` -> `eval_loss_after`):

- **Only 31 of 185 rounds (17 %) lowered the held-out loss**; mean change +0.0185 (worse), median +0.0065. Early on it worked: the first rounds start from about 2.396 and the best adapter, round 164 (3 Nov), reached 2.1198. **Since round 164 every round has been worse** (rounds 165-185: +0.06 to +0.75, e.g. round 182 +0.483), and round 186 (live, 7 Nov) too: 2.1198 -> 2.2442. That is 23 consecutive non-improving rounds. `latest_adapter.json` has stayed at round 164 since 3 Nov 23:48, so every new round restarts from the same adapter and is discarded.
- `valid: true` and a grant (cobblestones) are issued for every round: the validator replays one segment and checks it reproduces (`max_abs_divergence` 0.0). "Verified" means reproducibly computed, not that the model improved. 185 of 185 verdicts are valid.
- By domain (one domain per round, `DOMAIN_ORDER` cycle general, french, edu_web, news, biomed, math), share of rounds that improved the held-out loss: general 45 %, french 32 %, edu_web 12 %, math 6 %, biomed 3 %, news 0 % (31 rounds each). Defaults in `moe_work.build_train_spec`: lr 2e-4, LoRA rank 8, inner steps 40 (the work generator passes 80).
- Hypothesis, **not tested**: one single-domain shard per round at this learning rate overwrites what the other domains need (forgetting), and the greedy accept-only-if-better rule then leaves the chain at its plateau. The data show the domain dependence; they do not show the cause.
- Cost: in the owner's current window (17:00-22:00 Paris) the lockout of §3.1 allows about one 3.5-minute round per 34 minutes, i.e. about 6 rounds before 22:00, each of which is expected to be discarded. The first round of the window ran 17:30:26-17:34Z on 7 Nov; at 17:34:27Z, 30 s after the task ended, the validator's GPU use made the governor yield (`busy_trigger: gpu`, quiet timer reset), exactly as in §3.1: a second live observation of the same mechanism.

**Update 2026-11-08 (read-only, after the owner's redeployment).** The node scripts were rewritten at 16:59:07Z on 7 Nov (`t4_governor.py`, `t4_monitor.py`, `t4_workgen.py`; `expected_governor_args.txt` now `--night-start-h 17 --night-end-h 22`, so the training window is 17:00-22:00 Paris) and root created the operator `HOLD` file at 21:00Z, so the governor has been `held` (training paused) since. Corrections and additions to the paragraphs above:
- The deployed `t4_workgen.py` grew from 4,640 to 6,208 bytes and now has `--mix` ("stratified rounds over the six evaluated domains instead of one domain per round (the single-domain pattern failed the train-to-serve gate)"), `--night-only` and `--profile big`. That is the domain-balanced option (a) below: the owner's side already has it. **The running unit's `ExecStart` passes no flags, so `--mix` is not enabled**; rounds are still single-domain.
- Rounds 186-190 ran in the 7 Nov window (17:30-19:49Z), one per 32-37 min as predicted: 186 +0.124 (2.1198 -> 2.2442), 187 +0.187 (before 2.0367 -> 2.2236), **188 +0.051 (2.0367 -> 2.0876), promoted as the new head** (`latest_adapter.json` = hmq_00188, 18:39Z; its after-loss 2.0876 beats the previous head's 2.1198 although it is worse than its own start), 189 +0.070, 190 +0.309. So "stuck at round 164" and "every round worse since 164" are no longer true: one round was promoted, for a head improvement of 0.032 on the held-out loss. Unexplained: the `before` loss changed from 2.1198 (round 186) to 2.0367 (round 187) without a corresponding promotion in the ledger; it coincides with the redeployment and I did not find the cause.
- The rate of improvement is otherwise as stated: of the 190 rounds with losses, 31 (16 %) lowered the held-out loss relative to their own starting loss; the 23 rounds 165-187 did not, round 188 was worse than its own start but beat the old head's recorded loss and was promoted, and rounds 189-190 were worse than their start (the new head).
- I did not remove the `HOLD`; training stays paused until the owner decides.

What I did not do: change the training configuration, the work generator or the governor (all owner-controlled). Options for the owner, none implemented: (a) a domain-balanced shard per round (mix all six domains), (b) a lower learning rate or fewer inner steps, (c) promote on a held-out aggregate that includes a per-domain no-regression check, (d) before any change, a small preregistered offline comparison of (a)-(c) from the round-164 adapter on three rounds each, which needs the T4 for about an hour.

## 4. Incident: research jobs on the shared T4 caused user-facing errors (disclosure)

Between 17:57Z and 21:22Z on 2026-10-31 I ran OptimalEnergyAI experiments (Qwen2.5 0.5B/1.5B/3B) on this GPU. From 19:38Z to 21:21Z I also **stopped `hivemind-live-bee-llama.service`** (authorised by the owner) so that a 3B fp32 job (about 13 GiB) could fit; it was restarted automatically at 21:21Z. I did not stop or touch Ollama.

`t4_node.jsonl` (`chat_completions_since_last`, by route and status), 30-minute bins, UTC:

| bin | 200 | 502 | 429 | 499 |
|---|---|---|---|---|
| 14:30 ... 19:00 (my 0.5B and 1.5B jobs ran from 17:57) | 20, 9, 9, 23, 18, 16, 18, 16, 19, 15 | 0 | 0 | 0 |
| 19:30 (3B job and Bee stop from about 19:38) | 6 | 3 | 0 | 0 |
| 20:00 | 8 | 8 | 1 | 1 |
| 20:30 | 15 | 10 | 19 | 5 |
| 21:00 (3B finished about 21:21) | 4 | 2 | 0 | 0 |

**23 requests received 502 and 20 received 429 (6 client aborts) in the 1 h 43 min; the preceding 5 hours had none.** The errors were on the `coder` and `u` routes (Ollama-served), not on the Bee route I stopped. The most likely mechanism is VRAM: a 13 GiB job leaves no room for the 7B coder model, so Ollama could not load it. That is a hypothesis; I did not capture Ollama's logs. I do not know whether these were external users or the node's own producer heartbeats.
The governor itself was unaffected (state `yielded` throughout, no BOINC task running or killed).

**Mitigation proposed**: a research job that needs more than the free VRAM reported by `nvidia-smi` minus the largest resident-on-demand model (7B coder, about 5.5 GiB) must not run while serving is live. Concretely: (a) run such jobs only inside the owner-approved night window with the `HOLD` file set; (b) add a 5xx/429 alarm to `t4_monitor.py` (it already counts statuses per route); (c) for OptimalEnergyAI, 3B fp32 runs wait for that window or use a larger GPU.

## 5. Changes in this PR (all tested, none deployed)

1. **Telemetry (policy-neutral)**: every governor event carries `schema: 2`; a yield with trigger `gpu` carries `other_procs` (pid, name, MiB, 80-char command line of the non-BOINC GPU processes). Nothing is renamed or removed.
2. **`scripts/t4_governor_report.py`** and `tests/test_t4_governor_telemetry.py` (6 tests, stdlib only).
3. **README**: the stale "30 s of quiet" corrected to the owner's policy.
4. **FOR OWNER REVIEW, separate commit**: `--yield-grace-s` default 1.0 s to 0.1 s (a constant `YIELD_GRACE_S`). Expected yield latency about 0.22 s; acceptance after install: at least 20 real yields with p99 < 450 ms in the report, else revert. It changes the governor code hash, so it takes effect only after `sudo bash deploy/t4_node/install_node.sh`, which clears and restores the immutable flag; I did not run it.

5. **SUPERSEDED on 2026-11-08: `main` already contains this fix** (`pipeline_gpu_pids` / `PIPELINE_MARKERS`, with the comment "measured on 2026-11-05 ... training about 6% of the time", written independently of this audit). This branch was merged with `main` and keeps `main`'s version; my own commit for it was dropped in the conflict resolution. **The running node still has the older governor**: the lockout was observed live on 7 Nov at 17:34Z, after the 16:59Z redeployment, so the action is to deploy `main` (`sudo bash deploy/t4_node/install_node.sh`, owner). My original proposal, kept for the record: `other_gpu_busy` no longer counts the BOINC project's own server-script GPU processes (`real_client.boinc_server_scripts`, the validator replay) as a foreign user (new `boinc_pipeline_gpu_pids`). A real foreign user still triggers a yield (tested), and on a yield only `moe_app` is ended. Effect: rounds can follow each other at night until a real user touches the GPU, so the training duty cycle rises (this is the point, and also the cost: more night-time GPU use). Risk: a user request arriving during the ~20 s validator replay shares the card with it, since the validator is not killed; it already waits for an idle gateway (`wait_for_idle`, README). Acceptance after install: bursts with a real task per night in `t4_governor_report.py` rise from about 1 per 32 min to several per hour on a quiet night, and no user-visible slowdown in `chat_completions_since_last` (no 5xx/429); revert otherwise. Same install step as item 4. Not run.

Existing governor, drift and policy tests still pass (6 selected tests of `test_boinc_node_layer.py`); 8 new tests in `tests/test_t4_governor_telemetry.py`.

## 6. Proposals that need the owner or a measurement first

**Day MoE.** Only gpt-oss:20b (12.3 GiB) has a T4 number, and it leaves about 3 GiB, less than one training task needs; by day training is off anyway (7200 s), so the binding constraint is the night hand-over: at 22:00 the day mix must have released enough VRAM for 4300 MiB free. Proposed measurement before choosing a day MoE tier: for 2-3 candidates (for example OLMoE-1B-7B and granite-3.1-3b-a800m, both already in the local HF cache, and gpt-oss:20b) record tok/s, TTFT, peak VRAM and time to unload under the Ollama 10-min keep-alive, in one table, with positive and negative controls as in the OptimalEnergyAI protocol.

**Night QLoRA.** The owner's rule yields at most about 1 h per night (§3.1). If more training is wanted, that is the owner's decision to take with the evidence in §3.1; a safe, measurable alternative that does not touch the thresholds is to make each burst count (longer rounds when a burst is granted, since a burst now ends at about 2 min of a 3-4 min round) and to find out what ends the empty bursts (`other_procs`).

**Federation (nothing exists; measure before building).** A ladder with kill criteria, each step preregistered:
- F0: the RTX 2070 agent (`hivemind-rtx2070-agent.service`) is a unit on this host, but `nvidia-smi` here shows only the T4: establish where the 2070 physically is and measure RTT and bandwidth to it. Kill if RTT is above 5 ms or bandwidth below 1 Gbit/s (prediction, Tier C: pipeline-parallel decode over a slower link is bandwidth-bound).
- F1: a two-device layer split of one MoE with llama.cpp's RPC backend (llama.cpp has one; whether the local build under `gwaya-lite/llama.cpp` enables it is not checked) against the same model on one GPU: tok/s and TTFT, 3 prompts x 5 repeats. Kill if the split is slower than the single GPU for the model that fits one GPU.
- F2: expert placement driven by `expert_registry.py` residency planning with the measured numbers, only if F1 shows a gain for a model that does not fit one GPU.
- F3: failure injection (drop the remote node mid-request) with the shield's fallback.
No federation code should be written before F1.

## 7. Reproduce

```
python3 scripts/t4_governor_report.py                 # the table in §3
python3 scripts/t4_governor_report.py --json
python3 -m pytest tests/test_t4_governor_telemetry.py tests/test_boinc_node_layer.py -k "governor or policy or drift or terminate or pmon or night or telemetry or report or grace or other_gpu"
```
