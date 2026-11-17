# NIGHT-INTENSE: a plan for intense night QLoRA with larger bases on the single T4 (nothing applied, nothing run)

Status: **proposal and pre-registration. No GPU was used, no model was loaded, no service was touched, no governor setting was changed.**
Pre-registration: `artifacts/experiments/NIGHT_INTENSE_1_prereg.json` (written and committed before any run). Estimator: `scripts/qlora_vram_estimate.py`.

Request (owner, 2026-10-27): prepare to conduct intense new MoE QLoRA and larger-model training during the nights on the one Tesla T4 (15,360 MiB) that also serves users.

**What this is not.** The owner's governor policy (idle 1800 s at night, 7200 s by day, 4300 MiB free) stays as it is, and it is changed only through the reviewed
repo template (`deploy/t4_node/node/hivemind-t4-governor.service`). An earlier 5-second "intense" experiment was rejected by the owner: it produced 113 yields for
3.93 h of training (node telemetry aggregate, section 1). This proposal is different: it keeps the 1800 s quiet rule and asks only whether the genuinely quiet
night hours can host a larger base.

Convention: every number cites a file or says "node telemetry aggregate". "Not tested." marks every claim that nobody has measured.

---

## 1. Facts

### 1.1 Node telemetry aggregates (small sample)

All figures in this subsection are **node telemetry aggregate**, computed on 2026-10-27 at about 21:20 UTC from the node's own governor and node logs. The sample is
**2 to 3 days**. It is a snapshot, not a trend; every hour-of-day figure rests on n = 2 (n = 3 for 20:00 and 21:00).

| Fact | Value (node telemetry aggregate) |
|---|---|
| Real chat completions per Paris hour, 23:00 to 06:59 | 0 in every hour except 03:00 (mean 2, min 0, max 3). Sum of the eight hourly means: about 2 per night |
| 22:00 and 07:00 | 22:00: mean 2 (0 to 4); 07:00: mean 9 |
| Real chat completions per Paris hour, 08:00 to 21:00 | hourly means 19 to 32, mean of the 14 hourly means about 26 |
| nginx request rate | about 1.2 requests/s, which is health-check polling, not users (not used as a load measure) |
| Hours per governor state, rule 1800 s / 7200 s | 7.32 h training, 43.03 h yielded, 22 resumes, 19 yields |
| Hours per governor state, 5 s / 5 s experiment | 3.93 h training, 0.42 h yielded, 115 resumes, **113 yields** |
| Night window (22:00 to 07:00) of 2026-10-27 | 5.86 h training, 1.08 h yielded, 2 resumes, 1 yield |
| GPU at the time of writing | 4,839 MiB used of 15,360 MiB; only the bee llama-server (4,836 MiB) resident; the 7B coder Ollama model had unloaded when idle |
| Policy in force | `idle_start_s` 1800, `day_idle_s` 7200, `min_free_mib` 4300 |

Caveats that matter. (a) The night-window row of 2026-10-26 (3.25 h training, 84 yields) mixes the rejected 5-second period with the restored 1800 s rule, so only the
"by idle rule" and 2026-10-27 rows say anything about the owner's policy. (b) Two to three days cannot show a weekly pattern: a weekend, a launch or a bot would change
the picture. (c) Whether the roughly 2 night completions are coder or bee requests is **not split** in the aggregates. (d) The 03:00 hour is the only non-zero night hour
and has two observations.

### 1.2 Repo-sourced measurements

| Measurement | Value | Source |
|---|---|---|
| Granite-1B-a400m, resident after load | 2,444 MiB | `docs/T4_NIGHT_MODEL_DECISION.md` |
| Granite-1B peak under the 3,950 MiB cap (seq 128, micro-batch 2, rank 8) | 3,409 MiB allocated, 3,522 MiB reserved; 1.9 s per step, 134 tokens/s (lower bound, a package build shared the CPU) | `docs/T4_NIGHT_MODEL_DECISION.md` |
| Training process peak on the node | **3,716 MiB in every one of about 20 runs**; whole GPU peak with Bee + 7B + training 14,017 MiB (about 1.3 GB free) | `deploy/t4_node/live/README.md` |
| Qwen3-4B-Instruct-2507 | resident 2,578 MiB; out of memory during training at 3.97 GiB at micro-batch 2 | `docs/T4_NIGHT_MODEL_DECISION.md` |
| Phi-4-mini-instruct | resident 2,764 MiB; out of memory during training at 4.0 GiB | `docs/T4_NIGHT_MODEL_DECISION.md` |
| Granite-3B-a800m, Phi-tiny-MoE, OLMoE-1B-7B | about 6.0, 6.1 and 12.5 GiB resident (fused experts stay 16-bit); the first two ran out of memory loading, OLMoE was not run | `docs/T4_NIGHT_MODEL_DECISION.md` |
| Qwen2.5-Coder-7B QLoRA (NIGHT-MOE-1) | 10.23 GiB peak, about 33 s per step on the cyber stage; settings not stated in the doc | `docs/NIGHT_MOE_2_PLAN.md` |
| Cold reload of the 7B coder | 52 to 120 s; one 504 measured 2026-10-25; "over 120 s" in the issue | `deploy/t4_bee_expert/start.sh` (comment), issue #193 |
| Coder warm latency (short answer, router path) | 0.63 to 0.81 s; bee-stars 0.35 to 0.51 s | `deploy/t4_node/node/README.md` |
| One real yield | 1,135 ms from request to GPU free, `ended_by_governor` true, n = 1 | `deploy/t4_node/live/README.md` |
| Plateau of the Granite run | 98 chained rounds, plateau after about 30, mean held-out loss -5.8 % (2.2705 to 2.1380; best-of-97 on the same sets, one seed) | `deploy/t4_node/live/README.md` |
| Big profile | rank 16, seq 512, micro-batch 4, plan class `cuda_hivemind_big`, `min_gpu_ram_mb` 10,000; **no run recorded** | `src/hivemind/boinc/real_client/moe_profiles.py`, `docs/BOINC_RTX_GPU_INSTALL.md` |
| Thermal limits in code | governor starts training only below `--max-temp-c` 82; design ceiling 83 C soft, 84 C hard cut | `scripts/t4_governor.py`, `docs/SWARM_GPU_AND_DATASTORE_HARDENING.md` (HW-01) |

Measured temperature and power of this T4 during sustained QLoRA are **not recorded in any file read**: Not tested.

---

## 2. What "intense" can mean on this card: three VRAM tiers

| Tier | Free VRAM | When | What fits (estimate, see section 3) | Needs from the owner |
|---|---|---|---|---|
| **N0 current** | 4,300 MiB free is the start rule; training process peak 3,716 MiB measured | any time the policy allows | the Granite-1B MoE night base (measured); at micro-batch 1 plus checkpointing a dense 4B is conceivable but marginal (estimate 3,922 MiB, error up to 20 %): Not tested | nothing; this is what runs |
| **N1 night, coder unloaded** | 15,360 - 4,839 = **10,521 MiB** free with only the bee server resident | quiet night hours, after the 7B coder has left the GPU | Granite-3B-a800m (estimate 6,714 MiB at R2), Qwen3-4B with a higher rank, 4B dense at sequence 512 with checkpointing | an unloaded coder at night (section 7, option B) |
| **N2 night, bee also stopped** | about 15 GB (the 14,000 MiB budget used in the table leaves about 1.3 GB, as the whole-GPU peak in `deploy/t4_node/live/README.md` did) | only if the bee server is paused | OLMoE-1B-7B is still marginal at best (estimate 13,522 MiB at R2); a 7B dense with checkpointing at micro-batch 1 (estimate 7,479 MiB) | a serving decision: the bee-stars expert is offline for the window |

**How thin is the N1 margin against the big plan class.** `cuda_hivemind_big` declares `min_gpu_ram_mb` 10,000 (`moe_profiles.py`). If that number is read as
"10,000 MB of free memory", then 10,521 MiB free leaves a margin of **521 MiB (5 %) if MB means MiB, or 984 MiB (9 %) if MB means 10^6 bytes**; the repo does not say which. Either
margin is smaller than the 20 % uncertainty of the estimator, so a profile sized to the 10,000 MB floor could fail at its first step. Two points soften this. First, in BOINC
`min_gpu_ram_mb` filters on the card's total memory, not on free memory (the docstring of `moe_profiles.py` describes it as keeping large workunits off small GPUs), so the T4
qualifies whatever is resident; the only free-VRAM check is the governor's `--min-free-mib`. Second, that is exactly the gap: nothing today stops a big workunit from starting
beside a resident 7B coder. `deploy/t4_node/node/README.md` says a per-workunit VRAM requirement in the governor is "not built or tested". Not tested.

---

## 3. Candidate bases

Source of every row: `config.json` files in the local Hugging Face cache (read-only; no weights were opened). Parameters are computed from the config by the plain transformer
formulas in `scripts/qlora_vram_estimate.py`, except Qwen3.5-4B (hybrid attention plus a vision tower), where the dense 4.7 B of `docs/T4_NIGHT_MODEL_DECISION.md` is used.
**Licence: unknown for every model.** The cache snapshots hold no README, LICENSE or licence field in `config.json`, and the plan does not fetch anything from the network.
The licence of each model, and of the six night datasets, must be read from the model card before any adapter or number is published.

Caveat for every MoE row, from `docs/T4_NIGHT_MODEL_DECISION.md`: with transformers 5.17 the experts are fused 3-D tensors that bitsandbytes does not quantise, so a MoE is
resident at about two bytes per **total** parameter. Only a MoE of roughly 1.5 B total parameters fits the 4300 MiB policy. The checkpoint headers show per-expert linear layers,
which is misleading. LoRA on `gate/up/down` does not apply to fused experts; adapters go on attention (and the router is not adapted).

Fit key: "fits" = estimate with a 20 % margin is inside the budget; "marginal" = the point estimate is inside but not with the margin; "NO" = over. Budgets are free-VRAM
figures of the three tiers (4,300 / 10,000 / 14,000 MiB). Peak = estimated process peak (allocated + 307 MiB context allowance). Three recipes:
**R0** = the production workunit (seq 128, micro-batch 2, rank 8, q/k/v/o, no checkpointing); **R1** = the big profile (seq 512, micro-batch 4, rank 16, q/k/v/o);
**R2** = seq 512, micro-batch 1, rank 16, all linear layers, gradient checkpointing.

| Base | Total / active B (config) | Licence | Weights MiB | R0 peak | R0: 4300 / 10000 / 14000 | R1 peak | R1: 4300 / 10000 / 14000 | R2 peak | R2: 4300 / 10000 / 14000 |
|---|---|---|---|---|---|---|---|---|---|
| granite-3.1-1b-a400m-base (MoE) | 1.33 / 0.43 | unknown | 2,439 | 3,150 | fits / fits / fits | 5,825 | NO / fits / fits | 3,074 | fits / fits / fits |
| granite-3.1-3b-a800m-base (MoE) | 3.30 / 0.88 | unknown | 6,007 | 6,908 | NO / fits / fits | 10,764 | NO / NO / fits | 6,714 | NO / fits / fits |
| OLMoE-1B-7B-0924 (MoE) | 6.92 / 1.28 | unknown | 12,818 | 13,642 | NO / NO / marginal | 16,956 | NO / NO / NO | 13,522 | NO / NO / marginal |
| Qwen3-4B-Instruct-2507 (dense) | 4.02 / 4.02 | unknown | 2,579 | 4,388 | NO / fits / fits | 13,699 | NO / NO / marginal | 4,359 | NO / fits / fits |
| Qwen3.5-4B (dense, hybrid) | 4.70 / 4.70 (decision doc) | unknown | 3,267 | 5,164 | NO / fits / fits | 15,186 | NO / NO / NO | 5,428 | NO / fits / fits |
| Phi-tiny-MoE-instruct (MoE) | 3.75 / 1.29 | unknown | 6,218 | 7,424 | NO / fits / fits | 12,763 | NO / NO / marginal | 7,080 | NO / fits / fits |
| Phi-4-mini-instruct (dense) | 3.84 / 3.84 | unknown | 2,801 | 4,619 | NO / fits / fits | 14,048 | NO / NO / NO | 4,741 | NO / fits / fits |
| Qwen2.5-Coder-7B-Instruct (dense, the served coder) | 7.62 / 7.62 | unknown | 5,378 | 7,638 | NO / fits / fits | 19,040 | NO / NO / NO | 7,479 | NO / fits / fits |

Reproduce with `python3 scripts/qlora_vram_estimate.py`. The weights column reproduces the three measured resident sizes within 2.1 % (section 3.1); the peaks do not
have that accuracy.

Reading the table: **the big profile as written (R1, micro-batch 4 at sequence 512) does not fit N1 for any 4B dense base**; the logits alone are
tokens x vocabulary x 10 bytes. The estimator says gradient checkpointing and micro-batch 1 (R2) is what makes the 4B class feasible, and that R2 may even approach the
4300 MiB limit for Qwen3-4B (4,359 MiB, just over the 4,300 limit). None of this was measured: Not tested.

### 3.1 Calibration and its honest error

Calibration points (all `docs/T4_NIGHT_MODEL_DECISION.md`, settings = R0): Granite-1B **exact** 3,409 MiB allocated; Qwen3-4B and Phi-4-mini are **out-of-memory lower bounds**
under a 3,950 MiB cap (reported at 3.97 and 4.0 GiB, used as 4,065 and 4,096 MiB). One scalar (`act_scale`, the multiplier on activations) is fitted by least squares on relative
error; leave-one-out means fitting on two points and predicting the third.

| Point | Measured (MiB) | Leave-one-out prediction | LOO error | Weights-only error vs resident |
|---|---|---|---|---|
| Granite-1B-a400m | 3,409 | 2,810 | **-17.6 %** | 0.0 % |
| Qwen3-4B | 4,065 (OOM bound) | 4,095 | +0.7 % | +0.9 % |
| Phi-4-mini | 4,096 (OOM bound) | 4,508 | +10.1 % | +2.1 % |
| Granite-1B process peak (nvidia-smi view) | 3,716 | 3,150 (2,843 allocated + 307 allowance; the allowance was derived from this pair) | -15.2 % in-sample | not applicable |

Maximum leave-one-out error **17.6 %**, mean 9.5 %, fitted `act_scale` 0.99. This is inside the 20 % threshold but it is a **fit of one parameter to three points, two of them
censored**: it says the formula is not absurd, not that it is validated. Specific weaknesses: (i) the one exact point is the MoE and the estimator under-predicts it by
about 17 %, so MoE estimates are biased low; (ii) the 307 MiB context allowance comes from a single pair (3,716 process peak minus 3,409 allocated peak) that may not come from
the same run; (iii) nothing was calibrated at sequence 256 or 512, micro-batch 1 or 4, rank above 8, MLP targets or checkpointing, which is where this plan lives, so every R1 and R2
number is an extrapolation; (iv) an informal out-of-sample check on the 7B coder (rank 16, all linear, sequence 300 assumed, no checkpointing) gives 8,455 MiB at micro-batch 1 and
10,351 MiB at micro-batch 2 against the 10.23 GiB (10,476 MiB) of NIGHT-MOE-1, whose settings are not stated: consistent only if micro-batch 2, which is a guess. The table therefore
uses a **20 % margin, never the 17.6 % figure**, and must be re-checked by the memory probe of section 6 before any training is trusted.

---

## 4. Recipe knobs for "intense"

| Knob | Current (measured) | Proposed range | Evidence level |
|---|---|---|---|
| LoRA rank | 8, alpha 16, 1,376,256 trainable parameters (`docs/BOINC_CORE_NODE.md`) | 16 or 32 (A1, A2, A3) | rank 16 exists only as the unrun big profile: Not tested |
| Target modules | `q_proj, k_proj, v_proj, o_proj` (`moe_app.py`) | add `gate/up/down` on dense bases only; on a MoE the experts are fused and cannot take an adapter | Not tested |
| Sequence length | 128 | 256 (arms), 512 only with checkpointing | VRAM cost estimated, not measured |
| Micro-batch and accumulation | 2, no accumulation (`moe_profiles.py`) | 1 to 2 with accumulation to an effective 4 to 8 | gradient accumulation is listed as not used yet in `deploy/t4_node/node/README.md`; Not tested |
| Gradient checkpointing | off | on for 4B+ | named as the second step in `docs/T4_NIGHT_MODEL_DECISION.md`; Not tested |
| Optimizer | AdamW restarted at each segment (`replay_inner_segment`) | keep: replay validation depends on it | measured by the E-series replay tests only for Granite |
| Rounds per night | 98 rounds in 9.4 h under the 5 s rule (mean 105 s per round); 5.86 h of training in the 2026-10-27 night under 1800 s | one 3.0 h block per arm in NIGHT-INTENSE-1 | node telemetry aggregate and `deploy/t4_node/live/README.md` |
| Checkpoint cadence | segment checkpoint every 10 steps; a yield loses up to about 30 s | keep; a 4B step is slower so lose-on-yield grows | measured for Granite only (`deploy/t4_node/node/README.md`) |
| Thermal limit | start gate 82 C; alert level 83 C | stop the block at 83 C | no measured sustained-training temperature in the repo: Not tested |
| Power | no cap set | leave as is; log power every 5 s | Not tested |
| Throughput | Granite 134 tokens/s lower bound; 7B coder about 33 s per step (cyber, NIGHT-MOE-1) | 3B and 4B arms are expected to be several times slower per token than Granite; the table of `tokens/hour` is an output of the experiment | Not tested |

Why not "more rounds of the same thing": the Granite run plateaued after about 30 of 98 rounds and "expect little more from the same model and data; the lever left is data, not
model size" (`docs/T4_NIGHT_MODEL_DECISION.md`). Plateau on a 1.3 B model with tiny held-out sets may be a capacity limit or a data limit; the experiment below separates rank and
base size, but **it does not address data**, which the decision doc names as the other lever.

---

## 5. Evaluation

- **Metric.** Held-out loss per domain, expressed in **bits per byte** so that bases with different tokenizers compare (`docs/T4_NIGHT_MODEL_DECISION.md`). Fixed held-out slices of
  the six night domains (general, french, edu_web, news, biomed, math from `moe_work.py`), never trained on. The earlier run used 8 sequences x 128 tokens per domain, which that
  run's own caveat calls tiny; this plan uses 64 spans of 512 bytes per domain (384 total). Whether every domain has 64 usable held-out rows is Not verified.
- **Paired exact test against a control.** The control is the current recipe on the current base (arm A0) run in the same harness. The test is the exact paired sign test of
  `hivemind.swarm.verified_update_gate.exact_paired_p`, applied to per-span differences of gain per hour. The gate itself takes pass/fail outcomes on identical items; using its
  exact test on continuous per-span differences is an **adaptation**, stated in the pre-registration. The gate proper (`decide`) is used for a non-regression check.
- **Multiplicity.** Three comparisons (A1, A2, A3 against A0), alpha 0.05 / 3 = 0.0167, fixed now; 216 of 384 non-tied spans (56 %) is the smallest concordance that reaches it.
- **Plateau.** The earlier run plateaued after about 30 rounds (`deploy/t4_node/live/README.md`: later blocks improve by 0.0005 to 0.005, std of the last 20 rounds 0.0055). A 3.0 h
  block of a slower, larger model will do far fewer than 30 rounds, so it measures **early-training gain per hour**, not the plateau level. The experiment says nothing about the
  plateau of a larger base. Not tested.
- **No MBPP or HumanEval for training.** They are the repository's evaluation benchmarks (`deploy/t4_node/node/README.md`), cached but never training data.
- **What loss does not tell.** Held-out loss is not answer quality, and the Granite base is not a model the gateway serves; a better adapter on it has no user-facing gain today
  (`deploy/t4_node/live/README.md`). A larger-base adapter becomes user-facing only through a separate serving decision.

---

## 6. Pre-registered experiment NIGHT-INTENSE-1

Full text: `artifacts/experiments/NIGHT_INTENSE_1_prereg.json`. Summary.

**Question.** Does a larger base or a higher LoRA rank reduce held-out loss per night-hour compared with the current Granite-1B base within the same quiet night window, without
degrading morning serving latency?

**Arms** (one 3.0 h wall-clock block each, one arm per night, same ordered data, one seed):

| Arm | Base | Recipe | Tier | Estimate |
|---|---|---|---|---|
| A0 control | granite-3.1-1b-a400m-base | rank 8, q/k/v/o, seq 128, mb 2 | N0 | 3,150 MiB (measured process peak 3,716) |
| A1 rank | same | rank 32, q/k/v/o, seq 256, mb 2 | N0 | 3,592 MiB: marginal against 4,300 with the margin |
| A2 larger MoE | granite-3.1-3b-a800m-base | rank 16, q/k/v/o, seq 256, mb 2, checkpointing | N1 | 6,714 MiB |
| A3 larger dense | Qwen3-4B-Instruct-2507 | rank 16, all linear, seq 256, mb 1, checkpointing | N1 | 3,922 MiB: N0 conceivable but marginal |

**Metrics.** M1 primary: held-out bits-per-byte gain per training hour, pooled and per domain, against the arm's own untrained base. M2 absolute end bpb (descriptive: a bigger base is better
before training). M3 tokens/s and bytes/h. M4 VRAM peak (nvidia-smi, 5 s), temperature, power, RSS, disk. M5 interruptions. **M6 time to first token of the coder tier**, through the router path with the
persistent probe account: 20 requests 90 s apart from 23:00 to 23:30 with the coder warm (baseline, inside the window) and 20 requests over the first 30 minutes after training has stopped and left the GPU; the first
request after the stop is recorded apart (cold or warm).

**Acceptance rules, fixed now.**
- R1 validity: at least 2.5 h of training, starting no earlier than 00:00 and ending before 06:30, VRAM within the tier budget, temperature below 83 C, held-out sets unchanged. Otherwise INVALID (one re-run).
- R2 efficacy: pooled gain per hour at least 1.25 x A0, exact paired p < 0.0167, no domain with negative gain, and the gate's non-regression check against A0's own per-span map returns ACCEPT.
- R3 serving: warm-series median TTFT at most 1.25 x the pre-block median and at most 1.0 s; no 5xx; the first request after the stop answers within 120 s (a cold reload of 52 to 120 s is expected for N1 arms and is reported, not penalised below 120 s).
- Verdicts: WIN (R1+R2+R3), NO_GAIN, SERVING_DAMAGE (R3 fails: unusable whatever its loss), DOES_NOT_FIT, INVALID. One seed, so a WIN is labelled indicative.

**Probe protocol.** One candidate at a time. Only inside 23:00 to 06:30 Paris. Only when the **owner** has set the HOLD switch and says go. The commands below are listed so the owner can read them;
**none of them has been run, and this PR runs none.** Placeholders are written as `$NAME`; no host, user name or address appears in this file.

```bash
# 0. Owner: pause training so the governor cannot start a task (documented switch, deploy/t4_node/node/README.md)
touch /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD

# 1. 23:00 to 23:30 (inside the window): baseline TTFT of the coder tier, 20 requests 90 s apart (coder resident), via the router with the probe account.
#    These probe requests are traffic: they reset any quiet clock, which is why training cannot start before 00:00.
curl -s -N -o /dev/null -w '%{time_starttransfer}\n' -H "Authorization: Bearer $(cat $PROBE_KEY_FILE)" -H 'Content-Type: application/json' \
  -d '{"model":"coder","messages":[{"role":"user","content":"Reply with OK."}],"max_tokens":1,"temperature":0,"stream":true}' "$GATEWAY_URL/v1/chat/completions"

# 2. At 23:30, N1 arms only: show what is resident, then unload the coder (then 30 minutes with no request of any kind, until 00:00) (Ollama: keep_alive 0 unloads; the repo's fake mimics it, scripts/e16_fake_ollama.py)
nvidia-smi --query-gpu=memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits
curl -s localhost:11434/api/ps
curl -s localhost:11434/api/generate -d '{"model":"qwen2.5-coder:7b-instruct","keep_alive":0}'

# 3. Memory probe of the one candidate, with a cap of the tier budget minus the CUDA context (existing flags only: rank is fixed at 8, no checkpointing flag)
python3 scripts/t4_model_probe.py Qwen/Qwen3-4B-Instruct-2507 --cap-mib 9650 --micro-batch 1 --seq-len 256
#    Expectation before the run: the estimator, with its 20 % margin. DOES_NOT_FIT if the probe fails.

# 4. 00:00: training block (3.0 h, ends 03:00): NOT AVAILABLE. Needs the probe runner listed below.

# 5. After the block (hard stop 06:30; planned end about 03:00 plus evaluation): confirm the GPU is free, then measure the 20 TTFT requests of step 1 again over 30 minutes
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
# (repeat the curl of step 1, 20 times, 90 s apart; record the first one separately)

# 6. Owner: allow the governor again only after step 5 is complete
rm /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD
```

**Prerequisites that do not exist yet (not built, not in this PR):** (a) a probe runner that trains one arm for a fixed wall-clock budget and **stops within about 1 s** on any nginx in-flight
request or other-process GPU activity, reusing the governor's `stub_status` and GPU-busy checks, and writes before/after held-out bpb; (b) Qwen3 held-out sets and, for a BOINC route, a
model archive for each base; (c) rank and checkpointing options in `scripts/t4_model_probe.py`. Step 4 therefore cannot be executed from the repository today.

---

## 7. Governor and policy options (for the owner)

### Option A: keep as is
No change. Training stays on Granite-1B at N0 under the 1800 s / 7200 s / 4300 MiB policy. Trade-off: zero risk to serving; the plateau (about 30 rounds) limits the value of further nights.
Within A, arms A1 and A3 are *conceivably* runnable at N0 with no policy change (estimates 3,592 and 3,922 MiB, both marginal), as one-off supervised probes under HOLD. Not tested.

### Option B: a "night-big window", 23:00 to 06:30
Proposal: inside 23:00 to 06:30 Paris, a big-profile workunit may start only when the quiet time is at least 1800 s (**the rule is not shortened**) **and** the coder is unloaded so that at least
10,000 MiB are free. Outside the window nothing changes.

The current governor cannot express this: it has one night window (22:00 to 07:00), one `--min-free-mib`, and no notion of a workunit size (`scripts/t4_governor.py`). So B has two parts:
a code change to the governor and the work generator (not in this PR) and the template change below. **The flags in the diff do not exist yet; the diff is the proposed interface only and is
applied nowhere.**

```diff
--- a/deploy/t4_node/node/hivemind-t4-governor.service
+++ b/deploy/t4_node/node/hivemind-t4-governor.service
@@ -13 +13,2 @@
-ExecStart=/usr/bin/python3 /opt/hivemind/t4/scripts/t4_governor.py --idle-start-s 1800 --day-idle-s 7200 --min-free-mib 4300 --night-tz Europe/Paris --night-start-h 22 --night-end-h 7
+ExecStart=/usr/bin/python3 /opt/hivemind/t4/scripts/t4_governor.py --idle-start-s 1800 --day-idle-s 7200 --min-free-mib 4300 --night-tz Europe/Paris --night-start-h 22 --night-end-h 7 \
+  --big-window-start 23:00 --big-window-end 06:30 --big-min-free-mib 10000 --big-idle-s 1800
```

Meaning of the new flags (proposed): `--big-window-*` is the window in which the big profile may run; `--big-min-free-mib` is the free-VRAM test for it, checked at start and at each resume;
`--big-idle-s` is the quiet time for it and is set equal to the existing 1800 s on purpose. The work generator would queue a big workunit (`hmqb_`, chain `latest_adapter_big`) only inside the window.
Changes go through the repo and `install_node.sh` (which makes the unit immutable), never through a drop-in: a drop-in silently overrode the policy on 2026-10-25 (`deploy/t4_node/live/README.md`).

**What unloads the coder, and who does it.** Two ways, with different consequences:
1. *Ollama keep-alive expiry.* No code: Ollama drops an idle model after `OLLAMA_KEEP_ALIVE` (10 minutes per the comment in `deploy/t4_bee_expert/start.sh`; the telemetry shows the coder unloaded at night).
   Weakness: the same script pins the coder with `keep_alive: -1` when it runs, and any pin defeats B silently (the governor's free-VRAM test would simply refuse to start). Whether the live pin is active
   now is **not verified**: the telemetry says unloaded, issue #193 says pinned "as a stopgap", and both can be true at different times. A user request at 22:55 re-loads it for another 10 minutes.
2. *Explicit unload* (`keep_alive: 0` on `/api/generate`): deterministic, and the owner or the governor would send it at window start after 1800 s of quiet. Weakness: the governor would then touch a serving
   component; a bug there is a serving bug. A separate small unit run by the owner is safer than putting it in the governor.
Either way **the owner decides who**; this plan recommends expiry first (nothing new to break) and an explicit unload only if expiry proves unreliable. At the end of the block, a pre-warm of the coder at 06:30
(one `max_tokens: 1` request, the same idea as the warm-up in PR #249) puts the 52 to 120 s reload at a time when nobody waits. Not tested.

**Rollback.** Revert the template commit, run the installer, restart the governor, check `systemctl show -p ExecStart hivemind-t4-governor.service` and the governor `start` event shows the previous values;
immediate pause is `touch .../HOLD`. The unit is immutable by design (`chattr +i`), so rollback is a deliberate, logged step.

**Monitoring.** (i) Yields and resumes per night from `governor.jsonl` (the 5 s experiment is the cautionary number: 113 yields for 3.93 h); (ii) morning TTFT by M6 every night of the experiment and then weekly;
(iii) `model_loading` and 5xx counts from the router for 22:00 to 08:00 (PR #249 added the `model_loading` answer); (iv) GPU temperature and power from the node monitor; (v) disk and RSS.
Stop rule: any SERVING_DAMAGE verdict or any user-facing 5xx attributable to a cold coder pauses option B until the owner re-decides.

### Option C: also pause the bee server at night (N2)
Frees about 4.8 GB more (about 15 GB). Trade-off: the bee-stars expert (served by the llama-server, its own producer and tier) is offline in the window; any night request to it fails or reroutes. Only worth it if an
arm truly needs more than N1, which the table says only OLMoE-1B-7B or a 7B dense at long sequence do. It is a **serving decision for the owner**, not a training decision; this plan does not recommend it before B has produced a result.

---

## 8. User-impact analysis from the facts

- **Night traffic is almost nil in this sample.** About 0 real completions per hour from 23:00 to 06:59, about 2 per night in total, against about 26 per hour by day (node telemetry aggregate, n = 2 days per hour). Expected
  users touching the GPU during a 23:00 to 06:30 block: roughly 2 per night, with a range from 0 to a few. Small sample: a different week could differ, and one busy night is enough to produce several cold starts.
- **What a yield costs the trainer.** Not the user: the governor sets BOINC to `never` and ends the task, freeing the GPU in 1,135 ms once measured (n = 1), 0.77 s and 7.1 s in the 5 s experiment (`deploy/t4_node/node/hivemind-t4-governor.service` comment);
  work since the last segment checkpoint (up to about 30 s) is redone.
- **What a yield costs the user at N1.** The trainer leaves quickly, but if the coder was unloaded to make room, that request meets a **cold 7B reload of 52 to 120 s** (`deploy/t4_bee_expert/start.sh` comment; one 504 measured, issue #193). PR #249 is merged, but its own body says it was not deployed and the live node was not touched, so whether the running gateway behaves as follows is **Not tested**. As written in that PR: when every candidate reports the model not loaded and the cold start is at least `cold_start_wait_budget_s` (default 20 s; 60 s is assumed when the producer reports none), the gateway answers `model_loading` (503 with `Retry-After`) with no charge and starts a one-token warm-up; a producer that reports no residency at all is still forwarded to with the existing 120 s upstream timeout, which is exactly the cold-coder case if residency is not reported. The user is then told to retry rather than timing out, if and only if residency reporting is live. A request for the bee model is not affected by the coder being unloaded. Which model the roughly 2 night requests ask for is not split in the data: Not tested.
- **Morning.** 07:00 shows a mean of 9 completions per hour and 08:00 a mean of 24 (node telemetry aggregate, n = 2 days). A block that stops by 06:30 followed by a pre-warm has about 30 minutes of margin before the first user wave; a block that overruns would put the cold reload in front of the first users. This is why M6 measures the first 30 minutes after the stop.
- **Day policy is unchanged** (2 h silence), so a daytime user at about 26 completions per hour never sees a trainer.

---

## 9. Risks

| Risk | Detail | Mitigation / status |
|---|---|---|
| Spot VM preemption | the VM can be reclaimed at any moment; Cloud Scheduler restarts it; a block loses at most the last segment (about 30 s) for Granite (`deploy/t4_node/node/README.md`); a 4B step is slower | segment checkpoints; the probe runner must write its own; Not tested for larger bases |
| Thermal | the card is shared with serving; alert 83 C, start gate 82 C (`scripts/t4_governor.py`); no sustained-training temperature recorded | stop the block at 83 C; log temperature and power; Not tested |
| Disk | the data disk held 1.6 GB free in October (`docs/NIGHT_MOE_2_PLAN.md`); the present free space is Not verified; a 4B base adds about 8 GB of 16-bit weights to the cache (`docs/T4_NIGHT_MODEL_DECISION.md`) | check free disk before and after each arm (M4) |
| Host RAM | loading a 4B model reads about 8 GB of 16-bit weights through CPU memory, took 3 to 4 minutes, and a probe next to a package build exhausted memory and stopped both (`docs/T4_NIGHT_MODEL_DECISION.md`) | one probe at a time, never beside a build; current free host RAM is Not verified |
| Replay-validation differences between hardware | the BOINC verifier replays segments and rejects a divergence above 1e-4; cross-device replay tolerance (E12) compared one T4 with its host CPU (`docs/benchmarks/T4_10h_campaign_report.md`); other GPUs and any larger base are not covered | any larger base needs its own replay check before it is offered to volunteers; Not tested |
| Licences | model licences are unknown here (no card in the cache); dataset licences for the six night domains are not recorded in the files read (FineWeb-Edu and Wikipedia "carry their own terms", `docs/BEE_MOE_NUMERIC_SOLVERS_PROPOSAL.md`) | read the cards before any publication of adapters or results |
| No peer review | the plan, the estimator and the pre-registration were written by one agent session and reviewed by nobody | the owner's review of this PR is the first review |
| Estimator error | up to 17.6 % leave-one-out on three points, biased low for the MoE; extrapolated beyond the calibrated shape | 20 % margin; mandatory probe before any training |
| Pinned or reloaded coder | a pinned coder, or a user request at 22:55, leaves less than 10,000 MiB free | the free-VRAM test refuses to start; the block simply does not run |
| Adapter lineage | an adapter is specific to its base; a new base starts a new chain and cannot continue the Granite adapters | stated in the pre-registration's not-claimed list |
| Gaming of the metric | held-out loss per hour favours fast small models in raw hours | M1 is per own-base gain; M2 reports absolute loss; no claim beyond these six domains |

---

## 10. Decisions for the owner

1. Do you want NIGHT-INTENSE-1 to run at all? If yes, which arm first? (This plan suggests A0, then A1, then A2, then A3; A1 and A3 may be possible under the unchanged policy.)
2. For N1 arms: do you accept the option B window (23:00 to 06:30, 1800 s quiet, coder unloaded, 10,000 MiB free) as a **proposal to be implemented and reviewed in code**, or do you prefer option A only?
3. Who unloads the coder at night: Ollama's own keep-alive expiry (recommended first), an explicit unload by you, or by the governor? And is the coder currently pinned (`keep_alive -1`)?
4. Is pausing the bee server in the window (option C) acceptable at all, or off the table?
5. Do you accept a cold coder reload of 52 to 120 s for the roughly 2 night requests per night this sample suggests, with the `model_loading` answer from PR #249 as the user-facing behaviour?
6. Do you accept the acceptance rules of the pre-registration as written (including the serving-latency rule R3), or do you want a different latency bar?
7. Who builds the probe runner (a stop-within-1-s trainer, Qwen3 held-out sets, rank and checkpointing in the memory probe), and who reviews it?
8. Who reads the model and dataset licences before any result or adapter leaves the node?
9. Do you want this also tried on a second seed before any serving decision, since one seed can only be indicative?
