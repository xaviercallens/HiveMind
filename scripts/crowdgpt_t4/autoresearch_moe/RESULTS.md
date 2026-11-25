# autoresearch-MoE: results (21 Nov 2026)

The screen ran as pre-registered in `program.md`, including Amendment A1. The raw table is `results_2026-11-21.tsv`,
and the judge receipts are on the T4 in `hivemind_work/autoresearch_moe/T*_judge.json`. J is the mean mid-row NLL on
240 never-used chunks (lower is better). The base Crowd-v1 scores 4.286385.

| Trial | Change | J | vs reference (95 % CI) | Guard | Outcome |
|---|---|---|---|---|---|
| T1 | baseline (concatenated, seq 64) | 4.301917 | n/a | n/a | starting champion |
| T2 | same seed and data | n/a | n/a | n/a | bit-identical final state (`65d156a53c8a`) |
| T2b | baseline on data slice 2 | 4.278310 | −0.0236 [−0.0305, −0.0166] vs T1 | ok | noise control |
| T3 | interleaved shards (1024) | 4.301611 | −0.0003 [−0.0675, +0.0692] vs T1 | **fails** | **discard** |
| T4 | seq_len 128, same tokens | 4.281046 | −0.0209 [−0.0265, −0.0152] vs T1 | ok | keep |
| T5 | T4 + start from run 2 round 49, unseen data | 4.230899 | −0.0501 [−0.0635, −0.0375] vs T4 | ok | keep |
| T6 | T5 + inner lr 4e-4 | 4.206387 | −0.0245 [−0.0299, −0.0191] vs T5 | ok | keep, **champion** |
| T7 | T6 recipe on slice 2 | 4.226036 | −0.0523 [−0.0653, −0.0396] vs T2b | ok | **CONFIRMED** |

**Recommended recipe for campaign 2:** concatenated layout, `--seq-len 128`, `--lr 4e-4`, starting from run 2 round 49.

## How to read it

- **The winner beats the base model on content.** T6 scores −0.080 nats against the base model
  (CI [−0.101, −0.059]) and passes the +1 % guard against the base. T7 (slice 2) scores −0.060 [−0.081, −0.040], with
  maths at +1.43 % against the base, so that guard fails there.
- **Noise.** Two data slices of the same baseline recipe differ by 0.024 nats (T2b against T1). T4's gain, 0.021, is
  inside that. The cumulative gain confirmed on slice 2 is 0.052, about twice the noise gap.
- **T5 is not a like-for-like "start point" effect.** Round 49 has 75 more rounds of training than the upcycled start,
  so part of T5's gain is simply more training. What the screen shows is that continuing from round 49 with longer
  sequences and a higher learning rate keeps improving content.
- **Interleaving (T3) moved domains strongly in both directions.** Against the base model, stars moved −20.3 % and
  exoplanets −6.0 %, but general text +17.5 %, code +2.7 % and security +2.8 %. The mean is unchanged and the guard
  fails, so it is discarded under the rule. The size of the stars gain makes a gentler interleave (a smaller outer
  step, or interleaving only inside a learner's domain group) a candidate for a follow-up screen.
- **Budget.** Each trial used 30,720 tokens per learner, so these are short-run effects.
