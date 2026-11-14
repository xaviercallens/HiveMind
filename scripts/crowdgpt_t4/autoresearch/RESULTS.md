# autoresearch results: LLM-JEPA tuning for Crowd-v1 (2026-11-14, 07:50–09:25 CET, T4)

Protocol: `program.md`, written before any run. Every run used:
- 600 s of training wall-clock;
- seed 0;
- LoRA r64.

Metric: `val_ce`, the answer-token CE on 500 dev examples (lower is better). Guard rail: `far_loss` ≤ 3.396 + 0.05 = 3.446.
Raw log: `results.tsv`.

| run | val_ce | Δ vs SFT | far_loss | guard | steps | JEPA term at end | status |
|---|---|---|---|---|---|---|---|
| sft (baseline) | **2.5034** | — | 3.396 | — | 496 | — | keep |
| jepa λ=1 (baseline) | 2.7348 | +0.231 | 4.159 | ✗ | 385 | 0.0025 (collapsed) | discard |
| H1 λ=0.1 + LD 0.5 | 2.5128 | +0.009 | 3.474 | ✗ (by 0.028) | 545 | 0.023 | discard |
| H2 stop-grad on target | 2.8030 | +0.300 | 3.399 | ✓ | 365 | 0.081 | discard |
| H3 centered cosine | 2.9624 | +0.459 | 3.323 | ✓ | 355 | 0.007 (centered) | discard |
| H4 InfoNCE (T = 0.07) | 3.2738 | +0.770 | 3.253 | ✓ | 305 | 0.108 | discard |
| H6 layer-16 alignment | 2.6085 | +0.105 | 4.352 | ✗ | 482 | 0.0017 (collapsed) | discard |
| *x H1+H2 (post-hoc)* | 2.5403 | +0.037 | 3.430 | ✓ | 546 | 0.179 | discard |
| *x H1+H3 (post-hoc)* | 2.5048 | **+0.001** | 3.452 | ✗ (by 0.006) | 572 | 0.197 | discard |

Dev exact match (300 examples) is 0 % in every run: 600 s is too short to produce exact regexes. P1 needed 1,500 steps to
reach 17 %.

## Pre-registered selection: the 3 best of the 5 tested

Ranked by `val_ce`, guard rail applied:
1. **H2, stop-gradient**: 2.803, far 3.399.
2. **H3, centered cosine**: 2.962, far 3.323.
3. **H4, InfoNCE**: 3.274, far 3.253.

They are the three that **stop the damage to general language modeling**. P1's main failure was +0.90 nat; these runs end
at or below SFT's far_loss. At λ = 1 they all slow task learning at equal compute.

## What the evidence says (including the post-hoc runs)

1. **No variant beats plain SFT at equal compute.** The best, H1+H3, ties it (+0.001 nat, far below single-seed noise).
   The paper's "+14 points" does not appear in this setting.
2. **Two levers, and they combine.**
   - Weaker pressure (λ = 0.1 with loss dropout, H1) keeps task learning close to SFT.
   - Anti-collapse (stop-grad H2, or centering H3) removes or reduces the general-loss damage.
   - Together (x H1+H2) they reach val_ce 2.540 while passing the guard rail: the best guard-passing JEPA variant.
3. **Collapse is the failure mode, and it is fixable.** Every λ = 1 run without anti-collapse ends with 1 − cos ≈ 0.002
   and +0.76 to +0.96 nat of general loss; moving the alignment to layer 16 (H6) does not prevent it. With stop-grad,
   centering or InfoNCE, the term stays at 0.08–0.20 and general loss stays at SFT level or better.
4. **Caveats.**
   - One seed, 600 s, LoRA r64. Step counts vary (305–572) with the JEPA overhead and with contention on the shared T4.
   - Differences under about 0.02 nat should be treated as ties.
   - The selection is a screening, not a confirmation.

## Recommended next step (not run; the T4 is reserved for the weekend training)

Confirm at the full P1 protocol (3 epochs, 3 seeds, test-set exact match), against SFT:
- **(a) x H1+H2**, the best guard-passing variant;
- **(b) x H1+H3**, the best val_ce;
- **(c) H2 stop-grad with λ = 0.1** (= a), plus H4 InfoNCE with λ = 0.1.

**H8** (LoRA r512, where the paper's LoRA gain is reported) needs about 6 GB for adapter + Adam. That means a bigger GPU
than the T4 can spare, such as a community 24 GB card. It is the hypothesis most likely to reproduce the paper.
