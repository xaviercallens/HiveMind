# H-Net chunking + tree-of-thought orchestration + multi-MoE calls: what is supported, what is not, and the one experiment worth running

Written 2026-11-01 (evening) from the full text of Hwang, Wang, Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence
Modeling* (arXiv 2507.07955), the HN1 run recorded in `artifacts/experiments/HN1_boundary_signal_report.json`, and what
HiveMind's router and nodes do today. Nothing in this file is a measurement of the proposal; the measurements it names are
labelled with their artifact or NOT_RUN.

## 1. Claim-by-claim check of the proposal

| Proposed element | Status | Why |
|---|---|---|
| Hierarchical encoder/decoder on the edge node, main network remote | Paper-supported architecture, **not implemented anywhere in HiveMind** | H-Net is a single model trained end to end; HiveMind nodes exchange text over OpenAI-compatible HTTP. A latent-exchange protocol would be new. |
| "Macro-chunks reduce network traffic 5x-9x" | **Overstated** | The paper's ratios are bytes per chunk (4.5-6 for 1-stage, about 9 for 2-stage on code). Against the token-level exchange HiveMind already does, 1-stage is about 1x and 2-stage about 2x on code (`docs/HNET_T4_PROBE_PLAN.md`). |
| Boundary probability `p_t` as the trigger to open a tree branch ("Boundary-Triggered Tree Search") | **Testable, currently unsupported** | HN1 looked for the same cue in Qwen2.5-0.5B's hidden states; the run was INVALID by its own controls (newline positive control failed on english, random-init negative control failed on french). No served model exposes a 2-stage `p_t`. |
| "Latent Multi-MoE Consensus": interpolating hidden vectors from several MoEs inside the dechunking layer | **Unsupported** | It requires every expert to share one latent space. Different backbones (Qwen-Coder, DeepSeek-R1-Distill) do not. Appendix F gives a shared encoder for *one* frozen backbone at 189B bytes of training; it does not give a common space across backbones. |
| Reasoning in the latent space of chunks (Coconut-style) over the network | **Unsupported on today's stack** | Needs the latent protocol above plus a model trained for latent recurrence; the paper only lists these as related directions in Section 4. |
| Adaptive ratio loss for "System 1 / System 2" modes | **Design idea, no evidence** | The ratio loss targets a compression rate during training; the paper does not change it at inference. |
| Generator-verifier loop across experts, prune failed branches before continuing | **Runnable today, in token space** | HiveMind already verifies code against tests (GWAYA, QueenBee verified outcomes). Branching K candidates across expert pools and keeping only what passes needs no new model. This is the piece to measure first. |
| Domain-split routing of sub-tasks to specialist experts | **Runnable today** | `hivemind/auto` already routes by intent per request; per-sub-task routing is an orchestration change, not a model change. |

Conclusion: of the three "innovations", only the generator-verifier loop can be built and measured on HiveMind now. The
boundary trigger is a research question with one invalid run behind it. Latent consensus has no path without a shared
encoder per backbone, which costs more training than the project can run.

## 2. Recommendation on the owner's question (which level first)

**The fusion/verification mechanism first, but in token space, not latent space.** The meta-MoE routing protocol only pays
off if branching plus verification actually raises the verified pass rate per remote call; that is unmeasured. The
experiment below answers it with models already on this machine. If it fails, the routing protocol has nothing to route.

## 3. TM1: verifier-pruned branching across expert pools (preregistered, NOT_RUN)

Preregistration: `artifacts/experiments/TM1_verifier_pruned_branching_prereg.json` (written before any run; the report
records its sha256).

- **Tasks:** 40 MBPP problems, seeded sample, hidden tests from the dataset (local Hugging Face cache; no download).
- **Models (all local, Ollama):** planner `qwen2.5-coder:7b`; executor pools `qwen2.5-coder:3b` and `qwen2.5-coder:1.5b`.
  No DeepSeek-R1 model is on this host; the planner role is filled by the 7b coder and the report says so.
- **Arms:** A, one call to the 3b executor. B, planner writes a plan with 3 approaches; 3 branches run on the two pools;
  the hidden tests pick the first passing branch. C, 3 samples from the 3b executor at temperature 0.8, same verifier, no
  planner (isolates planning from best-of-3 sampling).
- **Measures:** verified pass rate per arm; model calls per solved task; generated tokens per solved task.
- **Hypotheses:** H1, B exceeds A by at least 10 points of pass rate. H2, B exceeds C by at least 5 points (otherwise the
  planner adds nothing over sampling). H3, B costs at most 3x A's tokens per task.
- **Positive control:** an oracle arm that stops at the first passing branch must score at least A on every seed; if not,
  the harness is broken. **Negative controls:** (i) verifier replaced by "accept the first branch": B's pass rate judged by
  the real hidden tests must fall to the single-branch rate; (ii) hidden tests swapped with another problem's tests: pass
  rate must be near 0, or the verifier is tautological.
- **Decision rule:** H1-H3 each reported as supported or not; the controls decide validity. Only if H1 and H2 hold does the
  meta-MoE routing protocol get designed.
- **Not claimed:** anything about latent chunks, H-Net, DeepSeek-R1, or network traffic.
- **Blocker named before the run:** generated code must execute to be verified. The GWAYA bubblewrap sandbox is BLOCKED on
  this host (AppArmor user-namespace restriction). The run needs either a host with a working sandbox or the owner's
  explicit acceptance of `python -I` in a temporary directory with a timeout and resource limits, which is weaker.

## 4. Text proposed for the foundation paper (issue #452; do not paste numbers until TM1 has a report)

> *Hierarchical chunking and tree search.* Dynamic chunking (Hwang et al., 2025) learns byte-level boundaries and compresses
> about 4.5-6 bytes per chunk, close to a BPE tokenizer; against HiveMind's token-level exchange it does not by itself reduce
> traffic between nodes. Its distillation recipe keeps a frozen backbone but costs 189B bytes of training and loses about
> three accuracy points, which is outside this project's compute. We therefore treat the chunker's boundary signal as a
> research question (experiment HN1, first run invalid by its preregistered controls) and test the orchestration idea it
> motivates, verifier-pruned branching across expert pools, in token space (experiment TM1, preregistered, not yet run).
