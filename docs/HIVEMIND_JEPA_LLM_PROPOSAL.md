# HiveMind JEPA-LLM: proposal (2026-11-15)

**Status:** proposal, plus an opt-in library (`src/hivemind/jepa/`, CPU-tested). **Nothing is wired into production
training.** The BOINC QLoRA workunits, DiLoCo replay, the governor and the live services are unchanged. The experiment
plan below is written before any of its runs. It will run after the weekend, when the T4 is free.

## 1. Why, and what we already know

HiveMind trains small models with scarce data and volunteer compute, so an objective that extracts more from each token
is worth having. Two papers from the same group are relevant:

| Paper | Objective | Needs | Training cost | Claim |
|---|---|---|---|---|
| **LLM-JEPA**, arXiv 2509.14252 | NTP + λ·(1 − cos) between the last-token states of two *paired views* | paired views (Q↔A, NL↔code) | ≈ 2× (an extra pass) | large fine-tuning gains on Instruct models; +11.4 points at LoRA r64 on NL-RX-SYNTH (Table 8) |
| **Semantic Tube Prediction (STP)**, arXiv 2602.22617 | NTP + λ·[1 − cos(h_t − h_r, h_r − h_s)] for random s < r < t, last-layer states | **nothing**: one sequence | **≈ 1×** ("negligible computational overhead") | matches full-data fine-tuning accuracy with **16× less data** on NL-RX-SYNTH; fine-tuning only, no pretraining tested |

Reference code for both: `github.com/galilai-group/llm-jepa` (Apache-2.0; `finetune.py` for LLM-JEPA, `stp.py` for STP).

**Our own measurements** (HiveMind #907, T4, Crowd-v1 + LoRA r64, NL-RX-SYNTH):
- LLM-JEPA at λ = 1 gave **no gain**: 16.90 vs 17.07 % exact match, 3 seeds.
- General held-out loss rose by **+0.90 nat**, because the cosine term collapsed (1 − cos ≈ 5e-4).
- Stop-gradient, centering or InfoNCE removed the damage, but no variant beat plain SFT at equal compute.
- Whether this is specific to Crowd-v1 is open (P2, ready).
- Separately, Crowd-v1's biggest weakness is the **data mix**: the gap to Qwen3-0.6B is +3.84 nat on code and +2.44 on
  math.

**Consequence for HiveMind.** STP is the better first candidate. It costs ≈ 1×, needs no paired data, so it fits raw
token streams, and it compares *displacements* of hidden states. A shared offset cancels in a difference, so it is
immune to the "all final states drift to one direction" shortcut that broke our LLM-JEPA runs. That last point is our
reasoning, not a measurement.

## 2. What the paper says vs the released code

The paper's experiments use the **three-index form**: Section 4, "pick (random) indices s<r<t, and compute
1−cos(h_t−h_r, h_r−h_s)".

The released `stp.py --linear=random_span` computes something else:
- a random patch inside the user + assistant tokens;
- 1 − cos(displacement of the patch, displacement of the rest);
- optional length weights.

The paper does not describe this, and the two are **not assumed equivalent**. The library implements the paper's form
as the default (`stp_variant="geodesic"`) and ports the released form for comparison (`stp.random_span_cos`).

λ is tuned per task in the paper (Table 2): 0.02 on NL-RX-SYNTH, ranging from 0.0025 to 0.16. Its text gives
"typically peaking between 0.01 and 0.08". Default here: **λ = 0.02**.

## 3. Design

### 3.1 Two objectives, one spec (`hivemind.jepa`)

| Module | Content |
|---|---|
| `spec.JepaSpec` | Versioned and validated; `spec_sha256()` binds it into a signed workunit. A client that does not know `version` refuses the workunit. Fields: `mode` ∈ {none, stp, pairs}, `lam`, `stp_variant`, `stp_samples`, `min_segment`, `eos_id`, `stopgrad`, `center`, `dropout`, `seed`, `collapse_floor`. |
| `stp` | The paper's geodesic loss, document-aware for packed streams; the released random_span variant; a `health()` check. |
| `pairs` | LLM-JEPA view term with the measured anti-collapse defaults (stop-gradient), plus centering and InfoNCE. |
| `hf.ntp_and_stp` | Next-token loss + STP from **one** forward pass of a Hugging Face / PEFT causal LM. It calls the decoder body once and applies the LM head only where labels exist, instead of `output_hidden_states=True`, which keeps every layer alive. |

### 3.2 Properties HiveMind needs (and that the library enforces)

1. **Document boundaries.** BOINC, DiLoCo and CrowdGPT streams are packed with EOS between unrelated documents. A
   triple spanning an EOS would join two texts, so triples are sampled **inside one document segment**. Rows with no
   segment of at least `min_segment` tokens contribute nothing. The paper did not face this: it only fine-tuned on
   single examples.
2. **Replayability.** Triples and loss-dropout decisions come from `sha256(seed, step, row, k)`, never a device RNG, so
   a validator replaying a segment computes the **same** loss. This is tested.
3. **Collapse watch.**
   - A perfectly straight trajectory satisfies STP while discarding information. The paper reports the term at about
     1.4 (λ = 0) and 0.6 (λ = 0.08), never near 0.
   - `health()` flags values under `collapse_floor` (0.05).
   - Every run must also pass the general-loss guard rail used in #907: held-out NTP ≤ SFT + 0.05 nat.
4. **Acceptance stays NTP.** The held-out next-token loss (`eval_by_domain`, the CrowdGPT loss check) remains the only
   promotion metric. Inference never uses the auxiliary term.

### 3.3 Integration points (proposed, not applied)

| Path | Objective | Change |
|---|---|---|
| **DiLoCo / BOINC replay** (`core/diloco_inner.replay_inner_segment`, `boinc/real_client/moe_app.py`) | STP on raw token streams | Optional `jepa` dict in the signed workunit spec (`JepaSpec.to_json()` + `spec_sha256`), and in the replay loop replace `model(input_ids, labels).loss` with `hf.ntp_and_stp(model, ids, labels, spec, step)`. Old clients refuse unknown versions. **Gated (§5).** |
| **Chat SFT** (`scripts/e17_train.py`, `qwen35_qlora_train.py`, Bee / coder grafts) | STP, or `pairs` for Q↔A / docstring↔code | A flag in the training script; same spec. |
| **OpenTrainer workers** | same as BOINC | Through the same signed spec. |
| **CrowdGPT client** | STP | Through `crowd_jepa.py` in Vxtzq/CrowdGPT#103, if E-STP-1 is positive; a round-config flag. |

Sketch of the replay patch (for review, not applied):

```python
# core/diloco_inner.py, replay_inner_segment
spec = JepaSpec.from_dict(workunit["jepa"]) if workunit.get("jepa") else JepaSpec()
...
loss, info = ntp_and_stp(model, input_ids, labels, spec, step=global_step)   # was: model(input_ids, labels).loss
loss.backward()
```

## 4. Pre-registered experiments (after the weekend; T4 under `t4guard`)

Common settings:
- Crowd-v1 rev 5ac3916 + engram (frozen), LoRA r64;
- general-loss guard rail ≤ SFT + 0.05 nat;
- STP health flag at < 0.05;
- 3 seeds, paired.

| ID | Question | Design | Success, decided now |
|---|---|---|---|
| **E-STP-1** | Does STP's data efficiency hold on Crowd-v1? | NL-RX-SYNTH, SFT vs SFT + STP (geodesic, λ = 0.02) at data fractions 1, 1/4, 1/16; lr 2e-5, 4 epochs (paper/LLM-JEPA recipe); exact match on the first 1,000 test examples | STP beats SFT by > 2 × pooled sd at ≥ 1 fraction, with no guard-rail or health failure |
| **E-STP-2** | Specific to Crowd-v1? | Same on Qwen3-0.6B (post-trained checkpoint) | Same rule; read jointly with E-STP-1 as in P2 |
| **E-STP-3** | Does STP help on **raw text** (never tested by the paper)? | Continued LoRA training on the Python-code domain (P0's largest gap, +3.84 nat), packed with EOS, SFT-on-text vs + STP (document-aware), equal tokens; metric: held-out code NTP loss, plus general-loss guard rail | STP lowers held-out code loss by > 2 × pooled sd without guard-rail failure |
| **P2 + STP arm** (amendment) | Add an STP arm to the ready P2 queue (≈ 1× cost) | `night_jepa.py --arm stp` (to be added before the run) | as P2 |

Compute: STP costs ≈ 1× SFT. E-STP-1 is 18 runs of about 25 min at 4 epochs, so it fits about two T4 nights.

## 5. Gate

- **Production wiring** (§3.3 replay path) only if E-STP-1 or E-STP-2 shows a gain under its pre-registered rule, and
  E-STP-3 shows no harm to held-out loss.
- **Other models** (`docs/JEPA_FOR_HIVEMIND_MODELS.md`) follow the same gate.
- If nothing passes, the library stays as an opt-in research tool and the data mix (#907 P0) remains the main lever.

## 6. Risks

| Risk | Mitigation |
|---|---|
| Straight-line shortcut (STP's analogue of P1's collapse) | `health()` + general-loss guard rail; report the term's trajectory |
| λ sensitivity (the paper tunes it per task; its text and appendix disagree on the range) | Start at 0.02 (NL-RX); a small sweep {0.005, 0.02, 0.08} only after E-STP-1 |
| Raw-text STP is an extrapolation | E-STP-3 tests it before any use |
| Replay mismatch between client versions | Versioned spec, hash bound into the workunit, refusal of unknown versions |
| Memory on the shared T4 | One-pass loss (no `output_hidden_states`); same 2.7 GB cap and ≥ 4,300 MiB free-VRAM guard as #907 |

## 7. Files

- `src/hivemind/jepa/{spec,stp,pairs,hf}.py`.
- `tests/test_jepa_llm.py`: 9 CPU tests, no download, run with the 3.11 venv:
  - spec validation and hashing;
  - deterministic dropout;
  - EOS-safe triples;
  - known-trajectory values (straight = 0, reversed = 2);
  - the released-variant split against a hand computation;
  - stop-gradient behaviour;
  - one-pass loss equal to the HF loss when off, and replayable when on.
- This document.
