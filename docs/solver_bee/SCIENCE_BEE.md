# Science Bee: ask in plain language, get a solved, checked and explained answer

Solver Bee stage 2, task P5. Code: `src/hivemind/solverbee/science_bee.py` and `families.py`. Experiment: S-3, pre-registered in `artifacts/experiments/S3_formulation_prereg.json`.

## What it does

A student, a citizen scientist or a researcher asks a question in English or French ("a pendulum with damping 0.1 started at 0.5 rad: where is it after 1, 2 and 5 seconds?").

1. A language model PROPOSES a data-only specification (`hivemind-solver-spec/1`, a JSON object). It is told only the allowed grammar and the families, which are generated from `families.py`.
2. A strict validator checks the proposal: unknown fields rejected, size limited to 64 KiB, every number finite, `t_out` strictly increasing, parameters known to the family. If it fails, the exact validation message is sent back and the model tries again, at most 3 retries (so at most 4 model calls). After that the answer is "could not formulate": it never guesses.
3. The accepted spec goes to the sandboxed `hivemind-solver` binary and to the verifier (the same `sandbox.py` and `verifier.py` as the Solver Bee service, imported lazily; any failure gives UNVERIFIED).
4. A deterministic template (no model) turns the result into plain language: what was solved, the key numbers at the requested times, the checks that passed, the verdict wording, and the mandatory limits sentence.

> The numbers solve the equations that were set up; whether those equations describe your real situation is not checked.

## What it does not do

* Nothing a user or a model writes is ever evaluated as code. A spec is data. The tests include hostile model output with code inside strings: it is rejected by the validator and never executed.
* The model is not trusted. Trust comes from the solver and the verifier. The model can still pick the wrong equations or a wrong number; the checks cannot see that. An AUDITED or VERIFIED answer to a mis-formulated question is possible, and the explanation prints the full setup (family, parameters, initial state, tolerances) so a person can read what was actually solved.
* Physical-model validity is never checked: whether a pendulum equation, a drag law or an SIR model fits your real situation is up to you.
* No silent unit conversion by the explainer. The solver works on plain numbers; the explainer says which units your question mentions and that all numbers were assumed to be in mutually consistent units. If the model converted units while writing the spec (minutes to seconds, a length to `g_over_l`), the converted numbers are visible in the setup lines.
* Linear systems come back labelled `x1, x2, ...` (the spec has no field for names). For the presets the state orders are fixed in `families.py`: series RLC `[q, i]`, spring-mass-damper `[x, v]`.
* "VERIFIED" needs a hit in the reference problem library; free-form specs reach at best AUDITED (no reference, but shape, hashes, invariants if any and the 100x tighter re-solve agree).
* Tolerance: if the proposal omits `rtol`/`atol`, the Science Bee adds `rtol 1e-10`, `atol 1e-12` (shown in the answer).

## Families

Named families (Rust CLI): `pendulum`, `projectile_drag`, `kepler`, `logistic` (the four new ones of the contract addendum) and the original six. Presets built by `families.py` from the generic kinds: SIR and SEIR (`mass_action`), decay chains (`mass_action`), series RLC and spring-mass-damper (`linear`). Each builder validates every argument strictly (`FamilyError`).

Note: at the time of writing the released CLI binary lists only the original six families, so questions that need the four new named families end UNVERIFIED (solver error) until a binary that lists them is used. The record of every S-3 run stores the binary hash and its `families` output.

## Example dialogues

These two are real outputs of the pipeline (real binary, real verifier). To keep them reproducible without a model they were produced with the keyword baseline formulator, not a language model; the explanation text is the same for any formulator.

English (question `q10` of the S-3 set):

```
Q: A series RLC circuit has R = 10, L = 0.5, C = 0.01 and a constant voltage V = 5. It starts with q0 = 0 and i0 = 0. Give q and i at t = 0.1, 0.5, 1 seconds.

What was solved: a linear system x' = A x + b. A = [[0, 1]; [-200, -20]], b = [0, 10].
Initial state: x1 = 0, x2 = 0.
Solver tolerances: rtol = 1e-10, atol = 1e-12.
Key numbers at the requested times:
  t = 0.1 : x1 = 0.0245837, x2 = 0.30956
  t = 0.5 : x1 = 0.0502275, x2 = -0.00646118
  t = 1 : x1 = 0.0500031, x2 = -2.46985e-05
Units: the solver works on plain numbers, so results are in the units of the numbers in the setup above. Your question mentions (time: seconds). I did not convert any unit silently; ...
Checks: the result has the right shape: passed; the input and result fingerprints match: passed; agreement with a stored independent reference solution: not applicable; conserved quantities stay conserved: not applicable; re-solving with 100 times tighter tolerances gives the same answer: passed.
AUDITED: no stored reference exists for a free-form problem, so the answer was audited instead: the checks listed above passed, but nothing compared it with a known solution.
The numbers solve the equations that were set up; whether those equations describe your real situation is not checked.
```

French (question `q09`), abridged:

```
Q: Une épidémie SEIR avec beta = 0,6, sigma = 0,3, gamma = 0,1, s0 = 0,98, e0 = 0,01, i0 = 0,01. Donne S, E, I, R à t = 10, 20, 40.

Problème résolu : réseau de réactions à action de masse. Espèces : S, E, I, R. Réactions : S + I -> E + I (k = 0.6); E -> I (k = 0.3); I -> R (k = 0.1).
Résultats aux instants demandés :
  t = 10 : S = 0.787508, E = 0.0879205, I = 0.0881251, R = 0.0364465
  ...
AUDITED : il n'existe pas de référence enregistrée pour un problème libre ; ...
The numbers solve the equations that were set up; whether those equations describe your real situation is not checked.
Les nombres résolvent les équations qui ont été posées ; que ces équations décrivent votre situation réelle n'est pas vérifié.
```

Follow-up (needs a model; the message flow, not a measured run): after the first answer, "and if the damping doubles?" (or "et si l'amortissement double ?") is sent to the model together with the previous VALIDATED spec and the instruction to return a new FULL spec; that spec goes through the same validation loop, the same solver and the same verifier. Follow-ups are detected by a short list of openers ("and if", "what if", "et si", ...) or forced with `ask(question, follow_up=True)`.

## How to run

Python:

```python
from hivemind.solverbee import science_bee as sb

backend = sb.OllamaBackend("qwen2.5:7b")            # local Ollama only, num_gpu=0 (CPU) by default
pipeline = sb.SolverPipeline("/path/to/hivemind-solver", problems_dir="solvers/problems")
bee = sb.ScienceBee(backend, pipeline)
answer = bee.ask("Un pendule avec g_over_l = 9,81 et damping = 0,1 part de theta0 = 0,5. Que valent theta et omega à t = 1, 2, 5 ?")
print(answer["text"])
answer = bee.ask("et si l'amortissement double ?")  # modifies the previous validated spec
```

The backend talks to an Ollama that is already running; it never starts, stops or configures it. Its base URL must be a loopback address (`localhost`, `127.x.x.x`, `::1`); look-alike hosts and URLs with credentials are refused, proxy variables are ignored, redirects are not followed, no streaming. For tests and dry runs use `sb.FakeBackend([...scripted replies...])`.

Experiment S-3 (CPU, one model arm, then the analysis):

```
python3 scripts/run_experiment_S3_formulation.py --arm baseline --binary BIN
python3 scripts/run_experiment_S3_formulation.py --arm model --model MODEL --cpu --binary BIN
python3 scripts/s3_analyze.py --baseline artifacts/experiments/S3_formulation_baseline_baseline_cpu.json --model-cpu artifacts/experiments/S3_formulation_model_MODEL_cpu.json
```

The GPU arm is the same command with `--gpu` and is run only by the lead in the owner's night window.

## How S-3 is scored

Full rules are frozen in `S3_formulation_prereg.json` (written and pushed before any model run). In short:

* 40 questions (24 English, 16 French), tiers A direct (13), B paraphrased (15), C implicit: unit conversion or derived parameter (12); 10 problem types. Expected specs and expected numeric answers come from `artifacts/experiments/s3_reference.py` (closed forms, exact matrix exponentials, mpmath `odefun` at 30 digits; nonlinear answers cross-checked against scipy DOP853 to 3e-11 relative). The file hash is in `S3_questions.sha256`.
* Primary metric: a question is correct if the final spec, once solved, reproduces the expected answers within the library tolerance policy (factor 100 on rtol 1e-8, atol 1e-10, fixed, not the spec's own tolerances) AND the verdict is VERIFIED or AUDITED. "Could not formulate", FAILED and UNVERIFIED are not correct.
* Secondary: exact spec match, retries, "could not formulate" rate.
* Arms: keyword/template baseline, model on CPU (`num_gpu=0`), model on GPU (lead only). Paired exact McNemar test model against baseline, alpha 0.05, per tier.
* Acceptance: A1 model CPU >= baseline on tier A with zero regressions; A2 tiers B and C reported separately; A3 zero wrong answers presented as VERIFIED (a hard failure otherwise; wrong-but-AUDITED answers are counted and reported next to it); A4 every answer contains the limits sentence.

## Results so far (no real model has been run)

Binary used: the released CLI build, which lists only the original six families (so the 18 questions on pendulum, projectile, Kepler and logistic cannot be solved by any arm yet). Receipt hashes are in the arm files.

| Arm (CPU) | Correct (40) | Tier A (13) | Tier B (15) | Tier C (12) | Exact spec match | Could not formulate |
|---|---|---|---|---|---|---|
| Keyword baseline (`S3_formulation_baseline_baseline_cpu.json`) | 4 | 4 | 0 | 0 | 13 | 26 |
| Scripted fake backend answering the expected spec (`S3_formulation_model_fake_cpu.json`, harness ceiling, not a model) | 19 | 5 | 8 | 6 | 40 | 0 |

How to read it:

* The baseline answers 13 questions with the exact expected spec, 12 of them in tier A, and fails the paraphrased and implicit questions (tier B: 1 exact match, tier C: 0), as expected for regular expressions. Its single tier-A miss is a question whose last sentence ends with "(same time unit as the rates)": its time-extraction rule stops on the word "time". The baseline was frozen before the questions and is not tuned.
* Of the 22 questions that the current binary can solve, the harness ceiling is 19 correct: for three oscillatory problems (`q11`, `q25`, `q36`) the verifier's tightening check returns FAILED even though the fake backend supplies the exact right spec and the numbers agree with the references. This is a property of the verifier at the house tolerances (a conservative check on a correct spec), reported as measured and not adjusted; it applies equally to every arm.
* Of the 18 new-family questions, all end UNVERIFIED (solver error) with this binary. The pre-registered rule counts them as not correct. A binary that lists all ten families is needed for the real model arms.
* No wrong answer was presented as VERIFIED or AUDITED in either file (A3), and every answer contained the limits sentence (A4). A1 cannot be judged until a real model arm exists; with the scripted fake backend the analysis script reports PASS only because the fake backend is a perfect formulator.

## Privacy

Questions stay on the machine. The only network traffic is the loopback request to the local Ollama; the backend refuses non-loopback addresses. A remote model could only be used by writing another `Backend` class, which is a deliberate decision of whoever configures it; the questions would then leave the machine.

## Files

| File | Purpose |
|---|---|
| `src/hivemind/solverbee/families.py` | catalogue (parameters, defaults, meaning and units of each state variable, invariants) and spec builders / presets |
| `src/hivemind/solverbee/science_bee.py` | backends, JSON extraction, strict validator, formulation loop, pipeline, explainer, conversation memory, keyword baseline |
| `artifacts/experiments/S3_questions.json`, `.sha256`, `s3_reference.py`, `S3_formulation_prereg.json` | frozen S-3 inputs |
| `scripts/run_experiment_S3_formulation.py`, `scripts/s3_analyze.py` | runner and analysis |
| `tests/test_solverbee_families.py`, `test_solverbee_science_bee.py`, `test_solverbee_s3.py` | tests |
