# E8 stage 2: joint QLoRA rounds on a tiny Mixture-of-Experts, across machines

Follows `docs/E8_TWO_HOST_REAL_TEST.md`. Stage 1 validated the network protocol with a CPU toy. Stage 2 uses the real QLoRA stack (bitsandbytes NF4 + PEFT LoRA + a Hugging Face Granite-style MoE) and exchanges LoRA **deltas** between machines. It is `scripts/e8_qlora_moe_round.py`; tests are `tests/test_e8_qlora_moe_round.py`.

## 1. What it is, exactly

* **Model:** a tiny `GraniteMoeForCausalLM` (hidden 64, 2 layers, 4 experts, top-2 routing, vocabulary 64), **random weights**, shipped by the syncer so every machine starts from identical bytes. It is a stand-in for the production Granite 1B-A400M, not a replacement for it.
* **QLoRA:** with `--quant nf4` the linear layers are NF4-quantised (bitsandbytes, fp16 compute); LoRA (r=4, alpha 8) sits on `q_proj` and `v_proj`; the output head is trained in full (it is not quantised; without it the toy task is barely learnable, measured). The experts are **not adapted**: in the Hugging Face Granite MoE they are fused parameters, not `nn.Linear`.
* **Data:** each learner has its own non-IID shard (an affine next-token rule with 10 % noise, a different rule per learner); the held-out set is the mixture of all of them, from a seed no shard uses.
* **Round:** every learner starts from the global LoRA state, trains `--inner-steps` AdamW steps, and sends back only its delta; the syncer takes the token-weighted mean and applies an outer step with momentum (outer lr 0.7, mu 0.65, as in the E8 pilot). Learners wait for each other (K = M).
* **Controls built in:** `--shuffle-labels` (targets shuffled: the global loss must **not** improve) and `baseline` (the same algorithm in one process, for the exactness comparison).
* **Guards:** a learner on a GPU refuses to start if it would leave less than `--min-free-mib` (default 4,300, the T4's floor in `AGENTS.md`) free, and an unreadable GPU is a refusal; the syncer binds one address (loopback by default, `0.0.0.0` refused) and payloads are loaded with `weights_only=True`.

## 2. Limits (do not overstate)

* Toy scale (about 200 k parameters). It shows that the QLoRA delta round works across machines; it says nothing about Granite-scale quality, VRAM, or speed.
* **Two learners:** Multi-Krum keeps everyone at n < 5, so this is not a Byzantine test. That needs at least five learners (E9).
* The held-out loss is on the synthetic task; "learns" means "loss falls far below chance on that task", not downstream capability.
* GPU arithmetic is not bit-reproducible across machines or even runs: compare final losses with a tolerance, not for equality.
* If the other host cannot run bitsandbytes (it is the part most likely to fail on Windows), it must run with `--quant none`; then the two hosts train **different numerical models** (NF4 versus full precision) and the result is a LoRA round with mixed precision, not a QLoRA-on-both result. State that in the report.

## 3. Running it (the numbers)

Same repository commit on both machines. Python with `torch`, `transformers`, `peft` (and `bitsandbytes` for NF4). Overlay or tunnel exactly as in stage 1: syncer on the T4, learner 1 on the 2070 host through the SSH port-forward or WireGuard (`10.77.0.4:8765`).

Common settings for the first cell: `--inner-steps 40 --inner-lr 2e-2 --seed S` for seeds 1 to 3, `--rounds 4`.

T4 (syncer, then learner 0 on the GPU; use the BOINC virtualenv, which has bitsandbytes):

```bash
python scripts/e8_qlora_moe_round.py syncer --bind 10.77.0.4 --port 8765 --learners 2 --rounds 4 --seed S --inner-steps 40 --inner-lr 2e-2 --out artifacts/experiments/e8_real/stage2_wan_seedS.json
python scripts/e8_qlora_moe_round.py learner --syncer http://10.77.0.4:8765 --learner-id 0 --quant nf4 --device cuda --host-label t4-socreateai-agora-hermes-node1 --seed S --inner-steps 40 --inner-lr 2e-2
```

RTX 2070 host (learner 1; `--syncer` is the local end of the port-forward if you use SSH):

```bash
python scripts/e8_qlora_moe_round.py learner --syncer http://127.0.0.1:18766 --learner-id 1 --quant nf4 --device cuda --host-label rtx2070-XAVGAMSRV --seed S --inner-steps 40 --inner-lr 2e-2
```

Single-process baseline and negative control (either machine, same seed): `python scripts/e8_qlora_moe_round.py baseline --quant nf4 --device cuda --rounds 4 --seed S --inner-steps 40 --inner-lr 2e-2 --out ...` and the same with `--shuffle-labels`.

The learner retries its first connection for 300 s (`--connect-timeout`), so it may be started before the syncer is ready.

## 4. What to report

Raw JSON (the syncer report names both hosts and records, per round, the held-out loss each learner saw, the delta sizes, the arrival times and each learner's peak VRAM), the baseline and shuffled-label runs, free VRAM before and after on both machines (from `nvidia-smi`), and whether NF4 loaded on each host. A result is: the paired difference between the two-host final loss and the baseline's with its spread over seeds, **and** that the shuffled-label control did not improve.

## 5. Results so far (T4 only; the cross-host cell has not been run)

Measured on the T4 (Tesla T4, NF4, `--inner-steps 40 --inner-lr 2e-2 --rounds 4 --seed 1`, 2 learners). Mean held-out loss each round (learners' own NF4 evaluation), then the final value; chance for a 64-token vocabulary is ln 64 = 4.159.

| run | round 0 | round 1 | round 2 | round 3 | final |
|---|---|---|---|---|---|
| single process (baseline), real labels | 4.1721 | 2.0106 | 1.6994 | 1.7079 | **1.7772** |
| two learners over HTTP on loopback, both on the T4 GPU | 4.1721 | 2.0106 | 1.6994 | 1.7079 | **1.7772** |
| negative control, shuffled labels | 4.1721 | 4.1920 | 4.2408 | 4.2341 | **4.2034** |

* The two-learner round and the single-process baseline agree to every printed digit (difference 0) on the GPU with NF4. Both learners ran on the same GPU, so this shows the protocol is exact, not that two different GPUs agree.
* The negative control stays at chance, so the fall is learning, not an artifact of the pipeline.
* The base model is built from a platform-independent generator (numpy PCG64 keyed by seed and parameter name). An earlier version used torch's RNG; two learners in one process then built different bases, which the hash check in the AMCP version caught. The numbers above were re-measured after that fix.
* Training only the attention LoRA with frozen random embeddings and head reached a final loss of about 3.9; training the output head as well gives the numbers above (same settings): that is why the head is trained.
* The loss rises slightly after round 2 with these settings (the outer step is aggressive); gentler settings (`--inner-lr 1e-2 --outer-lr 0.5 --outer-mu 0.5`) kept improving in the AMCP soak (see `docs/FEDERATION_V2_AMCP_AGLETS.md`). Neither setting was tuned.
* Peak VRAM per learner as PyTorch allocated it: 27.8 MiB; free VRAM stayed near 10,000 MiB. Round latency on loopback: about 19 s for round 0 (both learners start, quantise and load), then 2 to 3 s per round.
* Unit tests: 13 pass, including a real HTTP round across two learners compared with the baseline within 1e-4 on CPU, and the negative control. One test caught a real bug in my first shuffle implementation.

**Not yet run: the cross-host cell** (T4 plus the 2070 host). It needs the 2070 host to run learner 1 with this script; whether NF4 loads there is unknown.

## 6. RTX 2070 Host (`XAVGAMSRV`) NF4 CUDA Verification, Measured Baselines & Live Auto-Learner (2026-11-01)

* **Environment on `XAVGAMSRV`**: `D:\xdev\HiveMind\.venv\Scripts\python.exe` (`torch=2.6.0+cu124`, `cuda=True`, `bitsandbytes` 4-bit NF4 GPU quantization enabled, `peft`, `transformers`, NVIDIA GeForce RTX 2070 8 GB SM 7.5).
* **NF4 Status on RTX 2070**: **Loads and executes natively on the RTX 2070 GPU (`--quant nf4 --device cuda`)** with `6,553 MiB` free VRAM before and after (`1,454 MiB` used / `8,192 MiB` total, `25-27 °C`).
* **Unit Tests (`tests/test_e8_qlora_moe_round.py`)**: **`13 passed in 14.47s`** after fixing `getattr(args, "connect_timeout", 300.0)` in `scripts/e8_qlora_moe_round.py` (when `Namespace` is constructed directly in `test_a_real_http_round_across_two_learners_matches_the_single_process_baseline`).

### 6.1 Measured Single-Process NF4 CUDA Baselines & Shuffled-Labels Control on RTX 2070 (`XAVGAMSRV`, `--quant nf4 --device cuda --inner-steps 40 --inner-lr 2e-2 --rounds 4`)

Raw JSON files are committed under `artifacts/experiments/e8_real/stage2_rtx2070_baseline_seed{1,2,3}.json` and `artifacts/experiments/e8_real/stage2_rtx2070_shuffled_seed1.json`:

| Run on RTX 2070 (`XAVGAMSRV`, NF4 CUDA) | Round 0 | Round 1 | Round 2 | Round 3 | Final eval loss |
|---|---:|---:|---:|---:|---:|
| `baseline`, seed 1, real labels | `4.1825` | `2.0188` | `1.7055` | `1.7043` | **`1.7099`** |
| `baseline`, seed 2, real labels | `4.1765` | `2.0398` | `1.7352` | `1.7431` | **`1.6895`** |
| `baseline`, seed 3, real labels | `4.1718` | `1.9182` | `1.5810` | `1.6250` | **`1.5849`** |
| Negative control (`--shuffle-labels`), seed 1 | `4.1825` | `4.1967` | `4.2332` | `4.2145` | **`4.1906`** |

### 6.2 Live Two-Host Stage 2 Readiness (`XAVGAMSRV` <-> `socreateai-agora-hermes-node1`)

The SSH port-forward tunnel from `XAVGAMSRV` (`127.0.0.1:18766 -> 10.77.0.4:8765` and `127.0.0.1:18765 -> 127.0.0.1:8765`) is **Established**, and the background auto-learner daemon on `XAVGAMSRV` polls `/state` every 2 seconds and automatically executes:
```bash
D:\xdev\HiveMind\.venv\Scripts\python.exe scripts/e8_qlora_moe_round.py learner --syncer http://127.0.0.1:18766 --learner-id 1 --quant nf4 --device cuda --host-label rtx2070-XAVGAMSRV --seed <S> --inner-steps 40 --inner-lr 2e-2
```
as soon as the T4 syncer opens a Stage 2 round.
