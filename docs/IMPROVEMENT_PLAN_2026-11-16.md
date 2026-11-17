# HiveMind Improvement Plan (2026-11-16 post-weekend update)

> Successor to [`IMPROVEMENT_PLAN_2026-11-05.md`](IMPROVEMENT_PLAN_2026-11-05.md), grounded in the **14–16 November 2026** CrowdGPT × HiveMind weekend run telemetry, [`LL.md`](../LL.md) (v3.4.2 + PR `#926`), and [`MVP_SCORECARD.md`](MVP_SCORECARD.md) (`3/11` passing).
>
> **Purpose:** Give **other Claude / Antigravity coding sessions** crisp, testable PR briefs (`IP-293`–`IP-297`, `IP-300`) and give **end-users / community contributors** turnkey onboarding & self-contained run paths (`IP-298`, `IP-299`). Nothing here hides failed gates or no-op runs: every metric comes directly from measured logs on `socreateai-agora-hermes-node1`, `Vxtzq/Crowd-v1` HuggingFace telemetry, and `hivemind-openrouter`.
>
> **Legend:** **R** = directly observed in files/logs/Cloud Run outputs on 2026-11-16; **A** = inherited from `origin/main` PRs / `LL.md`.

---

## 1. Closed or In-Flight Since 2026-11-14 (`v3.4.2`, `#916`–`#926`)

| Item / PR | Status | Evidence / Notes |
|---|---|---|
| `#916` (`--crowdgpt` contributor wrapper) | **Merged** | `scripts/hivemind_opentrainer_agent.py --crowdgpt` wraps Vxtzq's `client.py` with pause/stop/thermal guards (`docs/OPENTRAINER_CROWDGPT_CONTRIBUTOR.md`). |
| `#917` (public signup / redirect / live starter grant) | **Merged** | Removed stale `500 HMT` claims & parked MoE split ads on `/raid`; redirects trailing `)`/`]` on `/signup`. |
| `#918`, `#919` (honest egress & account receipts) | **Merged** | Labels `modeled` vs `measured` (`~0.9 GB/round`, `~1.5 GB/h`) and separates real accounts from seeded fixtures (`LL.md` lesson 4). |
| `#920`, `#922`, `#907` (JEPA P1 null + E-STP pilot) | **Merged** | 3-seed JEPA r64 pilot on `Crowd-v1` returned null (`16.90%` vs `17.07%` SFT, `far_loss +0.90 nat`, cosine collapse); `scripts/crowdgpt_t4/run_estp.py` queued for `>= 2026-11-16 07:00 CET`. |
| `#923`, `#924` (Multi-Krum & unified L2 cap) | **Merged** | Round close now runs Multi-Krum before outer aggregation; `DEFAULT_MAX_DELTA_L2 = 25.0` unified across router/orchestrator/CLI. |
| `#925` (`v3.4.2` + `LL.md`) | **Merged** | Documents GPU floor rules, `T4_CONTRIBUTIONS_STOPPED`, and parking of `Crowd-v1` MoE upcycle (`PARK_CROWD_V1_MOE_UPCYCLE`) & T4 co-training (`PARK_T4_COTRAINING_WITH_INFERENCE`). |
| `#926` (`--cuda-mem-cap-mib` & `rounds.jsonl`) | **Open PR** | Hard `set_per_process_memory_fraction` cap + per-round flushed `rounds.jsonl` so watchdog `SIGTERM` never drops round metrics (`LL.md` lessons 10–11). |
| `#927` (`feat/opentrainer-turnkey-train-mode`) | **Open PR (this batch)** | Adds `--train` execution + pinned multi-key Crowd-v1 check to `hivemind_opentrainer_agent.py` & fixes `/join` copy-paste command (`IP-298`). |
| `#928` (`fix/crowdgpt-fail-closed-safetensors-keys`) | **Open PR (this batch)** | Fails closed (`RuntimeError`) in `gcp_t4_crowd_diloco_worker.py` when `0` tensor keys match (`IP-294`). |

---

## 2. Measured Weekend Reality (2026-11-14 to 2026-11-16 05:55 UTC)

- **Upstream `Vxtzq/Crowd-v1`:** `12` applied rounds (`368–379`), **`3.576B` applied tokens** across `916` deltas (`41.30B` cumulative tokens, `122` registered users, `17` active GPUs at `05:55 UTC`). Our T4 landed **5 completed chunked uploads** (`868–949 MB` each) on Nov 14 before stopping (`T4_CONTRIBUTIONS_STOPPED`).
- **T4 Experimental OpenDiLoCo (`gcpcrowd/` + `gcpcrowd_moe/`):**
  - Pre-fix `CROWDGPT-GCP-T4-DILOCO`: `111` logged rounds loaded `0` matching keys -> stuck at `eval_loss = 11.929456` (`gcpcrowd_uninit_backup_r1_114/`).
  - Post-fix `CROWDGPT-MOE-8X-UPCYCLE-V1` (pinned `58a33099424e`): Rounds `1–6` accepted (`3.6703 -> 0.5689`, `-84.5%` on the 2-window probe), then **Rounds `7–17` held identically at `0.5692452`** (`ratchet=False`, `outer_delta_l2 ≈ 38.0`).
  - Nocturnal `run_20261115T201714Z_night_full`: saved `250` checkpoints (`20:18Z–02:00Z`) cleanly under the `4,300 MiB` floor until another cron job (`02:00 UTC` `AutoevolveAI/scripts/run_cron_dream.sh`) dipped free VRAM to `365 MiB` (`02:02:12Z` watchdog exit). All `250` checkpoints matched Round 1 (`L2 = 101.891411`, `diff = 0.0`) because every candidate step missed the monotone ratchet threshold.
- **Community Adoption on HiveMind:** `0` new external `trainer` accounts over the weekend (`86` accounts total, `0` `trainer`); `2` OpenRouter inference requests (`953` tokens) in the last 24h; `3` live fleet endpoints across `2` physical GPUs (T4 + owner RTX 2070).

---

## 3. Track A — Tasks for Other Claude Sessions

| IP-id | R/A | Area | Finding | Definition of Done | Negative control | Proposed PR |
|---|---|---|---|---|---|---|
| **IP-293** | R | `swarm` / `scripts` | `src/hivemind/swarm/crowdgpt_moe_opendiloco.py:1123-1135`: when `accepted_by_loss_ratchet` is `False`, `step_decoupled_round` rolls back `flat_master_cpu` and leaves `round_index` unchanged, with no feedback to advance the training shard window or decay `lr`/`outer_lr`. Re-running with the same seed/slice reproduces the identical rejected step indefinitely (`Rounds 7–17` stuck at `0.5692452`; `Rounds 1–250` of `night_full` stuck at `L2=101.891411`). | On ratchet rejection: (a) advance shard window offset or step `lr`/`outer_lr` decay (e.g. `×0.5`), and (b) stop/backoff cleanly after `K` consecutive rejections (default `3`) with explicit `ratchet_stall_reason` in `rounds.jsonl`. Also verify LoRA `delta_l2` calibration against `DEFAULT_MAX_DELTA_L2 = 25.0` (`src/hivemind/router/opentrainer.py:127`) since T4 MoE rounds measured `L2 = 32.9–43.2`. | Synthetic 5-round test where Round 2+ overshoots: `lr` halves and consecutive-rejection counter trips cleanly instead of repeating identical rejected weights 5 times. | `PR-IP-34` |
| **IP-294** | R | `scripts` | `scripts/gcp_t4_crowd_diloco_worker.py:131-180,248` (`origin/main`): `_stream_safetensors_no_mmap` skips unmatched keys silently and never raises when `0` keys match (`model_bf16.safetensors` single-key raw blob produced `111` `eval_loss=11.929` rounds). | Track `matched_keys` and raise `RuntimeError` if `0` keys match; prefer multi-key (`>= 2` header keys) cached files when resolving weights. | Load a 1-key `{"weights": ...}` safetensors file -> raises `RuntimeError("Refusing uninitialized Crowd-v1 load")`. | Closed in PR `#928` (`PR-IP-35`) |
| **IP-295** | R | `router` | `src/hivemind/router/gateway.py:3933-3937` & `3112`: `/v1/hivemind/opentrainer/moe-self-training` still reports `latest_eval_loss: 11.929456` from pre-fix uninitialized `CROWDGPT-GCP-T4-DILOCO` rounds alongside `CROWDGPT-MOE-8X-UPCYCLE-V1`, and `/v1/hivemind/t4/status` returns `telemetry_status: "NO_TELEMETRY_RECEIVED"` with static `vram_used_mib: 4841` fallback literals even during live runs. | Quarantine/filter invalid pre-fix (`CROWDGPT-GCP-T4-DILOCO` uninitialized) telemetry when summarizing `CROWDGPT-MOE-8X-UPCYCLE-V1` (expose per-campaign summaries instead of mixing campaigns), and push live T4 NVML telemetry timer to `/v1/hivemind/t4/telemetry`. | Submit a `CROWDGPT-GCP-T4-DILOCO` `11.9` outlier and a `CROWDGPT-MOE-8X-UPCYCLE-V1` `0.57` receipt: `moe-self-training` reports per-campaign series without showing `11.9` as latest loss for `UPCYCLE-V1`. | `PR-IP-36` |
| **IP-296** | R | `ops` / `scripts` | `LL.md` lesson 10 + `scripts/crowdgpt_t4/run_estp.py:12-23,36-43`: other projects' GPU cron jobs on T4 (`0 2 * * * AutoevolveAI/scripts/run_cron_dream.sh`) caused the `02:02Z` `365 MiB` floor dip, while `/etc/cron.d/hivemind-crowdgpt-intense` kept launching 15-min `CROWDGPT-MOE-8X-UPCYCLE-V1` ticks even after `LL.md` documented all T4 training stopped (`2026-11-15 ~19:45 Paris`). `estp_queue.jsonl` also targets `2026-11-16 07:00 CET`. | Add a shared cross-project NVML free-VRAM preflight guard (`/mnt/disks/disk-socrateai-local-1/hivemind_telemetry/gpu_lease` or `flock`) checked by *every* T4 GPU launcher (`AutoevolveAI`, `crowdgpt_t4`, `hivemind`), and reconcile `/etc/cron.d/hivemind-crowdgpt-intense` state explicitly with owner policy in `LL.md`. | Attempt to launch a second GPU worker when free VRAM minus estimated peak `< 4,300 MiB`: refuses cleanly with exit code `3` before touching CUDA. | `PR-IP-36` |
| **IP-297** | R | `research` | `crowdgpt_exp/results/p1/SUMMARY.md` & `criteria.json`: Weekend LLM-JEPA pilot (`LoRA r=64`, `Crowd-v1 rev 5ac3916`, `NL-RX-SYNTH`, 3 seeds) failed `c2_success` (`16.90%` vs `17.07%` SFT; `far_loss +0.897 nat`; cosine collapsed `0.68 -> 0.0004`). `E-STP` queue (`run_estp.py`) starts `>= 07:00 CET` only under `HOLD`. | Execute `E-STP-1` / `E-STP-2` (`Crowd-v1` vs `Qwen3-0.6B`) strictly when `HOLD` is active (never competing with inference), log metrics to `results/estp/SUMMARY.md`, and keep negative controls visible. | `run_estp.py` invoked without `HOLD` file exits/waits without allocating CUDA memory (`free_mib` unchanged). | `PR-IP-38` |
| **IP-300** | R | `mvp` / `boinc` | `docs/MVP_SCORECARD.md:5-18`: `3/11` passing. `B3` (`train-to-serve gate`, `servable=False`, `gain=-0.00403`) failed for BOINC Granite (`hmq_00147`), `C1`, `C2`, `E1`, `E2` still fail or `NOT-RUN`, and `UPCYCLE-V1` (`CROWDGPT-MOE-8X-UPCYCLE-V1`) has only been evaluated on its internal `2`-window held-out probe, not an independent paired base-vs-MoE benchmark. | Run an independent held-out paired evaluation (`N >= 200` chunks, McNemar / paired bootstrap CI) comparing pinned base `Crowd-v1` vs `CROWDGPT-MOE-8X-UPCYCLE-V1` Round 6 checkpoint; wire `C2` & `E2` receipts to `scripts/mvp_scorecard.py`. | Unpromoted checkpoint (`gain <= 0`) is refused by the train-to-serve router gate (`servable=False`). | `PR-IP-39` |

---

## 4. Track B — Tasks for End-Users & Volunteer Onboarding

| IP-id | R/A | Area | Finding | Definition of Done | Negative control | Proposed PR |
|---|---|---|---|---|---|---|
| **IP-298** | R | `onboarding` / `scripts` | `src/hivemind/router/join_page.py:318-321` (`origin/main`): `/join` Section 3b instructed end-users to run `hivemind_opentrainer_agent.py ... --submit-delta --once` without `--train` (errors immediately without a pre-existing `--round-report`) and stated `"Night 3 runs Monday 2 November 22:00 to Tuesday 07:00 Paris"`. Zero external `trainer` accounts joined during the weekend. | Provide working `--max-vram-gb 5.5 --train --once` flow in `hivemind_opentrainer_agent.py`, `<14 GB` desktop `1536 MiB` reserve default in `hivemind_opentrainer_worker.py`, and accurate `/join` copy. | `agent.main(["--train", "--small-arch", "--cpu-only", "--offline", "--once"])` produces real `.safetensors` + JSON report and submits matching SHA-256 delta; missing cache fails closed (`rc == 2`). | Closed in PR `#927` (`PR-IP-37`) |
| **IP-299** | R | `onboarding` / `mvp` | `docs/MVP_SCORECARD.md:17` (`E1`: `"One-command onboarding; GPU yielded to the owner within seconds"` = **`NOT-RUN`**); end-users on Windows/Linux gaming PCs still face manual `git clone`, `uv`/`venv` setup, and separate preflight/register/heartbeat steps (`LL.md` lesson 7: ~1,000 clones, 0 outside issues/PRs). | Deliver a 3-command volunteer quickstart (Docker image or `uvx`/one-shot bootstrap script covering GPU preflight -> `register` -> `--train`/`--crowdgpt` with game yield `< 1.5s`) plus an automated `E1` verification run producing `artifacts/mvp/E1_*.json`. | Simulate game process / inference load while volunteer worker runs on an 8 GB GPU: worker yields VRAM in `<= 2.0s` and writes passing `E1` JSON receipt. | `PR-IP-37b` |

---

## 5. Immediate Execution Order for Claiming Sessions

1. **Merge in-flight safety foundations first:** PR `#926` (`--cuda-mem-cap-mib` + `rounds.jsonl`), PR `#927` (`--train` & `/join` fix), and PR `#928` (fail-closed safetensors key-match check).
2. **Stop wasting GPU cycles on deterministic ratchet retries (`IP-293` / `PR-IP-34`)** and add the cross-project T4 lease lock (`IP-296` / `PR-IP-36`) before starting new nightly MoE or `E-STP` queues.
3. **Ship `E1` 3-command end-user onboarding (`IP-299` / `PR-IP-37b`)** before any next community announcement, and measure `B3` paired gains (`IP-300` / `PR-IP-39`) before promoting any trained adapter to live routing.
