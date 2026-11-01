# The T4 node's batch layer: restart after preemption, telemetry, and QLoRA training only when idle

Installed on 2026-10-25 at the owner's request. Inference always has priority; this layer must never slow end users down.

## What exists now

| Piece | What it does | State |
| --- | --- | --- |
| Cloud Scheduler job `hm-t4-keepalive` | every 3 min calls the Compute API `start` on this Spot VM (no-op while it runs) | **enabled**, first attempt succeeded |
| GCE `startup-script` metadata | keeps the pre-existing `systemctl restart ssh` line, then runs `/opt/hivemind/t4/bin/startup.sh` in the background | **set** |
| `bin/startup.sh` | idempotent boot orchestrator: ollama + nginx, then the inference producers, then the BOINC layer, then governor + monitor; waits until the gateway lists the tiers; logs to `hivemind_telemetry/startup.log` | installed |
| systemd units (`node/*.service`) | BOINC db / Apache (loopback) / daemons / client, governor, work generator, monitor | **installed, enabled**; db, apache, daemons, client, workgen, governor, monitor are **running** |
| `t4_monitor.py` (root) | every 15 s one JSON line (GPU, load, disk, nginx in-flight, unit states, gateway-visible tiers, governor state, QLoRA progress) in `hivemind_telemetry/t4_node.jsonl`; restarts enabled-but-down units (max 3 per 10 min); writes a `PREEMPTION_NOTICE` event the moment the metadata server announces one | running |
| `t4_governor.py` | polls nginx's loopback `stub_status` once a second; on an in-flight request it sets BOINC's GPU mode to `never` (the task stops and later resumes from its segment checkpoint) and records how long the GPU took to free; after 30 min of quiet inside the night window (22:00-07:00 Paris) or 2 h outside it (the owner's policy below; the first versions used 30 s), and only with enough free VRAM and a cool card, it sets `always`; `scripts/t4_governor_report.py` summarises what it did | running, **held** (see below) |
| `t4_workgen.py` | keeps one chained QLoRA workunit queued; each starts from the latest validated adapter (`latest_adapter.pt`), the validator requires the chain to start there | running; round 1 queued |
| Validator | its GPU replay waits for a quiet moment too (nginx idle for 5 s, max 30 min) | active |

## Policy (owner, 2026-10-25): serving first, training only after long silence, mostly at night

- The governor starts in `never` on every (re)start. Training resumes only after **30 minutes** without an in-flight request
  **inside the night window** (22:00-07:00 `Europe/Paris`, an assumption about the owner's night; `--night-tz/-start-h/-end-h`), or after
  **2 hours** of silence outside it. Any request suspends it immediately.
- `…/hivemind_telemetry/HOLD` is an operator switch: while it exists nothing trains (`touch` to pause, `rm` to allow).
- The validator's GPU replay also waits for a quiet moment.
- A `yield` log line is a real measurement only when `had_boinc_gpu_task` is true; `gpu_freed_ms` of a yield with no task is meaningless.

**How the governor decides the node is busy (two signals, polled once a second).** (1) nginx in-flight requests: the gateway is a push router (it calls the producer's
registered HTTPS endpoint; the agent only heartbeats), so router traffic does pass through nginx, and the 13:12 yield was triggered that way (`in_flight: 1`). (2) GPU
activity by anything that is not the BOINC task: whole-GPU utilisation >= 10 % when no task runs, or the highest per-process SM % among the other processes
(`nvidia-smi pmon`) while one does. This second signal exists because callers that skip nginx (local clients of Ollama on :11434, of the Bee server, of the gwaya-lite
gateway) and requests shorter than one poll would otherwise be invisible. Any trigger restarts the quiet timer; yield events record `trigger`. Still blind: a request
that starts and ends between two polls without a GPU-utilisation sample.

**VRAM at the first night run.** The governor starts training only if free VRAM (not counting what BOINC itself holds) is >= 4300 MiB, and the QLoRA app peaks at ~3.4 GiB (+ CUDA context). With the Bee server (4.8 GiB) alone resident
(Ollama unloaded after its 10-minute keep-alive, as at 19:15Z: 4,841 MiB used) about 10.5 GiB is free and training starts if the node has been silent for 30 minutes. With the 7B also
resident (10.3 GiB used) ~5 GiB is free and it still starts, leaving ~1.6 GiB. With both coders resident (~13 GiB) the guard refuses, so training does not run: the safe failure mode. A pinned
model (`keep_alive: -1` or `OLLAMA_KEEP_ALIVE=-1`) is good for latency but means Ollama never unloads, so night training would only run while the free-VRAM guard passes.

**Not measured yet (do not quote numbers):** the time from "request arrives" to "GPU free" with a task running, the effect of a
running task on inference latency, and the QLoRA loss trend over chained rounds. The first night run produces the first
real yield measurement. The 13:15Z idle latency baseline (router path, temperature 0) taken by a teammate session: bee-stars
0.35-0.51 s, coder (short) 0.63-0.81 s, auto 0.27-0.28 s, coder 320 tokens 9.8-9.9 s at ~32 tok/s.

## What the night rounds train on

One domain per round, cycling `general` (wikitext), `french` (Wikipedia fr), `edu_web` (FineWeb-Edu), `news` (CNN/DailyMail), `biomed`
(PubMed), `math` (GSM8K), all from the local Hugging Face cache (no downloads). Every domain has a fixed held-out slice that never appears in
training; each round reports the held-out loss for **all** domains, so a round trained on French also shows its effect on English and math
(`eval_by_domain_before/after`, `qlora_progress` in the telemetry). MBPP and HumanEval are cached too but are the repository's evaluation
benchmarks and are never training data. A round is 80 inner steps (~3-4 min). Each round starts from the previous validated adapter.

**More MoE models: what is true today.** `granite-3.1-3b-a800m-base` is fully cached (6.2 GB) and is the natural next model; it will not fit
beside the daytime mix (~5 GB free) but could at night once Ollama has unloaded its model (~10 GB free beside the Bee expert). That is a
different adapter lineage (adapters are model-specific) and needs a per-workunit VRAM requirement in the governor; **not built or
tested**. `OLMoE-1B-7B` and `Phi-tiny-MoE` are config-only in the cache (8 KB): they would need a download, and OLMoE's experts would not
fit this card anyway. Quality levers not used yet (each is a separate, measurable change): a learning-rate schedule across rounds,
gradient accumulation, longer sequences (VRAM-limited), expert-level quantisation.

## Known limits

- The governor sees nginx in-flight requests only: traffic that bypasses nginx (local clients of Ollama on 127.0.0.1:11434) is invisible to it.
- A yield kills the task process; work since the last segment checkpoint (up to ~30 s) is redone on resume.
- With the full inference mix resident (~10.3 GiB), one task (~3.5 GiB) fits; the validator's replay is serialised behind idle for that reason.
- The older bee/coder producers still run as session-started processes; their units are installed and enabled for the next boot, but the
  watchdog only reports them (use `--manage-legacy` after the cutover described in `../live/README.md`). A teammate session's `@reboot` cron line starts the
  same services, so the cutover (removing that line and stopping the hand-started processes) is to be done together, in a quiet window.
- The BOINC app launchers point at the `/opt/hivemind/t4/src` snapshot (app version 101); re-run `install_node.sh` after code changes.
- Everything is Spot-VM-bound: the Scheduler job restarts the VM; the services come back by the units and `startup.sh`.

## Operating

```bash
/opt/hivemind/t4/bin/boinccmd_node.sh --get_tasks          # the node's BOINC client
tail -f /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/governor.jsonl
cat    /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/latest.json
touch/rm /mnt/disks/disk-socrateai-local-1/hivemind_telemetry/HOLD   # pause / allow training
gcloud scheduler jobs pause hm-t4-keepalive --location=us-east4      # before stopping the VM on purpose
```
