# T4 GPU Contention vs Reload & QLoRA Training Window Attribution (`#591`, `#589`, `#608`, `#570`, `#404`)

## 1. Slow Scripted Chats (`5–34 s`) During `86–98%` T4 Utilisation (`#591`, replaces `#589`)

### Measured Evidence
- Across the `12:45Z`, `13:15Z`, and `13:45Z` monitor windows, the slowest scripted chats (`5.3 s`, `33.2 s`, `19.2 s`; `1,644–4,580 B` SSE responses) occurred on `POST /v1/chat/completions` (`model="hivemind/coder"`, `stream=True`, 25–30 SSE chunks) while Tesla T4 GPU utilisation read **`86–98%`** and VRAM read **`10,509 MiB`**.
- Earlier windows (`06:15` and `07:00–07:30 CET`) showed VRAM dropping to **`5,049 MiB`** when Ollama evicted an idle model after its default `keep_alive` window, followed by **`10,509 MiB`** once `qwen2.5-coder:7b` was resident again.
- Therefore:
  1. In the `12:45Z–13:45Z` slots where VRAM was already `10,509 MiB` and GPU SM utilisation was `86–98%`, the `5.3–33.2 s` latency on `hivemind/coder` (`4.6 kB` streamed reply) was driven by **concurrent GPU compute contention + multi-chunk SSE generation (`25–30` chunks)** on the shared T4 (`llama-server` + `ollama` + peer probe), not a cold disk reload.
  2. In the `06:15` / `07:00–07:30 CET` transition where VRAM rose from `5,049 MiB -> 10,509 MiB`, the initial latency spike included **Ollama model reload**.

### Recorded Operator Decision (`#591`)
- Keep primary serving tiers pinned (`keep_alive: 24h`) when serving mode is active, separate monitor probe chats from non-probe user chats (`split_probe_and_user_chats`), and attribute slow chats per window via `attribute_slow_chats()` and `latency_by_model_and_producer()` in `scripts/live_user_monitor.py`.

---

## 2. Paris 21 Window: `state="training"`, `171 -> 172`, `5,201 MiB` VRAM at `0%` GPU Util (`#608`)

### 15 Training Snapshots Table (`2026-11-04` Paris 21 Window)

| Snapshot # | Governor State | GPU Util (`util_pct`) | GPU VRAM (`vram_mib`) | Total VRAM (`total_mib`) |
| :--- | :--- | :--- | :--- | :--- |
| 1–15 (all 15 samples) | `training` | `0%` | `5,201 MiB` | `15,360 MiB` |

### Attribution Statement (`#608`)
- Across all 15 `state="training"` snapshots in the Paris 21 window (`5,201 / 15,360 MiB` VRAM, `0%` GPU utilisation, `0` active `moe_app` GPU PIDs), **no GPU training kernel was executing during those 15 samples**; the `assimilated` counter step `171 -> 172` came from the BOINC assimilator (`hivemind_assimilate`) recording a completed workunit in `assimilated.jsonl` while the governor had unlocked BOINC GPU mode (`always`) with no new runnable GPU task actively occupying the SMs during the 10-second snapshot ticks.

---

## 3. Daytime (`5–8 s`) and Night (`28–57 s`) Router `p95` Attribution (`#570`, `#404`)

- **Daytime `5–8 s` `p95` (`#570`)**: Caused by the scripted monitor probe's `hivemind/coder` streaming request (`stream=True`, `1.6–4.6 kB` SSE payload, `4.8–8.1 s`) while synchronous `hivemind/bee-stars` and `hivemind/auto` probes complete in `0.22–0.73 s` (`p50 = 0.22–0.40 s`). `tribute` stays at `p95 = 0.54–0.73 s` whenever its coder stream hits a warm worker without contention.
- **Night `28–57 s` `p95` (`#404`)**: Caused by `hivemind/coder` long/streaming completions queueing behind concurrent T4 GPU load or waiting on a single producer stream (`dominant_cause = "long_generation"` when `response_bytes >= 1,500 B`, or `"slow_or_overloaded_producer"` when small responses queue behind an active generation).
- Implemented in `scripts/live_user_monitor.py` via `attribute_slow_chats()`, `latency_by_model_and_producer()`, and `attribute_t4_window_snapshots()`.
