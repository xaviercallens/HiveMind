# Identification of the `:32` Scripted Chat Pair at `:15` / `:45` Windows (#590, #588)

## Summary of Finding

The two `POST /v1/chat/completions` requests observed in Cloud Run logs at `:15:32Z` and `:45:32Z` on both `hivemind-at-home-ruche` (`ruche`) and `hivemind-openrouter` (`router`) come from `scripts/live_user_monitor.py:run_probes` itself, forwarded through `deploy/ruche_router_overlay/server.py:forward_to_router`:

1. **Why `:32` of every `:15` / `:45` slot?**
   - `scripts/live_user_monitor.py` starts at `:15:00` and `:45:00`.
   - `main()` first executes `traffic_summary(since)`, which invokes `gcloud logging read ... --limit 5000 --format=json` and takes ~30–32 seconds.
   - `run_probes()` therefore begins its sequential loop over `SERVICES = ("ruche", "tribute", "router")` at `:15:32` / `:45:32`.
2. **Why do both `ruche` and `router` log the exact same pair from `python-httpx/0.28.1` at `:32`, followed 20–35 seconds later by the remaining 13 probe requests?**
   - Cloud Run's `httpRequest` log entry records `timestamp` at **request arrival**.
   - On the first service (`ruche`), `deploy/ruche_router_overlay/server.py:75-86` proxies `/v1/*` unchanged to `router` via `httpx.AsyncClient` (`python-httpx/0.28.1`).
   - Case 1 (`model="hivemind/bee-stars"`, `stream=False`) arrives at `:32.0`, returns a single JSON completion (~1.1 kB) in ~0.4 s.
   - Case 2 (`model="hivemind/coder"`, `stream=True`) arrives immediately after at `:32.4`. Because `stream=True` emits 25–30 Server-Sent Events (`data: {"choices":[{"delta":{"content":...}}]}\n\n`), the HTTP response size is 1,644–4,580 bytes, and when the coder tier is cold or contending on the T4 it takes 5–34.5 s (`19.2 s` at `13:45:32Z`, `33.2–34.5 s` at `13:15:32Z`).
   - Because `run_probes()` is synchronous (`for svc, base in services.items(): for model, ... in cases:`), it blocks on Case 2 for those 19–35 seconds before sending Case 3 (`hivemind/auto`), the `404` and `401` controls on `ruche`, and the 10 subsequent requests on `tribute` and `router` (which therefore arrive at `:52`–`:07`).

## Monitor Separation Contract

`scripts/live_user_monitor.py` now:
1. Tags all monitor probe requests with `User-Agent: hivemind-live-user-monitor/1.0 (python-httpx)` (`PROBE_USER_AGENT`).
2. Separates log-derived chat statistics via `split_probe_and_user_chats()` into `non_probe` (`other` / `users`) and `probe` groups (`chat_non_probe`, `chat_probe`, `chat_by_source`).
3. Reports per-tier probe latency (`latency_by_tier`) directly from `run_probes()`.
4. Prints `users chat 0 (no p95 for users)` when a window has zero non-probe chats.
