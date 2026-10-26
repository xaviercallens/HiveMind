# Live end-user tests

The record of the first real-life test is `artifacts/live_tests/FIRST_REAL_USER_TEST_20261025.json`.

## What passed on 2026-10-25
- A brand-new account at the default max price makes its first request through each public service (Ruche, Tribute, router)
  to each tier (`hivemind/bee-stars`, `hivemind/coder`, `hivemind/auto`), in JSON and SSE: 17 of 18 calls served,
  answered correctly and charged in the shared ledger. The failure was a cold coder (504 after 120 s, not charged).
- The owner's first console request (`hivemind/auto`, 294 tokens, 36 tok/s, first byte 273 ms) was served and settled.
  **Its answer was wrong code**; serving works, answer quality on open coding prompts is not yet measured.

## What was fixed to get there
- 402 on a new user's first request: idle accounts that raised their max price kept the price above the default.
  The router now prices among accounts active in the last 900 s (PR #182, `HIVEMIND_BID_WINDOW_S`).
- Cold coder: the 7B coder is pinned in VRAM (`keep_alive=-1`), repeated at boot by `deploy/t4_bee_expert/start.sh`.

## Monitor
```bash
python scripts/live_user_monitor.py --key-file ~/.config/hivemind/probe_consumer_key --out <dir>/monitor.jsonl
```
Appends one JSON line per run:
- **traffic:** request-log counts for the three services (chat latency p50/p95, 402, 5xx, registrations, browser vs scripted
  callers; no IP addresses, keys or prompts);
- **probes:** each service × tier is asked a question with a checked answer, plus an unknown model (must be 404) and a
  request without a key (must be 401). Before probing, each checker must reject a known-wrong answer; if one does not,
  the run is INVALID (exit 2).

The probe account is `probe.live-monitor`. Its calls appear in the traffic counts as scripted requests.
