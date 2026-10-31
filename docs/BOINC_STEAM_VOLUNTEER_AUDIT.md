# BOINC and Steam volunteer path: audit of 2026-10-31

What was checked, what was found, what was changed, and what was **not** verified. Written after a request to "ensure the BOINC and Steam fat clients work, run intense tests, and improve the documentation" of the join page's `#train` section.

## 1. Verified working (against the real thing)

| Claim | How it was checked | Result |
|---|---|---|
| The La Ruche coordinator is online and serves its public pages | `GET /`, `/night-moe`, `/boinc/status`, `/community`, `/api/v1/federated/ledger` | all HTTP 200 |
| A real BOINC client accepts the project's configuration | throw-away BOINC 7.18.1 client (own data directory, no account, nothing attached), `boinccmd --get_project_config <url>` | the client fetched `get_project_config.php` and parsed name, username login and minimum password length |
| Stock BOINC settings keep the GPU out of a player's way | `scripts/boinc_volunteer_config_e2e.py`, one fresh real client per scenario | 6 of 6 checks; see below |
| The listed game stops GPU work and the work resumes when it exits | same script, a process named like the game | "suspended: an exclusive app is running", then "not suspended" after it exits with an unlisted program still running |

Measured detail worth knowing: **on Linux the client sees process names cut to 15 characters.** A 16-character game name listed in full never matched; its 15-character form did. `hivemind_volunteer_config.py` therefore writes both forms.
The activity rule (`run_gpu_if_user_active=0`) was exercised when the client considered the host "in use"; on a headless host that state follows terminal activity, so the check asserts it only when it occurred (the report says `activity_rule_exercised`).

## 2. Defects found in the idle-GPU helper (`scripts/hivemind_boinc_idle_worker.py`) and fixed here

1. **Unmeasurable input counted as idle.** With no `xprintidle` (Wayland, minimal Linux, headless) or a failed call it answered 10 s, so the helper kept working **during a game**. Now unknown means "the user is active"; a dedicated, never-played machine can opt back in with `--assume-idle-when-unmeasurable`.
2. **An invented GPU.** When `nvidia-smi` failed it returned a made-up "NVIDIA GeForce RTX 2070, 8 % load, 46 °C", which hid the failure and made the PC look idle. Now it reports `available: false`, busy and hot, and the helper does not start. `[N/A]` fields are no longer filled with made-up numbers.
3. **A pause that did nothing, reported as done.** Without `boinccmd` it returned "paused". Now it says `boinccmd_not_found` (or `boinccmd_failed_rc<N>`).
4. **A made-up idle time sent to the hub.** The reported `gpu_idle_seconds` was raised to threshold + 0.5 s whenever the helper ran. Now it is the measured value.
5. **A test that passed on fake data.** `test_rtx_gpu_status_and_telemetry` required "NVIDIA" in the card name and so failed on a real Tesla T4 while passing on the invented card. It now skips when no card is readable and accepts real names.

Tests: `tests/test_idle_worker_safety.py` (10 tests; 8 of them fail on the original script, checked) and the repaired `tests/test_rtx_boinc_and_steam_fat_client.py`.

## 3. Claims on the join page that the code or a third party contradicts (corrected on the page)

* **"Free-to-Play Steam Edition, Steam AppID 3133700, available today."** Steam's public store API: AppID 3133700 is *Fantasy Grounds - Raiders of the Ghost Zone*, a DLC by SmiteWorks USA. There is no HiveMind game on Steam, and the repository contains no Steam client binary (the `--self-test` fat-client test is skipped for that reason). The page now says so. `docs/STEAM_STATUS.md` already said it; the page contradicted the repo's own document.
* **"Fold TESS/JWST light curves."** The helper folds a 16,384-point **synthetic** array with a dip injected at fixed positions, and its kernel "recovers" that same dip. It is a self-check of the GPU path, not an analysis of telescope data.
* **"Rank-4 QLoRA outer-delta SVD verification" / "compute QLoRA updates".** `compute_deterministic_qlora_outer_delta` builds its matrices from a SHA-256 of the workunit id and returns **fixed** loss numbers (hold-out 2.215 to 1.798, cosine 0.9996). No model is trained. The helper submits that result to the coordinator. The page now calls it a deterministic placeholder and the helper's own report carries `placeholder_not_training: true`.
* **"0 % GPU overhead" and "a locked 60 FPS".** Nothing measures frame rate or overhead; the tests only check that a code path allocates nothing. Removed; the page says the effect on frame rate has not been measured.

## 4. Open: needs an owner decision, not changed here

* The helper **submits the placeholder result to the production coordinator, which may credit points for it** (the page's "250 HMT and 45 Cobblestones per workunit" appears only in a launch-scaling model, `international_launch_scaler.py`, so what the coordinator really pays was not verified). Options: stop submitting until real work exists, or have the coordinator refuse results flagged as placeholders.
* The coordinator takes the `--handle` as given: nothing stops one person from submitting as many handles as they like.
* The helper switches the BOINC GPU mode itself (`never`, then `auto`), so it overrides a mode the volunteer set by hand, and a crash while paused leaves the GPU mode at `never`.
* `portal_page.py`, `steam_edition.py`, `fat_steam_astro_builder.py` and `game/__init__.py` still describe a "Free-to-Play Steam" game and the same AppID.

## 5. Not verified

Windows and macOS (the BOINC settings were tested on Linux only); any real game; frame-rate effect; attaching a real client to the project with a real account and receiving a real workunit from the scheduler (needs an account on the production project; only the configuration fetch was tested); what points the coordinator actually grants; the browser/WebGPU path.

## 6. How to repeat

```bash
python scripts/hivemind_volunteer_config.py gamer --game YourGame.exe        # prints the two settings files
python scripts/boinc_volunteer_config_e2e.py --boinc-root <extracted BOINC package> --work /tmp/vc_e2e
python -m pytest tests/test_idle_worker_safety.py tests/test_volunteer_config.py tests/test_join_page.py
```
