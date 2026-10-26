# Acceptance criteria — real BOINC client on the T4 (written before the first run)

Scope: an engineering integration test, not a scientific hypothesis. It is **tier 3** ("real-client end-to-end") of
three honestly separated tiers:

1. unit tests (`tests/test_boinc_real_client.py`): XML builders, upload protocol, validator bridge, credit;
2. standalone app run: the GPU app executed directly (no BOINC), deterministic across two runs;
3. real-client end-to-end: a stock `boinc` 7.18.1 client (Ubuntu package, unpacked, private data dir, no system
   daemon) talks to the local project server (`project_server.py`, 127.0.0.1 only), on this VM's Tesla T4.

What is **not** claimed: this is not the production BOINC server (no MySQL/Apache/daemons, no code signing; the app
is delivered through the client's *anonymous platform* mechanism), not a public project, not a volunteer network.

| ID | Criterion (all must hold for "PASS") |
| --- | --- |
| A1 | The client's own startup log reports an NVIDIA GPU `Tesla T4` and CUDA (captured verbatim). |
| A2 | The client's scheduler request carries that GPU in `<coprocs>` (captured by the server) and the server only hands out work then. |
| A3 | A genuine workunit is downloaded by the client over HTTP, run by the client on the GPU (`device_name == Tesla T4`), and its output is uploaded through the upload-handler protocol. |
| A4 | The result is reported with `exit_status 0`, acknowledged by the server, and **validated**: same-class (GPU) replay is bit-exact and the CPU replay is within 1 Q16 unit. |
| A5 | Credit is granted through `verified_credit` (cobblestones > 0, `capped_by == "verified"`). |
| A6 | Red-team control: a workunit whose command line makes the app report a perturbed adapter (`--inject-fault`) is **rejected** and earns 0 credit. |
| A7 | Preemption: killing the client mid-task and restarting it resumes from the app checkpoint (`resumed_from_step > 0`) and the final result is bit-identical to an uninterrupted run of the same workunit. |
| A8 | The T4 is idle (no BOINC or app process) after the test; no system service was installed. |

A failure of any row is reported as such, with the log, not adjusted away.
