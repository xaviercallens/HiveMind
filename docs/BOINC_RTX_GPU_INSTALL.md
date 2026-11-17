# Installing and joining the HiveMind BOINC project with an NVIDIA RTX GPU

This guide is for a volunteer with a consumer NVIDIA RTX card (RTX 20, 30 or 40 series, 8 to 24 GB).
It is written for newcomers. It is also strict about what has been checked.

**Read this first.** The only machine this project has ever run on is one Tesla T4 (Turing, sm_75, 15 GB) under
Linux. Nobody has run the HiveMind BOINC app on an RTX card. Anything about RTX in this page is either a plain
fact taken from a repository file (cited in brackets as `(file)`) or it is marked **Not tested.**
If you join with an RTX card today, you are the first test, not a regular user.

Points and credit in this project are non-monetary accounting units. Nothing here promises a reward.

## 1. Status at a glance

| Item | Verified on | Status |
| --- | --- | --- |
| Linux + Tesla T4 (15 GB, sm_75) | The project's own T4: a stock BOINC 7.18.1 client attached to a real BOINC 7.24.4 server, with no `app_info.xml`, ran the app, uploaded, and the server validated it by replay (`docs/BOINC_PRODUCTION_SERVER.md`, rows C1, C2, C4; `artifacts/boinc_prod_server/e2e_report.json`). | Verified, on one host, one GPU, five results. |
| Self-contained app download (runtime, code, model as signed BOINC files, unpacked once) | The script `scripts/boinc_distribution_e2e.py` exists to check it on the T4. Its output file `artifacts/boinc_distribution/report.json` is **not in the repository**. | The repo does not contain a record of a passing run. Treat as **Not tested.** by this repo's files. |
| Linux + RTX 20-series | Never run. An RTX 2070 (sm_75) is named in the E12 plan and listed under `not_run` in `artifacts/experiments/E12_replay_divergence_report.json`. The RTX 2070 shown in `README.md` is an inference producer, not a BOINC training host. | Same architecture as the T4, but **Not tested.** |
| RTX 30-series and 40-series | Never run. These are different architectures from the T4. | **Not tested.** The repo does not verify that the pinned torch and bitsandbytes builds in `deploy/boinc_dist/runtime-requirements.txt` support them. |
| Windows and macOS | Never run. The app launcher is a bash script that needs `tar`, `xz` and `flock` (`deploy/boinc_dist/launcher.sh.in`). The only platform the project publishes is `x86_64-pc-linux-gnu` (`src/hivemind/boinc/real_client/server_stack.py`). | **Not supported** by the current distribution. |
| Public project URL | Does not exist yet. | Not open. The server is loopback-only; no TLS, no public hostname (`docs/BOINC_PRODUCTION_SERVER.md`, "Limits"). Making it reachable is the owner's decision (`deploy/t4_node/README.md`). |

## 2. What you are signing up for

**What the app does.** It fine-tunes a small mixture-of-experts language model with QLoRA. The model is
`ibm-granite/granite-3.1-1b-a400m-base`, a 1.3 B-parameter MoE and a base model, not a chat model
(`docs/BOINC_CORE_NODE.md`, `src/hivemind/boinc/real_client/moe_work.py`). The attention projections are 4-bit NF4, the
LoRA adapter is fp32 with rank 8 and 1,376,256 trainable parameters. The experts stay frozen in fp16
(`docs/BOINC_CORE_NODE.md`). Each night round trains one domain, cycling through `general`, `french`, `edu_web`,
`news`, `biomed` and `math`. A round is 80 inner steps and takes about 3 to 4 minutes on the T4
(`deploy/t4_node/node/README.md`). Each round starts from the previous validated adapter.

**Download size.** Not stated by the repo. The app version has four files: a launcher, a Python runtime archive, a
source archive and a model archive (`scripts/boinc_publish_app_version.py`). The repo records no sizes for them. Ask the
project for the sizes before you join if your connection is metered.

**Disk.** Not stated by the repo. The archives are stored and then unpacked into `hivemind_env/` inside the BOINC
project directory, so plan for both copies (`deploy/boinc_dist/launcher.sh.in`). Unpacked versions unused for a week are
deleted by the launcher (same file).

**GPU memory.** Measured on the T4: the training app peaked at 3,409 MiB (40 steps, run directly,
`docs/BOINC_CORE_NODE.md`). The server schedules the standard app on cards with at least 3,500 MB
(`src/hivemind/boinc/real_client/moe_profiles.py`, key `min_gpu_ram_mb`). The node's own rule is "start only with
at least 4,300 MiB free" (`deploy/t4_node/node/README.md`). A second, larger profile exists in code (rank 16, sequence
length 512, plan class `cuda_hivemind_big`, at least 10,000 MB) (`moe_profiles.py`). No run of it is recorded in the
repo, so its memory use is **Not tested.**

**When it runs.** Community work is only offered in the European night, 22:00 to 07:00 `Europe/Paris`
(`src/hivemind/boinc/real_client/contributor_policy.py`). That window is a project policy about the shared server GPU. It
is not tied to your local time zone. On the project's own T4 node, a separate governor suspends training whenever an
inference request arrives and resumes it after 30 minutes of quiet in the night window
(`deploy/t4_node/node/README.md`). **That governor is the owner's node software. The repo does not state that your
machine gets any such idle detection.** What your stock BOINC client does when you use the PC (pause when you are
active) depends on your BOINC preferences. The test setup in the repo allowed running while the user was active
(`run_if_user_active` and `run_gpu_if_user_active` set to 1 in `client_setup.py`), so check your own preferences.

**Checking of results.** Every result is replayed by the server. A new contributor's result is replayed in full, every
segment, until the host has 5 validated results (`contributor_policy.py`, `DEFAULT_PROBATION = 5`; community mode is
switched on by `HIVEMIND_COMMUNITY=1`). A result that differs from the replay by more than 1e-4 is rejected with zero
credit (`docs/BOINC_CORE_NODE.md`, row B4). A rejected result may be retried by BOINC (`docs/BOINC_PRODUCTION_SERVER.md`,
finding 8).

**Nothing system-wide.** The launcher says it installs nothing system-wide and fetches nothing from the internet at run
time (`deploy/boinc_dist/launcher.sh.in`). It unpacks into `<project dir>/hivemind_env/`. It sets
`HF_HUB_OFFLINE=1` and ignores your own Python packages.

**What leaves your machine.** From the code: the task's result files only. The T4 runs uploaded `out.json` (938 B) and
`out.pt` (27.9 MB), the adapter weights (`docs/BOINC_CORE_NODE.md`, row B1). The standard BOINC client also sends
the usual scheduler request: host and GPU description (the log shows the T4 in `<coprocs>`,
`docs/BOINC_REAL_CLIENT.md`). The repo has not been audited for anything else, so **Not tested.** beyond these.

## 3. Before you start

* An NVIDIA driver that supports your card. The exact minimum driver version is not specified by the repo.
* Run `nvidia-smi`. It should list your card and show the memory in use. Write down the card name, total memory and
  driver version for the checklist in section 6.
* Free GPU memory: close games, video editors and anything else using the GPU. The app needs about 3.4 GiB for itself
  on the T4 (section 2). Not measured on any other card.
* Free disk: see section 2. The size is not stated.
* Linux with `tar`, `xz` and `flock` (util-linux), `x86_64` (`deploy/boinc_dist/launcher.sh.in`,
  `server_stack.py`).
* Optional check tool: a volunteer preflight is proposed in PR #255 (not merged yet). It adds
  `scripts/hivemind_volunteer_preflight.py` and `docs/community/VOLUNTEER_PREFLIGHT.md`. Those files are not in `main` and
  are not described here. Until it is merged, do the checks above by hand.

## 4. Install (Linux)

These are the steps the repo documents for a stock BOINC client. They were exercised on the T4 only. The steps with
no file behind them are marked.

1. **Install the BOINC client from your distribution.** The tests used Debian/Ubuntu `boinc-client` 7.18.1, unpacked
   without a system service (`docs/BOINC_REAL_CLIENT.md`, "How to reproduce"). Running it as a normal service from your
   distribution package is the usual way, but this repo did not test that. **Not tested.**
2. **Get the project URL.** It is not public yet. Where it would go, this guide writes
   `<PROJECT_URL (not public yet)>`. The server listens on `127.0.0.1:8081` only, has no TLS and no public hostname.
   Exposing it is the owner's step (`docs/BOINC_PRODUCTION_SERVER.md`; `deploy/t4_node/README.md`).
3. **Create an account on the project's web page.** The test used the server's `create_account.php`, which returns an
   authenticator (`docs/BOINC_PRODUCTION_SERVER.md`, row C1).
4. **Attach the client to the project.** The tests wrote an account file for the client with the master URL and the
   authenticator (`prepare_data_dir(..., install_apps=False)` in `src/hivemind/boinc/real_client/client_setup.py`). With
   no `app_info.xml`, the client asks the project for its signed app versions itself. For a normal install, use the BOINC
   Manager's "Add project" with `<PROJECT_URL (not public yet)>`, or `boinccmd --project_attach URL AUTHENTICATOR`.
   Those two forms are ordinary BOINC features and are not used in this repo. **Not tested.**
5. **Beta flag.** New app versions are published as beta. Only accounts with `allow_beta_work` in their project preferences
   receive them until a version is promoted (`scripts/boinc_publish_app_version.py`; the distribution test sets
   `allow_beta_work` for its account). Ask the project whether the current version is promoted.
6. **The downloads.** The client downloads the launcher and three archives through the project's normal download URL.
   They are signed by the project's key (`boinc_publish_app_version.py`, via BOINC `update_versions`). The file names
   contain a content hash, so an unchanged runtime or model is not downloaded again after a code-only update.
7. **First task start.** The launcher unpacks the three archives once into `<project dir>/hivemind_env/`, under a lock,
   atomically (`launcher.sh.in`). If unpacking fails it prints `hivemind: unpacking <file> failed` and exits 195
   (see section 7).
8. **The runtime that is unpacked.** A relocatable CPython 3.10 with pinned libraries, including `torch==2.13.0` and
   `bitsandbytes==0.50.2`, built from the CUDA 11.8 wheel index (`deploy/boinc_dist/runtime-requirements.txt`,
   `scripts/boinc_build_distribution.sh`). Whether that build runs on your card is **Not tested.** (section 5).

**Check that the task runs.** The repo reads the client's state with `boinccmd`. On a default install run
`boinccmd --get_tasks`. For a client with its own data directory and port, `client_setup.py` runs
`boinccmd --host 127.0.0.1:PORT --passwd PASSWORD ...` with the password from `gui_rpc_auth.cfg`. Look at the task name
(`hmq_...`) and its state. The test scripts also use `--get_state`, `--get_cc_status`, and
`--project MASTER_URL update` to ask for work now (`scripts/boinc_core_node_e2e.py`, `scripts/boinc_distribution_e2e.py`).
In the project directory of the BOINC data folder you should see the downloaded `hivemind_*` files and a
`hivemind_env/` folder (`boinc_distribution_e2e.py` lists exactly these).

## 5. RTX-specific notes and risks

**(a) Library versions are pinned because results are compared to about 1e-4.** The comment at the top of
`deploy/boinc_dist/runtime-requirements.txt` says: "Pinned to exactly what the production node validates with (replay
validation compares to ~1e-4, so library versions must match the server's)." The launcher sets
`PYTHONNOUSERSITE=1` for the same reason (`launcher.sh.in`). Do not replace torch or bitsandbytes with your own build.
Note: the pin was raised from `torch 2.7.1+cu118` to `2.13.0` by a dependency bump (git commit `5cf111f`). The E12 report
below ran on `torch 2.7.1+cu118`. This repo does not state which torch version the T4 node validated with after the bump.

**(b) The repo's own measurement of cross-device divergence (E12).** In `artifacts/experiments/E12_replay_divergence_report.json`
(hardware: Tesla T4, torch 2.7.1+cu118, CUDA 11.8; `eps` = 0.0001; 50-step segments, 9 per cell):

| Cell (field in `summary`) | What was compared | max `linf` | Segments above eps |
| --- | --- | --- | --- |
| `B_separate_process_T4` | T4 replay vs T4 replay in a separate process | 0.0 | 0 of 9 |
| `C_cross_T4_vs_cpu` | fp32 T4 vs this host's x86 CPU | 0.000626 (min 0.000205) | 9 of 9 |
| `D_proxy_fp64_snap_T4_vs_cpu` | fp64 plus 2^-20 snapping proxy | 0.000634 (min 0.000198) | 9 of 9 |

So the same T4 reproduced itself exactly, and a T4 against a CPU class diverged by 2e-4 to 6e-4 (rounded from the
`min_linf` and `max_linf` fields), above the 1e-4 rule, in all 9 segments. The research log words it this way:
"same-T4 exact, every class crossing > ε, fp64+snap proxy does not help" (`docs/RESEARCH_PROGRAM.md`, status log of
2026-10-24). The pilot (`artifacts/experiments/E12_pilot_report.json`) also shows a CPU-vs-CPU build difference of
1.5e-4 (`remote_cpu.linf_vs_ref`, `passes_eps: false`) and a bf16 cell at 3.3e-3 (`bf16_autocast.linf_vs_ref`).

What this means for you: the server validator replays your segment on its own GPU (a T4) and requires a match
within 1e-4 (`src/hivemind/boinc/real_client/moe_bridge.py`). An RTX card is a different execution class. The repo has
**not** measured whether RTX results replay-validate against the T4 server. The E12 report lists the RTX 2070 under
`not_run`. The research log says validators are assigned by accelerator class and cross-class replays are "not
comparable" (`HM-S10-08`, `docs/RESEARCH_PROGRAM.md`), but this repo does not show that rule in the BOINC validator
code. So the first RTX volunteers are an experiment. A result may be rejected for a reason that is not your fault.
**Not tested.**

**(c) Torch and bitsandbytes per architecture.** The runtime is built from the CUDA 11.8 wheel index with
`bitsandbytes==0.50.2`. Whether those builds run NF4 4-bit on Turing RTX, Ampere (RTX 30) or Ada (RTX 40) is not verified
by this repo. Only sm_75 on the T4 has run. **Not tested.**

**(d) Laptop GPUs and heat.** General guidance, not measured here: laptop GPUs often have lower power limits and throttle
when hot, which makes a task slower; a long task on a hot laptop may be paused by the client's own temperature
settings. A task that runs a few minutes per round on a T4 (section 2) may take longer. The repo has no laptop run.

**(e) 8 GB cards.** The measured peak (3,409 MiB) is well under 8 GB, but it was measured on a 15 GB T4 with other
processes on the card, and never on an 8 GB card. Windows display drivers also reserve memory. The code constant
for the Windows reserve is 6,656 MiB (`src/hivemind/core/qwen35_usage_moe_orchestrator.py`), but this guide
covers Linux only. Status: **Not tested.** on 8 GB.

The planned plan classes `std_8gb`, `big_12gb` and `xl_16gb_plus` (issues #238 to #244) are planned. A profile table
for them exists in `qwen35_usage_moe_orchestrator.py`, but the BOINC server stack only registers the plan classes in
`moe_profiles.py` (`cuda_hivemind` and `cuda_hivemind_big`). Treat the 8/12/16 GB split as **not implemented**, and a
different model family (Qwen) as **Not tested.** for BOINC.

## 6. First RTX machine validation checklist

Run this on the first RTX machine and send the answers to the maintainers. Do not skip a step that fails: report
which step it was. The maintainers use the answers to decide whether RTX is supported.

1. **Hardware facts.** Card name, total VRAM, laptop or desktop, CPU model, RAM, free disk.
2. **Driver.** Output of `nvidia-smi` (the whole top table).
3. **OS.** Linux distribution and version, `uname -m` (must be `x86_64`), and `tar --version`, `xz --version`,
   `flock --version`.
4. **Free memory.** `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` before you start. Close other GPU
   programs.
5. **BOINC client.** `boinc --version` (the tests used 7.18.1), and how you installed it.
6. **Attach.** Did the client attach to the project? Quote the line from the client log
   (the repo's tests keep it in `client_stdout.log`; on a normal install it is the BOINC client's own log, whose location this repo does not document, so say where you found it). Did it detect your GPU? The T4 run logged a line of the form `CUDA: NVIDIA GPU 0: Tesla T4 (...)` (`docs/BOINC_REAL_CLIENT.md`).
7. **Download and unpack.** Did the four `hivemind_*` files download? Did `hivemind_env/` appear in the project
   directory? If not, which step failed?
8. **Task state.** `boinccmd --get_tasks` while the task runs and after it ends. Report the task name, state and
   exit status.
9. **Peak VRAM.** While the task runs: `nvidia-smi --query-gpu=memory.used --format=csv -l 5` and report the highest
   value. Compare with 3,409 MiB (T4).
10. **If it failed:** the tail of the task's `stderr` (the last 50 lines). The place it is stored depends on your BOINC
    install, so give the path you found; do not guess.
11. **Validation outcome.** Did the project show the result as valid or invalid? For a rejection, give the reason text the
    project shows. The validator's verdict text includes `max_diff` (for example `verified segment k/K: max_diff=...`,
    `src/hivemind/boinc/real_client/moe_bridge.py`), so report it if the project shows it. If you cannot see it, say so.
12. **Time.** How long did the task take, and did you see throttling (`nvidia-smi -q -d PERFORMANCE`)?

## 7. Troubleshooting

Only failures documented in the repo are listed.

* **The task restarts again and again** ("exited with zero status but no 'finished' file"). The app must leave the
  `boinc_finish_called` file. Fixed in the app after the first test run (`docs/BOINC_REAL_CLIENT.md`, finding 1).
* **`hivemind: unpacking ... failed`.** `tar` could not unpack an archive. The launcher removes the partial folder and
  stops (`deploy/boinc_dist/launcher.sh.in`). Check free disk and that `tar` and `xz` exist. Exit code 195 also means
  the unpack directory could not be created or locked.
* **Task stops when you use the GPU or when the node is busy.** On the project's node, the governor sets the BOINC GPU
  mode to `never`, the task stops, and it later resumes from its last segment checkpoint. Up to about 30 seconds of work
  are redone (`deploy/t4_node/node/README.md`, "Known limits").
* **Killed or crashed mid-task.** The T4 runs resumed from the last checkpoint after a SIGKILL, with a result identical to an
  uninterrupted run (`docs/BOINC_CORE_NODE.md`, row B5).
* **Out of GPU memory.** The repo documents the guard, not an error text: the node refuses to start training when less
  than 4,300 MiB is free (`deploy/t4_node/node/README.md`). Close other GPU programs. Behaviour on an actual OOM on an
  8 GB card is **Not tested.**
* **A result is rejected.** A result that differs from the replay by more than 1e-4 gets 0 credit, and BOINC may retry
  the work (`docs/BOINC_CORE_NODE.md`, B4; `docs/BOINC_PRODUCTION_SERVER.md`, finding 8). See section 5(b) for why an
  RTX result might be rejected without any fault of yours.
* **The client says "Couldn't resolve host name" first.** Seen once on the T4 stack because of a placeholder entry in the
  server's scheduler list. The client fell back to the right one. A server-side issue (`docs/BOINC_PRODUCTION_SERVER.md`, finding 6).
* **Credit looks high or low.** BOINC's native credit and the project's `verified_credit` disagree by an order of
  magnitude (267.7 against 56.9 for the same workunit). This is a known open point (see section 9).

## 8. Privacy and safety

* **What the server sees.** Your BOINC account, host description (including the GPU), and the result files you upload
  (section 2). The repo has not audited more than that. **Not tested.**
* **Your network.** The project's own server and the T4 node bind to `127.0.0.1` (`docs/BOINC_PRODUCTION_SERVER.md`,
  C8: no non-loopback listener). Your client makes outgoing connections to the project. The repo does not say that the
  client opens an incoming port; the stock client's own GUI RPC port is local and password-protected
  (`client_setup.py` writes `gui_rpc_auth.cfg`).
* **Signed files.** The launcher and the three archives are signed with the project's code-signing key by BOINC's
  `update_versions` (`boinc_publish_app_version.py`). The signing private key currently sits on the same server as the
  project, which BOINC itself warns is a weakness for a public project (`docs/BOINC_PRODUCTION_SERVER.md`). TLS is not set
  up there yet.
* **Stop temporarily.** The repo uses `boinccmd --set_gpu_mode never` to make the client release the GPU at the next
  checkpoint, and `--set_gpu_mode always` to allow it again (`deploy/t4_node/README.md`; `scripts/boinc_distribution_e2e.py`).
  `boinccmd --project URL suspend` and `detach` are standard BOINC commands that this repo does not use. **Not tested.**
* **Stop completely.** `boinccmd --quit` ends the client (`client_setup.py`, `Client.quit`).
* **Uninstall.** The docs do not give an uninstall procedure. What the files show: everything this project adds is
  inside the BOINC project directory (`hivemind_*` files and `hivemind_env/`), and the launcher installs nothing
  system-wide. Detaching from the project in BOINC and removing the BOINC data directory (and, if you installed it, the
  BOINC client package) would remove it. That is an inference from `launcher.sh.in`, not a documented, tested procedure.
  **Not tested.**

## 9. What is still open

Owner decisions:

* A public project URL and exposing the server to the internet (egress, TLS). Today it is loopback-only
  (`docs/BOINC_PRODUCTION_SERVER.md`, `deploy/t4_node/README.md`).
* The credit mismatch between BOINC's native credit and `verified_credit` (`docs/BOINC_PRODUCTION_SERVER.md`, "Two things
  to decide").
* Key custody: the code-signing private key is on the server, and there is no TLS (same file, "Limits").

Engineering:

* A resident validator: the validator loads the model for every result (about 10 to 25 s) and shares the GPU with the
  volunteer's task (`docs/BOINC_PRODUCTION_SERVER.md`).
* Plan classes for 8, 12 and 16 GB cards: planned in issues #238 to #244, not implemented in the BOINC server stack.
* Windows (and macOS) support: the launcher is bash and needs `tar`, `xz` and `flock`.
* Measuring cross-architecture replay (RTX against T4): E12 did not run an RTX card (`not_run` in the E12 report), and the
  fixed-point clause is undecided (`analysis.P12-fixed-point` in the same report).
* A recorded run of the self-contained download path (`artifacts/boinc_distribution/report.json` is not in the repo).


## 10. Turnkey 1-Command RTX BOINC Server & Client Deployment (`scripts/hivemind_rtx_boinc_deploy.py`)

For developers, volunteers, or low-cost VM operators who want an immediate, zero-MariaDB/zero-Apache deployment of both the **RTX BOINC Server** and **RTX BOINC Client** on Windows 10/11, Linux, or Docker:

### A. Detect Your NVIDIA RTX GPU & BOINC Plan Class
```bash
python scripts/hivemind_rtx_boinc_deploy.py detect
```
Automatically detects your NVIDIA RTX GPU, driver version, total/free VRAM, and assigns:
- `cuda_hivemind` (`hivemind_qlora_moe`, rank 8, `< 6,656 MiB` WDDM ceiling, `>= 1,536 MiB` free VRAM floor) for 8 GB–9.9 GB cards (e.g., RTX 2060, RTX 2070, RTX 2080, RTX 3060 Ti, RTX 3070, RTX 4060).
- `cuda_hivemind_big` (`hivemind_qlora_moe_big`, rank 16, `>= 2,560 MiB` free VRAM floor on 12 GB cards / `>= 4,300 MiB` free VRAM floor on 16 GB T4) for 10 GB–24 GB+ cards (e.g., RTX 3060 12GB, RTX 4070, RTX 4080, RTX 3090, RTX 4090, Tesla T4).

### B. Start the Turnkey RTX BOINC Server (Standalone or Docker Compose)
```bash
# Direct Python server (speaks BOINC XML RPC + JSON/HTTPS RTX Workunit API, backed by SQLite):
python scripts/hivemind_rtx_boinc_deploy.py server --host 0.0.0.0 --port 8097

# Or via Docker Compose (starts RTX BOINC Server on :8097 + Permanent OpenTrainer Orchestrator on :8096):
docker compose -f deploy/rtx_boinc/docker-compose.rtx-boinc.yml up -d
```

### C. Deploy the RTX BOINC Client & Run a Verified CUDA Workunit
```bash
# 1. Generate cross-platform BOINC client XMLs (cc_config.xml, global_prefs_override.xml, app_info.xml,
#    app_config.xml, account_*.xml) + Windows PowerShell & Linux systemd --user service manifests:
python scripts/hivemind_rtx_boinc_deploy.py deploy-client --server http://127.0.0.1:8095 --handle your-handle

# 2. Run one verified CUDA QLoRA + PTX BLS workunit (remove --once to run continuously when idle):
python scripts/hivemind_rtx_boinc_deploy.py client --server http://127.0.0.1:8095 --handle your-handle --once

# 3. Run full end-to-end Server + Client + CUDA self-test:
python scripts/hivemind_rtx_boinc_deploy.py self-test --device cuda
```
