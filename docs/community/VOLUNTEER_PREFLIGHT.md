# Volunteer preflight: is my machine ready?

`scripts/hivemind_volunteer_preflight.py` checks a machine **before** you install anything and says whether it is *ready*, *ready with warnings* or *not ready*, and why. Python 3.10+, standard library only.

```bash
python3 scripts/hivemind_volunteer_preflight.py              # human table
python3 scripts/hivemind_volunteer_preflight.py --json       # machine output
python3 scripts/hivemind_volunteer_preflight.py --explain    # what is read, then exit
python3 scripts/hivemind_volunteer_preflight.py --data-dir /big/disk/boinc --project-url https://PROJECT_HOST/
```

Exit code: 0 ready, 1 ready with warnings, 2 not ready.

## Privacy

It reads hardware facts and sends nothing anywhere unless `--project-url` is given; it never installs anything and writes nothing. Without that flag it opens no network connection at all. It never prints or stores your hostname, user name, MAC address, serial numbers or GPU UUID; your home directory is shown as `~`.

## What it checks and how to read each result

| Check | OK | WARN | FAIL |
| --- | --- | --- | --- |
| OS / architecture | x86_64 Linux | | anything else. The only published app version is platform `x86_64-pc-linux-gnu` (`docs/BOINC_PRODUCTION_SERVER.md`); Windows and macOS builds do not exist yet |
| Python | informational | older than 3.10 | |
| NVIDIA GPU | card has at least 3,500 MB (plan class `cuda_hivemind`) and most of it is free | another program holds more than half of the VRAM or leaves less than 3,500 MB free (a game): training peaks near 3,409 MiB (`docs/BOINC_CORE_NODE.md`); or `nvidia-smi` output was unreadable | no NVIDIA GPU, or under 3,500 MB. CPU-only volunteering is not supported by the QLoRA app |
| Free disk | at least 2 x 4.4 GB | below 8.8 GB | below 6.6 GB (1.5 x) |
| RAM / CPU | 8 GB and 4 threads or more | below either | |
| Existing BOINC client | informational: tells you to add the project, or to install a client first | | |
| Network (only with `--project-url`) | https, public address, reachable | slow link (over 2 h) or HTTP error | `http://`, private or non-public address, unreachable |
| Laptop / battery / sleep | informational hints, including the community night window 22:00-07:00 Europe/Paris (`contributor_policy.py`) | | never fails |

The GPU line also tells you which serving tiers of the HiveMind OpenRouter your card can hold (from `src/hivemind/router/catalog.py`, `min_vram_gb`) and which are too big.

## First download size

About **4.4 GB**: runtime 1.76 GB + base model 2.67 GB + small files (the portable distribution built by `scripts/boinc_build_distribution.sh`, launcher in `deploy/boinc_dist/`). This figure was supplied by the project owner; no repo file states it, so re-measure it when the distribution changes. The launcher unpacks the archives into the project directory, so the script asks for 1.5 x (fail) and 2 x (warn) of it. The RAM (8 GB) and CPU (4 threads) thresholds are the project's rule of thumb; no repo document gives figures.

## Example output

```
[ OK ] NVIDIA GPU
       NVIDIA GeForce RTX 3060: 11776 of 12288 MiB free, driver 550.54. Enough for training ...
Verdict: READY
```
```
[WARN] NVIDIA GPU
       NVIDIA GeForce RTX 3060: 1788 of 12288 MiB free; another program (a game?) is using the card ...
       -> Close games and GPU-heavy apps, or let BOINC run only while the PC is idle.
Verdict: READY WITH WARNINGS
```
```
[FAIL] Operating system
       Windows is not supported: the only published app version is for x86_64 Linux ...
Verdict: NOT READY
```

## Limits

Not wired into any installer. Tested with fakes on Linux; the Windows and macOS probe code paths (RAM, battery) have not been run. See also `docs/BOINC_REAL_CLIENT.md`, `docs/BOINC_CORE_NODE.md`, `docs/BOINC_PRODUCTION_SERVER.md`.
