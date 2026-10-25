---
name: project-boinc-core-node
description: "State, locations and open decisions of the real-BOINC work (stock client + source-built server, QLoRA MoE apps on the T4), 2026-10-25"
metadata:
  node_type: memory
  type: project
  originSessionId: 5c0bcf79-7259-42eb-8295-f29db822ef58
  modified: 2026-10-25T11:26:37.590Z
---

Branches/PRs (draft, stacked): #178 `worktree-feat-boinc-real-client` (emulator + toy GPU app), #181 `feat/boinc-core-node` (base = #178's branch; QLoRA MoE apps, core node, real server). Merge is the owner's; merge #178 first then retarget #181 to main. Docs: `docs/BOINC_REAL_CLIENT.md`, `BOINC_CORE_NODE.md`, `BOINC_PRODUCTION_SERVER.md`, `deploy/t4_node/README.md`. Acceptance files were written before runs, each under `artifacts/boinc_*/ACCEPTANCE.md`.

Locations (data disk, not in git): `/mnt/disks/disk-socrateai-local-1/boinc/` = `root/` (unpacked boinc-client 7.18.1 + libboinc7, run with LD_LIBRARY_PATH, never installed), `src/` (BOINC 7.24.4 source, server built there), `venv/` (system-site-packages + peft + bitsandbytes; the shared ~/.local site-packages were deliberately NOT touched because another session shares them), `stack/` (private MariaDB data, apache.conf, project `hmproj`, shims/python). Run records: `core1*`, `prod2*`.

Running as of 2026-10-25 ~11:10 UTC (dies with the session/VM, no systemd): private MariaDB (socket only, no TCP), Apache on 127.0.0.1:8081, 7 BOINC daemons. After a reboot: `sudo install -d -o $USER /run/mysqld` then `python3 scripts/boinc_server_stack.py start ...` (flags in the doc). The VM's own nginx (80/443) is not mine; do not touch. apt packages mariadb/apache2/php are installed but system services are disabled.

**Why:** the user asked to launch BOINC as a core node for QLoRA MoE training/inference, then to "conduct the production BOINC server" after the tests, and to make the T4 available to new users with the best MoE models.

**How to apply / open decisions:** (1) CORRECTED 2026-10-25 by another session's report (I did not verify it): the T4 is ALREADY served to new users through the gateway (public https endpoint, two producer accounts: coder tiers + a "bee" expert); I had wrongly said a key/tunnel was missing. The gateway catalog is live. Open choice is only what the one card serves: current mix (~4.8 GB bee expert + 5.6 GB 7B coder), gpt-oss:20b (needs both stopped) or BOINC training. With the mix resident, a BOINC task fits but the validator replay cannot overlap it. DECISION 2026-10-25 (owner): keep the current mix (bee expert + coder tiers) on the T4 for now; the gpt-oss:20b switch and BOINC validation are to be started later that night by the other sessions, not by me; do not start GPU work or stop the live services. Credit question (BOINC 267.7 vs verified 56.9) still unanswered. PR #178 merged 2026-10-25 11:24 UTC; #181 retargeted to main, ready, merge still pending the owner (this session never merges). gpt-oss:20b (12.3 GiB) is the only measured servable MoE and cannot share the card with BOINC (3.5 GiB/task); switch with `boinccmd --set_gpu_mode never`. (2) BOINC native credit (267.7) vs `verified_credit` (56.9) disagree; reconciling is open. (3) QLoRA = NF4 on attention only, experts frozen fp16 (granite experts are not nn.Linear). (4) Gotchas found: `hivemind.config.ensure_local_drive_caches()` redirects HF cache on import (use cache_dir); app must write `boinc_finish_called`; the shared T4 is also used by another session's gwaya-lite llama-server and Ollama (check nvidia-smi before GPU runs). (5) Scores that were rescored (not hidden): core B7/B8, prod C3; originals kept in the reports.
