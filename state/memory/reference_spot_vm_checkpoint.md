---
name: reference-spot-vm-checkpoint
description: How the T4 campaign state is checkpointed against Spot-VM preemption and how to resume from it
metadata:
  node_type: memory
  type: reference
  originSessionId: 22ff63cf-3b37-455e-984e-5fca864875cb
  modified: 2026-10-24T09:05:05.259Z
---

The T4 machine is a GCP Spot VM (hermes-node1): it can be preempted with ~30 s notice; processes, /tmp and the running Claude job die; the git remote and (usually) the disks survive.

Checkpoint mechanism: `scripts/checkpoint_state.py` (git plumbing with a temporary index, never touches the worktree or the checked-out branch). Remote branch `checkpoint/t4-campaign-state` on origin (fast-forward only), ref `refs/checkpoints/t4-campaign` locally, tar.gz copies in `/mnt/disks/disk-socrateai-local-1/hivemind_checkpoints/` (last 6). It holds the git-ignored/volatile state under `state/` (artifacts/experiments/E15, E16 stage files and journals, job logs, queue state, memory notes, watch scripts), `state/MANIFEST.json` (sha256, skipped secret-like files) and `RESUME.md` with the restart commands. Secrets are excluded by filename denylist and a content scan; model weights and LoRA .pt are excluded (E15's train stage reruns from the RFT journal).

A background watcher runs `python3 scripts/checkpoint_state.py --watch --every 1200` (every 20 min, plus an immediate checkpoint when the GCE metadata server reports preemption). If a session finds it dead, restart it, take a manual `--once`, and relaunch the queue as written in `RESUME.md` (jobs are resumable).

**Why:** the user asked on 2026-10-24 for a checkpoint in case the Spot image fails.

**How to apply:** before long unattended GPU work, check the watcher is alive; after a preemption read `RESUME.md` on the checkpoint branch first. Code and top-level `artifacts/experiments/*.json` are protected by normal pushes of `worktree-feat-s10-e11-e12-t4`. See [[project-t4-campaign-2026-10-24]].
