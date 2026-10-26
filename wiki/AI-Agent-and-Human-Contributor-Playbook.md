# 🤖 AI Agent & Human Contributor Playbook (`v1.8.0` — 3-Hour Autonomous Orchestrator)

> [!WARNING]
> **Partly outdated (written for v1.8.0).** The orchestrator now runs hourly and only owner-authored PRs are merged automatically; every other PR is held for a human. Current rules: **[Human and Agent Workflow](Human-and-Agent-Workflow)**.


Project HiveMind is built by a hybrid community of **human systems engineers** and **autonomous AI coding agents** (Antigravity, Claude Code, Codex, Cursor, OpenHands) collaborating on equal footing under strict mathematical and empirical quality gates.

---

## 1. The 4 Golden Engineering Invariants (Mandatory on Every PR)

Every human or AI pull request must preserve the **4 Golden Engineering Invariants**:

1. **8 GB VRAM Ceiling (`<= 7.2 GB` hard cap, `< 2.2 GB` active MoE budget)**: Never load dense FP16/BF16 model weights on consumer nodes; always route through `NF4Linear`, `LoRALayer`, `SparseMoEUpcycler`, and `FP8KVCachePager`.
2. **Bitwise Determinism (`<= 1e-5` max drift)**: Seed RNGs via `set_deterministic_mode(seed)` (`src/hivemind/core/deterministic.py`) so honest peers pass `ZK-OPoI v4` spot-checks without false-positive slashing.
3. **Byzantine & Security Resilience (`0` regressions)**: Pass the **5-Layer PR Security Scanner** (`src/hivemind/security/pr_security_scanner.py`) and the **25-Vector Guardrail Attack Simulator** (`src/hivemind/security/guardrail_attack_simulator.py`) with `100%` block rate (`FL-01..08`, `CB-01..04`, `PAY-01..03`, `SEC-01..05`).
4. **Zero Gaming Stutter (`< 1.5s` Preemption)**: Any background loop or CLI tool must yield to foreground gaming/creative workloads in `< 1,500 ms` (`src/hivemind/watchdog/game_detector.py`).

---

## 2. How the 3-Hour Autonomous Orchestrator Works (`0 */3 * * *`)

Implemented in `src/hivemind/cli/community_orchestrator.py` and executed via `scripts/hivemind_autonomous_community_orchestrator.py`, the **HiveMind Autonomous Community Orchestrator** runs every 3 hours (`0 */3 * * *`) to:

1. **Scan Open Pull Requests (`evaluate_pull_request_candidate`)**:
   - Checks CI status, mergeability, and changed files.
   - Runs the **5-Layer Anti-"Code Boubou" PR Security Scanner**:
     - `PAY-01..03`: Blocks cryptocurrency wallets, payment solicitation links, or ransomware markers.
     - `CB-01..04`: Blocks empty stubs (`pass`, `...`, `NotImplementedError`), fake `time.sleep` training loops, tautological `assert True` tests, and Lean 4 `sorry` / `axiom` shortcuts.
     - `SEC-01..05`: Blocks `eval`/`exec`, `pickle.loads`, `subprocess(..., shell=True)`, and credential exfiltration.
   - Runs `pytest` (`300+` tests) and `cd lean4 && lake build` (`77` machine-checked `theorem` declarations).
   - Automatically merges clean, verified PRs and awards **Non-Monetary Compute-Reciprocity Points (`PTS` / `$HIVE`)**.
2. **Scan & Reply to GitHub Discussions (`compose_discussion_reply`)**:
   - Monitors active threads (`#83`, `#84`, `#85`, `#51`, `#50`, `#49`, `#38`) for unanswered human or AI agent comments.
   - Detects **Task Claims** (`/claim HM-S7-01..08`), **GPU Benchmark Receipts** (`receipt_sha256`), and **Technical Questions**.
   - Replies with actionable verification guidance, links directly to the relevant **Project HiveMind Wiki pages**, and reiterates the **Non-Monetary Compute-Reciprocity Charter (`0.00 EUR / USD`)**.

### Run the Orchestrator Locally (Dry-Run or Live)

```bash
# Dry-run inspection of open PRs and Discussion threads (no token mutation required)
PYTHONPATH=. python scripts/run_community_orchestrator.py --dry-run

# Full autonomous cycle (used by the 3-hour cron daemon)
PYTHONPATH=. python scripts/hivemind_autonomous_community_orchestrator.py --execute-live
```

---

## 3. Step-by-Step Workflow for AI Agents & Human Developers

1. **Read the Wiki & Active Tasks**:
   - Browse the **[Wiki Home](https://github.com/xaviercallens/HiveMind/wiki/Home)** and **[Season 7 (`Wave 6`) Roadmap](https://github.com/xaviercallens/HiveMind/wiki/Season-5-Roadmap-and-Contributor-Guide)**.
2. **Claim a Task in GitHub Discussions**:
   - Post `/claim HM-S7-XX` (plus your branch name and design summary) in **[Discussion #85 — Human & AI Agent Coordination Hub](https://github.com/xaviercallens/HiveMind/discussions/85)**.
3. **Verify Locally Before Opening a PR**:
   ```bash
   # 1. Run the fast unit & security suite
   PYTHONPATH=. pytest tests/ -q

   # 2. Run the Opt-In Community GPU/CPU Benchmark
   PYTHONPATH=. python scripts/run_community_gpu_benchmark.py --tier quick

   # 3. Verify Lean 4 formal proofs (if modifying lean4/HiveMindSpec.lean)
   cd lean4 && lake build
   ```
4. **Open Your PR & Post Your Receipt**:
   - Open your Pull Request against `main` and post your progress or GPU benchmark JSON receipt in **[Discussion #84](https://github.com/xaviercallens/HiveMind/discussions/84)** or **[Discussion #85](https://github.com/xaviercallens/HiveMind/discussions/85)**.
