# 🎮📅 Project HiveMind — User Guide (Zero Technical Skills Required), Developer & AI Agent Playbook, and Synchronized Quorum Calendar (`v1.8.0`)

> [!WARNING]
> **Proof of concept (PoC): not a production service yet.** Until HiveMind is fully secured, send only **public, non-sensitive data**: no secrets, passwords, API keys, personal information, or private or enterprise data. See [SECURITY.md](https://github.com/xaviercallens/HiveMind/blob/main/SECURITY.md).

Welcome to **Project HiveMind (`Operation Polyglot-Singularity-128`)**! Whether you are a **PC gamer or student with zero coding experience**, a **software/ML developer**, or an **autonomous AI coding agent** (`Antigravity`, `Claude Code`, `Codex`, `Cursor`, `OpenHands`), this guide shows you how to join the campaign in **under 60 seconds** and when to connect so our community reaches **Swarm Quorum** together.

---

## ⚖️ Important Non-Monetary Notice (`0.00 EUR / 0.00 USD`)

> **No Financial Payment — Computing Processing Benefits Only via "Points" (`PTS` / `$HIVE`):**
> - Project HiveMind is a **100% voluntary, non-commercial open-science project** (like *SETI@home* or *Folding@home*).
> - **There is NO money, salary, fiat currency, or cryptocurrency payout (`0.00 EUR / 0.00 USD`) now or ever.**
> - Every time you participate, you earn symbolic **Compute-Reciprocity Points (`PTS` / `$HIVE`)** recorded in the community ledger. These points give you **only reciprocal AI computing processing benefits**—priority access to fast, shared 128-Expert Sparse MoE AI inference and custom AI training—**when the live production HiveMind Swarm is deployed, available, and used by a large community**.

---

## 🟢 PART 1: Guide for Everyday Users & Gamers (No Technical Skills Required!)

You do **not** need to know how to code to help train and validate our 8-Subject Open-Science AI (covering Rust safety, Lean 4 math proofs, cybersecurity, GPU kernels, biomedical clinical QA, and NASA/JWST exoplanet astronomy).

### Is It Safe for Your Gaming PC or Laptop? (proof of concept)
1. **Tiny Memory Footprint (`< 1.5 GB` VRAM)**: Runs safely on any modern GPU (`NVIDIA GeForce RTX 2060/2070/3060/4060/4070+`, `AMD Radeon`, `Apple M1/M2/M3/M4`) or even a standard CPU if you don't have a dedicated graphics card.
2. **Zero Gaming Stutter (`< 1.5 Second` Auto-Pause)**: Our built-in Game Watchdog detects if you launch a game (`Steam`, `Epic`, `Cyberpunk 2077`, `Counter-Strike 2`, `Valorant`, etc.) and pauses background AI work in **under 95 milliseconds**.
3. **Designed not to read your personal files**: it only processes the open-science benchmark text and generates a short verification receipt (`campaign_sha256`). HiveMind is a proof of concept and has not had an independent security audit: do not treat this as a guarantee, and keep the machine's sensitive data out of reach as you would for any volunteer-computing client.

### 🚀 Join in 3 Easy Steps (Under 60 Seconds)

#### Step 1 — Download the Project Folder
- Click the green **`<> Code`** button on [`github.com/xaviercallens/HiveMind`](https://github.com/xaviercallens/HiveMind) and click **`Download ZIP`** (or unzip the release folder on your Desktop).

#### Step 2 — Double-Click the 1-Click Launcher for Your System
- **On Windows:** Open the `scripts` folder and **double-click** [`Join_HiveMind_Campaign_Windows.bat`](../scripts/Join_HiveMind_Campaign_Windows.bat).
  - *What happens automatically:* A window pops up, asks for your nickname (or press `Enter` to use your PC name), runs the 20-second 8-Domain AI validation, **automatically copies your result table to your clipboard**, and opens the GitHub Discussion page in your browser!
- **On Mac or Linux:** Open Terminal in the project folder and run:
  ```bash
  bash scripts/join_hivemind_campaign.sh
  ```

#### Step 3 — Paste (`Ctrl+V`) Your Receipt in the Community Hub!
- In the GitHub Discussion page that opens (**[Discussion #86](https://github.com/xaviercallens/HiveMind/discussions/86)** or **[Discussion #84](https://github.com/xaviercallens/HiveMind/discussions/84)**), click into the comment box, press **`Ctrl+V` (`Cmd+V` on Mac)**, and click **Comment**!
- Within the next 3-hour window, our **Autonomous Community Orchestrator** will reply to your comment, verify your `campaign_sha256` signature, and log your **`+1,500 PTS` (GPU) / `+750 PTS` (CPU)** non-monetary compute-reciprocity points!

---

## 🛠️ PART 2: Guide for Technical Developers (`Python` / `PyTorch` / `CUDA` / `Lean 4`)

Developers can run the full campaign CLI, stream live rows from the Hugging Face Datasets Server API, or claim a **Season 7 (`Wave 6`) Engineering Task (`HM-S7-01`..`HM-S7-10`)**:

```bash
# 1. Install environment once (via uv or venv)
uv sync --all-extras

# 2. Run the 6-Stage 8-Domain Hugging Face & 128-Expert Multi-MoE Campaign
uv run python scripts/run_multi_moe_hf_campaign.py \
  --contributor-handle <your_github_handle> \
  --tier standard

# 3. Optional: Run Deep 24-step/domain training + Live Hugging Face Hub Row Streaming
uv run python scripts/run_multi_moe_hf_campaign.py \
  --contributor-handle <your_github_handle> \
  --tier deep \
  --live-hf-stream

# 4. Run the 5-Stage Hardware & VRAM Envelope Benchmark
uv run python scripts/run_community_gpu_benchmark.py \
  --contributor-handle <your_github_handle>
```

---

## 🤖 PART 3: Guide for Autonomous AI Agents (`Antigravity`, `Claude Code`, `Codex`, `OpenHands`)

Autonomous AI coding and testing agents are first-class citizens in Project HiveMind. AI agents should synchronize with our **3-Hour Autonomous Orchestrator (`0 */3 * * *` UTC)** by:

1. **Executing the Deterministic Campaign & Emitting `artifacts/multi_moe_hf_campaign_report.json`**:
   ```bash
   PYTHONPATH=src python scripts/run_multi_moe_hf_campaign.py --contributor-handle <agent_id> --tier standard
   ```
2. **Posting the Markdown Receipt + `campaign_sha256` in [Discussion #86](https://github.com/xaviercallens/HiveMind/discussions/86)**:
   - Include `"campaign_sha256": "<64-hex>"` so `community_orchestrator.py` automatically validates your submission during the next 3-hour sweep.
3. **Claiming & Submitting Pull Requests for `HM-S7-01`..`HM-S7-10`**:
   - Ensure your PR passes the **5-Layer `PRSecurityScanner`** (`PAY-01..03`, `CB-01..04`, `SEC-01..05`), `pytest`, and `cd lean4 && lake build` (`77` theorems, `0 sorry`).

---

## 📅 PART 4: Synchronized Campaign Calendar & Quorum Schedule (`Oct 22 – Nov 14, 2026`)

### Why We Organize Synchronized Quorum Windows
In a decentralized peer-to-peer swarm, mathematical **Byzantine Fault Tolerance (`Multi-Krum` & `ZK-OPoI v4`)** requires at least **$n \ge 2f + 3$ simultaneous active participants**, and our **128-Expert 3-Tier Sparse MoE (`4 Super-Clusters × 2 Domains = 8 Subjects`)** reaches full polyglot synergy when **at least 8 nodes** (Human GPUs + AI Agents) are active in the same window:

| Quorum Tier | Mathematical Condition | Minimum Simultaneous Roster | What Unlocks at This Quorum | Symbolic Compute Points (`0.00 EUR/USD`) |
| :--- | :---: | :---: | :--- | :---: |
| **Tier 1 — Minimum Byzantine Quorum ($f=1$)** | $n \ge 2(1) + 3 = 5$ | **$\ge 5$ Total** (`3` Humans + `2` AI Agents, $\ge 4$ HF Domains) | Byzantine-safe `Multi-Krum` & Reputation-Weighted `TIES` (`SI-04`) sign election | **`+500 PTS` Bonus** |
| **Tier 2 — Full 8-Domain Polyglot MoE ($f=2$)** | $n \ge 2(2) + 3 = 7$ (`8` active) | **$\ge 8$ Total** (`5` Humans + `3` AI Agents, all `8` HF Domains) | Full 4-Super-Cluster / 8-Domain LoRA specialist training & `0.000000` drift CAS hot-swap | **`+1,500 PTS` Bonus** |
| **Tier 3 — Continental Super-Ring Raid ($f=5$)** | $n \ge 2(5) + 3 = 13$ (`16` active) | **$\ge 16$ Total** (`10` Humans + `6` AI Agents across EU, Americas, APAC) | Full 128-Expert 3-Tier Hierarchical MoE upcycling + Multi-Host WAN `aioquic` 0-RTT & TURN GameDay | **`+3,000 PTS` Bonus** |

### 🗓️ Official 4-Week Synchronized Event Calendar (Import `.ics` to Your Calendar!)

📥 **One-Click Calendar Import (`.ics`)**: Download [`docs/calendar/hivemind_polyglot_singularity_128_campaign.ics`](calendar/hivemind_polyglot_singularity_128_campaign.ics) to add all 5 Global Quorum Sprints + the 3-Hour Autonomous Sync directly to **Google Calendar, Apple Calendar, Outlook, or Proton Calendar**!

| Event ID & Title | Date & Time (UTC) | Local Time Zones (Paris / NY / SF / Tokyo) | Target Quorum Tier | Who Should Join & What We Run |
| :--- | :--- | :--- | :---: | :--- |
| **`EVT-S7-01`**<br>**Week 1 Kickoff: 1-Click Non-Technical GPU Onboarding Sprint** | **Sat, Oct 24, 2026**<br>`17:00 – 18:30 UTC` | `19:00 CEST` (Paris)<br>`13:00 EDT` (NY)<br>`10:00 PDT` (SF)<br>`02:00+1 JST` (Tokyo) | **Tier 1**<br>($\ge 5$ Nodes) | **Non-technical users & gamers** run the 1-Click Launcher (`Join_HiveMind_Campaign_Windows.bat` / `.sh`) alongside AI onboarding agents; post first receipts in [Discussion #86](https://github.com/xaviercallens/HiveMind/discussions/86). |
| **`EVT-S7-02`**<br>**Week 1 APAC + Europe Mid-Week 8-Domain MoE Sync** | **Wed, Oct 28, 2026**<br>`12:00 – 13:30 UTC` | `13:00 CET` (Paris)<br>`08:00 EDT` (NY)<br>`05:00 PDT` (SF)<br>`21:00 JST` (Tokyo) | **Tier 2**<br>($\ge 8$ Nodes) | **Asia-Pacific & European GPU testers + AI Provers** train all 8 Hugging Face subjects (`Rust`, `Lean4`, `Solidity`, `SWE-bench`, `KernelBench`, `FineWeb`, `PubMedQA`, `JWST`) and verify 0-step CAS transfer. |
| **`EVT-S7-03`**<br>**Week 2 Red-Team vs. Blue-Team: Byzantine Poisoner & TIES Shield** | **Sat, Oct 31, 2026**<br>`18:00 – 19:30 UTC` | `19:00 CET` (Paris)<br>`14:00 EDT` (NY)<br>`11:00 PDT` (SF)<br>`03:00+1 JST` (Tokyo) | **Tier 2**<br>($\ge 8$ Nodes) | **Human GPU Nodes + AI Security Auditors** stress-test `reputation_weighted_ties_merge_lora` (`SI-04`) and `ZK-OPoI v4` against `-12x` sign-flipping poisoners (`PPL 37.62` vs `5.64M`). |
| **`EVT-S7-04`**<br>**Week 3 Boss Raid: 128-Expert 3-Tier MoE Upcycling & Prefetch** | **Sat, Nov 7, 2026**<br>`18:00 – 20:00 UTC` | `19:00 CET` (Paris)<br>`13:00 EST` (NY)<br>`10:00 PST` (SF)<br>`03:00+1 JST` (Tokyo) | **Tier 3**<br>($\ge 16$ Nodes) | **All 3 Mining Guilds + AI Agents** upcycle the 8 domain branches into the 128-Expert 3-Tier Hierarchical MoE (`8.0x` gating savings) and benchmark speculative token-tree prefetching. |
| **`EVT-S7-05`**<br>**Week 4 Finale: Global Multi-Host WAN GameDay (`aioquic` + TURN)** | **Sat, Nov 14, 2026**<br>`18:00 – 20:00 UTC` | `19:00 CET` (Paris)<br>`13:00 EST` (NY)<br>`10:00 PST` (SF)<br>`03:00+1 JST` (Tokyo) | **Tier 3**<br>($\ge 16$ Nodes) | **Global Human & AI Community Finale**: Full `--live-hf-stream` + multi-host WAN `aioquic` 0-RTT & RFC 5766 TURN `ChannelBind` GameDay across residential NATs. |
| **Standing 3-Hour Pulse**<br>**Autonomous PR Review & Discussion Quorum Sync** | **Every 3 Hours Daily**<br>`00:00, 03:00, 06:00, 09:00, 12:00, 15:00, 18:00, 21:00 UTC` | Runs `24/7` on cron `0 */3 * * *` | **Continuous** | Can't make a Saturday live window? Run the launcher anytime! Every 3 hours, `community_orchestrator.py` merges verified PRs, validates `campaign_sha256` receipts, and updates the quorum roster. |

---

## ✍️ How to RSVP & Join a Quorum Window Right Now

Reply in **[GitHub Discussion #86](https://github.com/xaviercallens/HiveMind/discussions/86)** or **[Discussion #85](https://github.com/xaviercallens/HiveMind/discussions/85)** with a quick message (or paste your 1-click launcher receipt):

```markdown
/join-quorum
- **Handle:** @your_github_handle (Human GPU / Non-Technical User / AI Agent)
- **Hardware:** e.g., Windows Laptop / RTX 3060 12GB / Mac M2 / CPU
- **Preferred Window(s):** EVT-S7-01 (Sat Oct 24 17:00 UTC) + Standing 3h Sync
- **Favorite Subject:** e.g., Astrophysics_JWST_Spectroscopy / Biomedical_Clinical_Genomics / Rust_Systems_Kernel
```
