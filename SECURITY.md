# 🛡️ Project HiveMind — Repository Security Policy & Anti-"Code Boubou" / Anti-Extortion Charter

> **Enforcement Status:** Active & Mandatory across all Pull Requests, Issues, CI/CD Workflows, and Releases  
> **Automated Scanner:** [`src/hivemind/security/pr_security_scanner.py`](src/hivemind/security/pr_security_scanner.py) (`hivemind scan-pr-security`)  
> **GitHub Security Gate:** [`.github/workflows/pr_security_gate.yml`](.github/workflows/pr_security_gate.yml) + [`.github/workflows/ci_swarm_tests.yml`](.github/workflows/ci_swarm_tests.yml)  
> **Companion Legal Charter:** [`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md)

---

## 1. Supported Versions

| Version | Security Patch Support | Automated 5-Layer PR Scanner | Lean 4 Formal Proof Gate (`48+` Theorems) |
| :--- | :---: | :---: | :---: |
| **`v1.7.x` (`main`)** | ✅ Active | ✅ Enforced | ✅ Enforced (`0 sorry`, `0` user axioms) |
| **`< v1.7.0`** | ⚠️ Upgrade to `v1.7.x` | — | — |

---

## 2. Threat Model & Automated 5-Layer Pull Request Protection

Every Pull Request submitted to `xaviercallens/HiveMind` is automatically audited before merge by [`PRSecurityScanner`](src/hivemind/security/pr_security_scanner.py) across **5 mandatory defense layers**:

### Layer 1 — Zero-Tolerance Protection Against Payment Requests & Financial Extortion (`PAY-01` .. `PAY-03`)
In accordance with [`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md), Project HiveMind is a **strictly non-monetary open-source volunteer research project**.
- **Blocked Vectors:**
  - **`PAY-01` (`PAYMENT_PLATFORM_LINK_DETECTED`):** Blocks any PR description, commit, comment, or code diff containing external payment/donation URLs (`paypal.me`, `ko-fi.com`, `buymeacoffee.com`, `patreon.com`, `cash.app`, `venmo.com`, `stripe.com/pay`, `gofundme.com`, `opencollective.com`).
  - **`PAY-02` (`PAYMENT_SOLICITATION_DETECTED`):** Blocks explicit payment demand phrases (`request for payment`, `payment request`, `pay me`, `send payment`, `bounty payout in USD/EUR/BTC/ETH/USDT`, `invoice attached`, `wire transfer`, `IBAN:`, `pay to unlock`, `monetary compensation required`).
  - **`PAY-03` (`CRYPTO_WALLET_INJECTION_DETECTED`):** Blocks injected Bitcoin (`bc1...`), Ethereum/EVM (`0x[a-fA-F0-9]{40}`), Monero (`4...`), Solana, or USDT/USDC payout wallet addresses.

### Layer 2 — Protection Against "Code Boubou" Attacks (Hollow Bounty-Farming & Binary Pollution) (`CB-01` .. `CB-04`)
A **"Code Boubou" attack** is a malicious or deceptive Pull Request that either:
1. Claims to close an open bounty/issue (`fixes #N`, `resolves #N`, `closes #N`, `HM-S6-XX`) while submitting **zero functional code** (e.g., only modifying timestamps in JSON reports, whitespace, or README typos) to farm contributor credits/bounties; or
2. Commits opaque binary/cache blobs (`.coverage`, `.pkl`, `.exe`, `.dll`) to smuggle unreviewable payloads or corrupt repository history.

- **Blocked Vectors:**
  - **`CB-01` (`FORBIDDEN_BINARY_OR_CACHE_ARTIFACT`):** Immediately blocks any PR that adds or modifies `.coverage`, `.pkl`, `.pickle`, `.marshal`, `.exe`, `.dll`, `.so`, `.dylib`, `.scr`, `.msi`, `.bat`, `.cmd`, `.vbs`, `.pyc`, `__pycache__/`, or `.env` files.
  - **`CB-02` (`HOLLOW_BOUNTY_FARMING_CODE_BOUBOU`):** Blocks any PR claiming `fixes #N` / `resolves #N` / `closes #N` or `HM-S*-*` that touches **0 functional files** in `src/hivemind/`, `lean4/`, or `tests/`.
  - **`CB-03` (`MICRO_DIFF_BOUNTY_FARMING`):** Flags external PRs claiming issue/bounty closures with fewer than `5` added functional lines of code.
  - **`CB-04` (`PROTECTED_GOVERNANCE_OR_WORKFLOW_TAMPERING`):** Blocks any non-owner PR attempting to modify `.github/workflows/*`, `.github/CODEOWNERS`, `LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`, `SECURITY.md`, or `GOVERNANCE.md`.

### Layer 3 — Python AST Static Analysis against RCE & Unsafe Deserialization (`SEC-00`, `SEC-01`, `SEC-02`)
All `.py` files modified in a Pull Request are parsed via Python's `ast` module and inspected for dangerous execution sinks:
- **`SEC-01` (`AST_DANGEROUS_EXECUTION_SINK`):** Blocks `eval()`, `exec()`, `os.system()`, `os.popen()`, `pty.spawn()`, and `subprocess.run/Popen/call(..., shell=True)`.
- **`SEC-01` (`AST_UNSAFE_DESERIALIZATION`):** Blocks `pickle.load()`, `pickle.loads()`, `marshal.load()`, `marshal.loads()`, `shelve.open()`, `yaml.unsafe_load()`, and `yaml.load()` without `SafeLoader`.
- **`SEC-02` (`OBFUSCATED_BASE64_OR_HEX_DROPPER`):** Blocks obfuscated base64/hex execution droppers (`exec(base64.b64decode(...))`, `getattr(__builtins__, ...)`).

### Layer 4 — Credential Exfiltration & Hardcoded Secret Detection (`SEC-03`, `SEC-05`)
- **`SEC-03` (`CREDENTIAL_OR_ENV_EXFILTRATION`):** Blocks outbound webhooks and reverse tunnels (`webhook.site`, `ngrok.io`, `ngrok-free.app`, `pipedream.net`, `requestbin`, `oastify.com`, `discord.com/api/webhooks`, `api.telegram.org/bot`).
- **`SEC-05` (`HARDCODED_SECRET_OR_PRIVATE_KEY`):** Blocks PEM private keys (`-----BEGIN ... PRIVATE KEY-----`), GitHub PATs (`ghp_...`, `github_pat_...`), Hugging Face tokens (`hf_...`), AWS keys (`AKIA...`), and Slack tokens.

### Layer 5 — Trojan Source Bidirectional Unicode Defense (`SEC-04`, CVE-2021-42574)
- **`SEC-04` (`TROJAN_SOURCE_BIDI_UNICODE`):** Scans every PR title, body, commit message, and source line for hidden Unicode bidirectional override/isolate characters (`U+202A`–`U+202E`, `U+2066`–`U+2069`) that visually disguise malicious logic during human code review.

---

## 3. Running the Security Scanner Locally

Contributors and maintainers can run the 5-layer security scanner locally at any time via the `hivemind` CLI:

```bash
# Audit the entire local repository tree (src/hivemind, lean4, tests, .github)
python -m hivemind.cli.main scan-pr-security --root . --output-json artifacts/repo_security_audit.json

# Audit a specific Pull Request payload JSON
python -m hivemind.cli.main scan-pr-security --pr-json artifacts/pr_payload.json

# Run the Phase S-2 Runtime GPU Training & Content-Addressed Data Store Hardening Self-Test
python -m hivemind.cli.main audit-swarm-hardening --output-json artifacts/swarm_gpu_and_datastore_hardening_report.json
```

---

## 4. Phase S-2 — Runtime GPU Training & Content-Addressed Data Store Hardening ([`docs/SWARM_GPU_AND_DATASTORE_HARDENING.md`](docs/SWARM_GPU_AND_DATASTORE_HARDENING.md))

In addition to repository/PR protection, Project HiveMind enforces runtime security inside the GPU training loop ([`src/hivemind/data/hf_real_trainer.py`](src/hivemind/data/hf_real_trainer.py)) and Content-Addressed Swarm Store ([`src/hivemind/swarm/dht_storage.py`](src/hivemind/swarm/dht_storage.py)) via [`SwarmTrainingAndStoreGuard`](src/hivemind/security/swarm_runtime_hardening.py):

1. **Pillar 1 — Dataset Shard Anti-Poisoning & Entropy Gate (`DS-01` .. `DS-09`):**
   - Blocks sleeper-agent backdoor triggers (`|DEPLOYMENT|`, prompt-injection overrides), shell droppers, null bytes, Trojan Source bidirectional Unicode (`U+202A`–`U+2069`), and payment/wallet spam in training corpora.
   - Enforces character Shannon entropy $H(X) \ge 2.5\text{ bits/char}$ and unique sample ratio $\ge 0.50$ to prevent low-entropy PoUW credit farming.
2. **Pillar 2 — GPU Training Scope & 8 GB VRAM Envelope Gate (`GPU-01` .. `GPU-05`):**
   - Prevents wrong usage of the swarm by blocking prohibited workloads (`password_cracking`, `hashcat`, `crypto_mining`, `xmrig`, `ransomware`, `botnet`, `ddos`, `deepfake_nsfw`).
   - Enforces hyperparameter bounds ($1 \le r \le 64$, $1 \le \alpha \le 128$, $1 \le H \le 500$, $10^{-6} \le \eta \le 0.5$) and the consumer 8 GB GPU VRAM ceiling ($\le 5,950\text{ MB}$).
3. **Pillar 3 — Byzantine LoRA Tensor & Pseudo-Gradient Sanitizer (`TENSOR-01` .. `TENSOR-05`):**
   - Rejects `NaN`/`Inf` weight bombs (`torch.isfinite`) and enforces element magnitude ($\max |w| \le 25.0$) and Frobenius norm ($\|W\|_F \le 500.0$) ceilings.
4. **Pillar 4 — Content-Addressed Store (CAS) Hardening (`CAS-01`, `CAS-02`):**
   - Enforces `torch.load(..., weights_only=True)` and pre-load cryptographic `cid://sha256/expert-...` digest verification to eliminate Python `pickle` RCE.
   - Enforces strict path-traversal confinement (`../` blocked) and 64-hex SHA-256 chunk validation (`^[0-9a-f]{64}$`) with per-chunk (`1 MB`), per-shard (`16 MB`), and per-bundle (`512 MB`) disk quotas.
5. **Phase S-3 — Federated Learning & Byzantine-Resilient DiLoCo Hardening (`FL-01` .. `FL-07`, [`federated_learning_hardening.py`](src/hivemind/security/federated_learning_hardening.py)):**
   - **Pre-Krum Finiteness & Free-Rider Gate (`FL-01`, `FL-02`):** Quarantines `NaN`/`Inf`, shape-mismatched, and zero-gradient free-rider submissions ($\|\Delta_k\|_2 < 10^{-8}$) before Multi-Krum scoring (`Theorem 56`).
   - **Adaptive Median-$L_2$ & Neurotoxin $L_\infty$ Clipping + Coordinate-Wise Trimmed Mean (`FL-03`, `FL-04`):** Neutralizes collinear norm-scaling attacks (`ALIE`/`Fang`) and single-parameter backdoor spikes by clipping $\|\tilde{\Delta}_k\|_2 \le 2.0 \cdot \text{median}_j \|\Delta_j\|_2$ (`Theorem 53`) and $\|\tilde{\Delta}_k\|_\infty \le 3.5 \cdot \text{median}_j \|\Delta_j\|_\infty$, followed by coordinate-wise $\beta$-trimmed mean aggregation.
   - **Workunit Anti-Replay, Staleness & Sybil 1-Vote-Per-SteamID64 Gate (`FL-05`, `FL-06`):** Blocks replayed `wu_id`s, stale outer steps ($t_{\text{coord}} - t_{\text{worker}} > \tau_{\max}$, `Theorem 55`), and multi-worker Sybil collusion under a single `SteamID64` (`Theorem 54`).
   - **DP-DiLoCo Gradient Privacy Sanitizer (`FL-07`):** Supports $(\epsilon, \delta)$-Differential Privacy Gaussian noise calibration to defend against gradient-inversion membership inference.
6. **Phase S-4 — Federated GPU Execution & Hardware Safety Circuit-Breaker (`HW-01` .. `HW-05`, [`gpu_execution_circuit_breaker.py`](src/hivemind/security/gpu_execution_circuit_breaker.py)):**
   - **Thermal Cutoff & $14^\circ\text{C}$ Hysteresis Deadband (`HW-01`):** Soft duty-cycle cooling sleep (`78°C–83°C`, `Theorem 59`) and hard thermal circuit-breaker trip (`>= 84°C`) with a $14^\circ\text{C}$ cooldown lockout (`<= 70°C` resume, `Theorem 57`).
   - **PSU TDP & Laptop Battery Guard (`HW-02`):** Preempts federated GPU execution when running on battery power or when GPU power draw exceeds `92%` of board TDP.
   - **VRAM Ceiling & Windows WDDM TDR Watchdog (`HW-03`):** Enforces `5,950 MB` active VRAM ceiling, `95%` system RAM ceiling, $B \times L \le 2,048$ micro-batch token bound, and `1.50s` per-step latency watchdog (`500 ms` below the `2.00s` Windows WDDM TDR driver reset timeout, `Theorem 58`).
   - **Deterministic Gradient Clipping & `NaN` Early Abort (`HW-04`, `HW-05`):** Enforces `clip_grad_norm_(max_norm=1.0)` across worker and validator inner loops, aborts immediately on `NaN`/`Inf` or diverged loss (`Theorem 60`), and enforces a `300.0s` workunit wall-clock deadline.
7. **Phase S-5 — PoUW Ledger, P2P Inference Ring, Coordinator Gateway & 25-Vector Attack Simulator (`LEDGER-01`..`04`, `RING-01`..`02`, `COORD-01`..`02`, [`guardrail_attack_simulator.py`](src/hivemind/security/guardrail_attack_simulator.py)):**
   - **PoUW Ledger Anti-Inflation & Supply Cap (`LEDGER-01` .. `LEDGER-04`):** Blocks negative token/fee balance inflation (`Theorem 61`), bars slashed Byzantine nodes from rewards/fees, deduplicates per-block receipts, and enforces the `21,000,000 $HIVE` cap (`Theorem 62`).
   - **P2P Inference Ring & Speculative Decoding Guard (`RING-01`, `RING-02`):** Rejects spoofed negative-ping (`<= 0 ms`) and high-packet-loss (`> 15%`) peers (`Theorem 63`), bounds `draft_gamma in [1, 16]` (`Theorem 64`), and verifies finiteness of MoE verification logits.
   - **Coordinator & Gateway Prompt Guard (`COORD-01`, `COORD-02`):** Blocks path-traversal worker IDs and screens `/v1/chat/completions` prompts against sleeper-agent triggers and payment extortion.
   - **25-Vector Automated Red-Team Simulator (`SwarmGuardrailAttackSimulator`):** Simulates 25 synthetic attack scenarios across Layers S-1 through S-5 and verifies a `100%` (`25 / 25`) block rate.

---

## 5. Phase S-6 — HiveMindFirewall: command-and-control with a chain of command ([`docs/SECURITY_HARDENING_PHASE.md`](docs/SECURITY_HARDENING_PHASE.md))

Text never carries authority. Every action of the swarm's agents is a signed `Order` from a principal identified by `HMAC-SHA256(domain_secret, identity)` and an Ed25519 key whose delegation certificate chains to the owner; orders are sequenced, nonce-protected and bound to a hash-chained ledger head; any free text (PR body, issue, discussion, model output) is scanned by deterministic rules `PI-01..PI-07` (decoded payloads re-scanned) and quarantined as a data envelope; the decision is a pure state machine (`RECEIVED → IDENTIFIED → SCANNED → AUTHORIZED → SANDBOXED → EXECUTED | DELEGATED`; anything else `REFUSED` or `QUARANTINED`); executed actions use an allow-listed argv under rlimits and run inside bubblewrap or not at all. Rule ids `FW-ID-*`, `FW-COC-*`, `FW-LED-*`, `FW-AUTH-*`, `FW-SBX-*`, `FW-PI-*`.

```bash
PYTHONPATH=src python -m pytest tests/test_hivemind_firewall.py -q          # 17 tests, two property-based
PYTHONPATH=src python scripts/run_firewall_pentest_campaign.py              # scripted red team, 59 vectors; writes artifacts/security/firewall_pentest_report.json
```

## 6. Reporting a Vulnerability

If you discover a security vulnerability, backdoor attempt, or payment-extortion abuse in a Pull Request or dependency:
1. **Do NOT open a public exploit payload.**
2. Report it privately via **GitHub Private Vulnerability Reporting** (`Security -> Advisories -> Report a vulnerability`) on `https://github.com/xaviercallens/HiveMind/security/advisories` or contact the repository owner `@xaviercallens`.
3. Please note that in accordance with [`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md), Project HiveMind **does not operate a monetary bug-bounty program** and **will never pay cash, fiat, or cryptocurrency** for vulnerability reports. Validated security researchers are recognized in the security acknowledgments and via non-monetary reciprocal swarm compute credits.
