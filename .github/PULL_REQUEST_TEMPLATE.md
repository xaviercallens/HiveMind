## Summary & Parallel Track

- **Task / Issue ID:** `HM-S6-__` (or `HM-M_-__` / Custom)
- **Closes Issue:** `#__`
- **Subsystem Track:**
  - [ ] `feat/*` — Core Swarm Fabric (`src/hivemind/swarm/`)
  - [ ] `expert/*` — Sparse MoE & Compression (`src/hivemind/core/`, `src/hivemind/compression/`, `src/hivemind/data/`)
  - [ ] `game/*` — Steam Hack-Miner, Web HUD & Steamworks (`src/hivemind/game/`, `src/hivemind/client/`)
  - [ ] `formal/*` — Lean 4 Formal Specification & Paper (`lean4/`, `paper/`)
  - [ ] `security/*` / `docs/*` / `fix/*` — Security, Documentation, CI/CD, or Bug Fix

---

## Architectural & Functional Changes

<!-- Describe the technical implementation, algorithmic complexity, and wire/VRAM impact.
     NOTE: Hollow "Code Boubou" PRs (claiming issue closure without real code in src/hivemind/, lean4/, or tests/) are automatically blocked by CI. -->

---

## Mandatory Quality, Security & Legal Verification Checklist

- [ ] **Full Test Suite & `>=95%` Coverage Gate Green:** `python -m pytest tests/ -v --tb=short` passes with `0` failures.
- [ ] **Lean 4 Formal Invariants Intact:** `lean4/HiveMindSpec.lean` contains `0 sorry` and `0` user-defined `axiom` declarations (`>= 48` machine-checked theorems).
- [ ] **8 GB Consumer GPU VRAM Envelope:** Peak active VRAM remains strictly `<= 5,950 MB` (`<= 5.81 GB`) with CPU fallback when CUDA/Triton/Steamworks DLLs are absent.
- [ ] **5-Layer Security Gate Passed (`hivemind scan-pr-security`):**
  - No forbidden binary/cache blobs (`.coverage`, `.pkl`, `.exe`, `.dll`, `__pycache__`).
  - No dangerous AST sinks (`eval`, `exec`, `pickle.loads`, `shell=True`, `os.system`), obfuscated base64 droppers, or Trojan Source Unicode characters.
  - Real functional implementation included (not a hollow "Code Boubou" timestamp/whitespace PR).
- [ ] **Binding Non-Monetary & Zero-Payment Attestation ([`LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md`](../LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md)):**
  - I confirm that this contribution is strictly voluntary and non-monetary, contains **zero payment links, invoices, or cryptocurrency wallet addresses**, and that I waive any claim to fiat, cash, or monetary remuneration.
- [ ] **Contributor License Agreement & Dual-License IP Protection ([`CLA.md`](../CLA.md)):**
  - `I have read the CLA Document and I hereby sign the CLA` (I have read and agree to the Project Hive Mind Contributor License Agreement `CLA.md` v1.0, including Section 2 Dual-Licensing & Commercial Rights, Article L. 121-7 French IP Code, and Section 9 French Jurisdiction, and confirm that I have the legal authority to grant the rights therein).
