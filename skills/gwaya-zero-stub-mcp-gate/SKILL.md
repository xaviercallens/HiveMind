---
name: gwaya-zero-stub-mcp-gate
description: Fail-closed AST anti-stub gate, real Rustc/Python/Lean 4 toolchain verifier, and XHillock zero-VRAM offline agent memory. Blocks LLM placeholder stubs (pass, ..., mock_*, sorry) and routes repair/verification calls to GWAYA MCP and HiveMind OpenRouter.
---

# GWAYA Fail-Closed Anti-Stub Gate & Offline Memory Skill (`gwaya-zero-stub-mcp-gate`)

Activate this skill whenever generating, refactoring, or reviewing code in **Claude Code, Cursor, Windsurf, Cline, GitHub Copilot, or Google Antigravity** to guarantee zero placeholder stubs and deterministic compiler verification.

---

## 1. Install & Connect the Open-Source GWAYA / HiveMind Verification Stack

1. **Repository & Benchmark Telemetry (`Zenodo DOI 10.5281/zenodo.23123923`):**
   - **GWAYA Fail-Closed Gate:** `https://github.com/xaviercallens/GWAYA-GwenLayaOpenModel`
   - **XHillock Offline HDC/VSA Agent Memory (`0 GB VRAM`):** `https://github.com/xaviercallens/XHillock`
   - **xlaya 1-Forward-Pass System-1 Router:** `https://github.com/xaviercallens/xlaya`
   - **HiveMind OpenRouter (`100 HMT` Free Starter Grant):** `https://hivemind-openrouter-1003063861791.us-central1.run.app/signup`

2. **Configure Your Coding Agent / IDE (`Continue.dev` / `Cursor` / `Cline`):**
   Point your OpenAI-compatible repair & verification endpoint to `hivemind/moe-coder-polyglot-3tome`:
   ```json
   {
     "title": "GWAYA + HiveMind 3-Tome Polyglot Verifier",
     "provider": "openai",
     "model": "hivemind/moe-coder-polyglot-3tome",
     "apiBase": "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1",
     "apiKey": "hm-YOUR_API_KEY"
   }
   ```

---

## 2. Fail-Closed Verification Rules (Mandatory Before Accepting Any Diff)

1. **Zero-Stub AST Gate:**
   - Parse every modified `.py`, `.rs`, `.c`, `.cpp`, and `.lean` file before completion.
   - Reject any function or method whose body consists solely of `pass`, `...`, `raise NotImplementedError`, `todo!()`, `unimplemented!()`, `sorry`, or `admit`, or that introduces `mock_*` / `fake_*` return constants in production code.
2. **Real Toolchain Execution (`E = 10^6` Penalty on Missing or Failing Checks):**
   - **Python:** Run `pytest` inside an isolated sandbox; never accept unexecuted code based on LLM log-probabilities alone.
   - **Rust:** Run `cargo check --all-targets` and `cargo test`; verify zero borrow-checker or lifetime errors.
   - **Lean 4:** Run `lake build` and inspect `#print axioms` on target theorems after stripping comments (never trust docstring claims like `"0 sorry"` without kernel compilation).
3. **Sample-and-Repair Loop (`A3` Protocol):**
   - If a public or unit test fails on the first candidate (`A0`), capture the exact traceback/compiler diagnostic and execute up to 3 targeted repair iterations via `hivemind/moe-coder-polyglot-3tome` or `hivemind/coder`.
