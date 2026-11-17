# Solver Bee MCP server (Wave 1, stage 2 P4)

A Model Context Protocol server over stdio that lets an MCP client (an assistant that speaks MCP) solve ODE initial-value problems with the pinned `hivemind-solver` binary and check every answer with the independent verifier. Stdlib only, no MCP SDK. Code: `src/hivemind/solverbee/mcp.py` (protocol and tools), `scripts/solver_bee_mcp.py` (the launcher), `tests/test_solverbee_mcp.py`. It reuses the HTTP service pipeline (`src/hivemind/solverbee/service.py`) and the sandbox (`src/hivemind/solverbee/sandbox.py`); see `docs/solver_bee/SERVICE.md` and `docs/solver_bee/VERIFIER.md`.

## What a verdict means (read this first)

- **VERIFIED**: the listed checks passed (hashes, the library reference for a library problem, invariants where defined, and agreement with a rerun at 100 times tighter tolerances). It does **not** mean the physical model is right. Whether the equations, parameters and units describe your system is your responsibility. The solver does not know or convert units.
- **AUDITED**: no library reference exists for this spec id, but the other checks passed.
- **FAILED**: a check failed. The reasons are listed.
- **UNVERIFIED**: the answer could not be obtained or checked (fail closed). Examples: the solver crashed, the sandbox refused, the verifier is missing.

Specs and results are **data**. They are parsed as JSON and never evaluated. A string such as `__import__('os')` inside a spec is an inert value.

## Run it

```
python3 scripts/solver_bee_mcp.py --binary /path/to/hivemind-solver [--problems solvers/problems] [--timeout 15] [--allow-unsandboxed]
```

- `--binary` is the pinned `hivemind-solver` executable (build it from `solvers/hivemind-solver`).
- `--problems` is the reference library, read at startup and never written. It defaults to `solvers/problems` in this repository.
- `--timeout` is the wall-clock limit for one solver run. A solve makes at most two runs (the answer, then the tightened rerun), so the total is about 30 s plus process start-up.
- Without user namespaces the server **refuses to start**, unless `--allow-unsandboxed` is given. With that flag the solver runs without namespaces (resource limits and a scrubbed environment only), and **every tool result says `"sandbox": "none"`**.

Stdout carries only protocol messages, one JSON-RPC object per line, ASCII. Logs and diagnostics go to stderr. Exit code 0 at end of input, 3 for a usage or environment error.

## Client configuration

Client-specific configuration syntax: verify against your client's documentation. The JSON below shows the command and arguments that every stdio MCP client needs.

Sandboxed (the default):

```json
{
  "mcpServers": {
    "solver-bee": {
      "command": "python3",
      "args": [
        "/path/to/HiveMind/scripts/solver_bee_mcp.py",
        "--binary", "/path/to/hivemind-solver",
        "--problems", "/path/to/HiveMind/solvers/problems"
      ]
    }
  }
}
```

Unsandboxed (only where user namespaces are unavailable and you accept `sandbox: "none"`):

```json
{
  "mcpServers": {
    "solver-bee": {
      "command": "python3",
      "args": [
        "/path/to/HiveMind/scripts/solver_bee_mcp.py",
        "--binary", "/path/to/hivemind-solver",
        "--problems", "/path/to/HiveMind/solvers/problems",
        "--allow-unsandboxed"
      ]
    }
  }
}
```

## Tools

| Tool | Arguments | Returns |
| --- | --- | --- |
| `solve_ode` | `spec` (object, `hivemind-solver-spec/1`) | Two text blocks: a plain summary with the verdict, and the JSON with `verdict`, `sandbox`, `result` (the trajectory), `certificate` and `solver_bee`. The same object is also in `structuredContent`. |
| `list_problems` | none | The reference problems: `id`, `description`, `kind`, `dimension`. |
| `list_families` | none | The named families with dimension, equations (documentation text, never executed), parameters and defaults, plus the spec kinds and limits. It runs `hivemind-solver families`. |
| `verify_result` | `spec` (object), `result` (object, `hivemind-solver-result/1`) | A certificate for a result produced elsewhere. It reruns the pinned solver at tighter tolerances. |

Tool input schemas are JSON Schema (`additionalProperties: false`) and carry descriptions for the model. The spec schema is the same as the HTTP service's: `docs/solver_bee/SERVICE.md` and `docs/solver_bee/VERIFIER.md` have the field list.

`verify_result` hashes the spec as compact JSON (separators `,` and `:`, keys in the order given, UTF-8). The result's `input_sha256` must equal that hash. The easy route is `solve_ode` with the same spec object, then `verify_result` with both unchanged. A result computed from a pretty-printed file will fail the hashes check. That is intended.

## Errors and limits

| Situation | Response |
| --- | --- |
| Not JSON, duplicate keys, `NaN` | JSON-RPC `-32700` (parse error), id `null` |
| Not an object, batch array, missing `jsonrpc` or `method` | `-32600` (invalid request) |
| Unknown method (for example `resources/list`) | `-32601` (method not found) |
| Missing or extra arguments, wrong argument types, unknown tool, `initialize` without `protocolVersion` | `-32602` (invalid params) |
| Call before `initialize` | `-32600` |
| Spec rejected by the solver contract (unknown field, bad id, shapes, and so on) | Tool result with `isError: true` and the reason, so the model can fix the spec |
| Spec over 64 KiB | Tool result with `isError: true` |
| A second solve while one is running | Tool result with `isError: true` ("busy") |
| Solver failure, timeout, crash, verifier missing | Tool result, verdict `UNVERIFIED` (not `isError`) |

Other limits: one solve at a time per server process (the server handles one message at a time); one JSON-RPC message at most 2 MiB; no network; the solver never reads files other than its input on stdin, and the problems directory is read-only. Each result carries `"sandbox"`: `namespace` (the sandbox was active), `none` (unsandboxed mode) or `unavailable`.

The sandbox caveats of `sandbox.py` apply: no seccomp filter, a shared host kernel, and no cgroup memory limit (the address-space rlimit is per process).

Protocol versions accepted: `2025-06-18`, `2025-03-26` and `2024-11-05`. The server answers an unknown version with its newest one. Batch requests are not supported. The server offers tools only (no resources or prompts).

## Example conversation

The user asks for the undamped oscillator from the reference library. The assistant calls `solve_ode`, the server runs the real pinned solver in the namespace sandbox, and the answer is checked against the library reference. The numbers below are from a real run of this server against the release binary (`harmonic_undamped`, omega = 2, x(0) = 1, v(0) = 0.5).

> **User:** Integrate the undamped oscillator from the library at t = 0.5, 1, 2, 5 and 10.
>
> **Assistant:** *(calls `solve_ode` with the `harmonic_undamped` spec)*
>
> **Tool result (summary):** Solver Bee verdict for spec harmonic_undamped: VERIFIED (sandbox: namespace, 159 ms). Checks: shape pass, hashes pass, reference pass, invariants pass, tightening pass. Reasons: shape, hashes, reference, invariants, tightening passed. VERIFIED means the listed checks passed. In the E16 measurement 5-7% of VERIFIED code answers still failed hidden tests; it never means 'correct'.
>
> **Assistant:** The solver and verifier agree with the library reference. The position at t = 0.5 is 0.75067, and at t = 10 it is 0.63632 (velocity -1.62185). Here the reference is the closed form x(t) = cos(2t) + 0.25 sin(2t), which I can check by hand: at t = 0.5 it gives 0.7507. The verdict covers the checks only. Whether this oscillator is the one in your system, with those units and parameters, is for you to confirm.

## Tests

`tests/test_solverbee_mcp.py` runs scripted JSON-RPC sessions in-process against a fake solver executable (contract-shaped results and a family list) and with the injected or the real verifier. It covers the handshake and version negotiation, notifications, every error code, tool schemas, malformed and oversize input, hostile strings, `isError` and busy handling, the sandbox flags, and the startup refusal. It also runs a session against a subprocess of the real script, checking that every stdout line is a JSON-RPC message. A real-binary session (`harmonic_undamped`, expected VERIFIED) is skipped when the release binary is missing.
