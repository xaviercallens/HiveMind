# Solver Bee service (Wave 1, W1-4 and W1-5)

A loopback-only HTTP service that takes an ODE initial-value problem as **data**, runs the pinned `hivemind-solver` binary in a namespace sandbox, has the independent verifier check the result, and answers with the result plus a certificate. Stdlib only. Code: `src/hivemind/solverbee/sandbox.py`, `src/hivemind/solverbee/service.py`, `scripts/solver_bee.py`. The Rust binary (PR-A) and the verifier (PR-B) are separate; this service codes against the contract (`hivemind-solver-spec/1`, `hivemind-solver-result/1`, `hivemind-solver-cert/1`) and imports the verifier lazily. Without a verifier every answer is UNVERIFIED (fail closed).

## Run

```
python3 scripts/solver_bee.py serve --binary /path/to/hivemind-solver [--problems solvers/problems] [--port 8100] [--max-concurrent 2] [--timeout 30]
```

Binds `127.0.0.1` only. Another host is refused unless both `--nonlocal` and `--api-key-file FILE` (a Bearer token of at least 16 characters) are given, the same rule as `scripts/gwaya_lite.py`. If user namespaces are unavailable the service refuses to run solvers (every answer UNVERIFIED with the reason) unless `--allow-unsandboxed` is given; every answer then says `sandbox: "none"`.

`GET /health` returns `{status, sandbox: namespace|none|unavailable, verifier: bool, problems, max_concurrent, in_flight, auth}`.

## Requests

Chat (OpenAI-compatible). The last user message must contain exactly one fenced block tagged `ode-spec` holding a spec; the bytes between the fences are hashed as received.

````
curl -s http://127.0.0.1:8100/v1/chat/completions -H 'Content-Type: application/json' -d '{
 "messages":[{"role":"user","content":"Integrate this.\n```ode-spec\n{\"schema\":\"hivemind-solver-spec/1\",\"id\":\"demo.1\",\"problem\":{\"kind\":\"linear\",\"A\":[[-1,0],[0,-2]],\"b\":[0,0]},\"y0\":[1,1],\"t_out\":[0.5,1.0]}\n```"}]}'
````

The reply is a `chat.completion` object whose assistant message is a one-paragraph summary, a fenced `json` block (the solver result) and a fenced `hivemind-cert` block (the certificate). The response also carries `solver_bee`: `{verdict, input_sha256, result_sha256, sandbox: "namespace"|"none", wall_ms}`. `usage` counts **bytes**, not model tokens (`"unit": "bytes"`). `stream: true` is rejected.

Plain: `POST /v1/solve` with `{"spec": {...}}` returns `{verdict, summary, result, certificate, solver_bee}`. Here the input hash is of the compact re-serialisation of the spec object, not of the client's original bytes (the chat route hashes the exact block bytes).

Errors use the OpenAI shape `{"error":{"message","type"}}`: 400 malformed request or spec (unknown field, bad JSON, NaN, duplicate key, sizes, two blocks, no block), 401 bad key, 403 Host/Origin not loopback, 404, 411/413 body framing and size (body limit 512 KiB, spec limit 64 KiB), 429 all slots busy (`Retry-After: 1`, no queueing on the host CPU). Verification outcomes, including every failure to run or verify, are HTTP 200 with a certificate.

## Verdicts

Only the verifier's verdict is used. Vocabulary as in GWAYA-Lite:

| Verdict | Meaning |
|---|---|
| VERIFIED | reference (problem is in the library) and invariants, tightening, hashes and shape checks all passed. It does not mean "correct": in the E16 measurement 5-7% of VERIFIED code answers still failed hidden tests; the caveat is in every certificate and summary. |
| AUDITED | no reference for this spec; shape, hashes, invariants-if-any and tightening passed. |
| FAILED | at least one check failed. |
| UNVERIFIED | the result could not be obtained or checked: solver crash, timeout or limit kill, bad output, solver `status=error`, missing binary, refused sandbox, missing verifier, verifier exception, or a certificate whose input or result hash does not match the request (fail closed). The reasons are listed. The service builds this certificate itself (no checks). |

If the spec id matches a `--problems` entry, the verifier receives that entry as `problem`, but only when the submitted `problem`, `y0`, `t0` and `t_out` equal the library spec's; otherwise the reference does not apply and the answer is at best AUDITED. The verifier gets a `rerun` callable that validates the tightened spec and runs it through the same sandbox.

## Security properties and limits

- Specs are data. They are validated structurally (unknown fields rejected, sizes bounded, finite numbers only) and handed to the binary on stdin; nothing in a spec is evaluated, formatted into a command, or used as a path. No `eval`, `exec`, `pickle` or `shell=True` in the code (a test checks the AST). Log lines carry only verdict, sandbox mode, timing and byte counts, never bodies.
- The binary runs under the same primitives as the E15 gate: user, mount, net, pid, ipc and uts namespaces; tmpfs over `/home`, `/root`, `/tmp` and other host directories; minimal `/dev`; all capabilities dropped, no_new_privs, non-dumpable; environment reduced to `PATH`; RLIMIT_AS (2 GiB default), CPU, NPROC, FSIZE, NOFILE, CORE; wall-clock timeout (30 s default) with process-group kill; stdout capped at 8 MiB (kill on overflow); stdin at most 64 KiB; the binary is copied into the tmpfs after its sha256 is checked.
- Loopback only, Host and Origin guard against DNS rebinding and drive-by pages, at most N solves in flight.
- Caveats inherited from E15: **no seccomp filter**, **shared host kernel**, **no cgroup memory or pids limit** (address space is limited per process; NPROC is counted per real uid), the work tmpfs is writable up to a bound, and the `memory` limit label is a heuristic (set when stderr shows an allocation failure). Single host, no TLS, no authentication on loopback: any local process can use the port.
- Floating-point results across different CPUs are not claimed identical (experiment S-1 measures that).

## Later: the OpenRouter producer shield

Not implemented here. The intended plug-in point is the producer shield treating this service as one more local OpenAI-compatible upstream bound to loopback: the shield forwards a chat request, keeps the `solver_bee` object and the certificate block with the answer, and never upgrades a verdict. Wiring, pricing and routing are separate owner decisions.
