# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Model Context Protocol server for the Solver Bee (stdio transport, stdlib only, no MCP SDK).

Transport: newline-delimited JSON-RPC 2.0 on stdin/stdout, one UTF-8 JSON object per line. Nothing but protocol messages is ever
written to stdout (the launcher script moves stdout to stderr); logs go to stderr. See docs/solver_bee/MCP.md.

Tools: ``solve_ode``, ``list_problems``, ``list_families``, ``verify_result``. Every solve goes through the same pipeline as the HTTP
service (``service.SolverBeeService``): spec validation, a sandboxed run of the pinned ``hivemind-solver`` binary, the independent
verifier (imported lazily; any failure gives UNVERIFIED, fail closed).

Specs and results are DATA. They are parsed as JSON and never evaluated. A VERIFIED verdict means the listed checks passed; it never
means the physical model is right. Units are the caller's responsibility.

Error policy: malformed JSON-RPC and malformed arguments (wrong shape, missing or extra keys, unknown tool) give JSON-RPC errors
(-32700, -32600, -32601, -32602). Spec content that the solver contract rejects, an oversize spec and a busy server give a tool
result with ``isError: true``, so the model can read the reason and correct its input.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import threading
from typing import Any, BinaryIO, Callable, Dict, List, Optional

from hivemind.solverbee import sandbox as sbx
from hivemind.solverbee import service as svc_mod

log = logging.getLogger("hivemind.solverbee.mcp")

PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")  # newest first; the first entry is the fallback
SERVER_NAME = "hivemind-solver-bee"
SERVER_VERSION = "0.1.0"
MAX_LINE_BYTES = 2 * 1024 * 1024  # one JSON-RPC message; a 64 KiB spec and a large result fit well inside
DEFAULT_TIMEOUT_S = 15.0  # per solver run; a solve makes at most two runs (the solve and the tightened rerun), so the total is about 30 s

E_PARSE, E_INVALID_REQUEST, E_METHOD_NOT_FOUND, E_INVALID_PARAMS, E_INTERNAL = -32700, -32600, -32601, -32602, -32603

INSTRUCTIONS = (
    "Solver Bee: certified numeric ODE solving with a pinned solver and an independent verifier. Specs and results are data and are never "
    "executed. VERIFIED means the listed checks passed; it does NOT mean the physical model is right. Units are the caller's responsibility."
)

_SPEC_DESC = (
    "A hivemind-solver-spec/1 object. Required: schema (must be \"hivemind-solver-spec/1\"); id (1 to 64 characters from A-Za-z0-9_.-); "
    "problem, which is one of {\"kind\":\"named\",\"family\":...,\"params\":{...}} (use list_families for the names and parameters), "
    "{\"kind\":\"linear\",\"A\":[[...]],\"b\":[...]} (x' = A x + b, n <= 20), or {\"kind\":\"mass_action\",\"species\":[...],\"reactions\":[...]} "
    "(at most 20 species and 50 reactions); y0 (finite numbers, one per variable); t_out (strictly increasing numbers, all greater than t0, "
    "at most 1000). Optional: t0 (default 0), rtol (default 1e-8), atol (default 1e-10; a number or a list of n numbers), method (\"bdf\" "
    "default, or \"adams\"), max_steps (1 to 1000000, default 100000). Unknown fields are rejected. At most 64 KiB. "
    "UNITS ARE YOUR RESPONSIBILITY: the solver does not know or convert units. This object is DATA ONLY: it is parsed as JSON and never executed or "
    "evaluated, so any text in it is inert."
)

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "solve_ode",
        "description": (
            "Solve an ordinary differential equation initial-value problem with the pinned rusty-SUNDIALS CVODE solver, run in a sandbox "
            "(no network, no file access), then check the answer with an independent verifier. Returns a plain-language summary, the "
            "trajectory (result) and a certificate. Verdicts: VERIFIED = the listed checks passed (for a library problem the reference matched, "
            "a tighter-tolerance rerun agreed, and invariants held where defined); AUDITED = no library reference for this id, but the other "
            "checks passed; FAILED = a check failed; UNVERIFIED = the answer could not be checked (fail closed). VERIFIED does NOT mean the "
            "physical model is right: whether the equations, parameters and units describe your system is your responsibility."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {"spec": {"type": "object", "description": _SPEC_DESC}},
            "required": ["spec"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_problems",
        "description": (
            "List the reference problems in the Solver Bee library: id and a one-line description. A spec with the same id and the same "
            "problem, y0, t0 and t_out as a library entry is checked against that entry's reference by solve_ode. Read-only."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_families",
        "description": (
            "List the named ODE families the solver knows, with dimension, equations (documentation text, never executed), parameters and "
            "their defaults, plus the spec kinds and limits. Use it to write a spec with problem.kind \"named\". Read-only."
        ),
        "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "verify_result",
        "description": (
            "Check a solver result that was produced elsewhere. The pinned solver is run again at tighter tolerances, and the same verifier "
            "rules as solve_ode apply. The result's input_sha256 must be the SHA-256 of the spec serialised as compact JSON (separators ',' "
            "and ':', keys in the order given, UTF-8, no ASCII escaping); otherwise the hashes check fails. The easiest way to get a matching "
            "result is solve_ode with the same spec object. Verdicts are the same as solve_ode. VERIFIED does NOT mean the physical model is "
            "right; the model and the units are your responsibility. Spec and result are data and are never executed."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec": {"type": "object", "description": _SPEC_DESC},
                "result": {
                    "type": "object",
                    "description": (
                        "A hivemind-solver-result/1 object as returned by the solver: schema, id, status, t, y, stats, solver and input_sha256. "
                        "Only the verdict is returned; a result that does not match the spec, or that was edited, is not verified."
                    ),
                },
            },
            "required": ["spec", "result"],
            "additionalProperties": False,
        },
    },
]
for _t in TOOLS:
    _t["annotations"] = {"readOnlyHint": True, "openWorldHint": False}
TOOL_NAMES = tuple(t["name"] for t in TOOLS)


class McpError(Exception):
    """A JSON-RPC error to return to the client."""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code, self.message = code, message


def spec_bytes_of(spec: Any) -> bytes:
    """The bytes a spec object is hashed and solved as: compact JSON, keys in the order given, UTF-8, no ASCII escaping."""
    return json.dumps(spec, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _rpc_error(rid: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}


def _tool_result(text: str, payload: Dict[str, Any], *, is_error: bool = False) -> Dict[str, Any]:
    return {
        "content": [{"type": "text", "text": text}, {"type": "text", "text": json.dumps(payload, ensure_ascii=False, indent=1, allow_nan=False)}],
        "structuredContent": payload,
        "isError": is_error,
    }


def _check_args(schema: Dict[str, Any], args: Dict[str, Any]) -> None:
    """Shape check against the tool's inputSchema: required keys present, no unknown keys, object-typed values are objects."""
    props = schema.get("properties", {})
    for key in schema.get("required", []):
        if key not in args:
            raise McpError(E_INVALID_PARAMS, f"missing required argument '{key}'")
    for key, value in args.items():
        if key not in props:
            raise McpError(E_INVALID_PARAMS, f"unknown argument '{str(key)[:40]}'")
        if props[key].get("type") == "object" and not isinstance(value, dict):
            raise McpError(E_INVALID_PARAMS, f"argument '{key}' must be a JSON object")


def _clean(text: Any, limit: int = 200) -> str:
    return svc_mod._clean(text, limit)


class SolverBeeMcp:
    """One MCP session. ``handle_line`` takes one raw message and returns the response object, or None for notifications."""

    def __init__(self, service: svc_mod.SolverBeeService, *, sandbox_mode: str, problems_dir: Optional[str]) -> None:
        if sandbox_mode not in ("namespace", "none", "unavailable"):
            raise ValueError("sandbox_mode must be 'namespace', 'none', or 'unavailable'")
        self.svc = service
        self.sandbox_mode = sandbox_mode  # "namespace" | "none" | "unavailable"; reported in every tool result
        self.problems_dir = problems_dir
        self._initialized = False
        self._busy = threading.Lock()  # one solve at a time per server process
        self._families: Optional[Dict[str, Any]] = None
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
            "solve_ode": self._solve_ode,
            "list_problems": self._list_problems,
            "list_families": self._list_families,
            "verify_result": self._verify_result,
        }

    # ------------------------------------------------------------ JSON-RPC framing
    def handle_line(self, raw: bytes) -> Optional[Dict[str, Any]]:
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError("raw line must be bytes")
        if len(raw) > MAX_LINE_BYTES:
            return _rpc_error(None, E_INVALID_REQUEST, f"message larger than {MAX_LINE_BYTES} bytes")
        try:
            msg = json.loads(raw.decode("utf-8"), parse_constant=svc_mod._reject_constant, object_pairs_hook=svc_mod._no_dup_pairs)
        except (UnicodeDecodeError, ValueError, RecursionError):
            return _rpc_error(None, E_PARSE, "parse error: not valid JSON (or duplicate keys / non-finite numbers)")
        return self.handle_message(msg)

    def handle_message(self, msg: Any) -> Optional[Dict[str, Any]]:
        if isinstance(msg, list):
            return _rpc_error(None, E_INVALID_REQUEST, "batch requests are not supported")
        if not isinstance(msg, dict):
            return _rpc_error(None, E_INVALID_REQUEST, "request must be a JSON object")
        has_id = "id" in msg
        rid = msg.get("id")
        if has_id and rid is not None and (isinstance(rid, bool) or not isinstance(rid, (str, int))):
            return _rpc_error(None, E_INVALID_REQUEST, "id must be a string, an integer or null")
        if msg.get("jsonrpc") != "2.0":
            return _rpc_error(rid if has_id else None, E_INVALID_REQUEST, "jsonrpc must be '2.0'")
        if "method" not in msg:
            if "result" in msg or "error" in msg:
                return None  # a response to a request this server never sent
            return _rpc_error(rid if has_id else None, E_INVALID_REQUEST, "missing method")
        method = msg["method"]
        if not isinstance(method, str):
            return _rpc_error(rid if has_id else None, E_INVALID_REQUEST, "method must be a string")
        params = msg.get("params")
        if params is not None and not isinstance(params, (dict, list)):
            return _rpc_error(rid if has_id else None, E_INVALID_PARAMS, "params must be an object or an array") if has_id else None
        if not has_id:
            return None  # notifications (for example notifications/initialized) never get a response
        try:
            return {"jsonrpc": "2.0", "id": rid, "result": self._dispatch(method, params)}
        except McpError as exc:
            return _rpc_error(rid, exc.code, exc.message)
        except Exception as exc:  # noqa: BLE001  never leak internals to the client
            log.error("internal error in %s: %s", method, type(exc).__name__)
            return _rpc_error(rid, E_INTERNAL, "internal error")

    def _dispatch(self, method: str, params: Any) -> Dict[str, Any]:
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            self._require_initialized()
            return {"tools": TOOLS}
        if method == "tools/call":
            self._require_initialized()
            return self._call_tool(params)
        raise McpError(E_METHOD_NOT_FOUND, "method not found: " + method[:80])

    def _require_initialized(self) -> None:
        if not self._initialized:
            raise McpError(E_INVALID_REQUEST, "server not initialized: send initialize first")

    def _initialize(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict) or not isinstance(params.get("protocolVersion"), str):
            raise McpError(E_INVALID_PARAMS, "initialize needs params.protocolVersion (a string)")
        requested = params["protocolVersion"]
        chosen = requested if requested in PROTOCOL_VERSIONS else PROTOCOL_VERSIONS[0]
        self._initialized = True
        return {
            "protocolVersion": chosen,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": INSTRUCTIONS + " Sandbox: " + self.sandbox_mode + ".",
        }

    def _call_tool(self, params: Any) -> Dict[str, Any]:
        if not isinstance(params, dict):
            raise McpError(E_INVALID_PARAMS, "tools/call needs params")
        name = params.get("name")
        if not isinstance(name, str):
            raise McpError(E_INVALID_PARAMS, "tools/call needs params.name (a string)")
        if name not in self._handlers:
            raise McpError(E_INVALID_PARAMS, "unknown tool: " + name[:80])
        args = params.get("arguments", {})
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise McpError(E_INVALID_PARAMS, "arguments must be a JSON object")
        schema = next(t["inputSchema"] for t in TOOLS if t["name"] == name)
        _check_args(schema, args)
        try:
            return self._handlers[name](args)
        except McpError:
            raise
        except Exception as exc:  # noqa: BLE001  a tool failure is a tool result, never a crash
            log.error("tool %s failed: %s", name, type(exc).__name__)
            return self._tool_error("internal error; treat the answer as UNVERIFIED")

    def _tool_error(self, message: str) -> Dict[str, Any]:
        msg = _clean(message, 300)
        return _tool_result("Error: " + msg, {"error": msg, "sandbox": self.sandbox_mode}, is_error=True)

    # ------------------------------------------------------------ tools
    def _solve_ode(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sb = spec_bytes_of(args["spec"])
        except (TypeError, ValueError):
            return self._tool_error("spec is not serialisable JSON")
        if not self._busy.acquire(blocking=False):
            return self._tool_error("busy: a solve is already running; retry when it finishes")
        try:
            code, rep = self.svc.solve(sb)
        finally:
            self._busy.release()
        if code != 200:
            return self._tool_error(rep["error"]["message"])
        payload = {"verdict": rep["verdict"], "sandbox": self.sandbox_mode, "result": rep["result"], "certificate": rep["certificate"],
                   "solver_bee": rep["solver_bee"]}
        return _tool_result(rep["summary"], payload)

    def _list_problems(self, args: Dict[str, Any]) -> Dict[str, Any]:
        del args
        items: List[Dict[str, Any]] = []
        for pid in sorted(self.svc.library):
            p = self.svc.library[pid]
            spec = p.get("spec") if isinstance(p.get("spec"), dict) else {}
            desc = p.get("description") if isinstance(p.get("description"), str) else ""
            items.append({"id": pid, "description": desc[:200], "kind": (spec.get("problem") or {}).get("kind") if isinstance(spec.get("problem"), dict) else None,
                          "dimension": len(spec["y0"]) if isinstance(spec.get("y0"), list) else None})
        if self.problems_dir is None:
            text = "No problems directory is configured (start the server with --problems DIR)."
        else:
            text = f"{len(items)} reference problems:\n" + "\n".join(f"- {i['id']}: {i['description']}" for i in items)
        return _tool_result(text, {"problems": items, "count": len(items), "sandbox": self.sandbox_mode})

    def _list_families(self, args: Dict[str, Any]) -> Dict[str, Any]:
        del args
        if self._families is None:
            o = sbx.run_binary([self.svc.binary, "families"], b"", limits=self.svc.limits)
            if o.refused or not o.ok or o.rc != 0 or o.limit:
                return self._tool_error("could not list families: " + _clean(o.error or f"exit {o.rc} {o.limit or ''}"))
            try:
                data = json.loads(o.stdout.decode("utf-8"))
                fams = [f for f in data["families"] if isinstance(f, dict)]
            except (UnicodeDecodeError, ValueError, KeyError, TypeError, RecursionError):
                return self._tool_error("could not list families: the solver output is not the expected JSON")
            self._families = {"families": fams, "kinds": data.get("kinds", []), "limits": data.get("limits", {})}
        lines = []
        for f in self._families["families"]:
            ps = ", ".join(f"{_clean(p.get('name'), 30)}={_clean(p.get('default'), 30)}" for p in (f.get("params") or []) if isinstance(p, dict))
            lines.append(f"- {_clean(f.get('family'), 40)} (dimension {_clean(f.get('dim'), 8)}): {ps or 'no parameters'}")
        text = f"{len(lines)} named families (parameters with defaults):\n" + "\n".join(lines)
        return _tool_result(text, {**self._families, "sandbox": self.sandbox_mode})

    def _verify_result(self, args: Dict[str, Any]) -> Dict[str, Any]:
        try:
            sb = spec_bytes_of(args["spec"])
            spec = svc_mod.validate_spec_bytes(sb)
        except svc_mod.SpecError as exc:
            return self._tool_error(str(exc))
        except (TypeError, ValueError):
            return self._tool_error("spec is not serialisable JSON")
        if not self._busy.acquire(blocking=False):
            return self._tool_error("busy: a solve is already running; retry when it finishes")
        try:
            cert = self._certify(sb, spec, args["result"])
        finally:
            self._busy.release()
        checks = ", ".join(f"{_clean(c.get('name'), 30)} {_clean(c.get('status'), 10)}" for c in (cert.get("checks") or []) if isinstance(c, dict))
        reasons = "; ".join(_clean(r, 160) for r in (cert.get("reasons") or [])[:4])
        text = f"Certificate for spec {_clean(spec['id'], 64)}: {cert['verdict']} (sandbox: {self.sandbox_mode})."
        if checks:
            text += f" Checks: {checks}."
        if reasons:
            text += f" Reasons: {reasons}."
        text += " " + svc_mod.CAVEAT
        return _tool_result(text, {"verdict": cert["verdict"], "sandbox": self.sandbox_mode, "certificate": cert})

    def _certify(self, spec_bytes: bytes, spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """The verifier's certificate for a result produced elsewhere. Same rules as the service pipeline; the rerun uses the pinned binary."""
        svc = self.svc
        input_sha = hashlib.sha256(spec_bytes).hexdigest()
        verifier, why = svc.verifier()
        if verifier is None:
            return svc._unverified(spec["id"], input_sha, [why], result, None)

        def rerun(tight: bytes) -> Dict[str, Any]:
            svc_mod.validate_spec_bytes(tight)
            return svc._run_solver(tight, tail={})

        try:
            cert = verifier.verify(spec_bytes, result, problem=svc._library_problem(spec), rerun=rerun, binary_sha256=None)
            return svc._check_cert(cert, input_sha, result)
        except Exception as exc:  # noqa: BLE001  fail closed
            return svc._unverified(spec["id"], input_sha, [f"verification failed with {type(exc).__name__}: {_clean(exc, 120)}"], result, None)


def build_server(binary: str, *, problems_dir: Optional[str] = None, timeout_s: float = DEFAULT_TIMEOUT_S, allow_unsandboxed: bool = False,
                 verifier: Any = None) -> SolverBeeMcp:
    """The server for one binary. With ``allow_unsandboxed`` every solver run is forced without namespaces and every result says sandbox 'none'."""
    svc = svc_mod.SolverBeeService(binary, problems_dir=problems_dir, verifier=verifier, max_concurrent=1, timeout_s=timeout_s,
                                   allow_unsandboxed=allow_unsandboxed)
    if allow_unsandboxed:
        svc.limits = dataclasses.replace(svc.limits, force_none=True)
        mode = sbx.SANDBOX_NONE
    else:
        mode = sbx.SANDBOX_NAMESPACE if sbx.namespace_available() else "unavailable"
    return SolverBeeMcp(svc, sandbox_mode=mode, problems_dir=problems_dir)


def read_frame(stream: BinaryIO, limit: int = MAX_LINE_BYTES) -> Optional[bytes]:
    """The next line without its newline, or None at end of input. An oversize line is drained and returned truncated, so it is rejected."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    line = stream.readline(limit + 1)
    if line == b"":
        return None
    if len(line) > limit and not line.endswith(b"\n"):
        while True:
            chunk = stream.readline(65536)
            if chunk == b"" or chunk.endswith(b"\n"):
                break
    return line.rstrip(b"\r\n")


def encode(resp: Dict[str, Any]) -> bytes:
    """One protocol line: ASCII only (no U+2028 or other line separators), compact, no raw newlines."""
    if not isinstance(resp, dict):
        raise ValueError("resp must be a dict")
    try:
        text = json.dumps(resp, ensure_ascii=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        text = json.dumps(_rpc_error(resp.get("id"), E_INTERNAL, "internal error: response not serialisable"), separators=(",", ":"))
    return (text + "\n").encode("ascii")
