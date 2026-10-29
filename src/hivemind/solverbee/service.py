# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Solver Bee service: a LOOPBACK-ONLY HTTP front for the sandboxed solver binary plus the independent verifier (stdlib only).

Pipeline per request: validate the spec bytes (size, JSON, schema/id, shapes; unknown fields rejected; nothing is ever evaluated as code) -> run the solver binary in
the sandbox (``sandbox.run_binary``) -> parse the result -> run the verifier (with a ``rerun`` that sends the tightened spec through the same sandbox) -> respond.

Verdict rule: ONLY the verifier's verdict is reported. Any exception, timeout, crash, missing binary, refused sandbox or missing verifier gives UNVERIFIED with the
reason (fail closed). Verification outcomes are HTTP 200 with the certificate; malformed requests are HTTP 4xx; busy is 429; never a bare 500.
VERIFIED means the stated checks passed, never "correct" (E16: 5-7% of VERIFIED code answers still failed hidden tests).

HTTP conventions mirror ``scripts/gwaya_lite.py``: Host and Origin must be loopback (anti DNS-rebinding), optional Bearer key, body size limit, OpenAI-style error
objects, metadata-only logs (request bodies are never logged), non-loopback binds refused unless a non-local flag AND an API key file are both given.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import ipaddress
import json
import math
import os
import re
import secrets
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from hivemind.solverbee import sandbox as sbx

SPEC_SCHEMA, RESULT_SCHEMA, CERT_SCHEMA = "hivemind-solver-spec/1", "hivemind-solver-result/1", "hivemind-solver-cert/1"
PROBLEM_SCHEMA = "hivemind-solver-problem/1"
SPEC_MAX_BYTES = 64 * 1024
MAX_BODY_BYTES = 512 * 1024  # a 64 KiB spec can grow about 6x when JSON-escaped inside a chat message
ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
VERDICTS = ("VERIFIED", "AUDITED", "FAILED", "UNVERIFIED")
MODEL_ID = "hivemind-solver-bee"
CAVEAT = ("VERIFIED means the listed checks passed. In the E16 measurement 5-7% of VERIFIED code answers still failed hidden tests; it never means 'correct'.")
FAMILIES = (
    "robertson",
    "van_der_pol",
    "brusselator",
    "lotka_volterra",
    "harmonic",
    "stiff_chain",
    "pendulum",
    "projectile_drag",
    "kepler",
    "logistic",
)
TOP_KEYS = {"schema", "id", "problem", "y0", "t0", "t_out", "rtol", "atol", "method", "max_steps"}
SPEC_BLOCK_RE = re.compile(r"^```ode-spec[ \t]*\r?\n(.*?)^```[ \t]*\r?$", re.MULTILINE | re.DOTALL)
SPEC_OPEN_RE = re.compile(r"^```ode-spec\b", re.MULTILINE)
VERIFIER_MODULE = "hivemind.solverbee.verifier"


class SpecError(ValueError):
    """The request or the spec is malformed (HTTP 400). Never carries request content beyond short field names."""


class SolverRunError(RuntimeError):
    """A solver run could not produce a usable result (used by ``rerun``)."""


# ---------------------------------------------------------------- spec validation (data only)
def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _reject_constant(name: str) -> Any:
    raise SpecError("non-finite number literal %s is not allowed" % name)


def _no_dup_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise SpecError("duplicate key in JSON object")
        d[k] = v
    return d


def _num_list(x: Any, name: str, max_len: int, min_len: int = 1) -> List[float]:
    if not isinstance(x, list) or not (min_len <= len(x) <= max_len) or not all(_is_num(v) for v in x):
        raise SpecError(f"'{name}' must be a list of {min_len}..{max_len} finite numbers")
    return x


def validate_spec_bytes(raw: bytes) -> Dict[str, Any]:
    """Parse and validate a hivemind-solver-spec/1. Returns the parsed dict (for library lookup and summaries only); raises SpecError."""
    if not isinstance(raw, (bytes, bytearray)):
        raise SpecError("spec must be bytes")
    if len(raw) > SPEC_MAX_BYTES:
        raise SpecError(f"spec larger than {SPEC_MAX_BYTES} bytes")
    if not raw.strip():
        raise SpecError("empty spec")
    try:
        spec = json.loads(bytes(raw).decode("utf-8"), parse_constant=_reject_constant, object_pairs_hook=_no_dup_pairs)
    except SpecError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise SpecError("spec is not valid UTF-8 JSON") from None
    if not isinstance(spec, dict):
        raise SpecError("spec must be a JSON object")
    unknown = sorted(set(spec) - TOP_KEYS)
    if unknown:
        raise SpecError("unknown spec field(s): " + ", ".join(repr(k[:40]) for k in unknown[:5]))
    if spec.get("schema") != SPEC_SCHEMA:
        raise SpecError(f"schema must be {SPEC_SCHEMA!r}")
    sid = spec.get("id")
    if not isinstance(sid, str) or not ID_RE.match(sid):
        raise SpecError("id must match [A-Za-z0-9_.-]{1,64}")
    for k in ("problem", "y0", "t_out"):
        if k not in spec:
            raise SpecError(f"missing field '{k}'")
    y0 = _num_list(spec["y0"], "y0", 20)
    n = len(y0)
    prob = spec["problem"]
    if not isinstance(prob, dict) or prob.get("kind") not in ("named", "linear", "mass_action"):
        raise SpecError("problem.kind must be named, linear or mass_action")
    kind = prob["kind"]
    if kind == "named":
        if set(prob) - {"kind", "family", "params"}:
            raise SpecError("unknown field in problem")
        if prob.get("family") not in FAMILIES:
            raise SpecError("problem.family must be one of " + ", ".join(FAMILIES))
        params = prob.get("params", {})
        if not isinstance(params, dict) or len(params) > 32:
            raise SpecError("problem.params must be an object with at most 32 entries")
        for pk, pv in params.items():
            if not isinstance(pk, str) or len(pk) > 64:
                raise SpecError("bad params key")
            if not (_is_num(pv) or isinstance(pv, bool) or (isinstance(pv, str) and len(pv) <= 64) or (isinstance(pv, list) and len(pv) <= 1000 and all(_is_num(v) for v in pv))):
                raise SpecError("params values must be numbers, booleans, short strings or number lists")
    elif kind == "linear":
        if set(prob) - {"kind", "A", "b"}:
            raise SpecError("unknown field in problem")
        a, b = prob.get("A"), prob.get("b")
        if not isinstance(a, list) or not 1 <= len(a) <= 20:
            raise SpecError("A must be an n x n matrix with 1 <= n <= 20")
        for row in a:
            _num_list(row, "A row", 20)
            if len(row) != len(a):
                raise SpecError("A must be square")
        _num_list(b, "b", 20)
        if len(b) != len(a) or len(a) != n:
            raise SpecError("A, b and y0 must have the same dimension")
    else:
        if set(prob) - {"kind", "species", "reactions"}:
            raise SpecError("unknown field in problem")
        sp, rx = prob.get("species"), prob.get("reactions")
        if not isinstance(sp, list) or not 1 <= len(sp) <= 20 or not all(isinstance(s, str) and 1 <= len(s) <= 64 for s in sp) or len(set(sp)) != len(sp):
            raise SpecError("species must be 1..20 distinct strings of at most 64 characters")
        if not isinstance(rx, list) or not 1 <= len(rx) <= 50:
            raise SpecError("reactions must be a list of 1..50 reactions")
        for r in rx:
            if not isinstance(r, dict) or set(r) - {"reactants", "products", "k"} or not _is_num(r.get("k")) or r["k"] < 0:
                raise SpecError("each reaction needs reactants, products and a finite k >= 0, nothing else")
            for side in ("reactants", "products"):
                m = r.get(side)
                if not isinstance(m, dict) or len(m) > 20:
                    raise SpecError(f"reaction {side} must be an object")
                for s, c in m.items():
                    if s not in sp or isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 10:
                        raise SpecError("stoichiometric coefficients must be small positive integers over declared species")
        if len(sp) != n:
            raise SpecError("y0 must have one entry per species")
    t0 = spec.get("t0", 0.0)
    if not _is_num(t0):
        raise SpecError("t0 must be a finite number")
    t_out = _num_list(spec["t_out"], "t_out", 1000)
    prev = t0
    for t in t_out:
        if not t > prev:
            raise SpecError("t_out must be strictly increasing and greater than t0")
        prev = t
    if "rtol" in spec and not (_is_num(spec["rtol"]) and spec["rtol"] > 0):
        raise SpecError("rtol must be a positive number")
    if "atol" in spec:
        at = spec["atol"]
        if isinstance(at, list):
            _num_list(at, "atol", 20)
            if len(at) != n or not all(v > 0 for v in at):
                raise SpecError("atol array must have n positive entries")
        elif not (_is_num(at) and at > 0):
            raise SpecError("atol must be a positive number or an array")
    if "method" in spec and spec["method"] not in ("bdf", "adams"):
        raise SpecError("method must be bdf or adams")
    if "max_steps" in spec:
        ms = spec["max_steps"]
        if isinstance(ms, bool) or not isinstance(ms, int) or not 1 <= ms <= 1_000_000:
            raise SpecError("max_steps must be an integer in 1..1000000")
    return spec


# ---------------------------------------------------------------- helpers
def canonical_sha256(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def _clean(text: Any, limit: int = 200) -> str:
    """Printable, single-line, bounded: for reasons derived from stderr or exceptions."""
    s = "".join(ch if ch.isprintable() else " " for ch in str(text))
    return " ".join(s.split())[:limit]


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(p.get("text", "")) for p in content if isinstance(p, dict) and p.get("type") == "text")
    return ""


def extract_spec_block(text: str) -> bytes:
    """The bytes of the single ```ode-spec fenced block (content verbatim, minus the newline before the closing fence)."""
    blocks = SPEC_BLOCK_RE.findall(text)
    opens = len(SPEC_OPEN_RE.findall(text))
    if not blocks:
        raise SpecError("the last user message must contain one ```ode-spec fenced block" + (" (found an unterminated one)" if opens else ""))
    if len(blocks) != 1 or opens != 1:
        raise SpecError("the last user message must contain exactly ONE ```ode-spec block")
    body = blocks[0]
    if body.endswith("\r\n"):
        body = body[:-2]
    elif body.endswith("\n"):
        body = body[:-1]
    return body.encode("utf-8")


def _is_loopback_host(h: str) -> bool:
    h = h.strip("[]").lower()
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _hostname_of(host_header: str) -> str:
    h = host_header.strip()
    if h.startswith("["):
        return h[1:].split("]")[0]
    return h.rsplit(":", 1)[0] if h.count(":") == 1 else h


class ServiceError(Exception):
    pass


def check_bind(host: str, allow_nonlocal: bool, api_key_file: Optional[str]) -> Optional[str]:
    """Returns the API key to enforce (or None). Refuses a non-loopback bind unless BOTH allow_nonlocal and an api key file are given (as gwaya_lite does)."""
    key = None
    if api_key_file:
        try:
            key = Path(api_key_file).read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ServiceError(f"cannot read --api-key-file: {exc.strerror}") from None
        if len(key) < 16:
            raise ServiceError("the API key must be at least 16 characters")
    if not _is_loopback_host(host):
        if not (allow_nonlocal and key):
            raise ServiceError(f"refusing to bind to non-loopback address {host!r}: pass BOTH --nonlocal and --api-key-file (Bearer auth is then required)")
    return key


def load_problem_library(directory: Optional[str]) -> Dict[str, Dict[str, Any]]:
    lib: Dict[str, Dict[str, Any]] = {}
    if not directory:
        return lib
    for p in sorted(Path(directory).glob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and d.get("schema") == PROBLEM_SCHEMA and isinstance(d.get("id"), str) and ID_RE.match(d["id"]) and isinstance(d.get("spec"), dict):
            lib[d["id"]] = d
    return lib


# ---------------------------------------------------------------- the service core
class SolverBeeService:
    def __init__(self, binary: str, *, problems_dir: Optional[str] = None, verifier: Any = None, max_concurrent: int = 2, timeout_s: float = 30.0,
                 allow_unsandboxed: bool = False, api_key: Optional[str] = None, loopback_only: bool = True, mem_mb: int = 2048,
                 log: Callable[[str], None] = lambda m: None) -> None:
        self.binary = binary
        self.library = load_problem_library(problems_dir)
        self._verifier = verifier
        self.limits = sbx.Limits(timeout_s=float(timeout_s), mem_mb=int(mem_mb), allow_unsandboxed=bool(allow_unsandboxed))
        self.max_concurrent = int(max_concurrent)
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self.api_key, self.loopback_only, self.log = api_key, loopback_only, log
        self._in_flight = 0
        self._lock = threading.Lock()

    # -- verifier (lazy; fail closed)
    def verifier(self) -> Tuple[Any, str]:
        """(verifier object or None, reason when None). The real module is imported only when none was injected."""
        if self._verifier is not None:
            return self._verifier, ""
        try:
            mod = importlib.import_module(VERIFIER_MODULE)
        except Exception as exc:  # noqa: BLE001  ImportError or anything raised while importing
            return None, f"verifier unavailable ({type(exc).__name__})"
        if not callable(getattr(mod, "verify", None)):
            return None, "verifier unavailable (module has no verify())"
        return mod, ""

    def authorised(self, header: str) -> bool:
        if not self.api_key:
            return True
        m = re.match(r"Bearer\s+(.+)$", header or "")
        return bool(m) and hmac.compare_digest(m.group(1).strip().encode(), self.api_key.encode())

    def health(self) -> Dict[str, Any]:
        v, why = self.verifier()
        ns = sbx.namespace_available()
        return {"status": "ok", "service": "solver-bee", "sandbox": "namespace" if ns else ("none" if self.limits.allow_unsandboxed else "unavailable"),
                "verifier": v is not None, "problems": len(self.library), "max_concurrent": self.max_concurrent, "in_flight": self._in_flight, "auth": bool(self.api_key),
                **({"verifier_reason": why} if why else {})}

    # -- solver runs
    def _binary_sha(self) -> Optional[str]:
        try:
            return sbx.file_sha256(os.path.realpath(self.binary))
        except OSError:
            return None

    def _run_solver(self, spec_bytes: bytes, *, tail: Dict[str, str]) -> Dict[str, Any]:
        """Run once; returns the parsed result dict or raises SolverRunError(reason). ``tail['sandbox']`` records the mode that was active."""
        o = sbx.run_binary([self.binary, "run"], spec_bytes, limits=self.limits)
        tail["sandbox"] = o.sandbox
        if o.refused or not o.ok:
            raise SolverRunError("sandbox: " + _clean(o.error))
        if o.limit:
            raise SolverRunError(f"solver killed by the {o.limit} limit")
        if o.rc != 0:
            raise SolverRunError(f"solver exited with code {o.rc}" + (": " + _clean(o.stderr_tail) if o.stderr_tail.strip() else ""))
        try:
            res = json.loads(o.stdout.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError):
            raise SolverRunError("solver output is not valid JSON") from None
        if not isinstance(res, dict) or res.get("schema") != RESULT_SCHEMA:
            raise SolverRunError(f"solver output is not a {RESULT_SCHEMA}")
        return res

    def _unverified(self, spec_id: str, input_sha: str, reasons: List[str], result: Optional[Dict[str, Any]], bin_sha: Optional[str]) -> Dict[str, Any]:
        return {"schema": CERT_SCHEMA, "id": spec_id, "verdict": "UNVERIFIED", "checks": [], "reasons": [_clean(r) for r in reasons], "input_sha256": input_sha,
                "result_sha256": canonical_sha256(result) if result is not None else None, "binary_sha256": bin_sha,
                "solver_rev": ((result or {}).get("solver") or {}).get("rev") if isinstance(result, dict) else None, "verifier_version": None, "caveat": CAVEAT}

    def solve(self, spec_bytes: bytes) -> Tuple[int, Dict[str, Any]]:
        """(HTTP status, report). Report: {verdict, summary, result, certificate, solver_bee}. 400 for malformed specs, 429 when busy, else 200."""
        t0 = time.monotonic()
        try:
            spec = validate_spec_bytes(spec_bytes)
        except SpecError as exc:
            return 400, {"error": {"message": str(exc), "type": "invalid_request_error"}}
        if not self._sem.acquire(blocking=False):
            return 429, {"error": {"message": "busy: all solver slots are in use, retry later", "type": "rate_limit_error"}}
        with self._lock:
            self._in_flight += 1
        try:
            return 200, self._pipeline(spec_bytes, spec, t0)
        finally:
            with self._lock:
                self._in_flight -= 1
            self._sem.release()

    def _pipeline(self, spec_bytes: bytes, spec: Dict[str, Any], t0: float) -> Dict[str, Any]:
        input_sha = hashlib.sha256(spec_bytes).hexdigest()
        bin_sha = self._binary_sha()
        info: Dict[str, str] = {"sandbox": "none"}
        result: Optional[Dict[str, Any]] = None
        cert: Optional[Dict[str, Any]] = None
        try:
            result = self._run_solver(spec_bytes, tail=info)
            if result.get("status") != "ok":
                cert = self._unverified(spec["id"], input_sha, ["solver reported status=%s: %s" % (_clean(result.get("status"), 20), _clean(result.get("error", ""), 120))], result, bin_sha)
            else:
                v, why = self.verifier()
                if v is None:
                    cert = self._unverified(spec["id"], input_sha, [why], result, bin_sha)
                else:
                    def rerun(tight_spec_bytes: bytes) -> Dict[str, Any]:
                        validate_spec_bytes(tight_spec_bytes)  # the verifier's tightened spec goes through the same checks
                        return self._run_solver(tight_spec_bytes, tail=info)

                    cert = v.verify(spec_bytes, result, problem=self._library_problem(spec), rerun=rerun, binary_sha256=bin_sha)
                    cert = self._check_cert(cert, input_sha, result)
        except SolverRunError as exc:
            cert = self._unverified(spec["id"], input_sha, [str(exc)], result, bin_sha)
        except Exception as exc:  # noqa: BLE001  fail closed on anything: verifier bug, rerun failure, bad cert
            cert = self._unverified(spec["id"], input_sha, [f"verification failed with {type(exc).__name__}: {_clean(exc, 120)}"], result, bin_sha)
        wall_ms = int((time.monotonic() - t0) * 1000)
        verdict = cert["verdict"]
        self.log(f"solve verdict={verdict} sandbox={info['sandbox']} wall_ms={wall_ms} spec_bytes={len(spec_bytes)}")
        return {"verdict": verdict, "summary": self._summary(spec["id"], cert, info["sandbox"], wall_ms), "result": result, "certificate": cert,
                "solver_bee": {"verdict": verdict, "input_sha256": input_sha, "result_sha256": canonical_sha256(result) if result is not None else None,
                               "sandbox": info["sandbox"], "wall_ms": wall_ms}}

    def _library_problem(self, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """The library entry for this spec id, but only when the submitted problem/y0/t0/t_out equal the library spec's (otherwise the reference does not apply)."""
        p = self.library.get(spec["id"])
        if p is None:
            return None
        ls = p["spec"]
        same = all(canonical_sha256(spec.get(k, 0.0 if k == "t0" else None)) == canonical_sha256(ls.get(k, 0.0 if k == "t0" else None)) for k in ("problem", "y0", "t0", "t_out"))
        return p if same else None

    @staticmethod
    def _check_cert(cert: Any, input_sha: str, result: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(cert, dict) or cert.get("verdict") not in VERDICTS:
            raise ValueError("verifier returned no valid certificate")
        if cert.get("input_sha256") not in (None, input_sha):
            raise ValueError("certificate input hash does not match the request")
        if cert.get("result_sha256") not in (None, canonical_sha256(result)):
            raise ValueError("certificate result hash does not match the result")
        return cert

    @staticmethod
    def _summary(spec_id: str, cert: Dict[str, Any], sandbox: str, wall_ms: int) -> str:
        checks = ", ".join("%s %s" % (_clean(c.get("name"), 30), _clean(c.get("status"), 10)) for c in (cert.get("checks") or [])[:8] if isinstance(c, dict))
        reasons = "; ".join(_clean(r, 160) for r in (cert.get("reasons") or [])[:4])
        out = f"Solver Bee verdict for spec {spec_id}: {cert['verdict']} (sandbox: {sandbox}, {wall_ms} ms)."
        if checks:
            out += f" Checks: {checks}."
        if reasons:
            out += f" Reasons: {reasons}."
        return out + " " + CAVEAT

    # -- endpoints
    def chat(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        def err(code: int, msg: str) -> Tuple[int, Dict[str, Any]]:
            return code, {"error": {"message": msg, "type": "invalid_request_error"}}

        if not isinstance(body, dict) or not isinstance(body.get("messages"), list) or not body["messages"]:
            return err(400, "body must be a JSON object with a non-empty 'messages' list")
        if body.get("stream"):
            return err(400, "streaming is not supported")
        users = [m for m in body["messages"] if isinstance(m, dict) and m.get("role") == "user"]
        if not users:
            return err(400, "no user message")
        text = _content_text(users[-1].get("content"))
        try:
            spec_bytes = extract_spec_block(text)
        except SpecError as exc:
            return err(400, str(exc))
        code, rep = self.solve(spec_bytes)
        if code != 200:
            return code, rep
        content = rep["summary"] + "\n\nResult:\n```json\n" + json.dumps(rep["result"], ensure_ascii=False) + "\n```\n\nCertificate:\n```hivemind-cert\n" + \
            json.dumps(rep["certificate"], ensure_ascii=False, indent=1) + "\n```\n"
        pt = sum(len(_content_text(m.get("content")).encode("utf-8")) for m in body["messages"] if isinstance(m, dict))
        ct = len(content.encode("utf-8"))
        return 200, {"id": "chatcmpl-solverbee-" + secrets.token_hex(6), "object": "chat.completion", "created": int(time.time()), "model": MODEL_ID,
                     "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct, "unit": "bytes"}, "solver_bee": rep["solver_bee"]}

    def solve_endpoint(self, body: Any) -> Tuple[int, Dict[str, Any]]:
        if not isinstance(body, dict) or not isinstance(body.get("spec"), dict) or set(body) - {"spec"}:
            return 400, {"error": {"message": "body must be {\"spec\": {...}}", "type": "invalid_request_error"}}
        spec_bytes = json.dumps(body["spec"], separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        code, rep = self.solve(spec_bytes)
        if code != 200:
            return code, rep
        return 200, {"verdict": rep["verdict"], "summary": rep["summary"], "result": rep["result"], "certificate": rep["certificate"], "solver_bee": rep["solver_bee"]}


# ---------------------------------------------------------------- HTTP
def make_handler(svc: SolverBeeService) -> type:
    class Handler(BaseHTTPRequestHandler):
        server_version = "SolverBee/1"
        timeout = 30  # slow clients cannot hold a thread forever

        def log_message(self, *a: Any) -> None:  # request lines and bodies are never logged here
            del a

        def _send(self, code: int, obj: Dict[str, Any], extra: Optional[Dict[str, str]] = None) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.send_header("Connection", "close")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(b)

        def _err(self, code: int, msg: str, typ: str = "invalid_request_error") -> None:
            self._send(code, {"error": {"message": msg, "type": typ}})

        def _guard(self) -> bool:
            if svc.loopback_only:  # anti DNS-rebinding and anti drive-by-from-a-web-page
                if not _is_loopback_host(_hostname_of(self.headers.get("Host", ""))):
                    self._err(403, "Host header is not loopback", "forbidden")
                    return False
                origin = self.headers.get("Origin")
                if origin is not None and not _is_loopback_host(_hostname_of(re.sub(r"^[a-z]+://", "", origin))):
                    self._err(403, "cross-origin request refused", "forbidden")
                    return False
            if not svc.authorised(self.headers.get("Authorization", "")):
                self._err(401, "missing or wrong Bearer token", "invalid_api_key")
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if not self._guard():
                return
            if self.path.split("?")[0] == "/health":
                self._send(200, svc.health())
            else:
                self._err(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            if not self._guard():
                return
            path = self.path.split("?")[0]
            if path not in ("/v1/chat/completions", "/v1/solve"):
                return self._err(404, "not found")
            if "chunked" in self.headers.get("Transfer-Encoding", "").lower():
                return self._err(411, "Content-Length required (chunked bodies are not supported)")
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self._err(411, "Content-Length required")
            if n < 0 or n > MAX_BODY_BYTES:
                return self._err(413, f"request body limited to {MAX_BODY_BYTES} bytes")
            try:
                body = json.loads(self.rfile.read(n).decode("utf-8"))
            except (UnicodeDecodeError, ValueError, RecursionError):
                return self._err(400, "invalid JSON")
            try:
                code, obj = svc.chat(body) if path == "/v1/chat/completions" else svc.solve_endpoint(body)
            except Exception as exc:  # noqa: BLE001  never a bare 500 with internals
                svc.log(f"internal error: {type(exc).__name__}")
                code, obj = 200, {"verdict": "UNVERIFIED", "error": {"message": "internal error; treat as UNVERIFIED", "type": "server_error"}}
            self._send(code, obj, {"Retry-After": "1"} if code == 429 else None)

    return Handler


def build_server(svc: SolverBeeService, host: str, port: int) -> ThreadingHTTPServer:
    class Server(ThreadingHTTPServer):
        daemon_threads = True
        address_family = socket.AF_INET6 if ":" in host else socket.AF_INET

    return Server((host, port), make_handler(svc))
