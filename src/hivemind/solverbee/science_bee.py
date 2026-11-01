# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Science Bee: plain-language questions -> data-only solver specs -> solved, verified, explained (stdlib only).

The language model's ONLY role is to PROPOSE a ``hivemind-solver-spec/1`` JSON object. Nothing a user or a model writes is ever evaluated as code. The proposal
goes through a strict validator, then through the solver and the verifier; trust comes from those two. Whether the equations describe the user's real situation
is never checked, and every answer says so.

Layout: backends (LLM access) | validator and JSON extraction | formulation loop | pipeline | explainer | conversation | keyword baseline (experiment S-3 only).
"""

from __future__ import annotations

import hashlib
import importlib
import ipaddress
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import families as fam
from .families import LIMITS_SENTENCE, SPEC_SCHEMA

MAX_SPEC_BYTES = 64 * 1024
MAX_RAW_CHARS = 200_000
MAX_RETRIES = 3
HOUSE_RTOL = 1e-10
HOUSE_ATOL = 1e-12
RAW_KEEP = 600
TOP_KEYS = {"schema", "id", "problem", "y0", "t0", "t_out", "rtol", "atol", "method", "max_steps"}
LIMITS_SENTENCE_FR = ("Les nombres résolvent les équations qui ont été posées ; que ces équations décrivent votre situation réelle n'est pas vérifié.")


class SpecInvalid(ValueError):
    """The proposed spec violates the grammar. The message is fed back to the model verbatim (it never echoes long model text)."""


class BackendError(RuntimeError):
    """The language model could not be reached or answered unusably."""


# ======================================================================== backends
class Backend:
    """LLM access: ``complete(messages) -> str`` where messages are [{"role": "system"|"user"|"assistant", "content": str}]."""

    name = "backend"

    def complete(self, messages: List[Dict[str, str]]) -> str:  # pragma: no cover - interface
        raise BackendError(f"{self.name}: no language model backend configured")


class FakeBackend(Backend):
    """Scripted backend for tests and dry runs: a list of replies (consumed in order) or a callable(messages) -> str."""

    name = "fake"

    def __init__(self, script: Any) -> None:
        self._script = script
        self._i = 0
        self.calls: List[List[Dict[str, str]]] = []

    def complete(self, messages: List[Dict[str, str]]) -> str:
        self.calls.append([dict(m) for m in messages])
        if callable(self._script):
            return str(self._script(messages))
        if self._i >= len(self._script):
            raise BackendError("fake backend script exhausted")
        out = self._script[self._i]
        self._i += 1
        return str(out)


def check_loopback_url(url: str) -> str:
    """Return the normalised base URL, or raise BackendError unless the host is literally localhost, 127.x.x.x or ::1 (no userinfo, no look-alike names)."""
    if not isinstance(url, str) or len(url) > 200:
        raise BackendError("base URL must be a short string")
    try:
        parts = urllib.parse.urlsplit(url.strip())
        host, port = parts.hostname, parts.port
    except ValueError:
        raise BackendError("base URL is not valid") from None
    if parts.scheme not in ("http", "https") or parts.username is not None or parts.password is not None or host is None:
        raise BackendError("base URL must be http(s)://host[:port] without credentials")
    if parts.path not in ("", "/") or parts.query or parts.fragment:
        raise BackendError("base URL must have no path, query or fragment")
    if host == "localhost":
        ip = "127.0.0.1"
    else:
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            raise BackendError("the model backend must be on this machine: host must be localhost or a loopback IP literal") from None
        if not addr.is_loopback:
            raise BackendError("the model backend must be on this machine: host must be a loopback address")
        ip = host
    netloc = f"[{ip}]" if ":" in ip else ip
    if port is not None:
        netloc += f":{port}"
    return f"{parts.scheme}://{netloc}"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    last_refused_url: str = ""
    last_refused_code: int = 0

    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:  # type: ignore[override]
        self.last_refused_url = str(newurl) if newurl is not None else ""
        self.last_refused_code = int(code) if isinstance(code, int) else 0
        return None


class OllamaBackend(Backend):
    """Native /api/chat of a LOCAL Ollama (loopback only, no streaming, no proxy, no redirects). num_gpu=0 means CPU only (the default).

    This class only talks to an already running server; it never starts, stops or configures Ollama.
    """

    name = "ollama"

    def __init__(self, model: str, *, base_url: str = "http://127.0.0.1:11434", num_gpu: int = 0, timeout_s: float = 300.0, temperature: float = 0.0,
                 seed: int = 20261027, num_predict: int = 2048, think: Optional[bool] = None, max_response_bytes: int = 4 * 1024 * 1024) -> None:
        if not isinstance(model, str) or not re.fullmatch(r"[A-Za-z0-9_.:/@-]{1,100}", model):
            raise BackendError("model name has unexpected characters")
        if isinstance(num_gpu, bool) or not isinstance(num_gpu, int) or num_gpu < 0:
            raise BackendError("num_gpu must be an integer >= 0")
        self.base_url = check_loopback_url(base_url)
        self.model, self.num_gpu, self.timeout_s = model, num_gpu, float(timeout_s)
        self.temperature, self.seed, self.num_predict, self.think = float(temperature), int(seed), int(num_predict), think
        self.max_response_bytes = int(max_response_bytes)
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)

    def payload(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                "options": {"num_gpu": self.num_gpu, "temperature": self.temperature, "seed": self.seed, "num_predict": self.num_predict}}
        if self.think is not None:
            body["think"] = bool(self.think)
        return body

    def complete(self, messages: List[Dict[str, str]]) -> str:
        data = json.dumps(self.payload(messages), ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(self.base_url + "/api/chat", data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with self._opener.open(req, timeout=self.timeout_s) as resp:  # noqa: S310 - scheme and host checked in check_loopback_url
                raw = resp.read(self.max_response_bytes + 1)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise BackendError(f"model request failed ({type(exc).__name__})") from None
        if len(raw) > self.max_response_bytes:
            raise BackendError("model response too large")
        try:
            obj = json.loads(raw.decode("utf-8"))
            content = obj["message"]["content"]
        except (ValueError, KeyError, TypeError):
            raise BackendError("model response is not an Ollama chat reply") from None
        if not isinstance(content, str):
            raise BackendError("model reply content is not text")
        return content


# ======================================================================== JSON extraction and strict validator
def _reject_constant(name: str) -> Any:
    raise SpecInvalid(f"non-finite number {name} is not allowed")


def _no_dup_pairs(pairs: List[Tuple[str, Any]]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for k, v in pairs:
        if k in d:
            raise SpecInvalid("duplicate key " + repr(k[:40]))
        d[k] = v
    return d


def _loads(text: str) -> Any:
    try:
        return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_no_dup_pairs)
    except SpecInvalid:
        raise
    except (ValueError, RecursionError):
        raise SpecInvalid("the JSON could not be parsed") from None


def _balanced_objects(text: str) -> List[str]:
    """Top-level balanced {...} slices (string-aware), in order."""
    out: List[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth, j, in_str, esc = 0, i, False, False
        while j < n:
            c = text[j]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
            elif c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0 and j < n:
            out.append(text[i:j + 1])
            i = j + 1
        else:
            i += 1
    return out


def extract_json(text: str) -> Dict[str, Any]:
    """Exactly one JSON object found in the model's reply (prose and code fences around it are ignored). Raises SpecInvalid otherwise."""
    if not isinstance(text, str) or not text.strip():
        raise SpecInvalid("the reply was empty; reply with ONE JSON object")
    if len(text) > MAX_RAW_CHARS:
        raise SpecInvalid("the reply is too long")
    found: List[Dict[str, Any]] = []
    for cand in _balanced_objects(text):
        try:
            obj = _loads(cand)
        except SpecInvalid:
            continue
        if isinstance(obj, dict):
            found.append(obj)
    if not found:
        raise SpecInvalid("no JSON object found in the reply; reply with ONE JSON object")
    if len(found) > 1:
        raise SpecInvalid("the reply contains more than one JSON object; reply with exactly ONE")
    return found[0]


def _is_num(x: Any) -> bool:
    return not isinstance(x, bool) and isinstance(x, (int, float)) and math.isfinite(x) and abs(x) <= 1e150


def _numlist(x: Any, name: str, max_len: int, min_len: int = 1) -> List[float]:
    if not isinstance(x, list) or not min_len <= len(x) <= max_len:
        raise SpecInvalid(f"{name} must be a list of {min_len}..{max_len} numbers")
    if not all(_is_num(v) for v in x):
        raise SpecInvalid(f"{name} must contain only finite numbers")
    return x


def validate_spec(obj: Any) -> Dict[str, Any]:
    """Strict check of a ``hivemind-solver-spec/1`` object. Returns it unchanged; raises SpecInvalid with a short, model-safe message."""
    if not isinstance(obj, dict):
        raise SpecInvalid("the spec must be a JSON object")
    unknown = sorted(set(obj) - TOP_KEYS)
    if unknown:
        raise SpecInvalid("unknown field(s): " + ", ".join(repr(str(k)[:40]) for k in unknown[:5]) + "; allowed: " + ", ".join(sorted(TOP_KEYS)))
    if obj.get("schema") != SPEC_SCHEMA:
        raise SpecInvalid(f'"schema" must be "{SPEC_SCHEMA}"')
    sid = obj.get("id")
    if not isinstance(sid, str) or not fam.ID_RE.match(sid):
        raise SpecInvalid('"id" must match [A-Za-z0-9_.-]{1,64}')
    for k in ("problem", "y0", "t_out"):
        if k not in obj:
            raise SpecInvalid(f'missing field "{k}"')
    y0 = _numlist(obj["y0"], "y0", 20)
    n = len(y0)
    prob = obj["problem"]
    if not isinstance(prob, dict) or prob.get("kind") not in ("named", "linear", "mass_action"):
        raise SpecInvalid('"problem.kind" must be "named", "linear" or "mass_action"')
    kind = prob["kind"]
    if kind == "named":
        if set(prob) - {"kind", "family", "params"}:
            raise SpecInvalid('unknown field in "problem" (named allows kind, family, params)')
        family = prob.get("family")
        if family not in fam.FAMILIES:
            raise SpecInvalid('"problem.family" must be one of ' + ", ".join(fam.list_families()))
        params = prob.get("params", {})
        if not isinstance(params, dict) or len(params) > 32:
            raise SpecInvalid('"problem.params" must be an object')
        try:
            fam.check_params(family, params)
            dim = fam.family_dim(family, params)
        except fam.FamilyError as exc:
            raise SpecInvalid(str(exc)[:200]) from None
        if dim is not None and dim != n:
            raise SpecInvalid(f'family "{family}" needs y0 of length {dim}, got {n}')
        if dim is None:
            raise SpecInvalid(f'family "{family}": params.n must be an integer in 2..20')
    elif kind == "linear":
        if set(prob) - {"kind", "A", "b"}:
            raise SpecInvalid('unknown field in "problem" (linear allows kind, A, b)')
        a, b = prob.get("A"), prob.get("b")
        if not isinstance(a, list) or not 1 <= len(a) <= 20:
            raise SpecInvalid('"A" must be an n x n matrix with 1 <= n <= 20')
        for row in a:
            _numlist(row, "A row", 20)
            if len(row) != len(a):
                raise SpecInvalid('"A" must be square')
        _numlist(b, "b", 20)
        if len(b) != len(a) or len(a) != n:
            raise SpecInvalid('"A", "b" and "y0" must have the same dimension')
    else:
        if set(prob) - {"kind", "species", "reactions"}:
            raise SpecInvalid('unknown field in "problem" (mass_action allows kind, species, reactions)')
        sp, rx = prob.get("species"), prob.get("reactions")
        if not isinstance(sp, list) or not 1 <= len(sp) <= 20 or not all(isinstance(s, str) and 1 <= len(s) <= 64 for s in sp) or len(set(sp)) != len(sp):
            raise SpecInvalid('"species" must be 1..20 distinct strings of at most 64 characters')
        if not isinstance(rx, list) or not 1 <= len(rx) <= 50:
            raise SpecInvalid('"reactions" must be a list of 1..50 reactions')
        for r in rx:
            if not isinstance(r, dict) or set(r) - {"reactants", "products", "k"} or not _is_num(r.get("k")) or r["k"] < 0:
                raise SpecInvalid('each reaction needs "reactants", "products" and a finite "k" >= 0, nothing else')
            for side in ("reactants", "products"):
                m = r.get(side)
                if not isinstance(m, dict) or len(m) > 20:
                    raise SpecInvalid(f'reaction "{side}" must be an object {{species: count}}')
                for s, c in m.items():
                    if s not in sp or isinstance(c, bool) or not isinstance(c, int) or not 1 <= c <= 10:
                        raise SpecInvalid("stoichiometric coefficients must be integers 1..10 over declared species")
        if len(sp) != n:
            raise SpecInvalid('"y0" must have one entry per species')
    if "t0" in obj and not _is_num(obj["t0"]):
        raise SpecInvalid('"t0" must be a finite number')
    t0 = obj.get("t0", 0.0)
    t_out = _numlist(obj["t_out"], "t_out", 1000)
    prev = t0
    for t in t_out:
        if not t > prev:
            raise SpecInvalid('"t_out" must be strictly increasing and every value greater than "t0"')
        prev = t
    if "rtol" in obj and not (_is_num(obj["rtol"]) and 0 < obj["rtol"] <= 1):
        raise SpecInvalid('"rtol" must be a number in (0, 1]')
    if "atol" in obj:
        at = obj["atol"]
        if isinstance(at, list):
            _numlist(at, "atol", 20)
            if len(at) != n or not all(v > 0 for v in at):
                raise SpecInvalid('"atol" array must have one positive entry per state variable')
        elif not (_is_num(at) and at > 0):
            raise SpecInvalid('"atol" must be a positive number')
    if "method" in obj and obj["method"] not in ("bdf", "adams"):
        raise SpecInvalid('"method" must be "bdf" or "adams"')
    if "max_steps" in obj:
        ms = obj["max_steps"]
        if isinstance(ms, bool) or not isinstance(ms, int) or not 1 <= ms <= 1_000_000:
            raise SpecInvalid('"max_steps" must be an integer in 1..1000000')
    if len(canonical_bytes(obj)) > MAX_SPEC_BYTES:
        raise SpecInvalid("the spec is larger than 64 KiB")
    return obj


def canonical_bytes(spec: Dict[str, Any]) -> bytes:
    """The exact bytes sent to the solver (compact JSON, key order as proposed)."""
    return json.dumps(spec, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def with_house_tolerances(spec: Dict[str, Any]) -> Dict[str, Any]:
    """A copy of the validated spec with the Science Bee house tolerances added when the proposal omitted them (tighter than the CLI defaults 1e-8 / 1e-10)."""
    out = dict(spec)
    out.setdefault("rtol", HOUSE_RTOL)
    out.setdefault("atol", HOUSE_ATOL)
    return out


def spec_sha256(spec: Dict[str, Any]) -> str:
    return hashlib.sha256(canonical_bytes(spec)).hexdigest()


def is_slot_proposal(obj: Any) -> bool:
    """True when the model proposed a `hivemind-solver-slots/1` slot object rather than a raw spec."""
    return isinstance(obj, dict) and (
        obj.get("schema") == fam.SLOT_SCHEMA
        or ("slot_family" in obj and "problem" not in obj)
    )


def compile_or_validate_proposal(obj: Any) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Validate a raw `hivemind-solver-spec/1` or compile+validate a `hivemind-solver-slots/1` proposal."""
    if is_slot_proposal(obj):
        try:
            compiled, trace = fam.compile_slot_spec(obj, return_trace=True)
        except fam.FamilyError as exc:
            raise SpecInvalid(str(exc)[:200]) from None
        return validate_spec(compiled), trace
    return validate_spec(obj), None



# ======================================================================== prompts
SYSTEM_PROMPT_HEAD = """You turn a science question (English or French) into ONE data-only specification for a numeric ODE solver.
You only PROPOSE the specification; a solver and an independent verifier decide whether the numbers hold. You never compute the answer yourself.

Reply with exactly ONE JSON object and nothing else (no code, no formulas as strings, no comments). Nothing you write is evaluated as code.
If the question cannot be expressed in the grammar below, reply with {"cannot_formulate": "<short reason>"} instead. Never guess missing numbers.

GRAMMAR (unknown fields are rejected; all numbers finite):
{"schema":"hivemind-solver-spec/1","id":"<[A-Za-z0-9_.-]{1,64}>","problem":<PROBLEM>,"y0":[<n numbers, n<=20>],"t0":0,"t_out":[<strictly increasing numbers > t0, at most 1000>]}
optional extra keys: "rtol", "atol" (if you omit them the Science Bee adds rtol 1e-10 and atol 1e-12; do not loosen them), "method" ("bdf" or "adams"), "max_steps" (1..1000000).
<PROBLEM> is ONE of:
 1. {"kind":"named","family":"<family>","params":{<only the parameters listed for that family>}}   (omitted params take the stated defaults)
 2. {"kind":"linear","A":[[n x n numbers]],"b":[n numbers]}           meaning x' = A x + b
 3. {"kind":"mass_action","species":["S1",...],"reactions":[{"reactants":{"S1":1},"products":{"S2":1},"k":0.5}]}   (stoichiometry = integers 1..10, k >= 0)

NAMED FAMILIES (the state vector y0 follows the order shown):
"""

SYSTEM_PROMPT_TAIL = """
PRESETS you must encode with the generic kinds (state order matters, y0 follows it):
{presets}
 - SIR (fractions summing to 1): species ["S","I","R"], reactions S+I->2I as {{"reactants":{{"S":1,"I":1}},"products":{{"I":2}},"k":beta}} and I->R as {{"reactants":{{"I":1}},"products":{{"R":1}},"k":gamma}}.
 - SEIR: species ["S","E","I","R"], S+I->E+I (k=beta), E->I (k=sigma), I->R (k=gamma).
 - series RLC with constant voltage V: kind linear, state [q, i], A=[[0,1],[-1/(L*C),-R/L]], b=[0,V/L].
 - spring-mass-damper with constant force F: kind linear, state [x, v], A=[[0,1],[-k/m,-c/m]], b=[0,F/m].
 - decay chain A->B->C: kind mass_action, species ["A","B","C"], one reaction per step with its rate.

UNITS: the solver is unit-free. Use SI units (seconds, metres, kilograms, radians) unless the question uses one consistent unit system throughout. If the question
gives other units (minutes, hours, km, degrees, litres...) CONVERT them yourself so that every number in the spec is mutually consistent, and put t_out in the time
unit you chose. Derived parameters (for example g_over_l = g / length, or (2*pi/period)^2) must be computed by you as plain numbers, never as expressions.
Always list ALL requested output times in t_out, in increasing order.
"""

SLOT_SYSTEM_PROMPT_TAIL = """
S-3b SLOT-FILLING SCHEMA (PREFERRED FOR SMALL MODELS <= 4B):
Instead of computing unit conversions, algebraic ratios, or A/b matrices yourself, you may propose a `hivemind-solver-slots/1` object and let deterministic code compile the solver spec:
{"schema":"hivemind-solver-slots/1","id":"<optional id>","slot_family":"<slot_family>","slots":{<named parameters and t_out / t_out_ms / t_out_min / t_out_h / t_out_days>}}
Allowed slot_family values:
 - sir (R0 or beta, infectious_period or gamma, i0 or i0_percent, s0)
 - seir (R0 or beta, incubation_period or sigma, infectious_period or gamma, e0/e0_percent, i0/i0_percent)
 - series_rlc (R, L or L_mH, C or C_mF or C_uF, V, q0, i0)
 - spring_mass_damper (m or m_g, k, c or damper_force+damper_speed, F, x0 or x0_cm, v0)
 - pendulum (theta0 or theta0_deg, omega0, g_over_l or length_m/length_cm/period_s/period_ms, damping)
 - projectile_drag (x0, y0, vx0+vy0 or speed/speed_kmh + angle_deg or slope_rise+slope_run, g, k)
 - kepler (mu, circular_radius or x0+y0+vx0+vy0)
 - logistic (r or r_percent_per_min, K, n0)
 - decay_chain (a0, rates or rates_per_min or time_constants_s)
 - yukawa_crystal / qf_yukawa (kappa, damping, x0, v0)
 - gpe_vortex / qf_pgpe (g_int, omega_trap, zeta, x0, v0)
 - art_exo_genetics / chromatin_remodeling (k_pol, k_exo, k_ext, T0, M0, P0)
 - bao_cosmology / k3_quintessence (hubble, m_eff, phi0, dphi0, source)
"""


def build_system_prompt(*, prefer_slots: bool = False) -> str:
    """The allowed grammar and families, generated from families.py."""
    base = SYSTEM_PROMPT_HEAD + fam.catalogue_text() + "\n" + SYSTEM_PROMPT_TAIL.format(presets=fam.presets_text())
    if prefer_slots:
        return base + SLOT_SYSTEM_PROMPT_TAIL
    return base


def build_first_messages(question: str, *, prefer_slots: bool = False) -> List[Dict[str, str]]:
    return [{"role": "system", "content": build_system_prompt(prefer_slots=prefer_slots)},
            {"role": "user", "content": "Question (between the markers):\n<<<\n" + question.strip() + "\n>>>\nReply with ONE JSON object."}]


def build_followup_messages(question: str, previous_spec: Dict[str, Any], previous_question: Optional[str], *, prefer_slots: bool = False) -> List[Dict[str, str]]:
    prev_q = (previous_question or "").strip()
    return [{"role": "system", "content": build_system_prompt(prefer_slots=prefer_slots)},
            {"role": "user", "content": ("The previous question was:\n<<<\n" + prev_q + "\n>>>\nIts VALIDATED specification was:\n" + json.dumps(previous_spec, separators=(",", ":"), ensure_ascii=False)
                                         + "\n\nThe user now asks a follow-up (between the markers):\n<<<\n" + question.strip() + "\n>>>\n"
                                         "Return the FULL new specification (not a patch): copy the previous one and change only what the follow-up changes. Reply with ONE JSON object.")}]


# ======================================================================== formulation loop
def formulate(messages: List[Dict[str, str]], backend: Backend, *, max_retries: int = MAX_RETRIES) -> Dict[str, Any]:
    """Ask the model for a spec; validate; on failure feed the exact validation error back, at most ``max_retries`` times (so at most 1 + max_retries calls).

    Returns {ok, spec, spec_bytes, attempts, retries, errors, raw_outputs, reason}. Never guesses: after the last failure ok is False.
    """
    msgs = [dict(m) for m in messages]
    errors: List[str] = []
    raws: List[str] = []
    for attempt in range(1, max_retries + 2):
        try:
            raw = backend.complete(msgs)
        except BackendError as exc:
            errors.append(f"backend error: {exc}")
            return _fail(errors, raws, attempt, "the language model could not be reached")
        raws.append(raw[:RAW_KEEP] if isinstance(raw, str) else "")
        try:
            obj = extract_json(raw)
            if set(obj) == {"cannot_formulate"}:
                why = obj["cannot_formulate"]
                reason = why[:300] if isinstance(why, str) else "no reason given"
                errors.append("model declined: " + reason)
                return _fail(errors, raws, attempt, reason)
            validated, slot_trace = compile_or_validate_proposal(obj)
            spec = with_house_tolerances(validated)
            return {"ok": True, "spec": spec, "spec_bytes": canonical_bytes(spec), "slot_trace": slot_trace, "attempts": attempt,
                    "retries": attempt - 1, "errors": errors, "raw_outputs": raws, "reason": ""}
        except SpecInvalid as exc:
            msg = str(exc)[:300]
            errors.append(msg)
            msgs.append({"role": "assistant", "content": raw[:2000] if isinstance(raw, str) else ""})
            msgs.append({"role": "user", "content": "Your reply was rejected by the validator: " + msg + "\nReply again with ONE corrected JSON object only."})
    return _fail(errors, raws, max_retries + 1, "the model's proposals did not pass validation: " + (errors[-1] if errors else ""))


def _fail(errors: List[str], raws: List[str], attempts: int, reason: str) -> Dict[str, Any]:
    return {"ok": False, "spec": None, "spec_bytes": None, "slot_trace": None, "attempts": attempts, "retries": max(0, attempts - 1),
            "errors": errors, "raw_outputs": raws, "reason": reason}


# ======================================================================== pipeline (sandbox -> solver -> verifier)
class SolverPipeline:
    """spec bytes -> sandboxed solver -> verifier -> certificate. sandbox, verifier and service helpers are imported lazily; any failure gives UNVERIFIED."""

    def __init__(self, binary: str, *, problems_dir: Optional[str] = None, timeout_s: float = 30.0, mem_mb: int = 2048, allow_unsandboxed: bool = False) -> None:
        self.binary, self.problems_dir = binary, problems_dir
        self.timeout_s, self.mem_mb, self.allow_unsandboxed = float(timeout_s), int(mem_mb), bool(allow_unsandboxed)
        self._library: Optional[Dict[str, Dict[str, Any]]] = None
        self._bin_sha: Optional[str] = None

    @staticmethod
    def _mod(name: str) -> Any:
        return importlib.import_module("hivemind.solverbee." + name)

    def _unverified(self, spec_id: str, reasons: List[str], result: Optional[Dict[str, Any]], t0: float) -> Dict[str, Any]:
        cert = {"schema": "hivemind-solver-cert/1", "id": spec_id, "verdict": "UNVERIFIED", "checks": [], "reasons": [r[:300] for r in reasons]}
        return {"verdict": "UNVERIFIED", "result": result, "certificate": cert, "wall_ms": int((time.monotonic() - t0) * 1000)}

    def _library_problem(self, spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.problems_dir:
            return None
        svc = self._mod("service")
        if self._library is None:
            self._library = svc.load_problem_library(self.problems_dir)
        p = self._library.get(spec.get("id"))
        if p is None:
            return None
        ls = p["spec"]
        same = all(svc.canonical_sha256(spec.get(k, 0.0 if k == "t0" else None)) == svc.canonical_sha256(ls.get(k, 0.0 if k == "t0" else None)) for k in ("problem", "y0", "t0", "t_out"))
        return p if same else None

    def run(self, spec_bytes: bytes) -> Dict[str, Any]:
        t0 = time.monotonic()
        spec_id = "unknown"
        result: Optional[Dict[str, Any]] = None
        try:
            spec = json.loads(spec_bytes.decode("utf-8"))
            spec_id = str(spec.get("id", "unknown"))
            sbx, ver = self._mod("sandbox"), self._mod("verifier")
            if self._bin_sha is None:
                self._bin_sha = sbx.file_sha256(self.binary)
            limits = sbx.Limits(timeout_s=self.timeout_s, mem_mb=self.mem_mb, allow_unsandboxed=self.allow_unsandboxed)

            def solve(raw: bytes) -> Dict[str, Any]:
                out = sbx.run_binary([self.binary, "run"], raw, limits=limits, expected_sha256=self._bin_sha)
                if not out.ok or out.limit or out.rc != 0:
                    raise RuntimeError("sandbox/solver failure: " + str(out.error or out.stderr_tail or out.limit or out.rc)[:200])
                return json.loads(out.stdout.decode("utf-8"))

            result = solve(spec_bytes)
            if not isinstance(result, dict) or result.get("status") != "ok":
                why = str((result or {}).get("error", "unknown"))[:200] if isinstance(result, dict) else "bad result"
                return self._unverified(spec_id, ["solver reported an error: " + why], result if isinstance(result, dict) else None, t0)
            cert = ver.verify(spec_bytes, result, problem=self._library_problem(spec), rerun=solve, binary_sha256=self._bin_sha)
            if not isinstance(cert, dict) or cert.get("verdict") not in ("VERIFIED", "AUDITED", "FAILED", "UNVERIFIED"):
                raise ValueError("verifier returned no valid certificate")
            if cert.get("input_sha256") not in (None, hashlib.sha256(spec_bytes).hexdigest()):
                raise ValueError("certificate input hash does not match the request")
            return {"verdict": cert["verdict"], "result": result, "certificate": cert, "wall_ms": int((time.monotonic() - t0) * 1000)}
        except Exception as exc:  # noqa: BLE001 - fail closed on anything
            return self._unverified(spec_id, [f"pipeline failed closed ({type(exc).__name__}): {str(exc)[:160]}"], result, t0)


# ======================================================================== explainer (deterministic, no model)
_FR_WORDS = re.compile(r"\b(quel|quelle|quels|quelles|donne|donnez|calcule|combien|avec|une?|des|les|le|la|du|de|est|sont|si|et|ou|apr[eè]s|secondes?|ressort|"
                       r"pendule|épidémie|epidemie|vitesse|masse|taux|population|à|où)\b", re.IGNORECASE)
_EN_WORDS = re.compile(r"\b(what|which|give|compute|how|with|the|is|are|if|and|or|after|seconds?|spring|pendulum|epidemic|speed|mass|rate)\b", re.IGNORECASE)


def detect_language(text: str) -> str:
    fr, en = len(_FR_WORDS.findall(text)), len(_EN_WORDS.findall(text))
    return "fr" if fr > en else "en"


_UNIT_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "time": [(r"\b(?:ms|milliseconds?|millisecondes?)\b", "milliseconds"), (r"\b(?:seconds?|secondes?|sec|s)\b", "seconds"), (r"\b(?:minutes?|min)\b", "minutes"),
             (r"\b(?:hours?|heures?|h)\b", "hours"), (r"\b(?:days?|jours?|j|d)\b", "days"), (r"\b(?:years?|ans?|années?)\b", "years")],
    "length": [(r"\b(?:km|kilomet(?:er|re)s?)\b", "kilometres"), (r"\b(?:cm|centimet(?:er|re)s?)\b", "centimetres"), (r"\b(?:m|met(?:er|re)s?|mètres?)\b", "metres")],
    "angle": [(r"\b(?:rad|radians?)\b", "radians"), (r"\b(?:degrees?|degrés?|deg)\b|°", "degrees")],
}
_UNIT_AFTER_NUMBER = r"(?<![\w.])[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?\s*"


def units_mentioned(question: str) -> Dict[str, List[str]]:
    """Units the question itself uses, found right after a number (never converted)."""
    found: Dict[str, List[str]] = {}
    for dim, pats in _UNIT_PATTERNS.items():
        for pat, label in pats:
            if re.search(_UNIT_AFTER_NUMBER + "(?:" + pat + ")", question, re.IGNORECASE) and label not in found.setdefault(dim, []):
                found[dim].append(label)
    return {k: v for k, v in found.items() if v}


def _g(x: Any) -> str:
    return "%.6g" % x if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x)


def _describe_setup(spec: Dict[str, Any], lang: str) -> List[str]:
    prob, y0 = spec["problem"], spec["y0"]
    lines: List[str] = []
    fr = lang == "fr"
    if prob["kind"] == "named":
        e = fam.FAMILIES[prob["family"]]
        used = fam.check_params(prob["family"], prob.get("params", {}))
        pw = []
        for k, (d, meaning) in e["params"].items():
            if k in used:
                pw.append(f"{k} = {_g(used[k])} ({meaning})")
            else:
                pw.append(f"{k} = {_g(d)} ({meaning}; " + ("valeur par défaut" if fr else "default value") + ")")
        lines.append(("Problème résolu : " if fr else "What was solved: ") + e["summary"] + ".")
        lines.append(("Paramètres : " if fr else "Parameters: ") + "; ".join(pw) + ".")
        names = [s["name"] for s in e["state"]] or [f"y{i + 1}" for i in range(len(y0))]
        meanings = [f"{s['name']} = {s['meaning']} [{s['unit']}]" for s in e["state"]]
        lines.append(("État initial : " if fr else "Initial state: ") + ", ".join(f"{nm} = {_g(v)}" for nm, v in zip(names, y0)) + ".")
        if meanings:
            lines.append(("Variables : " if fr else "Variables: ") + "; ".join(meanings) + ".")
    elif prob["kind"] == "mass_action":
        sp = prob["species"]
        rx = []
        for r in prob["reactions"]:
            lhs = " + ".join((f"{c} " if c != 1 else "") + s for s, c in r["reactants"].items()) or "nothing"
            rhs = " + ".join((f"{c} " if c != 1 else "") + s for s, c in r["products"].items()) or "nothing"
            rx.append(f"{lhs} -> {rhs} (k = {_g(r['k'])})")
        lines.append(("Problème résolu : réseau de réactions à action de masse. " if fr else "What was solved: a mass-action reaction network. ")
                     + ("Espèces : " if fr else "Species: ") + ", ".join(sp) + ". " + ("Réactions : " if fr else "Reactions: ") + "; ".join(rx) + ".")
        lines.append(("État initial : " if fr else "Initial state: ") + ", ".join(f"{s} = {_g(v)}" for s, v in zip(sp, y0)) + ".")
    else:
        rows = "; ".join("[" + ", ".join(_g(v) for v in row) + "]" for row in prob["A"])
        lines.append(("Problème résolu : système linéaire x' = A x + b. " if fr else "What was solved: a linear system x' = A x + b. ")
                     + f"A = [{rows}], b = [" + ", ".join(_g(v) for v in prob["b"]) + "].")
        lines.append(("État initial : " if fr else "Initial state: ") + ", ".join(f"x{i + 1} = {_g(v)}" for i, v in enumerate(y0)) + ".")
    tol = f"rtol = {_g(spec.get('rtol', fam.DEFAULT_RTOL))}, atol = {_g(spec.get('atol', fam.DEFAULT_ATOL))}"
    lines.append(("Tolérances du solveur : " if fr else "Solver tolerances: ") + tol + ".")
    return lines


def _state_names(spec: Dict[str, Any]) -> List[str]:
    prob = spec["problem"]
    if prob["kind"] == "named":
        names = [s["name"] for s in fam.FAMILIES[prob["family"]]["state"]]
        return names or [f"y{i + 1}" for i in range(len(spec["y0"]))]
    if prob["kind"] == "mass_action":
        return list(prob["species"])
    return [f"x{i + 1}" for i in range(len(spec["y0"]))]


def echo_interpreted_problem(
    spec: Dict[str, Any],
    *,
    language: str = "en",
    slot_trace: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Echo the interpreted problem back to the user for confirmation before relying on numbers."""
    lang = "fr" if language == "fr" else "en"
    names = _state_names(spec)
    setup_lines = _describe_setup(spec, lang)
    conversions = list((slot_trace or {}).get("conversions", []))
    derived = list((slot_trace or {}).get("derived_parameters", []))
    prompt = (
        "Veuillez confirmer que ce problème interprété (équations, conversions d'unités, état initial et instants de sortie) correspond bien à votre situation réelle."
        if lang == "fr"
        else "Please confirm that this interpreted problem (equations, unit conversions, initial state, and output times) matches your intended setup."
    )
    return {
        "schema": "hivemind-problem-echo/1",
        "id": str(spec.get("id", "science-bee")),
        "language": lang,
        "setup_lines": setup_lines,
        "state_variables": names,
        "initial_state": {nm: float(v) for nm, v in zip(names, spec["y0"])},
        "t0": float(spec.get("t0", 0.0)),
        "t_out": [float(t) for t in spec["t_out"]],
        "conversions_applied": conversions,
        "derived_parameters": derived,
        "confirmation_prompt": prompt,
        "limits_notice": LIMITS_SENTENCE_FR if lang == "fr" else LIMITS_SENTENCE,
    }


_CHECK_WORDS = {
    "shape": ("the result has the right shape", "le résultat a la bonne forme"),
    "hashes": ("the input and result fingerprints match", "les empreintes de l'entrée et du résultat concordent"),
    "reference": ("agreement with a stored independent reference solution", "accord avec une solution de référence indépendante enregistrée"),
    "invariants": ("conserved quantities stay conserved", "les quantités conservées restent conservées"),
    "tightening": ("re-solving with 100 times tighter tolerances gives the same answer", "résoudre avec des tolérances 100 fois plus strictes donne la même réponse"),
}
_VERDICT_WORDS = {
    "VERIFIED": ("VERIFIED: this exact problem is in the reference library and every listed check passed. VERIFIED never means that the answer is correct for your real situation.",
                 "VERIFIED : ce problème exact figure dans la bibliothèque de référence et tous les contrôles listés ont réussi. VERIFIED ne signifie jamais que la réponse est juste pour votre situation réelle."),
    "AUDITED": ("AUDITED: no stored reference exists for a free-form problem, so the answer was audited instead: the checks listed above passed, but nothing compared it with a known solution.",
                "AUDITED : il n'existe pas de référence enregistrée pour un problème libre ; la réponse a été auditée : les contrôles listés ci-dessus ont réussi, mais rien ne l'a comparée à une solution connue."),
    "FAILED": ("FAILED: at least one check failed. Do not use these numbers.", "FAILED : au moins un contrôle a échoué. N'utilisez pas ces nombres."),
    "UNVERIFIED": ("UNVERIFIED: the result could not be obtained or checked, so no number is offered as an answer.",
                   "UNVERIFIED : le résultat n'a pas pu être obtenu ou contrôlé, donc aucun nombre n'est proposé comme réponse."),
}


def explain(question: str, spec: Optional[Dict[str, Any]], report: Optional[Dict[str, Any]], *, language: Optional[str] = None, failure_reason: str = "") -> str:
    """Deterministic plain-language answer (no model). Always ends with the limits sentence (English; French translation added for French questions)."""
    lang = language or detect_language(question)
    fr = lang == "fr"
    out: List[str] = []
    if spec is None or report is None:
        out.append(("Je n'ai pas pu transformer votre question en un problème bien défini, donc je ne donne aucun nombre. " if fr
                    else "I could not turn your question into a well-defined problem, so I give no numbers. ") + (f"({failure_reason[:200]})" if failure_reason else ""))
    else:
        out.extend(_describe_setup(spec, lang))
        verdict = report.get("verdict", "UNVERIFIED")
        result = report.get("result") or {}
        cert = report.get("certificate") or {}
        if verdict in ("VERIFIED", "AUDITED") and isinstance(result.get("y"), list) and result.get("t"):
            names = _state_names(spec)
            ts, ys = result["t"], result["y"]
            idx = list(range(len(ts))) if len(ts) <= 8 else [0, 1, 2, 3, len(ts) - 4, len(ts) - 3, len(ts) - 2, len(ts) - 1]
            out.append("Résultats aux instants demandés :" if fr else "Key numbers at the requested times:")
            for i in idx:
                out.append("  t = %s : " % _g(ts[i]) + ", ".join(f"{nm} = {_g(v)}" for nm, v in zip(names, ys[i])))
            if len(ts) > 8:
                out.append(f"  (… {len(ts) - 8} " + ("instants intermédiaires omis)" if fr else "intermediate times omitted)"))
        um = units_mentioned(question)
        if fr:
            tr = {"time": "temps", "length": "longueur", "angle": "angle", "milliseconds": "millisecondes", "seconds": "secondes", "minutes": "minutes", "hours": "heures",
                  "days": "jours", "years": "années", "kilometres": "kilomètres", "centimetres": "centimètres", "metres": "mètres", "radians": "radians", "degrees": "degrés"}
            said = "; ".join(f"{tr[d]} : {', '.join(tr[u] for u in v)}" for d, v in um.items())
        else:
            said = "; ".join(f"{d}: {', '.join(v)}" for d, v in um.items())
        if fr:
            out.append("Unités : le solveur travaille sur des nombres sans unité ; les résultats sont dans les unités des nombres de l'énoncé ci-dessus. "
                       + (f"Votre question mentionne ({said}). " if said else "Votre question ne cite aucune unité reconnue. ")
                       + "Je n'ai converti aucune unité en silence ; si la proposition a converti des unités, le résultat de cette conversion est visible dans les paramètres ci-dessus : vérifiez-le. "
                       "J'ai supposé que tous les nombres de l'énoncé sont exprimés dans des unités cohérentes entre elles.")
        else:
            out.append("Units: the solver works on plain numbers, so results are in the units of the numbers in the setup above. "
                       + (f"Your question mentions ({said}). " if said else "Your question mentions no recognised unit. ")
                       + "I did not convert any unit silently; if the proposal converted units, the converted numbers are visible in the parameters above: check them. "
                       "I assumed that all numbers in the setup are in units consistent with each other.")
        checks = [c for c in (cert.get("checks") or []) if isinstance(c, dict)]
        if checks:
            parts = []
            for c in checks:
                w = _CHECK_WORDS.get(str(c.get("name")), (str(c.get("name"))[:40],) * 2)[1 if fr else 0]
                st = {"pass": "réussi" if fr else "passed", "fail": "ÉCHEC" if fr else "FAILED", "skip": "non applicable" if fr else "not applicable"}.get(str(c.get("status")), "?")
                parts.append(f"{w}: {st}")
            out.append(("Contrôles : " if fr else "Checks: ") + "; ".join(parts) + ".")
        out.append(_VERDICT_WORDS.get(verdict, _VERDICT_WORDS["UNVERIFIED"])[1 if fr else 0])
        reasons = [str(r)[:160] for r in (cert.get("reasons") or [])[:3]]
        if reasons and verdict in ("FAILED", "UNVERIFIED"):
            out.append(("Raisons : " if fr else "Reasons: ") + "; ".join(reasons))
    out.append(LIMITS_SENTENCE)
    if fr:
        out.append(LIMITS_SENTENCE_FR)
    return "\n".join(out)


# ======================================================================== conversation
_FOLLOWUP_RE = re.compile(r"^\s*(and if|and with|and what if|what if|what about|same (?:but|with)|now with|and when|et si|et avec|et quand|que se passe-t-il si|même chose (?:mais|avec)|"
                          r"maintenant avec|avec maintenant)\b", re.IGNORECASE)


class ScienceBee:
    """One conversation: remembers the last VALIDATED spec so that a follow-up modifies it through the same validation loop."""

    def __init__(self, backend: Backend, pipeline: Any, *, max_retries: int = MAX_RETRIES, prefer_slots: bool = False) -> None:
        self.backend, self.pipeline, self.max_retries = backend, pipeline, int(max_retries)
        self.prefer_slots = bool(prefer_slots)
        self.last_spec: Optional[Dict[str, Any]] = None
        self.last_question: Optional[str] = None

    def reset(self) -> None:
        self.last_spec = None
        self.last_question = None

    def ask(self, question: str, *, follow_up: Optional[bool] = None, language: Optional[str] = None) -> Dict[str, Any]:
        t0 = time.monotonic()
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("question must be a non-empty string of at most 4000 characters")
        is_follow = (bool(_FOLLOWUP_RE.match(question)) if follow_up is None else bool(follow_up)) and self.last_spec is not None
        msgs = (
            build_followup_messages(question, self.last_spec, self.last_question, prefer_slots=self.prefer_slots)
            if is_follow
            else build_first_messages(question, prefer_slots=self.prefer_slots)
        )
        f = formulate(msgs, self.backend, max_retries=self.max_retries)
        lang = language or detect_language(question)
        ans: Dict[str, Any] = {"question": question, "language": lang, "follow_up": is_follow, "formulated": f["ok"], "spec": f["spec"],
                               "slot_trace": f.get("slot_trace"), "interpreted_problem_echo": None, "attempts": f["attempts"],
                               "retries": f["retries"], "errors": f["errors"], "raw_outputs": f["raw_outputs"], "failure_reason": f["reason"], "report": None,
                               "verdict": None, "spec_sha256": None}
        if f["ok"]:
            self.last_spec, self.last_question = f["spec"], question
            ans["spec_sha256"] = hashlib.sha256(f["spec_bytes"]).hexdigest()
            ans["interpreted_problem_echo"] = echo_interpreted_problem(f["spec"], language=lang, slot_trace=f.get("slot_trace"))
            ans["report"] = self.pipeline.run(f["spec_bytes"])
            ans["verdict"] = ans["report"].get("verdict")
        ans["text"] = explain(question, f["spec"], ans["report"], language=lang, failure_reason=f["reason"])
        ans["wall_seconds"] = round(time.monotonic() - t0, 3)
        return ans


# ======================================================================== exact spec match (secondary metric of S-3)
def _canon_num(x: float) -> float:
    return float(x)


def _canon_problem(spec: Dict[str, Any]) -> Tuple[Any, ...]:
    prob = spec["problem"]
    if prob["kind"] == "named":
        e = fam.FAMILIES[prob["family"]]
        used = fam.check_params(prob["family"], prob.get("params", {}))
        full = {k: float(used.get(k, d)) for k, (d, _m) in e["params"].items()}
        return ("named", prob["family"], tuple(sorted(full.items())))
    if prob["kind"] == "linear":
        return ("linear", tuple(tuple(float(v) for v in row) for row in prob["A"]), tuple(float(v) for v in prob["b"]))
    rx = sorted((tuple(sorted(r["reactants"].items())), tuple(sorted(r["products"].items())), float(r["k"])) for r in prob["reactions"])
    return ("mass_action", tuple(prob["species"]), tuple(rx))


def _close(a: Any, b: Any, rel: float) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= rel * max(abs(a), abs(b)) + 1e-300
    if isinstance(a, tuple) and isinstance(b, tuple):
        return len(a) == len(b) and all(_close(x, y, rel) for x, y in zip(a, b))
    return bool(a == b)


def specs_match(a: Dict[str, Any], b: Dict[str, Any], rel: float = 1e-6) -> bool:
    """Exact spec match as pre-registered: same problem (defaults filled in), y0, t0 (default 0) and t_out within relative 1e-6; id, tolerances and method ignored."""
    try:
        if not _close(_canon_problem(a), _canon_problem(b), rel):
            return False
        pairs = [(a["y0"], b["y0"]), (a["t_out"], b["t_out"]), ([a.get("t0", 0.0)], [b.get("t0", 0.0)])]
        return all(len(x) == len(y) and all(abs(float(u) - float(v)) <= rel * max(abs(float(u)), abs(float(v))) + 1e-12 for u, v in zip(x, y)) for x, y in pairs)
    except (KeyError, TypeError, ValueError, fam.FamilyError):
        return False


# ======================================================================== keyword/template baseline (comparison arm of S-3 only)
_NUM = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SEP = r"\s*(?:=|:|of|is|at|equals|de|à|a|vaut|égal à|egal a)?\s*"


def _fr_decimals(text: str) -> str:
    return re.sub(r"(?<=\d),(?=\d)", ".", text)


def _find(text: str, synonyms: Sequence[str]) -> Optional[float]:
    for syn in synonyms:
        m = re.search(r"(?<![\w])(?:" + syn + r")" + _SEP + "(" + _NUM + ")", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


_TIME_MARK = re.compile(r"(?:\bt\s*=|\bat times?\b|\bat\b|\bafter\b|\bà\b|\baux instants\b|\bapr[eè]s\b|\btimes?\b)", re.IGNORECASE)


def _times(text: str) -> Optional[List[float]]:
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    for s in reversed(sentences):
        marks = list(_TIME_MARK.finditer(s))
        if marks:
            tail = s[marks[-1].end():]
            nums = [float(x) for x in re.findall(_NUM, tail)]
            if nums:
                return nums
    return None


def _increasing(ts: Optional[List[float]]) -> Optional[List[float]]:
    if not ts or any(b <= a for a, b in zip(ts, ts[1:])) or ts[0] <= 0:
        return None
    return ts


def baseline_formulate(question: str) -> Optional[Dict[str, Any]]:
    """Keyword/template formulator: regular expressions over a tiny vocabulary. Returns a validated spec, or None (could not formulate). No unit conversion."""
    q = _fr_decimals(question)
    ts = _increasing(_times(q))
    if ts is None:
        return None
    low = q.lower()
    try:
        spec: Optional[Dict[str, Any]] = None
        if re.search(r"\bseir\b", low):
            beta, sig, gam = _find(q, ["beta", "β"]), _find(q, ["sigma", "σ"]), _find(q, ["gamma", "γ"])
            i0, e0 = _find(q, ["i0", "i_0"]), _find(q, ["e0", "e_0"])
            s0 = _find(q, ["s0", "s_0"])
            if None in (beta, sig, gam, i0):
                return None
            e0 = 0.0 if e0 is None else e0
            s0 = 1.0 - i0 - e0 if s0 is None else s0
            spec = fam.seir_spec(beta, sig, gam, s0, e0, i0, ts)
        elif re.search(r"\bsir\b", low):
            beta, gam, i0, s0 = _find(q, ["beta", "β"]), _find(q, ["gamma", "γ"]), _find(q, ["i0", "i_0"]), _find(q, ["s0", "s_0"])
            if None in (beta, gam, i0):
                return None
            s0 = 1.0 - i0 if s0 is None else s0
            spec = fam.sir_spec(beta, gam, s0, i0, ts)
        elif re.search(r"\brlc\b|circuit", low):
            r, l_, c = _find(q, ["R", "resistance", "résistance"]), _find(q, ["L", "inductance"]), _find(q, ["C", "capacitance", "capacité"])
            v, q0, i0 = _find(q, ["V", "voltage", "tension"]), _find(q, ["q0", "q_0"]), _find(q, ["i0", "i_0"])
            if None in (r, l_, c):
                return None
            spec = fam.series_rlc_spec(r, l_, c, 0.0 if v is None else v, 0.0 if q0 is None else q0, 0.0 if i0 is None else i0, ts)
        elif re.search(r"spring|ressort", low):
            m, c, k = _find(q, ["m", "mass", "masse"]), _find(q, ["c", "damping", "amortissement"]), _find(q, ["k", "spring constant", "raideur"])
            f, x0, v0 = _find(q, ["F", "force"]), _find(q, ["x0", "x_0", "displacement", "déplacement"]), _find(q, ["v0", "v_0", "velocity", "vitesse"])
            if None in (m, k, x0):
                return None
            spec = fam.spring_mass_damper_spec(m, 0.0 if c is None else c, k, 0.0 if f is None else f, x0, 0.0 if v0 is None else v0, ts)
        elif re.search(r"pendul", low):
            g, d = _find(q, ["g_over_l", r"g\s*/\s*l"]), _find(q, ["damping", "friction", "amortissement", "frottement"])
            th, om = _find(q, ["theta0", "theta_0", "θ0", "initial angle", "angle initial", "angle"]), _find(q, ["omega0", "omega_0", "ω0", "angular velocity", "vitesse angulaire"])
            if th is None:
                return None
            spec = fam.pendulum_spec(th, 0.0 if om is None else om, ts, g_over_l=9.81 if g is None else g, damping=0.0 if d is None else d)
        elif re.search(r"projectile|ball\b|balle|cannonball|obus", low):
            g, k = _find(q, ["g"]), _find(q, ["drag coefficient", "drag", "k", "traînée", "trainee"])
            x0, y0 = _find(q, ["x0", "x_0"]), _find(q, ["y0", "y_0", "height", "hauteur"])
            vx, vy = _find(q, ["vx0", "vx_0", "vx"]), _find(q, ["vy0", "vy_0", "vy"])
            if vx is None or vy is None:
                return None
            spec = fam.projectile_drag_spec(0.0 if x0 is None else x0, 0.0 if y0 is None else y0, vx, vy, ts, g=9.81 if g is None else g, k=0.0 if k is None else k)
        elif re.search(r"kepler|orbit|orbite|planet|planète|two-body", low):
            mu = _find(q, ["mu", "μ"])
            x0, y0, vx, vy = (_find(q, ["x0", "x_0"]), _find(q, ["y0", "y_0"]), _find(q, ["vx0", "vx_0", "vx"]), _find(q, ["vy0", "vy_0", "vy"]))
            if None in (x0, y0, vx, vy):
                return None
            spec = fam.kepler_spec(x0, y0, vx, vy, ts, mu=1.0 if mu is None else mu)
        elif re.search(r"logisti", low):
            r, kk = _find(q, ["r", "growth rate", "taux de croissance"]), _find(q, ["K", "carrying capacity", "capacité"])
            n0 = _find(q, ["n0", "n_0", "initial population", "population initiale"])
            if n0 is None:
                return None
            spec = fam.logistic_spec(n0, ts, r=1.0 if r is None else r, K=1.0 if kk is None else kk)
        elif re.search(r"decay|chain|chaîne|chaine|stiff|raide|désintégration|desintegration", low):
            ks = [_find(q, [f"k{i}", f"k_{i}"]) for i in (1, 2, 3)]
            rates = [k for k in ks if k is not None]
            a0 = _find(q, ["a0", "a_0", "initial amount", "quantité initiale", "quantite initiale"])
            if len(rates) < 1 or a0 is None or ks[: len(rates)] != rates:
                return None
            spec = fam.decay_chain_spec(rates, [a0] + [0.0] * len(rates), ts)
        if spec is None:
            return None
        return validate_spec(spec)
    except (fam.FamilyError, SpecInvalid, ValueError, TypeError):
        return None


class BaselineBackend(Backend):
    """Adapter so the baseline runs through the same pipeline: the question is read from the last user message."""

    name = "baseline"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        text = messages[-1]["content"]
        m = re.search(r"<<<\n(.*?)\n>>>", text, re.DOTALL)
        spec = baseline_formulate(m.group(1) if m else text)
        return json.dumps(spec) if spec is not None else json.dumps({"cannot_formulate": "no template matched"})


# ======================================================================== named-slot formulator (pre-converting unit & derived-parameter slots)
def _find_with_unit(text: str, synonyms: Sequence[str], unit_re: str) -> Optional[float]:
    for syn in synonyms:
        m = re.search(r"(?<![\w])(?:" + syn + r")" + _SEP + "(" + _NUM + r")\s*(?:" + unit_re + r")(?![\w])", text, re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def _find_before_unit(text: str, unit_re: str, noun_re: Optional[str] = None) -> Optional[float]:
    tail = (r"\s+(?:" + noun_re + r")") if noun_re else ""
    m = re.search(r"(" + _NUM + r")\s*(?:" + unit_re + r")(?![\w])" + tail, text, re.IGNORECASE)
    return float(m.group(1)) if m else None


_SLOT_TIME_MARK = re.compile(
    r"(?:\bt\s*=|\bat times?\b|\bat\b|\bafter\b|\bà\b|\baux instants\b|\baux jours?\b|\bon days?\b|\bapr[eè]s\b|\btimes?\b)",
    re.IGNORECASE,
)


def _slot_times(text: str) -> Optional[List[float]]:
    m_grid = re.search(
        r"(?:over|simulate|pendant|sur)\s+(" + _NUM + r")\s*(?:seconds?|days?|minutes?|s|d|min)\b.*?\((\d+)\s*points?\)",
        text,
        re.IGNORECASE,
    )
    if m_grid:
        t_end = float(m_grid.group(1))
        n_pts = int(m_grid.group(2))
        if t_end > 0 and n_pts >= 2:
            return [round(i * t_end / (n_pts - 1), 12) for i in range(1, n_pts)]
    sentences = re.split(r"(?<=[.?!])\s+", text.strip())
    for s in reversed(sentences):
        marks = list(_SLOT_TIME_MARK.finditer(s))
        if marks:
            tail = s[marks[-1].end():]
            nums = [float(x) for x in re.findall(_NUM, tail)]
            if nums:
                return nums
    return None


def slot_formulate(question: str) -> Optional[Dict[str, Any]]:
    """Extract a `hivemind-solver-slots/1` slot dictionary (including unit conversions & derived parameters) and validate it via `compile_slot_spec`."""
    q = _fr_decimals(question)
    ts = _increasing(_slot_times(q))
    if ts is None:
        return None
    low = q.lower()
    sentences = re.split(r"(?<=[.?!])\s+", q.strip())
    last_sent = sentences[-1] if sentences else q
    time_in_ms = bool(re.search(r"\b(?:ms|milliseconds?|millisecondes?)\b", last_sent, re.IGNORECASE))
    t_key = "t_out_ms" if time_in_ms else "t_out"

    slot_obj: Optional[Dict[str, Any]] = None
    try:
        if re.search(r"\bseir\b|s\s*,\s*e\s*,\s*i\s*,\s*r\b|incubation", low):
            beta = _find(q, ["beta", "β", "transmission rate", "taux de transmission"])
            r0_repro = _find(q, ["R0", "R_0", "basic reproduction number", "taux de reproduction"])
            sig = _find(q, ["sigma", "σ"])
            inc_p = _find(q, ["incubation period", "incubation", "période d'incubation", "periode d'incubation"])
            gam = _find(q, ["gamma", "γ", "recovery rate", "taux de guérison"])
            inf_p = _find(q, ["average infectious period", "infectious period", "infectious for", "période infectieuse", "periode infectieuse"])
            e0_pct = _find_with_unit(q, ["e0", "e_0", "exposed", "exposés", "exposes"], r"%|percent|pourcent")
            i0_pct = _find_with_unit(q, ["i0", "i_0", "infectious", "infected", "infectés", "infectes"], r"%|percent|pourcent")
            e0 = _find(q, ["e0", "e_0"]) if e0_pct is None else None
            i0 = _find(q, ["i0", "i_0"]) if i0_pct is None else None
            if e0_pct is None and e0 is None:
                m_exp = re.search(r"(" + _NUM + r")\s*%\s*(?:are\s+|sont\s+)?(?:exposed|expos[eé]s?)", q, re.IGNORECASE)
                if m_exp:
                    e0_pct = float(m_exp.group(1))
            if i0_pct is None and i0 is None:
                m_inf = re.search(r"(" + _NUM + r")\s*%\s*(?:are\s+|sont\s+)?(?:infectious|infected|infect[eé]s?)", q, re.IGNORECASE)
                if m_inf:
                    i0_pct = float(m_inf.group(1))
            s0 = _find(q, ["s0", "s_0"])
            slots_d: Dict[str, Any] = {t_key: ts}
            if beta is not None:
                slots_d["beta"] = beta
            elif r0_repro is not None:
                slots_d["R0"] = r0_repro
            if sig is not None:
                slots_d["sigma"] = sig
            elif inc_p is not None:
                slots_d["incubation_period"] = inc_p
            if gam is not None:
                slots_d["gamma"] = gam
            elif inf_p is not None:
                slots_d["infectious_period"] = inf_p
            if e0_pct is not None:
                slots_d["e0_percent"] = e0_pct
            elif e0 is not None:
                slots_d["e0"] = e0
            if i0_pct is not None:
                slots_d["i0_percent"] = i0_pct
            elif i0 is not None:
                slots_d["i0"] = i0
            if s0 is not None:
                slots_d["s0"] = s0
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "seir", "slots": slots_d}

        elif re.search(r"\bsir\b|s\s*,\s*i\s*,\s*r\b", low):
            beta = _find(q, ["beta", "β", "transmission rate", "taux de transmission"])
            r0_repro = _find(q, ["R0", "R_0", "basic reproduction number", "taux de reproduction"])
            gam = _find(q, ["gamma", "γ", "recovery rate", "taux de guérison"])
            inf_p = _find(q, ["average infectious period", "infectious period", "infectious for", "période infectieuse", "periode infectieuse"])
            i0_pct = _find_with_unit(q, ["i0", "i_0", "infectious", "infected", "infectés", "infectes"], r"%|percent|pourcent")
            i0 = _find(q, ["i0", "i_0"]) if i0_pct is None else None
            if i0_pct is None and i0 is None:
                m_inf = re.search(
                    r"(" + _NUM + r")\s*%\s*(?:de\s+la\s+population\s+)?(?:are\s+|est\s+|sont\s+|initially\s+)?(?:infectious|infected|infect[eé]e?s?)",
                    q,
                    re.IGNORECASE,
                )
                if m_inf:
                    i0_pct = float(m_inf.group(1))
            s0 = _find(q, ["s0", "s_0"])
            slots_d = {t_key: ts}
            if beta is not None:
                slots_d["beta"] = beta
            elif r0_repro is not None:
                slots_d["R0"] = r0_repro
            if gam is not None:
                slots_d["gamma"] = gam
            elif inf_p is not None:
                slots_d["infectious_period"] = inf_p
            if i0_pct is not None:
                slots_d["i0_percent"] = i0_pct
            elif i0 is not None:
                slots_d["i0"] = i0
            if s0 is not None:
                slots_d["s0"] = s0
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "sir", "slots": slots_d}

        elif re.search(r"\brlc\b|circuit", low):
            r = _find(q, ["R", "resistance", "résistance"]) or _find_before_unit(q, r"ohms?|Ω")
            l_mh = _find_with_unit(q, ["L", "inductance"], r"mH") or _find_before_unit(q, r"mH")
            l_ = None if l_mh is not None else (_find(q, ["L", "inductance"]) or _find_before_unit(q, r"H", r"inductor|inductance"))
            c_mf = _find_with_unit(q, ["C", "capacitance", "capacité"], r"mF") or _find_before_unit(q, r"mF")
            c_uf = _find_with_unit(q, ["C", "capacitance", "capacité"], r"uF|µF") or _find_before_unit(q, r"uF|µF")
            c = None if (c_mf is not None or c_uf is not None) else (_find(q, ["C", "capacitance", "capacité"]) or _find_before_unit(q, r"F", r"capacitor|condensateur"))
            v = _find(q, ["V", "voltage", "tension"]) or _find_before_unit(q, r"V|volts?")
            q0 = _find(q, ["q0", "q_0"])
            i0 = _find(q, ["i0", "i_0"])
            if r is None:
                return None
            slots_d = {"R": r, "V": 0.0 if v is None else v, "q0": 0.0 if q0 is None else q0, "i0": 0.0 if i0 is None else i0, t_key: ts}
            if l_mh is not None:
                slots_d["L_mH"] = l_mh
            elif l_ is not None:
                slots_d["L"] = l_
            if c_mf is not None:
                slots_d["C_mF"] = c_mf
            elif c_uf is not None:
                slots_d["C_uF"] = c_uf
            elif c is not None:
                slots_d["C"] = c
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "series_rlc", "slots": slots_d}

        elif re.search(r"spring|ressort", low):
            m_g = _find_with_unit(q, ["m", "mass", "masse"], r"g|grams?|grammes?") or _find_before_unit(q, r"g|grams?|grammes?", r"mass|masse|block|bloc")
            m = None if m_g is not None else (_find(q, ["m", "mass", "masse"]) or _find_before_unit(q, r"kg", r"mass|masse|block|bloc"))
            c = _find(q, ["c", "damping coefficient", "coefficient d'amortissement", "damping", "amortissement"])
            k = _find(q, ["k", "spring constant", "raideur", "stiffness"]) or _find_before_unit(q, r"N/m")
            f = _find(q, ["F", "constant force", "force"]) or _find_before_unit(q, r"N", r"force")
            x0_cm = (
                _find_with_unit(q, ["x0", "x_0", "displacement", "déplacement", "pulled", "tiré", "tire", "stretched", "écarte", "ecarte", "écarté", "ecarte"], r"cm|centimet(?:er|re)s?")
                or _find_before_unit(q, r"cm|centimet(?:er|re)s?")
            )
            x0 = None if x0_cm is not None else _find(q, ["x0", "x_0", "x", "displacement", "déplacement", "pulled", "stretched"])
            v0 = _find(q, ["v0", "v_0", "velocity", "vitesse"])
            if k is None or (m is None and m_g is None):
                return None
            slots_d = {"k": k, "F": 0.0 if f is None else f, "v0": 0.0 if v0 is None else v0, t_key: ts}
            if m_g is not None:
                slots_d["m_g"] = m_g
            else:
                slots_d["m"] = m
            m_damp = re.search(
                r"damper\s+(?:exerts|produces|gives|resists\s+with)\s+(" + _NUM + r")\s*N\s+(?:at|when\s+[\w\s]+moves\s+at)\s+(" + _NUM + r")\s*m/s",
                q,
                re.IGNORECASE,
            )
            if m_damp:
                slots_d["damper_force"] = float(m_damp.group(1))
                slots_d["damper_speed"] = float(m_damp.group(2))
            elif c is not None:
                slots_d["c"] = c
            if x0_cm is not None:
                slots_d["x0_cm"] = x0_cm
            elif x0 is not None:
                slots_d["x0"] = x0
            else:
                slots_d["x0"] = 0.0
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "spring_mass_damper", "slots": slots_d}

        elif re.search(r"pendul", low):
            g_over_l = _find(q, ["g_over_l", r"g\s*/\s*l"])
            length_m = _find_with_unit(q, ["length", "longueur", "L", "l"], r"m|met(?:er|re)s?|mètres?")
            length_cm = _find_with_unit(q, ["length", "longueur", "L", "l"], r"cm|centimet(?:er|re)s?")
            period_ms = _find_with_unit(q, ["period", "small-angle period", "période", "periode", "T"], r"ms|milliseconds?|millisecondes?")
            period_s = None if period_ms is not None else _find_with_unit(q, ["period", "small-angle period", "période", "periode", "T"], r"s|sec|seconds?|secondes?")
            g_val = _find(q, ["g", "gravity", "gravité"])
            d = _find(q, ["damping", "friction", "amortissement", "frottement", "gamma"])
            th_deg = (
                _find_with_unit(q, ["theta0", "theta_0", "θ0", "initial angle", "angle initial", "angle", "at"], r"deg|degrees?|degrés?|°")
                or _find_before_unit(q, r"deg|degrees?|degrés?|°")
            )
            th = None if th_deg is not None else _find(q, ["theta0", "theta_0", "θ0", "initial angle", "angle initial", "angle"])
            om = _find(q, ["omega0", "omega_0", "ω0", "angular velocity", "vitesse angulaire"])
            if th is None and th_deg is None:
                return None
            slots_d = {"omega0": 0.0 if om is None else om, "damping": 0.0 if d is None else d, t_key: ts}
            if g_over_l is not None:
                slots_d["g_over_l"] = g_over_l
            elif period_ms is not None:
                slots_d["period_ms"] = period_ms
            elif period_s is not None:
                slots_d["period_s"] = period_s
            else:
                if length_m is not None:
                    slots_d["length_m"] = length_m
                elif length_cm is not None:
                    slots_d["length_cm"] = length_cm
                if g_val is not None:
                    slots_d["g"] = g_val
            if th_deg is not None:
                slots_d["theta0_deg"] = th_deg
            else:
                slots_d["theta0"] = th
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "pendulum", "slots": slots_d}

        elif re.search(r"projectile|ball\b|balle|cannonball|obus", low):
            g_val = _find(q, ["g", "gravity", "gravité"])
            k = _find(q, ["drag coefficient", "quadratic drag coefficient", "traînée quadratique de coefficient", "trainee quadratique de coefficient", "coefficient", "drag", "k", "traînée", "trainee"])
            x0 = _find(q, ["x0", "x_0"])
            y0 = _find(q, ["y0", "y_0", "height", "hauteur"])
            vx = _find(q, ["vx0", "vx_0", "vx"])
            vy = _find(q, ["vy0", "vy_0", "vy"])
            speed_kmh = (
                _find_with_unit(q, ["speed", "vitesse", "v0", "launched at", "thrown at", "horizontally at", "lancée à", "lancee a", "lancée", "lancee"], r"km/h|kph")
                or _find_before_unit(q, r"km/h|kph")
            )
            speed = None if speed_kmh is not None else (
                _find(q, ["speed", "vitesse", "v0", "launched at", "thrown at"])
                or _find_before_unit(q, r"m/s")
            )
            slots_d = {"x0": 0.0 if x0 is None else x0, "y0": 0.0 if y0 is None else y0,
                       "g": 9.81 if g_val is None else g_val, "k": 0.0 if k is None else k, t_key: ts}
            if vx is not None and vy is not None:
                slots_d["vx0"], slots_d["vy0"] = vx, vy
            else:
                if speed_kmh is not None:
                    slots_d["speed_kmh"] = speed_kmh
                elif speed is not None:
                    slots_d["speed"] = speed
                else:
                    return None
                m_slope = re.search(r"(?:rising|rises)\s+(" + _NUM + r")\s+\w+\s+for\s+every\s+(" + _NUM + r")", q, re.IGNORECASE)
                if m_slope:
                    slots_d["slope_rise"] = float(m_slope.group(1))
                    slots_d["slope_run"] = float(m_slope.group(2))
                else:
                    deg = _find_with_unit(q, ["angle", "at"], r"deg|degrees?|degrés?|°") or _find_before_unit(q, r"deg|degrees?|degrés?|°")
                    if deg is not None:
                        slots_d["angle_deg"] = deg
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "projectile_drag", "slots": slots_d}

        elif re.search(r"kepler|orbit|orbite|planet|planète|two-body", low):
            mu = _find(q, ["mu", "μ"])
            r0 = _find(q, ["r0", "r_0", "radius", "rayon"])
            x0, y0, vx, vy = (_find(q, ["x0", "x_0"]), _find(q, ["y0", "y_0"]), _find(q, ["vx0", "vx_0", "vx"]), _find(q, ["vy0", "vy_0", "vy"]))
            slots_d = {"mu": 1.0 if mu is None else mu, t_key: ts}
            if re.search(r"circular|circulaire", low) and r0 is not None:
                slots_d["circular_radius"] = r0
            elif None not in (x0, y0, vx, vy):
                slots_d.update({"x0": x0, "y0": y0, "vx0": vx, "vy0": vy})
            else:
                return None
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "kepler", "slots": slots_d}

        elif re.search(r"logisti", low):
            m_pct_min = re.search(r"(" + _NUM + r")\s*%\s*per\s*minute", q, re.IGNORECASE)
            r = _find(q, ["r", "growth rate", "taux de croissance"])
            kk = _find(q, ["K", "carrying capacity", "capacité"])
            n0 = _find(q, ["n0", "n_0", "p0", "initial population", "population initiale", "culture", "population"])
            if n0 is None:
                return None
            slots_d = {"K": 1.0 if kk is None else kk, "n0": n0, t_key: ts}
            if m_pct_min:
                slots_d["r_percent_per_min"] = float(m_pct_min.group(1))
            elif r is not None:
                slots_d["r"] = r
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "logistic", "slots": slots_d}

        elif re.search(r"decay|chain|chaîne|chaine|stiff|raide|désintégration|desintegration|turns\s+into", low):
            ks = [_find(q, [f"k{i}", f"k_{i}"]) for i in (1, 2, 3)]
            rates = [k for k in ks if k is not None]
            a0 = _find(q, ["a0", "a_0", "initial amount", "quantité initiale", "quantite initiale", "start with"])
            if a0 is None:
                m_a0 = re.search(r"start(?:ing)?\s+(?:with|from)\s+(" + _NUM + r")\s*(?:units?|mols?|moles?)?\s+(?:of\s+)?A\b", q, re.IGNORECASE)
                if m_a0:
                    a0 = float(m_a0.group(1))
            if len(rates) >= 1 and a0 is not None and ks[: len(rates)] == rates:
                per_min = bool(re.search(r"per\s+minute|par\s+minute|/min", low))
                slots_d = {"a0": a0, ("rates_per_min" if per_min else "rates"): rates, t_key: ts}
                slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "decay_chain", "slots": slots_d}
            elif a0 is not None:
                # Check for "turns into ... at X per minute" or "time constant of X ms / s"
                m_per_min = re.findall(r"turns\s+into\s+[B-Z]\s+at\s+(" + _NUM + r")\s*per\s*minute", q, re.IGNORECASE)
                m_tau = re.findall(r"time\s+constant\s+of\s+(" + _NUM + r")\s*(ms|s|seconds?|milliseconds?)\b", q, re.IGNORECASE)
                if m_per_min:
                    slots_d = {"a0": a0, "rates_per_min": [float(v) for v in m_per_min], t_key: ts}
                    slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "decay_chain", "slots": slots_d}
                elif m_tau:
                    taus_s = [float(v) / 1000.0 if u.lower().startswith("m") else float(v) for v, u in m_tau]
                    slots_d = {"a0": a0, "time_constants_s": taus_s, t_key: ts}
                    slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "decay_chain", "slots": slots_d}

        elif re.search(r"yukawa|qf_yukawa", low):
            kappa = _find(q, ["kappa", "κ", "screening"])
            damping = _find(q, ["damping", "gamma", "γ"])
            x0 = _find(q, ["x0", "x_0", "r0", "r_0"])
            v0 = _find(q, ["v0", "v_0", "vr0", "v_r0"])
            slots_d = {t_key: ts}
            for k_name, v_val in (("kappa", kappa), ("damping", damping), ("x0", x0), ("v0", v0)):
                if v_val is not None:
                    slots_d[k_name] = v_val
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "yukawa_crystal", "slots": slots_d}

        elif re.search(r"gross-pitaevskii|\bgpe\b|bogoliubov|condensate|qf_pgpe", low):
            omega_trap = _find(q, ["omega_trap", "omega", "ω", "trap frequency"])
            g_int = _find(q, ["g_int", "interaction", "g"])
            zeta = _find(q, ["zeta", "damping", "gamma_diss", "gamma", "γ"])
            x0 = _find(q, ["x0", "x_0", "psi_re0"])
            v0 = _find(q, ["v0", "v_0", "psi_im0"])
            slots_d = {t_key: ts}
            for k_name, v_val in (("omega_trap", omega_trap), ("g_int", g_int), ("zeta", zeta), ("x0", x0), ("v0", v0)):
                if v_val is not None:
                    slots_d[k_name] = v_val
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "gpe_vortex", "slots": slots_d}

        elif re.search(r"art-exo|art_exo|proofreading|exonuclease|chromatin_remodeling", low):
            k_pol = _find(q, ["k_pol", "kpol", "polymerization rate"])
            k_exo = _find(q, ["k_exo", "kexo", "exonuclease rate", "proofreading rate"])
            k_ext = _find(q, ["k_ext", "kext", "k_lig", "extension rate"])
            t0_c = _find(q, ["T0", "t0_conc", "c0", "c_0"])
            m0_c = _find(q, ["M0", "m0_conc", "m0", "m_0"])
            p0_c = _find(q, ["P0", "p0_conc", "p0", "p_0"])
            slots_d = {t_key: ts}
            for k_name, v_val in (("k_pol", k_pol), ("k_exo", k_exo), ("k_ext", k_ext), ("T0", t0_c), ("M0", m0_c), ("P0", p0_c)):
                if v_val is not None:
                    slots_d[k_name] = v_val
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "art_exo_genetics", "slots": slots_d}

        elif re.search(r"quintessence|klein-gordon|bao_cosmology|k3_quintessence", low):
            m_eff = _find(q, ["m_eff", "m_phi", "mphi", "mass", "scalar mass"])
            hubble = _find(q, ["hubble", "h_hub", "H"])
            phi0 = _find(q, ["phi0", "phi_0", "φ0"])
            dphi0 = _find(q, ["dphi0", "dphi_0"])
            slots_d = {t_key: ts}
            for k_name, v_val in (("m_eff", m_eff), ("hubble", hubble), ("phi0", phi0), ("dphi0", dphi0)):
                if v_val is not None:
                    slots_d[k_name] = v_val
            slot_obj = {"schema": fam.SLOT_SCHEMA, "slot_family": "bao_cosmology", "slots": slots_d}

        if slot_obj is None:
            return None
        # Ensure it compiles and validates cleanly before returning the slot proposal
        compile_or_validate_proposal(slot_obj)
        return slot_obj
    except (fam.FamilyError, SpecInvalid, ValueError, TypeError):
        return None


class SlotBackend(Backend):
    """Named-slot backend that emits `hivemind-solver-slots/1` proposals with unit and derived-parameter slots."""

    name = "slot"

    def complete(self, messages: List[Dict[str, str]]) -> str:
        text = messages[-1]["content"]
        m = re.search(r"<<<\n(.*?)\n>>>", text, re.DOTALL)
        slot_obj = slot_formulate(m.group(1) if m else text)
        return json.dumps(slot_obj) if slot_obj is not None else json.dumps({"cannot_formulate": "missing or ambiguous parameters for slot schema"})

