#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Experiment S-3 runner: plain-language question -> spec -> solve -> verify, scored against the frozen answers (stdlib only).

Pre-registered in artifacts/experiments/S3_formulation_prereg.json; the question set is artifacts/experiments/S3_questions.json (sha256 checked at start).

    run_experiment_S3_formulation.py --arm baseline --binary BIN
    run_experiment_S3_formulation.py --arm model --model qwen2.5:7b --cpu --binary BIN        (CPU only: options.num_gpu = 0)
    run_experiment_S3_formulation.py --arm model --model qwen2.5:7b --gpu --binary BIN        (GPU arm: run by the lead, in the owner's night window only)
    run_experiment_S3_formulation.py --arm model --model fake --binary BIN                    (dry run: a scripted fake backend answers the expected spec)

The script never starts, stops or configures Ollama; it only talks to an already running loopback server. Nothing here changes a tolerance or a threshold:
the agreement rule is fixed in the pre-registration (factor 100, rtol 1e-8, atol 1e-10, independent of the spec's own tolerances).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess  # noqa: S404 - fixed argv list, never a shell
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from hivemind.solverbee import science_bee as sb  # noqa: E402

EXP = REPO / "artifacts" / "experiments"
QUESTIONS = EXP / "S3_questions.json"
PREREG = EXP / "S3_formulation_prereg.json"
DEFAULT_BINARY = "/mnt/disks/disk-socrateai-local-1/solver-build/target-a/release/hivemind-solver"
SCHEMA = "hivemind-s3-arm/1"
POLICY_FACTOR, POLICY_RTOL, POLICY_ATOL = 100.0, 1e-8, 1e-10
RAW_KEEP = 600


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_questions(path: Path = QUESTIONS) -> Dict[str, Any]:
    """The frozen question set, after checking its sha256 against S3_questions.sha256 next to it."""
    expected = path.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    actual = sha256_file(path)
    if actual != expected:
        raise SystemExit(f"question set hash mismatch: {actual} != {expected}")
    return json.loads(path.read_text(encoding="utf-8"))


def agreement(result: Optional[Dict[str, Any]], expected: Dict[str, Any]) -> bool:
    """Pre-registered agreement: same times (rel 1e-9), same shape, every component within 100*(1e-8*|ref|+1e-10)."""
    try:
        if not isinstance(result, dict) or result.get("status") != "ok":
            return False
        t, y = result["t"], result["y"]
        et, ey = expected["t"], expected["y"]
        if len(t) != len(et) or len(y) != len(ey):
            return False
        for a, b in zip(t, et):
            if abs(a - b) > 1e-9 * max(abs(a), abs(b)) + 1e-15:
                return False
        for row, ref in zip(y, ey):
            if len(row) != len(ref):
                return False
            for a, b in zip(row, ref):
                if not isinstance(a, (int, float)) or isinstance(a, bool) or a != a or abs(a) == float("inf"):
                    return False
                if abs(a - b) > POLICY_FACTOR * (POLICY_RTOL * abs(b) + POLICY_ATOL):
                    return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def binary_info(binary: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path_basename": Path(binary).name, "sha256": None, "families": [], "version": None}
    try:
        info["sha256"] = sha256_file(Path(binary))
        out = subprocess.run([binary, "families"], capture_output=True, timeout=30, check=False)  # noqa: S603
        info["families"] = sorted(f["family"] for f in json.loads(out.stdout.decode("utf-8"))["families"])
        out = subprocess.run([binary, "version"], capture_output=True, timeout=30, check=False)  # noqa: S603
        info["version"] = json.loads(out.stdout.decode("utf-8"))
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        info["error"] = type(exc).__name__
    return info


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


def expected_family(q: Dict[str, Any]) -> Optional[str]:
    prob = q["expected_spec"]["problem"]
    return prob["family"] if prob["kind"] == "named" else None


def final_family(spec: Optional[Dict[str, Any]]) -> Optional[str]:
    if spec and spec["problem"]["kind"] == "named":
        return spec["problem"]["family"]
    return None


class OracleFake(sb.FakeBackend):
    """Dry-run backend: answers the expected spec of the question found in the prompt (a harness check, not a model)."""

    def __init__(self, questions: List[Dict[str, Any]]) -> None:
        self._by_text = {q["text"]: q["expected_spec"] for q in questions}
        super().__init__(self._answer)

    def _answer(self, messages: List[Dict[str, str]]) -> str:
        m = re.search(r"<<<\n(.*?)\n>>>", messages[-1]["content"], re.DOTALL)
        spec = self._by_text.get(m.group(1)) if m else None
        return json.dumps(spec) if spec else json.dumps({"cannot_formulate": "unknown question"})


def run_question(q: Dict[str, Any], bee: "sb.ScienceBee", supported: List[str]) -> Dict[str, Any]:
    bee.reset()
    ans = bee.ask(q["text"], follow_up=False, language=q["lang"])
    spec, report = ans["spec"], ans["report"]
    result = report.get("result") if report else None
    verdict = ans["verdict"] if ans["formulated"] else "COULD_NOT_FORMULATE"
    agree = agreement(result, q["expected_answers"]) if ans["formulated"] else False
    correct = bool(agree and verdict in ("VERIFIED", "AUDITED"))
    fam_final, fam_exp = final_family(spec), expected_family(q)
    return {
        "id": q["id"], "tier": q["tier"], "lang": q["lang"], "type": q["type"],
        "formulated": ans["formulated"], "attempts": ans["attempts"], "retries": ans["retries"],
        "raw_outputs": [r[:RAW_KEEP] for r in ans["raw_outputs"]], "validation_errors": ans["errors"][:4],
        "final_spec_sha256": ans["spec_sha256"], "exact_match": bool(spec is not None and sb.specs_match(spec, q["expected_spec"])),
        "verdict": verdict, "agreement": agree, "correct": correct,
        "verified_but_wrong": bool(verdict == "VERIFIED" and not agree), "audited_but_wrong": bool(verdict == "AUDITED" and not agree),
        "expected_family_supported_by_binary": fam_exp is None or fam_exp in supported,
        "final_family_supported_by_binary": fam_final is None or fam_final in supported,
        "answer_contains_limits_sentence": sb.LIMITS_SENTENCE in ans["text"],
        "wall_seconds": ans["wall_seconds"],
    }


def summarise(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    def block(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(rs)
        return {"n": n, "correct": sum(r["correct"] for r in rs), "exact_match": sum(r["exact_match"] for r in rs),
                "could_not_formulate": sum(not r["formulated"] for r in rs), "retries_total": sum(r["retries"] for r in rs),
                "verified_but_wrong": sum(r["verified_but_wrong"] for r in rs), "audited_but_wrong": sum(r["audited_but_wrong"] for r in rs),
                "wall_seconds_mean": round(sum(r["wall_seconds"] for r in rs) / n, 3) if n else None,
                "correct_among_supported_families": sum(r["correct"] for r in rs if r["expected_family_supported_by_binary"]),
                "n_supported_families": sum(1 for r in rs if r["expected_family_supported_by_binary"])}
    return {"overall": block(records),
            "by_tier": {t: block([r for r in records if r["tier"] == t]) for t in ("A", "B", "C")},
            "by_lang": {lg: block([r for r in records if r["lang"] == lg]) for lg in ("en", "fr")},
            "verdict_counts": {v: sum(1 for r in records if r["verdict"] == v) for v in sorted({r["verdict"] for r in records})},
            "all_answers_contain_limits_sentence": all(r["answer_contains_limits_sentence"] for r in records)}


def build_backend(args: argparse.Namespace, questions: List[Dict[str, Any]]) -> "sb.Backend":
    if args.arm == "baseline":
        return sb.BaselineBackend()
    if args.model == "fake":
        return OracleFake(questions)
    num_gpu = 999 if args.gpu else 0
    return sb.OllamaBackend(args.model, base_url=args.base_url, num_gpu=num_gpu, timeout_s=args.timeout, think=False if args.no_think else None)


def server_version(backend: "sb.Backend") -> Optional[str]:
    """Read-only GET /api/version of the already running local Ollama (never starts or configures it)."""
    if not isinstance(backend, sb.OllamaBackend):
        return None
    try:
        with backend._opener.open(backend.base_url + "/api/version", timeout=10) as resp:  # noqa: S310 - loopback checked in OllamaBackend
            return str(json.loads(resp.read(4096).decode("utf-8")).get("version"))
    except (OSError, ValueError):
        return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=("baseline", "model"), required=True)
    ap.add_argument("--model", default="none", help="Ollama model name for the model arm, or 'fake' for the scripted dry run")
    dev = ap.add_mutually_exclusive_group()
    dev.add_argument("--cpu", action="store_true", help="CPU only (num_gpu = 0); the default")
    dev.add_argument("--gpu", action="store_true", help="GPU arm (num_gpu = 999); only the lead runs this, in the owner's night window")
    ap.add_argument("--binary", default=DEFAULT_BINARY)
    ap.add_argument("--problems", default=str(REPO / "solvers" / "problems"), help="problem library directory (reference lookup)")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--ids", default="", help="comma separated question ids (reduced set)")
    ap.add_argument("--out", default="", help="output file (default artifacts/experiments/S3_formulation_<arm>_<model>_<device>.json)")
    ap.add_argument("--base-url", default="http://127.0.0.1:11434")
    ap.add_argument("--timeout", type=float, default=600.0, help="per model call, seconds")
    ap.add_argument("--no-think", action="store_true", help="send think=false (for reasoning models)")
    ap.add_argument("--allow-unsandboxed", action="store_true", help="only when the namespace sandbox is unavailable; recorded in the file")
    args = ap.parse_args(argv)
    if args.arm == "model" and args.model == "none":
        ap.error("--model is required for the model arm")
    qdoc = load_questions(Path(args.questions))
    questions = qdoc["questions"]
    if args.ids:
        wanted = set(args.ids.split(","))
        questions = [q for q in questions if q["id"] in wanted]
    device = "gpu" if args.gpu else "cpu"
    label_model = "baseline" if args.arm == "baseline" else re.sub(r"[^A-Za-z0-9_.-]", "-", args.model)
    out = Path(args.out) if args.out else EXP / f"S3_formulation_{args.arm}_{label_model}_{device}.json"
    backend = build_backend(args, questions)
    pipeline = sb.SolverPipeline(args.binary, problems_dir=args.problems, allow_unsandboxed=args.allow_unsandboxed)
    bee = sb.ScienceBee(backend, pipeline)
    binfo = binary_info(args.binary)
    t0 = time.monotonic()
    records = []
    for q in questions:
        rec = run_question(q, bee, binfo["families"])
        records.append(rec)
        print(f"{rec['id']} {rec['tier']} {rec['lang']} {rec['type']:<10} {rec['verdict']:<20} agree={int(rec['agreement'])} exact={int(rec['exact_match'])} retries={rec['retries']}", flush=True)
    doc: Dict[str, Any] = {
        "schema": SCHEMA, "experiment_id": "S3-science-bee-formulation", "arm": args.arm, "model": label_model, "device": device,
        "num_gpu": (999 if args.gpu else 0) if args.arm == "model" and args.model != "fake" else None,
        "is_dry_run_with_scripted_fake_backend": bool(args.arm == "model" and args.model == "fake"),
        "ollama_version": server_version(backend), "unsandboxed_allowed": bool(args.allow_unsandboxed),
        "binary": binfo, "questions_sha256": sha256_file(Path(args.questions)), "prereg_sha256": sha256_file(PREREG) if PREREG.exists() else None,
        "science_bee_py_sha256": sha256_file(REPO / "src" / "hivemind" / "solverbee" / "science_bee.py"),
        "environment": {"python": platform.python_version(), "cpu": cpu_model(), "cpus": __import__("os").cpu_count()},
        "n_questions_run": len(records), "reduced_set": bool(args.ids), "total_wall_seconds": round(time.monotonic() - t0, 2),
        "summary": summarise(records), "records": records,
    }
    doc["receipt_sha256"] = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    s = doc["summary"]
    print(f"\n{args.arm}/{label_model}/{device}: correct {s['overall']['correct']}/{s['overall']['n']}  " + "  ".join(f"{t}: {b['correct']}/{b['n']}" for t, b in s["by_tier"].items())
          + f"  exact-match {s['overall']['exact_match']}  could-not-formulate {s['overall']['could_not_formulate']}  verified_but_wrong {s['overall']['verified_but_wrong']}")
    print(f"wrote {out}  receipt {doc['receipt_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
