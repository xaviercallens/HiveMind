"""Audit every Python file under ``src/`` and ``scripts/`` with GWAYA's zero-stub AST auditor
(``gwaya.ast_audit``, GWAYA-GwenLayaOpenModel, MIT) and write ``artifacts/gwaya_stub_audit.json``.

Part of the Definition of Done in ``docs/ENGINEERING_COORDINATION_BRIEF.md``: a file with a stub
(``pass`` / ``...`` / ``NotImplementedError`` body outside tests) fails the audit. Requires the
``gwaya`` package (``uv sync --extra gwaya`` in AutoevolveAI, or ``pip install`` from the GWAYA repo).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
THIRD_PARTY = PROJECT_ROOT / "third_party"
if THIRD_PARTY.is_dir() and str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))
OUT = PROJECT_ROOT / "artifacts" / "gwaya_stub_audit.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["src", "scripts"])
    args = ap.parse_args()
    try:
        from gwaya.ast_audit import ZeroStubAudit  # type: ignore
    except ModuleNotFoundError:
        report = {"status": "BLOCKED", "reason": "gwaya package not importable in this interpreter", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))
        return 2
    files: List[Path] = []
    for root in args.roots:
        files.extend(sorted((PROJECT_ROOT / root).rglob("*.py")))
    results: List[Dict[str, Any]] = []
    for f in files:
        res = ZeroStubAudit.audit_python_code(f.read_text(encoding="utf-8"))
        d = res.__dict__ if hasattr(res, "__dict__") else dict(res)  # dataclass-like result
        passed = bool(d.get("passed", d.get("is_clean", not d.get("stubs"))))
        results.append({"file": str(f.relative_to(PROJECT_ROOT)), "passed": passed, "detail": {k: (v if isinstance(v, (int, float, str, bool, list, dict)) else str(v)) for k, v in d.items()}})
    failed = [r["file"] for r in results if not r["passed"]]
    report = {
        "schema_version": "hivemind-gwaya-stub-audit/v1.0",
        "auditor": "gwaya.ast_audit.ZeroStubAudit (GWAYA-GwenLayaOpenModel)",
        "files_audited": len(results), "files_failed": failed, "all_passed": not failed,
        "results": results, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    payload = json.dumps({k: v for k, v in report.items() if k != "receipt_sha256"}, sort_keys=True).encode()
    report["receipt_sha256"] = hashlib.sha256(payload).hexdigest()
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"files_audited": len(results), "files_failed": failed, "all_passed": not failed}, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
