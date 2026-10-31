# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Public results page for the NIGHT-MOE-1 expert-adapter experiment.

Reads the JSON that ``scripts/build_night_moe_report.py`` writes and renders it as a self-contained HTML page. Nothing is computed
here: every number is copied from the report, and every string is HTML-escaped before it reaches the page.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_PATH = Path("artifacts/experiments/NIGHT_MOE_results.json")


def load_night_moe(path: Path | str = DEFAULT_RESULTS_PATH) -> dict[str, Any] | None:
    """The report as a dict, or ``None`` when the file is absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and "subjects" in data else None


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4g}"
    return escape(str(value))


def render_night_moe_html(report: dict[str, Any] | None) -> str:
    """Self-contained HTML for the report (an explanatory stub page when ``report`` is ``None``)."""
    head = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>NIGHT-MOE-1 results</title><style>body{font:15px system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:16px;"
        "background:#10141a;color:#e6e9ee}table{border-collapse:collapse;width:100%;margin:12px 0;display:block;overflow-x:auto}"
        "th,td{border:1px solid #2c3440;padding:5px 8px;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}"
        ".VALID,.GAIN{color:#6fd08c}.VOID,.HARM{color:#ff7b72}.NO_GAIN,.NS,.NOT_RUN{color:#c9b458}a{color:#7cb7ff}</style></head><body>"
    )
    if report is None:
        return head + "<h1>NIGHT-MOE-1 results</h1><p>No results file is deployed with this service yet.</p></body></html>"
    rows = []
    for name, s in report["subjects"].items():
        own, ctl = s["own"], s.get("control")
        rows.append(
            f"<tr><td>{escape(name)}</td><td>{escape(s['own_pack'])} (n={own['n']})</td><td>{_fmt(own['base_accuracy_same_items'])}</td>"
            f"<td>{_fmt(own['accuracy'])}</td><td>{_fmt(own['delta_vs_base'])}</td><td>{_fmt(own['mcnemar_exact_p'])}</td>"
            f"<td>{_fmt(ctl['accuracy']) if ctl else '-'} <span class='{escape(ctl['flag']) if ctl else ''}'>{escape(ctl['flag']) if ctl else ''}</span></td>"
            f"<td class='{escape(s['verdict'])}'><b>{escape(s['verdict'])}</b></td></tr>"
        )
    cells = []
    for name, s in report["subjects"].items():
        for pack, c in s["other_packs"].items():
            cells.append(
                f"<tr><td>{escape(name)}</td><td>{escape(pack)}</td><td>{escape(c['kind'])}</td><td>{c['n']}</td>"
                f"<td>{_fmt(c['base_accuracy_same_items'])}</td><td>{_fmt(c['accuracy'])}</td><td>{_fmt(c['delta_vs_base'])}</td>"
                f"<td>{_fmt(c['mcnemar_exact_p'])}</td><td class='{escape(c['flag'])}'>{escape(c['flag'])}</td>"
                f"<td>{'yes' if c.get('survives_bonferroni') else 'no'}</td><td>{_fmt(c['unparsable'])}</td></tr>"
            )
    notes = []
    for name, s in report["subjects"].items():
        notes.append(f"<li><b>{escape(name)}</b>: {escape(s['verdict'])}, {escape(s['reason'])}</li>")
        if s.get("control_note"):
            notes.append(f"<li style='list-style:none'>control: {escape(s['control_note'])}</li>")
        if s.get("missing_cells"):
            notes.append(f"<li style='list-style:none'>no result for: {escape(', '.join(s['missing_cells']))}</li>")
    return (
        head
        + f"<h1>NIGHT-MOE-1 results</h1><p>{escape(str(report['base_model']))}. Exact paired McNemar against the base model on identical items; "
        f"alpha {_fmt(report['alpha'])}, Bonferroni alpha {_fmt(report['bonferroni_alpha'])} over {report['n_other_cells']} transfer/interference cells. "
        "These are adapters evaluated offline on a T4; none is served by the router. <a href='/api/v1/night-moe'>JSON</a> · "
        "<a href='/join?lang=en#train'>Join Night Training (#train)</a> · <a href='/boinc/status'>BOINC Status</a></p>"
        "<h2>Night Training Roles (22:00–07:00 Paris Time)</h2>"
        "<ul>"
        "<li><b>For End-Users (Volunteers, Developers &amp; Gamers):</b> Leave your PC or laptop running overnight (or in automatic idle-GPU pause mode via "
        "<code>scripts/hivemind_boinc_idle_worker.py</code> or <code>boinccmd</code>). Your machine leases nocturnal BOINC workunits (<code>physics_sim</code>, "
        "<code>code_generation</code>, <code>math_reasoning</code>), folds 16,384-point TESS/JWST exoplanet light curves, and computes rank-4 QLoRA inner-loop updates, "
        "earning <b>250 HMT Reflexion Points</b> and <b>45 Cobblestones</b> per verified workunit to spend the next day on all 15+ models.</li>"
        "<li><b>For HiveMind Nodes (Dedicated GPU Producers, HiveMiners &amp; Owner Fleet — RTX 2070 8 GB + GCP Tesla T4 16 GB):</b> "
        "During the day (07:00–22:00), HiveMind nodes serve low-latency OpenRouter inference (<code>hivemind/coder</code>, <code>hivemind/bee-stars</code>, etc.). "
        "At night (22:00–07:00), HiveMind nodes automatically switch to <b>Nocturnal DiLoCo Outer-Step Aggregation, Multi-Krum Byzantine Verification, and SVD "
        "Teacher-to-Student Distillation</b> (<a href='/v1/hivemind/bees/owner-fleet-status'>Live Node Fleet Telemetry</a>).</li>"
        "</ul>"
        "<h2>Subjects</h2><table><tr><th>Subject</th><th>Own pack</th><th>Base</th><th>Adapter</th><th>Delta</th><th>p</th><th>Shuffled control</th>"
        "<th>Verdict</th></tr>" + "".join(rows) + "</table><h2>Transfer and interference</h2><table><tr><th>Adapter</th><th>Pack</th><th>Kind</th>"
        "<th>n</th><th>Base (same items)</th><th>Adapter</th><th>Delta</th><th>p</th><th>Flag</th><th>Bonferroni</th><th>Unparsable</th></tr>"
        + "".join(cells) + "</table><h2>Reasons and gaps</h2><ul>" + "".join(notes) + "</ul></body></html>"
    )
