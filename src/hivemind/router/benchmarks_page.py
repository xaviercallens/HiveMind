# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Benchmarks page: MEASURED results only.

Served by the gateway at ``GET /benchmarks`` and ``GET /comparatif`` (HTML, ``fr``/``en``/``es``/``de``/``zh``),
``GET /v1/hivemind/benchmarks`` (JSON) and ``GET /v1/hivemind/benchmarks/infographic.svg``.

Until 2026-11-03 this module held about 1,100 lines of hard-coded latency, quality and "beats GPT-4o / Claude / Gemini" figures for 18 bees,
none of which had a measurement behind it, plus tiers that no producer serves. They are gone (git history keeps them). What remains is
honest by construction:

* a result is shown only if it is in ``measured_results.json`` (next to this file, so it ships in the gateway image) together with its
  sample size, date, hardware, method and source artifact;
* every catalog tier without a measured result is listed as "not measured";
* no comparison with a proprietary model is shown, because no such comparison has been run.

To add a result, run the harness (``scripts/night_eval.py`` for quality, ``scripts/mvp_e2e_smoke.py`` for the metering path), keep its
artifact, and add an entry with the same figures to ``measured_results.json``.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

SUPPORTED_BENCHMARK_LANGS: tuple[str, ...] = ("fr", "en", "es", "de", "zh")
MEASURED_RESULTS_PATH = Path(__file__).with_name("measured_results.json")

POC_NOTICE = {
    "en": "Proof of concept (PoC): not a production service yet. Send only public, non-sensitive data (no secrets, personal, private or enterprise data).",
    "fr": "Preuve de concept (PoC) : pas encore un service de production. N'envoyez que des données publiques et non sensibles (ni secrets, ni données personnelles, privées ou d'entreprise).",
    "es": "Prueba de concepto (PoC): todavía no es un servicio de producción. Envíe solo datos públicos y no sensibles (sin secretos, datos personales, privados ni empresariales).",
    "de": "Proof of Concept (PoC): noch kein Produktionsdienst. Senden Sie nur öffentliche, nicht sensible Daten (keine Geheimnisse, personenbezogenen, privaten oder Unternehmensdaten).",
    "zh": "概念验证 (PoC)：尚未成为生产服务。请仅发送公开的非敏感数据（不要发送机密、个人、私人或企业数据）。",
}

_TEXT: dict[str, dict[str, str]] = {
    "en": {
        "title": "HiveMind benchmarks: measured results only",
        "intro": "Only figures that were actually measured are shown, each with its sample size, date, hardware and method. Everything else says 'not measured'. HiveMind does not compare itself with GPT-4o, Claude or Gemini: no such comparison has been run.",
        "measured": "Measured results",
        "none": "No measured result is published in this build.",
        "unmeasured": "Tiers without a measured result",
        "unmeasured_note": "These tiers are served by the off-the-shelf model shown. No quality or latency figure is published for them.",
        "col_tier": "Tier", "col_served": "Served by", "col_status": "Status", "not_measured": "not measured",
        "reproduce": "Reproduce: scripts/night_eval.py (quality) and scripts/mvp_e2e_smoke.py (metering path).",
        "live": "Live gateway figures", "n_a": "n/a",
    },
    "fr": {
        "title": "Benchmarks HiveMind : résultats mesurés uniquement",
        "intro": "Seuls les chiffres réellement mesurés sont affichés, avec taille d'échantillon, date, matériel et méthode. Tout le reste indique « non mesuré ». HiveMind ne se compare pas à GPT-4o, Claude ou Gemini : aucune comparaison de ce type n'a été réalisée.",
        "measured": "Résultats mesurés",
        "none": "Aucun résultat mesuré n'est publié dans cette version.",
        "unmeasured": "Paliers sans résultat mesuré",
        "unmeasured_note": "Ces paliers sont servis par le modèle du commerce indiqué. Aucun chiffre de qualité ou de latence n'est publié pour eux.",
        "col_tier": "Palier", "col_served": "Servi par", "col_status": "Statut", "not_measured": "non mesuré",
        "reproduce": "Reproduire : scripts/night_eval.py (qualité) et scripts/mvp_e2e_smoke.py (chemin de facturation).",
        "live": "Chiffres en direct de la passerelle", "n_a": "n/d",
    },
    "es": {
        "title": "Benchmarks de HiveMind: solo resultados medidos",
        "intro": "Solo se muestran cifras realmente medidas, con tamaño de muestra, fecha, hardware y método. Todo lo demás indica 'no medido'. HiveMind no se compara con GPT-4o, Claude ni Gemini: no se ha realizado tal comparación.",
        "measured": "Resultados medidos",
        "none": "No se publica ningún resultado medido en esta versión.",
        "unmeasured": "Niveles sin resultado medido",
        "unmeasured_note": "Estos niveles los sirve el modelo comercial indicado. No se publica cifra de calidad ni de latencia para ellos.",
        "col_tier": "Nivel", "col_served": "Servido por", "col_status": "Estado", "not_measured": "no medido",
        "reproduce": "Reproducir: scripts/night_eval.py (calidad) y scripts/mvp_e2e_smoke.py (ruta de facturación).",
        "live": "Cifras en vivo de la pasarela", "n_a": "n/d",
    },
    "de": {
        "title": "HiveMind-Benchmarks: nur gemessene Ergebnisse",
        "intro": "Gezeigt werden nur tatsächlich gemessene Werte, jeweils mit Stichprobengröße, Datum, Hardware und Methode. Alles andere heißt 'nicht gemessen'. HiveMind vergleicht sich nicht mit GPT-4o, Claude oder Gemini: ein solcher Vergleich wurde nicht durchgeführt.",
        "measured": "Gemessene Ergebnisse",
        "none": "In dieser Version ist kein gemessenes Ergebnis veröffentlicht.",
        "unmeasured": "Stufen ohne gemessenes Ergebnis",
        "unmeasured_note": "Diese Stufen werden vom angegebenen Standardmodell bedient. Für sie wird keine Qualitäts- oder Latenzangabe veröffentlicht.",
        "col_tier": "Stufe", "col_served": "Bedient von", "col_status": "Status", "not_measured": "nicht gemessen",
        "reproduce": "Nachvollziehen: scripts/night_eval.py (Qualität) und scripts/mvp_e2e_smoke.py (Abrechnungspfad).",
        "live": "Live-Zahlen des Gateways", "n_a": "k. A.",
    },
    "zh": {
        "title": "HiveMind 基准：仅显示实测结果",
        "intro": "仅显示真正测得的数据，并附样本量、日期、硬件和方法。其余一律标注“未测量”。HiveMind 不与 GPT-4o、Claude 或 Gemini 比较：从未进行过此类比较。",
        "measured": "实测结果",
        "none": "此版本未发布任何实测结果。",
        "unmeasured": "尚无实测结果的档位",
        "unmeasured_note": "这些档位由所示的现成模型提供服务，未发布质量或延迟数据。",
        "col_tier": "档位", "col_served": "提供模型", "col_status": "状态", "not_measured": "未测量",
        "reproduce": "复现：scripts/night_eval.py（质量）与 scripts/mvp_e2e_smoke.py（计费路径）。",
        "live": "网关实时数据", "n_a": "无",
    },
}


def load_measured_results(path: Path | None = None) -> list[dict[str, Any]]:
    """The measured results shipped with the gateway (empty if the file is missing or malformed: nothing is invented)."""
    try:
        data = json.loads((path or MEASURED_RESULTS_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    results = data.get("results") if isinstance(data, dict) else None
    return [r for r in results if isinstance(r, dict) and r.get("id") and r.get("model_id") and r.get("source")] if isinstance(results, list) else []


def _unmeasured_tiers(measured: list[dict[str, Any]]) -> list[dict[str, str]]:
    from hivemind.router.catalog import TIERS

    done = {str(r["model_id"]).removeprefix("hivemind/") for r in measured}
    return [{"model_id": f"hivemind/{t.tier_id}", "served_by": t.ollama_tag, "status": "not measured"} for t in TIERS if t.tier_id not in done]


def _fmt_result(r: dict[str, Any]) -> str:
    """One readable line of what was measured (never more precise than the data)."""
    if "accuracy" in r and "n" in r:
        lo, hi = (r.get("wilson95") or [None, None])[:2]
        ci = f", 95% Wilson {lo:.3f}-{hi:.3f}" if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) else ""
        return f"{r.get('correct')}/{r['n']} correct = {float(r['accuracy']):.3f}{ci}"
    if "latency_seconds" in r:
        lat = ", ".join(f"{float(x):.2f} s" for x in r["latency_seconds"])
        return f"end-to-end latency over n={r.get('n')}: {lat}"
    return "see source"


def benchmarks_payload(live_telemetry: dict[str, Any] | None = None) -> dict[str, Any]:
    """JSON for ``GET /v1/hivemind/benchmarks``: measured results and the list of unmeasured tiers; no comparison with other vendors."""
    if live_telemetry is not None and not isinstance(live_telemetry, dict):
        raise ValueError("live_telemetry must be a dict or None")
    measured = load_measured_results()
    return {
        "object": "hivemind.benchmarks",
        "status": "measured_only",
        "notice": POC_NOTICE["en"],
        "supported_languages": list(SUPPORTED_BENCHMARK_LANGS),
        "measured_results": measured,
        "tiers_without_measured_result": _unmeasured_tiers(measured),
        "comparison_with_other_vendors": "none: no such comparison has been run",
        "live_telemetry": dict(live_telemetry) if live_telemetry else {},
        "reproduce": ["scripts/night_eval.py", "scripts/mvp_e2e_smoke.py"],
    }


def _live_line(live_telemetry: dict[str, Any] | None, t: dict[str, str]) -> str:
    lt = live_telemetry or {}
    accounts = (lt.get("accounts") or {}).get("total_registered_accounts")
    models = (lt.get("models") or {}).get("live_models_count")
    show = lambda v: html.escape(str(v)) if v is not None else html.escape(t["n_a"])  # noqa: E731
    return f"accounts: {show(accounts)} | live models: {show(models)}"


def render_benchmarks_page(lang: str = "en", nonce: str = "", live_telemetry: dict[str, Any] | None = None) -> str:
    """The HTML page. Every number on it comes from ``measured_results.json`` or from the live telemetry argument."""
    if not isinstance(lang, str) or not isinstance(nonce, str):
        raise ValueError("lang and nonce must be strings")
    if live_telemetry is not None and not isinstance(live_telemetry, dict):
        raise ValueError("live_telemetry must be a dict or None")
    lang = lang if lang in SUPPORTED_BENCHMARK_LANGS else "en"
    t = _TEXT[lang]
    measured = load_measured_results()
    cards = []
    for r in measured:
        caveats = f"<p class='cv'>{html.escape(str(r['caveats']))}</p>" if r.get("caveats") else ""
        control = f"<p>{html.escape(str(r['control']))}</p>" if r.get("control") else ""
        baseline = f"<p>{html.escape(str(r['baseline']))}</p>" if r.get("baseline") else ""
        cards.append(
            f"<article class='card'><h3>{html.escape(str(r['model_id']))}</h3><p><strong>{html.escape(_fmt_result(r))}</strong></p>"
            f"<p>{html.escape(str(r.get('what', '')))}</p>{baseline}{control}"
            f"<p class='meta'>{html.escape(str(r.get('date', '')))} · {html.escape(str(r.get('hardware', '')))} · {html.escape(str(r.get('method', '')))}"
            f" · <code>{html.escape(str(r['source']))}</code></p>{caveats}</article>"
        )
    rows = "".join(
        f"<tr><td>{html.escape(x['model_id'])}</td><td><code>{html.escape(x['served_by'])}</code></td><td>{html.escape(t['not_measured'])}</td></tr>"
        for x in _unmeasured_tiers(measured)
    )
    css = (
        "body{font-family:system-ui,sans-serif;margin:0;background:#0d0e11;color:#e4e2e6;line-height:1.55}main{max-width:960px;margin:0 auto;padding:1.5rem}"
        ".poc{background:#2a2208;color:#ffc943;padding:.7rem 1rem;font-size:.9rem}.card{border:1px solid #333;border-radius:8px;padding:1rem;margin:1rem 0;background:#14161a}"
        ".meta,.cv{font-size:.82rem;color:#9aa0a6}table{width:100%;border-collapse:collapse}td,th{border-bottom:1px solid #2a2d33;padding:.4rem;text-align:left;font-size:.88rem}"
        "a{color:#38bdf8}code{font-size:.82rem}"
    )
    return (
        f"<!doctype html><html lang='{lang}'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(t['title'])}</title><style nonce='{html.escape(nonce)}'>{css}</style></head><body>"
        f"<p class='poc' role='note' id='poc-notice'><strong>&#9888;</strong> {html.escape(POC_NOTICE[lang])}</p>"
        "<nav style='padding:.6rem 1.5rem'><a href='/'>Home</a> · <a href='/portal'>Portal</a> · <a href='/join'>Join</a> · "
        "<a href='/signup'>Sign up</a> · <a href='/manifesto'>Manifesto</a></nav><main>"
        f"<h1>{html.escape(t['title'])}</h1><p>{html.escape(t['intro'])}</p>"
        f"<h2>{html.escape(t['measured'])}</h2>{''.join(cards) if cards else '<p>' + html.escape(t['none']) + '</p>'}"
        f"<h2>{html.escape(t['unmeasured'])}</h2><p>{html.escape(t['unmeasured_note'])}</p>"
        f"<table><thead><tr><th>{html.escape(t['col_tier'])}</th><th>{html.escape(t['col_served'])}</th><th>{html.escape(t['col_status'])}</th></tr></thead><tbody>{rows}</tbody></table>"
        f"<h2>{html.escape(t['live'])}</h2><p id='benchmarks-live-telemetry' data-source='/v1/hivemind/live-telemetry'>{_live_line(live_telemetry, t)}</p>"
        f"<p class='meta'>{html.escape(t['reproduce'])}</p></main>"
        f"<script nonce='{html.escape(nonce)}'>{_LIVE_JS}</script></body></html>"
    )


# Refreshes the live line from the gateway's own telemetry endpoint; values are written with textContent (never innerHTML).
_LIVE_JS = (
    "(function(){var el=document.getElementById('benchmarks-live-telemetry');if(!el||!window.fetch)return;"
    "fetch('/v1/hivemind/live-telemetry',{cache:'no-store'}).then(function(r){return r.ok?r.json():null}).then(function(d){"
    "if(!d)return;var a=d.accounts&&d.accounts.total_registered_accounts;var m=d.models&&d.models.live_models_count;"
    "el.textContent='accounts: '+(a==null?'n/a':a)+' | live models: '+(m==null?'n/a':m);}).catch(function(){});})();"
)


def render_hivemind_capacity_infographic_svg(live_telemetry: dict[str, Any] | None = None) -> str:
    """A small SVG listing what is measured. It contains no capacity or speed claim that is not in ``measured_results.json``."""
    measured = load_measured_results()
    lines = ["PROJECT HIVEMIND: measured results only", POC_NOTICE["en"][:96]]
    for r in measured:
        lines.append(f"{r['model_id']}: {_fmt_result(r)} ({r.get('date', '')})")
    lines.append(f"{len(_unmeasured_tiers(measured))} catalog tiers are not measured yet; no comparison with other vendors has been run.")
    height = 40 + 26 * len(lines)
    body = "".join(f"<text x='24' y='{40 + 26 * i}' fill='#e4e2e6' font-family='sans-serif' font-size='15'>{html.escape(s)}</text>" for i, s in enumerate(lines))
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='1100' height='{height}' viewBox='0 0 1100 {height}'><rect width='1100' height='{height}' fill='#0d0e11'/>{body}</svg>"
