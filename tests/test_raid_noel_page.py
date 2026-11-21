# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, DOM-execution, and gateway integration tests for SLICE-04 (/raid-noel & /raid)."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import LINK_TYPO_PAGES, create_router_app, link_typo_target
from hivemind.router.raid_noel_page import (
    _RAID_NOEL_JS,
    DEFAULT_GATEWAY_URL,
    FRIENDLY_GUILDS,
    PERSONAL_PC_RULE_EN,
    PERSONAL_PC_RULE_FR,
    _opt_float,
    _opt_int,
    build_raid_noel_status,
    estimate_gpu_electricity_cost,
    pick_raid_language,
    raid_noel_csp,
    render_raid_noel_page,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
RAID_NOEL_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "raid_noel_page.py"
GATEWAY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"
STATIC_FIGURE_RE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
SECRET = b"slice04-raid-noel-test-secret32!"

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_slice04",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)


class _StrictDomInspector(HTMLParser):
    """Verify CSP-clean attributes (no inline style= or on*= handlers) and collect element text by id."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inline_styles: list[str] = []
        self.inline_handlers: list[str] = []
        self.style_nonces: list[str] = []
        self.script_nonces: list[str] = []
        self.by_id_text: dict[str, str] = {}
        self._active_id_stack: list[tuple[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        amap = {k.lower(): (v or "") for k, v in attrs}
        if "style" in amap:
            self.inline_styles.append(tag)
        for k in amap:
            if k.startswith("on"):
                self.inline_handlers.append(f"{tag}.{k}")
        if tag == "style":
            self.style_nonces.append(amap.get("nonce", ""))
        if tag == "script":
            self.script_nonces.append(amap.get("nonce", ""))
        el_id = amap.get("id")
        if el_id:
            self.by_id_text.setdefault(el_id, "")
        if tag not in {"meta", "input", "br", "hr", "img", "link"}:
            self._active_id_stack.append((tag, el_id))

    def handle_data(self, data: str) -> None:
        for _tag, el_id in reversed(self._active_id_stack):
            if el_id is not None:
                self.by_id_text[el_id] = self.by_id_text.get(el_id, "") + data
                break

    def handle_endtag(self, tag: str) -> None:
        for idx in range(len(self._active_id_stack) - 1, -1, -1):
            if self._active_id_stack[idx][0] == tag:
                del self._active_id_stack[idx:]
                break


def test_agpl_header_no_static_telemetry_no_unsafe_dom_and_allowed_hosts() -> None:
    source = RAID_NOEL_PATH.read_text(encoding="utf-8")
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    assert expected_header in source[:300]
    assert expected_header in GATEWAY_PATH.read_text(encoding="utf-8")[:300]

    # Zero static telemetry figures in source
    assert STATIC_FIGURE_RE.findall(source) == []

    # Zero unsafe HTML sinks
    unsafe_sinks = (
        "inner" + "HTML",
        "outer" + "HTML",
        "insertAdjacent" + "HTML",
        "document.write",
    )
    for sink in unsafe_sinks:
        assert sink not in source

    # Lateral movement / allowed hosts check
    assert lateral_guard.audit_source_for_lateral_movement(source, "src/hivemind/router/raid_noel_page.py") == []
    for match in re.findall(r"https?://[^\s'\"<>]+", source):
        host = (urlparse(match).hostname or "").lower()
        assert (
            host.endswith(".run.app")
            or host.endswith(".googleapis.com")
            or host == "datasets-server.huggingface.co"
            or host in {"127.0.0.1", "localhost"}
        ), f"Disallowed host {host!r} in {match!r}"

    # 5-Layer PRSecurityScanner passes with zero findings on both source and test files
    scanner = PRSecurityScanner()
    for rel, path in (
        ("src/hivemind/router/raid_noel_page.py", RAID_NOEL_PATH),
        ("tests/test_raid_noel_page.py", Path(__file__).resolve()),
    ):
        text = path.read_text(encoding="utf-8")
        assert scanner.scan_python_ast_security(rel, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel) == []
        assert scanner.scan_unicode_trojan_source(text, rel) == []


def test_raid_noel_csp_and_language_selection() -> None:
    csp = raid_noel_csp("Nonce_RaidNoel-2026+ok=")
    assert "default-src 'none'" in csp
    assert "style-src 'nonce-Nonce_RaidNoel-2026+ok='" in csp
    assert "script-src 'nonce-Nonce_RaidNoel-2026+ok='" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp
    assert "'unsafe-eval'" not in csp

    for bad_nonce in ("", "short", "bad nonce", "nonce;script-src *", None, True, "a" * 85):
        with pytest.raises(ValueError, match="nonce"):
            raid_noel_csp(bad_nonce)  # type: ignore[arg-type]

    assert pick_raid_language("fr", "en-US,en;q=0.9", default="en") == "fr"
    assert pick_raid_language("en-GB", "fr-FR,fr;q=0.9", default="fr") == "en"
    assert pick_raid_language("es", "en-GB;q=0.7, fr-FR;q=0.9", default="en") == "fr"
    assert pick_raid_language(None, "fr;q=bad, en;q=0.8", default="fr") == "en"
    assert pick_raid_language(None, "de-DE,es;q=0.8", default="fr") == "fr"
    assert pick_raid_language(None, "de-DE,es;q=0.8", default="en") == "en"
    assert pick_raid_language(None, None, default="fr") == "fr"
    assert pick_raid_language(None, "fr;q=0", default="en") == "en"

    with pytest.raises(ValueError, match="query_lang"):
        pick_raid_language(True, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="accept_language"):
        pick_raid_language(None, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default"):
        pick_raid_language(None, None, default="es")
    with pytest.raises(ValueError, match="default"):
        pick_raid_language(None, None, default=False)  # type: ignore[arg-type]


def test_electricity_estimator_math_and_validation() -> None:
    est = estimate_gpu_electricity_cost(120.0, 4.0, 0.25)
    assert est == {
        "gpu_watts": 120.0,
        "hours": 4.0,
        "kwh_tariff_eur": 0.25,
        "energy_wh": 480.0,
        "energy_kwh": 0.48,
        "estimated_eur": 0.12,
    }

    zero_est = estimate_gpu_electricity_cost(0, 0, 0)
    assert zero_est["energy_wh"] == 0.0
    assert zero_est["estimated_eur"] == 0.0

    desktop_est = estimate_gpu_electricity_cost(220, 5.5, 0.2516)
    assert desktop_est["energy_wh"] == 1210.0
    assert desktop_est["energy_kwh"] == 1.21
    assert desktop_est["estimated_eur"] == round(1.21 * 0.2516, 4)

    for kwargs in (
        {"gpu_watts": True, "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": "120", "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": float("nan"), "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": float("inf"), "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": -5.0, "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 2500.0, "hours": 4.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 120.0, "hours": -1.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 120.0, "hours": 9000.0, "kwh_tariff_eur": 0.25},
        {"gpu_watts": 120.0, "hours": 4.0, "kwh_tariff_eur": -0.1},
        {"gpu_watts": 120.0, "hours": 4.0, "kwh_tariff_eur": 12.0},
    ):
        with pytest.raises(ValueError):
            estimate_gpu_electricity_cost(**kwargs)  # type: ignore[arg-type]


def test_build_raid_noel_status_honesty_and_validation() -> None:
    assert _opt_int(None) is None
    assert _opt_int(True) is None
    assert _opt_int(3.5) is None
    assert _opt_int("4") is None
    assert _opt_int(7.0) == 7
    assert _opt_float(None) is None
    assert _opt_float(False) is None
    assert _opt_float("bad") is None
    assert _opt_float(float("nan")) is None
    assert _opt_float(4.66666, decimals=2) == 4.67

    empty = build_raid_noel_status(live_telemetry=None, feedback_summary=None)
    assert empty["object"] == "hivemind.raid_noel.status"
    assert empty["compliance"]["personal_pc_only"] is True
    assert empty["compliance"]["monetary_cost_eur"] == "0.00 EUR"
    assert empty["compliance"]["monetary_cost_usd"] == "0.00 USD"
    assert empty["swarm"]["telemetry_status"] == "UNAVAILABLE"
    for key in (
        "active_trainers",
        "registered_trainer_nodes",
        "registered_trainer_accounts",
        "live_producers",
        "registered_producers",
        "registered_accounts",
        "active_accounts_24h",
        "accepted_opentrainer_rounds",
        "live_models",
    ):
        assert empty["swarm"][key] is None
    assert empty["onboarding_feedback"] == {
        "total_responses": None,
        "mean_ease_rating": None,
        "zero_game_lag_ratio": None,
        "by_profile": {"dev": None, "gamer": None},
    }
    guild_labels = [g["label"] for g in empty["friendly_guilds"]]
    assert guild_labels == ["Team Python", "Team Rust/C++", "Team Steam RPG", "Team Night Owl"]
    assert all(g["symbolic_fee"] == "0.00 EUR" for g in empty["friendly_guilds"])

    populated = build_raid_noel_status(
        live_telemetry={
            "status": "LIVE_GCP_PRODUCTION",
            "accounts": {"total_registered_accounts": 14, "active_accounts_24h": 6},
            "producers": {"live_hiveminders": 3, "total_hiveminders": 5},
            "opentrainer": {
                "active_trainer_nodes": 2,
                "total_registered_nodes": 4,
                "registered_trainer_accounts_count": 3,
                "accepted_rounds_count": 9,
            },
            "models": {"live_models_count": 4},
        },
        feedback_summary={
            "total_responses": 5,
            "mean_ease_rating": 4.8,
            "zero_game_lag_ratio": 1.0,
            "by_profile": {"dev": 3, "gamer": 2},
        },
    )
    assert populated["swarm"]["telemetry_status"] == "LIVE_GCP_PRODUCTION"
    assert populated["swarm"]["active_trainers"] == 2
    assert populated["swarm"]["registered_trainer_accounts"] == 3
    assert populated["swarm"]["live_producers"] == 3
    assert populated["swarm"]["registered_accounts"] == 14
    assert populated["swarm"]["accepted_opentrainer_rounds"] == 9
    assert populated["onboarding_feedback"]["total_responses"] == 5
    assert populated["onboarding_feedback"]["mean_ease_rating"] == 4.8
    assert populated["onboarding_feedback"]["zero_game_lag_ratio"] == 1.0

    with pytest.raises(ValueError, match="live_telemetry"):
        build_raid_noel_status(live_telemetry="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="feedback_summary"):
        build_raid_noel_status(feedback_summary=["bad"])  # type: ignore[arg-type]


def test_render_raid_noel_page_bilingual_structure_and_validation() -> None:
    # 1. French default with None telemetry -> 'n/a' everywhere on scoreboard & feedback
    fr_empty = render_raid_noel_page("fr", "NonceFR_12345678", path="/raid-noel")
    assert STATIC_FIGURE_RE.findall(fr_empty) == []
    assert '<html lang="fr">' in fr_empty
    assert "Merci aux Premiers Pionniers HiveMind !" in fr_empty
    for g in FRIENDLY_GUILDS:
        assert g["label"] in fr_empty

    dom_fr = _StrictDomInspector()
    dom_fr.feed(fr_empty)
    assert dom_fr.by_id_text["personalPcComplianceBanner"] == PERSONAL_PC_RULE_FR
    assert dom_fr.inline_styles == []
    assert dom_fr.inline_handlers == []
    assert dom_fr.style_nonces == ["NonceFR_12345678"]
    assert dom_fr.script_nonces == ["NonceFR_12345678"]
    for stat_id in (
        "raidLiveTrainers",
        "raidLiveProducers",
        "raidRegisteredAccounts",
        "raidAcceptedRounds",
        "raidFeedbackResponses",
        "raidFeedbackEase",
        "raidFeedbackZeroLag",
    ):
        assert dom_fr.by_id_text[stat_id] == "n/a"
    assert dom_fr.by_id_text["calcResultWh"] == "480.0 Wh (0.480 kWh)"
    assert dom_fr.by_id_text["calcResultEur"] == "0.12 EUR"

    # 2. English version with live telemetry + feedback summary
    en_live = render_raid_noel_page(
        "en",
        "NonceEN_87654321",
        gateway_url=DEFAULT_GATEWAY_URL,
        path="/raid",
        live_telemetry={
            "status": "LIVE_GCP_PRODUCTION",
            "accounts": {"total_registered_accounts": 21, "active_accounts_24h": 8},
            "producers": {"live_hiveminders": 4, "total_hiveminders": 7},
            "opentrainer": {
                "active_trainer_nodes": 3,
                "total_registered_nodes": 5,
                "registered_trainer_accounts_count": 4,
                "accepted_rounds_count": 12,
            },
            "models": {"live_models_count": 2},
        },
        feedback_summary={
            "total_responses": 4,
            "mean_ease_rating": 4.75,
            "zero_game_lag_ratio": 0.75,
            "by_profile": {"dev": 2, "gamer": 2},
        },
    )
    assert STATIC_FIGURE_RE.findall(en_live) == []
    assert '<html lang="en">' in en_live
    assert "Thank You HiveMind Pioneers!" in en_live
    assert PERSONAL_PC_RULE_EN in en_live
    dom_en = _StrictDomInspector()
    dom_en.feed(en_live)
    assert dom_en.by_id_text["raidLiveTrainers"] == "3"
    assert dom_en.by_id_text["raidLiveProducers"] == "4"
    assert dom_en.by_id_text["raidRegisteredAccounts"] == "21"
    assert dom_en.by_id_text["raidAcceptedRounds"] == "12"
    assert dom_en.by_id_text["raidFeedbackResponses"] == "4"
    assert dom_en.by_id_text["raidFeedbackEase"] == "4.75 / 5"
    assert dom_en.by_id_text["raidFeedbackZeroLag"] == "75.0%"
    assert dom_en.by_id_text["raidFeedbackProfiles"] == "dev: 2 | gamer: 2"

    # 3. Validation errors
    with pytest.raises(ValueError, match="lang"):
        render_raid_noel_page(True, "NonceOK_12345678")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gateway_url"):
        render_raid_noel_page("fr", "NonceOK_12345678", gateway_url="")
    with pytest.raises(ValueError, match="gateway_url"):
        render_raid_noel_page("fr", "NonceOK_12345678", gateway_url="https://bad host.run.app")
    with pytest.raises(ValueError, match="gateway_url"):
        render_raid_noel_page("fr", "NonceOK_12345678", gateway_url="ftp://files.example.org")
    with pytest.raises(ValueError, match="gateway_url"):
        render_raid_noel_page("fr", "NonceOK_12345678", gateway_url="https://u:p@example.run.app")
    with pytest.raises(ValueError, match="path"):
        render_raid_noel_page("fr", "NonceOK_12345678", path="raid-noel")
    with pytest.raises(ValueError, match="path"):
        render_raid_noel_page("fr", "NonceOK_12345678", path=False)  # type: ignore[arg-type]


@pytest.mark.parametrize("lang_mode", ["fr", "en"])
def test_raid_noel_js_calculator_guilds_and_live_refresh_textcontent_only(lang_mode: str) -> None:
    forbidden_prop = "inner" + "HTML"
    harness = f"""
const elements = {{}};
function makeEl(id, initVal = '') {{
  const listeners = {{}};
  const attrs = {{}};
  const el = {{
    id,
    _text: '',
    value: initVal,
    className: '',
    getAttribute(k) {{ return Object.prototype.hasOwnProperty.call(attrs, k) ? attrs[k] : null; }},
    setAttribute(k, v) {{ attrs[k] = String(v); }},
    addEventListener(type, fn) {{
      if (!listeners[type]) listeners[type] = [];
      listeners[type].push(fn);
    }},
    async trigger(type) {{
      for (const fn of (listeners[type] || [])) {{
        await fn({{ preventDefault() {{}} }});
      }}
    }},
  }};
  Object.defineProperty(el, 'textContent', {{
    get() {{ return this._text; }},
    set(v) {{ this._text = String(v); }},
  }});
  Object.defineProperty(el, {json.dumps(forbidden_prop)}, {{
    get() {{ throw new Error('Forbidden HTML getter'); }},
    set() {{ throw new Error('Forbidden HTML setter'); }},
  }});
  return el;
}}

[
  'raidNoelApp',
  'raidLiveTrainers', 'raidRegisteredTrainers',
  'raidLiveProducers', 'raidRegisteredProducers',
  'raidRegisteredAccounts', 'raidActiveAccounts24h',
  'raidAcceptedRounds', 'raidLiveModels',
  'refreshRaidBtn', 'raidLiveRefreshNote',
  'guildBtn-team-python', 'guildBtn-team-rust-cpp', 'guildBtn-team-steam-rpg', 'guildBtn-team-night-owl',
  'guildCheckinStatus',
  'calcPresetEco', 'calcPresetWhisper', 'calcPresetDesktop',
  'calcWatts', 'calcHours', 'calcTariff', 'calcComputeBtn',
  'calcResultWh', 'calcResultEur', 'calcResultNote',
  'raidFeedbackResponses', 'raidFeedbackEase', 'raidFeedbackZeroLag', 'raidFeedbackProfiles'
].forEach((id) => {{ elements[id] = makeEl(id); }});

elements.raidNoelApp.setAttribute('data-lang', {json.dumps(lang_mode)});
elements.calcWatts.value = '120';
elements.calcHours.value = '4';
elements.calcTariff.value = '0.25';

const store = new Map([['hivemind_holiday_guild', 'team-night-owl']]);
globalThis.window = {{
  localStorage: {{
    getItem: (k) => (store.has(k) ? store.get(k) : null),
    setItem: (k, v) => {{ store.set(k, String(v)); }},
  }},
}};
globalThis.document = {{
  getElementById: (id) => elements[id] || null,
}};

let fetchMode = 'ok';
globalThis.fetch = async (url, opts = {{}}) => {{
  if (url !== '/v1/hivemind/raid/status') throw new Error('Unexpected fetch url: ' + url);
  if (fetchMode === 'net_err') throw new Error('offline');
  if (fetchMode === 'http_err') return {{ ok: false, status: 503, async json() {{ return {{}}; }} }};
  if (fetchMode === 'nulls') {{
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{ swarm: {{}}, onboarding_feedback: {{ by_profile: {{}} }} }};
      }},
    }};
  }}
  return {{
    ok: true,
    status: 200,
    async json() {{
      return {{
        swarm: {{
          active_trainers: 5,
          registered_trainer_accounts: 8,
          live_producers: 3,
          registered_producers: 6,
          registered_accounts: 30,
          active_accounts_24h: 11,
          accepted_opentrainer_rounds: 17,
          live_models: 3,
        }},
        onboarding_feedback: {{
          total_responses: 10,
          mean_ease_rating: 4.9,
          zero_game_lag_ratio: 1.0,
          by_profile: {{ dev: 6, gamer: 4 }},
        }},
      }};
    }},
  }};
}};
"""
    driver = r"""
(async () => {
  const restoredGuildStatus = elements.guildCheckinStatus.textContent;
  const initWh = elements.calcResultWh.textContent;
  const initEur = elements.calcResultEur.textContent;

  // Presets
  await elements.calcPresetEco.trigger('click');
  const ecoWh = elements.calcResultWh.textContent;
  await elements.calcPresetDesktop.trigger('click');
  const deskWh = elements.calcResultWh.textContent;
  await elements.calcPresetWhisper.trigger('click');
  const whisperWh = elements.calcResultWh.textContent;

  // Custom calculation: 150 W * 2 h * 0.30 EUR/kWh = 300.0 Wh & 0.09 EUR
  elements.calcWatts.value = '150';
  elements.calcHours.value = '2';
  elements.calcTariff.value = '0.30';
  await elements.calcWatts.trigger('input');
  const customWh = elements.calcResultWh.textContent;
  const customEur = elements.calcResultEur.textContent;

  // Out-of-range calculation -> 'n/a'
  elements.calcWatts.value = '-10';
  await elements.calcComputeBtn.trigger('click');
  const badWh = elements.calcResultWh.textContent;
  const badEur = elements.calcResultEur.textContent;

  // Guild check-in selection
  await elements['guildBtn-team-rust-cpp'].trigger('click');
  const rustStatus = elements.guildCheckinStatus.textContent;
  const rustPressed = elements['guildBtn-team-rust-cpp'].getAttribute('aria-pressed');
  const owlPressed = elements['guildBtn-team-night-owl'].getAttribute('aria-pressed');
  const savedGuild = store.get('hivemind_holiday_guild');

  // Live refresh success
  await elements.refreshRaidBtn.trigger('click');
  const syncedTrainers = elements.raidLiveTrainers.textContent;
  const syncedRounds = elements.raidAcceptedRounds.textContent;
  const syncedEase = elements.raidFeedbackEase.textContent;
  const syncedLag = elements.raidFeedbackZeroLag.textContent;

  // Live refresh with null values -> 'n/a'
  fetchMode = 'nulls';
  await elements.refreshRaidBtn.trigger('click');
  const nullTrainers = elements.raidLiveTrainers.textContent;
  const nullEase = elements.raidFeedbackEase.textContent;

  // HTTP error & network error fallbacks
  fetchMode = 'http_err';
  await elements.refreshRaidBtn.trigger('click');
  const httpErrNote = elements.raidLiveRefreshNote.textContent;

  fetchMode = 'net_err';
  await elements.refreshRaidBtn.trigger('click');
  const netErrNote = elements.raidLiveRefreshNote.textContent;

  console.log(JSON.stringify({
    restoredGuildStatus,
    initWh,
    initEur,
    ecoWh,
    deskWh,
    whisperWh,
    customWh,
    customEur,
    badWh,
    badEur,
    rustStatus,
    rustPressed,
    owlPressed,
    savedGuild,
    syncedTrainers,
    syncedRounds,
    syncedEase,
    syncedLag,
    nullTrainers,
    nullEase,
    httpErrNote,
    netErrNote,
  }));
})().catch((err) => { console.error(err); process.exit(1); });
"""
    proc = subprocess.run(
        ["node", "-e", f"{harness}\n{_RAID_NOEL_JS}\n{driver}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "Team Night Owl" in out["restoredGuildStatus"]
    assert out["initWh"] == "480.0 Wh (0.480 kWh)"
    assert out["initEur"] == "0.12 EUR"
    assert out["ecoWh"] == "180.0 Wh (0.180 kWh)"
    assert out["deskWh"] == "880.0 Wh (0.880 kWh)"
    assert out["whisperWh"] == "480.0 Wh (0.480 kWh)"
    assert out["customWh"] == "300.0 Wh (0.300 kWh)"
    assert out["customEur"] == "0.09 EUR"
    assert out["badWh"] == "n/a"
    assert out["badEur"] == "n/a"
    assert "Team Rust/C++" in out["rustStatus"]
    assert out["rustPressed"] == "true"
    assert out["owlPressed"] == "false"
    assert out["savedGuild"] == "team-rust-cpp"
    assert out["syncedTrainers"] == "5"
    assert out["syncedRounds"] == "17"
    assert out["syncedEase"] == "4.90 / 5"
    assert out["syncedLag"] == "100.0%"
    assert out["nullTrainers"] == "n/a"
    assert out["nullEase"] == "n/a"
    assert out["httpErrNote"] == out["netErrNote"]


def test_gateway_raid_noel_raid_status_and_link_typos_end_to_end(tmp_path: Path) -> None:
    assert "/raid-noel" in LINK_TYPO_PAGES
    assert "/raid" in LINK_TYPO_PAGES
    assert link_typo_target("/raid-noel)") == "/raid-noel"
    assert link_typo_target("/raid)") == "/raid"

    store = RouterStore(tmp_path / "raid_noel.sqlite3", secret=SECRET)
    client = TestClient(create_router_app(store, open_registration=True, allow_private_producers=True))

    # 1. Link-typo redirects work on the live gateway
    r_typo_fr = client.get("/raid-noel)?lang=fr", follow_redirects=False)
    assert r_typo_fr.status_code == 307
    assert r_typo_fr.headers["location"] == "/raid-noel?lang=fr"

    r_typo_en = client.get("/raid]", follow_redirects=False)
    assert r_typo_en.status_code == 307
    assert r_typo_en.headers["location"] == "/raid"

    # 2. /raid-noel defaults to French, /raid defaults to English; HEAD succeeds on both
    r_fr = client.get("/raid-noel")
    assert r_fr.status_code == 200
    assert r_fr.headers["content-language"] == "fr"
    assert '<html lang="fr">' in r_fr.text
    dom_gw_fr = _StrictDomInspector()
    dom_gw_fr.feed(r_fr.text)
    assert dom_gw_fr.by_id_text["personalPcComplianceBanner"] == PERSONAL_PC_RULE_FR
    csp_fr = r_fr.headers["content-security-policy"]
    assert "default-src 'none'" in csp_fr
    nonce_fr = csp_fr.split("'nonce-")[1].split("'")[0]
    assert f'nonce="{nonce_fr}"' in r_fr.text
    assert r_fr.headers["x-frame-options"] == "DENY"
    assert r_fr.headers["x-content-type-options"] == "nosniff"
    assert client.head("/raid-noel").status_code == 200

    r_en = client.get("/raid")
    assert r_en.status_code == 200
    assert r_en.headers["content-language"] == "en"
    assert '<html lang="en">' in r_en.text
    assert "Thank You HiveMind Pioneers!" in r_en.text
    assert PERSONAL_PC_RULE_EN in r_en.text
    csp_en = r_en.headers["content-security-policy"]
    nonce_en = csp_en.split("'nonce-")[1].split("'")[0]
    assert nonce_en != nonce_fr
    assert f'nonce="{nonce_en}"' in r_en.text
    assert client.head("/raid").status_code == 200

    # Language query overrides work both ways
    assert '<html lang="en">' in client.get("/raid-noel?lang=en").text
    assert '<html lang="fr">' in client.get("/raid?lang=fr").text
    assert '<html lang="fr">' in client.get("/raid", headers={"Accept-Language": "fr-FR,fr;q=0.9"}).text

    # 3. Initial /v1/hivemind/raid/status reflects empty store & zero feedback honestly
    r_status_0 = client.get("/v1/hivemind/raid/status")
    assert r_status_0.status_code == 200
    assert r_status_0.headers["cache-control"] == "no-store"
    s0 = r_status_0.json()
    assert s0["object"] == "hivemind.raid_noel.status"
    assert s0["swarm"]["registered_accounts"] == 0
    assert s0["swarm"]["accepted_opentrainer_rounds"] == 0
    assert s0["onboarding_feedback"]["total_responses"] == 0
    assert s0["onboarding_feedback"]["mean_ease_rating"] is None

    # 4. Register trainer + prosumer accounts and submit onboarding feedback -> live readings update!
    assert client.post("/v1/hivemind/register", json={"handle": "trainer_owl", "role": "trainer"}).status_code in (200, 201)
    assert client.post("/v1/hivemind/register", json={"handle": "gamer_fox", "role": "prosumer"}).status_code in (200, 201)
    assert (
        client.post(
            "/v1/hivemind/onboarding/feedback",
            json={"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "linux"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/v1/hivemind/onboarding/feedback",
            json={"profile": "gamer", "ease_rating": 4, "zero_game_lag": True, "platform": "windows11"},
        ).status_code
        == 200
    )

    s1 = client.get("/v1/hivemind/raid/status").json()
    assert s1["swarm"]["registered_accounts"] == 2
    assert s1["swarm"]["registered_trainer_accounts"] == 1
    assert s1["onboarding_feedback"]["total_responses"] == 2
    assert s1["onboarding_feedback"]["mean_ease_rating"] == 4.5
    assert s1["onboarding_feedback"]["zero_game_lag_ratio"] == 1.0
    assert s1["onboarding_feedback"]["by_profile"] == {"dev": 1, "gamer": 1}

    dom_after = _StrictDomInspector()
    dom_after.feed(client.get("/raid-noel").text)
    assert dom_after.by_id_text["raidRegisteredAccounts"] == "2"
    assert dom_after.by_id_text["raidRegisteredTrainers"] == "1"
    assert dom_after.by_id_text["raidFeedbackResponses"] == "2"
    assert dom_after.by_id_text["raidFeedbackEase"] == "4.50 / 5"
    assert dom_after.by_id_text["raidFeedbackZeroLag"] == "100.0%"

    # 5. Pioneer archive route /crowdgpt-raid remains intact
    assert client.get("/crowdgpt-raid?lang=fr").status_code == 200
