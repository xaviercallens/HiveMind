# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, statistical, security, DOM-execution, and gateway integration tests for SLICE-05 (/arena-noel)."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import random
import re
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import pytest
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.blind_ab_arena import (
    _ARENA_NOEL_JS,
    ARENA_TASKS_BY_ID,
    CURATED_ARENA_TASKS,
    DEFAULT_GATEWAY_URL,
    PERSONAL_PC_RULE_EN,
    PERSONAL_PC_RULE_FR,
    Z_WILSON_95,
    BlindArenaCollector,
    _format_count,
    _format_wilson,
    _format_win_ratio,
    blind_ab_arena_csp,
    build_blind_task_cards,
    pick_arena_language,
    render_blind_ab_arena_page,
    sign_blind_ballot,
    verify_blind_ballot,
    wilson_95_lower_bound,
)
from hivemind.router.gateway import LINK_TYPO_PAGES, create_router_app, link_typo_target
from hivemind.security.pr_security_scanner import PRSecurityScanner

REPO_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

from gwaya.ast_audit import ZeroStubAudit  # noqa: E402

ARENA_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "blind_ab_arena.py"
GATEWAY_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "gateway.py"
STATIC_FIGURE_RE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
TEST_STORE_KEY = b"slice05-arena-noel-router-key32!"
TEST_BALLOT_KEY = b"slice05-arena-ballot-hmac-key32!"

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_slice05",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)


class _StrictDomInspector(HTMLParser):
    """Verify CSP-clean attributes (zero inline style= or on*= handlers) and capture element attrs/text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.inline_styles: list[str] = []
        self.inline_handlers: list[str] = []
        self.style_nonces: list[str] = []
        self.script_nonces: list[str] = []
        self.by_id_text: dict[str, str] = {}
        self.by_id_attrs: dict[str, dict[str, str]] = {}
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
            self.by_id_attrs[el_id] = amap
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


def _collect_all_dict_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_all_dict_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_all_dict_keys(item))
    return found


def test_compliance_headers_zero_vram_ast_allowed_hosts_and_curated_tasks() -> None:
    source = ARENA_PATH.read_text(encoding="utf-8")
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    assert expected_header in source[:300]
    assert expected_header in GATEWAY_PATH.read_text(encoding="utf-8")[:300]
    assert expected_header in Path(__file__).resolve().read_text(encoding="utf-8")[:300]

    # Zero typed-in performance/traffic figures
    assert STATIC_FIGURE_RE.findall(source) == []

    # Zero unsafe HTML sinks
    for sink in ("inner" + "HTML", "outer" + "HTML", "insertAdjacent" + "HTML", "document.write"):
        assert sink not in source

    # Zero GPU / ML / live network inference imports in blind_ab_arena.py
    tree = ast.parse(source, filename="blind_ab_arena.py")
    forbidden_gpu_or_http_pkgs = {
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "httpx",
        "requests",
        "aiohttp",
    }
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_roots.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint(forbidden_gpu_or_http_pkgs)

    # Zero lateral movement and strictly allowed hosts
    assert lateral_guard.audit_source_for_lateral_movement(source, "src/hivemind/router/blind_ab_arena.py") == []
    for match in re.findall(r"https?://[^\s'\"<>]+", source):
        host = (urlparse(match).hostname or "").lower()
        assert (
            host.endswith(".run.app")
            or host.endswith(".googleapis.com")
            or host == "datasets-server.huggingface.co"
            or host in {"127.0.0.1", "localhost"}
        ), f"Disallowed host {host!r} in {match!r}"

    # ZeroStubAudit + 5-Layer PRSecurityScanner clean on source & curated Rust/Lean snippets
    assert ZeroStubAudit.audit_python_code(source).is_clean is True
    for task in CURATED_ARENA_TASKS:
        if task["domain"] == "lean4":
            assert ZeroStubAudit.audit_lean_code(task["holiday_checkpoint_output"]).is_clean is True
            assert ZeroStubAudit.audit_lean_code(task["base_checkpoint_output"]).is_clean is True
        elif task["domain"] == "rust":
            assert ZeroStubAudit.audit_rust_code(task["holiday_checkpoint_output"]).is_clean is True
            assert ZeroStubAudit.audit_rust_code(task["base_checkpoint_output"]).is_clean is True

    scanner = PRSecurityScanner()
    for rel, path in (
        ("src/hivemind/router/blind_ab_arena.py", ARENA_PATH),
        ("tests/test_blind_ab_arena.py", Path(__file__).resolve()),
    ):
        txt = path.read_text(encoding="utf-8")
        assert scanner.scan_python_ast_security(rel, txt) == []
        assert scanner.scan_text_for_payment_solicitation(txt, rel) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(txt, rel) == []
        assert scanner.scan_unicode_trojan_source(txt, rel) == []

    # Verify all 4 required curated domains are present and distinct
    domains = [t["domain"] for t in CURATED_ARENA_TASKS]
    assert domains == ["python", "rust", "lean4", "exoplanet"]
    for t in CURATED_ARENA_TASKS:
        assert t["holiday_checkpoint_output"] != t["base_checkpoint_output"]


def test_csp_and_language_negotiation() -> None:
    csp = blind_ab_arena_csp("ArenaNonce_2026+ok=")
    assert "default-src 'none'" in csp
    assert "style-src 'nonce-ArenaNonce_2026+ok='" in csp
    assert "script-src 'nonce-ArenaNonce_2026+ok='" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp
    assert "'unsafe-eval'" not in csp

    for bad_nonce in ("", "short", "bad nonce", "nonce;script-src *", None, True, "n" * 85):
        with pytest.raises(ValueError, match="nonce"):
            blind_ab_arena_csp(bad_nonce)  # type: ignore[arg-type]

    assert pick_arena_language("fr", "en-US,en;q=0.9", default="en") == "fr"
    assert pick_arena_language("en_US", "fr-FR,fr;q=0.9", default="fr") == "en"
    assert pick_arena_language("de", "en-GB;q=0.6, fr-FR;q=0.9", default="en") == "fr"
    assert pick_arena_language(None, "fr;q=invalid, en;q=0.8", default="fr") == "en"
    assert pick_arena_language(None, "de-DE,es;q=0.8", default="fr") == "fr"
    assert pick_arena_language(None, "de-DE,es;q=0.8", default="en") == "en"
    assert pick_arena_language(None, None, default="fr") == "fr"
    assert pick_arena_language(None, "fr;q=0", default="en") == "en"

    with pytest.raises(ValueError, match="query_lang"):
        pick_arena_language(False, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="accept_language"):
        pick_arena_language(None, 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default"):
        pick_arena_language(None, None, default="de")
    with pytest.raises(ValueError, match="default"):
        pick_arena_language(None, None, default=True)  # type: ignore[arg-type]


def test_wilson_95_lower_bound_math_and_validation() -> None:
    assert wilson_95_lower_bound(0, 0) == 0.0
    assert wilson_95_lower_bound(0, 15) == 0.0

    # Exact closed-form check for wins=8, trials=10
    p = 0.8
    n = 10.0
    z = Z_WILSON_95
    z2 = z * z
    expected_8_10 = (p + z2 / (2.0 * n) - z * math.sqrt((p * (1.0 - p) / n) + z2 / (4.0 * n * n))) / (1.0 + z2 / n)
    assert wilson_95_lower_bound(8, 10) == pytest.approx(expected_8_10, rel=1e-12)
    assert 0.49 < wilson_95_lower_bound(8, 10) < 0.80

    # Conservative small-sample property & asymptotic convergence
    lb_10 = wilson_95_lower_bound(10, 10)
    lb_100 = wilson_95_lower_bound(100, 100)
    assert 0.72 < lb_10 < lb_100 < 1.0

    for bad_args in (
        (True, 5, 1.96),
        (3, False, 1.96),
        (2.5, 5, 1.96),
        (-1, 5, 1.96),
        (6, 5, 1.96),
        (3, -1, 1.96),
        (3, 5, 0.0),
        (3, 5, -1.0),
        (3, 5, float("nan")),
        (3, 5, float("inf")),
        (3, 5, True),
    ):
        with pytest.raises(ValueError):
            wilson_95_lower_bound(*bad_args)  # type: ignore[arg-type]


def test_blind_label_randomization_and_hmac_ballot_verification() -> None:
    # Sign & verify both A and B sides with explicit and random salts
    tok_a = sign_blind_ballot("python-sliding-median", "A", TEST_BALLOT_KEY, salt="0123456789ab")
    tok_b = sign_blind_ballot("python-sliding-median", "B", TEST_BALLOT_KEY, salt="0123456789ab")
    assert tok_a != tok_b
    assert verify_blind_ballot(tok_a, "python-sliding-median", TEST_BALLOT_KEY) == "A"
    assert verify_blind_ballot(tok_b, "python-sliding-median", TEST_BALLOT_KEY) == "B"

    # Cross-task replay is rejected
    with pytest.raises(ValueError, match="signature"):
        verify_blind_ballot(tok_a, "rust-frame-decoder", TEST_BALLOT_KEY)

    # Tampered signature or masked bit is rejected
    salt_part, bit_part, mac_part = tok_a.split(".")
    flipped_bit = "0" if bit_part == "1" else "1"
    with pytest.raises(ValueError, match="signature"):
        verify_blind_ballot(f"{salt_part}.{flipped_bit}.{mac_part}", "python-sliding-median", TEST_BALLOT_KEY)

    # Validation errors on sign_blind_ballot and verify_blind_ballot
    with pytest.raises(ValueError, match="task_id"):
        sign_blind_ballot("unknown-task", "A", TEST_BALLOT_KEY)
    with pytest.raises(ValueError, match="holiday_side"):
        sign_blind_ballot("python-sliding-median", "C", TEST_BALLOT_KEY)
    with pytest.raises(ValueError, match="secret"):
        sign_blind_ballot("python-sliding-median", "A", b"short")
    with pytest.raises(ValueError, match="salt"):
        sign_blind_ballot("python-sliding-median", "A", TEST_BALLOT_KEY, salt="BAD_SALT!")
    with pytest.raises(ValueError, match="task_id"):
        verify_blind_ballot(tok_a, True, TEST_BALLOT_KEY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ballot_token"):
        verify_blind_ballot(123, "python-sliding-median", TEST_BALLOT_KEY)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="malformed ballot_token"):
        verify_blind_ballot("only-two.parts", "python-sliding-median", TEST_BALLOT_KEY)
    with pytest.raises(ValueError, match="malformed ballot_token fields"):
        verify_blind_ballot("0123456789ab.9." + ("a" * 24), "python-sliding-median", TEST_BALLOT_KEY)

    # Randomized blind placement over 64 seeded draws: every task visits both A and B, never leaking holiday_side
    rng = random.Random(20261225)
    a_counts = {tid: 0 for tid in ARENA_TASKS_BY_ID}
    for _ in range(64):
        cards = build_blind_task_cards(lang="en", rng=rng, secret=TEST_BALLOT_KEY)
        assert len(cards) == 4
        for card in cards:
            tid = card["task_id"]
            spec = ARENA_TASKS_BY_ID[tid]
            assert "holiday_side" not in card
            assert {card["candidate_a"], card["candidate_b"]} == {
                spec["holiday_checkpoint_output"],
                spec["base_checkpoint_output"],
            }
            recovered = verify_blind_ballot(card["ballot_token"], tid, TEST_BALLOT_KEY)
            if recovered == "A":
                assert card["candidate_a"] == spec["holiday_checkpoint_output"]
                assert card["candidate_b"] == spec["base_checkpoint_output"]
                a_counts[tid] += 1
            else:
                assert card["candidate_b"] == spec["holiday_checkpoint_output"]
                assert card["candidate_a"] == spec["base_checkpoint_output"]
    for tid, count_a in a_counts.items():
        assert 16 <= count_a <= 48, f"Expected balanced blind A/B randomization for {tid}, got {count_a}/64"

    with pytest.raises(ValueError, match="lang"):
        build_blind_task_cards(lang=True)  # type: ignore[arg-type]


def test_collector_vote_recording_summary_math_validation_and_zero_score_key() -> None:
    with pytest.raises(ValueError, match="max_votes"):
        BlindArenaCollector(max_votes=0)
    with pytest.raises(ValueError, match="max_votes"):
        BlindArenaCollector(max_votes=True)  # type: ignore[arg-type]

    collector = BlindArenaCollector(secret=TEST_BALLOT_KEY, max_votes=10)
    empty = collector.summary()
    assert empty["object"] == "hivemind.blind_ab_arena.summary"
    assert empty["zero_vram_inference"] is True
    assert empty["total_votes"] == 0
    assert empty["decisive_votes"] == 0
    assert empty["holiday_wins"] == 0
    assert empty["baseline_wins"] == 0
    assert empty["ties"] == 0
    assert empty["win_ratio"] is None
    assert empty["wilson_95_lower_bound"] == 0.0
    assert empty["by_profile"] == {"dev": 0, "gamer": 0}
    assert "score" not in _collect_all_dict_keys(empty)

    # Ties-only check: decisive_votes == 0 -> win_ratio is None & wilson_95_lower_bound == 0.0
    tie_collector = BlindArenaCollector(secret=TEST_BALLOT_KEY)
    tie_res = tie_collector.record_vote(task_id="python-sliding-median", chosen_side="tie", profile="dev")
    assert tie_res["total_votes"] == 1
    assert tie_res["decisive_votes"] == 0
    assert tie_res["ties"] == 1
    assert tie_res["win_ratio"] is None
    assert tie_res["wilson_95_lower_bound"] == 0.0
    assert "score" not in _collect_all_dict_keys(tie_res)

    # Record 5 votes on main collector: 3 holiday wins, 1 baseline win, 1 tie
    b_py_b = collector.issue_ballot("python-sliding-median", "B", salt="aabbccddeeff")
    r1 = collector.record_vote(
        task_id="python-sliding-median",
        chosen_side="B",
        profile="dev",
        ballot_token=b_py_b,
    )
    assert r1["last_vote"]["preferred_checkpoint"] == "holiday_checkpoint"
    assert r1["last_vote"]["revealed_holiday_side"] == "B"

    # Vote via explicit holiday_side='A' choosing 'a' (case-normalized) -> holiday win #2
    r2 = collector.record_vote(
        task_id="rust-frame-decoder",
        chosen_side="a",
        profile="gamer",
        holiday_side="A",
    )
    assert r2["last_vote"]["preferred_checkpoint"] == "holiday_checkpoint"

    # Stateless vote without ballot_token on lean4-energy-descent (default_holiday_side='A'), choosing 'A' -> holiday win #3
    r3 = collector.record_vote(
        task_id="lean4-energy-descent",
        chosen_side="A",
        profile="dev",
    )
    assert r3["last_vote"]["preferred_checkpoint"] == "holiday_checkpoint"

    # Baseline win on exoplanet-transit-bls (default_holiday_side='B'), choosing 'A' -> baseline win #1
    r4 = collector.record_vote(
        task_id="exoplanet-transit-bls",
        chosen_side="A",
        profile="dev",
    )
    assert r4["last_vote"]["preferred_checkpoint"] == "baseline_checkpoint"

    # Tie vote -> tie #1
    r5 = collector.record_vote(
        task_id="exoplanet-transit-bls",
        chosen_side="tie",
        profile="gamer",
    )
    assert r5["recorded"] is True
    assert "score" not in _collect_all_dict_keys(r5)

    sm = collector.summary()
    assert "score" not in _collect_all_dict_keys(sm)
    assert sm["total_votes"] == 5
    assert sm["decisive_votes"] == 4
    assert sm["holiday_wins"] == 3
    assert sm["baseline_wins"] == 1
    assert sm["ties"] == 1
    assert sm["chosen_side_counts"] == {"A": 3, "B": 1, "tie": 1}
    assert sm["by_profile"] == {"dev": 3, "gamer": 2}
    assert sm["win_ratio"] == 0.75
    assert sm["wilson_95_lower_bound"] == round(wilson_95_lower_bound(3, 4), 4)
    assert sm["by_task"]["exoplanet-transit-bls"]["total_votes"] == 2
    assert sm["by_task"]["exoplanet-transit-bls"]["baseline_wins"] == 1
    assert sm["by_task"]["exoplanet-transit-bls"]["ties"] == 1

    # Validation errors on record_vote
    for bad_kwargs in (
        {"task_id": True, "chosen_side": "A", "profile": "dev"},
        {"task_id": "missing-task", "chosen_side": "A", "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": True, "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": "C", "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": "A", "profile": False},
        {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "corporate"},
        {
            "task_id": "python-sliding-median",
            "chosen_side": "A",
            "profile": "dev",
            "ballot_token": b_py_b,
            "holiday_side": "B",
        },
        {
            "task_id": "python-sliding-median",
            "chosen_side": "A",
            "profile": "dev",
            "holiday_side": "Z",
        },
    ):
        with pytest.raises(ValueError):
            collector.record_vote(**bad_kwargs)  # type: ignore[arg-type]


def test_render_blind_ab_arena_page_bilingual_dom_and_formatters() -> None:
    assert _format_win_ratio(None) == "n/a"
    assert _format_win_ratio(True) == "n/a"
    assert _format_win_ratio(float("nan")) == "n/a"
    assert _format_win_ratio(0.875) == "87.5%"
    assert _format_wilson(None) == "n/a"
    assert _format_wilson(False) == "n/a"
    assert _format_wilson("bad") == "n/a"
    assert _format_wilson(0.61234) == "0.6123"
    assert _format_count(None) == "n/a"
    assert _format_count(True) == "n/a"
    assert _format_count(2.5) == "n/a"
    assert _format_count(9.0) == "9"

    # French default with None summary -> 'n/a'
    fr_html = render_blind_ab_arena_page("fr", "NonceFR_12345678", rng=random.Random(7))
    assert STATIC_FIGURE_RE.findall(fr_html) == []
    assert '<html lang="fr">' in fr_html
    dom_fr = _StrictDomInspector()
    dom_fr.feed(fr_html)
    assert dom_fr.inline_styles == []
    assert dom_fr.inline_handlers == []
    assert dom_fr.style_nonces == ["NonceFR_12345678"]
    assert dom_fr.script_nonces == ["NonceFR_12345678"]
    assert dom_fr.by_id_text["personalPcComplianceBanner"] == PERSONAL_PC_RULE_FR
    assert dom_fr.by_id_text["arenaTotalVotes"] == "n/a"
    assert dom_fr.by_id_text["arenaWinRatio"] == "n/a"
    assert dom_fr.by_id_text["arenaWilsonLower"] == "n/a"
    for tid in ARENA_TASKS_BY_ID:
        assert f"arenaCard-{tid}" in dom_fr.by_id_attrs
        assert dom_fr.by_id_attrs[f"arenaCard-{tid}"]["data-ballot"]

    # English version with live summary
    collector = BlindArenaCollector(secret=TEST_BALLOT_KEY)
    collector.record_vote(task_id="python-sliding-median", chosen_side="A", profile="dev", holiday_side="A")
    collector.record_vote(task_id="rust-frame-decoder", chosen_side="B", profile="gamer", holiday_side="B")
    en_html = render_blind_ab_arena_page(
        "en",
        "NonceEN_87654321",
        gateway_url=DEFAULT_GATEWAY_URL,
        path="/arena-noel",
        arena_summary=collector.summary(),
        cards=collector.build_cards(lang="en", rng=random.Random(11)),
    )
    assert STATIC_FIGURE_RE.findall(en_html) == []
    assert '<html lang="en">' in en_html
    dom_en = _StrictDomInspector()
    dom_en.feed(en_html)
    assert dom_en.by_id_text["personalPcComplianceBanner"] == PERSONAL_PC_RULE_EN
    assert dom_en.by_id_text["arenaTotalVotes"] == "2"
    assert dom_en.by_id_text["arenaRecordBreakdown"] == "2 / 0 / 0"
    assert dom_en.by_id_text["arenaWinRatio"] == "100.0%"
    assert dom_en.by_id_text["arenaProfileBreakdown"] == "dev: 1 | gamer: 1"

    # Validation errors on render_blind_ab_arena_page
    with pytest.raises(ValueError, match="lang"):
        render_blind_ab_arena_page(False, "NonceOK_12345678")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="gateway_url"):
        render_blind_ab_arena_page("fr", "NonceOK_12345678", gateway_url="")
    with pytest.raises(ValueError, match="gateway_url"):
        render_blind_ab_arena_page("fr", "NonceOK_12345678", gateway_url="https://bad host.run.app")
    with pytest.raises(ValueError, match="gateway_url"):
        render_blind_ab_arena_page("fr", "NonceOK_12345678", gateway_url="ftp://files.example.org")
    with pytest.raises(ValueError, match="gateway_url"):
        render_blind_ab_arena_page("fr", "NonceOK_12345678", gateway_url="https://u:p@example.run.app")
    with pytest.raises(ValueError, match="path"):
        render_blind_ab_arena_page("fr", "NonceOK_12345678", path="arena-noel")
    with pytest.raises(ValueError, match="arena_summary"):
        render_blind_ab_arena_page("fr", "NonceOK_12345678", arena_summary="invalid")  # type: ignore[arg-type]


@pytest.mark.parametrize("lang_mode", ["fr", "en"])
def test_arena_js_controller_uses_textcontent_only_for_votes_and_summary(lang_mode: str) -> None:
    forbidden_prop = "inner" + "HTML"
    harness = f"""
const elements = {{}};
function makeEl(id) {{
  const listeners = {{}};
  const attrs = {{}};
  const el = {{
    id,
    _text: '',
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
  'arenaNoelApp',
  'arenaTotalVotes', 'arenaRecordBreakdown', 'arenaWinRatio', 'arenaWilsonLower', 'arenaProfileBreakdown',
  'refreshArenaBtn', 'arenaRefreshNote',
  'profileBtn-dev', 'profileBtn-gamer', 'arenaProfileStatus'
].forEach((id) => {{ elements[id] = makeEl(id); }});

[
  'python-sliding-median',
  'rust-frame-decoder',
  'lean4-energy-descent',
  'exoplanet-transit-bls'
].forEach((tid) => {{
  elements['arenaCard-' + tid] = makeEl('arenaCard-' + tid);
  elements['arenaCard-' + tid].setAttribute('data-ballot', '0123456789ab.1.' + 'a'.repeat(24));
  elements['voteBtnA-' + tid] = makeEl('voteBtnA-' + tid);
  elements['voteBtnB-' + tid] = makeEl('voteBtnB-' + tid);
  elements['voteBtnTie-' + tid] = makeEl('voteBtnTie-' + tid);
  elements['voteStatus-' + tid] = makeEl('voteStatus-' + tid);
}});

elements.arenaNoelApp.setAttribute('data-lang', {json.dumps(lang_mode)});
globalThis.document = {{ getElementById: (id) => elements[id] || null }};

const postedVotes = [];
let fetchMode = 'ok';
globalThis.fetch = async (url, opts = {{}}) => {{
  if (fetchMode === 'net_err') throw new Error('offline');
  if (fetchMode === 'http_err') return {{ ok: false, status: 503, async json() {{ return {{}}; }} }};
  if (url === '/v1/hivemind/arena/summary') {{
    if (fetchMode === 'nulls') {{
      return {{
        ok: true,
        status: 200,
        async json() {{ return {{ by_profile: {{}} }}; }},
      }};
    }}
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          total_votes: 8,
          holiday_wins: 6,
          baseline_wins: 1,
          ties: 1,
          win_ratio: 0.8571,
          wilson_95_lower_bound: 0.4869,
          by_profile: {{ dev: 5, gamer: 3 }},
        }};
      }},
    }};
  }}
  if (url === '/v1/hivemind/arena/vote') {{
    const body = JSON.parse(opts.body);
    postedVotes.push(body);
    return {{
      ok: true,
      status: 200,
      async json() {{
        return {{
          total_votes: postedVotes.length,
          holiday_wins: postedVotes.length,
          baseline_wins: 0,
          ties: 0,
          win_ratio: 1.0,
          wilson_95_lower_bound: 0.2065,
          by_profile: {{ dev: 0, gamer: postedVotes.length }},
          last_vote: {{
            task_id: body.task_id,
            chosen_side: body.chosen_side,
            profile: body.profile,
            revealed_holiday_side: 'B',
            preferred_checkpoint: 'holiday_checkpoint',
          }},
        }};
      }},
    }};
  }}
  throw new Error('Unexpected fetch url: ' + url);
}};
"""
    driver = r"""
(async () => {
  await elements['profileBtn-gamer'].trigger('click');
  const gamerPressed = elements['profileBtn-gamer'].getAttribute('aria-pressed');
  const devPressed = elements['profileBtn-dev'].getAttribute('aria-pressed');

  await elements['voteBtnB-python-sliding-median'].trigger('click');
  const voteStatusOk = elements['voteStatus-python-sliding-median'].textContent;
  const voteBtnBPressed = elements['voteBtnB-python-sliding-median'].getAttribute('aria-pressed');

  await elements.refreshArenaBtn.trigger('click');
  const syncedTotal = elements.arenaTotalVotes.textContent;
  const syncedRatio = elements.arenaWinRatio.textContent;
  const syncedWilson = elements.arenaWilsonLower.textContent;

  fetchMode = 'nulls';
  await elements.refreshArenaBtn.trigger('click');
  const nullRatio = elements.arenaWinRatio.textContent;

  fetchMode = 'http_err';
  await elements['voteBtnTie-rust-frame-decoder'].trigger('click');
  const voteStatusErr = elements['voteStatus-rust-frame-decoder'].textContent;

  fetchMode = 'net_err';
  await elements.refreshArenaBtn.trigger('click');
  const netErrNote = elements.arenaRefreshNote.textContent;

  console.log(JSON.stringify({
    gamerPressed,
    devPressed,
    postedVotes,
    voteStatusOk,
    voteBtnBPressed,
    syncedTotal,
    syncedRatio,
    syncedWilson,
    nullRatio,
    voteStatusErr,
    netErrNote,
  }));
})().catch((err) => { console.error(err); process.exit(1); });
"""
    proc = subprocess.run(
        ["node", "-e", f"{harness}\n{_ARENA_NOEL_JS}\n{driver}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["gamerPressed"] == "true"
    assert out["devPressed"] == "false"
    assert len(out["postedVotes"]) == 1
    assert out["postedVotes"][0]["task_id"] == "python-sliding-median"
    assert out["postedVotes"][0]["chosen_side"] == "B"
    assert out["postedVotes"][0]["profile"] == "gamer"
    assert out["postedVotes"][0]["ballot_token"].startswith("0123456789ab.")
    assert "B" in out["voteStatusOk"]
    assert out["voteBtnBPressed"] == "true"
    assert out["syncedTotal"] == "8"
    assert out["syncedRatio"] == "85.7%"
    assert out["syncedWilson"] == "0.4869"
    assert out["nullRatio"] == "n/a"
    assert out["voteStatusErr"] != ""
    assert out["netErrNote"] != ""


def test_gateway_arena_noel_endpoints_zero_gpu_egress_and_rate_limit(tmp_path: Path) -> None:
    assert "/arena-noel" in LINK_TYPO_PAGES
    assert link_typo_target("/arena-noel)") == "/arena-noel"

    upstream_calls: list[str] = []

    def _forbid_upstream(req: httpx.Request) -> httpx.Response:
        upstream_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_gpu_call"})

    now = [1_700_000_000.0]
    store = RouterStore(tmp_path / "arena_noel.sqlite3", secret=TEST_STORE_KEY)
    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_upstream)),
        clock=lambda: now[0],
        open_registration=True,
    )
    client = TestClient(app)

    # 1. Link typo redirect
    r_typo = client.get("/arena-noel)?lang=en", follow_redirects=False)
    assert r_typo.status_code == 307
    assert r_typo.headers["location"] == "/arena-noel?lang=en"

    # 2. GET / HEAD /arena-noel in FR (default) and EN, with unique per-request CSP nonces
    r_fr = client.get("/arena-noel")
    assert r_fr.status_code == 200
    assert r_fr.headers["content-language"] == "fr"
    assert '<html lang="fr">' in r_fr.text
    csp_fr = r_fr.headers["content-security-policy"]
    nonce_fr = csp_fr.split("'nonce-")[1].split("'")[0]
    assert f'nonce="{nonce_fr}"' in r_fr.text
    assert r_fr.headers["x-frame-options"] == "DENY"
    assert r_fr.headers["x-content-type-options"] == "nosniff"
    assert client.head("/arena-noel").status_code == 200

    r_en = client.get("/arena-noel?lang=en")
    assert r_en.status_code == 200
    assert r_en.headers["content-language"] == "en"
    assert '<html lang="en">' in r_en.text
    nonce_en = r_en.headers["content-security-policy"].split("'nonce-")[1].split("'")[0]
    assert nonce_en != nonce_fr
    assert '<html lang="en">' in client.get("/arena-noel", headers={"Accept-Language": "en-US,en;q=0.9"}).text

    # 3. Initial summary is clean and has zero 'score' keys
    s0 = client.get("/v1/hivemind/arena/summary")
    assert s0.status_code == 200
    assert s0.headers["cache-control"] == "no-store"
    s0_json = s0.json()
    assert s0_json["total_votes"] == 0
    assert s0_json["win_ratio"] is None
    assert s0_json["wilson_95_lower_bound"] == 0.0
    assert "score" not in _collect_all_dict_keys(s0_json)

    # 4. Submit blind vote using real ballot_token extracted from rendered DOM + direct API votes
    dom = _StrictDomInspector()
    dom.feed(r_fr.text)
    py_ballot = dom.by_id_attrs["arenaCard-python-sliding-median"]["data-ballot"]
    py_cand_a = dom.by_id_text["candA-python-sliding-median"]
    py_spec = ARENA_TASKS_BY_ID["python-sliding-median"]
    pick_holiday_side = "A" if py_cand_a == py_spec["holiday_checkpoint_output"] else "B"

    v1 = client.post(
        "/v1/hivemind/arena/vote",
        json={
            "task_id": "python-sliding-median",
            "chosen_side": pick_holiday_side,
            "profile": "dev",
            "ballot_token": py_ballot,
        },
    )
    assert v1.status_code == 200
    v1_json = v1.json()
    assert v1_json["recorded"] is True
    assert v1_json["holiday_wins"] == 1
    assert v1_json["last_vote"]["preferred_checkpoint"] == "holiday_checkpoint"
    assert "score" not in _collect_all_dict_keys(v1_json)

    # Direct stateless vote without ballot_token (rust-frame-decoder default_holiday_side='B')
    v2 = client.post(
        "/v1/hivemind/arena/vote",
        json={"task_id": "rust-frame-decoder", "chosen_side": "B", "profile": "gamer"},
    )
    assert v2.status_code == 200
    v3 = client.post(
        "/v1/hivemind/arena/vote",
        json={"task_id": "lean4-energy-descent", "chosen_side": "B", "profile": "dev"},
    )
    assert v3.status_code == 200
    v4 = client.post(
        "/v1/hivemind/arena/vote",
        json={"task_id": "exoplanet-transit-bls", "chosen_side": "tie", "profile": "gamer"},
    )
    assert v4.status_code == 200

    s1 = client.get("/v1/hivemind/arena/summary").json()
    assert "score" not in _collect_all_dict_keys(s1)
    assert s1["total_votes"] == 4
    assert s1["decisive_votes"] == 3
    assert s1["holiday_wins"] == 2
    assert s1["baseline_wins"] == 1
    assert s1["ties"] == 1
    assert s1["win_ratio"] == round(2 / 3, 4)
    assert s1["wilson_95_lower_bound"] == round(wilson_95_lower_bound(2, 3), 4)
    assert s1["by_profile"] == {"dev": 2, "gamer": 2}

    # 5. Validation errors -> HTTP 400
    for bad_payload in (
        {"task_id": "unknown-task", "chosen_side": "A", "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": "Z", "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "enterprise"},
        {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "dev", "ballot_token": "forged.0.deadbeef"},
        {"task_id": "python-sliding-median", "chosen_side": "A"},
        {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "dev", "email": "pii@example.com"},
    ):
        resp_bad = client.post("/v1/hivemind/arena/vote", json=bad_payload)
        assert resp_bad.status_code == 400

    # Zero GPU / producer upstream calls were made during the entire session
    assert upstream_calls == []

    # 6. Rate limiter triggers 429 when exceeding 60 requests in a 60s window
    now[0] += 120.0
    statuses = [
        client.post(
            "/v1/hivemind/arena/vote",
            json={"task_id": "python-sliding-median", "chosen_side": "A", "profile": "dev"},
        ).status_code
        for _ in range(61)
    ]
    assert statuses[:60] == [200] * 60
    assert statuses[60] == 429
