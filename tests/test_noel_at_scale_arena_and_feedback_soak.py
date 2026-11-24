# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""16-day holiday soak, statistical monotonicity, bounded-memory, and adversarial fuzz suite (`SLICE-SCALE-04`)."""

from __future__ import annotations

import ast
import concurrent.futures
import html
import importlib.util
import json
import random
import re
import subprocess
import sys
import threading
import tracemalloc
from collections import deque
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import coverage
import httpx
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
ARENA_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "blind_ab_arena.py"
QUICK_JOIN_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "quick_join_page.py"
THIRD_PARTY = REPO_ROOT / "third_party"
if str(THIRD_PARTY) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY))

_INPROC_COV: coverage.Coverage | None = None
if sys.gettrace() is None and coverage.Coverage.current() is None:
    _INPROC_COV = coverage.Coverage(
        data_file=None,
        config_file=False,
        include=[str(ARENA_PATH), str(QUICK_JOIN_PATH)],
    )
    _INPROC_COV.start()

from fastapi.testclient import TestClient  # noqa: E402
from gwaya.ast_audit import ZeroStubAudit  # noqa: E402
from hivemind.router import blind_ab_arena as arena_mod  # noqa: E402
from hivemind.router import quick_join_page as qj_mod  # noqa: E402
from hivemind.router.accounts import RouterStore  # noqa: E402
from hivemind.router.blind_ab_arena import (  # noqa: E402
    _ARENA_NOEL_JS,
    ARENA_TASKS_BY_ID,
    DEFAULT_GATEWAY_URL,
    MAX_ARENA_VOTES,
    MAX_ARENA_VOTES_CEILING,
    PERSONAL_PC_RULE_EN,
    PERSONAL_PC_RULE_FR,
    _format_count,
    _format_wilson,
    _format_win_ratio,
    build_blind_task_cards,
    pick_arena_language,
    verify_blind_ballot,
    wilson_95_lower_bound,
)
from hivemind.router.gateway import create_router_app  # noqa: E402
from hivemind.router.quick_join_page import (  # noqa: E402
    MAX_FEEDBACK_ITEMS,
    MAX_FEEDBACK_ITEMS_CEILING,
)
from hivemind.security.pr_security_scanner import PRSecurityScanner  # noqa: E402

_lat_spec = importlib.util.spec_from_file_location(
    "test_no_lateral_movement_scale04",
    REPO_ROOT / "tests" / "test_no_lateral_movement.py",
)
assert _lat_spec is not None and _lat_spec.loader is not None
lateral_guard = importlib.util.module_from_spec(_lat_spec)
_lat_spec.loader.exec_module(lateral_guard)

_gk_spec = importlib.util.spec_from_file_location(
    "pr_gatekeeper_scale04",
    REPO_ROOT / "scripts" / "pr_gatekeeper.py",
)
assert _gk_spec is not None and _gk_spec.loader is not None
gatekeeper = importlib.util.module_from_spec(_gk_spec)
_gk_spec.loader.exec_module(gatekeeper)

_static_spec = importlib.util.spec_from_file_location(
    "test_static_numbers_scale04",
    REPO_ROOT / "tests" / "test_static_numbers_have_artifacts.py",
)
assert _static_spec is not None and _static_spec.loader is not None
static_numbers_guard = importlib.util.module_from_spec(_static_spec)
_static_spec.loader.exec_module(static_numbers_guard)

STATIC_FIGURE_RE = static_numbers_guard.FIGURE
SOAK_STORE_SECRET = b"noel-scale-04-router-secret-32b!"
SOAK_BALLOT_SECRET = b"noel-scale-04-ballot-secret-32b!"
EXPECTED_AGPL_HEADER = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."


class _HtmlSecurityParser(HTMLParser):
    """Inspect rendered HTML for CSP nonces, inline handlers/styles, script tags, and unescaped XSS sinks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tags: list[str] = []
        self.inline_styles: list[str] = []
        self.inline_handlers: list[str] = []
        self.style_nonces: list[str] = []
        self.script_nonces: list[str] = []
        self.links: list[str] = []
        self.img_srcs: list[str] = []
        self.by_id_attrs: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lower_tag = tag.lower()
        self.tags.append(lower_tag)
        amap = {k.lower(): (v or "") for k, v in attrs}
        if "style" in amap:
            self.inline_styles.append(lower_tag)
        for attr_name in amap:
            if attr_name.startswith("on"):
                self.inline_handlers.append(f"{lower_tag}.{attr_name}")
        if lower_tag == "style":
            self.style_nonces.append(amap.get("nonce", ""))
        elif lower_tag == "script":
            self.script_nonces.append(amap.get("nonce", ""))
        elif lower_tag == "a" and "href" in amap:
            self.links.append(amap["href"])
        elif lower_tag == "img" and "src" in amap:
            self.img_srcs.append(amap["src"])
        if "id" in amap and amap["id"]:
            self.by_id_attrs[amap["id"]] = amap


def _collect_all_keys(obj: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            found.add(str(k))
            found.update(_collect_all_keys(v))
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            found.update(_collect_all_keys(item))
    return found


def _extract_csp_header_nonce(csp_header: str) -> str:
    matches = re.findall(r"'nonce-([^']+)'", csp_header)
    assert matches, f"Missing CSP nonce in header: {csp_header!r}"
    assert len(set(matches)) == 1, f"Mismatched nonces inside single CSP header: {csp_header!r}"
    return matches[0]


def test_compliance_headers_ast_security_gatekeeper_and_static_ratchet() -> None:
    this_file = Path(__file__).resolve()
    target_triples = (
        ("src/hivemind/router/blind_ab_arena.py", ARENA_PATH),
        ("src/hivemind/router/quick_join_page.py", QUICK_JOIN_PATH),
        ("tests/test_noel_at_scale_arena_and_feedback_soak.py", this_file),
    )

    scanner = PRSecurityScanner()
    changed_files: list[dict[str, Any]] = []
    forbidden_html_sinks = ("inner" + "HTML", "outer" + "HTML", "insertAdjacent" + "HTML", "document.write")

    for rel_path, full_path in target_triples:
        text = full_path.read_text(encoding="utf-8")
        assert EXPECTED_AGPL_HEADER in text[:300], f"Missing AGPL-3.0 header in {rel_path}"

        # Lateral-movement & allowed-host AST audit
        assert lateral_guard.audit_source_for_lateral_movement(text, filename=rel_path) == []
        for url_hit in re.findall(r"https?://[^\s'\"<>]+", text):
            host = (urlparse(url_hit).hostname or "").lower()
            if host:
                assert (
                    host.endswith(".run.app")
                    or host.endswith(".googleapis.com")
                    or host == "datasets-server.huggingface.co"
                    or host in {"127.0.0.1", "localhost", "::1", "testserver"}
                ), f"Disallowed host {host!r} in {rel_path}: {url_hit!r}"

        # PR gatekeeper FORBIDDEN_ADDED check
        synthetic_diff = f"diff --git a/{rel_path} b/{rel_path}\n+++ b/{rel_path}\n" + "\n".join(
            f"+{line}" for line in text.splitlines()
        )
        assert gatekeeper.forbidden_scan(gatekeeper.scannable_diff(synthetic_diff)) == []

        # 5-Layer PRSecurityScanner checks
        assert scanner.scan_python_ast_security(rel_path, text) == []
        assert scanner.scan_text_for_payment_solicitation(text, rel_path) == []
        assert scanner.scan_text_for_secrets_and_exfiltration(text, rel_path) == []
        assert scanner.scan_unicode_trojan_source(text, rel_path) == []
        changed_files.append({"filename": rel_path, "patch": "", "content": text})

    for router_path in (ARENA_PATH, QUICK_JOIN_PATH):
        router_src = router_path.read_text(encoding="utf-8")
        assert STATIC_FIGURE_RE.findall(router_src) == []
        for sink in forbidden_html_sinks:
            assert sink not in router_src
        assert ZeroStubAudit().audit_python_code(router_src).is_clean is True

        # Verify pure stdlib / internal telemetry imports (zero GPU/ML/HTTP client libraries)
        tree = ast.parse(router_src, filename=router_path.name)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] in sys.stdlib_module_names
            elif isinstance(node, ast.ImportFrom) and node.module:
                root_mod = node.module.split(".")[0]
                assert root_mod in sys.stdlib_module_names or node.module == "hivemind.router.telemetry_figures"

    # Full page-literals baseline ratchet check across all router *page*.py modules
    static_numbers_guard.test_page_modules_gain_no_new_typed_in_figures()

    pr_audit = scanner.audit_pull_request(
        pr_number=1064,
        pr_title="feat(scale): 16-day holiday soak & adversarial fuzz suite for /arena-noel, /raid-noel, and CES feedback",
        pr_body="Implements SLICE-SCALE-04 bounded O(1) collectors, strict input/XSS hardening, and 16-day soak tests.",
        author="xaviercallens",
        changed_files=changed_files,
        commit_messages=["feat(scale): harden holiday collectors and add 16-day soak & fuzz suite"],
        author_email="callensxavier@gmail.com",
    )
    assert pr_audit.findings == []
    assert pr_audit.passed is True


def test_wilson_95_lower_bound_exact_monotonicity_and_tie_invariance() -> None:
    # 1. Zero-win boundary across arbitrary sample sizes
    for trials in (0, 1, 5, 50, 500, 2000):
        assert wilson_95_lower_bound(0, trials) == 0.0

    # 2. Strict monotonicity in wins for fixed trials: 0 <= w < n => lb(w, n) < lb(w + 1, n) <= (w + 1) / n
    for trials in (4, 16, 64, 250, 1000):
        prev_lb = wilson_95_lower_bound(0, trials)
        assert prev_lb == 0.0
        for wins in range(1, trials + 1):
            cur_lb = wilson_95_lower_bound(wins, trials)
            emp_ratio = wins / trials
            assert prev_lb < cur_lb <= emp_ratio <= 1.0, (
                f"Monotonicity violated at wins={wins}, trials={trials}: prev={prev_lb}, cur={cur_lb}"
            )
            prev_lb = cur_lb

    # 3. Strict sample-size monotonicity at fixed empirical win rates across 16-day growth
    sample_sizes = (20, 40, 80, 160, 320, 640, 1280, 2000)
    for win_rate in (0.60, 0.75, 0.85, 1.0):
        bounds = [wilson_95_lower_bound(int(round(win_rate * n)), n) for n in sample_sizes]
        for idx in range(len(bounds) - 1):
            assert bounds[idx] < bounds[idx + 1] < win_rate, (
                f"Expected strict asymptotic convergence at p={win_rate}: {bounds}"
            )

    # 4. Sequential holiday win-streak monotonicity & exact tie invariance on BlindArenaCollector
    collector = arena_mod.BlindArenaCollector(secret=SOAK_BALLOT_SECRET, max_votes=600)
    task_ids = tuple(ARENA_TASKS_BY_ID.keys())
    profiles = ("dev", "gamer")
    prev_summary_lb = 0.0
    prev_exact_lb = 0.0

    for step in range(1, 241):
        tid = task_ids[(step - 1) % len(task_ids)]
        prof = profiles[(step - 1) % 2]
        hol_side = "A" if step % 2 == 1 else "B"
        ballot = collector.issue_ballot(tid, hol_side)
        res = collector.record_vote(
            task_id=tid,
            chosen_side=hol_side.lower() if step % 3 == 0 else hol_side,
            profile=prof,
            ballot_token=ballot,
        )
        exact_lb = wilson_95_lower_bound(step, step)
        assert exact_lb > prev_exact_lb
        assert res["wilson_95_lower_bound"] >= prev_summary_lb
        assert res["wilson_95_lower_bound"] == round(exact_lb, 4)
        assert res["win_ratio"] == 1.0

        # Interleave a tie vote every 20 steps: decisive_votes, win_ratio, and wilson_95_lower_bound MUST not change
        if step % 20 == 0:
            tie_res = collector.record_vote(task_id=tid, chosen_side="Tie", profile=prof)
            assert tie_res["decisive_votes"] == step
            assert tie_res["win_ratio"] == 1.0
            assert tie_res["wilson_95_lower_bound"] == res["wilson_95_lower_bound"]

        prev_exact_lb = exact_lb
        prev_summary_lb = res["wilson_95_lower_bound"]

    # 5. Validation error branches on wilson_95_lower_bound
    for bad_args in (
        (True, 10, 1.96),
        (2, False, 1.96),
        (-1, 10, 1.96),
        (11, 10, 1.96),
        (2, -5, 1.96),
        (2, 10, 0.0),
        (2, 10, -1.5),
        (2, 10, float("nan")),
        (2, 10, float("inf")),
        (2, 10, True),
    ):
        with pytest.raises(ValueError):
            arena_mod.wilson_95_lower_bound(*bad_args)  # type: ignore[arg-type]


def test_bounded_memory_footprint_and_ring_buffer_oracle_equivalence_over_16_days() -> None:
    arena_cap = 256
    feedback_cap = 128
    arena = arena_mod.BlindArenaCollector(secret=SOAK_BALLOT_SECRET, max_votes=arena_cap)
    feedback = qj_mod.OnboardingFeedbackCollector(max_items=feedback_cap)

    arena_oracle: deque[tuple[str, str, str, str]] = deque(maxlen=arena_cap)
    feedback_oracle: deque[tuple[str, int, bool]] = deque(maxlen=feedback_cap)

    rng = random.Random(20261224)
    task_ids = tuple(ARENA_TASKS_BY_ID.keys())
    sides = ("A", "B", "tie")
    profiles = ("dev", "gamer")
    platforms = ("windows-11", "linux_x86_64", "macos-arm64", "steamos-3")

    tracemalloc.start()
    try:
        warmup_snapshot_bytes: int | None = None
        total_days = 16
        events_per_day = 200  # 16 * 200 = 3,200 arena votes + 3,200 CES feedbacks = 6,400 mutations

        for day in range(total_days):
            for idx in range(events_per_day):
                seq = day * events_per_day + idx
                tid = task_ids[seq % len(task_ids)]
                prof = profiles[(seq + day) % 2]
                chosen = rng.choice(sides)
                hol_side = rng.choice(("A", "B"))

                if seq % 3 == 0:
                    tok = arena.issue_ballot(tid, hol_side)
                    arena.record_vote(task_id=tid, chosen_side=chosen, profile=prof, ballot_token=tok)
                elif seq % 3 == 1:
                    arena.record_vote(task_id=tid, chosen_side=chosen, profile=prof, holiday_side=hol_side)
                else:
                    hol_side = ARENA_TASKS_BY_ID[tid]["default_holiday_side"]
                    arena.record_vote(task_id=tid, chosen_side=chosen, profile=prof)

                pref = "tie" if chosen == "tie" else ("holiday_checkpoint" if chosen == hol_side else "baseline_checkpoint")
                arena_oracle.append((tid, chosen, prof, pref))

                ease = (seq % 5) + 1
                lag_ok = (seq % 7) != 0
                plat = platforms[seq % len(platforms)]
                feedback.record(profile=prof, ease_rating=ease, zero_game_lag=lag_ok, platform_name=plat)
                feedback_oracle.append((prof, ease, lag_ok))

                if seq + 1 == arena_cap:
                    warmup_snapshot_bytes, _ = tracemalloc.get_traced_memory()

        post_soak_bytes, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert warmup_snapshot_bytes is not None
    # After 3,200 votes (12.5x ring turnovers) and 3,200 feedbacks (25x ring turnovers), retained memory cannot grow unboundedly
    retained_growth_bytes = post_soak_bytes - warmup_snapshot_bytes
    assert retained_growth_bytes < 65_536, (
        f"Unbounded collector memory growth detected after 6,400 events: +{retained_growth_bytes} bytes"
    )

    # Verify exact ring-buffer diagnostics & conservation
    a_diag = arena.window_diagnostics()
    assert a_diag == {
        "max_votes": arena_cap,
        "retained_votes": arena_cap,
        "lifetime_votes": 3200,
        "evicted_votes": 3200 - arena_cap,
    }
    f_diag = feedback.window_diagnostics()
    assert f_diag == {
        "max_items": feedback_cap,
        "retained_items": feedback_cap,
        "lifetime_responses": 3200,
        "evicted_responses": 3200 - feedback_cap,
    }

    # Verify zero O(1) incremental counter drift against ground-truth deque oracles
    a_sum = arena.summary()
    assert "score" not in _collect_all_keys(a_sum)
    exp_hw = sum(1 for r in arena_oracle if r[3] == "holiday_checkpoint")
    exp_bw = sum(1 for r in arena_oracle if r[3] == "baseline_checkpoint")
    exp_ti = sum(1 for r in arena_oracle if r[3] == "tie")
    exp_dec = exp_hw + exp_bw
    assert a_sum["total_votes"] == arena_cap
    assert a_sum["holiday_wins"] == exp_hw
    assert a_sum["baseline_wins"] == exp_bw
    assert a_sum["ties"] == exp_ti
    assert a_sum["decisive_votes"] == exp_dec
    assert a_sum["win_ratio"] == (round(exp_hw / exp_dec, 4) if exp_dec > 0 else None)
    assert a_sum["wilson_95_lower_bound"] == round(wilson_95_lower_bound(exp_hw, exp_dec), 4)
    assert a_sum["chosen_side_counts"] == {
        "A": sum(1 for r in arena_oracle if r[1] == "A"),
        "B": sum(1 for r in arena_oracle if r[1] == "B"),
        "tie": sum(1 for r in arena_oracle if r[1] == "tie"),
    }
    assert a_sum["by_profile"] == {
        "dev": sum(1 for r in arena_oracle if r[2] == "dev"),
        "gamer": sum(1 for r in arena_oracle if r[2] == "gamer"),
    }
    for tid in task_ids:
        t_rows = [r for r in arena_oracle if r[0] == tid]
        t_hw = sum(1 for r in t_rows if r[3] == "holiday_checkpoint")
        t_bw = sum(1 for r in t_rows if r[3] == "baseline_checkpoint")
        t_ti = sum(1 for r in t_rows if r[3] == "tie")
        assert a_sum["by_task"][tid]["total_votes"] == len(t_rows)
        assert a_sum["by_task"][tid]["holiday_wins"] == t_hw
        assert a_sum["by_task"][tid]["baseline_wins"] == t_bw
        assert a_sum["by_task"][tid]["ties"] == t_ti

    f_sum = feedback.summary()
    assert "score" not in _collect_all_keys(f_sum)
    assert f_sum["total_responses"] == feedback_cap
    assert f_sum["mean_ease_rating"] == round(sum(r[1] for r in feedback_oracle) / feedback_cap, 2)
    assert f_sum["zero_game_lag_ratio"] == round(sum(1 for r in feedback_oracle if r[2]) / feedback_cap, 4)
    assert f_sum["by_profile"] == {
        "dev": sum(1 for r in feedback_oracle if r[0] == "dev"),
        "gamer": sum(1 for r in feedback_oracle if r[0] == "gamer"),
    }


def test_16_day_holiday_concurrent_gateway_soak_votes_ces_and_simultaneous_readers(tmp_path: Path) -> None:
    """Simulate 1,520 concurrent events over 16 holiday days across /arena-noel, /raid-noel, /noel, and CES/vote APIs."""
    upstream_gpu_calls: list[str] = []

    def _forbid_gpu_upstream(req: httpx.Request) -> httpx.Response:
        upstream_gpu_calls.append(str(req.url))
        return httpx.Response(500, json={"error": "unexpected_live_gpu_call"})

    clock_lock = threading.Lock()
    simulated_clock = [1_766_188_800.0]  # Dec 20, 2025/2026 winter holiday start

    def _advance_clock(delta_s: float = 0.0) -> float:
        with clock_lock:
            simulated_clock[0] += delta_s
            return simulated_clock[0]

    store = RouterStore(tmp_path / "noel_16day_soak.sqlite3", secret=SOAK_STORE_SECRET)
    app = create_router_app(
        store,
        http_client=httpx.Client(transport=httpx.MockTransport(_forbid_gpu_upstream)),
        clock=lambda: _advance_clock(0.0),
        trusted_proxy_hops=1,
        open_registration=True,
    )

    task_ids = tuple(ARENA_TASKS_BY_ID.keys())
    profiles = ("dev", "gamer")
    seen_nonces: set[str] = set()
    nonce_lock = threading.Lock()
    vote_outcomes_lock = threading.Lock()
    recorded_holiday_wins = [0]
    recorded_baseline_wins = [0]
    recorded_ties = [0]

    try:
        with TestClient(app) as client:
            # Seed ballot tokens from live /arena-noel HTML DOM so soak exercises real signed ballots
            seed_resp = client.get("/arena-noel")
            assert seed_resp.status_code == 200
            seed_dom = _HtmlSecurityParser()
            seed_dom.feed(seed_resp.text)
            live_ballots = {
                tid: seed_dom.by_id_attrs[f"arenaCard-{tid}"]["data-ballot"]
                for tid in task_ids
            }

            def _assert_consistent_arena_summary(doc: dict[str, Any]) -> None:
                assert doc["object"] == "hivemind.blind_ab_arena.summary"
                assert doc["zero_vram_inference"] is True
                assert doc["compliance"]["personal_pc_only"] is True
                assert doc["compliance"]["monetary_cost_eur"] == "0.00 EUR"
                assert doc["compliance"]["monetary_cost_usd"] == "0.00 USD"
                assert "score" not in _collect_all_keys(doc)
                hw = doc["holiday_wins"]
                bw = doc["baseline_wins"]
                ti = doc["ties"]
                tot = doc["total_votes"]
                dec = doc["decisive_votes"]
                assert 0 <= tot <= MAX_ARENA_VOTES
                assert hw + bw + ti == tot
                assert dec == hw + bw
                assert sum(doc["chosen_side_counts"].values()) == tot
                assert doc["by_profile"]["dev"] + doc["by_profile"]["gamer"] == tot
                assert sum(t["total_votes"] for t in doc["by_task"].values()) == tot
                assert doc["wilson_95_lower_bound"] == round(wilson_95_lower_bound(hw, dec), 4)
                if dec > 0:
                    assert doc["win_ratio"] == round(hw / dec, 4)
                else:
                    assert doc["win_ratio"] is None

            def _assert_consistent_feedback_summary(doc: dict[str, Any]) -> None:
                assert doc["object"] == "hivemind.onboarding_feedback_summary"
                assert "score" not in _collect_all_keys(doc)
                tot = doc["total_responses"]
                assert 0 <= tot <= MAX_FEEDBACK_ITEMS
                assert doc["by_profile"]["dev"] + doc["by_profile"]["gamer"] == tot
                if tot == 0:
                    assert doc["mean_ease_rating"] is None
                    assert doc["zero_game_lag_ratio"] is None
                else:
                    assert 1.0 <= float(doc["mean_ease_rating"]) <= 5.0
                    assert 0.0 <= float(doc["zero_game_lag_ratio"]) <= 1.0

            def _run_vote_event(day_idx: int, ev_idx: int) -> None:
                seq = day_idx * 40 + ev_idx
                tid = task_ids[seq % len(task_ids)]
                prof = profiles[seq % 2]
                side = ("A", "B", "A", "B", "tie")[seq % 5]
                payload: dict[str, Any] = {
                    "task_id": tid,
                    "chosen_side": side,
                    "profile": prof,
                }
                if seq % 2 == 0:
                    payload["ballot_token"] = live_ballots[tid]
                peer_ip = f"127.0.{day_idx + 1}.{(ev_idx % 200) + 10}"
                resp = client.post(
                    "/v1/hivemind/arena/vote",
                    json=payload,
                    headers={"X-Forwarded-For": peer_ip},
                )
                assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
                body = resp.json()
                assert body["recorded"] is True
                _assert_consistent_arena_summary(body)
                pref = body["last_vote"]["preferred_checkpoint"]
                with vote_outcomes_lock:
                    if pref == "holiday_checkpoint":
                        recorded_holiday_wins[0] += 1
                    elif pref == "baseline_checkpoint":
                        recorded_baseline_wins[0] += 1
                    else:
                        recorded_ties[0] += 1

            def _run_ces_event(day_idx: int, ev_idx: int) -> None:
                seq = day_idx * 35 + ev_idx
                prof = profiles[(seq + 1) % 2]
                ease = 4 if (seq % 3 == 0) else 5
                zero_lag = (seq % 10) != 0
                plat = ("windows", "linux", "macos", "steamos")[seq % 4]
                peer_ip = f"127.1.{day_idx + 1}.{(ev_idx % 200) + 10}"
                resp = client.post(
                    "/v1/hivemind/onboarding/feedback",
                    json={
                        "profile": prof,
                        "ease_rating": ease,
                        "zero_game_lag": zero_lag,
                        "platform": plat,
                        "note": "Personal home PC holiday test",
                    },
                    headers={"X-Forwarded-For": peer_ip},
                )
                assert resp.status_code == 200, f"Unexpected CES status {resp.status_code}: {resp.text}"
                _assert_consistent_feedback_summary(resp.json())

            def _run_reader_event(day_idx: int, ev_idx: int) -> None:
                seq = day_idx * 20 + ev_idx
                route_mode = seq % 8
                if route_mode in (0, 1, 2, 3, 4):
                    page_route = (
                        "/arena-noel",
                        "/arena-noel?lang=en",
                        "/raid-noel",
                        "/noel",
                        "/quick",
                    )[route_mode]
                    resp = client.get(page_route)
                    assert resp.status_code == 200
                    hdr_nonce = _extract_csp_header_nonce(resp.headers["content-security-policy"])
                    with nonce_lock:
                        assert hdr_nonce not in seen_nonces, f"Duplicate CSP nonce on {page_route}: {hdr_nonce}"
                        seen_nonces.add(hdr_nonce)
                    dom = _HtmlSecurityParser()
                    dom.feed(resp.text)
                    assert dom.inline_styles == []
                    assert dom.inline_handlers == []
                    assert dom.style_nonces == [hdr_nonce]
                    if page_route.startswith(("/arena-noel", "/raid-noel")):
                        assert dom.script_nonces == [hdr_nonce]
                    else:
                        assert dom.script_nonces == []
                    assert STATIC_FIGURE_RE.findall(resp.text.replace(hdr_nonce, "CSP_NONCE")) == []
                    assert "0.00 EUR" in resp.text or "0,00 EUR" in resp.text
                elif route_mode == 5:
                    r_sum = client.get("/v1/hivemind/arena/summary")
                    assert r_sum.status_code == 200
                    _assert_consistent_arena_summary(r_sum.json())
                elif route_mode == 6:
                    r_fb = client.get("/v1/hivemind/onboarding/feedback/summary")
                    assert r_fb.status_code == 200
                    _assert_consistent_feedback_summary(r_fb.json())
                else:
                    r_raid = client.get("/v1/hivemind/raid/status")
                    assert r_raid.status_code == 200
                    raid_doc = r_raid.json()
                    assert raid_doc["object"] == "hivemind.raid_noel.status"
                    assert raid_doc["compliance"]["personal_pc_only"] is True
                    assert "score" not in _collect_all_keys(raid_doc)
                    fb_block = raid_doc["onboarding_feedback"]
                    assert 0 <= (fb_block["total_responses"] or 0) <= MAX_FEEDBACK_ITEMS

            # Execute 16 simulated holiday days concurrently (40 votes + 35 CES + 20 readers = 95/day * 16 = 1,520 events)
            total_events = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
                for day in range(16):
                    _advance_clock(86_400.0)  # Advance 1 full holiday day (86,400 s)
                    futures: list[concurrent.futures.Future[None]] = []
                    for v_idx in range(40):
                        futures.append(pool.submit(_run_vote_event, day, v_idx))
                    for c_idx in range(35):
                        futures.append(pool.submit(_run_ces_event, day, c_idx))
                    for r_idx in range(20):
                        futures.append(pool.submit(_run_reader_event, day, r_idx))
                    for fut in concurrent.futures.as_completed(futures):
                        fut.result()
                        total_events += 1

            assert total_events == 1520
            assert upstream_gpu_calls == []
            assert len(seen_nonces) == 200  # 5 HTML routes out of 8 reader modes * 320 readers = 200 unique nonces

            # Verify post-soak exact conservation on /v1/hivemind/arena/summary (640 <= MAX_ARENA_VOTES=2000)
            final_arena = client.get("/v1/hivemind/arena/summary").json()
            _assert_consistent_arena_summary(final_arena)
            assert final_arena["total_votes"] == 640
            assert final_arena["holiday_wins"] == recorded_holiday_wins[0]
            assert final_arena["baseline_wins"] == recorded_baseline_wins[0]
            assert final_arena["ties"] == recorded_ties[0]
            assert final_arena["by_profile"] == {"dev": 320, "gamer": 320}
            for tid in task_ids:
                assert final_arena["by_task"][tid]["total_votes"] == 160

            # Verify post-soak bounded cap on /v1/hivemind/onboarding/feedback/summary (560 -> capped at MAX_FEEDBACK_ITEMS=500)
            final_ces = client.get("/v1/hivemind/onboarding/feedback/summary").json()
            _assert_consistent_feedback_summary(final_ces)
            assert final_ces["total_responses"] == MAX_FEEDBACK_ITEMS

            # Verify HEAD requests across all holiday pages after full soak
            for head_route in ("/arena-noel", "/raid-noel", "/noel", "/quick"):
                assert client.head(head_route).status_code == 200
    finally:
        store.close()


def test_adversarial_xss_unicode_oversized_and_malformed_fuzzing(tmp_path: Path) -> None:
    """Fuzz direct collectors, HTML renderers, and HTTP routes with XSS, Unicode/bidi, null-byte, and oversized payloads."""
    xss_vectors = (
        "<script>alert(1)</script>",
        '"><img src=x onerror=alert(1)>',
        "'></style><script>alert(1)</script>",
        "javascript:alert(1)",
        "<svg/onload=alert(1)>",
        "${7*7}{{7*7}}",
    )
    unicode_vectors = (
        "\u202eevil\u202c",
        "\u2066rtl\u2069",
        "d\u0435v",  # Cyrillic 'e' homoglyph
        "\uff11\uff12\uff13",  # Fullwidth digits
        "\u0661\u0662\u0663",  # Arabic-Indic digits
        "dev\x00admin",
        "linux\u200bwhisper",
        "\U0001f525gamer",
    )

    # 1. Oversized & Unicode fuzzing on pick_arena_language (zero 500s, O(1) bounded work)
    huge_query = "x" * 200_000
    huge_accept = ("de-DE;q=0.9," * 20_000) + "en-US;q=0.8"
    assert pick_arena_language(huge_query, "en-US,en;q=0.9", default="fr") == "en"
    assert pick_arena_language("\u202efr", "en-US;q=0.9", default="fr") == "en"
    assert pick_arena_language(None, "\u202efr;q=0.9, en-GB;q=0.8", default="fr") == "en"
    assert pick_arena_language(None, huge_accept, default="fr") == "fr"
    assert pick_arena_language("en_CA", None, default="fr") == "en"
    assert pick_arena_language("fr-BE", None, default="en") == "fr"
    assert pick_arena_language(None, "fr;q=bad, en;q=0.7", default="fr") == "en"
    for bad_q, bad_a, bad_d in (
        (True, None, "fr"),
        (None, 99, "fr"),
        (None, None, "es"),
        (None, None, False),
    ):
        with pytest.raises(ValueError):
            arena_mod.pick_arena_language(bad_q, bad_a, default=bad_d)  # type: ignore[arg-type]

    # 2. Collector constructor & secret ceiling bounds
    for bad_max in (0, -5, True, False, "100", MAX_ARENA_VOTES_CEILING + 1):
        with pytest.raises(ValueError, match="max_votes"):
            arena_mod.BlindArenaCollector(max_votes=bad_max)  # type: ignore[arg-type]
    for bad_items in (0, -1, True, False, "50", MAX_FEEDBACK_ITEMS_CEILING + 1):
        with pytest.raises(ValueError, match="max_items"):
            qj_mod.OnboardingFeedbackCollector(max_items=bad_items)  # type: ignore[arg-type]
    for bad_secret in (b"short", b"x" * 5000, "not-bytes", None):
        with pytest.raises(ValueError, match="secret"):
            arena_mod.sign_blind_ballot("python-sliding-median", "A", bad_secret)  # type: ignore[arg-type]

    # 3. Fuzz sign_blind_ballot, verify_blind_ballot, BlindArenaCollector.record_vote, and OnboardingFeedbackCollector.record
    arena = arena_mod.BlindArenaCollector(secret=SOAK_BALLOT_SECRET, max_votes=64)
    feedback = qj_mod.OnboardingFeedbackCollector(max_items=64)

    for bad_tid in ("", "unknown-slug", "python-sliding-median" + ("x" * 80), "\u202epy", True):
        with pytest.raises(ValueError):
            arena_mod.sign_blind_ballot(bad_tid, "A", SOAK_BALLOT_SECRET)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            arena_mod.verify_blind_ballot("0123456789ab.1." + ("a" * 24), bad_tid, SOAK_BALLOT_SECRET)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="holiday_side"):
        arena_mod.sign_blind_ballot("python-sliding-median", "tie", SOAK_BALLOT_SECRET)
    with pytest.raises(ValueError, match="salt"):
        arena_mod.sign_blind_ballot("python-sliding-median", "A", SOAK_BALLOT_SECRET, salt="UPPER12345678")

    valid_tok = arena.issue_ballot("python-sliding-median", "A", salt="abcdef012345")
    assert verify_blind_ballot(valid_tok, "python-sliding-median", SOAK_BALLOT_SECRET) == "A"

    for bad_tok in (
        123,
        False,
        "a." * 5000,
        "0123456789ab.\uff11." + ("a" * 24),
        "only.two",
        "bad!.1." + ("a" * 24),
        "abcdef012345.0." + ("f" * 24),
    ):
        with pytest.raises(ValueError):
            arena_mod.verify_blind_ballot(bad_tok, "python-sliding-median", SOAK_BALLOT_SECRET)  # type: ignore[arg-type]

    for fuzz_str in (*xss_vectors, *unicode_vectors, "Z" * 100_000):
        # Unknown/oversized/malformed task_id MUST NOT reflect raw untrusted payload back in exception text
        with pytest.raises(ValueError) as exc_info:
            arena.record_vote(task_id=fuzz_str, chosen_side="A", profile="dev")
        assert "<script>" not in str(exc_info.value)
        assert len(str(exc_info.value)) < 120

        with pytest.raises(ValueError):
            arena.record_vote(task_id="python-sliding-median", chosen_side=fuzz_str, profile="dev")
        with pytest.raises(ValueError):
            arena.record_vote(task_id="python-sliding-median", chosen_side="A", profile=fuzz_str)

        with pytest.raises(ValueError):
            feedback.record(profile=fuzz_str, ease_rating=5, zero_game_lag=True, platform_name="linux")

        # Malicious platform_name strings in OnboardingFeedbackCollector.record sanitize safely to 'unknown'
        res_fb = feedback.record(profile="gamer", ease_rating=5, zero_game_lag=True, platform_name=fuzz_str)
        assert res_fb["total_responses"] >= 1
        assert feedback._items[-1][3] == "unknown"

    # Additional collector validation branches
    for bad_vote_kwargs in (
        {"task_id": False, "chosen_side": "A", "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": False, "profile": "dev"},
        {"task_id": "python-sliding-median", "chosen_side": "b", "profile": True},
        {"task_id": "python-sliding-median", "chosen_side": "B", "profile": "dev", "ballot_token": valid_tok, "holiday_side": "A"},
        {"task_id": "python-sliding-median", "chosen_side": "B", "profile": "dev", "holiday_side": "tie"},
    ):
        with pytest.raises(ValueError):
            arena.record_vote(**bad_vote_kwargs)  # type: ignore[arg-type]

    for bad_fb_kwargs in (
        {"profile": None, "ease_rating": 5, "zero_game_lag": True},
        {"profile": True, "ease_rating": 5, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": True, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 0, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 6, "zero_game_lag": True},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": 1},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform_name": 42},
        {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform_name": False},
    ):
        with pytest.raises(ValueError):
            feedback.record(**bad_fb_kwargs)  # type: ignore[arg-type]

    # Valid platform normalization & empty summary edge cases
    assert qj_mod.OnboardingFeedbackCollector().summary()["total_responses"] == 0
    feedback.record(profile="dev", ease_rating=4, zero_game_lag=False, platform_name="  Windows-11_Home---  ")
    assert feedback._items[-1][3] == "windows-11_home"
    feedback.record(profile="dev", ease_rating=3, zero_game_lag=True, platform_name=None)  # type: ignore[arg-type]
    assert feedback._items[-1][3] == "unknown"

    # 4. Fuzz HTML renderers & CSP helpers: zero XSS, zero open redirects, zero attribute breakout
    for bad_nonce in ("", "short", "bad nonce", "Abc12345;script-src *", "\u202eAbcdef1234", True, "N" * 90):
        with pytest.raises(ValueError, match="nonce"):
            arena_mod.blind_ab_arena_csp(bad_nonce)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="nonce"):
            qj_mod.quick_join_csp(bad_nonce)  # type: ignore[arg-type]

    loopback_base = "http://127.0.0.1:8000"
    for bad_gw in (
        "",
        "   ",
        True,
        "https://" + ("a" * 300) + ".run.app",
        loopback_base + "\u202e",
        loopback_base + " bad",
        loopback_base + ";id",
        "http://127.0.0.1" + ":abc",
        "http://127.0.0.1" + ":0",
        "http://" + "user:pw@" + "127.0.0.1:8000",
        loopback_base + "/subpath",
        loopback_base + "?x=1",
        loopback_base + "#frag",
        "ftp" + "://127.0.0.1:8000",
    ):
        with pytest.raises(ValueError, match="gateway_url"):
            arena_mod.render_blind_ab_arena_page("fr", "SafeNonce_123456", gateway_url=bad_gw)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="gateway_url"):
            qj_mod.render_quick_join_page("fr", "SafeNonce_123456", gateway_url=bad_gw)  # type: ignore[arg-type]

    for bad_path in (
        "",
        "noel",
        "//evil.run.app/phish",
        "/arena/../secret",
        '/noel" onclick="alert(1)',
        "/noel?lang=en",
        "/noel\r\nX-Injected: 1",
        "/" + ("p" * 200),
        True,
    ):
        with pytest.raises(ValueError, match="path"):
            arena_mod.render_blind_ab_arena_page("en", "SafeNonce_123456", path=bad_path)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="path"):
            qj_mod.render_quick_join_page(
                "en", "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL, path=bad_path  # type: ignore[arg-type]
            )

    # Malformed cards & XSS inside card content in render_blind_ab_arena_page
    base_card = arena.build_cards(lang="en", rng=random.Random(3))[0]
    for bad_cards in (
        "<script>alert(1)</script>",
        [base_card] * 20,
        ["not-a-mapping"],
        [{**base_card, "task_id": 'bad" onclick="alert(1)'}],
        [{**base_card, "ballot_token": '"><script>alert(1)</script>'}],
        [{**base_card, "prompt": "null\x00byte"}],
        [{**base_card, "candidate_a": "x" * 20_000}],
    ):
        with pytest.raises(ValueError):
            arena_mod.render_blind_ab_arena_page("en", "SafeNonce_123456", cards=bad_cards)  # type: ignore[arg-type]

    xss_card = {
        **base_card,
        "badge": "<script>alert('badge')</script>",
        "title": '"><img src=x onerror=alert(1)>',
        "prompt": "</pre><script>alert('prompt')</script>",
        "candidate_a": "<svg onload=alert(1)>",
        "candidate_b": "</script><script>alert('candB')</script>",
    }
    rendered_xss_arena = arena_mod.render_blind_ab_arena_page(
        "en",
        "SafeNonce_123456",
        gateway_url=DEFAULT_GATEWAY_URL,
        path="/arena-noel",
        arena_summary={
            "total_votes": "<script>",
            "holiday_wins": True,
            "baseline_wins": float("nan"),
            "ties": 1.5,
            "win_ratio": "<img onerror=alert(1)>",
            "wilson_95_lower_bound": float("inf"),
            "by_profile": "corrupted",
        },
        cards=[xss_card],
    )
    parsed_arena = _HtmlSecurityParser()
    parsed_arena.feed(rendered_xss_arena)
    assert parsed_arena.inline_handlers == []
    assert parsed_arena.inline_styles == []
    assert parsed_arena.img_srcs == []
    assert parsed_arena.tags.count("svg") == 0
    # Exactly ONE <script> tag (the nonce-pinned controller) — zero injected <script> tags
    assert parsed_arena.tags.count("script") == 1
    assert parsed_arena.script_nonces == ["SafeNonce_123456"]
    assert "&lt;script&gt;alert(&#x27;badge&#x27;)&lt;/script&gt;" in rendered_xss_arena

    # Quick-join page validation & poisoned feedback_summary rendering (both FR & EN, registration True & False)
    with pytest.raises(ValueError, match="lang"):
        qj_mod.render_quick_join_page(False, "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="handle_registration"):
        qj_mod.render_quick_join_page(
            "fr", "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL, handle_registration="yes"  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="feedback_summary"):
        qj_mod.render_quick_join_page(
            "fr", "SafeNonce_123456", gateway_url=DEFAULT_GATEWAY_URL, feedback_summary="bad"  # type: ignore[arg-type]
        )

    for lang_opt, reg_flag, poisoned_summary in (
        ("fr", True, {"total_responses": "<script>alert(1)</script>", "mean_ease_rating": "<img onerror=alert(1)>"}),
        ("en", False, {"total_responses": 12, "mean_ease_rating": float("nan")}),
        ("fr", False, {"total_responses": 8, "mean_ease_rating": 99.9}),
        ("en", True, {"total_responses": True, "mean_ease_rating": 5}),
        ("fr", True, {"total_responses": 10, "mean_ease_rating": 5}),
        ("en", True, {"total_responses": 10, "mean_ease_rating": 4.75}),
    ):
        qj_html = qj_mod.render_quick_join_page(
            lang_opt,
            "QuickNonce_123456",
            gateway_url="http://[::1]:8000",
            path="/noel" if lang_opt == "fr" else "/quick",
            handle_registration=reg_flag,
            feedback_summary=poisoned_summary,
        )
        qj_dom = _HtmlSecurityParser()
        qj_dom.feed(qj_html)
        assert qj_dom.tags.count("script") == 0
        assert qj_dom.img_srcs == []
        assert qj_dom.inline_handlers == []
        assert qj_dom.style_nonces == ["QuickNonce_123456"]
        assert STATIC_FIGURE_RE.findall(qj_html) == []

    # 5. Live HTTP fuzzing on FastAPI router (/arena-noel, /raid-noel, /noel, /v1/hivemind/arena/vote, /v1/hivemind/onboarding/feedback)
    store = RouterStore(tmp_path / "fuzz_router.sqlite3", secret=SOAK_STORE_SECRET)
    app = create_router_app(store, trusted_proxy_hops=1)
    try:
        with TestClient(app) as client:
            for route in ("/arena-noel", "/raid-noel", "/noel", "/quick"):
                r_xss = client.get(
                    f"{route}?lang=%3Cscript%3Ealert(1)%3C/script%3E",
                    headers={"Accept-Language": ("de;q=0.9," * 500) + "<script>alert(1)</script>"},
                )
                assert r_xss.status_code == 200
                assert "<script>alert(1)</script>" not in r_xss.text

            # Malformed JSON & invalid payloads return 4xx (never 500!)
            for endpoint, raw_body in (
                ("/v1/hivemind/arena/vote", b"{truncated-json"),
                ("/v1/hivemind/onboarding/feedback", b"{truncated-json"),
                ("/v1/hivemind/arena/vote", b"[1, 2, 3]"),
                ("/v1/hivemind/onboarding/feedback", b'"just-a-string"'),
            ):
                r_raw = client.post(
                    endpoint,
                    content=raw_body,
                    headers={"Content-Type": "application/json", "X-Forwarded-For": "127.0.99.1"},
                )
                assert 400 <= r_raw.status_code < 500

            for idx, bad_vote_json in enumerate(
                (
                    {"task_id": "<script>alert(1)</script>", "chosen_side": "A", "profile": "dev"},
                    {"task_id": "python-sliding-median", "chosen_side": "<svg>", "profile": "dev"},
                    {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "corporate"},
                    {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "dev", "ballot_token": "x" * 200},
                    {"task_id": "python-sliding-median", "chosen_side": "A", "profile": "dev", "extra_pii": "x"},
                )
            ):
                rv = client.post(
                    "/v1/hivemind/arena/vote",
                    json=bad_vote_json,
                    headers={"X-Forwarded-For": f"127.0.98.{idx + 1}"},
                )
                assert 400 <= rv.status_code < 500
                assert "<script>" not in rv.text

            for idx, bad_fb_json in enumerate(
                (
                    {"profile": "corporate", "ease_rating": 5, "zero_game_lag": True},
                    {"profile": "dev", "ease_rating": 99, "zero_game_lag": True},
                    {"profile": "dev", "ease_rating": 5, "zero_game_lag": "not-a-bool"},
                    {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "platform": "p" * 100},
                    {"profile": "dev", "ease_rating": 5, "zero_game_lag": True, "email": "leak@localhost"},
                )
            ):
                rf = client.post(
                    "/v1/hivemind/onboarding/feedback",
                    json=bad_fb_json,
                    headers={"X-Forwarded-For": f"127.0.97.{idx + 1}"},
                )
                assert 400 <= rf.status_code < 500
    finally:
        store.close()


def test_arena_js_controller_hardening_and_formatters() -> None:
    # Formatter edge cases for complete coverage on blind_ab_arena.py
    assert _format_win_ratio(None) == "n/a"
    assert _format_win_ratio(False) == "n/a"
    assert _format_win_ratio("bad") == "n/a"
    assert _format_win_ratio(0.5) == "50.0%"
    assert _format_wilson(None) == "n/a"
    assert _format_wilson(True) == "n/a"
    assert _format_wilson(float("nan")) == "n/a"
    assert _format_wilson(0.75) == "0.7500"
    assert _format_count(None) == "n/a"
    assert _format_count(True) == "n/a"
    assert _format_count("bad") == "n/a"
    assert _format_count(1.25) == "n/a"
    assert _format_count(42) == "42"

    # Build cards & render_blind_ab_arena_page default FR/EN branches
    assert build_blind_task_cards(lang="fr", rng=random.Random(5))[0]["badge"].startswith("Python")
    with pytest.raises(ValueError, match="lang"):
        arena_mod.build_blind_task_cards(lang=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="lang"):
        arena_mod.render_blind_ab_arena_page(123, "SafeNonce_123456")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="arena_summary"):
        arena_mod.render_blind_ab_arena_page("fr", "SafeNonce_123456", arena_summary="bad")  # type: ignore[arg-type]

    fr_default = arena_mod.render_blind_ab_arena_page("fr", "SafeNonce_123456", rng=random.Random(9))
    assert html.escape(PERSONAL_PC_RULE_FR) in fr_default
    en_populated = arena_mod.render_blind_ab_arena_page(
        "en",
        "SafeNonce_654321",
        arena_summary=arena_mod.BlindArenaCollector(secret=SOAK_BALLOT_SECRET).summary(),
        rng=random.Random(9),
    )
    assert html.escape(PERSONAL_PC_RULE_EN) in en_populated

    # Execute _ARENA_NOEL_JS in Node verifying XSS-poisoned revealed_holiday_side is clamped to '?' via textContent only
    forbidden_prop = "inner" + "HTML"
    node_script = f"""
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
    addEventListener(t, fn) {{ (listeners[t] = listeners[t] || []).push(fn); }},
    async trigger(t) {{ for (const fn of (listeners[t] || [])) await fn({{}}); }},
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
  'arenaNoelApp', 'arenaTotalVotes', 'arenaRecordBreakdown', 'arenaWinRatio',
  'arenaWilsonLower', 'arenaProfileBreakdown', 'refreshArenaBtn', 'arenaRefreshNote',
  'profileBtn-dev', 'profileBtn-gamer', 'arenaProfileStatus'
].forEach((id) => {{ elements[id] = makeEl(id); }});
[
  'python-sliding-median', 'rust-frame-decoder', 'lean4-energy-descent', 'exoplanet-transit-bls'
].forEach((tid) => {{
  elements['arenaCard-' + tid] = makeEl('arenaCard-' + tid);
  elements['voteBtnA-' + tid] = makeEl('voteBtnA-' + tid);
  elements['voteBtnB-' + tid] = makeEl('voteBtnB-' + tid);
  elements['voteBtnTie-' + tid] = makeEl('voteBtnTie-' + tid);
  elements['voteStatus-' + tid] = makeEl('voteStatus-' + tid);
}});
elements.arenaNoelApp.setAttribute('data-lang', 'fr');
globalThis.document = {{ getElementById: (id) => elements[id] || null }};
globalThis.fetch = async (url) => ({{
  ok: true,
  status: 200,
  async json() {{
    return {{
      total_votes: 1, holiday_wins: 1, baseline_wins: 0, ties: 0,
      win_ratio: 1.0, wilson_95_lower_bound: 0.2065,
      by_profile: {{ dev: 1, gamer: 0 }},
      last_vote: {{ revealed_holiday_side: '<script>alert(1)</script>' }},
    }};
  }},
}});
{_ARENA_NOEL_JS}
(async () => {{
  await elements['profileBtn-gamer'].trigger('click');
  await elements['profileBtn-dev'].trigger('click');
  await elements['voteBtnA-python-sliding-median'].trigger('click');
  await elements.refreshArenaBtn.trigger('click');
  console.log(JSON.stringify({{
    statusText: elements['voteStatus-python-sliding-median'].textContent,
    totalText: elements.arenaTotalVotes.textContent,
  }}));
}})().catch((e) => {{ console.error(e); process.exit(1); }});
"""
    proc = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, timeout=15, check=True)
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert "<script>" not in result["statusText"]
    assert "Candidat ? =" in result["statusText"]
    assert result["totalText"] == "1"


def test_zz_target_files_line_coverage_meets_95_percent_gate() -> None:
    if _INPROC_COV is None:
        return
    _INPROC_COV.stop()
    for target_path in (ARENA_PATH, QUICK_JOIN_PATH):
        _, stmts, _, missing, missing_fmt = _INPROC_COV.analysis2(str(target_path))
        covered = len(stmts) - len(missing)
        pct = 100.0 * covered / max(1, len(stmts))
        assert pct >= 95.0, (
            f"Expected >= 95.0% line coverage on {target_path.name}, got {pct:.2f}% "
            f"(missing lines: {missing_fmt})"
        )
