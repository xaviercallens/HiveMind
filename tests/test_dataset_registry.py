# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Tests for the S21 dataset registry: licence gate, Hub lookup, slice mapping, decontamination and the mixture caps.

Only network I/O is replaced (an injected ``fetch`` or a patched ``urlopen``); every check has a positive and a negative case.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hivemind.data import dataset_registry as dr

REPO = Path(__file__).resolve().parent.parent
SYS = "system prompt"


def spec(mapping: dict[str, Any], **kw: Any) -> dr.SourceSpec:
    base = {"id": kw.pop("id", "org/ds"), "domain": kw.pop("domain", "python_coding"), "config": "default", "split": "train",
            "mapping": mapping, "verified_signal": kw.pop("verified_signal", True), "signal_basis": "test", **kw}
    return dr.SourceSpec.from_dict(base)


FIELDS = {"kind": "fields", "prompt_fields": ["q", "ctx"], "completion_field": "a", "key_field": "id"}


# ---------------------------------------------------------------- licences
@pytest.mark.parametrize("raw", ["mit", "apache-2.0", ["cc-by-4.0"], ["mit", "apache-2.0"], "CC-BY-SA-4.0", "odc-by"])
def test_allowlisted_licences_are_accepted_with_conditions(raw: object) -> None:
    d = dr.classify_license(raw)
    assert d.status == dr.ACCEPTED
    assert d.conditions and d.license


@pytest.mark.parametrize("raw,needle", [
    (None, "no licence"), ("", "no licence"), ([], "no licence"), ("cc-by-nc-4.0", "non-commercial"),
    (["cc-by-nc-3.0"], "non-commercial"), ("cc-by-nc-sa-4.0", "non-commercial"), ("cc-by-nd-4.0", "derivatives"),
    ("other", "custom or unknown"), ("unknown", "custom or unknown"), ("llama3", "not on the allowlist"),
    (["mit", "cc-by-nc-4.0"], "non-commercial"), (42, "unreadable"), ([""], "unreadable"), ({"a": 1}, "unreadable"),
])
def test_unusable_licences_are_rejected_with_a_reason(raw: object, needle: str) -> None:
    d = dr.classify_license(raw)
    assert d.status == dr.REJECTED
    assert needle in d.reason
    assert d.conditions is None


# ---------------------------------------------------------------- candidates file
def test_candidates_file_parses_and_covers_six_domains_with_rejections_recorded() -> None:
    doc, specs = dr.load_candidates(REPO / "configs" / "datasets_s21_candidates.json")
    per_domain = {d: [s for s in specs if s.domain == d] for d in doc["domains"]}
    assert set(per_domain) == {"python_coding", "maths", "physics", "exoplanet", "stars", "cyber"}
    assert all(len(v) >= 2 for v in per_domain.values())
    assert abs(sum(d["weight"] for d in doc["domains"].values()) - 1.0) < 1e-9


def test_generated_registry_matches_candidates_and_carries_api_metadata() -> None:
    reg = json.loads((REPO / "configs" / "datasets_s21.json").read_text(encoding="utf-8"))
    _, specs = dr.load_candidates(REPO / "configs" / "datasets_s21_candidates.json")
    assert [(e["id"], e["config"]) for e in reg["registry"]] == [(s.dataset_id, s.config) for s in specs]
    for e in reg["registry"]:
        assert e["status"] in (dr.ACCEPTED, dr.REJECTED) and e["reason"]
        if e["status"] == dr.ACCEPTED:
            assert len(e["revision_sha"]) == 40 and e["retrieved_utc"].endswith("Z")
            assert dr.classify_license(e["license_as_read"]).status == dr.ACCEPTED


@pytest.mark.parametrize("bad,err", [
    ({"verified_signal": "yes"}, "must be a boolean"), ({"mapping": {"kind": "nope"}}, "unknown mapping kind"), ({"split": None}, "lacks"),
])
def test_candidate_validation_rejects_malformed_entries(bad: dict[str, Any], err: str) -> None:
    d = {"id": "x/y", "domain": "maths", "config": "c", "split": "s", "mapping": FIELDS, "verified_signal": True, "signal_basis": "b", **bad}
    d = {k: v for k, v in d.items() if v is not None}
    with pytest.raises((ValueError, TypeError), match=err):
        dr.SourceSpec.from_dict(d)


def test_load_candidates_rejects_undeclared_domain_and_duplicates(tmp_path: Path) -> None:
    c = {"id": "x/y", "domain": "maths", "config": "c", "split": "s", "mapping": FIELDS, "verified_signal": True, "signal_basis": "b"}
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"domains": {}, "candidates": [c]}))
    with pytest.raises(ValueError, match="not declared"):
        dr.load_candidates(p)
    p.write_text(json.dumps({"domains": {"maths": {}}, "candidates": [c, c]}))
    with pytest.raises(ValueError, match="twice"):
        dr.load_candidates(p)
    p.write_text(json.dumps({"domains": {"maths": {}}, "candidates": [c]}))
    assert dr.load_candidates(p)[1][0].dataset_id == "x/y"


# ---------------------------------------------------------------- Hub lookup (network mocked)
def hub(meta: dict[str, Any]) -> dr.Fetch:
    def fetch(url: str) -> bytes:
        assert url.startswith(dr.HUB_API)
        return json.dumps(meta).encode()
    return fetch


META = {"sha": "a" * 40, "lastModified": "2024-01-01T00:00:00.000Z", "usedStorage": 123, "gated": False,
        "cardData": {"license": "mit", "size_categories": ["10K<n<100K"]}}


def test_registry_entry_accepts_and_copies_api_fields() -> None:
    e = dr.registry_entry(spec(FIELDS), hub(META), "2026-10-28T00:00:00Z")
    assert e["status"] == dr.ACCEPTED
    assert (e["revision_sha"], e["license"], e["used_storage_bytes"], e["size_categories"]) == ("a" * 40, "mit", 123, ["10K<n<100K"])


@pytest.mark.parametrize("meta,needle", [
    ({**META, "cardData": {}}, "no licence"), ({**META, "cardData": None}, "no licence"),
    ({**META, "cardData": {"license": "cc-by-nc-4.0"}}, "non-commercial"), ({**META, "gated": "manual"}, "gated"),
    ({**META, "disabled": True}, "disabled"),
])
def test_registry_entry_rejects_with_reason(meta: dict[str, Any], needle: str) -> None:
    e = dr.registry_entry(spec(FIELDS), hub(meta), "t")
    assert e["status"] == dr.REJECTED
    assert needle in e["reason"]


def test_registry_entry_records_api_failure_instead_of_dropping() -> None:
    def boom(url: str) -> bytes:
        raise OSError("HTTP 404")
    e = dr.registry_entry(spec(FIELDS), boom, "t")
    assert e["status"] == dr.REJECTED and "Hub API lookup failed" in e["reason"]
    assert e["id"] == "org/ds"


def test_default_fetch_sends_user_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    class Resp:
        def __enter__(self) -> Resp:  # noqa: PYI034 - local test double
            return self

        def __exit__(self, *a: object) -> None:
            return None

        def read(self) -> bytes:
            return b"payload"

    def fake_urlopen(req: Any, timeout: float) -> Resp:
        seen["ua"], seen["timeout"] = req.get_header("User-agent"), timeout
        return Resp()

    monkeypatch.setattr(dr.urllib.request, "urlopen", fake_urlopen)
    assert dr.default_fetch("https://example.invalid/x", timeout_s=5) == b"payload"
    assert seen == {"ua": dr.USER_AGENT, "timeout": 5}


def test_utc_now_is_iso_utc() -> None:
    t = dr.utc_now()
    assert len(t) == 20 and t.endswith("Z") and t[4] == "-"


def test_split_size_finds_split_and_reports_missing_or_error() -> None:
    doc = {"size": {"splits": [{"config": "default", "split": "train", "num_rows": 7, "num_bytes_parquet_files": 99}]}}
    ok = dr.split_size(spec(FIELDS), lambda u: json.dumps(doc).encode())
    assert ok == {"status": dr.PASSED, "num_rows": 7, "num_bytes_parquet_files": 99}
    other = dr.split_size(dr.SourceSpec("o/d", "m", "default", "test", FIELDS, True, "b"), lambda u: json.dumps(doc).encode())
    assert other["status"] == dr.BLOCKED and "not listed" in other["reason"]

    def boom(url: str) -> bytes:
        raise TimeoutError("slow")
    assert dr.split_size(spec(FIELDS), boom)["status"] == dr.BLOCKED


def rows_server(total: int, fail_at: int | None = None) -> tuple[dr.Fetch, list[str]]:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)
        off, length = int(q["offset"][0]), int(q["length"][0])
        if fail_at is not None and off >= fail_at:
            raise OSError("502")
        rows = [{"row": {"id": i, "q": f"question {i}", "a": f"answer {i}"}} for i in range(off, min(total, off + length))]
        return json.dumps({"rows": rows}).encode()
    return fetch, calls


def test_slice_rows_pages_and_hashes_every_page() -> None:
    fetch, calls = rows_server(1000)
    res = dr.slice_rows(spec(FIELDS, slice_rows=250), fetch)
    assert res.status == dr.PASSED and len(res.rows) == 250
    assert [s.n_rows for s in res.shards] == [100, 100, 50] and len(calls) == 3
    assert len({s.sha256 for s in res.shards}) == 3


def test_slice_rows_stops_at_end_and_blocks_on_error_or_empty() -> None:
    fetch, _ = rows_server(120)
    short = dr.slice_rows(spec(FIELDS), fetch, n_rows=300)
    assert short.status == dr.PASSED and len(short.rows) == 120
    fetch, _ = rows_server(1000, fail_at=100)
    broken = dr.slice_rows(spec(FIELDS), fetch, n_rows=300)
    assert broken.status == dr.BLOCKED and "offset 100" in broken.reason and len(broken.rows) == 100
    fetch, _ = rows_server(0)
    empty = dr.slice_rows(spec(FIELDS), fetch)
    assert empty.status == dr.BLOCKED and empty.rows == []


def test_read_jsonl_shards_hashes_exact_bytes(tmp_path: Path) -> None:
    p = tmp_path / "a.jsonl"
    p.write_bytes(b'{"x": 1}\n\n{"x": 2}\n')
    rows, shards = dr.read_jsonl_shards([p])
    assert rows == [{"x": 1}, {"x": 2}]
    import hashlib
    assert shards[0].sha256 == hashlib.sha256(p.read_bytes()).hexdigest() and shards[0].to_dict()["rows"] == 2


# ---------------------------------------------------------------- mapping
def test_fields_mapping_positive_and_negative() -> None:
    s = spec(FIELDS)
    out = dr.map_row(s, {"id": 5, "q": " Sort a list ", "ctx": "in place", "a": "xs.sort()"}, SYS, 0)
    assert out is not None and out["key"] == "org/ds:5"
    assert out["messages"][1]["content"] == "Sort a list\n\nin place" and out["completion"] == "xs.sort()"
    with pytest.raises(dr.Unmappable):
        dr.map_row(s, {"q": "", "a": "x"}, SYS, 0)
    with pytest.raises(dr.Unmappable):
        dr.map_row(s, {"q": "x", "a": None}, SYS, 0)


def test_filters_and_exclude_pattern_drop_rows_by_design() -> None:
    s = spec({**FIELDS, "filter_in": {"lang": ["python"]}, "exclude_pattern": "malware"})
    assert dr.map_row(s, {"q": "x", "a": "y", "lang": "rust"}, SYS, 0) is None
    assert dr.map_row(s, {"q": "write malware", "a": "y", "lang": "python"}, SYS, 0) is None
    assert dr.map_row(s, {"q": "x", "a": "drop the MALWARE here", "lang": "python"}, SYS, 0) is None
    kept = dr.map_row(s, {"q": "x", "a": "y", "lang": "python", "unused_column": "malware"}, SYS, 3)
    assert kept is not None and kept["key"] == "org/ds:3"  # columns not trained on are not searched


def test_messages_mapping() -> None:
    s = spec({"kind": "messages", "messages_field": "m"})
    ok = dr.map_row(s, {"m": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}, SYS, 0)
    assert ok is not None and ok["messages"][1]["content"] == "q" and ok["completion"] == "a"
    sharegpt = dr.map_row(s, {"m": [{"from": "human", "value": "q2"}, {"from": "gpt", "value": "a2"}]}, SYS, 0)
    assert sharegpt is not None and sharegpt["completion"] == "a2"
    for bad in ({"m": []}, {"m": "text"}, {"m": [{"role": "user", "content": "q"}]}, {"m": [{"role": "assistant", "content": "a"}]}):
        with pytest.raises(dr.Unmappable):
            dr.map_row(s, bad, SYS, 0)


MCQ = {"kind": "mcq", "question_field": "question", "choices_field": "choices", "answer_field": "answerKey"}


def test_mcq_mapping_and_its_schema_errors() -> None:
    s = spec(MCQ, domain="physics")
    row = {"question": "Q?", "choices": {"text": ["a", "b"], "label": ["A", "B"]}, "answerKey": "B"}
    out = dr.map_row(s, row, SYS, 0)
    assert out is not None and out["completion"] == "Answer: B" and out["messages"][1]["content"] == "Q?\nA. a\nB. b"
    for bad in ({**row, "answerKey": "C"}, {**row, "choices": ["a", "b"]}, {**row, "choices": {"text": ["a"], "label": ["A", "B"]}}):
        with pytest.raises(dr.Unmappable):
            dr.map_row(s, bad, SYS, 0)


LABEL = {"kind": "label", "label_field": "disp", "label_map": {"PLANET": "Verdict: CONFIRMED", "FALSE_POSITIVE": "Verdict: FALSE POSITIVE"},
         "prompt_header": "Signal:", "question": "Real?", "min_features": 2, "missing_values": [-9999],
         "features": {"p": "period", "d": "depth", "r": "radius"}, "entity_key": {"field": "koi", "style": "koi"}, "key_field": "koi"}


def test_label_mapping_formats_features_skips_missing_and_sets_entity_key() -> None:
    s = spec(LABEL, domain="exoplanet")
    out = dr.map_row(s, {"p": 12.345, "d": 256.0, "r": -9999, "disp": "PLANET", "koi": "5218.01"}, SYS, 0)
    assert out is not None and out["completion"] == "Verdict: CONFIRMED"
    assert out["messages"][1]["content"] == "Signal:\n- period: 12.3\n- depth: 256\nReal?"
    assert out["entity_key"] == "K05218.01"
    assert dr.map_row(s, {"p": 1, "d": 2, "disp": "NO_SIGNAL"}, SYS, 0) is None
    assert dr.map_row(s, {"p": 1, "d": 2, "disp": None}, SYS, 0) is None
    with pytest.raises(dr.Unmappable, match="too few"):
        dr.map_row(s, {"p": 1, "d": float("nan"), "disp": "PLANET"}, SYS, 0)
    with pytest.raises(dr.Unmappable, match="type"):
        dr.map_row(s, {"p": [1], "d": 2, "disp": "PLANET"}, SYS, 0)


def test_label_mapper_called_directly_rejects_a_non_string_label() -> None:
    with pytest.raises(dr.Unmappable, match="label"):
        dr.MAPPERS["label"]({"p": 1, "d": 2, "disp": 3}, LABEL)
    assert dr.MAPPERS["label"]({"p": 1, "d": 2, "disp": "PLANET"}, LABEL)[1] == "Verdict: CONFIRMED"


def test_label_first_char_and_system_override() -> None:
    m = {**LABEL, "label_first_char": True, "label_map": {"G": "Class: G"}, "system_prompt": "override", "entity_key": None}
    s = spec(m, domain="stars")
    out = dr.map_row(s, {"p": 1, "d": "faint", "disp": "g2"}, SYS, 0)
    assert out is not None and out["completion"] == "Class: G" and out["messages"][0]["content"] == "override"
    assert "- depth: faint" in out["messages"][1]["content"]
    assert dr.map_row(s, {"p": 1, "d": 2, "disp": "K5"}, SYS, 0) is None


@pytest.mark.parametrize("value,style,expected", [
    ("K04170.01", "koi", "K04170.01"), ("4170.01", "koi", "K04170.01"), ("k00012.02", "koi", "K00012.02"),
    ("TOI-1.01", "koi", None), ("", "koi", None), (None, "star", None), ("HD 1", "star", "star-HD 1"), (" x ", "raw", "x")])
def test_normalize_entity_key(value: Any, style: str, expected: str | None) -> None:
    assert dr.normalize_entity_key(value, style) == expected


@pytest.mark.parametrize("v,s", [(0, "0"), (6360.4, "6360"), (2.4812, "2.48"), (0.01234, "0.0123"), (1.2e-5, "1.20e-05"), (3.2e8, "3.200e+08")])
def test_fmt_number(v: float, s: str) -> None:
    assert dr.fmt_number(v) == s


def test_validate_slice_passes_clean_slices_and_fails_broken_schemas() -> None:
    s = spec(FIELDS)
    good = [{"id": i, "q": f"q{i}", "a": f"a{i}"} for i in range(20)]
    rep, mapped = dr.validate_slice(s, good, SYS)
    assert rep["status"] == dr.PASSED and rep["rows_mapped"] == 20 and len(mapped) == 20
    mostly_bad = good[:5] + [{"id": i, "question": "wrong field"} for i in range(15)]
    rep2, _ = dr.validate_slice(s, mostly_bad, SYS)
    assert rep2["status"] == dr.FAILED and rep2["rows_unmappable"] == 15 and "unmappable" in rep2["reason"]
    rep3, _ = dr.validate_slice(spec({**FIELDS, "filter_in": {"lang": ["python"]}}), good, SYS)
    assert rep3["status"] == dr.FAILED and rep3["reason"] == "no row mapped" and rep3["rows_filtered_by_design"] == 20


# ---------------------------------------------------------------- decontamination
def row(user: str, completion: str = "c", key: str = "k", **kw: Any) -> dict[str, Any]:
    return {"key": key, "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user}], "completion": completion, **kw}


EVAL_Q = "Natalia sold clips to forty eight of her friends in April and then she sold half as many clips in May how many"


def test_ngram_overlap_is_removed_and_counted_but_unrelated_text_kept() -> None:
    idx = dr.EvalIndex()
    idx.add("gsm8k", [EVAL_Q], ["eval-1"])
    leaked = row("Rephrased: " + EVAL_Q.upper() + "!!", key="a")
    clean = row("A train leaves the station at noon travelling sixty miles per hour toward a city two hundred miles away", key="b")
    short = row("Natalia sold clips", key="c")  # under 13 words of overlap: kept
    kept, counts = dr.decontaminate([leaked, clean, short], idx)
    assert [r["key"] for r in kept] == ["b", "c"]
    assert counts == {"eval_entity_key": 0, "eval_exact": 0, "eval_ngram": 1, "exact_duplicate": 0, "kept": 2}


def test_exact_eval_prompt_entity_key_and_duplicates_are_removed() -> None:
    idx = dr.EvalIndex()
    idx.add("pack", ["What is 2 + 2?"], ["K00001.01"])
    rows = [row("what is 2+2", key="exact"), row("unrelated", key="ent", entity_key="K00001.01"),
            row("Fresh question here", key="d1"), row("fresh   QUESTION here!", key="d2"), row("other", key="ok", entity_key="K00002.01")]
    kept, counts = dr.decontaminate(rows, idx)
    assert [r["key"] for r in kept] == ["d1", "ok"]
    assert (counts["eval_exact"], counts["eval_entity_key"], counts["exact_duplicate"]) == (1, 1, 1)


def test_boilerplate_ngrams_are_ignored_so_shared_scaffolding_does_not_wipe_a_source() -> None:
    scaffold = "Kepler transit signal orbital period days transit duration hours transit depth ppm planet radius earth radii"
    idx = dr.EvalIndex(max_doc_freq=3)
    idx.add("pack", [f"{scaffold} value {i}" for i in range(5)])
    kept, counts = dr.decontaminate([row(scaffold + " value 999")], idx)
    assert counts["eval_ngram"] == 0 and len(kept) == 1
    assert idx.boilerplate_ngrams > 0
    strict = dr.EvalIndex(max_doc_freq=10)
    strict.add("pack", [f"{scaffold} value {i}" for i in range(5)])
    assert dr.decontaminate([row(scaffold + " value 999")], strict)[1]["eval_ngram"] == 1


def test_add_eval_pack_reads_user_turns_and_keys(tmp_path: Path) -> None:
    pack = tmp_path / "maths"
    pack.mkdir()
    (pack / "eval.jsonl").write_text(json.dumps(row(EVAL_Q, key="gsm8k-test-0")) + "\n" + json.dumps(row("", key="empty")) + "\n")
    idx = dr.EvalIndex()
    idx.add_eval_pack(pack)
    assert idx.sources == {"night_pack:maths": 1} and "gsm8k-test-0" in idx.keys
    assert dr.decontaminate([row(EVAL_Q)], idx)[1]["eval_exact"] == 1


@given(st.text(min_size=1, max_size=200))
@settings(max_examples=60, deadline=None)
def test_text_hash_ignores_case_and_punctuation(s: str) -> None:
    assert dr.text_hash(s) == dr.text_hash("  " + s.upper() + " ?! ")
    assert dr.text_hash(s + " zqxj") != dr.text_hash(s) or not dr.normalize_tokens(s + " zqxj")


@given(st.lists(st.sampled_from(["a", "b", "c", "d"]), min_size=0, max_size=40), st.integers(min_value=1, max_value=15))
@settings(max_examples=60, deadline=None)
def test_ngram_count_bound(tokens: list[str], n: int) -> None:
    hs = dr.ngram_hashes(tokens, n)
    assert len(hs) <= max(0, len(tokens) - n + 1)
    assert (len(hs) == 0) == (len(tokens) < n)


# ---------------------------------------------------------------- mixture
def test_mixture_config_rejects_bad_values() -> None:
    for kw in ({"total_rows": 0}, {"domain_weights": {}}, {"domain_weights": {"a": -1.0}}, {"per_source_cap": 0},
               {"unverified_cap": 1.0}):
        args = {"total_rows": 10, "domain_weights": {"a": 1.0}, **kw}
        with pytest.raises(ValueError):
            dr.MixtureConfig(**args)
    assert dr.MixtureConfig(10, {"a": 1.0}).unverified_cap == 0.30


def test_plan_enforces_unverified_cap_and_source_cap() -> None:
    avail = {"v1": ("code", True, 1000), "v2": ("code", True, 30), "u1": ("code", False, 5000), "u2": ("code", False, 5000)}
    plan = dr.plan_mixture(avail, dr.MixtureConfig(1000, {"code": 1.0}, per_source_cap=0.5))
    a = plan["allocation"]
    assert a["v1"] == 500 and a["v2"] == 30  # source cap 500; v2 limited by availability
    assert a["u1"] + a["u2"] == 227  # floor(0.3/0.7 * 530)
    d = plan["domains"]["code"]
    assert d["unverified_fraction"] <= 0.30 and d["status"] == "SHORT" and "short" in d["reason"]


def test_plan_blocks_a_domain_without_verified_rows_and_builds_a_full_one() -> None:
    avail = {"u": ("cyber", False, 10_000), "v": ("maths", True, 10_000), "v2": ("maths", True, 10_000), "u2": ("maths", False, 10_000)}
    plan = dr.plan_mixture(avail, dr.MixtureConfig(1000, {"cyber": 0.5, "maths": 0.5}, per_source_cap=0.5))
    assert plan["domains"]["cyber"]["status"] == dr.BLOCKED and plan["allocation"]["u"] == 0
    assert plan["domains"]["maths"]["status"] == dr.BUILT and plan["domains"]["maths"]["unverified_rows"] == 0
    assert plan["unverified_fraction"] == 0.0 and plan["total_rows"] == 500


def test_plan_admits_unverified_when_verified_alone_is_capped() -> None:
    avail = {"v": ("m", True, 10_000), "u": ("m", False, 10_000)}
    plan = dr.plan_mixture(avail, dr.MixtureConfig(100, {"m": 1.0}, per_source_cap=0.7))
    assert plan["allocation"] == {"u": 30, "v": 70}
    assert plan["domains"]["m"]["status"] == dr.BUILT and plan["unverified_fraction"] == pytest.approx(0.30)


@given(st.dictionaries(st.text(alphabet="abcdef", min_size=1, max_size=3),
                       st.tuples(st.sampled_from(["x", "y"]), st.booleans(), st.integers(min_value=0, max_value=500)), max_size=8),
       st.integers(min_value=1, max_value=2000), st.floats(min_value=0.05, max_value=1.0))
@settings(max_examples=80, deadline=None)
def test_plan_invariants(avail: dict[str, tuple[str, bool, int]], total: int, cap: float) -> None:
    cfg = dr.MixtureConfig(total, {"x": 1.0, "y": 2.0}, per_source_cap=cap)
    plan = dr.plan_mixture(avail, cfg)
    for s, n in plan["allocation"].items():
        assert 0 <= n <= avail[s][2]
    for info in plan["domains"].values():
        assert info["verified_rows"] + info["unverified_rows"] <= info["target"]
        if info["unverified_fraction"] is not None:
            assert info["unverified_fraction"] <= 0.30 + 1e-9
    assert sum(plan["allocation"].values()) == plan["total_rows"]


def test_sample_rows_is_deterministic_and_order_independent() -> None:
    rows = [{"key": f"k{i}"} for i in range(50)]
    a = dr.sample_rows(rows, 10, 1, "s")
    b = dr.sample_rows(list(reversed(rows)), 10, 1, "s")
    assert a == b and len(a) == 10
    assert dr.sample_rows(rows, 10, 2, "s") != a
    assert dr.sample_rows(rows, -1, 1, "s") == []


def test_build_mixture_end_to_end_manifest() -> None:
    idx = dr.EvalIndex()
    idx.add("pack", [EVAL_Q], [])
    ver = [row(f"verified question number {i}", key=f"v{i}") for i in range(40)] + [row(EVAL_Q, key="leak")]
    unv = [row(f"unverified question number {i}", key=f"u{i}") for i in range(40)] + [row("verified question number 0", key="dup")]
    shard = dr.Shard("mem://a", "0" * 64, 10, 41)
    mix, man = dr.build_mixture({"V": ver, "U": unv}, {"V": True, "U": False}, {"V": "m", "U": "m"},
                                dr.MixtureConfig(40, {"m": 1.0}, per_source_cap=1.0), idx, [shard])
    assert man["decontamination"]["per_source"]["V"]["eval_exact"] == 1
    assert man["decontamination"]["per_source"]["U"]["exact_duplicate"] == 1  # duplicate of a V row, seen first
    assert man["plan"]["allocation"] == {"U": 0, "V": 40} and len(mix) == 40
    assert man["shards"][0]["sha256"] == "0" * 64 and len(man["mixture_sha256"]) == 64
    _, man2 = dr.build_mixture({"V": ver, "U": unv}, {"V": True, "U": False}, {"V": "m", "U": "m"},
                                  dr.MixtureConfig(40, {"m": 1.0}, per_source_cap=1.0), idx, [shard])
    assert man2["mixture_sha256"] == man["mixture_sha256"]


# ---------------------------------------------------------------- retries
def flaky(errors: list[BaseException]) -> tuple[dr.Fetch, list[str]]:
    calls: list[str] = []

    def fetch(url: str) -> bytes:
        calls.append(url)
        if errors:
            raise errors.pop(0)
        return b"ok"
    return fetch, calls


def http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {}, None)  # type: ignore[arg-type]


def test_retrying_fetch_retries_rate_limits_and_transient_errors() -> None:
    sleeps: list[float] = []
    fetch, calls = flaky([http_error(429), http_error(503), urllib.error.URLError("reset"), TimeoutError()])
    assert dr.retrying_fetch(fetch, retries=5, base_delay_s=1.0, sleep=sleeps.append)("u") == b"ok"
    assert len(calls) == 5 and sleeps == [1.0, 2.0, 3.0, 4.0]


def test_retrying_fetch_does_not_retry_client_errors_and_gives_up_after_retries() -> None:
    sleeps: list[float] = []
    fetch, calls = flaky([http_error(404)])
    with pytest.raises(urllib.error.HTTPError):
        dr.retrying_fetch(fetch, sleep=sleeps.append)("u")
    assert len(calls) == 1 and sleeps == []
    fetch, calls = flaky([http_error(429)] * 3)
    with pytest.raises(urllib.error.HTTPError):
        dr.retrying_fetch(fetch, retries=2, base_delay_s=0.5, sleep=sleeps.append)("u")
    assert len(calls) == 3 and sleeps == [0.5, 1.0]
    fetch, calls = flaky([urllib.error.URLError("down")] * 2)
    with pytest.raises(urllib.error.URLError):
        dr.retrying_fetch(fetch, retries=1, base_delay_s=0.0, sleep=sleeps.append)("u")
    assert len(calls) == 2
