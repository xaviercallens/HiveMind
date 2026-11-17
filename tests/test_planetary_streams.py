"""Validation of the planetary-scale plan manifest (docs/roadmap/planetary_streams.json).

These tests are what keep the plan honest as contributors edit it: every stream has an objective, an explanation,
directions and a definition of done; every definition-of-done item is closed by at least one validation entry with a
command, a receipt and a pass condition; dependencies exist and form no cycle; paper references are real section
labels; tools marked as existing exist; parked ideas cite receipts that exist; and the generated markdown matches.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "roadmap" / "planetary_streams.json"
TEX = ROOT / "paper" / "hivemind_foundation_article.tex"
STREAM_FIELDS = {
    "id", "title", "objective", "explanation", "directions", "dependencies", "definition_of_done", "validation",
    "owner_decisions", "constraints", "prereg_required", "paper_refs", "roadmap_ids", "labels", "skills",
}
# Tools a validation command runs. Receipts under artifacts/ are outputs the command creates, so they are not required to exist.
PATH_TOKEN = re.compile(r"(?<![\w/])((?:scripts|tests|paper|lean4|src)/[\w./-]+)")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def streams(manifest: dict) -> list[dict]:
    return manifest["streams"]


def test_every_stream_has_all_fields_and_nonempty_text(streams: list[dict]) -> None:
    assert len(streams) >= 8
    for s in streams:
        assert STREAM_FIELDS <= set(s), (s.get("id"), STREAM_FIELDS - set(s))
        for k in ("title", "objective", "explanation", "skills"):
            assert isinstance(s[k], str) and len(s[k].strip()) >= 10, (s["id"], k)
        assert s["directions"] and s["definition_of_done"] and s["validation"] and s["labels"], s["id"]
        assert re.fullmatch(r"PS-\d{2}", s["id"]), s["id"]


def test_stream_ids_and_dod_ids_are_unique(streams: list[dict]) -> None:
    ids = [s["id"] for s in streams]
    assert len(ids) == len(set(ids))
    dods = [d["id"] for s in streams for d in s["definition_of_done"]]
    assert len(dods) == len(set(dods))
    for s in streams:
        assert all(d["id"].startswith(s["id"] + "-D") for d in s["definition_of_done"]), s["id"]


def test_dependencies_exist_and_form_no_cycle(streams: list[dict]) -> None:
    graph = {s["id"]: set(s["dependencies"]) for s in streams}
    for sid, deps in graph.items():
        assert deps <= set(graph), (sid, deps - set(graph))
        assert sid not in deps, sid
    order, seen = [], set()
    while len(order) < len(graph):
        ready = [s for s, d in graph.items() if s not in seen and d <= seen]
        assert ready, f"dependency cycle among {set(graph) - seen}"
        order.extend(ready)
        seen.update(ready)


def test_every_dod_item_is_closed_by_a_validation_entry(streams: list[dict]) -> None:
    for s in streams:
        dod_ids = {d["id"] for d in s["definition_of_done"]}
        covered = set()
        for v in s["validation"]:
            assert {"dod", "command", "receipt", "pass", "exists"} <= set(v), (s["id"], v)
            assert v["dod"] in dod_ids, (s["id"], v["dod"])
            assert v["command"].strip() and v["receipt"].strip() and len(v["pass"].strip()) >= 8, (s["id"], v["dod"])
            covered.add(v["dod"])
        assert covered == dod_ids, (s["id"], dod_ids - covered)


def test_tools_marked_existing_exist(streams: list[dict]) -> None:
    for s in streams:
        for v in s["validation"]:
            if not v["exists"]:
                continue
            for token in PATH_TOKEN.findall(v["command"]):
                if "<" in token:
                    continue
                path = ROOT / token.rstrip("/.")
                assert path.exists(), (s["id"], v["dod"], token)


def test_paper_refs_are_real_section_labels(streams: list[dict], manifest: dict) -> None:
    labels = set(re.findall(r"\\label\{(sec:[^}]+)\}", TEX.read_text(encoding="utf-8")))
    for s in streams:
        for r in s["paper_refs"]:
            assert r in labels, (s["id"], r)
    for p in manifest["parked"]:
        assert p["paper_ref"] in labels, p["idea"]


def test_baseline_receipts_exist(manifest: dict) -> None:
    for f in manifest["baseline"]["facts"]:
        if f["receipt"].startswith(("artifacts/", "paper/")):
            assert (ROOT / f["receipt"]).exists(), f["receipt"]


def test_parked_entries_cite_existing_receipts_and_reopen_conditions(manifest: dict) -> None:
    assert manifest["parked"]
    for p in manifest["parked"]:
        assert (ROOT / p["receipt"]).exists(), p["receipt"]
        assert len(p["reopen_if"]) >= 20 and len(p["reason"]) >= 20, p["idea"]


def test_waves_are_ordered_and_gated(manifest: dict, streams: list[dict]) -> None:
    waves = manifest["waves"]
    targets = [w["nodes_target"] for w in waves]
    assert targets == sorted(targets) and waves[-1]["nodes_target"] >= 10000
    ids = {s["id"] for s in streams}
    for w in waves[1:]:
        assert w["entry"], w["id"]
        for ref in re.findall(r"PS-\d{2}", " ".join(w["entry"])):
            assert ref in ids, (w["id"], ref)


def test_rendered_markdown_is_in_sync() -> None:
    spec = importlib.util.spec_from_file_location("render_planetary_plan", ROOT / "scripts" / "render_planetary_plan.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    m = mod.load()
    assert mod.PLAN_MD.read_text(encoding="utf-8") == mod.render_plan(m) + "\n", "run scripts/render_planetary_plan.py"
    assert mod.PARKED_MD.read_text(encoding="utf-8") == mod.render_parked(m), "run scripts/render_planetary_plan.py"
