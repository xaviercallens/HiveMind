"""QLoRA MoE training domains: registry, fetch manifest, null-safe documents, and the held-out tail staying out of training."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hivemind.boinc.real_client import moe_work  # noqa: E402

_spec = importlib.util.spec_from_file_location("moe_fetch_domains", Path(__file__).resolve().parents[1] / "scripts" / "moe_fetch_domains.py")
fetch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fetch)

ORIGINAL = ["general", "french", "edu_web", "news", "biomed", "math"]


def test_original_cycle_is_a_prefix_and_new_domains_are_registered() -> None:
    assert moe_work.DOMAIN_ORDER[:6] == ORIGINAL
    assert len(moe_work.DOMAIN_ORDER) == len(set(moe_work.DOMAIN_ORDER)) == 16
    assert set(moe_work.DOMAIN_ORDER) == set(moe_work._SOURCES)


def test_every_added_dataset_is_in_the_pinned_fetch_manifest() -> None:
    cached = {fetch.cache_dir_name(r).removeprefix("datasets--") for r in fetch.DATASETS}
    for d in moe_work.DOMAIN_ORDER[6:]:
        assert moe_work._SOURCES[d]["ds"] in cached, d
    for repo, (sha, files) in fetch.DATASETS.items():
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), repo  # pinned to a commit, not a moving branch
        assert files and all(f.endswith(".parquet") for f in files)


def test_every_source_file_pattern_matches_a_manifest_file() -> None:
    for d in moe_work.DOMAIN_ORDER[6:]:
        s = moe_work._SOURCES[d]
        repo = next(r for r in fetch.DATASETS if fetch.cache_dir_name(r).removeprefix("datasets--") == s["ds"])
        names = fetch.DATASETS[repo][1]
        for key in ("train", "eval"):
            assert any(Path(n).match(s[key]) or n == s[key] for n in names), (d, key, s[key], names)


def test_benchmarks_are_not_training_sources() -> None:
    joined = " ".join(s["ds"] for s in moe_work._SOURCES.values()).lower()
    assert not any(b in joined for b in ("mbpp", "humaneval", "bigcodebench", "livecodebench", "mmlu"))


def test_null_columns_are_skipped_not_written_as_none() -> None:
    docs = moe_work._docs([{"q": "why?", "exp": None}, {"q": "how?", "exp": "because"}], ["q", "exp"])
    assert docs == ["why?", "how?\nbecause"]
    assert all("None" not in d for d in docs)


def _single_file_source(tmp_path: Path, n: int, rows: int, eval_start: int):
    f = tmp_path / "datasets--x--y" / "snapshots" / "abc" / "d.parquet"
    f.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": [f"row{i}" for i in range(n)]}), f, row_group_size=50)
    moe_work.HUB = tmp_path
    moe_work._SOURCES["_t"] = {"ds": "x--y", "train": "d.parquet", "eval": "d.parquet", "cols": ["text"], "rows": rows, "eval_start": eval_start}
    return f


def test_a_cursor_past_the_last_shard_wraps_and_never_reads_the_held_out_tail(tmp_path: Path) -> None:
    old_hub = moe_work.HUB
    try:
        f = _single_file_source(tmp_path, n=1000, rows=100, eval_start=-200)
        tok = lambda s: type("T", (), {"input_ids": [1] * len(s)})()  # noqa: E731 - one token per character
        held_out = {f"row{i}" for i in range(800, 1000)}
        for shard in range(0, 40):  # 8 whole shards fit before the held-out tail; the cursor must wrap, not run on
            ids = moe_work.domain_ids(tok, "_t", "train", shard, 64 * 128)
            assert ids, shard
        # the rows a shard reads: shard 8 wraps to shard 0, shard 9 to shard 1
        start0 = [r["text"] for r in moe_work._rows(f, ["text"], 0, 100)]
        assert start0[0] == "row0" and not (set(start0) & held_out)
        got = {d: len(moe_work.domain_ids(tok, "_t", "train", d, 64 * 128)) for d in (0, 8, 16)}
        assert got[0] == got[8] == got[16]
        assert moe_work._total_rows(f) == 1000
    finally:
        moe_work.HUB = old_hub
        moe_work._SOURCES.pop("_t", None)


def test_wrap_matches_the_unwrapped_read_for_shards_that_fit(tmp_path: Path) -> None:
    old_hub = moe_work.HUB
    try:
        f = _single_file_source(tmp_path, n=1000, rows=100, eval_start=-200)
        tok = lambda s: type("T", (), {"input_ids": [ord(c) for c in s]})()  # noqa: E731
        for shard in range(8):
            want = [r["text"] for r in moe_work._rows(f, ["text"], shard * 100, 100)]
            got = moe_work.domain_ids(tok, "_t", "train", shard, 64 * 128)
            assert got == [ord(c) for c in "".join(w + "\n\n" for w in want)]
    finally:
        moe_work.HUB = old_hub
        moe_work._SOURCES.pop("_t", None)


@pytest.mark.parametrize("domain", ["science", "cyber", "olympiad_math"])
def test_spec_per_domain_has_positive_rows_and_valid_eval_start(domain: str) -> None:
    s = moe_work._SOURCES[domain]
    assert s["rows"] > 0 and s["cols"] and isinstance(s["eval_start"], int)
