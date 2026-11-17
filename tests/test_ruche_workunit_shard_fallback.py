import json
from pathlib import Path

from hivemind.boinc import hivemind_at_home_dashboard as dash


def test_shards_from_cas_reads_domain_from_file_names(tmp_path: Path) -> None:
    cas = tmp_path / "mined_shards_cas" / "dataset_shards"
    cas.mkdir(parents=True)
    (cas / "code_python_shard_1_abc.json").write_text(json.dumps({"texts": ["x"]}), encoding="utf-8")
    (cas / "bio_med_shard_6_def.json").write_text(json.dumps({"texts": ["y"]}), encoding="utf-8")
    (cas / "notes.txt").write_text("ignored", encoding="utf-8")
    shards = dash._shards_from_cas(tmp_path)
    assert [s["domain_name"] for s in shards] == ["bio_med", "code_python"]


def test_shards_from_cas_is_empty_when_directory_missing(tmp_path: Path) -> None:
    assert dash._shards_from_cas(tmp_path) == []
