"""Big-GPU training profile: a separate app and plan class, larger workunits, its own adapter chain, and no way to mix it with the standard one."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.boinc.real_client import boinc_server_scripts as bss  # noqa: E402
from hivemind.boinc.real_client import moe_app, moe_profiles, moe_work  # noqa: E402
from hivemind.boinc.real_client.moe_bridge import MoEValidator  # noqa: E402
from hivemind.boinc.real_client.server_stack import Stack  # noqa: E402

STD, BIG = moe_profiles.PROFILES["standard"], moe_profiles.PROFILES["big"]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_profiles_are_separate_apps_with_a_strictly_bigger_big_one() -> None:
    assert STD["app"] != BIG["app"] and STD["plan_class"] != BIG["plan_class"]
    assert STD["prefix"] != BIG["prefix"] and STD["chain_stem"] != BIG["chain_stem"]
    assert not BIG["prefix"].startswith(STD["prefix"]) and not STD["prefix"].startswith(BIG["prefix"])  # the standard pending() regex must not match big names
    for k in ("min_gpu_ram_mb", "seq_len", "n_seq", "micro_batch", "rank", "alpha"):
        assert BIG[k] > STD[k], k
    assert moe_profiles.TRAIN_APPS == (STD["app"], BIG["app"])
    assert moe_profiles.profile_for_app(BIG["app"]) is BIG
    with pytest.raises(KeyError):
        moe_profiles.profile_for_app("hivemind_moe_infer")


def test_flops_estimate_keeps_the_projects_reference_for_standard_and_scales_with_tokens() -> None:
    assert moe_profiles.flops_estimate("standard") == pytest.approx(4.8e14)
    tokens = lambda p: p["inner_steps"] * p["micro_batch"] * p["seq_len"]  # noqa: E731
    assert moe_profiles.flops_estimate("big") / moe_profiles.flops_estimate("standard") == pytest.approx(tokens(BIG) / tokens(STD))


def test_the_three_node_gpus_land_in_the_right_class() -> None:
    gib = {"RTX 3090": 24, "RTX 4070": 12, "RTX 2070": 8, "T4": 15}
    can = lambda gpu, p: gib[gpu] * 1024 >= p["min_gpu_ram_mb"]  # noqa: E731
    assert [g for g in gib if can(g, BIG)] == ["RTX 3090", "RTX 4070", "T4"]  # a T4 (15 GB) also qualifies; a 2070 (8 GB) stays on standard
    assert all(can(g, STD) for g in gib)


def test_plan_class_spec_declares_both_classes_with_their_memory_floor(tmp_path: Path) -> None:
    st = Stack(tmp_path, tmp_path, "python3", tmp_path)
    st.root.mkdir(parents=True, exist_ok=True)
    st.write_plan_class_spec()
    xml = (st.root / "plan_class_spec.xml").read_text()
    for p in (STD, BIG):
        block = xml.split(f"<name>{p['plan_class']}</name>")[1].split("</plan_class>")[0]
        assert f"<min_gpu_ram_mb>{p['min_gpu_ram_mb']}</min_gpu_ram_mb>" in block
    assert xml.count("<plan_class>") == 3 and "<name>cpu_hivemind_solver</name>" in xml


def test_install_app_puts_the_big_version_under_its_own_plan_class(tmp_path: Path) -> None:
    st = Stack(tmp_path, tmp_path, "python3", tmp_path)
    st.root.mkdir(parents=True, exist_ok=True)
    (st.root / "templates").mkdir()
    st.install_app(BIG["app"], "#!/bin/sh\n", 1, ["out.json", "out.pt"], 1000, version=101, plan_class=BIG["plan_class"], gpu_ram_mb=BIG["min_gpu_ram_mb"])
    st.install_app(STD["app"], "#!/bin/sh\n", 1, ["out.json", "out.pt"], 1000, version=101)
    big_v = st.root / "apps" / BIG["app"] / "101" / f"x86_64-pc-linux-gnu__{BIG['plan_class']}" / "version.xml"
    std_v = st.root / "apps" / STD["app"] / "101" / "x86_64-pc-linux-gnu__cuda_hivemind" / "version.xml"
    assert f"<plan_class>{BIG['plan_class']}</plan_class>" in big_v.read_text() and "<gpu_ram>10000000000</gpu_ram>" in big_v.read_text()
    assert "<plan_class>cuda_hivemind</plan_class>" in std_v.read_text() and "<gpu_ram>3500000000</gpu_ram>" in std_v.read_text()


def test_gpu_gate_refuses_a_small_gpu_and_an_unknown_app() -> None:
    spec = {"app": BIG["app"], "min_gpu_ram_mb": BIG["min_gpu_ram_mb"]}
    assert "needs a GPU with 10000 MiB" in moe_app.gpu_gate(spec, "train", 8192)
    assert moe_app.gpu_gate(spec, "train", 12288) is None
    assert moe_app.gpu_gate(spec, "infer", 100) is None
    assert "unknown training app" in moe_app.gpu_gate({"app": "evil"}, "train", 99999)
    assert moe_app.gpu_gate({"wu_id": "old-style spec without app or floor"}, "train", 1) is None  # workunits made before this change still run


def _fake_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(moe_work, "domain_ids", lambda tok, domain, role, shard, need: list(range(need)))


def test_big_spec_has_big_sizes_and_the_profile_overrides_explicit_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tokenizer(monkeypatch)
    spec = moe_work.build_train_spec("hmqb_00001", domain="science", profile="big", inner_steps=3, rank=1, seq_len=7)
    assert spec["app"] == BIG["app"] and spec["min_gpu_ram_mb"] == BIG["min_gpu_ram_mb"] and spec["profile"] == "big"
    assert spec["rank"] == 16 and spec["alpha"] == 32 and spec["inner_steps"] == 80 and spec["micro_batch"] == 4
    assert len(spec["train_tokens"]) == 128 and all(len(c) == 512 for c in spec["train_tokens"])
    assert set(spec["eval_sets"]) == set(moe_work.DOMAIN_ORDER) and all(len(c) == 512 for e in spec["eval_sets"].values() for c in e)


def test_a_spec_without_a_profile_is_exactly_the_old_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_tokenizer(monkeypatch)
    spec = moe_work.build_train_spec("hmq_00001", domain="science")
    assert "app" not in spec and "min_gpu_ram_mb" not in spec and "profile" not in spec
    assert spec["rank"] == 8 and len(spec["train_tokens"]) == 64 and len(spec["train_tokens"][0]) == 128


def test_a_bigger_shard_reads_proportionally_more_rows_and_stays_disjoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    f = tmp_path / "datasets--x--y" / "snapshots" / "abc" / "d.parquet"
    f.parent.mkdir(parents=True)
    pq.write_table(pa.table({"text": [f"r{i}" for i in range(4000)]}), f, row_group_size=100)
    monkeypatch.setattr(moe_work, "HUB", tmp_path)
    monkeypatch.setitem(moe_work._SOURCES, "_t", {"ds": "x--y", "train": "d.parquet", "eval": "d.parquet", "cols": ["text"], "rows": 10, "eval_start": -200})
    seen: list[tuple[int, int]] = []
    real = moe_work._rows
    monkeypatch.setattr(moe_work, "_rows", lambda path, cols, start, n: (seen.append((start, n)), real(path, cols, start, n))[1])
    tok = lambda s: type("T", (), {"input_ids": [1] * len(s)})()  # noqa: E731
    moe_work.domain_ids(tok, "_t", "train", 0, 64 * 128)
    moe_work.domain_ids(tok, "_t", "train", 2, 128 * 512)  # 8x the tokens of the base shard
    assert seen == [(0, 10), (2 * 80, 80)]
    moe_work.domain_ids(tok, "_t", "train", 10**6, 128 * 512)
    start, n = seen[-1]
    assert start + n <= 4000 - 200  # a runaway cursor wraps inside the training region


def _result_files(tmp_path: Path, app: str, wu: str) -> list[str]:
    state = {"w": torch.arange(4, dtype=torch.float32)}
    pt = tmp_path / f"{wu}.pt"
    torch.save({"states": [state, state]}, pt)
    js = tmp_path / f"{wu}.json"
    js.write_text(json.dumps({"app": app, "wu_id": wu, "eval_loss_before": 2.0, "eval_loss_after": 1.5, "domain": "science"}))
    return [str(js), str(pt)]


def test_each_app_publishes_its_own_chain_head(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ledger = tmp_path / "ledger"
    monkeypatch.setattr(bss, "LEDGER_DIR", ledger)
    assert bss.assimilate(_result_files(tmp_path, BIG["app"], "hmqb_00001")) == 0
    assert (ledger / "latest_adapter_big.pt").exists() and not (ledger / "latest_adapter.pt").exists()
    assert bss.assimilate(_result_files(tmp_path, STD["app"], "hmq_00001")) == 0
    head_std, head_big = (json.loads((ledger / f"{s}.json").read_text()) for s in ("latest_adapter", "latest_adapter_big"))
    assert head_std["wu_id"] == "hmq_00001" and head_big["wu_id"] == "hmqb_00001"
    rows = [json.loads(line) for line in (ledger / "assimilated.jsonl").read_text().splitlines()]
    assert [r["app"] for r in rows] == [BIG["app"], STD["app"]]


def test_a_result_for_the_wrong_app_is_rejected_by_the_validator(tmp_path: Path) -> None:
    meta, states = tmp_path / "m.json", tmp_path / "s.pt"
    meta.write_text(json.dumps({"wu_id": "hmqb_00001", "app": STD["app"]}))  # a standard-app result presented for a big workunit
    torch.save({"states": []}, states)
    verdict = MoEValidator().validate_train({"wu_id": "hmqb_00001", "app": BIG["app"]}, meta, states, "host")
    assert verdict.valid is False and "not for this workunit/app" in verdict.reason
    meta.write_text(json.dumps({"wu_id": "hmqb_00001", "app": BIG["app"]}))  # right app, so it gets past this check and fails a later one instead
    later = MoEValidator().validate_train({"wu_id": "hmqb_00001", "app": BIG["app"], "inner_steps": 80, "segment_steps": 10}, meta, states, "host")
    assert "not for this workunit/app" not in later.reason


def test_workgen_big_profile_uses_its_own_prefix_state_and_chain(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    wg = _load("t4_workgen")
    calls: list[str] = []

    class FakeStack:
        def __init__(self, *a, **k) -> None:
            self.root = tmp_path / "stack"
            self.dbname = "db"

        def ensure_chain_template(self, app: str = "hivemind_qlora_moe") -> str:
            return f"{app}_chain_in"

        def sql(self, q: str) -> str:
            calls.append(q)
            return "0\n"

        def create_work(self, app, name, inputs, fpops, command_line="", template=None) -> None:
            calls.append(f"create {app} {name} {fpops:.4g} {template} {sorted(inputs)}")

    monkeypatch.setattr(wg, "Stack", FakeStack)
    monkeypatch.setattr(wg, "build_train_spec", lambda name, **k: {"name": name, "profile": k.get("profile")})
    (tmp_path / "stack" / "hivemind_specs").mkdir(parents=True)
    monkeypatch.setattr(sys, "argv", ["t4_workgen.py", "--profile", "big", "--once", "--stack-base", str(tmp_path), "--boinc-src", str(tmp_path)])
    assert wg.main() == 0
    assert any("a.name='hivemind_qlora_moe_big'" in c and "regexp '^hmqb_'" in c for c in calls)
    made = [c for c in calls if c.startswith("create")][0]
    assert made.startswith("create hivemind_qlora_moe_big hmqb_00001 3.84e+15 None ['in.json']")  # first round: no chain head, the app's own default template
    ledger = tmp_path / "stack" / "hivemind_ledger"
    assert (ledger / "workgen_state_big.json").exists() and not (ledger / "workgen_state.json").exists()
    # a standard-chain head must be ignored by the big generator; only latest_adapter_big.* continues the big chain
    (ledger / "latest_adapter.json").write_text(json.dumps({"sha256": "std"}))
    (ledger / "latest_adapter.pt").write_bytes(b"std")
    (ledger / "latest_adapter_big.json").write_text(json.dumps({"sha256": "big"}))
    (ledger / "latest_adapter_big.pt").write_bytes(b"big")
    calls.clear()
    assert wg.main() == 0
    second = [c for c in calls if c.startswith("create")][0]
    assert second.startswith("create hivemind_qlora_moe_big hmqb_00002 ") and "hivemind_qlora_moe_big_chain_in" in second and "'adapter.pt'" in second
