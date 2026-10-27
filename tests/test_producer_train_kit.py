"""Producer training kit: which GPUs may train what, and the governor/attach files it renders."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hivemind.boinc.real_client.moe_profiles import PROFILES, profiles_for_gpu  # noqa: E402

_spec = importlib.util.spec_from_file_location("producer_train_kit", ROOT / "scripts" / "producer_train_kit.py")
kit = importlib.util.module_from_spec(_spec)
sys.modules["producer_train_kit"] = kit
_spec.loader.exec_module(kit)


@pytest.mark.parametrize("name,mib,expected", [
    ("RTX 3090", 24576, ["big", "standard"]), ("RTX 4070", 12282, ["big", "standard"]), ("T4", 15360, ["big", "standard"]),
    ("RTX 2070", 8192, ["standard"]), ("GTX 1050", 2048, []),
])
def test_profiles_per_gpu(name: str, mib: int, expected: list[str]) -> None:
    assert profiles_for_gpu(mib) == expected, name


def test_the_floor_is_inclusive_and_exact() -> None:
    big, std = PROFILES["big"]["min_gpu_ram_mb"], PROFILES["standard"]["min_gpu_ram_mb"]
    assert profiles_for_gpu(big) == ["big", "standard"] and profiles_for_gpu(big - 1) == ["standard"]
    assert profiles_for_gpu(std) == ["standard"] and profiles_for_gpu(std - 1) == []


def test_parse_nvidia_smi_takes_the_largest_card_and_ignores_junk() -> None:
    assert kit.parse_gpu_mib("NVIDIA GeForce RTX 3090, 24576\nNVIDIA GeForce RTX 2070, 8192\n") == 24576
    assert kit.parse_gpu_mib("Tesla T4, 15360") == 15360
    assert kit.parse_gpu_mib("") is None and kit.parse_gpu_mib("No devices were found") is None and kit.parse_gpu_mib("x, n/a") is None


def test_governor_unit_carries_the_owner_policy_and_this_hosts_paths() -> None:
    unit = kit.render_governor_unit(Path("/opt/hivemind/HiveMind"), 24576)
    for frag in ("--idle-start-s 1800", "--day-idle-s 7200", "--night-tz Europe/Paris", "--night-start-h 22", "--night-end-h 7",
                 "--boinccmd /usr/bin/boinccmd", "--boinc-dir /var/lib/boinc-client", "--rpc-port 31416", "/opt/hivemind/HiveMind/scripts/t4_governor.py"):
        assert frag in unit, frag
    assert f"--min-free-mib {PROFILES['standard']['min_gpu_ram_mb'] + kit.FREE_VRAM_MARGIN_MIB}" in unit  # starts as soon as the smallest profile fits
    assert "Requires=boinc-client.service" in unit and "Restart=on-failure" in unit


def test_governor_unit_refuses_a_gpu_that_is_too_small() -> None:
    with pytest.raises(ValueError):
        kit.render_governor_unit(Path("/x"), 2048)


def test_setup_text_never_contains_a_key_and_reads_it_from_the_environment() -> None:
    txt = kit.render_setup("https://boinc.example/hmproj/", 12282, Path("/tmp/k"), Path("/opt/hivemind/HiveMind"))
    assert 'boinccmd --project_attach https://boinc.example/hmproj/ "$HIVEMIND_BOINC_ACCOUNT_KEY"' in txt
    assert "big (hivemind_qlora_moe_big" in txt and "standard (hivemind_qlora_moe" in txt
    assert "touch /var/lib/hivemind-producer/HOLD" in txt


def test_cli_writes_the_three_files_and_reports_the_profiles(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert kit.main(["--server-url", "https://b/p/", "--out", str(tmp_path / "k"), "--gpu-mib", "8192"]) == 0
    assert sorted(p.name for p in (tmp_path / "k").iterdir()) == ["SETUP.txt", "governor.service", "nginx_status.conf"]
    assert "eligible: standard" in capsys.readouterr().out
    assert "stub_status" in (tmp_path / "k" / "nginx_status.conf").read_text()


def test_cli_exit_codes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert kit.main(["--server-url", "u", "--out", str(tmp_path / "a"), "--gpu-mib", "1024"]) == 3  # too small
    monkeypatch.setattr(kit, "detect_gpu_mib", lambda: None)
    assert kit.main(["--server-url", "u", "--out", str(tmp_path / "b")]) == 2  # no GPU found
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()  # nothing written when refusing
