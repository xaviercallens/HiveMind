# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The stock-BOINC settings generator: what a gaming PC gets must pause the GPU for the player, and a game name can never inject markup."""

from __future__ import annotations

import importlib.util
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_volunteer_config.py"
_spec = importlib.util.spec_from_file_location("hivemind_volunteer_config", _PATH)
assert _spec is not None and _spec.loader is not None
vc = importlib.util.module_from_spec(_spec)
sys.modules["hivemind_volunteer_config"] = vc
_spec.loader.exec_module(vc)


def prefs(profile: str) -> dict[str, str]:
    return {e.tag: (e.text or "") for e in ET.fromstring(vc.prefs_xml(profile))}


def test_gamer_never_runs_gpu_work_while_the_user_is_active_and_frees_gpu_memory_when_paused() -> None:
    p = prefs("gamer")
    assert p["run_gpu_if_user_active"] == "0" and p["run_if_user_active"] == "0"
    assert float(p["idle_time_to_run"]) >= 3  # not the instant the mouse stops: a cutscene is not idle
    assert p["leave_apps_in_memory"] == "0"  # a paused task must give its video memory back to the game


def test_night_adds_the_window_and_dedicated_ignores_activity() -> None:
    n = prefs("night")
    assert (n["start_hour"], n["end_hour"]) == ("22", "7") and n["run_gpu_if_user_active"] == "0"
    d = prefs("dedicated")
    assert d["run_gpu_if_user_active"] == "1" and "start_hour" not in d


def test_every_profile_is_well_formed_and_the_gamer_one_is_the_most_conservative() -> None:
    for name in vc.PROFILES:
        assert ET.fromstring(vc.prefs_xml(name)).tag == "global_preferences"
    assert float(prefs("gamer")["max_ncpus_pct"]) <= 50 < float(prefs("dedicated")["max_ncpus_pct"])


def test_listed_games_become_exclusive_gpu_apps_and_the_default_is_empty() -> None:
    assert [e.text for e in ET.fromstring(vc.cc_config_xml(["cs2", "eldenring.exe"])).iter("exclusive_gpu_app")] == ["cs2", "eldenring.exe"]
    assert list(ET.fromstring(vc.cc_config_xml([])).iter("exclusive_gpu_app")) == []  # no invented list of games


@pytest.mark.parametrize("bad", ["", "<x/>", "a/b", "..\\x", "a&b", "x" * 65, "-lead", "a\nb", "<!--", "a..b"])
def test_a_game_name_that_is_not_a_plain_program_name_is_refused(bad: str) -> None:
    with pytest.raises(vc.ConfigError):
        vc.check_game(bad)
    with pytest.raises(vc.ConfigError):
        vc.cc_config_xml([bad])


def test_apply_keeps_a_backup_and_refuses_a_directory_that_is_not_a_boinc_dir(tmp_path: Path) -> None:
    (tmp_path / "cc_config.xml").write_text("<cc_config>mine</cc_config>")
    done = vc.apply(tmp_path, vc.prefs_xml("gamer"), vc.cc_config_xml(["cs2"]))
    assert len(done) == 2 and (tmp_path / "cc_config.xml.hivemind-backup").read_text() == "<cc_config>mine</cc_config>"
    with pytest.raises(vc.ConfigError, match="not a directory"):
        vc.apply(tmp_path / "missing", "", "")


def test_cli_prints_both_files_and_refuses_bad_input(capsys: pytest.CaptureFixture[str]) -> None:
    assert vc.main(["gamer", "--game", "cs2"]) == 0
    out = capsys.readouterr().out
    assert "global_prefs_override.xml" in out and "<exclusive_gpu_app>cs2</exclusive_gpu_app>" in out
    assert vc.main(["gamer", "--game", "<bad>"]) == 2
    assert "refused" in capsys.readouterr().err


def test_long_game_names_also_get_their_15_character_form_because_linux_truncates_process_names() -> None:
    """Measured on the real client: it compares what the kernel reports, which is cut to 15 characters, so a 16+ character name would never match."""
    assert vc.with_linux_names(["eldenring.exe"]) == ["eldenring.exe"]  # 13: unchanged
    assert vc.with_linux_names(["hivemindfakegame"]) == ["hivemindfakegame", "hivemindfakegam"]
    games = [e.text for e in ET.fromstring(vc.cc_config_xml(["cyberpunk2077.exe"])).iter("exclusive_gpu_app")]
    assert games == ["cyberpunk2077.exe", "cyberpunk2077.e"]
    assert vc.with_linux_names(["a" * 15, "a" * 15]) == ["a" * 15]  # no duplicates
