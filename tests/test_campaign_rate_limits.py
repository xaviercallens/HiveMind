import os

from hivemind.boinc import hivemind_at_home_dashboard as dash


def test_env_int_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("HIVEMIND_RATE_GET_PER_MIN", raising=False)
    assert dash._env_int("HIVEMIND_RATE_GET_PER_MIN", 3000) == 3000


def test_env_int_reads_valid_value(monkeypatch):
    monkeypatch.setenv("HIVEMIND_RATE_GET_PER_MIN", "5000")
    assert dash._env_int("HIVEMIND_RATE_GET_PER_MIN", 3000) == 5000


def test_env_int_rejects_garbage_and_non_positive(monkeypatch):
    monkeypatch.setenv("HIVEMIND_RATE_POST_PER_MIN", "lots")
    assert dash._env_int("HIVEMIND_RATE_POST_PER_MIN", 900) == 900
    monkeypatch.setenv("HIVEMIND_RATE_POST_PER_MIN", "0")
    assert dash._env_int("HIVEMIND_RATE_POST_PER_MIN", 900) == 900
