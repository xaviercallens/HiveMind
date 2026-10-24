"""The consumer connector: editor settings that parse, keys that are never written by accident."""

from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

import httpx
import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "hivemind_connect.py"
_spec = importlib.util.spec_from_file_location("hivemind_connect", _PATH)
assert _spec is not None and _spec.loader is not None
connect: ModuleType = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(connect)

GATEWAY = "https://gw.example.org"


def _json_body(text: str) -> dict[str, object]:
    return json.loads("\n".join(line for line in text.splitlines() if not line.startswith("//")))


def test_json_snippets_are_valid_json_with_the_right_base_url_and_model() -> None:
    opencode = _json_body(connect.render_snippet("opencode", gateway=GATEWAY + "/", model="coder"))
    options = opencode["provider"]["hivemind"]["options"]  # type: ignore[index]
    assert options["baseURL"] == GATEWAY + "/v1" and options["apiKey"] == connect.KEY_PLACEHOLDER
    assert "hivemind/coder" in opencode["provider"]["hivemind"]["models"]  # type: ignore[index]
    zed = _json_body(connect.render_snippet("zed", gateway=GATEWAY, model="hivemind/moe-coder", max_tokens=4096))
    entry = zed["language_models"]["openai_compatible"]["HiveMind"]  # type: ignore[index]
    assert entry["api_url"] == GATEWAY + "/v1" and entry["available_models"][0]["name"] == "hivemind/moe-coder"
    assert entry["available_models"][0]["max_output_tokens"] == 4096


def test_continue_aider_python_and_env_snippets_carry_url_model_and_limit() -> None:
    cont = connect.render_snippet("continue", gateway=GATEWAY, model="coder-plus", max_tokens=3000)
    assert "provider: openai" in cont and f"apiBase: {GATEWAY}/v1" in cont and "model: hivemind/coder-plus" in cont and "maxTokens: 3000" in cont
    assert "aider --model openai/hivemind/coder" in connect.render_snippet("aider", gateway=GATEWAY, model="coder")
    py = connect.render_snippet("python", gateway=GATEWAY, model="auto")
    compile(py, "snippet", "exec")  # the Python snippet is valid Python
    assert 'model="hivemind/auto"' in py
    assert connect.render_snippet("env", gateway=GATEWAY, model="coder") == f"export OPENAI_BASE_URL={GATEWAY}/v1\nexport OPENAI_API_KEY=hm-YOUR_KEY\n"


def test_every_listed_tool_renders_and_unknown_tools_and_bad_limits_are_rejected() -> None:
    for tool in connect.TOOLS:
        assert GATEWAY + "/v1" in connect.render_snippet(tool, gateway=GATEWAY, model="coder")
    with pytest.raises(ValueError):
        connect.render_snippet("emacs", gateway=GATEWAY, model="coder")
    with pytest.raises(ValueError):
        connect.render_snippet("continue", gateway=GATEWAY, model="coder", max_tokens=0)


def test_snippet_command_uses_a_placeholder_unless_asked_to_embed_the_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv(connect.API_KEY_ENV, "hm-real-secret-key")
    assert connect.main(["--gateway", GATEWAY, "snippet", "env"]) == 0
    default = capsys.readouterr().out
    assert "hm-real-secret-key" not in default and connect.KEY_PLACEHOLDER in default
    assert connect.main(["--gateway", GATEWAY, "snippet", "env", "--embed-key"]) == 0
    assert "hm-real-secret-key" in capsys.readouterr().out


def test_register_prints_the_key_once_or_saves_it_owner_only_and_never_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        return httpx.Response(200, json={"account_id": "acc1", "api_key": "hm-new-key-123", "role": body["role"]})

    transport = httpx.MockTransport(handler)
    assert connect.main(["--gateway", GATEWAY, "register", "--handle", "me"], transport=transport) == 0
    assert "hm-new-key-123" in capsys.readouterr().out
    target = tmp_path / "key"
    assert connect.main(["--gateway", GATEWAY, "register", "--handle", "me", "--save-key", str(target)], transport=transport) == 0
    assert target.read_text() == "hm-new-key-123" and stat.S_IMODE(os.stat(target).st_mode) == 0o600
    assert "hm-new-key-123" not in capsys.readouterr().out
    assert connect.main(["--gateway", GATEWAY, "register", "--handle", "me", "--save-key", str(target)], transport=transport) == 1


def test_check_reports_balance_live_models_and_tiers_and_rejects_a_bad_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    tiers = [{"public_id": "hivemind/coder", "moe": False, "min_vram_gb": 6.0, "good_for": "python"},
             {"public_id": "hivemind/moe-coder", "moe": True, "min_vram_gb": 24.0, "good_for": "agents"}]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/v1/hivemind/me":
            if request.headers["authorization"] != "Bearer hm-good":
                return httpx.Response(401, json={})
            return httpx.Response(200, json={"account_id": "a1", "role": "prosumer", "balance": 0.0, "max_price_per_1k": 2.0})
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "hivemind/coder"}]})
        return httpx.Response(200, json={"tiers": tiers})

    transport = httpx.MockTransport(handler)
    monkeypatch.setenv(connect.API_KEY_ENV, "hm-good")
    assert connect.main(["--gateway", GATEWAY, "check"], transport=transport) == 0
    out = capsys.readouterr().out
    assert "hivemind/coder" in out and "[live]" in out and "no producer yet" in out and "balance is 0" in out
    monkeypatch.setenv(connect.API_KEY_ENV, "hm-bad")
    assert connect.main(["--gateway", GATEWAY, "check"], transport=transport) == 1


def test_register_google_with_token_saves_key_or_prints(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/hivemind/register/google"
        body = json.loads(request.content)
        assert body["id_token"] == "valid.google.jwt"
        return httpx.Response(200, json={"account_id": "google:acc99", "api_key": "hm-gkey-999", "role": body["role"], "created": True})

    transport = httpx.MockTransport(handler)
    # direct print
    assert connect.main(["--gateway", GATEWAY, "register-google", "--token", "valid.google.jwt"], transport=transport) == 0
    out = capsys.readouterr().out
    assert "Account created with Google!" in out and "hm-gkey-999" in out and "account google:acc99" in out

    # save-key with 0o600 mode
    target = tmp_path / "gkey"
    assert connect.main(["--gateway", GATEWAY, "login-google", "--token", "valid.google.jwt", "--save-key", str(target)], transport=transport) == 0
    assert target.read_text() == "hm-gkey-999" and stat.S_IMODE(os.stat(target).st_mode) == 0o600


def test_register_google_handles_errors_and_missing_token(capsys: pytest.CaptureFixture[str]) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Google sign-in failed. Sign in again."}})

    transport = httpx.MockTransport(handler)
    # error from gateway
    assert connect.main(["--gateway", GATEWAY, "register-google", "--token", "expired.jwt"], transport=transport) == 1
    err = capsys.readouterr().err
    assert "Google sign-in failed (401)" in err

    # missing token
    with pytest.raises(SystemExit):
        connect.main(["--gateway", GATEWAY, "register-google"])

