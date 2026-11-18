"""GWAYA-Lite assistant (scripts/gwaya_lite.py): verdict wording, ladder, residency, gateway.  CPU only, fake Ollama (scripts/e16_fake_ollama.py), no real model."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import re
import signal
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

_CUDA_VISIBLE_DEVICES_BEFORE = os.environ.get("CUDA_VISIBLE_DEVICES")
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # for the imports below only; restored after them
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import e15_gate as gate_mod  # noqa: E402
import e16_fake_ollama as fk  # noqa: E402
import e16_ollama as oll  # noqa: E402
import gwaya_lite as G  # noqa: E402

# Restored right after the imports above: setting it at module level hid the GPU from every test module collected
# after this one (IP-179); the autouse fixture below keeps this module's own tests CPU-only at run time.
if _CUDA_VISIBLE_DEVICES_BEFORE is None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = _CUDA_VISIBLE_DEVICES_BEFORE


@pytest.fixture(autouse=True)
def _cpu_only_for_this_module(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

PROMPT = "Write a function add_two(a, b) that returns the sum of a and b."
REF = "def add_two(a, b):\n    return a + b\n"
T1, T2 = "assert add_two(1, 2) == 3", "assert add_two(2, 2) == 4"
M15, M3, M7 = G.TIER_MODELS["1.5b"], G.TIER_MODELS["3b"], G.TIER_MODELS["7b"]

needs_sandbox = pytest.mark.skipif(not gate_mod.gate_available(), reason="E15 sandbox unavailable on this host")


@pytest.fixture
def fake_factory():
    made = []

    def make(**kw):
        f = fk.FakeOllama({PROMPT: REF}, **kw)
        f.url = f.start()
        made.append(f)
        return f

    yield make
    for f in made:
        f.stop()
    gate_mod.set_sandbox_enabled(True)


def asst(f, ladder=("1.5b", "3b", "7b"), rounds=3, **kw):
    return G.Assistant(oll.Client(f.url, retries=0), ladder, 4096, rounds, **kw)


def run_main(argv, f):
    return G.main(argv, oll.Client(f.url, retries=0))


# ---------------------------------------------------------------- verdicts
@needs_sandbox
def test_verified_with_tests(fake_factory):
    f = fake_factory(ability={"1.5b": 1.0})
    r = asst(f).solve(PROMPT, [T1, T2])
    assert r.verdict == G.VERIFIED and r.tier == "1.5b" and r.candidates == 1 and r.repair_rounds == 0 and r.n_tests == 2
    assert r.detail.startswith("VERIFIED (zero-stub audit clean AND 2 test(s) passed in the sandbox)")
    assert "add_two" in r.code and r.gate_verdict == "ACCEPTED" and r.tokens > 0
    assert G.EXIT_CODES[r.verdict] == 0


@needs_sandbox
def test_audited_without_tests_never_says_verified(fake_factory):
    f = fake_factory(ability={"1.5b": 1.0})
    r = asst(f).solve(PROMPT, [])
    assert r.verdict == G.AUDITED
    assert r.detail == "AUDITED (syntax + zero-stub audit only; NOT tested)"
    assert "VERIFIED" not in G.render_provenance(r).split("NOTE:")[0]


def test_unverified_when_sandbox_disabled(fake_factory, capsys):
    f = fake_factory(ability={"1.5b": 1.0})
    gate_mod.set_sandbox_enabled(False)
    r = asst(f).solve(PROMPT, [T1])
    assert r.verdict == G.UNVERIFIED and r.detail.startswith("UNVERIFIED (the gate could not run")
    assert r.tier == "1.5b" and r.candidates == 1  # fail-closed: stops, does not escalate or claim anything
    assert run_main(["ask", PROMPT, "--test", T1], f) == 2
    out = capsys.readouterr().out
    assert "UNVERIFIED" in out and "verdict: VERIFIED" not in out


@needs_sandbox
def test_failed_after_whole_ladder(fake_factory, capsys):
    f = fake_factory(ability={"1.5b": 0, "3b": 0, "7b": 0}, repair_ability=0, wrong="stub")
    r = asst(f).solve(PROMPT, [T1])
    assert r.verdict == G.FAILED and r.detail.startswith("FAILED (no candidate passed the gate")
    assert [t.tier for t in r.tiers] == ["1.5b", "3b", "7b"]
    assert all(t.candidates == 6 and t.repair_rounds == 3 and not t.passed for t in r.tiers)  # greedy + 2 samples + 3 repairs per tier
    assert r.error and "Zero-stub" in r.error
    assert run_main(["ask", PROMPT, "--test", T1], f) == 1
    assert "FAILED" in capsys.readouterr().out


@needs_sandbox
def test_hidden_second_test_blocks_cheating_solution(fake_factory):
    f = fake_factory(ability={"1.5b": 0, "3b": 0, "7b": 0}, repair_ability=0, wrong="cheat")  # returns the constant of the visible test
    r = asst(f, ladder=("1.5b",)).solve(PROMPT, [T1, T2])
    assert r.verdict == G.FAILED and "Test run failed" in r.error and "test 1 failed" in r.error


# ---------------------------------------------------------------- ladder
@needs_sandbox
def test_escalation_order_and_stop(fake_factory):
    f = fake_factory(ability={"1.5b": 0, "3b": 0, "7b": 1.0}, repair_ability=0, wrong="stub")
    r = asst(f).solve(PROMPT, [T1])
    assert r.verdict == G.VERIFIED and r.tier == "7b"
    order = [c["model"] for c in f.calls]
    assert order == [M15] * 6 + [M3] * 6 + [M7]  # a tier's whole loop runs before the next tier starts; 7b stops at its first pass


@needs_sandbox
def test_no_escalation_when_first_tier_passes(fake_factory):
    f = fake_factory(ability={"1.5b": 1.0, "3b": 1.0, "7b": 1.0})
    asst(f).solve(PROMPT, [T1])
    assert {c["model"] for c in f.calls} == {M15}


@needs_sandbox
def test_repair_round_fixes_and_zero_rounds_option(fake_factory):
    f = fake_factory(ability={"1.5b": 0, "3b": 0, "7b": 0}, repair_ability=1.0, wrong="stub")
    r = asst(f, ladder=("1.5b",)).solve(PROMPT, [T1])
    assert r.verdict == G.VERIFIED and r.repair_rounds == 1 and r.candidates == 4  # greedy + 2 samples + repair 1
    r0 = asst(f, ladder=("1.5b",), rounds=0).solve(PROMPT, [T1])
    assert r0.verdict == G.FAILED and r0.candidates == 3 and r0.repair_rounds == 0


@needs_sandbox
def test_candidate_protocol_matches_e16(fake_factory):
    f = fake_factory(ability={"1.5b": 0}, repair_ability=0, wrong="stub")
    asst(f, ladder=("1.5b",)).solve(PROMPT, [T1])
    temps = [c["temperature"] for c in f.calls]
    assert temps == [0.0, 0.7, 0.7, 0.0, 0.0, 0.0]
    assert len({c["seed"] for c in f.calls}) == 6 and all(c["ctx"] == 4096 for c in f.calls)


def test_parse_ladder_and_validate_tests():
    assert G.parse_ladder("1.5B, 7b,1.5b") == ["1.5b", "7b"]
    with pytest.raises(G.GwayaLiteError):
        G.parse_ladder("1.5b,70b")
    assert G.validate_tests(["assert f(1)==2", "", "# note"]) == ["assert f(1)==2"]
    for bad in (["import os"], ["assert x\nimport os"], ["print(1)"], [5]):
        with pytest.raises(G.GwayaLiteError):
            G.validate_tests(bad)


# ---------------------------------------------------------------- residency / GPU discipline
TABLE = {"1.5b": {4096: 100}, "3b": {4096: 200}, "7b": {4096: 700}}


def test_planning_table_comes_from_e16_report():
    t = G.load_planning_table()
    assert (t["1.5b"][4096], t["3b"][4096], t["7b"][4096]) == (1651, 2713, 5483)
    assert t["7b"][8192] == 6387 and G.need_mib(t, "7b", 8192) == 6387
    assert G.need_mib(t, "3b", 16384) > t["3b"][4096]
    assert G.load_planning_table(ROOT / "nonexistent.json")["7b"][4096] == 5483


def test_choose_unloads_and_plan():
    assert G.choose_unloads({"a": 100, "b": 700}, 200, 800) == ["b"]
    assert G.choose_unloads({"a": 100}, 200, 800) == []
    assert G.choose_unloads({"a": 300, "b": 300}, 300, 800) == ["a"]
    plan = G.plan_sequence(["1.5b", "3b", "7b"], 4096, 7500, G.load_planning_table())
    assert [s["unload_first"] for s in plan] == [[], [], ["3b"]] and plan[-1]["resident_after"] == ["1.5b", "7b"] and plan[-1]["total_mib"] == 7134
    assert all(s["fits"] for s in plan)


def test_residency_unloads_when_co_residence_exceeds_budget(fake_factory):
    f = fake_factory(preloaded=[(M7, 4096)])
    res = G.Residency(oll.Client(f.url, retries=0))
    assert res.ensure("3b", 4096) == [M7]  # 5483 + 2713 > 7500
    assert not f.loaded
    f2 = fake_factory(preloaded=[(M7, 4096)])
    assert G.Residency(oll.Client(f2.url, retries=0)).ensure("1.5b", 4096) == []  # 5483 + 1651 fits
    assert M7 in f2.loaded


def test_residency_with_fake_memory_table_and_foreign_model(fake_factory):
    f = fake_factory(preloaded=[(M3, 4096), (M15, 4096)])
    res = G.Residency(oll.Client(f.url, retries=0), budget_mib=800, table=TABLE)
    assert res.ensure("7b", 4096) == [M3]  # largest first: 3b (200) leaves 100 + 700 = 800
    assert set(f.loaded) == {M15}
    with pytest.raises(G.ResidencyError):
        G.Residency(oll.Client(f.url, retries=0), budget_mib=500, table=TABLE).ensure("7b", 4096)  # alone above the budget


def test_residency_refuses_when_unload_cannot_be_verified(fake_factory):
    f = fake_factory(preloaded=[(M7, 4096)], stuck=[M7])
    res = G.Residency(oll.Client(f.url, retries=0, sleep=lambda s: None), unload_wait_s=0.2)
    with pytest.raises(G.ResidencyError):
        res.ensure("3b", 4096)


# ---------------------------------------------------------------- unload on exit
@needs_sandbox
def test_ask_unloads_after_run_and_keep_warm_does_not(fake_factory, capsys):
    f = fake_factory(ability={"1.5b": 1.0})
    run_main(["ask", PROMPT, "--test", T1], f)
    assert not f.loaded and M15 in f.ever_unloaded
    f2 = fake_factory(ability={"1.5b": 1.0})
    run_main(["ask", PROMPT, "--test", T1, "--keep-warm"], f2)
    assert M15 in f2.loaded


def test_unload_on_injected_exception(fake_factory):
    f = fake_factory(preloaded=[(M15, 4096)])
    client = oll.Client(f.url, retries=0)

    def boom(*a, **k):
        raise RuntimeError("injected")

    client.chat = boom
    args = G.build_parser().parse_args(["ask", PROMPT, "--test", T1])
    with pytest.raises(RuntimeError):
        G.cmd_ask(args, client)
    assert not f.loaded and M15 in f.ever_unloaded


def test_sigterm_becomes_systemexit_so_finally_runs():
    with pytest.raises(SystemExit) as ei:
        with G.signal_guard():
            os.kill(os.getpid(), signal.SIGTERM)
            time.sleep(1)
    assert ei.value.code == 128 + signal.SIGTERM


# ---------------------------------------------------------------- gateway
class Gw:
    def __init__(self, f, api_key=None, log_prompts=False, ladder=("1.5b",), **kw):
        self.logs = []
        self.asst = asst(f, ladder, **kw)
        self.gw = G.Gateway(self.asst, api_key, log_prompts, True, self.logs.append)
        self.httpd = G.build_server(self.gw, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def req(self, method, path, body=None, headers=None, raw=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=60)
        h = {"Content-Type": "application/json", **(headers or {})}
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        try:
            c.request(method, path, body=data, headers=h)
            r = c.getresponse()
            return r.status, r.read().decode()
        finally:
            c.close()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def gw_factory():
    made = []

    def make(f, **kw):
        g = Gw(f, **kw)
        made.append(g)
        return g

    yield make
    for g in made:
        g.close()


def chat_body(content=PROMPT, **kw):
    return {"model": "gwaya-lite", "messages": [{"role": "system", "content": "x"}, {"role": "user", "content": content}], **kw}


def test_health_and_models(fake_factory, gw_factory):
    g = gw_factory(fake_factory())
    s, b = g.req("GET", "/health")
    assert s == 200 and json.loads(b)["status"] == "ok"
    s, b = g.req("GET", "/v1/models")
    assert s == 200 and json.loads(b)["data"][0]["id"] == "gwaya-lite"
    assert g.req("GET", "/nope")[0] == 404


@needs_sandbox
def test_chat_non_stream_and_tests_block(fake_factory, gw_factory):
    f = fake_factory(ability={"1.5b": 1.0})
    g = gw_factory(f)
    content = PROMPT + f"\n\n```tests\n{T1}\n{T2}\n```\n"
    s, b = g.req("POST", "/v1/chat/completions", chat_body(content))
    o = json.loads(b)
    assert s == 200 and o["object"] == "chat.completion"
    msg = o["choices"][0]["message"]["content"]
    assert msg.startswith("```python\ndef add_two") and "GWAYA-Lite: VERIFIED" in msg and "5-7%" in msg
    assert o["gwaya"]["verdict"] == "VERIFIED" and o["gwaya"]["tier"] == "1.5b" and o["gwaya"]["tests"] == 2 and o["gwaya"]["candidates"] == 1
    # the tests block never reached the model
    assert "```tests" not in json.dumps(f.calls)
    # same via the gwaya extension field, no tests -> AUDITED
    s, b = g.req("POST", "/v1/chat/completions", chat_body(gwaya={"tests": [T1], "ladder": ["1.5b"]}))
    assert json.loads(b)["gwaya"]["verdict"] == "VERIFIED"
    s, b = g.req("POST", "/v1/chat/completions", chat_body())
    assert json.loads(b)["gwaya"]["verdict"] == "AUDITED"


@needs_sandbox
def test_chat_sse_stream(fake_factory, gw_factory):
    g = gw_factory(fake_factory(ability={"1.5b": 1.0}))
    c = http.client.HTTPConnection("127.0.0.1", g.port, timeout=60)
    c.request("POST", "/v1/chat/completions", json.dumps(chat_body(PROMPT + f"\n```tests\n{T1}\n```", stream=True)), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read().decode()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    frames = [x[len("data: "):] for x in raw.split("\n\n") if x.startswith("data: ")]
    assert frames[-1] == "[DONE]"
    objs = [json.loads(x) for x in frames[:-1]]
    assert all(o["object"] == "chat.completion.chunk" for o in objs) and objs[-1]["choices"][0]["finish_reason"] == "stop"
    text = "".join(o["choices"][0]["delta"].get("content", "") for o in objs)
    assert text.startswith("```python") and "VERIFIED" in text and len(objs) > 5
    assert objs[-1]["gwaya"]["verdict"] == "VERIFIED"


def test_tests_block_extraction():
    p, t = G.extract_tests_block("Do it.\n```tests\nassert f(1)==2\nassert f(2)==3\n```\nthanks")
    assert "```" not in p and "assert" not in p and p.startswith("Do it.") and t == ["assert f(1)==2", "assert f(2)==3"]
    assert G.extract_tests_block("no block") == ("no block", [])


def test_bad_requests(fake_factory, gw_factory):
    g = gw_factory(fake_factory())
    assert g.req("POST", "/v1/chat/completions", raw=b"{nope")[0] == 400
    assert g.req("POST", "/v1/chat/completions", {"messages": []})[0] == 400
    s, b = g.req("POST", "/v1/chat/completions", chat_body(PROMPT + "\n```tests\nimport os\n```"))
    assert s == 400 and "assert" in b
    assert g.req("POST", "/v1/chat/completions", chat_body(max_tokens=0))[0] == 400
    assert g.req("POST", "/v1/other", {})[0] == 404


def test_body_limit_64kib(fake_factory, gw_factory):
    g = gw_factory(fake_factory())
    s, b = g.req("POST", "/v1/chat/completions", raw=b'{"messages":[{"role":"user","content":"' + b"a" * 70000 + b'"}]}')
    assert s == 413 and "65536" in b


def test_max_tokens_clamped_and_requests_serialised(fake_factory, gw_factory):
    f = fake_factory(ability={"1.5b": 1.0})
    g = gw_factory(f)
    seen, active, peak = [], [0], [0]
    orig = g.asst.client.chat
    lk = threading.Lock()

    def spy(model, messages, options=None, keep_alive="2m"):
        with lk:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        seen.append(options["num_predict"])
        time.sleep(0.15)
        try:
            return orig(model, messages, options, keep_alive)
        finally:
            with lk:
                active[0] -= 1

    g.asst.client.chat = spy
    gate_mod.set_sandbox_enabled(False)  # fast path: gate says UNVERIFIED after one candidate; the lock behaviour is what is tested
    ths = [threading.Thread(target=g.req, args=("POST", "/v1/chat/completions", chat_body(max_tokens=99999))) for _ in range(3)]
    [t.start() for t in ths]
    [t.join(60) for t in ths]
    assert seen == [1024, 1024, 1024] and peak[0] == 1
    assert g.asst.num_predict == G.DEFAULT_TOKENS  # restored after each request


def test_auth_host_and_origin_guards(fake_factory, gw_factory):
    f = fake_factory()
    g = gw_factory(f, api_key="k" * 20)
    assert g.req("GET", "/health")[0] == 401
    assert g.req("GET", "/health", headers={"Authorization": "Bearer wrong"})[0] == 401
    assert g.req("GET", "/health", headers={"Authorization": "Bearer " + "k" * 20})[0] == 200
    g2 = gw_factory(f)
    assert g2.req("GET", "/health")[0] == 200  # loopback default: no auth
    assert g2.req("GET", "/health", headers={"Origin": "https://evil.example"})[0] == 403
    assert g2.req("GET", "/health", headers={"Origin": "http://localhost:3000"})[0] == 200
    assert g2.req("GET", "/health", headers={"Host": "evil.example"})[0] == 403


def test_bind_rules(tmp_path):
    key = tmp_path / "key"
    key.write_text("s" * 24 + "\n")
    short = tmp_path / "short"
    short.write_text("abc")
    assert G.check_bind("127.0.0.1", False, None) is None and G.check_bind("localhost", False, None) is None and G.check_bind("::1", False, None) is None
    for host in ("0.0.0.0", "192.168.1.5", "example.com"):
        with pytest.raises(G.GwayaLiteError):
            G.check_bind(host, False, None)
        with pytest.raises(G.GwayaLiteError):
            G.check_bind(host, True, None)  # --allow-nonlocal alone is not enough
        with pytest.raises(G.GwayaLiteError):
            G.check_bind(host, False, str(key))  # nor is the key alone
    assert G.check_bind("0.0.0.0", True, str(key)) == "s" * 24
    with pytest.raises(G.GwayaLiteError):
        G.check_bind("127.0.0.1", False, str(short))
    with pytest.raises(G.GwayaLiteError):
        G.check_bind("127.0.0.1", False, str(tmp_path / "missing"))


def test_serve_refuses_nonlocal_exit_3(fake_factory, capsys):
    f = fake_factory()
    assert run_main(["serve", "--host", "0.0.0.0", "--port", "0"], f) == 3
    assert "refusing to bind" in capsys.readouterr().err


def test_serve_shutdown_unloads_models(fake_factory):
    f = fake_factory(preloaded=[(M15, 4096)])
    holder = {}
    args = G.build_parser().parse_args(["serve", "--port", "0"])
    th = threading.Thread(target=lambda: holder.setdefault("rc", G.cmd_serve(args, oll.Client(f.url, retries=0), ready=lambda h: holder.setdefault("h", h))))
    th.start()
    for _ in range(100):
        if "h" in holder:
            break
        time.sleep(0.05)
    port = holder["h"].server_address[1]
    assert json.loads(urllib.request.build_opener(urllib.request.ProxyHandler({})).open(f"http://127.0.0.1:{port}/health", timeout=10).read())["status"] == "ok"
    holder["h"].shutdown()
    th.join(30)
    assert holder["rc"] == 0 and not f.loaded and M15 in f.ever_unloaded


@needs_sandbox
def test_no_prompt_logging_by_default(fake_factory, gw_factory):
    secret = "zz_secret_marker_function"
    f = fake_factory(ability={"1.5b": 1.0})
    f.answers = {secret: "def zz_secret_marker_function():\n    return 1\n"}
    g = gw_factory(f)
    g.req("POST", "/v1/chat/completions", chat_body(f"Write {secret}."))
    text = "\n".join(g.logs)
    assert "verdict=" in text and secret not in text and "prompt=" not in text
    g2 = gw_factory(f, log_prompts=True)
    g2.req("POST", "/v1/chat/completions", chat_body(f"Write {secret}."))
    assert secret in "\n".join(g2.logs)


# ---------------------------------------------------------------- doctor, exit codes, static checks
def test_doctor_with_missing_models(fake_factory, monkeypatch):
    f = fake_factory(missing=[M7, M3])
    monkeypatch.setattr(oll, "gpu_info", lambda *a, **k: {"name": "FakeGPU", "memory_total_mib": 15360, "driver": "0"})
    monkeypatch.setattr(oll, "gpu_memory_used_mib", lambda *a, **k: f.gpu_read())
    lines = []
    args = G.build_parser().parse_args(["doctor"])
    rc = G.cmd_doctor(args, oll.Client(f.url, retries=0), lines.append)
    out = "\n".join(lines)
    assert rc == 2 and f"ollama pull {M7}" in out and f"ollama pull {M3}" in out and f"ollama pull {M15}" not in out
    assert "gate / sandbox self-test" in out and "FakeGPU" in out and "max-resident budget 7500" in out and "status: NOT ready" in out
    assert "load   7b" in out and "resident after ['1.5b', '7b'] = 7134 MiB" in out


def test_doctor_ollama_down_is_exit_3(monkeypatch):
    lines = []
    args = G.build_parser().parse_args(["doctor", "--ollama-url", "http://127.0.0.1:9"])
    assert G.cmd_doctor(args, None, lines.append) == 3 and "not reachable" in lines[1]


def test_exit_3_cases(fake_factory, capsys):
    f = fake_factory(missing=[M3])
    for argv in ([], ["ask"], ["ask", "x", "--ladder", "9b"], ["bogus"], ["ask", "x", "--test", "import os"], ["ask", "x", "--ollama-url", "http://example.com:11434"]):
        try:
            rc = G.main(argv)
        except SystemExit as e:
            rc = e.code
        assert rc == 3, argv
    assert run_main(["ask", PROMPT, "--ladder", "3b", "--test", T1], f) == 3  # model not installed
    assert f"ollama pull {M3}" in capsys.readouterr().err


@needs_sandbox
def test_json_output_and_exit_0(fake_factory, capsys):
    f = fake_factory(ability={"1.5b": 1.0})
    assert run_main(["ask", PROMPT, "--test", T1, "--test", T2, "--json"], f) == 0
    d = json.loads(capsys.readouterr().out)
    assert d["verdict"] == "VERIFIED" and d["tier"] == "1.5b" and d["tests"] == 2 and "5-7%" in d["caveat"] and d["generated_tokens"] > 0
    assert "artifacts/experiments/E16_gwaya_lite_t4_report.json" in d["caveat"]


@needs_sandbox
def test_provenance_block_has_required_fields(fake_factory, capsys):
    f = fake_factory(ability={"1.5b": 0}, repair_ability=0, wrong="stub")
    run_main(["ask", PROMPT, "--test", T1, "--ladder", "1.5b"], f)
    out = capsys.readouterr().out
    for needle in ("PROVENANCE", "tier used:", "candidates tried: 6", "repair rounds: 3", "generated tokens:", "seconds:", "gate:", "5-7%", "E16_gwaya_lite_t4_report.json", "NOT GWAYA's Bubblewrap"):
        assert needle in out, needle


def test_no_dynamic_execution_or_shell_in_module():
    src = (ROOT / "scripts" / "gwaya_lite.py").read_text()
    assert not re.search(r"\b(exec|eval)\s*\(", src) and "shell=True" not in src and "import subprocess" not in src and "os.system" not in src
