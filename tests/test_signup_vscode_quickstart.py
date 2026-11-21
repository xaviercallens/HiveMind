# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Unit, security, and DOM-execution tests for SLICE-02 (/signup VS Code Continue.dev quickstart & /v1/models key check)."""

from __future__ import annotations

import json
import re
import subprocess
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import pytest
import yaml
from fastapi.testclient import TestClient
from hivemind.router.accounts import RouterStore
from hivemind.router.gateway import create_router_app
from hivemind.router.signup_page import (
    _AUTH_JS,
    _GOOGLE_JS,
    DEFAULT_KEY_PLACEHOLDER,
    DEFAULT_PUBLIC_API_BASE,
    PERSONAL_PC_COMPLIANCE_NOTE,
    _reading,
    build_continue_config_yaml,
    content_security_policy,
    render_signup_page,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SIGNUP_PAGE_PATH = REPO_ROOT / "src" / "hivemind" / "router" / "signup_page.py"
STATIC_FIGURE_RE = re.compile(r"\d[\d.,]*\s?(?:ms\b|tok/s|HTTP Reqs|Completions|\bs/round)")
TEST_GOOGLE_CLIENT_ID = "1234567890-abcdef123456.apps.googleusercontent.com"
SECRET = b"slice02-signup-vscode-test-secret-32b!"


class _SectionInspector(HTMLParser):
    """Lightweight HTML5 tree inspector verifying nesting of #result and nav links."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._stack: list[tuple[str, dict[str, str]]] = []
        self.ids_inside_result: set[str] = set()
        self.nav_links: list[tuple[str, str]] = []
        self._in_quick_nav = False
        self._current_nav_href: str | None = None
        self._current_nav_text: list[str] = []
        self.continue_code_parent_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {k: (v or "") for k, v in attrs}
        in_result = any(t == "section" and a.get("id") == "result" for t, a in self._stack)
        if in_result and attr_map.get("id"):
            self.ids_inside_result.add(attr_map["id"])
        if tag == "code" and attr_map.get("id") == "continueYamlSnippet":
            self.continue_code_parent_tags = [t for t, _ in self._stack]
        if tag == "nav" and attr_map.get("data-testid") == "signup-quick-nav":
            self._in_quick_nav = True
        if self._in_quick_nav and tag == "a":
            self._current_nav_href = attr_map.get("href", "")
            self._current_nav_text = []
        if tag not in {"meta", "input", "br", "hr", "img", "link"}:
            self._stack.append((tag, attr_map))

    def handle_data(self, data: str) -> None:
        if self._in_quick_nav and self._current_nav_href is not None:
            self._current_nav_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_quick_nav and tag == "a" and self._current_nav_href is not None:
            self.nav_links.append((self._current_nav_href, "".join(self._current_nav_text).strip()))
            self._current_nav_href = None
            self._current_nav_text = []
        if tag == "nav" and self._in_quick_nav:
            self._in_quick_nav = False
        for idx in range(len(self._stack) - 1, -1, -1):
            if self._stack[idx][0] == tag:
                del self._stack[idx:]
                break


def test_agpl_header_static_numbers_and_allowed_hosts_invariants() -> None:
    source = SIGNUP_PAGE_PATH.read_text(encoding="utf-8")
    expected_header = "# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0."
    assert expected_header in source[:300]

    # Static performance/telemetry claims gate (must remain 0)
    assert STATIC_FIGURE_RE.findall(source) == []

    # Never use unsafe DOM sinks or POST /v1/chat/completions
    for forbidden in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "/v1/chat/completions"):
        assert forbidden not in source

    # Check all embedded literal URLs in signup_page.py reference only permitted hosts
    for match in re.findall(r"https?://[^\s'\"<>]+", source):
        host = (urlparse(match).hostname or "").lower()
        assert (
            host.endswith(".run.app")
            or host.endswith(".googleapis.com")
            or host == "accounts.google.com"
            or host == "datasets-server.huggingface.co"
            or host in {"127.0.0.1", "localhost"}
        ), f"Unexpected host {host!r} in {match!r}"


def test_build_continue_config_yaml_valid_outputs_and_normalization() -> None:
    yaml_text = build_continue_config_yaml(
        "hm-AbCdEf0123456789_xyz",
        "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app",
    )
    assert PERSONAL_PC_COMPLIANCE_NOTE in yaml_text
    parsed = yaml.safe_load(yaml_text)
    assert parsed["name"] == "HiveMind OpenRouter"
    assert parsed["schema"] == "v1"
    assert len(parsed["models"]) == 2
    coder = parsed["models"][0]
    assert coder["provider"] == "openai"
    assert coder["model"] == "hivemind/coder"
    assert coder["apiBase"] == "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app/v1"
    assert coder["apiKey"] == "hm-AbCdEf0123456789_xyz"
    assert coder["roles"] == ["chat", "edit", "apply"]
    assert coder["defaultCompletionOptions"]["maxTokens"] == 2048

    # Already ending in /v1 or /v1/ does not duplicate /v1, and hm_live_ keys are accepted
    yaml_v1 = build_continue_config_yaml("  hm_live_xavier_994a102f88c  ", "http://127.0.0.1:8080/v1/")
    parsed_v1 = yaml.safe_load(yaml_v1)
    assert parsed_v1["models"][0]["apiBase"] == "http://127.0.0.1:8080/v1"
    assert parsed_v1["models"][0]["apiKey"] == "hm_live_xavier_994a102f88c"


@pytest.mark.parametrize(
    ("bad_key", "bad_base", "err_match"),
    [
        (None, DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("   ", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("sk-not-hivemind-key", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("hm-key\n    injected: true", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("hm-key with spaces", DEFAULT_PUBLIC_API_BASE, "api_key"),
        ("hm-valid_key_123", None, "api_base"),
        ("hm-valid_key_123", "   ", "api_base"),
        ("hm-valid_key_123", "https://bad host.run.app/v1", "api_base"),
        ("hm-valid_key_123", "https://bad.run.app/v1\ninjected: 1", "api_base"),
        ("hm-valid_key_123", 'https://bad.run.app/"quote"/v1', "api_base"),
        ("hm-valid_key_123", "javascript:alert(1)", "api_base"),
        ("hm-valid_key_123", "ftp://files.example.org/v1", "api_base"),
        ("hm-valid_key_123", "not-a-valid-url", "api_base"),
        ("hm-valid_key_123", "https://user:secret@example.run.app/v1", "api_base"),
        ("hm-valid_key_123", "https://example.run.app/v1?token=1", "api_base"),
        ("hm-valid_key_123", "https://example.run.app/v1#frag", "api_base"),
    ],
)
def test_build_continue_config_yaml_rejects_invalid_inputs(bad_key: object, bad_base: object, err_match: str) -> None:
    with pytest.raises(ValueError, match=err_match):
        build_continue_config_yaml(bad_key, bad_base)  # type: ignore[arg-type]


def test_csp_and_reading_validation() -> None:
    csp = content_security_policy("Nonce_Token-2026+ok=")
    assert "default-src 'none'" in csp
    assert "script-src 'nonce-Nonce_Token-2026+ok=' https://accounts.google.com/gsi/client" in csp
    assert "connect-src 'self' https://accounts.google.com/gsi/" in csp
    assert "frame-ancestors 'none'" in csp
    assert "'unsafe-inline'" not in csp and "'unsafe-eval'" not in csp

    for invalid_nonce in ("", "bad nonce", "nonce;script-src *", None):
        with pytest.raises(ValueError, match="nonce"):
            content_security_policy(invalid_nonce)  # type: ignore[arg-type]

    assert _reading({"n": None}, "n") == "n/a"
    assert _reading({"n": 12}, "n") == "12"
    assert _reading({"n": 4609}, "n", thousands=True) == "4,609"
    with pytest.raises(ValueError, match="source"):
        _reading([], "n")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="key"):
        _reading({}, "")
    with pytest.raises(ValueError, match="thousands"):
        _reading({}, "n", thousands="yes")  # type: ignore[arg-type]


def test_render_signup_page_structure_links_compliance_and_all_branches() -> None:
    page = render_signup_page(
        None,
        "nonce-slice02-check",
        starter_grant_hmt=25.0,
        registration_bonus_hmt=10.0,
        allow_bonuses=True,
        default_role="consumer",
    )
    assert STATIC_FIGURE_RE.findall(page) == []
    assert "innerHTML" not in page
    assert "/v1/chat/completions" not in page
    assert PERSONAL_PC_COMPLIANCE_NOTE in page
    assert "Starter grant on this gateway: 25.00 HMT + 10.00 HMT registration bonus (35.00 HMT total" in page
    assert "60-Second IDE & CLI Quick Setup" in page
    assert "📋 Copy VS Code Continue.dev Config" in page
    assert "⚡ Test Key in Browser (/v1/models)" in page
    assert "python hivemind_connect.py check" in page
    assert "python hivemind_connect.py snippet continue --model coder" in page
    assert DEFAULT_KEY_PLACEHOLDER in page

    inspector = _SectionInspector()
    inspector.feed(page)
    assert ("/noel", "🎄 Noël Quick Start") in inspector.nav_links
    assert {"ideQuickSetup", "continueYamlSnippet", "copyContinueBtn", "verifyKeyModelsBtn", "verifyKeyStatus"}.issubset(
        inspector.ids_inside_result
    )
    assert inspector.continue_code_parent_tags[-1] == "pre"

    # Existing IDs and data-testids remain intact
    for preserved in (
        'id="poc-notice"',
        'id="starter-grant"',
        'data-testid="signup-quick-nav"',
        'id="signupLiveTelemetry"',
        'id="existingBrowserSessionBanner"',
        'data-testid="existing-browser-session-banner"',
        'id="result"',
        'id="key"',
        'id="copy"',
        'data-testid="btn-open-v31-authenticated"',
        'data-testid="signup-claim-bonus-btn"',
        'data-testid="banner-claim-bonus-btn"',
        'data-testid="signup-rotate-key-btn"',
        'data-testid="signup-transfer-hmt-form"',
        'id="loginForm"',
        'id="registerForm"',
        'id="changePasswordForm"',
        'id="opentrainerCard"',
    ):
        assert preserved in page

    # Cover remaining starter_grant / registration_bonus / auth_result / Google client_id / error branches
    only_grant = render_signup_page(None, "nonce-g", starter_grant_hmt=15.0)
    assert "Starter grant on this gateway: 15.00 HMT per new account" in only_grant

    only_reg_bonus = render_signup_page(None, "nonce-r", registration_bonus_hmt=5.0)
    assert "Registration bonus on this gateway: 5.00 HMT per new account" in only_reg_bonus

    authed_created = render_signup_page(
        TEST_GOOGLE_CLIENT_ID,
        "nonce-google-1",
        auth_result={
            "created": True,
            "api_key": "hm-CreatedRealKey_987654321",
            "account_id": "acct-1",
            "handle": "noel_dev",
            "role": "prosumer",
            "balance_hmt": 50.0,
        },
        live_telemetry={
            "accounts": {"total_registered_accounts": 10, "by_role": {"prosumer": 4, "producer": 2, "consumer": 3, "trainer": 1}},
            "models": {"live_models_count": 3},
            "t4_boinc": {"vram_free_mib": 8192},
            "opentrainer": {"total_registered_nodes": 2, "accepted_rounds_count": 5},
        },
    )
    assert "Your account was created." in authed_created
    assert "apiKey: hm-CreatedRealKey_987654321" in authed_created
    assert TEST_GOOGLE_CLIENT_ID in authed_created
    assert '<section id="result" class="card ok-banner">' in authed_created

    authed_restored = render_signup_page(
        None,
        "nonce-restored",
        auth_result={"created": False, "api_key": "hm-RestoredKey_123456", "account_id": "acct-2", "balance": 12.5},
        error_message="<bad>",
        default_role="unknown-fallback",
    )
    assert "Welcome back! Your previous account & HMT balance have been restored." in authed_restored
    assert "apiKey: hm-RestoredKey_123456" in authed_restored
    assert "&lt;bad&gt;" in authed_restored


def test_render_signup_page_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="nonce"):
        render_signup_page(None, "")
    with pytest.raises(ValueError, match="Google client id"):
        render_signup_page("bad-client-id", "nonce-ok")
    with pytest.raises(ValueError, match="starter_grant_hmt"):
        render_signup_page(None, "nonce-ok", starter_grant_hmt=-1.0)
    with pytest.raises(ValueError, match="registration_bonus_hmt"):
        render_signup_page(None, "nonce-ok", registration_bonus_hmt=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="allow_bonuses"):
        render_signup_page(None, "nonce-ok", allow_bonuses="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="auth_result"):
        render_signup_page(None, "nonce-ok", auth_result="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="error_message"):
        render_signup_page(None, "nonce-ok", error_message=404)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="live_telemetry"):
        render_signup_page(None, "nonce-ok", live_telemetry="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_role"):
        render_signup_page(None, "nonce-ok", default_role=123)  # type: ignore[arg-type]


def test_gateway_signup_and_v1_models_integration(tmp_path: Path) -> None:
    store = RouterStore(tmp_path / "router.db", secret=SECRET)
    client = TestClient(create_router_app(store, open_registration=True, starter_grant_hmt=25.0))

    get_resp = client.get("/signup")
    assert get_resp.status_code == 200
    csp = get_resp.headers["content-security-policy"]
    nonce = csp.split("'nonce-")[1].split("'")[0]
    assert f'nonce="{nonce}"' in get_resp.text
    assert 'href="/noel"' in get_resp.text
    assert PERSONAL_PC_COMPLIANCE_NOTE in get_resp.text

    post_resp = client.post(
        "/signup",
        data={"handle": "noel_vscode_tester", "password": "HolidaySafePassword!2026", "role": "consumer"},
    )
    assert post_resp.status_code == 200
    key_match = re.search(r'<code id="key">(hm-[A-Za-z0-9_-]+)</code>', post_resp.text)
    assert key_match is not None
    issued_key = key_match.group(1)
    assert f"apiKey: {issued_key}" in post_resp.text

    models_resp = client.get("/v1/models", headers={"Authorization": f"Bearer {issued_key}"})
    assert models_resp.status_code == 200
    models_payload = models_resp.json()
    assert models_payload["object"] == "list"
    assert isinstance(models_payload["data"], list) and len(models_payload["data"]) >= 1


def test_js_flows_update_continue_yaml_via_textcontent_and_verify_v1_models() -> None:
    g_js = _GOOGLE_JS.replace("%CLIENT_ID%", json.dumps(TEST_GOOGLE_CLIENT_ID))
    harness = r"""
const elements = {};
function makeEl(id) {
  const listeners = {};
  const el = {
    id,
    _text: '',
    value: '',
    className: '',
    hidden: id === 'result' || id === 'existingBrowserSessionBanner',
    checked: id === 'role_prosumer',
    addEventListener(type, fn) { listeners[type] = fn; },
    async trigger(type, ev) {
      if (listeners[type]) await listeners[type](ev || { preventDefault() {} });
    },
  };
  Object.defineProperty(el, 'textContent', {
    get() { return this._text; },
    set(v) { this._text = String(v); },
  });
  Object.defineProperty(el, 'innerHTML', {
    get() { throw new Error('innerHTML getter forbidden'); },
    set() { throw new Error('innerHTML setter forbidden'); },
  });
  return el;
}
[
  'status', 'key', 'continueYamlSnippet', 'verifyKeyStatus', 'acct', 'role', 'bal', 'kind', 'result',
  'loginForm', 'loginIdentifier', 'loginPassword', 'registerForm', 'regHandle', 'regEmail', 'regPassword',
  'copy', 'copyContinueBtn', 'verifyKeyModelsBtn', 'signupLiveTelemetry', 'activeSessionHandle',
  'activeSessionBalance', 'existingBrowserSessionBanner', 'rotateKeyBtn', 'changePasswordForm',
  'resetTokenOrKey', 'newResetPassword', 'signOutSessionBtn', 'claimBonusBtn', 'bannerClaimBonusBtn',
  'transferPointsForm', 'transferRecipient', 'transferAmount', 'transferMemo', 'signin', 'gbtn'
].forEach((id) => { elements[id] = makeEl(id); });

const storage = new Map();
globalThis.localStorage = {
  getItem: (k) => (storage.has(k) ? storage.get(k) : null),
  setItem: (k, v) => { storage.set(k, String(v)); },
  removeItem: (k) => { storage.delete(k); },
};
globalThis.window = {
  location: { origin: 'https://hivemind-test.a.run.app', search: '', hash: '' },
};
globalThis.document = {
  getElementById: (id) => elements[id] || null,
  querySelector: () => ({ value: 'consumer', checked: true }),
};
let clipboardText = '';
let failClipboard = false;
Object.defineProperty(globalThis, 'navigator', {
  configurable: true,
  writable: true,
  value: {
    clipboard: {
      async writeText(t) {
        if (failClipboard) throw new Error('clipboard denied');
        clipboardText = String(t);
      },
    },
  },
});
const fetchCalls = [];
let modelsMode = 'ok';
globalThis.fetch = async (url, opts = {}) => {
  fetchCalls.push({ url, method: opts.method || 'GET', headers: opts.headers || {} });
  if (String(url).includes('/v1/chat/completions')) {
    throw new Error('Forbidden call to /v1/chat/completions');
  }
  if (url === '/v1/hivemind/live-telemetry') {
    return { ok: true, status: 200, async json() { return { accounts: { total_registered_accounts: 5, by_role: {} }, models: { live_models_count: 2 }, t4_boinc: { vram_free_mib: 4096 }, opentrainer: {} }; } };
  }
  if (url === '/api/v1/auth/session') {
    return { ok: true, status: 200, async json() { return { authenticated: true, user: { handle: 'sess_user', role: 'consumer', balance_hmt: 42 } }; } };
  }
  if (url === '/api/v1/auth/rotate-key') {
    return { ok: true, status: 200, async json() { return { token: 'hm-RotatedKey_222', handle: 'sess_user' }; } };
  }
  if (url === '/api/v1/auth/change-password') {
    return { ok: true, status: 200, async json() { return { api_key: 'hm-PassChangedKey_333', handle: 'sess_user' }; } };
  }
  if (url === '/v1/hivemind/register/google') {
    return { ok: true, status: 200, async json() { return { api_key: 'hm-GoogleKey_444', account_id: 'acct-g', role: 'consumer', created: true }; } };
  }
  if (url === '/v1/models') {
    if (modelsMode === 'network_err') throw new Error('offline');
    if (modelsMode === 'http_err') {
      return { ok: false, status: 401, async json() { return { error: { message: 'Invalid API key' } }; } };
    }
    return {
      ok: true,
      status: 200,
      async json() {
        return { object: 'list', data: [{ id: 'hivemind/auto' }, { id: 'hivemind/coder' }] };
      },
    };
  }
  return { ok: true, status: 200, async json() { return {}; } };
};
"""
    driver = r"""
(async () => {
  // 0. Unauthenticated verifyKeyModelsBtn shows friendly guidance without network call
  await elements.verifyKeyModelsBtn.trigger('click');
  const unauthStatus = elements.verifyKeyStatus.textContent;

  // 1. showSessionResult()
  showSessionResult({ api_key: 'hm-SessionKey_111', account_id: 'acct-1', handle: 'alice', role: 'consumer', balance_hmt: 25, created: true });
  const afterShow = elements.continueYamlSnippet.textContent;

  // 2. checkActiveBrowserSession()
  storage.set('hivemind_api_key', 'hm-BrowserSessionKey_155');
  elements.key.textContent = '';
  await checkActiveBrowserSession();
  const afterCheckSession = elements.continueYamlSnippet.textContent;

  // 3. rotateKeyBtn
  await elements.rotateKeyBtn.trigger('click');
  const afterRotate = elements.continueYamlSnippet.textContent;

  // 4. changePasswordForm
  elements.resetTokenOrKey.value = 'hm-RotatedKey_222';
  elements.newResetPassword.value = 'NewStrongPassword!2026';
  await elements.changePasswordForm.trigger('submit');
  const afterPassChange = elements.continueYamlSnippet.textContent;

  // 5. onCredential()
  await onCredential({ credential: 'fake-google-jwt' });
  const afterGoogle = elements.continueYamlSnippet.textContent;

  // 6. copyContinueBtn (success + fallback)
  await elements.copyContinueBtn.trigger('click');
  const copiedLabel = elements.copyContinueBtn.textContent;
  failClipboard = true;
  await elements.copyContinueBtn.trigger('click');
  const fallbackCopyLabel = elements.copyContinueBtn.textContent;

  // 7. verifyKeyModelsBtn (success + HTTP error + network error)
  await elements.verifyKeyModelsBtn.trigger('click');
  const verifiedStatus = elements.verifyKeyStatus.textContent;
  modelsMode = 'http_err';
  await elements.verifyKeyModelsBtn.trigger('click');
  const httpErrStatus = elements.verifyKeyStatus.textContent;
  modelsMode = 'network_err';
  await elements.verifyKeyModelsBtn.trigger('click');
  const netErrStatus = elements.verifyKeyStatus.textContent;

  console.log(JSON.stringify({
    unauthStatus,
    afterShow,
    afterCheckSession,
    afterRotate,
    afterPassChange,
    afterGoogle,
    clipboardText,
    copiedLabel,
    fallbackCopyLabel,
    verifiedStatus,
    httpErrStatus,
    netErrStatus,
    fetchCalls,
  }));
})().catch((e) => { console.error(e); process.exit(1); });
"""
    proc = subprocess.run(
        ["node", "-e", f"{harness}\n{_AUTH_JS}\n{g_js}\n{driver}"],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    expected_origin_v1 = "https://hivemind-test.a.run.app/v1"

    assert "Sign in or create an account first" in out["unauthStatus"]
    assert out["afterShow"] == build_continue_config_yaml("hm-SessionKey_111", expected_origin_v1)
    assert out["afterCheckSession"] == build_continue_config_yaml("hm-BrowserSessionKey_155", expected_origin_v1)
    assert out["afterRotate"] == build_continue_config_yaml("hm-RotatedKey_222", expected_origin_v1)
    assert out["afterPassChange"] == build_continue_config_yaml("hm-PassChangedKey_333", expected_origin_v1)
    assert out["afterGoogle"] == build_continue_config_yaml("hm-GoogleKey_444", expected_origin_v1)
    assert out["clipboardText"] == out["afterGoogle"]
    assert "Copied Continue.dev Config" in out["copiedLabel"]
    assert "manually" in out["fallbackCopyLabel"]
    assert "2 live model(s)" in out["verifiedStatus"]
    assert "hivemind/auto, hivemind/coder" in out["verifiedStatus"]
    assert "Invalid API key" in out["httpErrStatus"]
    assert "Could not reach /v1/models" in out["netErrStatus"]

    model_calls = [c for c in out["fetchCalls"] if c["url"] == "/v1/models"]
    assert len(model_calls) == 3
    for call in model_calls:
        assert call["method"] == "GET"
        assert call["headers"]["Authorization"] == "Bearer hm-GoogleKey_444"
    assert not any("/v1/chat/completions" in str(c["url"]) for c in out["fetchCalls"])
