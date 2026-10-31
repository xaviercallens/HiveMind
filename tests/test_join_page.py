# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The public multilingual join page (FR default, EN, ES, DE, ZH): same structure in all languages, no currency, no script, and every command matches the real scripts."""

from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hivemind.router import RouterStore, create_router_app
from hivemind.router.catalog import TIERS, catalog_payload
from hivemind.router.join_page import (
    COORDINATOR_URL,
    DEFAULT_API_URL,
    LANG_NAMES,
    LANGS,
    PATHS,
    REPO_URL,
    _safe_href,
    join_content_security_policy,
    pick_language,
    render_join_page,
)
from hivemind.router.points_wording import currency_leftovers

ROOT = Path(__file__).resolve().parents[1]
NONCE = "abcdefgh12345678"
TIER_ROWS = [{"title": t.title, "tier_id": t.tier_id, "min_vram_gb": t.min_vram_gb, "weights_gb": t.weights_gb} for t in TIERS]
OFFICIAL_POINTS_NAMES = {
    "fr": "Points de Réflexion HiveMind (HMT)",
    "en": "HiveMind Reflexion Points (HMT)",
    "es": "Puntos de Reflexión HiveMind (HMT)",
    "de": "HiveMind-Reflexionspunkte (HMT)",
    "zh": "HiveMind 反思积分 (HMT)",
}


def page(lang: str = "fr", tiers=TIER_ROWS) -> str:  # noqa: ANN001
    return render_join_page(lang, NONCE, tiers=tiers)


class Outline(HTMLParser):
    """Collects what the structure-parity and safety checks need; also checks that every tag is closed."""

    VOID = {"meta", "link", "br", "img", "hr", "input"}

    def __init__(self) -> None:
        super().__init__()
        self.stack: list[str] = []
        self.ids: list[str] = []
        self.hrefs: list[str] = []
        self.tags: list[str] = []
        self.pre: list[str] = []
        self.text: list[str] = []
        self.problems: list[str] = []
        self._in_pre = False

    def handle_starttag(self, tag, attrs):  # noqa: ANN001
        self.tags.append(tag)
        a = dict(attrs)
        if "id" in a:
            self.ids.append(a["id"])
        if tag in ("a", "link") and "href" in a:
            self.hrefs.append(a["href"])
        if "src" in a:
            self.problems.append(f"<{tag} src={a['src']}>")
        if tag == "pre":
            self._in_pre = True
            self.pre.append("")
        if tag not in self.VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):  # noqa: ANN001
        if tag == "pre":
            self._in_pre = False
        if tag in self.VOID:
            return
        if not self.stack or self.stack[-1] != tag:
            self.problems.append(f"unbalanced </{tag}>")
        else:
            self.stack.pop()

    def handle_data(self, data):  # noqa: ANN001
        self.text.append(data)
        if self._in_pre:
            self.pre[-1] += data


def outline(lang: str) -> Outline:
    parser = Outline()
    parser.feed(page(lang))
    parser.close()
    return parser


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# all five languages (fr, en, es, de, zh) have the exact same page structure, commands and links
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_all_five_languages_have_the_same_structure_the_same_commands_and_the_same_links() -> None:
    assert LANGS == ("fr", "en", "es", "de", "zh")
    ref = outline("fr")
    assert ref.problems == [] and ref.stack == []
    assert len(ref.pre) >= 7 and len(ref.ids) >= 6
    for lang in LANGS:
        cur = outline(lang)
        assert cur.problems == [] and cur.stack == [], lang
        assert cur.ids == ref.ids and cur.tags == ref.tags, lang
        assert cur.pre == ref.pre, lang  # commands are language-neutral across all 5 languages
        assert [h for h in cur.hrefs if not h.startswith("/join")] == [h for h in ref.hrefs if not h.startswith("/join")], lang


def test_the_languages_really_differ_declare_themselves_and_french_is_default() -> None:
    default_html = render_join_page(tiers=TIER_ROWS)
    assert '<html lang="fr">' in default_html and "Rejoindre HiveMind La Ruche" in default_html
    en, fr, es, de, zh = page("en"), page("fr"), page("es"), page("de"), page("zh")
    assert '<html lang="en">' in en and '<html lang="fr">' in fr and '<html lang="es">' in es and '<html lang="de">' in de and '<html lang="zh">' in zh
    assert "Join HiveMind La Ruche" in en and "Rejoindre HiveMind La Ruche" in fr
    assert "Unirse a HiveMind La Ruche" in es and "HiveMind La Ruche beitreten" in de and "加入 HiveMind La Ruche" in zh
    assert "Disponible maintenant" in fr and "Available now" in en and "Disponible ahora" in es and "Jetzt verfügbar" in de and "现已开放" in zh
    for code, name in LANG_NAMES.items():
        assert f'hreflang="{code}"' in fr and f'href="/join?lang={code}"' in fr and name in fr


def test_train_and_play_paths_are_available_now_with_working_commands() -> None:
    assert [cls for _anchor, _title, cls, _status, _summary in PATHS] == ["ok", "ok", "ok", "ok"]
    en, fr = page("en"), page("fr")
    assert "pas encore ouvert au public" not in fr and "not open to the public yet" not in en
    assert "En développement" not in fr and "In development" not in en
    assert "Disponible maintenant · unités MoE nocturnes et astrophysique" in fr
    assert "Disponible maintenant · jeu Steam gratuit et pause GPU automatique" in fr
    assert "Available now · night MoE &amp; astrophysics workunits" in en
    assert "Available now · Free-to-Play Steam &amp; automatic idle-GPU pause" in en
    assert "hivemind_boinc_idle_worker.py" in fr and "AppID 3133700" in fr


def test_every_section_link_points_at_a_section_that_exists() -> None:
    for lang in LANGS:
        o = outline(lang)
        anchors = [h[1:] for h in o.hrefs if h.startswith("#")]
        assert anchors and all(a in o.ids for a in anchors)


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# wording, safety
# ---------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lang", LANGS)
def test_no_euro_or_dollar_amount_appears_and_the_points_are_named_the_official_way(lang: str) -> None:
    text = html.unescape(re.sub(r"<[^>]+>", " ", page(lang)))
    assert currency_leftovers(text) == []
    assert OFFICIAL_POINTS_NAMES[lang] in text


@pytest.mark.parametrize("lang", LANGS)
def test_no_script_no_external_resource_and_only_trusted_links(lang: str) -> None:
    body = page(lang)
    assert "<script" not in body and "javascript:" not in body.lower() and "onerror" not in body.lower()
    allowed = (REPO_URL, "https://ollama.com", COORDINATOR_URL, "/", "#")
    lang_hrefs = {f"/join?lang={c}" for c in LANGS}
    for href in outline(lang).hrefs:
        assert href.startswith(allowed) or href in lang_hrefs, href
    csp = join_content_security_policy(NONCE)
    assert f"style-src 'nonce-{NONCE}'" in csp and "script-src" not in csp and "default-src 'none'" in csp
    assert f'<style nonce="{NONCE}">' in body


def test_catalog_text_is_escaped_and_bad_inputs_are_refused() -> None:
    evil = [{"title": "<script>alert(1)</script>", "tier_id": "x\"><b>", "min_vram_gb": 4, "weights_gb": 1.5}]
    out = page("en", evil)
    assert "<script>alert" not in out and "&lt;script&gt;" in out and '"><b>' not in out
    for bad in (dict(lang="it"), dict(lang="en", nonce="short"), dict(lang="en", api_url="http://x.example"), dict(lang="en", repo_url="javascript:alert(1)"),
                dict(lang="en", path="/a b")):
        kwargs = {"nonce": NONCE, **bad}
        with pytest.raises(ValueError):
            render_join_page(kwargs.pop("lang"), kwargs.pop("nonce"), **kwargs)
    assert not _safe_href("javascript:alert(1)") and not _safe_href("//evil.example") and _safe_href("/signup") and _safe_href("#use")


def test_without_a_catalog_the_page_still_renders_and_points_at_the_live_list() -> None:
    assert "/v1/hivemind/catalog" in page("en", None) and "<table" not in page("en", None)


def test_the_model_table_is_the_catalog_with_european_decimal_commas() -> None:
    fr, es, de, en, zh = page("fr"), page("es"), page("de"), page("en"), page("zh")
    assert all(f"<code>{t.tier_id}</code>" in en for t in TIERS)
    assert "<td class=\"n\">4,5</td>" in fr and "<td class=\"n\">4,5</td>" in es and "<td class=\"n\">4,5</td>" in de
    assert "<td class=\"n\">4.5</td>" in en and "<td class=\"n\">4.5</td>" in zh


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the instructions are true: every command and flag exists in the code the page tells people to run
# ---------------------------------------------------------------------------------------------------------------------------------------------------


def test_every_script_and_flag_the_page_tells_people_to_use_exists() -> None:
    commands = "\n".join(outline("en").pre).replace("\\\n", " ")
    scripts = set(re.findall(r"scripts/([a-z_]+\.py)", commands))
    assert scripts == {
        "hivemind_boinc_idle_worker.py",
        "hivemind_producer_agent.py",
        "hivemind_producer_shield.py",
        "hivemind_volunteer_preflight.py",
    }
    for name in scripts:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        for line in (l for l in commands.splitlines() if f"scripts/{name}" in l):
            for flag in re.findall(r"(--[a-z][a-z-]+)", line):
                assert f'"{flag}"' in source, f"{name} has no {flag}"


def test_the_preset_and_model_tag_in_the_examples_are_a_real_tier_that_fits_a_small_card() -> None:
    commands = "\n".join(outline("en").pre)
    preset = re.search(r"--preset (\S+)", commands).group(1)  # type: ignore[union-attr]
    tier = next(t for t in TIERS if t.tier_id == preset)
    assert tier.ollama_tag in commands and f"ollama pull {tier.ollama_tag}" in commands and tier.min_vram_gb <= 4.5


def test_the_boinc_numbers_match_the_project_configuration() -> None:
    from hivemind.boinc.real_client.moe_profiles import PROFILES, STANDARD

    assert PROFILES[STANDARD]["min_gpu_ram_mb"] == 3500  # "at least 3.5 GB of video memory"
    assert "x86_64-pc-linux-gnu" in (ROOT / "src/hivemind/boinc/real_client/server_stack.py").read_text(encoding="utf-8")  # "Linux on an x86-64 processor"
    assert "22" in (ROOT / "scripts/hivemind_volunteer_preflight.py").read_text(encoding="utf-8")  # the night window the page names


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# language negotiation and the gateway routes
# ---------------------------------------------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("query,header,default,expected", [
    (None, None, "fr", "fr"),
    ("fr", "en", "en", "fr"),
    ("en", "fr", "fr", "en"),
    ("es", "fr", "fr", "es"),
    ("de", "fr", "fr", "de"),
    ("zh", "fr", "fr", "zh"),
    ("it", "fr-CA", "en", "fr"),
    (None, "fr-CA,fr;q=0.9,en;q=0.8", "en", "fr"),
    (None, "es-ES,es;q=0.9,fr;q=0.8", "fr", "es"),
    (None, "de-DE,de;q=0.9,en;q=0.8", "fr", "de"),
    (None, "zh-CN,zh;q=0.9,en;q=0.8", "fr", "zh"),
    (None, "en-US,en;q=0.9,fr;q=0.95", "en", "en"),
    (None, "en;q=0.5,fr;q=0.95", "en", "fr"),
    (None, "en-US,fr;q=0.2", "fr", "en"),
    (None, "it,pt", "fr", "fr"),
    (None, "fr;q=0", "en", "en"),
    (None, "fr;q=abc, en", "en", "en"),
])
def test_language_negotiation(query, header, default, expected) -> None:  # noqa: ANN001
    assert pick_language(query, header, default) == expected


@pytest.fixture()
def client() -> TestClient:
    return TestClient(create_router_app(RouterStore(":memory:", b"g" * 32)))


def test_gateway_serves_french_by_default_and_negotiates_all_five_languages_with_strict_headers(client: TestClient) -> None:
    default_join = client.get("/join", headers={"Accept-Language": ""})
    assert default_join.status_code == 200 and default_join.headers["content-language"] == "fr"
    assert "Rejoindre HiveMind La Ruche" in default_join.text
    en = client.get("/join", headers={"Accept-Language": "en-US,en;q=0.9"})
    fr = client.get("/join", headers={"Accept-Language": "fr-CA,fr;q=0.9"})
    es = client.get("/join?lang=es")
    de = client.get("/join?lang=de")
    zh = client.get("/join?lang=zh")
    assert en.status_code == fr.status_code == es.status_code == de.status_code == zh.status_code == 200
    assert en.headers["content-language"] == "en" and fr.headers["content-language"] == "fr"
    assert es.headers["content-language"] == "es" and de.headers["content-language"] == "de" and zh.headers["content-language"] == "zh"
    assert "Rejoindre" in fr.text and "Join HiveMind La Ruche" in en.text and "Unirse a HiveMind La Ruche" in es.text
    assert "HiveMind La Ruche beitreten" in de.text and "加入 HiveMind La Ruche" in zh.text
    assert en.headers["vary"] == "Accept-Language"
    nonce = re.search(r"style-src 'nonce-([^']+)'", en.headers["content-security-policy"]).group(1)  # type: ignore[union-attr]
    assert f'<style nonce="{nonce}">' in en.text and en.headers["x-content-type-options"] == "nosniff"
    assert "<td class=\"n\">24</td>" in en.text and "moe-coder" in en.text  # the live catalog's biggest tier, from catalog_payload()
    assert {t["tier_id"] for t in catalog_payload()["tiers"]} <= set(re.findall(r"<code>([a-z0-9._-]+)</code>", en.text))


def test_rejoindre_defaults_to_french_but_the_query_string_wins_and_head_works(client: TestClient) -> None:
    assert "Rejoindre" in client.get("/rejoindre").text
    assert "Join HiveMind La Ruche" in client.get("/rejoindre?lang=en").text
    assert "Unirse a HiveMind La Ruche" in client.get("/rejoindre?lang=es").text
    assert "Join HiveMind La Ruche" in client.get("/rejoindre", headers={"Accept-Language": "en"}).text
    assert client.head("/join").status_code == 200
    assert 'href="/rejoindre?lang=en"' in client.get("/rejoindre").text  # the switch stays on the French path


def test_the_signup_page_links_to_the_join_page(client: TestClient) -> None:
    text = client.get("/signup").text
    assert 'href="/join"' in text and 'href="/rejoindre"' in text


def test_default_api_url_is_a_valid_https_origin() -> None:
    assert DEFAULT_API_URL.startswith("https://") and "/" not in DEFAULT_API_URL.removeprefix("https://")


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# the account steps describe only the sign-up routes this gateway really has (the live gateway has handle registration and NO Google sign-in)
# ---------------------------------------------------------------------------------------------------------------------------------------------------

REGISTER_CMD = re.compile(r"-d '(\{[^']+\})'")


def flagged(lang: str, google: bool, handles: bool) -> str:
    return render_join_page(lang, NONCE, tiers=TIER_ROWS, google_enabled=google, handle_registration=handles)


@pytest.mark.parametrize("lang", LANGS)
def test_default_describes_handle_registration_and_does_not_promise_google(lang: str) -> None:
    text = page(lang)
    assert "/v1/hivemind/register" in text and not re.search(r"(?i)(sign in|connect\w+|connexion) (with|avec) Google", text)  # the live gateway has no Google sign-in
    assert "your-handle" in text and "api_key" in text


def test_every_combination_is_well_formed_and_all_five_languages_stay_in_step() -> None:
    for google in (False, True):
        for handles in (False, True):
            ref = Outline()
            ref.feed(flagged("fr", google, handles))
            assert ref.problems == [] and ref.stack == []
            for lang in LANGS:
                cur = Outline()
                cur.feed(flagged(lang, google, handles))
                assert cur.problems == [] and cur.stack == [], (lang, google, handles)
                assert cur.ids == ref.ids and cur.tags == ref.tags and cur.pre == ref.pre, (lang, google, handles)


def test_google_only_closed_and_both_each_say_the_right_thing() -> None:
    google_only = flagged("en", True, False)
    assert "/v1/hivemind/register" not in google_only and "sign in with Google" in google_only and "closed" not in google_only
    closed = flagged("en", False, False)
    assert "Registration is closed" in closed and "/v1/hivemind/register" not in closed and "sign in with Google" not in closed
    both = flagged("en", True, True)
    assert "Registration is closed" not in both and "/v1/hivemind/register" in both and "sign in with Google" in both
    assert "L'inscription est fermée" in html.unescape(flagged("fr", False, False))


def test_the_registration_example_is_a_request_the_real_endpoint_accepts() -> None:
    from hivemind.router.gateway import RegisterRequest

    body = REGISTER_CMD.search("\n".join(outline("en").pre))
    assert body is not None
    request = RegisterRequest.model_validate_json(body.group(1))  # extra fields are forbidden by the model, so a typo would fail here
    assert request.role == "consumer" and request.kind is None and request.test is False
    for role in ("producer", "prosumer"):  # the roles the producer step names: both must be roles the endpoint accepts
        RegisterRequest.model_validate({"handle": "x", "role": role})
        assert f'"{role}"' in html.unescape(page("en"))


def test_the_handle_rule_on_the_page_matches_the_stores_rule() -> None:
    store = RouterStore(":memory:", b"s" * 32)
    store.register("Good_handle.1@x+y-z", "consumer")  # the characters the page lists
    with pytest.raises(Exception):  # noqa: B017, PT011 - the store raises its own security/reserved errors
        store.register("bad handle\n", "consumer")
    assert "ASCII letters, digits and . _ @ + -" in page("en") and "64" in page("en")


def test_no_step_numbers_are_hard_coded_because_the_account_step_can_have_one_or_two_items() -> None:
    for lang in LANGS:
        assert not re.search(r"(?i)\b(step|étape|paso|schritt) \d", page(lang))


def test_the_french_status_label_has_the_space_before_the_colon() -> None:
    assert "État : " in page("fr") and "Status: " in page("en") and "Estado: " in page("es") and "Status: " in page("de") and "状态：" in page("zh")


def test_the_gateway_page_reflects_its_own_configuration() -> None:
    open_no_google = TestClient(create_router_app(RouterStore(":memory:", b"g" * 32))).get("/join?lang=en").text
    assert "/v1/hivemind/register" in open_no_google and "sign in with Google" not in open_no_google
    google_only = TestClient(create_router_app(RouterStore(":memory:", b"g" * 32), open_registration=False,
                                               google_client_id="123456789-abc.apps.googleusercontent.com")).get("/join?lang=en").text
    assert "/v1/hivemind/register" not in google_only and "sign in with Google" in google_only


def test_boinc_train_section_documents_plan_classes_dry_run_and_status_endpoints(client: TestClient) -> None:
    en = client.get("/join?lang=en").text
    assert "cuda_hivemind" in en and "cuda_hivemind_big" in en and "cpu_hivemind_solver" in en
    assert "--dry-run" in en and "--allow-cpu-solver" in en and "boinccmd --project_attach" in en
    assert "/boinc/status" in en and "/api/v1/federated/ledger" in en
    cfg = client.get("/get_project_config.php")
    assert cfg.status_code == 200 and "<project_config>" in cfg.text and "cuda_hivemind" in cfg.text
    st = client.get("/boinc/status")
    assert st.status_code == 200 and st.json()["status"] == "ONLINE"
