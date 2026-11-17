# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Links pasted with trailing punctuation ("/signup)") redirect to the page instead of a 404."""
import pytest

from hivemind.router.gateway import link_typo_target


@pytest.mark.parametrize("path, expected", [
    ("/signup)", "/signup"),
    ("/signup](https://hivemind-openrouter-1003063861791.us-central1.run.app/signup)", "/signup"),
    ("/signup%29", "/signup"),
    ("/signup%5D", "/signup"),
    ("/signup.", "/signup"),
    ("/rejoindre,", "/rejoindre"),
    ("/)", "/"),
])
def test_known_pages_with_trailing_link_punctuation_redirect(path: str, expected: str) -> None:
    assert link_typo_target(path) == expected


@pytest.mark.parametrize("path", [
    "/signup", "/nope)", "/v1/chat/completions)", "/admin)", "/signupDiscord", "//evil.example)", "/signup/../admin)", "",
])
def test_other_paths_are_not_redirected(path: str) -> None:
    assert link_typo_target(path) is None


def test_gateway_redirects_a_pasted_link_and_keeps_real_404s(tmp_path) -> None:
    from fastapi.testclient import TestClient

    from hivemind.router.accounts import RouterStore
    from hivemind.router.gateway import create_router_app

    client = TestClient(create_router_app(RouterStore(tmp_path / "router.db", b"k" * 32), allow_private_producers=True))
    r = client.get("/signup)?role=trainer", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"] == "/signup?role=trainer"
    assert client.head("/signup%5D", follow_redirects=False).status_code == 307
    missing = client.get("/no-such-page)", follow_redirects=False)
    assert missing.status_code == 404 and missing.json()["error"]["code"] == "http_404"
    assert client.post("/signup)", follow_redirects=False).status_code in (404, 405)
