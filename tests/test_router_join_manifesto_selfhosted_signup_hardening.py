# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Input-validation hardening tests for router join, manifesto, selfhosted promises, and signup pages."""

from __future__ import annotations

import pytest

from hivemind.router.join_page import pick_language, render_join_page
from hivemind.router.manifesto_page import pick_manifesto_language
from hivemind.router.selfhosted_p2p_promises import (
    audit_selfhosted_reddit_1xjc76k_promises,
    evaluate_producer_anti_farming_gate,
    wilson_lower_bound,
)
from hivemind.router.signup_page import _reading, render_signup_page


def test_join_page_pick_language_and_render_reject_invalid_inputs() -> None:
    assert pick_language("en", None) == "en"
    with pytest.raises(ValueError, match="query_lang"):
        pick_language(True, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="accept_language"):
        pick_language(None, 123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default"):
        pick_language(None, None, default="it")

    assert "<html" in render_join_page("fr", "abcdefgh12345678")
    with pytest.raises(ValueError, match="tiers"):
        render_join_page("fr", "abcdefgh12345678", tiers="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonce"):
        render_join_page("fr", None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="api_url"):
        render_join_page("fr", "abcdefgh12345678", api_url=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repo_url"):
        render_join_page("fr", "abcdefgh12345678", repo_url=None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="path"):
        render_join_page("fr", "abcdefgh12345678", path=None)  # type: ignore[arg-type]


def test_manifesto_page_pick_language_rejects_invalid_inputs() -> None:
    assert pick_manifesto_language("en", None) == "en"
    with pytest.raises(ValueError, match="query_lang"):
        pick_manifesto_language(True, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="accept_language"):
        pick_manifesto_language(None, False)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default"):
        pick_manifesto_language(None, None, default=True)  # type: ignore[arg-type]


def test_selfhosted_p2p_promises_rejects_non_numeric_and_invalid_types() -> None:
    assert wilson_lower_bound(18, 20) > 0.65
    with pytest.raises(ValueError, match="successes and trials"):
        wilson_lower_bound(18.5, 20)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="successes and trials"):
        wilson_lower_bound(18, "20")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="z must be a finite positive float"):
        wilson_lower_bound(18, 20, z="1.96")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="max_logprob_drift must be a finite float"):
        evaluate_producer_anti_farming_gate(
            canary_successes=19,
            canary_trials=20,
            max_logprob_drift="0.05",  # type: ignore[arg-type]
            replay_loss_before=3.0,
            replay_loss_after=2.5,
        )

    with pytest.raises(ValueError, match="signup_html_trainer"):
        audit_selfhosted_reddit_1xjc76k_promises(signup_html_trainer=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="public_fleet_payload"):
        audit_selfhosted_reddit_1xjc76k_promises(public_fleet_payload="not-a-dict")  # type: ignore[arg-type]


def test_signup_page_reading_and_render_reject_invalid_inputs() -> None:
    assert _reading({"count": 42}, "count") == "42"
    with pytest.raises(ValueError, match="source"):
        _reading(None, "count")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="key"):
        _reading({}, "")
    with pytest.raises(ValueError, match="thousands"):
        _reading({}, "count", thousands=1)  # type: ignore[arg-type]

    assert "<html" in render_signup_page(None, "valid-nonce-123456")
    with pytest.raises(ValueError, match="error_message"):
        render_signup_page(None, "valid-nonce-123456", error_message=123)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_role"):
        render_signup_page(None, "valid-nonce-123456", default_role=None)  # type: ignore[arg-type]
