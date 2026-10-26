"""Serving experts: routing next to the LoRA domain experts, isolation from Season 3 training, and the optional GWAYA hook."""

from __future__ import annotations

import sys

import pytest
from hypothesis import given, settings, strategies as st

from hivemind.core.btm_moe import SEASON_3_DOMAIN_EXPERT_SPECS
from hivemind.core.serving_experts import (
    PYTHON_CODE_GWAYA,
    SERVING_EXPERT_SPECS,
    load_gwaya_code_expert,
    route_with_serving_experts,
)


def test_python_prompt_routes_to_the_gwaya_code_expert_first() -> None:
    domains, weights = route_with_serving_experts("Fix this traceback in my python script: def f(x): assert x", top_k=2)
    assert domains[0] == PYTHON_CODE_GWAYA
    assert weights[PYTHON_CODE_GWAYA] > 0.5
    assert abs(sum(weights.values()) - 1.0) < 1e-3


def test_other_domains_are_not_hijacked() -> None:
    solidity, _ = route_with_serving_experts("Is this solidity contract safe from reentrancy? msg.sender uint256 require", top_k=1)
    rust, _ = route_with_serving_experts("rust fn with pub mut Arc Mutex and a borrow lifetime", top_k=1)
    assert solidity == ["Solidity_Smart_Contracts"]
    assert rust == ["Rust_Systems_Security"]
    assert PYTHON_CODE_GWAYA not in solidity + rust


def test_serving_experts_never_enter_the_lora_domain_table() -> None:
    # Registering one must not give the Season 3 trainer a fifth domain to train (it has no dataset and no adapter).
    assert PYTHON_CODE_GWAYA not in SEASON_3_DOMAIN_EXPERT_SPECS
    assert len(SEASON_3_DOMAIN_EXPERT_SPECS) == 4
    assert not set(SERVING_EXPERT_SPECS) & set(SEASON_3_DOMAIN_EXPERT_SPECS)
    assert all(spec["kind"] == "serving_verified" and "lora_rank_r" not in spec for spec in SERVING_EXPERT_SPECS.values())


@settings(max_examples=100, deadline=None)
@given(st.text(max_size=200), st.integers(min_value=1, max_value=6))
def test_routing_weights_are_a_distribution_over_known_domains(prompt: str, top_k: int) -> None:
    domains, weights = route_with_serving_experts(prompt, top_k=top_k)
    known = set(SEASON_3_DOMAIN_EXPERT_SPECS) | set(SERVING_EXPERT_SPECS)
    assert len(domains) <= min(top_k, len(known))
    assert set(domains) <= known and set(weights) == set(domains)
    if domains:  # a prompt without any keyword hit abstains: ([], {})
        assert abs(sum(weights.values()) - 1.0) < 1e-3 * len(domains)
    else:
        assert weights == {}


def test_loader_reports_a_missing_gwaya_clearly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "gwaya", None)
    monkeypatch.setitem(sys.modules, "gwaya.hivemind_expert", None)
    with pytest.raises(ImportError, match="GWAYA Code expert"):
        load_gwaya_code_expert("http://127.0.0.1:8080")


def test_spec_keywords_match_gwaya_when_installed() -> None:
    gwaya_expert = pytest.importorskip("gwaya.hivemind_expert")
    assert SERVING_EXPERT_SPECS[PYTHON_CODE_GWAYA]["keywords"] == list(gwaya_expert.KEYWORDS)
    assert gwaya_expert.DOMAIN_NAME == PYTHON_CODE_GWAYA
    prompt = "python def import pytest"
    hits = sum(1 for kw in SERVING_EXPERT_SPECS[PYTHON_CODE_GWAYA]["keywords"] if kw in prompt.lower())
    assert hits >= 3 and gwaya_expert.score(prompt) == 1.0 + 2.5 * hits  # same scale as route_domain_query


def test_loader_builds_the_expert_with_seven_b_only_by_default() -> None:
    gwaya_expert = pytest.importorskip("gwaya.hivemind_expert")
    expert = load_gwaya_code_expert("http://127.0.0.1:8080")
    assert isinstance(expert, gwaya_expert.GwayaCodeExpert)
    assert expert.small is None and expert.small_first is False
    assert expert.large.name == "qwen2.5-coder:7b" and expert.large.url.endswith("/v1")


def test_python_keywords_do_not_route_a_school_class() -> None:
    # "class " fired on 38 of 1319 GSM8K word problems ("a class of 20 students"); a real class statement also carries def/import.
    assert "class " not in SERVING_EXPERT_SPECS[PYTHON_CODE_GWAYA]["keywords"]
