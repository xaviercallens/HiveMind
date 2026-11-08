# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Negative-control and invariant hardening tests for amcp_collab, amcp_github, pr_first_openrouter_orchestrator, and ef_compression_benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from hivemind.swarm.amcp_collab import (
    AGENT,
    BEE,
    HUMAN,
    CollabError,
    EventRule,
    Field,
    check_publish,
    dev_keyring,
)
from hivemind.swarm.amcp_github import SeenEvents, gh_fetch, poll, to_payload
from hivemind.swarm.ef_compression_benchmark import (
    _local_steps,
    _mse,
    _topk_fp16,
    build_problem,
    run_benchmark,
)
from hivemind.swarm.pr_first_openrouter_orchestrator import (
    CANONICAL_5_ROADMAP_PRS,
    PRFirstOpenRouterCodingOrchestrator,
    PRVerificationReceipt,
    RoadmapPRSpec,
    scan_python_ast_security,
)


# ---------------------------------------------------------------------------
# 1. amcp_collab hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "code,msg",
    [
        ("", "valid"),
        ("   ", "valid"),
        (123, "valid"),
        ("valid", ""),
        ("valid", "   "),
        ("valid", None),
    ],
)
def test_collab_error_rejects_empty_or_non_string_code_and_message(code: object, msg: object) -> None:
    with pytest.raises(ValueError):
        CollabError(code, msg)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kind": "invalid_kind"},
        {"kind": "str", "max_len": 0},
        {"kind": "str", "max_len": -5},
        {"kind": "str", "max_len": True},
        {"kind": "str", "required": 1},  # type: ignore[dict-item]
        {"kind": "int", "lo": True, "hi": 10},
        {"kind": "int", "lo": 0, "hi": False},
        {"kind": "int", "lo": 20, "hi": 10},
        {"kind": "choice", "choices": ()},
        {"kind": "choice", "choices": ("ok", "")},
        {"kind": "choice", "choices": ["a", "b"]},  # type: ignore[dict-item]
    ],
)
def test_field_post_init_rejects_invalid_specs(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Field(**kwargs)


def test_field_check_rejects_empty_field_name() -> None:
    f = Field("str", 64)
    with pytest.raises(ValueError, match="field name"):
        f.check("", "hello")
    with pytest.raises(ValueError, match="field name"):
        f.check("   ", "hello")


@pytest.mark.parametrize(
    "fields,kinds,direct_only",
    [
        ("not-a-dict", frozenset({HUMAN}), False),
        ({"": Field("str")}, frozenset({HUMAN}), False),
        ({"x": "not-a-field"}, frozenset({HUMAN}), False),
        ({"x": Field("str")}, frozenset(), False),
        ({"x": Field("str")}, frozenset({"unknown_kind"}), False),
        ({"x": Field("str")}, [HUMAN], False),
        ({"x": Field("str")}, frozenset({HUMAN}), "yes"),
    ],
)
def test_event_rule_post_init_rejects_invalid_specs(fields: object, kinds: object, direct_only: object) -> None:
    with pytest.raises(ValueError):
        EventRule(fields=fields, kinds=kinds, direct_only=direct_only)  # type: ignore[arg-type]


def test_dev_keyring_rejects_non_bool_create(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="create must be a boolean"):
        dev_keyring(tmp_path / "k", create=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs,expected_code",
    [
        ({"role": "consumer", "sender_kind": HUMAN, "topic_key": "human_agent", "event_type": "io.hivemind.amcp.message", "payload": {}, "via_gateway": 0}, "payload_invalid"),
        ({"role": "consumer", "sender_kind": HUMAN, "topic_key": "human_agent", "event_type": "", "payload": {}}, "payload_invalid"),
        ({"role": "consumer", "sender_kind": HUMAN, "topic_key": "human_agent", "event_type": "io.hivemind\x00.msg", "payload": {}}, "payload_invalid"),
        ({"role": "consumer", "sender_kind": HUMAN, "topic_key": "human_agent", "event_type": "io.hivemind.amcp.custom", "payload": ["not", "a", "dict"]}, "payload_invalid"),
        ({"role": "consumer", "sender_kind": HUMAN, "topic_key": "human_agent", "event_type": "io.hivemind.amcp.message", "payload": "not-a-dict"}, "payload_invalid"),
        ({"role": 123, "sender_kind": HUMAN, "topic_key": "human_agent", "event_type": "io.hivemind.amcp.message", "payload": {}}, "unknown_role"),
        ({"role": "consumer", "sender_kind": 123, "topic_key": "human_agent", "event_type": "io.hivemind.amcp.message", "payload": {}}, "unknown_kind"),
        ({"role": "consumer", "sender_kind": HUMAN, "topic_key": 123, "event_type": "io.hivemind.amcp.message", "payload": {}}, "unknown_topic"),
    ],
)
def test_check_publish_hardened_guards(kwargs: dict, expected_code: str) -> None:
    with pytest.raises(CollabError) as exc_info:
        check_publish(**kwargs)
    assert exc_info.value.code == expected_code


# ---------------------------------------------------------------------------
# 2. amcp_github hardening
# ---------------------------------------------------------------------------


def test_to_payload_rejects_non_mapping_and_handles_corrupted_nested_types() -> None:
    with pytest.raises(TypeError, match="event must be a Mapping"):
        to_payload("not-a-dict")  # type: ignore[arg-type]

    for bad_event in [
        {"type": "IssuesEvent", "payload": "not-a-dict", "actor": {"login": "alice"}, "repo": {"name": "a/b"}},
        {"type": "IssuesEvent", "payload": {"action": "opened", "issue": "not-a-dict"}, "actor": {"login": "alice"}, "repo": {"name": "a/b"}},
        {"type": "PullRequestEvent", "payload": {"action": "opened", "pull_request": 123}, "actor": {"login": "alice"}, "repo": {"name": "a/b"}},
        {"type": "IssueCommentEvent", "payload": {"action": "created", "issue": {}, "comment": "bad"}, "actor": {"login": "alice"}, "repo": {"name": "a/b"}},
        {"type": "PullRequestReviewEvent", "payload": {"action": "submitted", "pull_request": {}, "review": "bad"}, "actor": {"login": "alice"}, "repo": {"name": "a/b"}},
        {"type": "ReleaseEvent", "payload": {"action": "published", "release": "bad"}, "actor": {"login": "alice"}, "repo": {"name": "a/b"}},
    ]:
        assert to_payload(bad_event) is None


@pytest.mark.parametrize("bad_keep", [0, -1, True, False, "500"])
def test_seen_events_rejects_invalid_keep(bad_keep: object) -> None:
    with pytest.raises(ValueError, match="keep must be a positive integer"):
        SeenEvents(None, keep=bad_keep)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_id", ["", "   ", None, 123])
def test_seen_events_add_rejects_invalid_event_id(bad_id: object) -> None:
    seen = SeenEvents(None, keep=10)
    with pytest.raises(ValueError, match="event_id must be a non-empty string"):
        seen.add(bad_id)  # type: ignore[arg-type]


def test_seen_events_corrupted_file_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "corrupt.json"
    bad_file.write_text('{"ids": "not-a-list"}', encoding="utf-8")
    with pytest.raises(ValueError, match="Corrupted SeenEvents"):
        SeenEvents(bad_file)


def test_poll_and_gh_fetch_validate_arguments_and_skip_idless_events() -> None:
    seen = SeenEvents(None)
    with pytest.raises(TypeError, match="fetch must be callable"):
        poll("not-callable", "owner/repo", seen, lambda _: None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="repo must match"):
        poll(lambda _: [], "../invalid/repo", seen, lambda _: None)
    with pytest.raises(TypeError, match="seen must be a SeenEvents"):
        poll(lambda _: [], "owner/repo", "not-seen", lambda _: None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="publish must be callable"):
        poll(lambda _: [], "owner/repo", seen, "not-callable")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="backfill must be a boolean"):
        poll(lambda _: [], "owner/repo", seen, lambda _: None, backfill="yes")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_per_cycle must be a positive integer"):
        poll(lambda _: [], "owner/repo", seen, lambda _: None, max_per_cycle=0)
    with pytest.raises(ValueError, match="max_per_cycle must be a positive integer"):
        poll(lambda _: [], "owner/repo", seen, lambda _: None, max_per_cycle=True)

    # Events missing id or with boolean/empty id are ignored safely
    res = poll(lambda _: [{"type": "IssuesEvent"}, {"id": "", "type": "IssuesEvent"}, {"id": True}], "owner/repo", seen, lambda _: None, backfill=True)
    assert res == {"fetched": 3, "new": 0, "published": 0, "skipped": 0}

    with pytest.raises(ValueError, match="repo must match"):
        gh_fetch("owner/repo/extra")


# ---------------------------------------------------------------------------
# 3. pr_first_openrouter_orchestrator hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pr_order": 0, "roadmap_id": "HM-1", "title": "T", "module_path": "a.py", "test_path": "t.py"},
        {"pr_order": True, "roadmap_id": "HM-1", "title": "T", "module_path": "a.py", "test_path": "t.py"},
        {"pr_order": 1, "roadmap_id": "", "title": "T", "module_path": "a.py", "test_path": "t.py"},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "  ", "module_path": "a.py", "test_path": "t.py"},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "T", "module_path": "/abs/a.py", "test_path": "t.py"},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "T", "module_path": "../escape.py", "test_path": "t.py"},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "T", "module_path": "a.py", "test_path": "../escape_test.py"},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "T", "module_path": "a.py", "test_path": "t.py", "openrouter_model": ""},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "T", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": "   "},
        {"pr_order": 1, "roadmap_id": "HM-1", "title": "T", "module_path": "a.py", "test_path": "t.py", "openrouter_request_id": ""},
    ],
)
def test_roadmap_pr_spec_rejects_invalid_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        RoadmapPRSpec(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"roadmap_id": "", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": True, "forbidden_calls_found": (), "line_count": 60, "ready_to_merge": True},
        {"roadmap_id": "HM-1", "module_path": "../a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": True, "forbidden_calls_found": (), "line_count": 60, "ready_to_merge": True},
        {"roadmap_id": "HM-1", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": True, "forbidden_calls_found": ("eval",), "line_count": 60, "ready_to_merge": False},
        {"roadmap_id": "HM-1", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": False, "forbidden_calls_found": (), "line_count": 60, "ready_to_merge": False},
        {"roadmap_id": "HM-1", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": False, "forbidden_calls_found": ("eval",), "line_count": 60, "ready_to_merge": True},
        {"roadmap_id": "HM-1", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": True, "forbidden_calls_found": (), "line_count": 20, "ready_to_merge": True},
        {"roadmap_id": "HM-1", "module_path": "a.py", "test_path": "t.py", "openrouter_completion_id": None, "openrouter_request_id": None, "ast_security_clean": True, "forbidden_calls_found": (), "line_count": True, "ready_to_merge": False},
    ],
)
def test_pr_verification_receipt_rejects_inconsistent_fields(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        PRVerificationReceipt(**kwargs)


def test_scan_python_ast_security_detects_builtins_and_rejects_non_str() -> None:
    with pytest.raises(TypeError, match="source_code must be a string"):
        scan_python_ast_security(None)  # type: ignore[arg-type]

    clean, found = scan_python_ast_security("builtins.eval('1')\nos.popen('ls')\n")
    assert clean is False
    assert "builtins.eval" in found and "os.popen" in found


def test_orchestrator_validates_repo_root_and_specs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="repo_root"):
        PRFirstOpenRouterCodingOrchestrator("")  # type: ignore[arg-type]

    orch = PRFirstOpenRouterCodingOrchestrator(tmp_path)
    with pytest.raises(TypeError, match="RoadmapPRSpec"):
        orch.verify_spec("not-a-spec")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty sequence"):
        orch.verify_all_roadmap_prs(())
    with pytest.raises(ValueError, match="Duplicate roadmap_id"):
        orch.verify_all_roadmap_prs([CANONICAL_5_ROADMAP_PRS[0], CANONICAL_5_ROADMAP_PRS[0]])


# ---------------------------------------------------------------------------
# 4. ef_compression_benchmark hardening
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"seed": True},
        {"num_workers": 0},
        {"num_workers": False},
        {"samples_per_worker": 0},
        {"d_in": 0},
        {"d_out": 0},
        {"noise_std": -0.1},
        {"noise_std": float("nan")},
        {"noise_std": True},
    ],
)
def test_build_problem_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        build_problem(**kwargs)


def test_mse_local_steps_and_topk_fp16_validate_tensors() -> None:
    w = torch.zeros(4, 8)
    x = torch.ones(10, 8)
    y = torch.ones(10, 4)
    assert _mse(w, x, y) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="Incompatible tensor shapes"):
        _mse(torch.zeros(4), x, y)
    with pytest.raises(ValueError, match="finite"):
        _mse(torch.full((4, 8), float("nan")), x, y)

    with pytest.raises(ValueError, match="steps"):
        _local_steps(w, x, y, 0, 0.05)
    with pytest.raises(ValueError, match="lr"):
        _local_steps(w, x, y, 2, 0.0)
    with pytest.raises(ValueError, match="lr"):
        _local_steps(w, x, y, 2, float("inf"))

    with pytest.raises(ValueError, match="non-empty finite"):
        _topk_fp16(torch.tensor([]), 0.1)
    with pytest.raises(ValueError, match="topk_ratio"):
        _topk_fp16(torch.ones(16), 0.0)
    with pytest.raises(ValueError, match="topk_ratio"):
        _topk_fp16(torch.ones(16), 1.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": -1},
        {"num_workers": 0},
        {"outer_rounds": 0},
        {"inner_steps": 0},
        {"log_every": 0},
        {"inner_lr": 0.0},
        {"outer_lr": -1.0},
        {"inner_lr": float("nan")},
        {"schemes": ()},
        {"schemes": "dense"},
        {"schemes": [123]},  # type: ignore[list-item]
    ],
)
def test_run_benchmark_rejects_invalid_parameters(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        run_benchmark(**kwargs)
