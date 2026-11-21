#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""HiveMind Autonomous Control Agent powered by the Google Antigravity SDK.

Connects the operator's HiveMind account (``~/.hivemind/credentials.json`` or ``HIVEMIND_API_KEY``)
and GitHub account with a stateful Google Antigravity SDK ``Agent`` to:
  1. Track real external vs. internal adoption (registered accounts by role, active trainers/producers,
     14-day GitHub unique visitors & clones, onboarding ease ratings, and zero-game-lag comfort).
  2. Monitor active campaigns (``Winter GPU Raid Noel 2026``, ``CROWDGPT-MOE-AUTORESEARCH``,
     ``CROWDGPT-MOE-8X-UPCYCLE``, ``OpenTrainer`` accepted rounds, and T4 VRAM headroom) without
     ever sending inference traffic to ``hivemind/coder``.
  3. Continuously self-improve onboarding & platform quality by diagnosing UX friction, queuing
     small-perimeter test-gated slices (``>= 95%`` coverage), and persisting session memory via
     ``LocalAgentConfig(save_dir=..., conversation_id=...)``.

Compliance & Safety Rules:
  - Personal home PCs and personal hobby/open-source projects ONLY (never corporate laptops/data).
  - Symbolic compute points have 0.00 EUR / 0.00 USD monetary value.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_GATEWAY = "https://hivemind-openrouter-cfhmhmvv5a-uc.a.run.app"
DEFAULT_OPERATOR_HANDLE = "xavier-agy-control"
DEFAULT_GITHUB_REPO = "xaviercallens/HiveMind"
NOEL_LAUNCH_DATE = date(2026, 12, 19)
NOEL_END_DATE = date(2027, 1, 3)
HANDLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{1,47}$")

HttpJsonFn = Callable[[str, str, dict[str, Any] | None, dict[str, str] | None], tuple[int, dict[str, Any]]]
GhJsonFn = Callable[[list[str]], dict[str, Any] | list[Any] | None]


class SelfImprovementAction(BaseModel):
    """Concrete, small-perimeter improvement proposed or queued by the Control Agent."""

    slice_id: str = Field(description="Short identifier such as SLICE-03 or SLICE-DYN-01")
    title: str = Field(description="Concise PR-ready title")
    priority: str = Field(description="P0, P1, or P2")
    target_metric: str = Field(description="Measured adoption or comfort KPI improved by this slice")
    hypothesis: str = Field(description="Why this change reduces user drop-off or improves reliability")
    test_gate: str = Field(description="Automated pytest and >=95% coverage verification criteria")


class CampaignStatusSummary(BaseModel):
    """Live summary of an active training or adoption campaign."""

    campaign_id: str
    phase: str
    headline_metric: str
    healthy: bool
    blocker: str = ""


class ControlCycleReport(BaseModel):
    """Structured executive output returned by an Antigravity SDK control cycle."""

    generated_utc: str
    operator_handle: str
    adoption_stage: str = Field(description="pre_launch, canary_alpha, holiday_raid, or post_raid")
    external_adoption_verdict: str
    days_until_noel_launch: int
    key_metrics: dict[str, str]
    campaigns: list[CampaignStatusSummary]
    top_friction_bottleneck: str
    next_actions: list[SelfImprovementAction]


def default_state_dir() -> Path:
    return Path.home() / ".hivemind" / "agy_control_agent"


def default_credentials_path() -> Path:
    return Path.home() / ".hivemind" / "credentials.json"


def mask_api_key(token: str) -> str:
    """Mask an ``hm-...`` API key so secrets are never written to stdout or logs."""
    cleaned = (token or "").strip()
    if not cleaned:
        return "<unset>"
    if len(cleaned) <= 8:
        return "hm-****"
    prefix = cleaned[:3] if cleaned.startswith("hm-") else "key-"
    return f"{prefix}****{cleaned[-4:]}"


def validate_gateway_url(url: str) -> str:
    cleaned = (url or "").strip().rstrip("/")
    if not (
        (cleaned.startswith("https://") and cleaned.split("/")[2].endswith(".run.app"))
        or cleaned.startswith("http://127.0.0.1")
        or cleaned.startswith("http://localhost")
    ):
        raise ValueError(
            f"Refusing untrusted gateway URL {url!r}: expected an HTTPS *.run.app gateway or localhost."
        )
    return cleaned


def sanitize_handle(raw: str) -> str:
    candidate = re.sub(r"[^A-Za-z0-9._+-]+", "-", (raw or "").strip()).strip("-.")
    if len(candidate) < 2:
        candidate = f"agy-{secrets.token_hex(3)}"
    candidate = candidate[:48]
    if not HANDLE_RE.match(candidate) or candidate.lower().startswith(("google.", "github.", "oidc.")):
        raise ValueError("Invalid HiveMind handle.")
    return candidate


def default_http_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    validate_gateway_url(url)
    req_headers = {"Accept": "application/json", "User-Agent": "hivemind-agy-control-agent/1.0"}
    if headers:
        req_headers.update(headers)
    body_bytes: bytes | None = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body_bytes, headers=req_headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15.0) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(raw) if raw.strip() else {}
            return int(resp.status), parsed if isinstance(parsed, dict) else {"data": parsed}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            parsed = {"raw_error": raw[:300]}
        return int(exc.code), parsed if isinstance(parsed, dict) else {"data": parsed}
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return 0, {"network_error": str(exc)}


def default_gh_json(args: list[str]) -> dict[str, Any] | list[Any] | None:
    """Run read-only ``gh`` CLI commands with ``GITHUB_TOKEN`` and ``GH_TOKEN`` unset."""
    env = {k: v for k, v in os.environ.items() if k not in ("GITHUB_TOKEN", "GH_TOKEN")}
    try:
        proc = subprocess.run(
            ["gh", *args],
            env=env,
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def ensure_operator_account(
    *,
    gateway: str = DEFAULT_GATEWAY,
    preferred_handle: str = DEFAULT_OPERATOR_HANDLE,
    credentials_path: Path | None = None,
    http_fn: HttpJsonFn = default_http_json,
) -> dict[str, Any]:
    """Load, verify, or provision the operator's HiveMind account credentials safely."""
    gw = validate_gateway_url(gateway)
    cred_file = credentials_path or default_credentials_path()
    api_key = os.environ.get("HIVEMIND_API_KEY", "").strip()
    saved: dict[str, Any] = {}

    if not api_key and cred_file.exists():
        try:
            loaded = json.loads(cred_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                saved = loaded
                api_key = str(loaded.get("api_key", "")).strip()
        except (OSError, json.JSONDecodeError):
            saved = {}

    created_new = False
    handle = str(saved.get("handle") or preferred_handle)

    if api_key:
        code, me_data = http_fn(
            "GET",
            f"{gw}/v1/hivemind/me",
            None,
            {"Authorization": f"Bearer {api_key}"},
        )
        if code == 200:
            handle = str(me_data.get("handle") or handle)
            return {
                "authenticated": True,
                "created": False,
                "gateway": gw,
                "handle": handle,
                "account_id": str(me_data.get("account_id", saved.get("account_id", ""))),
                "role": str(me_data.get("role", saved.get("role", "prosumer"))),
                "masked_key": mask_api_key(api_key),
                "credentials_path": str(cred_file),
            }

    clean_handle = sanitize_handle(preferred_handle)
    reg_code, reg_data = http_fn(
        "POST",
        f"{gw}/v1/hivemind/register",
        {"handle": clean_handle, "role": "prosumer"},
        None,
    )
    if reg_code in (200, 201) and reg_data.get("api_key"):
        api_key = str(reg_data["api_key"])
        handle = str(reg_data.get("handle") or clean_handle)
        created_new = True
    else:
        fallback_handle = sanitize_handle(f"{clean_handle[:36]}-{secrets.token_hex(2)}")
        reg_code, reg_data = http_fn(
            "POST",
            f"{gw}/v1/hivemind/register",
            {"handle": fallback_handle, "role": "prosumer"},
            None,
        )
        if reg_code not in (200, 201) or not reg_data.get("api_key"):
            return {
                "authenticated": False,
                "created": False,
                "gateway": gw,
                "handle": clean_handle,
                "masked_key": "<unset>",
                "error": f"Registration returned HTTP {reg_code}",
            }
        api_key = str(reg_data["api_key"])
        handle = str(reg_data.get("handle") or fallback_handle)
        created_new = True

    cred_file.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "gateway": gw,
        "handle": handle,
        "account_id": str(reg_data.get("account_id", "")),
        "role": str(reg_data.get("role", "prosumer")),
        "api_key": api_key,
        "saved_at_utc": datetime.now(UTC).isoformat(),
    }
    cred_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if os.name != "nt":
        cred_file.chmod(0o600)

    return {
        "authenticated": True,
        "created": created_new,
        "gateway": gw,
        "handle": handle,
        "account_id": record["account_id"],
        "role": record["role"],
        "masked_key": mask_api_key(api_key),
        "credentials_path": str(cred_file),
    }


def collect_platform_snapshot(
    *,
    gateway: str = DEFAULT_GATEWAY,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    today: date | None = None,
    http_fn: HttpJsonFn = default_http_json,
    gh_fn: GhJsonFn = default_gh_json,
) -> dict[str, Any]:
    """Gather read-only adoption, campaign, and satisfaction telemetry and compute deltas."""
    gw = validate_gateway_url(gateway)
    sdir = state_dir or default_state_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    history_path = sdir / "telemetry_history.jsonl"

    now_dt = datetime.now(UTC)
    ref_day = today or now_dt.date()
    days_to_launch = (NOEL_LAUNCH_DATE - ref_day).days
    if ref_day < date(2026, 12, 5):
        stage = "pre_launch"
    elif ref_day < NOEL_LAUNCH_DATE:
        stage = "canary_alpha"
    elif ref_day <= NOEL_END_DATE:
        stage = "holiday_raid"
    else:
        stage = "post_raid"

    health_code, _ = http_fn("GET", f"{gw}/health", None, None)
    models_code, models_body = http_fn("GET", f"{gw}/v1/models", None, None)
    live_code, live_body = http_fn("GET", f"{gw}/v1/hivemind/live-telemetry", None, None)
    fleet_code, fleet_body = http_fn("GET", f"{gw}/v1/hivemind/producers/fleet", None, None)
    t4_code, t4_body = http_fn("GET", f"{gw}/v1/hivemind/t4/status", None, None)
    fb_code, fb_body = http_fn("GET", f"{gw}/v1/hivemind/onboarding/feedback/summary", None, None)

    accounts_info = live_body.get("accounts") if isinstance(live_body.get("accounts"), dict) else {}
    by_role = accounts_info.get("by_role") if isinstance(accounts_info.get("by_role"), dict) else {}
    opentrainer_info = live_body.get("opentrainer") if isinstance(live_body.get("opentrainer"), dict) else {}
    models_list = models_body.get("data") if isinstance(models_body.get("data"), list) else []

    producers = fleet_body.get("producers") if isinstance(fleet_body.get("producers"), list) else []
    online_producers = sum(
        1 for p in producers if isinstance(p, dict) and str(p.get("status", "")).lower() in ("online", "healthy", "ready")
    )

    repo_meta = gh_fn(["api", f"repos/{github_repo}"])
    views_meta = gh_fn(["api", f"repos/{github_repo}/traffic/views"])
    clones_meta = gh_fn(["api", f"repos/{github_repo}/traffic/clones"])
    open_prs_meta = gh_fn(["pr", "list", "-R", github_repo, "--state", "open", "--json", "number,isDraft", "--limit", "50"])

    stars = int(repo_meta.get("stargazers_count", 0)) if isinstance(repo_meta, dict) else None
    forks = int(repo_meta.get("forks_count", 0)) if isinstance(repo_meta, dict) else None
    views_uniques_14d = int(views_meta.get("uniques", 0)) if isinstance(views_meta, dict) else None
    clones_uniques_14d = int(clones_meta.get("uniques", 0)) if isinstance(clones_meta, dict) else None
    open_prs_count = len(open_prs_meta) if isinstance(open_prs_meta, list) else None

    total_accounts = accounts_info.get("total_registered_accounts")
    accepted_rounds = opentrainer_info.get("accepted_rounds_count")
    trainer_nodes = opentrainer_info.get("total_registered_nodes")

    prev_snapshot: dict[str, Any] | None = None
    if history_path.exists():
        lines = [ln for ln in history_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        if lines:
            try:
                prev_snapshot = json.loads(lines[-1])
            except json.JSONDecodeError:
                prev_snapshot = None

    def _delta(curr: int | None, prev_key: str) -> int | None:
        if curr is None or not isinstance(prev_snapshot, dict):
            return None
        prev_val = prev_snapshot.get(prev_key)
        return (int(curr) - int(prev_val)) if isinstance(prev_val, int) else None

    snapshot: dict[str, Any] = {
        "captured_utc": now_dt.isoformat(),
        "adoption_stage": stage,
        "days_until_noel_launch": days_to_launch,
        "gateway_healthy": health_code == 200 and models_code == 200,
        "live_models_count": len(models_list),
        "total_registered_accounts": int(total_accounts) if isinstance(total_accounts, int) else None,
        "accounts_by_role": by_role,
        "online_producers": online_producers,
        "trainer_nodes": int(trainer_nodes) if isinstance(trainer_nodes, int) else None,
        "accepted_opentrainer_rounds": int(accepted_rounds) if isinstance(accepted_rounds, int) else None,
        "t4_status_http": t4_code,
        "t4_vram_free_mib": (t4_body.get("gpu") or {}).get("vram_free_mib")
        if isinstance(t4_body.get("gpu"), dict)
        else t4_body.get("vram_free_mib"),
        "onboarding_feedback_deployed": fb_code == 200,
        "onboarding_feedback_total": int(fb_body.get("total_responses", 0)) if fb_code == 200 else 0,
        "onboarding_average_ease_rating": fb_body.get("average_ease_rating") if fb_code == 200 else None,
        "onboarding_zero_game_lag_ratio": fb_body.get("zero_game_lag_ratio") if fb_code == 200 else None,
        "github_stars": stars,
        "github_forks": forks,
        "github_views_uniques_14d": views_uniques_14d,
        "github_clones_uniques_14d": clones_uniques_14d,
        "github_open_prs": open_prs_count,
        "deltas_since_previous": {
            "accounts": _delta(int(total_accounts) if isinstance(total_accounts, int) else None, "total_registered_accounts"),
            "accepted_rounds": _delta(int(accepted_rounds) if isinstance(accepted_rounds, int) else None, "accepted_opentrainer_rounds"),
            "github_stars": _delta(stars, "github_stars"),
            "github_views_uniques_14d": _delta(views_uniques_14d, "github_views_uniques_14d"),
        },
        "live_telemetry_http": live_code,
        "fleet_http": fleet_code,
    }

    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, sort_keys=True) + "\n")

    return snapshot


def evaluate_readiness_gates(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Evaluate objective adoption, UX satisfaction, and campaign readiness gates."""
    avg_ease = snapshot.get("onboarding_average_ease_rating")
    zero_lag = snapshot.get("onboarding_zero_game_lag_ratio")
    fb_deployed = bool(snapshot.get("onboarding_feedback_deployed"))
    gw_ok = bool(snapshot.get("gateway_healthy"))
    by_role = snapshot.get("accounts_by_role") if isinstance(snapshot.get("accounts_by_role"), dict) else {}
    role_trainers = int(by_role.get("trainer", 0)) if isinstance(by_role.get("trainer"), int) else 0
    online_prods = int(snapshot.get("online_producers", 0)) if isinstance(snapshot.get("online_producers"), int) else 0
    active_volunteer_gpus = max(role_trainers, online_prods)

    gates = {
        "gateway_and_models_online": gw_ok,
        "holiday_routes_deployed_on_cloud_run": fb_deployed,
        "satisfaction_ces_target_met": isinstance(avg_ease, (int, float)) and float(avg_ease) >= 4.2,
        "gamer_zero_lag_target_met": isinstance(zero_lag, (int, float)) and float(zero_lag) >= 0.95,
        "external_volunteer_gpu_target_noel": active_volunteer_gpus >= 8,
    }
    blockers: list[str] = []
    if not gw_ok:
        blockers.append("Cloud Run gateway /health or /v1/models is unreachable.")
    if not fb_deployed:
        blockers.append(
            "Cloud Run revision has not yet been redeployed with PR #949 (/noel, /quick, /v1/hivemind/onboarding/feedback)."
        )
    if avg_ease is None:
        blockers.append("Zero onboarding satisfaction responses recorded yet (target CES >= 4.2/5).")
    elif float(avg_ease) < 4.2:
        blockers.append(f"Average ease_rating {avg_ease} is below 4.2/5 target.")
    if zero_lag is not None and float(zero_lag) < 0.95:
        blockers.append(f"Zero-game-lag comfort ratio {zero_lag} is below 0.95 target.")
    if active_volunteer_gpus < 8:
        blockers.append(
            f"Active external trainer/online producer GPUs ({active_volunteer_gpus}) below Christmas target (>= 8 external friend GPUs)."
        )

    return {
        "gates": gates,
        "passed_count": sum(1 for v in gates.values() if v),
        "total_gates": len(gates),
        "blockers": blockers,
    }


def is_unsafe_shell_command(args: dict[str, Any]) -> bool:
    """Safety policy predicate blocking destructive commands and live coder inference."""
    cmd = str(args.get("CommandLine") or args.get("command") or "").strip()
    if not cmd:
        return False
    lower = cmd.lower()
    tokens = set(re.split(r"\s+", lower))
    if "rm" in tokens and ("-rf" in tokens or "-fr" in tokens):
        return True
    if "git" in tokens and "push" in tokens and ("--force" in tokens or "-f" in tokens):
        return True
    if any(t.startswith("--no-") and t.endswith("verify") for t in tokens):
        return True
    if ("curl" in tokens or "wget" in tokens) and "|" in lower and ("sh" in tokens or "bash" in tokens):
        return True
    if "/v1/chat/completions" in lower and "hivemind/coder" in lower:
        return True
    return False


def load_or_read_backlog(backlog_path: Path | None = None) -> dict[str, Any]:
    if backlog_path and backlog_path.exists():
        try:
            data = json.loads(backlog_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass
    return {"slices": []}


def append_improvement_proposal(
    *,
    slice_id: str,
    title: str,
    priority: str,
    target_metric: str,
    hypothesis: str,
    test_plan: str,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Persist a concrete, test-gated improvement action proposed by the Control Agent."""
    if priority not in ("P0", "P1", "P2"):
        raise ValueError("priority must be P0, P1, or P2")
    if not slice_id.strip() or not title.strip():
        raise ValueError("slice_id and title must be non-empty")
    sdir = state_dir or default_state_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    record = {
        "recorded_utc": datetime.now(UTC).isoformat(),
        "slice_id": slice_id.strip(),
        "title": title.strip(),
        "priority": priority,
        "target_metric": target_metric.strip(),
        "hypothesis": hypothesis.strip(),
        "test_plan": test_plan.strip(),
    }
    out_file = sdir / "improvement_proposals.jsonl"
    with out_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
    return record


def build_control_tools(
    *,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    credentials_path: Path | None = None,
    http_fn: HttpJsonFn = default_http_json,
    gh_fn: GhJsonFn = default_gh_json,
) -> list[Callable[..., str]]:
    """Create the custom tool functions registered with ``LocalAgentConfig(tools=...)``."""
    sdir = state_dir or default_state_dir()

    def check_operator_account() -> str:
        """Verify and return the operator's authenticated HiveMind account status (masked key, handle, role)."""
        res = ensure_operator_account(
            gateway=gateway,
            preferred_handle=operator_handle,
            credentials_path=credentials_path,
            http_fn=http_fn,
        )
        return json.dumps(res, indent=2, sort_keys=True)

    def collect_adoption_and_campaign_snapshot() -> str:
        """Collect live HiveMind gateway, T4 training, onboarding feedback, and GitHub adoption metrics."""
        snap = collect_platform_snapshot(
            gateway=gateway,
            github_repo=github_repo,
            state_dir=sdir,
            http_fn=http_fn,
            gh_fn=gh_fn,
        )
        return json.dumps(snap, indent=2, sort_keys=True)

    def evaluate_campaign_readiness_gates() -> str:
        """Evaluate objective readiness gates and blockers for the Christmas Holiday adoption campaign."""
        snap = collect_platform_snapshot(
            gateway=gateway,
            github_repo=github_repo,
            state_dir=sdir,
            http_fn=http_fn,
            gh_fn=gh_fn,
        )
        gates = evaluate_readiness_gates(snap)
        return json.dumps(gates, indent=2, sort_keys=True)

    def inspect_backlog_and_pr_velocity() -> str:
        """Inspect the Christmas onboarding improvement backlog and open PRs on GitHub."""
        backlog = load_or_read_backlog(backlog_path)
        slices = backlog.get("slices") if isinstance(backlog.get("slices"), list) else []
        counts: dict[str, int] = {"merged": 0, "in_pr": 0, "pending": 0}
        next_pending: list[dict[str, Any]] = []
        for item in slices:
            if not isinstance(item, dict):
                continue
            st = str(item.get("status", "pending"))
            counts[st] = counts.get(st, 0) + 1
            if st == "pending" and len(next_pending) < 3:
                next_pending.append({"id": item.get("id"), "title": item.get("title"), "week": item.get("week")})
        prs = gh_fn(["pr", "list", "-R", github_repo, "--state", "open", "--json", "number,title,isDraft", "--limit", "15"])
        return json.dumps(
            {
                "slice_counts": counts,
                "next_pending_slices": next_pending,
                "open_prs_sample": prs if isinstance(prs, list) else [],
            },
            indent=2,
            sort_keys=True,
        )

    def record_self_improvement_action(
        slice_id: str,
        title: str,
        priority: str,
        target_metric: str,
        hypothesis: str,
        test_plan: str,
    ) -> str:
        """Record a concrete, test-gated UX or reliability self-improvement proposal.

        Args:
            slice_id: Identifier such as SLICE-03 or SLICE-DYN-01.
            title: Concise PR-ready title.
            priority: One of P0, P1, or P2.
            target_metric: Exact adoption or comfort metric targeted.
            hypothesis: Why this slice reduces user friction or improves model/platform quality.
            test_plan: Automated pytest and >=95% line coverage plan.
        """
        saved = append_improvement_proposal(
            slice_id=slice_id,
            title=title,
            priority=priority,
            target_metric=target_metric,
            hypothesis=hypothesis,
            test_plan=test_plan,
            state_dir=sdir,
        )
        return json.dumps(saved, indent=2, sort_keys=True)

    return [
        check_operator_account,
        collect_adoption_and_campaign_snapshot,
        evaluate_campaign_readiness_gates,
        inspect_backlog_and_pr_velocity,
        record_self_improvement_action,
    ]


SYSTEM_INSTRUCTIONS = """You are the HiveMind Autonomous Control Agent for Xavier Callens (@xaviercallens).
Your mission is to:
1. Track honest, measured adoption and satisfaction (never invent numbers; report 'n/a' when unavailable).
2. Monitor active training & holiday campaigns (Winter GPU Raid Noel 2026 starting 2026-12-19, MoE Autoresearch, OpenTrainer) WITHOUT ever sending inference prompts to hivemind/coder while GPU training is active.
3. Drive continuous, small-perimeter self-improvement focused on extreme ease of installation and connection (<60s Dev VS Code setup, <5min Windows 11 Gamer-Whisper setup, <1.5s Steam GPU yield, <=70 C quiet fans, >=95% pytest coverage).
4. Strictly enforce the Amadeus / Corporate Compliance Rule: 100% personal home PCs and personal hobby/open-source projects only (never corporate laptops or enterprise code/data), and 0.00 EUR / 0.00 USD symbolic points only.

Always call your tools (`check_operator_account`, `collect_adoption_and_campaign_snapshot`, `evaluate_campaign_readiness_gates`, `inspect_backlog_and_pr_velocity`) before producing conclusions, and call `record_self_improvement_action` for your highest-priority concrete next improvement."""


def load_saved_conversation_id(state_dir: Path) -> str | None:
    session_file = state_dir / "session_state.json"
    if not session_file.exists():
        return None
    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        cid = str(data.get("conversation_id", "")).strip()
        return cid or None
    except (OSError, json.JSONDecodeError):
        return None


def save_conversation_id(state_dir: Path, conversation_id: str | None) -> None:
    if not conversation_id:
        return
    state_dir.mkdir(parents=True, exist_ok=True)
    session_file = state_dir / "session_state.json"
    session_file.write_text(
        json.dumps(
            {
                "conversation_id": conversation_id,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def build_antigravity_agent_config(
    *,
    tools: list[Callable[..., Any]],
    state_dir: Path,
    workspace_dir: Path,
    response_schema: type[BaseModel] | None = ControlCycleReport,
    resume_session: bool = True,
    periodic_interval_s: float | None = None,
    vertex: bool = False,
    project: str | None = None,
    location: str = "us-central1",
    model: str | None = None,
) -> Any:
    """Build a verified ``google.antigravity.LocalAgentConfig`` with subagents, policies, and persistence."""
    from google.antigravity import CapabilitiesConfig, CompactionConfig, LocalAgentConfig, types
    from google.antigravity.hooks import policy
    from google.antigravity.triggers import TriggerContext, every

    save_dir = (state_dir / "conversations").resolve()
    app_data_dir = (state_dir / "app_data").resolve()
    save_dir.mkdir(parents=True, exist_ok=True)
    app_data_dir.mkdir(parents=True, exist_ok=True)

    subagents = [
        types.SubagentConfig(
            name="adoption_analyst",
            description="Analyzes external vs internal adoption funnel, GitHub unique visitors, and CES satisfaction.",
            system_instructions="Analyze measured adoption & UX satisfaction data objectively. Never invent metrics.",
            capabilities=types.SubagentCapabilities(
                agent_behavior=types.AgentBehavior.AUTONOMOUS,
                enabled_tools=[types.BuiltinTools.VIEW_FILE, types.BuiltinTools.LIST_DIR],
            ),
        ),
        types.SubagentConfig(
            name="campaign_tracker",
            description="Monitors MoE Autoresearch, OpenTrainer rounds, and Winter GPU Raid readiness.",
            system_instructions="Monitor training & holiday campaign readiness without querying hivemind/coder.",
            capabilities=types.SubagentCapabilities(
                agent_behavior=types.AgentBehavior.AUTONOMOUS,
                enabled_tools=[types.BuiltinTools.VIEW_FILE],
            ),
        ),
        types.SubagentConfig(
            name="ux_self_improver",
            description="Designs small-perimeter onboarding & reliability slices with >=95% pytest coverage.",
            system_instructions="Propose minimal, high-impact onboarding improvements for personal home PCs only.",
            capabilities=types.SubagentCapabilities(
                agent_behavior=types.AgentBehavior.AUTONOMOUS,
                enabled_tools=[types.BuiltinTools.VIEW_FILE, types.BuiltinTools.SEARCH_DIR],
            ),
        ),
    ]

    safety_policies = [
        policy.deny(
            "run_command",
            when=is_unsafe_shell_command,
            name="block_destructive_or_live_coder_commands",
            reason="Destructive git/shell operations and live hivemind/coder inference are forbidden.",
        ),
        policy.confirm_run_command(),
    ]

    triggers_list: list[Any] = []
    if periodic_interval_s is not None and periodic_interval_s > 0:

        async def _periodic_control_tick(ctx: TriggerContext) -> None:
            await ctx.send(
                "Run a scheduled HiveMind control check: inspect operator account, collect adoption & campaign "
                "telemetry, evaluate readiness gates, inspect the backlog, and record the next self-improvement action."
            )

        triggers_list.append(every(periodic_interval_s, _periodic_control_tick))

    conv_id = load_saved_conversation_id(state_dir) if resume_session else None

    cfg_kwargs: dict[str, Any] = {
        "system_instructions": SYSTEM_INSTRUCTIONS,
        "tools": tools,
        "subagents": subagents,
        "policies": safety_policies,
        "triggers": triggers_list or None,
        "workspaces": [str(workspace_dir.resolve())],
        "save_dir": str(save_dir),
        "app_data_dir": str(app_data_dir),
        "conversation_id": conv_id,
        "response_schema": response_schema,
        "capabilities": CapabilitiesConfig(
            enable_subagents=True,
            agent_behavior=types.AgentBehavior.AUTONOMOUS,
            max_subagent_depth=2,
            allowed_subagents=["adoption_analyst", "campaign_tracker", "ux_self_improver"],
        ),
        "budget_config": types.BudgetConfig(
            max_model_calls=25,
            max_tool_calls=40,
            max_total_tokens=250_000,
        ),
        "compaction_config": CompactionConfig(token_threshold=80_000),
    }
    if vertex:
        cfg_kwargs["vertex"] = True
        if project:
            cfg_kwargs["project"] = project
        cfg_kwargs["location"] = location
    if model:
        cfg_kwargs["model"] = model

    return LocalAgentConfig(**cfg_kwargs)


def format_report_markdown(report: dict[str, Any]) -> str:
    """Render a ``ControlCycleReport`` dictionary into concise GitHub-flavored Markdown."""
    lines = [
        f"# HiveMind Antigravity Control Report ({report.get('generated_utc', 'n/a')})",
        "",
        f"- **Operator Account**: `{report.get('operator_handle', 'n/a')}`",
        f"- **Campaign Stage**: `{report.get('adoption_stage', 'n/a')}` (`J-{report.get('days_until_noel_launch', 'n/a')}` before 19 Dec 2026 launch)",
        f"- **Adoption Verdict**: {report.get('external_adoption_verdict', 'n/a')}",
        f"- **Top Friction Bottleneck**: {report.get('top_friction_bottleneck', 'n/a')}",
        "",
        "## Key Measured Metrics",
    ]
    for k, v in (report.get("key_metrics") or {}).items():
        lines.append(f"- **{k}**: `{v}`")
    lines.append("")
    lines.append("## Active Campaigns")
    for camp in report.get("campaigns") or []:
        if isinstance(camp, dict):
            icon = "OK" if camp.get("healthy") else "WARN"
            blk = f" — Blocker: {camp.get('blocker')}" if camp.get("blocker") else ""
            lines.append(
                f"- **[{icon}] `{camp.get('campaign_id')}`** ({camp.get('phase')}): {camp.get('headline_metric')}{blk}"
            )
    lines.append("")
    lines.append("## Prioritized Self-Improvement Actions")
    for act in report.get("next_actions") or []:
        if isinstance(act, dict):
            lines.append(
                f"- **[{act.get('priority')}] `{act.get('slice_id')}` — {act.get('title')}** "
                f"(Target: *{act.get('target_metric')}*, Gate: *{act.get('test_gate')}*)"
            )
    return "\n".join(lines) + "\n"


async def run_sdk_audit_cycle(
    *,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    workspace_dir: Path | None = None,
    resume_session: bool = True,
    vertex: bool = False,
    project: str | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Execute one full Google Antigravity SDK control & self-improvement audit turn."""
    from google.antigravity import Agent

    sdir = (state_dir or default_state_dir()).resolve()
    wdir = (workspace_dir or Path.cwd()).resolve()
    tools = build_control_tools(
        gateway=gateway,
        operator_handle=operator_handle,
        github_repo=github_repo,
        state_dir=sdir,
        backlog_path=backlog_path,
    )
    config = build_antigravity_agent_config(
        tools=tools,
        state_dir=sdir,
        workspace_dir=wdir,
        response_schema=ControlCycleReport,
        resume_session=resume_session,
        vertex=vertex,
        project=project,
        model=model,
    )

    prompt = (
        "Run a complete HiveMind Control & Adoption Audit cycle now:\n"
        "1. Call `check_operator_account` to confirm Xavier's HiveMind operator account.\n"
        "2. Call `collect_adoption_and_campaign_snapshot` and `evaluate_campaign_readiness_gates`.\n"
        "3. Call `inspect_backlog_and_pr_velocity`.\n"
        "4. Call `record_self_improvement_action` for the single highest-leverage next UX/adoption improvement.\n"
        "5. Return the structured ControlCycleReport grounded strictly in the measured tool outputs."
    )

    async with Agent(config=config) as agent:
        response = await agent.chat(prompt)
        structured = await response.structured_output()
        if structured is None:
            raw_text = await response.text()
            structured = json.loads(raw_text)
        save_conversation_id(sdir, getattr(agent, "conversation_id", None))

    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "latest_report.json").write_text(json.dumps(structured, indent=2) + "\n", encoding="utf-8")
    (sdir / "latest_report.md").write_text(format_report_markdown(structured), encoding="utf-8")
    return structured


async def run_sdk_daemon(
    *,
    interval_seconds: float = 10800.0,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    workspace_dir: Path | None = None,
    vertex: bool = False,
    project: str | None = None,
    model: str | None = None,
) -> None:  # pragma: no cover - long-running daemon entrypoint
    """Run the persistent Antigravity SDK Control Agent with periodic ``every(...)`` triggers."""
    from google.antigravity import Agent

    sdir = (state_dir or default_state_dir()).resolve()
    wdir = (workspace_dir or Path.cwd()).resolve()
    tools = build_control_tools(
        gateway=gateway,
        operator_handle=operator_handle,
        github_repo=github_repo,
        state_dir=sdir,
        backlog_path=backlog_path,
    )
    config = build_antigravity_agent_config(
        tools=tools,
        state_dir=sdir,
        workspace_dir=wdir,
        response_schema=None,
        periodic_interval_s=interval_seconds,
        vertex=vertex,
        project=project,
        model=model,
    )
    async with Agent(config=config) as agent:
        save_conversation_id(sdir, getattr(agent, "conversation_id", None))
        await asyncio.Event().wait()


async def run_sdk_interactive_chat(
    *,
    gateway: str = DEFAULT_GATEWAY,
    operator_handle: str = DEFAULT_OPERATOR_HANDLE,
    github_repo: str = DEFAULT_GITHUB_REPO,
    state_dir: Path | None = None,
    backlog_path: Path | None = None,
    workspace_dir: Path | None = None,
    vertex: bool = False,
    project: str | None = None,
    model: str | None = None,
) -> None:  # pragma: no cover - interactive TTY loop
    """Launch an interactive Antigravity SDK terminal session with all HiveMind control tools."""
    from google.antigravity import Agent
    from google.antigravity.utils.interactive import run_interactive_loop

    sdir = (state_dir or default_state_dir()).resolve()
    wdir = (workspace_dir or Path.cwd()).resolve()
    tools = build_control_tools(
        gateway=gateway,
        operator_handle=operator_handle,
        github_repo=github_repo,
        state_dir=sdir,
        backlog_path=backlog_path,
    )
    config = build_antigravity_agent_config(
        tools=tools,
        state_dir=sdir,
        workspace_dir=wdir,
        response_schema=None,
        vertex=vertex,
        project=project,
        model=model,
    )
    async with Agent(config=config) as agent:
        save_conversation_id(sdir, getattr(agent, "conversation_id", None))
        await run_interactive_loop(agent)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="HiveMind Autonomous Control Agent (Google Antigravity SDK)."
    )
    parser.add_argument(
        "--mode",
        choices=("status", "audit", "daemon", "chat"),
        default="status",
        help="Execution mode: deterministic status check, SDK audit cycle, periodic SDK daemon, or interactive chat.",
    )
    parser.add_argument("--gateway", default=DEFAULT_GATEWAY, help="HiveMind Cloud Run gateway URL.")
    parser.add_argument("--handle", default=DEFAULT_OPERATOR_HANDLE, help="Operator HiveMind handle.")
    parser.add_argument("--github-repo", default=DEFAULT_GITHUB_REPO, help="GitHub owner/repo to track.")
    parser.add_argument("--state-dir", type=Path, default=None, help="Directory for persistent agent state.")
    parser.add_argument("--backlog", type=Path, default=None, help="Optional JSON improvement backlog file.")
    parser.add_argument("--interval-seconds", type=float, default=10800.0, help="Daemon trigger interval (default 3h).")
    parser.add_argument("--fresh-session", action="store_true", help="Start a new Antigravity conversation_id.")
    parser.add_argument("--vertex", action="store_true", help="Authenticate via Vertex AI ADC.")
    parser.add_argument("--project", default=None, help="GCP project ID when --vertex is enabled.")
    parser.add_argument("--model", default=None, help="Optional explicit model identifier override.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sdir = (args.state_dir or default_state_dir()).resolve()

    if args.mode == "status":
        acct = ensure_operator_account(gateway=args.gateway, preferred_handle=args.handle)
        snap = collect_platform_snapshot(gateway=args.gateway, github_repo=args.github_repo, state_dir=sdir)
        gates = evaluate_readiness_gates(snap)
        print(json.dumps({"account": acct, "snapshot": snap, "readiness": gates}, indent=2, sort_keys=True))
        return 0 if acct.get("authenticated") and snap.get("gateway_healthy") else 1

    try:
        import google.antigravity  # noqa: F401
    except ImportError:
        print(
            "Error: google-antigravity SDK is not installed in this Python environment. "
            "Install it with: uv pip install google-antigravity",
            file=sys.stderr,
        )
        return 2

    if args.mode == "audit":
        report = asyncio.run(
            run_sdk_audit_cycle(
                gateway=args.gateway,
                operator_handle=args.handle,
                github_repo=args.github_repo,
                state_dir=sdir,
                backlog_path=args.backlog,
                resume_session=not args.fresh_session,
                vertex=args.vertex,
                project=args.project,
                model=args.model,
            )
        )
        print(format_report_markdown(report))
        return 0

    if args.mode == "daemon":  # pragma: no cover
        asyncio.run(
            run_sdk_daemon(
                interval_seconds=args.interval_seconds,
                gateway=args.gateway,
                operator_handle=args.handle,
                github_repo=args.github_repo,
                state_dir=sdir,
                backlog_path=args.backlog,
                vertex=args.vertex,
                project=args.project,
                model=args.model,
            )
        )
        return 0

    asyncio.run(  # pragma: no cover
        run_sdk_interactive_chat(
            gateway=args.gateway,
            operator_handle=args.handle,
            github_repo=args.github_repo,
            state_dir=sdir,
            backlog_path=args.backlog,
            vertex=args.vertex,
            project=args.project,
            model=args.model,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
