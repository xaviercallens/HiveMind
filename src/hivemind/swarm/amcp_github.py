# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""GitHub repository events -> AMCP ``io.hivemind.amcp.github.event`` (read-only).

Lets humans, coding agents and bees on the bus see what happens in the repository (issues, pull requests, reviews, releases) without polling GitHub
themselves. Deliberately small and defensive:

* **Read-only.** It reads public repository events. It never writes to GitHub.
* **No bodies.** Issue and comment text is written by anyone on the internet and would reach coding agents as if it were a message. Only the number, a sanitised
  title (<= 200 characters, no control characters), the https URL, the action and the actor's login are forwarded.
* **No backfill by default.** The first run records what already exists as seen and publishes nothing, so starting the bridge does not flood the bus.
* **Bounded.** At most ``max_per_cycle`` events per call, and unknown event types are ignored.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

EVENT_TYPE = "io.hivemind.amcp.github.event"
_LOGIN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})(?:\[bot\])?$")
_GITHUB_URL = re.compile(r"^https://github\.com/[A-Za-z0-9._/#-]{1,260}$")
_CTRL = re.compile(r"[\x00-\x1f\x7f]+")
_PR_ACTIONS = {"opened", "closed", "reopened", "ready_for_review"}
_ISSUE_ACTIONS = {"opened", "closed", "reopened"}
Fetch = Callable[[str], list[dict[str, Any]]]


def clean_title(value: Any) -> str:
    text = _CTRL.sub(" ", str(value or "")).strip()
    return text[:200]


def to_payload(event: Mapping[str, Any]) -> dict[str, Any] | None:
    """The bus payload for one GitHub event, or ``None`` if it is not a kind we forward or does not validate."""
    kind, body = event.get("type"), event.get("payload") or {}
    actor = (event.get("actor") or {}).get("login", "")
    repo = (event.get("repo") or {}).get("name", "")
    if not _LOGIN.match(str(actor)) or not re.fullmatch(r"[A-Za-z0-9._-]{1,39}/[A-Za-z0-9._-]{1,100}", str(repo)):
        return None
    if kind == "IssuesEvent" and body.get("action") in _ISSUE_ACTIONS:
        item, name, action = body.get("issue") or {}, "issue", body["action"]
    elif kind == "PullRequestEvent" and body.get("action") in _PR_ACTIONS:
        item, name = body.get("pull_request") or {}, "pull_request"
        action = "merged" if body["action"] == "closed" and item.get("merged") else body["action"]
    elif kind == "IssueCommentEvent" and body.get("action") == "created":
        item, name, action = body.get("issue") or {}, "comment", "created"
        item = {**item, "html_url": (body.get("comment") or {}).get("html_url", item.get("html_url"))}  # link to the comment, forward none of its text
    elif kind == "PullRequestReviewEvent" and body.get("action") == "submitted":
        item, name = body.get("pull_request") or {}, "review"
        action = str((body.get("review") or {}).get("state", "commented")).lower()
        item = {**item, "html_url": (body.get("review") or {}).get("html_url", item.get("html_url"))}
    elif kind == "ReleaseEvent" and body.get("action") == "published":
        release = body.get("release") or {}
        item, name, action = {"number": 0, "title": release.get("name") or release.get("tag_name"), "html_url": release.get("html_url")}, "release", "published"
    else:
        return None
    number, url = item.get("number"), str(item.get("html_url") or "")
    if isinstance(number, bool) or not isinstance(number, int) or not 0 <= number <= 10_000_000 or not _GITHUB_URL.match(url):
        return None
    payload: dict[str, Any] = {"repo": repo, "kind": name, "action": str(action)[:40], "number": number, "url": url, "actor": str(actor)}
    title = clean_title(item.get("title"))
    if title:
        payload["title"] = title
    return payload


class SeenEvents:
    """Which GitHub event ids were already handled; persisted so a restart does not repeat them."""

    def __init__(self, path: Path | None = None, keep: int = 500) -> None:
        self.path, self.keep = path, keep
        self.ids: list[str] = []
        self.first_run = True
        if path is not None and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            self.ids, self.first_run = [str(i) for i in data.get("ids", [])], False

    def add(self, event_id: str) -> None:
        if event_id not in self.ids:
            self.ids.append(event_id)
            del self.ids[: -self.keep]

    def save(self) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"ids": self.ids}), encoding="utf-8")
            self.first_run = False


def poll(fetch: Fetch, repo: str, seen: SeenEvents, publish: Callable[[dict[str, Any]], Any], *, backfill: bool = False, max_per_cycle: int = 20) -> dict[str, int]:
    """One cycle: fetch the repository's latest events, publish the new forwardable ones oldest-first, remember everything seen."""
    events = list(fetch(repo))
    fresh = [e for e in reversed(events) if str(e.get("id")) not in seen.ids]  # the API lists newest first
    published = skipped = 0
    for event in fresh:
        seen.add(str(event.get("id")))
        if seen.first_run and not backfill:
            skipped += 1
            continue
        payload = to_payload(event)
        if payload is None or published >= max_per_cycle:
            skipped += 1
            continue
        publish({"event": event, "payload": payload})
        published += 1
    seen.save()
    return {"fetched": len(events), "new": len(fresh), "published": published, "skipped": skipped}


def gh_fetch(repo: str) -> list[dict[str, Any]]:  # pragma: no cover - needs the GitHub CLI and network; exercised by the operator
    import subprocess

    out = subprocess.run(["gh", "api", f"repos/{repo}/events?per_page=30"], capture_output=True, text=True, timeout=60, check=True).stdout
    data = json.loads(out)
    return data if isinstance(data, list) else []
