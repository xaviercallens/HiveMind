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
_REPO = re.compile(r"^[A-Za-z0-9._-]{1,39}/[A-Za-z0-9._-]{1,100}$")
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
    if not isinstance(event, Mapping):
        raise TypeError("event must be a Mapping")
    kind = event.get("type")
    body = event.get("payload")
    actor_obj = event.get("actor")
    repo_obj = event.get("repo")
    if not isinstance(body, Mapping) or not isinstance(actor_obj, Mapping) or not isinstance(repo_obj, Mapping):
        return None
    actor = actor_obj.get("login", "")
    repo = repo_obj.get("name", "")
    if not isinstance(actor, str) or not isinstance(repo, str) or not _LOGIN.match(actor) or not _REPO.fullmatch(repo):
        return None
    if kind == "IssuesEvent" and body.get("action") in _ISSUE_ACTIONS:
        item, name, action = body.get("issue"), "issue", body["action"]
        if not isinstance(item, Mapping):
            return None
    elif kind == "PullRequestEvent" and body.get("action") in _PR_ACTIONS:
        item, name = body.get("pull_request"), "pull_request"
        if not isinstance(item, Mapping):
            return None
        action = "merged" if body["action"] == "closed" and item.get("merged") else body["action"]
    elif kind == "IssueCommentEvent" and body.get("action") == "created":
        raw_issue = body.get("issue")
        raw_comment = body.get("comment")
        if not isinstance(raw_issue, Mapping) or not isinstance(raw_comment, Mapping):
            return None
        name, action = "comment", "created"
        item = {**raw_issue, "html_url": raw_comment.get("html_url", raw_issue.get("html_url"))}  # link to the comment, forward none of its text
    elif kind == "PullRequestReviewEvent" and body.get("action") == "submitted":
        raw_pr = body.get("pull_request")
        raw_review = body.get("review")
        if not isinstance(raw_pr, Mapping) or not isinstance(raw_review, Mapping):
            return None
        name = "review"
        action = str(raw_review.get("state", "commented")).lower()
        item = {**raw_pr, "html_url": raw_review.get("html_url", raw_pr.get("html_url"))}
    elif kind == "ReleaseEvent" and body.get("action") == "published":
        release = body.get("release")
        if not isinstance(release, Mapping):
            return None
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
        if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
            raise ValueError("keep must be a positive integer")
        if path is not None and not isinstance(path, Path):
            path = Path(path)
        self.path, self.keep = path, keep
        self.ids: list[str] = []
        self.first_run = True
        if self.path is not None and self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(data, Mapping) or not isinstance(data.get("ids", []), list):
                raise ValueError("Corrupted SeenEvents state file")
            self.ids = [str(i) for i in data.get("ids", []) if str(i).strip()][ -self.keep :]
            self.first_run = False

    def add(self, event_id: str) -> None:
        if not isinstance(event_id, str) or not event_id.strip():
            raise ValueError("event_id must be a non-empty string")
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
    if not callable(fetch):
        raise TypeError("fetch must be callable")
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise ValueError("repo must match 'owner/name'")
    if not isinstance(seen, SeenEvents):
        raise TypeError("seen must be a SeenEvents instance")
    if not callable(publish):
        raise TypeError("publish must be callable")
    if not isinstance(backfill, bool):
        raise ValueError("backfill must be a boolean")
    if isinstance(max_per_cycle, bool) or not isinstance(max_per_cycle, int) or max_per_cycle < 1:
        raise ValueError("max_per_cycle must be a positive integer")
    events = list(fetch(repo))
    fresh: list[Mapping[str, Any]] = []
    for e in reversed(events):  # the API lists newest first
        if not isinstance(e, Mapping):
            continue
        raw_id = e.get("id")
        if raw_id is None or isinstance(raw_id, bool) or not str(raw_id).strip():
            continue
        if str(raw_id) not in seen.ids:
            fresh.append(e)
    published = skipped = 0
    for event in fresh:
        seen.add(str(event.get("id")).strip())
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
    if not isinstance(repo, str) or not _REPO.fullmatch(repo):
        raise ValueError("repo must match 'owner/name'")
    import subprocess

    out = subprocess.run(["gh", "api", f"repos/{repo}/events?per_page=30"], capture_output=True, text=True, timeout=60, check=True).stdout
    data = json.loads(out)
    return data if isinstance(data, list) else []
