#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Publish GitHub repository events (issues, pull requests, reviews, releases) on the AMCP bus, read-only.

  export HIVEMIND_KAFKA_BOOTSTRAP=127.0.0.1:9092
  python scripts/amcp_github_bridge.py --repo xaviercallens/HiveMind --once --dry-run   # show what it would publish
  python scripts/amcp_github_bridge.py --repo xaviercallens/HiveMind --dev-keyring      # keep running, every 60 s

Needs the GitHub CLI (`gh`) logged in; it only READS the repository's public events. Bodies of issues and comments are never forwarded (see
src/hivemind/swarm/amcp_github.py). A direct-tier component: it holds the signing keyring.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hivemind.swarm.amcp_collab import AGENT, check_publish, dev_keyring  # noqa: E402
from hivemind.swarm.amcp_github import EVENT_TYPE, SeenEvents, gh_fetch, poll  # noqa: E402
from hivemind.swarm.amcp_kafka_campaign_bus import build_amcp_kafka_event  # noqa: E402
from hivemind.swarm.amcp_kafka_transport import AMCPPublisher, HMACKeyring, KafkaBrokerTransport, KafkaConfigError, KafkaSettings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="xaviercallens/HiveMind")
    ap.add_argument("--state", type=Path, default=Path.home() / ".hivemind-amcp" / "github_bridge.json")
    ap.add_argument("--interval-s", type=float, default=60.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print the payloads, publish nothing, remember nothing")
    ap.add_argument("--backfill", action="store_true", help="publish the events that already exist on the first run (default: only new ones)")
    ap.add_argument("--dev-keyring", action="store_true", help="use ~/.hivemind-amcp/dev_keyring when HIVEMIND_AMCP_HMAC_KEYS is not set")
    a = ap.parse_args(argv)

    publisher: AMCPPublisher | None = None
    if not a.dry_run:
        try:
            try:
                ring = HMACKeyring.from_env()
            except KafkaConfigError:
                if not a.dev_keyring:
                    raise
                ring = dev_keyring()
            publisher = AMCPPublisher(KafkaBrokerTransport(KafkaSettings.from_env(), flush_timeout_s=20.0), ring, ttl_s=3600.0)
        except KafkaConfigError as exc:
            print(f"configuration problem: {exc}", file=sys.stderr)
            return 2

    def publish(item: dict[str, Any]) -> None:
        payload = item["payload"]
        check_publish(None, AGENT, "human_agent", EVENT_TYPE, payload, via_gateway=False)
        if publisher is None:
            print(json.dumps(payload))
            return
        event = build_amcp_kafka_event(topic_key="human_agent", event_type=EVENT_TYPE, sender_id="bridge:github", sender_kind=AGENT, recipient_id="broadcast",
                                       subject_id=f"gh-{a.repo}"[:64], correlation_id=f"gh-{payload['kind']}-{payload['number']}"[:64], payload=payload)
        publisher.publish(event)

    seen = SeenEvents(None if a.dry_run else a.state)
    while True:
        try:
            print(json.dumps(poll(gh_fetch, a.repo, seen, publish, backfill=a.backfill or a.dry_run)), flush=True)
        except Exception as exc:  # noqa: BLE001 - a network or gh failure must not kill a long-running bridge
            print(f"cycle failed: {type(exc).__name__}: {exc}"[:300], file=sys.stderr, flush=True)
            if a.once:
                return 1
        if a.once:
            return 0
        time.sleep(a.interval_s)


if __name__ == "__main__":
    raise SystemExit(main())
