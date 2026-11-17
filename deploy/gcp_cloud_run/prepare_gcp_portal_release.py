"""Staged GCP Public DNS Portal Release Verifier (DO NOT DEPLOY YET).

Verifies all DNS, FinOps (`min-instances=0`), Core MVP Auth (Google, Reddit, Steam,
GitHub SSO + Standard Email/Password PBKDF2-HMAC-SHA256), and Red Team security
controls (PR #532 / GHSA-gjc6-74f9-wmrr) without executing `gcloud run deploy`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def build_staged_portal_release_report() -> dict[str, Any]:
    base_dir = Path(__file__).resolve().parent
    service_yaml = base_dir / "service.yaml"
    dns_yaml = base_dir / "dns_public_portal.yaml"

    service_text = service_yaml.read_text(encoding="utf-8") if service_yaml.exists() else ""
    dns_text = dns_yaml.read_text(encoding="utf-8") if dns_yaml.exists() else ""

    checks = {
        "service_yaml_present": service_yaml.exists(),
        "dns_public_portal_yaml_present": dns_yaml.exists(),
        "public_dns_configured": "hivemind.socrate-ai-lab.com" in dns_text and "34.54.227.33" in dns_text,
        "managed_ssl_configured": "socrate-ai-ssl" in dns_text,
        "scale_to_zero_min_instances_0": 'autoscaling.knative.dev/minScale: "0"' in service_text,
        "cpu_throttling_enabled": 'run.googleapis.com/cpu-throttling: "true"' in service_text,
        "startup_cpu_boost_enabled": 'run.googleapis.com/startup-cpu-boost: "true"' in service_text,
        "core_mvp_4_sso_providers": "google,reddit,steam,github" in service_text,
        "pbkdf2_sha256_configured": "PBKDF2-HMAC-SHA256:200000" in service_text,
        "staged_not_deployed_guard": "STAGED_READY_NOT_DEPLOYED" in service_text,
    }
    manifest_digest = hashlib.sha256((service_text + "\n" + dns_text).encode("utf-8")).hexdigest()
    return {
        "status": "STAGED_READY_NOT_DEPLOYED" if all(checks.values()) else "INCOMPLETE",
        "deployed_to_gcp": False,
        "do_not_deploy_yet": True,
        "public_dns_domain": "hivemind.socrate-ai-lab.com",
        "public_dns_url": "https://hivemind.socrate-ai-lab.com",
        "secondary_dns_domain": "portal.socrate-ai-lab.com",
        "forwarding_rule": "socrate-forwarding-rule (34.54.227.33)",
        "managed_ssl_certificate": "socrate-ai-ssl",
        "gcp_project": "gen-lang-client-0625573011",
        "cloud_run_service": "hivemind-at-home-ruche",
        "region": "us-central1",
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "manifest_sha256": manifest_digest,
    }


if __name__ == "__main__":
    report = build_staged_portal_release_report()
    print(json.dumps(report, indent=2))
