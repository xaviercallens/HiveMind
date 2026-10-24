"""Deterministic Owner-Order Identification, Filtering, and Authorization Guardrail.

Enforces the strict governance invariant that administrative, operational, architectural,
release, merge, roadmap, and agent-execution orders are accepted **exclusively** from the
project creator and sole maintainer:

    ``callensxavier@gmail.com``

Deterministic Pipeline Stages
-----------------------------
1. **Sender Identity Normalization & Anti-Spoofing (`normalize_and_verify_owner_email`)**:
   - Rejects missing or non-string email identities (`OWN-01`).
   - Rejects RFC-5322 display-name / angle-bracket spoofing such as
     ``"callensxavier@gmail.com <attacker@evil.com>"``, multiple ``@`` signs, quotes,
     whitespace, or header control characters (`OWN-02`).
   - Rejects non-ASCII characters, Unicode homoglyphs (Cyrillic/Greek confusables),
     zero-width characters, and bidirectional control characters (`OWN-03`).
   - Rejects plus-addressing (``callensxavier+foo@gmail.com``), dot-variant aliases,
     and domain/subdomain spoofing (`OWN-04`), requiring an exact canonical match to
     ``callensxavier@gmail.com``.
2. **Request Intent Classification (`classify_request_intent`)**:
   - Deterministically inspects request text, command fields, and modified file paths
     to classify requests as ``DIRECTIVE_ORDER`` vs ``STANDARD_CONTRIBUTION``.
   - Detects administrative/operational imperatives (merge, release, deploy, push,
     override, configure, execute, close, delete, modify guardrail/roadmap/charter)
     and adversarial prompt-injection orders (``ignore previous instructions``,
     ``system override``, ``you must``, ``act as``, ``bypass security``).
3. **Order Filtering & Cryptographic Audit Receipt (`OwnerOrderGuardrail`)**:
   - Accepts orders **only** when ``is_authorized_owner is True``
     (``normalized_email == "callensxavier@gmail.com"``).
   - Emits an immutable, SHA-256-attested ``OwnerOrderDecision`` for every evaluated
     request and filters batches so zero unauthorized orders ever reach execution.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

from hivemind.security._exceptions import SwarmSecurityViolationError

AUTHORIZED_OWNER_EMAIL: Final[str] = "callensxavier@gmail.com"
AUTHORIZED_OWNER_LOCAL_PART: Final[str] = "callensxavier"
AUTHORIZED_OWNER_DOMAIN: Final[str] = "gmail.com"
AUTHORIZED_OWNER_GITHUB_LOGIN: Final[str] = "xaviercallens"

# Zero-width, format, and bidirectional control characters prohibited in identity strings
_FORBIDDEN_UNICODE_CODEPOINTS: Final[frozenset[int]] = frozenset(
    {
        0x200B,  # ZERO WIDTH SPACE
        0x200C,  # ZERO WIDTH NON-JOINER
        0x200D,  # ZERO WIDTH JOINER
        0x2060,  # WORD JOINER
        0xFEFF,  # ZERO WIDTH NO-BREAK SPACE / BOM
        0x202A,  # LRE
        0x202B,  # RLE
        0x202C,  # PDF
        0x202D,  # LRO
        0x202E,  # RLO
        0x2066,  # LRI
        0x2067,  # RLI
        0x2068,  # FSI
        0x2069,  # PDI
    }
)

# Strict ASCII email token: only lowercase ASCII letters, digits, and single '.' in domain
_STRICT_ASCII_EMAIL_RE: Final[re.Pattern[str]] = re.compile(
    r"^[a-z0-9_.-]+@[a-z0-9.-]+\.[a-z]{2,}$"
)

# RFC-5322 / header spoofing indicators: angle brackets, quotes, commas, semicolons, newlines, spaces
_HEADER_SPOOF_RE: Final[re.Pattern[str]] = re.compile(r"[<>\"',;\r\n\t\s()]")

# Deterministic patterns that identify an incoming message as a DIRECTIVE_ORDER
_DIRECTIVE_ORDER_PATTERNS: Final[tuple[tuple[str, re.Pattern[str]], ...]] = (
    (
        "ORDER_RELEASE_OR_MERGE",
        re.compile(
            r"\b(?:commit\s+and\s+push|merge\s+(?:this|pr|pull\s+request|branch|now)|"
            r"create\s+(?:a\s+)?release|publish\s+release|tag\s+and\s+release|"
            r"push\s+to\s+(?:origin|main|master)|deploy\s+to\s+production|"
            r"force\s+merge|auto[- ]?merge\s+now)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ORDER_GUARDRAIL_OR_SECURITY_CHANGE",
        re.compile(
            r"\b(?:add\s+(?:a\s+)?(?:new\s+)?guardrail|disable\s+guardrail|"
            r"bypass\s+(?:security|scanner|guardrail|gate)|override\s+(?:security|gate|scanner)|"
            r"accept\s+only\s+(?:the\s+)?order\s+from|whitelist\s+(?:user|author|email)|"
            r"add\s+trusted\s+maintainer|change\s+owner\s+email|modify\s+security\s+policy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ORDER_AGENT_ORCHESTRATION_DIRECTIVE",
        re.compile(
            r"\b(?:act\s+as\s+(?:orchestrator|arm|brain|admin)|"
            r"run\s+(?:the\s+)?orchestrator|execute\s+command|"
            r"schedule\s+(?:a\s+)?cron|every\s+\d+\s+hours?\s+run|"
            r"reorganize\s+(?:the\s+)?roadmap|update\s+governance|"
            r"execute\s+order|follow\s+my\s+order|obey\s+this\s+instruction)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "ORDER_PROMPT_INJECTION_OVERRIDE",
        re.compile(
            r"\b(?:ignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|orders|rules)|"
            r"system\s+override|developer\s+mode\s+enabled|sudo\s+merge|"
            r"you\s+must\s+(?:immediately\s+)?(?:merge|approve|execute|run|accept)|"
            r"new\s+owner\s+order|urgent\s+admin\s+order)\b",
            re.IGNORECASE,
        ),
    ),
)

# Protected repository paths whose modification always constitutes an administrative order
_PROTECTED_ORDER_PATHS: Final[frozenset[str]] = frozenset(
    {
        "SECURITY.md",
        "CODEOWNERS",
        "LEGAL_DISCLAIMER_AND_COMPUTE_RECIPROCITY.md",
        "ROADMAP.md",
        "pyproject.toml",
        "src/hivemind/security/owner_order_guardrail.py",
        "src/hivemind/security/pr_security_scanner.py",
        "src/hivemind/cli/community_orchestrator.py",
    }
)


class RequestIntentCategory(str, Enum):
    """Deterministic classification of an incoming request's operational intent."""

    DIRECTIVE_ORDER = "DIRECTIVE_ORDER"
    STANDARD_CONTRIBUTION = "STANDARD_CONTRIBUTION"


@dataclass(frozen=True)
class EmailIdentityVerification:
    """Deterministic verification outcome for a candidate sender email address."""

    raw_email: str
    normalized_email: str
    is_authorized_owner: bool
    rule_id: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestIntentClassification:
    """Deterministic classification outcome for an incoming request payload."""

    intent_category: RequestIntentCategory
    is_directive_order: bool
    matched_triggers: tuple[str, ...]
    protected_paths_touched: tuple[str, ...]
    payload_sha256: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["intent_category"] = self.intent_category.value
        data["matched_triggers"] = list(self.matched_triggers)
        data["protected_paths_touched"] = list(self.protected_paths_touched)
        return data


@dataclass(frozen=True)
class OwnerOrderRequest:
    """Canonical representation of an incoming order or request to evaluate."""

    request_id: str
    sender_email: str
    content: str
    sender_login: str = ""
    source_channel: str = "direct_order"
    changed_files: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OwnerOrderDecision:
    """Deterministic authorization decision and SHA-256 audit receipt for a request."""

    request_id: str
    accepted: bool
    status: str
    rule_id: str
    reason: str
    authorized_owner_email: str
    raw_sender_email: str
    normalized_sender_email: str
    sender_login: str
    is_authorized_owner: bool
    intent_category: str
    is_directive_order: bool
    matched_triggers: tuple[str, ...]
    protected_paths_touched: tuple[str, ...]
    source_channel: str
    payload_sha256: str
    receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["matched_triggers"] = list(self.matched_triggers)
        data["protected_paths_touched"] = list(self.protected_paths_touched)
        return data


@dataclass(frozen=True)
class OwnerOrderFilterBatchResult:
    """Batch filtering outcome separating accepted owner orders from rejected requests."""

    authorized_owner_email: str
    total_evaluated: int
    accepted_count: int
    rejected_count: int
    accepted_requests: tuple[OwnerOrderRequest, ...]
    decisions: tuple[OwnerOrderDecision, ...]
    batch_receipt_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_owner_email": self.authorized_owner_email,
            "total_evaluated": self.total_evaluated,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "accepted_request_ids": [r.request_id for r in self.accepted_requests],
            "decisions": [d.to_dict() for d in self.decisions],
            "batch_receipt_sha256": self.batch_receipt_sha256,
        }


def normalize_and_verify_owner_email(raw_email: str | None) -> EmailIdentityVerification:
    """Deterministically normalize and verify whether `raw_email` is `callensxavier@gmail.com`.

    Rejects:
    - `None` or empty strings (`OWN-01`)
    - Non-ASCII / Unicode homoglyphs / zero-width / bidi characters (`OWN-03`)
    - RFC-5322 display names, angle brackets `<...>`, quotes, spaces, or multiple `@` (`OWN-02`)
    - Plus-addressing (`callensxavier+...@gmail.com`), dot-variants, or any address other
      than `callensxavier@gmail.com` (`OWN-04`)
    """
    if not isinstance(raw_email, str) or not raw_email.strip():
        return EmailIdentityVerification(
            raw_email="" if raw_email is None else str(raw_email),
            normalized_email="",
            is_authorized_owner=False,
            rule_id="OWN-01",
            reason="Missing or empty sender email identity.",
        )

    # Check for forbidden Unicode control/zero-width/bidi characters or non-ASCII homoglyphs
    for ch in raw_email:
        cp = ord(ch)
        if cp in _FORBIDDEN_UNICODE_CODEPOINTS or unicodedata.category(ch) in {"Cf", "Cc"}:
            if ch not in {"\t", "\n", "\r"}:
                return EmailIdentityVerification(
                    raw_email=raw_email,
                    normalized_email="",
                    is_authorized_owner=False,
                    rule_id="OWN-03",
                    reason="Forbidden Unicode control, zero-width, or bidirectional character in email.",
                )
        if cp > 0x7F:
            return EmailIdentityVerification(
                raw_email=raw_email,
                normalized_email="",
                is_authorized_owner=False,
                rule_id="OWN-03",
                reason="Non-ASCII character or Unicode homoglyph detected in sender email.",
            )

    candidate = raw_email.strip()

    # Reject any RFC-5322 display-name wrapper, angle brackets, quotes, whitespace, or multiple '@'
    if _HEADER_SPOOF_RE.search(candidate) or candidate.count("@") != 1:
        return EmailIdentityVerification(
            raw_email=raw_email,
            normalized_email=candidate.lower(),
            is_authorized_owner=False,
            rule_id="OWN-02",
            reason=(
                "Malformed email header or RFC-5322 display-name spoofing detected; "
                "only a bare canonical address is permitted."
            ),
        )

    normalized = candidate.lower()
    if "+" in normalized or not _STRICT_ASCII_EMAIL_RE.match(normalized):
        return EmailIdentityVerification(
            raw_email=raw_email,
            normalized_email=normalized,
            is_authorized_owner=False,
            rule_id="OWN-04",
            reason=(
                f"Sender email '{normalized}' is not authorized (plus-addressing and non-standard "
                f"aliases are forbidden; required '{AUTHORIZED_OWNER_EMAIL}')."
            ),
        )

    local_part, domain_part = normalized.split("@", 1)
    if (
        normalized != AUTHORIZED_OWNER_EMAIL
        or local_part != AUTHORIZED_OWNER_LOCAL_PART
        or domain_part != AUTHORIZED_OWNER_DOMAIN
    ):
        return EmailIdentityVerification(
            raw_email=raw_email,
            normalized_email=normalized,
            is_authorized_owner=False,
            rule_id="OWN-04",
            reason=(
                f"Unauthorized sender email '{normalized}'; orders are accepted exclusively "
                f"from '{AUTHORIZED_OWNER_EMAIL}'."
            ),
        )

    return EmailIdentityVerification(
        raw_email=raw_email,
        normalized_email=AUTHORIZED_OWNER_EMAIL,
        is_authorized_owner=True,
        rule_id="OWN-00",
        reason=f"Verified sole authorized owner identity '{AUTHORIZED_OWNER_EMAIL}'.",
    )


def classify_request_intent(
    content: str,
    changed_files: Sequence[str] = (),
    source_channel: str = "direct_order",
) -> RequestIntentClassification:
    """Deterministically classify a request payload into `DIRECTIVE_ORDER` or `STANDARD_CONTRIBUTION`."""
    safe_content = content or ""
    matched_triggers: list[str] = []

    # Direct order intake channels are always treated as directive orders
    if source_channel in {"direct_order", "cli_order", "admin_order", "agent_directive"}:
        matched_triggers.append(f"CHANNEL:{source_channel}")

    for label, pattern in _DIRECTIVE_ORDER_PATTERNS:
        if pattern.search(safe_content):
            matched_triggers.append(label)

    protected_touched: list[str] = []
    for raw_path in changed_files:
        norm_path = str(raw_path).replace("\\", "/").strip("/")
        if (
            norm_path in _PROTECTED_ORDER_PATHS
            or norm_path.startswith(".github/workflows/")
            or norm_path.startswith("src/hivemind/security/")
        ):
            protected_touched.append(norm_path)

    if protected_touched:
        matched_triggers.append("PROTECTED_GOVERNANCE_OR_SECURITY_PATH")

    is_order = len(matched_triggers) > 0
    category = (
        RequestIntentCategory.DIRECTIVE_ORDER
        if is_order
        else RequestIntentCategory.STANDARD_CONTRIBUTION
    )
    payload_canon = json.dumps(
        {
            "content": safe_content,
            "changed_files": list(changed_files),
            "source_channel": source_channel,
        },
        sort_keys=True,
    )
    payload_sha256 = hashlib.sha256(payload_canon.encode("utf-8")).hexdigest()

    return RequestIntentClassification(
        intent_category=category,
        is_directive_order=is_order,
        matched_triggers=tuple(matched_triggers),
        protected_paths_touched=tuple(protected_touched),
        payload_sha256=payload_sha256,
    )


class OwnerOrderGuardrail:
    """Deterministic Guardrail that identifies requests and accepts orders ONLY from `callensxavier@gmail.com`."""

    def __init__(self, strict_orders_only: bool = True) -> None:
        """Initialize the OwnerOrderGuardrail.

        Args:
            strict_orders_only: When `True` (default), every request evaluated through
                this guardrail is treated as an order intake and requires
                `sender_email == "callensxavier@gmail.com"`. When `False` (used for
                public PR/Discussion triage), non-directive `STANDARD_CONTRIBUTION`
                items from the community are allowed to proceed to technical review,
                while any `DIRECTIVE_ORDER` not from `callensxavier@gmail.com` is
                deterministically blocked.
        """
        self.authorized_owner_email: str = AUTHORIZED_OWNER_EMAIL
        self.authorized_owner_github_login: str = AUTHORIZED_OWNER_GITHUB_LOGIN
        self.strict_orders_only: bool = bool(strict_orders_only)

    @staticmethod
    def coerce_request(
        item: OwnerOrderRequest | Mapping[str, Any],
        default_index: int = 0,
    ) -> OwnerOrderRequest:
        """Convert a mapping or `OwnerOrderRequest` into a canonical `OwnerOrderRequest`."""
        if isinstance(item, OwnerOrderRequest):
            return item
        req_id = str(item.get("request_id") or item.get("id") or f"req-{default_index}")
        sender_email = str(item.get("sender_email") or item.get("author_email") or item.get("email") or "")
        sender_login = str(item.get("sender_login") or item.get("author") or item.get("login") or "")
        title = str(item.get("title") or "")
        body = str(item.get("content") or item.get("body") or item.get("command") or "")
        combined_content = f"{title}\n{body}".strip() if title else body
        source_channel = str(item.get("source_channel") or "direct_order")
        raw_files = item.get("changed_files") or item.get("files") or ()
        file_names: list[str] = []
        for f in raw_files:
            if isinstance(f, Mapping):
                file_names.append(str(f.get("filename") or f.get("path") or ""))
            else:
                file_names.append(str(f))
        return OwnerOrderRequest(
            request_id=req_id,
            sender_email=sender_email,
            content=combined_content,
            sender_login=sender_login,
            source_channel=source_channel,
            changed_files=tuple(fn for fn in file_names if fn),
            metadata=dict(item.get("metadata") or {}),
        )

    def evaluate_request(
        self,
        request: OwnerOrderRequest | Mapping[str, Any],
    ) -> OwnerOrderDecision:
        """Deterministically identify request intent, verify sender email, and decide acceptance."""
        req = self.coerce_request(request)
        email_check = normalize_and_verify_owner_email(req.sender_email)
        intent_check = classify_request_intent(
            content=req.content,
            changed_files=req.changed_files,
            source_channel=req.source_channel,
        )

        # Secondary spoofing check: if someone claims the owner's GitHub login (`xaviercallens`)
        # while supplying a non-owner email address, flag explicit identity spoofing (`OWN-06`).
        login_clean = req.sender_login.strip().lower()
        if (
            login_clean == self.authorized_owner_github_login
            and req.sender_email.strip()
            and not email_check.is_authorized_owner
        ):
            accepted = False
            status = "REJECTED_OWNER_LOGIN_EMAIL_MISMATCH"
            rule_id = "OWN-06"
            reason = (
                f"Sender claims GitHub login '@{req.sender_login}' but provided non-owner "
                f"email '{req.sender_email}' (expected '{self.authorized_owner_email}')."
            )
        elif email_check.is_authorized_owner:
            accepted = True
            status = "ACCEPTED_AUTHORIZED_OWNER_ORDER"
            rule_id = "OWN-00"
            reason = (
                f"Order accepted: verified sender '{self.authorized_owner_email}' is the sole "
                "authorized project owner."
            )
        elif self.strict_orders_only or intent_check.is_directive_order:
            accepted = False
            status = "REJECTED_UNAUTHORIZED_ORDER_SENDER"
            rule_id = email_check.rule_id if email_check.rule_id != "OWN-00" else "OWN-05"
            reason = (
                f"Order rejected ({rule_id}): only orders from '{self.authorized_owner_email}' "
                f"are accepted. ({email_check.reason})"
            )
        else:
            accepted = True
            status = "ALLOWED_NON_DIRECTIVE_CONTRIBUTION"
            rule_id = "OWN-00-CONTRIB"
            reason = (
                "Non-directive community contribution contains no administrative orders or "
                "protected path modifications."
            )

        receipt_payload = {
            "request_id": req.request_id,
            "accepted": accepted,
            "status": status,
            "rule_id": rule_id,
            "authorized_owner_email": self.authorized_owner_email,
            "raw_sender_email": req.sender_email,
            "normalized_sender_email": email_check.normalized_email,
            "sender_login": req.sender_login,
            "is_authorized_owner": email_check.is_authorized_owner,
            "intent_category": intent_check.intent_category.value,
            "is_directive_order": intent_check.is_directive_order,
            "matched_triggers": list(intent_check.matched_triggers),
            "protected_paths_touched": list(intent_check.protected_paths_touched),
            "source_channel": req.source_channel,
            "payload_sha256": intent_check.payload_sha256,
        }
        receipt_sha256 = hashlib.sha256(
            json.dumps(receipt_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return OwnerOrderDecision(
            request_id=req.request_id,
            accepted=accepted,
            status=status,
            rule_id=rule_id,
            reason=reason,
            authorized_owner_email=self.authorized_owner_email,
            raw_sender_email=req.sender_email,
            normalized_sender_email=email_check.normalized_email,
            sender_login=req.sender_login,
            is_authorized_owner=email_check.is_authorized_owner,
            intent_category=intent_check.intent_category.value,
            is_directive_order=intent_check.is_directive_order,
            matched_triggers=intent_check.matched_triggers,
            protected_paths_touched=intent_check.protected_paths_touched,
            source_channel=req.source_channel,
            payload_sha256=intent_check.payload_sha256,
            receipt_sha256=receipt_sha256,
        )

    def assert_authorized_order(
        self,
        request: OwnerOrderRequest | Mapping[str, Any],
    ) -> OwnerOrderDecision:
        """Evaluate `request` and raise `SwarmSecurityViolationError` unless it is an authorized order from `callensxavier@gmail.com`."""
        decision = self.evaluate_request(request)
        if not decision.accepted or not decision.is_authorized_owner:
            raise SwarmSecurityViolationError(
                rule_id=decision.rule_id if decision.rule_id != "OWN-00" else "OWN-04",
                category="UNAUTHORIZED_OWNER_ORDER",
                message=(
                    f"{decision.reason} "
                    f"(request_id={decision.request_id}, sender_email={decision.raw_sender_email!r})"
                ),
            )
        return decision

    def filter_requests(
        self,
        requests: Sequence[OwnerOrderRequest | Mapping[str, Any]],
    ) -> OwnerOrderFilterBatchResult:
        """Deterministically filter a sequence of requests, keeping ONLY authorized requests."""
        coerced = [self.coerce_request(item, default_index=i) for i, item in enumerate(requests)]
        decisions: list[OwnerOrderDecision] = []
        accepted_reqs: list[OwnerOrderRequest] = []

        for req in coerced:
            dec = self.evaluate_request(req)
            decisions.append(dec)
            if dec.accepted and (not self.strict_orders_only or dec.is_authorized_owner):
                accepted_reqs.append(req)

        batch_canon = json.dumps(
            [d.receipt_sha256 for d in decisions],
            sort_keys=True,
        )
        batch_sha256 = hashlib.sha256(batch_canon.encode("utf-8")).hexdigest()

        return OwnerOrderFilterBatchResult(
            authorized_owner_email=self.authorized_owner_email,
            total_evaluated=len(coerced),
            accepted_count=len(accepted_reqs),
            rejected_count=len(coerced) - len(accepted_reqs),
            accepted_requests=tuple(accepted_reqs),
            decisions=tuple(decisions),
            batch_receipt_sha256=batch_sha256,
        )


def evaluate_owner_order_request(
    sender_email: str,
    content: str,
    *,
    request_id: str = "order-0",
    sender_login: str = "",
    source_channel: str = "direct_order",
    changed_files: Sequence[str] = (),
    strict_orders_only: bool = True,
) -> OwnerOrderDecision:
    """Convenience function to evaluate a single order against the `callensxavier@gmail.com` guardrail."""
    guardrail = OwnerOrderGuardrail(strict_orders_only=strict_orders_only)
    return guardrail.evaluate_request(
        OwnerOrderRequest(
            request_id=request_id,
            sender_email=sender_email,
            content=content,
            sender_login=sender_login,
            source_channel=source_channel,
            changed_files=tuple(changed_files),
        )
    )


def filter_authorized_owner_orders(
    requests: Sequence[OwnerOrderRequest | Mapping[str, Any]],
) -> OwnerOrderFilterBatchResult:
    """Deterministically filter a batch of orders, accepting ONLY those from `callensxavier@gmail.com`."""
    guardrail = OwnerOrderGuardrail(strict_orders_only=True)
    return guardrail.filter_requests(requests)


def verify_local_git_owner_identity(repo_root: Path | None = None) -> dict[str, Any]:
    """Verify that the local Git configuration and HEAD commit belong to `callensxavier@gmail.com`."""
    cwd = str(repo_root) if repo_root is not None else None
    cfg_res = subprocess.run(
        ["git", "config", "user.email"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    head_res = subprocess.run(
        ["git", "log", "-3", "--format=%ae|%ce"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    git_config_email = cfg_res.stdout.strip() if cfg_res.returncode == 0 else ""
    author_email = ""
    committer_email = ""
    recent_author_verified = False
    if head_res.returncode == 0 and "|" in head_res.stdout:
        lines = [ln.strip() for ln in head_res.stdout.splitlines() if "|" in ln]
        if lines:
            parts = lines[0].split("|", 1)
            author_email = parts[0].strip()
            committer_email = parts[1].strip()
        for ln in lines:
            a_em, c_em = ln.split("|", 1)
            if normalize_and_verify_owner_email(a_em.strip()).is_authorized_owner:
                recent_author_verified = True
                if not normalize_and_verify_owner_email(author_email).is_authorized_owner:
                    author_email = a_em.strip()
                    committer_email = c_em.strip()
                break

    cfg_verify = normalize_and_verify_owner_email(git_config_email)
    author_verify = normalize_and_verify_owner_email(author_email)
    committer_verify = normalize_and_verify_owner_email(committer_email)

    verified = bool(
        cfg_verify.is_authorized_owner
        or author_verify.is_authorized_owner
        or recent_author_verified
    )
    payload = {
        "authorized_owner_email": AUTHORIZED_OWNER_EMAIL,
        "git_config_email": git_config_email,
        "head_author_email": author_email,
        "head_committer_email": committer_email,
        "git_config_verified": cfg_verify.is_authorized_owner,
        "head_author_verified": author_verify.is_authorized_owner,
        "head_committer_verified": committer_verify.is_authorized_owner,
        "owner_identity_verified": verified,
    }
    payload["attestation_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload
