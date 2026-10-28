# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Bring Your Own Bee (BYOB): submission manifest, fail-closed static validation, gated evaluation, registry entry.

An external contributor offers a bee (a LoRA adapter, a GGUF file, a full model or a serving endpoint) by publishing it on
the Hugging Face Hub and sending a signed :class:`BeeManifest`. This module decides what the hive may do with it. The
design follows three rules that the rest of HiveMind already applies to code (``hivemind_firewall``, GWAYA):

1. **Fail closed, with a reason code.** Every check returns zero or more :class:`Reason` entries whose ``code`` is a
   :class:`ReasonCode`. Any reason rejects the submission; an external service that cannot be reached gives BLOCKED, never
   a pass. The checks are deterministic functions of the manifest and of what the Hub API returns at the pinned commit.
2. **Weights are data, never code.** Only ``.safetensors`` and ``.gguf`` weight files are allowed; every pickle-bearing
   format (``.bin``, ``.pt``, ``.pth``, ``.pkl``, ``.ckpt``, ...) rejects the whole repository, including a stray
   ``training_args.bin`` in a sub-folder, because a mirror cannot promise which files a later loader touches. Any
   ``auto_map``, ``custom_code`` tag or ``.py`` file rejects too: ``trust_remote_code`` is never used.
3. **Promotion is earned in stages.** Static validation can at best make a bee a *candidate*. Running a submitted model
   needs an isolated executor (sandbox and GPU); when none is configured the evaluation stage is BLOCKED. A passing
   held-out evaluation promotes candidate -> *shadow* only. *Live* requires an owner approval record signed with the
   owner key and bound to the exact manifest digest (:func:`approve_live`).

What is checked statically (reason codes in brackets): HMAC-SHA256 manifest signature by the contributor key
[SIGNATURE_*]; per-contributor rate limit [RATE_LIMITED]; field grammar [MANIFEST_*]; revision pinned to a 40-hex commit
that the Hub resolves to itself [REVISION_*]; license in a redistribution-compatible allowlist, equal to the license the
Hub card declares, and the declared base model's own license also allowlisted [LICENSE_*, BASE_LICENSE_*]; repository
public and not gated [REPO_*]; file types [PICKLE_FORMAT, FILE_TYPE_NOT_ALLOWED, PATH_*]; remote code [REMOTE_CODE];
required files per kind [MISSING_*]; ``adapter_config.json`` with ``task_type`` CAUSAL_LM [ADAPTER_TASK_NOT_CAUSAL_LM]
and a base equal to the declared base, that base being one the hive serves [BASE_*]; resident size within the declared
VRAM tier [VRAM_*]; bee-name collision [NAME_COLLISION]; injection scan of the free-text description
[DESCRIPTION_INJECTION]; endpoint URL shape for the ``serving_endpoint`` kind [ENDPOINT_*].

Known limits, stated so nobody reads more into a pass: file contents are not downloaded (only ``*.json`` configs are), so a
``.safetensors`` file is trusted to be safetensors by name until the isolated stage loads it with a safetensors-only
loader; the resident-size rule is a planning estimate (weights x 1.10 + 1 GB context for standalone weights, the catalog
tier's ``min_vram_gb`` + adapter size for adapters), not a measurement; the endpoint host check rejects literal private
addresses and ``localhost`` but cannot stop DNS rebinding, so the prober must run from a network without internal access.

The output of :func:`build_registry_entry` is a plain dict (schema ``hivemind.byob.registry_entry/v1``, fields listed in
``REGISTRY_ENTRY_FIELDS`` and in ``docs/S21_BRING_YOUR_OWN_BEE.md``) so the S21 expert registry can ingest it without
importing this module.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Final, Iterable, Mapping, Sequence

from hivemind.router.catalog import PREWIRED_PARTNER_MODELS, TIERS, normalize_model_alias, tier_by_id
from hivemind.security.hivemind_firewall import InjectionVerdict, canonical_json, scan_text_for_injection

REGISTRY_ENTRY_SCHEMA: Final[str] = "hivemind.byob.registry_entry/v1"
MANIFEST_SCHEMA: Final[str] = "hivemind.byob.manifest/v1"

KINDS: Final[tuple[str, ...]] = ("lora_adapter", "gguf", "full_model", "serving_endpoint")

# SPDX-style ids as the Hub writes them in ``cardData.license``. Each one allows redistribution of the weights and is
# compatible with shipping beside AGPL-3.0 code. Non-commercial (cc-by-nc-*), "other", custom model licences (gemma,
# llama*) and missing licences are rejected: the hive cannot promise contributors' users a right it does not have.
LICENSE_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "cc-by-4.0", "agpl-3.0", "gpl-3.0"}
)

# Declared VRAM tiers (GB) a bee may claim; the resident estimate must fit the declared one.
VRAM_TIERS_GB: Final[tuple[int, ...]] = (4, 6, 8, 12, 16, 24)

# Planning rule for standalone weights (gguf, full_model): resident = weights x 1.10 + 1 GB context allowance.
STANDALONE_WEIGHT_FACTOR: Final[float] = 1.10
STANDALONE_CONTEXT_GB: Final[float] = 1.0


@dataclass(frozen=True)
class ServedBase:
    """A Hugging Face base model behind a catalog tier the hive serves; LoRA adapters must target one of these."""

    hf_repo_id: str
    tier_id: str  # must exist in ``hivemind.router.catalog.TIERS`` (checked by a test)


# Base checkpoints of the dense catalog tiers whose Ollama tag is the instruct model of that Hugging Face repo. Repackaged
# copies (``unsloth/...``, ``*-bnb-4bit``) are deliberately absent: identical weights would be a claim we do not verify.
HIVE_SERVED_BASES: Final[tuple[ServedBase, ...]] = (
    ServedBase("Qwen/Qwen2.5-Coder-3B-Instruct", "coder-lite"),
    ServedBase("Qwen/Qwen2.5-Coder-7B-Instruct", "coder"),
    ServedBase("Qwen/Qwen2.5-Coder-14B-Instruct", "coder-plus"),
    ServedBase("Qwen/Qwen3-8B", "qwen35-core"),
    ServedBase("mistralai/Mistral-7B-Instruct-v0.3", "mistral-core"),
    ServedBase("google/gemma-3-4b-it", "gemma-core"),
)

PICKLE_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt", ".joblib", ".npy", ".npz", ".h5", ".msgpack", ".marshal", ".dill"}
)
WEIGHT_EXTENSIONS: Final[frozenset[str]] = frozenset({".safetensors", ".gguf"})
CODE_EXTENSIONS: Final[frozenset[str]] = frozenset({".py", ".pyc", ".pyo", ".so", ".dll", ".dylib", ".exe", ".sh", ".ipynb"})
# Everything else that may sit in a model repo; any other extension is rejected (fail closed).
AUXILIARY_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".json", ".jsonl", ".md", ".txt", ".model", ".tiktoken", ".jinja", ".yaml", ".yml", ".csv", ".png", ".jpg", ".jpeg",
     ".gitattributes", ".gitignore"}
)
EXTENSIONLESS_ALLOWED: Final[frozenset[str]] = frozenset({"LICENSE", "NOTICE", "README", "Modelfile", "COPYING"})

MAX_JSON_BYTES: Final[int] = 2_000_000
DEFAULT_RATE_LIMIT: Final[int] = 3
DEFAULT_RATE_WINDOW_S: Final[float] = 86_400.0
MIN_KEY_BYTES: Final[int] = 32

_BEE_NAME_RE: Final[re.Pattern[str]] = re.compile(r"^bee-[a-z0-9][a-z0-9-]{2,46}[a-z0-9]$")
_CONTRIBUTOR_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9._-]{2,63}$")
_REPO_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,95}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_SHA_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{40}$")
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9_-]{1,47}$")
_KEYWORD_RE: Final[re.Pattern[str]] = re.compile(r"^[a-z0-9][a-z0-9 ._+#-]{0,39}$")
_SUBFOLDER_RE: Final[re.Pattern[str]] = re.compile(r"^(?:[A-Za-z0-9_][A-Za-z0-9._-]{0,63})(?:/[A-Za-z0-9_][A-Za-z0-9._-]{0,63}){0,3}$")


class ReasonCode(str, Enum):
    """Why a submission was rejected or blocked. Stable strings: contributors and the registry key on them."""

    SIGNATURE_MISSING = "SIGNATURE_MISSING"
    SIGNATURE_UNKNOWN_CONTRIBUTOR = "SIGNATURE_UNKNOWN_CONTRIBUTOR"
    SIGNATURE_INVALID = "SIGNATURE_INVALID"
    RATE_LIMITED = "RATE_LIMITED"
    MANIFEST_FIELD_INVALID = "MANIFEST_FIELD_INVALID"
    MANIFEST_UNKNOWN_FIELD = "MANIFEST_UNKNOWN_FIELD"
    REVISION_NOT_PINNED = "REVISION_NOT_PINNED"
    REVISION_MISMATCH = "REVISION_MISMATCH"
    LICENSE_NOT_ALLOWED = "LICENSE_NOT_ALLOWED"
    LICENSE_MISMATCH = "LICENSE_MISMATCH"
    LICENSE_UNDECLARED_ON_HUB = "LICENSE_UNDECLARED_ON_HUB"
    BASE_LICENSE_NOT_ALLOWED = "BASE_LICENSE_NOT_ALLOWED"
    REPO_NOT_FOUND = "REPO_NOT_FOUND"
    REPO_GATED_OR_PRIVATE = "REPO_GATED_OR_PRIVATE"
    PICKLE_FORMAT = "PICKLE_FORMAT"
    FILE_TYPE_NOT_ALLOWED = "FILE_TYPE_NOT_ALLOWED"
    PATH_UNSAFE = "PATH_UNSAFE"
    REMOTE_CODE = "REMOTE_CODE"
    MISSING_REQUIRED_FILE = "MISSING_REQUIRED_FILE"
    BASE_MODEL_MISMATCH = "BASE_MODEL_MISMATCH"
    BASE_MODEL_NOT_SERVED = "BASE_MODEL_NOT_SERVED"
    BASE_MODEL_NOT_FOUND = "BASE_MODEL_NOT_FOUND"
    ADAPTER_TASK_NOT_CAUSAL_LM = "ADAPTER_TASK_NOT_CAUSAL_LM"
    VRAM_TIER_INVALID = "VRAM_TIER_INVALID"
    VRAM_TIER_EXCEEDED = "VRAM_TIER_EXCEEDED"
    NAME_COLLISION = "NAME_COLLISION"
    DESCRIPTION_INJECTION = "DESCRIPTION_INJECTION"
    ENDPOINT_URL_INVALID = "ENDPOINT_URL_INVALID"
    HUB_UNREACHABLE = "HUB_UNREACHABLE"
    NO_ISOLATED_EXECUTOR = "NO_ISOLATED_EXECUTOR"
    EXECUTOR_NOT_ISOLATED = "EXECUTOR_NOT_ISOLATED"
    NO_HELDOUT_PACK = "NO_HELDOUT_PACK"
    HELDOUT_PACK_MISMATCH = "HELDOUT_PACK_MISMATCH"
    HELDOUT_CONTROL_FAILED = "HELDOUT_CONTROL_FAILED"
    ENDPOINT_PROBE_FAILED = "ENDPOINT_PROBE_FAILED"
    EVAL_BELOW_THRESHOLD = "EVAL_BELOW_THRESHOLD"
    STATIC_NOT_PASSED = "STATIC_NOT_PASSED"
    OWNER_APPROVAL_INVALID = "OWNER_APPROVAL_INVALID"
    STAGE_NOT_SHADOW = "STAGE_NOT_SHADOW"


class Verdict(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    BLOCKED = "BLOCKED"


class Stage(str, Enum):
    """Lifecycle of a bee. Only ``approve_live`` produces LIVE."""

    REJECTED = "rejected"
    BLOCKED = "blocked"
    CANDIDATE = "candidate"
    SHADOW = "shadow"
    LIVE = "live"


@dataclass(frozen=True)
class Reason:
    code: ReasonCode
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code.value, "detail": self.detail}


class HubUnavailableError(RuntimeError):
    """The Hub could not be reached or answered with a server error: the stage is BLOCKED, not failed."""


class HubNotFoundError(LookupError):
    """The Hub answered 401/403/404 for a repo, revision or file."""


# ---------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------

_MANIFEST_FIELDS: Final[tuple[str, ...]] = (
    "schema", "bee_name", "contributor_id", "hf_repo_id", "revision", "kind", "base_model", "license", "domain",
    "vram_tier_gb", "eval_pack", "description", "routing_keywords", "endpoint_url", "subfolder", "primary_file", "signature",
)


@dataclass(frozen=True)
class BeeManifest:
    """What a contributor submits. ``signature`` is HMAC-SHA256 (hex) over :meth:`signing_payload` with the contributor key."""

    bee_name: str
    contributor_id: str
    hf_repo_id: str
    revision: str
    kind: str
    base_model: str
    license: str
    domain: str
    vram_tier_gb: int
    eval_pack: str
    description: str = ""
    routing_keywords: tuple[str, ...] = ()
    endpoint_url: str = ""
    subfolder: str = ""
    primary_file: str = ""
    signature: str = ""
    schema: str = MANIFEST_SCHEMA

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BeeManifest":
        """Build a manifest from JSON. Unknown keys raise ``ValueError`` (a typo must not silently drop a field)."""
        unknown = sorted(set(data) - set(_MANIFEST_FIELDS))
        if unknown:
            raise ValueError(f"{ReasonCode.MANIFEST_UNKNOWN_FIELD.value}: {unknown}")
        kw = dict(data)
        kw["routing_keywords"] = tuple(kw.get("routing_keywords", ()))
        vram = kw.get("vram_tier_gb")
        if isinstance(vram, bool) or not isinstance(vram, int):
            raise ValueError(f"{ReasonCode.MANIFEST_FIELD_INVALID.value}: vram_tier_gb must be an integer, got {vram!r}")
        for name in ("bee_name", "contributor_id", "hf_repo_id", "revision", "kind", "base_model", "license", "domain", "eval_pack"):
            if not isinstance(kw.get(name), str):
                raise ValueError(f"{ReasonCode.MANIFEST_FIELD_INVALID.value}: {name} must be a string")
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["routing_keywords"] = list(self.routing_keywords)
        return out

    def signing_payload(self) -> bytes:
        """Canonical bytes covered by the signature: every field except ``signature`` itself."""
        body = self.to_dict()
        body.pop("signature")
        return canonical_json(body)

    def digest(self) -> str:
        """SHA-256 of the signing payload; owner approvals and the registry entry bind to it."""
        return hashlib.sha256(self.signing_payload()).hexdigest()

    def signed(self, key: bytes) -> "BeeManifest":
        return replace(self, signature=hmac_hex(key, self.signing_payload()))


def hmac_hex(key: bytes, payload: bytes) -> str:
    if len(key) < MIN_KEY_BYTES:
        raise ValueError(f"HMAC key must be at least {MIN_KEY_BYTES} bytes")
    return hmac.new(key, payload, hashlib.sha256).hexdigest()


def verify_signature(manifest: BeeManifest, keyring: Mapping[str, bytes]) -> list[Reason]:
    if not manifest.signature:
        return [Reason(ReasonCode.SIGNATURE_MISSING, "manifest carries no signature")]
    key = keyring.get(manifest.contributor_id)
    if key is None:
        return [Reason(ReasonCode.SIGNATURE_UNKNOWN_CONTRIBUTOR, f"no key registered for {manifest.contributor_id!r}")]
    expected = hmac_hex(key, manifest.signing_payload())
    if not hmac.compare_digest(expected, manifest.signature):
        return [Reason(ReasonCode.SIGNATURE_INVALID, "HMAC-SHA256 does not match the manifest bytes")]
    return []


# ---------------------------------------------------------------------------------------------------------
# Pure field checks
# ---------------------------------------------------------------------------------------------------------


def _bad(field_name: str, value: Any, rule: str) -> Reason:
    return Reason(ReasonCode.MANIFEST_FIELD_INVALID, f"{field_name}={value!r}: {rule}")


def check_endpoint_url(url: str) -> list[Reason]:
    """https only, no credentials, no localhost and no literal private, loopback, link-local or reserved address."""
    parsed = urllib.parse.urlsplit(url)
    host = parsed.hostname or ""
    if parsed.scheme != "https" or not host:
        return [Reason(ReasonCode.ENDPOINT_URL_INVALID, f"{url!r}: must be an https URL with a host")]
    if parsed.username or parsed.password:
        return [Reason(ReasonCode.ENDPOINT_URL_INVALID, "credentials in the URL are not accepted")]
    if host == "localhost" or host.endswith(".localhost") or host.endswith(".internal") or host.endswith(".local"):
        return [Reason(ReasonCode.ENDPOINT_URL_INVALID, f"host {host!r} is local")]
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return []
    if not address.is_global:
        return [Reason(ReasonCode.ENDPOINT_URL_INVALID, f"address {host} is not globally routable")]
    return []


def check_manifest_fields(manifest: BeeManifest) -> list[Reason]:
    """Grammar and allowlist checks that need no network. Returns every failure, not just the first."""
    m = manifest
    reasons: list[Reason] = []
    if m.schema != MANIFEST_SCHEMA:
        reasons.append(_bad("schema", m.schema, f"must be {MANIFEST_SCHEMA}"))
    if not _BEE_NAME_RE.match(m.bee_name):
        reasons.append(_bad("bee_name", m.bee_name, "must match bee-[a-z0-9-], 8 to 52 characters"))
    if not _CONTRIBUTOR_RE.match(m.contributor_id):
        reasons.append(_bad("contributor_id", m.contributor_id, "lowercase letters, digits, '.', '_' or '-'"))
    if not _REPO_RE.match(m.hf_repo_id) or ".." in m.hf_repo_id:
        reasons.append(_bad("hf_repo_id", m.hf_repo_id, "must be <owner>/<name>"))
    if not _SHA_RE.match(m.revision):
        reasons.append(Reason(ReasonCode.REVISION_NOT_PINNED, f"revision {m.revision!r} is not a 40-hex commit sha"))
    if m.kind not in KINDS:
        reasons.append(_bad("kind", m.kind, f"one of {list(KINDS)}"))
    if not _REPO_RE.match(m.base_model) or ".." in m.base_model:
        reasons.append(_bad("base_model", m.base_model, "must be a Hugging Face repo id <owner>/<name>"))
    if m.license.lower() not in LICENSE_ALLOWLIST:
        reasons.append(Reason(ReasonCode.LICENSE_NOT_ALLOWED, f"{m.license!r} not in {sorted(LICENSE_ALLOWLIST)}"))
    if not _SLUG_RE.match(m.domain):
        reasons.append(_bad("domain", m.domain, "lowercase slug"))
    if not _SLUG_RE.match(m.eval_pack):
        reasons.append(_bad("eval_pack", m.eval_pack, "lowercase slug naming a hive-held evaluation pack"))
    if m.vram_tier_gb not in VRAM_TIERS_GB:
        reasons.append(Reason(ReasonCode.VRAM_TIER_INVALID, f"{m.vram_tier_gb} not in {list(VRAM_TIERS_GB)}"))
    if len(m.description) > 2000:
        reasons.append(_bad("description", f"<{len(m.description)} chars>", "at most 2000 characters"))
    elif m.description and scan_text_for_injection(m.description).verdict is InjectionVerdict.INJECTION:
        reasons.append(Reason(ReasonCode.DESCRIPTION_INJECTION, "description scans INJECTION (hivemind_firewall PI rules)"))
    if len(m.routing_keywords) > 32 or any(not isinstance(k, str) or not _KEYWORD_RE.match(k) for k in m.routing_keywords):
        reasons.append(_bad("routing_keywords", list(m.routing_keywords)[:4], "at most 32 lowercase keywords of <= 40 chars"))
    if m.subfolder and (not _SUBFOLDER_RE.match(m.subfolder) or ".." in m.subfolder.split("/")):
        reasons.append(Reason(ReasonCode.PATH_UNSAFE, f"subfolder {m.subfolder!r}"))
    if m.kind == "gguf" and not m.primary_file.endswith(".gguf"):
        reasons.append(_bad("primary_file", m.primary_file, "kind gguf must name the .gguf file to serve"))
    if m.primary_file and (m.primary_file.startswith("/") or ".." in m.primary_file.split("/")):
        reasons.append(Reason(ReasonCode.PATH_UNSAFE, f"primary_file {m.primary_file!r}"))
    if m.kind == "serving_endpoint":
        reasons.extend(check_endpoint_url(m.endpoint_url))
    elif m.endpoint_url:
        reasons.append(_bad("endpoint_url", m.endpoint_url, "only the serving_endpoint kind carries an endpoint"))
    return reasons


def normalised_name_key(name: str) -> str:
    """Collision key: lower-case alphanumerics with a leading ``hivemind`` and ``bee`` stripped (``bee-coder`` == ``coder``)."""
    key = re.sub(r"[^a-z0-9]", "", normalize_model_alias(name).lower())
    key = key.removeprefix("hivemind")
    return key.removeprefix("bee")


def existing_bee_names(extra: Iterable[str] = ()) -> frozenset[str]:
    """Names already taken: catalog tiers and partner aliases, serving and Season 3 experts, flagship bees, and ``extra``.

    ``extra`` is where the expert registry passes the bees it already holds. ``btm_moe`` (which imports torch) is imported
    here rather than at module load so that ``scripts/byob_submit.py`` starts quickly.
    """
    from hivemind.core.btm_moe import SEASON_3_DOMAIN_EXPERT_SPECS
    from hivemind.core.serving_experts import SERVING_EXPERT_SPECS
    from hivemind.swarm.rusty_sundials_scientific_bee import BEE_RUSTY_SUNDIALS_ID

    names: set[str] = {tier.tier_id for tier in TIERS}
    names.update(alias for partner in PREWIRED_PARTNER_MODELS for alias in partner.aliases)
    names.update(SERVING_EXPERT_SPECS)
    names.update(SEASON_3_DOMAIN_EXPERT_SPECS)
    names.add(BEE_RUSTY_SUNDIALS_ID)
    names.update(extra)
    return frozenset(names)


def check_name_collision(bee_name: str, taken: Iterable[str]) -> list[Reason]:
    key = normalised_name_key(bee_name)
    clashes = sorted(name for name in taken if normalised_name_key(name) == key)
    if clashes:
        return [Reason(ReasonCode.NAME_COLLISION, f"{bee_name!r} collides with {clashes[:3]}")]
    return []


def served_base(base_model: str) -> ServedBase | None:
    return next((b for b in HIVE_SERVED_BASES if b.hf_repo_id.lower() == base_model.lower()), None)


# ---------------------------------------------------------------------------------------------------------
# Repository tree checks (pure: they take the Hub tree listing as data)
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoFile:
    path: str
    size: int
    sha256: str  # LFS sha256 when the Hub reports one, else "" (small git blobs only carry a git oid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    if name.startswith(".") and name.count(".") == 1:
        return name  # ".gitattributes"
    return "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""


def parse_tree(entries: Sequence[Mapping[str, Any]]) -> list[RepoFile]:
    files: list[RepoFile] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        lfs = entry.get("lfs") or {}
        files.append(RepoFile(str(entry.get("path", "")), int(entry.get("size", 0)), str(lfs.get("oid", ""))))
    return files


def check_repo_files(files: Sequence[RepoFile]) -> list[Reason]:
    """Whole-repository file policy: safetensors/gguf weights only, no pickle, no code, no unsafe path."""
    reasons: list[Reason] = []
    for f in files:
        parts = f.path.split("/")
        if not f.path or f.path.startswith("/") or ".." in parts or "\\" in f.path:
            reasons.append(Reason(ReasonCode.PATH_UNSAFE, f.path))
            continue
        ext = _extension(f.path)
        if ext in PICKLE_EXTENSIONS:
            reasons.append(Reason(ReasonCode.PICKLE_FORMAT, f"{f.path} ({ext} can carry a pickle)"))
        elif ext in CODE_EXTENSIONS:
            reasons.append(Reason(ReasonCode.REMOTE_CODE, f"{f.path} is executable code"))
        elif ext in WEIGHT_EXTENSIONS or ext in AUXILIARY_EXTENSIONS:
            continue
        elif ext == "" and parts[-1] in EXTENSIONLESS_ALLOWED:
            continue
        else:
            reasons.append(Reason(ReasonCode.FILE_TYPE_NOT_ALLOWED, f"{f.path} ({ext or 'no extension'})"))
    return reasons


def _under(path: str, subfolder: str) -> bool:
    return not subfolder or path.startswith(subfolder.rstrip("/") + "/")


def _join(subfolder: str, name: str) -> str:
    return f"{subfolder.rstrip('/')}/{name}" if subfolder else name


def served_weight_files(manifest: BeeManifest, files: Sequence[RepoFile]) -> list[RepoFile]:
    """The weight files the hive would load for this kind (the size check and the registry entry use these)."""
    if manifest.kind == "gguf":
        return [f for f in files if f.path == manifest.primary_file]
    if manifest.kind == "lora_adapter":
        return [f for f in files if f.path == _join(manifest.subfolder, "adapter_model.safetensors")]
    return [f for f in files if _under(f.path, manifest.subfolder) and "/" not in f.path[len(manifest.subfolder):].strip("/")
            and f.path.endswith(".safetensors")]


def check_required_files(manifest: BeeManifest, files: Sequence[RepoFile]) -> list[Reason]:
    paths = {f.path for f in files}
    need: list[str] = []
    if manifest.kind == "lora_adapter":
        need = [_join(manifest.subfolder, "adapter_config.json"), _join(manifest.subfolder, "adapter_model.safetensors")]
    elif manifest.kind in ("full_model", "serving_endpoint"):
        need = [_join(manifest.subfolder, "config.json")]
    elif manifest.kind == "gguf":
        need = [manifest.primary_file]
    reasons = [Reason(ReasonCode.MISSING_REQUIRED_FILE, p) for p in need if p not in paths]
    if manifest.kind in ("full_model", "serving_endpoint") and not served_weight_files(manifest, files):
        reasons.append(Reason(ReasonCode.MISSING_REQUIRED_FILE, f"no .safetensors weights under {manifest.subfolder or '/'}"))
    return reasons


_PY_IDENT_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRANSFORMERS_MODULE_RE: Final[re.Pattern[str]] = re.compile(r"^transformers(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")


def peft_auto_mapping_is_library_only(mapping: Any) -> bool:
    """True when a peft ``auto_mapping`` only names a class inside the installed ``transformers`` package.

    peft writes ``{"base_model_class": "Qwen2ForCausalLM", "parent_library": "transformers.models.qwen2.modeling_qwen2"}``
    for ordinary adapters (Unsloth adds ``"unsloth_fixed": true``); that imports library code, not repository code. Any
    other ``parent_library`` (a repo module, ``os``, ...) or a non-identifier class name is treated as remote code.
    """
    if not isinstance(mapping, Mapping):
        return False
    cls_name = mapping.get("base_model_class")
    parent = mapping.get("parent_library")
    extra = set(mapping) - {"base_model_class", "parent_library", "unsloth_fixed"}
    return (isinstance(cls_name, str) and bool(_PY_IDENT_RE.match(cls_name)) and isinstance(parent, str)
            and bool(_TRANSFORMERS_MODULE_RE.match(parent)) and not extra)


def has_remote_code(config: Mapping[str, Any] | None) -> bool:
    """True when loading this config would import repository code.

    A transformers ``auto_map`` always points at modules shipped in the repo, so any non-empty one is remote code. A peft
    ``auto_mapping`` is remote code unless :func:`peft_auto_mapping_is_library_only` holds.
    """
    if not config:
        return False
    if config.get("auto_map"):
        return True
    mapping = config.get("auto_mapping")
    return bool(mapping) and not peft_auto_mapping_is_library_only(mapping)


def estimate_resident_gb(manifest: BeeManifest, weight_bytes: int) -> float | None:
    """Planning estimate of GPU residency in GB, or ``None`` when it cannot be computed (adapter on an unserved base)."""
    weights_gb = weight_bytes / 1e9
    if manifest.kind == "lora_adapter":
        base = served_base(manifest.base_model)
        tier = tier_by_id(base.tier_id) if base else None
        return None if tier is None else round(tier.min_vram_gb + weights_gb, 3)
    return round(weights_gb * STANDALONE_WEIGHT_FACTOR + STANDALONE_CONTEXT_GB, 3)


# ---------------------------------------------------------------------------------------------------------
# Hugging Face Hub access (the only network I/O of the static stage)
# ---------------------------------------------------------------------------------------------------------


class HfHttpHubClient:
    """Read-only client for the public Hub REST API (no token, no ``huggingface_hub`` dependency).

    ``revision_info`` -> ``GET /api/models/{repo}/revision/{rev}``; ``list_tree`` -> ``GET /api/models/{repo}/tree/{rev}
    ?recursive=true`` following ``Link: rel="next"`` pages; ``fetch_json`` -> ``GET /{repo}/resolve/{rev}/{path}`` capped
    at ``MAX_JSON_BYTES``. 401/403/404 raise :class:`HubNotFoundError`; anything else that fails raises
    :class:`HubUnavailableError`.
    """

    def __init__(self, endpoint: str = "https://huggingface.co", timeout_s: float = 20.0,
                 opener: Callable[..., Any] | None = None) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = timeout_s
        self._opener = opener

    def _open(self, request: urllib.request.Request, timeout: float) -> Any:
        opener = self._opener if self._opener is not None else urllib.request.urlopen
        return opener(request, timeout=timeout)

    def _get(self, url: str, limit: int) -> tuple[bytes, str]:
        request = urllib.request.Request(url, headers={"User-Agent": "hivemind-byob/1", "Accept": "application/json"})
        try:
            with self._open(request, timeout=self.timeout_s) as response:
                body = response.read(limit + 1)
                link = response.headers.get("Link", "") or ""
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise HubNotFoundError(f"{exc.code} for {url}") from exc
            raise HubUnavailableError(f"HTTP {exc.code} for {url}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise HubUnavailableError(f"{type(exc).__name__}: {exc}") from exc
        if len(body) > limit:
            raise HubUnavailableError(f"response from {url} exceeds {limit} bytes")
        return body, link

    def _get_json(self, url: str, limit: int = MAX_JSON_BYTES * 8) -> tuple[Any, str]:
        body, link = self._get(url, limit)
        try:
            return json.loads(body), link
        except ValueError as exc:
            raise HubUnavailableError(f"non-JSON answer from {url}") from exc

    def revision_info(self, repo_id: str, revision: str) -> dict[str, Any]:
        data, _ = self._get_json(f"{self.endpoint}/api/models/{repo_id}/revision/{urllib.parse.quote(revision, safe='')}")
        if not isinstance(data, dict):
            raise HubUnavailableError("revision info is not an object")
        return data

    def list_tree(self, repo_id: str, revision: str) -> list[dict[str, Any]]:
        url: str | None = f"{self.endpoint}/api/models/{repo_id}/tree/{urllib.parse.quote(revision, safe='')}?recursive=true"
        entries: list[dict[str, Any]] = []
        pages = 0
        while url:
            pages += 1
            if pages > 50:
                raise HubUnavailableError("tree listing has more than 50 pages")
            page, link = self._get_json(url)
            if not isinstance(page, list):
                raise HubUnavailableError("tree listing is not a list")
            entries.extend(page)
            match = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = match.group(1) if match else None
            if url and not url.startswith(self.endpoint + "/"):
                raise HubUnavailableError(f"pagination left the Hub: {url}")
        return entries

    def fetch_json(self, repo_id: str, revision: str, path: str) -> dict[str, Any]:
        quoted = urllib.parse.quote(path)
        data, _ = self._get_json(f"{self.endpoint}/{repo_id}/resolve/{revision}/{quoted}", MAX_JSON_BYTES)
        if not isinstance(data, dict):
            raise HubUnavailableError(f"{path} is not a JSON object")
        return data


# ---------------------------------------------------------------------------------------------------------
# Rate limit
# ---------------------------------------------------------------------------------------------------------


class SubmissionLedger:
    """Per-contributor sliding-window rate limit; optionally persisted as JSONL (one ``{contributor_id, ts}`` per line).

    Only submissions whose signature verified are recorded, so a forged manifest cannot spend a real contributor's quota.
    Rejected-but-signed submissions do count: re-submitting until a check passes is what the limit is for.
    """

    def __init__(self, path: Path | None = None, max_per_window: int = DEFAULT_RATE_LIMIT,
                 window_s: float = DEFAULT_RATE_WINDOW_S) -> None:
        if max_per_window < 1 or window_s <= 0:
            raise ValueError("max_per_window must be >= 1 and window_s > 0")
        self.path = path
        self.max_per_window = max_per_window
        self.window_s = window_s
        self._events: list[tuple[str, float]] = []
        if path is not None and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._events.append((str(row["contributor_id"]), float(row["ts"])))

    def count(self, contributor_id: str, now: float) -> int:
        return sum(1 for cid, ts in self._events if cid == contributor_id and now - self.window_s < ts <= now)

    def check(self, contributor_id: str, now: float) -> list[Reason]:
        used = self.count(contributor_id, now)
        if used >= self.max_per_window:
            return [Reason(ReasonCode.RATE_LIMITED, f"{used} submissions in the last {self.window_s:.0f}s (max {self.max_per_window})")]
        return []

    def record(self, contributor_id: str, now: float) -> None:
        self._events.append((contributor_id, now))
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"contributor_id": contributor_id, "ts": now}) + "\n")


# ---------------------------------------------------------------------------------------------------------
# Static validation
# ---------------------------------------------------------------------------------------------------------


@dataclass
class StaticReport:
    verdict: Verdict
    reasons: list[Reason] = field(default_factory=list)
    files: list[RepoFile] = field(default_factory=list)
    weight_files: list[RepoFile] = field(default_factory=list)
    weight_bytes: int = 0
    resident_gb: float | None = None
    hub_license: str = ""
    checked_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "reasons": [r.to_dict() for r in self.reasons],
            "file_count": len(self.files),
            "weight_files": [f.to_dict() for f in self.weight_files],
            "weight_bytes": self.weight_bytes,
            "resident_gb_estimate": self.resident_gb,
            "hub_license": self.hub_license,
            "checked_utc": self.checked_utc,
        }


def _utc(now: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


def _card_license(info: Mapping[str, Any]) -> str:
    card = info.get("cardData") or {}
    lic = card.get("license", "") if isinstance(card, Mapping) else ""
    if isinstance(lic, list):
        lic = lic[0] if len(lic) == 1 else ""
    if not lic:
        lic = next((t.split(":", 1)[1] for t in info.get("tags", []) or [] if isinstance(t, str) and t.startswith("license:")), "")
    return str(lic).lower()


def _hub_checks(manifest: BeeManifest, hub: HfHttpHubClient, report: StaticReport) -> None:
    """Every Hub-backed check; raises ``HubUnavailableError`` (-> BLOCKED) when the Hub cannot answer."""
    m = manifest
    try:
        info = hub.revision_info(m.hf_repo_id, m.revision)
    except HubNotFoundError as exc:
        report.reasons.append(Reason(ReasonCode.REPO_NOT_FOUND, f"{m.hf_repo_id}@{m.revision}: {exc}"))
        return
    if info.get("sha") != m.revision:
        report.reasons.append(Reason(ReasonCode.REVISION_MISMATCH, f"Hub resolved {m.revision} to {info.get('sha')!r}"))
    if info.get("private") or info.get("gated") or info.get("disabled"):
        report.reasons.append(Reason(ReasonCode.REPO_GATED_OR_PRIVATE, "the hive can only mirror public, ungated repos"))
    report.hub_license = _card_license(info)
    if not report.hub_license:
        report.reasons.append(Reason(ReasonCode.LICENSE_UNDECLARED_ON_HUB, "model card declares no license"))
    elif report.hub_license != m.license.lower():
        report.reasons.append(Reason(ReasonCode.LICENSE_MISMATCH, f"manifest says {m.license!r}, Hub card says {report.hub_license!r}"))
    hub_config = info.get("config") if isinstance(info.get("config"), Mapping) else None
    if "custom_code" in (info.get("tags") or []) or has_remote_code(hub_config):
        report.reasons.append(Reason(ReasonCode.REMOTE_CODE, "Hub marks the repo custom_code / config has auto_map"))
    try:
        base_info = hub.revision_info(m.base_model, "main")
    except HubNotFoundError:
        report.reasons.append(Reason(ReasonCode.BASE_MODEL_NOT_FOUND, m.base_model))
    else:
        base_license = _card_license(base_info)
        if base_license not in LICENSE_ALLOWLIST:
            report.reasons.append(Reason(ReasonCode.BASE_LICENSE_NOT_ALLOWED,
                                         f"{m.base_model} is licensed {base_license or 'undeclared'!r}"))

    report.files = parse_tree(hub.list_tree(m.hf_repo_id, m.revision))
    report.reasons.extend(check_repo_files(report.files))
    report.reasons.extend(check_required_files(m, report.files))
    paths = {f.path for f in report.files}

    for config_name in ("config.json", "tokenizer_config.json"):
        path = _join(m.subfolder, config_name)
        if path in paths and has_remote_code(hub.fetch_json(m.hf_repo_id, m.revision, path)):
            report.reasons.append(Reason(ReasonCode.REMOTE_CODE, f"{path} declares auto_map"))

    if m.kind == "lora_adapter":
        if served_base(m.base_model) is None:
            report.reasons.append(Reason(ReasonCode.BASE_MODEL_NOT_SERVED, f"{m.base_model} is not in HIVE_SERVED_BASES"))
        adapter_path = _join(m.subfolder, "adapter_config.json")
        if adapter_path in paths:
            adapter_cfg = hub.fetch_json(m.hf_repo_id, m.revision, adapter_path)
            declared_in_file = str(adapter_cfg.get("base_model_name_or_path", ""))
            if declared_in_file.lower() != m.base_model.lower():
                report.reasons.append(Reason(ReasonCode.BASE_MODEL_MISMATCH,
                                             f"adapter_config says {declared_in_file!r}, manifest says {m.base_model!r}"))
            task_type = adapter_cfg.get("task_type")
            if task_type != "CAUSAL_LM":
                report.reasons.append(Reason(ReasonCode.ADAPTER_TASK_NOT_CAUSAL_LM, f"task_type {task_type!r}: a bee must generate text"))
            if has_remote_code(adapter_cfg):
                report.reasons.append(Reason(ReasonCode.REMOTE_CODE, f"{adapter_path} declares auto_mapping"))
    else:
        card = info.get("cardData") or {}
        card_base = card.get("base_model") if isinstance(card, Mapping) else None
        card_bases = [card_base] if isinstance(card_base, str) else list(card_base or [])
        if card_bases and m.base_model.lower() not in {str(b).lower() for b in card_bases}:
            report.reasons.append(Reason(ReasonCode.BASE_MODEL_MISMATCH, f"model card base_model {card_bases[:3]} != {m.base_model!r}"))

    report.weight_files = served_weight_files(m, report.files)
    report.weight_bytes = sum(f.size for f in report.weight_files)
    if m.kind != "serving_endpoint":
        report.resident_gb = estimate_resident_gb(m, report.weight_bytes)
        if report.resident_gb is not None and report.resident_gb > m.vram_tier_gb:
            report.reasons.append(Reason(ReasonCode.VRAM_TIER_EXCEEDED,
                                         f"estimated {report.resident_gb} GB > declared {m.vram_tier_gb} GB"))


def static_validate(manifest: BeeManifest, *, keyring: Mapping[str, bytes], hub: HfHttpHubClient, ledger: SubmissionLedger,
                    taken_names: Iterable[str], now: float | None = None) -> StaticReport:
    """Run every static check. REJECT on any reason; BLOCKED when the Hub cannot be reached; ACCEPT otherwise.

    Order matters only for the signature: an unsigned or forged manifest stops there (nothing else is worth reporting to
    an unauthenticated sender) and is not recorded against the claimed contributor's rate limit.
    """
    ts = time.time() if now is None else now
    report = StaticReport(verdict=Verdict.REJECT, checked_utc=_utc(ts))
    report.reasons.extend(verify_signature(manifest, keyring))
    if report.reasons:
        return report
    report.reasons.extend(ledger.check(manifest.contributor_id, ts))
    ledger.record(manifest.contributor_id, ts)
    if report.reasons:
        return report
    report.reasons.extend(check_manifest_fields(manifest))
    report.reasons.extend(check_name_collision(manifest.bee_name, taken_names))
    if any(r.code in (ReasonCode.REVISION_NOT_PINNED, ReasonCode.MANIFEST_FIELD_INVALID) for r in report.reasons):
        return report  # an unpinned or malformed repo reference must not be sent to the Hub
    try:
        _hub_checks(manifest, hub, report)
    except HubUnavailableError as exc:
        report.reasons.append(Reason(ReasonCode.HUB_UNREACHABLE, str(exc)))
        report.verdict = Verdict.BLOCKED
        return report
    report.verdict = Verdict.REJECT if report.reasons else Verdict.ACCEPT
    return report


# ---------------------------------------------------------------------------------------------------------
# Evaluation stage
# ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeldOutItem:
    item_id: str
    prompt: str
    expected_regex: str
    reference_answer: str  # used only by the positive control; never sent to the bee


@dataclass(frozen=True)
class HeldOutPack:
    """A hive-held evaluation pack (kept outside the public repo). Loaded with :meth:`from_json_file`."""

    pack_id: str
    items: tuple[HeldOutItem, ...]

    @classmethod
    def from_json_file(cls, path: Path) -> "HeldOutPack":
        data = json.loads(path.read_text(encoding="utf-8"))
        items = tuple(HeldOutItem(str(i["item_id"]), str(i["prompt"]), str(i["expected_regex"]), str(i["reference_answer"]))
                      for i in data["items"])
        return cls(str(data["pack_id"]), items)

    def score(self, answers: Sequence[str]) -> float:
        if len(answers) != len(self.items) or not self.items:
            raise ValueError("one answer per item is required")
        hits = sum(1 for item, ans in zip(self.items, answers) if re.search(item.expected_regex, ans or ""))
        return hits / len(self.items)

    def control_reasons(self) -> list[Reason]:
        """Positive control: every reference answer must pass its own checker. Negative control: an empty answer and the
        prompt echoed back must both fail. A pack whose checker cannot fail (or cannot pass) measures nothing."""
        if not self.items:
            return [Reason(ReasonCode.HELDOUT_CONTROL_FAILED, "pack has no items")]
        reasons: list[Reason] = []
        for item in self.items:
            if not re.search(item.expected_regex, item.reference_answer):
                reasons.append(Reason(ReasonCode.HELDOUT_CONTROL_FAILED, f"{item.item_id}: reference answer fails its checker"))
            if re.search(item.expected_regex, "") or re.search(item.expected_regex, item.prompt):
                reasons.append(Reason(ReasonCode.HELDOUT_CONTROL_FAILED, f"{item.item_id}: checker accepts an empty or echoed answer"))
        return reasons


AnswerFn = Callable[[BeeManifest, Sequence[RepoFile], Sequence[str]], list[str]]


@dataclass(frozen=True)
class IsolatedExecutor:
    """Runs a submitted model inside a sandbox with a GPU and returns one answer per prompt, in order.

    ``answer_pack`` must load weights with safetensors / llama.cpp loaders only; ``isolated`` may be True only when the run
    is really confined (no network, read-only root). The default deployment configures none, so evaluation is BLOCKED.
    """

    name: str
    isolated: bool
    answer_pack: AnswerFn


EndpointProber = Callable[[str, str, str, float], str]


def https_chat_prober(url: str, model: str, prompt: str, timeout_s: float) -> str:
    """POST one OpenAI-format chat request to ``{url}/v1/chat/completions`` and return the first message content."""
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "User-Agent": "hivemind-byob/1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - https enforced by check_endpoint_url
        data = json.loads(response.read(1_000_000))
    return str(data["choices"][0]["message"]["content"])


@dataclass
class EvaluationReport:
    verdict: Verdict
    reasons: list[Reason] = field(default_factory=list)
    pack_id: str = ""
    n_items: int = 0
    pass_rate: float | None = None
    executor: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict.value, "reasons": [r.to_dict() for r in self.reasons], "pack_id": self.pack_id,
                "n_items": self.n_items, "pass_rate": self.pass_rate, "executor": self.executor}


def evaluate(manifest: BeeManifest, static: StaticReport, *, pack: HeldOutPack | None,
             executor: IsolatedExecutor | None = None, prober: EndpointProber | None = None,
             min_pass_rate: float = 0.6, timeout_s: float = 30.0) -> EvaluationReport:
    """Score the bee on a held-out pack. Never ACCEPT without a run: no executor/prober, no pack or a broken pack -> BLOCKED.

    Weight kinds need an ``IsolatedExecutor`` whose ``isolated`` is True. The ``serving_endpoint`` kind is probed through
    ``prober`` with ``timeout_s`` per item; nothing of the contributor's runs on hive hardware. The pack's own positive and
    negative controls run before any answer is scored.
    """
    if static.verdict is not Verdict.ACCEPT:
        return EvaluationReport(Verdict.REJECT if static.verdict is Verdict.REJECT else Verdict.BLOCKED,
                                [Reason(ReasonCode.STATIC_NOT_PASSED, f"static verdict {static.verdict.value}")])
    report = EvaluationReport(Verdict.BLOCKED)
    if manifest.kind == "serving_endpoint":
        if prober is None:
            report.reasons.append(Reason(ReasonCode.NO_ISOLATED_EXECUTOR, "no endpoint prober configured"))
            return report
        report.executor = "endpoint_prober"
    else:
        if executor is None:
            report.reasons.append(Reason(ReasonCode.NO_ISOLATED_EXECUTOR, "no sandboxed GPU executor configured on this host"))
            return report
        report.executor = executor.name
        if not executor.isolated:
            report.reasons.append(Reason(ReasonCode.EXECUTOR_NOT_ISOLATED, f"{executor.name} does not declare isolation"))
            return report
    if pack is None:
        report.reasons.append(Reason(ReasonCode.NO_HELDOUT_PACK, "no held-out pack configured"))
        return report
    report.pack_id, report.n_items = pack.pack_id, len(pack.items)
    if pack.pack_id != manifest.eval_pack:
        report.reasons.append(Reason(ReasonCode.HELDOUT_PACK_MISMATCH, f"{pack.pack_id} != {manifest.eval_pack}"))
        return report
    controls = pack.control_reasons()
    if controls:
        report.reasons.extend(controls)
        return report
    prompts = [item.prompt for item in pack.items]
    if manifest.kind == "serving_endpoint" and prober is not None:
        answers: list[str] = []
        for prompt in prompts:
            try:
                answers.append(prober(manifest.endpoint_url, manifest.bee_name, prompt, timeout_s))
            except Exception as exc:  # any transport, timeout or decoding failure of a contributor endpoint
                report.verdict = Verdict.REJECT
                report.reasons.append(Reason(ReasonCode.ENDPOINT_PROBE_FAILED, f"{type(exc).__name__}: {str(exc)[:120]}"))
                return report
    elif executor is not None:
        answers = executor.answer_pack(manifest, static.weight_files, prompts)
    report.pass_rate = pack.score(answers)
    if report.pass_rate >= min_pass_rate:
        report.verdict = Verdict.ACCEPT
    else:
        report.verdict = Verdict.REJECT
        report.reasons.append(Reason(ReasonCode.EVAL_BELOW_THRESHOLD, f"pass rate {report.pass_rate:.3f} < {min_pass_rate}"))
    return report


# ---------------------------------------------------------------------------------------------------------
# Registry entry and promotion
# ---------------------------------------------------------------------------------------------------------

REGISTRY_ENTRY_FIELDS: Final[tuple[str, ...]] = (
    "schema", "expert_id", "public_id", "contributor_id", "kind", "domain", "routing_keywords", "description",
    "source", "base_model", "base_tier_id", "license", "vram_tier_gb", "resident_gb_estimate", "eval_pack",
    "endpoint_url", "stage", "static", "evaluation", "reasons", "manifest_sha256", "owner_approval", "created_utc",
)


def _stage(static: StaticReport, evaluation: EvaluationReport | None) -> Stage:
    if static.verdict is Verdict.REJECT:
        return Stage.REJECTED
    if static.verdict is Verdict.BLOCKED:
        return Stage.BLOCKED
    if evaluation is None or evaluation.verdict is Verdict.BLOCKED:
        return Stage.CANDIDATE
    return Stage.SHADOW if evaluation.verdict is Verdict.ACCEPT else Stage.REJECTED


def build_registry_entry(manifest: BeeManifest, static: StaticReport, evaluation: EvaluationReport | None,
                         now: float | None = None) -> dict[str, Any]:
    """Registry-entry-shaped dict; the best stage it can carry is ``shadow``."""
    base = served_base(manifest.base_model)
    reasons = [r.to_dict() for r in static.reasons] + ([r.to_dict() for r in evaluation.reasons] if evaluation else [])
    entry = {
        "schema": REGISTRY_ENTRY_SCHEMA,
        "expert_id": manifest.bee_name,
        "public_id": f"hivemind/{manifest.bee_name}",
        "contributor_id": manifest.contributor_id,
        "kind": manifest.kind,
        "domain": manifest.domain,
        "routing_keywords": list(manifest.routing_keywords),
        "description": manifest.description,
        "source": {
            "hf_repo_id": manifest.hf_repo_id,
            "revision": manifest.revision,
            "subfolder": manifest.subfolder,
            "weight_files": [f.to_dict() for f in static.weight_files],
            "weight_bytes": static.weight_bytes,
        },
        "base_model": manifest.base_model,
        "base_tier_id": base.tier_id if base else None,
        "license": manifest.license.lower(),
        "vram_tier_gb": manifest.vram_tier_gb,
        "resident_gb_estimate": static.resident_gb,
        "eval_pack": manifest.eval_pack,
        "endpoint_url": manifest.endpoint_url or None,
        "stage": _stage(static, evaluation).value,
        "static": static.to_dict(),
        "evaluation": evaluation.to_dict() if evaluation else None,
        "reasons": reasons,
        "manifest_sha256": manifest.digest(),
        "owner_approval": None,
        "created_utc": _utc(time.time() if now is None else now),
    }
    return entry  # field order equals REGISTRY_ENTRY_FIELDS (asserted in tests/test_byob.py)


def owner_approval_payload(entry: Mapping[str, Any], approver: str, approved_utc: str) -> bytes:
    return canonical_json({"action": "byob_promote_live", "expert_id": entry["expert_id"],
                           "manifest_sha256": entry["manifest_sha256"], "approver": approver, "approved_utc": approved_utc})


def sign_owner_approval(entry: Mapping[str, Any], approver: str, approved_utc: str, owner_key: bytes) -> dict[str, str]:
    return {"approver": approver, "approved_utc": approved_utc,
            "manifest_sha256": str(entry["manifest_sha256"]),
            "signature": hmac_hex(owner_key, owner_approval_payload(entry, approver, approved_utc))}


def approve_live(entry: Mapping[str, Any], approval: Mapping[str, str], owner_key: bytes) -> tuple[dict[str, Any], list[Reason]]:
    """shadow -> live, only with an owner approval signed over this exact manifest digest. Returns (entry, reasons)."""
    out = dict(entry)
    if entry.get("stage") != Stage.SHADOW.value:
        return out, [Reason(ReasonCode.STAGE_NOT_SHADOW, f"stage is {entry.get('stage')!r}; only shadow can go live")]
    try:
        expected = hmac_hex(owner_key, owner_approval_payload(entry, approval["approver"], approval["approved_utc"]))
    except (KeyError, ValueError) as exc:
        return out, [Reason(ReasonCode.OWNER_APPROVAL_INVALID, f"malformed approval: {exc}")]
    if approval.get("manifest_sha256") != entry.get("manifest_sha256") or not hmac.compare_digest(expected, approval.get("signature", "")):
        return out, [Reason(ReasonCode.OWNER_APPROVAL_INVALID, "signature does not cover this manifest digest with the owner key")]
    out["stage"] = Stage.LIVE.value
    out["owner_approval"] = dict(approval)
    return out, []


def run_submission(manifest: BeeManifest, *, keyring: Mapping[str, bytes], hub: HfHttpHubClient, ledger: SubmissionLedger,
                   taken_names: Iterable[str], pack: HeldOutPack | None = None, executor: IsolatedExecutor | None = None,
                   prober: EndpointProber | None = None, min_pass_rate: float = 0.6,
                   now: float | None = None) -> dict[str, Any]:
    """Static validation, then evaluation when static passed, then the registry entry."""
    static = static_validate(manifest, keyring=keyring, hub=hub, ledger=ledger, taken_names=taken_names, now=now)
    evaluation = None
    if static.verdict is Verdict.ACCEPT:
        evaluation = evaluate(manifest, static, pack=pack, executor=executor, prober=prober, min_pass_rate=min_pass_rate)
    return build_registry_entry(manifest, static, evaluation, now=now)
