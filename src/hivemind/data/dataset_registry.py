# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""S21 dataset registry: more real training data per bee domain, with licences, decontamination and provenance.

The night experts (``scripts/night_pack_build.py``) each train on one public source. This module widens that to 2-4 sources
per domain without letting three failure modes in:

1. **Licence drift.** Every candidate is looked up on the Hugging Face Hub API (``/api/datasets/<id>``); the licence is
   read from ``cardData.license`` and classified against an explicit allowlist that is compatible with redistributing
   derived artefacts under the AGPL-3.0. Missing, unknown, custom (``other``) and non-commercial licences are recorded as
   ``REJECTED`` with the reason, never silently dropped. Revision sha, size and the retrieval timestamp come from the
   same API response, so no metadata in the registry is typed by hand.
2. **Eval contamination.** Rows are removed when their normalised text hashes equal another row (exact dedupe), equal a
   held-out eval prompt, share any word 13-gram with a held-out eval item, or name the same entity (a KOI, a host star)
   as an eval item. N-grams that occur in many eval items are boilerplate (shared prompt scaffolding) and are ignored;
   how many were ignored is reported. Every removal is counted per source and per reason.
3. **Dilution by unverified signal.** Each source carries a curator's judgement of whether its targets are verified
   (human/catalogue gold labels, answers checked by execution) or not (LLM-generated text). The mixture planner keeps the
   unverified share of every domain at or below ``unverified_cap`` (default 0.30), the rule used across the project.
   A domain with no verified source therefore receives no rows and is reported ``BLOCKED``: the cap is never inverted.

Sampling is deterministic: rows are ranked by ``sha256(seed:source:key)`` so the selection does not depend on the order in
which rows were read. The provenance manifest records the sha256 of every shard actually read (a datasets-server page or
a local JSONL file), the plan, the decontamination counts and the realised unverified fraction.

Network access goes through an injectable ``fetch`` callable (default: ``urllib`` with a timeout); nothing here downloads
a full dataset. ``slice_rows`` uses the datasets-server ``/rows`` endpoint, which serves the Hub's current parquet
conversion: that slice cannot be pinned to the recorded revision sha, and the manifest says so.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HUB_API = "https://huggingface.co/api/datasets/"
DATASETS_SERVER = "https://datasets-server.huggingface.co"
USER_AGENT = "hivemind-dataset-registry/1.0"
NGRAM_N = 13
DEFAULT_UNVERIFIED_CAP = 0.30
ACCEPTED, REJECTED = "ACCEPTED", "REJECTED"
PASSED, FAILED, BLOCKED, BUILT = "PASSED", "FAILED", "BLOCKED", "BUILT"

Fetch = Callable[[str], bytes]

# Licence ids (Hub spelling) accepted for training data whose derived artefacts are published under the AGPL-3.0.
# The value is the obligation the licence carries; it is copied into the registry entry.
LICENSE_ALLOWLIST: dict[str, str] = {
    "apache-2.0": "keep the licence and NOTICE with redistributed data",
    "mit": "keep the copyright notice",
    "bsd-2-clause": "keep the copyright notice",
    "bsd-3-clause": "keep the copyright notice",
    "cc0-1.0": "none",
    "pddl": "none",
    "cc-by-4.0": "attribution",
    "cc-by-3.0": "attribution",
    "odc-by": "attribution",
    "cdla-permissive-2.0": "keep the licence text",
    "cc-by-sa-4.0": "attribution; redistributed data stays CC-BY-SA-4.0 (one-way compatible with GPLv3 family)",
    "gpl-3.0": "copyleft, compatible with AGPL-3.0",
    "agpl-3.0": "same licence as the project",
    "lgpl-3.0": "copyleft, compatible with AGPL-3.0",
}
_NON_COMMERCIAL = re.compile(r"(^|-)nc(-|$)|non-?commercial", re.IGNORECASE)
_NO_DERIVATIVES = re.compile(r"(^|-)nd(-|$)", re.IGNORECASE)


@dataclass(frozen=True)
class LicenseDecision:
    """Outcome of classifying one dataset card licence."""

    status: str
    license: str | None
    reason: str
    conditions: str | None = None


def classify_license(raw: object) -> LicenseDecision:
    """Classify a ``cardData.license`` value (a string, a list of strings or missing) against the allowlist.

    A list is accepted only if every licence in it is allowlisted: the card does not say which part carries which licence.
    """
    if raw is None or raw == "" or raw == []:
        return LicenseDecision(REJECTED, None, "no licence in the dataset card")
    items = [raw] if isinstance(raw, str) else list(raw) if isinstance(raw, list) else None
    if items is None or not all(isinstance(x, str) and x.strip() for x in items):
        return LicenseDecision(REJECTED, None, f"unreadable licence field {raw!r}")
    ids = [x.strip().lower() for x in items]
    label = ",".join(ids)
    for lic in ids:
        if _NON_COMMERCIAL.search(lic):
            return LicenseDecision(REJECTED, label, f"{lic} is non-commercial: incompatible with AGPL-3.0 redistribution")
        if _NO_DERIVATIVES.search(lic):
            return LicenseDecision(REJECTED, label, f"{lic} forbids derivatives: a trained adapter is a derivative")
        if lic in ("other", "unknown", "unlicense-unknown"):
            return LicenseDecision(REJECTED, label, f"licence '{lic}' is custom or unknown: needs owner review")
        if lic not in LICENSE_ALLOWLIST:
            return LicenseDecision(REJECTED, label, f"licence '{lic}' is not on the allowlist")
    return LicenseDecision(ACCEPTED, label, "allowlisted", "; ".join(LICENSE_ALLOWLIST[x] for x in dict.fromkeys(ids)))


# ---------------------------------------------------------------- candidates
@dataclass(frozen=True)
class SourceSpec:
    """One hand-curated candidate. Only the id, where to read it and how to map it are typed by hand; metadata is fetched."""

    dataset_id: str
    domain: str
    config: str
    split: str
    mapping: Mapping[str, Any]
    verified_signal: bool
    signal_basis: str
    slice_rows: int = 300

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> SourceSpec:
        missing = [k for k in ("id", "domain", "config", "split", "mapping", "verified_signal", "signal_basis") if k not in d]
        if missing:
            raise ValueError(f"candidate {d.get('id', '?')} lacks {missing}")
        if not isinstance(d["verified_signal"], bool):
            raise TypeError(f"candidate {d['id']}: verified_signal must be a boolean")
        kind = d["mapping"].get("kind")
        if kind not in MAPPERS:
            raise ValueError(f"candidate {d['id']}: unknown mapping kind {kind!r}")
        return cls(str(d["id"]), str(d["domain"]), str(d["config"]), str(d["split"]), dict(d["mapping"]),
                   bool(d["verified_signal"]), str(d["signal_basis"]), int(d.get("slice_rows", 300)))


def load_candidates(path: Path) -> tuple[dict[str, Any], list[SourceSpec]]:
    """Read the candidates file; returns (the raw document, parsed specs). Domains used by a source must be declared."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    specs = [SourceSpec.from_dict(c) for c in doc["candidates"]]
    undeclared = sorted({s.domain for s in specs} - set(doc["domains"]))
    if undeclared:
        raise ValueError(f"domains used but not declared: {undeclared}")
    ids = [s.dataset_id + "/" + s.config + "/" + s.split for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError("a candidate (id, config, split) is listed twice")
    return doc, specs


# ---------------------------------------------------------------- Hub API
def default_fetch(url: str, timeout_s: float = 120.0) -> bytes:
    """HTTP GET with a project user agent. Errors propagate; callers turn them into REJECTED/BLOCKED records."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return bytes(resp.read())


RETRY_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def retrying_fetch(fetch: Fetch, retries: int = 5, base_delay_s: float = 10.0,
                   sleep: Callable[[float], None] = time.sleep) -> Fetch:
    """Wrap ``fetch`` so rate limits (429) and transient server errors are retried with linear backoff.

    A 404 or any other client error is not retried: it is a real answer and is reported as such.
    """
    def wrapped(url: str) -> bytes:
        for attempt in range(retries + 1):
            try:
                return fetch(url)
            except urllib.error.HTTPError as exc:
                if exc.code not in RETRY_HTTP_CODES or attempt == retries:
                    raise
            except (urllib.error.URLError, TimeoutError):
                if attempt == retries:
                    raise
            sleep(base_delay_s * (attempt + 1))
        raise RuntimeError("unreachable: the loop returns or raises")  # pragma: no cover

    return wrapped


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _err(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"[:240]


def registry_entry(spec: SourceSpec, fetch: Fetch, retrieved_utc: str) -> dict[str, Any]:
    """Look one candidate up on the Hub and decide ACCEPTED / REJECTED. Every field but the spec comes from the API."""
    entry: dict[str, Any] = {"id": spec.dataset_id, "domain": spec.domain, "config": spec.config, "split": spec.split,
                             "verified_signal": spec.verified_signal, "signal_basis": spec.signal_basis,
                             "retrieved_utc": retrieved_utc, "metadata_source": HUB_API + spec.dataset_id}
    try:
        meta = json.loads(fetch(HUB_API + urllib.parse.quote(spec.dataset_id, safe="/")))
    except Exception as exc:  # noqa: BLE001 - the Hub answered with an error or not at all: record, do not drop
        return {**entry, "status": REJECTED, "reason": f"Hub API lookup failed ({_err(exc)})"}
    card = meta.get("cardData") or {}
    decision = classify_license(card.get("license"))
    entry.update({"revision_sha": meta.get("sha"), "last_modified": meta.get("lastModified"),
                  "license_as_read": card.get("license"), "license": decision.license,
                  "license_conditions": decision.conditions, "size_categories": card.get("size_categories"),
                  "used_storage_bytes": meta.get("usedStorage"), "gated": meta.get("gated", False)})
    if meta.get("disabled") or meta.get("private"):
        return {**entry, "status": REJECTED, "reason": "dataset is disabled or private"}
    if decision.status != ACCEPTED:
        return {**entry, "status": REJECTED, "reason": decision.reason}
    if entry["gated"]:
        return {**entry, "status": REJECTED, "reason": f"gated ({entry['gated']}): access terms need owner review"}
    return {**entry, "status": ACCEPTED, "reason": decision.reason}


def split_size(spec: SourceSpec, fetch: Fetch) -> dict[str, Any]:
    """Row and byte counts for the configured split from datasets-server ``/size``; an error is returned, not raised."""
    url = f"{DATASETS_SERVER}/size?" + urllib.parse.urlencode({"dataset": spec.dataset_id, "config": spec.config})
    try:
        doc = json.loads(fetch(url))
    except Exception as exc:  # noqa: BLE001 - any failure is reported as BLOCKED, never raised past the registry
        return {"status": BLOCKED, "reason": _err(exc)}
    for s in doc.get("size", {}).get("splits", []):
        if s.get("config") == spec.config and s.get("split") == spec.split:
            return {"status": PASSED, "num_rows": s.get("num_rows"), "num_bytes_parquet_files": s.get("num_bytes_parquet_files")}
    return {"status": BLOCKED, "reason": f"split {spec.config}/{spec.split} not listed by datasets-server /size"}


@dataclass
class Shard:
    """One unit of bytes actually read, with its digest."""

    uri: str
    sha256: str
    n_bytes: int
    n_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {"uri": self.uri, "sha256": self.sha256, "bytes": self.n_bytes, "rows": self.n_rows}


@dataclass
class SliceResult:
    status: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    shards: list[Shard] = field(default_factory=list)
    reason: str = ""


def slice_rows(spec: SourceSpec, fetch: Fetch, n_rows: int | None = None, page: int = 100) -> SliceResult:
    """Stream the first ``n_rows`` rows of the split through datasets-server ``/rows`` (no full download)."""
    want = spec.slice_rows if n_rows is None else n_rows
    out = SliceResult(PASSED)
    offset = 0
    while offset < want:
        length = min(page, want - offset)
        url = f"{DATASETS_SERVER}/rows?" + urllib.parse.urlencode(
            {"dataset": spec.dataset_id, "config": spec.config, "split": spec.split, "offset": offset, "length": length})
        try:
            data = fetch(url)
            rows = [r["row"] for r in json.loads(data)["rows"]]
        except Exception as exc:  # noqa: BLE001 - network, HTTP or payload errors all make the slice BLOCKED
            return SliceResult(BLOCKED, out.rows, out.shards, f"datasets-server /rows at offset {offset}: {_err(exc)}")
        out.shards.append(Shard(url, hashlib.sha256(data).hexdigest(), len(data), len(rows)))
        out.rows.extend(rows)
        if len(rows) < length:
            break
        offset += length
    if not out.rows:
        return SliceResult(BLOCKED, [], out.shards, "datasets-server returned no rows")
    return out


def read_jsonl_shards(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], list[Shard]]:
    """Read local JSONL shards; each file's sha256 is taken over the exact bytes parsed."""
    rows: list[dict[str, Any]] = []
    shards: list[Shard] = []
    for p in paths:
        data = Path(p).read_bytes()
        part = [json.loads(ln) for ln in data.decode("utf-8").split("\n") if ln.strip()]
        rows.extend(part)
        shards.append(Shard(str(p), hashlib.sha256(data).hexdigest(), len(data), len(part)))
    return rows, shards


# ---------------------------------------------------------------- mapping to the night-pack format
class Unmappable(ValueError):
    """The row does not have the fields the mapping needs (a schema problem, counted separately from filters)."""


def _text(row: Mapping[str, Any], name: str) -> str:
    v = row.get(name)
    if not isinstance(v, str) or not v.strip():
        raise Unmappable(f"field {name!r} missing or empty")
    return v.strip()


def fmt_number(v: float) -> str:
    """Plain decimal text, the same convention as the night packs: 6360, 2.48, 0.0123, 1.2e-05."""
    if v == 0:
        return "0"
    if abs(v) >= 1e7:
        return f"{v:.3e}"
    if abs(v) >= 100:
        return f"{v:.0f}"
    return f"{v:.3g}" if abs(v) >= 1e-3 else f"{v:.2e}"


def _passes_filters(row: Mapping[str, Any], mapping: Mapping[str, Any]) -> bool:
    return all(row.get(k) in allowed for k, allowed in (mapping.get("filter_in") or {}).items())


def _excluded(prompt: str, completion: str, mapping: Mapping[str, Any]) -> bool:
    """``exclude_pattern`` is matched against the text that would be trained on (prompt and completion), not unused columns."""
    pat = mapping.get("exclude_pattern")
    return bool(pat and (re.search(pat, prompt, re.IGNORECASE) or re.search(pat, completion, re.IGNORECASE)))


def _map_fields(row: Mapping[str, Any], m: Mapping[str, Any]) -> tuple[str, str]:
    parts = [_text(row, f) for f in m["prompt_fields"][:1]]
    parts += [str(row[f]).strip() for f in m["prompt_fields"][1:] if isinstance(row.get(f), str) and row[f].strip()]
    return "\n\n".join(parts), m.get("completion_prefix", "") + _text(row, m["completion_field"])


def _map_messages(row: Mapping[str, Any], m: Mapping[str, Any]) -> tuple[str, str]:
    msgs = row.get(m["messages_field"])
    if not isinstance(msgs, list) or not msgs:
        raise Unmappable("messages field missing or empty")
    turns = [(t.get("role") or t.get("from"), t.get("content") or t.get("value")) for t in msgs if isinstance(t, dict)]
    users = [c for r, c in turns if r in ("user", "human") and isinstance(c, str) and c.strip()]
    if not turns or turns[-1][0] not in ("assistant", "gpt") or not isinstance(turns[-1][1], str) or not users:
        raise Unmappable("conversation does not end with an assistant reply after a user turn")
    return users[-1].strip(), turns[-1][1].strip()


def _map_mcq(row: Mapping[str, Any], m: Mapping[str, Any]) -> tuple[str, str]:
    q = _text(row, m["question_field"])
    ch = row.get(m["choices_field"])
    if not isinstance(ch, dict) or not isinstance(ch.get("text"), list) or not isinstance(ch.get("label"), list):
        raise Unmappable("choices must be {'text': [...], 'label': [...]}")
    if len(ch["text"]) != len(ch["label"]) or len(ch["label"]) < 2:
        raise Unmappable("choices text/label lengths differ or fewer than two options")
    gold = row.get(m["answer_field"])
    if gold not in ch["label"]:
        raise Unmappable(f"answer {gold!r} is not one of the option labels")
    opts = "\n".join(f"{lab}. {txt}" for lab, txt in zip(ch["label"], ch["text"]))
    return f"{q}\n{opts}", f"Answer: {gold}"


def _missing(v: Any, sentinels: Sequence[Any]) -> bool:
    return v is None or v in sentinels or (isinstance(v, float) and math.isnan(v))


def _map_label(row: Mapping[str, Any], m: Mapping[str, Any]) -> tuple[str, str]:
    sentinels = list(m.get("missing_values", []))
    lines = []
    for col, name in m["features"].items():
        v = row.get(col)
        if _missing(v, sentinels):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float, str)):
            raise Unmappable(f"feature {col!r} has type {type(v).__name__}")
        lines.append(f"- {name}: {fmt_number(float(v)) if isinstance(v, (int, float)) else v}")
    if len(lines) < int(m.get("min_features", 1)):
        raise Unmappable("too few features present")
    raw = row.get(m["label_field"])
    if not isinstance(raw, str):
        raise Unmappable(f"label {m['label_field']!r} missing")
    key = raw.strip()[:1].upper() if m.get("label_first_char") else raw.strip()
    return m["prompt_header"] + "\n" + "\n".join(lines) + "\n" + m["question"], m["label_map"][key]


def _label_filtered(row: Mapping[str, Any], m: Mapping[str, Any]) -> bool:
    """A label outside ``label_map`` (e.g. NO_SIGNAL, or a class the pack does not use) is a filter, not a schema error."""
    if m.get("kind") != "label":
        return False
    raw = row.get(m["label_field"])
    if not isinstance(raw, str) or _missing(raw, m.get("missing_values", [])):
        return True
    key = raw.strip()[:1].upper() if m.get("label_first_char") else raw.strip()
    return key not in m["label_map"]


MAPPERS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any]], tuple[str, str]]] = {
    "fields": _map_fields, "messages": _map_messages, "mcq": _map_mcq, "label": _map_label}


def normalize_entity_key(value: Any, style: str) -> str | None:
    """Canonical entity id used to match train rows with eval keys: ``koi`` turns '5218.01' and 'K05218.01' into one id."""
    if not isinstance(value, str) or not value.strip():
        return None
    v = value.strip()
    if style == "koi":
        mm = re.fullmatch(r"K?0*(\d+)\.(\d{2})", v, re.IGNORECASE)
        return f"K{int(mm.group(1)):05d}.{mm.group(2)}" if mm else None
    if style == "star":
        return "star-" + v
    return v


_KOI_KEY = re.compile(r"K(\d{5})\.\d{2}")


def entity_aliases(key: str) -> set[str]:
    """The ids an entity key matches on. A KOI also matches every other KOI of the same host star (``K05218.01`` and
    ``K05218.02`` share star features): the night exoplanet pack splits whole stars between train and eval, so a sibling
    KOI of an eval item is contamination too."""
    mm = _KOI_KEY.fullmatch(key)
    return {key, f"koi-host-K{mm.group(1)}"} if mm else {key}


def map_row(spec: SourceSpec, row: Mapping[str, Any], system_prompt: str, index: int) -> dict[str, Any] | None:
    """Map one raw row to ``{key, messages, completion}`` (the night-pack train format), or None if filtered by design.

    ``system_prompt`` is the domain's; a mapping may override it when its task differs (e.g. discovery method vs vetting).

    Raises ``Unmappable`` when the row lacks the fields the mapping needs.
    """
    m = spec.mapping
    if not _passes_filters(row, m) or _label_filtered(row, m):
        return None
    prompt, completion = MAPPERS[m["kind"]](row, m)
    if _excluded(prompt, completion, m):
        return None
    raw_key = row.get(m["key_field"]) if m.get("key_field") else None
    key = f"{spec.dataset_id}:{raw_key if raw_key not in (None, '') else index}"
    out: dict[str, Any] = {"key": key, "source": spec.dataset_id, "domain": spec.domain, "verified": spec.verified_signal,
                           "messages": [{"role": "system", "content": m.get("system_prompt", system_prompt)},
                                        {"role": "user", "content": prompt}],
                           "completion": completion}
    ek = m.get("entity_key")
    if ek:
        out["entity_key"] = normalize_entity_key(row.get(ek["field"]), ek.get("style", "raw"))
    return out


def validate_slice(spec: SourceSpec, rows: Sequence[Mapping[str, Any]], system_prompt: str,
                   max_unmappable_fraction: float = 0.10) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Map a streamed slice; PASSED iff at least one row maps and at most ``max_unmappable_fraction`` of unfiltered rows fail."""
    mapped: list[dict[str, Any]] = []
    positions: list[int] = []
    filtered = 0
    errors: dict[str, int] = {}
    for i, r in enumerate(rows):
        try:
            out = map_row(spec, r, system_prompt, i)
        except (Unmappable, KeyError) as exc:
            msg = str(exc)[:120]
            errors[msg] = errors.get(msg, 0) + 1
            continue
        if out is None:
            filtered += 1
        else:
            mapped.append(out)
            positions.append(i)
    # A key column that is not an id (e.g. a task-type field) gives many rows one key. Those rows get their position in
    # the split appended, so keys stay unique and sampling stays independent of read order; the count is reported.
    key_count: dict[str, int] = {}
    for r in mapped:
        key_count[r["key"]] = key_count.get(r["key"], 0) + 1
    rekeyed = 0
    for r, i in zip(mapped, positions):
        if key_count[r["key"]] > 1:
            r["key"] = f"{r['key']}#{i}"
            rekeyed += 1
    considered = len(rows) - filtered
    bad = sum(errors.values())
    ok = bool(mapped) and bad <= max_unmappable_fraction * considered
    report = {"status": PASSED if ok else FAILED, "rows_read": len(rows), "rows_mapped": len(mapped),
              "rows_filtered_by_design": filtered, "rows_unmappable": bad,
              "rows_rekeyed_duplicate_key": rekeyed,
              "unmappable_reasons": dict(sorted(errors.items(), key=lambda kv: -kv[1])[:5])}
    if not ok:
        report["reason"] = "no row mapped" if not mapped else f"{bad}/{considered} unfiltered rows unmappable"
    return report, mapped


# ---------------------------------------------------------------- decontamination
def normalize_tokens(text: str) -> list[str]:
    """NFKC, case-folded, alphanumeric word tokens: punctuation and whitespace differences do not hide a duplicate."""
    folded = unicodedata.normalize("NFKC", unicodedata.normalize("NFKC", text).upper().casefold())  # upper first: 'ı' and 'i' meet
    return re.findall(r"[^\W_]+", folded)


def text_hash(text: str) -> str:
    return hashlib.sha256(" ".join(normalize_tokens(text)).encode()).hexdigest()


def ngram_hashes(tokens: Sequence[str], n: int = NGRAM_N) -> set[int]:
    """8-byte hashes of every word n-gram (empty when the text is shorter than n words)."""
    return {int.from_bytes(hashlib.blake2b(" ".join(tokens[i:i + n]).encode(), digest_size=8).digest(), "big")
            for i in range(len(tokens) - n + 1)}


def _user_text(row: Mapping[str, Any]) -> str:
    return "\n".join(m["content"] for m in row.get("messages", []) if m.get("role") == "user")


@dataclass
class EvalIndex:
    """Held-out eval items to decontaminate against: prompt hashes, n-gram document frequencies and entity keys."""

    n: int = NGRAM_N
    max_doc_freq: int = 10
    exact: set[str] = field(default_factory=set)
    ngram_df: dict[int, int] = field(default_factory=dict)
    keys: set[str] = field(default_factory=set)
    sources: dict[str, int] = field(default_factory=dict)

    def add(self, name: str, texts: Iterable[str], keys: Iterable[str] = ()) -> None:
        count = 0
        for t in texts:
            toks = normalize_tokens(t)
            if not toks:
                continue
            count += 1
            self.exact.add(hashlib.sha256(" ".join(toks).encode()).hexdigest())
            for h in ngram_hashes(toks, self.n):
                self.ngram_df[h] = self.ngram_df.get(h, 0) + 1
        self.keys.update(a for k in keys if k for a in entity_aliases(k))
        self.sources[name] = self.sources.get(name, 0) + count

    def add_eval_pack(self, pack_dir: Path) -> None:
        """A night pack's ``eval.jsonl``: the user turns (system prompts are shared scaffolding) and the item keys."""
        rows, _ = read_jsonl_shards([Path(pack_dir) / "eval.jsonl"])
        self.add(f"night_pack:{Path(pack_dir).name}", (_user_text(r) for r in rows), (str(r.get("key", "")) for r in rows))

    @property
    def active_ngrams(self) -> set[int]:
        return {h for h, df in self.ngram_df.items() if df <= self.max_doc_freq}

    @property
    def boilerplate_ngrams(self) -> int:
        return sum(1 for df in self.ngram_df.values() if df > self.max_doc_freq)


def decontaminate(rows: Sequence[dict[str, Any]], index: EvalIndex,
                  seen: set[str] | None = None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Drop exact duplicates (across everything passed through ``seen``) and rows overlapping the eval index.

    Order of checks, first hit wins: entity key, exact eval prompt, 13-gram overlap, exact duplicate of an earlier row.
    """
    seen = set() if seen is None else seen
    active = index.active_ngrams
    counts = {"eval_entity_key": 0, "eval_exact": 0, "eval_ngram": 0, "exact_duplicate": 0, "kept": 0}
    kept: list[dict[str, Any]] = []
    for r in rows:
        user = _user_text(r)
        toks_user = normalize_tokens(user)
        toks_all = normalize_tokens(user + "\n" + str(r.get("completion", "")))
        if r.get("entity_key") and entity_aliases(r["entity_key"]) & index.keys:
            counts["eval_entity_key"] += 1
        elif hashlib.sha256(" ".join(toks_user).encode()).hexdigest() in index.exact:
            counts["eval_exact"] += 1
        elif ngram_hashes(toks_all, index.n) & active:
            counts["eval_ngram"] += 1
        else:
            h = hashlib.sha256(" ".join(toks_all).encode()).hexdigest()
            if h in seen:
                counts["exact_duplicate"] += 1
                continue
            seen.add(h)
            kept.append(r)
    counts["kept"] = len(kept)
    return kept, counts


# ---------------------------------------------------------------- mixture
@dataclass(frozen=True)
class MixtureConfig:
    total_rows: int
    domain_weights: Mapping[str, float]
    per_source_cap: float = 0.5
    unverified_cap: float = DEFAULT_UNVERIFIED_CAP
    seed: int = 20261028

    def __post_init__(self) -> None:
        if self.total_rows <= 0:
            raise ValueError("total_rows must be positive")
        if not self.domain_weights or any(w < 0 for w in self.domain_weights.values()) or sum(self.domain_weights.values()) <= 0:
            raise ValueError("domain_weights must be non-negative with a positive sum")
        if not 0 < self.per_source_cap <= 1:
            raise ValueError("per_source_cap must be in (0, 1]")
        if not 0 <= self.unverified_cap < 1:
            raise ValueError("unverified_cap must be in [0, 1)")


def _fill(targets: dict[str, int], avail: Mapping[str, int], cap: int, budget: int) -> dict[str, int]:
    """Water-fill ``budget`` rows over sources, each limited by its availability and the per-source cap."""
    got = dict(targets)
    while True:  # every round adds at least one row or stops, so this terminates
        room = {s: min(avail[s], cap) - got[s] for s in avail if min(avail[s], cap) > got[s]}
        left = budget - sum(got.values())
        if left <= 0 or not room:
            return got
        share = max(1, left // len(room))
        for s in sorted(room):
            got[s] += max(0, min(room[s], share, budget - sum(got.values())))


def plan_mixture(available: Mapping[str, tuple[str, bool, int]], cfg: MixtureConfig) -> dict[str, Any]:
    """Allocate rows per source. ``available`` maps source -> (domain, verified, rows available after decontamination).

    Per domain: target = round(weight share * total); every source is capped at ``per_source_cap`` of the target;
    verified rows are filled first, then unverified rows up to ``floor(cap / (1 - cap) * verified)`` so the unverified
    share never exceeds ``unverified_cap``. The shortfall against the target is reported, never backfilled silently.
    """
    wsum = sum(cfg.domain_weights.values())
    alloc: dict[str, int] = {}
    domains: dict[str, Any] = {}
    for dom, w in sorted(cfg.domain_weights.items()):
        target = round(cfg.total_rows * w / wsum)
        cap = max(1, math.floor(cfg.per_source_cap * target))
        ver = {s: a[2] for s, a in available.items() if a[0] == dom and a[1]}
        unv = {s: a[2] for s, a in available.items() if a[0] == dom and not a[1]}
        got_v = _fill({s: 0 for s in ver}, ver, cap, target)
        n_v = sum(got_v.values())
        unv_allowed = min(target - n_v, math.floor(cfg.unverified_cap / (1 - cfg.unverified_cap) * n_v + 1e-9))
        got_u = _fill({s: 0 for s in unv}, unv, cap, unv_allowed)
        n_u = sum(got_u.values())
        alloc.update(got_v)
        alloc.update(got_u)
        status = BUILT if n_v + n_u == target else BLOCKED if n_v == 0 else "SHORT"
        info: dict[str, Any] = {"target": target, "per_source_cap": cap, "verified_rows": n_v, "unverified_rows": n_u,
                                "unverified_fraction": (n_u / (n_v + n_u)) if n_v + n_u else None, "status": status}
        if status == BLOCKED:
            info["reason"] = "no verified rows: the unverified cap admits none"
        elif status == "SHORT":
            info["reason"] = f"{target - n_v - n_u} rows short of target (availability, source cap or unverified cap)"
        domains[dom] = info
    tot = sum(alloc.values())
    tot_u = sum(n for s, n in alloc.items() if not available[s][1])
    return {"allocation": dict(sorted(alloc.items())), "domains": domains, "total_rows": tot,
            "unverified_fraction": (tot_u / tot) if tot else None, "config": {
                "total_rows": cfg.total_rows, "domain_weights": dict(cfg.domain_weights), "per_source_cap": cfg.per_source_cap,
                "unverified_cap": cfg.unverified_cap, "seed": cfg.seed}}


def sample_rows(rows: Sequence[dict[str, Any]], k: int, seed: int, source: str) -> list[dict[str, Any]]:
    """Deterministic, order-independent sample: rank by sha256(seed:source:key) and keep the first ``k``.

    Ties (two rows with one key) are broken by the row's canonical JSON, so the result never depends on input order."""
    def rank(r: dict[str, Any]) -> tuple[str, str]:
        return hashlib.sha256(f"{seed}:{source}:{r['key']}".encode()).hexdigest(), json.dumps(r, sort_keys=True)

    return sorted(rows, key=rank)[:max(0, k)]


def build_mixture(rows_by_source: Mapping[str, Sequence[dict[str, Any]]], verified: Mapping[str, bool],
                  domain_of: Mapping[str, str], cfg: MixtureConfig, index: EvalIndex,
                  shards: Sequence[Shard]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Decontaminate (verified sources first, so a cross-source duplicate survives in its verified copy), plan and sample.

    Returns (mixture rows, provenance manifest).
    """
    seen: set[str] = set()
    clean: dict[str, list[dict[str, Any]]] = {}
    decon: dict[str, dict[str, int]] = {}
    for src in sorted(rows_by_source, key=lambda x: (not verified[x], x)):  # a duplicate is kept in the verified copy
        clean[src], decon[src] = decontaminate(rows_by_source[src], index, seen)
    plan = plan_mixture({s: (domain_of[s], verified[s], len(clean[s])) for s in clean}, cfg)
    mix: list[dict[str, Any]] = []
    for src in sorted(clean):
        mix.extend(sample_rows(clean[src], plan["allocation"].get(src, 0), cfg.seed, src))
    manifest = {"plan": plan, "decontamination": {"per_source": decon, "ngram_n": index.n,
                                                  "eval_sources": dict(index.sources), "eval_items_hashed": len(index.exact),
                                                  "eval_entity_keys": len(index.keys), "max_eval_doc_freq": index.max_doc_freq,
                                                  "boilerplate_ngrams_ignored": index.boilerplate_ngrams},
                "shards": [s.to_dict() for s in shards], "rows": len(mix),
                "mixture_sha256": hashlib.sha256("".join(json.dumps(r, sort_keys=True) for r in mix).encode()).hexdigest()}
    return mix, manifest
