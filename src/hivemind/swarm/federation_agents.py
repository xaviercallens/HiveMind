# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Signed *task agents* for federated co-training over AMCP, in the spirit of IBM Aglets without their danger.

An Aglet was a Java object that moved, code and state, from host to host and ran there. Moving executable code between untrusted machines is exactly what must NOT happen here.
What survives of the idea is the life cycle, carried by data instead of code:

  dispatch -> ``task_offer``    the coordinator sends a signed task description (round, hash of the model state, hash of the code version, parameters, deadline)
  arrival  -> ``check_offer``   the host verifies it BEFORE running anything: the code pin must equal the hash of the code the host already has from the repository,
                                the base model hash must match, the deadline must not have passed, the global state must hash to what the offer says
  clone    -> several learners   claim and run the same offer independently (``task_claim``)
  result   -> ``task_result``    the delta travels back, never the code; the coordinator checks the run, the round, the global state the learner started from, the code pin,
                                the allowed learner list and the shape and size of the delta (``check_result``) before it can enter an aggregate
  retract  -> ``task_abort``     a round that did not reach its quorum by the deadline is withdrawn and offered again

Everything is a signed AMCP envelope (HMAC, TTL, size cap of 64 KiB), so a participant needs only an OUTBOUND connection to the bus. Weights never travel in a message: the model state
is identified by a hash and rebuilt locally, and only a LoRA state small enough for the cap travels. Larger adapters need chunking or an object store (not built).
"""

from __future__ import annotations

import base64
import hashlib
import io
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from hivemind.swarm.amcp_kafka_campaign_bus import AMCPKafkaEvent, build_amcp_kafka_event
from hivemind.swarm.amcp_kafka_transport import MAX_ENVELOPE_BYTES

ALGO = "e8_qlora_moe_v1"
OFFER = "io.hivemind.amcp.fed.task_offer"
CLAIM = "io.hivemind.amcp.fed.task_claim"
RESULT = "io.hivemind.amcp.fed.task_result"
ABORT = "io.hivemind.amcp.fed.task_abort"
_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
ENVELOPE_HEADROOM = 4096  # the signed envelope adds fields around the payload


class FederationError(Exception):
    """Base class: a message was refused. The text says why, so a host operator can act on it."""


class OfferRefused(FederationError):
    pass


class ResultRejected(FederationError):
    pass


# ------------------------------------------------------------------------------------------------------------------------------ hashes and pins


def code_pin(paths: Sequence[Path | str]) -> str:
    """One hash over the files that implement the round, in a fixed order. A host runs a task only if its own pin equals the offer's.

    Line endings are normalised first: a Windows checkout (CRLF) of the same commit must pin the same as a Linux one, or every task would be refused for a reason nobody can see.
    """
    h = hashlib.sha256()
    for p in paths:
        p = Path(p)
        h.update(p.name.encode() + b"\0" + p.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def tensors_sha256(state: Mapping[str, Any]) -> str:
    """Canonical hash of a dict of tensors (names, shapes, dtypes, bytes), independent of insertion order."""
    h = hashlib.sha256()
    for k in sorted(state):
        t = state[k].detach().cpu().contiguous()
        h.update(k.encode() + b"\0" + str(tuple(t.shape)).encode() + b"\0" + str(t.dtype).encode() + b"\0" + t.numpy().tobytes())
    return h.hexdigest()


def encode_tensors(state: Mapping[str, Any]) -> str:
    import torch

    buf = io.BytesIO()
    torch.save({k: v.detach().cpu().contiguous() for k, v in state.items()}, buf)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_tensors(b64: str) -> dict[str, Any]:
    """Weights-only: a payload can carry tensors and plain containers, never code."""
    import torch

    raw = base64.b64decode(b64, validate=True)  # strict: anything that is not base64 is refused before torch sees it
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


# ------------------------------------------------------------------------------------------------------------------------------ events


def _check_size(event: AMCPKafkaEvent) -> AMCPKafkaEvent:
    import json

    size = len(json.dumps(event.payload, separators=(",", ":")).encode())
    if size > MAX_ENVELOPE_BYTES - ENVELOPE_HEADROOM:
        raise FederationError(f"payload of {size} bytes does not fit one AMCP envelope ({MAX_ENVELOPE_BYTES} bytes): this adapter is too large to send directly (needs chunking or an object store)")
    return event


def build_offer(*, run_id: str, rnd: int, sender_id: str, global_state: Mapping[str, Any], base_sha256: str, pin: str, inner_steps: int, deadline_unix: float,
                quorum: int, params: Mapping[str, Any] | None = None) -> AMCPKafkaEvent:
    if not _ID_RE.match(run_id):
        raise FederationError("run_id must be 1-64 characters of letters, digits . _ -")
    payload = {"run_id": run_id, "round": rnd, "algo": ALGO, "script_sha256": pin, "base_sha256": base_sha256, "global_sha256": tensors_sha256(global_state),
               "inner_steps": inner_steps, "deadline_unix": deadline_unix, "quorum": quorum, "params": dict(params or {}), "global_b64": encode_tensors(global_state)}
    return _check_size(build_amcp_kafka_event(topic_key="federation", event_type=OFFER, sender_id=sender_id, sender_kind="ai_agent", subject_id=run_id, payload=payload))


def build_claim(*, run_id: str, rnd: int, learner_id: str, host: str) -> AMCPKafkaEvent:
    return build_amcp_kafka_event(topic_key="federation", event_type=CLAIM, sender_id=learner_id, sender_kind="gpu_producer", subject_id=run_id,
                                  payload={"run_id": run_id, "round": rnd, "learner_id": learner_id, "host": host})


def build_result(*, run_id: str, rnd: int, learner_id: str, host: str, tokens: int, delta: Mapping[str, Any], delta_l2: float, eval_loss_global: float, pin: str,
                 global_sha256: str, vram_peak_mib: float | None = None) -> AMCPKafkaEvent:
    payload = {"run_id": run_id, "round": rnd, "learner_id": learner_id, "host": host, "tokens": tokens, "delta_l2": delta_l2, "eval_loss_global": eval_loss_global,
               "script_sha256": pin, "global_sha256": global_sha256, "delta_b64": encode_tensors(delta)}
    if vram_peak_mib is not None:
        payload["vram_peak_mib"] = float(vram_peak_mib)
    return _check_size(build_amcp_kafka_event(topic_key="federation", event_type=RESULT, sender_id=learner_id, sender_kind="gpu_producer", subject_id=run_id, payload=payload))


def build_abort(*, run_id: str, rnd: int, sender_id: str, reason: str) -> AMCPKafkaEvent:
    return build_amcp_kafka_event(topic_key="federation", event_type=ABORT, sender_id=sender_id, sender_kind="ai_agent", subject_id=run_id,
                                  payload={"run_id": run_id, "round": rnd, "reason": reason[:128]})


# ------------------------------------------------------------------------------------------------------------------------------ checks


def check_offer(payload: Mapping[str, Any], *, my_pin: str, my_base_sha256: str, now: float | None = None, run_id: str | None = None) -> dict[str, Any]:
    """Learner side. Verify BEFORE running anything; returns the decoded global state, or raises ``OfferRefused`` with the reason."""
    now = time.time() if now is None else now
    if run_id is not None and payload.get("run_id") != run_id:
        raise OfferRefused(f"offer is for run {payload.get('run_id')!r}, this host joined {run_id!r}")
    if payload.get("algo") != ALGO:
        raise OfferRefused(f"unknown algorithm {payload.get('algo')!r}")
    if payload.get("script_sha256") != my_pin:
        raise OfferRefused("the offer pins a different code version than the one on this host: update the checkout (code never travels in a message)")
    if payload.get("base_sha256") != my_base_sha256:
        raise OfferRefused("the base model on this host hashes differently from the offer's: refusing to train a different model")
    if float(payload.get("deadline_unix", 0)) <= now:
        raise OfferRefused("the offer's deadline has passed")
    state = decode_tensors(str(payload.get("global_b64", "")))
    if tensors_sha256(state) != payload.get("global_sha256"):
        raise OfferRefused("the global state does not hash to what the offer says")
    return state


def check_delta(delta: Mapping[str, Any], template: Mapping[str, Any], max_l2: float) -> float:
    """Same names and shapes as the global state, all values finite, and a bounded size. Returns the delta's L2 norm."""
    if set(delta) != set(template):
        raise ResultRejected("delta names differ from the global state")
    total = 0.0
    for k, v in delta.items():
        if tuple(v.shape) != tuple(template[k].shape):
            raise ResultRejected(f"delta {k!r} has shape {tuple(v.shape)}, expected {tuple(template[k].shape)}")
        f = v.float()
        if not bool(f.isfinite().all()):
            raise ResultRejected(f"delta {k!r} contains NaN or infinity")
        total += float((f * f).sum())
    norm = math.sqrt(total)
    if norm > max_l2:
        raise ResultRejected(f"delta norm {norm:.3g} exceeds the bound {max_l2:.3g}")
    return norm


def check_result(payload: Mapping[str, Any], *, run_id: str, rnd: int, global_sha256: str, pin: str, allowed: Sequence[str]) -> None:
    """Coordinator side, on the signed payload (the delta itself is checked with ``check_delta``)."""
    if payload.get("run_id") != run_id:
        raise ResultRejected("result is for another run")
    if int(payload.get("round", -1)) != rnd:
        raise ResultRejected(f"result is for round {payload.get('round')}, the open round is {rnd}")
    if payload.get("learner_id") not in allowed:
        raise ResultRejected(f"learner {payload.get('learner_id')!r} is not on the allowed list")
    if payload.get("global_sha256") != global_sha256:
        raise ResultRejected("the learner trained from a different global state than the one offered")
    if payload.get("script_sha256") != pin:
        raise ResultRejected("the learner ran a different code version")
    if not (float(payload.get("tokens", 0)) > 0):
        raise ResultRejected("no tokens trained")


# ------------------------------------------------------------------------------------------------------------------------------ round bookkeeping


@dataclass
class RoundBook:
    """Which results of one round are accepted, and when the round may close: when every allowed learner answered, or at the deadline with at least ``quorum`` results."""

    run_id: str
    rnd: int
    allowed: Sequence[str]
    quorum: int
    deadline_unix: float
    global_sha256: str
    pin: str
    max_delta_l2: float
    template: Mapping[str, Any]
    accepted: dict[str, dict[str, Any]] = field(default_factory=dict)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    claims: set[str] = field(default_factory=set)

    def claim(self, payload: Mapping[str, Any]) -> None:
        if payload.get("run_id") == self.run_id and int(payload.get("round", -1)) == self.rnd and payload.get("learner_id") in self.allowed:
            self.claims.add(str(payload["learner_id"]))

    def offer_result(self, payload: Mapping[str, Any]) -> bool:
        """Check and keep one result. A rejection is recorded with its reason and never raises, so one bad participant cannot stop a round."""
        who = str(payload.get("learner_id", "?"))
        try:
            check_result(payload, run_id=self.run_id, rnd=self.rnd, global_sha256=self.global_sha256, pin=self.pin, allowed=self.allowed)
            if who in self.accepted:
                raise ResultRejected("duplicate result for this round")
            delta = decode_tensors(str(payload.get("delta_b64", "")))
            norm = check_delta(delta, self.template, self.max_delta_l2)
        except FederationError as exc:
            self.rejected.append((who, str(exc)))
            return False
        except Exception as exc:  # noqa: BLE001 - a malformed payload is a rejection, not a crash
            self.rejected.append((who, f"unreadable result: {type(exc).__name__}"))
            return False
        self.accepted[who] = {"delta": delta, "tokens": int(payload["tokens"]), "eval_loss_global": float(payload["eval_loss_global"]), "delta_l2": norm,
                              "host": str(payload.get("host", "?")), "vram_peak_mib": payload.get("vram_peak_mib"), "t": time.time()}
        return True

    def status(self, now: float | None = None) -> str:
        """``open``, ``complete`` (everyone answered), ``quorum`` (deadline passed, enough results) or ``failed`` (deadline passed, too few)."""
        now = time.time() if now is None else now
        if len(self.accepted) >= len(self.allowed):
            return "complete"
        if now < self.deadline_unix:
            return "open"
        return "quorum" if len(self.accepted) >= self.quorum else "failed"
