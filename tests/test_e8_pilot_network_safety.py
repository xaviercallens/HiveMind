# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The E8 pilot talks to other machines. Its syncer has no authentication and its payloads come off the network: neither may be open to the world or able to run code."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import run_experiment_E8_pilot as pilot  # noqa: E402

RAN: list[str] = []


def _marker() -> None:
    RAN.append("code ran")


class _Evil:
    def __reduce__(self):  # noqa: ANN204 - what a malicious peer would put in a pickle
        return (_marker, ())


def test_tensors_roundtrip() -> None:
    d = {"a": torch.arange(6, dtype=torch.float32).reshape(2, 3), "b": torch.ones(2)}
    back = pilot.b64_to_tensors(pilot.tensors_to_b64(d))
    assert set(back) == {"a", "b"} and torch.equal(back["a"], d["a"]) and torch.equal(back["b"], d["b"])


def test_a_payload_that_carries_code_is_refused_and_never_runs() -> None:
    buf = io.BytesIO()
    torch.save({"w": _Evil()}, buf)
    RAN.clear()
    with pytest.raises(Exception):  # noqa: B017, PT011 - torch raises its UnpicklingError subclass; the point is that it raises
        pilot.b64_to_tensors(base64.b64encode(buf.getvalue()).decode("ascii"))
    assert RAN == []


def _run_syncer(monkeypatch: pytest.MonkeyPatch, *extra: str) -> list[tuple[str, int]]:
    seen: list[tuple[str, int]] = []

    class Stop(Exception):
        pass

    class FakeServer:
        def __init__(self, addr: tuple[str, int], handler: object) -> None:
            seen.append(addr)
            raise Stop

    monkeypatch.setattr(pilot, "ThreadingHTTPServer", FakeServer)
    monkeypatch.setattr(sys, "argv", ["run_experiment_E8_pilot.py", "syncer", "--port", "0", *extra])
    with pytest.raises(Stop):
        pilot.main()
    return seen


def test_the_syncer_listens_on_loopback_unless_told_an_address(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_syncer(monkeypatch) == [("127.0.0.1", 0)]
    assert _run_syncer(monkeypatch, "--bind", "10.77.0.4") == [("10.77.0.4", 0)]


@pytest.mark.parametrize("wide", ["0.0.0.0", "::", "[::]", ""])
def test_the_syncer_refuses_to_listen_on_every_interface(monkeypatch: pytest.MonkeyPatch, wide: str) -> None:
    monkeypatch.setattr(sys, "argv", ["run_experiment_E8_pilot.py", "syncer", "--bind", wide])
    with pytest.raises(SystemExit) as exc:
        pilot.main()
    assert exc.value.code == 2
