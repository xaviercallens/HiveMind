"""The owner-order guardrail must import without torch: the PR security scanner runs it in a CI without the GPU runtime."""

from __future__ import annotations

import subprocess
import sys


def test_owner_order_guardrail_import_does_not_pull_in_torch() -> None:
    code = (
        "import sys; import hivemind.security.owner_order_guardrail as g; "
        "print('torch' in sys.modules, g.AUTHORIZED_OWNER_EMAIL)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.split()
    assert out[0] == "False"  # torch was never imported
    assert out[1] == "callensxavier@gmail.com"  # and the guardrail is the real one


def test_exception_class_is_shared_not_duplicated() -> None:
    from hivemind.security import SwarmSecurityViolationError as from_package
    from hivemind.security._exceptions import SwarmSecurityViolationError as from_exceptions
    from hivemind.security.owner_order_guardrail import SwarmSecurityViolationError as from_guardrail
    from hivemind.security.swarm_runtime_hardening import SwarmSecurityViolationError as from_hardening

    assert from_exceptions is from_guardrail is from_hardening is from_package
    err = from_exceptions("OWN-04", "UNAUTHORIZED", "x")
    assert err.rule_id == "OWN-04" and str(err) == "[OWN-04:UNAUTHORIZED] x"
