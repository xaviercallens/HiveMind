"""The node endpoint logs request lines at DEBUG through the logging module, not to stderr and not nowhere."""

from __future__ import annotations

import logging

import pytest

from hivemind.boinc.real_client.node_endpoint import log_request


def test_request_line_is_logged_at_debug_with_client_and_formatted_text(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG, logger="hivemind.boinc.real_client.node_endpoint"):
        log_request("127.0.0.1", '"%s" %s', "POST /v1/chat/completions", "200")
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "127.0.0.1" in caplog.text and '"POST /v1/chat/completions" 200' in caplog.text


def test_nothing_is_recorded_above_debug_and_a_message_without_args_is_kept_verbatim(caplog: pytest.LogCaptureFixture) -> None:
    """Negative control: at INFO the line is silent; a literal percent sign without args must not raise."""
    with caplog.at_level(logging.INFO, logger="hivemind.boinc.real_client.node_endpoint"):
        log_request("127.0.0.1", "100% served")
    assert caplog.records == []
    with caplog.at_level(logging.DEBUG, logger="hivemind.boinc.real_client.node_endpoint"):
        log_request("10.0.0.2", "100% served")
    assert "100% served" in caplog.text
