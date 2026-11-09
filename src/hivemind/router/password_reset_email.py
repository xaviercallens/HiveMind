# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Password-reset e-mail delivery (v3.2).

Delivery uses SMTP with TLS only: implicit TLS on port 465, STARTTLS on any other port (refused if the server does not
offer it). It is configured by the same variables as the welcome e-mail:

    HIVEMIND_SMTP_HOST, HIVEMIND_SMTP_PORT (default 587), HIVEMIND_SMTP_USER, HIVEMIND_SMTP_PASSWORD, HIVEMIND_SMTP_FROM

and ``HIVEMIND_PUBLIC_WEB_URL`` (the https origin of the web portal that serves ``/auth/reset``). When any of host,
from-address or web URL is missing, :func:`reset_email_configured` is false and nothing is sent: the portal then says that
reset by e-mail is not available, instead of pretending a message left.

The reset token travels in the URL *fragment* (``#token=...``), so it never reaches a server log or a Referer header.
"""
from __future__ import annotations

import os
import smtplib
import ssl
from collections.abc import Mapping
from email.message import EmailMessage

SUBJECT = "Reset your HiveMind password"


def _env(env: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if env is None else env


def reset_email_configured(env: Mapping[str, str] | None = None) -> bool:
    e = _env(env)
    web = e.get("HIVEMIND_PUBLIC_WEB_URL", "").strip()
    return bool(e.get("HIVEMIND_SMTP_HOST", "").strip() and e.get("HIVEMIND_SMTP_FROM", "").strip() and web.startswith("https://"))


def reset_link(token: str, env: Mapping[str, str] | None = None) -> str:
    if not isinstance(token, str) or not token.strip():
        raise ValueError("token must be a non-empty string")
    base = _env(env).get("HIVEMIND_PUBLIC_WEB_URL", "").strip().rstrip("/")
    return f"{base}/auth/reset#token={token.strip()}"


def build_reset_message(recipient: str, link: str, sender: str) -> EmailMessage:
    if not isinstance(recipient, str) or "@" not in recipient.strip():
        raise ValueError("recipient must be a valid email address")
    if not isinstance(link, str) or not link.strip().startswith(("https://", "http://", "/")):
        raise ValueError("link must be a non-empty URL or path")
    if not isinstance(sender, str) or "@" not in sender.strip():
        raise ValueError("sender must be a valid email address")
    msg = EmailMessage()
    msg["Subject"] = SUBJECT
    msg["From"] = sender.strip()
    msg["To"] = recipient.strip()
    msg.set_content(
        "Someone asked to reset the password of the HiveMind account that uses this e-mail address.\n\n"
        f"To choose a new password, open this link within 30 minutes:\n{link.strip()}\n\n"
        "The link works once. Using it signs out every session and replaces your API key.\n"
        "If you did not ask for this, ignore this message: your password stays as it is.\n"
    )
    return msg


def send_reset_email(recipient: str, token: str, env: Mapping[str, str] | None = None) -> bool:
    """Send the reset link to ``recipient``. Returns True when the SMTP server accepted the message. Never raises and
    never logs the token or the link."""
    e = _env(env)
    if (
        not reset_email_configured(e)
        or not isinstance(recipient, str)
        or "@" not in recipient
        or not isinstance(token, str)
        or not token.strip()
    ):
        return False
    host = e["HIVEMIND_SMTP_HOST"].strip()
    port = int(e.get("HIVEMIND_SMTP_PORT", "587") or "587")
    user = e.get("HIVEMIND_SMTP_USER", "").strip()
    password = e.get("HIVEMIND_SMTP_PASSWORD", "")
    msg = build_reset_message(recipient.strip(), reset_link(token, e), e["HIVEMIND_SMTP_FROM"].strip())
    context = ssl.create_default_context()
    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=10, context=context) as server:
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=10) as server:
                server.ehlo()
                if not server.has_extn("starttls"):
                    return False  # never send a reset link in clear text
                server.starttls(context=context)
                server.ehlo()
                if user and password:
                    server.login(user, password)
                server.send_message(msg)
        return True
    except Exception:
        return False
