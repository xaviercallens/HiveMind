#!/usr/bin/env python3
"""HiveMind connector: register, check your connection and print ready-to-paste settings for your editor.

Only dependency: ``pip install httpx``. It prints settings with a key placeholder by default, so nothing secret is
written anywhere unless you ask (--embed-key, --save-key).

    python hivemind_connect.py register --handle my-name            # shows your API key once
    export HIVEMIND_API_KEY=hm-...
    python hivemind_connect.py check                                # balance, live models, catalog
    python hivemind_connect.py snippet continue --model coder       # paste into Continue's config.yaml
    python hivemind_connect.py snippet --list

Snippets target open standards: the OpenAI chat-completions protocol for the tools, plain JSON/YAML for the config.
Each editor changes its settings format over time: verify a snippet against that tool's current documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

DEFAULT_GATEWAY: str = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
GATEWAY_ENV: str = "HIVEMIND_GATEWAY"
API_KEY_ENV: str = "HIVEMIND_API_KEY"
KEY_PLACEHOLDER: str = "hm-YOUR_KEY"
DEFAULT_MODEL: str = "coder"
DEFAULT_MAX_TOKENS: int = 2048
HTTP_TIMEOUT_S: float = 30.0
TOOLS: tuple[str, ...] = ("continue", "opencode", "aider", "cline", "zed", "jetbrains", "python", "env")


def public_model_id(model: str) -> str:
    """``coder`` and ``hivemind/coder`` both mean the gateway model id ``hivemind/coder`` (``auto`` stays ``hivemind/auto``)."""
    return model if model.startswith("hivemind/") else f"hivemind/{model}"


def render_snippet(tool: str, *, gateway: str, model: str, api_key: str = KEY_PLACEHOLDER, max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Settings text for ``tool`` pointing at ``gateway``; pure function, no I/O."""
    base = gateway.rstrip("/") + "/v1"
    mid = public_model_id(model)
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if tool == "continue":
        return (
            "# Continue (VS Code / JetBrains): add to config.yaml\n"
            "models:\n"
            f"  - name: HiveMind {model}\n"
            "    provider: openai\n"
            f"    model: {mid}\n"
            f"    apiBase: {base}\n"
            f"    apiKey: {api_key}\n"
            "    roles: [chat, edit, apply]\n"
            "    defaultCompletionOptions:\n"
            f"      maxTokens: {max_tokens}\n"
        )
    if tool == "opencode":
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {"hivemind": {
                "npm": "@ai-sdk/openai-compatible", "name": "HiveMind OpenRouter",
                "options": {"baseURL": base, "apiKey": api_key},
                "models": {mid: {"name": f"HiveMind {model}"}},
            }},
        }
        return "// OpenCode: opencode.json\n" + json.dumps(config, indent=2) + "\n"
    if tool == "aider":
        return (
            "# Aider: environment, then run aider\n"
            f"export OPENAI_API_BASE={base}\n"
            f"export OPENAI_API_KEY={api_key}\n"
            f"aider --model openai/{mid}\n"
        )
    if tool == "cline":
        return (
            "Cline (VS Code extension): open its settings and choose\n"
            "  API Provider : OpenAI Compatible\n"
            f"  Base URL     : {base}\n"
            f"  API Key      : {api_key}\n"
            f"  Model ID     : {mid}\n"
            f"  (set the maximum output tokens to {max_tokens} if the tool offers it)\n"
        )
    if tool == "zed":
        config = {"language_models": {"openai_compatible": {"HiveMind": {
            "api_url": base,
            "available_models": [{"name": mid, "display_name": f"HiveMind {model}", "max_tokens": 32768, "max_output_tokens": max_tokens}],
        }}}}
        return ("// Zed: settings.json (the key is entered in Zed's UI or via the HIVEMIND_API_KEY environment variable)\n"
                + json.dumps(config, indent=2) + "\n")
    if tool == "jetbrains":
        return (
            "PyCharm / IntelliJ: install the Continue plugin and use the `continue` snippet, or, in Settings > Tools > AI\n"
            "Assistant > Models, add an OpenAI-compatible provider with\n"
            f"  URL   : {base}\n"
            f"  Key   : {api_key}\n"
            f"  Model : {mid}\n"
        )
    if tool == "python":
        return (
            "# pip install openai\n"
            "from openai import OpenAI\n\n"
            f'client = OpenAI(base_url="{base}", api_key="{api_key}")\n'
            "reply = client.chat.completions.create(\n"
            f'    model="{mid}",\n'
            f"    max_tokens={max_tokens},\n"
            '    messages=[{"role": "user", "content": "Write a pygame window that bounces a ball."}],\n'
            ")\n"
            "print(reply.choices[0].message.content)\n"
            'print(reply.model_dump().get("hivemind"))  # price, charge and producer of this call\n'
        )
    if tool == "env":
        return f"export OPENAI_BASE_URL={base}\nexport OPENAI_API_KEY={api_key}\n"
    raise ValueError(f"unknown tool {tool!r}; choose one of: {', '.join(TOOLS)}")


def _gateway(args: argparse.Namespace) -> str:
    return (args.gateway or os.environ.get(GATEWAY_ENV) or DEFAULT_GATEWAY).rstrip("/")


def _key(parser: argparse.ArgumentParser) -> str:
    key = os.environ.get(API_KEY_ENV, "")
    if not key:
        parser.error(f"set {API_KEY_ENV} to your HiveMind API key (run `register` first)")
    return key


def _save_or_print_key(body: dict[str, Any], args: argparse.Namespace, *, is_google: bool = False) -> int:
    if args.save_key:
        path = Path(args.save_key).expanduser()
        if path.exists():
            print(f"refusing to overwrite {path}", file=sys.stderr)
            return 1
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(descriptor, "w") as handle:
            handle.write(body["api_key"])
        print(f"key saved to {path} (owner-only). Then: export {API_KEY_ENV}=$(cat {path})")
    else:
        if is_google:
            header = "Account created with Google!" if body.get("created") else "Welcome back: previous key rotated."
            print(f"{header}\nYour API key (shown once, store it safely):\n  {body['api_key']}\nThen: export {API_KEY_ENV}=<that key>")
        else:
            print(f"Your API key (shown once, store it safely):\n  {body['api_key']}\nThen: export {API_KEY_ENV}=<that key>")
    print(f"account {body['account_id']} role {body['role']}. New accounts hold 0 HMT: ask the operator for a grant, or earn by sharing a GPU.")
    return 0


def cmd_register(client: httpx.Client, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    response = client.post(_gateway(args) + "/v1/hivemind/register", json={"handle": args.handle, "role": args.role})
    if response.status_code != 200:
        print(f"registration failed ({response.status_code}): {response.text[:300]}", file=sys.stderr)
        return 1
    return _save_or_print_key(response.json(), args, is_google=False)


def cmd_register_google(client: httpx.Client, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    gateway = _gateway(args)
    token = args.token or os.environ.get("GOOGLE_ID_TOKEN", "")

    if not token and getattr(args, "gcloud", False):
        try:
            import subprocess
            cmd = ["gcloud", "auth", "print-identity-token", "--include-email"]
            if getattr(args, "audience", None):
                cmd.append(f"--audiences={args.audience}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and proc.stdout.strip():
                token = proc.stdout.strip()
            else:
                print(f"gcloud identity token failed: {proc.stderr.strip()}", file=sys.stderr)
        except (FileNotFoundError, OSError):
            print("gcloud command not found on PATH.", file=sys.stderr)

    if not token and getattr(args, "browser", False):
        signup_url = f"{gateway}/signup"
        print(f"Opening browser for Google Sign-in: {signup_url}")
        try:
            import webbrowser
            webbrowser.open(signup_url)
        except Exception:
            pass
        print("Complete Google sign-in in your browser.")
        print("Paste the API key displayed on the page, or enter your Google ID token:")
        try:
            entered = input("API Key or Google ID Token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return 1
        if entered.startswith("hm-"):
            return _save_or_print_key({"account_id": "google-user", "api_key": entered, "role": args.role, "created": False}, args, is_google=True)
        token = entered

    if not token:
        parser.error("provide a Google ID token via --token, set GOOGLE_ID_TOKEN, use --gcloud, or use --browser")

    response = client.post(gateway + "/v1/hivemind/register/google", json={"id_token": token, "role": args.role})
    if response.status_code != 200:
        print(f"Google sign-in failed ({response.status_code}): {response.text[:300]}", file=sys.stderr)
        return 1
    return _save_or_print_key(response.json(), args, is_google=True)


def cmd_check(client: httpx.Client, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    gateway = _gateway(args)
    headers = {"Authorization": f"Bearer {_key(parser)}"}
    me = client.get(gateway + "/v1/hivemind/me", headers=headers)
    if me.status_code != 200:
        print(f"key rejected ({me.status_code}). Check {API_KEY_ENV}.", file=sys.stderr)
        return 1
    account = me.json()
    live = [m.get("id") for m in client.get(gateway + "/v1/models", headers=headers).json().get("data", [])]
    tiers = client.get(gateway + "/v1/hivemind/catalog").json().get("tiers", [])
    print(f"account {account['account_id']}  role {account['role']}  balance {account['balance']} HMT  max price {account['max_price_per_1k']}/1k")
    print(f"live models now: {', '.join(live) if live else 'none'}")
    for tier in tiers:
        flag = "live" if tier["public_id"] in live else "no producer yet"
        print(f"  {tier['public_id']:<22} {'MoE ' if tier['moe'] else '    '}needs ~{tier['min_vram_gb']:g} GB  [{flag}]  {tier['good_for']}")
    if account["balance"] <= 0:
        print("balance is 0: requests will be refused until you are granted HMT or earn some as a producer.")
    return 0


def cmd_snippet(client: httpx.Client, args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    if args.list or not args.tool:
        print("tools: " + ", ".join(TOOLS))
        return 0
    key = os.environ.get(API_KEY_ENV, "") if args.embed_key else KEY_PLACEHOLDER
    if args.embed_key and not key:
        parser.error(f"--embed-key needs {API_KEY_ENV} to be set")
    try:
        print(render_snippet(args.tool, gateway=_gateway(args), model=args.model, api_key=key, max_tokens=args.max_tokens), end="")
    except ValueError as exc:
        parser.error(str(exc))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Connect your editor or Python code to the HiveMind OpenRouter.")
    parser.add_argument("--gateway", help=f"Gateway URL (default: ${GATEWAY_ENV} or the public HiveMind gateway)")
    sub = parser.add_subparsers(dest="command", required=True)
    reg = sub.add_parser("register", help="Create an account and get an API key")
    reg.add_argument("--handle", required=True)
    reg.add_argument("--role", choices=["consumer", "producer", "prosumer"], default="prosumer",
                     help="prosumer (default) can both use models and share a GPU")
    reg.add_argument("--save-key", metavar="FILE", help="Write the key to FILE (owner-only) instead of printing it")
    reg.set_defaults(func=cmd_register)

    reg_g = sub.add_parser("register-google", help="Sign in with Google (ID token, gcloud, or browser)")
    reg_g.add_argument("--token", help="Google ID token (or set GOOGLE_ID_TOKEN)")
    reg_g.add_argument("--gcloud", action="store_true", help="Obtain Google ID token using gcloud CLI")
    reg_g.add_argument("--audience", help="Optional audience parameter for gcloud identity token")
    reg_g.add_argument("--browser", action="store_true", help="Open /signup page in your browser")
    reg_g.add_argument("--role", choices=["consumer", "producer", "prosumer"], default="prosumer",
                       help="prosumer (default) can both use models and share a GPU")
    reg_g.add_argument("--save-key", metavar="FILE", help="Write the key to FILE (owner-only) instead of printing it")
    reg_g.set_defaults(func=cmd_register_google)

    login_g = sub.add_parser("login-google", help="Alias for register-google (returns rotated API key)")
    login_g.add_argument("--token", help="Google ID token (or set GOOGLE_ID_TOKEN)")
    login_g.add_argument("--gcloud", action="store_true", help="Obtain Google ID token using gcloud CLI")
    login_g.add_argument("--audience", help="Optional audience parameter for gcloud identity token")
    login_g.add_argument("--browser", action="store_true", help="Open /signup page in your browser")
    login_g.add_argument("--role", choices=["consumer", "producer", "prosumer"], default="prosumer")
    login_g.add_argument("--save-key", metavar="FILE", help="Write the key to FILE (owner-only) instead of printing it")
    login_g.set_defaults(func=cmd_register_google)
    chk = sub.add_parser("check", help="Verify your key and show balance, live models and tiers")
    chk.set_defaults(func=cmd_check)
    snip = sub.add_parser("snippet", help="Print settings for an editor or library")
    snip.add_argument("tool", nargs="?", choices=TOOLS)
    snip.add_argument("--model", default=DEFAULT_MODEL, help="tier id such as coder, coder-lite, moe-coder, or auto")
    snip.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    snip.add_argument("--embed-key", action="store_true", help=f"Put ${API_KEY_ENV} into the snippet instead of a placeholder")
    snip.add_argument("--list", action="store_true")
    snip.set_defaults(func=cmd_snippet)
    return parser


def main(argv: Sequence[str] | None = None, *, transport: httpx.BaseTransport | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    with httpx.Client(timeout=HTTP_TIMEOUT_S, transport=transport) as client:
        try:
            return int(args.func(client, args, parser))
        except httpx.HTTPError as exc:
            print(f"cannot reach the gateway: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1


if __name__ == "__main__":
    sys.exit(main())
