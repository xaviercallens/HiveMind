# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Loopback OpenAI-compatible endpoint serving the latest assimilated QLoRA MoE adapter.

BOINC is a batch system, so interactive inference is not a BOINC job: this is a small server next to it that
loads the base MoE (NF4 attention) plus the adapter the project assimilated, and answers ``/v1/chat/completions``
with greedy decoding (non-streaming). The prompt is the last user message, tokenised as-is (the model is a *base*
model with no chat template), so the output equals the batch inference app's for the same prompt.

Binds to 127.0.0.1 only. Exposing it is the owner's decision (see docs/BOINC_CORE_NODE.md).
"""

from __future__ import annotations

import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from hivemind.boinc.real_client.moe_app import HF_CACHE, load_peft_model, state_sha


_LOG = logging.getLogger(__name__)


def log_request(client: str, fmt: str, *args: Any) -> None:
    """Record one HTTP request line at DEBUG level instead of writing it to stderr (the handler's default)."""
    _LOG.debug("node endpoint %s: %s", client, fmt % args if args else fmt)


class NodeEndpoint:
    def __init__(self, spec: dict[str, Any], adapter_path: Path, port: int = 0, served_name: str = "hivemind-granite-moe-qlora") -> None:
        import torch
        from transformers import AutoTokenizer

        self.spec, self.name = spec, served_name
        state = torch.load(adapter_path, map_location="cpu")["state"]
        self.adapter_sha = state_sha(state)
        self.model = load_peft_model(spec, state)
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(spec["model_id"], cache_dir=HF_CACHE)
        self.lock = threading.Lock()
        node = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: Any) -> None:
                log_request(self.address_string(), format, *args)

            def _json(self, obj: Any, code: int = 200) -> None:
                b = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

            def do_GET(self) -> None:
                if self.path in ("/health", "/healthz"):
                    return self._json({"ok": True, "adapter_sha256": node.adapter_sha})
                if self.path == "/v1/models":
                    return self._json({"object": "list", "data": [{"id": node.name, "object": "model", "owned_by": "hivemind-local"}]})
                self._json({"error": "not found"}, 404)

            def do_POST(self) -> None:
                if self.path != "/v1/chat/completions":
                    return self._json({"error": "not found"}, 404)
                try:
                    req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                    prompt = [m for m in req["messages"] if m.get("role") == "user"][-1]["content"]
                    out = node.complete(prompt, int(req.get("max_tokens", node.spec.get("max_new_tokens", 24))))
                except Exception as exc:
                    return self._json({"error": {"message": str(exc)}}, 400)
                self._json(out)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.port = self.httpd.server_address[1]

    def complete(self, prompt: str, max_new_tokens: int) -> dict[str, Any]:
        import torch

        ids = self.tok(prompt, return_tensors="pt").input_ids.to(0)
        with self.lock, torch.no_grad():
            gen = self.model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.eos_token_id)
        new = gen[0, ids.size(1) :].tolist()
        text = self.tok.decode(new, skip_special_tokens=True)
        return {
            "id": f"hm-{int(time.time() * 1000)}", "object": "chat.completion", "created": int(time.time()), "model": self.name,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length" if len(new) >= max_new_tokens else "stop"}],
            "usage": {"prompt_tokens": ids.size(1), "completion_tokens": len(new), "total_tokens": ids.size(1) + len(new)},
            "hivemind": {"token_ids": new, "adapter_sha256": self.adapter_sha},
        }

    def start(self) -> None:
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
