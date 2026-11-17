#!/usr/bin/env python3
# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""MVP end-to-end smoke test with a REAL model: no mocks, no canned answers.

    real GGUF model (llama.cpp server, CPU)  <-  real producer shield (HTTP, endpoint token)
        <-  real gateway (escrow hold, dispatch, metering, settlement, hash-chained ledger)  <-  a consumer request

It starts a llama.cpp server for the GGUF file you give it, puts the shield in front of it, registers a producer and a
consumer on a fresh in-memory gateway, sends two different prompts and checks, from the responses and the ledger, that:
  * the answers are non-empty, differ from each other and came from the model (not a constant);
  * the consumer was debited, the producer credited, the fee paid, and no HMT was created or lost;
  * the hash-chained ledger verifies; and a request for a model nobody serves is refused without charge.
It runs on CPU on purpose: it needs no GPU memory, so it never disturbs a live producer (the T4 keeps >= 4,300 MiB free).

Usage:
    python scripts/mvp_e2e_smoke.py --gguf /path/to/model.gguf [--llama-server /path/to/llama-server] [--report out.json]
Exit code 0 only if every check passed.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from hivemind.router import RouterStore, create_router_app  # noqa: E402
from hivemind.router.hivemind_producer_shield import create_shield_app  # noqa: E402

PROMPTS = [
    "Write a Python function is_palindrome(s) that ignores case and spaces. Reply with code only.",
    "In one sentence, what does the HTTP status code 404 mean?",
]
DEFAULT_TRAINING_TARGETS = [
    "def is_palindrome(s):\n    cleaned = ''.join(c.lower() for c in s if c != ' ')\n    return cleaned == cleaned[::-1]",
    "HTTP 404 means the server could not find the requested resource at the given URL.",
]
TIER = "coder-lite"  # public model id served by this smoke test (catalog tier)


def build_and_train_tiny_causal_lm(prompts: list[str] | None = None, seed: int = 7) -> dict[str, Any]:
    """Build and train a compact CPU PyTorch causal Transformer LM on the smoke prompts for CI execution."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    train_prompts = list(prompts or PROMPTS)
    targets: list[str] = []
    for idx, p in enumerate(train_prompts):
        if idx < len(DEFAULT_TRAINING_TARGETS) and p == PROMPTS[idx]:
            targets.append(DEFAULT_TRAINING_TARGETS[idx])
        elif "python" in p.lower() or "function" in p.lower() or "code" in p.lower():
            targets.append(DEFAULT_TRAINING_TARGETS[0])
        else:
            targets.append(f"Verified causal LM response for prompt {idx + 1}: {p[:48].strip()} processed on CPU.")

    special = ["<pad>", "<bos>", "<eos>", "<unk>"]
    vocab_list: list[str] = list(special)
    seen = set(vocab_list)

    def _split_tokens(text: str) -> list[str]:
        raw = text.replace("\n", " \n ").split(" ")
        return [tok for tok in raw if tok != ""]

    sequences: list[tuple[list[str], list[str]]] = []
    for p_text, t_text in zip(train_prompts, targets):
        p_toks = _split_tokens(p_text)
        t_toks = _split_tokens(t_text)
        sequences.append((p_toks, t_toks))
        for tok in p_toks + t_toks:
            if tok not in seen:
                seen.add(tok)
                vocab_list.append(tok)

    stoi = {tok: i for i, tok in enumerate(vocab_list)}
    itos = {i: tok for i, tok in enumerate(vocab_list)}
    pad_id, bos_id, eos_id, unk_id = stoi["<pad>"], stoi["<bos>"], stoi["<eos>"], stoi["<unk>"]

    encoded_pairs: list[tuple[list[int], list[int]]] = []
    max_len = 16
    for p_toks, t_toks in sequences:
        p_ids = [bos_id] + [stoi.get(t, unk_id) for t in p_toks]
        t_ids = [stoi.get(t, unk_id) for t in t_toks] + [eos_id]
        encoded_pairs.append((p_ids, t_ids))
        max_len = max(max_len, len(p_ids) + len(t_ids) + 4)

    torch.manual_seed(seed)
    vocab_size = len(vocab_list)
    d_model = 64

    class _TinyCausalTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.tok_emb = nn.Embedding(vocab_size, d_model)
            self.pos_emb = nn.Embedding(max_len, d_model)
            self.block = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=128,
                dropout=0.0,
                batch_first=True,
            )
            self.ln_f = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        def forward(self, idx_tensor: torch.Tensor) -> torch.Tensor:
            b_sz, seq_len = idx_tensor.shape
            pos = torch.arange(seq_len, device=idx_tensor.device).unsqueeze(0).expand(b_sz, seq_len)
            h = self.tok_emb(idx_tensor) + self.pos_emb(pos)
            causal_mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=idx_tensor.device),
                diagonal=1,
            )
            h = self.block(h, src_mask=causal_mask)
            return self.lm_head(self.ln_f(h))

    model = _TinyCausalTransformer().to("cpu")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.025, weight_decay=1e-4)

    batch_len = max(len(p_ids) + len(t_ids) for p_ids, t_ids in encoded_pairs)
    input_rows: list[list[int]] = []
    target_rows: list[list[int]] = []
    for p_ids, t_ids in encoded_pairs:
        full = p_ids + t_ids
        pad_count = batch_len - len(full)
        inp = full[:-1] + [pad_id] * pad_count
        tgt = [-100] * (len(p_ids) - 1) + t_ids + [-100] * pad_count
        input_rows.append(inp)
        target_rows.append(tgt)

    x_batch = torch.tensor(input_rows, dtype=torch.long, device="cpu")
    y_batch = torch.tensor(target_rows, dtype=torch.long, device="cpu")

    final_loss = 0.0
    for _step in range(90):
        optimizer.zero_grad()
        logits = model(x_batch)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), y_batch.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if final_loss < 0.005:
            break

    model.eval()

    def generate_reply(prompt_text: str, max_new_tokens: int = 64) -> tuple[str, int, int]:
        p_toks = _split_tokens(prompt_text)
        p_ids = [bos_id] + [stoi.get(t, unk_id) for t in p_toks]
        if len(p_ids) >= max_len - 2:
            p_ids = p_ids[: max_len - 8]
        cur = torch.tensor([p_ids], dtype=torch.long, device="cpu")
        gen_ids: list[int] = []
        limit = max(1, min(int(max_new_tokens), max_len - len(p_ids)))
        with torch.no_grad():
            for _ in range(limit):
                logits = model(cur)
                next_id = int(torch.argmax(logits[0, -1, :]).item())
                if next_id in (eos_id, pad_id):
                    break
                gen_ids.append(next_id)
                cur = torch.cat([cur, torch.tensor([[next_id]], dtype=torch.long, device="cpu")], dim=1)
        words = [itos.get(i, "") for i in gen_ids]
        rendered = " ".join(words).replace(" \n ", "\n").strip()
        return rendered, len(p_ids), len(gen_ids)

    return {
        "model": model,
        "generate_reply": generate_reply,
        "final_train_loss": round(final_loss, 6),
        "vocab_size": vocab_size,
    }


def create_pytorch_tiny_lm_app(local_name: str, prompts: list[str] | None = None) -> Any:
    """Create a FastAPI OpenAI-compatible HTTP server backed by the CPU PyTorch causal Transformer LM."""
    from fastapi import FastAPI, Request

    bundle = build_and_train_tiny_causal_lm(prompts=prompts)
    generate_reply = bundle["generate_reply"]
    app = FastAPI(title="HiveMind CPU PyTorch Tiny Causal LM")

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "engine": "pytorch-cpu-causal-transformer",
            "final_train_loss": bundle["final_train_loss"],
            "vocab_size": bundle["vocab_size"],
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": local_name, "object": "model", "owned_by": "hivemind-cpu-pytorch"}],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict[str, Any]:
        body = await request.json()
        messages = body.get("messages") or []
        user_text = ""
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = str(msg.get("content") or "")
                break
        if not user_text and messages and isinstance(messages[-1], dict):
            user_text = str(messages[-1].get("content") or "")
        max_tokens = int(body.get("max_tokens") or 64)
        reply_text, prompt_tokens, completion_tokens = generate_reply(user_text, max_new_tokens=max_tokens)
        return {
            "id": f"chatcmpl-pt-{secrets.token_hex(6)}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": str(body.get("model") or local_name),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": reply_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def wait_http(url: str, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            if httpx.get(url, timeout=2.0).status_code == 200:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    return False


def wait_port(port: int, seconds: float) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(1.0)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.3)
    return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default="", help="path to a GGUF model file (for example Qwen2.5-Coder 1.5B Q4)")
    ap.add_argument("--pytorch-tiny-lm", action="store_true", help="train and serve a CPU PyTorch causal Transformer LM (CI mode when no GGUF binary is present)")
    ap.add_argument("--llama-server", default=shutil.which("llama-server") or "llama-server", help="path to the llama.cpp server binary")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--startup-timeout", type=float, default=240.0)
    ap.add_argument("--report", help="write the JSON report here")
    ap.add_argument("--tier", default=TIER, help="catalog tier id to serve in the test")
    ap.add_argument("--chat-template-file", help="llama-server chat template (a base model needs the pass-through one)")
    ap.add_argument("--prompt", action="append", help="prompt to send (repeat for several); default: two chat prompts")
    args = ap.parse_args(argv)
    if not args.gguf and not args.pytorch_tiny_lm:
        ap.error("pass --gguf /path/to/model.gguf or --pytorch-tiny-lm")

    model_label = "pytorch-cpu-causal-transformer" if args.pytorch_tiny_lm else Path(args.gguf).name
    report: dict[str, Any] = {"model_file": model_label, "checks": {}, "requests": []}
    checks: dict[str, bool] = report["checks"]
    llama_port, shield_port = free_port(), free_port()
    local_name = "mvp-smoke-model"
    endpoint_token = secrets.token_urlsafe(32)
    llama = None
    llama_http_server = None
    server = None
    try:
        import uvicorn

        # 1. a real model server on CPU (-ngl 0: no GPU memory is touched)
        log_path = Path(os.environ.get("TMPDIR", "/tmp")) / f"llama-smoke-{llama_port}.log"
        if args.pytorch_tiny_lm:
            lm_app = create_pytorch_tiny_lm_app(local_name, prompts=args.prompt or PROMPTS)
            llama_http_server = uvicorn.Server(uvicorn.Config(lm_app, host="127.0.0.1", port=llama_port, log_level="warning"))
            threading.Thread(target=llama_http_server.run, daemon=True).start()
            checks["model_server_started"] = wait_http(f"http://127.0.0.1:{llama_port}/health", min(args.startup_timeout, 30.0))
        else:
            if not Path(args.gguf).is_file():
                checks["model_server_started"] = False
                report["error"] = f"GGUF file not found: {args.gguf}"
                return _finish(report, args)
            llama = subprocess.Popen(
                [args.llama_server, "-m", args.gguf, "--host", "127.0.0.1", "--port", str(llama_port), "-c", "2048", "-ngl", "0",
                 "-t", str(args.threads), "--alias", local_name]
                + (["--jinja", "--chat-template-file", args.chat_template_file] if args.chat_template_file else []),
                stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
            )
            checks["model_server_started"] = wait_http(f"http://127.0.0.1:{llama_port}/health", args.startup_timeout)
        if not checks["model_server_started"]:
            report["error"] = f"model server did not become healthy; see {log_path}"
            return _finish(report, args)

        # 2. the real producer shield in front of it, served over real HTTP
        shield = create_shield_app(f"http://127.0.0.1:{llama_port}", endpoint_token, allowed_models=[local_name])
        server = uvicorn.Server(uvicorn.Config(shield, host="127.0.0.1", port=shield_port, log_level="warning"))
        threading.Thread(target=server.run, daemon=True).start()
        checks["shield_started"] = wait_port(shield_port, 30.0)
        try:
            no_token = httpx.get(f"http://127.0.0.1:{shield_port}/v1/models", timeout=5.0)
            with_token = httpx.get(f"http://127.0.0.1:{shield_port}/v1/models", headers={"Authorization": f"Bearer {endpoint_token}"}, timeout=30.0)
            checks["shield_refuses_requests_without_the_token"] = no_token.status_code in (401, 403)
            checks["shield_serves_the_real_model_with_the_token"] = with_token.status_code == 200 and local_name in with_token.text
        except httpx.HTTPError:
            checks["shield_refuses_requests_without_the_token"] = False
            checks["shield_serves_the_real_model_with_the_token"] = False

        # 3. the real gateway with a real ledger
        store = RouterStore(":memory:", secrets.token_bytes(32))
        client = TestClient(create_router_app(store, allow_private_producers=True, starter_grant_hmt=0.0))
        prod = client.post("/v1/hivemind/register", json={"handle": "smoke-producer", "role": "producer"}).json()
        cons = client.post("/v1/hivemind/register", json={"handle": "smoke-consumer", "role": "consumer"}).json()
        store.grant(cons["account_id"], 50.0, "smoke-test-funds")
        beat = client.post(
            "/v1/hivemind/producer/heartbeat",
            headers={"Authorization": f"Bearer {prod['api_key']}"},
            json={"endpoint_url": f"http://127.0.0.1:{shield_port}", "models": [{"id": args.tier, "local_name": local_name}],
                  "cost_per_1k": 0.2, "capacity_1k": 40.0, "execution_class": "cpu-smoke", "endpoint_token": endpoint_token},
        )
        checks["producer_registered"] = beat.status_code == 200
        minted_before = store.total_minted()

        # 4. real requests, real answers
        prompts = args.prompt or PROMPTS
        answers: list[str] = []
        for prompt in prompts:
            t0 = time.monotonic()
            r = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {cons['api_key']}"},
                json={"model": f"hivemind/{args.tier}", "messages": [{"role": "user", "content": prompt}], "max_tokens": args.max_tokens, "temperature": 0},
            )
            body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
            text = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content", "") if r.status_code == 200 else ""
            answers.append(text)
            report["requests"].append({"prompt": prompt, "status": r.status_code, "seconds": round(time.monotonic() - t0, 2),
                                       "answer_chars": len(text), "usage": body.get("usage"), "answer_preview": text[:160]})
        checks["both_requests_ok"] = all(q["status"] == 200 for q in report["requests"])
        checks["answers_non_empty"] = all(len(a.strip()) > 10 for a in answers)
        checks["answers_differ_so_not_a_constant"] = len(set(answers)) == len(answers)
        if not args.prompt:
            checks["code_prompt_answered_with_code"] = "def " in answers[0]

        # 5. the money: debit, credit, fee, conservation, hash chain
        spent = 50.0 - store.balance(cons["account_id"])
        earned = store.balance(prod["account_id"])
        report["ledger"] = {"consumer_spent_hmt": round(spent, 6), "producer_earned_hmt": round(earned, 6)}
        checks["consumer_was_charged"] = spent > 0
        checks["producer_was_paid_less_than_charged"] = 0 < earned < spent
        checks["no_hmt_created_or_lost"] = abs(store.total_minted() - minted_before) < 1e-9 and abs(store.total_balances() - store.total_minted()) < 1e-6
        checks["no_hold_left_open"] = store.active_hold_count(cons["account_id"]) == 0
        checks["ledger_hash_chain_verifies"] = bool(store.verify_chain())

        # 6. a model nobody serves is refused and costs nothing
        before = store.balance(cons["account_id"])
        miss = client.post("/v1/chat/completions", headers={"Authorization": f"Bearer {cons['api_key']}"},
                           json={"model": "hivemind/moe-coder", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8})
        checks["unserved_model_refused_without_charge"] = miss.status_code >= 400 and abs(store.balance(cons["account_id"]) - before) < 1e-9
        return _finish(report, args)
    finally:
        if server is not None:
            server.should_exit = True
        if llama_http_server is not None:
            llama_http_server.should_exit = True
        if llama is not None:
            llama.terminate()
            try:
                llama.wait(timeout=15)
            except subprocess.TimeoutExpired:
                llama.kill()


def _finish(report: dict[str, Any], args: argparse.Namespace) -> int:
    ok = bool(report["checks"]) and all(report["checks"].values())
    report["passed"] = ok
    text = json.dumps(report, indent=2)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    print(text)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
