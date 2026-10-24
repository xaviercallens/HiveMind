# HiveMind OpenRouter — build spec (interface contract for the parallel builders)

Owner's brief (2026-10-24): an OpenRouter-style gateway on GCP where end users register and use HiveMind MoE models
through the standard OpenAI endpoint protocol from their IDE (OpenCode, Hermes, VS Code); producers contribute GPU
MoE inference and earn **HiveMind points (HMT)** they can spend later; a Nash equilibrium makes it win-win.

Package: `src/hivemind/router/` (torch-free, imports only stdlib, fastapi, pydantic, httpx, and
`hivemind.economy.compute_coin`, `hivemind.security.hivemind_firewall`). Python ≥3.11, type-annotated, no stubs.

## 1. `market.py` — the win-win two-sided equilibrium (pure functions)

Units: price in **HMT per 1,000 tokens**. Fee `f = validator_share + burn_share` (defaults 0.10 + 0.05 = 0.15);
producers receive `(1 - f)` of every payment.

Model per MoE model id, per epoch:
- Consumers `j` have linear value `v_j` (HMT per 1k tokens, their declared max price) for capacity.
- Producers `i` have cost `c_i` (HMT per 1k tokens) and capacity `k_i` (1k-token units per epoch).
- Consumers play the **Kelly proportional-allocation game** over the supplied capacity `S`. With linear values, the
  equilibrium per-token price `ω` solves `Σ_j max(0, 1 − ω / v_j) = 1` (requires ≥ 2 consumers with v > 0); shares
  `x_j = max(0, 1 − ω/v_j)`; bids `w_j = ω x_j S`. Key fact (prove in the docstring): `ω` does not depend on `S`.
  Reuse `hivemind.economy.compute_coin.kelly_equilibrium(values)`: its `total_payment` W with capacity 1 *is* ω.
- Producers therefore face a constant marginal revenue `(1 − f) ω` per unit: supplying full capacity is a dominant
  strategy iff `c_i < (1 − f) ω`, and zero otherwise (`c_i == (1−f)ω` → indifferent, supply 0). `S = Σ_{active} k_i`.
- **Reserve fallback**: let `r = min_i c_i / (1 − f)` (cheapest producer's break-even price). If fewer than 2
  consumers, or `ω ≤ r` (no producer would supply), the mechanism is `posted_price` at `p = r · (1 + margin)` with
  `margin = 0.05`: consumers with `v_j ≥ p` are served, capacity of producers with `c_i < (1−f)p` is split equally
  among served consumers (deterministic). Report `mechanism = "kelly" | "posted_price" | "no_trade"`.
- **Win-win certificate** `certify(eq) -> Certificate`: (a) every served consumer's surplus `(v_j − p) x_j S ≥ 0`,
  (b) every active producer's profit `((1−f)p − c_i) k_i > 0`, inactive producers' profit 0, (c) no producer gains by
  any unilateral supply change on a grid of 41 points in `[0, k_i]`, (d) in the kelly mechanism, no consumer gains
  by changing its bid `w_j` on a log grid (utility `u_j(w) = v_j S · w/(w + W_−j) − w`), tolerance `1e-9` relative,
  (e) conservation: `Σ payments = Σ producer revenue + validator fees + burn` within 1e-9.
  `Certificate` fields: `individually_rational_consumers`, `individually_rational_producers`, `producer_nash`,
  `consumer_nash` (None for posted_price), `conserved`, `max_deviation_gain`, `win_win` (all True).
- Public API (exact names):
  ```python
  @dataclass(frozen=True) class ConsumerBid: account_id: str; max_price_per_1k: float
  @dataclass(frozen=True) class ProducerOffer: account_id: str; cost_per_1k: float; capacity_1k: float
  @dataclass(frozen=True) class FeePolicy: validator_share: float = 0.10; burn_share: float = 0.05; posted_margin: float = 0.05
  @dataclass(frozen=True) class MarketEquilibrium: model_id: str; mechanism: str; price_per_1k: float; supply_1k: float;
      consumer_shares: Mapping[str, float]; consumer_surplus: Mapping[str, float]; active_producers: tuple[str, ...];
      producer_profit: Mapping[str, float]; producer_revenue_per_1k: float; fee: FeePolicy  (+ to_dict())
  def clear_market(model_id: str, bids: Sequence[ConsumerBid], offers: Sequence[ProducerOffer], fee: FeePolicy = FeePolicy()) -> MarketEquilibrium
  def certify(eq: MarketEquilibrium, bids, offers) -> Certificate
  def split_payment(price_per_1k: float, tokens: int, fee: FeePolicy) -> tuple[float, float, float, float]  # (charge, producer, validators, burn); charge = price * tokens / 1000
  ```
- Tests `tests/test_router_market.py`: closed-form two-consumer case (v = 4, 4 → ω = 2), dominant-strategy supply,
  reserve fallback with one consumer, no_trade when every v_j < r·(1+margin), Hypothesis properties: certificate
  `win_win` for random instances (2–10 consumers, 1–8 producers, values/costs log-uniform), conservation of split.

## 2. `accounts.py` — accounts, API keys, HMT ledger (SQLite)

- `RouterStore(path: Path | ":memory:", secret: bytes)`; secret ≥ 32 bytes else `ValueError`.
- API keys: `"hm-" + secrets.token_urlsafe(32)`; stored only as `HMAC-SHA256(secret, key)` hex. Returned once.
- Tables: `accounts(account_id PK, handle UNIQUE, role CHECK in ('consumer','producer','prosumer','admin'), key_hash UNIQUE, created_utc, max_price_per_1k REAL)`;
  `ledger(seq INTEGER PK, account_id, delta REAL, kind, ref, prev_hash, entry_hash, utc)` hash-chained
  (`entry_hash = sha256(prev_hash | canonical json of row)`); `producers(account_id PK, endpoint_url, models_json, cost_per_1k, capacity_1k, last_heartbeat_utc, execution_class)`;
  `usage(request_id PK, consumer_id, producer_id, model_id, prompt_tokens, completion_tokens, billed_tokens, price_per_1k, charge, utc)`.
  Special system accounts: `sys:validators`, `sys:burn` (balances only grow).
- `account_id` = `HMAC-SHA256(secret, normalise_identity(handle))[:32]` using `hivemind.security.hivemind_firewall.normalise_identity`.
- Methods (exact names): `register(handle, role) -> tuple[str, str]` (account_id, api_key); `authenticate(api_key) -> Account | None` (constant-time via HMAC compare of hashes);
  `balance(account_id) -> float`; `grant(account_id, amount, ref)` (admin mint; the only way HMT is created);
  `settle(request_id, consumer_id, producer_id, model_id, prompt_tokens, completion_tokens, billed_tokens, price_per_1k, fee: FeePolicy) -> Settlement`
  (atomic transaction: debit consumer `charge`, credit producer, `sys:validators`, `sys:burn`; raises `InsufficientBalance` before writing; duplicate request_id raises);
  `set_max_price`, `upsert_producer(...)`, `active_producers(model_id, now_utc, ttl_s=120) -> list[ProducerRecord]`, `verify_chain() -> bool`, `total_minted() -> float`.
- Invariant (Hypothesis test): after any sequence of grants and settlements, `Σ balances (incl. sys) == total_minted()` within 1e-9, no balance negative except none, `verify_chain()` True; tampering a row makes `verify_chain()` False.
- Tests `tests/test_router_accounts.py`.

## 3. `gateway.py` — OpenAI-compatible FastAPI app

`create_router_app(store: RouterStore, *, http_client: httpx.Client | None = None, fee: FeePolicy = FeePolicy(), allow_private_producers: bool = False, admin_key_hash: str | None = None, clock=time.time) -> FastAPI`.

Endpoints:
- `GET /v1/models` (OpenAI list format; ids are `hivemind/<model>` for each model served by ≥1 live producer, plus `hivemind/auto`).
- `POST /v1/chat/completions` — Bearer auth (`Authorization: Bearer hm-...`). Body is the OpenAI chat request. Resolve model
  (`hivemind/auto` → the live model with most supply); compute current equilibrium (`clear_market` over all consumer
  accounts' `max_price_per_1k` (default 2.0) and live producers for that model); if consumer's max price < price → 402
  with OpenAI-style error JSON; reserve check: balance ≥ price × (prompt bytes + max_tokens or 1024)/1000 else 402.
  Pick producer: among `active_producers` of the equilibrium, deterministic weighted round-robin by capacity.
  Forward to `{endpoint_url}/v1/chat/completions` with the body (model rewritten to the producer's local name),
  timeout 120 s; on failure try the next producer; all fail → 502, no charge.
  Metering (anti-inflation): `billed_tokens = min(reported prompt+completion, utf8_bytes(prompt msgs) + utf8_bytes(completion text))`
  and completion part also ≤ `max_tokens` when given. Settle via `store.settle`. Response = producer JSON with `model`
  rewritten to `hivemind/<model>` and an extra top-level `hivemind` object `{price_per_1k, charged_hmt, producer_id, mechanism, request_id}`.
  `stream: true`: proxy SSE chunks (`data: ...\n\n`, final `data: [DONE]`), count completion bytes from deltas, settle at the end.
- `POST /v1/hivemind/register` `{handle, role}` → `{account_id, api_key, role}` (rate-limited 10/min per client IP, in memory).
- `GET /v1/hivemind/me` → account, balance, role, recent usage.
- `POST /v1/hivemind/consumer/max-price` `{max_price_per_1k}`.
- `POST /v1/hivemind/producer/heartbeat` (producer/prosumer only) `{endpoint_url, models: [{id, local_name}], cost_per_1k, capacity_1k, execution_class}`.
  SSRF guard: `https` required and every resolved IP must be global (`ipaddress.ip_address(x).is_global`), unless `allow_private_producers`.
- `GET /v1/hivemind/market?model=` → equilibrium + certificate JSON (public, no auth).
- `POST /v1/hivemind/admin/grant` with `X-Admin-Key` whose HMAC matches `admin_key_hash` → `store.grant`.
- `GET /healthz`.
Errors use OpenAI shape `{"error": {"message", "type", "code"}}`. Never log API keys or prompt text.
Tests `tests/test_router_gateway.py` using `fastapi.testclient.TestClient` and `httpx.MockTransport` producers (external I/O):
register → grant → chat completion charged correctly and producer credited; inflation attempt (producer reports 10× usage) billed by bytes cap;
402 on low balance; 502 with no charge when producer down; failover to second producer; streaming; SSRF rejects `https://127.0.0.1`, `http://`, `https://10.0.0.1`;
`/v1/models`; official `openai` Python client (installed, v3.x) against the TestClient via its `http_client` param if feasible.

## 4. Deploy + docs
- `deploy/gcp_openrouter/{Dockerfile,server.py,requirements.txt,deploy.sh}`: env `HIVEMIND_ROUTER_SECRET` (Secret Manager), `HIVEMIND_ROUTER_DB` (default `/tmp/router.db`, documented as ephemeral on Cloud Run — production needs Cloud SQL/Firestore, task HM-S13-03), `HIVEMIND_ROUTER_ADMIN_KEY_HASH`. `deploy.sh` runs `gcloud run deploy hivemind-openrouter --source . --region us-central1 --min-instances 0 --max-instances 2` (do not run it; owner decides).
- `scripts/hivemind_producer_agent.py`: a producer daemon that heartbeats to the gateway, advertising a local OpenAI-compatible server (Ollama/vLLM/llama.cpp) — real code, CLI flags.
- `docs/HIVEMIND_OPENROUTER.md`: architecture, the equilibrium and why it is win-win, HMT lifecycle, IDE setup (OpenCode `opencode.json` provider via `@ai-sdk/openai-compatible` with `baseURL`/`apiKey`; VS Code via the Continue extension `config.yaml` `provider: openai` + `apiBase`; Hermes and any other OpenAI-compatible client via base URL + key / `OPENAI_BASE_URL` + `OPENAI_API_KEY`) — mark each IDE snippet "verify against the tool's current docs".
