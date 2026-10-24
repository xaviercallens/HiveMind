# Changelog

## v2.10.1 — 2026-10-24 — OpenRouter Google Flow: CLI integration, UI showcase & e2e verification

- `scripts/hivemind_connect.py`: added `register-google` and `login-google` subcommands supporting direct Google ID token (`--token`), automatic `gcloud auth print-identity-token --include-email` token generation (`--gcloud`), and browser-directed registration (`--browser`).
- `src/hivemind/boinc/hivemind_at_home_dashboard.py` & `HiveMindatHome.github.io`: added Screen 02C showcase card for the OpenRouter and Google Sign-in gateway, featuring 5 capability tiers (`coder-lite`, `coder`, `coder-plus`, `moe-general`, `moe-coder`) and direct link to `/signup`, localized across 5 languages (`en`, `fr`, `zh`, `es`, `de`).
- `scripts/test_router_google_flow.py`: end-to-end verification suite covering `/signup` CSP, RS256 token verification, anti-replay, key rotation, negative controls, anti-squatting handle protection, and CLI connector. 10/10 checks passing with cryptographic receipt SHA-256 `78631759c841adfa1c847af172cc47229e276a0bcf1b2fa0c75bdba926b6baad`.
- GWAYA zero-stub audit passing 110/110 files.

## v2.10.0 — 2026-10-24 — Easy registration: Sign in with Google

- `hivemind.router.google_auth`: ID-token verification with the standard library plus `cryptography` only. RS256 is pinned (no `none`, no HS256-with-public-key confusion), keys come only from Google's published set (cached, refetched at most once a minute on an unknown key id), audience must be our client id, issuer Google, token fresh and unexpired, e-mail verified, and each token works once. 33 tests, one negative control per rule; three reintroduced bugs (no audience check, any algorithm, no signature check) each made a test fail.
- `POST /v1/hivemind/register/google` and the `/signup` page (strict content-security policy with a per-request nonce, text written with `textContent` only). The store keeps a keyed hash of Google's `sub`, never the e-mail; a returning user keeps account, balance and role and gets a fresh key (old key retired). `google.` handles are reserved against squatting. Optional `HIVEMIND_OPEN_REGISTRATION=0` closes handle-only registration.
- Off until the operator creates an OAuth web client in the Google Cloud console and sets `HIVEMIND_GOOGLE_CLIENT_ID` (steps in `docs/HIVEMIND_OPENROUTER.md`). Not exercised against real Google yet: tests use a fake key set.
- Also: `uv.lock` refreshed for the dev extra (PR #143).

## v2.9.0 — 2026-10-24 — Open-Router for Python and game coders: tiers, GPU presets, producer shield, editor connector

- Capability tiers (`hivemind.router.catalog`, served publicly at `/v1/hivemind/catalog`): `coder-lite` (3B), `coder` (7B), `coder-plus` (14B) and two mixture-of-experts tiers, `moe-general` (gpt-oss 20B, 16 GB) and `moe-coder` (Qwen3-Coder 30B, 24 GB). One market per tier keeps liquidity deep; the tier id is a claim until `HM-S13-13`.
- Producer agent: `--suggest --vram-gb/--detect-vram`, `--preset TIER`, preflight that refuses to advertise a model the endpoint does not list, optional endpoint token. The gateway stores the token and presents it as a bearer secret to the producer's own server (additive column; databases created earlier are upgraded in place, tested on an old-schema file).
- Producer shield (`scripts/hivemind_producer_shield.py`): only `GET /v1/models` and `POST /v1/chat/completions`, constant-time token check, body and concurrency caps, optional model pin, and a fallback that lists models for older Ollama releases. The live run found that fallback was needed: this host's Ollama has no `/v1/models`.
- Connector (`scripts/hivemind_connect.py`): register (key printed once or saved owner-only), `check`, and editor snippets for Continue, Cline, Aider, OpenCode, Zed, JetBrains, Python and environment variables.
- Guide `docs/HIVEMIND_PYTHON_GAME_DEVS.md` with game-development recipes and an honest list of what does not exist yet. ROADMAP §3.11 gains `HM-S13-09..14`.
- Live end-to-end on this host (`scripts/router_gamer_e2e.py`): connector, gateway, shield and real Ollama; negative controls refuse a request without the token and Ollama's management API.

## v2.8.0 — 2026-10-24 — Operation Open-Router: an OpenAI-compatible MoE gateway with a producer/consumer equilibrium, deployed on GCP

- `hivemind.router`: `market` (Kelly consumer game with a price independent of supply, producer dominant-strategy supply, posted-price fallback, a win-win certificate), `accounts` (HMAC-hashed API keys, hash-chained HMT ledger, escrow holds, conservation invariant), `gateway` (`/v1/chat/completions` incl. streaming, `/v1/models`, registration, producer heartbeat with SSRF guard, public market certificate). Works with any OpenAI client; IDE setup for OpenCode, VS Code (Continue) and generic clients in `docs/HIVEMIND_OPENROUTER.md`.
- Producer daemon `scripts/hivemind_producer_agent.py` (refuses a non-https gateway so the key is never sent in clear).
- Adversarial security review found 5 high and 6 medium issues (sybil bids setting the price, double-spend of one balance, conflicting token limits, unbounded billing and response size, `n`/`best_of` multiplication, capacity weighting, unauthenticated costly market endpoint, body limit after parse, rate limit behind a proxy, empty-choice billing); all fixed with regression tests. Low findings fixed: NAT64/6to4/Teredo/IPv4-compatible SSRF bypass, unmetered request fields (allow-list), producer-agent scheme check, single instance.
- 144 tests; live end-to-end against local Ollama through the official `openai` client. Public Cloud Run service with a Litestream-replicated ledger in a private versioned GCS bucket; account and balance verified to survive a revision rollout.
- ROADMAP §3.11 Season 13 (`HM-S13-01..08`). HMT are points with no cash value; producers can read the prompts they serve; output correctness is not yet verified (`HM-S13-05`).


## v2.7.0 — 2026-10-24 — Operation Fair-Market: BOINC computing coin and Nash-equilibrium compute market (proposal + code + simulation)

- `hivemind.economy.compute_coin`: BOINC cobblestone credit granted only for verified work (spot-check + Multi-Krum; class-peak and canonical-replica caps); work-backed $HIVE issuance with geometric decay under the 21 M cap and an 80/15/5 fee split with burn; Kelly consumer equilibrium (efficiency ≥ 0.75 by the 4/3 price-of-anarchy bound), Tullock worker equilibrium (closed form, ε-Nash certified), damped best-response routing across queues with a deviation certificate. 12 tests, 5 property-based.
- Pre-registered simulation (seed 2026): Kelly efficiency min 0.800; 300/300 Tullock certified; routing 55/60 undamped (hypothesis refuted, run kept) → 60/60 damped; controls passed. Status: proposed, not adopted (`HM-S12-02`).
- ROADMAP §3.10 Season 12 (`HM-S12-01..06`); proposal `docs/COMPUTE_COIN_AND_MARKET.md`.


## v2.6.0 — 2026-10-24 — Operation Hardened-Hive: HiveMindFirewall, chain of command, scripted red team

- `hivemind.security.hivemind_firewall`: keyed-hash identities + Ed25519 delegation certificates rooted at the owner; signed, sequenced, hash-chained orders with nonce burning; deterministic prompt-injection scan (PI-01..07, decoded payloads re-scanned) and data envelopes; pure state machine with QUARANTINED as the default for anything undefined; allow-listed argv sandbox (bubblewrap or BLOCKED); hash-chained audit log. 17 tests (two Hypothesis properties).
- Scripted red team `scripts/run_firewall_pentest_campaign.py` over 59 pre-registered vectors: first run 47/49 attacks blocked (two real defects: percent-encoding threshold, owner-delegation rule order), second run 49/49, 0 false positives, controls passed.
- Gatekeeper records a report-only injection scan of PR text and the GitHub signature-verification status of every commit; policy switch to require signed owner commits. ROADMAP §3.9 Season 11 (`HM-S11-01..08`, campaigns `HM-S11-PT-1..3`); SECURITY.md §5; design in `docs/SECURITY_HARDENING_PHASE.md`.


## v2.5.2 — 2026-10-24 — preprint v3.7.2 (not yet deposited): Lean gate at 128; T4 branch review; main locked to PRs

- Independent Lean gate re-run on the 128-theorem `HiveMindSpec.lean` that reached `main` by direct push (4474802): 128/128, 0 sorry, 0 untrusted axioms; Theorems 125 and 127 flagged as not constraining their mechanism (`HM-S10-09`). Paper inventory table, limitations and README updated to 128.
- Scientific lead's review of the T4 experiment branch before its PR: `docs/review/t4_branch_review_2026-10-24.md`.
- Governance: `main` requires a pull request and enforces admins (PR #125 adds the gatekeeper's merge retry); the shared checkout's git e-mail reset to the coordinator's address.


## v2.5.1 — 2026-10-24 — preprint v3.7.1: E12 pilot complete (bf16 cell)

- E12-pilot P12d supported: bf16 autocast replayed by an fp32 validator diverges to 3.3e-3 with 96/96 tensors above ε, 2.7× the tampered control; dtype is a class boundary (validators assigned within dtype class; bf16 ε to be measured on GPU, `HM-S10-08`). PROV-34; ledger P12d; paper §4.1, §E12 and limitations updated. No other numbers changed.


## v2.5.0 — 2026-10-23 — preprint v3.7.0: E12 replay-divergence pilot; hourly PR gatekeeper

- E12-pilot (pre-registered): the Byzantine spot-check's ε = 1e-4 replay rule fails on CPU for a thread-count change alone (2.6e-4) and for a second host (1.5e-4); bit-identical execution reproduces exactly; tampered control 1.25e-3 with 96/96 tensors above ε. Rule re-specified as execution-class matched with a per-class ε and a tensor-count term. bf16 cell pending at build time.
- `scripts/pr_gatekeeper.py` (hourly, owner-only merge authority; PR text is data); branch protection on `main`.


## v2.4.0 — 2026-10-23 — preprint v3.6.0 after an external review of v3.5.0

- Review recorded verbatim (`paper/review/external_review_v3.5.0_2026-10-23.md`, provenance unstated, no score) with a point-by-point response (`response_to_external_review_v3.5.0.md`).
- Paper v3.6.0: architecture figure (worker inner step; outer WAN round), hardware-heterogeneity caveat and architecture-matched rule for the ε = 1e-4 spot-check (new E12 / RQ11), compounding-approximations and PoC-scale limitations, Skip-Token-as-dropout note, PA-CGAD-under-quorum note. Numbers unchanged.
- Kept in the manuscript, on purpose: the correction that versions 2.9–3.1 labelled a scripted audit a "peer review" (the error is on public records; so is its correction).
- Zenodo: new version of the concept record; Lean: `lean4/mathlib/RDAMultiplier.lean` (3 Mathlib theorems, independent gate).

