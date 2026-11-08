# HiveMind Improvement Plan (2026-11-05 update)

> Successor to [`IMPROVEMENT_PLAN_2026-11-04.md`](IMPROVEMENT_PLAN_2026-11-04.md). Daily read-only audit of `main` at `816badd` (2026-11-05, about 10:30 CET): five parallel area audits for **new findings only** (IP-1 to IP-269, `docs/MVP_REALITY_AUDIT.md` and the open owner decisions excluded by brief). Nothing is implemented here; the items are for others to claim. **R** = re-checked by the author in this run (executed or read against the code); **A** = agent-reported with the quoted or executed evidence the agent gave, not re-run by the author.
>
> **Status on `main` (2026-11-07):** Every non-protected item in this plan (`IP-270..IP-281`, `IP-283..IP-292`) is closed on `main` (only `IP-282` in `src/hivemind/security/hivemind_firewall.py` is an owner-only protected path under `scripts/pr_gatekeeper.py`):
> - `IP-270`, `IP-271` (PoUW validator pool tied to verified work/storage/spot-check receipts; cumulative minted supply cap across slashing): closed in `#702`.
> - `IP-272`, `IP-273`, `IP-274`, `IP-275` (GWAYA multi-statement stub detection, DHT lease NaN time guards, DHT chunk CID digest verification on path load, DARL finite positive `alpha`): closed in `#699`.
> - `IP-276`, `IP-277`, `IP-278`, `IP-279` (threadpool + work budget + per-IP limiters on compute routes, `bees/trade` sliding-window limiter & open-order cap, 400 validation on malformed anonymous input, Cloud Run `--min-instances 0 --max-instances 1` doc alignment): closed in `#703`.
> - `IP-280`, `IP-281`, `IP-283`, `IP-284`, `IP-285` (`KnowledgeMarket` stateful per-epoch dedupe/cap, `LatticeQuantizer` finite guard, `FP8E4M3PageCodec` finite/underflow guard, `write_kv_page` encode-before-evict, `compress_tensor_topk_fp8` finite guard): closed in `#704`.
> - `IP-286`, `IP-287`, `IP-288`, `IP-289`, `IP-290` (decoupled exoplanet discovery dedupe set + tag cap, `planet-forecast` `mean_tokens_per_req` 422 validation, `compute_coin` finite & workunit dedupe guards, `swarm_lab` finite scenario validation, generated token counting & authenticated producer registration): closed in `#700` and `#701`.
> - `IP-291`, `IP-292` (`rtx2070_failover_orchestrator.py` live GPU probe without hardcoded telemetry literals, fail-closed `opentrainer_heartbeat_ok`): closed in `#705`.

## 1. Closed or narrowed since the 2026-11-04 plan

| Item | Status | By |
| :-- | :-- | :-- |
| IP-250, IP-251 (scanner missed most key formats, echoed the match) | closed: the formats are detected and the evidence is a label, line and hash | 35064fe |
| IP-252 (fullwidth directive scanned CLEAN) | closed for NFKC; hidden-character forms are IP-282 below | ab9477f |
| IP-253 (KV page codecs overflow/underflow) | closed for the int8 and lattice codecs; the FP8 codec is IP-283 below | #597, #611 |
| IP-254 (CrowdGPT loss check accepts an empty recompute) | closed | #579 |
| IP-255, IP-256 (DARL NaN times, eco ledger NaN) | closed | c2863d2, 743dac5 |
| IP-257 (docstring-only body certified) | closed for the single-statement case; wider stub shapes are IP-272 below | #605, 5ffbdd3 |
| IP-258 (fee split rounding) | closed: integer micro-units | #594 |
| IP-259 (OpenTrainer delta replay and rate) | closed: 409 on replay, 20 per minute per account | #616 |
| IP-260 (bee capacity market) | narrowed: an account's orders never match each other (#617); escrow, `bee_id` ownership and an open-order cap remain (issue #577) | #617 |
| IP-261, IP-262 (campaign poll event and rate, usage-capture scan) | closed | #606, #613, #612 |
| IP-263, IP-264 (Erlang-C overflow, route_supply NaN) | closed | #596, #593 |
| IP-269 (TTS voice name 500) | closed: 400 for a malformed voice or lang | #621 |
| Not IP-numbered | failed-producer log names the reason (#602, #620; effective on the live router only after a redeploy); campaign poll limiter; `--retired-models` monitor flag (#607) | |

Still open from the previous plan: IP-265, IP-266, IP-267, IP-268.

## 2. Actions that need the owner (not PRs)

1. **Close the stale issues** fixed on main: #513 (OpenTrainer status; the `/boinc/status` Ruche probe part was not re-checked), #519, #520, #523, #554, #573, #574, #575, #582, #583. The agent token cannot comment or close.
2. **Decide the PR-gate owner e-mail change** (#512 with #511): the workflow calls `audit_pull_request` without `author_email`, so fixing the scanner alone would make every owner PR unverified.
3. **Rebase or close the four PRs the gatekeeper marks ACCEPT but never merges**: #442, #434, #369, #366 (issue #569).
4. **Authentication for the text-to-speech route (#522) and the monitor telemetry route (#521)**; the producer pool registration in IP-290 depends on the same decision.
5. **Redeploy the router** (owner call) so the 502 logging from #602 and #620 takes effect.

## 3. Counts

Raw agent findings: 23 across five areas (swarm 5, router 5, game/economy/lab 5, core/security 6, tests/CI/scripts/Lean 2). Recorded: 23 rows (P0 0, P1 9, P2 9, P3 5); R 7 (IP-270, IP-272, IP-278, IP-279, IP-281, IP-288, IP-291), A 16. Two splits and one fold: the swarm agent's first finding became IP-270 and IP-271, and the router agent's "thin tests" item is folded into the negative controls of IP-276 to IP-278. Dropped as already covered or too weak: the bare `pytest.raises(Exception)` tests, the 30 ms TTL margin in `test_swarm_nectar_cache.py`, the cockpit `best_period_days` literal (same family as IP-28/89). The Lean audit found no `sorry`, `admit` or user `axiom` in `lean4/HiveMindSpec.lean` and all 161 theorems covered by `#print axioms`; no secret literals in tests, scripts, workflows or Lean files; no `continue-on-error` or `|| true` in the workflows. The client and cli packages were not reviewed.

Table columns: id | R/A | area | finding with file:line and evidence | Definition of Done | negative control (fails on `main` today) | proposed PR.

## P1 — wrong results, minting, unsafe defaults

| IP-id | R/A | Area | Finding | Definition of Done | Negative control | PR |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| IP-270 | R | swarm | `src/hivemind/swarm/ledger.py:258-266`: `mine_pouw_block` credits `validator_pool / len(unique_validators)` to every id in `validator_node_ids`, a bare list of strings; no spot check, receipt or identity is required. Executed: three `mine_pouw_block(i, "cid://x", [], [], ["sybil"])` calls gave `get_balance("sybil") == 15.0` and `verify_chain_integrity() == True`. | validators earn only when tied to a verified work, storage or spot-check receipt in the block; a block with none mints 0 | three empty-receipt blocks with validator `sybil` leave its balance 0 | PR-IP-31 |
| IP-271 | A | swarm | `ledger.py:150-152` (`block_reward_at_height`): the supply cap is measured on live balances, so a balance at the cap gives reward 0.0 and halving it (the slashing path) gives 50.0 again; slashed coins become mintable. | cap against cumulative minted supply, not balances | set a balance to `MAX_SUPPLY_HIVE`, halve it, assert the reward at height 5 is 0.0 (50.0 today) | PR-IP-31 |
| IP-276 | A | router | `gateway.py` about line 3591: `POST /v1/hivemind/bees/rusty-sundials/sweep` is anonymous and `async def` running `run_wave2_parameter_sweep_quorum` inline (no `run_in_threadpool`, unlike `/rescue`): 13.56 s of blocked loop for `{"steps":128,"parameter_grid":[0.1]*16}`; `science-bee/ask`, `slot-compile`, `scale-100-simulate` (1.12 s at 1000/5000) and `surrogate` do inline compute too. One instance serves everything. | these handlers use `run_in_threadpool` plus a per-IP limiter or auth; the sweep is bounded by a work budget | start the sweep and assert `GET /v1/models` answers in under 1 s meanwhile | PR-IP-32 |
| IP-277 | A | router | `gateway.py:2832`: `POST /v1/hivemind/bees/trade` has no limiter and no per-account cap; each call re-clears the whole 24 h book and writes a never-pruned event. One consumer posted 480 accepted orders in 14.69 s, 0 x 429; `GET /campaigns` and `/bees/market` embed the same clearing. (A different defect from #617; part of the open-order cap in #577.) | per-account sliding-window limiter, per-account open-order cap, clearing cached or capped | 100 orders from one key: some answer 429 | PR-IP-32 |
| IP-280 | A | queenbee | `src/hivemind/queenbee/knowledge_market.py:59-81`: `admit_share` is stateless; the per-identity cap is the caller-supplied `prior_points_this_epoch`, nothing dedups on `(account_id, text_sha256)`, `epoch` is never compared. 100 identical calls gave `knowledge_points == {"u1": 100}`; `settle_knowledge_bonus` then paid u1 49.50 of a 50.0 pool. | admission tracks per-epoch counts, rejects a repeated `text_sha256`, rejects an epoch that differs from the settlement epoch | admit the same share 3 times: points == 1 (3 today) | PR-IP-33 |
| IP-281 | R | compress | `src/hivemind/compress/lattice_quantizer.py:195-197, 215-217`: `rel = ... if vn > 0.0 else 0.0` with `vn` NaN reports `relative_mse 0.0`; with `error_feedback=True` the NaN residual is stored in `_carry` and poisons every later call; no `isfinite` check in the file. Executed: E8, scale 0.1, one NaN in 16 elements gave `relative_mse 0.0`. | `quantize` raises `ValueError` on non-finite input and never stores a non-finite residual | a vector with a NaN raises; NaN then clean data stays finite with EF on | PR-IP-33 |
| IP-286 | A | game | `game/exoplanet_hunter_engine.py:1801-1805, 1853`: the IP-202 dedupe reads only the retained ledger (`ledger_entries[:200]`); `account_key` is the free-text `gamer_tag` with no length cap. 200 junk tags minted 120,000 points and evicted a real row, so the same real tag earned 600 again; a 1 MB tag wrote a 2.1 MB ledger file. | awards tied to an authenticated account, `gamer_tag` and `notes` capped, dedupe kept apart from the display ledger | Alice once, 200 junk tags, Alice again: second call earns 0 (600 today) | PR-IP-27 |
| IP-287 | A | economy | `economy/international_launch_model.py:792-820` via `boinc/hivemind_at_home_dashboard.py:9250-9307`: `mean_tokens_per_req` is not clamped on the public `planet-forecast` route; `0`, `nan`, `1e308`, `-1` gave 500 (1e308 returns inf in 15 fields). Different parameter from IP-248. | clamp to a finite range, map `ValueError` to 422 | GET with `0`, `nan`, `1e308`: 422, or 200 with every field finite | PR-IP-26 |
| IP-291 | R | scripts | `scripts/rtx2070_failover_orchestrator.py:774-785`: posts invented GPU telemetry to `/v1/hivemind/t4/telemetry` with the producer key (`used_mib` 1812/4841, `total_mib` 8192/15360, `temp_c` 34.0/66.0, `util_pct` 12.0, `power_w` 42.04/30.2, `hold_file_present` False, `in_flight_requests` 0); only the `t4_reachable` flag is live. The router side (IP-143) now reads these keys, so the literals reach the public pages as T4 figures. | read NVML or `detect_gpu`; send nothing (BLOCKED) when the probe fails | stub the probe to different used/free/temp: the POST body changes; no readable GPU: no POST | PR-IP-24 |

## P2 — correctness and gates that cannot fail

| IP-id | R/A | Area | Finding | Definition of Done | Negative control | PR |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| IP-272 | R | swarm | `swarm/s35_02_gwaya_laya_failclosed_gate.py:87` (`if len(body) == 1:`): the AST anti-stub check returns `[]` for `def f(x): return None`, `x=1; raise NotImplementedError('later')`, `class A: pass`, `return NotImplemented`, `return ...` after a docstring and a constant lambda. Executed for the first three. | flag a last or only effective statement that is `pass`, `...`, a `NotImplementedError`/`NotImplemented` raise or a bare/`None` return with an unused argument; flag pass-only classes and constant lambdas | the six snippets each return a non-empty list | PR-IP-25 |
| IP-273 | A | swarm | `swarm/dht_storage.py:738-739, 761-762, 778-781`: lease heartbeat, commit and reclaim compare `now > expires_at_s`; with `now_s=nan` every comparison is False, so a lease granted at 0.0 commits as `committed`. The IP-255 defect, unfixed here. | `ValueError` on non-finite `now_s` in grant, heartbeat, commit, reclaim | grant at 0.0, commit with `nan` raises | PR-IP-25 |
| IP-274 | A | swarm | `swarm/dht_storage.py:293`: the digest check runs only when the payload's own `cid` starts with `cid://sha256/expert-`; on a path load nothing is checked. A checkpoint saved with `cid="forged"` loads. (`weights_only=True` still blocks pickle execution.) | reject a payload whose `cid` does not match the pattern; always recompute and compare the digest | a `cid="forged"` checkpoint raises `ChunkIntegrityError` | PR-IP-25 |
| IP-278 | R | router | `gateway.py` (about 1818, `rusty-sundials/rescue`, `science-bee/ask` at 3694): malformed anonymous input is a 500, not a 400. Executed with TestClient: `scale-100-simulate {"active_producers":"x"}`, `rescue {"steps":"x"}` and `science-bee/ask` with a 4001-character question each gave 500; a JSON list body to `surrogate`/`sweep` gave 500 (agent-reported). | parse through a model or shared helper; invalid input is 400 | parametrised POSTs of those cases answer 400 | PR-IP-32 |
| IP-282 | A | security | `security/hivemind_firewall.py:394-429, 828-835`; `firewall_pentest_corpus.py:49`: a directive split by a zero-width space or soft hyphen is only `SUSPICIOUS` (rule PI-03, not in the hard set) and continues to `SCAN_CLEAN`; the corpus expects `SUSPICIOUS_OR_INJECTION`. HTML-comment and spaced-letter forms scan CLEAN. | also scan the hidden-character-stripped form and raise to `INJECTION`; tighten the corpus expectation | `scan_text_for_injection` of the zero-width-split directive is `INJECTION` | PR-IP-29 |
| IP-283 | A | core | `core/kv_page_codecs.py:42-54`: `FP8E4M3PageCodec` (the default of `PluggableKVPageCache`) has none of the guards the other codecs got in #597/#611: a page with a NaN decodes with a non-finite row and still reports `payload_bits = 2112`; `randn*1e-9` decodes to zeros. | `encode` calls both guards | NaN page and 1e-9 page raise `ValueError` | PR-IP-33 |
| IP-288 | R | economy | `economy/compute_coin.py:163-185, 127-131`: `settle_epoch` and `EmissionSchedule` check only `< 0.0`. Executed: a grant of `inf` cobblestones gave `minted_by_node == {'n': nan}`; per the agent a NaN payment gives NaN fees and `initial_hive_per_cobblestone=nan/inf` is accepted; the same grant list passed twice mints twice. | reject non-finite inputs, drop non-finite grants, dedupe by `(node_id, workunit_id)` | `inf` grant and `nan` payment raise or return finite values | PR-IP-26 |
| IP-289 | A | lab | `lab/swarm_lab.py:119-135`: `Scenario.validate` uses comparisons NaN passes; `eta=nan`, `sigma=inf`, `outer_lr=nan` give `final_loss nan`, `eta=50` gives `inf`, `grace=nan` raises an uncaught `IndexError`; `--json` prints `NaN`. | `math.isfinite` on every float knob; `simulate` raises when the loss curve is non-finite | `Scenario(eta=float('nan')).validate()` raises | PR-IP-26 |
| IP-290 | A | game | `game/exoplanet_hunter_engine.py:2117, 1998-2026`; `cockpit.py:3126-3143, 3190`: `total_tokens_processed += input_token_count + int(max_new_tokens)` counts the requested, not the generated, tokens (`max_new_tokens=10**9` gave a counter of 1000000002 for 191 characters of text); `/api/v1/producers/register` is unauthenticated, uncapped and can set the anchor producer `offline`; 300 registrations rewrote the whole registry each time. | count generated tokens, clamp `max_new_tokens`, authenticate and cap registration | `serve_inference(..., max_new_tokens=10**9)` then `total_tokens_processed < 10_000`; 1,000 registrations answer 401/413 | PR-IP-27 |

## P3 — hygiene

| IP-id | R/A | Area | Finding | Definition of Done | Negative control | PR |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| IP-275 | A | swarm | `swarm/s36_05_darl_cas_shard_leasing.py:61, 74`: `alpha` is not validated; `alpha=inf` gives `expires_at_s=inf` (never reclaimed), `alpha=-5` or NaN clamps to 1.0. | constructor requires finite alpha > 0 | `alpha=float('inf')` raises | PR-IP-25 |
| IP-279 | R | deploy | `docs/HIVEMIND_OPENROUTER_SPEC.md:105`, `wiki/HiveMind-at-Home-GCP-Dashboard-and-La-Ruche-Vault.md:20`, `.agents/skills/gcp-cost-autoscaler-ops/SKILL.md:40, 48` disagree with `deploy/gcp_openrouter/deploy.sh:58` (`--max-instances 1`, also required by `AGENTS.md:17`): they give `--max-instances=2` or `--min-instances=1`; two writers on one Litestream ledger. Grep-confirmed for the wiki page and the skill file; the spec line was matched but not read in full. | all four state `--min-instances 0 --max-instances 1` | `scripts/mvp_scorecard.py` greps those docs for `max-instances[ =]2` and fails | PR-IP-32 |
| IP-284 | A | core | `kv_page_codecs.py` about 170-177: `write_kv_page` evicts the oldest page before it encodes, so a rejected NaN write leaves the cache empty. | encode first, then evict and store | a good page, a NaN write inside `raises`, then the first page is still cached | PR-IP-33 |
| IP-285 | A | core | `core/compression.py:130-140`: `compress_tensor_topk_fp8` with one NaN or inf gives a NaN scale and a payload that is all NaN; the single inf is not preserved. | raise `ValueError` on non-finite input (also the DCT 2-bit variant) | `compress_tensor_topk_fp8(t_with_inf)` raises | PR-IP-33 |
| IP-292 | A | scripts | `scripts/rtx2070_failover_orchestrator.py:798`: `"opentrainer_heartbeat_ok": bool(hb_ack.get("ok", True))` reports OK when the ack is missing; `send_heartbeat`'s own error handling was not read. | default to `False` | `send_heartbeat` returning `{}` reports `False` | PR-IP-24 |

## 4. Proposed pull requests (new) and issues to open

New groups: PR-IP-31 (ledger minting and supply cap: IP-270, IP-271), PR-IP-32 (router anonymous routes: IP-276, IP-277, IP-278, IP-279), PR-IP-33 (compress/KV/knowledge market: IP-280, IP-281, IP-283, IP-284, IP-285). The other rows join PR-IP-24 to PR-IP-29 as listed. Issues opened by this run are listed in roadmap section 3.26f.

## 5. Limits of this update

- Read-only; nothing implemented. Only the R rows were run or read by the author; the A rows carry the agent's quoted or executed evidence. The R re-checks were run in a scratch process against a read-only worktree.
- The live services were not probed. The client and cli packages were not reviewed; the swarm audit read about 1,500 lines of the package; the scripts audit sampled by grep.
- The audit agents ran scratch scripts against a read-only worktree; nothing was written inside it.
