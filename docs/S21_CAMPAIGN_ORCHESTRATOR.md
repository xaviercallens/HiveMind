# S21 `orch`: campaign joins and the night dream phase

Code: `src/hivemind/swarm/campaign_orchestrator.py`, CLI `scripts/dream_phase_plan.py`, tests `tests/test_campaign_orchestrator.py`.
Status: planning and admission only. Nothing here trains, deploys or touches a GPU, a VM or a BOINC server.

## 1. Joining a campaign

The six campaigns are the existing `JOINT_SUBJECT_CAMPAIGNS` (`amcp_kafka_campaign_bus.py`). A participant sends a
`CampaignRequest`:

| field | meaning |
|---|---|
| `action` | `join` or `leave` |
| `participant_id`, `participant_kind` | `human_researcher`, `gpu_producer`, `ai_agent` or `bee_specialist` |
| `campaign_id` | a `SubjectCampaign.subject_id` |
| `declared_vram_mib`, `availability` | as declared; availability is one of the poll's four slots |
| `issued_at_utc`, `nonce` | freshness and replay protection |
| `signature` | HMAC-SHA256 (hex) over the canonical JSON of every other field (sorted keys, no spaces) |

`CampaignOrchestrator.handle()` checks, in this order, and stops at the first failure:
field validity, `|now - issued_at_utc| <= 900 s`, signature, nonce, campaign exists, campaign open now
(`dtstart - join_lead_s <= now < dtend`; `join_lead_s` is 0 by default), the participant's quota of concurrent campaigns (2),
the campaign's capacity (64). The limits are `CampaignPolicy` defaults, not measured values.

Idempotence and replay:
- a join from an active member returns `ALREADY_MEMBER` and changes nothing;
- an exact resend (same nonce, same content) returns the stored decision and emits no second event;
- the same nonce with different content is `REJECTED`;
- a forged request is rejected before its nonce is recorded, so it cannot burn a genuine participant's nonce.

`leave` returns `LEFT` or `NOT_MEMBER`; `expire(now)` turns memberships whose campaign has ended into `expired`.
Every decision, rejections included, is published as an AMCP event on `hivemind.amcp.campaigns.v1` with type
`io.hivemind.amcp.campaign.<decision>`, the participant as recipient and the nonce as correlation id.

**Transport interface.** The orchestrator takes one callable, `publish(AMCPKafkaEvent) -> metadata`. Today that is
`InMemoryKafkaBridge().publish`. The S21 `kafka` PR's transport plugs in the same way (follow-up wiring, one line at the call
site). If publishing raises, the decision stands (it is already persisted) and `delivery.status` is `FAILED` with the error.

**State** is one JSON file (`schema_version: 1`, `memberships`, `nonces`), written atomically: temp file in the same directory,
`fsync`, `os.replace`. A file with the wrong schema is refused (`ValueError`), not repaired. Nonces older than twice the skew
window are pruned.

## 2. The dream phase planner

`plan_dream_phase(now_utc, members, ledger, hold_file, governor_state_path, policy)` returns a plan dict; the CLI prints it.

**Night window.** 22:00-07:00 Europe/Paris by default (the same window as `scripts/t4_governor.py`), built with `zoneinfo` and
measured in UTC, so its length is 9 h on ordinary nights, 8 h on the night into the last Sunday of March and 10 h on the night into
the last Sunday of October. (Subtracting two datetimes that share one `ZoneInfo` gives wall-clock time and misses the DST hour; the
code subtracts in UTC, and the tests for 2026-03-28/29 and 2026-10-24/25 fail if that is changed.) At any instant the planner uses
the window that contains it, or else the next one; `usable_h` is the time from now (or the window start) to the window end.

**Episode ledger** (`--ledger`): `{"domains": [{"domain", "verified_accept", "verified_reject", "unverified",
"heldout_eval_items", "task_kind", "consumed_verified"}]}`. The counts must be real verdict counts from the verification gates;
this module does not produce them. Producing the ledger from the episode store, and the dataset licences and decontamination
behind `heldout_eval_items`, belong to the S21 `datasets` PR.

**Domain selection.** A domain is refused when its new verified episodes (`verified_accept - consumed_verified`) are below
`min_new_verified` (200), or its held-out eval set is below `min_heldout_eval_items` (300, the per-pack n NIGHT-MOE-2 asks for the
primary paired test). The unverified rows mixed in are capped at 30% of the training mix, computed in integers:
`u_max = floor(3 v / 7)`, the largest `u` with `u / (v + u) <= 0.3` (a Hypothesis test checks both that bound and that `u_max + 1`
breaks it). Rows above the cap are dropped and the count is reported. Domains are ranked by new verified episodes; at most
`max_jobs` (8) are selected and the rest are `DEFERRED`.

**Controls.** Every job has a shuffled-label control (same rows, recipe and seed; labels permuted). NIGHT-MOE-1 showed that for a
generative task this control cannot fail (a shuffled numeric answer is never right), so a `generative` domain also gets a
cross-domain control (same recipe on another qualifying domain's rows), marked as the primary comparison; a generative domain with
no other qualifying domain is refused. Each job then has an `eval_heldout` stage (base, adapter and every control, exact paired
McNemar, alpha 0.05, unparsable counts reported) and a `promotion_gate` with `auto_promote: false` that lists what must hold before
promotion: adapter beats the primary control, adapter beats base on items both parsed, the verified update gate on the other packs,
and a decontamination report.

**Assignment.** Eligible producers are active `gpu_producer` memberships whose availability includes `eu_evening_cet` or
`24x7_autonomous` and whose declared VRAM fits a BOINC training profile (`profiles_for_gpu`: `big` from 10000 MiB, `standard` from
3500 MiB). A producer in several campaigns is planned with the smallest VRAM it declared. Each job's budget is
`train_cap_h + n_controls * control_cap_h + (1 + n_controls) * eval_cap_h` (1.0, 0.25, 0.75 h by default: caps taken from the
NIGHT-MOE-1 orchestrator plus an eval allowance, not measured durations). A job goes to a producer with that much night left,
preferring producers joined to a campaign whose `domain_key` is the domain, then the least loaded; one producer's jobs run back to
back. The job carries the profile's BOINC app, plan class, workunit prefix, minimum GPU RAM and LoRA rank.

BOINC `BOINCWorkunit` specs are **not** emitted: a workunit needs the SHA-256 and HMAC of a real Parquet shard, and no shard exists
at planning time. The plan names the app and prefix so the work generator can create the workunits once the shard is built.

**BLOCKED.** The plan is `BLOCKED`, with no jobs and every reason listed, when any of these holds: the HOLD file exists; the
governor state file is missing, unparsable, has no `state` or `t`, is older than 900 s, or says `held`; no domain qualifies; no
selected domain could be assigned to a producer. The CLI also blocks when the campaign state or the ledger cannot be read. Exit
status 0 = PLANNED, 2 = BLOCKED. `--dry-run` is required and is the only mode.

## 3. Measured

- `tests/test_campaign_orchestrator.py`: 48 tests pass locally (see the PR for the command and output).
- The CLI on this workstation, 2026-10-28, with no state, ledger or governor file present: exit 2, `BLOCKED` with four reasons
  (campaign state, ledger, governor state not readable; no domain qualifies); night window 2026-10-28T21:00Z to
  2026-10-29T06:00Z, 9.0 h. This is the expected result off the producer host, not a plan.

## 4. Not done

- No real ledger exists yet; no plan has been made from real counts.
- Not wired to the S21 Kafka transport, the expert registry (`moescale`) or the dataset registry (`datasets`); see the PR body.
- No HTTP endpoint for joins; the router does not call the orchestrator yet.
- Nothing executes a plan. Executing one stays with `scripts/night_orchestrator.py`-style runners under their own preconditions.
