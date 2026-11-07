# External Review of Project HiveMind v4.3.0 (`doi:10.5281/zenodo.24043016`) & Planetary-Scale Readiness (Received 2026-11-07)

## Summary Assessment

Project HiveMind v4.3.0 makes an important shift toward radical empirical honesty: acknowledging the post-`hmq_00171` BOINC QLoRA-MoE validation-loss regression in Section 10, separating off-the-shelf served tiers from community-trained adapters in the catalog (`PS-03-D2`), and pre-registering falsifiable gates across ten planetary-scale engineering streams (`PS-01` through `PS-10`). Before promoting the Planetary-Scale Architecture from pre-registered design to empirical validation, ten concrete verification requirements must be satisfied with reproducible receipts:

## 1. Attribution & Remediation of the Post-`hmq_00171` Chain Regression (`PS-01`)
- **Critique:** Since round `hmq_00171`, single-domain workunits with 80 inner steps and unstratified mixing caused cross-domain catastrophic interference across the 6-domain evaluation suite (`math`, `python`, `rust`, `sql`, `cybersec`, `general_reasoning`).
- **Required Evidence:** (a) Verify that the evaluation harness itself has zero drift (`max_abs_noop_delta_nats <= 1e-6` when replaying `hmq_00170`), (b) run a 2x2 factorial attribution experiment (`E24`) isolating the acceptance ratchet, step count (`64` vs `80`), and 6-domain stratified mixing, and (c) demonstrate a 20-round consecutive window (`hmq_00216`..`hmq_00235`) with `median_change_nats <= 0.0` and `share_worse <= 0.50`.

## 2. Enforced Safety & Resource Invariants in Code and Lean 4 (`PS-02`)
- **Critique:** A night probe previously breached the Tesla T4 4,300 MiB free-VRAM floor, and volunteer safety guarantees (explicit owner consent, single-heartbeat pause preemption, no peer-to-peer lateral movement, and bounded network egress) must be enforced in both runtime code and formal specification.
- **Required Evidence:** Enforce the 4,300 MiB VRAM floor in `night_probe_queue.py` and trainer entrypoints (`PS-02-D1`), prove `no_node_starts_without_stored_owner_enrolment` and `pause_implies_no_new_work_within_one_heartbeat` with zero `sorry` in `lean4/HiveMindSpec.lean` alongside an integration test (`PS-02-D2`), enforce an AST + runtime no-lateral-movement CI gate (`PS-02-D3`), and publish per-node-type egress budgets (`PS-02-D4`).

## 3. Receipt-Backed Public Claims & Prompt-Visibility Notice (`PS-03`)
- **Critique:** Every public page offering or documenting the chat API must state clearly that volunteer producer nodes process cleartext prompts, and every benchmark number must cite an existing repository receipt file without exposing any personal data.
- **Required Evidence:** Publish a zero-PII accounts receipt (`PS-03-D1`), strip unverified trained labels from off-the-shelf tiers (`PS-03-D2`), enforce a CI test (`tests/test_public_claims_have_receipts.py`) checking the PoC notice and receipt paths (`PS-03-D3`), and keep the README top TL;DR concise (`<= 14` lines, `PS-03-D4`).

## 4. Volunteer OpenTrainer Ergonomics & Telemetry-Driven Economics (`PS-04`)
- **Critique:** The economics model previously relied on static `18.0 h/day` idle and `175.0 W` power assumptions rather than measured volunteer telemetry.
- **Required Evidence:** Demonstrate clean-environment install-to-first-workunit in `< 900 s` (`PS-04-D1`), measure VRAM peak (`< ceiling_mib`) and game-launch pause latency (`< 1,500 ms`) across at least 3 distinct GPU classes (`PS-04-D2`), and compute the consumer GPU economics receipt from `n >= 10` enrolled trainers (`PS-04-D3`).

## 5. Sub-Unity Verification Overhead via Canary Probes & Stratified Auditing (`PS-05`)
- **Critique:** Full bit-exact replay of every workunit yields a verification-to-training FLOP ratio $\ge 1.0$, negating the throughput gain of volunteer training.
- **Required Evidence:** Implement deterministic canary probes catching `>= 95%` of planted sign-flip, random-noise, zero-delta, scale-inflation, and stale-replay faults at `<= 5%` false-positive rate (`PS-05-D1`), and execute pre-registered experiment `E21` showing `verification_to_training_flop_ratio <= 0.25` (`PS-05-D2`, `PS-05-D3`).

## 6. Residential Uplink Compression (`E26`, `PS-06`)
- **Critique:** Top-1% sparse + INT8 + EF21 error-feedback compression must be benchmarked across realistic asymmetric residential broadband and cellular uplinks (`5` to `50 Mbps` up) without degrading held-out bits-per-byte (`bpb`) beyond `+0.010 bpb`.
- **Required Evidence:** Commit `E26` pre-registration and report across `>= 5` residential link profiles demonstrating `>= 50x` compression and `max_bpb_delta_vs_control <= 0.010` (`PS-06-D1`, `PS-06-D2`).

## 7. Decentralized Peer Discovery (`E22`) & W3C `did:key` Identity Bridge (`PS-07`)
- **Critique:** Relying on a single bootstrap coordinator creates a single point of failure, and node identity across HiveMind and KwaaiNet requires a shared cryptographic credential format.
- **Required Evidence:** Implement `did:key:z6Mk...` (`Ed25519Signature2020`) verifiable credentials with shared test vectors (`PS-07-D1`), demonstrate `< 30 s` p95 peer discovery when the primary bootstrap node is offline (`E22`, `PS-07-D2`), and record a Petals/libp2p swarm liveness receipt (`PS-07-D3`).

## 8. AMCP Event Mesh, Read-Only Credit Ledger & Agent Mobility (`E23`, `PS-08`)
- **Critique:** Workunit validation/assimilation notifications and agent checkpoint handoffs across NAT boundaries must not permit mutations of the BOINC credit ledger.
- **Required Evidence:** Publish `validated` and `assimilated` events over AMCP topics with read-only ledger enforcement (`PS-08-D1`), verify exact zero-drift credit reconciliation across 100 workunits (`PS-08-D2`), and demonstrate `>= 99%` state continuity across 100 NAT handoffs with `< 5 s` p95 resume latency (`E23`, `PS-08-D3`).

## 9. 1,000-Node Heterogeneous Byzantine Testbed (`E27`, `PS-09`)
- **Critique:** Prior `MESH-SCALE-1` runs tested only 1 run per size without adversarial gradient poisoning.
- **Required Evidence:** Execute `>= 3` seeds at each scale (`N in {100, 300, 1000}`) with CUDA/ROCm/Metal hardware profiles, residential link traces, and up to `30%` Byzantine nodes, maintaining `<= 1,800 s` epoch convergence, `<= 15%` verifier overhead, `<= 250 ms` p95 routing latency, and `0` coordinator OOMs within the `$150/month` budget (`PS-09-D1`, `PS-09-D2`).

## 10. Reproducible Cost Control & Open-Weights Resilience (`PS-10`)
- **Critique:** Monthly cloud spend must remain at or below `$150/month` under the scheduled T4 training window and shrunk disks, and all base models (`Granite`, `Qwen`, `Gemma`) must have verified local/GCS mirror checksums and permissive/open licenses.
- **Required Evidence:** Verify Zenodo deposit checksums (`PS-10-D1`), produce a passing `<= $150/month` cost receipt from `scripts/d1_cost_estimate.py` (`PS-10-D2`), verify base-weights mirror SHA-256 checksums (`PS-10-D3`), and archive this external review with a point-by-point response in `paper/review/` (`PS-10-D4`).
