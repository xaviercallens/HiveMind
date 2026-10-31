# E8 on two real hosts: T4 and RTX 2070

Instructions for the session that runs on the **RTX 2070 machine** (and for whoever runs the T4 side), derived from the Foundation Paper v4.1.0 (Zenodo 10.5281/zenodo.23776302, LaTeX source `hivemind_foundation_article.tex`): experiment **E8** (multi-host WAN, RQ1) and its first pilot, **E8-pilot** (paper section "First two-host measurement"). Written 2026-10-31.

## 1. What the paper requires, and what this test is (and is not)

E8 asks one question: what does a decoupled outer round cost over a real network, against single-host workers on the same hardware. Its own design names the controls: real tokens, `K=ceil(M/2)` (later), adaptive grace, RDA, **a localhost negative control** ("the same learners over localhost must match single-host within CI, otherwise the gap is not the network"), and baselines on the same hardware. The pilot was one learner on the author's workstation plus one on a Cloud TPU VM (CPU-only), three seeds, 12 outer rounds, the E1 toy model, `scripts/run_experiment_E8_pilot.py`.

**This document asks for Stage 1: repeat the pilot's protocol-validation cell between the T4 and the RTX 2070 over a private tunnel.** It tests the network protocol (shard and parameter shipping, remote inner loop, RDA merge) across two real machines. Be exact about its limits:

* The pilot's learners use the **toy model on CPU**. Stage 1 does **not** exercise either GPU, does not train a LoRA, and says nothing about VRAM. Do not describe it as a GPU or QLoRA result.
* With **two** learners, `multi_krum_select_and_aggregate` (`src/hivemind/swarm/crowdgpt_bridge.py`) falls back to `f_byzantine = 0` when `n < 2f+3` and keeps everyone. A sign-flipped delta would **not** be blocked. Byzantine filtering cannot be tested with two learners: it needs at least five deltas for `f=1` (paper E9 uses `f/n` of 0.1 to 0.3). Do not run a "positive control" for Multi-Krum at `n=2` and do not report it.
* `K=ceil(M/2)` was refuted at this size by the paper's own E13 (single GPU), and the pilot's `K=1` cells failed instructively. Use `K=2` (wait for both), as the pilot's validation cell did.
* Both machines are Turing (sm_75): this is not a heterogeneous-hardware result.

Stage 2 (a separate PR, new code): a real LoRA inner phase on each host that exports a delta, a transfer, and aggregation with at least five learners. Not specified here beyond the guards in section 6.

## 2. The numbers

| | Value |
|---|---|
| Overlay network | WireGuard `10.77.0.0/24` (UDP 51820) |
| T4 (this project's T4 VM) | overlay `10.77.0.4`, interface `wg-hivemind` already up |
| RTX 2070 machine | overlay **`10.77.0.5`** (node id 5, reserved) |
| Syncer | on the T4, `10.77.0.4:8765` (HTTP, no authentication: overlay address only) |
| Learner 0 / learner 1 | T4 / RTX 2070 |
| Protocol cell | `K=2`, `--grace off`, `--merge rda`, 12 rounds, `HM_INNER_STEPS=400`, seeds 1, 2, 3 |
| Hosts | two separate machines: the T4 VM has one Tesla T4 and the "RTX 2070 agent" running on it is **not** a second host |

## 3. Federating the 2070 (what a person with project access must do first)

The 2070 only needs a tunnel to the T4 for Stage 1. The Kafka quorum is not involved.

1. **Open UDP 51820 to the T4 (operator, needs approval):** `gcloud compute instances add-tags socreateai-agora-hermes-node1 --zone us-east4-b --tags hm-kafka`. The existing rule `hm-kafka-wireguard` then allows UDP 51820 to it (authenticated by key; no other port).
2. **On the 2070 (this session):** generate a key pair (`wg genkey`), keep the private key in a root-only file, and send the **public** key to the operator. Never paste or commit a private key.
3. **On the T4 (operator):** `sudo wg set wg-hivemind peer <2070 public key> allowed-ips 10.77.0.5/32`, and add the same `[Peer]` block to `/etc/wireguard/wg-hivemind.conf` so it survives a restart.
4. **On the 2070:** `/etc/wireguard/wg-hivemind.conf` with `Address = 10.77.0.5/24`, a single `[Peer]` for the T4 (`AllowedIPs = 10.77.0.4/32`, `Endpoint = <T4 public IP>:51820`, `PersistentKeepalive = 25`), then `wg-quick up wg-hivemind`. Check: `ping 10.77.0.4` answers and `wg show` reports a recent handshake.

The T4's public address is whatever `gcloud compute instances describe socreateai-agora-hermes-node1 --zone us-east4-b` reports as `natIP`; the operator gives it to the 2070 session.

## 4. Running the cell

Same repository commit on both machines (this document's PR merged), same Python, torch present on both, **the same `HM_INNER_STEPS` on syncer, learners and baseline**.

On the T4 (per seed `S` in 1, 2, 3):

```bash
HM_INNER_STEPS=400 python scripts/run_experiment_E8_pilot.py syncer --bind 10.77.0.4 --port 8765 --learners 2 --k 2 --merge rda --grace off --rounds 12 --seed S --timeout 1800 --out artifacts/experiments/e8_real/wan_seedS_rda.json
HM_INNER_STEPS=400 python scripts/run_experiment_E8_pilot.py learner --syncer http://10.77.0.4:8765 --learner-id 0 --host-label t4-socreateai-agora-hermes-node1
```

On the 2070:

```bash
HM_INNER_STEPS=400 python scripts/run_experiment_E8_pilot.py learner --syncer http://10.77.0.4:8765 --learner-id 1 --host-label rtx2070-<hostname>
```

Then the single-host baseline with identical seed and hyper-parameters (T4 or either machine): `HM_INNER_STEPS=400 python scripts/run_experiment_E8_pilot.py baseline --learners 2 --merge rda --rounds 12 --seed S --out artifacts/experiments/e8_real/baseline_seedS_rda.json`.
The **negative control** (required by the paper): the same two learners over loopback on one machine (syncer `--bind 127.0.0.1`, both learners `http://127.0.0.1:8765`) must match the baseline; if it does not, any cross-host difference is not the network.

`--bind` is new in this PR: the syncer used to listen on every interface with no authentication, and it now refuses `0.0.0.0`. Payloads are loaded with `torch.load(weights_only=True)` so a peer cannot ship code.

## 5. What to report

Raw output, not a summary: the three `wan_seed*_rda.json`, the baselines, the loopback control, `wg show` handshake lines, and the round latency distribution (p50, p95) the syncer records. State the paired final-loss difference with its interval (n=3). The paper's pilot reported `-0.00000 +/- 0.00000` for this cell; **any other value is a finding to report as it is, not a failure to tune away.** Do not claim more than "two real hosts, toy model, K=2".

## 6. Guards (do not work around)

* Do not touch `/opt/hivemind/t4/`, the T4 governor, or its HOLD file. Policy changes go through the repository and installer by the owner.
* The T4 must keep at least 4,300 MiB of free VRAM (`AGENTS.md`). Stage 1 uses no GPU memory.
* No secrets in the repository, the PR or a log: WireGuard private keys, API keys, handles with keys.
* No new paid static IP, no public listener. The syncer listens on the overlay address only.
* Do not credit points, register producers or write to the router for this test.

## 7. Review of the paper against what was observed (2026-10-31)

Things the author should reconcile before the next release; each is from the LaTeX source read against the repository and live systems, not an accusation.

1. **"47 registered nodes ... live WAN swarm" (section on live deployment).** Observed on 2026-10-31: 3 to 4 live producer accounts, every one heartbeated from the T4 VM. The "RTX 2070" producer's heartbeat process runs on the T4 VM and advertises the T4's own public address (`34.48.48.154`, the project's static IP) as its endpoint. The text also lists RTX 3060/3090/4070/5060, ROCm and Apple Silicon classes; nothing observed backs those. If "nodes" counts registered accounts (it includes 22 consumers and 3 admins), say so.
2. **"The live `hivemind/coder` producer agent on an 8 GB consumer RTX 2070 operates at 5,462 MiB."** The only agent visible on that name runs on the T4 VM (a Tesla T4 with 15,360 MiB), so this figure needs its source host named.
3. **Scope contradiction.** The introduction says the system "has not yet been evaluated across multiple machines"; the limitations section says "from single-host simulation to 47-node live WAN deployment (resolved)". Both are in the same version.
4. **Lemma count.** 120, 141 and 155 appear in different places.
5. **E8 is a toy-model, CPU-only, two-host pilot**, and the paper says so. It should not be cited as GPU or LoRA evidence.
6. **Production credit.** The BOINC/idle-worker path submits a hash-derived placeholder "QLoRA delta" to the production coordinator (see `docs/BOINC_STEAM_VOLUNTEER_AUDIT.md`); numbers derived from credited workunits (Cobblestones, "attested compute") inherit that caveat until the credit path is gated.

## 8. RTX 2070 Host (`XAVGAMSRV`) Readiness, Public Keys & Measured Local Controls (2026-11-01)

The RTX 2070 host (`XAVGAMSRV`, Windows 11, Python 3.11.9, PyTorch 2.7.0+cpu in the pilot environment, NVIDIA GeForce RTX 2070 8 GB SM 7.5) has merged this PR (`474e56c`), verified `tests/test_e8_pilot_network_safety.py` (`7 passed`), generated its federation public keys (private keys kept outside the repository in local restricted storage), and executed the required single-host baselines and loopback negative controls.

### 8.1 Public Keys for `XAVGAMSRV` (RTX 2070 Host)

* **WireGuard Public Key (`10.77.0.5/32`)**:
  ```text
  WOsqyc4wXDumYjyWDgIz+rVBabIccgk3+RDVoxxxRFk=
  ```
  On the T4 (`socreateai-agora-hermes-node1`):
  ```bash
  sudo wg set wg-hivemind peer WOsqyc4wXDumYjyWDgIz+rVBabIccgk3+RDVoxxxRFk= allowed-ips 10.77.0.5/32
  ```
* **User-Space SSH Tunnel Public Key (Zero-UAC / Zero-Firewall-Change Option for Windows 11 `XAVGAMSRV`)**:
  Because `XAVGAMSRV` runs in a non-elevated Windows 11 user session (creating a kernel `wintun` interface requires interactive Windows UAC elevation) and TCP `34.48.48.154:22` is already reachable from `XAVGAMSRV` (`Test-NetConnection 34.48.48.154 -Port 22 == True`), the T4 session can also authorize this dedicated Ed25519 public key in `~/.ssh/authorized_keys` on `socreateai-agora-hermes-node1`:
  ```text
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIHR6g/jG45IOKFDWyC7hw1ETcY6uAddg5vCpec0SqKcI rtx2070-XAVGAMSRV-e8
  ```
  The background auto-learner daemon running on `XAVGAMSRV` polls both `http://10.77.0.4:8765` (WireGuard overlay) and an automatic SSH tunnel to `34.48.48.154:22` (`-L 18765:127.0.0.1:8765` and `-L 18766:10.77.0.4:8765`), and immediately launches `learner --learner-id 1 --host-label rtx2070-XAVGAMSRV` with `HM_INNER_STEPS=400` as soon as the T4 syncer opens a round.

### 8.2 Measured Single-Host Baselines & Loopback Negative Controls on `XAVGAMSRV` (`HM_INNER_STEPS=400`, `K=2`, `--grace off`, `--merge rda`, 12 rounds)

Raw JSON artifacts are committed under `artifacts/experiments/e8_real/baseline_seed{1,2,3}_rda.json` and `artifacts/experiments/e8_real/loopback_seed{1,2,3}_rda.json`:

| Seed | `baseline` final loss | `baseline` receipt SHA-256 | `loopback` (`127.0.0.1`) final loss | Paired diff (`loopback - baseline`) | `loopback` round latency `p50` / `p95` | `loopback` HTTP RTT `p50` (`L0` / `L1`) | `loopback` receipt SHA-256 |
|---:|---:|---|---:|---:|---|---|---|
| `1` | `2.33977365` | `1386cf1892acfee2...` | `2.33977365` | `+0.00000000` | `0.806 s` / `2.471 s` | `2.30 ms` / `2.32 ms` | `786fead3c769f681...` |
| `2` | `2.28465152` | `b9e774196bf53c29...` | `2.28465152` | `+0.00000000` | `0.806 s` / `2.596 s` | `2.29 ms` / `2.31 ms` | `7bc9bc2345a9209e...` |
| `3` | `2.75701427` | `850356727cb66860...` | `2.75701427` | `+0.00000000` | `0.806 s` / `2.605 s` | `2.37 ms` / `2.36 ms` | `a525582f7ef46144...` |

* **Paired final-loss difference (`loopback - baseline`, $n=3$)**: `0.00000000 +/- 0.00000000` (exact match across all 3 seeds, confirming the required negative control before the two-host WAN run).
