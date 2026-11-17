# One Tesla T4 as a HiveMind node: training (BOINC) or public inference, never both at once

Nothing in this directory is installed or started by the repository. Every unit is a **template** for the machine's
owner to review, edit (paths, user) and enable. Nothing here opens a port to the internet: all services bind to
`127.0.0.1`; making one reachable is a separate, deliberate step (below).

## The two modes (the card has 15 GiB, shared)

| Mode | What runs | GPU memory | Use |
| --- | --- | --- | --- |
| **train** | stock BOINC client + `hivemind_qlora_moe` / `hivemind_moe_infer` apps (`docs/BOINC_CORE_NODE.md`) | ~3.5 GiB per task | QLoRA of a MoE, batch inference, validated and credited |
| **serve-moe** | private Ollama with `gpt-oss:20b` (MoE, 3.6 B active) behind the producer shield | **12.3 GiB** measured (`docs/HIVEMIND_OPENROUTER.md`) | interactive inference for new users through the gateway |
| serve-coder (alternative) | Ollama `qwen2.5-coder:7b` (dense) | 5.5 GiB at 4k ctx (E16) | leaves room for one BOINC task; not a MoE |

`serve-moe` leaves ~3.0 GiB free, which is **less than one BOINC task needs**. Switch with
`boinccmd --set_gpu_mode never` (BOINC lets go of the GPU at the next checkpoint) before starting `serve-moe`, and
`always` again after stopping it. Swapping the 20 B model in costs a cold load of ~285 s.

## Which MoE models are actually available here

Measured or pulled on this machine, from the repo's own reports (nothing newer was tested):

1. `gpt-oss:20b` — MoE, pulled in Ollama, the only one with a serving measurement on this T4 (12,331 MiB; 19–45 tok/s; 6/6 vs 3/6 for the 7B coder on six coding tasks, n=6, direction only). It "thinks first": send `reasoning_effort: low` and a generous `max_tokens`.
2. `ibm-granite/granite-3.1-1b-a400m-base` — MoE (1.3 B), cached; used by the QLoRA node for training/batch inference. It is a **base** model, not a chat model: do not advertise it as a general assistant.
3. Other MoE checkpoints named in the experiments (`granite-3.1-3b-a800m`, `OLMoE-1B-7B`, `Phi-tiny-MoE`) were training subjects; this machine has not been verified to serve them.
4. `qwen3-coder:30b` (MoE) does **not** fit a T4 (19 GB weights).

> **Status correction (2026-10-25).** An earlier version of this page said the T4 was not yet reachable by new users. The
> owner's other session reports that it already is: this T4 serves through the gateway via a public HTTPS endpoint, with
> two producer accounts (the coder tiers, and a separate "bee" expert). The author of this page did not verify that and
> did not create it. What is left open is only **what the one card serves**: the current mix (a ~4.8 GB expert in
> llama.cpp plus the 7B coder, ~5.6 GB), `gpt-oss:20b` (12.3 GiB, needs both stopped), or BOINC training. The steps below
> describe how to do it for a *new* machine.
>
> **BOINC next to the live mix:** with ~10.4 GiB already resident, one BOINC task (~3.5 GiB) fits, but the server-side
> validator's replay (another ~3.5 GiB of model) cannot overlap with it (≈17.4 GiB > the card's 15.4). The recorded runs
> did not overlap them. Serialise validation, or validate on a different card.

## Exposing the node to new users (a new machine; this one is reported as already done)

The public gateway (`docs/HIVEMIND_OPENROUTER.md`, "The public deployment") routes only to producers that
(a) hold a producer API key and (b) advertise a **public https endpoint** (its SSRF guard rejects http and private
addresses). New users register at the gateway's `/signup` or `POST /v1/hivemind/register`; they start with 0 HMT until
the owner grants some. So, in order:

1. `export HIVEMIND_ENDPOINT_TOKEN=$(python3 -c "import secrets;print(secrets.token_urlsafe(32))")` (never commit it).
2. Start the private model server and the shield (units below), `--allow-model gpt-oss:20b`.
3. Put a TLS tunnel or reverse proxy of your choice in front of `127.0.0.1:8787` — this is the step that makes the GPU reachable from the internet, so choose it knowingly.
4. Register a producer account (`POST /v1/hivemind/register` with `role: producer`), keep the key in `HIVEMIND_API_KEY`.
5. `python scripts/hivemind_producer_agent.py --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app --endpoint https://YOUR_TUNNEL --preset moe-general --execution-class t4`

Producers see the full prompts of the people they serve and the gateway does not verify output correctness
(HM-S13-05): say so to your users.

## Units (templates)

- `hivemind-t4-ollama-moe.service` — private Ollama on `127.0.0.1:11436` with the measured settings.
- `hivemind-t4-shield.service` — the authenticating proxy on `127.0.0.1:8787`; reads the token from `/etc/hivemind/t4-node.env`.
- `hivemind-t4-producer.service` — the heartbeat agent; reads `HIVEMIND_API_KEY` from the same file.

`/etc/hivemind/t4-node.env` (mode 0600, owner root) holds `HIVEMIND_ENDPOINT_TOKEN`, `HIVEMIND_API_KEY`, `HIVEMIND_GATEWAY`, `HIVEMIND_ENDPOINT`.
