# Co-training nights: calendar and how to join

Add the calendar to your agenda: [`cotraining_nights_2026-11.ics`](cotraining_nights_2026-11.ics) (import it, or subscribe to the raw file URL). All times are given in Paris time (CET, UTC+1 since 25 October) and in UTC.

| Night | Paris | UTC | What runs | Open to |
|---|---|---|---|---|
| **2** | Sun 1 Nov 22:00 → Mon 2 Nov 07:00 | 21:00 → 06:00 | run `night-002`: resumes from night 1 (round 13), two learners on the T4 | **closed** (its learner list is fixed when it starts) |
| **3** | Mon 2 Nov 22:00 → Tue 3 Nov 07:00 | 21:00 → 06:00 | run `night-003`: registered trainers; the T4 GPU is reserved for the idle-only **H-Net checkpoint probe** (issue #448), the coordinator runs on the T4 CPU | trainers with a slot (`t01`-`t08`) |
| **4** | Tue 3 Nov 22:00 → Wed 4 Nov 07:00 | 21:00 → 06:00 | run `night-004`: the T4 learners plus registered trainers | trainers with a slot |

**Slot requests close at 18:00 Paris (17:00 UTC) on the day of the night**, so that the operator can add you to the overlay before the run starts. Without a registered trainer, night 3 runs only the H-Net probe.

## Trainers: what you need

1. A machine with an NVIDIA GPU (any recent driver; Windows or Linux), or a CPU for a smoke test. Python 3.11 with `torch`, `transformers` 5.17, `peft` 0.21, `confluent-kafka`, and `bitsandbytes` for NF4.
2. **A slot.** Comment on issue #431 with your WireGuard **public** key, GPU model, free VRAM and operating system. Never post a private key or any key the operator sends you. You get back a learner id, a shard index, an overlay address and the commit to check out.
3. **The overlay.** One WireGuard peer (the T4 hub). Steps and a ready config: `docs/OVERLAY_OPEN_SOURCE_TUNNELS.md` section 2.
4. **The signing key.** It is handed over out of band, never in an issue, a chat or a log.
5. **Join.** Follow `docs/FEDERATION_V2_AMCP_AGLETS.md` section 8:
   `python scripts/e8_amcp_round.py learner --run-id night-003 --learner-id <your id> --index <your shard> --quant nf4 --device cuda --night-only`

Your learner verifies every task before it runs anything. It never runs code from a message and only claims tasks in the night window. On a machine with the governor log it waits while inference is in flight, and it refuses to start if your GPU would drop below its free-VRAM floor (`--min-free-mib`, 4,300 MiB by default; lower it only on a machine you do not use for anything else). Stop it any time with Ctrl+C.

What you should know before you join:

- The model is a toy (about 200 k parameters). These nights test the protocol and the infrastructure, not model quality.
- No points are earned. With fewer than five learners, no filter can verify a delta.
- Everyone on the overlay is trusted by the Kafka cluster, which has no authentication of its own yet. That is why slots are handed out one by one.

## End users

You do not need to train to take part. Use the models any time through the gateway (`/join` on the router: sign-up, API key, editor setup). Follow the nights on issue #431, where the results are posted the next morning. Report anything odd in a new issue.

## Results

Every run writes `artifacts/experiments/e8_real/<run>.json`. Per round, it records who was accepted and from which host, who was rejected and why, the held-out loss, the delta sizes and the latency.
