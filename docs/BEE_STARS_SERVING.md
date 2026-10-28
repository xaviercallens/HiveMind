# Serving the Bee stars expert on the T4, and the Ruche front door

## What is served

`hivemind/bee-stars` is the NIGHT-MOE-1 stars adapter: a QLoRA adapter on Qwen2.5-Coder-7B-Instruct, the only adapter the night
run rated VALID (stellar classification 0.489 to 0.893). It is one specialist expert, not a routed mixture, and it has not
passed the regression gate on other packs (issue #170). It is offered as an experimental tier.

Path of a request:

```
client -> Ruche (/v1/*) -> HiveMind OpenRouter (accounts, HMT ledger, market)
       -> https://34-48-48-154.sslip.io/bee/ (nginx, T4 VM) -> shield :8789 (bearer token) -> llama-server :8294 (base + --lora)
```

Ollama 0.34 on the VM rejects LoRA adapters ("LoRA adapters are no longer supported"). The adapter is therefore served by
llama.cpp with the same base file, flag and context as the night evaluation. `deploy/t4_bee_expert/start.sh` starts the
three processes idempotently; a `@reboot` crontab entry runs it after a VM restart. The coder tiers keep their own shield
(:8788) and producer.

## Measured (artifacts/experiments/BEE_STARS_serving_check.json)

| Path | stars pack, n=317 |
|---|---|
| night evaluation (2026-10-25 00:16Z) | 283 (0.8927) |
| served llama-server, direct | 283 (0.8927); 317/317 replies identical to the night run |
| through the public router | 282 (0.8896); 316/317 identical |

The agreement check can fail: the night run's shuffled-label control adapter matches the night replies on 104/317 items. The
one routed difference (F -> G on one star) is likely llama.cpp prompt caching: the gateway forwards `seed` but not
`cache_prompt`. Not tested.

## Ruche front door (`deploy/ruche_router_overlay`)

Revision 00010 of `hivemind-at-home-ruche` answered `/v1/chat/completions` with template text for any model and prompt, with
no key, and its sign-up created accounts in a ledger separate from the router. The overlay (revision 00014, live since
2026-10-25) removes those routes. In their place:

- `/v1/*` is forwarded unchanged (streaming included) to the router;
- `/signup` and `/login` redirect to the router's sign-up;
- `/openrouter` lists the models with a live producer and the measured result above;
- `/night-moe` is restored;
- `/join` and `/rejoindre` serve the public bilingual (French/English) "how to join" page (`src/hivemind/router/join_page.py`, also served by the
  router itself). The model table is read from the router's public catalog at request time; if the router is down the page still renders.
  **Before building the image, copy `src/hivemind/router/join_page.py` into `deploy/ruche_router_overlay/`** (the Dockerfile `COPY`s it, like `night_moe_page.py`).

The rest of the dashboard is the 00010 image unchanged.

Known issue: Ollama swaps the two coder models on the T4 next to the resident llama-server (4.8 GB), so a cold `coder` call
took 52-89 s in the smoke tests; warm calls take under a second.
