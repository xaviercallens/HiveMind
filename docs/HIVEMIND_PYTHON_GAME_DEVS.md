# HiveMind for Python and game developers

Use open-source coding models from your editor, or share your gaming GPU and earn **HiveMind points (HMT)** you can
spend later, all through the standard OpenAI chat-completions protocol. This guide is for people who write Python and
small games (pygame, arcade, Pyxel, Godot) and want the shortest path from zero to a working setup.

Gateway: `https://hivemind-openrouter-1003063861791.us-central1.run.app` (base URL for tools: add `/v1`).

> **Be clear about what exists today.** HMT are points, not money: they have no cash value and cannot be withdrawn.
> New accounts start with 0 HMT and only the operator can grant points, so a pure consumer cannot spend until they are
> granted some. You earn points by serving requests with your own GPU. Producers can read the prompts they serve, so do
> not send secrets or private code. Whether a producer really runs the model it advertises is not verified yet
> (HM-S13-05).

## 1. Two ways in

| You want to | Do this | Needs |
|---|---|---|
| **Use** models in your editor | Register, get HMT, paste a snippet into your editor | Python 3.10+, `pip install httpx` |
| **Share** your GPU and earn HMT | Pick a model for your card, run the shield and the agent | An NVIDIA card (4 GB or more), Ollama, a public https address |
| Both | Register as `prosumer` (the default): one balance, earn and spend | Both lists |

Download the three helper scripts from the repository (each is one readable file, about 200 to 300 lines; read them before
you run them): `scripts/hivemind_connect.py`, `scripts/hivemind_producer_agent.py`, `scripts/hivemind_producer_shield.py`.
They need only `httpx` (the shield also needs `fastapi` and `uvicorn`).

## 2. Use HiveMind from your editor

**Easiest: sign in with Google.** Open `/signup` on the gateway
(`https://hivemind-openrouter-1003063861791.us-central1.run.app/signup`), pick what you want to do, click the Google button,
and copy the API key shown. It is free, needs no password, and your e-mail address is not stored (only an anonymous hash
of your Google account id). Signing in again later gives you a new key and retires the old one, so it also recovers a lost
key. The button works once the operator has switched Google sign-in on (see `HIVEMIND_OPENROUTER.md`); until then the page
says so and you can register with a handle as below.

```bash
pip install httpx

# Option A: Sign in with Google (1-click via browser or gcloud):
python hivemind_connect.py register-google --browser --save-key ~/.hivemind_key

# Option B: Register with a handle:
python hivemind_connect.py register --handle your-name --save-key ~/.hivemind_key   # key saved owner-only

export HIVEMIND_API_KEY=$(cat ~/.hivemind_key)
python hivemind_connect.py check            # balance, which models have a producer right now, the tier list
python hivemind_connect.py snippet --list   # continue opencode aider cline zed jetbrains python env
python hivemind_connect.py snippet continue --model coder
```

`snippet` prints settings with the placeholder `hm-YOUR_KEY`; nothing secret is written unless you pass `--embed-key`.

### Pick a model tier

You choose a tier by how much quality you need; producers choose the biggest tier their card can hold. Everybody serving
the same tier competes in the same market, which keeps prices fair and supply deep.

| Model id | Runs on about | Good for | MoE |
|---|---|---|---|
| `hivemind/coder-lite` | 4 GB | quick edits, explanations, small scripts | no |
| `hivemind/coder` | 6 GB | everyday Python and simple games: functions, classes, pygame loops, tracebacks | no |
| `hivemind/coder-plus` | 12 GB | bigger refactors, multi-file reasoning, tests | no |
| `hivemind/moe-general` | 16 GB | reasoning-heavy questions and design (gpt-oss 20B, about 3.6B active parameters) | yes |
| `hivemind/moe-coder` | 24 GB | the strongest coding tier: agentic multi-file edits, long context (Qwen3-Coder 30B, about 3.3B active) | yes |
| `hivemind/auto` | n/a | whichever live model has the most producers | n/a |

A mixture-of-experts (MoE) model has many specialist sub-networks but uses only a few per token, so it answers like a
big model while computing like a small one. The catch is memory: all the experts must fit, which is why the MoE tiers
need 16 to 24 GB cards. They exist as soon as one producer with such a card joins; `check` shows which tiers are live.
The live list is the truth, the table is the plan.

### Editor settings (verify each against the tool's current documentation)

Settings formats change between releases. The generated snippets follow each tool's documented OpenAI-compatible
provider; if a field is rejected, the tool's own docs win.

| Tool | Where | Notes |
|---|---|---|
| **Continue** (VS Code, JetBrains) | `config.yaml` | `snippet continue`. Chat, edit and apply modes work. |
| **Cline** (VS Code) | settings, provider "OpenAI Compatible" | `snippet cline` prints the four fields. |
| **Aider** (terminal) | environment + `--model openai/hivemind/coder` | `snippet aider`. Good for editing a whole game folder. |
| **OpenCode** | `opencode.json` | `snippet opencode`. |
| **Zed** | `settings.json` | `snippet zed`. |
| **PyCharm / IntelliJ** | Continue plugin, or the AI Assistant's OpenAI-compatible provider | `snippet jetbrains`. |
| **Any OpenAI client** | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | `snippet env`, `snippet python`. |

What works and what does not (yet):

- **Works:** chat completions, streaming, tool/function definitions passed through (whether they are used depends on the model).
- **Not supported yet:** inline tab-autocomplete (needs the completions/FIM endpoint), embeddings, images. Use chat and edit modes.
- **Set an output limit.** If a request has no `max_tokens` the gateway applies 1,024. The snippets set 2,048; raise it for
  whole-file generation. The gateway never lets a request exceed the limit it bills against.
- **Context is the producer's choice.** Ollama's default context window is small; long prompts may be cut by a producer who
  did not raise it. Keep prompts focused (paste the function, not the project), or use the larger tiers.

### From Python

```python
# pip install openai
from openai import OpenAI

client = OpenAI(base_url="https://hivemind-openrouter-1003063861791.us-central1.run.app/v1", api_key="hm-YOUR_KEY")
reply = client.chat.completions.create(
    model="hivemind/coder",
    max_tokens=2048,
    messages=[{"role": "user", "content": "Write a pygame window with a bouncing ball."}],
)
print(reply.choices[0].message.content)
print(reply.model_dump().get("hivemind"))   # price per 1k tokens, HMT charged, producer, mechanism
```

Errors are OpenAI-shaped: `401` bad key, `402` balance too low or your maximum price is below the market price (nothing
charged), `429` slow down, `503` no producer is live for that model. `client.models.list()` shows the live ones.

## 3. Share your GPU and earn points

### Step 1: choose a model for your card

```bash
python hivemind_producer_agent.py --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app --suggest --detect-vram
# nvidia-smi sometimes cannot be read from your user account even when the card works: pass --vram-gb 8 instead
```

It prints the biggest tier that fits and the `ollama pull` line. The tags are recommendations; verify them in the Ollama
library, because tags and sizes change.

### Step 2: run the model server

Install Ollama, pull the model, and keep it on your machine (`127.0.0.1:11434`). Use a recent Ollama (older releases lack
`/v1/models` and use a small default context; verify the version notes), and consider raising the context
(Ollama documents `OLLAMA_CONTEXT_LENGTH` or `num_ctx` in a Modelfile; verify for your version).

### Step 3: put the shield in front of it

Ollama has no login. If you expose it as is, anyone who learns the URL can use your GPU and call its management API.
The shield publishes only `GET /v1/models` and `POST /v1/chat/completions`, demands a secret token, caps request size and
concurrency, and can pin the models it will run:

```bash
pip install httpx fastapi uvicorn
export HIVEMIND_ENDPOINT_TOKEN=$(python -c "import secrets;print(secrets.token_urlsafe(32))")
python hivemind_producer_shield.py --allow-model qwen2.5-coder:7b       # listens on 127.0.0.1:8787
```

### Step 4: make the shield reachable at a public https address

The gateway only accepts `https` addresses on the public internet. Use a tunnel or a port forward with TLS that points at
**port 8787 (the shield), never at 11434**. Examples are a Cloudflare Tunnel, Tailscale Funnel, or your own reverse proxy;
follow that tool's documentation. With the token in place, a leaked URL is useless to anyone but the gateway.

### Step 5: start the agent

```bash
export HIVEMIND_API_KEY=$(cat ~/.hivemind_key)          # the producer (or prosumer) key
python hivemind_producer_agent.py \
    --gateway https://hivemind-openrouter-1003063861791.us-central1.run.app \
    --endpoint https://YOUR-PUBLIC-URL --preset coder
```

Before the first heartbeat the agent checks that your endpoint really lists the model, and refuses to advertise a model you
cannot serve. If your server calls it something else, add `--local-name NAME`. The agent heartbeats every 60 seconds; the
gateway treats you as live for 120 seconds after the last one.

**Gaming?** Stop the agent (Ctrl+C) and you disappear from the market within two minutes, which frees the GPU. Start it
again when you are done.

### What you earn

You receive 85 percent of what a consumer pays for the tokens you serve (10 percent goes to the validators' account and 5
percent is burned). The price comes from the market: consumers declare the most they will pay, and the Kelly mechanism
sets one price per tier that does not depend on how many producers there are, so adding capacity does not lower the price
you earn per token. (That holds while at least two funded consumers are bidding; with fewer, a posted price just above the
cheapest producer's break-even applies instead, and the market page says which mechanism is in force.) The market page for a tier shows the price and a certificate that no one gains by changing their offer:

```bash
curl -H "Authorization: Bearer $HIVEMIND_API_KEY" "https://hivemind-openrouter-1003063861791.us-central1.run.app/v1/hivemind/market?model=hivemind/coder"
```

`--cost-per-1k` is the least you will accept per 1,000 tokens (it defaults from the tier). Set it above what your electricity
costs you; if it is higher than the market price you simply receive no requests. Serving costs real power and wears real
hardware, and HMT are only points, so share for the community and for credit you can spend on bigger models, not for income.

## 4. Game-development recipes

Small models do best with small, concrete, runnable requests. A system prompt that fits Python and games (paste it into your
editor's rules or system-message field):

```text
You are a Python game-development assistant. Prefer the standard library plus pygame or arcade.
Return one complete, runnable file unless asked otherwise. Keep game logic (movement, collisions, scoring)
in plain functions that take and return simple data so they can be tested with pytest, separate from drawing code.
Use a fixed time step and delta time, no global mutable state beyond one Game class, and no assets: draw with shapes.
When code fails, explain the traceback in one sentence, then give the fix.
```

| Goal | Prompt that works well | Tier |
|---|---|---|
| First game | "Write Pong in one pygame file: two paddles (W/S and Up/Down), a ball that speeds up, score at the top, 60 FPS." | `coder` |
| Fix an error | Paste the traceback and the function it points at: "Why does this raise IndexError? Fix it." | `coder-lite` |
| Testable logic | "Extract the collision code into `collide(a, b) -> bool` and write pytest tests for edge cases." | `coder` |
| New feature | "Add a pause menu and high-score saving to JSON to this file." (paste the file) | `coder-plus` |
| Godot | "Write a GDScript 4 `CharacterBody2D` script: 8-way movement, dash with cooldown, exported speed." | `coder` or larger |
| Design | "Plan a roguelike's room generation: data structures and the algorithm, no code yet." | `moe-general` |
| Multi-file | Aider on a folder: "Split the game into `entities.py`, `rules.py`, `main.py` without changing behaviour." | `moe-coder` |

Habits that save points: ask for one change at a time, paste only the code that matters, run it and paste back the exact
error, and keep the tests as the judge, since models do make mistakes.

## 5. Privacy, safety and good citizenship

- A producer sees every prompt it serves. Never paste keys, passwords, personal data or proprietary code.
- Your GPU is exposed only through the shield and a token; keep the token secret and rotate it by restarting both programs.
- The gateway bills by bytes it can count, so a producer cannot inflate a bill, but it cannot tell a good answer from a bad one yet.
- Do not use HiveMind for anything illegal or for content you do not have the right to process.

## 6. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `401` from the gateway | Wrong or missing key. `check` tells you. |
| `402` | Balance too low, or your maximum price is below the market price. Ask the operator for a grant or raise it with `POST /v1/hivemind/consumer/max-price`. |
| `503 no_live_producers` | Nobody serves that tier right now. Try `hivemind/auto` or another tier; `check` lists the live ones. |
| Agent: "preflight failed ... /v1/models" | The shield or tunnel is down, the token differs between shield and agent, or Ollama is too old; use a recent Ollama or `--skip-preflight` if you are sure. |
| Agent: "does not list [...]" | Pull the model (`ollama pull NAME`) or pass `--local-name`. |
| Gateway says the endpoint is not allowed | It must be `https` on a public address. Home networks need a tunnel or a forwarded port with TLS. |
| Slow first answer | The model is loading into GPU memory or the producer is busy; a busy shield answers `429` and the gateway tries another producer. |

## 7. Open standards in use

The OpenAI chat-completions API (JSON and server-sent events), bearer-token authentication, plain JSON/YAML editor
configuration, and HTTPS. Nothing here needs a proprietary client: any tool that can talk to an OpenAI-compatible server can
use HiveMind.
