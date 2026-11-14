# OpenTrainer: CrowdGPT contributor mode

Contributor mode lets a HiveMinder lend a GPU to **Crowd-v1**, the shared dense model that
CrowdGPT trains. The training itself is done by CrowdGPT's own signed client (`client.py`).
HiveMind's OpenTrainer agent only supervises that client, so your GPU is given back when you
need it.

- Training rounds go to **CrowdGPT's coordinator**, not to HiveMind. No HMT is credited for
  these rounds, and nothing is submitted to the HiveMind gateway.
- Each finished round uploads about **0.9 GB**. On a metered connection, check your data cap
  before you start.
- The client runs no code supplied by the server. Model weights are loaded as `safetensors`.
- You are **anonymous by default**: the client starts with `--anon`.
- The automatic memory, heat and GPU-busy guards are **NVIDIA-only**, because they read
  `nvidia-smi`. `--pause-for` and `--stop-at` work on every OS.

## 1. Install CrowdGPT (once)

The agent never installs anything for you.

```bash
git clone https://github.com/Vxtzq/CrowdGPT
cd CrowdGPT
pip install -r requirements_cuda.txt      # NVIDIA. AMD: requirements_rocm.txt, Windows DirectML: requirements_directml.txt
python client.py --doctor --no-update     # client.py must match the signed release
```

You also need a HiveMind checkout (for `scripts/hivemind_opentrainer_agent.py`) and `psutil`
(`pip install psutil`).

## 2. Start with one command

Linux (NVIDIA):

```bash
python scripts/hivemind_opentrainer_agent.py --crowdgpt --crowdgpt-dir ~/CrowdGPT
```

Windows (NVIDIA), pausing while a game runs:

```powershell
python scripts\hivemind_opentrainer_agent.py --crowdgpt --crowdgpt-dir C:\CrowdGPT --pause-for eldenring.exe,cs2.exe
```

To try it out first, `--dry-run` runs the preflight and prints the exact client command
without starting anything. Add `--once` to train a single round and then exit.

Other GPUs (AMD, Intel, Apple) have no automatic memory release. You must give the budget
yourself:

```bash
python scripts/hivemind_opentrainer_agent.py --crowdgpt --crowdgpt-dir ~/CrowdGPT --backend rocm --vram-gb 6
```

| Option | Default | Meaning |
|---|---|---|
| `--reserve-mib` | 1500 | GPU memory always left free for your other programs. A HiveMind Tesla T4 node must pass `4300`. |
| `--vram-gb` | auto | Client budget, computed as (free − reserve − 700 MiB context) rounded down to 0.5 GB. Required on non-NVIDIA GPUs. |
| `--backend` | auto | `cuda`, `rocm`, `xpu`, `directml`, `cpu` or `mps`. |
| `--pause-for` | none | Comma-separated process names. Training pauses while any of them runs. |
| `--hot-c` / `--cool-c` | 82 / 74 | Pause at or above 82 °C; resume only at or below 74 °C. |
| `--stop-at` | none | Local time `HH:MM` at which to stop, for example `07:30`. |
| `--data-dir` | `~/.hivemind/crowdgpt` | Holds the receipts, the client log and `HF_HOME`. |
| `--once` | off | Run one round, then exit. |
| `--dry-run` | off | Preflight and plan only. |

## 3. What happens to your GPU

Before the first launch, the agent runs `client.py --doctor --json --no-update`. It refuses
to start unless `client.py` matches CrowdGPT's signed release manifest and your local version
equals the released version.

| Situation | What the agent does |
|---|---|
| GPU hot (≥ `--hot-c`), a `--pause-for` program starts, or another program starts GPU compute work | **Pause.** The client is suspended and keeps its round, but it **keeps its GPU memory**. |
| GPU quiet, and at or below `--cool-c`, for 60 s | **Resume** where it left off. |
| Free GPU memory below `--reserve-mib` for 10 s (a game or AI app needs memory) | **Stop.** The client is terminated so its memory is freed. **The round in progress is lost**, because the client cannot resume mid-round. |
| GPU idle for 5 min after a stop | **Relaunch** with a budget recomputed from the free memory at that moment. Repeated quick stops double the wait, up to 8×. |

Pausing alone is not enough when another program needs memory, because a suspended client
still holds all of its VRAM. That is why memory pressure leads to a stop rather than a pause.

On Windows, Direct3D and Vulkan games do not appear in `nvidia-smi`'s process list. Such a game
is still caught by the memory and heat guards, but name it in `--pause-for` to pause as soon as
it starts. On Windows a stop is a hard terminate, so an upload that is already in progress
can be lost as well.

## 4. Receipts and privacy

Each round adds one line to `<data-dir>/crowdgpt_receipts.jsonl`, for example:
`{"round": 3, "steps": 250, "mean_loss": 4.2369, "submitted": true, "evidence": "client_reported", ...}`

- `submitted` is `true` only when the client printed `Submitted successfully`.
- Lost or uncounted rounds are recorded with `submitted: false` and the reason.

To contribute under a CrowdGPT account instead of anonymously, set `CROWDGPT_TOKEN` (or both
`CROWDGPT_USER` and `CROWDGPT_PASSWORD`, or `CROWDGPT_USE_SAVED_SESSION=1`) in the
environment. Secrets are never put on the command line or written to the logs.

## 5. How to stop

- Press **Ctrl+C**, or use `--stop-at HH:MM`. The client is stopped once, gracefully where the
  OS allows it, and the round in progress is lost.
- With `--once`, the agent exits by itself after one round has been uploaded.

## 6. Troubleshooting

| Symptom | What to do |
|---|---|
| `CrowdGPT client not found` | Clone the repository (step 1) and pass `--crowdgpt-dir`. |
| `Preflight: REFUSED … differs from the signed release` | Run `git pull` in the CrowdGPT folder (or re-clone), then `python client.py --doctor --no-update`. |
| `version mismatch` | Your checkout is older than the release. Update it. |
| `pass --vram-gb explicitly` | No `nvidia-smi` was found. Give a budget, for example `--vram-gb 4`. |
| `waiting: budget below 1.0 GB` | Close other GPU programs or lower `--reserve-mib`. |
| Client exits with `authentication failed` | Remove the account variables to stay anonymous, or fix the token. |

`--dry-run` shows the detected GPU, the computed budget and the exact command, without starting
anything. `python client.py --doctor` in the CrowdGPT folder shows CrowdGPT's own diagnostics.

## Démarrage rapide (français)

1. Installez le client CrowdGPT : `git clone https://github.com/Vxtzq/CrowdGPT`, puis
   `pip install -r requirements_cuda.txt` (cartes NVIDIA).
2. Testez sans rien lancer :
   `python scripts/hivemind_opentrainer_agent.py --crowdgpt --crowdgpt-dir ~/CrowdGPT --dry-run`
3. Lancez :
   `python scripts/hivemind_opentrainer_agent.py --crowdgpt --crowdgpt-dir ~/CrowdGPT --pause-for jeu.exe`
   (sous Windows : `scripts\hivemind_opentrainer_agent.py` et `C:\CrowdGPT`).

Ce qu'il faut savoir :

- Chaque round envoie environ **0,9 Go** au serveur de CrowdGPT, pas à HiveMind.
- Vous êtes **anonyme** par défaut.
- Le client est **mis en pause** quand le GPU chauffe ou quand un jeu tourne. Il est **arrêté**,
  et le round en cours est perdu, quand un autre programme a besoin de mémoire GPU. Il redémarre
  après 5 minutes de GPU inactif.
- Les protections automatiques ne fonctionnent qu'avec NVIDIA. Sur AMD ou Intel, indiquez
  `--vram-gb`.
- Pour arrêter : **Ctrl+C** ou `--stop-at 07:30`.
