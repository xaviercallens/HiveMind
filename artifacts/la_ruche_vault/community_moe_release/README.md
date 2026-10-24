---
library_name: hivemind-la-ruche
license: apache-2.0
tags:
  - hivemind
  - mixture-of-experts
  - qlora
  - diloco
  - la-ruche
---

# 🐝 La Ruche (`RUCHE-MOE-V2_6_0-fc6c9ccb`) — Community BTM Sparse MoE Model Vault

- **Architecture:** `HiveMind-BTM-UpcycledSparseMoE-8xQLoRA` (`8` Honeycomb Cells, Top-`2` active)
- **Router & Depth Core:** `SigmaMoE-PeriLN + ThreeTierHotColdHierarchy` + `OrthogonalInputInjector`
- **Combined Safetensors SHA-256:** `fc6c9ccbd11dbd4702a5147bf159b65781f08c0411ac77e9cc82fa0a8795c28c` (`661688` bytes)
- **Mean Holdout Perplexity:** `251.62 -> 234.76` (`-6.70%` reduction)
- **Contributors:** @clark719664, @pedromineo, @radialmonster, @xaviercallens
- **Charter:** Strictly non-monetary open-source project (0.00 EUR / 0.00 USD). All experts and MoE weights are openly shared under reciprocal community compute.

## Honeycomb Cells (`Alvéoles`) Inventory

| Cell ID | Domain | Contributor | Hardware | Holdout PPL (`Before -> After`) | Wire Codec | SHA-256 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ALVEOLE-01-CODE-PYTHON` | `code_python` | `@xaviercallens` | `cpu` | `248.94 -> 232.52` (`-6.60%`) | `e8_2bit` | `14800ad814ea...` |
| `ALVEOLE-02-MATH-REASONING` | `math_reasoning` | `@clark719664` | `cpu` | `254.28 -> 237.01` (`-6.79%`) | `e8_2bit` | `f20fe944e906...` |
| `ALVEOLE-03-PHYSICS-SIM` | `physics_sim` | `@radialmonster` | `cpu` | `252.67 -> 235.53` (`-6.78%`) | `e8_2bit` | `7eece19d44d0...` |
| `ALVEOLE-04-SYSTEMS-RUST` | `systems_rust` | `@pedromineo` | `cpu` | `249.04 -> 233.94` (`-6.06%`) | `e8_2bit` | `a3475700045d...` |
| `ALVEOLE-05-CYBER-SEC` | `cyber_sec` | `@xaviercallens` | `cpu` | `252.63 -> 235.40` (`-6.82%`) | `e8_2bit` | `efffb8cb3a92...` |
| `ALVEOLE-06-BIO-MED` | `bio_med` | `@clark719664` | `cpu` | `254.57 -> 236.24` (`-7.20%`) | `e8_2bit` | `2307ebe37413...` |
| `ALVEOLE-07-GENERAL-CHAT` | `general_chat` | `@radialmonster` | `cpu` | `253.53 -> 237.17` (`-6.45%`) | `e8_2bit` | `a5c49e0b6d1a...` |
| `ALVEOLE-08-FORMAL-LEAN4` | `formal_lean4` | `@pedromineo` | `cpu` | `247.27 -> 230.27` (`-6.88%`) | `e8_2bit` | `1d62f2c9e019...` |

## Quickstart: Load & Run the Community MoE Model

```python
import torch
from hivemind.swarm.la_ruche_vault import load_community_moe_from_ruche

moe_model, manifest = load_community_moe_from_ruche('artifacts/la_ruche_vault')
tokens = torch.tensor([[2, 5, 9, 14, 21, 33, 44, 55]])
outputs = moe_model(input_ids=tokens)
print('Logits shape:', outputs.logits.shape)
```
