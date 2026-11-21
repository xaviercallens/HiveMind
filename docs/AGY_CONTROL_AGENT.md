# Agent de Contrôle HiveMind — Google Antigravity SDK (`hivemind_agy_control_agent.py`)

Cet agent autonome s'exécute avec **ton compte HiveMind** (`~/.hivemind/credentials.json` ou `HIVEMIND_API_KEY`) et **ton compte GitHub (`@xaviercallens`)** via le **Google Antigravity Python SDK (`google-antigravity`)** afin de :
1. **Suivre l'adoption réelle et la satisfaction** (comptes par rôle, producteurs/trainers externes actifs, visiteurs et clones GitHub sur 14 jours, score de facilité CES `ease_rating >= 4.2/5`, confort gamer `zero_game_lag >= 95 %`).
2. **Piloter les campagnes en cours** (*Winter GPU Raid Noël 2026* dès le `19 décembre 2026`, *MoE Autoresearch*, *OpenTrainer*) **sans jamais solliciter `hivemind/coder` en inférence** pendant les entraînements GPU sur la T4.
3. **S'auto-améliorer en continu** en identifiant le goulot de friction principal, en enregistrant des actions de faible périmètre (`<= 3` fichiers, couverture `pytest >= 95 %`), et en conservant sa mémoire de session (`conversation_id` + `save_dir` dans `~/.hivemind/agy_control_agent/`).

> [!IMPORTANT]
> **Règle Conformité Amadeus / Entreprise :** Usage strictement limité aux **PC personnels à domicile et projets personnels / open-source**, avec `0,00 EUR / 0,00 USD` de contrepartie financière.

## Commandes Rapides

```bash
# 1. Installer le SDK Google Antigravity dans l'environnement Python
uv pip install google-antigravity

# 2. Vérifier instantanément ton compte HiveMind + télémétrie adoption & campagnes (sans appel LLM)
python scripts/hivemind_agy_control_agent.py --mode status

# 3. Lancer un cycle complet d'audit & d'auto-amélioration via Google Antigravity SDK
#    (utilise GEMINI_API_KEY ou --vertex --project gen-lang-client-0625573011)
python scripts/hivemind_agy_control_agent.py --mode audit

# 4. Lancer le démon autonome toutes les 3 heures (trigger natif `every(10800, ...)`)
python scripts/hivemind_agy_control_agent.py --mode daemon --interval-seconds 10800

# 5. Discuter en direct dans le terminal avec ton agent de contrôle HiveMind
python scripts/hivemind_agy_control_agent.py --mode chat
```

## Fichiers d'État Locaux (`~/.hivemind/`)
- `~/.hivemind/credentials.json` (`chmod 0600`) : clé `hm-...` de ton compte opérateur (toujours masquée `hm-****xxxx` dans les sorties).
- `~/.hivemind/agy_control_agent/session_state.json` : `conversation_id` Antigravity persistant entre les cycles.
- `~/.hivemind/agy_control_agent/telemetry_history.jsonl` : historique horodaté de l'adoption et calcul automatique des deltas.
- `~/.hivemind/agy_control_agent/improvement_proposals.jsonl` : propositions d'auto-amélioration priorisées avec plan de tests.
- `~/.hivemind/agy_control_agent/latest_report.md` & `latest_report.json` : dernier rapport exécutif structuré (`ControlCycleReport`).
