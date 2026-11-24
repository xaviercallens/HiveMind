# Plan d'Adoption HiveMind — Campagne Vacances de Noël 2026

> **Périmètre :** Prototype communautaire open-source (`AGPL-3.0`), **0,00 EUR / 0,00 USD** (aucune rémunération financière ni cryptomonnaie).  
> **Règle stricte de conformité :** Usage **100 % personnel**, hors temps de travail, uniquement sur **machines personnelles à domicile** (PC gaming perso, laptop perso) et sur **projets personnels** (Advent of Code, projets hobby, open source). Ne jamais installer sur un poste professionnel ni envoyer de code ou données d'entreprise.

---

## 1. Objectif Simple pour Noël (19 décembre 2026 – 3 janvier 2027)

Au lieu d'un parcours complexe demandant de cloner un monorepo de recherche et de configurer un environnement Python complet, nous réduisons le périmètre à **deux usages immédiats, simples et sans gêne** pour un premier cercle d'amis développeurs et joueurs PC :

1. **Parcours Développeur ("Connecté en 60 secondes, 0 dépendance") :**
   - Une seule commande Python standard (`python3 scripts/hivemind_connect.py --handle <pseudo>`) ou la page courte `/noel` (`/quick`).
   - Crée et sauvegarde la clé `hm-...` dans `~/.hivemind/credentials.json`, vérifie la connexion à la passerelle, et génère la configuration prête à l'emploi pour **VS Code (`Continue.dev`)**.
2. **Parcours Gamer PC Windows 11 / Linux ("Mode Silencieux Gamer Whisper") :**
   - Script natif Windows PowerShell (`.\scripts\join_hivemind.ps1 -Mode GamerWhisper`) et équivalent Linux/macOS (`./scripts/join_hivemind_campaign.sh --gamer-whisper`).
   - Plafond thermique doux (`70 °C`, reprise à `64 °C`), réserve mémoire vidéo dédiée au système (`2 048 Mio`), et **arrêt franc du sous-processus d'entraînement en `< 1,5 s`** (libérant 100 % de la VRAM) dès qu'un jeu Steam/plein écran démarre ou que la charge GPU externe dépasse `15 %`.
   - Désinstallation / suppression locale des identifiants en une seule commande (`--uninstall`).

---

## 2. Roadmap sur 6 Semaines (21 novembre 2026 -> 3 janvier 2027)

| Semaine | Dates | Livrables concrets | Critère de passage |
| :--- | :--- | :--- | :--- |
| **S1 — Outils Simples** | `21 – 27 nov.` | `scripts/hivemind_quickstart.py` (0 dépendance tierce), `scripts/join_hivemind.ps1` (Windows 11), mise à jour `join_hivemind_campaign.sh`, page courte `/noel` & `/quick`, API de mesure satisfaction `/v1/hivemind/onboarding/feedback`. | Tests automatisés verts + connexion réelle en `< 60 s`. |
| **S2 — Confort Gamer & Télémétrie** | `28 nov. – 4 déc.` | Vérification Windows 11 + Steam du mode `--gamer-whisper` (`<= 70 °C`, arrêt `< 1,5 s`), activation télémétrie T4 en direct sur Cloud Run, sélection recette MoE Campagne 2. | `0` saccade mesurée au lancement d'un jeu. |
| **S3 — Alpha Privée (3–5 amis)** | `5 – 11 déc.` | Test à froid par 3 à 5 amis proches (1 dev Linux/macOS, 2 gamers Windows RTX) sans aide au clavier ; correction immédiate des messages bloquants. | Temps clic -> 1re clé `<= 2 min` ; 1er round `<= 5 min`. |
| **S4 — Tableau d'Équipe & Kit Noël** | `12 – 18 déc.` | Page légère de suivi collectif de Noël (rounds validés par pseudo, Wh estimés honnêtement, évolution réelle de la loss) + fiche transparente d'invitation. | `ease_rating` Alpha `>= 4,0 / 5`. |
| **S5 — Lancement Vacances S1** | `19 – 26 déc.` | Ouverture du raid amical de Noël le samedi 19 décembre ; bulletin automatique quotidien (5 lignes) sur le canal du groupe. | `8 à 15` amis actifs (API ou GPU). |
| **S6 — Sprint Final & Bilan** | `27 déc. – 3 janv.` | Comparaison aveugle A/B par les amis développeurs entre le modèle de départ et l'adaptateur entraîné pendant les congés ; publication HuggingFace + bilan mesuré. | `ease_rating >= 4,2 / 5` et `0` plainte FPS en jeu. |

---

## 3. TODO List Opérationnelle

### Livrés dans cette première brique ("Focus Facilité")
- [x] `scripts/hivemind_quickstart.py` : assistant multiplateforme en bibliothèque standard Python uniquement (inscription/chargement clé `hm-...`, test `/v1/models`, génération config VS Code `Continue.dev`, mode surveillance `gamer-whisper`, nettoyage `--uninstall`).
- [x] `scripts/join_hivemind.ps1` : lanceur natif Windows 11 PowerShell inspectable (sans `WSL2` ni exécution distante opaque).
- [x] `scripts/join_hivemind_campaign.sh` : ajout des modes `--connect`, `--train` et `--gamer-whisper` avec chargement automatique de `~/.hivemind/credentials.json`.
- [x] `src/hivemind/router/quick_join_page.py` + routes `/noel` et `/quick` : page d'accueil minimale (< 1 écran, bilingue FR/EN, bandeau conformité PC perso).
- [x] `POST /v1/hivemind/onboarding/feedback` & `GET /v1/hivemind/onboarding/feedback/summary` : collecte anonyme et agrégée du retour utilisateur (`ease_rating` de 1 à 5, indicateur `zero_game_lag`, profil `dev`/`gamer`).

### Étapes suivantes (Semaines 2 à 4)
- [ ] Tester `join_hivemind.ps1` sur une machine Windows 11 RTX réelle avec lancement d'un jeu Steam pendant un round.
- [ ] Déployer la révision Cloud Run incluant `/noel`, `/quick` et la télémétrie T4 en direct (`scripts/push_t4_telemetry.py`).
- [ ] Organiser la session Alpha privée avec 3 à 5 amis développeurs/gamers entre le 5 et le 11 décembre.

---

## 4. Mesure Objective de la Facilité d'Utilisation et de la Satisfaction

1. **Temps d'activation Dev (`Time-to-First-Key`) :** durée entre l'ouverture de `/noel` (ou lancement de `hivemind_quickstart.py`) et une clé `hm-...` fonctionnelle -> cible **`<= 60 secondes`**.
2. **Temps d'activation Gamer (`Time-to-First-Round`) :** durée entre le pré-contrôle GPU et le premier round OpenTrainer accepté -> cible **`<= 5 minutes`**.
3. **Garantie "Zéro Gêne en Jeu" (`Yield Latency`) :** arrêt complet du worker (`0 Mio` de VRAM retenue) en **`< 1,5 seconde`** dès détection d'un exécutable de jeu ou d'une charge GPU externe `> 15 %`.
4. **Indice de Facilité & Confort (`/v1/hivemind/onboarding/feedback/summary`) :**
   - `ease_rating` (`1` = très difficile à `5` = immédiat) -> cible **`>= 4,2 / 5`**.
   - `zero_game_lag` (`true` = aucun ralentissement ni bruit gênant constaté) -> cible **`100 %` après correction Alpha**.
