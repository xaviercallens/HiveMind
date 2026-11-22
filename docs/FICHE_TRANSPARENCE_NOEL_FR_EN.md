# Fiche Transparence Noël 2026 (FR / EN) — Raid Amical HiveMind

> **Charte en 1 ligne (FR) :** Prototype citoyen open-source (`AGPL-3.0`), **0,00 EUR / 0,00 USD**, **100 % sur PC personnel à domicile et projets perso uniquement** (conformité stricte Amadeus / employeurs : jamais sur un PC pro ni avec du code d'entreprise).
>
> **1-Line Charter (EN):** Citizen-science open-source prototype (`AGPL-3.0`), **0.00 EUR / 0.00 USD**, **100% personal home PCs and personal hobby/open-source projects ONLY** (strict Amadeus / employer compliance: never on corporate machines or with work code).

---

## Partie FR — Les 4 Questions Légitimes (Amis Développeurs & Gamers PC)

### 1. Est-ce 100 % perso, bénévole et conforme (règle PC domicile / conformité Amadeus, 0,00 EUR) ?
- **100 % PC personnel à domicile uniquement :** HiveMind (`AGPL-3.0`) s'utilise exclusivement hors temps de travail, sur ta propre tour gaming ou ton laptop personnel, et pour tes projets personnels ou open-source (Advent of Code, scripts hobby, science citoyenne).
- **Règle stricte de conformité Amadeus / employeur :** n'installe **jamais** les scripts sur un ordinateur professionnel, ne te connecte jamais depuis un réseau ou VPN d'entreprise, et n'envoie **jamais** de code source, prompt ou donnée appartenant à un employeur (Amadeus ou tout autre employeur).
- **0,00 EUR / 0,00 USD :** il n'y a aucun argent en jeu — ni abonnement, ni rémunération, ni cryptomonnaie, ni jeton spéculatif. Les compteurs de contribution restent purement symboliques et amicaux.

### 2. Combien ça me coûte vraiment en électricité à la maison (Wh et centimes) ?
- **Mode Développeur API seule (`--connect` / `/noel` dans VS Code) :** **`0 Wh`** de consommation GPU supplémentaire sur ton PC (**`0,00 EUR`**).
- **Mode Prêt GPU doux :** la formule honnête au compteur électrique est `Puissance GPU additionnelle (W) × Durée (h) = Wh`.
- **Exemple concret mesurable :** **`~60 W × 2 h = 120 Wh = 0,12 kWh`**. Au tarif résidentiel moyen d'environ `0,25 EUR / kWh`, une session de 2 heures coûte **`~0,03 EUR` (environ 3 centimes d'euro)**.
- Même sur un GPU de bureau plus musclé en mode silencieux `Gamer-Whisper` (`~120 W × 2 h = 240 Wh = 0,24 kWh`, plafond thermique `70 °C`), cela représente **`~0,06 EUR` (~6 centimes)**. Un calculateur interactif avec ton propre tarif est disponible sur `/raid-noel`.

### 3. Qu'est-ce qui sort exactement de mon PC sur Internet ?
- **Moins de `50 Mo/h` (`< 50 MB/h`) de deltas mathématiques LoRA :** seuls des tenseurs numériques compressés d'adaptation de poids (deltas mathématiques QLoRA / DiLoCo) et un battement de cœur technique minimal (modèle GPU, Mio de VRAM libre, état du round) sortent en **HTTPS chiffré sur le port TCP `443`** vers les seuls hôtes autorisés (`*.run.app`, `*.googleapis.com`, et lecture publique sans jeton sur `datasets-server.huggingface.co`).
- **Zéro port entrant, zéro trafic local :** `0` port ouvert sur ta box, aucune écoute `0.0.0.0`, aucun scan de ton réseau local (LAN).
- **Zéro fichier personnel ni prompt local lu sur ton disque :** aucun document, fichier personnel, historique navigateur, capture d'écran ou variable secrète ne quitte ta machine. Tous les scripts (`hivemind_quickstart.py`, `hivemind_doctor.py`, `join_hivemind.ps1`) sont en texte clair (bibliothèque standard Python, zéro exécutable opaque).
- *Rappel API VS Code :* si tu envoies volontairement des requêtes de code vers `/v1/chat/completions`, garde-les strictement sur des projets perso (jamais de mots de passe, clés privées, données personnelles ou code professionnel).

### 4. Comment mettre en pause en `< 1,5 s` pour Steam ou tout désinstaller en `10 s` (`--uninstall`) ?
- **Pause automatique en `< 1,5 s` dès qu'un jeu démarre (`Gamer-Whisper`) :** le garde local plafonne la carte à **`70 °C`** (reprise à `64 °C`), réserve **`2 048 Mio` de VRAM libre** pour ton bureau, et surveille les jeux Steam/PC (`cs2.exe`, `eldenring.exe`, `bg3.exe`, `cyberpunk2077.exe`, `helldivers2.exe`, `dota2.exe`, `valorant.exe`, etc., ou `--game monjeu.exe`) ainsi que toute charge GPU externe `> 15 %`. Dès détection, le sous-processus d'entraînement est coupé net en **`< 1,5 seconde`** (sous la marge Windows WDDM TDR de `2,0 s`), libérant **100 % de ta VRAM et de tes FPS** (ou `Ctrl+C` à tout instant).
- **Désinstallation propre en `10 secondes` (`--uninstall`) :** aucun service caché, aucun pilote système, aucune tâche au démarrage de Windows/Linux n'est installée. Une seule commande supprime tes identifiants locaux (`~/.hivemind/credentials.json`) et la config VS Code (`~/.continue/hivemind_provider.yaml`) :
  ```bash
  # Linux / macOS / Windows (Python standard)
  python3 scripts/hivemind_quickstart.py --uninstall
  ```
  ```powershell
  # Windows 11 (PowerShell natif)
  .\scripts\join_hivemind.ps1 -Mode Uninstall
  ```
  *(Optionnel : si tu as exécuté un round d'entraînement GPU local, supprime aussi le dossier de cache `~/.cache/hivemind/opentrainer` pour récupérer l'espace disque.)*

---

## EN Section — The 4 Honest Answers (Developer & PC Gamer Friends)

### 1. Is this strictly personal-PC only, compliant (Amadeus rule), and 0.00 EUR / 0.00 USD?
- **100% personal home PCs only:** HiveMind (`AGPL-3.0`) is run strictly in personal free time, on your own home gaming PC or personal laptop, and exclusively for personal hobby or open-source projects (Advent of Code, personal learning, citizen science).
- **Strict Amadeus / Employer Compliance Rule:** **never** install or run HiveMind on a corporate laptop/workstation, **never** connect from an employer network or VPN, and **never** submit employer source code, prompts, or confidential data (Amadeus or any other employer).
- **0.00 EUR / 0.00 USD:** zero money changes hands — no paid tiers, no payouts, no cryptocurrency, and no financial value of any kind. Contribution counters are purely symbolic compute-reciprocity tallies among friends.

### 2. What is the real home electricity cost in Wh and cents?
- **Developer API-only mode (`--connect` / `/quick` in VS Code):** **`0 Wh`** of extra local GPU draw (**`0.00 EUR / 0.00 USD`**).
- **Quiet GPU lending mode:** the honest wall-meter formula is `Extra GPU power (W) × Hours (h) = Wh`.
- **Concrete example:** **`~60 W × 2 h = 120 Wh = 0.12 kWh`**. At a typical European residential rate of `~0.25 EUR / kWh`, a 2-hour session costs **`~0.03 EUR` (~3 euro cents)**.
- Even on a desktop card in quiet `Gamer-Whisper` mode (`~120 W × 2 h = 240 Wh = 0.24 kWh`, `70 °C` thermal cap), 2 hours comes out to **`~0.06 EUR` (~6 cents)**. You can check your exact wattage and local tariff on `/raid` (`/raid-noel`).

### 3. What actually leaves my PC over the network?
- **Under `50 MB/h` of LoRA mathematical weight deltas:** only compressed numerical QLoRA / DiLoCo adapter weight tensors plus minimal health heartbeats (GPU model, free VRAM MiB, round state) leave your PC over **outbound HTTPS on TCP port `443`** strictly to allowlisted endpoints (`*.run.app`, `*.googleapis.com`, and read-only public training rows on `datasets-server.huggingface.co`).
- **Zero inbound ports & zero LAN traffic:** `0` inbound firewall ports opened, zero `0.0.0.0` listeners, and zero local-network (LAN) scanning or peer connections.
- **Zero personal files or local prompts ever leave your disk:** no documents, personal files, browser data, screenshots, or environment secrets are ever read or uploaded. Every helper script (`hivemind_quickstart.py`, `hivemind_doctor.py`, `join_hivemind.ps1`) is plain-text standard-library code you can inspect before running.
- *VS Code Chat API reminder:* prompts you voluntarily send to `/v1/chat/completions` are served by volunteer nodes — keep them strictly to personal hobby/open-source code (never secrets, passwords, personal data, or employer code).

### 4. How do I pause in `< 1.5 s` for Steam games or uninstall in `10 s` (`--uninstall`)?
- **Automatic `< 1.5 s` pause when Steam or any PC game launches (`Gamer-Whisper`):** the local guard enforces a quiet **`70 °C`** fan/thermal cap (`64 °C` resume), keeps **`2,048 MiB` of free VRAM** reserved for your OS, and watches for Steam/PC game executables (`cs2.exe`, `eldenring.exe`, `bg3.exe`, `cyberpunk2077.exe`, `helldivers2.exe`, `dota2.exe`, `valorant.exe`, etc., plus `--game mygame.exe`) or external GPU load `> 15%`. When triggered, the training process tree stops cleanly in **`< 1.5 seconds`** (safely under the `2.0 s` Windows WDDM TDR timeout), returning **100% of your VRAM and FPS** immediately (or hit `Ctrl+C` anytime).
- **Complete `10-second` uninstall (`--uninstall`):** zero background services, zero startup hooks, and zero system drivers are installed. A single command removes your local credentials (`~/.hivemind/credentials.json`) and VS Code snippet (`~/.continue/hivemind_provider.yaml`):
  ```bash
  # Linux / macOS / Windows (Python stdlib)
  python3 scripts/hivemind_quickstart.py --uninstall
  ```
  ```powershell
  # Windows 11 (native PowerShell)
  .\scripts\join_hivemind.ps1 -Mode Uninstall
  ```
  *(Optional: if you ran local GPU training rounds, you can also delete `~/.cache/hivemind/opentrainer` to reclaim cached adapter disk space.)*
