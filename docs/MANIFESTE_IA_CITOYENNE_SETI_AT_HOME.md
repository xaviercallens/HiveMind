# 🇫🇷 Pour une IA citoyenne et décentralisée en souvenir de SETI@Home

> **Discussions fondatrices de la communauté :**
> - **Manifeste sur `r/artificielle` :** [Pour une IA citoyenne et décentralisée en souvenir de SETI@Home](https://www.reddit.com/r/artificielle/comments/1xctxbf/pour_une_ai_citoyenne_et_d%C3%A9centralis%C3%A9e_en/)
> - **La question d'origine sur `r/artificielle` :** [Pourquoi n'existe-t-il pas une IA décentralisée et citoyenne, sur le modèle de Wikipédia, Linux ou Folding@home ?](https://www.reddit.com/r/artificielle/comments/1x6vijg/comment/phgj7z2/)
> - **Échanges techniques avec les développeurs sur `r/developpeurs` :** [Discussion architecture distribuée, PyTorch, P2P & RTX 2070](https://www.reddit.com/r/developpeurs/comments/1xhit4c/comment/19rtqukepw39f/)

---

## 🚀 Rejoindre l'Essaim en 30 secondes (Pour tout le monde / Easy to Join for Everyone)

Que vous soyez **simple citoyen curieux (sans carte graphique dédiée)**, **joueur PC (RTX 8–12 Go)**, **volontaire BOINC / Folding@home**, ou **développeur / chercheur**, vous pouvez tester et rejoindre **Project HiveMind** immédiatement :

| Votre Profil | Action en 1 clic | Lien Direct |
| :--- | :--- | :--- |
| 🌐 **1. Découvrir & Jouer sans aucune installation** *(Tous publics, navigateur web)* | Explorez la galaxie 3D (6 000+ exoplanètes NASA/JWST) et aidez la science via **WebGPU `@compute`** directement dans votre navigateur | 👉 [**Lancer le Jeu Scientifique WebGPU (Zéro Installation)**](https://xaviercallens.github.io/HiveMindatHome.github.io/webgpu.html) *(ou [`/webgpu`](https://hivemind-openrouter-1003063861791.us-central1.run.app/webgpu))* |
| 💬 **2. Tester l'IA Citoyenne gratuitement** *(Utilisateurs, étudiants, développeurs)* | Obtenez une clé API (`Bearer hm-...`) en 10 secondes (sans carte bancaire) et interrogez les 15 modèles MoE & Abeilles spécialisées en 5 langues (`FR`, `EN`, `ES`, `DE`, `ZH`) | 👉 [**Tester sur HiveMind OpenRouter**](https://hivemind-openrouter-1003063861791.us-central1.run.app/?lang=fr) |
| 🐝 **3. Prêter son GPU quand il dort (BOINC / Gamer RTX)** *(Cartes 8–16 Go : RTX 2060, 2070, 3060, 4070...)* | Entraînez un petit module expert QLoRA la nuit ou pendant que votre PC est inactif (pause automatique `< 1,5 s` dès que vous bougez la souris ou lancez un jeu Steam) | 👉 [**Guide Rejoindre & Entraîner (`/join#train`)**](https://hivemind-openrouter-1003063861791.us-central1.run.app/join#train) · [**Version Française (`/rejoindre#train`)**](https://hivemind-openrouter-1003063861791.us-central1.run.app/rejoindre#train) |
| 🏛️ **4. Explorer La Ruche & Contribuer au Code** *(Chercheurs, juristes, médecins, devs PyTorch/P2P)* | Suivez les 24 Abeilles MoE en direct sur **La Ruche** et contribuez aux datasets citoyens ou au code open-source | 👉 [**Portail La Ruche**](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/) · [**Dépôt GitHub `xaviercallens/HiveMind`**](https://github.com/xaviercallens/HiveMind) |

---

## 📖 Le Récit Fondateur : Pourquoi maintenant ?

Il y a quelques jours sur [`r/artificielle`](https://www.reddit.com/r/artificielle/comments/1x6vijg/comment/phgj7z2/), un utilisateur non-informaticien a posé une question toute simple mais finalement fondamentale :

> *« Puisque les IA actuelles ont appris grâce aux textes produits par tout le monde sur Internet, pourquoi les modèles les plus puissants appartiennent-ils tous à une poignée de géants (OpenAI, Anthropic, Google) ? Pourquoi n'existe-t-il pas une IA décentralisée et citoyenne, sur le modèle de Wikipédia, Linux ou Folding@home ? »*

En lui répondant en vulgarisant l'état de la recherche, je me suis dit que le sujet méritait d'être partagé et débattu, d'autant que j'ai codé un prototype open-source complet sur ma propre machine (**une vieille RTX 2070 de 8 Go**) pour vérifier que les briques techniques tenaient réellement la route.

Voici pourquoi c'était techniquement impossible il y a encore deux ans, et pourquoi **trois percées récentes** changent complètement la donne.

---

### 1. Pourquoi nos PC domestiques ne pouvaient pas rivaliser avec un Data Center ?

Jusqu'ici, entraîner un grand modèle de langage (LLM) se heurtait à **deux murs physiques** :

1. **Le mur de la mémoire (VRAM) :** Un grand modèle monolithique (70 milliards de paramètres et plus) demande des centaines de gigaoctets de mémoire vidéo ultra-rapide. Une carte graphique de joueur (**8 à 12 Go de VRAM**) est incapable de le charger en entier.
2. **Le mur de la latence (notre box fibre) :** Dans un projet citoyen classique (comme *SETI@home* ou *Folding@home*), chaque PC calcule dans son coin pendant des heures et renvoie son résultat à la fin. Mais un réseau de neurones classique fonctionne comme un seul cerveau : jusqu'ici, toutes les cartes graphiques devaient s'échanger des gigaoctets de données **à chaque milliseconde** (le paradigme *SPMD*). Dans un hangar rempli de câbles optiques à très haut débit (*InfiniBand*), ça marche. Sur Internet entre des box fibre domestiques (**20 à 50 ms de latence**), les cartes graphiques passeraient **99 % de leur temps à s'attendre mutuellement**.

---

### 2. Ce qui vient de changer : DiLoCo + les « Essaims d'Experts » (MoE)

Trois avancées scientifiques récentes permettent de contourner ces deux murs sans avoir besoin d'un supercalculateur centralisé :

#### A. L'entraînement par « îlots » asynchrones (*DiLoCo* et *Decoupled DiLoCo*)
Des chercheurs (notamment chez DeepMind, puis repris en open-source avec *OpenDiLoCo*) ont mis au point une méthode qui casse l'obligation de communiquer à chaque milliseconde :
- Au lieu que 1 000 ordinateurs travaillent sur la même phrase en se téléphonant à chaque mot, chaque PC (ou petit groupe de PC) travaille de façon autonome pendant **des centaines d'étapes (`H=500`)** sur son propre sujet, avec **zéro communication réseau**.
- Ce n'est que périodiquement qu'ils mettent leurs apprentissages en commun.
- **Résultat :** cela divise par **environ 500** le besoin en bande passante. Mieux encore, la version *découplée* (*Decoupled DiLoCo*) permet de ne plus attendre les machines lentes ou celles qui se déconnectent : **si un particulier éteint son PC ou lance un jeu, le reste de l'essaim continue sans jamais bloquer**.

#### B. Découper le géant en petits spécialistes (*Branch-Train-Mix* & *Mixture-of-Experts*)
Plutôt que d'entraîner un bloc monolithique géant que personne ne peut faire tourner chez soi, on part d'un **socle commun léger (compressé en 4 bits NF4 pour tenir en mémoire)** et on divise l'intelligence en une multitude de **petits modules experts spécialisés (LoRA)** d'une quinzaine de mégaoctets chacun : un expert en code, un en mathématiques, un en rédaction française, un en médecine/droit, un en astrophysique, etc.
- Chaque carte graphique grand public (**même avec seulement 8 Go de VRAM**) peut entraîner son petit module expert dans son coin.
- Une fois entraînés, ces spécialistes sont fusionnés dans une architecture dite **Mixture-of-Experts (MoE)** : quand vous posez une question, un aiguilleur (« routeur ») n'active que les **2 experts pertinents**.

#### C. Éliminer les tricheurs sans refaire deux fois le calcul
Sur un réseau ouvert sur Internet, comment vérifier qu'un participant n'envoie pas de faux calculs (carte graphique instable ou acte malveillant) sans obliger deux personnes à calculer la même chose ?
- En enregistrant des **empreintes cryptographiques (SHA-256)** à intervalles réguliers pendant l'apprentissage, un validateur n'a besoin de rejouer qu'un tout petit segment tiré au sort (**~10 % du travail**) pour vérifier l'honnêteté du calcul, combiné à un filtre statistique (**Multi-Krum**) qui écarte automatiquement les mises à jour aberrantes.

---

### 3. À quoi ressemble une vraie « IA Citoyenne » (et le test concret sur ma RTX 2070) ?

L'idée a aussi un **intérêt écologique évident** : des millions de cartes graphiques (**RTX 2060, 2070, 3060, 4070...**) ont **déjà été fabriquées** et dorment dans nos PC la nuit ou quand nous travaillons, pendant qu'on construit des datacenters géants supplémentaires.

Le fonctionnement repose sur **4 piliers**, inspirés des communs (*Wikipédia*, *BitTorrent*, *SETI@home / Folding@home*) :

1. **📚 La Ruche à Données (façon Wikipédia / Common Voice) :** Pas besoin de gros GPU pour contribuer. Des citoyens, chercheurs, juristes, médecins ou développeurs constituent et vérifient des jeux de données propres et ouverts dans leur domaine, hébergés en pair-à-pair ([La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/)).
2. **🛡️ Zéro intrusion sur le PC :** Un petit démon logiciel qui ne travaille que lorsque le PC est totalement inactif. Dès qu'on bouge la souris ou qu'on lance un jeu, il sauvegarde son état sur le disque SSD en **moins d'une seconde et demie** et libère **100 % de la carte graphique**.
3. **⚖️ La réciprocité (façon ratio BitTorrent / Preuve de Calcul Utile) :** Pas de spéculation financière ni de minage inutile qui brûle de l'électricité dans le vide : quand votre PC aide à entraîner ou héberger un expert la nuit, vous accumulez des **Points de Réflexion HiveMind (HMT)** qui vous permettent d'interroger l'essaim gratuitement quand vous en avez besoin.
4. **⚡ Le test concret sur une RTX 2070 (8 Go de VRAM) (`Project HiveMind`) :**
   - Le socle compressé en 4-bit (`NF4`) ne pèse que **176 Mo** de poids (moins d'1 Go de contexte CUDA actif).
   - **4 branches spécialisées** (*Code*, *Mathématiques*, *Sciences/STEM*, *Français*) ont été entraînées selon la logique *DiLoCo-QLoRA* (chaque mise à jour compressée ne pèse que **5 Mo** à envoyer sur le réseau), avec rejet automatique de nœuds empoisonnés par *Multi-Krum*.
   - Une fois fusionnés en un modèle **Sparse MoE (96 modules experts FFN répartis sur 24 couches)**, l'ensemble culmine à **1,5 Go de VRAM** sur les 8 Go de la carte, tout en divisant par **5** l'impact de la latence réseau lors de la génération de texte grâce au décodage spéculatif.

---

## 🛠️ Guide Rapide : Comment Participer en 1 Commande ?

### Option 1 — Sans aucune installation (Navigateur WebGPU)
Ouvrez simplement [**https://xaviercallens.github.io/HiveMindatHome.github.io/webgpu.html**](https://xaviercallens.github.io/HiveMindatHome.github.io/webgpu.html) (ou [**https://hivemind-openrouter-1003063861791.us-central1.run.app/webgpu**](https://hivemind-openrouter-1003063861791.us-central1.run.app/webgpu)) pour parcourir la galaxie, poser vos sondes sur de vraies exoplanètes NASA/JWST et prêter votre GPU en arrière-plan via WebGPU `@compute`.

### Option 2 — Tester l'API OpenAI-compatible ou l'Aire de Jeu (5 langues)
Rendez-vous sur [**https://hivemind-openrouter-1003063861791.us-central1.run.app/fr**](https://hivemind-openrouter-1003063861791.us-central1.run.app/fr) (ou [`/signup`](https://hivemind-openrouter-1003063861791.us-central1.run.app/signup)) pour créer un compte gratuit en un clic et tester les modèles MoE dans votre navigateur, dans **Cursor**, **Continue.dev**, ou via `curl`.

### Option 3 — Rejoindre l'entraînement nocturne BOINC / Partager son GPU (Linux, Windows, macOS)
Consultez le guide interactif [**https://hivemind-openrouter-1003063861791.us-central1.run.app/join#train**](https://hivemind-openrouter-1003063861791.us-central1.run.app/join#train) (ou [**`/rejoindre#train`**](https://hivemind-openrouter-1003063861791.us-central1.run.app/rejoindre#train)) :

```bash
# Télécharger et lancer le travailleur BOINC / GPU-inactif (pause automatique dès que vous jouez) :
curl -sLO https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/scripts/hivemind_boinc_idle_worker.py
python3 hivemind_boinc_idle_worker.py \
  --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app \
  --handle votre-pseudo \
  --domain physics_sim \
  --once
```

---

## 🌍 English Summary (For the Geeks & International Contributors)

**Project HiveMind** is an open-source distributed training and inference framework (think **SETI@home / Folding@home** meets **Petals** and **DiLoCo**) designed to aggregate idle consumer gaming GPUs (**8 GB–12 GB VRAM**) and local NVMe storage into a community-owned **Sparse Mixture-of-Experts (MoE)** cluster:

- **La Ruche Community Dashboard:** [https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/)
- **To Test (5-Language OpenRouter & Playground):** [https://hivemind-openrouter-1003063861791.us-central1.run.app](https://hivemind-openrouter-1003063861791.us-central1.run.app)
- **To Join (BOINC Night Training & Idle-GPU Sharing):** [https://hivemind-openrouter-1003063861791.us-central1.run.app/join#train](https://hivemind-openrouter-1003063861791.us-central1.run.app/join#train)
- **To Join Without Installation & Play a Scientific Game:** [https://xaviercallens.github.io/HiveMindatHome.github.io/webgpu.html](https://xaviercallens.github.io/HiveMindatHome.github.io/webgpu.html)
- **GitHub Repository:** [https://github.com/xaviercallens/HiveMind](https://github.com/xaviercallens/HiveMind)
