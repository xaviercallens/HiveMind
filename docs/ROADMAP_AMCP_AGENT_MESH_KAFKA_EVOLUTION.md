# 🗺️ Roadmap Evolution: The Bi-Layer Honeybee Agent Mesh (AMCP v1.5 + Kafka & ABC MoE)

> **Document Version:** 1.0.0 — Post-S21 Evolution  
> **Status:** APPROVED & ACTIVE  
> **Core Authors:** Xavier Callens (`@xaviercallens` / `u/OrganizationTop9026`), Claude Code Agent & Antigravity CLI Agent  
> **Invariants:** 8 GB VRAM ceiling, zero EUR/USD fiat (Kelly-market HMT compute reciprocity), <1.5s gamer preemption, bitwise determinism.

---

## 1. Executive Summary & Architectural Leap

The recent implementation of **AMCP v1.5 over Kafka** (`amcp_kafka_transport.py`, `amcp_kafka_campaign_bus.py`) provided HiveMind with a secure, signed CloudEvents 1.0 transport (HMAC-SHA256, DLQ, replay protection).

However, coordinating heterogeneous, decentralized MoE agents (Mixture of Experts) across citizen hardware requires more than static point-to-point handoffs. We are taking the definitive biological leap: **The Bi-Layer Honeybee Hive Mind Mesh**.

```
┌────────────────────────────────────────────────────────────────────────┐
│  Couche 1 : Consensus Réseau & Routage (Modèle de Scrutin de Nidification)│
│  • Waggle Dance sur Kafka (Durée & Intensité = tau_0 * Q)              │
│  • Recrutement Aveugle (MCL) & Verrouillage par Quorum Dynamique       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Élection Consumer Group Dédié
┌───────────────────────────────────▼────────────────────────────────────┐
│  Couche 2 : Résolution Agent & Outils (Artificial Bee Colony - ABC)    │
│  • Abeilles Employées : Exploitation interne des experts du MoE        │
│  • Abeilles Observatrices : Critique de qualité & détection de rejet   │
│  • Abeilles Éclaireuses : Exploration d'outils externes (Web, Z3, RAG) │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Phase-by-Phase Evolutionary Roadmap

```mermaid
flowchart LR
    S21[S21: AMCP Transport<br>CloudEvents + DLQ] --> S22[S22: Dynamic Quorum Sensing<br>Waggle Dance & MCL]
    S22 --> S23[S23: Distributed ABC MoE Mesh<br>Employed / Observer / Scout]
    S23 --> S24[S24: Swarm Nectar Cache<br>Cross-Bee Food Sources]
    S24 --> S25[S25: Lean 4 Quorum Proofs<br>Formal Convergence Invariants]
    S25 --> S26[S26: LayerStoRm x HiveMind<br>PCIe Expert Streaming & Two-Zone VRAM]
    S26 --> S34[S34: SwaT + Nocturnal Sleep<br>Wake KV Telepathy & NREM/REM Consolidation]
```

### 🔹 Milestone 1 (S22.1) : Moteur de Quorum Sensing & Danse Frétillante Kafka
- **Objectif :** Transformer le topic `hivemind.mesh.waggle_dance.v1` en système de consensus décentralisé en temps réel.
- **Livrables :**
  1. **Broadcast Dynamique :** Chaque agent éclaireur publiant une proposition de tâche module sa durée d'émission selon $\Delta t = \tau_0 \times Q_{\text{site}}$.
  2. **Recrutement Stochastique MCL :** Les pairs écoutent la danse et s'enrôlent si $\text{Affinité} \times Q \ge 0,35$.
  3. **Verrouillage par Quorum :** Dès que $N_{\text{recruits}} \ge K_{\text{quorum}}$, élection automatique d'un Kafka Consumer Group dédié (`cg-hivemind-<domain>-<task_id>`).
- **Métrique Gate :** Latence d'élection de quorum $< 450\text{ ms}$ sur 50 nœuds concurrents avec 0 collision de partition.

### 🔹 Milestone 2 (S22.2) : Boucle ABC Adaptative sur Petits Modèles MoE Edge
- **Objectif :** Déployer la boucle *Artificial Bee Colony* (ABC) sur notre nouvelle flotte de petits modèles (<4B) et nœuds consommateurs (CPU et GPU 8 Go).
- **Livrables :**
  1. **Abeilles Employées :** Routage d'experts MoE internes sur `hivemind/qwen35-4b`, `hivemind/phi4-mini`, `hivemind/gemma4-e4b`, `hivemind/qwen35-sub2b` pour extraire et formuler les premières hypothèses.
  2. **Abeilles Observatrices :** Calcul de la fitness $f(\text{source}) = 0,6 \cdot \text{Couverture} + 0,4 \cdot \text{NectarRichness}$. Si la fitness stagne après $N$ essais ($\text{abandonment\_limit} = 2$), la source est marquée épuisée.
  3. **Abeilles Éclaireuses :** Déclenchement automatique d'outils externes (MCP Web Search, solveurs Z3/SciPy, modèles de diffusion) pour ramener du nectar frais dans le prompt contextuel.
- **Métrique Gate :** Taux de complétion des tâches de recherche complexe $> 85\%$ sans intervention humaine, avec scoring de conformité JSON $> 95\%$.

### 🔹 Milestone 3 (S23) : Le Bus de Partage de Nectar (« Swarm Nectar Cache »)
- **Objectif :** Éliminer les redondances de calcul et de recherche web à travers la ruche via le topic `hivemind.mesh.food_sources.v1`.
- **Livrables :**
  1. **Cache P2P de Nectar Informationnel :** Dès qu'une éclaireuse valide un extrait web, une preuve de solveur ou une extraction de table, elle publie un artefact immuable hashé (SHA-256) sur le bus.
  2. **Réutilisation Inter-Abeilles :** Une autre abeille traitant une sous-tâche similaire consomme directement ce nectar sans requêter à nouveau le web ou ré-exécuter le solveur.
  3. **Compensation PoUW HMT :** L'abeille ayant découvert le nectar reçoit des micro-crédits HMT chaque fois que son nectar est réutilisé par l'essaim.
- **Métrique Gate :** Réduction de $60\%$ des requêtes web redondantes et division par 2 de la consommation de tokens sur les tâches répétitives.

### 🔹 Milestone 4 (S24) : Preuve Formelle de Convergence en Lean 4
- **Objectif :** Démontrer mathématiquement la stabilité et la convergence du Scrutin de Nidification sur réseau pair-à-pair asynchrone.
- **Livrables :**
  1. Spécification formelle dans `formal/ANSE/WaggleDanceQuorum.lean`.
  2. Théorème `theorem waggle_dance_quorum_convergence` : sous une distribution de bruits gaussiens sur l'évaluation $Q$, la probabilité d'élire le meilleur expert tend vers $1$ lorsque la taille du quorum $K \ge 3$.
  3. Intégration dans le compilateur CI `lake build`.
- **Métrique Gate :** `lake build` passe avec 0 `sorry` et 0 avertissement.

### 🔹 Milestone 5 (S25) : Collaboration LayerStoRm × HiveMind — Streaming PCIe d'Experts MoE & Cache VRAM Bi-Zone
- **Objectif :** Intégrer les mécanismes de streaming PCIe DMA et de placement NUMA en RAM hôte de **LayerStoRm** ([`kkontosis/LayerStoRm`](https://github.com/kkontosis/LayerStoRm)) comme backend d'exécution haut de gamme dans l'écosystème HiveMind, permettant d'héberger des modèles MoE géants (DeepSeek-V2/V3, GLM-4-MoE, Qwen3-MoE-30B+) sur des machines grand public sans dépasser l'enveloppe VRAM.
- **Livrables techniques clés (`HM-S25-01` à `HM-S25-05`) :**
  1. **Streaming PCIe DMA NUMA-Aware (`HM-S25-01`) :** Garder l'intégralité des poids d'experts routés en RAM hôte épinglée (pinned NUMA RAM 64–128 Go) et ne transférer sur le bus PCIe 4.0/5.0 que les $k$ experts activés par token.
  2. **Cache VRAM Bi-Zone couplé au Quorum Waggle Dance (`HM-S25-02`) :** Maintenir en VRAM les couches d'attention denses, le KV cache et un cache d'experts à deux zones (Zone A : experts chauds épinglés ; Zone B : anneau dynamique LRU pré-chargé dès l'élection du quorum Kafka `hivemind.mesh.waggle_dance.v1`).
  3. **Ordonnanceur DMA Waterline & FoMoE Skip-Attention (`HM-S25-03`) :** Recouvrir les transferts PCIe H2D de la couche $L+1$ avec le calcul d'attention de la couche $L$ grâce au pré-routage spéculatif FoMoE ($85,8\%$ de précision pré-attention).
  4. **Compression KV Hybride (`HiSparse` + `SnapMLA` + `E8 Lattice`, `HM-S25-04`) :** Fusionner le délestage KV épars de LayerStoRm avec le quantificateur sur réseau $E_8$ de HiveMind pour supporter des contextes de 32k–128k tokens sous le plafond de $6\,656\text{ MiB}$.
  5. **Adaptateur Producteur BYOB `layerstorm-frontier-moe` (`HM-S25-05`) :** Connecteur dans `hivemind_producer_agent.py` avec préemption gamer `< 0,5\text{ s}$ (libération immédiate des files DMA épinglées au lancement d'un jeu Steam/Blender) et rémunération en points de réflexion HMT.
- **Métrique Gate (Campagne pré-enregistrée `E19`) :** Débit de génération $> 18\text{ tok/s}$ sur modèles MoE $>30\text{B}$ avec 0 dépassement du plafond VRAM configuré et réduction de $60\%$ de la bande passante PCIe consommée sur les préfixes récurrents grâce au *Swarm Nectar Cache* (S23) placé en amont.

### 🔹 Milestone 6 (S34) : « Swarm Telepathy » (`SwaT` + `Attention Matching`) — Transfert Direct de KV-Cache Inter-Familles Sans Pré-Remplissage (`arXiv:2610.15095`)
- **Objectif :** Remplacer les résumés textuels (`T2T`) coûteux entre Abeilles de familles différentes (`Qwen3.5`, `SmolLM3`, `Gemma-3/4`, `Granite-3.3`, `Llama-3.2`, `Phi-4`) par le transfert direct et compacté de leur état latent interne (**KV-Cache**) via **SwaT** ([`Ar-Kareem/SwaT`](https://github.com/Ar-Kareem/SwaT), cloné dans [`third_party/SwaT/`](../third_party/SwaT/), [`docs/S34_SWAT_SWARM_TELEPATHY_INTEGRATION.md`](S34_SWAT_SWARM_TELEPATHY_INTEGRATION.md)).
- **Livrables techniques clés (`HM-S34-01` à `HM-S34-07`) :**
  1. **Traducteur de Cache `CacheRendererV2` & Alignement RoPE par Empans de Caractères (`HM-S34-01`) :** Dé-application du RoPE émetteur (`flatten_sender_cache`), cross-attention causale alignée sur les empans de caractères ($p_i = \max(0, j + \text{clip}_{[0,1]}((m_i - a^{\mathcal{R}}_j)/(b^{\mathcal{R}}_j - a^{\mathcal{R}}_j)) - 1/2)$) avec masque préfixe strict ($b^{\mathcal{S}}_i \le b^{\mathcal{R}}_j$), et recalibrage RMS par couche/tête (`bound_to_stats`).
  2. **Nectar Latent Compacté (`SwaT + AM_U + FP8/E8`, `HM-S34-02`) :** Combinaison de `SwaT` avec la compaction **Attention Matching (`AM`, `arXiv:2602.16284`)** et nos codecs KV `FP8` / réseau de Gosset $E_8$ sur le bus `SwarmNectarCache` (`hivemind.mesh.telepathy_kv.v1`), offrant en moyenne **$+40\%$ ($+48,8\%$ géométrique) de précision** et un **gain de vitesse $\times 4$** face au résumé textuel `T2T`.
  3. **Espace Latent Universel de la Ruche ($2n$ vs $n(n-1)$ Mappers, `HM-S34-03`) :** Passage d'une complexité quadratique $n(n-1)$ de paires émetteur-récepteur à $2n$ adaptateurs via un espace KV canonique partagé par toutes les abeilles BYOB.
  4. **Télépathie Zéro-Shot Vision $\to$ Texte (`HM-S34-04`) & Extension de Fenêtre Contextuelle `28K -> 5.6K` (`HM-S34-05`) :** Transfert de capacités visuelles d'une abeille VLM (`Gemma-3` / `Reddit-Vision`) vers une abeille purement textuelle (`Coder`/`Solver`) et compression de longs contextes pré-remplis sur `T4`/`LayerStoRm` vers les GPU citoyens `8 Go` (`RTX 2070`, `< 6 656 MiB`).
- **Métrique Gate (Campagne pré-enregistrée `E22`) :** Rétention zéro-shot $\ge 90\%$ (et $> 100\%$ en mode `Large -> Small`), accélération TTFT récepteur $\ge 2,5\times$ face à `T2T` à budget de tokens égal, zéro oubli catastrophique ($\Delta \mathcal{L}_{\text{guard}} \le 0$) + gain $\ge +2,5$ pts de `Knowledge Seeding + Dreaming` face au SFT nocturne classique (`P22d`–`P22e`), et preuve formelle Lean 4 (`Theorems 162–165`, `0 sorry`).
- **Extension Sommeil Nocturne à Deux Stades (`HM-S34-08` à `HM-S34-10`, *Language Models Need Sleep*, `arXiv:2606.03979v2`) :**
  5. **Stade 1 du Sommeil (NREM Consolidation — `Compute–Consolidate–Update` & `Knowledge Seeding`, `HM-S34-08`) :** Hiérarchie **Continuum Memory System (`CMS`, `1K -> 5K -> 10K`)** déverrouillant dynamiquement des experts MoE bas-rang ($r=64, \alpha=128$) dans les blocs lents, distillation ascendante (`KS` = `On-Policy GKD` + RL `Learning-to-Imitate` `LTI`) avec gel des anciens paramètres contre l'oubli catastrophique, puis remise à zéro/élagage synaptique des experts rapides.
  6. **Stade 2 du Sommeil (REM `Dreaming` — Mixage Aléatoire d'Experts MoE & Filtrage par Norme de Gradient, `HM-S34-09` & `HM-S34-10`) :** Génération de $m$ rêves synthétiques avec injection d'un expert bas-rang aléatoire hors-domaine par routeur MoE, sélection Top-$k$ + $b$ par norme de gradient SFT $\omega^{(i)} = \|\nabla_\theta \mathcal{L}_{\text{SFT}}(\text{DREAM}^{(i)}, \theta)\|$ (`4x` moins coûteux que SEAL), et récompense binaire `ReST^EM` ($r=1$ ssi le score aval $\tau(\cdot)$ progresse), consolidant le nectar KV d'éveil (`SwaT+AM`) en mémoire permanente.

---

## 3. Matrice de Responsabilités et Déploiement

| Rôle / Entité | Responsabilité Clé dans la Feuille de Route |
| :--- | :--- |
| **Routeur HiveMind OpenRouter** | Ingestion des tâches sur `hivemind.mesh.task.inbox.v1`, filtrage amont via *Swarm Nectar Cache*, et retour client sur `hivemind.mesh.result.v1`. |
| **Nœuds Consommateurs / Gamers (8 Go)** | Exécution des abeilles employées et observatrices en arrière-plan avec préemption instantanée (<1,5 s). |
| **Nœuds Edge / Laptops (CPU / 4-8 Go)** | Exécution des modèles légers (`qwen35-4b`, `phi4-mini`) en mode éclaireuse et extraction sémantique. |
| **Nœuds Prosumers Multi-GPU / LayerStoRm (16-24 Go VRAM + 64 Go+ RAM)** | Exécution des modèles MoE de frontière via streaming PCIe DMA (`HM-S25`) en tant que *Macro-Cluster Super-Bees*. |
| **Brokers Kafka / AMCP v1.5** | Acheminement ordonné, persistance rejouable, filtrage anti-rejeu et files DLQ. |
| **Lean 4 Prover** | Certification mathématique formelle des invariants du protocole et du budget mémoire Bi-Zone. |

---

## 4. Statut d'Implémentation Actuel (POC Validé)

Le prototype fonctionnel complet a été implémenté et validé :
- Moteur bi-couche : [`src/hivemind/swarm/abc_moe_agent_mesh.py`](file:///home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/HiveMind/src/hivemind/swarm/abc_moe_agent_mesh.py)
- Cache P2P de Nectar : [`src/hivemind/swarm/swarm_nectar_cache.py`](file:///home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/HiveMind/src/hivemind/swarm/swarm_nectar_cache.py)
- Suite de tests : [`tests/test_abc_moe_agent_mesh.py`](file:///home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/HiveMind/tests/test_abc_moe_agent_mesh.py) & [`tests/test_swarm_nectar_cache.py`](file:///home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/HiveMind/tests/test_swarm_nectar_cache.py)
- Documentation d'architecture : [`honeybee_mesh_kafka_abc_architecture.md`](file:///home/xavkal/.gemini/antigravity-cli/brain/3cbf7fb8-27af-4c04-b750-146f1c80a9c2/honeybee_mesh_kafka_abc_architecture.md)
- Plan détaillé de collaboration S25 : [`docs/S25_LAYERSTORM_COLLABORATION_PLAN.md`](file:///home/xavkal/xdev/AutoevolveAI/sub_projects/hivemind/HiveMind/docs/S25_LAYERSTORM_COLLABORATION_PLAN.md)

---

## 5. Plan de Collaboration LayerStoRm × HiveMind (Jours 1–3 à Semaines 1–4)

| Horizon Temporel | Phase | Actions Concrètes & Livrables Conjoints | Critère de Succès (Gate) |
| :--- | :--- | :--- | :--- |
| **Jours 1 à 3** | **Phase 1 : Prise de contact, Alignement d'Interface & Pré-enregistrement `E19`** | • Prise de contact avec `@kkontosis` sur `r/LocalLLaMA` et GitHub.<br>• Définition du contrat d'interface JSON/HTTP entre `hivemind_producer_agent.py` et le serveur local LayerStoRm.<br>• Rédaction du protocole expérimental pré-enregistré **`E19`** (`artifacts/experiments/E19_layerstorm_pcie_nectar_prereg.json`). | Schéma d'API validé et pré-enregistrement `E19` commité avec contrôles négatifs. |
| **Semaine 1** | **Phase 2 : Connecteur Producteur `LayerStoRmBackend` & Onboarding BYOB (`HM-S25-01`, `HM-S25-05`)** | • Implémentation de `src/hivemind/swarm/layerstorm_backend.py` (sonde de télémétrie VRAM + RAM hôte NUMA, arrêt DMA `< 0,5 s` sur signal de préemption jeu).<br>• Ouverture du tier `hivemind/frontier-moe-layerstorm` sur la Gateway OpenRouter. | Tests unitaires et d'intégration passés ; préemption vérifiée `< 500 ms`. |
| **Semaine 2** | **Phase 3 : Couplage Waggle Dance + Cache VRAM Bi-Zone & Swarm Nectar Cache (`HM-S25-02`, `HM-S25-03`)** | • Pré-chargement (prefetch PCIe) des experts de la Zone B dès la notification de quorum Kafka `hivemind.mesh.waggle_dance.v1` avant l'arrivée du premier token.<br>• Couplage du *Swarm Nectar Cache* (S23) en amont de LayerStoRm pour court-circuiter les préfixes de prompts connus. | Réduction mesurée $\ge 50\%$ des transferts PCIe redondants sur les requêtes multi-tours. |
| **Semaines 3 & 4** | **Phase 4 : Campagne Expérimentale `E19`, Preuve Lean 4 & Publication Conjointe (`HM-S25-04`)** | • Exécution de la campagne `E19` sur configurations grand public (RTX 3090/4070/5070 Ti + 64 Go RAM).<br>• Preuve formelle Lean 4 (`theorem layerstorm_two_zone_vram_ceiling` dans `formal/ANSE/HiveMindSpec.lean`).<br>• Publication conjointe dans le Foundation Paper `v4.1.0` sur Zenodo avec co-crédit complet. | `lake build` à 0 `sorry`, rapport `E19` signé SHA-256 et publié. |
