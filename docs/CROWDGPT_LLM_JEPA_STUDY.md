# Étude : compléter CrowdGPT (`Vxtzq/Crowd-v1`) avec LLM-JEPA

Rédigée le 13 novembre 2026. Source : Huang, LeCun, Balestriero, *LLM-JEPA: Large Language Models Meet Joint Embedding
Predictive Architectures*, arXiv:2509.14252v2 (lu en HTML ; code `github.com/rbalestr-lab/llm-jepa`). Contexte CrowdGPT :
`HiveMind/docs/CROWDGPT_GPU_LOSS_CHECK.md`, `CROWDGPT_COLLABORATION_AGREEMENT.md`, `CROWDSWARM_JOINT_WORK_PROPOSAL.md`,
`src/hivemind/swarm/crowdgpt_loss_check.py`.

**Règle appliquée** (accord CrowdGPT × HiveMind, §2) : aucun chiffre non mesuré. Tous les gains cités ci-dessous sont **ceux du
papier, sur d'autres modèles** (surtout Llama-3.2-1B-*Instruct*). Ce ne sont pas des prévisions pour Crowd-v1.

---

## 0. En bref

1. **La piste est bonne, mais pas pour la raison avancée dans le message.** Le papier ne montre pas que LLM-JEPA remplace des
   dizaines de milliards de tokens de pré-entraînement. Ses résultats solides viennent tous du **fine-tuning** d'un modèle déjà
   compétent, sur des données qui ont **deux vues naturelles** d'un même contenu (texte ↔ regex, texte ↔ SQL, question ↔ réponse).
   Côté pré-entraînement, il n'y a que des résultats préliminaires : environ +1 point, sur un corpus de paraphrases.
2. **Là où ça colle vraiment** : la phase SFT de Crowd-v1 et les greffons QLoRA de domaine de la Ruche (Layer 2 de la
   proposition commune). Le papier montre que LoRA + JEPA donne de très bons résultats.
3. **Ce que ça coûte** : environ ×2 de calcul à l'entraînement, ramené à ×(2 − α) avec le *loss dropout*. L'inférence ne coûte
   rien de plus. Deux hyperparamètres s'ajoutent (λ, k).
4. **Mesuré le 13/11 sur le T4 (E0, §3)** : Crowd-v1 (révision `5ac3916`, step 366) est sain. Sa loss held-out vaut
   **3,03–3,19 nats** sur des chunks CrowdGPT et **4,86** sur wikitext-2, contre **2,54–2,78** et **3,02** pour Qwen3-0.6B.
   Les « 8 à 11 nats » d'avant venaient d'une ancienne révision mesurée par un loss check **qui ignorait la table engram** (sans
   elle : +0,35 nat). L'hypothèse de mémorisation est **réfutée** : le mode cohorte actuel fait 0,78 passe par sous-tranche, et
   l'écart vu / non vu n'est que de 0,16 nat. Le correctif utile est dans le loss check (PR HiveMind), pas dans le client.
5. Deux corrections au message d'origine. Crowd-v1 compte **657 787 392 paramètres**, mesurés sur les poids réels, et non 1B.
   Et le papier n'est pas « tout juste sorti » : la v1 date de septembre 2025.
6. **Pilote P1 sur le T4 (nuit du 13 au 14/11, §9) : résultat négatif.** Crowd-v1 + LoRA r64 sur NL-RX-SYNTH, 3 graines par
   bras : SFT 17,07 ± 0,93 %, SFT + JEPA 16,90 ± 1,06 % (Δ −0,17 point, différences appariées −0,3 / 0,0 / −0,2). La loss générale
   se dégrade de **+0,90 nat** avec JEPA. Aucun des critères écrits d'avance n'est atteint. Le terme JEPA s'effondre
   (1 − cos → 0,0005). Ça ne réfute pas le papier (pleine mise au point, modèle Instruct), mais la méthode ne se transpose pas
   telle quelle à Crowd-v1 + LoRA r64.
7. **autoresearch (façon Karpathy, 14/11, §10)** : 10 hypothèses, 5 testées à budget de calcul fixe.
   - **Aucune ne bat le SFT.** La meilleure combinaison (λ 0,1 + loss dropout + cosinus centré) l'égale à 0,001 nat près.
   - Les 3 retenues selon la règle écrite d'avance (stop-gradient, cosinus centré, InfoNCE) sont celles qui **suppriment
     l'effondrement et la dégradation de la loss générale**.

---

## 1. Ce que fait exactement LLM-JEPA (d'après le papier)

**Perte** :

    L = Σ_ℓ L_NTP(Text, Code)  +  λ · d( Pred(Enc(Text)), Enc(Code) )      avec d = 1 − cos

- **Enc(x)** : l'état caché de la **dernière couche** au **dernier token** de la vue x.
- **Pred** : il n'y a pas de réseau séparé. On ajoute k tokens spéciaux `[PRED]` après Text, et l'état caché du dernier `[PRED]`
  sert de prédiction. Les poids sont ceux du LLM, donc partagés. Avec k = 0, le prédicteur est l'identité.
- **Métrique** : le cosinus est nettement meilleur que les autres. Sur NL-RX-SYNTH, l'ablation donne 71,46 % avec le cosinus,
  2,22 % avec ℓ2 et 70,64 % avec MSE.
- **Anti-effondrement** : c'est la perte next-token qui l'assure. Sans elle (γ = 0), le modèle ne produit que des sorties vides.
  Le terme JEPA joue donc le rôle d'un **régulariseur**, pas d'un objectif autonome.
- **Implémentation** : Text et Code sont **empaquetés dans la même fenêtre**, avec un masque d'attention **bloc-causal** (causal à
  l'intérieur de chaque vue, aucune attention d'une vue vers l'autre). Il n'y a pas de passes avant séparées.
- **Coût** : le papier se contredit sur le décompte des passes (« une passe en plus » en §3.1, « deux au lieu de trois » en
  §3.2). Sa conclusion parle d'un coût **×2** à l'entraînement. Le **loss dropout** (LD = α) saute le terme JEPA sur une fraction
  α des batches, ce qui donne un coût de (2 − α) × SFT. Le papier tolère α = 0,5–0,75 en gardant λ·(1 − α) à peu près constant.
- **Hyperparamètres** : la grille de fine-tuning est k ∈ {0..4} × λ ∈ {0,5, 1, 2, 4}. Le λ optimal varie beaucoup selon la
  tâche (jusqu'à 1024 sur NQ-Open), et les auteurs n'ont pas de méthode de recherche efficace.

**Résultats publiés, Llama-3.2-1B-Instruct, SFT → LLM-JEPA (précision %)**

| Tâche (vues) | SFT | LLM-JEPA | λ, k |
|---|---|---|---|
| NL-RX-SYNTH (texte ↔ regex) | 57,29 ± 5,32 | 71,46 ± 1,34 | 1, 1 |
| NL-RX-TURK | 22,49 ± 1,91 | 30,94 ± 1,13 | 1, 1 |
| GSM8K (vues non décrites dans le papier) | 32,36 ± 0,58 | 36,36 ± 0,20 | 0,5, 4 |
| Spider (schéma + question ↔ SQL) | 47,52 ± 2,44 | 50,55 ± 2,08 | 1, 3 |
| NQ-Open (question ↔ réponse) | 20,12 ± 0,41 | 21,59 ± 0,40 | 1024, 0 |
| HellaSwag (contexte ↔ suite) | 69,40 ± 0,99 | 70,51 ± 1,20 | 1, 3 |

Les gains sont gros sur les tâches à vues très structurées (regex), et faibles, de l'ordre de 1 à 3 points, ailleurs. Ils sont
aussi faibles sur les modèles déjà bien entraînés : OLMo-2-1B passe de 87,09 à 87,52, Qwen3-1.7B de 44,32 à 45,00.

**LoRA** : au rang 512, LoRA + JEPA atteint 72,41 ± 2,94, contre 50,18 pour LoRA + SFT seul et 70,42 pour un fine-tuning complet
avec JEPA. Le papier note aussi que JEPA « résiste au surapprentissage » sur six époques, alors que le SFT seul surapprend.

**Pré-entraînement** (préliminaire, selon les auteurs) :
- Llama-3.2-1B entraîné depuis zéro sur NL-RX-SYNTH : 54,38 → 60,59.
- Pré-entraînement sur `cestwc/paraphrase` (la paraphrase i prédit la i+1), puis fine-tuning : Rotten Tomatoes 56,57 → 57,76,
  Yelp 26,46 → 27,15.

Aucune expérience ne porte sur un corpus brut de type FineWeb.

---

## 2. Ce que ça veut dire pour l'objectif « rivaliser avec Qwen 0.8B »

- Le corpus de Crowd-v1 (Ultra-FineWeb-L3, fineweb) est du texte brut, **sans vues appariées**. Le papier dit lui-même qu'il
  faudrait, pour des données quelconques, construire des vues « à la manière d'une augmentation de données ». C'est une question
  de recherche ouverte, pas une recette.
- Le papier ne fournit donc **aucune preuve** que LLM-JEPA réduise le nombre de tokens de pré-entraînement nécessaires. Le
  présenter comme un raccourci vers 100 milliards de tokens serait une affirmation non étayée.
- En revanche, pour un modèle de 658M, la phase SFT et les greffons de domaine sont exactement le cas d'usage du papier : même
  taille (~1B), données appariées, LoRA, et un bénéfice anti-surapprentissage appréciable quand les données sont rares.

**Recommandation** : JEPA en **SFT et greffons (Crowd-v1)**, et en **expérience de pré-entraînement sur la seule partie code**
(Crowd-v2), comme projet de recherche avec un critère de succès écrit d'avance. Pas de « pré-entraînement JEPA » sur tout le corpus.

---

## 3. Mesures E0 (T4, 13/11/2026) : l'état réel de Crowd-v1

**Ce que j'avais supposé, et qui était faux.** La version précédente de ce document attribuait l'écart entre `lastLoss` = 0,079 et
une loss held-out de 8 à 11 nats à une mémorisation de la sous-tranche de 10 Mio par les clients. Les mesures ci-dessous la réfutent.

**État actuel** (`status.json`, 13/11 21:22 UTC) : `lastLoss` 2,98, 37 G tokens vus, step 366, mode cohorte actif avec
`targetBatchSize` 2 et `targetLocalSteps` 500. Ça donne 500 × 2 = 1 000 fenêtres pour 1 280 par sous-tranche, soit **0,78 passe** :
aucune répétition. Le mécanisme « jusqu'à 25 passes » (500 micro-batches × bs) n'existe qu'hors cohorte, sur des GPU de 40–80 Go.

**E0 : loss held-out mesurée** (`scripts/crowdgpt_t4/e0_heldout_sanity.py`, résultats bruts dans
`artifacts/experiments/CROWDGPT_E0_heldout_2026-11-13.json`). 12 fenêtres de 2 048 tokens par source, graine 20261113, même
tokenizer Qwen3 pour tous les modèles :

| Modèle | chunk 1026, octet 0 (« vu », pool du 12/11) | chunk 1026, octet 200 Mo (« loin ») | wikitext-2 test |
|---|---|---|---|
| Crowd-v1 + engram, fp16 / fp32 / bf16 | **3,030** / 3,030 / 3,030 | **3,194** / 3,194 / 3,194 | **4,855** / 4,855 / 4,856 |
| Crowd-v1 **sans** engram, fp16 | 3,376 | 3,570 | 5,055 |
| Qwen3-0.6B, fp16 (référence) | 2,536 | 2,776 | 3,020 |

Ce qu'on en tire :
1. **Le modèle apprend.** 3,0–3,2 nats sur ses propres données, loin de l'uniforme (11,93). Le statut « vu » du chunk 1026 n'est
   qu'une inférence (log des pools) ; l'écart vu / loin (0,16 nat) est du même ordre que l'écart chez Qwen (0,24), donc rien
   n'indique de mémorisation.
2. **L'engram compte : +0,35 nat sans lui.** Le loss check HiveMind (`crowdgpt_loss_check.py`) l'ignorait : il mesurait un autre
   modèle. Corrigé dans cette PR. Le reçu couvre maintenant aussi la table engram (le SHA-256 porte sur tronc ‖ engram, sans
   changement de schéma).
3. **fp16 = fp32 = bf16** à 10⁻³ près sur le T4. On peut entraîner en fp16 (natif sur sm75) sans débordement visible.
4. **Où est l'écart avec Qwen3-0.6B** : environ 0,45 nat sur la distribution CrowdGPT, et **1,8 nat sur wikitext** (texte
   encyclopédique). Le second écart dépasse nettement le premier, ce qui suggère un **mélange de données pauvre en texte
   encyclopédique ou de référence**. C'est une hypothèse, à tester (§5, E1).
5. Coût sur le T4 : pic de 1,7 Go (fp16) à 2,9 Go (fp32) en inférence ; VRAM libre jamais sous 6 545 Mio (garde : 4 300 Mio).

**Le code client reste à surveiller** (lu dans `client.py` `d1983aa`, mais **sans effet mesurable aujourd'hui**) :
- hors cohorte, `STEPS_PER_SUBCHUNK = 500` compte des micro-batches, donc le nombre de passes croît avec le bs ;
- les fenêtres sont alignées sur des multiples fixes de `tps` ;
- `lastLoss` est la loss d'entraînement du dernier micro-batch.

Ce sont des améliorations d'hygiène, pas la priorité.

## 4. Intégration dans CrowdGPT : ce qui change, ce qui ne change pas

| Élément | Changement | Remarque |
|---|---|---|
| **Format des données** | Les shards `.bin` plats n'ont pas de frontières de vues. Il faut un format « paire » (tokens + offsets `text_start, text_end, code_start, code_end`), ou un index JSON à côté du shard. | Les shards texte existants restent utilisables pour la perte NTP seule. |
| **Masque d'attention** | `F.scaled_dot_product_attention(..., is_causal=True, enable_gqa)` devient un `attn_mask` bloc-causal explicite. | **Perte probable du chemin fusionné rapide** (Flash) avec un masque arbitraire, ce qui coûte plus de mémoire et de temps. Le client a déjà un chemin à masque manuel (`_chunked_attn`, fallback ROCm) qu'on peut adapter. Alternative : deux passes séparées (Text, Code), plus simples. À mesurer. |
| **Token `[PRED]`** | Ajouter une ligne au vocabulaire changerait `vocabSize`, les offsets du tenseur plat, `CROWD_V1_MODEL_SIZE`, `n_flat` et la borne `ln(vocab) + 5` du loss check. | **À éviter.** Réutiliser un token spécial Qwen3 réservé et inutilisé (vérifier lequel n'apparaît jamais dans les shards), ou partir sur **k = 0** (prédicteur identité, aucun token ajouté). k = 0 est un choix valide : c'est le meilleur réglage sur NQ-Open. |
| **Perte côté client** | `loss = γ·ce + λ·(1 − mean(cos(h_pred, h_code)))` avec loss dropout α. | Le code de référence n'a **pas de stop-gradient** : le gradient passe par les deux vues (`rbalestr-lab/llm-jepa` `finetune.py` l. 710–739, commit `ea0017c`, aucun `detach`). Le dépôt propose aussi des variantes ℓ2, MSE et InfoNCE (température 0,07). |
| **Agrégation** (Multi-Krum, trimmed mean) | Aucun changement. | Les deltas restent des deltas de poids. |
| **Loss check / ratchet** | Aucun changement. Le gate reste la **cross-entropy next-token held-out pure**. | L'inférence étant inchangée, c'est la bonne mesure ; la perte JEPA ne doit pas entrer dans le gate. Pour les rounds JEPA, ajouter éventuellement une métrique de tâche (précision NL-RX…) à côté. |
| **Inférence / service** (`hivemind/crowd-v1`) | Aucun changement. | Le token `[PRED]` n'est jamais émis. |

**Mémoire GPU (estimation, pas une mesure)** : en fine-tuning complet, il faut 658M × (4 o de poids + 4 de gradients + 8 d'états
Adam), soit environ 10,5 Go avant les activations, qui doublent avec deux vues empaquetées. Ça ne tient pas sur la RTX 2070
(8 Go). Deux voies :
- **LoRA ou QLoRA**, la voie naturelle, et celle où le papier est le plus convaincant ;
- bf16 + optimiseur 8-bit. Le client a déjà `AdamW8bit` (bitsandbytes) et le checkpointing d'activations, à mesurer avec les
  deux vues empaquetées.

Sur le T4, la règle AGENTS.md s'applique : **≥ 4 300 MiB de VRAM libres** à tout moment, et passage par le gouverneur
(inférence d'abord, calcul seulement dans les fenêtres d'inactivité).

---

## 5. Plan d'expériences proposé

**P0 — Données (hors JEPA, une nuit de T4)** : E0 montre un écart de 1,8 nat avec Qwen3-0.6B sur wikitext, contre 0,45 sur la
distribution CrowdGPT. Il faut mesurer la loss held-out de Crowd-v1 par **source** (code, web, encyclopédique, français) pour savoir
où ajouter des données. C'est le levier le moins cher : il ne change ni le client ni la perte.

**P1 — Pilote LLM-JEPA sur le T4** (`scripts/crowdgpt_t4/`, sous la garde VRAM ≥ 4 300 Mio, la nuit, en cédant à l'inférence)
- Modèle : Crowd-v1 `5ac3916` + engram (gelé), **LoRA** rang 64 sur toutes les projections des blocs, fp16 + GradScaler.
  Budget mémoire mesuré : environ 5 Go disponibles sans toucher aux services d'inférence (le fine-tuning complet n'y tient pas).
- Données : NL-RX-SYNTH (texte ↔ regex), la tâche au plus fort signal du papier ; jeu fourni par `rbalestr-lab/llm-jepa`.
- Bras : SFT seul / SFT + JEPA (λ = 1, k = 0 : prédicteur identité, aucun token ajouté au vocabulaire), **3 graines par bras**.
  L'écart type du baseline du papier est de ± 5,3 : une seule graine ne dit rien.
- **Seuil plancher, écrit d'avance** : si la précision SFT seule est < 5 %, le pilote est déclaré non informatif (le modèle est
  trop faible pour la tâche) et on passe à une tâche plus facile, au lieu d'interpréter du bruit.
- **Critère de succès, écrit d'avance** : JEPA est retenu si (moyenne JEPA − moyenne SFT) > 2 × l'écart type groupé des graines,
  **et** si la loss held-out générale (chunk « loin » de E0) ne se dégrade pas de plus de 0,05 nat.
- Mesurer aussi le coût réel (s/pas, VRAM pic) pour chaque bras.

**P2 — Greffons de domaine de la Ruche** : appliquer le même protocole à un domaine où les vues sont naturelles. Candidats : code
↔ docstring, énoncé Lean ↔ preuve, question ↔ réponse en français.

**P3 — Crowd-v2, pré-entraînement partiel (recherche)** : sur la fraction code du corpus, construire des vues **docstring/commentaire
↔ corps de fonction** et activer JEPA sur ces seuls batches, avec LD élevé. Comparer la loss held-out *et* une tâche code à budget
de tokens égal. C'est le test honnête de l'hypothèse « moins de tokens », et il peut échouer.

**Piste à lire avant P3** : le même dépôt contient `stp.py` (« Semantic Tube Prediction », `--linear=random_span`). C'est une
variante de fine-tuning qui travaille sur des spans à l'intérieur d'une seule passe empaquetée, toujours à partir de données
user/assistant. Elle ne supprime donc pas le besoin de vues appariées, mais elle pourrait réduire le surcoût. Pas encore évaluée ici.

**Distribué (DiLoCo / rounds de durée fixée par le serveur, `max_round_hours`, 2 h par défaut côté client)** : le terme JEPA est local au client et n'exige aucune synchronisation supplémentaire. Le
coordinateur doit seulement distribuer des shards « paires » et publier (λ, k, α) dans la config du round, pour que tous les
clients optimisent la même perte.

---

## 6. Licences et précautions

- Le papier est sous CC BY-NC-SA 4.0 : on peut s'inspirer de la méthode, mais pas recopier le texte ou les figures dans une doc
  commerciale.
- Le code `rbalestr-lab/llm-jepa` est sous **Apache-2.0** (`LICENSE`, commit `ea0017c`), compatible avec une inclusion dans
  `Vxtzq/CrowdGPT` (GPL-3.0) à condition de garder l'avis de licence. Réimplémenter la perte (une dizaine de lignes) reste plus
  simple.
- Dans toute communication (Suite #3, r/artificielle), présenter les gains comme « résultats du papier sur Llama-3.2-1B », jamais
  comme des résultats Crowd-v1 tant que P1 n'a pas tourné.

## 7. Questions ouvertes pour @Vxtzq

1. Quel jeu SFT est prévu pour Crowd-v1 ? A-t-il des vues naturelles (instruction ↔ réponse, texte ↔ code) ?
2. Es-tu d'accord pour ajouter un format de shard « paire » (offsets de vues) à côté des `.bin` actuels ?
3. Token `[PRED]` : réutiliser un token spécial Qwen3 réservé (lequel ?), ou partir sur k = 0 ?
4. Mélange de données : quelle part d'encyclopédique / de référence dans les chunks ? E0 montre 1,8 nat d'écart avec Qwen3-0.6B
   sur wikitext, contre 0,45 sur tes propres chunks.
5. Qui fait tourner P1 : un GPU HiveMind (T4 sous gouverneur / RTX 2070), un contributeur CrowdGPT, ou les deux en parallèle pour
   recouper ?

## 8. Proposition d'implémentation (PRs du 13/11/2026)

**HiveMind** (`crowdgpt/llm-jepa-study-t4`)
- `src/hivemind/swarm/crowdgpt_loss_check.py` : le loss check mesure le modèle avec engram (`CROWD_V1_ENGRAM_CONFIG`,
  `load_flat(trunk, engram=…)` ou tronc ‖ engram en un tenseur ; le reçu lie alors aussi l'engram). Le défaut reste tronc seul,
  pour la compatibilité. CLI : `--base-engram` / `--candidate-engram`. Tests : `tests/test_crowdgpt_loss_check_engram.py`.
- `scripts/crowdgpt_t4/` : garde T4 (`t4guard.py` : VRAM ≥ 4 300 Mio, plafond mémoire du processus, pause sur requêtes nginx,
  verrou) et E0 (`e0_heldout_sanity.py`). Résultats : `artifacts/experiments/CROWDGPT_E0_heldout_2026-11-13.json`.
- Ce document, et la correction du « shard de 1 Mo » dans `CROWDSWARM_JOINT_WORK_PROPOSAL.md`.

**CrowdGPT** (`xaviercallens/CrowdGPT` → `Vxtzq/CrowdGPT`, `feat/llm-jepa-opt-in`)
- `crowd_jepa.py` : perte LLM-JEPA autonome et **opt-in** (cosinus entre l'état caché final du dernier token de chaque vue, k = 0
  ou k tokens prédicteurs, loss dropout, masque bloc-causal pour deux vues empaquetées). Aucun changement de `client.py`, du
  format des shards ni de l'inférence : à brancher seulement si P1 réussit.
- `tests/test_crowd_jepa.py` (CPU, sans réseau).
- `docs/LLM_JEPA_PROPOSAL.md` : la proposition, et les mesures E0 qui concernent CrowdGPT (dont l'engram dans le loss check).

## 9. Résultats du pilote P1 (T4, 13–14/11/2026)

Protocole et critères écrits **avant** tout résultat : `scripts/crowdgpt_t4/P1_GOAL.md`. Les changements faits avant le
premier résultat, après la revue indépendante (workflow de 3 relecteurs), y sont listés. Code : `scripts/crowdgpt_t4/`
(`night_jepa.py`, `run_queue.py`, `criteria.py`, `t4guard.py`). Données brutes :
`artifacts/experiments/CROWDGPT_P1_llm_jepa_nlrx_2026-11-14.json`.

**Réglages.**
- Modèle : Crowd-v1 `5ac3916` + engram, gelé, fp16.
- LoRA r64 sur toutes les projections des blocs, lr 2e-4, batch effectif 16, 3 époques sur 8 000 exemples.
- Bras JEPA : λ = 1, k = 0. Vue 1 = la requête seule (comme le code de référence), vue 2 = la regex, empaquetées avec un
  masque bloc-causal ; l'entrée bigramme de l'engram est remise à zéro au début de la vue 2.
- Évaluation : correspondance exacte sur les 1 000 premiers exemples de test. Loss générale sur les 12 fenêtres « loin »
  d'E0 ; modèle de base : 3,194.

| graine | SFT % | JEPA % | Δ apparié | loss générale SFT | loss générale JEPA |
|---|---|---|---|---|---|
| 0 | 16,0 | 15,7 | −0,3 | 3,789 | 4,605 |
| 1 | 17,7 | 17,7 | 0,0 | 3,719 | 4,652 |
| 2 | 17,5 | 17,3 | −0,2 | 3,830 | 4,770 |
| **moyenne** | **17,07 ± 0,93** | **16,90 ± 1,06** | **−0,17** | **3,779** | **4,676** |

**Critères.**
1. Plancher (≥ 5 %) : passé, avec 16,0 %.
2. Succès : **non atteint**. Δ = −0,17, alors qu'il faut > 2 × 0,996. La condition sur la loss générale échoue aussi
   (+0,90 nat, limite +0,05).
3. Gain « comme le papier » (≥ +10 points ; le papier rapporte +14,2) : **non atteint**.
4. Les 3 graines par bras sont complètes : le verdict vaut pour ce montage.

**Exploratoire.**
- **λ = 4 (graine 0)** : 15,4 %, loss générale 5,012. Pire sur les deux plans.
- **Correspondance de préfixe** (métrique secondaire, graines 1–2) : SFT 20,6 / 19,4, JEPA 19,5 / 19,8. Pas d'écart.
- **Coût** : JEPA prend environ 1,8 × le temps de SFT, cohérent avec le ~2 × du papier.

**Lecture.** Le terme JEPA passe de 0,68 à environ 0,0005 dans chaque run JEPA : les états cachés finaux de la requête et
de la réponse deviennent presque identiques, pendant que la loss générale se dégrade de 0,9 nat par rapport au SFT seul.
Ça ressemble à un raccourci : la LoRA aligne les derniers états sur une direction commune au lieu d'apprendre la
correspondance requête → regex. **Non vérifié.** Le test qui trancherait : le cosinus entre les embeddings d'exemples
*différents*.

Pistes, si on poursuit (pas faites) :
- le rang LoRA 512, celui où le papier rapporte son gain LoRA ;
- un λ bien plus petit, avec du loss dropout ;
- un modèle déjà ajusté aux instructions.

En l'état, **on ne propose pas de brancher `crowd_jepa.py` dans `client.py`.** Le levier le moins cher reste le mélange de
données (§3 : 1,8 nat d'écart avec Qwen3-0.6B sur wikitext).

**Exploitation du T4.**
- File de 23:02 à 02:54. Le T4 est libre dès 02:54, bien avant l'échéance de 06:30, pour l'entraînement du week-end.
- Le modèle ollama d'un autre travail se chargeait et se déchargeait. Deux arrêts de la garde VRAM pendant sft_s0, repris
  depuis un checkpoint. La VRAM libre n'est jamais restée sous 4 300 Mio.

## 10. autoresearch : 10 hypothèses, 5 testées, 3 retenues (14/11/2026, T4)

Méthode adaptée de `karpathy/autoresearch` (mars 2026) :
- une évaluation fixe et une seule métrique ;
- un budget fixe de **600 s d'entraînement** par expérience (5 min sur H100 ≈ 10 min sur T4) ;
- un commit git par expérience, gardée ou annulée (`git reset`) ;
- `results.tsv` non versionné.

Protocole et 10 hypothèses écrits avant tout run : `scripts/crowdgpt_t4/autoresearch/program.md`. Résultats :
`scripts/crowdgpt_t4/autoresearch/RESULTS.md` et `artifacts/experiments/CROWDGPT_autoresearch_jepa_2026-11-14.json`.

**Réglages.**
- Métrique `val_ce` : CE sur les tokens de réponse de 500 exemples de dev, jamais vus à l'entraînement ; le test n'est pas
  touché.
- Garde-fou : loss générale ≤ SFT + 0,05.
- Seuil pour garder une hypothèse : battre le SFT.

| run | val_ce | Δ vs SFT | loss générale | garde-fou | terme JEPA final |
|---|---|---|---|---|---|
| SFT | **2,503** | — | 3,396 | — | — |
| JEPA λ=1 | 2,735 | +0,231 | 4,159 | ✗ | 0,0025 (effondré) |
| H1 λ=0,1 + LD 0,5 | 2,513 | +0,009 | 3,474 | ✗ (de 0,028) | 0,023 |
| H2 stop-gradient | 2,803 | +0,300 | 3,399 | ✓ | 0,081 |
| H3 cosinus centré | 2,962 | +0,459 | 3,323 | ✓ | 0,007 |
| H4 InfoNCE | 3,274 | +0,770 | 3,253 | ✓ | 0,108 |
| H6 couche 16 | 2,608 | +0,105 | 4,352 | ✗ | 0,0017 (effondré) |
| *H1+H2 (a posteriori)* | 2,540 | +0,037 | 3,430 | ✓ | 0,179 |
| *H1+H3 (a posteriori)* | 2,505 | **+0,001** | 3,452 | ✗ (de 0,006) | 0,197 |

**Les 3 retenues (règle écrite d'avance).** Classées par `val_ce` parmi celles qui passent le garde-fou : **H2
stop-gradient, H3 cosinus centré, H4 InfoNCE.**

**Ce que ça montre.**
- Deux leviers se combinent :
  - une pression faible (λ 0,1 + loss dropout) garde l'apprentissage de la tâche au niveau du SFT ;
  - un anti-effondrement (stop-gradient, centrage, InfoNCE) supprime la dégradation de la loss générale.
- H1+H2 est la meilleure variante qui passe le garde-fou (2,540).
- **Aucun gain sur le SFT à calcul égal** : le « +14 points » du papier n'apparaît pas ici.

**Limites.**
- 1 graine, 600 s, LoRA r64.
- Le nombre de pas varie de 305 à 572, selon le surcoût JEPA et la charge du T4 partagé.
- Un écart inférieur à environ 0,02 nat est à traiter comme une égalité.
- C'est un criblage, pas une confirmation.

**Prochaine étape proposée (non lancée, le T4 est réservé à l'entraînement du week-end).**
- Confirmer H1+H2 et H1+H3 au protocole complet de P1 (3 graines, 3 époques, précision sur le test).
- Tester **H8 (LoRA r512)** sur un GPU de 24 Go : c'est le réglage où le papier rapporte son gain LoRA, et celui qui a le
  plus de chances de le reproduire.
