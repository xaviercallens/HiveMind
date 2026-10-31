# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The public multilingual "join HiveMind" page, in French (default), English, Spanish, German and Chinese (``/join``, ``/rejoindre``).

Standard library only and no imports from the rest of ``hivemind``: the gateway serves it, and the La Ruche front door (``deploy/ruche_router_overlay``) copies
this one file next to its ``server.py``.

Rules this page follows, on purpose:
* Every sentence is written with matching inline markup in all five supported languages (``fr``, ``en``, ``es``, ``de``, ``zh``), so every language has the exact same sections, steps, links and commands.
* French (``fr``) is the default language when no ``?lang=`` or matching ``Accept-Language`` header is provided.
* All four participation paths (Using models, Sharing GPU via Ollama, Training at night with BOINC, and Gamers/Steam with automatic idle-GPU pause) are live and documented with working commands.
* HiveMind runs on HiveMind Reflexion Points (HMT) only: no euro or dollar amount appears (see ``points_wording.py``).
* No script, no external resource: the content security policy allows only the inline style carrying the nonce.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REPO_URL = "https://github.com/xaviercallens/HiveMind"
DEFAULT_API_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
COORDINATOR_URL = "https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"
LANGS: tuple[str, ...] = ("fr", "en", "es", "de", "zh")
LANG_NAMES: Mapping[str, str] = {
    "fr": "Français",
    "en": "English",
    "es": "Español",
    "de": "Deutsch",
    "zh": "中文",
}
Pair = tuple[str, str]  # (English, French) base pair; Español, Deutsch and 中文 are looked up in TRANSLATIONS

_URL_OK = re.compile(r"^https://[a-z0-9.-]+(?::\d+)?(?:/[A-Za-z0-9._~%/-]*)?$")


def pick_language(query_lang: str | None, accept_language: str | None, default: str = "fr") -> str:
    """``?lang=fr|en|es|de|zh`` wins; otherwise the first supported language in ``Accept-Language`` by quality; otherwise ``default`` (``fr``)."""
    if query_lang in LANGS:
        return str(query_lang)
    best: tuple[float, int, str] | None = None
    for position, part in enumerate((accept_language or "").split(",")):
        tag, _, params = part.strip().partition(";")
        primary = tag.strip().lower().split("-")[0]
        if primary not in LANGS:
            continue
        quality = 1.0
        if "q" in params.lower():  # a quality value that is present but malformed makes the entry unusable rather than top priority
            match = re.fullmatch(r"\s*q\s*=\s*([01](?:\.[0-9]{0,3})?)\s*", params, re.I)
            quality = float(match.group(1)) if match else 0.0
        if quality > 0 and (best is None or (quality, -position) > (best[0], -best[1])):
            best = (quality, position, primary)
    return best[2] if best else default


def join_content_security_policy(nonce: str) -> str:
    return "; ".join([
        "default-src 'none'", f"style-src 'nonce-{nonce}'", "img-src 'self' data:", "base-uri 'none'", "form-action 'none'", "frame-ancestors 'none'",
    ])


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# content: (English, French) + TRANSLATIONS for ("es", "de", "zh"). Inline markup: `code` and [text](url). {API} and {REPO} are replaced before rendering.
# ---------------------------------------------------------------------------------------------------------------------------------------------------

TITLE: Pair = ("Join HiveMind La Ruche", "Rejoindre HiveMind La Ruche")
DESCRIPTION: Pair = (
    "Use community AI models, or share your graphics card with HiveMind La Ruche: what works today, step by step, in French, English, Spanish, German and Chinese.",
    "Utilisez des modèles d'IA communautaires ou partagez votre carte graphique avec HiveMind La Ruche : ce qui fonctionne aujourd'hui, étape par étape, en français, anglais, espagnol, allemand et chinois.",
)
LEAD: Pair = (
    "HiveMind is a community network. People share the power of their graphics cards (GPUs), and anyone can use the resulting AI models through one "
    "OpenAI-compatible API. Contributors earn HiveMind Reflexion Points (HMT). The points have no cash value.",
    "HiveMind est un réseau communautaire. Des gens partagent la puissance de leur carte graphique (GPU), et tout le monde peut utiliser les modèles d'IA "
    "obtenus grâce à une seule API compatible avec celle d'OpenAI. Les contributeurs gagnent des Points de Réflexion HiveMind (HMT). Ces points n'ont "
    "aucune valeur monétaire.",
)

# (anchor, title, status class, status text, summary)
PATHS: Sequence[tuple[str, Pair, str, Pair, Pair]] = (
    ("use", ("Use the models", "Utiliser les modèles"), "ok", ("Available now", "Disponible maintenant"),
     ("Create an account, get a key, plug it into your editor or your code.",
      "Créez un compte, obtenez une clé et branchez-la à votre éditeur ou à votre code.")),
    ("share", ("Share your GPU to answer questions", "Partager votre GPU pour répondre aux questions"), "ok",
     ("Available now · you need to be comfortable with a terminal", "Disponible maintenant · il faut être à l'aise avec un terminal"),
     ("Serve a model from your own machine and earn points when people use it.",
      "Servez un modèle depuis votre propre ordinateur et gagnez des points quand on l'utilise.")),
    ("train", ("Train models at night (BOINC)", "Entraîner des modèles la nuit (BOINC)"), "ok",
     ("Available now · night MoE & astrophysics workunits", "Disponible maintenant · unités MoE nocturnes et astrophysique"),
     ("Lend your GPU to night QLoRA training and BOINC astrophysics workunits on Linux, Windows or macOS.",
      "Prêtez votre GPU aux tâches d'entraînement QLoRA nocturnes et aux unités BOINC d'astrophysique sous Linux, Windows ou macOS.")),
    ("play", ("Gamers and Steam", "Joueurs et Steam"), "ok",
     ("Available now · Free-to-Play Steam & automatic idle-GPU pause", "Disponible maintenant · jeu Steam gratuit et pause GPU automatique"),
     ("Give GPU time only while you are not playing. The automatic idle-GPU pause and the Free-to-Play Steam game (AppID 3133700) are available today.",
      "Donnez du temps de GPU seulement quand vous ne jouez pas. La pause GPU automatique et le jeu Steam gratuit (AppID 3133700) sont disponibles dès aujourd'hui.")),
)

@dataclass(frozen=True)
class Flags:
    """What this gateway offers right now, so the page never describes a sign-up route that is not there."""

    google: bool = False  # "Sign in with Google" (or another provider) is enabled
    handles: bool = True  # registering with a handle through the API is open


_HANDLE_RULE: Pair = (
    "The handle is public (it can appear in lists and on the message bus), so pick a pseudonym: 1 to 64 ASCII letters, digits and . _ @ + -",
    "Le pseudonyme est public (il peut apparaître dans des listes et sur le bus de messages) : choisissez-en un, de 1 à 64 lettres ASCII, chiffres ou . _ @ + -",
)
_CLOSED: Pair = (
    "Registration is closed on this gateway for now. Ask for an account in a [GitHub issue]({REPO}/issues).",
    "L'inscription est fermée sur cette passerelle pour l'instant. Demandez un compte dans un [billet GitHub]({REPO}/issues).",
)
_GOOGLE: Pair = (
    "You can also sign in with Google on the [sign-up page](/signup); we then keep only an anonymous hash of your Google id, not your e-mail address.",
    "Vous pouvez aussi vous connecter avec Google sur la [page d'inscription](/signup); nous ne gardons alors qu'une empreinte anonyme de votre identifiant "
    "Google, pas votre courriel.",
)


def _account_items(flags: Flags, role: str, intro: Pair) -> list[Pair]:
    if not flags.handles and not flags.google:
        return [_CLOSED]
    items: list[Pair] = []
    if flags.handles:
        items.append((f"{intro[0]} {_HANDLE_RULE[0]}.", f"{intro[1]} {_HANDLE_RULE[1]}."))
    if flags.google:
        items.append(_GOOGLE)
    return items


def _use_account(flags: Flags) -> list[Pair]:
    return _account_items(flags, "consumer", ("Create your account with a handle of your choice.", "Créez votre compte avec un pseudonyme de votre choix."))


def _use_register_cmd(flags: Flags) -> str:
    return 'curl -s {API}/v1/hivemind/register -H "Content-Type: application/json" -d \'{"handle":"your-handle","role":"consumer"}\'' if flags.handles else ""


def _producer_account(flags: Flags) -> list[Pair]:
    if not flags.handles and not flags.google:
        return [_CLOSED]
    items: list[Pair] = []
    if flags.handles:
        items.append(("Create a producer account the same way, with `\"role\":\"producer\"` (or `\"prosumer\"` if you also want to use models), and keep the key.",
                      "Créez un compte producteur de la même façon, avec `\"role\":\"producer\"` (ou `\"prosumer\"` si vous voulez aussi utiliser les modèles), "
                      "et gardez la clé."))
    if flags.google:
        items.append(("On the [sign-up page](/signup) with Google, choose “Share my GPU” or “Both”.",
                      "Sur la [page d'inscription](/signup) avec Google, choisissez « Share my GPU » ou « Both »."))
    return items


# Blocks per section: ("p", pair) | ("ol", [pair, ...] or flags -> [pair, ...]) | ("ul", [pair, ...]) | ("code", text or flags -> text) | ("note", pair) | ("tiers", None)
SECTIONS: Sequence[tuple[str, Pair, Sequence[tuple[str, Any]]]] = (
    ("use", ("1. Use the models", "1. Utiliser les modèles"), (
        ("ol", _use_account),
        ("code", _use_register_cmd),
        ("ol", [
            ("The reply holds your API key (`api_key`). It is shown once and cannot be recovered: store it somewhere safe. If you lose it, register a new handle.",
             "La réponse contient votre clé d'API (`api_key`). Elle n'est affichée qu'une fois et ne peut pas être récupérée : gardez-la en lieu sûr. "
             "Si vous la perdez, inscrivez un nouveau pseudonyme."),
            ("A new account holds 0 points unless the operator grants some. Points come from sharing a GPU (path 2) or from a grant: if you only want to try, "
             "ask for one in a [GitHub issue]({REPO}/issues).",
             "Un nouveau compte a 0 point, sauf don de l'opérateur. Les points viennent du partage d'un GPU (voie 2) ou d'un don : si vous voulez seulement "
             "essayer, demandez-en un dans un [billet GitHub]({REPO}/issues)."),
            ("Test your key:", "Testez votre clé :"),
        ]),
        ("code", "export HIVEMIND_API_KEY=hm-YOUR_KEY\n"
                 "curl {API}/v1/chat/completions \\\n"
                 "  -H \"Authorization: Bearer $HIVEMIND_API_KEY\" -H \"Content-Type: application/json\" \\\n"
                 "  -d '{\"model\":\"hivemind/auto\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello!\"}]}'"),
        ("note", ("In an editor (VS Code, Continue, …) follow the [connection guide]({REPO}/blob/main/docs/HIVEMIND_CONNECT_IDE.md).",
                  "Dans un éditeur (VS Code, Continue, …), suivez le [guide de connexion]({REPO}/blob/main/docs/HIVEMIND_CONNECT_IDE.md).")),
    )),
    ("share", ("2. Share your GPU to answer questions", "2. Partager votre GPU pour répondre aux questions"), (
        ("p", ("Your machine runs a model, the gateway sends it questions, and you earn points for the tokens you serve. You will use a terminal.",
               "Votre ordinateur fait tourner un modèle, la passerelle lui envoie des questions, et vous gagnez des points pour les jetons (tokens) servis. "
               "Vous utiliserez un terminal.")),
        ("ol", [
            ("Get the code (Python 3.11 or 3.12):", "Obtenez le code (Python 3.11 ou 3.12) :"),
        ]),
        ("code", "git clone {REPO}.git && cd HiveMind\npip install httpx fastapi uvicorn"),
        ("ol", [
            ("Find the model that fits your card (the table below lists all of them). This command reads your card’s memory and needs no account:",
             "Trouvez le modèle qui convient à votre carte (le tableau plus bas les liste tous). Cette commande lit la mémoire de votre carte et ne "
             "demande aucun compte :"),
        ]),
        ("code", "python scripts/hivemind_producer_agent.py --gateway {API} --suggest --detect-vram"),
        ("ol", [
            ("Install [Ollama](https://ollama.com), then download the model it suggested, for example:",
             "Installez [Ollama](https://ollama.com), puis téléchargez le modèle suggéré, par exemple :"),
        ]),
        ("code", "ollama pull qwen2.5-coder:3b"),
        ("ol", _producer_account),
        ("ol", [
            ("Your machine must be reachable over https at a public address: the gateway refuses plain http and private addresses. At home that means a "
             "tunnel or a reverse proxy in front of the producer shield, a small program that exposes only the two routes the gateway needs and demands "
             "its token. Check with `curl https://YOUR_ENDPOINT/v1/models`.",
             "Votre ordinateur doit être joignable en https à une adresse publique : la passerelle refuse le http simple et les adresses privées. À la "
             "maison, cela veut dire un tunnel ou un mandataire inverse devant le « shield » du producteur, un petit programme qui n'expose que les deux "
             "routes dont la passerelle a besoin et exige son jeton. Vérifiez avec `curl https://VOTRE_POINT/v1/models`."),
            ("Start the shield and the agent (all options: the [producer guide]({REPO}/blob/main/docs/HIVEMIND_OPENROUTER.md)). Use `--once` first to see "
             "the gateway’s answer:",
             "Démarrez le shield et l'agent (toutes les options : le [guide du producteur]({REPO}/blob/main/docs/HIVEMIND_OPENROUTER.md)). Utilisez "
             "d'abord `--once` pour voir la réponse de la passerelle :"),
        ]),
        ("code", "export HIVEMIND_API_KEY=hm-YOUR_PRODUCER_KEY\n"
                 "export HIVEMIND_ENDPOINT_TOKEN=\"$(python -c 'import secrets;print(secrets.token_urlsafe(32))')\"\n"
                 "python scripts/hivemind_producer_shield.py --upstream http://127.0.0.1:11434 --port 8787 --allow-model qwen2.5-coder:3b &\n"
                 "python scripts/hivemind_producer_agent.py --gateway {API} --endpoint https://YOUR_ENDPOINT --preset coder-lite --once"),
        ("note", ("On Windows (PowerShell) set variables with `$env:NAME=\"value\"` instead of `export`. Remove `--once` to keep the agent running.",
                  "Sous Windows (PowerShell), définissez les variables avec `$env:NOM=\"valeur\"` au lieu de `export`. Retirez `--once` pour laisser l'agent "
                  "en marche.")),
        ("p", ("You can stop at any time: after about two minutes without a heartbeat the gateway stops sending you requests.",
               "Vous pouvez arrêter en tout temps : après environ deux minutes sans signal de vie, la passerelle cesse de vous envoyer des requêtes.")),
        ("tiers", None),
    )),
    ("train", ("3. Train models at night with BOINC", "3. Entraîner des modèles la nuit avec BOINC"), (
        ("p", ("This path is open to the public: the federated BOINC & Night-MoE coordinator is live at [HiveMind La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app) "
               "(live Night-MoE telemetry at [/night-moe](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe), BOINC coordinator status at "
               "[/boinc/status](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc/status) and citizen-science leaderboard at "
               "[/community](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)). You can check your hardware, run a local dry-run (`--dry-run`), and lease workunits today.",
               "Cette voie est ouverte au public : le coordinateur fédéré BOINC et Night-MoE est en ligne sur [HiveMind La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app) "
               "(télémétrie Night-MoE en direct sur [/night-moe](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe), état du coordinateur BOINC sur "
               "[/boinc/status](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc/status) et classement de science participative sur "
               "[/community](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)). Vous pouvez vérifier votre matériel, lancer un test à blanc (`--dry-run`) et calculer des unités dès aujourd'hui.")),
        ("ul", [
            ("Linux on an x86-64 processor for the native BOINC portable bundle, or Linux, Windows and macOS with the Python BOINC/BLS worker",
             "Linux sur processeur x86-64 pour l'archive BOINC native, ou Linux, Windows et macOS avec l'agent Python BOINC/BLS"),
            ("An NVIDIA card with at least 3.5 GB of video memory (`cuda_hivemind`; 12 GB+ for `cuda_hivemind_big`), or a 4+ thread CPU with 8 GB RAM for CPU/BLS transit sweeps (`cpu_hivemind_solver`)",
             "Une carte NVIDIA avec au moins 3,5 Go de mémoire vidéo (`cuda_hivemind` ; 12 Go+ pour `cuda_hivemind_big`), ou un processeur 4+ fils avec 8 Go de RAM pour les balayages CPU/BLS (`cpu_hivemind_solver`)"),
            ("A first download of about 4.4 GB, reused afterwards", "Un premier téléchargement d'environ 4,4 Go, réutilisé ensuite"),
            ("Community work is offered at night, 22:00–07:00 Paris time, when the network is least busy",
             "Le travail communautaire est offert la nuit, de 22 h à 7 h, heure de Paris, quand le réseau est le moins occupé"),
            ("The standard BOINC client; the project’s files are digitally signed", "Le client BOINC standard; les fichiers du projet sont signés numériquement"),
        ]),
        ("ol", [
            ("Run the read-only hardware and coordinator network preflight check first (it installs nothing; add `--allow-cpu-solver` on a machine without an NVIDIA driver):",
             "Lancez d'abord la vérification matérielle et réseau en lecture seule (elle n'installe rien ; ajoutez `--allow-cpu-solver` sur une machine sans pilote NVIDIA) :"),
        ]),
        ("code", "python3 scripts/hivemind_volunteer_preflight.py --project-url https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app"),
        ("ol", [
            ("Run a complete local dry-run test (`--dry-run`) to verify 16,384-point TESS/JWST light-curve folding, rank-4 QLoRA outer-delta SVD verification, and coordinator reachability:",
             "Exécutez un test à blanc complet sur votre machine (`--dry-run`) pour vérifier le pliage de courbe de lumière TESS/JWST (16 384 points), la validation SVD QLoRA de rang 4 et la connexion au coordinateur :"),
        ]),
        ("code", "python3 scripts/hivemind_boinc_idle_worker.py --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app --handle your-handle --domain physics_sim --dry-run"),
        ("ol", [
            ("Lease, fold and submit a live federated BOINC workunit (`physics_sim`, `code_generation` or `math_reasoning`) from the public coordinator "
             "(remove `--once` to keep folding at night):",
             "Louez, calculez et soumettez une unité de travail BOINC fédérée en direct (`physics_sim`, `code_generation` ou `math_reasoning`) auprès du coordinateur public "
             "(retirez `--once` pour continuer le calcul la nuit) :"),
        ]),
        ("code", "python3 scripts/hivemind_boinc_idle_worker.py --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app --handle your-handle --domain physics_sim --once"),
        ("ol", [
            ("Or attach a standard BOINC client (`boinccmd`) directly to the public project URL and inspect verified receipts on [/api/v1/federated/ledger](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger):",
             "Ou attachez directement un client BOINC standard (`boinccmd`) à l'URL publique du projet et consultez les reçus vérifiés sur [/api/v1/federated/ledger](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger) :"),
        ]),
        ("code", "boinccmd --get_project_config https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app\n"
                 "boinccmd --project_attach https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app YOUR_ACCOUNT_KEY"),
        ("note", ("Night training for end-users and HiveMind nodes (22:00–07:00 Paris time): End-users (volunteers, developers and gamers) leave their PC or laptop running overnight to fold exoplanet transits and compute rank-4 QLoRA inner-loop updates, earning 250 HMT and 45 Cobblestones per workunit to spend the next day on all models. Dedicated HiveMind nodes (GPU producers, HiveMiners and the owner fleet) switch from daytime OpenRouter inference (07:00–22:00) to nocturnal DiLoCo outer-step aggregation, Multi-Krum verification and SVD teacher-to-student distillation (live node telemetry at [/v1/hivemind/bees/owner-fleet-status](/v1/hivemind/bees/owner-fleet-status)).",
                  "Entraînement nocturne pour les utilisateurs finaux et les nœuds HiveMind (22 h – 7 h, heure de Paris) : Les utilisateurs finaux (bénévoles, développeurs et joueurs) laissent leur PC ou portable calculer la nuit des transits d'exoplanètes et des mises à jour QLoRA de rang 4, gagnant 250 HMT et 45 Cobblestones par unité pour interroger tous les modèles le lendemain. Les nœuds dédiés HiveMind (producteurs GPU, HiveMiners et flotte principale) passent de l'inférence OpenRouter diurne (7 h – 22 h) à l'agrégation nocturne DiLoCo, la vérification Multi-Krum et la distillation SVD enseignant-étudiant (télémétrie des nœuds en direct sur [/v1/hivemind/bees/owner-fleet-status](/v1/hivemind/bees/owner-fleet-status)).")),
    )),
    ("play", ("4. Gamers and Steam", "4. Joueurs et Steam"), (
        ("p", ("Your GPU folds astrophysics and AI workunits only while you are not playing. Both the Free-to-Play Steam Edition "
               "(`USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter`, Steam `AppID 3133700`) and the automatic idle-GPU pause worker are available today, "
               "connecting gamers and citizen scientists on the [live community hub](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community).",
               "Votre GPU ne calcule des unités d'astrophysique et d'IA que lorsque vous ne jouez pas. L'édition Steam gratuite "
               "(`USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter`, Steam `AppID 3133700`) et l'agent de pause GPU automatique sont disponibles dès aujourd'hui, "
               "réunissant joueurs et scientifiques citoyens sur le [portail communautaire en direct](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community).")),
        ("p", ("While you play (`idle < 4 s` of keyboard/mouse input or GPU load above 45%), BOINC GPU compute pauses automatically (`boinccmd --set_gpu_mode never`, "
               "0% GPU overhead) so your game keeps 100% of your graphics card at 60 FPS. After 4 seconds of inactivity (with GPU load below 45% and temperature below 78 °C), "
               "it automatically resumes folding TESS/JWST Box-Least-Squares (BLS) exoplanet transits.",
               "Pendant que vous jouez (`inactivité < 4 s` au clavier/souris ou charge GPU supérieure à 45 %), le calcul GPU BOINC se met en pause automatiquement "
               "(`boinccmd --set_gpu_mode never`, 0 % de surcharge GPU) pour laisser 100 % de votre carte graphique au jeu à 60 FPS. Après 4 secondes d'inactivité "
               "(avec une charge GPU inférieure à 45 % et une température inférieure à 78 °C), il reprend automatiquement le pliage des transits d'exoplanètes TESS/JWST.")),
        ("ol", [
            ("Start the automatic idle-GPU worker (remove `--once` to leave the automatic pause/resume daemon running in the background while you play):",
             "Démarrez l'agent de pause GPU automatique (retirez `--once` pour laisser le démon de pause/reprise automatique tourner en arrière-plan pendant vos sessions de jeu) :"),
        ]),
        ("code", "python3 scripts/hivemind_boinc_idle_worker.py --coordinator https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app --handle your-handle --idle-threshold-seconds 4 --once"),
        ("ol", [
            ("You can also trigger an immediate manual GPU pause or resume from a Steam launch option:",
             "Vous pouvez aussi déclencher une pause ou une reprise GPU immédiate depuis une option de lancement Steam :"),
        ]),
        ("code", "python3 scripts/hivemind_boinc_idle_worker.py --pause-now"),
        ("ol", [
            ("If you also run an Ollama producer from path 2, a loaded language model keeps its video memory even when nobody is asking; before launching a heavy game, "
             "stop the producer agent (Ctrl+C) and unload the model (replace the name with your model’s):",
             "Si vous faites aussi tourner un producteur Ollama de la voie 2, un modèle de langue chargé garde sa mémoire vidéo même quand personne ne pose de question ; "
             "avant de lancer un jeu gourmand, arrêtez l'agent producteur (Ctrl+C) et libérez la mémoire (remplacez le nom par celui de votre modèle) :"),
        ]),
        ("code", "ollama stop qwen2.5-coder:3b"),
        ("ol", [
            ("After you play: start the producer agent again, with the same command as in path 2 (the BOINC idle-GPU worker resumes by itself).",
             "Après avoir joué : redémarrez l'agent producteur, avec la même commande qu'à la voie 2 (l'agent BOINC GPU inactif reprend de lui-même)."),
        ]),
    )),
    ("know", ("Good to know", "Bon à savoir"), (
        ("ul", [
            ("Producers see the questions sent to them. Do not run a producer on a machine that holds secrets, and do not send confidential text to the network.",
             "Les producteurs voient les questions qui leur sont envoyées. N'exploitez pas un producteur sur une machine qui contient des secrets, et "
             "n'envoyez pas de texte confidentiel au réseau."),
            ("Points have no cash value and are not a payment.", "Les points n'ont aucune valeur monétaire et ne sont pas un paiement."),
            ("Sharing uses electricity while your GPU answers. Measure your card with `nvidia-smi --query-gpu=power.draw --format=csv` and multiply by "
             "the hours it works.",
             "Le partage consomme de l'électricité pendant que votre GPU répond. Mesurez votre carte avec `nvidia-smi --query-gpu=power.draw --format=csv` "
             "et multipliez par les heures de travail."),
            ("The software is open source (AGPL-3.0). Read the code before you run it.", "Le logiciel est libre (AGPL-3.0). Lisez le code avant de l'exécuter."),
            ("The network is young and run by volunteers; availability is not guaranteed.",
             "Le réseau est jeune et exploité par des bénévoles : la disponibilité n'est pas garantie."),
        ]),
    )),
    ("help", ("Questions and help", "Questions et aide"), (
        ("p", ("Open an issue on [GitHub]({REPO}/issues), in English or French. For a security problem, read the [security policy]({REPO}/security/policy) first.",
               "Ouvrez un billet sur [GitHub]({REPO}/issues), en français ou en anglais. Pour un problème de sécurité, lisez d'abord la "
               "[politique de sécurité]({REPO}/security/policy).")),
    )),
)

LABELS: Mapping[str, Pair] = {
    "skip": ("Skip to content", "Aller au contenu"),
    "choose": ("Choose your way in", "Choisissez votre façon de participer"),
    "other_lang": ("Français", "English"),
    "tiers_summary": ("Which model for which card? (all models)", "Quel modèle pour quelle carte? (tous les modèles)"),
    "col_model": ("Model", "Modèle"), "col_id": ("Name for --preset", "Nom pour --preset"), "col_mem": ("Minimum memory (GB)", "Mémoire minimale (Go)"),
    "col_dl": ("Download (GB)", "Téléchargement (Go)"),
    "tiers_missing": ("The live list of models is at [/v1/hivemind/catalog](/v1/hivemind/catalog).",
                      "La liste à jour des modèles est à [/v1/hivemind/catalog](/v1/hivemind/catalog)."),
    "tiers_note": ("Names come from the project’s catalog. The minimum memory is the smallest card we recommend, including the context cache.",
                   "Les noms viennent du catalogue du projet. La mémoire minimale est celle de la plus petite carte que nous recommandons, cache de contexte compris."),
    "footer": ("HiveMind La Ruche · open source (AGPL-3.0) · points have no cash value",
               "HiveMind La Ruche · logiciel libre (AGPL-3.0) · les points n'ont aucune valeur monétaire"),
    "status_prefix": ("Status: ", "État : "),  # read by screen readers before the badge; French puts a space before the colon
}

# Complete translations for Español (es), Deutsch (de) and 中文 (zh), keyed by the English text.
TRANSLATIONS: Mapping[str, Mapping[str, str]] = {
    TITLE[0]: {
        "es": "Unirse a HiveMind La Ruche",
        "de": "HiveMind La Ruche beitreten",
        "zh": "加入 HiveMind La Ruche",
    },
    DESCRIPTION[0]: {
        "es": "Usa modelos de IA comunitarios o comparte tu tarjeta gráfica con HiveMind La Ruche: lo que funciona hoy, paso a paso, en francés, inglés, español, alemán y chino.",
        "de": "Nutzen Sie Community-KI-Modelle oder teilen Sie Ihre Grafikkarte mit HiveMind La Ruche: Was heute funktioniert, Schritt für Schritt, auf Französisch, Englisch, Spanisch, Deutsch und Chinesisch.",
        "zh": "使用社区 AI 模型，或与 HiveMind La Ruche 共享您的显卡（GPU）：今日可用功能分步指南，支持法语、英语、西班牙语、德语和中文。",
    },
    LEAD[0]: {
        "es": "HiveMind es una red comunitaria. Los participantes comparten la potencia de sus tarjetas gráficas (GPU) y cualquiera puede usar los modelos de IA resultantes mediante una única API compatible con OpenAI. Los colaboradores ganan Puntos de Reflexión HiveMind (HMT). Estos puntos no tienen valor monetario.",
        "de": "HiveMind ist ein Community-Netzwerk. Teilnehmende teilen die Rechenleistung ihrer Grafikkarten (GPUs), und alle können die entstehenden KI-Modelle über eine einzige OpenAI-kompatible API nutzen. Mitwirkende erhalten HiveMind-Reflexionspunkte (HMT). Diese Punkte haben keinen Geldwert.",
        "zh": "HiveMind 是一个社区计算网络。参与者共享显卡（GPU）算力，任何人都可以通过统一的 OpenAI 兼容 API 使用社区 AI 模型。贡献者将获得 HiveMind 反思积分 (HMT)。这些积分没有任何货币价值。",
    },
    "Use the models": {"es": "Usar los modelos", "de": "Modelle nutzen", "zh": "使用 AI 模型"},
    "Available now": {"es": "Disponible ahora", "de": "Jetzt verfügbar", "zh": "现已开放"},
    "Create an account, get a key, plug it into your editor or your code.": {
        "es": "Crea una cuenta, obtén una clave y conéctala a tu editor o a tu código.",
        "de": "Erstellen Sie ein Konto, holen Sie sich einen Schlüssel und verbinden Sie ihn mit Ihrem Editor oder Code.",
        "zh": "创建账户，获取 API 密钥，并将其接入您的代码编辑器或应用程序。",
    },
    "Share your GPU to answer questions": {
        "es": "Compartir tu GPU para responder preguntas",
        "de": "GPU teilen, um Fragen zu beantworten",
        "zh": "共享您的 GPU 回答问题",
    },
    "Available now · you need to be comfortable with a terminal": {
        "es": "Disponible ahora · requiere soltura con la terminal",
        "de": "Jetzt verfügbar · Grundkenntnisse im Terminal erforderlich",
        "zh": "现已开放 · 需要熟悉命令行终端",
    },
    "Serve a model from your own machine and earn points when people use it.": {
        "es": "Sirve un modelo desde tu propio equipo y gana puntos cuando la comunidad lo utilice.",
        "de": "Stellen Sie ein Modell auf Ihrem eigenen Rechner bereit und sammeln Sie Punkte, wenn es genutzt wird.",
        "zh": "在您自己的电脑上运行模型，当他人调用时即可获得积分。",
    },
    "Train models at night (BOINC)": {
        "es": "Entrenar modelos por la noche (BOINC)",
        "de": "Modelle nachts trainieren (BOINC)",
        "zh": "夜间训练模型 (BOINC)",
    },
    "Available now · night MoE & astrophysics workunits": {
        "es": "Disponible ahora · unidades MoE nocturnas y astrofísica",
        "de": "Jetzt verfügbar · nächtliche MoE- & Astrophysik-Aufgaben",
        "zh": "现已开放 · 夜间 MoE 与天体物理计算任务",
    },
    "Lend your GPU to night QLoRA training and BOINC astrophysics workunits on Linux, Windows or macOS.": {
        "es": "Presta tu GPU para el entrenamiento nocturno QLoRA y unidades de astrofísica BOINC en Linux, Windows o macOS.",
        "de": "Stellen Sie Ihre GPU nachts für QLoRA-Training und BOINC-Astrophysik-Aufgaben unter Linux, Windows oder macOS bereit.",
        "zh": "在 Linux、Windows 或 macOS 上将您的 GPU 用于夜间 QLoRA 训练与 BOINC 天体物理计算任务。",
    },
    "Gamers and Steam": {
        "es": "Jugadores y Steam",
        "de": "Gamer und Steam",
        "zh": "游戏玩家与 Steam",
    },
    "Available now · Free-to-Play Steam & automatic idle-GPU pause": {
        "es": "Disponible ahora · juego Steam gratuito y pausa automática de GPU",
        "de": "Jetzt verfügbar · kostenloses Steam-Spiel & automatische GPU-Pause",
        "zh": "现已开放 · Steam 免费游戏与 GPU 空闲自动暂停",
    },
    "Give GPU time only while you are not playing. The automatic idle-GPU pause and the Free-to-Play Steam game (AppID 3133700) are available today.": {
        "es": "Dona tiempo de GPU solo cuando no estés jugando. La pausa automática de GPU inactiva y el juego gratuito de Steam (AppID 3133700) están disponibles desde hoy.",
        "de": "Teilen Sie GPU-Zeit nur,Während Sie nicht spielen. Die automatische GPU-Pause und das kostenlose Steam-Spiel (AppID 3133700) sind ab sofort verfügbar.",
        "zh": "仅在您不玩游戏时贡献 GPU 算力。GPU 空闲自动暂停功能与 Steam 免费游戏（AppID 3133700）今日即可使用。",
    },
    f"Create your account with a handle of your choice. {_HANDLE_RULE[0]}.": {
        "es": "Crea tu cuenta con el seudónimo que prefieras. El seudónimo es público (puede aparecer en listas y en el bus de mensajes), así que elige uno de 1 a 64 letras ASCII, cifras o . _ @ + -.",
        "de": "Erstellen Sie Ihr Konto mit einem frei gewählten Pseudonym. Das Pseudonym ist öffentlich (es kann in Listen und auf dem Nachrichtenbus erscheinen); wählen Sie 1 bis 64 ASCII-Buchstaben, Ziffern oder . _ @ + -.",
        "zh": "使用您选择的昵称创建账户。该昵称是公开的（可能出现在列表和消息总线上），请选择 1 到 64 个 ASCII 字母、数字或 . _ @ + -。",
    },
    _CLOSED[0]: {
        "es": "El registro está cerrado en esta pasarela por ahora. Solicita una cuenta en una [incidencia de GitHub]({REPO}/issues).",
        "de": "Die Registrierung ist auf diesem Gateway derzeit geschlossen. Fragen Sie in einem [GitHub-Issue]({REPO}/issues) nach einem Konto.",
        "zh": "此网关目前暂停开放注册。请在 [GitHub Issue]({REPO}/issues) 中申请账户。",
    },
    _GOOGLE[0]: {
        "es": "También puedes iniciar sesión con Google en la [página de registro](/signup); solo guardamos un hash anónimo de tu identificador de Google, nunca tu correo electrónico.",
        "de": "Sie können sich auch auf der [Registrierungsseite](/signup) mit Google anmelden; wir speichern dann nur einen anonymen Hash Ihrer Google-ID, nicht Ihre E-Mail-Adresse.",
        "zh": "您也可以在[注册页面](/signup)使用 Google 登录；我们仅保存您 Google ID 的匿名哈希值，绝不存储您的电子邮箱。",
    },
    "Create a producer account the same way, with `\"role\":\"producer\"` (or `\"prosumer\"` if you also want to use models), and keep the key.": {
        "es": "Crea una cuenta de productor del mismo modo, con `\"role\":\"producer\"` (o `\"prosumer\"` si también quieres usar modelos), y guarda la clave.",
        "de": "Erstellen Sie auf dieselbe Weise ein Produzentenkonto mit `\"role\":\"producer\"` (oder `\"prosumer\"`, wenn Sie auch Modelle nutzen möchten), und bewahren Sie den Schlüssel auf.",
        "zh": "以相同方式创建生产者账户，设置 `\"role\":\"producer\"`（若同时想调用模型可设为 `\"prosumer\"`），并妥善保存密钥。",
    },
    "On the [sign-up page](/signup) with Google, choose “Share my GPU” or “Both”.": {
        "es": "En la [página de registro](/signup) con Google, elige «Share my GPU» o «Both».",
        "de": "Wählen Sie auf der [Registrierungsseite](/signup) mit Google „Share my GPU“ oder „Both“.",
        "zh": "在 Google [注册页面](/signup)上，选择“Share my GPU”或“Both”。",
    },
    "1. Use the models": {"es": "1. Usar los modelos", "de": "1. Modelle nutzen", "zh": "1. 使用 AI 模型"},
    "The reply holds your API key (`api_key`). It is shown once and cannot be recovered: store it somewhere safe. If you lose it, register a new handle.": {
        "es": "La respuesta contiene tu clave de API (`api_key`). Se muestra una sola vez y no se puede recuperar: guárdala en un lugar seguro. Si la pierdes, registra un nuevo seudónimo.",
        "de": "Die Antwort enthält Ihren API-Schlüssel (`api_key`). Er wird nur einmal angezeigt und kann nicht wiederhergestellt werden: Bewahren Sie ihn sicher auf. Bei Verlust registrieren Sie ein neues Pseudonym.",
        "zh": "响应中包含您的 API 密钥（`api_key`）。密钥仅显示一次且无法找回，请妥善保管。如果丢失，请注册一个新昵称。",
    },
    "A new account holds 0 points unless the operator grants some. Points come from sharing a GPU (path 2) or from a grant: if you only want to try, ask for one in a [GitHub issue]({REPO}/issues).": {
        "es": "Una cuenta nueva tiene 0 puntos salvo concesión del operador. Los puntos se obtienen compartiendo una GPU (vía 2) o mediante una concesión: si solo quieres probar, pídela en una [incidencia de GitHub]({REPO}/issues).",
        "de": "Ein neues Konto startet mit 0 Punkten, sofern der Betreiber keine gutschreibt. Punkte erhalten Sie durch das Teilen einer GPU (Weg 2) oder eine Gutschrift: Zum Ausprobieren können Sie in einem [GitHub-Issue]({REPO}/issues) danach fragen.",
        "zh": "新账户初始为 0 积分（除非管理员授予初始积分）。积分可通过共享 GPU（方式 2）或申请试用获得：如果您只想体验，请在 [GitHub Issue]({REPO}/issues) 中申请。",
    },
    "Test your key:": {"es": "Prueba tu clave:", "de": "Testen Sie Ihren Schlüssel:", "zh": "测试您的密钥："},
    "In an editor (VS Code, Continue, …) follow the [connection guide]({REPO}/blob/main/docs/HIVEMIND_CONNECT_IDE.md).": {
        "es": "En un editor (VS Code, Continue, …), sigue la [guía de conexión]({REPO}/blob/main/docs/HIVEMIND_CONNECT_IDE.md).",
        "de": "Folgen Sie in einem Editor (VS Code, Continue, …) der [Verbindungsanleitung]({REPO}/blob/main/docs/HIVEMIND_CONNECT_IDE.md).",
        "zh": "在代码编辑器（VS Code、Continue 等）中，请参阅[连接指南]({REPO}/blob/main/docs/HIVEMIND_CONNECT_IDE.md)。",
    },
    "2. Share your GPU to answer questions": {
        "es": "2. Compartir tu GPU para responder preguntas",
        "de": "2. GPU teilen, um Fragen zu beantworten",
        "zh": "2. 共享您的 GPU 回答问题",
    },
    "Your machine runs a model, the gateway sends it questions, and you earn points for the tokens you serve. You will use a terminal.": {
        "es": "Tu ordenador ejecuta un modelo, la pasarela le envía preguntas y ganas puntos por los tokens servidos. Usarás una terminal.",
        "de": "Ihr Rechner führt ein Modell aus, das Gateway sendet Fragen dorthin, und Sie erhalten Punkte für die gelieferten Tokens. Sie arbeiten im Terminal.",
        "zh": "您的电脑运行模型，网关向其转发提问，您将根据生成的词元（tokens）获得积分。此过程需要使用命令行终端。",
    },
    "Get the code (Python 3.11 or 3.12):": {
        "es": "Obtén el código (Python 3.11 o 3.12):",
        "de": "Code herunterladen (Python 3.11 oder 3.12):",
        "zh": "获取代码（需要 Python 3.11 或 3.12）：",
    },
    "Find the model that fits your card (the table below lists all of them). This command reads your card’s memory and needs no account:": {
        "es": "Encuentra el modelo adecuado para tu tarjeta (la tabla inferior los muestra todos). Este comando lee la memoria de tu tarjeta y no requiere cuenta:",
        "de": "Finden Sie das passende Modell für Ihre Karte (die Tabelle unten listet alle auf). Dieser Befehl liest den Grafikspeicher aus und benötigt kein Konto:",
        "zh": "查找适合您显卡的模型（下方表格列出了全部模型）。此命令仅读取显存容量，无需账户：",
    },
    "Install [Ollama](https://ollama.com), then download the model it suggested, for example:": {
        "es": "Instala [Ollama](https://ollama.com) y descarga el modelo sugerido, por ejemplo:",
        "de": "Installieren Sie [Ollama](https://ollama.com) und laden Sie das vorgeschlagene Modell herunter, zum Beispiel:",
        "zh": "安装 [Ollama](https://ollama.com)，然后下载建议的模型，例如：",
    },
    "Your machine must be reachable over https at a public address: the gateway refuses plain http and private addresses. At home that means a tunnel or a reverse proxy in front of the producer shield, a small program that exposes only the two routes the gateway needs and demands its token. Check with `curl https://YOUR_ENDPOINT/v1/models`.": {
        "es": "Tu equipo debe ser accesible por https en una dirección pública: la pasarela rechaza http simple y direcciones privadas. En casa, eso implica un túnel o proxy inverso delante del «shield» del productor, un pequeño programa que expone solo las dos rutas necesarias y exige su token. Compruébalo con `curl https://YOUR_ENDPOINT/v1/models`.",
        "de": "Ihr Rechner muss über https unter einer öffentlichen Adresse erreichbar sein: Das Gateway lehnt unverschlüsseltes http und private Adressen ab. Zu Hause bedeutet das einen Tunnel oder Reverse-Proxy vor dem Producer-Shield. Prüfen Sie dies mit `curl https://YOUR_ENDPOINT/v1/models`.",
        "zh": "您的电脑必须能通过公共 https 地址访问：网关拒绝普通 http 和私有内网地址。在家庭网络中，您可以在生产者安全盾（shield）前配置隧道或反向代理。请使用 `curl https://YOUR_ENDPOINT/v1/models` 进行验证。",
    },
    "Start the shield and the agent (all options: the [producer guide]({REPO}/blob/main/docs/HIVEMIND_OPENROUTER.md)). Use `--once` first to see the gateway’s answer:": {
        "es": "Inicia el shield y el agente (todas las opciones en la [guía del productor]({REPO}/blob/main/docs/HIVEMIND_OPENROUTER.md)). Usa `--once` primero para ver la respuesta de la pasarela:",
        "de": "Starten Sie den Shield und den Agenten (alle Optionen im [Produzenten-Leitfaden]({REPO}/blob/main/docs/HIVEMIND_OPENROUTER.md)). Nutzen Sie zuerst `--once`, um die Antwort des Gateways zu prüfen:",
        "zh": "启动安全盾与代理程序（完整参数见[生产者指南]({REPO}/blob/main/docs/HIVEMIND_OPENROUTER.md)）。首次运行可加上 `--once` 查看网关响应：",
    },
    "On Windows (PowerShell) set variables with `$env:NAME=\"value\"` instead of `export`. Remove `--once` to keep the agent running.": {
        "es": "En Windows (PowerShell), define las variables con `$env:NAME=\"value\"` en lugar de `export`. Quita `--once` para mantener el agente en ejecución.",
        "de": "Unter Windows (PowerShell) setzen Sie Variablen mit `$env:NAME=\"value\"` statt `export`. Entfernen Sie `--once`, damit der Agent dauerhaft läuft.",
        "zh": "在 Windows (PowerShell) 中，请使用 `$env:NAME=\"value\"` 代替 `export` 设置环境变量。移除 `--once` 即可保持代理持续运行。",
    },
    "You can stop at any time: after about two minutes without a heartbeat the gateway stops sending you requests.": {
        "es": "Puedes detenerlo en cualquier momento: tras unos dos minutos sin señal de vida, la pasarela deja de enviarte peticiones.",
        "de": "Sie können jederzeit stoppen: Nach etwa zwei Minuten ohne Heartbeat sendet das Gateway keine Anfragen mehr.",
        "zh": "您可以随时停止：在心跳信号中断约两分钟后，网关会自动停止向您发送请求。",
    },
    "3. Train models at night with BOINC": {
        "es": "3. Entrenar modelos por la noche con BOINC",
        "de": "3. Modelle nachts mit BOINC trainieren",
        "zh": "3. 夜间使用 BOINC 训练模型",
    },
    "This path is open to the public: the federated BOINC & Night-MoE coordinator is live at [HiveMind La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app) (live Night-MoE telemetry at [/night-moe](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe), BOINC coordinator status at [/boinc/status](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc/status) and citizen-science leaderboard at [/community](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)). You can check your hardware, run a local dry-run (`--dry-run`), and lease workunits today.": {
        "es": "Esta vía está abierta al público: el coordinador federado BOINC y Night-MoE está activo en [HiveMind La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app) (telemetría Night-MoE en vivo en [/night-moe](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe), estado del coordinador BOINC en [/boinc/status](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc/status) y clasificación de ciencia ciudadana en [/community](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)). Hoy mismo puedes verificar tu hardware, ejecutar una prueba local (`--dry-run`) y procesar unidades.",
        "de": "Dieser Weg ist öffentlich zugänglich: Der föderierte BOINC- & Night-MoE-Koordinator ist unter [HiveMind La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app) online (Live-Telemetrie unter [/night-moe](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe), BOINC-Koordinatorstatus unter [/boinc/status](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc/status) und Citizen-Science-Bestenliste unter [/community](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)). Sie können Ihre Hardware prüfen, einen lokalen Testlauf (`--dry-run`) starten und sofort Aufgaben berechnen.",
        "zh": "此通道已向公众开放：联邦 BOINC 与 Night-MoE 协调器已在 [HiveMind La Ruche](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app) 上线（实时 Night-MoE 遥测见 [/night-moe](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/night-moe)，BOINC 协调器状态见 [/boinc/status](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/boinc/status)，公民科学排行榜见 [/community](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)）。您今天即可检测硬件、运行本地演练（`--dry-run`）并领取计算任务。",
    },
    "Linux on an x86-64 processor for the native BOINC portable bundle, or Linux, Windows and macOS with the Python BOINC/BLS worker": {
        "es": "Linux en procesador x86-64 para el paquete nativo BOINC, o Linux, Windows y macOS con el agente Python BOINC/BLS",
        "de": "Linux auf einem x86-64-Prozessor für das native BOINC-Paket oder Linux, Windows und macOS mit dem Python-BOINC/BLS-Worker",
        "zh": "原生 BOINC 便携包支持 x86-64 Linux，Python BOINC/BLS 工作节点支持 Linux、Windows 和 macOS",
    },
    "An NVIDIA card with at least 3.5 GB of video memory (`cuda_hivemind`; 12 GB+ for `cuda_hivemind_big`), or a 4+ thread CPU with 8 GB RAM for CPU/BLS transit sweeps (`cpu_hivemind_solver`)": {
        "es": "Una tarjeta NVIDIA con al menos 3,5 GB de memoria de vídeo (`cuda_hivemind`; 12 GB+ para `cuda_hivemind_big`), o una CPU de 4+ hilos con 8 GB de RAM para barridos CPU/BLS (`cpu_hivemind_solver`)",
        "de": "Eine NVIDIA-Grafikkarte mit mindestens 3,5 GB Videospeicher (`cuda_hivemind`; 12 GB+ für `cuda_hivemind_big`) oder eine CPU mit 4+ Threads und 8 GB RAM für CPU/BLS-Transitsuchen (`cpu_hivemind_solver`)",
        "zh": "至少配备 3.5 GB 显存的 NVIDIA 显卡（`cuda_hivemind`；12 GB 以上支持 `cuda_hivemind_big`），或配备 4 线程以上 CPU 与 8 GB 内存以运行 CPU/BLS 凌星扫描（`cpu_hivemind_solver`）",
    },
    "A first download of about 4.4 GB, reused afterwards": {
        "es": "Una primera descarga de unos 4,4 GB, reutilizada después",
        "de": "Ein erster Download von rund 4,4 GB, der anschließend wiederverwendet wird",
        "zh": "首次下载约 4.4 GB，后续任务可重复使用",
    },
    "Community work is offered at night, 22:00–07:00 Paris time, when the network is least busy": {
        "es": "El trabajo comunitario se ofrece por la noche, de 22:00 a 07:00 (hora de París), cuando la red está menos ocupada",
        "de": "Community-Aufgaben werden nachts von 22:00 bis 07:00 Uhr Pariser Zeit vergeben, wenn das Netzwerk am wenigsten ausgelastet ist",
        "zh": "社区训练任务在巴黎时间夜间 22:00–07:00 网络空闲时段开放",
    },
    "The standard BOINC client; the project’s files are digitally signed": {
        "es": "El cliente BOINC estándar; los archivos del proyecto están firmados digitalmente",
        "de": "Der Standard-BOINC-Client; die Projektdateien sind digital signiert",
        "zh": "兼容标准 BOINC 客户端；所有项目文件均带有数字签名",
    },
    "Run the read-only hardware and coordinator network preflight check first (it installs nothing; add `--allow-cpu-solver` on a machine without an NVIDIA driver):": {
        "es": "Ejecuta primero la comprobación de hardware y red de solo lectura (no instala nada; añade `--allow-cpu-solver` en un equipo sin controlador NVIDIA):",
        "de": "Führen Sie zuerst die schreibgeschützte Hardware- und Netzwerkprüfung aus (installiert nichts; fügen Sie `--allow-cpu-solver` auf einem Rechner ohne NVIDIA-Treiber hinzu):",
        "zh": "首先运行只读硬件与协调器网络预检脚本（不安装任何软件；若未安装 NVIDIA 驱动可加上 `--allow-cpu-solver`）：",
    },
    "Run a complete local dry-run test (`--dry-run`) to verify 16,384-point TESS/JWST light-curve folding, rank-4 QLoRA outer-delta SVD verification, and coordinator reachability:": {
        "es": "Ejecuta una prueba local completa (`--dry-run`) para verificar el plegado de curvas de luz TESS/JWST de 16 384 puntos, la validación SVD QLoRA de rango 4 y la conexión con el coordinador:",
        "de": "Führen Sie einen vollständigen lokalen Testlauf (`--dry-run`) aus, um die 16.384-Punkte-TESS/JWST-Lichtkurvenfaltung, die Rang-4-QLoRA-SVD-Prüfung und die Erreichbarkeit des Koordinators zu bestätigen:",
        "zh": "运行完整的本地演练测试（`--dry-run`），验证 16,384 点 TESS/JWST 光变曲线折叠、4 秩 QLoRA 外层增量 SVD 校验以及协调器连通性：",
    },
    "Lease, fold and submit a live federated BOINC workunit (`physics_sim`, `code_generation` or `math_reasoning`) from the public coordinator (remove `--once` to keep folding at night):": {
        "es": "Solicita, calcula y envía una unidad BOINC federada en vivo (`physics_sim`, `code_generation` o `math_reasoning`) al coordinador público (quita `--once` para seguir calculando por la noche):",
        "de": "Leasen, berechnen und senden Sie eine föderierte Live-BOINC-Aufgabe (`physics_sim`, `code_generation` oder `math_reasoning`) an den öffentlichen Koordinator (entfernen Sie `--once` für den Dauerbetrieb in der Nacht):",
        "zh": "从公共协调器领取、计算并提交实时联邦 BOINC 任务（支持 `physics_sim`、`code_generation` 或 `math_reasoning`，移除 `--once` 可在夜间持续计算）：",
    },
    "Or attach a standard BOINC client (`boinccmd`) directly to the public project URL and inspect verified receipts on [/api/v1/federated/ledger](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger):": {
        "es": "O conecta directamente un cliente BOINC estándar (`boinccmd`) a la URL pública del proyecto y consulta los recibos verificados en [/api/v1/federated/ledger](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger):",
        "de": "Oder verbinden Sie einen Standard-BOINC-Client (`boinccmd`) direkt mit der öffentlichen Projekt-URL und prüfen Sie verifizierte Quittungen unter [/api/v1/federated/ledger](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger):",
        "zh": "或者使用标准 BOINC 客户端（`boinccmd`）直接挂载公共项目 URL，并在 [/api/v1/federated/ledger](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/api/v1/federated/ledger) 查看经验证的计算回执：",
    },
    "Night training for end-users and HiveMind nodes (22:00–07:00 Paris time): End-users (volunteers, developers and gamers) leave their PC or laptop running overnight to fold exoplanet transits and compute rank-4 QLoRA inner-loop updates, earning 250 HMT and 45 Cobblestones per workunit to spend the next day on all models. Dedicated HiveMind nodes (GPU producers, HiveMiners and the owner fleet) switch from daytime OpenRouter inference (07:00–22:00) to nocturnal DiLoCo outer-step aggregation, Multi-Krum verification and SVD teacher-to-student distillation (live node telemetry at [/v1/hivemind/bees/owner-fleet-status](/v1/hivemind/bees/owner-fleet-status)).": {
        "es": "Entrenamiento nocturno para usuarios finales y nodos de HiveMind (22:00–07:00, hora de París): Los usuarios finales (voluntarios, desarrolladores y jugadores) dejan su PC o portátil calculando por la noche tránsitos de exoplanetas y actualizaciones QLoRA de rango 4, ganando 250 HMT y 45 Cobblestones por unidad para usar todos los modelos al día siguiente. Los nodos dedicados de HiveMind (productores GPU, HiveMiners y la flota principal) pasan de la inferencia diurna de OpenRouter (07:00–22:00) a la agregación nocturna DiLoCo, la verificación Multi-Krum y la destilación SVD profesor-estudiante (telemetría de nodos en vivo en [/v1/hivemind/bees/owner-fleet-status](/v1/hivemind/bees/owner-fleet-status)).",
        "de": "Nachttraining für Endnutzer und HiveMind-Knoten (22:00–07:00 Uhr Pariser Zeit): Endnutzer (Frewillige, Entwickler und Gamer) lassen ihren PC oder Laptop über Nacht Exoplaneten-Transits und Rang-4-QLoRA-Updates berechnen und erhalten 250 HMT sowie 45 Cobblestones pro Aufgabe, um am nächsten Tag alle Modelle zu nutzen. Dedizierte HiveMind-Knoten (GPU-Produzenten, HiveMiners und die Hauptflotte) wechseln von der OpenRouter-Tagesinferenz (07:00–22:00 Uhr) zur nächtlichen DiLoCo-Aggregation, Multi-Krum-Prüfung und SVD-Lehrer-Schüler-Destillation (Live-Knotentelemetrie unter [/v1/hivemind/bees/owner-fleet-status](/v1/hivemind/bees/owner-fleet-status)).",
        "zh": "终端用户与 HiveMind 节点的夜间协同训练（巴黎时间 22:00–07:00）：终端用户（志愿者、开发者与游戏玩家）在夜间让个人电脑或笔记本自动折叠系外行星凌星光变曲线并计算 4 秩 QLoRA 内层更新，每完成一个任务即可获得 250 HMT 与 45 Cobblestones，次日可用于调用全部模型。专用 HiveMind 节点（GPU 生产者、HiveMiners 与核心集群节点）则从日间 OpenRouter 低延迟推理（07:00–22:00）自动切换至夜间 DiLoCo 外层聚合、Multi-Krum 拜占庭校验与 SVD 师生蒸馏（节点实时遥测见 [/v1/hivemind/bees/owner-fleet-status](/v1/hivemind/bees/owner-fleet-status)）。",
    },
    "4. Gamers and Steam": {
        "es": "4. Jugadores y Steam",
        "de": "4. Gamer und Steam",
        "zh": "4. 游戏玩家与 Steam",
    },
    "Your GPU folds astrophysics and AI workunits only while you are not playing. Both the Free-to-Play Steam Edition (`USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter`, Steam `AppID 3133700`) and the automatic idle-GPU pause worker are available today, connecting gamers and citizen scientists on the [live community hub](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community).": {
        "es": "Tu GPU calcula unidades de astrofísica e IA solo cuando no estás jugando. Tanto la edición gratuita de Steam (`USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter`, Steam `AppID 3133700`) como el agente de pausa automática de GPU están disponibles hoy, uniendo a jugadores y científicos ciudadanos en el [centro comunitario en vivo](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community).",
        "de": "Ihre GPU berechnet Astrophysik- und KI-Aufgaben nur, während Sie nicht spielen. Sowohl die kostenlose Steam-Edition (`USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter`, Steam `AppID 3133700`) als auch der automatische GPU-Pausen-Worker sind heute verfügbar und verbinden Gamer und Citizen Scientists im [Live-Community-Hub](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community).",
        "zh": "您的 GPU 仅在您不玩游戏时计算天体物理与 AI 任务。Steam 免费版（`USS HiveMind NCC-31337: Astrophysics Exoplanet Hunter`，Steam `AppID 3133700`）与 GPU 空闲自动暂停代理今日均已开放，在[实时社区中心](https://hivemind-at-home-ruche-cfhmhmvv5a-uc.a.run.app/community)将游戏玩家与公民科学家紧密连接。",
    },
    "While you play (`idle < 4 s` of keyboard/mouse input or GPU load above 45%), BOINC GPU compute pauses automatically (`boinccmd --set_gpu_mode never`, 0% GPU overhead) so your game keeps 100% of your graphics card at 60 FPS. After 4 seconds of inactivity (with GPU load below 45% and temperature below 78 °C), it automatically resumes folding TESS/JWST Box-Least-Squares (BLS) exoplanet transits.": {
        "es": "Mientras juegas (`inactividad < 4 s` de teclado/ratón o carga de GPU superior al 45 %), el cálculo GPU de BOINC se pausa automáticamente (`boinccmd --set_gpu_mode never`, 0 % de sobrecarga) para que tu juego conserve el 100 % de la tarjeta gráfica a 60 FPS. Tras 4 segundos de inactividad (con carga de GPU inferior al 45 % y temperatura bajo 78 °C), reanuda automáticamente el análisis de tránsitos de exoplanetas TESS/JWST.",
        "de": "Während Sie spielen (`Inaktivität < 4 s` bei Tastatur/Maus oder GPU-Last über 45 %), pausiert die BOINC-GPU-Berechnung automatisch (`boinccmd --set_gpu_mode never`, 0 % GPU-Overhead), sodass Ihr Spiel 100 % der Grafikkarte bei 60 FPS nutzt. Nach 4 Sekunden Inaktivität (bei unter 45 % GPU-Last und unter 78 °C) wird die TESS/JWST-Exoplaneten-Transitfaltung automatisch fortgesetzt.",
        "zh": "当您正在游戏时（键盘/鼠标`空闲 < 4 秒`或 GPU 负载高于 45%），BOINC GPU 计算会自动暂停（`boinccmd --set_gpu_mode never`，0% GPU 额外开销），确保游戏独占 100% 显卡性能并稳定 60 FPS。在空闲满 4 秒后（且 GPU 负载低于 45%、温度低于 78 °C），程序会自动恢复折叠 TESS/JWST 系外行星凌星光变曲线。",
    },
    "Start the automatic idle-GPU worker (remove `--once` to leave the automatic pause/resume daemon running in the background while you play):": {
        "es": "Inicia el agente automático de GPU inactiva (quita `--once` para dejar el demonio de pausa/reanudación automática en segundo plano mientras juegas):",
        "de": "Starten Sie den automatischen Idle-GPU-Worker (entfernen Sie `--once`, damit der Pausen-/Fortsetzungs-Dienst im Hintergrund weiterläuft, während Sie spielen):",
        "zh": "启动 GPU 空闲自动暂停工作节点（移除 `--once` 可让自动暂停/恢复守护进程在您游戏期间持续后台运行）：",
    },
    "You can also trigger an immediate manual GPU pause or resume from a Steam launch option:": {
        "es": "También puedes activar una pausa o reanudación inmediata de la GPU desde una opción de lanzamiento de Steam:",
        "de": "Sie können die GPU-Pause oder -Fortsetzung auch sofort über eine Steam-Startoption auslösen:",
        "zh": "您也可以在 Steam 启动选项中立即手动触发 GPU 暂停或恢复：",
    },
    "If you also run an Ollama producer from path 2, a loaded language model keeps its video memory even when nobody is asking; before launching a heavy game, stop the producer agent (Ctrl+C) and unload the model (replace the name with your model’s):": {
        "es": "Si también ejecutas un productor Ollama de la vía 2, un modelo cargado conserva su memoria de vídeo aunque nadie pregunte; antes de abrir un juego pesado, detén el agente (Ctrl+C) y libera la memoria (reemplaza el nombre por el de tu modelo):",
        "de": "Wenn Sie zusätzlich einen Ollama-Produzenten aus Weg 2 betreiben, belegt ein geladenes Sprachmodell auch im Leerlauf Videospeicher; beenden Sie vor einem anspruchsvollen Spiel den Agenten (Strg+C) und geben Sie den Speicher frei (ersetzen Sie den Modellnamen):",
        "zh": "如果您同时运行了方式 2 的 Ollama 生产者节点，已加载的语言模型在无请求时仍会占用显存；在启动大型游戏前，请先停止代理（Ctrl+C）并卸载显存中的模型（将名称替换为您的模型）：",
    },
    "After you play: start the producer agent again, with the same command as in path 2 (the BOINC idle-GPU worker resumes by itself).": {
        "es": "Después de jugar: vuelve a iniciar el agente productor con el mismo comando de la vía 2 (el agente BOINC de GPU inactiva se reanuda por sí solo).",
        "de": "Nach dem Spielen: Starten Sie den Produzenten-Agenten wieder mit demselben Befehl wie in Weg 2 (der BOINC-Idle-GPU-Worker setzt von selbst fort).",
        "zh": "游戏结束后：使用方式 2 中的相同命令重新启动生产者代理（BOINC 空闲 GPU 工作节点会自动恢复运行）。",
    },
    "Good to know": {"es": "Bueno saberlo", "de": "Gut zu wissen", "zh": "重要须知"},
    "Producers see the questions sent to them. Do not run a producer on a machine that holds secrets, and do not send confidential text to the network.": {
        "es": "Los productores ven las preguntas que se les envían. No ejecutes un productor en una máquina con secretos ni envíes texto confidencial a la red.",
        "de": "Produzenten sehen die an sie gesendeten Fragen. Betreiben Sie keinen Produzenten auf einem Rechner mit geheimen Daten und senden Sie keine vertraulichen Texte an das Netzwerk.",
        "zh": "生产者节点可以看到发送给它们的问题。请勿在存有机密数据的机器上运行生产者节点，也不要向网络发送机密文本。",
    },
    "Points have no cash value and are not a payment.": {
        "es": "Los puntos no tienen valor monetario y no constituyen un pago.",
        "de": "Punkte haben keinen Geldwert und sind keine Bezahlung.",
        "zh": "积分没有任何货币价值，也不构成任何形式的金钱报酬。",
    },
    "Sharing uses electricity while your GPU answers. Measure your card with `nvidia-smi --query-gpu=power.draw --format=csv` and multiply by the hours it works.": {
        "es": "Compartir consume electricidad mientras tu GPU responde. Mide tu tarjeta con `nvidia-smi --query-gpu=power.draw --format=csv` y multiplícalo por las horas de trabajo.",
        "de": "Das Teilen verbraucht Strom, während Ihre GPU antwortet. Messen Sie Ihre Karte mit `nvidia-smi --query-gpu=power.draw --format=csv` und multiplizieren Sie den Wert mit den Betriebsstunden.",
        "zh": "当您的 GPU 处理任务时会消耗电力。您可以使用 `nvidia-smi --query-gpu=power.draw --format=csv` 测量显卡功耗并乘以运行小时数。",
    },
    "The software is open source (AGPL-3.0). Read the code before you run it.": {
        "es": "El software es de código abierto (AGPL-3.0). Lee el código antes de ejecutarlo.",
        "de": "Die Software ist Open Source (AGPL-3.0). Lesen Sie den Quellcode, bevor Sie ihn ausführen.",
        "zh": "本软件完全开源（AGPL-3.0 协议）。建议您在运行前审阅源代码。",
    },
    "The network is young and run by volunteers; availability is not guaranteed.": {
        "es": "La red es joven y está gestionada por voluntarios; la disponibilidad no está garantizada.",
        "de": "Das Netzwerk ist jung und wird von Freiwilligen betrieben; eine durchgehende Verfügbarkeit wird nicht garantiert.",
        "zh": "该网络仍处于早期阶段并由志愿者维护，不保证全天候可用性。",
    },
    "Questions and help": {"es": "Preguntas y ayuda", "de": "Fragen und Hilfe", "zh": "问题与帮助"},
    "Open an issue on [GitHub]({REPO}/issues), in English or French. For a security problem, read the [security policy]({REPO}/security/policy) first.": {
        "es": "Abre una incidencia en [GitHub]({REPO}/issues), en español, francés o inglés. Para un problema de seguridad, lee primero la [política de seguridad]({REPO}/security/policy).",
        "de": "Eröffnen Sie ein Issue auf [GitHub]({REPO}/issues) (auf Deutsch, Französisch oder Englisch). Lesen Sie bei Sicherheitsproblemen zuerst die [Sicherheitsrichtlinie]({REPO}/security/policy).",
        "zh": "请在 [GitHub]({REPO}/issues) 上提交 Issue（支持中文、法语或英语）。如遇安全问题，请先阅读[安全策略]({REPO}/security/policy)。",
    },
    LABELS["skip"][0]: {"es": "Ir al contenido", "de": "Zum Inhalt springen", "zh": "跳转到正文"},
    LABELS["choose"][0]: {"es": "Elige cómo participar", "de": "Wählen Sie Ihren Einstieg", "zh": "选择您的参与方式"},
    LABELS["other_lang"][0]: {"es": "Français", "de": "Français", "zh": "Français"},
    LABELS["tiers_summary"][0]: {
        "es": "¿Qué modelo para qué tarjeta? (todos los modelos)",
        "de": "Welches Modell für welche Grafikkarte? (alle Modelle)",
        "zh": "不同显卡适合运行哪些模型？（全部模型列表）",
    },
    LABELS["col_model"][0]: {"es": "Modelo", "de": "Modell", "zh": "模型"},
    LABELS["col_id"][0]: {"es": "Nombre para --preset", "de": "Name für --preset", "zh": "--preset 参数名"},
    LABELS["col_mem"][0]: {"es": "Memoria mínima (GB)", "de": "Mindestspeicher (GB)", "zh": "最低显存 (GB)"},
    LABELS["col_dl"][0]: {"es": "Descarga (GB)", "de": "Download (GB)", "zh": "下载大小 (GB)"},
    LABELS["tiers_missing"][0]: {
        "es": "La lista actualizada de modelos está en [/v1/hivemind/catalog](/v1/hivemind/catalog).",
        "de": "Die aktuelle Modellliste finden Sie unter [/v1/hivemind/catalog](/v1/hivemind/catalog).",
        "zh": "实时模型列表位于 [/v1/hivemind/catalog](/v1/hivemind/catalog)。",
    },
    LABELS["tiers_note"][0]: {
        "es": "Los nombres provienen del catálogo del proyecto. La memoria mínima es la tarjeta más pequeña recomendada, incluida la caché de contexto.",
        "de": "Die Namen stammen aus dem Projektkatalog. Der Mindestspeicher bezeichnet die kleinste empfohlene Karte inklusive Kontext-Cache.",
        "zh": "模型名称来自项目目录。最低显存包含上下文缓存在内的推荐显卡下限。",
    },
    LABELS["footer"][0]: {
        "es": "HiveMind La Ruche · código abierto (AGPL-3.0) · los puntos no tienen valor monetario",
        "de": "HiveMind La Ruche · Open Source (AGPL-3.0) · Punkte haben keinen Geldwert",
        "zh": "HiveMind La Ruche · 开源软件 (AGPL-3.0) · 积分没有任何货币价值",
    },
    LABELS["status_prefix"][0]: {"es": "Estado: ", "de": "Status: ", "zh": "状态："},
}

_CSS = """
:root{color-scheme:light dark;--bg:#fafafa;--fg:#1a1a1a;--mut:#555;--card:#fff;--line:#d8d8d8;--acc:#1a56db;--ok:#0b6b2f;--prev:#8a4b00;--dev:#5b2a86}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#ececec;--mut:#a8a8a8;--card:#1c1c1c;--line:#3a3a3a;--acc:#8ab4f8;--ok:#7be0a0;--prev:#ffc27a;--dev:#d3a8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--acc)}a:focus-visible,summary:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.skip{position:absolute;left:-999px}.skip:focus{left:1rem;top:1rem;background:var(--card);padding:.5rem;z-index:2}
header,main,footer{max-width:46rem;margin:0 auto;padding:0 1rem}header{padding-top:1.2rem}
.lang{float:right;font-size:.92rem;display:flex;gap:.45rem;flex-wrap:wrap;align-items:center}
.lang a{text-decoration:none;padding:.1rem .4rem;border-radius:.35rem;border:1px solid transparent}
.lang a[aria-current="page"]{font-weight:700;border-color:var(--acc);background:var(--card)}
h1{font-size:1.9rem;line-height:1.2;margin:.4rem 0 .6rem}h2{font-size:1.35rem;margin:2.2rem 0 .6rem;scroll-margin-top:1rem}
.lead{font-size:1.08rem}.cards{display:grid;gap:.8rem;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));padding:0;list-style:none;margin:1rem 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:.6rem;padding:.9rem 1rem}.card h3{margin:.1rem 0 .3rem;font-size:1.05rem}
.card a.more{display:inline-block;margin-top:.3rem}
.badge{display:inline-block;font-size:.82rem;font-weight:600;border:1px solid currentColor;border-radius:1rem;padding:.05rem .6rem;margin:.1rem 0 .4rem}
.ok{color:var(--ok)}.preview{color:var(--prev)}.dev{color:var(--dev)}
pre{background:var(--card);border:1px solid var(--line);border-radius:.5rem;padding:.8rem;overflow-x:auto;font:.88rem/1.45 ui-monospace,Consolas,monospace}
code{font:.92em ui-monospace,Consolas,monospace;background:var(--card);border:1px solid var(--line);border-radius:.3rem;padding:0 .25rem}pre code{border:0;padding:0;background:none}
.note{border-left:4px solid var(--acc);padding:.2rem .8rem;color:var(--mut)}ol,ul{padding-left:1.4rem}li{margin:.35rem 0}
details{margin:1rem 0}summary{cursor:pointer;font-weight:600}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:.93rem}
th,td{border:1px solid var(--line);padding:.35rem .5rem;text-align:left}th{background:var(--card)}td.n,th.n{text-align:right;white-space:nowrap}
footer{color:var(--mut);font-size:.9rem;padding-bottom:2.5rem;margin-top:2.5rem}
"""


def _t(pair: Pair, lang: str) -> str:
    if lang == "en":
        return pair[0]
    if lang == "fr":
        return pair[1]
    extra = TRANSLATIONS.get(pair[0])
    if extra and lang in extra:
        return extra[lang]
    return pair[1]


def _safe_href(url: str) -> bool:
    return url.startswith(("https://", "/", "#")) and not url.startswith("//")


_LINK = re.compile(r"\[([^\]]+)\]\(([^)\s]+)\)")
_CODE = re.compile(r"`([^`]+)`")


def _inline(text: str, subs: Mapping[str, str]) -> str:
    for key, value in subs.items():
        text = text.replace("{" + key + "}", value)
    text = html.escape(text, quote=True)

    def link(match: re.Match[str]) -> str:
        return f'<a href="{match.group(2)}">{match.group(1)}</a>' if _safe_href(html.unescape(match.group(2))) else match.group(1)

    return _CODE.sub(r"<code>\1</code>", _LINK.sub(link, text))


def _num(value: float, lang: str) -> str:
    text = f"{value:g}"
    return text.replace(".", ",") if lang in ("fr", "es", "de") else text


def _tiers_html(tiers: Sequence[Mapping[str, Any]] | None, lang: str, subs: Mapping[str, str]) -> str:
    summary = html.escape(_t(LABELS["tiers_summary"], lang))
    if not tiers:
        return f'<p class="note">{_inline(_t(LABELS["tiers_missing"], lang), subs)}</p>'
    rows = "".join(
        f'<tr><td>{html.escape(str(t.get("title", "")))}</td><td><code>{html.escape(str(t.get("tier_id", "")))}</code></td>'
        f'<td class="n">{_num(float(t.get("min_vram_gb", 0)), lang)}</td><td class="n">{_num(float(t.get("weights_gb", 0)), lang)}</td></tr>'
        for t in tiers
    )
    numeric = ' class="n"'
    head = "".join(f"<th{numeric if i > 1 else ''}>{html.escape(_t(LABELS[k], lang))}</th>" for i, k in enumerate(("col_model", "col_id", "col_mem", "col_dl")))
    return (f'<details><summary>{summary}</summary><div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
            f'<p class="note">{html.escape(_t(LABELS["tiers_note"], lang))}</p></details>')


def _block(kind: str, payload: Any, lang: str, subs: Mapping[str, str], tiers: Sequence[Mapping[str, Any]] | None, steps_done: int = 0) -> str:
    if kind == "code" and not payload:
        return ""  # a flag-dependent command that does not apply on this gateway
    if kind == "p":
        return f"<p>{_inline(_t(payload, lang), subs)}</p>"
    if kind == "note":
        return f'<p class="note">{_inline(_t(payload, lang), subs)}</p>'
    if kind in ("ol", "ul"):
        start = f' start="{steps_done + 1}"' if kind == "ol" and steps_done else ""  # numbering continues across the code blocks between steps
        return f"<{kind}{start}>" + "".join(f"<li>{_inline(_t(item, lang), subs)}</li>" for item in payload) + f"</{kind}>"
    if kind == "code":
        text = payload
        for key, value in subs.items():
            text = text.replace("{" + key + "}", value)
        return f"<pre><code>{html.escape(text)}</code></pre>"
    if kind == "tiers":
        return _tiers_html(tiers, lang, subs)
    raise ValueError(f"unknown block kind {kind!r}")


def render_join_page(lang: str = "fr", nonce: str = "abcdefgh12345678", *, tiers: Sequence[Mapping[str, Any]] | None = None,
                     api_url: str = DEFAULT_API_URL, repo_url: str = REPO_URL, path: str = "/join",
                     google_enabled: bool = False, handle_registration: bool = True) -> str:
    """The complete multilingual page (FR default, EN, ES, DE, ZH). ``tiers`` is the ``tiers`` list of ``/v1/hivemind/catalog``."""
    flags = Flags(google=google_enabled, handles=handle_registration)
    if lang not in LANGS:
        raise ValueError("lang must be one of 'fr', 'en', 'es', 'de', 'zh'")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", nonce):
        raise ValueError("nonce must be 8-64 URL-safe characters")
    if not _URL_OK.match(api_url) or not _URL_OK.match(repo_url) or not re.fullmatch(r"/[a-z-]*", path):
        raise ValueError("api_url and repo_url must be plain https URLs and path a simple absolute path")
    subs = {"API": api_url.rstrip("/"), "REPO": repo_url.rstrip("/")}
    esc = html.escape
    cards = "".join(
        f'<li class="card"><h3>{esc(_t(title, lang))}</h3><span class="badge {cls}"><span class="visually">{esc(_t(LABELS["status_prefix"], lang))}</span>'
        f'{esc(_t(status, lang))}</span><p>{esc(_t(summary, lang))}</p><a class="more" href="#{anchor}">→ {esc(_t(title, lang))}</a></li>'
        for anchor, title, cls, status, summary in PATHS
    )

    def section(sid: str, title: Pair, blocks: Sequence[tuple[str, Any]]) -> str:
        done, out = 0, []
        for kind, payload in blocks:
            if callable(payload):
                payload = payload(flags)
            out.append(_block(kind, payload, lang, subs, tiers, done))
            if kind == "ol":
                done += len(payload)
        return f'<section id="{sid}"><h2>{esc(_t(title, lang))}</h2>{"".join(out)}</section>'

    body = "".join(section(sid, title, blocks) for sid, title, blocks in SECTIONS)
    alternates = "".join(f'<link rel="alternate" hreflang="{code}" href="{path}?lang={code}">' for code in LANGS)
    current_attr = ' aria-current="page"'
    lang_links = "".join(
        f'<a href="{path}?lang={code}" hreflang="{code}" lang="{code}"'
        f'{current_attr if code == lang else ""}>{esc(LANG_NAMES[code])}</a>'
        for code in LANGS
    )
    return (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(_t(TITLE, lang))}</title><meta name=\"description\" content=\"{esc(_t(DESCRIPTION, lang), quote=True)}\">"
        f'<link rel="canonical" href="{path}?lang={lang}">{alternates}'
        f'<link rel="alternate" hreflang="x-default" href="{path}?lang=fr">'
        f'<style nonce="{nonce}">{_CSS}.visually{{position:absolute;left:-999px}}</style></head><body>'
        f'<a class="skip" href="#main">{esc(_t(LABELS["skip"], lang))}</a>'
        f'<header><nav class="lang" aria-label="Language / Langue">{lang_links}</nav>'
        f'<h1>{esc(_t(TITLE, lang))}</h1><p class="lead">{esc(_t(LEAD, lang))}</p></header>'
        f'<main id="main"><h2>{esc(_t(LABELS["choose"], lang))}</h2><ul class="cards">{cards}</ul>{body}</main>'
        f'<footer><p>{esc(_t(LABELS["footer"], lang))}</p></footer></body></html>'
    )
