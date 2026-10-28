# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The public "join HiveMind" page, in French and English (``/join``, ``/rejoindre``).

Standard library only and no imports from the rest of ``hivemind``: the gateway serves it, and the La Ruche front door (``deploy/ruche_router_overlay``) copies
this one file next to its ``server.py``.

Rules this page follows, on purpose:
* Every sentence is written ONCE as an ``(English, French)`` pair, so both languages always have the same sections, steps and commands.
* It only describes what works today and says plainly what does not (BOINC is a preview, the Steam "pause while I play" mode is not released).
* HiveMind runs on HiveMind Reflexion Points (HMT) only: no euro or dollar amount appears (see ``points_wording.py``).
* No script, no external resource: the content security policy allows only the inline style carrying the nonce.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping, Sequence
from typing import Any

REPO_URL = "https://github.com/xaviercallens/HiveMind"
DEFAULT_API_URL = "https://hivemind-openrouter-1003063861791.us-central1.run.app"
LANGS: tuple[str, str] = ("en", "fr")
Pair = tuple[str, str]  # (English, French)

_URL_OK = re.compile(r"^https://[a-z0-9.-]+(?::\d+)?(?:/[A-Za-z0-9._~%/-]*)?$")


def pick_language(query_lang: str | None, accept_language: str | None, default: str = "en") -> str:
    """``?lang=fr|en`` wins; otherwise the first of fr/en in ``Accept-Language`` by quality; otherwise ``default``."""
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
# content: (English, French). Inline markup: `code` and [text](url). {API} and {REPO} are replaced before rendering.
# ---------------------------------------------------------------------------------------------------------------------------------------------------

TITLE: Pair = ("Join HiveMind La Ruche", "Rejoindre HiveMind La Ruche")
DESCRIPTION: Pair = (
    "Use community AI models, or share your graphics card with HiveMind La Ruche: what works today, step by step, in English and French.",
    "Utilisez des modèles d'IA communautaires ou partagez votre carte graphique avec HiveMind La Ruche : ce qui fonctionne aujourd'hui, étape par étape, en français et en anglais.",
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
    ("train", ("Train models at night (BOINC)", "Entraîner des modèles la nuit (BOINC)"), "preview",
     ("Preview · not open to the public yet", "Aperçu · pas encore ouvert au public"),
     ("Lend your GPU to training jobs through the standard BOINC client. Linux and NVIDIA only for now.",
      "Prêtez votre GPU à des tâches d'entraînement avec le client BOINC standard. Linux et NVIDIA seulement pour l'instant.")),
    ("play", ("Gamers and Steam", "Joueurs et Steam"), "dev", ("In development", "En développement"),
     ("Give GPU time only while you are not playing. The automatic pause is not released yet; this page shows how to do it by hand today.",
      "Donnez du temps de GPU seulement quand vous ne jouez pas. La pause automatique n'est pas encore publiée; cette page montre comment le faire à la main dès aujourd'hui.")),
)

# Blocks per section: ("p", pair) | ("ol", [pair, ...]) | ("ul", [pair, ...]) | ("code", text) | ("note", pair) | ("tiers", None)
SECTIONS: Sequence[tuple[str, Pair, Sequence[tuple[str, Any]]]] = (
    ("use", ("1. Use the models", "1. Utiliser les modèles"), (
        ("ol", [
            ("Open the [sign-up page](/signup) and sign in with Google. Choose “Use models in my editor” (or “Both”). We keep only an anonymous hash of "
             "your Google account id, not your e-mail address.",
             "Ouvrez la [page d'inscription](/signup) et connectez-vous avec Google. Choisissez « Use models in my editor » (utiliser les modèles dans mon "
             "éditeur) ou « Both » (les deux). Nous ne gardons qu'une empreinte anonyme de votre identifiant Google, pas votre courriel."),
            ("Copy the API key. It is shown once; signing in again replaces it.",
             "Copiez la clé d'API. Elle n'est affichée qu'une fois; se reconnecter la remplace."),
            ("A new account holds 0 points. Points come from sharing a GPU (path 2) or from a grant by the operator: if you only want to try, ask for one "
             "in a [GitHub issue]({REPO}/issues).",
             "Un nouveau compte a 0 point. Les points viennent du partage d'un GPU (voie 2) ou d'un don de l'opérateur : si vous voulez seulement "
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
        ("p", ("Your machine runs a model, the gateway sends it questions, and you earn points for the tokens you serve. Plan on a terminal and about an "
               "hour the first time.",
               "Votre ordinateur fait tourner un modèle, la passerelle lui envoie des questions, et vous gagnez des points pour les jetons (tokens) servis. "
               "Prévoyez un terminal et environ une heure la première fois.")),
        ("ol", [
            ("Get the code (Python 3.10 or newer):", "Obtenez le code (Python 3.10 ou plus récent) :"),
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
        ("ol", [
            ("Create a producer account on the [sign-up page](/signup) (choose “Share my GPU” or “Both”) and keep the key.",
             "Créez un compte producteur sur la [page d'inscription](/signup) (choisissez « Share my GPU » ou « Both ») et gardez la clé."),
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
    ("train", ("3. Train models at night with BOINC (preview)", "3. Entraîner des modèles la nuit avec BOINC (aperçu)"), (
        ("p", ("This path is not open to the public yet: the training server is not reachable from the internet. We will announce it here when it is. "
               "You can already check whether your machine qualifies.",
               "Cette voie n'est pas encore ouverte au public : le serveur d'entraînement n'est pas accessible depuis Internet. Nous l'annoncerons ici "
               "quand il le sera. Vous pouvez déjà vérifier si votre machine est admissible.")),
        ("ul", [
            ("Linux on an x86-64 processor (Windows and macOS are not supported yet)",
             "Linux sur processeur x86-64 (Windows et macOS ne sont pas encore pris en charge)"),
            ("An NVIDIA card with at least 3.5 GB of video memory", "Une carte NVIDIA avec au moins 3,5 Go de mémoire vidéo"),
            ("A first download of about 4.4 GB, reused afterwards", "Un premier téléchargement d'environ 4,4 Go, réutilisé ensuite"),
            ("Community work is offered at night, 22:00–07:00 Paris time, when the network is least busy",
             "Le travail communautaire est offert la nuit, de 22 h à 7 h, heure de Paris, quand le réseau est le moins occupé"),
            ("The standard BOINC client; the project’s files are digitally signed", "Le client BOINC standard; les fichiers du projet sont signés numériquement"),
        ]),
        ("p", ("This check only reads facts about your machine. It installs nothing and sends nothing:",
               "Cette vérification ne fait que lire des faits sur votre machine. Elle n'installe rien et n'envoie rien :")),
        ("code", "python3 scripts/hivemind_volunteer_preflight.py"),
    )),
    ("play", ("4. Gamers and Steam", "4. Joueurs et Steam"), (
        ("p", ("The goal: your GPU answers questions only while you are not playing. What exists today is a Steam “Hack-Miner” prototype that runs locally "
               "on your PC. What does not exist yet is a release that pauses by itself when a game starts.",
               "L'objectif : votre GPU ne répond aux questions que lorsque vous ne jouez pas. Ce qui existe aujourd'hui, c'est un prototype Steam "
               "« Hack-Miner » qui tourne localement sur votre PC. Ce qui n'existe pas encore, c'est une version qui se met d'elle-même en pause quand "
               "un jeu démarre.")),
        ("p", ("Until then, do it by hand. A loaded model keeps its video memory even when nobody is asking, and a game needs that memory.",
               "D'ici là, faites-le à la main. Un modèle chargé garde sa mémoire vidéo même quand personne ne pose de question, et un jeu a besoin de "
               "cette mémoire.")),
        ("ol", [
            ("Before you play: stop the agent (Ctrl+C), then free the memory (replace the name with your model’s):",
             "Avant de jouer : arrêtez l'agent (Ctrl+C), puis libérez la mémoire (remplacez le nom par celui de votre modèle) :"),
        ]),
        ("code", "ollama stop qwen2.5-coder:3b"),
        ("ol", [
            ("After you play: start the agent again (step 6 of path 2).", "Après avoir joué : redémarrez l'agent (étape 6 de la voie 2)."),
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
    "status": ("Status", "État"),
}

_CSS = """
:root{color-scheme:light dark;--bg:#fafafa;--fg:#1a1a1a;--mut:#555;--card:#fff;--line:#d8d8d8;--acc:#1a56db;--ok:#0b6b2f;--prev:#8a4b00;--dev:#5b2a86}
@media (prefers-color-scheme:dark){:root{--bg:#121212;--fg:#ececec;--mut:#a8a8a8;--card:#1c1c1c;--line:#3a3a3a;--acc:#8ab4f8;--ok:#7be0a0;--prev:#ffc27a;--dev:#d3a8ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:17px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif}
a{color:var(--acc)}a:focus-visible,summary:focus-visible{outline:3px solid var(--acc);outline-offset:2px}
.skip{position:absolute;left:-999px}.skip:focus{left:1rem;top:1rem;background:var(--card);padding:.5rem;z-index:2}
header,main,footer{max-width:46rem;margin:0 auto;padding:0 1rem}header{padding-top:1.2rem}
.lang{float:right;font-size:.95rem}h1{font-size:1.9rem;line-height:1.2;margin:.4rem 0 .6rem}h2{font-size:1.35rem;margin:2.2rem 0 .6rem;scroll-margin-top:1rem}
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
    return pair[0] if lang == "en" else pair[1]


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
    return text.replace(".", ",") if lang == "fr" else text


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


def render_join_page(lang: str, nonce: str, *, tiers: Sequence[Mapping[str, Any]] | None = None, api_url: str = DEFAULT_API_URL,
                     repo_url: str = REPO_URL, path: str = "/join") -> str:
    """The complete page. ``tiers`` is the ``tiers`` list of ``/v1/hivemind/catalog`` (``None``: link to the live list instead)."""
    if lang not in LANGS:
        raise ValueError("lang must be 'en' or 'fr'")
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,64}", nonce):
        raise ValueError("nonce must be 8-64 URL-safe characters")
    if not _URL_OK.match(api_url) or not _URL_OK.match(repo_url) or not re.fullmatch(r"/[a-z-]*", path):
        raise ValueError("api_url and repo_url must be plain https URLs and path a simple absolute path")
    subs = {"API": api_url.rstrip("/"), "REPO": repo_url.rstrip("/")}
    other = "fr" if lang == "en" else "en"
    esc = html.escape
    cards = "".join(
        f'<li class="card"><h3>{esc(_t(title, lang))}</h3><span class="badge {cls}"><span class="visually">{esc(_t(LABELS["status"], lang))}: </span>'
        f'{esc(_t(status, lang))}</span><p>{esc(_t(summary, lang))}</p><a class="more" href="#{anchor}">→ {esc(_t(title, lang))}</a></li>'
        for anchor, title, cls, status, summary in PATHS
    )
    def section(sid: str, title: Pair, blocks: Sequence[tuple[str, Any]]) -> str:
        done, out = 0, []
        for kind, payload in blocks:
            out.append(_block(kind, payload, lang, subs, tiers, done))
            if kind == "ol":
                done += len(payload)
        return f'<section id="{sid}"><h2>{esc(_t(title, lang))}</h2>{"".join(out)}</section>'

    body = "".join(section(sid, title, blocks) for sid, title, blocks in SECTIONS)
    return (
        f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{esc(_t(TITLE, lang))}</title><meta name=\"description\" content=\"{esc(_t(DESCRIPTION, lang), quote=True)}\">"
        f'<link rel="canonical" href="{path}?lang={lang}"><link rel="alternate" hreflang="en" href="{path}?lang=en">'
        f'<link rel="alternate" hreflang="fr" href="{path}?lang=fr"><style nonce="{nonce}">{_CSS}.visually{{position:absolute;left:-999px}}</style></head><body>'
        f'<a class="skip" href="#main">{esc(_t(LABELS["skip"], lang))}</a>'
        f'<header><nav class="lang" aria-label="Language / Langue"><a href="{path}?lang={other}" hreflang="{other}" lang="{other}">'
        f'{esc(_t(LABELS["other_lang"], lang))}</a></nav><h1>{esc(_t(TITLE, lang))}</h1><p class="lead">{esc(_t(LEAD, lang))}</p></header>'
        f'<main id="main"><h2>{esc(_t(LABELS["choose"], lang))}</h2><ul class="cards">{cards}</ul>{body}</main>'
        f'<footer><p>{esc(_t(LABELS["footer"], lang))}</p></footer></body></html>'
    )
