# S21: Bring Your Own Bee (contributor guide and design note)

Module: `src/hivemind/swarm/byob.py` · CLI: `scripts/byob_submit.py` · Tests: `tests/test_byob.py`

A *bee* is a model the hive can route requests to. Bring Your Own Bee (BYOB) lets an external contributor offer one
without commit access: you publish the model on the Hugging Face Hub, sign a small JSON manifest with your contributor
key, and the hive checks it. This page is the contributor guide and records how the checks work.

The flagship bee `hivemind/bee-rusty-sundials` (`docs/BYOB_RUSTY_SUNDIALS_SCIENTIFIC_BEE.md`) is maintained in this
repository. This pipeline is for bees that live somewhere else.

## 1. Lifecycle

```
submitted --static checks--> rejected | blocked | candidate
candidate --held-out evaluation in isolation--> rejected | candidate (BLOCKED) | shadow
shadow --owner approval record (HMAC, bound to the manifest digest)--> live
```

* **Static validation** (no model is run) can reject a bee or make it a *candidate*. If the Hub cannot be reached, the
  result is BLOCKED. A check that could not run never counts as a pass.
* **Evaluation** runs your model on a held-out pack that the hive keeps outside this public repository. Running submitted
  weights needs an isolated executor (sandbox and GPU). When none is configured, the stage is **BLOCKED** and the bee
  stays a candidate. Today no host is configured, because the GWAYA sandbox (bubblewrap) is unavailable on the owner's
  hosts, so in practice no submission gets past *candidate* yet.
* A passing evaluation moves the bee to **shadow** at most: it may receive mirrored traffic whose answers are logged but
  never returned to users.
* Only `approve_live` can make a bee **live**. It requires an approval record signed with the owner key over the exact
  `manifest_sha256`. Any change to the manifest needs a new approval.

## 2. The manifest

```bash
python scripts/byob_submit.py template > my_bee.json   # then edit it
```

| field | rule |
|---|---|
| `schema` | `hivemind.byob.manifest/v1` |
| `bee_name` | `bee-` + lowercase letters, digits and `-`; 8 to 52 characters; must not collide (see 3.8) |
| `contributor_id` | your registered id (lowercase, 3 to 64 characters) |
| `hf_repo_id` | `<owner>/<name>` on the Hub |
| `revision` | **the 40-hex commit sha** (Hub, *Files and versions*, *History*). Branch names and tags are rejected |
| `kind` | `lora_adapter`, `gguf`, `full_model` or `serving_endpoint` |
| `base_model` | Hub repo id of the base. For `lora_adapter` it must be one the hive serves (3.5) |
| `license` | Hub license id; must be in the allowlist (3.1) and equal the model card's |
| `domain` | lowercase slug, for example `cpp` or `stiff-ode` |
| `vram_tier_gb` | one of 4, 6, 8, 12, 16, 24; the estimated residency must fit |
| `eval_pack` | slug of the hive-held evaluation pack you ask to be scored on |
| `description` | at most 2,000 characters; scanned with the firewall's prompt-injection rules |
| `routing_keywords` | at most 32 lowercase keywords of up to 40 characters each |
| `endpoint_url` | `serving_endpoint` only: public `https` URL of an OpenAI-format server |
| `subfolder` | optional folder of the repo that holds the adapter or model |
| `primary_file` | `gguf` only: the one `.gguf` file to serve |
| `signature` | HMAC-SHA256 (hex) over every other field in canonical JSON, keyed with your contributor key |

Unknown fields are refused. A misspelled field would otherwise be dropped without notice, and `trust_remote_code` is
not a field anyone can set.

### Keys, signing and self-check

Your contributor key is a secret of at least 32 bytes that you share with the hive maintainers when you register.
Keep it in a file only you can read:

```bash
python -c "import secrets; print(secrets.token_hex(32))" > ~/.hivemind/byob.key && chmod 600 ~/.hivemind/byob.key
python scripts/byob_submit.py sign  --manifest my_bee.json --key-file ~/.hivemind/byob.key
python scripts/byob_submit.py check --manifest my_bee.json --key-file ~/.hivemind/byob.key --out entry.json
```

`check` runs the same static validation the hive runs, against the live Hub, and prints the registry entry. Exit code 0
means static validation passed, 1 means REJECT and 2 means BLOCKED. On your machine the evaluation stage reports
`NO_ISOLATED_EXECUTOR`, so a self-check can reach `stage: candidate` at best.

## 3. Static checks and reason codes

Every check that fails adds a `{code, detail}` reason, and any reason rejects the bee. You get all the reasons in one
pass, not only the first. There are two exceptions. A missing or wrong signature stops validation immediately. A
manifest whose `revision` is unpinned or whose fields are malformed is never sent to the Hub.

1. **License** (`LICENSE_NOT_ALLOWED`, `LICENSE_MISMATCH`, `LICENSE_UNDECLARED_ON_HUB`, `BASE_LICENSE_NOT_ALLOWED`).
   The allowlist is apache-2.0, mit, bsd-2-clause, bsd-3-clause, cc-by-4.0, agpl-3.0 and gpl-3.0. Each allows
   redistribution and is compatible with shipping beside AGPL-3.0 code. Non-commercial licences, `other` and custom model
   licences are refused, and so is a base model under one of them: an adapter on a Gemma base inherits the Gemma terms,
   so it is rejected even if the adapter itself is Apache. The licence is read from `cardData.license`, falling back to a
   `license:` tag.
2. **Pinned revision** (`REVISION_NOT_PINNED`, `REVISION_MISMATCH`). The revision must be a 40-hex sha, and the Hub must
   resolve it to itself.
3. **Repository** (`REPO_NOT_FOUND`, `REPO_GATED_OR_PRIVATE`). The repository must be public and not gated: the hive
   mirrors what it serves.
4. **Files** (`PICKLE_FORMAT`, `REMOTE_CODE`, `FILE_TYPE_NOT_ALLOWED`, `PATH_UNSAFE`, `MISSING_REQUIRED_FILE`). The whole
   recursive tree at the pinned commit is checked, not only the folder that will be loaded:
   * Weights must be `.safetensors` or `.gguf`.
   * Formats that can carry a pickle (`.bin .pt .pth .pkl .pickle .ckpt .joblib .npy .npz .h5 .msgpack .marshal .dill`)
     reject the repository. This includes a `training_args.bin` or an `optimizer.pt` left in a checkpoint folder: delete
     them, or publish the bee from a clean repository.
   * Code files (`.py .sh .ipynb` and native libraries) count as remote code.
   * Other files are allowed only from a short list: JSON, Markdown, text, tokenizer, template, YAML, CSV and image
     formats, plus `LICENSE`, `NOTICE`, `README` and `Modelfile`.
   * Required files: `adapter_config.json` and `adapter_model.safetensors` for `lora_adapter`; `config.json` and at least
     one `.safetensors` for `full_model` and `serving_endpoint`; the named `primary_file` for `gguf`.
5. **Remote code** (`REMOTE_CODE`). The bee is rejected when any of these holds:
   * the Hub tags the repository `custom_code`;
   * the Hub's parsed config, `config.json` or `tokenizer_config.json` has an `auto_map`;
   * a peft `auto_mapping` names anything except a class inside the installed `transformers` package.

   Ordinary adapters do write `{"base_model_class": "Qwen2ForCausalLM", "parent_library":
   "transformers.models.qwen2.modeling_qwen2"}`, and that form is accepted. The hive never passes `trust_remote_code`.
6. **Base model** (`BASE_MODEL_MISMATCH`, `BASE_MODEL_NOT_SERVED`, `BASE_MODEL_NOT_FOUND`, `ADAPTER_TASK_NOT_CAUSAL_LM`).
   For adapters, `adapter_config.json`'s `base_model_name_or_path` must equal `base_model` (case-insensitive) and
   `task_type` must be `CAUSAL_LM`. The base must also be in `HIVE_SERVED_BASES`:

   | base (Hub id) | catalog tier |
   |---|---|
   | `Qwen/Qwen2.5-Coder-3B-Instruct` | `coder-lite` |
   | `Qwen/Qwen2.5-Coder-7B-Instruct` | `coder` |
   | `Qwen/Qwen2.5-Coder-14B-Instruct` | `coder-plus` |
   | `Qwen/Qwen3-8B` | `qwen35-core` |
   | `mistralai/Mistral-7B-Instruct-v0.3` | `mistral-core` |
   | `google/gemma-3-4b-it` | `gemma-core` (base licence not allowlisted, so adapters are rejected) |

   Repackaged copies such as `unsloth/Qwen2.5-Coder-7B-Instruct` are not on the list. That they hold identical weights is
   a claim nobody has checked. If you trained on such a copy, set `base_model_name_or_path` to the upstream id, and only
   if that is really the base. For the other kinds, a `base_model` in the model card must agree with the manifest.
7. **VRAM tier** (`VRAM_TIER_INVALID`, `VRAM_TIER_EXCEEDED`). The residency estimate is a planning rule, not a
   measurement:
   * adapter: the catalog tier's `min_vram_gb` for the base, plus the adapter file size;
   * `gguf` and `full_model`: weights × 1.10 + 1 GB context allowance;
   * `serving_endpoint`: no estimate, because the hardware is yours.
8. **Name collision** (`NAME_COLLISION`). Names are compared after lower-casing, dropping non-alphanumerics and stripping
   a leading `hivemind` and `bee`. So `bee-coder` collides with the `coder` tier, and `bee-python-code-gwaya` with the
   serving expert. The names checked against are catalog tiers, partner aliases, serving experts, Season 3 domain
   experts, the flagship bee, and whatever the registry passes in `extra`.
9. **Rate limit** (`RATE_LIMITED`). Each contributor gets 3 signed submissions per 24 hours by default
   (`SubmissionLedger`, JSONL-backed). Only a manifest whose signature verified is counted, so someone who forges your
   id cannot use up your quota. A submission that is signed but rejected does count.
10. **Description** (`DESCRIPTION_INJECTION`). The description goes through `hivemind_firewall.scan_text_for_injection`.
    Text that the scan rates INJECTION (directive phrases, chat-role markers, authority claims, ...) rejects the bee.
11. **Endpoint** (`ENDPOINT_URL_INVALID`). The URL must be `https`, carry no credentials, and not point at `localhost`,
    `*.internal` or `*.local`, or at a literal non-global IP address such as `169.254.169.254`. DNS rebinding is outside
    what a static check can stop. The prober must therefore run from a network that has no internal access.

What static validation does **not** do: it downloads no weights (only JSON configs), so a `.safetensors` file is trusted
to be safetensors only by its name. The isolated stage has to load files with safetensors or llama.cpp loaders only, and
those loaders refuse anything else.

## 4. Evaluation stage

`evaluate(manifest, static, pack=..., executor=..., prober=...)` checks its inputs in this order, and the first failure
decides the verdict:

| condition | verdict |
|---|---|
| static verdict is not ACCEPT | REJECT, or BLOCKED if static was BLOCKED |
| weight kinds: no `IsolatedExecutor` (`NO_ISOLATED_EXECUTOR`), or one with `isolated=False` (`EXECUTOR_NOT_ISOLATED`) | BLOCKED |
| `serving_endpoint`: no prober configured | BLOCKED |
| no held-out pack, or a pack whose id is not `eval_pack` | BLOCKED |
| the pack's own controls fail (`HELDOUT_CONTROL_FAILED`) | BLOCKED |
| endpoint error or timeout on any item (`ENDPOINT_PROBE_FAILED`) | REJECT |
| pass rate below `min_pass_rate` (default 0.6) | REJECT (`EVAL_BELOW_THRESHOLD`) |
| otherwise | ACCEPT, and the registry stage becomes `shadow` |

Each pack controls itself before any answer is scored. Every item's reference answer must pass its own checker
(positive control). An empty answer and the prompt echoed back must both fail it (negative control). A checker such as
`.*` would pass every bee, so a pack containing one is BLOCKED rather than used. Reference answers are never sent to the
bee. Endpoints are probed with one OpenAI-format chat request per item (`https_chat_prober`) under a per-item timeout.

## 5. Registry entry (interface to the S21 expert registry)

`build_registry_entry` returns a plain JSON dict, schema `hivemind.byob.registry_entry/v1`, with exactly these fields in
this order (`REGISTRY_ENTRY_FIELDS`; a test enforces the order):

| field | meaning |
|---|---|
| `schema` | `hivemind.byob.registry_entry/v1` |
| `expert_id`, `public_id` | `bee-...`, `hivemind/bee-...` |
| `contributor_id`, `kind`, `domain`, `routing_keywords`, `description` | copied from the manifest |
| `source` | `{hf_repo_id, revision, subfolder, weight_files: [{path, size, sha256}], weight_bytes}`; `sha256` is the Hub's LFS sha256 |
| `base_model`, `base_tier_id` | declared base, and the catalog tier when the base is served (else `null`) |
| `license`, `vram_tier_gb`, `resident_gb_estimate`, `eval_pack`, `endpoint_url` | as checked (`endpoint_url` is `null` except for endpoints) |
| `stage` | `rejected`, `blocked`, `candidate` or `shadow` (`live` only through `approve_live`) |
| `static`, `evaluation` | the full stage reports, including verdicts and reasons (`evaluation` is `null` when static did not pass) |
| `reasons` | static and evaluation reasons concatenated, `[{code, detail}]` |
| `manifest_sha256` | SHA-256 of the signed manifest bytes; owner approvals bind to it |
| `owner_approval` | `null`, or `{approver, approved_utc, manifest_sha256, signature}` after `approve_live` |
| `created_utc` | ISO-8601 UTC |

The registry should accept only `stage in {"shadow", "live"}` for routing, treat `candidate` as listed but not routable,
and key residency planning on `resident_gb_estimate` and `vram_tier_gb`.

## 6. Interfaces left for the other S21 pieces

* **Expert registry (moescale).** It ingests the entry dict above and passes its own expert ids to
  `existing_bee_names(extra=...)` so that names do not collide.
* **Campaign orchestrator (orch).** It can schedule the evaluation stage in a night window once an `IsolatedExecutor`
  exists on a GPU host.
* **Base model catalog (basemodels).** When Qwen3.5 bases are promoted, add their Hub ids to `HIVE_SERVED_BASES`. A test
  checks that each served base points at a real catalog tier.

## 7. Real check run on 2026-10-28

`python scripts/byob_submit.py check` was run against the live Hub on three public LoRA repositories, with a throwaway
contributor key. The verdicts are in the S21 BYOB pull request description, together with the reasons each one
returned.
