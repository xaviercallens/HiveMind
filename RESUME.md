# Resume the T4 campaign after a Spot-VM preemption

State captured: 2026-10-26 00:31:38 UTC from branch `chore/zenodo-v3.8.0-deposit` at `50ca854`.

1. Get the code: `git fetch origin chore/zenodo-v3.8.0-deposit` and work from that branch (all code, prereg files and top-level `artifacts/experiments/*.json` reports are
   committed there). This checkpoint branch holds only the ignored/volatile state under `state/`.
2. Restore the volatile state (optional but saves GPU time): copy `state/artifacts/experiments/E15`, `E16` (stage JSONs and journals) back into
   `artifacts/experiments/` so that `--resume` skips finished stages and problems.
3. Check the GPU is idle (`nvidia-smi`), the Ollama daemon is up (`curl -s 127.0.0.1:11434/api/tags`) and the models are present
   (`ollama pull qwen2.5-coder:3b`; the 1.5b and 7b-instruct tags were already pulled). Hugging Face caches re-download automatically.
4. Relaunch the queue: `python3 scripts/run_t4_benchmark_queue.py --hours 9 --queue artifacts/experiments/T4_benchmark_queue_ext.json --log artifacts/experiments/T4_benchmark_queue_ext_log.json`.
   Jobs are resumable (`--resume`; completed stages/cells are skipped; a job killed mid-run loses only its current cell). The LoRA weights of E15 are
   NOT checkpointed (large): its train stage reruns from the rejection-sampling journal.
5. Read `state/memory/` for the full plan, decisions and caveats (project memory notes), and `state/MANIFEST.json` for file hashes.
6. Remember the rules: no merge/tag by the assistant, no Zenodo publish before the provenance lock passes, the PR #21 of GWAYA is not touched without the owner.
