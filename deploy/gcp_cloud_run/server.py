"""Cloud Run entrypoint for HiveMind@home, La Ruche MoE Vault & Federated AI Gateway."""

from __future__ import annotations

import os
from pathlib import Path
import uvicorn

from hivemind.boinc.hivemind_at_home_dashboard import create_hivemind_at_home_app

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

app = create_hivemind_at_home_app(
    gpu_campaigns_report_path=ARTIFACTS_DIR
    / "season9_10_gpu_swarm_campaigns_report.json",
    mining_ruche_report_path=ARTIFACTS_DIR / "dataset_mining_and_ruche_report.json",
    vault_dir=ARTIFACTS_DIR / "la_ruche_vault",
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
