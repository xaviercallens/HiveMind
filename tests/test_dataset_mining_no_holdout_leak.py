# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""The dataset-mining campaign never tokenises a holdout document into a training batch."""

from __future__ import annotations

from pathlib import Path

import pytest

from hivemind.data import dataset_mining_campaign as dmc


def test_training_batches_contain_no_holdout_document(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, ...]] = []
    real_tokenize = dmc.tokenize_documents

    def recording_tokenize(texts, **kwargs):  # noqa: ANN001
        calls.append(tuple(texts))
        return real_tokenize(texts, **kwargs)

    monkeypatch.setattr(dmc, "tokenize_documents", recording_tokenize)
    dmc.run_dataset_mining_and_ruche_campaign(
        vault_dir=tmp_path / "vault", report_json_path=tmp_path / "report.json", training_steps=2, learning_rate=0.03
    )
    assert calls, "tokenize_documents was never called"
    for domain_name, spec in dmc.SCIENTIFIC_MINING_DOMAIN_CORPORA.items():
        train, holdout = tuple(spec["train"]), tuple(spec["holdout"])
        assert train in calls and holdout in calls, domain_name
        # Negative control of the leak: no tokenised batch mixes a holdout document with a train document.
        for batch in calls:
            if any(doc in holdout for doc in batch):
                assert all(doc in holdout for doc in batch), f"{domain_name}: holdout document inside a training batch"
    assert all(len(set(batch)) == len(batch) for batch in calls)
