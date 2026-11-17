# Copyright (c) 2024-2026 Xavier Callens / Project HiveMind. Licensed under the AGPL-3.0.
"""Season 3 (`HM-S3-04`) Hugging Face Domain Dataset & QLoRA Expert Trainer Package."""

from hivemind.data.hf_real_trainer import (
    CURATED_SEASON3_DOMAIN_CORPORA,
    train_and_publish_season3_domain_experts,
    train_single_domain_lora_adapter,
)

__all__ = [
    "CURATED_SEASON3_DOMAIN_CORPORA",
    "train_and_publish_season3_domain_experts",
    "train_single_domain_lora_adapter",
]
