"""Qwen3-0.6B (HF) with PEFT LoRA, exposing the same interface as night_jepa.Model (hidden / logits_at / lora)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class QwenModel:
    def __init__(self, r: int, alpha: float, name: str = "Qwen/Qwen3-0.6B"):
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM
        base = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda()
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
        self.m = get_peft_model(base, cfg)
        for n, p in self.m.named_parameters():
            if p.requires_grad:
                p.data = p.data.float()  # fp32 LoRA params (GradScaler needs fp32 grads)
        self.lora = [p for p in self.m.parameters() if p.requires_grad]
        self.engram = None

    def hidden(self, idx, mask=None, ckpt=False, view_starts=None, attention_mask=None):
        """Final hidden states after the final norm. `mask`/`view_starts` (packed views) are not supported for HF: the
        caller runs the two views as separate passes (the reference implementation's default mode)."""
        if mask is not None:
            raise ValueError("packed block-causal views are not used with the HF model; run views separately")
        self.m.train(ckpt)
        return self.m.get_base_model().model(input_ids=idx, attention_mask=attention_mask).last_hidden_state

    def logits_at(self, h):
        w = self.m.get_base_model().lm_head.weight
        return F.linear(h.to(w.dtype), w).float()
