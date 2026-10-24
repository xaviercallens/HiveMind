"""E10b diagnosis (CPU, 2 threads): does the inner loop learn with the E12-style LoRA init vs a PEFT-scale init?"""
import json, math, os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = "/home/callensxavier_gmail_com/HiveMind/.claude/worktrees/feat-s10-e11-e12-t4"
sys.path.insert(0, ROOT + "/scripts")
sys.argv = ["x"]
import torch

torch.set_num_threads(2)
import run_experiment_E10b_lattice_05b as e

STEPS = int(os.environ.get("DIAG_STEPS", "30"))
cfg = e.Cfg(rounds=1, learners=1, inner_steps=STEPS, seq_len=128, micro_batch=2, eval_seqs=16, eval_batch=4, eval_every=1)
shards, held, _ = e.real_data(cfg)
model = e.real_model("cpu")
params = e.lora_params(model)


@torch.no_grad()
def init(kind: str, seed: int = 1) -> None:
    gen = torch.Generator().manual_seed(seed)
    for layer in model.model.layers:
        for proj in (layer.self_attn.q_proj, layer.self_attn.v_proj):
            r, fan_in = proj.lora_A.shape
            if kind == "e12":
                a = torch.randn(r, fan_in, generator=gen) / math.sqrt(r)
            else:  # PEFT default: kaiming_uniform_(a=sqrt(5)) -> U(-1/sqrt(fan_in), 1/sqrt(fan_in))
                b = 1.0 / math.sqrt(fan_in)
                a = (torch.rand(r, fan_in, generator=gen) * 2 - 1) * b
            proj.lora_A.copy_(a)
            proj.lora_B.zero_()


out = {}
for kind, lr in (("e12", 5e-4), ("peft", 5e-4), ("peft", 2e-4)):
    init(kind)
    model.train()
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    h0 = e.eval_loss(model, held, 4, "cpu")
    tr = []
    t0 = time.time()
    for s in range(STEPS):
        b = shards[0][2 * s : 2 * s + 2]
        loss = model(input_ids=b, labels=b, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tr.append(round(float(loss), 4))
    h1 = e.eval_loss(model, held, 4, "cpu")
    key = f"{kind}_lr{lr:g}"
    out[key] = {"heldout_before": h0, "heldout_after": h1, "delta": h1 - h0, "train_first5": tr[:5], "train_last5": tr[-5:],
                "sec_per_step_cpu": round((time.time() - t0) / STEPS, 2), "A_std": float(torch.cat([p.view(-1) for n, p in model.named_parameters() if n.endswith("lora_A")]).std())}
    print(key, json.dumps(out[key]), flush=True)
json.dump(out, open("/home/callensxavier_gmail_com/.claude/jobs/22ff63cf/tmp/e10b_init_diag.json", "w"), indent=1)
print("done", flush=True)
