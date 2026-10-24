import os, sys, time, glob, math
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch, torch.nn as nn
torch.set_num_threads(2)
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers; print('transformers', transformers.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())
HUB=os.path.expanduser('~/.cache/huggingface/hub')
TEXT=("The Amazon River in South America is the largest river by discharge volume of water in the world, and the disputed longest river system in the world in comparison to the Nile. "
"The headwaters of the Apurimac River on Nevado Mismi had been considered for nearly a century as the Amazon basin's most distant source, until a 2014 study found it to be the headwaters of the Mantaro River on the Cordillera Rumi Cruz in Peru. "
"The Mantaro and Apurimac rivers join, and with other tributaries form the Ucayali River, which in turn meets the Marañón River upstream of Iquitos, Peru, forming what countries other than Brazil consider to be the main stem of the Amazon. "
"Brazilians call this section the Solimões River above its confluence with the Rio Negro forming what Brazilians call the Amazon at the Meeting of Waters at Manaus, the largest city on the river.")

class LoRALinear(nn.Module):
    def __init__(self, base, r, alpha, gen):
        super().__init__()
        self.base=base
        for p in self.base.parameters(): p.requires_grad_(False)
        bound=1/math.sqrt(base.in_features)
        a=(torch.rand(r, base.in_features, generator=gen)*2-1)*bound
        self.lora_A=nn.Parameter(a.to(base.weight.dtype))
        self.lora_B=nn.Parameter(torch.zeros(base.out_features, r, dtype=base.weight.dtype))
        self.scaling=alpha/r
    def forward(self,x):
        return self.base(x)+(x@self.lora_A.T@self.lora_B.T)*self.scaling

def path(r): return sorted(glob.glob(f"{HUB}/models--{r.replace('/','--')}/snapshots/*"))[0]

def run(r):
    p=path(r); print('=====',r)
    tok=AutoTokenizer.from_pretrained(p)
    ids=tok(TEXT,return_tensors='pt').input_ids[:,:128]; print('tokens',ids.shape)
    t=time.time(); m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.float32); m.eval(); print('class',type(m).__name__,'load s',round(time.time()-t,1))
    n=sum(x.numel() for x in m.parameters()); print('params',n)
    c=m.config; print('experts',c.num_local_experts,'topk',c.num_experts_per_tok,'hid',c.hidden_size,'L',c.num_hidden_layers,'vocab',c.vocab_size,'heads',c.num_attention_heads,'kv',c.num_key_value_heads, 'inter', c.intermediate_size)
    l0=m.model.layers[0]; print(l0)
    print('named params layer0:',[(k,tuple(v.shape)) for k,v in l0.named_parameters()])
    with torch.no_grad():
        o=m(ids,output_router_logits=True); print('flag w/o labels: router_logits',type(o.router_logits),'aux',o.aux_loss)
        try: m(ids,labels=ids,output_router_logits=True)
        except Exception as e: print('flag WITH labels ->',type(e).__name__,e)
        cap={}
        hs=[ly.block_sparse_moe.router.register_forward_hook(lambda mod,i,out,k=k: cap.__setitem__(k,out)) for k,ly in enumerate(m.model.layers)]
        t=time.time(); o=m(ids,labels=ids); print('fp32 NLL',o.loss.item(),'fwd s',round(time.time()-t,2))
        idx,w,rl=cap[0]; print('hook out: top_k_index',idx.shape,idx.dtype,'top_k_weights',w.shape,'router_logits',rl.shape,rl.dtype,'n layers captured',len(cap)); print('token0 experts',idx[0].tolist(),'wsum',w[0].sum().item())
        print('mean distinct experts used/layer0', idx.unique().numel())
        for h in hs: h.remove()
        # fp16 / bf16 check
        for dt in (torch.float16, torch.bfloat16):
            try:
                m2=m.to(dt); o2=m2(ids,labels=ids); print(dt,'NLL',o2.loss.item(),'finite logits',bool(torch.isfinite(o2.logits).all()))
            except Exception as e: print(dt,'ERR',e)
        m=m.to(torch.float32)
    # LoRA
    for q in m.parameters(): q.requires_grad_(False)
    gen=torch.Generator().manual_seed(0)
    for layer in m.model.layers:
        layer.self_attn.q_proj=LoRALinear(layer.self_attn.q_proj,8,16,gen)
        layer.self_attn.v_proj=LoRALinear(layer.self_attn.v_proj,8,16,gen)
    tr=[q for q in m.parameters() if q.requires_grad]; print('trainable',len(tr),sum(q.numel() for q in tr))
    opt=torch.optim.AdamW(tr,lr=1e-3)
    batch=torch.cat([ids,ids.roll(7,1)],0); m.train()
    for s in range(3):
        t=time.time(); opt.zero_grad(); o=m(batch,labels=batch); o.loss.backward()
        gn=math.sqrt(sum((q.grad.float()**2).sum().item() for q in tr if q.grad is not None))
        nA=sum(1 for q in tr if q.grad is not None and q.grad.abs().sum()>0)
        opt.step(); print('step',s,'loss',o.loss.item(),'finite',math.isfinite(o.loss.item()),'gradnorm',gn,'params with nonzero grad',nA,'/',len(tr),'sec',round(time.time()-t,1))
    # router grads? (frozen)
    del m

for r in sys.argv[1:]: run(r)
