import os, sys, glob, time, traceback
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(2)
sys.path.insert(0,'src')
from transformers import AutoModelForCausalLM, AutoTokenizer
from hivemind.core.btm_upcycle import upcycle_branches_to_sparse_moe
p=glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*'))[0]
tok=AutoTokenizer.from_pretrained(p)
TEXT=("The Amazon River in South America is the largest river by discharge volume of water in the world, and the disputed longest river system in the world in comparison to the Nile. "
"The headwaters of the Apurimac River on Nevado Mismi had been considered for nearly a century as the Amazon basin's most distant source, until a 2014 study found it to be the headwaters of the Mantaro River on the Cordillera Rumi Cruz in Peru. "
"The Mantaro and Apurimac rivers join, and with other tributaries form the Ucayali River, which in turn meets the Marañón River upstream of Iquitos, Peru, forming what countries other than Brazil consider to be the main stem of the Amazon. "
"Brazilians call this section the Solimões River above its confluence with the Rio Negro forming what Brazilians call the Amazon at the Meeting of Waters at Manaus, the largest city on the river.")
ids=tok(TEXT,return_tensors='pt').input_ids[:,:128]
def dense():
    m=AutoModelForCausalLM.from_pretrained(p,dtype=torch.float32).eval(); return m
m=dense(); print(type(m).__name__, 'params',sum(x.numel() for x in m.parameters()), 'cfg tie',m.config.tie_word_embeddings)
with torch.no_grad(): print('dense NLL',m(ids,labels=ids).loss.item())
doms=['a','b','c','d']
for inplace in (False,True):
    print('--- in_place',inplace)
    try:
        m=dense()
        import torch.nn as nn
        class _Inner(nn.Module):
            def __init__(self,mm): super().__init__(); self.model=mm
        class Shim(nn.Module):  # mimics PeftModel.base_model.model layout only (peft not installed)
            def __init__(self,mm): super().__init__(); self.base_model=_Inner(mm)
        H,I,L=m.config.hidden_size,m.config.intermediate_size,m.config.num_hidden_layers
        g=torch.Generator().manual_seed(1)
        dims={'self_attn.q_proj':(H,H),'self_attn.k_proj':(H,m.config.num_key_value_heads*(H//m.config.num_attention_heads)),'self_attn.v_proj':(H,m.config.num_key_value_heads*(H//m.config.num_attention_heads)),'self_attn.o_proj':(H,H),'mlp.gate_proj':(H,I),'mlp.up_proj':(H,I),'mlp.down_proj':(I,H)}
        states={}
        for d in doms:
            st={}
            for li in range(L):
                for nm,(fi,fo) in dims.items():
                    pre=f"base_model.model.model.layers.{li}.{nm}"
                    st[pre+".lora_A.default.weight"]=torch.randn(8,fi,generator=g)*0.02
                    st[pre+".lora_B.default.weight"]=torch.randn(fo,8,generator=g)*float(os.environ.get("SC","0"))
            states[d]=st
        m=Shim(m)
        t=time.time()
        moe,rep=upcycle_branches_to_sparse_moe(m,states,lora_rank_r=8,lora_alpha=16,top_k=2,in_place=inplace,target_dtype=torch.float32,target_device=torch.device('cpu'))
        print('built s',round(time.time()-t,1),rep)
        n=sum(x.numel() for x in moe.parameters()); print('upcycled params (unique)',n)
        moe.eval()
        with torch.no_grad(): o=moe(ids,labels=ids); print('upcycled NLL (incl aux coef*aux)',o.loss.item())
        with torch.no_grad(): o2=moe.backbone(input_ids=ids,labels=ids); print('backbone NLL',o2.loss.item())
        print('util',moe.get_expert_utilization())
        blk=moe.get_moe_blocks()[0]; print('block0 attrs: router.wg',tuple(blk.router.wg.weight.shape),'experts',len(blk.experts))
    except Exception:
        traceback.print_exc()
