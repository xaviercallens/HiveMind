import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import shutil
from huggingface_hub import HfApi
api=HfApi()
for q in ["granite-3.1-1b-a400m","granite-3.1-3b-a800m","granite-3.0-1b-a400m","granite-3.0-3b-a800m","OLMoE","Qwen1.5-MoE","Qwen3-30B","deepseek-moe","moe-tiny","tiny-moe","Mixtral","phi-tiny-moe","PhiMoE","lfm2","Qwen3-MoE"]:
    try:
        ms=list(api.list_models(search=q,author=None,limit=15,sort="downloads"))
        print(q,[m.id for m in ms])
    except Exception as e: print(q,'ERR',e)
for r in ["ibm-granite/granite-3.1-1b-a400m-base","ibm-granite/granite-3.1-3b-a800m-base","ibm-granite/granite-3.0-1b-a400m-base","ibm-granite/granite-3.0-3b-a800m-base","allenai/OLMoE-1B-7B-0924","microsoft/Phi-tiny-MoE-instruct","LiquidAI/LFM2-8B-A1B"]:
    try:
        i=api.model_info(r,files_metadata=True)
        sz=sum(s.size or 0 for s in i.siblings if s.rfilename.endswith('.safetensors'))
        print(r,'gated',i.gated,'st GB',sz/1e9,'params',getattr(i.safetensors,'total',None) if i.safetensors else None,[s.rfilename for s in i.siblings if s.rfilename.endswith('.safetensors')])
    except Exception as e: print(r,'ERR',str(e)[:100])
print(shutil.disk_usage(os.path.realpath(os.path.expanduser('~/.cache/huggingface/hub'))))
