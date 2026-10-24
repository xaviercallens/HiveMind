import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from huggingface_hub import snapshot_download, hf_hub_download
for r in ["ibm-granite/granite-3.1-1b-a400m-base","ibm-granite/granite-3.1-3b-a800m-base"]:
    p=snapshot_download(r,allow_patterns=["*.safetensors","*.json","tokenizer*","*.txt","merges.txt","vocab.json"])
    print(p, os.listdir(p))
    c=json.load(open(p+'/config.json')); print(c)
for r in ["microsoft/Phi-tiny-MoE-instruct","allenai/OLMoE-1B-7B-0924"]:
    c=json.load(open(hf_hub_download(r,'config.json'))); print(r,c.get('architectures'),c.get('auto_map'),c.get('model_type'),c.get('num_local_experts'),c.get('num_experts'))
