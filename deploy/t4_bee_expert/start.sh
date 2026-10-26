#!/bin/bash
# Idempotent start of the Bee stars expert: llama-server (base + stars LoRA) -> shield :8789 -> nginx /bee/ -> producer agent.
set -u
B=~/bee-stars
L=/home/callensxavier_gmail_com/gwaya-lite/llama.cpp/build/bin/llama-server
BASE=/mnt/disks/disk-socrateai-local-1/leanmaster/ollama/blobs/sha256-60e05f2100071479f596b964f89f510f057ce397ea22f2833a0cfe029bfc2463
ADAPTER=/mnt/disks/disk-socrateai-local-1/hivemind-train/adapters/stars.gguf
GATEWAY=https://hivemind-openrouter-1003063861791.us-central1.run.app
cd $B
[ -s endpoint_token ] || python3 -c "import secrets;print(secrets.token_urlsafe(32))" > endpoint_token
chmod 600 endpoint_token
if ! curl -s -m2 localhost:8294/health | grep -q ok; then
  (setsid nohup $L -m $BASE --lora $ADAPTER --alias hivemind-bee-stars -c 4096 -ngl 99 -np 1 --host 127.0.0.1 --port 8294 > llama_8294.log 2>&1 < /dev/null &)
  for i in $(seq 1 120); do curl -s -m2 localhost:8294/health | grep -q ok && break; sleep 3; done
fi
if ! ss -ltn | grep -q ":8789 "; then
  (HIVEMIND_ENDPOINT_TOKEN=$(cat endpoint_token) setsid nohup python3 ~/hivemind-producer/hivemind_producer_shield.py --upstream http://127.0.0.1:8294 \
     --host 127.0.0.1 --port 8789 --max-concurrent 3 --allow-model hivemind-bee-stars > shield.log 2>&1 < /dev/null &)
  sleep 3
fi
if [ -s api_key ] && ! pgrep -f "hivemind_producer_agent.py .*--endpoint https://34-48-48-154.sslip.io/bee" >/dev/null; then
  (HIVEMIND_API_KEY=$(cat api_key) HIVEMIND_ENDPOINT_TOKEN=$(cat endpoint_token) setsid nohup python3 ~/hivemind-producer/hivemind_producer_agent.py \
     --gateway $GATEWAY --endpoint https://34-48-48-154.sslip.io/bee --model bee-stars=hivemind-bee-stars \
     --cost-per-1k 0.4 --capacity-1k 30 --execution-class t4 > agent.log 2>&1 < /dev/null &)
  sleep 15
fi
# Keep the 7B coder resident: Ollama unloads it after OLLAMA_KEEP_ALIVE (10m) idle, and the reload made a user's first
# coding request take 52-120 s (one 504 measured 2026-10-25). keep_alive=-1 on a load request pins it without changing the
# shared service config; Ollama still evicts it when another model is needed (OLLAMA_MAX_LOADED_MODELS=1).
curl -s -m 300 localhost:11434/api/generate -d '{"model":"qwen2.5-coder:7b-instruct","prompt":"","keep_alive":-1}' > /dev/null
llama_health=$(curl -s -m2 localhost:8294/health)
shield_listening=$(ss -ltn | grep -c ':8789 ')
agent_count=$(pgrep -fc 'hivemind_producer_agent[.]py .*sslip[.]io/bee')
echo "llama: ${llama_health}; shield listening: ${shield_listening}; agent processes: ${agent_count}"
tail -n 2 agent.log 2>/dev/null
