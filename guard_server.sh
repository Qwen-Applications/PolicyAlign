vllm serve "Qwen3Guard-Gen-8B" \
  --host 0.0.0.0 \
  --port 8001 \
  --dtype auto \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.90 \
  --enforce-eager