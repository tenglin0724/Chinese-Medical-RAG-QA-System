#!/bin/bash

# 配置参数
MODEL_PATH="/data1/tenglin/models/Qwen3-1.7B" 
HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096

# 启动vLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name Qwen3-1.7B \
    --trust-remote-code

