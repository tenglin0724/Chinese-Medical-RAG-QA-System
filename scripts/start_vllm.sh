#!/bin/bash
# vLLM服务器启动脚本

# 配置参数
MODEL_PATH="/data1/tenglin/models/Qwen3-4B-Instruct-2507"  # 修改为你的模型路径
HOST="0.0.0.0"
PORT=8000
GPU_MEMORY_UTILIZATION=0.9
MAX_MODEL_LEN=4096

echo "启动vLLM服务器..."
echo "模型路径: $MODEL_PATH"
echo "监听地址: http://$HOST:$PORT"
echo ""

# 启动vLLM服务器
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype auto \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name Qwen3-4B-Instruct \
    --trust-remote-code

# 可选参数说明：
# --tensor-parallel-size 2  # 使用2个GPU进行张量并行
# --dtype half  # 使用半精度
# --quantization awq  # 使用AWQ量化
# --max-num-seqs 256  # 最大批处理大小
