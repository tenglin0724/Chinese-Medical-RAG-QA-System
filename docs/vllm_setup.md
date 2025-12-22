# 如何使用vLLM部署本地LLM

## 1. 安装vLLM

```bash
pip install vllm
```

## 2. 启动vLLM服务器

### 方式1: 使用脚本启动

```bash
chmod +x scripts/start_vllm.sh
./scripts/start_vllm.sh
```

### 方式2: 手动启动

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --gpu-memory-utilization 0.9
```

## 3. 测试vLLM服务器

```bash
curl http://localhost:8000/v1/models
```

或使用Python测试：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="your-model-name",
    messages=[
        {"role": "user", "content": "你好"}
    ]
)

print(response.choices[0].message.content)
```

## 4. 配置RAG系统使用vLLM

修改 `config.yaml`:

```yaml
llm:
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  model_name: "your-model-name"  # vLLM中加载的模型名称
  temperature: 0.7
  max_tokens: 2048
```

## 5. 启动RAG系统

```bash
python app.py
```

## 性能优化建议

### GPU显存优化

- 调整 `--gpu-memory-utilization` (0.8-0.95)
- 使用量化: `--quantization awq` 或 `--quantization gptq`
- 减小 `--max-model-len`

### 吞吐量优化

- 增加 `--max-num-seqs` (批处理大小)
- 使用多GPU: `--tensor-parallel-size 2`
- 启用连续批处理（默认开启）

### 延迟优化

- 减小 `--max-num-seqs`
- 使用更少的GPU
- 调整 `--swap-space` (CPU-GPU交换空间)

## 常见问题

### Q: 显存不足
A: 
- 使用较小的模型
- 启用量化
- 减小 `--gpu-memory-utilization`
- 减小 `--max-model-len`

### Q: 推理速度慢
A:
- 使用多GPU并行
- 启用FlashAttention (自动)
- 使用较小的模型
- 增加批处理大小

### Q: 连接超时
A:
- 检查防火墙设置
- 确认服务器已启动
- 检查端口是否被占用

## 支持的模型

vLLM支持大多数HuggingFace格式的模型：

- Qwen / Qwen2 / Qwen2.5
- LLaMA / LLaMA-2 / LLaMA-3
- ChatGLM
- Baichuan
- InternLM
- 等等

详细列表: https://docs.vllm.ai/en/latest/models/supported_models.html
