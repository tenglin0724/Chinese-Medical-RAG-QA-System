# 中文医疗RAG问答系统

基于大语言模型和检索增强生成(RAG)技术的中文医疗问答系统。

## 📋 项目说明

本项目是一个课程项目，实现了一个专业的中文医疗领域问答系统，具有以下特点：

- **数据集**: 使用cMedQA2中文医疗问答数据集(10万+问答对)
- **RAG架构**: 基于LangChain构建，支持检索增强生成
- **向量检索**: 使用FAISS向量数据库和BGE中文向量模型
- **大语言模型**: 集成通义千问Qwen-Plus
- **Web界面**: Gradio交互式界面
- **多轮对话**: 支持上下文记忆的多轮对话
- **来源追溯**: 显示答案的知识来源

## 🏗️ 项目结构

```
my_chinese_medical_rag/
├── README.md                   # 项目说明
├── requirements.txt            # Python依赖
├── config.yaml                 # 配置文件
├── .env.example               # 环境变量示例
├── preprocess.py              # 数据预处理脚本
├── app.py                     # Web应用主程序
├── RAG_Debug.ipynb            # 调试Notebook
├── src/                       # 源代码
│   ├── __init__.py
│   ├── data_loader.py         # 数据加载
│   ├── text_splitter.py       # 文本分块
│   ├── vector_store.py        # 向量数据库
│   ├── llm.py                 # LLM接口（支持Qwen/本地/vLLM）
│   ├── rag_system.py          # RAG核心逻辑
│   └── evaluation.py          # 评估模块
├── scripts/                   # 脚本
│   ├── evaluate.py            # 评估脚本
│   ├── test_rag.py            # RAG测试脚本
│   ├── test_local_llm.py      # 本地LLM测试脚本
│   └── start_vllm.sh          # vLLM启动脚本
├── docs/                      # 文档
│   └── vllm_setup.md          # vLLM部署指南
├── dataset/                   # 数据集目录
│   └── cMedQA2/              # cMedQA2数据集
├── data/                      # 处理后的数据
│   ├── processed/             # 预处理数据
│   └── vector_store/          # 向量数据库
└── logs/                      # 日志文件
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（推荐）
conda create -n medical_rag python=3.10
conda activate medical_rag

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置设置

复制环境变量示例文件并配置：

```bash
cp .env.example .env
```

编辑 `.env` 文件，设置通义千问API Key：

```
DASHSCOPE_API_KEY=your_api_key_here
```

> 💡 获取API Key: 访问 [阿里云DashScope](https://dashscope.aliyun.com/) 注册并获取API密钥

### 3. 数据预处理

运行数据预处理脚本，构建知识库和向量数据库：

```bash
python preprocess.py
```

这一步会：
- 加载cMedQA2数据集
- 构建知识库（QA对+答案）
- 进行文本分块
- 生成向量embedding
- 构建FAISS向量索引

⏱️ **预计耗时**: 30-60分钟（取决于硬件配置）

### 4. 启动Web应用

```bash
python app.py
```

访问 `http://localhost:7860` 使用Web界面进行问答。

## 📊 系统评估

运行评估脚本来测试系统性能：

```bash
python scripts/evaluate.py
```

评估指标包括：
- 准确率 (Accuracy)
- 检索F1分数
- 幻觉率 (Hallucination Rate)

## 🧪 快速测试

运行测试脚本快速验证系统功能：

```bash
# 测试RAG系统（使用配置的LLM）
python scripts/test_rag.py

# 测试本地LLM
python scripts/test_local_llm.py
```

## 📓 Jupyter调试

使用交互式Notebook进行调试：

```bash
jupyter notebook RAG_Debug.ipynb
```

Notebook包含：
- 数据加载测试
- 文本分块测试  
- 向量检索测试
- LLM生成测试
- 完整RAG流程测试
- 性能分析工具

## ⚙️ 配置说明

主要配置项在 `config.yaml` 中：

### 向量模型配置

```yaml
embedding:
  model_name: "BAAI/bge-large-zh-v1.5"
  model_path: "/data1/tenglin/models/bge-base-zh-v1.5"  # 修改为你的模型路径
  device: "cuda"  # 或 "cpu"
```

### LLM配置

支持三种LLM提供商：

**1. 通义千问（在线API）**
```yaml
llm:
  provider: "qwen"
  model_name: "qwen-plus"
  api_key_env: "DASHSCOPE_API_KEY"
  temperature: 0.7
  max_tokens: 2048
```

**2. 本地模型（Transformers）**
```yaml
llm:
  provider: "local"
  model_path: "/path/to/your/model"  # 例如: Qwen2.5-7B-Instruct
  device: "cuda"
  temperature: 0.7
  max_tokens: 2048
```

**3. vLLM服务器**
```yaml
llm:
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  model_name: "your-model-name"
  temperature: 0.7
  max_tokens: 2048
```

### 检索配置

```yaml
vector_store:
  type: "faiss"
  top_k: 5  # 检索文档数量
  score_threshold: 0.7  # 相似度阈值
```

## 🖥️ 使用本地LLM

### 方法1: 使用Transformers直接加载

```python
# 修改config.yaml
llm:
  provider: "local"
  model_path: "/data1/tenglin/models/Qwen3-4B-Instruct-2507"
  device: "cuda"
```

运行测试：
```bash
python scripts/test_local_llm.py
```

### 方法2: 使用vLLM服务器（推荐用于生产环境）

1. 启动vLLM服务器：
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/your/model \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto
```

2. 修改config.yaml：
```yaml
llm:
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  model_name: "your-model-name"
```

3. 运行系统：
```bash
python app.py
```

## 🎯 核心功能

### 1. 智能检索

- 基于BGE中文向量模型的语义检索
- FAISS高效向量相似度搜索
- 可配置的检索数量和阈值

### 2. 多轮对话

- 支持上下文记忆
- 保留最近3轮对话历史
- 可选择启用/禁用对话历史

### 3. 来源追溯

- 显示答案的知识来源
- 展示相似度分数
- 支持来源内容预览

### 4. 安全回答

- 识别无法回答的问题
- 明确说明信息不足
- 提醒用户咨询专业医生

## 📈 性能优化建议

### 提升检索效果

1. **调整分块策略**：修改 `config.yaml` 中的 `chunking` 参数
2. **更换向量模型**：尝试不同的中文向量模型
3. **优化检索参数**：调整 `top_k` 和 `score_threshold`

### 提升生成质量

1. **优化Prompt**：修改 `config.yaml` 中的 `prompt` 模板
2. **调整LLM参数**：修改 `temperature` 和 `max_tokens`
3. **使用更强模型**：切换到 `qwen-max` 等更强大的模型

### 提升系统性能

1. **使用GPU加速**：确保向量模型运行在GPU上
2. **批量处理**：增加 `batch_size` 参数
3. **缓存机制**：对常见问题添加缓存

## 📝 使用示例

### Python API

```python
from src.rag_system import MedicalRAGSystem
import yaml

# 加载配置
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 初始化系统
rag_system = MedicalRAGSystem(config)

# 查询
result = rag_system.query("感冒了怎么办？")
print(f"回答: {result['answer']}")
print(f"来源数: {result['num_sources']}")

# 流式查询
answer_gen, sources = rag_system.query_stream("发烧吃什么药？")
for chunk in answer_gen:
    print(chunk, end='', flush=True)
```

## 🔧 故障排除

### 1. 向量模型加载失败

**问题**: 找不到模型文件

**解决**: 
- 检查 `config.yaml` 中的 `model_path` 是否正确
- 确保已下载对应的向量模型
- 可以使用 `model_name` 自动从HuggingFace下载

### 2. API调用失败

**问题**: `API Key验证失败`

**解决**:
- 检查 `.env` 文件中的 `DASHSCOPE_API_KEY` 是否正确
- 确保API Key有效且有足够余额
- 检查网络连接

### 3. 内存不足

**问题**: 构建向量数据库时内存溢出

**解决**:
- 减小 `batch_size`
- 使用CPU模式（设置 `device: "cpu"`）
- 减少知识库大小

## 📚 参考资料

- [cMedQA2数据集](https://github.com/zhangsheng93/cMedQA2)
- [LangChain文档](https://python.langchain.com/)
- [通义千问API](https://help.aliyun.com/zh/dashscope/)
- [BGE向量模型](https://huggingface.co/BAAI/bge-large-zh-v1.5)

## 📄 课程项目要求对照

✅ **数据准备**: 使用cMedQA2数据集(10万+QA对)，完成清洗、分块、向量化  
✅ **模型选择**: 使用Qwen-2.5作为LLM，BGE作为向量模型  
✅ **核心功能**: 支持多轮对话、长上下文(32k tokens)、来源显示、不确定回答拒绝  
✅ **迭代优化**: 支持更新embedding模型、检索策略、prompt配置  
✅ **部署**: Gradio Web界面，支持实时交互  
✅ **评估**: 实现准确率、F1、幻觉率等评估指标  

## ⚠️ 免责声明

本系统仅供学习和研究使用，不能替代专业医疗建议。如有健康问题，请咨询专业医生。

---

**注意**: 确保在使用前正确配置所有必需的API密钥和模型路径。

