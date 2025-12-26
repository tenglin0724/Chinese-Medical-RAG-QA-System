# 中文医疗 RAG 问答系统 (Chinese Medical RAG System)

本项目实现了一个基于检索增强生成 (RAG) 技术的中文医疗问答系统。该系统结合了专业的医疗知识库 (cMedQA2) 和大语言模型 (Qwen)，旨在提供准确、可靠的医疗健康咨询服务。

## 题目要求与实现详情

### 1. 数据准备
*   **数据来源**: 使用 [cMedQA2](https://github.com/zhangyics/CMedQA2) 中文医疗问答数据集，包含超过 10 万条真实的医患问答对。
*   **数据处理**:
    *   **清洗**: 去除无效字符和重复数据。
    *   **分块 (Chunking)**: 使用自定义文本切分器，将长文档切分为 1024 token 的片段 (overlap=64)，以保留上下文完整性。
    *   **向量化**: 使用 `bge-base-zh-v1.5` 模型将文本转换为向量表示。
    *   **向量数据库**: 构建基于 FAISS (IndexFlatL2) 的向量索引，支持高效的相似度检索。

### 2. 模型选择与微调
*   **基座模型**: 选用 **Qwen3-1.7B** (通义千问) 作为基础生成模型。该模型在中文语境下表现优异，且参数量适中，适合本地或低资源环境部署。
*   **推理服务**: 使用 **vLLM** 部署模型服务，提供高吞吐量的推理能力。
*   **微调 (可选)**: 项目架构支持接入经过 LoRA 或 SFT 微调后的模型。只需在 `config.yaml` 中修改 `model_name` 和 `base_url` 即可无缝切换至微调后的领域专用模型。

### 3. 核心功能
*   **多轮对话**: 系统维护对话历史 (History)，能够理解上下文，支持连续追问。
*   **长上下文支持**: 基于 Qwen 系列模型的长文本能力，结合 RAG 的分块检索，有效处理长篇医疗文档。
*   **引用来源显示**: 每次回答均会标注参考的文档来源 (Source Documents)，增强回答的可信度。
*   **拒答机制**: 通过 Prompt 工程 (System Prompt)，指示模型在检索内容不足以回答问题时明确告知用户，避免产生幻觉。

### 4. 迭代优化
*   **模块化设计**: 系统的各个组件（Embedding 模型、LLM、检索策略、Prompt）均通过 `config.yaml` 进行配置，无需修改代码即可替换。
*   **检索策略**: 支持调整 `top_k` (默认 5) 和相似度阈值 (`score_threshold`)，以平衡召回率和准确率。
*   **Prompt 优化**: 系统提示词 (System Prompt) 可独立更新，方便进行 Prompt Engineering 以提升特定场景下的表现。

### 5. 部署
*   **Web Demo**: 使用 **Gradio** 构建了交互式 Web 界面 (`app.py`)。
*   **功能**:
    *   支持实时问答。
    *   侧边栏显示参数配置 (Top-K, 阈值等)。
    *   展示检索到的参考文档片段。
    *   支持流式输出 (Streaming)。

### 6. 评估
*   **评估脚本**: 提供 `scripts/evaluate.py` 用于自动化评估。
*   **评估指标**: 结合 `src/evaluation.py`，支持计算 RAG 系统的关键指标，如：
    *   **检索准确率 (Hit Rate)**: 正确文档是否被检索到。
    *   **生成质量**: 比较模型回答与标准答案的相似度。

## 快速开始

### 环境配置
1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```
2. 启动 vLLM 服务 (参考 `docs/vllm_setup.md`):
   ```bash
   bash scripts/start_vllm.sh
   ```

### 数据处理
运行预处理脚本构建向量库:
```bash
python preprocess.py
```

### 启动应用
启动 Gradio 界面:
```bash
python app.py
```

### 评估
运行评估脚本:
```bash
python scripts/evaluate.py
```

## 项目结构
```
.
├── app.py                 # Gradio Web 应用入口
├── config.yaml            # 系统配置文件
├── preprocess.py          # 数据预处理与向量库构建
├── dataset/               # 数据集目录
├── docs/                  # 文档
├── scripts/               # 辅助脚本 (评估, 启动服务)
└── src/                   # 源代码
    ├── data_loader.py     # 数据加载
    ├── evaluation.py      # 评估模块
    ├── llm.py             # LLM 接口 (vLLM/Local)
    ├── rag_system.py      # RAG 核心逻辑
    ├── text_splitter.py   # 文本切分
    └── vector_store.py    # 向量数据库管理
```
