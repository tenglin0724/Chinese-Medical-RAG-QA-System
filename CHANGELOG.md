# 项目优化更新日志

## 2024-12-22 更新

### ✨ 新增特性

#### 1. 多种LLM支持

项目现在支持三种LLM部署方式：

**通义千问（在线API）** - 适合快速开发和测试
```yaml
llm:
  provider: "qwen"
  model_name: "qwen-plus"
  api_key_env: "DASHSCOPE_API_KEY"
```

**本地模型（Transformers）** - 适合离线环境和隐私保护
```yaml
llm:
  provider: "local"
  model_path: "/path/to/your/model"
  device: "cuda"
```

**vLLM服务器** - 适合生产环境和高并发场景
```yaml
llm:
  provider: "vllm"
  base_url: "http://localhost:8000/v1"
  model_name: "your-model-name"
```

相关文件：
- `src/llm.py` - 新增LocalLLM和VLLMServer类
- `scripts/test_local_llm.py` - 本地LLM测试脚本
- `scripts/start_vllm.sh` - vLLM启动脚本
- `docs/vllm_setup.md` - vLLM详细部署指南

#### 2. 交互式调试Notebook

新增 `RAG_Debug.ipynb`，提供完整的调试工作流：

- ✅ 数据加载和探索
- ✅ 文本分块测试
- ✅ 向量检索测试
- ✅ LLM生成测试
- ✅ 完整RAG流程
- ✅ 多轮对话测试
- ✅ 性能分析工具
- ✅ 调试辅助函数

使用方法：
```bash
jupyter notebook RAG_Debug.ipynb
```

#### 3. 修正测试评估数据处理

**问题修复**：
- cMedQA2数据集格式为CSV（正负样本对），而非简单的标签文本
- 训练/测试集文件需要解压
- 评估方法需要使用相关性标签

**改进内容**：

1. **数据加载器** (`src/data_loader.py`)
   - ✅ 自动解压候选对文件
   - ✅ 正确解析CSV格式（question_id, pos_ans_id, neg_ans_id）
   - ✅ 构建正负样本对
   - ✅ 支持去重和标签保留

2. **评估模块** (`src/evaluation.py`)
   - ✅ 新增基于相关性标签的检索评估
   - ✅ 计算MRR（Mean Reciprocal Rank）
   - ✅ 计算Precision@5和Recall@5
   - ✅ 支持按问题分组评估

3. **测试数据格式**
   ```python
   {
       'question_id': '24731702',
       'answer_id': '11064',
       'question': '感冒了怎么办？',
       'answer': '多喝水，注意休息...',
       'label': 1,  # 1=相关，0=不相关
       'combined_text': '问题：...答案：...'
   }
   ```

### 🔧 配置更新

`config.yaml` 新增配置项：

```yaml
llm:
  provider: "qwen"  # 新增: local, vllm选项
  
  # 本地模型配置
  # model_path: "/path/to/model"
  # device: "cuda"
  
  # vLLM服务器配置
  # base_url: "http://localhost:8000/v1"
```

### 📦 依赖更新

`requirements.txt` 新增：
- `accelerate>=0.24.0` - 加速模型加载
- `vllm>=0.3.0` - vLLM服务器支持（可选）
- `jupyter>=1.0.0` - Jupyter Notebook支持
- `notebook>=7.0.0`
- `ipykernel>=6.25.0`

### 📚 文档更新

新增文档：
- `docs/vllm_setup.md` - vLLM详细部署指南
  - 安装步骤
  - 启动配置
  - 性能优化
  - 常见问题
  - 支持的模型列表

README.md 更新：
- ✅ 本地LLM使用说明
- ✅ vLLM部署指南
- ✅ Notebook调试说明
- ✅ 更新的项目结构

### 🎯 使用示例

#### 使用本地模型

```python
# 方法1: 直接修改配置
config['llm']['provider'] = 'local'
config['llm']['model_path'] = '/path/to/model'
config['llm']['device'] = 'cuda'

# 方法2: 运行测试脚本
python scripts/test_local_llm.py
```

#### 使用vLLM服务器

```bash
# 1. 启动vLLM
./scripts/start_vllm.sh

# 2. 修改配置
# config.yaml中设置provider为vllm

# 3. 运行系统
python app.py
```

#### 使用Notebook调试

```bash
# 启动Notebook
jupyter notebook RAG_Debug.ipynb

# 或在VSCode中打开
# 支持交互式调试和可视化
```

### 🚀 性能优化

**本地LLM优化建议**：
- 使用CUDA加速
- 启用FP16/BF16精度
- 使用Flash Attention
- 批处理推理

**vLLM优化建议**：
- 调整GPU显存利用率
- 使用量化（AWQ/GPTQ）
- 多GPU并行推理
- 动态批处理

### 📊 评估改进

新的评估指标：
- **MRR (Mean Reciprocal Rank)** - 首个相关答案的排名倒数
- **Precision@5** - 前5个结果中相关答案的比例
- **Recall@5** - 前5个结果覆盖的相关答案比例
- **按问题分组评估** - 更准确的检索性能评估

### 🐛 Bug修复

1. ✅ 修复cMedQA2数据加载格式问题
2. ✅ 修复候选对文件解析错误
3. ✅ 修复测试集标签处理
4. ✅ 修复评估时的相关性判断

### 📝 待办事项

- [ ] 添加更多评估指标（ROUGE、BLEU）
- [ ] 支持模型微调（LoRA/SFT）
- [ ] 添加缓存机制提升性能
- [ ] 支持多模态输入
- [ ] 添加更多示例和教程

### 🙏 使用建议

1. **首次使用**：推荐使用通义千问API快速验证
2. **本地开发**：使用Notebook进行调试和实验
3. **生产部署**：使用vLLM提供高性能服务
4. **隐私保护**：使用本地Transformers模型

### 📞 问题反馈

如遇到问题，请检查：
1. 配置文件是否正确
2. 依赖是否完整安装
3. API密钥或模型路径是否正确
4. 日志文件中的错误信息

---

**版本**: v1.1.0  
**更新日期**: 2024-12-22  
**维护者**: tenglin
