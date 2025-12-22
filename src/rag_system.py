"""RAG核心逻辑模块"""

from typing import List, Dict, Optional, Generator, Tuple
from loguru import logger

from .vector_store import VectorStore, Retriever
from .llm import BaseLLM, LLMFactory


class MedicalRAGSystem:
    """医疗RAG问答系统"""
    
    def __init__(self, config: dict, vector_store: VectorStore = None, llm: BaseLLM = None):
        """
        初始化RAG系统
        
        Args:
            config: 配置字典
            vector_store: 向量数据库（可选）
            llm: 语言模型（可选）
        """
        self.config = config
        
        # 初始化向量数据库
        if vector_store is None:
            self.vector_store = VectorStore(config)
            # 尝试加载已有索引
            vector_store_dir = config['data']['vector_store_dir']
            try:
                self.vector_store.load(vector_store_dir)
                logger.info("成功加载已有向量数据库")
            except Exception as e:
                logger.warning(f"未找到已有向量数据库: {e}")
        else:
            self.vector_store = vector_store
        
        # 初始化检索器
        self.retriever = Retriever(self.vector_store, config)
        
        # 初始化LLM
        if llm is None:
            self.llm = LLMFactory.create_llm(config)
        else:
            self.llm = llm
        
        # 加载Prompt模板
        self.system_prompt = config['prompt']['system_prompt']
        self.query_prompt_template = config['prompt']['query_prompt']
        
        # 对话历史
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info("医疗RAG系统初始化完成")
    
    def query(
        self, 
        question: str, 
        top_k: int = None,
        use_history: bool = True
    ) -> Dict[str, any]:
        """
        查询问答
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_history: 是否使用对话历史
            
        Returns:
            包含回答、来源等信息的字典
        """
        logger.info(f"收到查询: {question}")
        
        # 1. 检索相关文档
        retrieved_docs, contexts = self.retriever.retrieve(question, top_k)
        logger.info(f"检索到 {len(retrieved_docs)} 条相关文档")
        
        # 2. 格式化上下文
        formatted_context = self.retriever.format_contexts(contexts)
        
        # 3. 构建Prompt
        # 如果使用对话历史，加入最近的几轮对话
        conversation_context = ""
        if use_history and len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-3:]  # 最近3轮
            history_texts = []
            for h in recent_history:
                history_texts.append(f"用户: {h['question']}\n助手: {h['answer']}")
            conversation_context = "\n\n".join(history_texts) + "\n\n"
        
        query_prompt = self.query_prompt_template.format(
            context=formatted_context,
            question=question
        )
        
        if conversation_context:
            query_prompt = f"对话历史:\n{conversation_context}当前问题:\n{query_prompt}"
        
        # 4. 生成回答
        answer = self.llm.generate(
            prompt=query_prompt,
            system_prompt=self.system_prompt
        )
        
        # 5. 保存对话历史
        self.conversation_history.append({
            'question': question,
            'answer': answer,
            'sources': retrieved_docs
        })
        
        # 6. 返回结果
        result = {
            'question': question,
            'answer': answer,
            'sources': retrieved_docs,
            'num_sources': len(retrieved_docs)
        }
        
        logger.info("查询完成")
        return result
    
    def query_stream(
        self, 
        question: str, 
        top_k: int = None,
        use_history: bool = True
    ) -> Tuple[Generator[str, None, None], List[Dict]]:
        """
        流式查询问答
        
        Args:
            question: 用户问题
            top_k: 检索文档数量
            use_history: 是否使用对话历史
            
        Returns:
            (生成器, 检索文档列表)
        """
        logger.info(f"收到流式查询: {question}")
        
        # 1. 检索相关文档
        retrieved_docs, contexts = self.retriever.retrieve(question, top_k)
        logger.info(f"检索到 {len(retrieved_docs)} 条相关文档")
        
        # 2. 格式化上下文
        formatted_context = self.retriever.format_contexts(contexts)
        
        # 3. 构建Prompt
        conversation_context = ""
        if use_history and len(self.conversation_history) > 0:
            recent_history = self.conversation_history[-3:]
            history_texts = []
            for h in recent_history:
                history_texts.append(f"用户: {h['question']}\n助手: {h['answer']}")
            conversation_context = "\n\n".join(history_texts) + "\n\n"
        
        query_prompt = self.query_prompt_template.format(
            context=formatted_context,
            question=question
        )
        
        if conversation_context:
            query_prompt = f"对话历史:\n{conversation_context}当前问题:\n{query_prompt}"
        
        # 4. 流式生成回答
        def generate_with_history():
            full_answer = ""
            for chunk in self.llm.generate_stream(
                prompt=query_prompt,
                system_prompt=self.system_prompt
            ):
                full_answer += chunk
                yield chunk
            
            # 生成完成后保存历史
            self.conversation_history.append({
                'question': question,
                'answer': full_answer,
                'sources': retrieved_docs
            })
        
        return generate_with_history(), retrieved_docs
    
    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        logger.info("对话历史已清空")
    
    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.conversation_history


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 加载配置
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化RAG系统
    rag_system = MedicalRAGSystem(config)
    
    # 测试查询
    test_questions = [
        "感冒了怎么办？",
        "发烧需要吃什么药？",
        "咳嗽有痰怎么治疗？"
    ]
    
    for question in test_questions:
        logger.info(f"\n{'='*60}")
        result = rag_system.query(question)
        
        logger.info(f"问题: {result['question']}")
        logger.info(f"回答: {result['answer']}")
        logger.info(f"引用来源数: {result['num_sources']}")
        
        if result['sources']:
            logger.info("\n相关来源:")
            for i, source in enumerate(result['sources'][:3], 1):
                logger.info(f"\n[{i}] (相似度: {source['score']:.3f})")
                logger.info(source['content'][:200] + "...")
    
    logger.info("\nRAG系统测试完成！")
