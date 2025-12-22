#!/usr/bin/env python3
"""本地LLM使用示例"""

import yaml
from loguru import logger
from dotenv import load_dotenv

from src.rag_system import MedicalRAGSystem


def main():
    """主函数"""
    # 加载环境变量
    load_dotenv()
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 修改配置使用本地LLM
    # 方式1: 使用transformers加载本地模型
    config['llm']['provider'] = 'local'
    config['llm']['model_path'] = '/data1/tenglin/models/Qwen3-4B-Instruct-2507'  # 修改为你的模型路径
    config['llm']['device'] = 'cuda'  # 或 'cpu'
    
    # 方式2: 使用vLLM服务器（需要先启动vLLM服务）
    # config['llm']['provider'] = 'vllm'
    # config['llm']['base_url'] = 'http://localhost:8000/v1'
    # config['llm']['model_name'] = 'Qwen2.5-7B-Instruct'
    
    logger.info(f"使用LLM提供商: {config['llm']['provider']}")
    
    # 初始化RAG系统
    logger.info("初始化RAG系统...")
    rag_system = MedicalRAGSystem(config)
    
    # 测试问题
    test_questions = [
        "感冒了怎么办？",
        "发烧需要吃什么药？",
    ]
    
    logger.info("\n开始测试RAG系统...")
    logger.info("="*80)
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n测试 {i}/{len(test_questions)}")
        logger.info(f"问题: {question}")
        logger.info("-"*80)
        
        # 查询
        result = rag_system.query(question, use_history=False)
        
        # 打印答案
        logger.info(f"回答: {result['answer']}")
        
        # 打印来源
        logger.info(f"\n检索到 {len(result['sources'])} 条相关来源:")
        for j, source in enumerate(result['sources'][:2], 1):
            logger.info(f"\n  来源 {j} (相似度: {source['score']:.3f}):")
            content_preview = source['content'][:100].replace('\n', ' ')
            logger.info(f"  {content_preview}...")
        
        logger.info("\n" + "="*80)
    
    logger.info("\n测试完成！")


if __name__ == "__main__":
    main()
