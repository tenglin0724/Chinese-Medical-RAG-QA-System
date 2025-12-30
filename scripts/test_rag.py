#!/usr/bin/env python3

import yaml
from loguru import logger
from dotenv import load_dotenv

from src.rag_system import MedicalRAGSystem


def main():
    # 加载环境变量
    load_dotenv()
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化RAG系统
    logger.info("初始化RAG系统...")
    rag_system = MedicalRAGSystem(config)
    
    # 测试问题
    test_questions = [
        "感冒了怎么办？",
        "发烧需要吃什么药？",
        "咳嗽有痰怎么治疗？",
        "头痛是什么原因引起的？",
        "高血压患者饮食需要注意什么？"
    ]
    
    logger.info("\n开始测试RAG系统...")
    logger.info("="*80)
    
    for i, question in enumerate(test_questions, 1):
        logger.info(f"\n测试 {i}/{len(test_questions)}")
        logger.info(f"问题: {question}")
        logger.info("-"*80)
        
        # 流式查询
        answer_generator, sources = rag_system.query_stream(
            question=question,
            use_history=True
        )
        
        # 打印答案
        logger.info("回答: ", end='')
        full_answer = ""
        for chunk in answer_generator:
            print(chunk, end='', flush=True)
            full_answer = chunk
        print()
        
        # 打印来源
        logger.info(f"\n检索到 {len(sources)} 条相关来源:")
        for j, source in enumerate(sources[:3], 1):
            logger.info(f"\n  来源 {j} (相似度: {source['score']:.3f}):")
            content_preview = source['content'][:150].replace('\n', ' ')
            logger.info(f"  {content_preview}...")
        
        logger.info("\n" + "="*80)
    
    logger.info("\n测试完成！")


if __name__ == "__main__":
    main()
