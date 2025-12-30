#!/usr/bin/env python3
"""评估脚本"""

import yaml
import json
from loguru import logger
from dotenv import load_dotenv

from src.rag_system import MedicalRAGSystem
from src.evaluation import RAGEvaluator


def main():
    # 配置日志
    logger.add(
        "logs/evaluation.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    # 加载环境变量
    load_dotenv()
    
    logger.info("="*60)
    logger.info("开始RAG系统评估")
    logger.info("="*60)
    
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化RAG系统
    logger.info("\n初始化RAG系统...")
    rag_system = MedicalRAGSystem(config)
    
    # 加载测试数据
    test_data_path = 'data/processed/test_qa_pairs.json'
    logger.info(f"\n加载测试数据: {test_data_path}")
    
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 限制测试样本数量
    test_sample_size = config['evaluation'].get('test_sample_size', 100)
    test_data = test_data[:test_sample_size]
    
    logger.info(f"测试样本数: {len(test_data)}")
    
    # 评估
    evaluator = RAGEvaluator(rag_system, config)
    results = evaluator.comprehensive_evaluation(test_data)
    
    # 保存结果
    results_path = 'data/evaluation_results.json'
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n评估结果已保存到: {results_path}")
    
    # 打印摘要
    logger.info("\n"+"="*60)
    logger.info("评估结果摘要")
    logger.info("="*60)
    
    for metric_name, metric_values in results.items():
        logger.info(f"\n{metric_name.upper()}:")
        for key, value in metric_values.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")


if __name__ == "__main__":
    main()
