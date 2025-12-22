#!/usr/bin/env python3
"""数据预处理脚本 - 构建知识库和向量数据库"""

import yaml
import os
from loguru import logger

from src.data_loader import CMedQA2Loader
from src.text_splitter import ChineseTextSplitter
from src.vector_store import VectorStore


def main():
    """主函数"""
    # 配置日志
    logger.add(
        "logs/preprocessing.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    logger.info("="*60)
    logger.info("开始数据预处理流程")
    logger.info("="*60)
    
    # 1. 加载配置
    logger.info("\n步骤 1/5: 加载配置文件")
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    logger.info("配置加载完成")
    
    # 2. 加载和处理数据
    logger.info("\n步骤 2/5: 加载cMedQA2数据集")
    loader = CMedQA2Loader(config)
    
    # 构建知识库
    knowledge_base = loader.build_knowledge_base()
    loader.save_processed_data(knowledge_base, 'knowledge_base.json')
    
    # 构建测试集
    test_qa_pairs = loader.build_qa_pairs('test')
    loader.save_processed_data(test_qa_pairs, 'test_qa_pairs.json')
    
    # 构建开发集
    dev_qa_pairs = loader.build_qa_pairs('dev')
    loader.save_processed_data(dev_qa_pairs, 'dev_qa_pairs.json')
    
    logger.info("数据加载和保存完成")
    
    # 3. 文本分块
    logger.info("\n步骤 3/5: 文本分块")
    splitter = ChineseTextSplitter(config)
    chunks = splitter.split_documents(knowledge_base)
    
    # 保存分块结果
    import json
    output_dir = config['data']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    chunks_path = os.path.join(output_dir, 'chunks.json')
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"分块数据已保存到: {chunks_path}")
    
    # 4. 构建向量数据库
    logger.info("\n步骤 4/5: 构建向量数据库")
    logger.info("注意：这一步可能需要较长时间，具体取决于数据量和硬件性能")
    
    vector_store = VectorStore(config)
    vector_store.build_index(chunks)
    
    # 5. 保存向量数据库
    logger.info("\n步骤 5/5: 保存向量数据库")
    save_dir = config['data']['vector_store_dir']
    vector_store.save(save_dir)
    
    logger.info("\n"+"="*60)
    logger.info("数据预处理完成！")
    logger.info("="*60)
    logger.info(f"\n已处理:")
    logger.info(f"  - 知识库条目: {len(knowledge_base)}")
    logger.info(f"  - 文本块数量: {len(chunks)}")
    logger.info(f"  - 测试集大小: {len(test_qa_pairs)}")
    logger.info(f"  - 开发集大小: {len(dev_qa_pairs)}")
    logger.info(f"\n向量数据库已保存到: {save_dir}")
    logger.info("\n下一步:")
    logger.info("  1. 配置 .env 文件，设置 DASHSCOPE_API_KEY")
    logger.info("  2. 运行 python app.py 启动Web界面")
    logger.info("  3. 或运行 python scripts/evaluate.py 进行评估")


if __name__ == "__main__":
    main()
