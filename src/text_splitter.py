"""文本分块模块"""

from typing import List, Dict
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
from loguru import logger


class ChineseTextSplitter:
    """中文文本分块器"""
    
    def __init__(self, config: dict):
        """
        初始化分块器
        
        Args:
            config: 配置字典
        """
        self.config = config
        chunk_config = config['chunking']
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_config['chunk_size'],
            chunk_overlap=chunk_config['chunk_overlap'],
            separators=chunk_config['separators'],
            length_function=len,
            is_separator_regex=False
        )
    
    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        分割文档
        
        Args:
            documents: 文档列表
            
        Returns:
            分块后的文档列表
        """
        logger.info(f"开始分块 {len(documents)} 个文档...")
        
        chunks = []
        for doc in documents:
            content = doc.get('content', '')
            
            # 对于较短的文档，不需要分块
            if len(content) <= self.config['chunking']['chunk_size']:
                chunks.append({
                    'chunk_id': f"{doc['id']}_0",
                    'doc_id': doc['id'],
                    'content': content,
                    'metadata': doc
                })
            else:
                # 分块
                text_chunks = self.text_splitter.split_text(content)
                for i, chunk in enumerate(text_chunks):
                    chunks.append({
                        'chunk_id': f"{doc['id']}_{i}",
                        'doc_id': doc['id'],
                        'content': chunk,
                        'chunk_index': i,
                        'metadata': doc
                    })
        
        logger.info(f"分块完成，共生成 {len(chunks)} 个文本块")
        return chunks


if __name__ == "__main__":
    import yaml
    import json
    
    # 加载配置
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载知识库
    with open('../data/processed/knowledge_base.json', 'r', encoding='utf-8') as f:
        knowledge_base = json.load(f)
    
    # 分块
    splitter = ChineseTextSplitter(config)
    chunks = splitter.split_documents(knowledge_base)
    
    # 保存
    import os
    output_dir = config['data']['processed_data_dir']
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'chunks.json'), 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    
    logger.info("文本分块完成！")
