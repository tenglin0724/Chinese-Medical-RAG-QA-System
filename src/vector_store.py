"""向量数据库和检索模块"""

import os
import pickle
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer


class VectorStore:
    """向量数据库"""
    
    def __init__(self, config: dict):
        """
        初始化向量数据库
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.embedding_config = config['embedding']
        self.vector_store_config = config['vector_store']
        
        # 初始化embedding模型
        logger.info(f"加载Embedding模型: {self.embedding_config['model_path']}")
        self.embedding_model = SentenceTransformer(
            self.embedding_config['model_path'],
            device=self.embedding_config['device']
        )
        
        # 向量数据库
        self.index = None  # 存储所有文档的向量矩阵
        self.documents = []
        self.dimension = None
        
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            texts: 文本列表
            
        Returns:
            向量数组
        """
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.embedding_config['batch_size'],
            show_progress_bar=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def build_index(self, documents: List[Dict]):
        """
        构建向量索引
        
        Args:
            documents: 文档列表
        """
        logger.info(f"开始构建向量索引，共 {len(documents)} 个文档...")
        
        # 提取文本
        texts = [doc['content'] for doc in documents]
        
        # 生成向量并存储为矩阵
        embeddings = self.embed_texts(texts)
        self.dimension = embeddings.shape[1]
        
        # 直接存储向量矩阵（已经归一化）
        self.index = embeddings.astype('float32')
        
        # 保存文档
        self.documents = documents
        
        logger.info(f"向量索引构建完成，维度: {self.dimension}, 文档数: {len(documents)}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        检索相关文档（使用余弦相似度）
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            相关文档列表
        """
        if self.index is None:
            raise ValueError("索引未构建，请先调用build_index")
        
        if top_k is None:
            top_k = self.vector_store_config['top_k']
        
        # 查询向量化
        query_embedding = self.embed_texts([query])[0]  # 获取单个向量
        
        # 计算余弦相似度（向量已归一化，所以就是点积）
        similarities = np.dot(self.index, query_embedding)
        
        # 获取top_k个最相似的文档索引
        top_k = min(top_k, len(self.documents))
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 组织结果
        results = []
        for rank, idx in enumerate(top_indices, 1):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['score'] = float(similarities[idx])  # 余弦相似度分数
                doc['rank'] = rank
                results.append(doc)
        
        return results
    
    def save(self, save_dir: str):
        """
        保存向量数据库
        
        Args:
            save_dir: 保存目录
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存向量索引
        if self.index is not None:
            index_path = os.path.join(save_dir, 'embeddings.npy')
            np.save(index_path, self.index)
            logger.info(f"向量索引已保存到: {index_path}")
        
        # 保存文档
        docs_path = os.path.join(save_dir, 'documents.pkl')
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        logger.info(f"文档已保存到: {docs_path}")
        
        # 保存元数据
        metadata = {
            'dimension': self.dimension,
            'num_documents': len(self.documents)
        }
        metadata_path = os.path.join(save_dir, 'metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        logger.info(f"元数据已保存到: {metadata_path}")
    
    def load(self, load_dir: str):
        """
        加载向量数据库
        
        Args:
            load_dir: 加载目录
        """
        # 加载向量索引
        index_path = os.path.join(load_dir, 'embeddings.npy')
        if os.path.exists(index_path):
            self.index = np.load(index_path)
            logger.info(f"向量索引已加载: {index_path}")
        
        # 加载文档
        docs_path = os.path.join(load_dir, 'documents.pkl')
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
            logger.info(f"文档已加载: {len(self.documents)} 条")
        
        # 加载元数据
        metadata_path = os.path.join(load_dir, 'metadata.pkl')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            self.dimension = metadata['dimension']
            logger.info(f"向量维度: {self.dimension}")


class Retriever:
    """检索器"""
    
    def __init__(self, vector_store: VectorStore, config: dict):
        """
        初始化检索器
        
        Args:
            vector_store: 向量数据库
            config: 配置字典
        """
        self.vector_store = vector_store
        self.config = config
    
    def retrieve(self, query: str, top_k: int = None) -> Tuple[List[Dict], List[str]]:
        """
        检索相关文档
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            (检索结果, 上下文文本列表)
        """
        # 检索
        results = self.vector_store.search(query, top_k)
        
        # 过滤低分结果
        threshold = self.config['vector_store'].get('score_threshold', 0.0)
        filtered_results = [r for r in results if r['score'] >= threshold]
        
        # 提取上下文
        contexts = []
        for result in filtered_results:
            # 根据文档类型组织上下文
            doc_type = result.get('metadata', {}).get('type', 'answer')
            
            if doc_type == 'qa_pair':
                # QA对格式
                context = result['content']
            else:
                # 答案格式
                context = f"医疗知识：{result['content']}"
            
            contexts.append(context)
        
        return filtered_results, contexts
    
    def format_contexts(self, contexts: List[str], max_length: int = 4000) -> str:
        """
        格式化上下文
        
        Args:
            contexts: 上下文列表
            max_length: 最大长度
            
        Returns:
            格式化后的上下文字符串
        """
        formatted = []
        current_length = 0
        
        for i, context in enumerate(contexts, 1):
            context_str = f"[{i}] {context}\n"
            context_length = len(context_str)
            
            if current_length + context_length > max_length:
                break
            
            formatted.append(context_str)
            current_length += context_length
        
        return "\n".join(formatted)


if __name__ == "__main__":
    import yaml
    import json
    
    # 加载配置
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载文档块
    with open('../data/processed/chunks.json', 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # 构建向量数据库
    vector_store = VectorStore(config)
    vector_store.build_index(chunks)
    
    # 保存
    save_dir = config['data']['vector_store_dir']
    vector_store.save(save_dir)
    
    # 测试检索
    retriever = Retriever(vector_store, config)
    test_query = "感冒了怎么办？"
    results, contexts = retriever.retrieve(test_query, top_k=5)
    
    logger.info(f"\n测试查询: {test_query}")
    logger.info(f"检索到 {len(results)} 条结果")
    for i, result in enumerate(results, 1):
        logger.info(f"\n结果 {i} (相似度: {result['score']:.3f}):")
        logger.info(result['content'][:200])
    
    logger.info("\n向量数据库构建完成！")
