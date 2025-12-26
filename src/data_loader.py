"""数据加载和预处理模块"""

import pandas as pd
import os
from typing import List, Dict, Tuple
from loguru import logger
from tqdm import tqdm


class CMedQA2Loader:
    """cMedQA2数据集加载器"""
    
    def __init__(self, config: dict):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.dataset_path = config['data']['dataset_path']
        self.question_file = config['data']['question_file']
        self.answer_file = config['data']['answer_file']
        
    def load_questions(self) -> pd.DataFrame:
        """加载问题数据"""
        question_path = os.path.join(self.dataset_path, self.question_file)
        logger.info(f"加载问题数据: {question_path}")
        
        df = pd.read_csv(question_path)
        logger.info(f"成功加载 {len(df)} 条问题数据")
        return df
    
    def load_answers(self) -> pd.DataFrame:
        """加载答案数据"""
        answer_path = os.path.join(self.dataset_path, self.answer_file)
        logger.info(f"加载答案数据: {answer_path}")
        
        df = pd.read_csv(answer_path)
        logger.info(f"成功加载 {len(df)} 条答案数据")
        return df
    
    def load_candidates(self, split: str = 'train') -> List[Tuple[str, str, int]]:
        """
        加载候选对数据
        
        Args:
            split: 数据集划分 (train/dev/test)
            
        Returns:
            候选对列表，每个元素为 (question_id, answer_id, label)
        """
        file_mapping = {
            'train': self.config['data']['train_file'],
            'dev': self.config['data']['dev_file'],
            'test': self.config['data']['test_file']
        }
        
        candidate_file = file_mapping.get(split)
        if not candidate_file:
            raise ValueError(f"Invalid split: {split}")
        
        # 先尝试解压文件
        candidate_path = os.path.join(self.dataset_path, candidate_file)
        zip_path = candidate_path.replace('.txt', '.zip')
        
        if not os.path.exists(candidate_path) and os.path.exists(zip_path):
            logger.info(f"解压{candidate_file}...")
            import zipfile
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.dataset_path)
        
        logger.info(f"加载{split}候选对数据: {candidate_path}")
        
        candidates = []
        with open(candidate_path, 'r', encoding='utf-8') as f:
            # 跳过表头
            header = f.readline()
            
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    question_id = parts[0]
                    pos_ans_id = parts[1]
                    neg_ans_id = parts[2]
                    
                    # 正样本
                    candidates.append((question_id, pos_ans_id, 1))
                    # 负样本
                    candidates.append((question_id, neg_ans_id, 0))
        
        logger.info(f"成功加载 {len(candidates)} 条候选对数据")
        return candidates
    
    def build_qa_pairs(self, split: str = 'train', use_all_answers: bool = False) -> List[Dict[str, str]]:
        """
        构建QA对数据
        
        Args:
            split: 数据集划分
            use_all_answers: 是否使用所有答案（包括负样本）
            
        Returns:
            QA对列表，每个元素包含问题和答案
        """
        logger.info(f"构建{split}集QA对...")
        
        # 加载数据
        questions_df = self.load_questions()
        answers_df = self.load_answers()
        candidates = self.load_candidates(split)
        
        # 创建索引映射
        question_map = questions_df.set_index('question_id')['content'].to_dict()
        answer_map = answers_df.set_index('ans_id')['content'].to_dict()
        
        # 构建QA对
        qa_pairs = []
        seen_pairs = set()  # 用于去重
        
        for q_id, a_id, label in tqdm(candidates, desc="构建QA对"):
            q_id = int(q_id)
            a_id = int(a_id)
            # 默认只保留正样本，除非明确要求使用所有答案
            if label == 0:
                continue
            
            pair_key = (q_id, a_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            question = question_map[q_id]
            answer = answer_map[a_id]
            if question and answer:
                qa_pairs.append({
                    'question_id': q_id,
                    'answer_id': a_id,
                    'question': question,
                    'answer': answer,
                    'label': label,
                    'combined_text': f"问题：{question}\n答案：{answer}"
                })
        
        logger.info(f"成功构建 {len(qa_pairs)} 条QA对")
        return qa_pairs,seen_pairs
    
    def build_knowledge_base(self) -> List[Dict[str, str]]:
        """
        构建知识库（使用所有的答案作为知识库）
        
        Returns:
            知识条目列表
        """
        logger.info("构建知识库...")
        
        questions_df = self.load_questions()
        answers_df = self.load_answers()
        
        # 方式1: 使用所有答案作为知识库
        knowledge_base = []
        for _, row in tqdm(answers_df.iterrows(), total=len(answers_df), desc="处理答案"):
            knowledge_base.append({
                'id': row['ans_id'],
                'content': row['content'],
                'type': 'answer'
            })
        
        # 方式2: 也可以使用QA对作为知识库（更完整的上下文）
        # 这里我们使用训练集的QA对
        train_qa_pairs = self.build_qa_pairs('train')
        for qa in train_qa_pairs:
            knowledge_base.append({
                'id': f"{qa['question_id']}_{qa['answer_id']}",
                'content': qa['combined_text'],
                'type': 'qa_pair',
                'question': qa['question'],
                'answer': qa['answer']
            })
        
        logger.info(f"知识库共包含 {len(knowledge_base)} 条记录")
        return knowledge_base
    
    def save_processed_data(self, data: List[Dict], filename: str):
        """保存处理后的数据"""
        output_dir = self.config['data']['processed_data_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, filename)
        df = pd.DataFrame(data)
        df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        logger.info(f"数据已保存到: {output_path}")


def preprocess_text(text: str) -> str:
    """
    文本预处理
    
    Args:
        text: 原始文本
        
    Returns:
        预处理后的文本
    """
    if not text:
        return ""
    
    # 去除多余空白
    text = ' '.join(text.split())
    
    return text.strip()


if __name__ == "__main__":
    import yaml
    
    # 加载配置
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化加载器
    loader = CMedQA2Loader(config)
    
    # 构建知识库
    knowledge_base = loader.build_knowledge_base()
    loader.save_processed_data(knowledge_base, 'knowledge_base.json')
    
    # 构建测试集
    test_qa_pairs = loader.build_qa_pairs('test')
    loader.save_processed_data(test_qa_pairs, 'test_qa_pairs.json')
    
    logger.info("数据预处理完成！")
