"""评估模块"""

import json
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger
from tqdm import tqdm


class RAGEvaluator:
    """RAG系统评估器"""
    
    def __init__(self, rag_system, config: dict):
        """
        初始化评估器
        
        Args:
            rag_system: RAG系统实例
            config: 配置字典
        """
        self.rag_system = rag_system
        self.config = config
    
    def evaluate_accuracy(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        评估准确率
        
        Args:
            test_data: 测试数据，每条包含question和ground_truth_answer
            
        Returns:
            评估指标字典
        """
        logger.info(f"开始评估准确率，测试样本数: {len(test_data)}")
        
        correct = 0
        total = len(test_data)
        
        for item in tqdm(test_data, desc="评估准确率"):
            question = item['question']
            ground_truth = item.get('answer', '')
            
            # 生成答案
            result = self.rag_system.query(question, use_history=False)
            predicted_answer = result['answer']
            
            # 简单的匹配评估（实际应用中应该使用更复杂的方法）
            if self._check_answer_similarity(predicted_answer, ground_truth):
                correct += 1
        
        accuracy = correct / total if total > 0 else 0
        
        logger.info(f"准确率: {accuracy:.3f} ({correct}/{total})")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_retrieval_with_relevance(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        使用测试集的真实相关性标签评估检索性能
        
        Args:
            test_data: 测试数据，每条包含question、answer、label
            
        Returns:
            评估指标字典 (MRR, Precision@K, Recall@K)
        """
        logger.info(f"开始评估检索性能（带相关性标签），测试样本数: {len(test_data)}")
        
        # 按问题分组
        from collections import defaultdict
        questions_dict = defaultdict(list)
        
        for item in test_data:
            q_id = item.get('question_id')
            if q_id:
                questions_dict[q_id].append(item)
        
        logger.info(f"共有 {len(questions_dict)} 个不同的问题")
        
        mrr_scores = []
        precision_at_5 = []
        recall_at_5 = []
        
        for q_id, items in tqdm(questions_dict.items(), desc="评估检索"):
            # 获取问题（使用第一个item的问题）
            question = items[0]['question']
            
            # 获取所有相关答案ID（label=1的答案）
            relevant_answer_ids = set([
                item['answer_id'] for item in items if item.get('label') == 1
            ])
            
            if not relevant_answer_ids:
                continue
            
            # 检索
            try:
                result = self.rag_system.query(question, use_history=False)
                retrieved_sources = result['sources']
                
                # 提取检索到的答案ID
                retrieved_answer_ids = []
                for source in retrieved_sources:
                    # 从metadata中提取answer_id
                    metadata = source.get('metadata', {})
                    if metadata.get('type') == 'qa_pair':
                        answer_id = source.get('doc_id', '').split('_')[-1]
                        retrieved_answer_ids.append(answer_id)
                
                # 计算MRR
                for rank, answer_id in enumerate(retrieved_answer_ids, 1):
                    if answer_id in relevant_answer_ids:
                        mrr_scores.append(1.0 / rank)
                        break
                else:
                    mrr_scores.append(0.0)
                
                # 计算Precision@5和Recall@5
                retrieved_set = set(retrieved_answer_ids[:5])
                relevant_set = relevant_answer_ids
                
                if len(retrieved_set) > 0:
                    precision = len(retrieved_set & relevant_set) / len(retrieved_set)
                    precision_at_5.append(precision)
                
                if len(relevant_set) > 0:
                    recall = len(retrieved_set & relevant_set) / len(relevant_set)
                    recall_at_5.append(recall)
                
            except Exception as e:
                logger.warning(f"评估问题 {q_id} 时出错: {e}")
                continue
        
        avg_mrr = np.mean(mrr_scores) if mrr_scores else 0
        avg_p5 = np.mean(precision_at_5) if precision_at_5 else 0
        avg_r5 = np.mean(recall_at_5) if recall_at_5 else 0
        
        logger.info(f"检索指标 - MRR: {avg_mrr:.3f}, P@5: {avg_p5:.3f}, R@5: {avg_r5:.3f}")
        
        return {
            'mrr': avg_mrr,
            'precision_at_5': avg_p5,
            'recall_at_5': avg_r5,
            'num_questions': len(questions_dict)
        }
    
    def evaluate_retrieval(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        评估检索性能
        
        Args:
            test_data: 测试数据，每条包含question和relevant_doc_ids
            
        Returns:
            评估指标字典 (Precision, Recall, F1)
        """
        logger.info(f"开始评估检索性能，测试样本数: {len(test_data)}")
        
        precisions = []
        recalls = []
        f1_scores = []
        
        for item in tqdm(test_data, desc="评估检索"):
            question = item['question']
            relevant_ids = set(item.get('relevant_doc_ids', []))
            
            if not relevant_ids:
                continue
            
            # 检索
            result = self.rag_system.query(question, use_history=False)
            retrieved_ids = set([
                source.get('doc_id', source.get('id'))
                for source in result['sources']
            ])
            
            # 计算指标
            if len(retrieved_ids) > 0:
                precision = len(relevant_ids & retrieved_ids) / len(retrieved_ids)
                precisions.append(precision)
            
            if len(relevant_ids) > 0:
                recall = len(relevant_ids & retrieved_ids) / len(relevant_ids)
                recalls.append(recall)
            
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                f1_scores.append(f1)
        
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        avg_f1 = np.mean(f1_scores) if f1_scores else 0
        
        logger.info(f"检索指标 - Precision: {avg_precision:.3f}, Recall: {avg_recall:.3f}, F1: {avg_f1:.3f}")
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }
    
    def evaluate_hallucination(self, test_data: List[Dict]) -> Dict[str, float]:
        """
        评估幻觉率
        
        Args:
            test_data: 测试数据
            
        Returns:
            幻觉率指标
        """
        logger.info(f"开始评估幻觉率，测试样本数: {len(test_data)}")
        
        hallucination_count = 0
        total = len(test_data)
        
        for item in tqdm(test_data, desc="评估幻觉"):
            question = item['question']
            
            # 生成答案
            result = self.rag_system.query(question, use_history=False)
            answer = result['answer']
            sources = result['sources']
            
            # 检查答案是否基于检索到的上下文
            # 简单方法：检查是否有高相似度的来源
            if not sources or all(s.get('score', 0) < 0.5 for s in sources):
                # 如果没有相关来源但仍给出了肯定答案，可能是幻觉
                if not self._is_uncertain_answer(answer):
                    hallucination_count += 1
        
        hallucination_rate = hallucination_count / total if total > 0 else 0
        
        logger.info(f"幻觉率: {hallucination_rate:.3f} ({hallucination_count}/{total})")
        
        return {
            'hallucination_rate': hallucination_rate,
            'hallucination_count': hallucination_count,
            'total': total
        }
    
    def _check_answer_similarity(self, predicted: str, ground_truth: str) -> bool:
        """检查答案相似性（简单版本）"""
        # 这里使用简单的关键词匹配
        # 实际应用中应该使用更复杂的方法，如ROUGE、BLEU等
        
        # 提取关键词
        pred_keywords = set(predicted)
        gt_keywords = set(ground_truth)
        
        # 计算重叠率
        if len(gt_keywords) == 0:
            return False
        
        overlap = len(pred_keywords & gt_keywords) / len(gt_keywords)
        return overlap > 0.3  # 30%的重叠即认为匹配
    
    def _is_uncertain_answer(self, answer: str) -> bool:
        """判断是否是不确定的回答"""
        uncertain_phrases = [
            "不确定", "无法回答", "没有相关信息",
            "不清楚", "无法判断", "建议咨询",
            "需要更多信息", "无法确定"
        ]
        
        return any(phrase in answer for phrase in uncertain_phrases)
    
    def comprehensive_evaluation(self, test_data: List[Dict]) -> Dict[str, any]:
        """
        综合评估
        
        Args:
            test_data: 测试数据
            
        Returns:
            所有评估指标
        """
        logger.info("="*60)
        logger.info("开始综合评估")
        logger.info("="*60)
        
        results = {}
        
        # 准确率评估（如果有正确答案）
        # accuracy_results = self.evaluate_accuracy(test_data)
        # results['accuracy'] = accuracy_results
        
        # 检索性能评估（使用相关性标签）
        if any('label' in item for item in test_data):
            retrieval_results = self.evaluate_retrieval_with_relevance(test_data)
            results['retrieval'] = retrieval_results
        
        # 幻觉率评估
        # 对于测试集，只评估有正确答案的样本
        positive_samples = [item for item in test_data if item.get('label') == 1]
        if positive_samples:
            sample_size = min(len(positive_samples), 100)
            hallucination_results = self.evaluate_hallucination(positive_samples[:sample_size])
            results['hallucination'] = hallucination_results
        
        logger.info("="*60)
        logger.info("综合评估完成")
        logger.info("="*60)
        
        return results


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv
    from src.rag_system import MedicalRAGSystem
    
    # 加载环境变量和配置
    load_dotenv()
    
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 初始化RAG系统
    rag_system = MedicalRAGSystem(config)
    
    # 加载测试数据
    with open('../data/processed/test_qa_pairs.json', 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 限制测试样本数量
    test_sample_size = config['evaluation'].get('test_sample_size', 100)
    test_data = test_data[:test_sample_size]
    
    # 评估
    evaluator = RAGEvaluator(rag_system, config)
    results = evaluator.comprehensive_evaluation(test_data)
    
    # 保存结果
    with open('../data/evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info("\n评估结果已保存到: data/evaluation_results.json")
