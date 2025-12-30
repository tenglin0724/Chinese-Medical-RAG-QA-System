"""Gradio Web界面"""

import gradio as gr
import yaml
import os
from loguru import logger
from dotenv import load_dotenv

from src.rag_system import MedicalRAGSystem


class MedicalRAGWebUI:
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化Web界面
        
        Args:
            config_path: 配置文件路径
        """
        # 加载环境变量
        load_dotenv()
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # 初始化RAG系统
        logger.info("初始化RAG系统...")
        self.rag_system = MedicalRAGSystem(self.config)
        
        logger.info("Web界面初始化完成")
    
    def query_with_sources(self, question: str, use_history: bool, top_k: int):
        """
        查询并返回结果和来源
        
        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量
            
        Returns:
            (回答, 来源展示)
        """
        if not question.strip():
            return "请输入问题。", ""
        
        try:
            # 查询
            result = self.rag_system.query(
                question=question,
                top_k=top_k,
                use_history=use_history
            )
            
            # 格式化回答
            answer = result['answer']
            
            # 格式化来源
            sources_text = self._format_sources(result['sources'])
            
            return answer, sources_text
        
        except Exception as e:
            logger.error(f"查询出错: {e}")
            return f"抱歉，处理您的问题时出现错误：{str(e)}", ""
    
    def query_stream(self, question: str, use_history: bool, top_k: int):
        """
        流式查询
        
        Args:
            question: 用户问题
            use_history: 是否使用对话历史
            top_k: 检索文档数量
            
        Yields:
            (回答片段, 来源展示)
        """
        if not question.strip():
            yield "请输入问题。", ""
            return
        
        try:
            # 流式查询
            answer_generator, sources = self.rag_system.query_stream(
                question=question,
                top_k=top_k,
                use_history=use_history
            )
            
            # 格式化来源
            sources_text = self._format_sources(sources)
            
            # 流式返回答案
            full_answer = ""
            for chunk in answer_generator:
                full_answer += chunk
                yield full_answer, sources_text
        
        except Exception as e:
            logger.error(f"流式查询出错: {e}")
            yield f"抱歉，处理您的问题时出现错误：{str(e)}", ""
    
    def _format_sources(self, sources: list) -> str:
        """格式化来源信息"""
        if not sources:
            return "未找到相关来源"
        
        formatted = []
        for i, source in enumerate(sources, 1):
            score = source.get('score', 0)
            content = source.get('content', '')
            
            # 截取内容
            preview = content[:300] + "..." if len(content) > 300 else content
            
            formatted.append(
                f"**来源 {i}** (相似度: {score:.3f})\n"
                f"{preview}\n"
            )
        
        return "\n---\n".join(formatted)
    
    def clear_history(self):
        """清空对话历史"""
        self.rag_system.clear_history()
        return "对话历史已清空", ""
    
    def build_interface(self):
        """构建Gradio界面"""
        
        with gr.Blocks(title="中文医疗RAG问答系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown(
                """
                # 🏥 中文医疗RAG问答系统
                
                基于大语言模型和检索增强生成(RAG)技术的医疗问答助手。
                
                **注意**: 本系统仅供学习参考，不能替代专业医疗建议。如有健康问题，请咨询专业医生。
                """
            )
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    question_input = gr.Textbox(
                        label="请输入您的健康问题",
                        placeholder="例如：感冒了怎么办？发烧吃什么药？",
                        lines=3
                    )
                    
                    with gr.Row():
                        submit_btn = gr.Button("🔍 查询", variant="primary")
                        clear_btn = gr.Button("🗑️ 清空历史")
                    
                    # 高级选项
                    with gr.Accordion("⚙️ 高级选项", open=False):
                        use_history_checkbox = gr.Checkbox(
                            label="使用对话历史",
                            value=True,
                            info="启用后将考虑之前的对话上下文"
                        )
                        top_k_slider = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=5,
                            step=1,
                            label="检索文档数量",
                            info="增加数量可能提供更多信息，但也可能引入噪声"
                        )
                    
                    # 回答区域
                    answer_output = gr.Textbox(
                        label="💬 回答",
                        lines=10,
                        interactive=False
                    )
                
                with gr.Column(scale=1):
                    # 来源区域
                    sources_output = gr.Markdown(
                        label="📚 参考来源",
                        value="检索到的相关医疗知识将显示在这里"
                    )
            
            # 示例问题
            gr.Examples(
                examples=[
                    ["感冒了怎么办？"],
                    ["发烧需要吃什么药？"],
                    ["咳嗽有痰怎么治疗？"],
                    ["头痛是什么原因？"],
                    ["高血压患者饮食注意什么？"]
                ],
                inputs=question_input
            )
            
            # 系统信息
            gr.Markdown(
                """
                ---
                ### 📊 系统信息
                - **数据集**: cMedQA2 中文医疗问答数据集
                - **向量模型**: BGE-Large-ZH-V1.5
                - **语言模型**: 通义千问 Qwen-Plus
                - **检索方式**: FAISS向量相似度检索
                
                ### 💡 使用建议
                1. 问题尽量具体明确
                2. 可以进行多轮对话，系统会记住上下文
                3. 注意查看参考来源，了解答案依据
                4. 遇到不确定的问题，系统会明确说明
                """
            )
            
            # 绑定事件
            submit_btn.click(
                fn=self.query_stream,
                inputs=[question_input, use_history_checkbox, top_k_slider],
                outputs=[answer_output, sources_output]
            )
            
            clear_btn.click(
                fn=self.clear_history,
                outputs=[answer_output, sources_output]
            )
        
        return demo
    
    def launch(self, **kwargs):
        """启动Web界面"""
        demo = self.build_interface()
        
        # 获取部署配置
        deployment_config = self.config.get('deployment', {})
        
        launch_kwargs = {
            'server_name': deployment_config.get('host', '0.0.0.0'),
            'server_port': deployment_config.get('port', 7860),
            'share': deployment_config.get('share', False),
            **kwargs
        }
        
        logger.info(f"启动Web界面: http://{launch_kwargs['server_name']}:{launch_kwargs['server_port']}")
        demo.launch(**launch_kwargs)


if __name__ == "__main__":
    logger.add(
        "logs/web_ui.log",
        rotation="500 MB",
        retention="10 days",
        level="INFO"
    )
    
    web_ui = MedicalRAGWebUI()
    web_ui.launch()
