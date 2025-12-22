"""LLM模型接口模块"""

import os
from typing import List, Dict, Optional, Generator
from abc import ABC, abstractmethod
from loguru import logger
import dashscope
from http import HTTPStatus

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class BaseLLM(ABC):
    """LLM基类"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """生成回复"""
        pass
    
    @abstractmethod
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """流式生成回复"""
        pass


class QwenLLM(BaseLLM):
    """通义千问LLM"""
    
    def __init__(self, config: dict):
        """
        初始化通义千问模型
        
        Args:
            config: 配置字典
        """
        self.config = config
        llm_config = config['llm']
        
        # 获取API Key
        api_key = os.getenv(llm_config['api_key_env'])
        if not api_key:
            raise ValueError(f"请设置环境变量: {llm_config['api_key_env']}")
        
        dashscope.api_key = api_key
        self.model_name = llm_config['model_name']
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2048)
        
        logger.info(f"初始化通义千问模型: {self.model_name}")
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            response = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                result_format='message'
            )
            
            if response.status_code == HTTPStatus.OK:
                return response.output.choices[0].message.content
            else:
                error_msg = f"API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                return f"抱歉，生成回答时出现错误：{response.message}"
        
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"
    
    def generate_stream(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """
        流式生成回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            responses = dashscope.Generation.call(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                result_format='message',
                stream=True,
                incremental_output=True
            )
            
            for response in responses:
                if response.status_code == HTTPStatus.OK:
                    content = response.output.choices[0].message.content
                    yield content
                else:
                    error_msg = f"API调用失败: {response.code} - {response.message}"
                    logger.error(error_msg)
                    yield f"\n\n[错误: {response.message}]"
                    break
        
        except Exception as e:
            logger.error(f"流式生成时出错: {e}")
            yield f"\n\n[错误: {str(e)}]"


class LocalLLM(BaseLLM):
    """本地部署的LLM（使用Transformers）"""
    
    def __init__(self, config: dict):
        """
        初始化本地LLM
        
        Args:
            config: 配置字典
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("请安装transformers库: pip install transformers torch")
        
        self.config = config
        llm_config = config['llm']
        
        self.model_path = llm_config.get('model_path', llm_config['model_name'])
        self.device = llm_config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2048)
        
        logger.info(f"加载本地模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None
        )
        
        if self.device == 'cpu':
            self.model = self.model.to(self.device)
        
        self.model.eval()
        logger.info("本地模型加载完成")
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        try:
            # 构建消息
            messages = []
            if system_prompt:
                messages.append({'role': 'system', 'content': system_prompt})
            messages.append({'role': 'user', 'content': prompt})
            
            # 应用chat template
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Tokenize
            inputs = self.tokenizer([text], return_tensors='pt').to(self.model.device)
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get('max_tokens', self.max_tokens),
                    temperature=kwargs.get('temperature', self.temperature),
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # 解码
            response = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"
    
    def generate_stream(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """
        流式生成回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        # 本地模型的流式生成比较复杂，这里先返回完整结果
        # 可以使用TextIteratorStreamer实现真正的流式输出
        try:
            response = self.generate(prompt, system_prompt, **kwargs)
            yield response
        except Exception as e:
            logger.error(f"流式生成时出错: {e}")
            yield f"\n\n[错误: {str(e)}]"


class VLLMServer(BaseLLM):
    """vLLM服务器LLM（兼容OpenAI API）"""
    
    def __init__(self, config: dict):
        """
        初始化vLLM服务器连接
        
        Args:
            config: 配置字典
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("请安装openai库: pip install openai")
        
        self.config = config
        llm_config = config['llm']
        
        # vLLM服务器地址
        self.base_url = llm_config.get('base_url', 'http://localhost:8000/v1')
        self.model_name = llm_config.get('model_name', 'default')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 2048)
        
        # 创建OpenAI客户端连接到vLLM
        self.client = OpenAI(
            base_url=self.base_url,
            api_key='sk'  # vLLM不需要真实的API key
        )
        
        logger.info(f"连接到vLLM服务器: {self.base_url}")
        logger.info(f"使用模型: {self.model_name}")
    
    def generate(self, prompt: str, system_prompt: str = None, **kwargs) -> str:
        """
        生成回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Returns:
            生成的文本
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens)
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"生成回复时出错: {e}")
            return f"抱歉，生成回答时出现错误：{str(e)}"
    
    def generate_stream(self, prompt: str, system_prompt: str = None, **kwargs) -> Generator[str, None, None]:
        """
        流式生成回复
        
        Args:
            prompt: 用户提示
            system_prompt: 系统提示
            **kwargs: 其他参数
            
        Yields:
            生成的文本片段
        """
        messages = []
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=kwargs.get('temperature', self.temperature),
                max_tokens=kwargs.get('max_tokens', self.max_tokens),
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        
        except Exception as e:
            logger.error(f"流式生成时出错: {e}")
            yield f"\n\n[错误: {str(e)}]"


class LLMFactory:
    """LLM工厂类"""
    
    @staticmethod
    def create_llm(config: dict) -> BaseLLM:
        """
        创建LLM实例
        
        Args:
            config: 配置字典
            
        Returns:
            LLM实例
        """
        provider = config['llm']['provider']
        
        if provider == 'qwen':
            return QwenLLM(config)
        elif provider == 'local':
            return LocalLLM(config)
        elif provider == 'vllm':
            return VLLMServer(config)
        else:
            raise ValueError(f"不支持的LLM提供商: {provider}")


if __name__ == "__main__":
    import yaml
    from dotenv import load_dotenv
    
    # 加载环境变量
    load_dotenv()
    
    # 加载配置
    with open('../config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建LLM
    llm = LLMFactory.create_llm(config)
    
    # 测试生成
    test_prompt = "什么是感冒？"
    response = llm.generate(test_prompt)
    logger.info(f"\n问题: {test_prompt}")
    logger.info(f"回答: {response}")
    
    # 测试流式生成
    logger.info("\n流式生成测试:")
    for chunk in llm.generate_stream(test_prompt):
        print(chunk, end='', flush=True)
    print()
