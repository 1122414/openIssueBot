# -*- coding: utf-8 -*-
"""
LLM服务模块

提供统一的LLM服务接口，支持多种LLM提供商
"""

from typing import Optional, Dict, Any
from .config import Config
from .llm_analysis import LLMAnalyzer
from .utils import log_info, log_error, log_warning


class LLMService:
    """
    LLM服务类，提供统一的LLM调用接口
    """
    
    def __init__(self, config: Config):
        """
        初始化LLM服务
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.analyzer = None
        
        try:
            self.analyzer = LLMAnalyzer(config)
            log_info(f"LLM服务初始化成功，使用提供商: {config.LLM_PROVIDER}")
        except Exception as e:
            log_error(f"LLM服务初始化失败: {e}")
    
    def is_available(self) -> bool:
        """
        检查LLM服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        return self.analyzer is not None
    
    def generate_response(self, prompt: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        生成LLM响应
        
        Args:
            prompt: 输入提示词
            max_tokens: 最大token数
            
        Returns:
            Dict: 响应结果
        """
        if not self.analyzer:
            return {
                "success": False,
                "error": "LLM服务未初始化",
                "answer": "抱歉，AI服务暂时不可用。"
            }
        
        try:
            # 使用LLMAnalyzer的analyze_problem方法
            result = self.analyzer.analyze_problem(prompt, max_tokens=max_tokens)
            return result
        except Exception as e:
            log_error(f"LLM响应生成失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "抱歉，AI分析过程中出现错误。"
            }
    
    def analyze_with_context(self, query: str, context: Optional[Dict] = None, max_tokens: int = 1000) -> Dict[str, Any]:
        """
        基于上下文分析问题
        
        Args:
            query: 用户查询
            context: 上下文信息
            max_tokens: 最大token数
            
        Returns:
            Dict: 分析结果
        """
        if not self.analyzer:
            return {
                "success": False,
                "error": "LLM服务未初始化",
                "answer": "抱歉，AI服务暂时不可用。"
            }
        
        try:
            result = self.analyzer.analyze_problem(query, context, max_tokens)
            return result
        except Exception as e:
            log_error(f"LLM上下文分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "抱歉，AI分析过程中出现错误。"
            }


# 全局LLM服务实例
_llm_service_instance = None


def get_llm_service() -> Optional[LLMService]:
    """
    获取LLM服务实例（单例模式）
    
    Returns:
        Optional[LLMService]: LLM服务实例
    """
    global _llm_service_instance
    
    if _llm_service_instance is None:
        try:
            config = Config()
            _llm_service_instance = LLMService(config)
        except Exception as e:
            log_error(f"创建LLM服务实例失败: {e}")
            return None
    
    return _llm_service_instance


def reset_llm_service():
    """
    重置LLM服务实例（用于配置更新后重新初始化）
    """
    global _llm_service_instance
    _llm_service_instance = None
    log_info("LLM服务实例已重置")