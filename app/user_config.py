"""用户配置加载模块

支持从用户配置文件加载配置，并与环境变量配置合并
"""

import json
import os
from typing import Dict, Any, Optional
from .config import Config
from .utils import log_info, log_warning, log_error

class UserConfig:
    """用户配置管理类"""
    
    def __init__(self, config_file: str = "user_config.json"):
        """初始化用户配置
        
        Args:
            config_file: 用户配置文件路径
        """
        self.config_file = config_file
        self.user_config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """加载用户配置文件"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.user_config = json.load(f)
                log_info(f"成功加载用户配置文件: {self.config_file}")
            except Exception as e:
                log_error(f"加载用户配置文件失败: {e}")
                self.user_config = {}
        else:
            log_warning(f"用户配置文件不存在: {self.config_file}，将使用默认配置")
            self.user_config = {}
    
    def get_github_config(self) -> Dict[str, str]:
        """获取GitHub配置"""
        github_config = self.user_config.get('github', {})
        return {
            'token': github_config.get('token') or Config.GITHUB_TOKEN,
            'repo': github_config.get('repo') or Config.GITHUB_REPO
        }
    
    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM配置"""
        llm_config = self.user_config.get('llm', {})
        provider = llm_config.get('provider', Config.LLM_PROVIDER)
        
        # 获取指定提供商的配置
        models_config = llm_config.get('models', {})
        provider_config = models_config.get(provider, {})
        
        result = {
            'provider': provider,
            'api_key': '',
            'model': '',
            'embedding_model': ''
        }
        
        if provider == 'openai':
            result.update({
                'api_key': provider_config.get('api_key') or Config.OPENAI_API_KEY,
                'model': provider_config.get('model') or Config.OPENAI_MODEL,
                'embedding_model': provider_config.get('embedding_model') or Config.OPENAI_EMBEDDING_MODEL
            })
        elif provider == 'zhipu':
            result.update({
                'api_key': provider_config.get('api_key') or Config.ZHIPU_API_KEY,
                'model': provider_config.get('model') or Config.ZHIPU_MODEL,
                'embedding_model': provider_config.get('embedding_model') or Config.ZHIPU_EMBEDDING_MODEL
            })
        elif provider == 'qwen':
            result.update({
                'api_key': provider_config.get('api_key') or Config.QWEN_API_KEY,
                'model': provider_config.get('model') or Config.QWEN_MODEL,
                'embedding_model': provider_config.get('embedding_model') or Config.QWEN_EMBEDDING_MODEL
            })
        elif provider == 'baidu':
            result.update({
                'api_key': provider_config.get('api_key') or Config.BAIDU_API_KEY,
                'secret_key': provider_config.get('secret_key') or Config.BAIDU_SECRET_KEY,
                'model': provider_config.get('model') or Config.BAIDU_MODEL,
                'embedding_model': provider_config.get('embedding_model') or Config.BAIDU_EMBEDDING_MODEL
            })
        elif provider == 'deepseek':
            result.update({
                'api_key': provider_config.get('api_key') or Config.DEEPSEEK_API_KEY,
                'model': provider_config.get('model') or Config.DEEPSEEK_MODEL
            })
        
        return result
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """获取嵌入模型配置"""
        embedding_config = self.user_config.get('embedding', {})
        provider = embedding_config.get('provider', Config.EMBEDDING_PROVIDER)
        
        result = {
            'provider': provider,
            'model': ''
        }
        
        if provider == 'local':
            result['model'] = embedding_config.get('local_model') or Config.LOCAL_EMBEDDING_MODEL
        else:
            # 对于在线嵌入模型，从LLM配置中获取对应的嵌入模型
            llm_config = self.get_llm_config()
            if provider == llm_config['provider']:
                result['model'] = llm_config.get('embedding_model', '')
            else:
                # 如果嵌入提供商与LLM提供商不同，需要单独配置
                if provider == 'openai':
                    result['model'] = Config.OPENAI_EMBEDDING_MODEL
                elif provider == 'zhipu':
                    result['model'] = Config.ZHIPU_EMBEDDING_MODEL
                elif provider == 'qwen':
                    result['model'] = Config.QWEN_EMBEDDING_MODEL
                elif provider == 'baidu':
                    result['model'] = Config.BAIDU_EMBEDDING_MODEL
        
        return result
    
    def get_search_config(self) -> Dict[str, Any]:
        """获取搜索配置"""
        search_config = self.user_config.get('search', {})
        return {
            'top_k_results': search_config.get('top_k_results', Config.TOP_K_RESULTS),
            'similarity_threshold': search_config.get('similarity_threshold', Config.SIMILARITY_THRESHOLD)
        }
    
    def get_cache_config(self) -> Dict[str, Any]:
        """获取缓存配置"""
        cache_config = self.user_config.get('cache', {})
        return {
            'expiry_hours': cache_config.get('expiry_hours', Config.CACHE_EXPIRY_HOURS)
        }
    
    def get_web_config(self) -> Dict[str, Any]:
        """获取Web服务器配置"""
        web_config = self.user_config.get('web', {})
        return {
            'host': web_config.get('host', Config.FLASK_HOST),
            'port': web_config.get('port', Config.FLASK_PORT),
            'debug': web_config.get('debug', Config.FLASK_DEBUG)
        }
    
    def validate_config(self) -> bool:
        """验证用户配置"""
        errors = []
        
        # 检查GitHub配置
        github_config = self.get_github_config()
        if not github_config['token']:
            errors.append("GitHub token未配置")
        if not github_config['repo']:
            errors.append("GitHub仓库未配置")
        
        # 检查LLM配置
        llm_config = self.get_llm_config()
        if not llm_config['api_key'] and llm_config['provider'] != 'local':
            errors.append(f"LLM提供商 {llm_config['provider']} 的API Key未配置")
        
        # 检查嵌入模型配置
        embedding_config = self.get_embedding_config()
        if embedding_config['provider'] != 'local':
            # 对于在线嵌入模型，需要检查对应的API Key
            if embedding_config['provider'] == llm_config['provider']:
                # 如果嵌入提供商与LLM提供商相同，使用相同的API Key
                if not llm_config['api_key']:
                    errors.append(f"嵌入模型提供商 {embedding_config['provider']} 的API Key未配置")
        
        if errors:
            log_error("用户配置验证失败:")
            for error in errors:
                log_error(f"  - {error}")
            return False
        
        log_info("用户配置验证通过")
        return True

# 全局用户配置实例
user_config = UserConfig()