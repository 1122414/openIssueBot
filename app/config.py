# -*- coding: utf-8 -*-
"""
配置文件模块

用于集中管理配置项，包括：
- GitHub Token和仓库信息
- OpenAI API Key
- FAISS索引配置
- 日志配置等
"""

import os
from typing import Optional
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

class Config:
    """
    配置管理类
    
    支持从环境变量和配置文件中读取配置项
    """
    
    # GitHub 配置
    GITHUB_TOKEN: str = os.getenv("GITHUB_TOKEN", "")
    GITHUB_REPO: str = os.getenv("GITHUB_REPO", "facebook/react")  # 默认仓库
    GITHUB_API_BASE_URL: str = "https://api.github.com"
    
    # OpenAI 配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 本地嵌入模型配置（备选方案）
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "paraphrase-MiniLM-L6-v2")
    USE_LOCAL_EMBEDDING: bool = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() == "true"
    
    # FAISS 配置
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    FAISS_DIMENSION: int = int(os.getenv("FAISS_DIMENSION", "384"))  # MiniLM模型维度
    
    # 搜索配置
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # 缓存配置
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./data/cache")
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
    
    # Flask 配置
    FLASK_HOST: str = os.getenv("FLASK_HOST", "127.0.0.1")
    FLASK_PORT: int = int(os.getenv("FLASK_PORT", "5000"))
    FLASK_DEBUG: bool = os.getenv("FLASK_DEBUG", "true").lower() == "true"
    FLASK_SECRET_KEY: str = os.getenv("FLASK_SECRET_KEY", "openissuebot-secret-key-change-in-production")
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", "./logs/openissuebot.log")
    
    @classmethod
    def validate_config(cls) -> bool:
        """
        验证配置是否完整
        
        Returns:
            bool: 配置是否有效
        """
        errors = []
        
        # 检查必需的配置项
        if not cls.GITHUB_TOKEN:
            errors.append("GITHUB_TOKEN 未设置")
            
        if not cls.USE_LOCAL_EMBEDDING and not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY 未设置，且未启用本地嵌入模型")
            
        if not cls.GITHUB_REPO:
            errors.append("GITHUB_REPO 未设置")
            
        if errors:
            print("配置验证失败:")
            for error in errors:
                print(f"  - {error}")
            return False
            
        return True
    
    @classmethod
    def get_github_repo_info(cls) -> tuple[str, str]:
        """
        解析GitHub仓库信息
        
        Returns:
            tuple: (owner, repo)
        """
        if "/" not in cls.GITHUB_REPO:
            raise ValueError(f"无效的仓库格式: {cls.GITHUB_REPO}，应为 'owner/repo'")
            
        owner, repo = cls.GITHUB_REPO.split("/", 1)
        return owner, repo
    
    @classmethod
    def create_directories(cls) -> None:
        """
        创建必要的目录
        """
        directories = [
            os.path.dirname(cls.FAISS_INDEX_PATH),
            cls.CACHE_DIR,
            os.path.dirname(cls.LOG_FILE)
        ]
        
        for directory in directories:
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                print(f"创建目录: {directory}")

# 全局配置实例
config = Config()

# 在模块加载时验证配置
if __name__ == "__main__":
    if config.validate_config():
        print("配置验证通过")
        config.create_directories()
    else:
        print("配置验证失败，请检查环境变量设置")