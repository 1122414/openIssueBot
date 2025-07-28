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
    
    # LLM提供商配置
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")
    
    # OpenAI配置
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    OPENAI_EMBEDDING_API_KEY: str = os.getenv("OPENAI_EMBEDDING_API_KEY", "")
    
    # 在线LLM模型配置
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")  # openai, zhipu, qwen, baidu, local
    
    # 智谱AI配置
    ZHIPU_API_KEY: str = os.getenv("ZHIPU_API_KEY", "")
    ZHIPU_MODEL: str = os.getenv("ZHIPU_MODEL", "glm-4")
    ZHIPU_EMBEDDING_API_KEY: str = os.getenv("ZHIPU_EMBEDDING_API_KEY", "")
    
    # 阿里云通义千问配置
    QWEN_API_KEY: str = os.getenv("QWEN_API_KEY", "")
    QWEN_MODEL: str = os.getenv("QWEN_MODEL", "qwen-turbo")
    QWEN_EMBEDDING_API_KEY: str = os.getenv("QWEN_EMBEDDING_API_KEY", "")
    
    # 百度文心一言配置
    BAIDU_API_KEY: str = os.getenv("BAIDU_API_KEY", "")
    BAIDU_SECRET_KEY: str = os.getenv("BAIDU_SECRET_KEY", "")
    BAIDU_MODEL: str = os.getenv("BAIDU_MODEL", "ernie-bot-turbo")
    BAIDU_EMBEDDING_API_KEY: str = os.getenv("BAIDU_EMBEDDING_API_KEY", "")
    BAIDU_EMBEDDING_SECRET_KEY: str = os.getenv("BAIDU_EMBEDDING_SECRET_KEY", "")
    
    # DeepSeek配置
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    
    # 在线嵌入模型配置
    EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local")  # local, openai, zhipu, qwen, baidu
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")  # 当前选择的嵌入模型
    
    # OpenAI嵌入模型
    OPENAI_EMBEDDING_MODEL: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    
    # 智谱AI嵌入模型
    ZHIPU_EMBEDDING_MODEL: str = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-2")
    
    # 阿里云嵌入模型
    QWEN_EMBEDDING_MODEL: str = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1")
    
    # 百度嵌入模型
    BAIDU_EMBEDDING_MODEL: str = os.getenv("BAIDU_EMBEDDING_MODEL", "embedding-v1")
    
    # 本地嵌入模型配置（备选方案）
    LOCAL_EMBEDDING_MODEL: str = os.getenv("LOCAL_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    USE_LOCAL_EMBEDDING: bool = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() == "true"
    
    # FAISS 配置
    FAISS_INDEX_TYPE: str = os.getenv("FAISS_INDEX_TYPE", "IndexFlatIP")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "./data/faiss_index")
    FAISS_DIMENSION: int = int(os.getenv("FAISS_DIMENSION", "1536"))  # 默认维度，实际使用embedding服务的维度
    FAISS_CACHE_EXPIRY: int = int(os.getenv("FAISS_CACHE_EXPIRY", "3600"))
    
    # 向量数据库配置
    VECTOR_DB_TYPE: str = os.getenv("VECTOR_DB_TYPE", "faiss")  # faiss, zilliz, milvus
    
    # Zilliz 配置
    ZILLIZ_URI: str = os.getenv("ZILLIZ_URI", "")
    ZILLIZ_TOKEN: str = os.getenv("ZILLIZ_TOKEN", "")
    ZILLIZ_COLLECTION_NAME: str = os.getenv("ZILLIZ_COLLECTION_NAME", "issue_embeddings")
    ZILLIZ_DIMENSION: int = int(os.getenv("ZILLIZ_DIMENSION", "384"))
    
    # Milvus 配置
    MILVUS_HOST: str = os.getenv("MILVUS_HOST", "localhost")
    MILVUS_PORT: int = int(os.getenv("MILVUS_PORT", "19530"))
    MILVUS_COLLECTION_NAME: str = os.getenv("MILVUS_COLLECTION_NAME", "issue_embeddings")
    MILVUS_DIMENSION: int = int(os.getenv("MILVUS_DIMENSION", "384"))
    
    # 搜索配置
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
    
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
    
    @classmethod
    def reload_config(cls) -> None:
        """
        重新加载配置
        """
        # 重新加载环境变量
        load_dotenv(override=True)
        
        # 重新设置所有配置项
        cls.GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
        cls.GITHUB_REPO = os.getenv("GITHUB_REPO", "facebook/react")
        
        # LLM配置
        cls.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
        cls.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        cls.OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        cls.OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        cls.OPENAI_EMBEDDING_API_KEY = os.getenv("OPENAI_EMBEDDING_API_KEY", "")
        
        cls.ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
        cls.ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4")
        cls.ZHIPU_EMBEDDING_API_KEY = os.getenv("ZHIPU_EMBEDDING_API_KEY", "")
        
        cls.QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
        cls.QWEN_MODEL = os.getenv("QWEN_MODEL", "qwen-turbo")
        cls.QWEN_EMBEDDING_API_KEY = os.getenv("QWEN_EMBEDDING_API_KEY", "")
        
        cls.BAIDU_API_KEY = os.getenv("BAIDU_API_KEY", "")
        cls.BAIDU_SECRET_KEY = os.getenv("BAIDU_SECRET_KEY", "")
        cls.BAIDU_MODEL = os.getenv("BAIDU_MODEL", "ernie-bot-turbo")
        cls.BAIDU_EMBEDDING_API_KEY = os.getenv("BAIDU_EMBEDDING_API_KEY", "")
        cls.BAIDU_EMBEDDING_SECRET_KEY = os.getenv("BAIDU_EMBEDDING_SECRET_KEY", "")
        
        cls.DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
        cls.DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        # 嵌入模型配置
        cls.EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "local")
        cls.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        cls.ZHIPU_EMBEDDING_MODEL = os.getenv("ZHIPU_EMBEDDING_MODEL", "embedding-2")
        cls.QWEN_EMBEDDING_MODEL = os.getenv("QWEN_EMBEDDING_MODEL", "text-embedding-v1")
        cls.BAIDU_EMBEDDING_MODEL = os.getenv("BAIDU_EMBEDDING_MODEL", "embedding-v1")
        cls.USE_LOCAL_EMBEDDING = os.getenv("USE_LOCAL_EMBEDDING", "true").lower() == "true"
        
        # 系统配置
        cls.SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
        cls.CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", "24"))
        
        # Flask配置
        cls.FLASK_HOST = os.getenv("FLASK_HOST", "127.0.0.1")
        cls.FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))
        cls.FLASK_DEBUG = os.getenv("FLASK_DEBUG", "true").lower() == "true"

# 全局配置实例
config = Config()

# 在模块加载时验证配置
if __name__ == "__main__":
    if config.validate_config():
        print("配置验证通过")
        config.create_directories()
    else:
        print("配置验证失败，请检查环境变量设置")