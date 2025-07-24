# -*- coding: utf-8 -*-
"""
文本嵌入与向量计算模块

支持多种嵌入方式：
1. 本地SentenceTransformer模型（推荐，免费）
2. OpenAI Embedding API（需要API Key）

提供文本向量化、相似度计算等功能
"""

import numpy as np
import openai
from typing import List, Union, Optional, Tuple
from sentence_transformers import SentenceTransformer
import pickle
import os
import requests
import json
from .config import Config
from .utils import log_info, log_error, log_warning

class EmbeddingService:
    """
    文本嵌入服务
    
    支持本地模型和多种在线API嵌入方式
    """
    
    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        """
        初始化嵌入服务
        
        Args:
            provider: 嵌入提供商 (local, openai, zhipu, qwen, baidu)
            model_name: 模型名称
        """
        self.provider = provider or getattr(Config, 'EMBEDDING_PROVIDER', 'local')
        self.local_model = None
        self.openai_client = None
        self.zhipu_client = None
        self.qwen_client = None
        self.baidu_client = None
        
        if self.provider == "local":
            self.model_name = model_name or Config.LOCAL_EMBEDDING_MODEL
            self._init_local_model()
        elif self.provider == "openai":
            self.model_name = model_name or Config.OPENAI_EMBEDDING_MODEL
            self._init_openai_client()
        elif self.provider == "zhipu":
            self.model_name = model_name or getattr(Config, 'ZHIPU_EMBEDDING_MODEL', 'embedding-2')
            self._init_zhipu_client()
        elif self.provider == "qwen":
            self.model_name = model_name or getattr(Config, 'QWEN_EMBEDDING_MODEL', 'text-embedding-v1')
            self._init_qwen_client()
        elif self.provider == "baidu":
            self.model_name = model_name or getattr(Config, 'BAIDU_EMBEDDING_MODEL', 'embedding-v1')
            self._init_baidu_client()
        else:
            raise ValueError(f"不支持的嵌入提供商: {self.provider}")
    
    def _init_local_model(self) -> None:
        """
        初始化本地SentenceTransformer模型
        """
        try:
            log_info(f"正在加载本地嵌入模型: {self.model_name}")
            self.local_model = SentenceTransformer(self.model_name)
            log_info(f"本地模型加载成功，向量维度: {self.local_model.get_sentence_embedding_dimension()}")
        except Exception as e:
            log_error(f"加载本地模型失败: {e}")
            raise
    
    def _init_openai_client(self) -> None:
        """
        初始化OpenAI客户端
        """
        # 优先使用独立的嵌入模型API Key，如果没有则使用通用的API Key
        api_key = getattr(Config, 'OPENAI_EMBEDDING_API_KEY', None) or Config.OPENAI_API_KEY
        if not api_key:
            raise ValueError("使用OpenAI嵌入需要设置OPENAI_API_KEY或OPENAI_EMBEDDING_API_KEY")
            
        try:
            self.openai_client = openai.OpenAI(api_key=api_key)
            log_info(f"OpenAI客户端初始化成功，使用模型: {self.model_name}")
        except Exception as e:
            log_error(f"初始化OpenAI客户端失败: {e}")
            raise
    
    def _init_zhipu_client(self) -> None:
        """
        初始化智谱AI客户端
        """
        # 优先使用独立的嵌入模型API Key，如果没有则使用通用的API Key
        # 同时检查环境变量中的设置
        api_key = (getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', None) or 
                  os.getenv('ZHIPU_EMBEDDING_API_KEY') or 
                  getattr(Config, 'ZHIPU_API_KEY', None) or 
                  os.getenv('ZHIPU_API_KEY'))
        
        if not api_key:
            raise ValueError("使用智谱AI嵌入需要设置ZHIPU_API_KEY或ZHIPU_EMBEDDING_API_KEY")
            
        try:
            self.zhipu_api_key = api_key
            log_info(f"智谱AI客户端初始化成功，使用模型: {self.model_name}")
        except Exception as e:
            log_error(f"初始化智谱AI客户端失败: {e}")
            raise
    
    def _init_qwen_client(self) -> None:
        """
        初始化通义千问客户端
        """
        # 优先使用独立的嵌入模型API Key，如果没有则使用通用的API Key
        api_key = getattr(Config, 'QWEN_EMBEDDING_API_KEY', None) or getattr(Config, 'QWEN_API_KEY', None)
        if not api_key:
            raise ValueError("使用通义千问嵌入需要设置QWEN_API_KEY或QWEN_EMBEDDING_API_KEY")
            
        try:
            self.qwen_api_key = api_key
            log_info(f"通义千问客户端初始化成功，使用模型: {self.model_name}")
        except Exception as e:
            log_error(f"初始化通义千问客户端失败: {e}")
            raise
    
    def _init_baidu_client(self) -> None:
        """
        初始化百度文心客户端
        """
        # 优先使用独立的嵌入模型API Key，如果没有则使用通用的API Key
        api_key = getattr(Config, 'BAIDU_EMBEDDING_API_KEY', None) or getattr(Config, 'BAIDU_API_KEY', None)
        secret_key = getattr(Config, 'BAIDU_EMBEDDING_SECRET_KEY', None) or getattr(Config, 'BAIDU_SECRET_KEY', None)
        
        if not api_key or not secret_key:
            raise ValueError("使用百度文心嵌入需要设置BAIDU_API_KEY和BAIDU_SECRET_KEY或BAIDU_EMBEDDING_API_KEY和BAIDU_EMBEDDING_SECRET_KEY")
            
        try:
            self.baidu_api_key = api_key
            self.baidu_secret_key = secret_key
            log_info(f"百度文心客户端初始化成功，使用模型: {self.model_name}")
        except Exception as e:
            log_error(f"初始化百度文心客户端失败: {e}")
            raise
    
    def get_embeddings(self, texts: Union[str, List[str]], batch_size: int = 32) -> np.ndarray:
        """
        获取文本嵌入向量
        
        Args:
            texts: 单个文本或文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([])
        
        if self.provider == "local":
            return self._get_local_embeddings(texts, batch_size)
        elif self.provider == "openai":
            return self._get_openai_embeddings(texts, batch_size)
        elif self.provider == "zhipu":
            return self._get_zhipu_embeddings(texts, batch_size)
        elif self.provider == "qwen":
            return self._get_qwen_embeddings(texts, batch_size)
        elif self.provider == "baidu":
            return self._get_baidu_embeddings(texts, batch_size)
        else:
            raise ValueError(f"不支持的嵌入提供商: {self.provider}")
    
    def _get_local_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        使用本地模型获取嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            # 预处理文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # 批量处理
            all_embeddings = []
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                batch_embeddings = self.local_model.encode(
                    batch,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    normalize_embeddings=True  # 归一化向量
                )
                all_embeddings.append(batch_embeddings)
                
            return np.vstack(all_embeddings)
            
        except Exception as e:
            log_error(f"本地嵌入计算失败: {e}")
            raise
    
    def _get_openai_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        使用OpenAI API获取嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            # 预处理文本
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            all_embeddings = []
            
            # 批量处理（OpenAI API有批量限制）
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                response = self.openai_client.embeddings.create(
                    model=self.model_name,
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            
            embeddings_array = np.array(all_embeddings)
            
            # 归一化向量
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            return embeddings_array
            
        except Exception as e:
            log_error(f"OpenAI嵌入计算失败: {e}")
            raise
    
    def _get_zhipu_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        使用智谱AI API获取嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            processed_texts = [self._preprocess_text(text) for text in texts]
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                # 调用智谱AI API
                headers = {
                    'Authorization': f'Bearer {self.zhipu_api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': self.model_name,
                    'input': batch
                }
                
                response = requests.post(
                    'https://open.bigmodel.cn/api/paas/v4/embeddings',
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_embeddings = [item['embedding'] for item in result['data']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise Exception(f"智谱AI API调用失败: {response.text}")
            
            embeddings_array = np.array(all_embeddings)
            
            # 归一化向量
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            return embeddings_array
            
        except Exception as e:
            log_error(f"智谱AI嵌入计算失败: {e}")
            raise
    
    def _get_qwen_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        使用通义千问API获取嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            processed_texts = [self._preprocess_text(text) for text in texts]
            all_embeddings = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                # 调用通义千问API
                headers = {
                    'Authorization': f'Bearer {self.qwen_api_key}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'model': self.model_name,
                    'input': {
                        'texts': batch
                    }
                }
                
                response = requests.post(
                    'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding',
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_embeddings = [item['embedding'] for item in result['output']['embeddings']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise Exception(f"通义千问API调用失败: {response.text}")
            
            embeddings_array = np.array(all_embeddings)
            
            # 归一化向量
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            return embeddings_array
            
        except Exception as e:
            log_error(f"通义千问嵌入计算失败: {e}")
            raise
    
    def _get_baidu_embeddings(self, texts: List[str], batch_size: int) -> np.ndarray:
        """
        使用百度文心API获取嵌入向量
        
        Args:
            texts: 文本列表
            batch_size: 批处理大小
            
        Returns:
            np.ndarray: 嵌入向量数组
        """
        try:
            processed_texts = [self._preprocess_text(text) for text in texts]
            all_embeddings = []
            
            # 获取access_token
            access_token = self._get_baidu_access_token()
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                # 调用百度文心API
                headers = {
                    'Content-Type': 'application/json'
                }
                
                data = {
                    'input': batch
                }
                
                response = requests.post(
                    f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/embeddings/{self.model_name}?access_token={access_token}',
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    batch_embeddings = [item['embedding'] for item in result['data']]
                    all_embeddings.extend(batch_embeddings)
                else:
                    raise Exception(f"百度文心API调用失败: {response.text}")
            
            embeddings_array = np.array(all_embeddings)
            
            # 归一化向量
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            embeddings_array = embeddings_array / norms
            
            return embeddings_array
            
        except Exception as e:
            log_error(f"百度文心嵌入计算失败: {e}")
            raise
    
    def _get_baidu_access_token(self) -> str:
        """
        获取百度API的access_token
        
        Returns:
            str: access_token
        """
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {
            'grant_type': 'client_credentials',
            'client_id': self.baidu_api_key,
            'client_secret': self.baidu_secret_key
        }
        
        response = requests.post(url, params=params)
        if response.status_code == 200:
            return response.json()['access_token']
        else:
            raise Exception(f"获取百度access_token失败: {response.text}")
    
    def _preprocess_text(self, text: str) -> str:
        """
        预处理文本
        
        Args:
            text: 原始文本
            
        Returns:
            str: 处理后的文本
        """
        if not text:
            return ""
            
        # 基本清理
        text = text.strip()
        
        # 限制长度（避免过长文本）
        max_length = 8000 if self.provider == "local" else 8000
        if len(text) > max_length:
            text = text[:max_length] + "..."
            
        return text
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            embedding1: 向量1
            embedding2: 向量2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 确保向量是归一化的
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # 确保结果在[0, 1]范围内
        return max(0.0, min(1.0, (similarity + 1) / 2))
    
    def compute_similarities(self, query_embedding: np.ndarray, 
                           embeddings: np.ndarray) -> np.ndarray:
        """
        计算查询向量与多个向量的相似度
        
        Args:
            query_embedding: 查询向量
            embeddings: 向量矩阵
            
        Returns:
            np.ndarray: 相似度数组
        """
        if embeddings.size == 0:
            return np.array([])
            
        # 归一化
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # 计算余弦相似度
        similarities = np.dot(embeddings_norm, query_norm)
        
        # 转换到[0, 1]范围
        similarities = (similarities + 1) / 2
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """
        获取嵌入向量维度
        
        Returns:
            int: 向量维度
        """
        if self.provider == "local" and self.local_model:
            return self.local_model.get_sentence_embedding_dimension()
        elif self.provider == "openai":
            # OpenAI模型的维度
            if "text-embedding-3-small" in self.model_name:
                return 1536
            elif "text-embedding-3-large" in self.model_name:
                return 3072
            elif "text-embedding-ada-002" in self.model_name:
                return 1536
            else:
                return 1536  # 默认维度
        elif self.provider == "zhipu":
            return 1024  # 智谱AI嵌入维度
        elif self.provider == "qwen":
            return 1536  # 通义千问嵌入维度
        elif self.provider == "baidu":
            return 384   # 百度文心嵌入维度
        else:
            return getattr(Config, 'FAISS_DIMENSION', 1536)
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str) -> None:
        """
        保存嵌入向量到文件
        
        Args:
            embeddings: 嵌入向量数组
            filepath: 保存路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            log_info(f"嵌入向量已保存到: {filepath}")
        except Exception as e:
            log_error(f"保存嵌入向量失败: {e}")
            raise
    
    def load_embeddings(self, filepath: str) -> Optional[np.ndarray]:
        """
        从文件加载嵌入向量
        
        Args:
            filepath: 文件路径
            
        Returns:
            np.ndarray: 嵌入向量数组或None
        """
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    embeddings = pickle.load(f)
                log_info(f"嵌入向量已从 {filepath} 加载")
                return embeddings
            else:
                log_warning(f"嵌入向量文件不存在: {filepath}")
                return None
        except Exception as e:
            log_error(f"加载嵌入向量失败: {e}")
            return None
    
    def extract_issue_text(self, issue: dict) -> str:
        """
        从Issue数据中提取用于嵌入的文本
        
        Args:
            issue: Issue数据字典
            
        Returns:
            str: 提取的文本
        """
        # 组合标题和正文
        title = issue.get('title', '')
        body = issue.get('body', '') or ''
        
        # 可选：包含标签信息
        labels = issue.get('labels', [])
        label_text = ' '.join([label.get('name', '') for label in labels if isinstance(label, dict)])
        
        # 组合文本
        combined_text = f"{title}\n\n{body}"
        if label_text:
            combined_text += f"\n\nLabels: {label_text}"
            
        return combined_text
    
    def batch_extract_issue_texts(self, issues: List[dict]) -> List[str]:
        """
        批量提取Issues文本
        
        Args:
            issues: Issues列表
            
        Returns:
            List[str]: 提取的文本列表
        """
        return [self.extract_issue_text(issue) for issue in issues]

# 便捷函数
def create_embedding_service(provider: Optional[str] = None, 
                           model_name: Optional[str] = None) -> EmbeddingService:
    """
    创建嵌入服务实例
    
    Args:
        provider: 嵌入提供商 (local, openai, zhipu, qwen, baidu)
        model_name: 模型名称
        
    Returns:
        EmbeddingService: 嵌入服务实例
    """
    return EmbeddingService(provider, model_name)

def compute_text_similarity(text1: str, text2: str, 
                          embedding_service: Optional[EmbeddingService] = None) -> float:
    """
    计算两个文本的相似度
    
    Args:
        text1: 文本1
        text2: 文本2
        embedding_service: 嵌入服务实例
        
    Returns:
        float: 相似度分数
    """
    if embedding_service is None:
        embedding_service = create_embedding_service()
        
    embeddings = embedding_service.get_embeddings([text1, text2])
    return embedding_service.compute_similarity(embeddings[0], embeddings[1])