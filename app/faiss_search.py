# -*- coding: utf-8 -*-
"""
FAISS 向量搜索模块

提供高效的向量相似度搜索功能：
1. 构建和管理FAISS索引
2. 向量搜索和检索
3. 索引的保存和加载
4. 支持不同的索引类型
"""

import faiss
import numpy as np
import pickle
import os
from typing import List, Tuple, Optional, Dict, Any
from .config import Config
from .utils import log_info, log_error, log_warning

class FAISSSearchEngine:
    """
    FAISS向量搜索引擎
    
    提供向量索引构建、搜索、保存和加载功能
    """
    
    def __init__(self, dimension: Optional[int] = None, index_type: str = "flat"):
        """
        初始化FAISS搜索引擎
        
        Args:
            dimension: 向量维度
            index_type: 索引类型 ('flat', 'ivf', 'hnsw')
        """
        self.dimension = dimension or Config.FAISS_DIMENSION
        self.index_type = index_type
        self.index = None
        self.metadata = []  # 存储与向量对应的元数据
        self.is_trained = False
        
        # 创建索引
        self._create_index()
        
        log_info(f"FAISS搜索引擎初始化完成，维度: {self.dimension}, 类型: {index_type}")
    
    def _create_index(self) -> None:
        """
        创建FAISS索引
        """
        try:
            if self.index_type == "flat":
                # 暴力搜索，精确但较慢
                self.index = faiss.IndexFlatIP(self.dimension)  # 内积索引
                self.is_trained = True
                
            elif self.index_type == "ivf":
                # 倒排文件索引，快速但需要训练
                nlist = 100  # 聚类中心数量
                quantizer = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
                self.is_trained = False
                
            elif self.index_type == "hnsw":
                # 分层导航小世界图，快速且精确
                # 注意：HNSW默认使用L2距离，需要特殊处理来支持内积
                M = 16  # 连接数
                # 使用L2距离的HNSW，但我们会在搜索时处理内积
                self.index = faiss.IndexHNSWFlat(self.dimension, M)
                self.index.hnsw.efConstruction = 200
                self.index.hnsw.efSearch = 50
                self.is_trained = True
                log_warning("HNSW索引使用L2距离，相似度计算可能不准确，建议使用flat或ivf索引")
                
            else:
                raise ValueError(f"不支持的索引类型: {self.index_type}")
                
            log_info(f"FAISS索引创建成功: {self.index_type}")
            
        except Exception as e:
            log_error(f"创建FAISS索引失败: {e}")
            raise
    
    def add_vectors(self, vectors: np.ndarray, metadata: Optional[List[Dict]] = None) -> None:
        """
        添加向量到索引
        
        Args:
            vectors: 向量数组
            metadata: 对应的元数据列表
        """
        if vectors.size == 0:
            log_warning("尝试添加空向量数组")
            return
            
        try:
            # 确保向量维度正确
            if vectors.shape[1] != self.dimension:
                raise ValueError(f"向量维度不匹配: 期望 {self.dimension}, 实际 {vectors.shape[1]}")
            
            # 归一化向量（对于内积索引）
            vectors_normalized = self._normalize_vectors(vectors)
            
            # 训练索引（如果需要）
            if not self.is_trained and self.index_type == "ivf":
                if len(vectors_normalized) >= 100:  # 需要足够的训练数据
                    log_info("开始训练IVF索引...")
                    self.index.train(vectors_normalized)
                    self.is_trained = True
                    log_info("IVF索引训练完成")
                else:
                    log_warning("训练数据不足，使用Flat索引")
                    self._create_index()  # 回退到Flat索引
            
            # 添加向量
            self.index.add(vectors_normalized)
            
            # 添加元数据
            if metadata:
                self.metadata.extend(metadata)
            else:
                # 创建默认元数据
                start_id = len(self.metadata)
                default_metadata = [{"id": start_id + i} for i in range(len(vectors_normalized))]
                self.metadata.extend(default_metadata)
            
            log_info(f"成功添加 {len(vectors_normalized)} 个向量到索引")
            
        except Exception as e:
            log_error(f"添加向量到索引失败: {e}")
            raise
    
    def search(self, query_vector: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            k: 返回的结果数量
            
        Returns:
            Tuple[np.ndarray, np.ndarray, List[Dict]]: (相似度分数, 索引, 元数据)
        """
        if self.index.ntotal == 0:
            log_warning("索引为空，无法搜索")
            return np.array([]), np.array([]), []
            
        try:
            # 确保查询向量维度正确
            if query_vector.shape[0] != self.dimension:
                raise ValueError(f"查询向量维度不匹配: 期望 {self.dimension}, 实际 {query_vector.shape[0]}")
            
            # 归一化查询向量
            query_normalized = self._normalize_vectors(query_vector.reshape(1, -1))
            
            # 搜索
            k = min(k, self.index.ntotal)  # 确保k不超过索引中的向量数量
            scores, indices = self.index.search(query_normalized, k)
            
            # 获取对应的元数据
            result_metadata = []
            for idx in indices[0]:
                if 0 <= idx < len(self.metadata):
                    result_metadata.append(self.metadata[idx])
                else:
                    result_metadata.append({"id": idx, "error": "metadata not found"})
            
            # 转换分数（内积转余弦相似度）
            similarity_scores = self._convert_scores(scores[0])
            
            log_info(f"搜索完成，返回 {len(result_metadata)} 个结果")
            
            return similarity_scores, indices[0], result_metadata
            
        except Exception as e:
            log_error(f"向量搜索失败: {e}")
            return np.array([]), np.array([]), []
    
    def batch_search(self, query_vectors: np.ndarray, k: int = 5) -> List[Tuple[np.ndarray, np.ndarray, List[Dict]]]:
        """
        批量搜索
        
        Args:
            query_vectors: 查询向量数组
            k: 每个查询返回的结果数量
            
        Returns:
            List[Tuple]: 每个查询的搜索结果
        """
        results = []
        for query_vector in query_vectors:
            result = self.search(query_vector, k)
            results.append(result)
        return results
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """
        归一化向量
        
        Args:
            vectors: 向量数组
            
        Returns:
            np.ndarray: 归一化后的向量
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        # 避免除零
        norms = np.where(norms == 0, 1, norms)
        return vectors / norms
    
    def _convert_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        转换分数到相似度
        
        Args:
            scores: 原始分数（内积或L2距离）
            
        Returns:
            np.ndarray: 相似度分数 [0, 1]
        """
        # 调试信息：记录原始分数
        log_info(f"原始FAISS分数: {scores[:5] if len(scores) > 5 else scores}")
        log_info(f"索引类型: {self.index_type}")
        
        if self.index_type == "hnsw":
            # HNSW使用L2距离，距离越小相似度越高
            # L2距离范围是[0, +∞)，我们需要转换为相似度[0, 1]
            # 使用改进的公式: similarity = exp(-distance/2) 提供更好的区分度
            converted_scores = np.exp(-scores / 2.0)
        else:
            # flat和ivf索引使用内积，分数已经在[-1, 1]范围内（归一化向量）
            # 使用sigmoid函数提供更好的区分度: sigmoid(score * 4)
            converted_scores = 1.0 / (1.0 + np.exp(-scores * 4.0))
        
        # 调试信息：记录转换后的分数
        log_info(f"转换后的相似度分数: {converted_scores[:5] if len(converted_scores) > 5 else converted_scores}")
        
        return converted_scores
    
    def save_index(self, filepath: str) -> None:
        """
        保存索引到文件
        
        Args:
            filepath: 保存路径
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 保存FAISS索引
            index_file = f"{filepath}.index"
            faiss.write_index(self.index, index_file)
            
            # 保存元数据和配置
            metadata_file = f"{filepath}.metadata"
            with open(metadata_file, 'wb') as f:
                pickle.dump({
                    'metadata': self.metadata,
                    'dimension': self.dimension,
                    'index_type': self.index_type,
                    'is_trained': self.is_trained
                }, f)
            
            log_info(f"索引已保存到: {filepath}")
            
        except Exception as e:
            log_error(f"保存索引失败: {e}")
            raise
    
    def load_index(self, filepath: str) -> bool:
        """
        从文件加载索引
        
        Args:
            filepath: 文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            index_file = f"{filepath}.index"
            metadata_file = f"{filepath}.metadata"
            
            if not (os.path.exists(index_file) and os.path.exists(metadata_file)):
                log_warning(f"索引文件不存在: {filepath}")
                return False
            
            # 加载FAISS索引
            self.index = faiss.read_index(index_file)
            
            # 加载元数据和配置
            with open(metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.metadata = data['metadata']
                self.dimension = data['dimension']
                self.index_type = data['index_type']
                self.is_trained = data['is_trained']
            
            log_info(f"索引已从 {filepath} 加载，包含 {self.index.ntotal} 个向量")
            return True
            
        except Exception as e:
            log_error(f"加载索引失败: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取索引统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "total_vectors": self.index.ntotal if self.index else 0,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "is_trained": self.is_trained,
            "metadata_count": len(self.metadata)
        }
    
    def clear(self) -> None:
        """
        清空索引
        """
        self._create_index()
        self.metadata = []
        log_info("索引已清空")
    
    def remove_vectors(self, indices: List[int]) -> None:
        """
        移除指定索引的向量（注意：FAISS不直接支持删除，需要重建索引）
        
        Args:
            indices: 要移除的向量索引列表
        """
        log_warning("FAISS不支持直接删除向量，建议重建索引")
        # 这里可以实现重建索引的逻辑
        pass

# 便捷函数
def create_search_engine(dimension: Optional[int] = None, 
                        index_type: str = "flat") -> FAISSSearchEngine:
    """
    创建FAISS搜索引擎
    
    Args:
        dimension: 向量维度
        index_type: 索引类型
        
    Returns:
        FAISSSearchEngine: 搜索引擎实例
    """
    return FAISSSearchEngine(dimension, index_type)

def search_similar_issues(query_embedding: np.ndarray, 
                         issue_embeddings: np.ndarray,
                         issue_metadata: List[Dict],
                         k: int = 5) -> Tuple[np.ndarray, List[Dict]]:
    """
    搜索相似的Issues（便捷函数）
    
    Args:
        query_embedding: 查询向量
        issue_embeddings: Issue向量数组
        issue_metadata: Issue元数据
        k: 返回结果数量
        
    Returns:
        Tuple[np.ndarray, List[Dict]]: (相似度分数, Issue元数据)
    """
    # 创建临时搜索引擎
    search_engine = create_search_engine(dimension=issue_embeddings.shape[1])
    
    # 添加向量
    search_engine.add_vectors(issue_embeddings, issue_metadata)
    
    # 搜索
    scores, indices, metadata = search_engine.search(query_embedding, k)
    
    return scores, metadata