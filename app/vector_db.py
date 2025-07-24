import os
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from abc import ABC, abstractmethod

# 导入各种向量数据库客户端
try:
    import faiss
except ImportError:
    faiss = None

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
except ImportError:
    connections = Collection = FieldSchema = CollectionSchema = DataType = utility = None

try:
    from pymilvus import MilvusClient
except ImportError:
    MilvusClient = None

from .config import Config
from .utils import log_info, log_error, log_warning


class VectorDatabase(ABC):
    """向量数据库抽象基类"""
    
    @abstractmethod
    def create_collection(self, collection_name: str, dimension: int) -> bool:
        """创建集合"""
        pass
    
    @abstractmethod
    def insert_vectors(self, collection_name: str, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """插入向量"""
        pass
    
    @abstractmethod
    def search_vectors(self, collection_name: str, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """搜索向量"""
        pass
    
    @abstractmethod
    def delete_collection(self, collection_name: str) -> bool:
        """删除集合"""
        pass
    
    @abstractmethod
    def get_collection_info(self, collection_name: str) -> Dict:
        """获取集合信息"""
        pass


class FAISSDatabase(VectorDatabase):
    """FAISS向量数据库实现"""
    
    def __init__(self, config: Config):
        if faiss is None:
            raise ImportError("需要安装faiss-cpu或faiss-gpu包")
        
        self.config = config
        self.index_path = config.FAISS_INDEX_PATH
        self.index_type = config.FAISS_INDEX_TYPE
        self.indexes = {}
        
        # 确保索引目录存在
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        
        log_info("FAISS数据库初始化完成")
    
    def create_collection(self, collection_name: str, dimension: int) -> bool:
        """创建FAISS索引"""
        try:
            if self.index_type == "IndexFlatIP":
                index = faiss.IndexFlatIP(dimension)
            elif self.index_type == "IndexFlatL2":
                index = faiss.IndexFlatL2(dimension)
            elif self.index_type == "IndexIVFFlat":
                quantizer = faiss.IndexFlatL2(dimension)
                index = faiss.IndexIVFFlat(quantizer, dimension, 100)
            else:
                index = faiss.IndexFlatIP(dimension)
            
            self.indexes[collection_name] = {
                'index': index,
                'dimension': dimension,
                'metadata': []
            }
            
            log_info(f"FAISS集合 {collection_name} 创建成功，维度: {dimension}")
            return True
            
        except Exception as e:
            log_error(f"创建FAISS集合失败: {e}")
            return False
    
    def insert_vectors(self, collection_name: str, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """插入向量到FAISS索引"""
        try:
            if collection_name not in self.indexes:
                log_error(f"集合 {collection_name} 不存在")
                return False
            
            index_info = self.indexes[collection_name]
            index = index_info['index']
            
            # 确保向量是float32类型
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            
            # 添加向量到索引
            index.add(vectors)
            
            # 保存元数据
            index_info['metadata'].extend(metadata)
            
            log_info(f"向FAISS集合 {collection_name} 插入 {len(vectors)} 个向量")
            return True
            
        except Exception as e:
            log_error(f"插入向量到FAISS失败: {e}")
            return False
    
    def search_vectors(self, collection_name: str, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """在FAISS索引中搜索向量"""
        try:
            if collection_name not in self.indexes:
                log_error(f"集合 {collection_name} 不存在")
                return []
            
            index_info = self.indexes[collection_name]
            index = index_info['index']
            metadata = index_info['metadata']
            
            # 确保查询向量是正确的形状和类型
            if query_vector.ndim == 1:
                query_vector = query_vector.reshape(1, -1)
            if query_vector.dtype != np.float32:
                query_vector = query_vector.astype(np.float32)
            
            # 搜索
            scores, indices = index.search(query_vector, min(top_k, index.ntotal))
            
            # 构建结果
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx != -1 and idx < len(metadata):
                    result = metadata[idx].copy()
                    result['score'] = float(score)
                    result['rank'] = i + 1
                    results.append(result)
            
            return results
            
        except Exception as e:
            log_error(f"FAISS搜索失败: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除FAISS集合"""
        try:
            if collection_name in self.indexes:
                del self.indexes[collection_name]
                log_info(f"FAISS集合 {collection_name} 删除成功")
                return True
            return False
        except Exception as e:
            log_error(f"删除FAISS集合失败: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """获取FAISS集合信息"""
        if collection_name not in self.indexes:
            return {}
        
        index_info = self.indexes[collection_name]
        return {
            'name': collection_name,
            'dimension': index_info['dimension'],
            'total_vectors': index_info['index'].ntotal,
            'metadata_count': len(index_info['metadata']),
            'type': 'FAISS'
        }


class ZillizDatabase(VectorDatabase):
    """Zilliz向量数据库实现"""
    
    def __init__(self, config: Config):
        if MilvusClient is None:
            raise ImportError("需要安装pymilvus包")
        
        self.config = config
        self.uri = config.ZILLIZ_URI
        self.token = config.ZILLIZ_TOKEN
        
        if not self.uri or not self.token:
            raise ValueError("Zilliz URI和Token必须配置")
        
        try:
            self.client = MilvusClient(
                uri=self.uri,
                token=self.token
            )
            log_info("Zilliz数据库连接成功")
        except Exception as e:
            log_error(f"Zilliz数据库连接失败: {e}")
            raise
    
    def create_collection(self, collection_name: str, dimension: int) -> bool:
        """创建Zilliz集合"""
        try:
            # 检查集合是否已存在
            if self.client.has_collection(collection_name):
                log_info(f"Zilliz集合 {collection_name} 已存在")
                return True
            
            # 创建集合
            self.client.create_collection(
                collection_name=collection_name,
                dimension=dimension,
                metric_type="COSINE",
                auto_id=True
            )
            
            log_info(f"Zilliz集合 {collection_name} 创建成功，维度: {dimension}")
            return True
            
        except Exception as e:
            log_error(f"创建Zilliz集合失败: {e}")
            return False
    
    def insert_vectors(self, collection_name: str, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """插入向量到Zilliz"""
        try:
            # 准备数据
            data = []
            for i, (vector, meta) in enumerate(zip(vectors, metadata)):
                record = {
                    'vector': vector.tolist(),
                    **meta  # 展开元数据
                }
                data.append(record)
            
            # 插入数据
            result = self.client.insert(
                collection_name=collection_name,
                data=data
            )
            
            log_info(f"向Zilliz集合 {collection_name} 插入 {len(vectors)} 个向量")
            return True
            
        except Exception as e:
            log_error(f"插入向量到Zilliz失败: {e}")
            return False
    
    def search_vectors(self, collection_name: str, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """在Zilliz中搜索向量"""
        try:
            # 搜索
            results = self.client.search(
                collection_name=collection_name,
                data=[query_vector.tolist()],
                limit=top_k,
                output_fields=["*"]
            )
            
            # 处理结果
            formatted_results = []
            for i, hit in enumerate(results[0]):
                result = hit['entity']
                result['score'] = hit['distance']
                result['rank'] = i + 1
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            log_error(f"Zilliz搜索失败: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除Zilliz集合"""
        try:
            if self.client.has_collection(collection_name):
                self.client.drop_collection(collection_name)
                log_info(f"Zilliz集合 {collection_name} 删除成功")
                return True
            return False
        except Exception as e:
            log_error(f"删除Zilliz集合失败: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """获取Zilliz集合信息"""
        try:
            if not self.client.has_collection(collection_name):
                return {}
            
            stats = self.client.get_collection_stats(collection_name)
            return {
                'name': collection_name,
                'total_vectors': stats.get('row_count', 0),
                'type': 'Zilliz'
            }
        except Exception as e:
            log_error(f"获取Zilliz集合信息失败: {e}")
            return {}


class MilvusDatabase(VectorDatabase):
    """Milvus向量数据库实现"""
    
    def __init__(self, config: Config):
        if connections is None:
            raise ImportError("需要安装pymilvus包")
        
        self.config = config
        self.host = config.MILVUS_HOST
        self.port = config.MILVUS_PORT
        
        try:
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port
            )
            log_info(f"Milvus数据库连接成功 ({self.host}:{self.port})")
        except Exception as e:
            log_error(f"Milvus数据库连接失败: {e}")
            raise
    
    def create_collection(self, collection_name: str, dimension: int) -> bool:
        """创建Milvus集合"""
        try:
            # 检查集合是否已存在
            if utility.has_collection(collection_name):
                log_info(f"Milvus集合 {collection_name} 已存在")
                return True
            
            # 定义字段
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension),
                FieldSchema(name="metadata", dtype=DataType.JSON)
            ]
            
            # 创建集合
            schema = CollectionSchema(fields, f"Collection for {collection_name}")
            collection = Collection(collection_name, schema)
            
            # 创建索引
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("vector", index_params)
            
            log_info(f"Milvus集合 {collection_name} 创建成功，维度: {dimension}")
            return True
            
        except Exception as e:
            log_error(f"创建Milvus集合失败: {e}")
            return False
    
    def insert_vectors(self, collection_name: str, vectors: np.ndarray, metadata: List[Dict]) -> bool:
        """插入向量到Milvus"""
        try:
            collection = Collection(collection_name)
            
            # 准备数据
            data = [
                vectors.tolist(),
                metadata
            ]
            
            # 插入数据
            collection.insert(data)
            collection.flush()
            
            log_info(f"向Milvus集合 {collection_name} 插入 {len(vectors)} 个向量")
            return True
            
        except Exception as e:
            log_error(f"插入向量到Milvus失败: {e}")
            return False
    
    def search_vectors(self, collection_name: str, query_vector: np.ndarray, top_k: int = 10) -> List[Dict]:
        """在Milvus中搜索向量"""
        try:
            collection = Collection(collection_name)
            collection.load()
            
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 10}
            }
            
            # 搜索
            results = collection.search(
                data=[query_vector.tolist()],
                anns_field="vector",
                param=search_params,
                limit=top_k,
                output_fields=["metadata"]
            )
            
            # 处理结果
            formatted_results = []
            for i, hit in enumerate(results[0]):
                result = hit.entity.get('metadata', {})
                result['score'] = hit.score
                result['rank'] = i + 1
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            log_error(f"Milvus搜索失败: {e}")
            return []
    
    def delete_collection(self, collection_name: str) -> bool:
        """删除Milvus集合"""
        try:
            if utility.has_collection(collection_name):
                utility.drop_collection(collection_name)
                log_info(f"Milvus集合 {collection_name} 删除成功")
                return True
            return False
        except Exception as e:
            log_error(f"删除Milvus集合失败: {e}")
            return False
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """获取Milvus集合信息"""
        try:
            if not utility.has_collection(collection_name):
                return {}
            
            collection = Collection(collection_name)
            stats = collection.get_stats()
            
            return {
                'name': collection_name,
                'total_vectors': stats.get('row_count', 0),
                'type': 'Milvus'
            }
        except Exception as e:
            log_error(f"获取Milvus集合信息失败: {e}")
            return {}


class VectorDatabaseFactory:
    """向量数据库工厂类"""
    
    @staticmethod
    def create_database(config: Config) -> VectorDatabase:
        """根据配置创建向量数据库实例"""
        db_type = getattr(config, 'VECTOR_DB_TYPE', 'faiss').lower()
        
        if db_type == 'faiss':
            return FAISSDatabase(config)
        elif db_type == 'zilliz':
            return ZillizDatabase(config)
        elif db_type == 'milvus':
            return MilvusDatabase(config)
        else:
            raise ValueError(f"不支持的向量数据库类型: {db_type}")


# 便捷函数
def create_vector_database(config: Config) -> VectorDatabase:
    """创建向量数据库实例"""
    return VectorDatabaseFactory.create_database(config)