# -*- coding: utf-8 -*-
"""
主要业务逻辑模块

将所有功能模块整合在一起，完成输入报错 → 查找Issue → 提取摘要 → LLM分析的完整流程：
1. 问题搜索引擎
2. 结果排序和过滤
3. 多模态搜索策略
4. 缓存管理
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import os
import json
from datetime import datetime

from .config import Config
from .github_api import GitHubAPI
from .embedding import EmbeddingService
from .faiss_search import FAISSSearchEngine
from .summarizer import IssueSummarizer
from .llm_analysis import LLMAnalyzer
from .utils import log_info, log_error, log_warning

class IssueSearchEngine:
    """
    Issue搜索引擎
    
    整合所有功能模块，提供完整的问题搜索和分析服务
    """
    
    def __init__(self, 
                 github_token: Optional[str] = None,
                 github_repo: Optional[str] = None,
                 use_local_embedding: Optional[bool] = None,
                 openai_api_key: Optional[str] = None):
        """
        初始化搜索引擎
        
        Args:
            github_token: GitHub访问令牌
            github_repo: GitHub仓库
            use_local_embedding: 是否使用本地嵌入模型
            openai_api_key: OpenAI API密钥
        """
        # 初始化配置
        self.github_token = github_token or Config.GITHUB_TOKEN
        self.github_repo = github_repo or Config.GITHUB_REPO
        self.use_local_embedding = use_local_embedding if use_local_embedding is not None else Config.USE_LOCAL_EMBEDDING
        self.openai_api_key = openai_api_key or Config.OPENAI_API_KEY
        
        # 初始化各个服务组件
        self.github_api = GitHubAPI(self.github_token, self.github_repo)
        self.embedding_service = EmbeddingService(use_local=self.use_local_embedding)
        self.search_engine = FAISSSearchEngine(dimension=self.embedding_service.get_embedding_dimension())
        self.summarizer = IssueSummarizer()
        
        # LLM分析器（可选）
        self.llm_analyzer = None
        if self.openai_api_key:
            try:
                self.llm_analyzer = LLMAnalyzer(self.openai_api_key)
            except Exception as e:
                log_warning(f"LLM分析器初始化失败: {e}")
        
        # 缓存相关
        self.cache_dir = Config.CACHE_DIR
        self.index_cache_path = os.path.join(self.cache_dir, f"index_{self.github_repo.replace('/', '_')}")
        self.issues_cache_path = os.path.join(self.cache_dir, f"issues_{self.github_repo.replace('/', '_')}.json")
        
        # 运行时数据
        self.issues_data = []
        self.issues_metadata = []
        self.is_index_loaded = False
        
        log_info(f"Issue搜索引擎初始化完成，目标仓库: {self.github_repo}")
    
    def initialize(self, force_refresh: bool = False) -> bool:
        """
        初始化搜索引擎（加载或构建索引）
        
        Args:
            force_refresh: 是否强制刷新数据
            
        Returns:
            bool: 初始化是否成功
        """
        try:
            log_info("开始初始化搜索引擎...")
            
            # 尝试加载缓存的索引
            if not force_refresh and self._load_cached_index():
                log_info("成功加载缓存的索引")
                return True
            
            # 获取Issues数据
            log_info("获取Issues数据...")
            issues = self.github_api.fetch_issues(state="all", max_pages=20)
            
            if not issues:
                log_error("未能获取到Issues数据")
                return False
            
            # 构建索引
            log_info(f"开始构建索引，共 {len(issues)} 个Issues")
            success = self._build_index(issues)
            
            if success:
                # 保存索引到缓存
                self._save_index_cache()
                log_info("索引构建并缓存成功")
                return True
            else:
                log_error("索引构建失败")
                return False
                
        except Exception as e:
            log_error(f"初始化搜索引擎失败: {e}")
            return False
    
    def search(self, 
               query: str, 
               max_results: int = 5,
               similarity_threshold: float = 0.3,
               include_llm_analysis: bool = True) -> Dict[str, Any]:
        """
        搜索相关Issues
        
        Args:
            query: 查询字符串
            max_results: 最大结果数
            similarity_threshold: 相似度阈值
            include_llm_analysis: 是否包含LLM分析
            
        Returns:
            Dict: 搜索结果
        """
        try:
            log_info(f"开始搜索: {query[:100]}...")
            
            # 确保索引已加载
            if not self.is_index_loaded:
                if not self.initialize():
                    return self._create_error_result("搜索引擎未初始化")
            
            # 获取查询向量
            query_embedding = self.embedding_service.get_embeddings([query])[0]
            
            # 向量搜索
            scores, indices, metadata = self.search_engine.search(
                query_embedding, 
                k=min(max_results * 2, 20)  # 获取更多候选结果
            )
            
            # 过滤低相似度结果
            filtered_results = []
            for i, (score, idx, meta) in enumerate(zip(scores, indices, metadata)):
                if score >= similarity_threshold:
                    # 获取对应的Issue数据
                    if idx < len(self.issues_data):
                        issue_data = self.issues_data[idx]
                        issue_summary = self.summarizer.extract_issue_summary(issue_data)
                        issue_summary['similarity_score'] = float(score)
                        filtered_results.append(issue_summary)
            
            # 限制结果数量
            filtered_results = filtered_results[:max_results]
            
            # 排序结果
            similarity_scores = [r['similarity_score'] for r in filtered_results]
            ranked_results = self.summarizer.rank_summaries(filtered_results, similarity_scores)
            
            # 构建搜索结果
            search_result = {
                "success": True,
                "query": query,
                "total_found": len(filtered_results),
                "results": ranked_results,
                "has_high_similarity": len(filtered_results) > 0 and max(similarity_scores) > 0.7,
                "search_strategy": "vector_similarity",
                "timestamp": datetime.now().isoformat()
            }
            
            # LLM分析（如果需要且没有高相似度结果）
            if include_llm_analysis and self.llm_analyzer:
                if not search_result["has_high_similarity"]:
                    log_info("相似度较低，启动LLM分析")
                    llm_result = self._perform_llm_analysis(query, ranked_results)
                    search_result["llm_analysis"] = llm_result
                elif len(ranked_results) > 0:
                    # 即使有高相似度结果，也可以提供LLM增强分析
                    log_info("提供LLM增强分析")
                    llm_result = self.llm_analyzer.analyze_with_issues(
                        query, 
                        ranked_results[:3],  # 使用前3个结果
                        self._get_project_info()
                    )
                    search_result["llm_enhancement"] = llm_result
            
            log_info(f"搜索完成，找到 {len(filtered_results)} 个相关结果")
            return search_result
            
        except Exception as e:
            log_error(f"搜索失败: {e}")
            return self._create_error_result(str(e))
    
    def search_by_keywords(self, 
                          keywords: List[str], 
                          max_results: int = 10) -> List[Dict]:
        """
        基于关键词搜索Issues
        
        Args:
            keywords: 关键词列表
            max_results: 最大结果数
            
        Returns:
            List[Dict]: 搜索结果
        """
        try:
            # 使用GitHub API搜索
            query = " ".join(keywords)
            search_results = self.github_api.search_issues(query, max_results)
            
            # 提取摘要
            summaries = []
            for issue in search_results:
                summary = self.summarizer.extract_issue_summary(issue)
                summaries.append(summary)
            
            return summaries
            
        except Exception as e:
            log_error(f"关键词搜索失败: {e}")
            return []
    
    def hybrid_search(self, 
                     query: str, 
                     max_results: int = 5) -> Dict[str, Any]:
        """
        混合搜索（向量搜索 + 关键词搜索）
        
        Args:
            query: 查询字符串
            max_results: 最大结果数
            
        Returns:
            Dict: 搜索结果
        """
        try:
            # 向量搜索
            vector_results = self.search(query, max_results, include_llm_analysis=False)
            
            # 提取关键词
            if self.llm_analyzer:
                keywords = self.llm_analyzer.extract_keywords(query)
            else:
                # 简单关键词提取
                import re
                keywords = re.findall(r'\b[a-zA-Z]{3,}\b', query)
                keywords = list(set(keywords))[:5]
            
            # 关键词搜索
            keyword_results = self.search_by_keywords(keywords, max_results)
            
            # 合并和去重结果
            all_results = []
            seen_numbers = set()
            
            # 添加向量搜索结果
            for result in vector_results.get('results', []):
                number = result.get('number')
                if number and number not in seen_numbers:
                    result['search_method'] = 'vector'
                    all_results.append(result)
                    seen_numbers.add(number)
            
            # 添加关键词搜索结果
            for result in keyword_results:
                number = result.get('number')
                if number and number not in seen_numbers:
                    result['search_method'] = 'keyword'
                    result['similarity_score'] = 0.5  # 默认分数
                    all_results.append(result)
                    seen_numbers.add(number)
            
            # 重新排序
            similarity_scores = [r.get('similarity_score', 0.5) for r in all_results]
            ranked_results = self.summarizer.rank_summaries(all_results, similarity_scores)
            
            return {
                "success": True,
                "query": query,
                "total_found": len(ranked_results),
                "results": ranked_results[:max_results],
                "search_strategy": "hybrid",
                "keywords_used": keywords,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            log_error(f"混合搜索失败: {e}")
            return self._create_error_result(str(e))
    
    def _build_index(self, issues: List[Dict]) -> bool:
        """
        构建FAISS索引
        
        Args:
            issues: Issues数据列表
            
        Returns:
            bool: 是否成功
        """
        try:
            # 提取文本
            issue_texts = self.embedding_service.batch_extract_issue_texts(issues)
            
            # 生成嵌入向量
            log_info("生成嵌入向量...")
            embeddings = self.embedding_service.get_embeddings(issue_texts)
            
            # 准备元数据
            metadata = []
            for i, issue in enumerate(issues):
                meta = {
                    'id': issue.get('id'),
                    'number': issue.get('number'),
                    'title': issue.get('title', ''),
                    'state': issue.get('state', ''),
                    'url': issue.get('html_url', ''),
                    'index': i
                }
                metadata.append(meta)
            
            # 添加到搜索引擎
            self.search_engine.add_vectors(embeddings, metadata)
            
            # 保存数据
            self.issues_data = issues
            self.issues_metadata = metadata
            self.is_index_loaded = True
            
            log_info(f"索引构建完成，包含 {len(issues)} 个Issues")
            return True
            
        except Exception as e:
            log_error(f"构建索引失败: {e}")
            return False
    
    def _load_cached_index(self) -> bool:
        """
        加载缓存的索引
        
        Returns:
            bool: 是否成功加载
        """
        try:
            # 检查缓存文件是否存在
            if not (os.path.exists(f"{self.index_cache_path}.index") and 
                   os.path.exists(f"{self.index_cache_path}.metadata") and
                   os.path.exists(self.issues_cache_path)):
                return False
            
            # 加载FAISS索引
            if not self.search_engine.load_index(self.index_cache_path):
                return False
            
            # 加载Issues数据
            with open(self.issues_cache_path, 'r', encoding='utf-8') as f:
                cached_data = json.load(f)
                self.issues_data = cached_data['issues']
                self.issues_metadata = cached_data['metadata']
            
            self.is_index_loaded = True
            log_info(f"成功加载缓存索引，包含 {len(self.issues_data)} 个Issues")
            return True
            
        except Exception as e:
            log_error(f"加载缓存索引失败: {e}")
            return False
    
    def _save_index_cache(self) -> None:
        """
        保存索引到缓存
        """
        try:
            # 确保缓存目录存在
            os.makedirs(self.cache_dir, exist_ok=True)
            
            # 保存FAISS索引
            self.search_engine.save_index(self.index_cache_path)
            
            # 保存Issues数据
            cache_data = {
                'issues': self.issues_data,
                'metadata': self.issues_metadata,
                'timestamp': datetime.now().isoformat(),
                'repo': self.github_repo
            }
            
            with open(self.issues_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            log_info("索引缓存保存成功")
            
        except Exception as e:
            log_error(f"保存索引缓存失败: {e}")
    
    def _perform_llm_analysis(self, query: str, related_issues: List[Dict]) -> Dict[str, Any]:
        """
        执行LLM分析
        
        Args:
            query: 用户查询
            related_issues: 相关Issues
            
        Returns:
            Dict: LLM分析结果
        """
        try:
            # 获取项目信息
            project_info = self._get_project_info()
            
            # 执行分析
            result = self.llm_analyzer.analyze_with_issues(
                query, 
                related_issues, 
                project_info
            )
            
            return result
            
        except Exception as e:
            log_error(f"LLM分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": "抱歉，AI分析服务暂时不可用。"
            }
    
    def _get_project_info(self) -> Optional[Dict]:
        """
        获取项目信息
        
        Returns:
            Dict: 项目信息
        """
        try:
            repo_info = self.github_api.get_repo_info()
            if repo_info:
                return {
                    'name': repo_info.get('name', ''),
                    'description': repo_info.get('description', ''),
                    'language': repo_info.get('language', ''),
                    'stars': repo_info.get('stargazers_count', 0),
                    'forks': repo_info.get('forks_count', 0)
                }
        except Exception as e:
            log_error(f"获取项目信息失败: {e}")
        
        return None
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        创建错误结果
        
        Args:
            error_message: 错误信息
            
        Returns:
            Dict: 错误结果
        """
        return {
            "success": False,
            "error": error_message,
            "results": [],
            "total_found": 0,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取搜索引擎统计信息
        
        Returns:
            Dict: 统计信息
        """
        return {
            "repo": self.github_repo,
            "is_initialized": self.is_index_loaded,
            "total_issues": len(self.issues_data),
            "embedding_service": {
                "use_local": self.use_local_embedding,
                "dimension": self.embedding_service.get_embedding_dimension()
            },
            "search_engine": self.search_engine.get_stats(),
            "llm_available": self.llm_analyzer is not None
        }
    
    def clear_cache(self) -> None:
        """
        清除所有缓存
        """
        try:
            # 清除GitHub API缓存
            self.github_api.clear_cache()
            
            # 清除索引缓存
            cache_files = [
                f"{self.index_cache_path}.index",
                f"{self.index_cache_path}.metadata",
                self.issues_cache_path
            ]
            
            for cache_file in cache_files:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
            
            # 重置状态
            self.issues_data = []
            self.issues_metadata = []
            self.is_index_loaded = False
            self.search_engine.clear()
            
            log_info("所有缓存已清除")
            
        except Exception as e:
            log_error(f"清除缓存失败: {e}")

# 便捷函数
def create_search_engine(github_token: Optional[str] = None,
                        github_repo: Optional[str] = None,
                        use_local_embedding: Optional[bool] = None,
                        openai_api_key: Optional[str] = None) -> IssueSearchEngine:
    """
    创建Issue搜索引擎
    
    Args:
        github_token: GitHub访问令牌
        github_repo: GitHub仓库
        use_local_embedding: 是否使用本地嵌入模型
        openai_api_key: OpenAI API密钥
        
    Returns:
        IssueSearchEngine: 搜索引擎实例
    """
    return IssueSearchEngine(github_token, github_repo, use_local_embedding, openai_api_key)

def quick_search(query: str, 
                github_repo: Optional[str] = None,
                max_results: int = 5) -> Dict[str, Any]:
    """
    快速搜索（便捷函数）
    
    Args:
        query: 查询字符串
        github_repo: GitHub仓库
        max_results: 最大结果数
        
    Returns:
        Dict: 搜索结果
    """
    try:
        search_engine = create_search_engine(github_repo=github_repo)
        
        if not search_engine.initialize():
            return {
                "success": False,
                "error": "搜索引擎初始化失败",
                "results": []
            }
        
        return search_engine.search(query, max_results)
        
    except Exception as e:
        log_error(f"快速搜索失败: {e}")
        return {
            "success": False,
            "error": str(e),
            "results": []
        }