# -*- coding: utf-8 -*-
"""
GitHub API 交互模块

负责与GitHub API交互，拉取指定仓库的Issues，包括：
- 获取Issues列表
- 获取Issue详情和评论
- 缓存机制
- 错误处理和重试
"""

import requests
import json
import os
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from .config import Config
from .utils import log_info, log_error, log_warning

class GitHubAPI:
    """
    GitHub API 客户端
    
    提供与GitHub API交互的功能，包括Issues获取、缓存等
    """
    
    def __init__(self, token: Optional[str] = None, repo: Optional[str] = None):
        """
        初始化GitHub API客户端
        
        Args:
            token: GitHub访问令牌
            repo: 仓库名称，格式为 'owner/repo'
        """
        self.token = token or Config.GITHUB_TOKEN
        self.repo = repo or Config.GITHUB_REPO
        self.base_url = Config.GITHUB_API_BASE_URL
        self.cache_dir = Config.CACHE_DIR
        self.cache_expiry = timedelta(hours=Config.CACHE_EXPIRY_HOURS)
        
        # 设置请求头
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "OpenIssueBot/1.0"
        }
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        log_info(f"GitHub API客户端初始化完成，目标仓库: {self.repo}")
    
    def _get_cache_path(self, cache_key: str) -> str:
        """
        获取缓存文件路径
        
        Args:
            cache_key: 缓存键
            
        Returns:
            str: 缓存文件路径
        """
        return os.path.join(self.cache_dir, f"{cache_key}.json")
    
    def _is_cache_valid(self, cache_path: str) -> bool:
        """
        检查缓存是否有效
        
        Args:
            cache_path: 缓存文件路径
            
        Returns:
            bool: 缓存是否有效
        """
        if not os.path.exists(cache_path):
            return False
            
        # 检查文件修改时间
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        return datetime.now() - file_time < self.cache_expiry
    
    def _load_cache(self, cache_key: str) -> Optional[Any]:
        """
        加载缓存数据
        
        Args:
            cache_key: 缓存键
            
        Returns:
            缓存的数据或None
        """
        cache_path = self._get_cache_path(cache_key)
        
        if self._is_cache_valid(cache_path):
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    log_info(f"从缓存加载数据: {cache_key}")
                    return data
            except Exception as e:
                log_error(f"加载缓存失败: {e}")
                
        return None
    
    def _save_cache(self, cache_key: str, data: Any) -> None:
        """
        保存数据到缓存
        
        Args:
            cache_key: 缓存键
            data: 要缓存的数据
        """
        cache_path = self._get_cache_path(cache_key)
        
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                log_info(f"数据已缓存: {cache_key}")
        except Exception as e:
            log_error(f"保存缓存失败: {e}")
    
    def _make_request(self, url: str, params: Optional[Dict] = None, max_retries: int = 3) -> Optional[Dict]:
        """
        发送HTTP请求，带重试机制
        
        Args:
            url: 请求URL
            params: 请求参数
            max_retries: 最大重试次数
            
        Returns:
            响应数据或None
        """
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # 记录响应头信息用于调试
                if attempt == 0:  # 只在第一次尝试时记录详细信息
                    log_info(f"API请求: {url}, 参数: {params}")
                    log_info(f"响应状态: {response.status_code}")
                    # log_info(f"剩余请求次数: {response.headers.get('X-RateLimit-Remaining', 'N/A')}")
                
                # 检查API限制
                if response.status_code == 403:
                    reset_time = response.headers.get('X-RateLimit-Reset')
                    if reset_time:
                        reset_datetime = datetime.fromtimestamp(int(reset_time))
                        log_warning(f"API限制达到，重置时间: {reset_datetime}")
                        
                response.raise_for_status()
                data = response.json()
                
                # 记录返回数据的长度
                if isinstance(data, list) and attempt == 0:
                    log_info(f"返回数据长度: {len(data)}")
                
                return data
                
            except (requests.exceptions.SSLError, requests.exceptions.ConnectionError) as e:
                if attempt < max_retries:
                    wait_time = (attempt + 1) * 2  # 递增等待时间
                    log_warning(f"网络连接错误 (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                    log_info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    log_error(f"请求失败，已达最大重试次数: {url}, 错误: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                log_error(f"请求失败: {url}, 错误: {e}")
                return None
        
        return None
    
    def fetch_issues(self, state: str = "all", per_page: int = 100, max_pages: int = None) -> List[Dict]:
        """
        获取仓库的Issues
        
        Args:
            state: Issue状态 ('open', 'closed', 'all')
            per_page: 每页数量
            max_pages: 最大页数
            
        Returns:
            List[Dict]: Issues列表
        """
        cache_key = f"issues_{self.repo.replace('/', '_')}_{state}"
        
        # 尝试从缓存加载
        cached_data = self._load_cache(cache_key)
        if cached_data:
            return cached_data
        
        log_info(f"开始获取 {self.repo} 的Issues (状态: {state})")
        
        all_issues = []
        page = 1
        
        while max_pages is None or page <= max_pages:
            url = f"{self.base_url}/repos/{self.repo}/issues"
            params = {
                "state": state,
                "per_page": per_page,
                "page": page,
                "sort": "updated",
                "direction": "desc"
            }
            
            data = self._make_request(url, params)
            if not data:  # API请求失败
                log_warning(f"第 {page} 页请求失败，停止获取")
                break
                
            if len(data) == 0:  # 空页面，结束
                log_info(f"第 {page} 页为空，已获取所有Issues")
                break
                
            all_issues.extend(data)
            log_info(f"获取第 {page} 页，共 {len(data)} 个Issues")
            
            # 如果返回的数据少于per_page，说明已经是最后一页
            if len(data) < per_page:
                log_info(f"第 {page} 页数据不足 {per_page} 个，已到最后一页")
                break
                
            page += 1
            time.sleep(0.1)  # 避免请求过快
        
        log_info(f"总共获取 {len(all_issues)} 个Issues")
        
        # 缓存结果
        self._save_cache(cache_key, all_issues)
        
        return all_issues
    
    def fetch_issue_comments(self, issue_number: int) -> List[Dict]:
        """
        获取指定Issue的评论
        
        Args:
            issue_number: Issue编号
            
        Returns:
            List[Dict]: 评论列表
        """
        cache_key = f"comments_{self.repo.replace('/', '_')}_{issue_number}"
        
        # 尝试从缓存加载
        cached_data = self._load_cache(cache_key)
        if cached_data:
            return cached_data
        
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}/comments"
        
        all_comments = []
        page = 1
        
        while True:
            params = {"page": page, "per_page": 100}
            data = self._make_request(url, params)
            
            if not data:
                break
                
            all_comments.extend(data)
            
            if len(data) < 100:
                break
                
            page += 1
            time.sleep(0.1)
        
        # 缓存结果
        self._save_cache(cache_key, all_comments)
        
        return all_comments
    
    def get_issue_detail(self, issue_number: int) -> Optional[Dict]:
        """
        获取Issue详细信息（包括评论）
        
        Args:
            issue_number: Issue编号
            
        Returns:
            Dict: Issue详细信息
        """
        url = f"{self.base_url}/repos/{self.repo}/issues/{issue_number}"
        issue_data = self._make_request(url)
        
        if issue_data:
            # 获取评论
            comments = self.fetch_issue_comments(issue_number)
            issue_data['comments_data'] = comments
            
        return issue_data
    
    def search_issues(self, query: str, max_results: int = 50) -> List[Dict]:
        """
        搜索Issues
        
        Args:
            query: 搜索查询
            max_results: 最大结果数
            
        Returns:
            List[Dict]: 搜索结果
        """
        try:
            # 对查询进行预处理，避免GitHub API的422错误
            if not query or len(query.strip()) < 2:
                log_warning(f"搜索查询过短或为空: '{query}'")
                return []
            
            # 清理查询字符串
            cleaned_query = query.strip()
            
            # 如果查询太简单，添加更多上下文
            if len(cleaned_query.split()) == 1 and len(cleaned_query) < 4:
                # 对于单个短词，添加一些常见的Issue相关词汇
                cleaned_query = f"{cleaned_query} issue OR bug OR problem OR question"
            
            url = f"{self.base_url}/search/issues"
            params = {
                "q": f"{cleaned_query} repo:{self.repo}",
                "sort": "updated",
                "order": "desc",
                "per_page": min(max_results, 100)
            }
            
            log_info(f"GitHub搜索查询: {params['q']}")
            data = self._make_request(url, params)
            
            if data and 'items' in data:
                log_info(f"GitHub搜索返回 {len(data['items'])} 个结果")
                return data['items']
            else:
                log_warning(f"GitHub搜索无结果或请求失败")
                return []
                
        except Exception as e:
            log_error(f"GitHub搜索异常: {e}")
            return []
    
    def get_repo_info(self) -> Optional[Dict]:
        """
        获取仓库基本信息
        
        Returns:
            Dict: 仓库信息
        """
        url = f"{self.base_url}/repos/{self.repo}"
        return self._make_request(url)
    
    def clear_cache(self) -> None:
        """
        清除所有缓存
        """
        try:
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.json'):
                    os.remove(os.path.join(self.cache_dir, filename))
            log_info("缓存已清除")
        except Exception as e:
            log_error(f"清除缓存失败: {e}")

# 便捷函数
def create_github_client(token: Optional[str] = None, repo: Optional[str] = None) -> GitHubAPI:
    """
    创建GitHub API客户端
    
    Args:
        token: GitHub访问令牌
        repo: 仓库名称
        
    Returns:
        GitHubAPI: GitHub API客户端实例
    """
    return GitHubAPI(token, repo)