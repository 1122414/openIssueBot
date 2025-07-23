# -*- coding: utf-8 -*-
"""
OpenIssueBot - 开源项目智能问题分析助手

这是一个基于RAG技术的智能助手，能够：
1. 通过GitHub API获取项目Issues
2. 使用向量搜索匹配相关问题
3. 在无匹配时使用LLM生成解决方案
4. 提供Web界面和CLI工具

Author: OpenIssueBot Team
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "OpenIssueBot Team"
__description__ = "开源项目智能问题分析助手"

# 导入核心模块
from .config import Config
from .github_api import GitHubAPI
from .embedding import EmbeddingService
from .faiss_search import FAISSSearchEngine
from .summarizer import IssueSummarizer
from .llm_analysis import LLMAnalyzer
from .issue_search import IssueSearchEngine
from .utils import setup_logging, log_error, log_info

__all__ = [
    "Config",
    "GitHubAPI",
    "EmbeddingService",
    "FAISSSearchEngine",
    "IssueSummarizer",
    "LLMAnalyzer",
    "IssueSearchEngine",
    "setup_logging",
    "log_error",
    "log_info"
]