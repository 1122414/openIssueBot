# -*- coding: utf-8 -*-
"""
问题摘要提取与格式化模块

从匹配的Issue中提取并生成简洁的摘要，包括：
1. Issue基本信息提取
2. 内容摘要生成
3. 解决方案提取
4. 格式化输出
"""

import re
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from .utils import log_info, log_error, log_warning

class IssueSummarizer:
    """
    Issue摘要提取器
    
    提供Issue内容分析、摘要生成和格式化功能
    """
    
    def __init__(self, max_summary_length: int = 300, max_solution_length: int = 500):
        """
        初始化摘要提取器
        
        Args:
            max_summary_length: 最大摘要长度
            max_solution_length: 最大解决方案长度
        """
        self.max_summary_length = max_summary_length
        self.max_solution_length = max_solution_length
        
        # 解决方案关键词
        self.solution_keywords = [
            'solution', 'solve', 'fix', 'fixed', 'resolved', 'workaround',
            '解决', '修复', '解决方案', '解决办法', '修复方法', '临时解决',
            'answer', 'answered', 'working', 'works', 'success'
        ]
        
        # 问题关键词
        self.problem_keywords = [
            'error', 'bug', 'issue', 'problem', 'fail', 'crash', 'exception',
            '错误', '问题', '故障', '异常', '崩溃', '失败'
        ]
        
        log_info("Issue摘要提取器初始化完成")
    
    def extract_issue_summary(self, issue: Dict) -> Dict:
        """
        提取Issue摘要信息
        
        Args:
            issue: Issue数据字典
            
        Returns:
            Dict: 摘要信息
        """
        try:
            summary = {
                'id': issue.get('id'),
                'number': issue.get('number'),
                'title': issue.get('title', ''),
                'state': issue.get('state', 'unknown'),
                'url': issue.get('html_url', ''),
                'created_at': self._format_date(issue.get('created_at')),
                'updated_at': self._format_date(issue.get('updated_at')),
                'closed_at': self._format_date(issue.get('closed_at')),
                'author': self._extract_user_info(issue.get('user')),
                'labels': self._extract_labels(issue.get('labels', [])),
                'comments_count': issue.get('comments', 0),
                'body_summary': self._summarize_text(issue.get('body', ''), self.max_summary_length),
                'priority_score': self._calculate_priority_score(issue),
                'has_solution': self._has_solution_indicators(issue),
                'problem_type': self._classify_problem_type(issue),
                'solution_summary': None
            }
            
            # 如果有评论数据，提取解决方案
            if 'comments_data' in issue and issue['comments_data']:
                summary['solution_summary'] = self._extract_solution_from_comments(
                    issue['comments_data']
                )
            
            return summary
            
        except Exception as e:
            log_error(f"提取Issue摘要失败: {e}")
            return self._create_error_summary(issue)
    
    def batch_extract_summaries(self, issues: List[Dict]) -> List[Dict]:
        """
        批量提取Issue摘要
        
        Args:
            issues: Issues列表
            
        Returns:
            List[Dict]: 摘要列表
        """
        summaries = []
        for issue in issues:
            summary = self.extract_issue_summary(issue)
            summaries.append(summary)
        
        log_info(f"批量提取了 {len(summaries)} 个Issue摘要")
        return summaries
    
    def _format_date(self, date_str: Optional[str]) -> Optional[str]:
        """
        格式化日期字符串
        
        Args:
            date_str: ISO格式日期字符串
            
        Returns:
            str: 格式化后的日期
        """
        if not date_str:
            return None
            
        try:
            # 解析ISO格式日期
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d %H:%M')
        except Exception:
            return date_str
    
    def _extract_user_info(self, user: Optional[Dict]) -> Dict:
        """
        提取用户信息
        
        Args:
            user: 用户数据字典
            
        Returns:
            Dict: 用户信息
        """
        if not user:
            return {'login': 'unknown', 'url': ''}
            
        return {
            'login': user.get('login', 'unknown'),
            'url': user.get('html_url', ''),
            'avatar_url': user.get('avatar_url', '')
        }
    
    def _extract_labels(self, labels: List) -> List[str]:
        """
        提取标签信息
        
        Args:
            labels: 标签列表
            
        Returns:
            List[str]: 标签名称列表
        """
        if not labels:
            return []
            
        label_names = []
        for label in labels:
            if isinstance(label, dict):
                name = label.get('name', '')
                if name:
                    label_names.append(name)
            elif isinstance(label, str):
                label_names.append(label)
                
        return label_names
    
    def _summarize_text(self, text: str, max_length: int) -> str:
        """
        文本摘要
        
        Args:
            text: 原始文本
            max_length: 最大长度
            
        Returns:
            str: 摘要文本
        """
        if not text:
            return ''
            
        # 清理文本
        text = self._clean_text(text)
        
        # 如果文本长度小于限制，直接返回
        if len(text) <= max_length:
            return text
            
        # 尝试在句子边界截断
        sentences = re.split(r'[.!?。！？]', text)
        summary = ''
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(summary + sentence) <= max_length - 3:  # 留出"..."的空间
                summary += sentence + '. '
            else:
                break
        
        # 如果没有找到合适的句子边界，直接截断
        if not summary:
            summary = text[:max_length - 3]
            
        return summary.strip() + '...'
    
    def _clean_text(self, text: str) -> str:
        """
        清理文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            str: 清理后的文本
        """
        if not text:
            return ''
            
        # 移除Markdown语法
        text = re.sub(r'```[\s\S]*?```', '[代码块]', text)  # 代码块
        text = re.sub(r'`([^`]+)`', r'\1', text)  # 行内代码
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)  # 链接
        text = re.sub(r'!\[([^\]]*)\]\([^)]+\)', r'[图片: \1]', text)  # 图片
        text = re.sub(r'#{1,6}\s+', '', text)  # 标题
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # 粗体
        text = re.sub(r'\*([^*]+)\*', r'\1', text)  # 斜体
        
        # 清理多余的空白字符
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def _calculate_priority_score(self, issue: Dict) -> float:
        """
        计算Issue优先级分数
        
        Args:
            issue: Issue数据
            
        Returns:
            float: 优先级分数 (0-1)
        """
        score = 0.0
        
        # 状态权重
        if issue.get('state') == 'closed':
            score += 0.3  # 已关闭的问题更有参考价值
            
        # 评论数权重
        comments = issue.get('comments', 0)
        if comments > 0:
            score += min(0.2, comments * 0.02)  # 评论越多，讨论越充分
            
        # 标签权重
        labels = issue.get('labels', [])
        for label in labels:
            label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
            if any(keyword in label_name for keyword in ['bug', 'enhancement', 'question']):
                score += 0.1
            if 'solved' in label_name or 'fixed' in label_name:
                score += 0.2
                
        # 时间权重（较新的问题权重稍高）
        created_at = issue.get('created_at')
        if created_at:
            try:
                created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                days_ago = (datetime.now().replace(tzinfo=created_date.tzinfo) - created_date).days
                if days_ago < 30:
                    score += 0.1
                elif days_ago < 90:
                    score += 0.05
            except Exception:
                pass
                
        return min(1.0, score)
    
    def _has_solution_indicators(self, issue: Dict) -> bool:
        """
        检查Issue是否有解决方案指示
        
        Args:
            issue: Issue数据
            
        Returns:
            bool: 是否有解决方案
        """
        # 检查状态
        if issue.get('state') == 'closed':
            return True
            
        # 检查标签
        labels = issue.get('labels', [])
        for label in labels:
            label_name = label.get('name', '').lower() if isinstance(label, dict) else str(label).lower()
            if any(keyword in label_name for keyword in ['solved', 'fixed', 'resolved', 'duplicate']):
                return True
                
        # 检查标题和内容
        title = issue.get('title', '').lower()
        body = issue.get('body', '').lower()
        
        for keyword in self.solution_keywords:
            if keyword in title or keyword in body:
                return True
                
        return False
    
    def _classify_problem_type(self, issue: Dict) -> str:
        """
        分类问题类型
        
        Args:
            issue: Issue数据
            
        Returns:
            str: 问题类型
        """
        title = issue.get('title', '').lower()
        body = issue.get('body', '').lower()
        labels = [label.get('name', '').lower() if isinstance(label, dict) else str(label).lower() 
                 for label in issue.get('labels', [])]
        
        # 检查标签
        for label in labels:
            if 'bug' in label:
                return 'bug'
            elif 'enhancement' in label or 'feature' in label:
                return 'enhancement'
            elif 'question' in label or 'help' in label:
                return 'question'
            elif 'documentation' in label or 'docs' in label:
                return 'documentation'
                
        # 检查标题和内容
        text = title + ' ' + body
        
        if any(keyword in text for keyword in ['error', 'exception', 'crash', 'fail', 'bug']):
            return 'bug'
        elif any(keyword in text for keyword in ['how to', 'how can', 'question', '?']):
            return 'question'
        elif any(keyword in text for keyword in ['feature', 'enhancement', 'improve', 'add']):
            return 'enhancement'
        else:
            return 'other'
    
    def _extract_solution_from_comments(self, comments: List[Dict]) -> Optional[str]:
        """
        从评论中提取解决方案
        
        Args:
            comments: 评论列表
            
        Returns:
            str: 解决方案摘要
        """
        if not comments:
            return None
            
        solution_comments = []
        
        for comment in comments:
            body = comment.get('body', '').lower()
            
            # 检查是否包含解决方案关键词
            if any(keyword in body for keyword in self.solution_keywords):
                solution_comments.append(comment)
                
        if not solution_comments:
            return None
            
        # 选择最佳解决方案评论（通常是点赞最多的）
        best_comment = max(solution_comments, 
                          key=lambda x: x.get('reactions', {}).get('+1', 0))
        
        solution_text = best_comment.get('body', '')
        return self._summarize_text(solution_text, self.max_solution_length)
    
    def _create_error_summary(self, issue: Dict) -> Dict:
        """
        创建错误摘要（当提取失败时）
        
        Args:
            issue: Issue数据
            
        Returns:
            Dict: 错误摘要
        """
        return {
            'id': issue.get('id', 'unknown'),
            'number': issue.get('number', 'unknown'),
            'title': issue.get('title', 'Unknown Title'),
            'state': issue.get('state', 'unknown'),
            'url': issue.get('html_url', ''),
            'error': 'Failed to extract summary'
        }
    
    def format_summary_for_display(self, summary: Dict, include_solution: bool = True) -> str:
        """
        格式化摘要用于显示
        
        Args:
            summary: 摘要数据
            include_solution: 是否包含解决方案
            
        Returns:
            str: 格式化的摘要文本
        """
        lines = []
        
        # 标题和基本信息
        lines.append(f"📋 #{summary.get('number', 'N/A')} - {summary.get('title', 'No Title')}")
        lines.append(f"🔗 {summary.get('url', 'No URL')}")
        
        # 状态和时间
        state = summary.get('state', 'unknown')
        state_emoji = '✅' if state == 'closed' else '🔄' if state == 'open' else '❓'
        lines.append(f"{state_emoji} 状态: {state.upper()}")
        
        if summary.get('created_at'):
            lines.append(f"📅 创建时间: {summary['created_at']}")
            
        # 作者
        author = summary.get('author', {})
        if author.get('login'):
            lines.append(f"👤 作者: {author['login']}")
            
        # 标签
        labels = summary.get('labels', [])
        if labels:
            lines.append(f"🏷️ 标签: {', '.join(labels)}")
            
        # 问题类型和优先级
        problem_type = summary.get('problem_type', 'unknown')
        priority_score = summary.get('priority_score', 0)
        lines.append(f"📊 类型: {problem_type} | 优先级: {priority_score:.2f}")
        
        # 内容摘要
        body_summary = summary.get('body_summary', '')
        if body_summary:
            lines.append(f"📝 摘要: {body_summary}")
            
        # 解决方案
        if include_solution and summary.get('solution_summary'):
            lines.append(f"💡 解决方案: {summary['solution_summary']}")
            
        return '\n'.join(lines)
    
    def rank_summaries(self, summaries: List[Dict], 
                      query_scores: Optional[List[float]] = None) -> List[Dict]:
        """
        对摘要进行排序
        
        Args:
            summaries: 摘要列表
            query_scores: 查询相似度分数
            
        Returns:
            List[Dict]: 排序后的摘要列表
        """
        def calculate_rank_score(i: int, summary: Dict) -> float:
            score = 0.0
            
            # 查询相似度权重 (40%)
            if query_scores and i < len(query_scores):
                score += query_scores[i] * 0.4
                
            # Issue优先级权重 (30%)
            score += summary.get('priority_score', 0) * 0.3
            
            # 解决方案存在权重 (20%)
            if summary.get('has_solution') or summary.get('solution_summary'):
                score += 0.2
                
            # 状态权重 (10%)
            if summary.get('state') == 'closed':
                score += 0.1
                
            return score
        
        # 计算排序分数并排序
        scored_summaries = []
        for i, summary in enumerate(summaries):
            rank_score = calculate_rank_score(i, summary)
            summary['rank_score'] = rank_score
            scored_summaries.append(summary)
            
        # 按分数降序排序
        scored_summaries.sort(key=lambda x: x['rank_score'], reverse=True)
        
        return scored_summaries

# 便捷函数
def create_summarizer(max_summary_length: int = 300, 
                     max_solution_length: int = 500) -> IssueSummarizer:
    """
    创建摘要提取器
    
    Args:
        max_summary_length: 最大摘要长度
        max_solution_length: 最大解决方案长度
        
    Returns:
        IssueSummarizer: 摘要提取器实例
    """
    return IssueSummarizer(max_summary_length, max_solution_length)