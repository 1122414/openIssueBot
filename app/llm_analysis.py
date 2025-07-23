# -*- coding: utf-8 -*-
"""
LLM 分析模块

在没有匹配的Issue时，使用大语言模型分析源码/文档自动生成解决建议：
1. OpenAI GPT模型集成
2. 提示词工程
3. 上下文构建
4. 结果后处理
"""

import openai
from typing import List, Dict, Optional, Any
import json
from .config import Config
from .utils import log_info, log_error, log_warning

class LLMAnalyzer:
    """
    大语言模型分析器
    
    提供基于LLM的问题分析和解决方案生成功能
    """
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        初始化LLM分析器
        
        Args:
            api_key: OpenAI API密钥
            model: 使用的模型名称
        """
        self.api_key = api_key or Config.OPENAI_API_KEY
        self.model = model or Config.OPENAI_MODEL
        
        if not self.api_key:
            raise ValueError("需要设置OPENAI_API_KEY才能使用LLM分析功能")
            
        # 初始化OpenAI客户端
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # 系统提示词模板
        self.system_prompt = self._create_system_prompt()
        
        log_info(f"LLM分析器初始化完成，使用模型: {self.model}")
    
    def _create_system_prompt(self) -> str:
        """
        创建系统提示词
        
        Returns:
            str: 系统提示词
        """
        return """
你是一个专业的软件开发问题分析专家，特别擅长分析开源项目中的技术问题。

你的任务是：
1. 分析用户提供的错误信息或问题描述
2. 基于提供的上下文信息（如相关Issues、代码片段、文档等）
3. 提供准确、实用的解决方案和建议

回答要求：
- 使用中文回答
- 结构清晰，包含问题分析、可能原因、解决方案
- 提供具体的代码示例或配置建议（如果适用）
- 如果问题复杂，提供多种可能的解决方案
- 保持专业但易懂的语言风格

回答格式：
## 问题分析
[分析问题的本质和可能原因]

## 解决方案
[提供具体的解决步骤]

## 相关建议
[额外的建议或注意事项]
"""
    
    def analyze_problem(self, 
                       query: str, 
                       context: Optional[Dict] = None,
                       max_tokens: int = 1000) -> Dict[str, Any]:
        """
        分析问题并生成解决方案
        
        Args:
            query: 用户问题描述
            context: 上下文信息（相关Issues、代码等）
            max_tokens: 最大生成token数
            
        Returns:
            Dict: 分析结果
        """
        try:
            # 构建用户提示词
            user_prompt = self._build_user_prompt(query, context)
            
            log_info(f"开始LLM分析，查询长度: {len(query)}")
            
            # 调用OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            # 提取回答
            answer = response.choices[0].message.content
            
            # 构建结果
            result = {
                "success": True,
                "answer": answer,
                "model": self.model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "context_provided": context is not None,
                "query_length": len(query)
            }
            
            log_info(f"LLM分析完成，使用tokens: {result['tokens_used']}")
            
            return result
            
        except Exception as e:
            log_error(f"LLM分析失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "answer": self._generate_fallback_response(query)
            }
    
    def _build_user_prompt(self, query: str, context: Optional[Dict] = None) -> str:
        """
        构建用户提示词
        
        Args:
            query: 用户查询
            context: 上下文信息
            
        Returns:
            str: 构建的提示词
        """
        prompt_parts = []
        
        # 添加用户问题
        prompt_parts.append(f"用户问题：\n{query}\n")
        
        # 添加上下文信息
        if context:
            prompt_parts.append("相关上下文信息：")
            
            # 相关Issues
            if context.get('related_issues'):
                prompt_parts.append("\n相关Issues：")
                for i, issue in enumerate(context['related_issues'][:3], 1):  # 最多3个
                    title = issue.get('title', 'No Title')
                    url = issue.get('url', '')
                    summary = issue.get('body_summary', '')
                    state = issue.get('state', 'unknown')
                    
                    prompt_parts.append(f"{i}. [{state.upper()}] {title}")
                    if summary:
                        prompt_parts.append(f"   摘要: {summary}")
                    if url:
                        prompt_parts.append(f"   链接: {url}")
                    prompt_parts.append("")
            
            # 项目信息
            if context.get('project_info'):
                project = context['project_info']
                prompt_parts.append(f"\n项目信息：")
                prompt_parts.append(f"- 项目名称: {project.get('name', 'Unknown')}")
                prompt_parts.append(f"- 描述: {project.get('description', 'No description')}")
                prompt_parts.append(f"- 主要语言: {project.get('language', 'Unknown')}")
                prompt_parts.append("")
            
            # 代码片段
            if context.get('code_snippets'):
                prompt_parts.append("\n相关代码片段：")
                for snippet in context['code_snippets'][:2]:  # 最多2个
                    prompt_parts.append(f"```{snippet.get('language', '')}")
                    prompt_parts.append(snippet.get('content', ''))
                    prompt_parts.append("```\n")
            
            # 错误日志
            if context.get('error_logs'):
                prompt_parts.append("\n错误日志：")
                for log in context['error_logs'][:3]:  # 最多3个
                    prompt_parts.append(f"```")
                    prompt_parts.append(log)
                    prompt_parts.append("```\n")
        
        prompt_parts.append("\n请基于以上信息提供详细的分析和解决方案。")
        
        return "\n".join(prompt_parts)
    
    def _generate_fallback_response(self, query: str) -> str:
        """
        生成备用回答（当LLM调用失败时）
        
        Args:
            query: 用户查询
            
        Returns:
            str: 备用回答
        """
        return f"""
## 问题分析
抱歉，当前无法连接到AI分析服务。基于您的问题描述：
"{query[:200]}..."

## 建议的排查步骤
1. 检查错误信息中的关键词，在项目文档中搜索相关内容
2. 查看项目的Issues页面，搜索类似的问题
3. 检查项目的README和文档，确认环境配置是否正确
4. 尝试查看项目的示例代码或测试用例
5. 考虑在项目的讨论区或社区寻求帮助

## 相关建议
- 提供完整的错误堆栈信息有助于问题定位
- 说明您的环境信息（操作系统、版本等）
- 描述重现问题的具体步骤
"""
    
    def analyze_with_issues(self, 
                           query: str, 
                           related_issues: List[Dict],
                           project_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        基于相关Issues进行分析
        
        Args:
            query: 用户查询
            related_issues: 相关Issues列表
            project_info: 项目信息
            
        Returns:
            Dict: 分析结果
        """
        context = {
            'related_issues': related_issues,
            'project_info': project_info
        }
        
        return self.analyze_problem(query, context)
    
    def generate_solution_summary(self, 
                                 query: str, 
                                 solutions: List[str],
                                 max_tokens: int = 500) -> Dict[str, Any]:
        """
        生成解决方案摘要
        
        Args:
            query: 原始查询
            solutions: 解决方案列表
            max_tokens: 最大token数
            
        Returns:
            Dict: 摘要结果
        """
        try:
            prompt = f"""
用户问题：{query}

已找到以下解决方案：
{chr(10).join([f"{i+1}. {sol}" for i, sol in enumerate(solutions)])}

请将这些解决方案整合成一个清晰、简洁的综合解决方案，突出最重要的步骤。
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个技术文档整理专家，擅长将多个解决方案整合成清晰的指导。"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            
            return {
                "success": True,
                "summary": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            log_error(f"生成解决方案摘要失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "summary": "\n".join([f"• {sol}" for sol in solutions])
            }
    
    def classify_problem_type(self, query: str) -> Dict[str, Any]:
        """
        分类问题类型
        
        Args:
            query: 用户查询
            
        Returns:
            Dict: 分类结果
        """
        try:
            prompt = f"""
请分析以下问题并分类：
"{query}"

请从以下类别中选择最合适的一个：
1. bug - 软件缺陷或错误
2. configuration - 配置问题
3. usage - 使用方法问题
4. environment - 环境设置问题
5. dependency - 依赖问题
6. performance - 性能问题
7. feature - 功能请求
8. other - 其他

只需要返回类别名称，不需要解释。
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=50,
                temperature=0.1
            )
            
            category = response.choices[0].message.content.strip().lower()
            
            return {
                "success": True,
                "category": category,
                "confidence": 0.8  # 简化的置信度
            }
            
        except Exception as e:
            log_error(f"问题分类失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "category": "other",
                "confidence": 0.0
            }
    
    def extract_keywords(self, query: str) -> List[str]:
        """
        提取查询中的关键词
        
        Args:
            query: 用户查询
            
        Returns:
            List[str]: 关键词列表
        """
        try:
            prompt = f"""
从以下问题描述中提取最重要的技术关键词（最多10个）：
"{query}"

请只返回关键词，用逗号分隔，不需要其他解释。
例如：react, useState, error, component
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=100,
                temperature=0.3
            )
            
            keywords_text = response.choices[0].message.content.strip()
            keywords = [kw.strip() for kw in keywords_text.split(',') if kw.strip()]
            
            return keywords[:10]  # 最多返回10个关键词
            
        except Exception as e:
            log_error(f"关键词提取失败: {e}")
            # 简单的备用关键词提取
            import re
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query)
            return list(set(words))[:10]
    
    def validate_solution(self, query: str, solution: str) -> Dict[str, Any]:
        """
        验证解决方案的相关性
        
        Args:
            query: 原始查询
            solution: 解决方案
            
        Returns:
            Dict: 验证结果
        """
        try:
            prompt = f"""
问题：{query}

解决方案：{solution}

请评估这个解决方案对问题的相关性和有用性，用1-10分评分（10分最高）。
只需要返回数字分数，不需要解释。
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=10,
                temperature=0.1
            )
            
            try:
                score = float(response.choices[0].message.content.strip())
                score = max(1, min(10, score))  # 确保在1-10范围内
            except ValueError:
                score = 5.0  # 默认分数
            
            return {
                "success": True,
                "relevance_score": score,
                "is_relevant": score >= 6.0
            }
            
        except Exception as e:
            log_error(f"解决方案验证失败: {e}")
            return {
                "success": False,
                "error": str(e),
                "relevance_score": 5.0,
                "is_relevant": True
            }

# 便捷函数
def create_llm_analyzer(api_key: Optional[str] = None, 
                       model: Optional[str] = None) -> LLMAnalyzer:
    """
    创建LLM分析器
    
    Args:
        api_key: OpenAI API密钥
        model: 模型名称
        
    Returns:
        LLMAnalyzer: LLM分析器实例
    """
    return LLMAnalyzer(api_key, model)

def quick_analyze(query: str, 
                 related_issues: Optional[List[Dict]] = None) -> str:
    """
    快速分析问题（便捷函数）
    
    Args:
        query: 用户查询
        related_issues: 相关Issues
        
    Returns:
        str: 分析结果
    """
    try:
        analyzer = create_llm_analyzer()
        
        if related_issues:
            result = analyzer.analyze_with_issues(query, related_issues)
        else:
            result = analyzer.analyze_problem(query)
            
        if result['success']:
            return result['answer']
        else:
            return result.get('answer', '分析失败，请稍后重试。')
            
    except Exception as e:
        log_error(f"快速分析失败: {e}")
        return "抱歉，当前无法提供AI分析服务。请尝试手动搜索相关文档或Issues。"