# -*- coding: utf-8 -*-
"""
Flask Web应用模块

提供用户友好的Web界面：
1. 问题搜索页面
2. 结果展示页面
3. 配置管理页面
4. API接口
5. 实时搜索功能
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_cors import CORS
import os
import json
from datetime import datetime
from typing import Dict, Any, Optional

from .config import Config
from .issue_search import IssueSearchEngine, create_search_engine
from .utils import log_info, log_error, log_warning, validate_github_repo, format_time_ago

# 创建Flask应用
app = Flask(__name__, 
           template_folder='../templates',
           static_folder='../static')

# 配置
app.config['SECRET_KEY'] = Config.FLASK_SECRET_KEY or 'openissuebot-secret-key'
app.config['JSON_AS_ASCII'] = False

# 启用CORS
CORS(app)

# 全局搜索引擎实例
search_engine: Optional[IssueSearchEngine] = None

# ==================== 辅助函数 ====================

def get_search_engine() -> Optional[IssueSearchEngine]:
    """
    获取搜索引擎实例
    
    Returns:
        Optional[IssueSearchEngine]: 搜索引擎实例
    """
    global search_engine
    
    if search_engine is None:
        try:
            search_engine = create_search_engine()
            log_info("搜索引擎实例创建成功")
        except Exception as e:
            log_error(f"创建搜索引擎失败: {e}")
            return None
    
    return search_engine

def format_search_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    格式化搜索结果用于前端显示
    
    Args:
        result: 原始搜索结果
        
    Returns:
        Dict: 格式化后的结果
    """
    # 检查输入类型
    if not isinstance(result, dict):
        log_error(f"format_search_result收到非字典类型数据: {type(result)}, 值: {result}")
        return {
            'error': 'Invalid result format',
            'original_data': str(result),
            'similarity_score': 0.0,
            'similarity_percentage': '0.0%',
            'similarity_level': 'error',
            'similarity_label': '数据错误'
        }
    
    formatted = result.copy()
    
    # 格式化时间
    if 'created_at' in formatted:
        try:
            created_time = datetime.fromisoformat(formatted['created_at'].replace('Z', '+00:00'))
            formatted['created_at_formatted'] = format_time_ago(created_time)
        except Exception:
            formatted['created_at_formatted'] = '未知时间'
    
    if 'updated_at' in formatted:
        try:
            updated_time = datetime.fromisoformat(formatted['updated_at'].replace('Z', '+00:00'))
            formatted['updated_at_formatted'] = format_time_ago(updated_time)
        except Exception:
            formatted['updated_at_formatted'] = '未知时间'
    
    # 格式化相似度分数
    if 'similarity_score' in formatted:
        score = formatted['similarity_score']
        formatted['similarity_percentage'] = f"{score * 100:.1f}%"
        
        # 相似度等级
        if score >= 0.8:
            formatted['similarity_level'] = 'high'
            formatted['similarity_label'] = '高度相似'
        elif score >= 0.6:
            formatted['similarity_level'] = 'medium'
            formatted['similarity_label'] = '中等相似'
        elif score >= 0.3:
            formatted['similarity_level'] = 'low'
            formatted['similarity_label'] = '低度相似'
        else:
            formatted['similarity_level'] = 'very-low'
            formatted['similarity_label'] = '相似度很低'
    
    # 截断长文本
    if 'body' in formatted and formatted['body']:
        if len(formatted['body']) > 300:
            formatted['body_preview'] = formatted['body'][:300] + '...'
        else:
            formatted['body_preview'] = formatted['body']
    
    # 格式化标签
    if 'labels' in formatted and isinstance(formatted['labels'], list):
        formatted_labels = []
        for label in formatted['labels']:
            if isinstance(label, dict):
                # GitHub API返回的标签对象
                formatted_labels.append({
                    'name': label.get('name', ''),
                    'color': label.get('color', 'gray')
                })
            elif isinstance(label, str):
                # 简单的字符串标签
                formatted_labels.append({
                    'name': label,
                    'color': 'gray'
                })
        formatted['labels_formatted'] = formatted_labels
    
    # 格式化作者信息
    if 'author' in formatted:
        author = formatted['author']
        if isinstance(author, dict):
            # 作者是字典对象，保持原样
            pass
        elif isinstance(author, str):
            # 作者是字符串，转换为字典格式
            formatted['author'] = {
                'login': author,
                'url': f'https://github.com/{author}',
                'avatar_url': f'https://github.com/{author}.png'
            }
        else:
            # 其他类型，设置默认值
            formatted['author'] = {
                'login': 'unknown',
                'url': '',
                'avatar_url': ''
            }
    
    return formatted

def get_config_status() -> Dict[str, Any]:
    """
    获取配置状态
    
    Returns:
        Dict: 配置状态信息
    """
    # 检查GitHub Token是否有效
    github_token_valid = False
    github_token_error = None
    if Config.GITHUB_TOKEN:
        try:
            import requests
            headers = {
                "Authorization": f"token {Config.GITHUB_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
                "User-Agent": "OpenIssueBot/1.0"
            }
            response = requests.get("https://api.github.com/user", headers=headers, timeout=10)
            if response.status_code == 200:
                github_token_valid = True
            else:
                github_token_error = f"GitHub Token无效 (HTTP {response.status_code})"
        except Exception as e:
            github_token_error = f"GitHub Token验证失败: {str(e)}"
            github_token_valid = False
    else:
        github_token_error = "GitHub Token未配置"
    
    # 检查GitHub仓库格式是否有效
    github_repo_valid = False
    github_repo_error = None
    if Config.GITHUB_REPO:
        # 简单检查格式是否为 owner/repo
        parts = Config.GITHUB_REPO.split('/')
        if len(parts) == 2 and all(part.strip() for part in parts):
            github_repo_valid = True
            
            # 如果Token有效且仓库格式正确，进一步验证仓库是否可访问
            if github_token_valid:
                try:
                    import requests
                    headers = {
                        "Authorization": f"token {Config.GITHUB_TOKEN}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "OpenIssueBot/1.0"
                    }
                    response = requests.get(f"https://api.github.com/repos/{Config.GITHUB_REPO}", headers=headers, timeout=10)
                    if response.status_code == 200:
                        github_repo_valid = True
                    else:
                        github_repo_valid = False
                        github_repo_error = f"仓库不存在或无访问权限 (HTTP {response.status_code})"
                except Exception as e:
                    github_repo_valid = False
                    github_repo_error = f"仓库验证失败: {str(e)}"
        else:
            github_repo_error = "仓库格式无效，应为 owner/repo 格式"
    else:
        github_repo_error = "GitHub仓库未配置"
    
    # 检查当前LLM供应商的API Key是否有效
    llm_provider = getattr(Config, 'LLM_PROVIDER', 'openai')
    openai_api_valid = False
    openai_api_error = None
    
    # 只有当选择OpenAI作为LLM供应商时才验证OpenAI API Key
    if llm_provider == 'openai':
        if Config.OPENAI_API_KEY:
            try:
                import openai
                # 设置API Key
                openai.api_key = Config.OPENAI_API_KEY
                
                # 尝试调用API验证
                response = openai.models.list()
                if response and hasattr(response, 'data'):
                    openai_api_valid = True
                else:
                    openai_api_error = "OpenAI API响应异常"
            except Exception as e:
                openai_api_error = f"OpenAI API Key验证失败: {str(e)}"
                openai_api_valid = False
        else:
            openai_api_error = "OpenAI API Key未配置"
    else:
        # 如果不是OpenAI供应商，则不显示OpenAI相关错误
        openai_api_valid = True  # 设为True以避免显示错误状态
        openai_api_error = None
    
    return {
        'github_token': bool(Config.GITHUB_TOKEN),
        'github_repo': bool(Config.GITHUB_REPO),
        'github_token_configured': github_token_valid,
        'github_repo_configured': github_repo_valid,
        'github_repo_valid': github_repo_valid,
        'openai_api_key': bool(Config.OPENAI_API_KEY),
        'openai_configured': openai_api_valid,
        'openai_api_key_configured': openai_api_valid,
        'search_engine_ready': search_engine is not None and getattr(search_engine, 'is_index_loaded', False),
        'use_local_embedding': Config.USE_LOCAL_EMBEDDING,
        'cache_dir': Config.CACHE_DIR,
        'flask_host': Config.FLASK_HOST,
        'flask_port': Config.FLASK_PORT,
        # 错误信息
        'github_token_error': github_token_error,
        'github_repo_error': github_repo_error,
        'openai_api_error': openai_api_error
    }

# ==================== 路由定义 ====================

@app.route('/')
def index():
    """
    主页
    """
    config_status = get_config_status()
    return render_template('index.html', config_status=config_status)

@app.route('/search')
def search_page():
    """
    搜索页面
    """
    config_status = get_config_status()
    return render_template('search.html', config_status=config_status)

@app.route('/config')
def config_page():
    """
    配置页面
    """
    config_status = get_config_status()
    
    # 获取当前配置信息
    current_config = {
        'github_token': Config.GITHUB_TOKEN,
        'github_repo': Config.GITHUB_REPO,
        'llm_provider': getattr(Config, 'LLM_PROVIDER', 'openai'),
        'openai_api_key': Config.OPENAI_API_KEY,
        'openai_model': getattr(Config, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
        'zhipu_api_key': getattr(Config, 'ZHIPU_API_KEY', ''),
        'zhipu_model': getattr(Config, 'ZHIPU_MODEL', 'glm-4'),
        'qwen_api_key': getattr(Config, 'QWEN_API_KEY', ''),
        'qwen_model': getattr(Config, 'QWEN_MODEL', 'qwen-turbo'),
        'baidu_api_key': getattr(Config, 'BAIDU_API_KEY', ''),
        'baidu_secret_key': getattr(Config, 'BAIDU_SECRET_KEY', ''),
        'baidu_model': getattr(Config, 'BAIDU_MODEL', 'ernie-bot-turbo'),
        'deepseek_api_key': getattr(Config, 'DEEPSEEK_API_KEY', ''),
        'deepseek_model': getattr(Config, 'DEEPSEEK_MODEL', 'deepseek-chat'),
        'embedding_provider': getattr(Config, 'EMBEDDING_PROVIDER', 'local'),
        'embedding_model': getattr(Config, 'EMBEDDING_MODEL', 'all-MiniLM-L6-v2'),
        'zhipu_embedding_model': getattr(Config, 'ZHIPU_EMBEDDING_MODEL', 'embedding-2'),
        'zhipu_embedding_api_key': getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', ''),
        'qwen_embedding_model': getattr(Config, 'QWEN_EMBEDDING_MODEL', 'text-embedding-v1'),
        'qwen_embedding_api_key': getattr(Config, 'QWEN_EMBEDDING_API_KEY', ''),
        'baidu_embedding_model': getattr(Config, 'BAIDU_EMBEDDING_MODEL', 'embedding-v1'),
        'baidu_embedding_api_key': getattr(Config, 'BAIDU_EMBEDDING_API_KEY', ''),
        'baidu_embedding_secret_key': getattr(Config, 'BAIDU_EMBEDDING_SECRET_KEY', ''),
        'openai_embedding_model': getattr(Config, 'OPENAI_EMBEDDING_MODEL', 'text-embedding-3-small'),
        'openai_embedding_api_key': getattr(Config, 'OPENAI_EMBEDDING_API_KEY', ''),
        'use_local_embedding': Config.USE_LOCAL_EMBEDDING,
        'max_issues_per_fetch': getattr(Config, 'MAX_ISSUES_PER_FETCH', 100),
        'similarity_threshold': getattr(Config, 'SIMILARITY_THRESHOLD', 0.3),
        'cache_ttl': getattr(Config, 'CACHE_TTL', 3600),
        'flask_host': Config.FLASK_HOST,
        'flask_port': Config.FLASK_PORT,
        'flask_debug': Config.FLASK_DEBUG
    }
    
    # 扩展config_status以包含更多状态信息
    config_status.update({
        'openai_configured': bool(Config.OPENAI_API_KEY),
        'zhipu_configured': bool(Config.ZHIPU_API_KEY),
        'qwen_configured': bool(Config.QWEN_API_KEY),
        'baidu_configured': bool(Config.BAIDU_API_KEY and Config.BAIDU_SECRET_KEY),
        'deepseek_configured': bool(Config.DEEPSEEK_API_KEY),
        'search_engine_ready': search_engine is not None and getattr(search_engine, 'is_index_loaded', False),
        'all_configured': bool(Config.GITHUB_TOKEN) and bool(Config.GITHUB_REPO)
    })
    
    return render_template('config.html', 
                         config_status=config_status, 
                         current_config=current_config)

@app.route('/about')
def about_page():
    """
    关于页面
    """
    # 获取项目统计信息
    project_stats = {
        'total_issues': 0,
        'search_count': 0,
        'avg_response_time': 0,
        'accuracy_rate': 95
    }
    
    # 如果搜索引擎已初始化，获取实际统计信息
    engine = get_search_engine()
    if engine and hasattr(engine, 'get_stats'):
        try:
            stats = engine.get_stats()
            if stats:
                project_stats.update({
                    'total_issues': stats.get('total_issues', 0),
                    'search_count': stats.get('search_count', 0),
                    'avg_response_time': stats.get('avg_response_time', 0),
                    'accuracy_rate': stats.get('accuracy_rate', 95)
                })
        except Exception as e:
            log_error(f"获取项目统计信息失败: {e}")
    
    return render_template('about.html', project_stats=project_stats)

# ==================== API接口 ====================

@app.route('/api/search', methods=['POST'])
def api_search():
    """
    搜索API接口
    
    Returns:
        JSON: 搜索结果
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': '缺少查询参数'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        # 获取搜索参数
        max_results = min(data.get('max_results', 5), 20)  # 限制最大结果数
        search_type = data.get('search_type', 'vector')  # vector, keyword, hybrid
        similarity_threshold = data.get('similarity_threshold', 0.3)
        recall_threshold = data.get('recall_threshold', 0.0)  # 召回阈值，默认为0
        include_llm = data.get('include_llm_analysis', True)
        
        # 获取搜索引擎
        engine = get_search_engine()
        if not engine:
            return jsonify({
                'success': False,
                'error': '搜索引擎未初始化，请检查配置'
            }), 500
        
        # 初始化搜索引擎
        if not engine.is_index_loaded:
            log_info("正在初始化搜索引擎...")
            if not engine.initialize():
                return jsonify({
                    'success': False,
                    'error': '搜索引擎初始化失败，请检查GitHub配置和网络连接'
                }), 500
        
        # 执行搜索
        if search_type == 'hybrid':
            result = engine.hybrid_search(query, max_results)
        elif search_type == 'keyword':
            keyword_results = engine.search_by_keywords([query], max_results)
            # 调试信息：记录keyword搜索结果
            log_info(f"keyword搜索原始结果类型: {type(keyword_results)}, 长度: {len(keyword_results) if hasattr(keyword_results, '__len__') else 'N/A'}")
            if keyword_results and len(keyword_results) > 0:
                log_info(f"keyword第一个结果类型: {type(keyword_results[0])}, 内容预览: {str(keyword_results[0])[:200]}")
            
            result = {
                'success': True,
                'query': query,
                'total_found': len(keyword_results),
                'results': keyword_results,
                'search_strategy': 'keyword',
                'timestamp': datetime.now().isoformat()
            }
        else:  # vector search
            result = engine.search(query, max_results, similarity_threshold, include_llm)
        
        # 调试信息：记录原始结果结构
        log_info(f"搜索原始结果类型: {type(result)}, 键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
        if result.get('success') and 'results' in result:
            log_info(f"results字段类型: {type(result['results'])}, 长度: {len(result['results']) if hasattr(result['results'], '__len__') else 'N/A'}")
            if result['results'] and len(result['results']) > 0:
                log_info(f"第一个结果类型: {type(result['results'][0])}, 内容预览: {str(result['results'][0])[:200]}")
            
            # 格式化搜索结果并应用召回阈值过滤
            formatted_results = []
            filtered_count = 0
            for i, r in enumerate(result['results']):
                try:
                    formatted_result = format_search_result(r)
                    
                    # 应用召回阈值过滤
                    similarity_score = formatted_result.get('similarity_score', 0)
                    if similarity_score >= recall_threshold:
                        formatted_results.append(formatted_result)
                    else:
                        filtered_count += 1
                        log_info(f"结果被召回阈值过滤: 相似度{similarity_score:.3f} < 阈值{recall_threshold}")
                        
                except Exception as format_error:
                    log_error(f"格式化第{i}个结果时出错: {format_error}, 原始数据: {r}")
                    # 添加错误结果以便调试
                    formatted_results.append({
                        'error': f'格式化失败: {str(format_error)}',
                        'original_data': str(r),
                        'index': i
                    })
            
            result['results'] = formatted_results
            result['filtered_by_recall_threshold'] = filtered_count
            result['total_after_filter'] = len(formatted_results)
        
        log_info(f"搜索完成: {query[:50]}... -> {result.get('total_found', 0)} 个结果")
        return jsonify(result)
        
    except Exception as e:
        log_error(f"搜索API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'搜索过程中发生错误: {str(e)}'
        }), 500

@app.route('/api/initialize', methods=['POST'])
def api_initialize():
    """
    初始化搜索引擎API
    
    Returns:
        JSON: 初始化结果
    """
    try:
        data = request.get_json() or {}
        force_refresh = data.get('force_refresh', False)
        
        engine = get_search_engine()
        if not engine:
            return jsonify({
                'success': False,
                'error': '无法创建搜索引擎实例'
            }), 500
        
        log_info(f"开始初始化搜索引擎 (force_refresh={force_refresh})")
        success = engine.initialize(force_refresh=force_refresh)
        
        if success:
            stats = engine.get_stats()
            return jsonify({
                'success': True,
                'message': '搜索引擎初始化成功',
                'stats': stats
            })
        else:
            return jsonify({
                'success': False,
                'error': '搜索引擎初始化失败，请检查配置和网络连接'
            }), 500
            
    except Exception as e:
        log_error(f"初始化API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'初始化过程中发生错误: {str(e)}'
        }), 500

@app.route('/api/stats')
def api_stats():
    """
    获取统计信息API
    
    Returns:
        JSON: 统计信息
    """
    try:
        engine = get_search_engine()
        if not engine:
            return jsonify({
                'success': False,
                'error': '搜索引擎未初始化'
            }), 500
        
        stats = engine.get_stats()
        config_status = get_config_status()
        
        return jsonify({
            'success': True,
            'stats': stats,
            'config': config_status,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        log_error(f"统计API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'获取统计信息失败: {str(e)}'
        }), 500

@app.route('/api/clear_cache', methods=['POST'])
def api_clear_cache():
    """
    清除缓存API
    
    Returns:
        JSON: 清除结果
    """
    try:
        engine = get_search_engine()
        if engine:
            engine.clear_cache()
            
            # 重置全局搜索引擎实例
            global search_engine
            search_engine = None
        
        return jsonify({
            'success': True,
            'message': '缓存已清除'
        })
        
    except Exception as e:
        log_error(f"清除缓存API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'清除缓存失败: {str(e)}'
        }), 500

@app.route('/api/direct_ai_answer', methods=['POST'])
def api_direct_ai_answer():
    """
    直接AI回答API（无搜索结果时使用）
    
    Returns:
        JSON: AI回答结果
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data:
            return jsonify({
                'success': False,
                'error': '缺少查询参数'
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        # 获取LLM服务
        from app.llm_service import get_llm_service
        llm_service = get_llm_service()
        
        if not llm_service:
            return jsonify({
                'success': False,
                'error': 'LLM服务未配置，请检查API密钥设置'
            }), 500
        
        # 构建直接回答的提示词
        prompt = f"""
你是一个专业的技术助手。用户提出了以下问题，但在知识库中没有找到相关的搜索结果。
请基于你的知识和经验，为用户提供有用的回答和建议。

用户问题：{query}

请提供：
1. 对问题的分析和理解
2. 可能的解决方案或建议
3. 相关的最佳实践
4. 如果需要，提供进一步学习的方向

请用中文回答，内容要专业、准确、有帮助。
"""
        
        # 调用LLM生成回答
        response = llm_service.generate_response(prompt)
        
        if response.get('success'):
            return jsonify({
                'success': True,
                'answer': response.get('answer', ''),
                'provider': response.get('provider', 'AI'),
                'model': response.get('model', ''),
                'tokens_used': response.get('tokens_used', 0),
                'mode': 'direct'
            })
        else:
            return jsonify({
                'success': False,
                'error': response.get('error', 'AI回答生成失败')
            }), 500
            
    except Exception as e:
        log_error(f"直接AI回答API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'AI回答过程中发生错误: {str(e)}'
        }), 500

@app.route('/api/rag_analysis', methods=['POST'])
def api_rag_analysis():
    """
    RAG检索增强分析API（有搜索结果时使用）
    
    Returns:
        JSON: RAG分析结果
    """
    try:
        data = request.get_json()
        
        if not data or 'query' not in data or 'results' not in data:
            return jsonify({
                'success': False,
                'error': '缺少必要参数'
            }), 400
        
        query = data['query'].strip()
        results = data['results']
        
        if not query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        if not results or len(results) == 0:
            return jsonify({
                'success': False,
                'error': '没有搜索结果可供分析'
            }), 400
        
        # 获取LLM服务
        from app.llm_service import get_llm_service
        llm_service = get_llm_service()
        
        if not llm_service:
            return jsonify({
                'success': False,
                'error': 'LLM服务未配置，请检查API密钥设置'
            }), 500
        
        # 构建RAG增强的提示词
        context = "\n\n".join([
            f"结果 {i+1}:\n标题: {result.get('title', '')}\n内容: {result.get('body_preview', '')}\n相似度: {result.get('similarity_score', 0):.2f}\nURL: {result.get('url', '')}"
            for i, result in enumerate(results[:5])  # 限制最多5个结果
        ])
        
        prompt = f"""
你是一个专业的技术助手。用户提出了问题，我已经从知识库中检索到了相关的信息。
请基于这些检索结果，为用户提供准确、有用的回答。

用户问题：{query}

检索到的相关信息：
{context}

请基于上述检索结果：
1. 分析用户问题的核心需求
2. 结合检索结果提供具体的解决方案
3. 引用相关的信息来源
4. 如果检索结果不完全匹配，请说明并提供额外建议

请用中文回答，内容要专业、准确、有帮助。在回答中适当引用检索结果的内容。
"""
        
        # 调用LLM生成RAG增强回答
        response = llm_service.generate_response(prompt)
        
        if response.get('success'):
            return jsonify({
                'success': True,
                'answer': response.get('answer', ''),
                'provider': response.get('provider', 'AI'),
                'model': response.get('model', ''),
                'tokens_used': response.get('tokens_used', 0),
                'mode': 'rag',
                'context_results': len(results)
            })
        else:
            return jsonify({
                'success': False,
                'error': response.get('error', 'RAG分析生成失败')
            }), 500
            
    except Exception as e:
        log_error(f"RAG分析API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'RAG分析过程中发生错误: {str(e)}'
        }), 500

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    """
    配置管理API
    
    Returns:
        JSON: 配置信息或更新结果
    """
    if request.method == 'GET':
        # 获取配置
        return jsonify({
            'success': True,
            'config': get_config_status()
        })
    
    elif request.method == 'POST':
        # 更新配置（注意：这里只是演示，实际应用中需要更安全的配置管理）
        try:
            data = request.get_json()
            
            # 验证配置
            if 'github_repo' in data:
                repo = data['github_repo'].strip()
                if repo and not validate_github_repo(repo):
                    return jsonify({
                        'success': False,
                        'error': 'GitHub仓库格式无效，应为 owner/repo 格式'
                    }), 400
            
            # 验证嵌入模型提供商
            if 'embedding_provider' in data:
                provider = data['embedding_provider']
                valid_providers = ['local', 'openai', 'zhipu', 'qwen', 'baidu']
                if provider not in valid_providers:
                    return jsonify({
                        'success': False,
                        'error': f'无效的嵌入模型提供商: {provider}'
                    }), 400
            
            # 验证LLM提供商
            if 'llm_provider' in data:
                provider = data['llm_provider']
                valid_providers = ['openai', 'zhipu', 'qwen', 'baidu']
                if provider not in valid_providers:
                    return jsonify({
                        'success': False,
                        'error': f'无效的LLM提供商: {provider}'
                    }), 400
            
            # 保存配置到.env文件
            env_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
            
            # 读取现有的.env文件内容
            env_vars = {}
            if os.path.exists(env_file_path):
                with open(env_file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key.strip()] = value.strip()
            
            # 更新配置
            config_mapping = {
                'github_token': 'GITHUB_TOKEN',
                'github_repo': 'GITHUB_REPO',
                'llm_provider': 'LLM_PROVIDER',
                'openai_api_key': 'OPENAI_API_KEY',
                'openai_model': 'OPENAI_MODEL',
                'zhipu_api_key': 'ZHIPU_API_KEY',
                'zhipu_model': 'ZHIPU_MODEL',
                'qwen_api_key': 'QWEN_API_KEY',
                'qwen_model': 'QWEN_MODEL',
                'baidu_api_key': 'BAIDU_API_KEY',
                'baidu_secret_key': 'BAIDU_SECRET_KEY',
                'baidu_model': 'BAIDU_MODEL',
                'deepseek_api_key': 'DEEPSEEK_API_KEY',
                'deepseek_model': 'DEEPSEEK_MODEL',
                'embedding_provider': 'EMBEDDING_PROVIDER',
                'embedding_model': 'EMBEDDING_MODEL',
                'openai_embedding_model': 'OPENAI_EMBEDDING_MODEL',
                'openai_embedding_api_key': 'OPENAI_EMBEDDING_API_KEY',
                'zhipu_embedding_model': 'ZHIPU_EMBEDDING_MODEL',
                'zhipu_embedding_api_key': 'ZHIPU_EMBEDDING_API_KEY',
                'qwen_embedding_model': 'QWEN_EMBEDDING_MODEL',
                'qwen_embedding_api_key': 'QWEN_EMBEDDING_API_KEY',
                'baidu_embedding_model': 'BAIDU_EMBEDDING_MODEL',
                'baidu_embedding_api_key': 'BAIDU_EMBEDDING_API_KEY',
                'baidu_embedding_secret_key': 'BAIDU_EMBEDDING_SECRET_KEY',
                'use_local_embedding': 'USE_LOCAL_EMBEDDING',
                'max_issues_per_fetch': 'MAX_ISSUES_PER_FETCH',
                'similarity_threshold': 'SIMILARITY_THRESHOLD',
                'cache_ttl': 'CACHE_TTL',
                'flask_host': 'FLASK_HOST',
                'flask_port': 'FLASK_PORT',
                'flask_debug': 'FLASK_DEBUG'
            }
            
            # 更新环境变量字典
            for form_key, env_key in config_mapping.items():
                if form_key in data:
                    value = data[form_key]
                    if isinstance(value, bool):
                        env_vars[env_key] = 'true' if value else 'false'
                    else:
                        env_vars[env_key] = str(value)
            
            # 写回.env文件
            with open(env_file_path, 'w', encoding='utf-8') as f:
                f.write('# OpenIssueBot Configuration\n')
                f.write('# This file is automatically generated by the web interface\n\n')
                
                # 按类别组织配置
                f.write('# GitHub Configuration\n')
                for key in ['GITHUB_TOKEN', 'GITHUB_REPO']:
                    if key in env_vars:
                        f.write(f'{key}={env_vars[key]}\n')
                
                f.write('\n# LLM Configuration\n')
                for key in ['LLM_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_MODEL', 'ZHIPU_API_KEY', 'ZHIPU_MODEL', 
                           'QWEN_API_KEY', 'QWEN_MODEL', 'BAIDU_API_KEY', 'BAIDU_SECRET_KEY', 'BAIDU_MODEL',
                           'DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL']:
                    if key in env_vars:
                        f.write(f'{key}={env_vars[key]}\n')
                
                f.write('\n# Embedding Configuration\n')
                for key in ['EMBEDDING_PROVIDER', 'EMBEDDING_MODEL', 'OPENAI_EMBEDDING_MODEL', 'OPENAI_EMBEDDING_API_KEY',
                           'ZHIPU_EMBEDDING_MODEL', 'ZHIPU_EMBEDDING_API_KEY', 'QWEN_EMBEDDING_MODEL', 'QWEN_EMBEDDING_API_KEY',
                           'BAIDU_EMBEDDING_MODEL', 'BAIDU_EMBEDDING_API_KEY', 'BAIDU_EMBEDDING_SECRET_KEY',
                           'USE_LOCAL_EMBEDDING']:
                    if key in env_vars:
                        f.write(f'{key}={env_vars[key]}\n')
                
                f.write('\n# Search Configuration\n')
                for key in ['MAX_ISSUES_PER_FETCH', 'SIMILARITY_THRESHOLD', 'CACHE_TTL']:
                    if key in env_vars:
                        f.write(f'{key}={env_vars[key]}\n')
                
                f.write('\n# Web Server Configuration\n')
                for key in ['FLASK_HOST', 'FLASK_PORT', 'FLASK_DEBUG']:
                    if key in env_vars:
                        f.write(f'{key}={env_vars[key]}\n')
                
                # 写入其他未分类的配置
                f.write('\n# Other Configuration\n')
                for key, value in env_vars.items():
                    if key not in ['GITHUB_TOKEN', 'GITHUB_REPO', 'LLM_PROVIDER', 'OPENAI_API_KEY', 'OPENAI_MODEL',
                                  'ZHIPU_API_KEY', 'ZHIPU_MODEL', 'QWEN_API_KEY', 'QWEN_MODEL', 'BAIDU_API_KEY',
                                  'BAIDU_SECRET_KEY', 'BAIDU_MODEL', 'DEEPSEEK_API_KEY', 'DEEPSEEK_MODEL',
                                  'EMBEDDING_PROVIDER', 'EMBEDDING_MODEL', 'OPENAI_EMBEDDING_MODEL', 'OPENAI_EMBEDDING_API_KEY',
                                  'ZHIPU_EMBEDDING_MODEL', 'ZHIPU_EMBEDDING_API_KEY', 'QWEN_EMBEDDING_MODEL', 'QWEN_EMBEDDING_API_KEY',
                                  'BAIDU_EMBEDDING_MODEL', 'BAIDU_EMBEDDING_API_KEY', 'BAIDU_EMBEDDING_SECRET_KEY',
                                  'USE_LOCAL_EMBEDDING', 'MAX_ISSUES_PER_FETCH', 'SIMILARITY_THRESHOLD',
                                  'CACHE_TTL', 'FLASK_HOST', 'FLASK_PORT', 'FLASK_DEBUG']:
                        f.write(f'{key}={value}\n')
            
            # 重新加载配置
            try:
                from .config import Config
                Config.reload_config()
                log_info("配置已重新加载")
            except Exception as reload_error:
                log_error(f"重新加载配置失败: {reload_error}")
            
            return jsonify({
                'success': True,
                'message': '配置保存成功！页面将自动刷新以应用新配置。'
            })
            
        except Exception as e:
            log_error(f"配置API错误: {e}")
            return jsonify({
                'success': False,
                'error': f'配置更新失败: {str(e)}'
            }), 500

@app.route('/api/chat', methods=['POST'])
def api_chat():
    """
    基于搜索结果的AI对话分析API
    
    Returns:
        JSON: AI分析结果
    """
    try:
        data = request.get_json() or {}
        query = data.get('query', '').strip()
        search_results = data.get('search_results', [])
        
        if not query:
            return jsonify({
                'success': False,
                'error': '查询内容不能为空'
            }), 400
        
        # 获取LLM配置
        provider = Config.LLM_PROVIDER
        if not provider:
            return jsonify({
                'success': False,
                'error': '未配置LLM提供商，请先在配置页面设置LLM'
            }), 400
        
        # 导入LLM分析器
        from .llm_analysis import LLMAnalyzer
        
        # 创建LLM分析器实例
        analyzer = LLMAnalyzer(Config)
        
        # 执行AI分析
        analysis_result = analyzer.analyze_search_results(query, search_results)
        
        if analysis_result['success']:
            return jsonify({
                'success': True,
                'analysis': analysis_result.get('answer', analysis_result.get('analysis', '')),
                'provider': provider,
                'model': analysis_result.get('model', ''),
                'tokens_used': analysis_result.get('tokens_used', 0),
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'error': analysis_result.get('error', 'AI分析失败')
            }), 500
            
    except Exception as e:
        log_error(f"对话API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'AI分析过程中发生错误: {str(e)}'
        }), 500

@app.route('/api/test-llm', methods=['POST'])
def api_test_llm():
    """
    测试LLM连接API
    
    Returns:
        JSON: 测试结果
    """
    try:
        data = request.get_json() or {}
        provider = data.get('provider') or Config.LLM_PROVIDER
        api_key = data.get('api_key')
        model = data.get('model')
        
        if not provider:
            return jsonify({
                'success': False,
                'error': '缺少LLM提供商参数'
            }), 400
        
        # 根据提供商获取相应的API Key和模型
        if provider == 'openai':
            api_key = api_key or Config.OPENAI_API_KEY
            model = model or Config.OPENAI_MODEL
        elif provider == 'zhipu':
            api_key = api_key or Config.ZHIPU_API_KEY
            model = model or Config.ZHIPU_MODEL
        elif provider == 'qwen':
            api_key = api_key or Config.QWEN_API_KEY
            model = model or Config.QWEN_MODEL
        elif provider == 'baidu':
            api_key = api_key or Config.BAIDU_API_KEY
            model = model or Config.BAIDU_MODEL
        else:
            return jsonify({
                'success': False,
                'error': f'不支持的LLM提供商: {provider}'
            }), 400
        
        if not api_key:
            return jsonify({
                'success': False,
                'error': f'{provider} LLM需要API Key'
            }), 400
        
        # 测试LLM连接
        from .llm_analysis import LLMAnalyzer
        try:
            # 创建临时的LLM分析器进行测试
            if provider == 'openai':
                import openai
                client = openai.OpenAI(api_key=api_key)
                # 发送简单的测试请求
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=10
                )
                if response.choices and response.choices[0].message:
                    return jsonify({
                        'success': True,
                        'message': f'{provider} LLM连接测试成功',
                        'provider': provider,
                        'model': model
                    })
            elif provider == 'zhipu':
                # 智谱AI API测试
                import requests
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                test_data = {
                    'model': model,
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                    'max_tokens': 10
                }
                response = requests.post('https://open.bigmodel.cn/api/paas/v4/chat/completions', 
                                       headers=headers, json=test_data, timeout=10)
                if response.status_code == 200:
                    return jsonify({
                        'success': True,
                        'message': f'{provider} LLM连接测试成功',
                        'provider': provider,
                        'model': model
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': f'智谱AI API验证失败: {response.text}'
                    }), 400
            elif provider == 'qwen':
                # 通义千问API测试
                import requests
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                test_data = {
                    'model': model,
                    'input': {'messages': [{'role': 'user', 'content': 'Hello'}]},
                    'parameters': {'max_tokens': 10}
                }
                response = requests.post('https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation', 
                                       headers=headers, json=test_data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('output'):
                        return jsonify({
                            'success': True,
                            'message': f'{provider} LLM连接测试成功',
                            'provider': provider,
                            'model': model
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'通义千问API验证失败: {result.get("message", "未知错误")}'
                        }), 400
                else:
                    return jsonify({
                        'success': False,
                        'error': f'通义千问API验证失败: {response.text}'
                    }), 400
            elif provider == 'baidu':
                # 百度文心API测试
                import requests
                # 首先获取access_token
                token_url = 'https://aip.baidubce.com/oauth/2.0/token'
                token_params = {
                    'grant_type': 'client_credentials',
                    'client_id': api_key,
                    'client_secret': data.get('secret_key') or Config.BAIDU_SECRET_KEY
                }
                token_response = requests.post(token_url, params=token_params, timeout=10)
                if token_response.status_code != 200:
                    return jsonify({
                        'success': False,
                        'error': f'百度API Token获取失败: {token_response.text}'
                    }), 400
                
                token_data = token_response.json()
                if 'access_token' not in token_data:
                    return jsonify({
                        'success': False,
                        'error': f'百度API Token获取失败: {token_data.get("error_description", "未知错误")}'
                    }), 400
                
                access_token = token_data['access_token']
                
                # 测试聊天API
                chat_url = f'https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model}?access_token={access_token}'
                chat_data = {
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                    'max_output_tokens': 10
                }
                chat_response = requests.post(chat_url, json=chat_data, timeout=10)
                if chat_response.status_code == 200:
                    result = chat_response.json()
                    if result.get('result'):
                        return jsonify({
                            'success': True,
                            'message': f'{provider} LLM连接测试成功',
                            'provider': provider,
                            'model': model
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'百度API验证失败: {result.get("error_msg", "未知错误")}'
                        }), 400
                else:
                    return jsonify({
                        'success': False,
                        'error': f'百度API验证失败: {chat_response.text}'
                    }), 400
            elif provider == 'deepseek':
                # DeepSeek API测试
                import requests
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                test_data = {
                    'model': model,
                    'messages': [{'role': 'user', 'content': 'Hello'}],
                    'max_tokens': 10
                }
                response = requests.post('https://api.deepseek.com/v1/chat/completions', 
                                       headers=headers, json=test_data, timeout=10)
                if response.status_code == 200:
                    result = response.json()
                    if result.get('choices'):
                        return jsonify({
                            'success': True,
                            'message': f'{provider} LLM连接测试成功',
                            'provider': provider,
                            'model': model
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'error': f'DeepSeek API验证失败: 响应格式异常'
                        }), 400
                else:
                    return jsonify({
                        'success': False,
                        'error': f'DeepSeek API验证失败: {response.text}'
                    }), 400
            else:
                # 对于其他提供商，返回API key格式验证
                if len(api_key) < 10:
                    return jsonify({
                        'success': False,
                        'error': f'{provider} API Key格式无效，长度过短'
                    }), 400
                return jsonify({
                    'success': True,
                    'message': f'{provider} LLM配置验证成功（基础格式检查通过）',
                    'provider': provider,
                    'model': model
                })
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'LLM连接测试失败: {str(e)}'
            }), 400
        
    except Exception as e:
        log_error(f"LLM测试API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'LLM连接测试失败: {str(e)}'
        }), 500

@app.route('/api/test-github', methods=['POST'])
def api_test_github():
    """
    测试GitHub连接API
    
    Returns:
        JSON: 测试结果
    """
    try:
        data = request.get_json() or {}
        token = data.get('github_token') or data.get('token') or Config.GITHUB_TOKEN
        repo = data.get('github_repo') or data.get('repo') or Config.GITHUB_REPO
        
        if not token:
            return jsonify({
                'success': False,
                'error': '缺少GitHub Token'
            }), 400
        
        if not repo:
            return jsonify({
                'success': False,
                'error': '缺少GitHub仓库'
            }), 400
        
        # 验证仓库格式
        if not validate_github_repo(repo):
            return jsonify({
                'success': False,
                'error': 'GitHub仓库格式无效，应为 owner/repo 格式'
            }), 400
        
        # 测试GitHub API连接
        from .github_api import GitHubAPI
        github_api = GitHubAPI(token, repo)
        
        # 尝试获取仓库信息
        repo_info = github_api.get_repo_info()
        if not repo_info:
            return jsonify({
                'success': False,
                'error': 'GitHub连接失败，请检查Token和仓库权限'
            }), 400
        
        return jsonify({
            'success': True,
            'message': 'GitHub连接测试成功',
            'repo_name': repo_info.get('name', ''),
            'repo_description': repo_info.get('description', ''),
            'stars': repo_info.get('stargazers_count', 0)
        })
        
    except Exception as e:
        log_error(f"GitHub测试API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'GitHub连接测试失败: {str(e)}'
        }), 500

@app.route('/api/test-embedding', methods=['POST'])
def api_test_embedding():
    """
    测试嵌入模型连接API
    
    Returns:
        JSON: 测试结果
    """
    try:
        data = request.get_json() or {}
        provider = data.get('provider') or Config.EMBEDDING_PROVIDER
        api_key = data.get('api_key')
        model = data.get('model')
        
        if not provider:
            return jsonify({
                'success': False,
                'error': '缺少提供商参数'
            }), 400
        
        # 根据提供商获取相应的API Key和模型
        if provider == 'local':
            model = model or Config.LOCAL_EMBEDDING_MODEL
        elif provider == 'openai':
            api_key = api_key or Config.OPENAI_API_KEY
            model = model or Config.OPENAI_EMBEDDING_MODEL
        elif provider == 'zhipu':
            # 优先使用嵌入模型专用的API Key，如果没有则使用通用的API Key
            api_key = api_key or getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', None) or Config.ZHIPU_API_KEY
            model = model or Config.ZHIPU_EMBEDDING_MODEL
        elif provider == 'qwen':
            api_key = api_key or Config.QWEN_API_KEY
            model = model or Config.QWEN_EMBEDDING_MODEL
        elif provider == 'baidu':
            api_key = api_key or Config.BAIDU_API_KEY
            model = model or Config.BAIDU_EMBEDDING_MODEL
        else:
            return jsonify({
                'success': False,
                'error': f'不支持的嵌入模型提供商: {provider}'
            }), 400
        
        if provider != 'local' and not api_key:
            return jsonify({
                'success': False,
                'error': f'{provider} 嵌入模型需要API Key'
            }), 400
        
        # 测试嵌入模型连接
        from .embedding import EmbeddingService
        try:
            # 根据提供商设置API Key
            if provider == 'openai':
                os.environ['OPENAI_API_KEY'] = api_key
            elif provider == 'zhipu':
                os.environ['ZHIPU_EMBEDDING_API_KEY'] = api_key
                # 调试日志：检查API Key是否正确设置
                log_info(f"智谱AI嵌入模型测试 - API Key长度: {len(api_key) if api_key else 0}")
            elif provider == 'qwen':
                os.environ['QWEN_EMBEDDING_API_KEY'] = api_key
            elif provider == 'baidu':
                os.environ['BAIDU_EMBEDDING_API_KEY'] = api_key
            
            embedding_service = EmbeddingService(provider=provider, model_name=model)
            # 测试获取嵌入向量
            test_embeddings = embedding_service.get_embeddings(["测试文本"])
            # 修复数组真值判断错误：使用 test_embeddings.size > 0 而不是 len(test_embeddings) > 0
            if test_embeddings is not None and test_embeddings.size > 0:
                return jsonify({
                    'success': True,
                    'message': f'{provider} 嵌入模型连接测试成功',
                    'provider': provider,
                    'model': model,
                    'dimension': len(test_embeddings[0]) if test_embeddings[0] is not None else 0
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '嵌入模型返回空结果'
                }), 400
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'嵌入模型测试失败: {str(e)}'
            }), 400
        
    except Exception as e:
        log_error(f"嵌入模型测试API错误: {e}")
        return jsonify({
            'success': False,
            'error': f'嵌入模型连接测试失败: {str(e)}'
        }), 500

# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(error):
    """
    404错误处理
    """
    return render_template('error.html', 
                         error_code=404, 
                         error_message='页面未找到'), 404

@app.errorhandler(500)
def internal_error(error):
    """
    500错误处理
    """
    log_error(f"内部服务器错误: {error}")
    return render_template('error.html', 
                         error_code=500, 
                         error_message='内部服务器错误'), 500

@app.errorhandler(Exception)
def handle_exception(e):
    """
    通用异常处理
    """
    log_error(f"未处理的异常: {e}")
    return render_template('error.html', 
                         error_code=500, 
                         error_message='服务器发生未知错误'), 500

# ==================== 模板过滤器 ====================

@app.template_filter('datetime')
def datetime_filter(value):
    """
    时间格式化过滤器
    """
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return format_time_ago(dt)
        except Exception:
            return value
    return value

@app.template_filter('truncate_smart')
def truncate_smart_filter(value, length=100):
    """
    智能截断过滤器
    """
    if not value or len(value) <= length:
        return value
    
    # 尝试在单词边界截断
    truncated = value[:length]
    last_space = truncated.rfind(' ')
    
    if last_space > length * 0.8:  # 如果最后一个空格位置合理
        return truncated[:last_space] + '...'
    else:
        return truncated + '...'

@app.template_filter('highlight_keywords')
def highlight_keywords_filter(text, keywords):
    """
    关键词高亮过滤器
    """
    if not text or not keywords:
        return text
    
    import re
    
    # 确保keywords是列表
    if isinstance(keywords, str):
        keywords = [keywords]
    
    highlighted = text
    for keyword in keywords:
        if keyword:
            pattern = re.compile(re.escape(keyword), re.IGNORECASE)
            highlighted = pattern.sub(f'<mark>{keyword}</mark>', highlighted)
    
    return highlighted

# ==================== 启动函数 ====================

def create_app(config_override=None):
    """
    创建Flask应用
    
    Args:
        config_override: 配置覆盖
        
    Returns:
        Flask: Flask应用实例
    """
    if config_override:
        app.config.update(config_override)
    
    return app

def run_app(host=None, port=None, debug=None):
    """
    运行Flask应用
    
    Args:
        host: 主机地址
        port: 端口号
        debug: 调试模式
    """
    host = host or Config.FLASK_HOST
    port = port or Config.FLASK_PORT
    debug = debug if debug is not None else Config.FLASK_DEBUG
    
    log_info(f"启动Flask应用: http://{host}:{port}")
    
    try:
        app.run(host=host, port=port, debug=debug)
    except Exception as e:
        log_error(f"Flask应用启动失败: {e}")
        raise

if __name__ == '__main__':
    run_app()