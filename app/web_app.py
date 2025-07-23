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
        formatted['labels_formatted'] = [{
            'name': label.get('name', ''),
            'color': label.get('color', 'gray')
        } for label in formatted['labels']]
    
    return formatted

def get_config_status() -> Dict[str, Any]:
    """
    获取配置状态
    
    Returns:
        Dict: 配置状态信息
    """
    return {
        'github_token_configured': bool(Config.GITHUB_TOKEN),
        'github_repo_configured': bool(Config.GITHUB_REPO),
        'github_repo_valid': validate_github_repo(Config.GITHUB_REPO or ''),
        'openai_api_key_configured': bool(Config.OPENAI_API_KEY),
        'use_local_embedding': Config.USE_LOCAL_EMBEDDING,
        'cache_dir': Config.CACHE_DIR,
        'flask_host': Config.FLASK_HOST,
        'flask_port': Config.FLASK_PORT
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
        'github_repo': Config.GITHUB_REPO,
        'openai_model': getattr(Config, 'OPENAI_MODEL', 'gpt-3.5-turbo'),
        'use_local_embedding': Config.USE_LOCAL_EMBEDDING,
        'embedding_model': getattr(Config, 'EMBEDDING_MODEL', 'paraphrase-MiniLM-L6-v2'),
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
        
        # 格式化结果
        if result.get('success') and 'results' in result:
            result['results'] = [format_search_result(r) for r in result['results']]
        
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
            
            # 这里可以添加配置更新逻辑
            # 注意：在生产环境中，配置更新应该更加安全和持久化
            
            return jsonify({
                'success': True,
                'message': '配置更新成功（重启应用后生效）'
            })
            
        except Exception as e:
            log_error(f"配置API错误: {e}")
            return jsonify({
                'success': False,
                'error': f'配置更新失败: {str(e)}'
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