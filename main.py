#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenIssueBot 主程序入口

这是OpenIssueBot的主要入口文件，提供：
1. 命令行接口
2. Web服务启动
3. 配置管理
4. 交互式搜索
5. 批量处理功能

使用方法:
    python main.py --help                    # 查看帮助
    python main.py web                       # 启动Web服务
    python main.py search "error message"    # 命令行搜索
    python main.py init                      # 初始化搜索引擎
    python main.py config                    # 配置管理
"""

import argparse
import sys
import os
from typing import Optional, List

# 抑制TensorFlow和FAISS警告信息
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 抑制TensorFlow INFO和WARNING
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # 禁用oneDNN优化警告

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.config import Config
from app.user_config import user_config
from app.issue_search import IssueSearchEngine, create_search_engine, quick_search
from app.web_app import run_app
from app.utils import (
    log_info, log_error, log_warning,
    validate_github_repo, format_time_ago,
    Timer
)

def print_banner():
    """
    打印程序横幅
    """
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                        OpenIssueBot                          ║
║                                                              ║
║    基于RAG和大语言模型的GitHub Issue智能搜索助手              ║
║                                                              ║
║    功能特性:                                                 ║
║    • 向量化搜索GitHub Issues                                 ║
║    • AI驱动的问题分析                                        ║
║    • 智能摘要提取                                            ║
║    • Web界面支持                                             ║
║    • 多种搜索策略                                            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_config_status():
    """
    打印配置状态
    """
    print("\n📋 当前配置状态:")
    print(f"   GitHub Token: {'✅ 已配置' if Config.GITHUB_TOKEN else '❌ 未配置'}")
    print(f"   GitHub Repo: {Config.GITHUB_REPO or '❌ 未配置'}")
    print(f"   OpenAI API Key: {'✅ 已配置' if Config.OPENAI_API_KEY else '❌ 未配置'}")
    print(f"   本地嵌入模型: {'✅ 启用' if Config.USE_LOCAL_EMBEDDING else '❌ 禁用'}")
    print(f"   缓存目录: {Config.CACHE_DIR}")
    print(f"   Web服务: {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    
    # 检查配置完整性
    missing_configs = []
    if not Config.GITHUB_TOKEN:
        missing_configs.append("GITHUB_TOKEN")
    if not Config.GITHUB_REPO:
        missing_configs.append("GITHUB_REPO")
    elif not validate_github_repo(Config.GITHUB_REPO):
        missing_configs.append("GITHUB_REPO (格式无效)")
    
    if missing_configs:
        print(f"\n⚠️  缺少必要配置: {', '.join(missing_configs)}")
        print("   请设置环境变量或修改 .env 文件")
    else:
        print("\n✅ 配置检查通过")

def cmd_web(args):
    """
    启动Web服务
    
    Args:
        args: 命令行参数
    """
    print("🚀 启动Web服务...")
    
    try:
        host = args.host or Config.FLASK_HOST
        port = args.port or Config.FLASK_PORT
        debug = args.debug if args.debug is not None else Config.FLASK_DEBUG
        
        print(f"   服务地址: http://{host}:{port}")
        print(f"   调试模式: {'开启' if debug else '关闭'}")
        print("\n💡 提示: 在浏览器中打开上述地址即可使用Web界面")
        print("   按 Ctrl+C 停止服务\n")
        
        run_app(host=host, port=port, debug=debug)
        
    except KeyboardInterrupt:
        print("\n👋 Web服务已停止")
    except Exception as e:
        log_error(f"Web服务启动失败: {e}")
        print(f"❌ Web服务启动失败: {e}")
        sys.exit(1)

def cmd_search(args):
    """
    命令行搜索
    
    Args:
        args: 命令行参数
    """
    query = args.query
    max_results = args.max_results
    
    print(f"🔍 搜索: {query}")
    print(f"   最大结果数: {max_results}")
    print(f"   目标仓库: {Config.GITHUB_REPO}")
    
    try:
        with Timer("搜索"):
            # 使用快速搜索
            if args.quick:
                result = quick_search(query, max_results=max_results)
            else:
                # 使用完整搜索引擎
                search_engine = create_search_engine()
                
                # 初始化
                print("\n📚 初始化搜索引擎...")
                if not search_engine.initialize(force_refresh=args.refresh):
                    print("❌ 搜索引擎初始化失败")
                    sys.exit(1)
                
                # 搜索
                print("\n🔎 执行搜索...")
                if args.hybrid:
                    result = search_engine.hybrid_search(query, max_results)
                else:
                    result = search_engine.search(
                        query, 
                        max_results, 
                        similarity_threshold=args.threshold,
                        include_llm_analysis=args.llm
                    )
        
        # 显示结果
        print_search_results(result, args.verbose)
        
    except Exception as e:
        log_error(f"搜索失败: {e}")
        print(f"❌ 搜索失败: {e}")
        sys.exit(1)

def cmd_init(args):
    """
    初始化搜索引擎
    
    Args:
        args: 命令行参数
    """
    print("🔧 初始化搜索引擎...")
    
    try:
        # 验证用户配置
        if not user_config.validate_config():
            print("❌ 用户配置验证失败，请检查配置文件")
            sys.exit(1)
        
        # 获取用户配置
        github_config = user_config.get_github_config()
        embedding_config = user_config.get_embedding_config()
        
        # 根据命令行参数或用户配置确定嵌入模型提供商
        embedding_provider = getattr(args, 'embedding_provider', None) or embedding_config['provider']
        
        # 创建搜索引擎，使用用户配置
        search_engine = create_search_engine(
            github_token=github_config['token'],
            github_repo=github_config['repo'],
            embedding_provider=embedding_provider
        )
        
        with Timer("初始化"):
            success = search_engine.initialize(force_refresh=args.force)
        
        if success:
            stats = search_engine.get_stats()
            print("\n✅ 初始化成功!")
            print(f"   索引Issues数量: {stats.get('total_issues', 0)}")
            print(f"   嵌入向量维度: {stats.get('embedding_service', {}).get('dimension', 0)}")
            print(f"   搜索引擎状态: {'已就绪' if stats.get('is_initialized') else '未就绪'}")
            print(f"   使用的嵌入提供商: {embedding_provider}")
            print(f"   目标仓库: {github_config['repo']}")
        else:
            print("❌ 初始化失败")
            sys.exit(1)
            
    except Exception as e:
        log_error(f"初始化失败: {e}")
        print(f"❌ 初始化失败: {e}")
        sys.exit(1)

def cmd_config(args):
    """
    配置管理
    
    Args:
        args: 命令行参数
    """
    if args.show:
        print_config_status()
    
    if args.validate:
        print("\n🔍 验证配置...")
        
        # 验证GitHub配置
        if Config.GITHUB_TOKEN and Config.GITHUB_REPO:
            try:
                from app.github_api import GitHubAPI
                github_api = GitHubAPI(Config.GITHUB_TOKEN, Config.GITHUB_REPO)
                repo_info = github_api.get_repo_info()
                
                if repo_info:
                    print(f"✅ GitHub连接成功")
                    print(f"   仓库: {repo_info.get('full_name')}")
                    print(f"   描述: {repo_info.get('description', '无描述')}")
                    print(f"   Stars: {repo_info.get('stargazers_count', 0)}")
                else:
                    print("❌ GitHub连接失败")
            except Exception as e:
                print(f"❌ GitHub验证失败: {e}")
        else:
            print("❌ GitHub配置不完整")
        
        # 验证OpenAI配置
        if Config.OPENAI_API_KEY:
            try:
                from app.llm_analysis import LLMAnalyzer
                llm = LLMAnalyzer(Config.OPENAI_API_KEY)
                print("✅ OpenAI API配置有效")
            except Exception as e:
                print(f"❌ OpenAI API验证失败: {e}")
        else:
            print("⚠️  OpenAI API未配置（可选）")
        
        # 验证嵌入模型
        try:
            from app.embedding import EmbeddingService
            provider = getattr(Config, 'EMBEDDING_PROVIDER', 'local')
            embedding_service = EmbeddingService(provider=provider)
            dimension = embedding_service.get_embedding_dimension()
            print(f"✅ 嵌入服务正常 (维度: {dimension})")
        except Exception as e:
            print(f"❌ 嵌入服务验证失败: {e}")
    
    if args.clear_cache:
        print("\n🗑️  清除缓存...")
        try:
            search_engine = create_search_engine()
            search_engine.clear_cache()
            print("✅ 缓存已清除")
        except Exception as e:
            print(f"❌ 清除缓存失败: {e}")

def cmd_interactive(args):
    """
    交互式搜索模式
    
    Args:
        args: 命令行参数
    """
    print("🎯 进入交互式搜索模式")
    print("   输入 'quit' 或 'exit' 退出")
    print("   输入 'help' 查看帮助\n")
    
    try:
        # 初始化搜索引擎
        search_engine = create_search_engine()
        print("📚 初始化搜索引擎...")
        if not search_engine.initialize():
            print("❌ 搜索引擎初始化失败")
            return
        print("✅ 搜索引擎就绪\n")
        
        while True:
            try:
                query = input("🔍 请输入搜索内容: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 退出交互式模式")
                    break
                
                if query.lower() == 'help':
                    print_interactive_help()
                    continue
                
                # 执行搜索
                print(f"\n🔎 搜索: {query}")
                result = search_engine.search(query, max_results=5)
                print_search_results(result, verbose=False)
                print()
                
            except KeyboardInterrupt:
                print("\n👋 退出交互式模式")
                break
            except Exception as e:
                print(f"❌ 搜索错误: {e}")
                
    except Exception as e:
        log_error(f"交互式模式失败: {e}")
        print(f"❌ 交互式模式失败: {e}")

def print_interactive_help():
    """
    打印交互式模式帮助
    """
    help_text = """
📖 交互式搜索帮助:

   基本用法:
   • 直接输入错误信息或问题描述进行搜索
   • 支持中文和英文搜索
   • 搜索结果按相似度排序

   特殊命令:
   • help  - 显示此帮助信息
   • quit  - 退出交互式模式
   • exit  - 退出交互式模式
   • q     - 退出交互式模式

   搜索技巧:
   • 包含具体的错误信息效果更好
   • 可以搜索功能描述或技术关键词
   • 尝试不同的关键词组合
    """
    print(help_text)

def print_search_results(result, verbose=False):
    """
    打印搜索结果
    
    Args:
        result: 搜索结果
        verbose: 是否显示详细信息
    """
    if not result.get('success'):
        print(f"❌ 搜索失败: {result.get('error', '未知错误')}")
        return
    
    results = result.get('results', [])
    total = result.get('total_found', 0)
    
    print(f"\n📊 搜索结果: 找到 {total} 个相关Issue")
    
    if not results:
        print("   没有找到相关的Issues")
        
        # 显示LLM分析结果
        if 'llm_analysis' in result:
            llm_result = result['llm_analysis']
            if llm_result.get('success'):
                print("\n🤖 AI分析结果:")
                print(f"   {llm_result.get('answer', '无分析结果')}")
        return
    
    # 显示搜索结果
    for i, issue in enumerate(results, 1):
        print(f"\n{i}. #{issue.get('number')} - {issue.get('title', '无标题')}")
        print(f"   状态: {issue.get('state', '未知')}")
        print(f"   相似度: {issue.get('similarity_score', 0) * 100:.1f}%")
        print(f"   链接: {issue.get('url', '无链接')}")
        
        if verbose:
            # 显示详细信息
            if issue.get('labels'):
                labels = [label.get('name', '') for label in issue.get('labels', [])]
                print(f"   标签: {', '.join(labels)}")
            
            if issue.get('created_at'):
                print(f"   创建时间: {issue.get('created_at')}")
            
            if issue.get('body_preview'):
                print(f"   内容预览: {issue.get('body_preview')}")
            
            if issue.get('solution_summary'):
                print(f"   解决方案: {issue.get('solution_summary')}")
    
    # 显示LLM增强分析
    if 'llm_enhancement' in result:
        llm_result = result['llm_enhancement']
        if llm_result.get('success'):
            print("\n🤖 AI增强分析:")
            print(f"   {llm_result.get('answer', '无分析结果')}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='OpenIssueBot - 基于RAG和大语言模型的GitHub Issue智能搜索助手',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py web                           # 启动Web服务
  python main.py search "ImportError"          # 搜索错误信息
  python main.py search "如何使用API" --llm     # 搜索并启用LLM分析
  python main.py init --force                  # 强制重新初始化
  python main.py config --validate            # 验证配置
  python main.py interactive                   # 交互式搜索

更多信息请访问: https://github.com/1122414/openIssueBot
        """
    )
    
    # 全局选项
    parser.add_argument('--version', action='version', version='OpenIssueBot 1.0.0')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细信息')
    parser.add_argument('--quiet', '-q', action='store_true', help='静默模式')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # Web服务命令
    web_parser = subparsers.add_parser('web', help='启动Web服务')
    web_parser.add_argument('--host', default=None, help='服务主机地址')
    web_parser.add_argument('--port', type=int, default=None, help='服务端口号')
    web_parser.add_argument('--debug', action='store_true', help='启用调试模式')
    web_parser.set_defaults(func=cmd_web)
    
    # 搜索命令
    search_parser = subparsers.add_parser('search', help='命令行搜索')
    search_parser.add_argument('query', help='搜索查询')
    search_parser.add_argument('--max-results', '-n', type=int, default=5, help='最大结果数')
    search_parser.add_argument('--threshold', '-t', type=float, default=0.3, help='相似度阈值')
    search_parser.add_argument('--hybrid', action='store_true', help='使用混合搜索')
    search_parser.add_argument('--llm', action='store_true', help='启用LLM分析')
    search_parser.add_argument('--quick', action='store_true', help='快速搜索模式')
    search_parser.add_argument('--refresh', action='store_true', help='刷新索引')
    search_parser.set_defaults(func=cmd_search)
    
    # 初始化命令
    init_parser = subparsers.add_parser('init', help='初始化搜索引擎')
    init_parser.add_argument('--force', '-f', action='store_true', help='强制重新初始化')
    init_parser.add_argument('--embedding-provider', choices=['local', 'openai', 'zhipu', 'qwen', 'baidu'], help='指定嵌入模型提供商')
    init_parser.set_defaults(func=cmd_init)
    
    # 配置命令
    config_parser = subparsers.add_parser('config', help='配置管理')
    config_parser.add_argument('--show', '-s', action='store_true', help='显示当前配置')
    config_parser.add_argument('--validate', action='store_true', help='验证配置')
    config_parser.add_argument('--clear-cache', action='store_true', help='清除缓存')
    config_parser.set_defaults(func=cmd_config)
    
    # 交互式命令
    interactive_parser = subparsers.add_parser('interactive', help='交互式搜索模式')
    interactive_parser.set_defaults(func=cmd_interactive)
    
    # 解析参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助
    if not args.command:
        print_banner()
        print_config_status()
        print("\n💡 使用 'python main.py --help' 查看所有可用命令")
        print("   或者使用 'python main.py web' 启动Web界面")
        return
    
    # 设置日志级别
    if args.quiet:
        import logging
        logging.getLogger().setLevel(logging.ERROR)
    elif args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)
    
    # 执行命令
    try:
        if not args.quiet:
            print_banner()
        
        args.func(args)
        
    except KeyboardInterrupt:
        print("\n👋 程序已中断")
        sys.exit(0)
    except Exception as e:
        log_error(f"程序执行失败: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        else:
            print(f"❌ 程序执行失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()