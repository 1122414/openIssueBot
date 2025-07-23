# -*- coding: utf-8 -*-
"""
工具函数模块

提供各种辅助功能：
1. 日志记录
2. 文本处理
3. 文件操作
4. 数据验证
5. 时间处理
"""

import os
import re
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timedelta
import unicodedata

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('openissuebot.log', encoding='utf-8')
    ]
)

logger = logging.getLogger('OpenIssueBot')

def setup_logging(log_level: str = 'INFO', log_file: str = 'openissuebot.log') -> None:
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # 清除现有的处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 重新配置日志
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ],
        force=True
    )
    
    logger.setLevel(level)

# ==================== 日志记录函数 ====================

def log_info(message: str) -> None:
    """
    记录信息日志
    
    Args:
        message: 日志信息
    """
    logger.info(message)

def log_error(message: str) -> None:
    """
    记录错误日志
    
    Args:
        message: 错误信息
    """
    logger.error(message)

def log_warning(message: str) -> None:
    """
    记录警告日志
    
    Args:
        message: 警告信息
    """
    logger.warning(message)

def log_debug(message: str) -> None:
    """
    记录调试日志
    
    Args:
        message: 调试信息
    """
    logger.debug(message)

# ==================== 文本处理函数 ====================

def clean_text(text: str) -> str:
    """
    清理文本内容
    
    Args:
        text: 原始文本
        
    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""
    
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    
    # 移除多余的空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 移除特殊字符（保留基本标点）
    text = re.sub(r'[^\w\s\.,;:!?\-()\[\]{}"\']', '', text)
    
    # 标准化Unicode字符
    text = unicodedata.normalize('NFKC', text)
    
    return text.strip()

def extract_code_blocks(text: str) -> List[str]:
    """
    提取文本中的代码块
    
    Args:
        text: 包含代码块的文本
        
    Returns:
        List[str]: 代码块列表
    """
    # 匹配Markdown代码块
    code_pattern = r'```[\w]*\n([\s\S]*?)\n```'
    code_blocks = re.findall(code_pattern, text)
    
    # 匹配行内代码
    inline_code_pattern = r'`([^`]+)`'
    inline_codes = re.findall(inline_code_pattern, text)
    
    return code_blocks + inline_codes

def extract_error_messages(text: str) -> List[str]:
    """
    提取文本中的错误信息
    
    Args:
        text: 包含错误信息的文本
        
    Returns:
        List[str]: 错误信息列表
    """
    error_patterns = [
        r'Error[:\s]+([^\n]+)',
        r'Exception[:\s]+([^\n]+)',
        r'Failed[:\s]+([^\n]+)',
        r'TypeError[:\s]+([^\n]+)',
        r'ValueError[:\s]+([^\n]+)',
        r'AttributeError[:\s]+([^\n]+)',
        r'ImportError[:\s]+([^\n]+)',
        r'ModuleNotFoundError[:\s]+([^\n]+)'
    ]
    
    errors = []
    for pattern in error_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        errors.extend(matches)
    
    return list(set(errors))  # 去重

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """
    提取文本关键词
    
    Args:
        text: 输入文本
        max_keywords: 最大关键词数量
        
    Returns:
        List[str]: 关键词列表
    """
    if not text:
        return []
    
    # 清理文本
    clean = clean_text(text.lower())
    
    # 停用词列表
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your',
        'his', 'her', 'its', 'our', 'their', 'when', 'where', 'why', 'how', 'what', 'which', 'who',
        'if', 'then', 'else', 'so', 'as', 'than', 'too', 'very', 'just', 'now', 'here', 'there'
    }
    
    # 提取单词
    words = re.findall(r'\b[a-zA-Z]{3,}\b', clean)
    
    # 过滤停用词并计算频率
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) >= 3:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_words[:max_keywords]]

def truncate_text(text: str, max_length: int = 500, suffix: str = "...") -> str:
    """
    截断文本到指定长度
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀
        
    Returns:
        str: 截断后的文本
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def similarity_score(text1: str, text2: str) -> float:
    """
    计算两个文本的相似度（简单版本）
    
    Args:
        text1: 文本1
        text2: 文本2
        
    Returns:
        float: 相似度分数 (0-1)
    """
    if not text1 or not text2:
        return 0.0
    
    # 提取关键词
    keywords1 = set(extract_keywords(text1, 20))
    keywords2 = set(extract_keywords(text2, 20))
    
    if not keywords1 or not keywords2:
        return 0.0
    
    # 计算Jaccard相似度
    intersection = len(keywords1.intersection(keywords2))
    union = len(keywords1.union(keywords2))
    
    return intersection / union if union > 0 else 0.0

# ==================== 文件操作函数 ====================

def ensure_dir(directory: str) -> None:
    """
    确保目录存在
    
    Args:
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        log_info(f"创建目录: {directory}")

def safe_read_json(file_path: str, default: Any = None) -> Any:
    """
    安全读取JSON文件
    
    Args:
        file_path: 文件路径
        default: 默认值
        
    Returns:
        Any: JSON数据或默认值
    """
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        log_error(f"读取JSON文件失败 {file_path}: {e}")
    
    return default

def safe_write_json(file_path: str, data: Any) -> bool:
    """
    安全写入JSON文件
    
    Args:
        file_path: 文件路径
        data: 要写入的数据
        
    Returns:
        bool: 是否成功
    """
    try:
        # 确保目录存在
        ensure_dir(os.path.dirname(file_path))
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        log_error(f"写入JSON文件失败 {file_path}: {e}")
        return False

def get_file_hash(file_path: str) -> Optional[str]:
    """
    获取文件的MD5哈希值
    
    Args:
        file_path: 文件路径
        
    Returns:
        Optional[str]: 哈希值或None
    """
    try:
        if not os.path.exists(file_path):
            return None
        
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        log_error(f"计算文件哈希失败 {file_path}: {e}")
        return None

def get_file_size(file_path: str) -> int:
    """
    获取文件大小
    
    Args:
        file_path: 文件路径
        
    Returns:
        int: 文件大小（字节）
    """
    try:
        return os.path.getsize(file_path) if os.path.exists(file_path) else 0
    except Exception:
        return 0

# ==================== 数据验证函数 ====================

def validate_github_repo(repo: str) -> bool:
    """
    验证GitHub仓库格式
    
    Args:
        repo: 仓库名称 (owner/repo)
        
    Returns:
        bool: 是否有效
    """
    if not repo:
        return False
    
    pattern = r'^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$'
    return bool(re.match(pattern, repo))

def validate_url(url: str) -> bool:
    """
    验证URL格式
    
    Args:
        url: URL字符串
        
    Returns:
        bool: 是否有效
    """
    if not url:
        return False
    
    pattern = r'^https?://[^\s/$.?#].[^\s]*$'
    return bool(re.match(pattern, url))

def validate_email(email: str) -> bool:
    """
    验证邮箱格式
    
    Args:
        email: 邮箱地址
        
    Returns:
        bool: 是否有效
    """
    if not email:
        return False
    
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_valid_json(text: str) -> bool:
    """
    检查字符串是否为有效JSON
    
    Args:
        text: 字符串
        
    Returns:
        bool: 是否为有效JSON
    """
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False

# ==================== 时间处理函数 ====================

def parse_github_time(time_str: str) -> Optional[datetime]:
    """
    解析GitHub时间格式
    
    Args:
        time_str: GitHub时间字符串
        
    Returns:
        Optional[datetime]: 解析后的时间或None
    """
    try:
        # GitHub使用ISO 8601格式
        return datetime.fromisoformat(time_str.replace('Z', '+00:00'))
    except Exception:
        return None

def format_time_ago(dt: datetime) -> str:
    """
    格式化时间为"多久之前"
    
    Args:
        dt: 时间对象
        
    Returns:
        str: 格式化的时间字符串
    """
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    diff = now - dt
    
    if diff.days > 365:
        years = diff.days // 365
        return f"{years}年前"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months}个月前"
    elif diff.days > 0:
        return f"{diff.days}天前"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours}小时前"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes}分钟前"
    else:
        return "刚刚"

def is_recent(dt: datetime, days: int = 30) -> bool:
    """
    检查时间是否在最近指定天数内
    
    Args:
        dt: 时间对象
        days: 天数
        
    Returns:
        bool: 是否在最近指定天数内
    """
    now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
    return (now - dt).days <= days

# ==================== 数据处理函数 ====================

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    将列表分块
    
    Args:
        lst: 原始列表
        chunk_size: 块大小
        
    Returns:
        List[List[Any]]: 分块后的列表
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    扁平化嵌套字典
    
    Args:
        d: 嵌套字典
        parent_key: 父键名
        sep: 分隔符
        
    Returns:
        Dict[str, Any]: 扁平化后的字典
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def safe_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
    """
    安全获取嵌套字典的值
    
    Args:
        data: 字典数据
        path: 路径（用.分隔）
        default: 默认值
        
    Returns:
        Any: 获取的值或默认值
    """
    try:
        keys = path.split('.')
        result = data
        for key in keys:
            result = result[key]
        return result
    except (KeyError, TypeError, AttributeError):
        return default

def merge_dicts(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并多个字典
    
    Args:
        *dicts: 要合并的字典
        
    Returns:
        Dict[str, Any]: 合并后的字典
    """
    result = {}
    for d in dicts:
        if isinstance(d, dict):
            result.update(d)
    return result

# ==================== 格式化函数 ====================

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小
    
    Args:
        size_bytes: 字节数
        
    Returns:
        str: 格式化的大小字符串
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def format_number(num: Union[int, float]) -> str:
    """
    格式化数字（添加千分位分隔符）
    
    Args:
        num: 数字
        
    Returns:
        str: 格式化的数字字符串
    """
    return f"{num:,}"

def format_percentage(value: float, total: float) -> str:
    """
    格式化百分比
    
    Args:
        value: 数值
        total: 总数
        
    Returns:
        str: 百分比字符串
    """
    if total == 0:
        return "0%"
    
    percentage = (value / total) * 100
    return f"{percentage:.1f}%"

# ==================== 缓存辅助函数 ====================

def generate_cache_key(*args: Any) -> str:
    """
    生成缓存键
    
    Args:
        *args: 用于生成键的参数
        
    Returns:
        str: 缓存键
    """
    key_str = "_".join(str(arg) for arg in args)
    return hashlib.md5(key_str.encode()).hexdigest()

def is_cache_valid(cache_time: datetime, max_age_hours: int = 24) -> bool:
    """
    检查缓存是否有效
    
    Args:
        cache_time: 缓存时间
        max_age_hours: 最大有效时间（小时）
        
    Returns:
        bool: 缓存是否有效
    """
    now = datetime.now(cache_time.tzinfo) if cache_time.tzinfo else datetime.now()
    return (now - cache_time).total_seconds() < max_age_hours * 3600

# ==================== 错误处理装饰器 ====================

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """
    失败重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        log_warning(f"函数 {func.__name__} 第 {attempt + 1} 次尝试失败: {e}")
                        if delay > 0:
                            import time
                            time.sleep(delay)
                    else:
                        log_error(f"函数 {func.__name__} 重试 {max_retries} 次后仍然失败: {e}")
            
            raise last_exception
        
        return wrapper
    return decorator

def safe_execute(func, *args, default=None, **kwargs):
    """
    安全执行函数
    
    Args:
        func: 要执行的函数
        *args: 位置参数
        default: 默认返回值
        **kwargs: 关键字参数
        
    Returns:
        Any: 函数结果或默认值
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        log_error(f"安全执行函数 {func.__name__} 失败: {e}")
        return default

# ==================== 性能监控 ====================

class Timer:
    """
    简单的计时器类
    """
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        log_info(f"{self.name} 执行时间: {duration:.2f}秒")
    
    def elapsed(self) -> float:
        """
        获取已经过的时间
        
        Returns:
            float: 已经过的时间（秒）
        """
        if self.start_time:
            end = self.end_time or datetime.now()
            return (end - self.start_time).total_seconds()
        return 0.0

# ==================== 导出的便捷函数 ====================

__all__ = [
    # 日志函数
    'setup_logging', 'log_info', 'log_error', 'log_warning', 'log_debug',
    
    # 文本处理
    'clean_text', 'extract_code_blocks', 'extract_error_messages', 
    'extract_keywords', 'truncate_text', 'similarity_score',
    
    # 文件操作
    'ensure_dir', 'safe_read_json', 'safe_write_json', 
    'get_file_hash', 'get_file_size',
    
    # 数据验证
    'validate_github_repo', 'validate_url', 'validate_email', 'is_valid_json',
    
    # 时间处理
    'parse_github_time', 'format_time_ago', 'is_recent',
    
    # 数据处理
    'chunk_list', 'flatten_dict', 'safe_get', 'merge_dicts',
    
    # 格式化
    'format_file_size', 'format_number', 'format_percentage',
    
    # 缓存
    'generate_cache_key', 'is_cache_valid',
    
    # 错误处理
    'retry_on_failure', 'safe_execute',
    
    # 性能监控
    'Timer'
]