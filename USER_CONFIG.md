# 用户配置指南

## 概述

OpenIssueBot 支持通过用户配置文件自定义各种设置，包括 GitHub 仓库、LLM 提供商、嵌入模型等。这样可以避免在代码中硬编码配置，提供更灵活的使用体验。

## 配置文件设置

### 1. 创建配置文件

复制示例配置文件并根据需要修改：

```bash
cp user_config.example.json user_config.json
```

### 2. 配置项说明

#### GitHub 配置
```json
{
  "github": {
    "token": "your_github_token_here",
    "repo": "chatchat-space/Langchain-Chatchat"
  }
}
```

- `token`: GitHub Personal Access Token
- `repo`: 目标仓库，格式为 `owner/repo`

#### LLM 配置
```json
{
  "llm": {
    "provider": "zhipu",
    "zhipu": {
      "api_key": "your_zhipu_api_key",
      "model": "glm-4-plus"
    }
  }
}
```

支持的提供商：
- `openai`: OpenAI GPT 模型
- `zhipu`: 智谱 AI 模型
- `qwen`: 阿里云通义千问
- `baidu`: 百度文心一言
- `deepseek`: DeepSeek 模型

#### 嵌入模型配置
```json
{
  "embedding": {
    "provider": "zhipu",
    "zhipu": {
      "api_key": "your_zhipu_api_key",
      "model": "embedding-2"
    }
  }
}
```

支持的提供商：
- `local`: 本地 SentenceTransformer 模型
- `openai`: OpenAI 嵌入模型
- `zhipu`: 智谱 AI 嵌入模型
- `qwen`: 阿里云嵌入模型
- `baidu`: 百度嵌入模型

#### 其他配置
```json
{
  "search": {
    "max_results": 10,
    "similarity_threshold": 0.7,
    "enable_rerank": true
  },
  "cache": {
    "enabled": true,
    "ttl_hours": 24
  },
  "web": {
    "host": "127.0.0.1",
    "port": 5000,
    "debug": false
  }
}
```

## 使用方法

### 1. 初始化搜索引擎

```bash
# 使用用户配置初始化
python main.py init --force

# 或者指定特定的嵌入提供商
python main.py init --embedding-provider zhipu --force
```

### 2. 启动 Web 服务

```bash
python main.py web
```

### 3. 交互式搜索

```bash
python main.py interactive
```

## 环境变量支持

除了配置文件，系统仍然支持通过环境变量设置配置：

```bash
export GITHUB_TOKEN="your_token"
export GITHUB_REPO="owner/repo"
export ZHIPU_API_KEY="your_zhipu_key"
```

**注意**: 用户配置文件的优先级高于环境变量。

## 配置验证

系统会在启动时自动验证配置：

1. 检查必需的配置项是否存在
2. 验证 API 密钥格式
3. 检查仓库格式是否正确

如果配置验证失败，系统会显示详细的错误信息。

## 示例配置

### 使用智谱 AI
```json
{
  "github": {
    "token": "ghp_xxxxxxxxxxxx",
    "repo": "chatchat-space/Langchain-Chatchat"
  },
  "llm": {
    "provider": "zhipu",
    "zhipu": {
      "api_key": "your_zhipu_api_key",
      "model": "glm-4-plus"
    }
  },
  "embedding": {
    "provider": "zhipu",
    "zhipu": {
      "api_key": "your_zhipu_api_key",
      "model": "embedding-2"
    }
  }
}
```

### 使用本地嵌入模型 + OpenAI LLM
```json
{
  "github": {
    "token": "ghp_xxxxxxxxxxxx",
    "repo": "facebook/react"
  },
  "llm": {
    "provider": "openai",
    "openai": {
      "api_key": "sk-xxxxxxxxxxxx",
      "model": "gpt-4"
    }
  },
  "embedding": {
    "provider": "local",
    "local": {
      "model": "paraphrase-MiniLM-L6-v2"
    }
  }
}
```

## 故障排除

1. **配置文件不存在**: 确保 `user_config.json` 文件在项目根目录
2. **API 密钥无效**: 检查 API 密钥是否正确且有效
3. **仓库访问失败**: 确保 GitHub token 有访问目标仓库的权限
4. **模型加载失败**: 检查网络连接和模型名称是否正确

如有问题，请查看日志输出获取详细错误信息。