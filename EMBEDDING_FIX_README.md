# 嵌入模型测试问题修复说明

## 问题描述

用户在测试嵌入模型时遇到了两个问题：

1. **智谱AI 401错误**：虽然界面显示智谱AI的API Key是正确的，但在测试嵌入模型时报401未授权错误
2. **本地模型数组真值判断错误**：测试本地模型时出现 "The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()" 错误

## 问题分析

### 问题1：智谱AI 401错误

**根本原因**：
- API Key获取逻辑不完整，没有正确处理嵌入模型专用的API Key
- 环境变量设置和读取逻辑存在问题

**具体问题**：
1. `web_app.py` 中获取智谱AI API Key时，没有优先检查 `ZHIPU_EMBEDDING_API_KEY`
2. `embedding.py` 中初始化智谱AI客户端时，没有检查环境变量中的API Key
3. 前端JavaScript代码中获取API Key的逻辑不够健壮

### 问题2：本地模型数组真值判断错误

**根本原因**：
- 在判断numpy数组是否为空时，直接使用了 `if test_embeddings` 和 `len(test_embeddings) > 0`
- 当numpy数组有多个元素时，Python无法确定数组的真值，导致歧义错误

**具体问题**：
- `web_app.py` 第1055行：`if test_embeddings and len(test_embeddings) > 0:` 会触发数组真值判断错误

## 修复方案

### 修复1：智谱AI API Key获取逻辑

#### 1.1 修复 `web_app.py` 中的API Key获取

```python
# 修复前
elif provider == 'zhipu':
    api_key = api_key or Config.ZHIPU_API_KEY
    model = model or Config.ZHIPU_EMBEDDING_MODEL

# 修复后
elif provider == 'zhipu':
    # 优先使用嵌入模型专用的API Key，如果没有则使用通用的API Key
    api_key = api_key or getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', None) or Config.ZHIPU_API_KEY
    model = model or Config.ZHIPU_EMBEDDING_MODEL
```

#### 1.2 修复 `embedding.py` 中的客户端初始化

```python
# 修复前
api_key = getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', None) or getattr(Config, 'ZHIPU_API_KEY', None)

# 修复后
api_key = (getattr(Config, 'ZHIPU_EMBEDDING_API_KEY', None) or 
          os.getenv('ZHIPU_EMBEDDING_API_KEY') or 
          getattr(Config, 'ZHIPU_API_KEY', None) or 
          os.getenv('ZHIPU_API_KEY'))
```

#### 1.3 修复前端JavaScript代码

```javascript
// 修复前
apiKey = document.getElementById('zhipu-embedding-api-key').value || document.getElementById('zhipu-api-key').value;

// 修复后
const embeddingApiKey = document.getElementById('zhipu-embedding-api-key')?.value?.trim();
const llmApiKey = document.getElementById('zhipu-api-key')?.value?.trim();
apiKey = embeddingApiKey || llmApiKey;
```

### 修复2：数组真值判断错误

#### 2.1 修复 `web_app.py` 中的数组判断逻辑

```python
# 修复前
if test_embeddings and len(test_embeddings) > 0:

# 修复后
if test_embeddings is not None and test_embeddings.size > 0:
```

**关键改进**：
- 使用 `test_embeddings.size > 0` 替代 `len(test_embeddings) > 0`
- 添加 `is not None` 检查确保对象存在
- 避免了numpy数组的真值判断歧义

## 修复文件列表

1. `app/web_app.py` - 修复API Key获取和数组判断逻辑
2. `app/embedding.py` - 修复智谱AI客户端初始化
3. `templates/config.html` - 修复前端API Key获取逻辑
4. `test_embedding_fix.py` - 新增测试验证脚本

## 验证方法

### 方法1：运行测试脚本

```bash
python test_embedding_fix.py
```

### 方法2：Web界面测试

1. 启动应用：`python main.py`
2. 访问配置页面：`http://localhost:5000/config`
3. 配置智谱AI API Key
4. 点击"测试嵌入模型连接"按钮
5. 选择"智谱AI"提供商进行测试
6. 选择"本地模型"提供商进行测试

### 方法3：日志检查

查看应用日志，确认：
- 智谱AI API Key长度正确显示
- 没有401错误
- 没有数组真值判断错误

## 预期结果

修复后应该看到：

1. **智谱AI测试**：
   - ✅ 连接成功
   - 显示模型信息和向量维度
   - 无401错误

2. **本地模型测试**：
   - ✅ 连接成功
   - 显示模型信息和向量维度
   - 无数组真值判断错误

## 注意事项

1. **API Key配置**：确保在配置页面或环境变量中正确设置了智谱AI的API Key
2. **网络连接**：智谱AI测试需要网络连接到智谱AI服务器
3. **本地模型**：首次使用本地模型时需要下载模型文件，可能需要一些时间
4. **依赖包**：确保安装了所有必要的Python包（sentence-transformers, numpy等）

## 技术细节

### numpy数组真值判断问题详解

```python
# 错误的做法（会报错）
if array:  # ValueError: The truth value of an array with more than one element is ambiguous

# 正确的做法
if array.size > 0:     # 检查数组是否有元素
if len(array) > 0:     # 检查数组长度
if array is not None:  # 检查数组是否存在
```

### API Key优先级

1. 前端传入的API Key（最高优先级）
2. 嵌入模型专用API Key（ZHIPU_EMBEDDING_API_KEY）
3. 通用API Key（ZHIPU_API_KEY）
4. 环境变量中的API Key

这样的优先级设计确保了灵活性和向后兼容性。