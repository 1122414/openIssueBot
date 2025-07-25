/**
 * OpenIssueBot 前端应用脚本
 * 提供搜索、配置管理、状态监控等功能
 */

// 全局变量
let searchTimeout;
let isSearching = false;
let currentSearchId = null;
let statusCheckInterval = null;

// 应用初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * 应用初始化
 */
function initializeApp() {
    // 初始化组件
    initializeTooltips();
    initializeModals();
    initializeEventListeners();
    
    // 检查系统状态
    checkSystemStatus();
    
    // 启动状态监控
    startStatusMonitoring();
    
    // 加载统计数据
    loadStatistics();
    
    console.log('OpenIssueBot 应用已初始化');
}

/**
 * 初始化工具提示
 */
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

/**
 * 初始化模态框
 */
function initializeModals() {
    // 可以在这里添加模态框的自定义行为
}

/**
 * 初始化事件监听器
 */
function initializeEventListeners() {
    // 搜索表单
    const searchForm = document.getElementById('search-form');
    if (searchForm) {
        searchForm.addEventListener('submit', handleSearchSubmit);
    }
    
    // 配置表单
    const configForm = document.getElementById('config-form');
    if (configForm) {
        configForm.addEventListener('submit', handleConfigSubmit);
    }
    
    // 实时搜索
    const queryInput = document.getElementById('query');
    if (queryInput) {
        queryInput.addEventListener('input', handleQueryInput);
    }
    
    // 相似度阈值滑块
    const thresholdSlider = document.getElementById('similarity-threshold');
    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', handleThresholdChange);
    }
    
    // 键盘快捷键
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // 页面可见性变化
    document.addEventListener('visibilitychange', handleVisibilityChange);
}

/**
 * 处理搜索表单提交
 */
function handleSearchSubmit(event) {
    event.preventDefault();
    performSearch();
}

/**
 * 处理配置表单提交
 */
function handleConfigSubmit(event) {
    event.preventDefault();
    saveConfiguration();
}

/**
 * 处理查询输入
 */
function handleQueryInput(event) {
    const realTimeSearch = document.getElementById('real-time-search');
    if (realTimeSearch && realTimeSearch.checked) {
        clearTimeout(searchTimeout);
        searchTimeout = setTimeout(() => {
            if (event.target.value.trim().length > 3) {
                performSearch();
            }
        }, 1000);
    }
}

/**
 * 处理相似度阈值变化
 */
function handleThresholdChange(event) {
    const thresholdValue = document.getElementById('threshold-value');
    if (thresholdValue) {
        thresholdValue.textContent = event.target.value;
    }
}

/**
 * 处理键盘快捷键
 */
function handleKeyboardShortcuts(event) {
    // Ctrl+Enter 执行搜索
    if (event.ctrlKey && event.key === 'Enter') {
        event.preventDefault();
        performSearch();
    }
    
    // Escape 清空搜索
    if (event.key === 'Escape') {
        clearSearch();
    }
    
    // Ctrl+/ 聚焦搜索框
    if (event.ctrlKey && event.key === '/') {
        event.preventDefault();
        const queryInput = document.getElementById('query');
        if (queryInput) {
            queryInput.focus();
        }
    }
}

/**
 * 处理页面可见性变化
 */
function handleVisibilityChange() {
    if (document.hidden) {
        // 页面隐藏时停止状态监控
        stopStatusMonitoring();
    } else {
        // 页面显示时恢复状态监控
        startStatusMonitoring();
        checkSystemStatus();
        // 如果在配置页面，也刷新配置状态
        if (window.location.pathname === '/config') {
            refreshConfigurationStatus();
        }
    }
}

/**
 * 执行搜索
 */
async function performSearch() {
    if (isSearching) {
        showMessage('warning', '搜索正在进行中，请稍候...');
        return;
    }
    
    const query = document.getElementById('query')?.value?.trim();
    if (!query) {
        showMessage('warning', '请输入搜索内容');
        return;
    }
    
    // 生成搜索ID
    currentSearchId = generateSearchId();
    isSearching = true;
    
    try {
        // 显示加载状态
        showLoading(true);
        hideSearchTips();
        
        // 构建搜索参数
        const searchData = buildSearchData(query);
        
        // 记录搜索开始时间
        const startTime = Date.now();
        
        // 发送搜索请求
        const response = await fetch('/api/search', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-Search-ID': currentSearchId
            },
            body: JSON.stringify(searchData)
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const searchTime = Date.now() - startTime;
        
        // 显示搜索结果
        displaySearchResults(data, searchTime);
        
        // 记录搜索历史
        recordSearchHistory(query, data.results?.length || 0, searchTime);
        
    } catch (error) {
        console.error('搜索错误:', error);
        showMessage('danger', `搜索失败: ${error.message}`);
        displaySearchError(error);
    } finally {
        isSearching = false;
        showLoading(false);
    }
}

/**
 * 构建搜索数据
 */
function buildSearchData(query) {
    return {
        query: query,
        search_type: document.getElementById('search-type')?.value || 'vector',
        max_results: parseInt(document.getElementById('max-results')?.value || '5'),
        similarity_threshold: parseFloat(document.getElementById('similarity-threshold')?.value || '0.3'),
        recall_threshold: parseFloat(document.getElementById('recall-threshold')?.value || '0.2'),
        include_llm_analysis: document.getElementById('include-llm')?.checked || false,
        search_id: currentSearchId
    };
}

/**
 * 显示搜索结果
 */
function displaySearchResults(data, searchTime) {
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    const resultsCount = document.getElementById('results-count');
    const searchTimeSpan = document.getElementById('search-time');
    
    if (!data.success) {
        showMessage('danger', `搜索失败: ${data.error}`);
        return;
    }
    
    const results = data.results || [];
    
    // 更新结果统计
    if (resultsCount) resultsCount.textContent = results.length;
    if (searchTimeSpan) searchTimeSpan.textContent = `(${searchTime}ms)`;
    
    if (results.length === 0) {
        displayNoResults();
        // 当没有搜索结果时，使用AI自己的能力进行回答
        const includeLLM = document.getElementById('include-llm')?.checked;
        if (includeLLM) {
            performDirectAIAnswer(document.getElementById('query').value);
        } else {
            hideLLMAnalysis();
        }
    } else {
        displayResultsList(results);
        // 有搜索结果时，进行RAG检索增强回答
        const includeLLM = document.getElementById('include-llm')?.checked;
        if (includeLLM) {
            performRAGAnalysis(document.getElementById('query').value, results);
        } else if (data.llm_analysis || data.llm_enhancement) {
            displayLLMAnalysis(data.llm_analysis || data.llm_enhancement);
        } else {
            hideLLMAnalysis();
        }
    }
    
    // 显示结果容器
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }
}

/**
 * 显示无结果页面
 */
function displayNoResults() {
    const resultsContent = document.getElementById('results-content');
    if (resultsContent) {
        resultsContent.innerHTML = `
            <div class="text-center py-5">
                <i class="bi bi-search text-muted" style="font-size: 3rem;"></i>
                <h5 class="mt-3 text-muted">没有找到相关结果</h5>
                <p class="text-muted mb-4">尝试以下方法可能会有帮助：</p>
                <div class="row justify-content-center">
                    <div class="col-md-8">
                        <ul class="list-unstyled text-start">
                            <li><i class="bi bi-arrow-right text-primary"></i> 调整搜索关键词</li>
                            <li><i class="bi bi-arrow-right text-primary"></i> 降低相似度阈值</li>
                            <li><i class="bi bi-arrow-right text-primary"></i> 尝试使用混合搜索</li>
                            <li><i class="bi bi-arrow-right text-primary"></i> 启用AI分析获得建议</li>
                        </ul>
                    </div>
                </div>
                <button class="btn btn-outline-primary" onclick="showSearchTips()">
                    <i class="bi bi-lightbulb"></i> 查看搜索技巧
                </button>
            </div>
        `;
    }
}

/**
 * 显示结果列表
 */
function displayResultsList(results) {
    const resultsContent = document.getElementById('results-content');
    if (resultsContent) {
        resultsContent.innerHTML = results.map((result, index) => 
            createResultItem(result, index + 1)
        ).join('');
    }
}

/**
 * 创建结果项HTML
 */
function createResultItem(result, index) {
    const similarity = result.similarity_score || 0;
    const similarityPercent = (similarity * 100).toFixed(1);
    const similarityClass = getSimilarityClass(similarity);
    
    const labels = result.labels_formatted || [];
    const labelsHtml = labels.map(label => 
        `<span class="badge label-badge" style="background-color: #${label.color}; color: ${getContrastColor(label.color)}">
            ${escapeHtml(label.name)}
        </span>`
    ).join('');
    
    const bodyPreview = result.body_preview ? escapeHtml(result.body_preview) : '';
    const solutionSummary = result.solution_summary ? escapeHtml(result.solution_summary) : '';
    
    return `
        <div class="search-result-item p-3 mb-3 border rounded" data-result-index="${index}">
            <div class="d-flex justify-content-between align-items-start mb-2">
                <h6 class="mb-1">
                    <span class="badge bg-secondary me-2">#${result.number}</span>
                    <a href="${result.url}" target="_blank" class="text-decoration-none" 
                       onclick="trackResultClick(${index}, '${result.url}')">
                        ${escapeHtml(result.title || '无标题')}
                    </a>
                </h6>
                <div class="text-end">
                    <span class="badge bg-${result.state === 'open' ? 'success' : 'secondary'}">
                        ${result.state === 'open' ? '开放' : '已关闭'}
                    </span>
                </div>
            </div>
            
            <div class="mb-2">
                <div class="d-flex align-items-center mb-1">
                    <small class="text-muted me-2">相似度:</small>
                    <div class="flex-grow-1 me-2">
                        <div class="similarity-bar ${similarityClass}" 
                             style="width: ${similarityPercent}%" 
                             title="相似度: ${similarityPercent}%"></div>
                    </div>
                    <small class="text-muted">${similarityPercent}%</small>
                </div>
            </div>
            
            ${bodyPreview ? `
                <div class="mb-2">
                    <p class="text-muted small mb-1">内容预览:</p>
                    <div class="code-block">${bodyPreview}</div>
                </div>
            ` : ''}
            
            ${solutionSummary ? `
                <div class="mb-2">
                    <p class="text-success small mb-1"><i class="bi bi-check-circle"></i> 解决方案:</p>
                    <div class="alert alert-success py-2 small">${solutionSummary}</div>
                </div>
            ` : ''}
            
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    ${labelsHtml}
                </div>
                <div class="text-muted small">
                    <i class="bi bi-calendar"></i> ${result.created_at_formatted || ''}
                    ${result.updated_at_formatted ? ` • <i class="bi bi-arrow-repeat"></i> ${result.updated_at_formatted}` : ''}
                </div>
            </div>
            
            <div class="mt-2">
                <button class="btn btn-sm btn-outline-primary me-2" 
                        onclick="copyResultLink('${result.url}')">
                    <i class="bi bi-link"></i> 复制链接
                </button>
                <button class="btn btn-sm btn-outline-secondary" 
                        onclick="shareResult(${index})">
                    <i class="bi bi-share"></i> 分享
                </button>
            </div>
        </div>
    `;
}

/**
 * 显示LLM分析结果
 */
function displayLLMAnalysis(analysis) {
    const container = document.getElementById('llm-analysis-container');
    const content = document.getElementById('llm-analysis-content');
    
    if (!analysis || !analysis.success || !analysis.answer) {
        hideLLMAnalysis();
        return;
    }
    
    const answer = escapeHtml(analysis.answer).replace(/\n/g, '<br>');
    const problemType = analysis.problem_type ? escapeHtml(analysis.problem_type) : '';
    const keywords = analysis.keywords || [];
    const confidence = analysis.confidence || 0;
    
    if (content) {
        content.innerHTML = `
            <div class="mb-3">
                <h6><i class="bi bi-lightbulb"></i> 智能分析</h6>
                <div class="alert alert-light">
                    ${answer}
                </div>
            </div>
            
            ${problemType ? `
                <div class="mb-3">
                    <h6><i class="bi bi-tag"></i> 问题类型</h6>
                    <span class="badge bg-info">${problemType}</span>
                </div>
            ` : ''}
            
            ${keywords.length > 0 ? `
                <div class="mb-3">
                    <h6><i class="bi bi-key"></i> 关键词</h6>
                    ${keywords.map(keyword => 
                        `<span class="badge bg-secondary me-1">${escapeHtml(keyword)}</span>`
                    ).join('')}
                </div>
            ` : ''}
            
            ${confidence > 0 ? `
                <div class="mb-3">
                    <h6><i class="bi bi-graph-up"></i> 置信度</h6>
                    <div class="progress">
                        <div class="progress-bar" style="width: ${confidence * 100}%" 
                             title="置信度: ${(confidence * 100).toFixed(1)}%">
                            ${(confidence * 100).toFixed(1)}%
                        </div>
                    </div>
                </div>
            ` : ''}
            
            <div class="text-end">
                <button class="btn btn-sm btn-outline-primary" onclick="copyAnalysis()">
                    <i class="bi bi-clipboard"></i> 复制分析
                </button>
            </div>
        `;
    }
    
    if (container) {
        container.style.display = 'block';
    }
}

/**
 * 隐藏LLM分析结果
 */
function hideLLMAnalysis() {
    const container = document.getElementById('llm-analysis-container');
    if (container) {
        container.style.display = 'none';
    }
}

/**
 * 显示搜索错误
 */
function displaySearchError(error) {
    const resultsContainer = document.getElementById('results-container');
    const resultsContent = document.getElementById('results-content');
    
    if (resultsContent) {
        resultsContent.innerHTML = `
            <div class="text-center py-5">
                <i class="bi bi-exclamation-triangle text-danger" style="font-size: 3rem;"></i>
                <h5 class="mt-3 text-danger">搜索出现错误</h5>
                <p class="text-muted mb-4">${escapeHtml(error.message)}</p>
                <button class="btn btn-outline-primary" onclick="performSearch()">
                    <i class="bi bi-arrow-clockwise"></i> 重试搜索
                </button>
            </div>
        `;
    }
    
    if (resultsContainer) {
        resultsContainer.style.display = 'block';
    }
}

/**
 * 保存配置
 */
async function saveConfiguration() {
    const formData = new FormData(document.getElementById('config-form'));
    const configData = {};
    
    // 处理表单数据
    for (let [key, value] of formData.entries()) {
        if (value.trim()) {
            configData[key] = value;
        }
    }
    
    // 处理复选框
    configData.use_openai_embedding = document.getElementById('use-openai-embedding')?.checked || false;
    configData.flask_debug = document.getElementById('flask-debug')?.checked || false;
    
    try {
        showLoading(true);
        
        const response = await fetch('/api/config', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(configData)
        });
        
        const data = await response.json();
        
        if (data.success) {
            showMessage('success', '配置保存成功！正在验证配置...');
            
            // 立即刷新配置状态
            await refreshConfigurationStatus();
            
            // 延迟刷新页面以确保用户看到状态更新
            setTimeout(() => {
                location.reload();
            }, 2000);
        } else {
            showMessage('danger', `配置保存失败: ${data.error}`);
        }
        
    } catch (error) {
        console.error('配置保存错误:', error);
        showMessage('danger', `配置保存失败: ${error.message}`);
    } finally {
        showLoading(false);
    }
}

/**
 * 检查系统状态
 */
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        updateStatusIndicators(data);
        
    } catch (error) {
        console.error('状态检查错误:', error);
        // 网络错误时显示离线状态
        const statusIndicator = document.getElementById('status-indicator');
        if (statusIndicator) {
            statusIndicator.innerHTML = '<i class="bi bi-circle-fill text-danger"></i> 离线';
            statusIndicator.className = 'badge bg-danger';
        }
    }
}

/**
 * 更新状态指示器
 */
function updateStatusIndicators(statusData) {
    const statusIndicator = document.getElementById('status-indicator');
    if (statusIndicator) {
        if (statusData.success && statusData.stats) {
            const stats = statusData.stats;
            if (stats.is_initialized) {
                statusIndicator.innerHTML = '<i class="bi bi-circle-fill text-success"></i> 就绪';
                statusIndicator.className = 'badge bg-success';
            } else {
                statusIndicator.innerHTML = '<i class="bi bi-circle-fill text-warning"></i> 未初始化';
                statusIndicator.className = 'badge bg-warning';
            }
        } else {
            statusIndicator.innerHTML = '<i class="bi bi-circle-fill text-danger"></i> 错误';
            statusIndicator.className = 'badge bg-danger';
        }
    }
}

/**
 * 启动状态监控
 */
function startStatusMonitoring() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
    }
    
    // 每30秒检查一次状态
    statusCheckInterval = setInterval(checkSystemStatus, 30000);
}

/**
 * 停止状态监控
 */
function stopStatusMonitoring() {
    if (statusCheckInterval) {
        clearInterval(statusCheckInterval);
        statusCheckInterval = null;
    }
}

/**
 * 加载统计数据
 */
async function loadStatistics() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        if (data.success) {
            updateStatisticsDisplay(data.stats);
        }
        
    } catch (error) {
        console.log('统计数据加载失败:', error);
    }
}

/**
 * 更新统计数据显示
 */
function updateStatisticsDisplay(stats) {
    // 更新各种统计数据的显示
    const elements = {
        'total-issues': stats.total_issues,
        'search-count': stats.search_count,
        'avg-response-time': stats.avg_response_time,
        'accuracy-rate': stats.accuracy_rate
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            element.textContent = value;
        }
    });
}

/**
 * 刷新配置状态（供按钮调用）
 */
async function refreshConfigStatus() {
    await refreshConfigurationStatus();
    showMessage('info', '配置状态已刷新');
}

/**
 * 刷新配置状态
 */
async function refreshConfigurationStatus() {
    try {
        const response = await fetch('/api/config', {
            method: 'GET'
        });
        
        const data = await response.json();
        
        if (data.success && data.config) {
            updateConfigurationStatusDisplay(data.config);
        }
        
    } catch (error) {
        console.error('刷新配置状态错误:', error);
    }
}

/**
 * 更新配置状态显示
 */
function updateConfigurationStatusDisplay(configStatus) {
    // 更新GitHub Token状态
    updateStatusIndicator('github-token', configStatus.github_token_configured, configStatus.github_token_error);
    
    // 更新GitHub仓库状态
    updateStatusIndicator('github-repo', configStatus.github_repo_configured, configStatus.github_repo_error);
    
    // 更新OpenAI API状态
    updateStatusIndicator('openai-api', configStatus.openai_api_key_configured, configStatus.openai_api_error);
    
    // 更新搜索引擎状态
    updateStatusIndicator('search-engine', configStatus.search_engine_ready);
    
    // 更新配置不完整警告
    const warningAlert = document.querySelector('.alert-warning');
    if (warningAlert) {
        const allConfigured = configStatus.github_token_configured && 
                            configStatus.github_repo_configured;
        warningAlert.style.display = allConfigured ? 'none' : 'block';
    }
    
    // 显示验证结果消息
    if (configStatus.github_token_error && !configStatus.github_token_configured) {
        showMessage('danger', `GitHub Token: ${configStatus.github_token_error}`);
    }
    if (configStatus.github_repo_error && !configStatus.github_repo_configured) {
        showMessage('danger', `GitHub 仓库: ${configStatus.github_repo_error}`);
    }
    if (configStatus.openai_api_error && !configStatus.openai_api_key_configured) {
        showMessage('warning', `OpenAI API: ${configStatus.openai_api_error}`);
    }
}

/**
 * 更新单个状态指示器
 */
function updateStatusIndicator(type, isConfigured, errorMessage = null) {
    // 查找对应的状态指示器和文本
    const indicators = document.querySelectorAll('.status-indicator');
    const statusTexts = document.querySelectorAll('.small.text-muted');
    
    let targetIndex = -1;
    switch(type) {
        case 'github-token':
            targetIndex = 0;
            break;
        case 'github-repo':
            targetIndex = 1;
            break;
        case 'openai-api':
            targetIndex = 2;
            break;
        case 'search-engine':
            targetIndex = 3;
            break;
    }
    
    if (targetIndex >= 0 && indicators[targetIndex] && statusTexts[targetIndex]) {
        const indicator = indicators[targetIndex];
        const statusText = statusTexts[targetIndex];
        
        // 清除所有状态类
        indicator.classList.remove('status-success', 'status-warning', 'status-danger');
        
        // 设置新的状态类和文本
        if (isConfigured) {
            indicator.classList.add('status-success');
            if (type === 'openai-api') {
                statusText.textContent = '已配置';
            } else {
                statusText.textContent = type === 'search-engine' ? '就绪' : '已配置';
            }
            // 移除错误提示
            indicator.removeAttribute('title');
        } else {
            if (type === 'openai-api' || type === 'search-engine') {
                indicator.classList.add('status-warning');
                statusText.textContent = type === 'openai-api' ? '未配置（可选）' : '未初始化';
            } else {
                indicator.classList.add('status-danger');
                statusText.textContent = '未配置';
            }
            
            // 添加错误提示到title属性
            if (errorMessage) {
                indicator.setAttribute('title', errorMessage);
                // 如果是必需的配置项，显示具体错误
                if (type !== 'openai-api' && type !== 'search-engine') {
                    statusText.textContent = '配置错误';
                }
            }
        }
    }
}

/**
 * 工具函数
 */

// 生成搜索ID
function generateSearchId() {
    return 'search_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
}

// 获取相似度样式类
function getSimilarityClass(similarity) {
    if (similarity >= 0.8) return 'similarity-high';
    if (similarity >= 0.6) return 'similarity-medium';
    if (similarity >= 0.3) return 'similarity-low';
    return 'similarity-very-low';
}

// 获取对比色
function getContrastColor(hexColor) {
    const r = parseInt(hexColor.substr(0, 2), 16);
    const g = parseInt(hexColor.substr(2, 2), 16);
    const b = parseInt(hexColor.substr(4, 2), 16);
    const brightness = (r * 299 + g * 587 + b * 114) / 1000;
    return brightness > 128 ? '#000000' : '#ffffff';
}

// HTML转义
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// 显示/隐藏加载状态
function showLoading(show) {
    const loadingOverlay = document.getElementById('loading-overlay');
    if (loadingOverlay) {
        loadingOverlay.style.display = show ? 'flex' : 'none';
    }
}

// 显示/隐藏搜索提示
function showSearchTips() {
    const searchTips = document.getElementById('search-tips');
    if (searchTips) {
        searchTips.style.display = 'block';
    }
}

function hideSearchTips() {
    const searchTips = document.getElementById('search-tips');
    if (searchTips) {
        searchTips.style.display = 'none';
    }
}

// 清空搜索
function clearSearch() {
    const queryInput = document.getElementById('query');
    if (queryInput) {
        queryInput.value = '';
    }
    
    const resultsContainer = document.getElementById('results-container');
    if (resultsContainer) {
        resultsContainer.style.display = 'none';
    }
    
    hideLLMAnalysis();
    showSearchTips();
}

// 显示消息
function showMessage(type, message, duration = 5000) {
    // 创建消息容器（如果不存在）
    let messageContainer = document.getElementById('message-container');
    if (!messageContainer) {
        messageContainer = document.createElement('div');
        messageContainer.id = 'message-container';
        messageContainer.style.cssText = 'position: fixed; top: 20px; right: 20px; z-index: 9999;';
        document.body.appendChild(messageContainer);
    }
    
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
    alertDiv.style.cssText = 'min-width: 300px; margin-bottom: 10px;';
    alertDiv.innerHTML = `
        <i class="bi bi-${getMessageIcon(type)}"></i> ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    messageContainer.appendChild(alertDiv);
    
    // 自动移除
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, duration);
}

// 获取消息图标
function getMessageIcon(type) {
    const icons = {
        'success': 'check-circle',
        'danger': 'exclamation-triangle',
        'warning': 'exclamation-circle',
        'info': 'info-circle'
    };
    return icons[type] || 'info-circle';
}

// 复制结果链接
function copyResultLink(url) {
    navigator.clipboard.writeText(url).then(() => {
        showMessage('success', '链接已复制到剪贴板');
    }).catch(() => {
        showMessage('danger', '复制失败，请手动复制');
    });
}

// 分享结果
function shareResult(index) {
    const resultElement = document.querySelector(`[data-result-index="${index}"]`);
    if (resultElement) {
        const title = resultElement.querySelector('a').textContent;
        const url = resultElement.querySelector('a').href;
        
        if (navigator.share) {
            navigator.share({
                title: title,
                url: url
            });
        } else {
            copyResultLink(url);
        }
    }
}

// 复制分析结果
function copyAnalysis() {
    const analysisContent = document.getElementById('llm-analysis-content');
    if (analysisContent) {
        const text = analysisContent.textContent;
        navigator.clipboard.writeText(text).then(() => {
            showMessage('success', '分析结果已复制到剪贴板');
        }).catch(() => {
            showMessage('danger', '复制失败，请手动复制');
        });
    }
}

// 跟踪结果点击
function trackResultClick(index, url) {
    // 可以在这里添加分析跟踪代码
    console.log(`用户点击了第 ${index} 个结果: ${url}`);
}

// 记录搜索历史
function recordSearchHistory(query, resultCount, searchTime) {
    const history = JSON.parse(localStorage.getItem('searchHistory') || '[]');
    history.unshift({
        query: query,
        resultCount: resultCount,
        searchTime: searchTime,
        timestamp: new Date().toISOString()
    });
    
    // 只保留最近50次搜索
    if (history.length > 50) {
        history.splice(50);
    }
    
    localStorage.setItem('searchHistory', JSON.stringify(history));
}

// 获取搜索历史
function getSearchHistory() {
    return JSON.parse(localStorage.getItem('searchHistory') || '[]');
}

// 清空搜索历史
function clearSearchHistory() {
    localStorage.removeItem('searchHistory');
    showMessage('success', '搜索历史已清空');
}

// 导出搜索结果
function exportSearchResults() {
    const results = document.querySelectorAll('.search-result-item');
    const data = Array.from(results).map(result => {
        const title = result.querySelector('a').textContent;
        const url = result.querySelector('a').href;
        const similarity = result.querySelector('.similarity-bar').style.width;
        return { title, url, similarity };
    });
    
    const blob = new Blob([JSON.stringify(data, null, 2)], {
        type: 'application/json'
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `search-results-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
    
    showMessage('success', '搜索结果已导出');
}

// 页面卸载时清理
window.addEventListener('beforeunload', function() {
    stopStatusMonitoring();
});

// 错误处理
window.addEventListener('error', function(event) {
    console.error('页面错误:', event.error);
    showMessage('danger', '页面出现错误，请刷新重试');
});

// 网络状态监控
window.addEventListener('online', function() {
    showMessage('success', '网络连接已恢复');
    checkSystemStatus();
});

window.addEventListener('offline', function() {
    showMessage('warning', '网络连接已断开');
    stopStatusMonitoring();
});

/**
 * AI分析相关函数
 */

/**
 * 执行直接AI回答（无搜索结果时）
 */
async function performDirectAIAnswer(query) {
    try {
        // 显示加载状态
        showLLMAnalysisLoading();
        showMessage('info', '正在使用AI回答您的问题...');
        
        const response = await fetch('/api/direct_ai_answer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayLLMAnalysis({
                ...data,
                analysis: data.answer || data.analysis,
                mode: 'direct'
            });
            showMessage('success', 'AI回答完成');
        } else {
            throw new Error(data.error || '回答失败');
        }
        
    } catch (error) {
        console.error('直接AI回答错误:', error);
        showMessage('danger', `AI回答失败: ${error.message}`);
        displayLLMAnalysisError(error.message);
    }
}

/**
 * 执行RAG检索增强回答（有搜索结果时）
 */
async function performRAGAnalysis(query, results) {
    try {
        // 显示加载状态
        showLLMAnalysisLoading();
        showMessage('info', '正在进行RAG检索增强分析...');
        
        const response = await fetch('/api/rag_analysis', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: query,
                results: results
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.success) {
            displayLLMAnalysis({
                ...data,
                analysis: data.answer || data.analysis,
                mode: 'rag'
            });
            showMessage('success', 'RAG分析完成');
        } else {
            throw new Error(data.error || '分析失败');
        }
        
    } catch (error) {
        console.error('RAG分析错误:', error);
        showMessage('danger', `RAG分析失败: ${error.message}`);
        displayLLMAnalysisError(error.message);
    }
}

/**
 * 执行LLM分析（兼容旧版本）
 */
async function performLLMAnalysis(query, results) {
    // 根据是否有结果选择不同的分析模式
    if (!results || results.length === 0) {
        return performDirectAIAnswer(query);
    } else {
        return performRAGAnalysis(query, results);
    }
}

// 显示AI分析结果
function displayLLMAnalysis(data) {
    const container = document.getElementById('llm-analysis-container');
    const content = document.getElementById('llm-analysis-content');
    
    if (!container || !content) return;
    
    const analysis = data.analysis || data.answer || '';
    const provider = data.provider || 'AI';
    const model = data.model || '';
    const tokensUsed = data.tokens_used || 0;
    const mode = data.mode || 'unknown';
    const contextResults = data.context_results || 0;
    
    // 根据模式显示不同的标识
    let modeInfo = '';
    if (mode === 'direct') {
        modeInfo = '<span class="badge bg-warning me-2"><i class="bi bi-lightbulb"></i> 直接回答</span>';
    } else if (mode === 'rag') {
        modeInfo = `<span class="badge bg-primary me-2"><i class="bi bi-search"></i> RAG增强 (${contextResults}个结果)</span>`;
    }
    
    content.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-3">
            <div>
                <span class="badge bg-success me-2">${provider}</span>
                ${model ? `<span class="badge bg-info me-2">${model}</span>` : ''}
                ${modeInfo}
                ${tokensUsed > 0 ? `<span class="badge bg-secondary">${tokensUsed} tokens</span>` : ''}
            </div>
            <div>
                <button class="btn btn-sm btn-outline-success me-2" onclick="copyAnalysis()" title="复制分析结果">
                    <i class="bi bi-clipboard"></i>
                </button>
                <button class="btn btn-sm btn-outline-success" onclick="regenerateAnalysis()" title="重新生成分析">
                    <i class="bi bi-arrow-clockwise"></i>
                </button>
            </div>
        </div>
        <div class="analysis-content">
            ${formatAnalysisContent(analysis)}
        </div>
    `;
    
    container.style.display = 'block';
    container.scrollIntoView({ behavior: 'smooth' });
}

// 显示AI分析加载状态
function showLLMAnalysisLoading() {
    const container = document.getElementById('llm-analysis-container');
    const content = document.getElementById('llm-analysis-content');
    
    if (!container || !content) return;
    
    content.innerHTML = `
        <div class="text-center py-4">
            <div class="spinner-border text-success mb-3" role="status">
                <span class="visually-hidden">AI分析中...</span>
            </div>
            <h6 class="text-success">AI正在分析中...</h6>
            <p class="text-muted mb-0">请稍候，正在基于搜索结果生成智能分析</p>
        </div>
    `;
    
    container.style.display = 'block';
}

// 显示AI分析错误
function displayLLMAnalysisError(errorMessage) {
    const container = document.getElementById('llm-analysis-container');
    const content = document.getElementById('llm-analysis-content');
    
    if (!container || !content) return;
    
    content.innerHTML = `
        <div class="alert alert-danger" role="alert">
            <i class="bi bi-exclamation-triangle"></i>
            <strong>AI分析失败</strong>
            <p class="mb-2">${errorMessage}</p>
            <button class="btn btn-sm btn-outline-danger" onclick="regenerateAnalysis()">
                <i class="bi bi-arrow-clockwise"></i> 重试
            </button>
        </div>
    `;
    
    container.style.display = 'block';
}

// 隐藏AI分析结果
function hideLLMAnalysis() {
    const container = document.getElementById('llm-analysis-container');
    if (container) {
        container.style.display = 'none';
    }
}

// 格式化分析内容
function formatAnalysisContent(content) {
    if (!content) return '';
    
    // 将Markdown格式转换为HTML
    let formatted = content
        // 标题
        .replace(/^### (.*$)/gm, '<h5>$1</h5>')
        .replace(/^## (.*$)/gm, '<h4>$1</h4>')
        .replace(/^# (.*$)/gm, '<h3>$1</h3>')
        // 代码块
        .replace(/```([\s\S]*?)```/g, '<pre class="bg-light p-3 rounded"><code>$1</code></pre>')
        // 行内代码
        .replace(/`([^`]+)`/g, '<code class="bg-light px-1 rounded">$1</code>')
        // 粗体
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        // 斜体
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        // 链接
        .replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
        // 换行
        .replace(/\n/g, '<br>');
    
    return formatted;
}

// 重新生成分析
function regenerateAnalysis() {
    const query = document.getElementById('query')?.value?.trim();
    const results = getCurrentSearchResults();
    
    if (query && results.length > 0) {
        performLLMAnalysis(query, results);
    } else {
        showMessage('warning', '请先进行搜索');
    }
}

// 获取当前搜索结果
function getCurrentSearchResults() {
    const resultElements = document.querySelectorAll('.search-result-item');
    const results = [];
    
    resultElements.forEach(element => {
        const titleElement = element.querySelector('a');
        const similarityElement = element.querySelector('.similarity-bar');
        
        if (titleElement && similarityElement) {
            results.push({
                title: titleElement.textContent.replace(/^#\d+\s*/, ''),
                url: titleElement.href,
                similarity_score: parseFloat(similarityElement.style.width) / 100,
                body_summary: element.querySelector('.code-block')?.textContent || ''
            });
        }
    });
    
    return results;
 }
 
 // 文件更新时间: 2025-07-24 21:40:56