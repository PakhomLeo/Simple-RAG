const API_BASE_URL = 'http://localhost:8000'; // 与 FastAPI 后端地址一致

const fileInput = document.getElementById('fileInput');
const uploadStatus = document.getElementById('uploadStatus');
const questionInput = document.getElementById('questionInput');
const topKInput = document.getElementById('topKInput');
const ragAnswerDiv = document.getElementById('ragAnswer');
const ragSourcesDiv = document.getElementById('ragSources');
const llmOnlyAnswerDiv = document.getElementById('llmOnlyAnswer');
const errorDisplay = document.getElementById('errorDisplay');

// 新增: 相似度图表相关 DOM Elements
const similarityChartContainer = document.getElementById('similarityChartContainer');
const similarityChartCanvas = document.getElementById('similarityChart');
const similarityInfo = document.getElementById('similarityInfo');
let similarityChartInstance = null; // 用于存储图表实例，方便更新或销毁

async function uploadFile() {
    clearMessages();
    if (!fileInput.files || fileInput.files.length === 0) { // 检查是否有文件被选择
        displayError("请先选择一个或多个 .md 文件。");
        return;
    }

    const files = fileInput.files;
    const formData = new FormData();
    let hasValidMdFile = false;

    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (file.name.endsWith('.md')) {
            formData.append('files', file); // 使用相同的键 'files' 来追加所有文件
            hasValidMdFile = true;
        } else {
            // 可以选择在这里显示一个警告，告知用户非md文件将被忽略
            console.warn(`Skipping non-markdown file: ${file.name}`);
        }
    }

    if (!hasValidMdFile) {
        displayError("未选择任何有效的 .md 文件进行上传。");
        return;
    }

    uploadStatus.textContent = `正在上传 ${formData.getAll('files').length} 个文档... ⏳`;
    uploadStatus.className = 'loading';

    try {
        const response = await fetch(`${API_BASE_URL}/upload/`, {
            method: 'POST',
            body: formData, // FormData 会自动设置正确的 Content-Type for multipart/form-data
        });

        const results = await response.json(); // 后端现在返回一个结果列表

        if (response.ok) {
            let successMessages = [];
            let errorMessages = [];
            results.forEach(result => {
                if (result.message.startsWith("Skipped") || result.message.startsWith("Error")) {
                    errorMessages.push(`<li>${escapeHTML(result.filename)}: ${escapeHTML(result.message)}</li>`);
                } else {
                    successMessages.push(`<li>${escapeHTML(result.filename)}: ${escapeHTML(result.message)} (文本块: ${result.total_chunks})</li>`);
                }
            });

            let displayText = "";
            if (successMessages.length > 0) {
                displayText += `<strong>成功处理的文件:</strong><ul>${successMessages.join('')}</ul>`;
            }
            if (errorMessages.length > 0) {
                displayText += `<strong>处理中遇到问题的文件或跳过的文件:</strong><ul>${errorMessages.join('')}</ul>`;
            }

            uploadStatus.innerHTML = displayText || "没有文件被处理。"; // 使用 innerHTML 来渲染列表
            uploadStatus.className = errorMessages.length > 0 ? 'error' : 'success'; // 如果有错误，显示错误状态
        
        } else { // 处理整个请求失败的情况 (例如 500 错误)
            // results 可能包含一个 detail 字段
            const detail = results.detail || response.statusText || "上传请求失败";
            displayError(`上传失败: ${escapeHTML(detail)}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        displayError(`上传时发生网络或处理错误: ${error.message}`);
    } finally {
        uploadStatus.classList.remove('loading');
    }
}

async function submitQuestion() {
    clearMessages();
    const question = questionInput.value.trim();
    const topK = parseInt(topKInput.value, 10);

    if (!question) {
        displayError("请输入您的问题。");
        return;
    }
    if (isNaN(topK) || topK < 1) {
        displayError("Top-K 值必须是一个大于0的数字。");
        return;
    }

    ragAnswerDiv.textContent = '正在生成答案... ⏳';
    llmOnlyAnswerDiv.textContent = '正在生成答案... ⏳';
    ragSourcesDiv.innerHTML = '';
    ragAnswerDiv.className = 'loading';
    llmOnlyAnswerDiv.className = 'loading';

    try {
        const response = await fetch(`${API_BASE_URL}/query/`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question, top_k: topK }),
        });

        const result = await response.json();

        ragAnswerDiv.classList.remove('loading');
        llmOnlyAnswerDiv.classList.remove('loading');

        if (response.ok) {
            ragAnswerDiv.textContent = result.rag_answer || '未能生成RAG增强答案。';
            llmOnlyAnswerDiv.textContent = result.llm_only_answer || '未能生成直接LLM答案。';
            
            if (result.rag_sources && result.rag_sources.length > 0) {
                ragSourcesDiv.innerHTML = result.rag_sources.map(source => 
                    `<div>
                        <strong>来源文档: ${escapeHTML(source.source_document_name || 'N/A')} (ID: ${escapeHTML(source.chunk_id)})</strong>
                        <p>${escapeHTML(source.original_content)}</p>
                    </div>`
                ).join('');
            } else {
                ragSourcesDiv.textContent = '没有找到引用来源。';
            }

            // 新增: 处理并显示相似度图表
            if (result.similarity_scores && result.similarity_scores.length > 0) {
                similarityChartContainer.style.display = 'block';
                similarityInfo.style.display = 'none';
                renderSimilarityChart(result.similarity_scores);
            } else {
                similarityChartContainer.style.display = 'none';
                similarityInfo.style.display = 'block';
                if (result.rag_sources && result.rag_sources.length > 0) {
                    similarityInfo.textContent = '未能计算相似度分数 (可能答案或来源未能成功向量化)。';
                } else {
                    similarityInfo.textContent = '没有引用来源，无法计算相似度。';
                }
                if (similarityChartInstance) {
                    similarityChartInstance.destroy(); // 如果之前有图表，销毁它
                    similarityChartInstance = null;
                }
            }

        } else {
            displayError(`查询失败: ${result.detail || response.statusText}`);
            ragAnswerDiv.textContent = '查询出错。';
            llmOnlyAnswerDiv.textContent = '查询出错。';
        }
    } catch (error) {
        console.error('Query error:', error);
        displayError(`查询时发生网络错误: ${error.message}`);
        ragAnswerDiv.classList.remove('loading');
        llmOnlyAnswerDiv.classList.remove('loading');
        ragAnswerDiv.textContent = '查询出错。';
        llmOnlyAnswerDiv.textContent = '查询出错。';
    }
}

function displayError(message) {
    errorDisplay.textContent = message;
    errorDisplay.className = 'error-message';
    uploadStatus.textContent = ''; // 清除上传状态，如果之前有的话
}

function clearMessages() {
    errorDisplay.textContent = '';
    errorDisplay.className = '';
    // uploadStatus.textContent = ''; // 保留成功的上传消息，除非再次上传
    ragAnswerDiv.textContent = '';
    ragSourcesDiv.innerHTML = '';
    llmOnlyAnswerDiv.textContent = '';
}

// Basic HTML escaping function to prevent XSS
function escapeHTML(str) {
    const div = document.createElement('div');
    div.appendChild(document.createTextNode(str));
    return div.innerHTML;
}

// Initialize: Clear any lingering messages on page load (optional)
window.onload = () => {
    // clearMessages(); // Might be too aggressive, user might want to see previous status after refresh
};

// 新增: 渲染相似度图表的函数
function renderSimilarityChart(scores) {
    if (similarityChartInstance) {
        similarityChartInstance.destroy(); // 销毁旧图表实例以避免重叠
    }

    const exponent = 2; // 可以调整这个指数，例如 2 或 3

    const labels = scores.map(score => `来源块 ID: ${score.source_chunk_id.substring(0, 8)}... (${score.source_document_name || 'N/A'})`);
    const ragSimilarityData = scores.map(score => Math.pow(score.similarity_with_rag_answer, exponent));
    const llmOnlySimilarityData = scores.map(score => Math.pow(score.similarity_with_llm_only_answer, exponent));

    const data = {
        labels: labels,
        datasets: [
            {
                label: 'RAG答案与来源相似度',
                data: ragSimilarityData,
                backgroundColor: 'rgba(75, 192, 192, 0.6)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
                label: '纯LLM答案与来源相似度',
                data: llmOnlySimilarityData,
                backgroundColor: 'rgba(255, 99, 132, 0.6)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }
        ]
    };

    const config = {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            maintainAspectRatio: false, // 允许图表根据容器调整大小
            scales: {
                y: {
                    beginAtZero: true,
                    suggestedMax: 1, // Y轴最大值为1 (余弦相似度范围)
                    title: {
                        display: true,
                        text: `调整后的相似度 (原始值^${exponent})`
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: '引用来源文本块'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += context.parsed.y.toFixed(4); // 显示4位小数
                            }
                            return label;
                        }
                    }
                }
            }
        }
    };

    similarityChartInstance = new Chart(similarityChartCanvas, config);
} 