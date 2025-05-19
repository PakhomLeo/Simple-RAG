# 本地 RAG 程序

本项目旨在构建一个在本地运行的检索增强生成 (RAG) 程序。用户可以上传一个或多个Markdown文档，程序会将其处理、向量化并存储。随后，用户可以通过与本地LLM对话，基于文档内容进行问答。程序会展示RAG增强的答案（附带引用来源）、直接LLM的答案作为对比，并且新增了**引用来源与两种答案的相似度匹配图表**，帮助用户更直观地理解RAG的效果。

## 实际完成了哪些具体功能？

1.  **Markdown文档处理**:
    *   支持通过前端界面上传单个或多个 Markdown (`.md`) 文件。
    *   自动从 Markdown 文件中提取纯文本内容（使用 `markdown-it-py`）。
    *   对提取的文本进行智能切块 (chunking)，块大小和重叠可配置。

2.  **向量化与存储**:
    *   调用可配置的 Embedding 模型 API（如火山方舟 `doubao-embedding-text-240715`）将文本块转换为向量。
    *   使用 Milvus 作为向量数据库，存储文本块的向量及其唯一ID。
    *   使用 SQLite 数据库存储文本块的元数据（如块ID、来源文件名）和原始内容。
    *   **数据清理**: 每次通过 `/upload/` 端点上传新文件时，Milvus 和 SQLite 中的旧数据将被自动清空，确保 RAG 基于最新的上传内容。

3.  **问答与检索**:
    *   用户可在前端界面输入问题，并配置检索的 Top-K 相关文本块数量。
    *   后端接收问题后，将其向量化，并在 Milvus 中检索 Top-K 个最相关的文本块。

4.  **大语言模型 (LLM) 集成与响应生成**:
    *   集成 Ollama 服务，支持调用本地部署的 LLM（如 `llama3:8b`, `qwen3:4b`，模型可配置）。
    *   **RAG增强回答**: 将检索到的文本块作为上下文，与用户问题一起构建提示 (prompt)，交由 LLM 生成答案。
    *   **直接LLM回答**: 同时，将用户问题直接交由同一个 LLM 生成答案，作为对比。

5.  **结果展示与分析**:
    *   前端界面清晰展示 RAG 增强后的答案、其引用的来源文本块详情（内容、来源文档名、块ID）。
    *   并排展示直接 LLM 的回答，方便用户比较。
    *   **新增 - 相似度匹配图表**:
        *   将 RAG 增强答案和直接 LLM 答案分别进行向量化。
        *   计算这两个答案向量与每一个引用来源文本块向量之间的余弦相似度。
        *   使用 Chart.js 在前端通过条形图展示这些相似度分数，每条引用来源对应两条柱子（与RAG答案的相似度、与纯LLM答案的相似度）。

6.  **灵活配置**:
    *   通过项目根目录下的 `.env` 文件进行参数配置，包括API密钥、服务地址（Milvus, Ollama）、模型名称（Embedding, LLM）、文本处理参数（块大小、重叠）、数据库文件名等。

7.  **API服务**:
    *   提供基于 FastAPI 的后端 API 服务，包含文件上传、问答查询、健康检查等端点。

8.  **用户界面**:
    *   提供简洁美观、易于操作的前端界面 (HTML, CSS, JavaScript)，支持文件上传、参数调整、提问和结果的可视化展示。

## 开发环境与编程工具是什么？

*   **后端 (Backend)**:
    *   **编程语言**: Python 3.8+
    *   **Web 框架**: FastAPI
    *   **ASGI 服务器**: Uvicorn
    *   **向量数据库**: Milvus (通过 `pymilvus` 库连接)
    *   **元数据/原文数据库**: SQLite (通过 SQLAlchemy 进行ORM操作，辅以 `crud.py` 封装)
    *   **Embedding 服务客户端**: `requests` 库 (用于调用如火山方舟的HTTP API)
    *   **LLM 客户端**: `ollama` Python 库 (用于与本地 Ollama 服务交互)
    *   **Markdown 解析**: `markdown-it-py`
    *   **文件上传处理**: `python-multipart` (FastAPI 依赖)
    *   **配置管理**: `python-dotenv` (加载 `.env` 文件)
    *   **数值计算/向量操作**: `numpy` (用于余弦相似度计算)
    *   **Python 依赖管理**: `pip` 与 `requirements.txt`

*   **前端 (Frontend)**:
    *   **核心技术**: HTML, CSS, JavaScript (原生，无大型框架)
    *   **图表库**: Chart.js (通过 CDN 引入，用于展示相似度匹配图表)

*   **大语言模型 (LLM) 服务**:
    *   Ollama (本地部署，用户需自行下载和运行模型)

*   **Embedding 模型服务**:
    *   可配置，默认为火山方舟 Embedding API (外部 HTTP 服务)

*   **版本控制**:
    *   Git (项目通过 `git clone` 获取)

*   **推荐开发/运行辅助工具**:
    *   Python 虚拟环境 (`venv`)

## 项目结构

```
RAG_Project/
├── backend/                  # 后端 Python 代码
│   ├── __init__.py
│   ├── config.py             # 配置加载 (从 .env)
│   ├── crud.py               # SQLite CRUD 操作
│   ├── database.py           # SQLite 数据库设置与 SQLAlchemy 模型
│   ├── main.py               # FastAPI 应用主文件，API 路由
│   ├── models.py             # Pydantic 数据模型 (请求/响应体)
│   └── services.py           # 核心服务逻辑 (文本处理, Embedding, Milvus, Ollama, 相似度计算)
├── frontend/                 # 前端代码
│   ├── index.html            # 主 HTML 文件
│   ├── script.js             # JavaScript 逻辑 (API 调用, DOM 操作, 图表渲染)
│   └── style.css             # CSS 样式
├── .env                      # 环境变量 (需自行创建并填充，参考下文示例)
├── requirements.txt          # Python 依赖列表
└── README.md                 # 本项目说明文档
```

## 如何运行

1.  **环境准备**:
    *   确保已安装 Python 3.8 或更高版本。
    *   确保已安装所有依赖项。可以通过运行以下命令来安装所有依赖项：
        ```bash
        pip install -r requirements.txt
        ```

2.  **启动后端服务**:
    *   确保您的终端位于项目根目录 (`RAG_Project/`) 下。
    *   运行以下命令:
        ```bash
        uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
        ```
    *   后端服务成功启动后，您应该会在终端看到类似 `Uvicorn running on http://0.0.0.0:8000` 的信息。

3.  **访问前端界面**:
    *   file:///c%3A/Users/jiang/OneDrive/code/RAG/frontend/index.html

## 主要API端点 (后端: `http://localhost:8000`)

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/PakhomLeo/Simple-RAG)
