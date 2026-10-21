# Simple RAG

[English](README.md) | **简体中文**

一个可以在本地运行的检索增强生成（RAG）小程序。上传 Markdown 文档后提出问题，页面会并排给出两个答案：
一个是本地大模型参考检索到的文档片段后给出的答案，另一个是同一个模型在没有任何上下文时的直接回答。
条形图展示每个答案与每个引用片段的相似度。

- **后端：** Python、FastAPI、SQLite（SQLAlchemy）、Milvus（pymilvus）、Ollama，以及任意 `/embeddings` 接口
- **前端：** 原生 HTML、CSS、JavaScript，Chart.js
- **许可证：** [MIT](LICENSE)

![架构图](docs/assets/architecture.svg)

## 功能

- 在浏览器里上传一个或多个 `.md` 文件。用 `markdown-it-py` 提取文本，再按固定长度切块并保留重叠
  （`CHUNK_SIZE`、`CHUNK_OVERLAP`）。
- 每个文本块生成一个 UUID。原文和文件名存进 SQLite，向量存进 Milvus。
- 向量由 HTTP Embedding 接口生成。默认是火山方舟（`doubao-embedding-text-240715`，2560 维）；
  只要请求和响应格式相同（`{"model", "input"}` → `{"data": [{"embedding": [...]}]}`），其他接口也可以用。
- 提问时先把问题向量化，从 Milvus 取出最相近的 Top-K 个文本块（L2 距离），丢掉距离超过
  `MILVUS_SEARCH_MAX_DISTANCE` 的结果，剩下的作为提示词上下文。
- 大模型运行在 Ollama 上（默认 `qwen3:4b`），答案中的 `<think>…</think>` 会被去掉。
- 返回结果包括 RAG 答案、纯 LLM 答案、引用来源，以及每个答案与每个来源之间的余弦相似度。
  页面上画的是相似度的平方。
- 每次上传都会**替换**之前的向量：处理新文件前会先删除整个 Milvus 集合。
  SQLite 中的记录不会删除（见[已知限制](#已知限制)）。

![上传与问答流程](docs/assets/flow.svg)

## 快速开始

### 环境要求

- Python 3.9 及以上（CI 在 3.10 和 3.12 上测试）
- 要得到真实答案，还需要：正在运行的 [Milvus](https://milvus.io/docs/install_standalone-docker.md)（或 Milvus Lite）、
  已拉取模型的 [Ollama](https://ollama.com/download)，以及一个 Embedding 接口的 API key

### 1. 安装

```bash
git clone https://github.com/PakhomLeo/Simple-RAG.git
cd Simple-RAG
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
```

编辑 `.env`。至少要填 `ARK_API_KEY`，并检查 `MILVUS_HOST`/`MILVUS_PORT`、`OLLAMA_BASE_URL`、
`LLM_MODEL_NAME` 和 `MILVUS_DIMENSION`（必须和 Embedding 维度一致）。如果不想装 Milvus 服务，可以
`pip install milvus-lite` 并设置 `MILVUS_CONNECTION_URI=./milvus_lite.db`。

```bash
ollama pull qwen3:4b
```

### 3. 运行

```bash
python -m backend.main
# 或：uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

打开 <http://127.0.0.1:8000/>。界面由后端同源提供，交互式 API 文档在 <http://127.0.0.1:8000/docs>。

### 不用任何 key 和模型试一试

`scripts/fake_services.py` 是一个假的 Embedding 接口加假的 Ollama，返回确定性的向量和固定格式的答案，
可以离线跑通整条流程。它只用于演示和测试。

```bash
pip install -r requirements-dev.txt        # 额外安装 milvus-lite、pytest、ruff
python scripts/fake_services.py --port 18080 --dim 64 &

export ARK_API_KEY=placeholder
export EMBEDDING_API_URL=http://127.0.0.1:18080/api/v3/embeddings
export OLLAMA_BASE_URL=http://127.0.0.1:18080
export MILVUS_CONNECTION_URI=./milvus_lite.db
export MILVUS_DIMENSION=64
python -m backend.main
```

然后在页面上上传 `examples/sample.md` 并提问。

## 配置项

所有配置都是环境变量，当前目录下的 `.env` 会自动加载。完整列表见 [`.env.example`](.env.example)。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ARK_API_KEY` | *（必填）* | 发给 Embedding 接口的 Bearer token |
| `EMBEDDING_API_URL` | 方舟 `…/api/v3/embeddings` | Embedding 接口地址 |
| `EMBEDDING_MODEL_NAME` | `doubao-embedding-text-240715` | Embedding 模型 |
| `EMBEDDING_TIMEOUT_SECONDS` | `60` | 每次 Embedding 请求的超时 |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 地址 |
| `LLM_MODEL_NAME` | `qwen3:4b` | Ollama 模型 |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus 服务 |
| `MILVUS_CONNECTION_URI` | *（空）* | 优先于 host/port；填 `.db` 路径则使用 Milvus Lite |
| `MILVUS_COLLECTION_NAME` | `rag_collection` | 集合名 |
| `MILVUS_DIMENSION` | `2560` | 向量维度，必须与 Embedding 模型一致 |
| `MILVUS_METRIC_TYPE` / `MILVUS_INDEX_TYPE` | `L2` / `IVF_FLAT` | 索引参数 |
| `MILVUS_SEARCH_MAX_DISTANCE` | `10000.0` | 距离超过该值的结果会被丢弃 |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `50` | 切块参数（字符数） |
| `SQLITE_DATABASE_URL` | `sqlite:///./rag_sqlite.db` | 存原文的 SQLite |
| `FASTAPI_HOST` / `FASTAPI_PORT` | `127.0.0.1` / `8000` | `python -m backend.main` 使用 |
| `SERVE_FRONTEND` | `true` | 在 `/` 提供 `frontend/` 页面 |
| `SIMPLE_RAG_API_KEY` | *（空）* | 设置后 `/upload/` 和 `/query/` 需要 `X-API-Key` 请求头 |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | CORS 允许的浏览器来源 |
| `MAX_UPLOAD_FILES` | `20` | 每次最多上传的文件数 |
| `MAX_UPLOAD_FILE_BYTES` | `5242880` | 单个文件大小上限（5 MiB） |
| `MAX_QUESTION_CHARS` | `4000` | 问题长度上限 |
| `MAX_TOP_K` | `50` | `top_k` 上限 |

## API

| 方法 | 路径 | 请求体 | 返回 |
| --- | --- | --- | --- |
| `POST` | `/upload/` | `multipart/form-data`，一个或多个 `files` | 每个文件一条结果：`filename`、`message`、`total_chunks` |
| `POST` | `/query/` | JSON `{"question": "...", "top_k": 3}` | `rag_answer`、`rag_sources[]`、`llm_only_answer`、`similarity_scores[]` |
| `GET` | `/health` | – | `{"status": "healthy"}` |
| `GET` | `/` | – | 网页界面 |

```bash
curl -F files=@examples/sample.md http://127.0.0.1:8000/upload/
curl -H 'Content-Type: application/json' \
     -d '{"question": "向量存在哪里？", "top_k": 3}' \
     http://127.0.0.1:8000/query/
```

设置了 `SIMPLE_RAG_API_KEY` 时，请加上 `-H "X-API-Key: <key>"`。页面上也有一个可选的 API Key 输入框。

## 安全说明

- 服务默认只监听 `127.0.0.1`。用 `python -m backend.main` 启动时，如果 `FASTAPI_HOST` 不是回环地址且没有设置
  `SIMPLE_RAG_API_KEY`，程序会拒绝启动。自己用 `uvicorn --host 0.0.0.0` 启动时不会做这个检查，请先设置 key。
- 任何能访问 `/upload/` 的人都能清空向量集合。只要别人能访问到服务，就请启用 API key。
- 上传的文档会原样放进提示词。文档里可以夹带指令来左右答案（提示注入）。模型没有任何工具，读不到也改不了别的东西，
  但如果文档来源不可信，请不要直接信任答案。
- Embedding API key 只发给 `EMBEDDING_API_URL`，不会写进日志。
- 报告漏洞请看 [SECURITY.md](SECURITY.md)。

## 目录结构

```
backend/
  config.py      从环境变量 / .env 读取配置
  database.py    SQLite 引擎与 chunks 表
  crud.py        文本块的写入与查询
  models.py      请求与响应模型（pydantic）
  services.py    切块、Embedding、Milvus、Ollama、余弦相似度
  main.py        FastAPI 应用、路由、CORS、API key、静态页面
frontend/        index.html、script.js、style.css
scripts/
  fake_services.py   用于演示和测试的假 Embedding 接口 + 假 Ollama
tests/           pytest 测试（使用假服务和 Milvus Lite）
examples/        sample.md
Markdown/        原项目附带的示例文档（见下文说明）
docs/assets/     架构图和流程图
```

## 开发

```bash
pip install -r requirements-dev.txt
ruff check backend tests scripts
ruff format --check backend tests scripts
pytest
```

测试会在空闲端口上启动假服务，并使用临时的 Milvus Lite 文件，不需要联网、key 或模型。
贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 已知限制

- 上传时只删除 Milvus 集合，不清空 SQLite，旧文本块会一直留在 `rag_sqlite.db` 里，需要时请手动删除该文件。
- 后端使用 pymilvus 的 ORM 风格接口，pymilvus 3.1 会移除这套接口，所以 `requirements.txt` 限定了 `pymilvus<3.1`。
- 切块按字符数进行，不按句子或标题。
- Chart.js 从 jsDelivr 加载（固定版本并带 Subresource Integrity 校验），所以图表需要联网。

## 示例文档

`examples/sample.md` 是为本项目编写的，和代码一样采用 MIT 许可证。
`Markdown/` 目录下的文件收集自第三方公开来源，仅用于测试，不在本项目许可证范围内，版权归原作者所有。

## 许可证

[MIT](LICENSE) © 2025 PakhomLeo
