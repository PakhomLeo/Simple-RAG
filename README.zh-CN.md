# Simple RAG

[English](README.md) | **简体中文**

一个可以在本地运行的检索增强生成（RAG）小程序。上传 Markdown 文档后提出问题，页面会并排给出两个答案：
一个是本地大模型参考检索到的文档片段后给出的答案，另一个是同一个模型在没有任何上下文时的直接回答。
条形图展示每个答案与每个引用片段的相似度。

- **后端：** Python、FastAPI、SQLite（SQLAlchemy）、Milvus（pymilvus）、Ollama，以及任意 `/embeddings` 接口
- **前端：** 原生 HTML、CSS、JavaScript，Chart.js
- **许可证：** [MIT](LICENSE)

![Simple RAG 网页界面：提问、带检索上下文的答案、无上下文的答案和相似度图表](docs/assets/screenshot.png)

*截图连接的是离线假服务（`scripts/fake_services.py`），所以答案是固定格式的。*

## 目录

- [功能](#功能)
- [工作原理](#工作原理)
  - [架构](#架构)
  - [启动过程](#启动过程)
  - [上传：解析与切块](#上传解析与切块)
  - [Embedding](#embedding)
  - [存储：SQLite 与 Milvus](#存储sqlite-与-milvus)
  - [检索](#检索)
  - [生成](#生成)
  - [相似度对比](#相似度对比)
  - [每次操作的请求数](#每次操作的请求数)
- [快速开始](#快速开始)
- [用假服务离线演示](#用假服务离线演示)
- [使用网页界面](#使用网页界面)
- [配置](#配置)
- [API](#api)
  - [认证、CORS 与限制](#认证cors-与限制)
  - [POST /upload/](#post-upload)
  - [POST /query/](#post-query)
  - [GET /health](#get-health)
  - [网页界面与 OpenAPI 文档](#网页界面与-openapi-文档)
  - [错误码](#错误码)
- [开发与测试](#开发与测试)
- [项目结构](#项目结构)
- [部署与安全](#部署与安全)
- [已知限制](#已知限制)
- [常见问题](#常见问题)
- [参与贡献](#参与贡献)
- [许可证](#许可证)

## 功能

- 在浏览器里或用 `curl` 上传一个或多个 `.md` 文件。用 `markdown-it-py` 提取文本，再按固定字符数切块并保留重叠
  （`CHUNK_SIZE`、`CHUNK_OVERLAP`）。
- 每个文本块生成一个 UUID。原文和文件名存进 SQLite，向量存进 Milvus（Milvus 服务器或本地 Milvus Lite 文件）。
- 向量由 HTTP Embedding 接口生成。默认是火山方舟（`doubao-embedding-text-240715`，2560 维）；
  只要请求和响应格式相同，其他接口也可以用（见 [Embedding](#embedding)）。
- 提问时先把问题向量化，从 Milvus 取出最相近的 Top-K 个文本块（默认 L2 距离），丢掉距离超过
  `MILVUS_SEARCH_MAX_DISTANCE` 的结果，剩下的作为提示词上下文。
- 大模型运行在 Ollama 上（默认 `qwen3:4b`），答案中的 `<think>…</think>` 会被去掉。
- 每次提问返回两个答案：一个带检索上下文（RAG），一个只用问题本身（纯 LLM）；同时返回引用来源，
  以及每个答案与每个来源之间的余弦相似度。
- 每次上传都会**替换**之前的索引，SQLite 和 Milvus 对应的是同一批文档（边界情况见[已知限制](#已知限制)）。
- 默认安全：只监听 `127.0.0.1`；没有 API key 时拒绝在公网地址上启动；CORS 白名单；限制上传文件大小和数量。
- 自带假 Embedding 接口和假 Ollama（`scripts/fake_services.py`），不需要 key 和模型，就能离线跑通整个流程和测试。

## 工作原理

### 架构

![架构图：浏览器界面调用 FastAPI 后端，后端把文本块原文存进 SQLite、向量存进 Milvus，并调用 Embedding 接口和 Ollama](docs/assets/architecture.svg)

后端是一个 FastAPI 应用（`backend/main.py`），它在同源下提供 `frontend/` 里的网页，所以浏览器不需要 CORS 配置。
`backend/services.py` 负责切块、Markdown 解析、Embedding、Milvus、Ollama 和余弦相似度；
`backend/crud.py` 和 `backend/database.py` 负责 SQLite 表。

![上传和查询流程](docs/assets/flow.svg)

### 启动过程

下面这些步骤都在导入 `backend.main` 时完成，早于服务开始接收请求：

1. `backend/config.py` 读取环境变量和 `.env` 文件（见[配置](#配置)）。
   `ARK_API_KEY` 为空时导入失败，报 `ValueError: ARK_API_KEY is not set in the environment variables.`
2. 如果 SQLite 表 `chunks` 不存在就创建。
3. 连接 Milvus 并加载集合；集合不存在时按配置创建（schema 和索引）。这一步失败只记日志，服务照样启动，
   Milvus 的错误会在第一次请求时出现。
4. 向 Ollama 请求一次模型列表。连不上时只打一条警告，继续启动。
5. 在应用 lifespan 中运行**监听地址检查**：没有设置 `SIMPLE_RAG_API_KEY` 时，拒绝在非回环地址上启动
   （详见[部署与安全](#部署与安全)）。

### 上传：解析与切块

`POST /upload/` 按下面的顺序处理一批文件：

1. 如果启用了 API key，先检查 `X-API-Key` 请求头。文件数超过 `MAX_UPLOAD_FILES` 时直接返回 `413`，什么都不改。
2. **每批上传先清空一次索引。** 先删除（drop）Milvus 集合，成功后再删除 SQLite `chunks` 表里的所有行。
   Milvus 清不掉时请求返回 `500`，SQLite 保持原样。
3. 然后逐个处理文件。某个文件出问题只写进它自己的结果，继续处理下一个：
   - 文件名只保留最后一段（去掉用 `/` 或 `\` 分隔的目录部分），去掉首尾空白，截到 255 个字符；
     名字为空时用 `unnamed.md`。文件本身从不写到磁盘。
   - 只接受以 `.md` 结尾的文件名（不区分大小写），其他文件跳过。
   - 最多读取 `MAX_UPLOAD_FILE_BYTES + 1` 个字节，超过大小限制的文件跳过。
   - 按 UTF-8 解码，其他编码的文件处理失败。
   - 用 `markdown-it-py` 解析 Markdown，保留每个 `inline` token（标题、段落、列表项；`**粗体**` 这类行内标记
     会留在文本里）和每个围栏代码块的原始内容，再用换行连接起来。
   - 按 `CHUNK_SIZE` 个字符切块，每块的起点比上一块往后 `CHUNK_SIZE - CHUNK_OVERLAP` 个字符，
     相邻两块共享 `CHUNK_OVERLAP` 个字符。默认值（200/50）下每 150 个字符开始一个新块。
     切块只看字符数，不按词、句子或标题边界。
   - 每个块用新的 UUID4 作为 `chunk_id` 存进 SQLite；然后把这个文件的所有块放在一次请求里向量化；
     最后把向量插入 Milvus 并 flush。

### Embedding

`services.get_embeddings()` 向 `EMBEDDING_API_URL` 发送一次 `POST`：

```http
Authorization: Bearer <ARK_API_KEY>
Content-Type: application/json

{"model": "<EMBEDDING_MODEL_NAME>", "input": ["text 1", "text 2"], "encoding_format": "float"}
```

期望的响应是 `{"data": [{"embedding": [0.1, ...]}, ...]}`，每个输入对应一项，顺序一致。请求超时时间是
`EMBEDDING_TIMEOUT_SECONDS`。网络错误、HTTP 错误和格式不对的响应都会抛出 `ValueError`。
向量长度和 `MILVUS_DIMENSION` 不一致时只记一条警告，随后 Milvus 插入或搜索会失败。
`ARK_API_KEY` 只发给 `EMBEDDING_API_URL`，不会写进日志。

### 存储：SQLite 与 Milvus

两个存储通过 `chunk_id` 关联：

| 存储 | 存什么 | 结构 |
| --- | --- | --- |
| SQLite（`SQLITE_DATABASE_URL`） | 文本块原文和文件名 | 表 `chunks`：`chunk_id`（字符串，主键）、`original_content`（文本）、`source_document_name`（字符串）、`upload_timestamp`（UTC） |
| Milvus（`MILVUS_COLLECTION_NAME`） | 只存向量 | `chunk_id`（`VARCHAR(36)`，主键，字段名取自 `MILVUS_ID_FIELD_NAME`）、`embedding`（`MILVUS_DIMENSION` 维的 `FLOAT_VECTOR`，字段名取自 `MILVUS_VECTOR_FIELD_NAME`） |

创建集合时，会在向量字段上按 `MILVUS_INDEX_TYPE`、`MILVUS_METRIC_TYPE` 和 `nlist = MILVUS_INDEX_PARAM_NLIST`
建索引并加载集合。集合在启动时（不存在的话）创建，上传删掉集合后，在第一次插入时重新创建，
所以 schema 和索引相关的配置改动要在下一次上传后才生效。

设置 `MILVUS_CONNECTION_URI` 可以按 URI 连接：服务器地址如 `http://localhost:19530`，或者文件路径如
`./milvus_lite.db`（使用 Milvus Lite，需要 `pip install milvus-lite`）。为空时使用 `MILVUS_HOST` 和 `MILVUS_PORT`。

### 检索

`POST /query/` 先把问题向量化，再在向量字段上搜索，参数为 `metric_type = MILVUS_METRIC_TYPE`、
`nprobe = MILVUS_SEARCH_PARAM_NPROBE`、`limit = top_k`、`consistency_level = MILVUS_CONSISTENCY_LEVEL`。
`distance > MILVUS_SEARCH_MAX_DISTANCE` 的结果会被丢掉。这个过滤保留的是*小*距离，适合默认的 `L2`（越小越近）。
默认阈值 `10000.0` 非常大，实际上几乎不过滤任何结果；想去掉弱匹配就调小它。
剩下的 `chunk_id` 再到 SQLite 里查出原文和文件名。

### 生成

上下文是检索到的各块原文，用 `\n\n---\n\n` 连接。RAG 提示词是写死在 `backend/main.py` 里的中文模板：

```text
以下是相关背景信息：

<context>

---
基于以上信息，请回答问题：<question>
```

如果没有任何块通过过滤，`<context>` 就是 `没有在文档中找到相关信息。`，RAG 提示词照样发送。
纯 LLM 的提示词就是问题本身。两个提示词都作为一条 user 消息，通过 Ollama 的 chat 接口
（`ollama.Client(host=OLLAMA_BASE_URL).chat`）发给 `LLM_MODEL_NAME`，不使用流式输出。
答案里的 `<think>…</think>` 会被删除，并去掉首尾空白。

### 相似度对比

只要找到至少一个来源块，后端就会把 RAG 答案、纯 LLM 答案和每个来源块的原文分别向量化
（来源块的向量是重新计算的，不是从 Milvus 读出来的），再计算每个答案和每个来源块的余弦相似度：

```text
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

向量为空、长度不同或全为零时结果是 `0.0`。结果放在 `similarity_scores` 里：每个来源块一项，包含
`similarity_with_rag_answer` 和 `similarity_with_llm_only_answer`。没有来源，或者某个答案为空时，
`similarity_scores` 为 `null`。网页上画的是每个值的**平方**（`frontend/script.js` 中 `exponent = 2`），
让差别更明显；API 返回的是原始余弦值。

### 每次操作的请求数

| 操作 | Embedding 接口请求数 | Ollama 请求数 |
| --- | --- | --- |
| 上传 | 每个被接受且非空的文件 1 次（这个文件的所有块放在一次请求里） | 0 |
| 有 N 个来源的查询 | 1（问题）+ 2（两个答案）+ N（每个来源块 1 次） | 2 |
| 没有来源的查询 | 1 | 2 |

## 快速开始

### 环境要求

- Python 3.9 或更新版本（见 `pyproject.toml`；CI 在 3.10 和 3.12 上运行）
- 要得到真实答案，还需要：
  - 运行中的 [Milvus](https://milvus.io/docs/install_standalone-docker.md) 服务器，或 Milvus Lite（`pip install milvus-lite`）；
  - 安装了 [Ollama](https://ollama.com/download) 并拉取了模型；
  - 一个 Embedding 接口的 API key（默认是火山方舟）。
- [离线演示](#用假服务离线演示)和测试不需要任何 key 和模型。

### 1. 安装

```bash
git clone https://github.com/PakhomLeo/Simple-RAG.git
cd Simple-RAG
python -m venv .venv
source .venv/bin/activate          # Windows：.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 配置

```bash
cp .env.example .env
ollama pull qwen3:4b
```

编辑 `.env`：至少要设置 `ARK_API_KEY`，并检查 `MILVUS_HOST`/`MILVUS_PORT`（或 `MILVUS_CONNECTION_URI`）、
`OLLAMA_BASE_URL`、`LLM_MODEL_NAME` 和 `MILVUS_DIMENSION`（必须和 Embedding 维度一致）。
不想装 Milvus 服务器的话，`pip install milvus-lite` 后设置 `MILVUS_CONNECTION_URI=./milvus_lite.db`。

### 3. 运行

在项目根目录执行：

```bash
python -m backend.main
# 或者：uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`python -m backend.main` 监听 `FASTAPI_HOST:FASTAPI_PORT`（默认 `127.0.0.1:8000`），并开启自动重载。
**使用 Milvus Lite 时请用 `uvicorn` 命令启动**（见[已知限制](#已知限制)）。打开 <http://127.0.0.1:8000/>。交互式 API 文档在 <http://127.0.0.1:8000/docs>。

> **请始终通过后端的 `/` 打开界面。** 不支持直接从磁盘打开 `frontend/index.html`（`file://`），
> 浏览器会拦截 API 请求。

### 4. 试一试

上传 `examples/sample.md`（英文）或 `examples/sample.zh-CN.md`（中文），然后试试文件末尾列出的问题。

## 用假服务离线演示

`scripts/fake_services.py` 是一个小 HTTP 服务，同时扮演 Embedding 接口和 Ollama：

- `POST …/embeddings` 返回确定性的“字符袋”向量（每个非空白字符哈希到 `--dim` 个桶中的一个，再做归一化），
  所以共享字符越多的文本越接近。
- `POST /api/chat` 对 RAG 提示词返回 `[fake LLM, with RAG context] <上下文的前 160 个字符>`，
  其他提示词返回 `[fake LLM, no context] I received a N-character question.`。
- `GET /api/tags` 返回一个模型，让启动时的 Ollama 检查通过。

它不需要 key 和模型，只用于演示和测试。

```bash
pip install -r requirements-dev.txt        # 额外安装 milvus-lite、pytest、httpx、ruff
python scripts/fake_services.py --port 18080 --dim 64 &

export ARK_API_KEY=placeholder              # 不能为空；假服务不校验它
export EMBEDDING_API_URL=http://127.0.0.1:18080/api/v3/embeddings
export OLLAMA_BASE_URL=http://127.0.0.1:18080
export MILVUS_CONNECTION_URI=./milvus_lite.db
export MILVUS_DIMENSION=64                  # 必须和 --dim 一致
uvicorn backend.main:app --host 127.0.0.1 --port 8000   # 用 Milvus Lite 时不要用 python -m backend.main
```

然后打开 <http://127.0.0.1:8000/>，上传 `examples/sample.md` 或 `examples/sample.zh-CN.md` 并提问，
也可以用 [API](#api) 一节里的 `curl` 示例。顶部的截图就是这样截的。
用完后停掉两个进程（后端按 `Ctrl+C`，假服务用 `kill %1`）。演示会在当前目录生成 `milvus_lite.db` 和
`rag_sqlite.db`，两者都已被 git 忽略。

假服务的参数：`--host`（默认 `127.0.0.1`）、`--port`（默认 `18080`）、`--dim`（默认 `64`）。

## 使用网页界面

页面（`frontend/index.html`，中文界面）分四部分：

1. **上传 Markdown 文档**：选一个或多个 `.md` 文件后点上传，页面会列出每个文件的结果（文本块数，或被跳过的原因）。
   每次上传都会替换索引。
2. **配置与提问**：Top-K（表单里是 1–10；API 允许到 `MAX_TOP_K`）、可选的 API Key 输入框
   （只有后端设置了 `SIMPLE_RAG_API_KEY` 时才需要填，值保存在当前标签页的 `sessionStorage` 里），以及问题。
3. **结果展示**：RAG 答案和纯 LLM 答案并排显示，下面是相似度条形图（余弦相似度的平方，每个来源块一组柱子，
   标签是 `chunk_id` 前 8 个字符和文件名）。
4. **引用来源详情**：每个来源块的文件名、`chunk_id` 和完整原文。

服务端返回的文本在显示前都会做 HTML 转义。Chart.js 从 jsDelivr 加载（4.5.1 版，带子资源完整性校验），
所以图表需要联网。

## 配置

所有配置都是环境变量。`backend/config.py` 还会用 python-dotenv 加载 `.env` 文件：从 `backend/` 目录开始逐级往上找，
所以把 `.env` 放在项目根目录即可。环境里已经设置的变量优先于 `.env`。配置只在导入时读取一次，改完要重启服务。
[`.env.example`](.env.example) 列出了一般需要设置的所有变量。

### Embedding 接口

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ARK_API_KEY` | *（无，必填）* | Embedding 接口的 Bearer token。为空时应用无法启动。使用假服务时任意非空值即可。 |
| `EMBEDDING_API_URL` | `https://ark.cn-beijing.volces.com/api/v3/embeddings` | Embedding 接口地址（格式见 [Embedding](#embedding)） |
| `EMBEDDING_MODEL_NAME` | `doubao-embedding-text-240715` | 请求里 `model` 字段的值 |
| `EMBEDDING_TIMEOUT_SECONDS` | `60` | 每次 Embedding 请求的超时秒数（浮点数） |

### 大模型（Ollama）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama 服务地址 |
| `LLM_MODEL_NAME` | `qwen3:4b` | 两个答案都用这个 Ollama 模型 |

### Milvus

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MILVUS_CONNECTION_URI` | *（空）* | 设置后代替 host 和 port：可以是服务器 URI（`http://localhost:19530`），也可以是 Milvus Lite 文件（`./milvus_lite.db`）。不要用 `MILVUS_URI`：pymilvus 会自己读取这个变量，而且只接受 http URI。 |
| `MILVUS_HOST` | `localhost` | Milvus 服务器主机，`MILVUS_CONNECTION_URI` 为空时使用 |
| `MILVUS_PORT` | `19530` | Milvus 服务器端口，`MILVUS_CONNECTION_URI` 为空时使用 |
| `MILVUS_COLLECTION_NAME` | `rag_collection` | 集合名。每次上传都会删除并重建。 |
| `MILVUS_DIMENSION` | `2560` | 向量维度，必须和 Embedding 模型一致（`doubao-embedding-text-240715` 是 2560）。 |
| `MILVUS_METRIC_TYPE` | `L2` | 索引和搜索使用的度量。距离过滤假定越小越近（`L2`）。 |
| `MILVUS_INDEX_TYPE` | `IVF_FLAT` | 向量字段的索引类型 |
| `MILVUS_INDEX_PARAM_NLIST` | `128` | 索引参数 `nlist` |
| `MILVUS_SEARCH_PARAM_NPROBE` | `10` | 搜索参数 `nprobe` |
| `MILVUS_CONSISTENCY_LEVEL` | `Strong` | 搜索的一致性级别 |
| `MILVUS_SEARCH_MAX_DISTANCE` | `10000.0` | 距离大于此值的结果在进入提示词前被丢弃 |
| `MILVUS_ID_FIELD_NAME` | `chunk_id` | 主键字段名。`.env.example` 里没有，一般不需要改。 |
| `MILVUS_VECTOR_FIELD_NAME` | `embedding` | 向量字段名。`.env.example` 里没有，一般不需要改。 |
| `MILVUS_INDEX_FIELD_NAME` | `embedding` | `config.py` 会读取，但代码里没有用到 |
| `MILVUS_TEXT_FIELD_NAME` | `text` | `config.py` 会读取，但没有用到：原文存在 SQLite，不存在 Milvus |

### 文本处理与存储

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHUNK_SIZE` | `200` | 每块的字符数 |
| `CHUNK_OVERLAP` | `50` | 相邻两块共享的字符数。**必须小于 `CHUNK_SIZE`**，否则切块永远不会结束，上传会卡住。 |
| `SQLITE_DATABASE_URL` | `sqlite:///./rag_sqlite.db` | 文本块存储的 SQLAlchemy URL。相对路径相对于当前工作目录。只支持 SQLite（引擎传了 SQLite 专用的 `check_same_thread` 参数）。 |

### 服务

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FASTAPI_HOST` | `127.0.0.1` | `python -m backend.main` 的监听地址（`uvicorn` 命令行不读它） |
| `FASTAPI_PORT` | `8000` | `python -m backend.main` 的监听端口 |
| `SERVE_FRONTEND` | `true` | 是否在 `/` 提供 `frontend/`。`1`、`true`、`yes`、`on`（不区分大小写）表示开启，其他值都表示关闭。 |

### 安全

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SIMPLE_RAG_API_KEY` | *（空）* | 设置后，`/upload/` 和 `/query/` 必须带值相同的 `X-API-Key` 请求头。要监听非回环地址就必须设置。 |
| `SIMPLE_RAG_BIND_HOST` | *（空）* | 启动时监听地址检查所用的地址。`python -m backend.main` 会自动设置。在无法识别地址的启动方式下（gunicorn、自己写的 `uvicorn.run(...)`）请手动设置。 |
| `UVICORN_HOST` | *（空）* | 不是 Simple RAG 自己的配置，但 `uvicorn` 命令行没带 `--host` 时，监听地址检查会读它 |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | 允许跨域访问的浏览器来源，逗号分隔。自带的 `/` 页面是同源的，不需要配置。 |

### 限制

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_UPLOAD_FILES` | `20` | 每次上传的文件数上限；超过返回 `413` |
| `MAX_UPLOAD_FILE_BYTES` | `5242880` | 单个文件大小上限（5 MiB）；超过的文件被跳过 |
| `MAX_QUESTION_CHARS` | `4000` | 问题最大长度；超过返回 `422` |
| `MAX_TOP_K` | `50` | `top_k` 上限；超过返回 `422` |

## API

默认地址是 `http://127.0.0.1:8000`。下面的示例是在设置了 API key 的[离线演示](#用假服务离线演示)上实际运行得到的；
ID、文本块内容和分数来自那次运行，你那边会不一样。

| 方法 | 路径 | 认证 | 用途 |
| --- | --- | --- | --- |
| `POST` | [`/upload/`](#post-upload) | 启用时需 `X-API-Key` | 用一批 Markdown 文件替换索引 |
| `POST` | [`/query/`](#post-query) | 启用时需 `X-API-Key` | 提问，得到 RAG 答案和纯 LLM 答案 |
| `GET` | [`/health`](#get-health) | 无 | 存活检查 |
| `GET` | [`/`](#网页界面与-openapi-文档) | 无 | 网页界面（`SERVE_FRONTEND` 开启时） |
| `GET` | [`/docs`、`/redoc`、`/openapi.json`](#网页界面与-openapi-文档) | 无 | FastAPI 自动生成的 API 文档 |

注意 `/upload/` 和 `/query/` 末尾有斜杠。

### 认证、CORS 与限制

- **API key**：默认关闭。设置 `SIMPLE_RAG_API_KEY` 后，`/upload/` 和 `/query/` 需要 `X-API-Key: <key>`，
  比较是常数时间的。缺少或错误时返回 `401 {"detail": "Invalid or missing API key."}`。
  `/health`、`/` 和文档页面始终不需要 key。
- **CORS**：只有 `CORS_ALLOW_ORIGINS` 里的来源会得到 CORS 响应头。允许的方法是 `GET` 和 `POST`，
  允许的请求头是 `Content-Type` 和 `X-API-Key`，不允许携带凭据。其他来源的预检请求返回 `400`。
- **限制**：每次上传最多 `MAX_UPLOAD_FILES` 个文件，单个文件最大 `MAX_UPLOAD_FILE_BYTES`，
  `1 ≤ len(question) ≤ MAX_QUESTION_CHARS`，`1 ≤ top_k ≤ MAX_TOP_K`。没有限流。

没有设置 key 时，去掉示例中的 `-H "X-API-Key: …"` 这一行。

### POST /upload/

用上传的文件替换整个索引（Milvus 和 SQLite）。请求体为 `multipart/form-data`，包含一个或多个名为 `files` 的字段。

```bash
curl -H "X-API-Key: $SIMPLE_RAG_API_KEY" \
     -F files=@examples/sample.md \
     -F files=@examples/sample.zh-CN.md \
     -F files=@LICENSE \
     http://127.0.0.1:8000/upload/
```

响应 `200`：每个文件一个对象，顺序与请求一致。

```json
[
  {"filename": "sample.md", "message": "Document processed and stored successfully.", "total_chunks": 12},
  {"filename": "sample.zh-CN.md", "message": "Document processed and stored successfully.", "total_chunks": 5},
  {"filename": "LICENSE", "message": "Skipped: Invalid file type. Only .md files are allowed.", "total_chunks": 0}
]
```

单个文件出问题**不会**改变状态码：请求仍然返回 `200`，由该文件的 `message` 说明原因（`total_chunks` 为 `0`）：

| `message` | 原因 |
| --- | --- |
| `Document processed and stored successfully.` | 成功；`total_chunks` 是文本块数量 |
| `Skipped: Invalid file type. Only .md files are allowed.` | 文件名不以 `.md` 结尾 |
| `Skipped: file is larger than <N> bytes.` | 超过 `MAX_UPLOAD_FILE_BYTES` |
| `File content is empty or too short to be chunked.` | 解析后没有文本 |
| `Error processing document (ValueError): …` | 不是合法的 UTF-8，或者 Embedding 接口调用失败 |
| `Embedding generation failed or count mismatch.` | Embedding 接口返回的向量数量不对 |
| `Service connection error during processing: …` | 连接 Milvus 失败 |
| `Runtime error during processing: …` | 插入 Milvus 失败（例如维度不一致） |
| `An unexpected error occurred. See the server log for details.` | 其他错误 |

请求级错误：

| 状态码 | `detail` | 原因 |
| --- | --- | --- |
| `401` | `Invalid or missing API key.` | 启用了 API key，但请求头缺失或错误 |
| `413` | `Too many files. At most <N> files per upload.` | 文件数超过 `MAX_UPLOAD_FILES`；什么都没有改 |
| `422` | FastAPI 校验错误列表 | 表单里没有 `files` 字段 |
| `500` | `Failed to clear the vector store before upload.` | Milvus 清空失败；SQLite 没有动 |
| `500` | `Failed to clear the chunk store before upload.` | SQLite 清空失败（Milvus 已经清空） |

### POST /query/

请求体为 JSON `{"question": string, "top_k": integer}`。`top_k` 可选，默认 `3`。

```bash
curl -H "X-API-Key: $SIMPLE_RAG_API_KEY" \
     -H 'Content-Type: application/json' \
     -d '{"question": "Where are the vectors stored?", "top_k": 2}' \
     http://127.0.0.1:8000/query/
```

响应 `200`（文本块内容用 `…` 做了截断）：

```json
{
  "rag_answer": "[fake LLM, with RAG context] s the Milvus collection and then deletes all chunk rows from SQLite, …",
  "rag_sources": [
    {
      "chunk_id": "248db5b1-3ec3-47d3-ac38-56bba9ef905e",
      "original_content": "s the Milvus collection and\nthen deletes all chunk rows from SQLite, …",
      "source_document_name": "sample.md"
    },
    {
      "chunk_id": "43d2950f-bc0c-4f21-923e-dbe8830f9eae",
      "original_content": " the prompt. Showing the answer with context next to the answer without context\n…",
      "source_document_name": "sample.md"
    }
  ],
  "llm_only_answer": "[fake LLM, no context] I received a 29-character question.",
  "similarity_scores": [
    {
      "source_chunk_id": "248db5b1-3ec3-47d3-ac38-56bba9ef905e",
      "source_document_name": "sample.md",
      "similarity_with_rag_answer": 0.9887575046469967,
      "similarity_with_llm_only_answer": 0.8429327981243102
    },
    {
      "source_chunk_id": "43d2950f-bc0c-4f21-923e-dbe8830f9eae",
      "source_document_name": "sample.md",
      "similarity_with_rag_answer": 0.9180291468713477,
      "similarity_with_llm_only_answer": 0.8653134681885776
    }
  ]
}
```

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `rag_answer` | string | RAG 提示词（问题加检索上下文）的答案 |
| `rag_sources` | array | 通过距离过滤的检索结果：`chunk_id`、`original_content`、`source_document_name`。没找到时为空数组。顺序是 SQLite 返回行的顺序，不一定按相似度排名。 |
| `llm_only_answer` | string | 只用问题本身得到的答案 |
| `similarity_scores` | array 或 `null` | 每个答案与每个来源块的余弦相似度（见[相似度对比](#相似度对比)）；没有来源时为 `null` |

索引为空时查询也会成功：

```json
{"rag_answer": "[fake LLM, with RAG context] 没有在文档中找到相关信息。", "rag_sources": [],
 "llm_only_answer": "[fake LLM, no context] I received a 5-character question.", "similarity_scores": null}
```

错误：

| 状态码 | `detail` | 原因 |
| --- | --- | --- |
| `401` | `Invalid or missing API key.` | 启用了 API key，但请求头缺失或错误 |
| `422` | FastAPI 校验错误列表 | `question` 为空或过长、`top_k` 超出范围、JSON 不合法 |
| `400` | `Embedding API request failed: …` 或 `Invalid response from Embedding API: …` | Embedding 接口连不上、返回 HTTP 错误或格式不对（以 `ValueError` 抛出） |
| `503` | `Service connection error: …` | 连接 Milvus 失败 |
| `500` | `Runtime error during processing: …` | Milvus 搜索或 Ollama 调用失败 |
| `500` | `An unexpected error occurred. See the server log for details.` | 其他错误；细节只写在服务端日志里 |

校验错误示例（`"top_k": 0`）：

```json
{"detail": [{"type": "greater_than_equal", "loc": ["body", "top_k"], "msg": "Input should be greater than or equal to 1", "input": 0, "ctx": {"ge": 1}}]}
```

### GET /health

```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy"}
```

只要进程在运行就返回 `200`，不检查 Milvus、Ollama 或 Embedding 接口。

### 网页界面与 OpenAPI 文档

- `SERVE_FRONTEND` 开启且 `frontend/` 目录存在时，`GET /` 返回 `frontend/index.html`，同时提供 `/script.js` 和
  `/style.css`。`SERVE_FRONTEND=false` 时只提供 API。
- FastAPI 自动生成的文档在 `/docs`（Swagger UI）、`/redoc` 和 `/openapi.json`，和 `/health` 一样不需要 key。
  可以在 `/docs` 里直接试 `/query/`；启用了 key 时要加上 `X-API-Key` 请求头。

### 错误码

| 状态码 | 端点 | 含义 |
| --- | --- | --- |
| `200` | 全部 | 成功。`/upload/` 要逐个查看每个文件的 `message`。 |
| `400` | `/query/` | Embedding 接口失败。不在白名单里的来源发起 CORS 预检时也返回 `400`。 |
| `401` | `/upload/`、`/query/` | `X-API-Key` 缺失或错误 |
| `413` | `/upload/` | 文件数过多 |
| `422` | `/upload/`、`/query/` | 请求校验失败 |
| `500` | `/upload/`、`/query/` | 存储清空失败；Milvus 或 Ollama 出错；意外错误 |
| `503` | `/query/` | 连接 Milvus 失败 |

错误响应体是 `{"detail": "<message>"}`；`422` 例外，它的 `detail` 是 FastAPI 的校验错误列表。

## 开发与测试

```bash
pip install -r requirements-dev.txt
ruff check backend tests scripts
ruff format --check backend tests scripts
pytest
```

`requirements-dev.txt` 在运行依赖之外加上了 `pytest`、`httpx`、`ruff` 和 `milvus-lite`。

测试不需要网络、key 和模型。`tests/conftest.py` 会在后台线程里用一个空闲端口启动 `scripts/fake_services.py`，
让后端连到它，使用临时的 Milvus Lite 文件和临时 SQLite 文件（向量 32 维），并为监听地址检查设置
`SIMPLE_RAG_BIND_HOST=127.0.0.1`。测试共 46 个，覆盖：

- `tests/test_services.py`：切块重叠、Markdown 文本提取、余弦相似度的边界情况、Embedding 请求（超时、错误处理）、
  默认主机都是本地地址；
- `tests/test_api.py`：健康检查和同源网页、上传和查询端到端、文件名清理、文件大小/数量/UTF-8 处理、
  `top_k` 和问题校验、CORS、API key、意外错误不泄露细节、上传同时清空两个存储（以及 Milvus 清空失败时保留 SQLite）、
  多种启动命令下的监听地址检查。

CI（`.github/workflows/ci.yml`）在 Python 3.10 和 3.12 上运行 ruff 和 pytest，并用 gitleaks 扫描完整 git 历史。
提交 PR 前的检查清单见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## 项目结构

```text
backend/
  config.py      从环境变量 / .env 读取配置
  database.py    SQLite 引擎和 chunks 表
  crud.py        文本块的写入、查询和删除
  models.py      请求和响应模型（pydantic）
  services.py    切块、Markdown 解析、Embedding、Milvus、Ollama、余弦相似度
  main.py        FastAPI 应用、路由、CORS、API key、启动时监听地址检查、静态页面
frontend/        index.html、script.js、style.css
scripts/
  fake_services.py   用于演示和测试的假 Embedding 接口 + 假 Ollama
tests/           pytest 测试（使用假服务和 Milvus Lite）
examples/        sample.md（英文）、sample.zh-CN.md（中文）
docs/assets/     架构图、流程图和界面截图
.env.example     配置模板
```

## 部署与安全

Simple RAG 是一个本地、单用户的工具，没有针对公网暴露做加固。让别人能访问之前，请先读 [SECURITY.md](SECURITY.md)。

- **监听地址和启动检查**：默认只监听 `127.0.0.1`。应用在 lifespan 中判断自己的监听地址，没有设置
  `SIMPLE_RAG_API_KEY` 时**拒绝**在非回环地址（例如 `0.0.0.0`）上启动。`localhost`、`127.0.0.0/8`、`::1`
  和 unix socket 都算回环。地址按下面的顺序确定：
  1. `SIMPLE_RAG_BIND_HOST`。`python -m backend.main` 会用 `FASTAPI_HOST` 设置它，并且在启动 uvicorn 之前，
     就会拒绝没有 key 的公网 `FASTAPI_HOST`。
  2. 对 `uvicorn` 命令行：`--uds`（算回环）、`--fd`（无法判断）、`--host`、再看 `UVICORN_HOST`，
     都没有时用 uvicorn 的默认值 `127.0.0.1`。

  其他启动方式（例如 gunicorn 或自己写的 `uvicorn.run(...)`）无法识别监听地址，应用会改为打印一条显眼的
  `SECURITY WARNING`；把实际绑定的地址写进 `SIMPLE_RAG_BIND_HOST` 就能在这些方式下启用检查。
- **只要别人能访问服务器，就要启用 API key。** 任何能调用 `/upload/` 的人都能替换整个索引。
  只有一个共享 key，没有用户体系，也没有限流。
- **没有 TLS**：应用只提供明文 HTTP。如果 key 要经过网络传输，请放在带 TLS 的反向代理后面。
- **不要开自动重载**：`python -m backend.main` 总是以 `reload=True` 启动 uvicorn，这是为开发准备的。
  长期运行的服务请用 `uvicorn backend.main:app --host … --port …`，不要加 `--reload`。
- **无需认证的端点**：`/health`、`/`、`/docs`、`/redoc` 和 `/openapi.json` 永远不需要 key。
- **错误细节**：`/query/` 的 `400`、`500`（`Runtime error …`）和 `503` 响应，以及上传时每个文件的消息，
  都包含底层异常的文本，其中可能有内部主机和端口。意外错误只返回一条通用消息。
- **CORS**：`CORS_ALLOW_ORIGINS` 只放你信任的来源。自带页面是同源的，不需要它。
- **数据**：上传的文件从不写到磁盘；原文和文件名（只保留最后一段）存在 SQLite 文件里，向量存在 Milvus 里，
  两者都要保护好。Embedding API key 只发给 `EMBEDDING_API_URL`，不写进日志。不要提交 `.env`。
- **提示注入**：上传的文档原样放进大模型提示词，文档内容可以左右答案。大模型没有任何工具，影响仅限于答案文本。
  见 [SECURITY.md](SECURITY.md)。
- **报告漏洞**：请按 [SECURITY.md](SECURITY.md) 的说明私下报告。

## 已知限制

- **每个服务只有一个索引**：每次上传都会替换它，不能往已有索引里追加文档。请把所有文档放在一次上传里
  （最多 `MAX_UPLOAD_FILES` 个）。
- **先清空索引，再检查文件**：一次上传里即使所有文件都被跳过或处理失败，索引也已经被清空。清空之前只检查
  API key 和文件数（`413`）。
- **文件处理失败后 SQLite 里可能留下行**：文本块先写进 SQLite，再做向量化。某个文件的 Embedding 调用或
  Milvus 插入失败时，它的行会留在 SQLite 里、没有对应向量。这些行不会被检索到（检索在 Milvus 上进行），
  下一次上传时会被删除。
- **`CHUNK_OVERLAP` 必须小于 `CHUNK_SIZE`**：代码不校验这两个值，否则上传永远不会结束。
- **切块很简单**：只按字符数切，块的开头和结尾可能落在词中间；行内 Markdown 标记会留在文本里。
- **来源顺序**：`rag_sources` 和 `similarity_scores` 按 SQLite 返回行的顺序排列，不一定按距离排名。
- **距离过滤**：它假定越小越近，适用于 `L2`，不适用于 `IP`、`COSINE` 这类相似度度量。默认阈值 `10000.0`
  几乎不过滤任何结果。
- **维度不一致只是警告**：Embedding 返回时只记警告，随后 Milvus 插入或搜索才失败。
- **每个文件一次 Embedding 请求**：一个文件的所有块放在一次请求里，所以会受到服务商对单次输入数量的限制。
  一次查询要发 `3 + N` 次 Embedding 请求（N 为来源数）。
- **中文提示词模板**：RAG 提示词和“没有找到”的提示是中文，界面文字也是中文。
- **网页界面细节**：Top-K 输入框只允许 1–10，而 API 允许到 `MAX_TOP_K`。浏览器端的 `.md` 检查区分大小写
  （后端不区分）。上传结果列表只把以 `Skipped` 或 `Error` 开头的消息标为问题，其他失败消息会出现在成功列表里。
- **`python -m backend.main` 不能配合 Milvus Lite 使用**：它总是以自动重载方式运行 uvicorn。重载监控进程会先导入应用并打开
  Milvus Lite 文件，它启动的服务进程就无法再打开同一个文件（`Open local milvus failed`），上传会报
  `Failed to clear the vector store before upload.`。请改用 `uvicorn backend.main:app --host 127.0.0.1 --port 8000` 启动。
  这个问题只出现在 Milvus Lite 文件上。
- **不支持流式输出**：两个答案都完整生成后才返回。
- **启动比较宽松**：启动时 Milvus 或 Ollama 不可用，服务也照样启动，`/health` 始终报告 `healthy`。
- **pymilvus**：后端使用 ORM 风格的 pymilvus API，pymilvus 3.1 删除了这套 API，所以 `requirements.txt`
  固定为 `pymilvus<3.1`。
- **Chart.js 来自 CDN**：图表需要联网。

## 常见问题

**启动时报 `ARK_API_KEY is not set in the environment variables.` 退出。**
这个 key 在导入时就是必需的，请在 `.env` 或环境变量里设置。使用假服务时任意非空值即可。

**服务拒绝启动：`Refusing to start: listening on non-loopback address …`。**
你绑定了公网地址但没有设置 `SIMPLE_RAG_API_KEY`。设置 key，或者改为绑定 `127.0.0.1`。见[部署与安全](#部署与安全)。

**日志里出现 `Milvus dimension (…) might not match actual embedding dimension (…)`，上传失败。**
把 `MILVUS_DIMENSION` 设成 Embedding 模型实际返回的维度，重启后重新上传。上传会删掉旧集合，并按新维度重建。

**上传第二个文件后，第一个文件不见了。**
这是设计如此：每次上传都会替换索引。请在一次上传里选中所有文件。

**怎样全部重置？**
重新上传一批文件；或者停掉服务，删除 SQLite 文件（默认 `rag_sqlite.db`）和 Milvus Lite 文件或 Milvus 集合
（默认 `rag_collection`）。

**能换别的 Embedding 服务吗？**
可以，只要它接受带 `Authorization: Bearer <key>` 的 `POST`，请求体为 `{"model", "input", "encoding_format": "float"}`，
并返回 `{"data": [{"embedding": [...]}]}`——兼容 OpenAI 的 `/embeddings` 接口都是这样。
设置 `EMBEDDING_API_URL`、`EMBEDDING_MODEL_NAME`、`ARK_API_KEY`（名字是历史原因）和 `MILVUS_DIMENSION` 即可。

**能用 Ollama 以外的大模型吗？**
不改代码不行：`services.get_llm_response()` 用的是 Ollama 客户端。Ollama 能提供的任何模型都可以通过
`LLM_MODEL_NAME` 选择。

**上传报 `Failed to clear the vector store before upload.`，日志里有 `Open local milvus failed`。**
你用 `python -m backend.main` 启动并使用了 Milvus Lite 文件。停掉它，改用
`uvicorn backend.main:app --host 127.0.0.1 --port 8000` 启动（见[已知限制](#已知限制)）。

**应该设置 `MILVUS_URI` 吗？**
不要，请用 `MILVUS_CONNECTION_URI`。pymilvus 会自己读取 `MILVUS_URI`，而且只接受 http URI，
把 Milvus Lite 文件路径写在那里会导致连接失败。

**为什么答案以 `[fake LLM …]` 开头？**
`OLLAMA_BASE_URL` 指向了 `scripts/fake_services.py`。改成真实的 Ollama 服务才能得到真实答案。

**相似度图表是空的。**
图表需要从 jsDelivr 加载 Chart.js（需要联网），并且至少要有一个来源块。没有检索到来源时，页面会显示提示，不画图。

**为什么图表上的数比 API 返回的小？**
图表画的是余弦相似度的平方，API 返回的是原始值。

**可以直接打开 `frontend/index.html` 吗？**
不可以。请通过后端的 `/` 打开界面；在 `file://` 下浏览器会拦截 API 请求。

## 参与贡献

欢迎报告问题、提交修复和小而专注的功能，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
`examples/sample.md` 和 `examples/sample.zh-CN.md` 是为本项目编写的示例，和其余代码一样采用 MIT 许可证。

## 许可证

[MIT](LICENSE) © 2025 PakhomLeo
