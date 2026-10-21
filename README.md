# Simple RAG

**English** | [简体中文](README.zh-CN.md)

A small Retrieval-Augmented Generation (RAG) app that runs on your own machine.
Upload Markdown files, ask a question, and see two answers side by side: one from a local LLM
that gets the retrieved document chunks as context, and one from the same LLM without any context.
A bar chart shows how close each answer is to each source chunk.

- **Backend:** Python, FastAPI, SQLite (SQLAlchemy), Milvus (pymilvus), Ollama, any `/embeddings` API
- **Frontend:** plain HTML, CSS and JavaScript, Chart.js
- **License:** [MIT](LICENSE)

![Architecture](docs/assets/architecture.svg)

## Features

- Upload one or more `.md` files from the browser. Text is extracted with `markdown-it-py`
  and split into fixed-size chunks with overlap (`CHUNK_SIZE`, `CHUNK_OVERLAP`).
- Each chunk gets a UUID. Its text and file name go to SQLite; its vector goes to Milvus.
- Embeddings come from an HTTP embedding API. The default is Volcengine Ark
  (`doubao-embedding-text-240715`, 2560 dimensions); any endpoint with the same
  request and response shape (`{"model", "input"}` → `{"data": [{"embedding": [...]}]}`) works.
- A question is embedded, the Top-K closest chunks are fetched from Milvus (L2 distance),
  chunks farther than `MILVUS_SEARCH_MAX_DISTANCE` are dropped, and the rest become the prompt context.
- The LLM runs in Ollama (default `qwen3:4b`). `<think>…</think>` blocks are removed from answers.
- The response contains the RAG answer, the LLM-only answer, the source chunks, and the cosine
  similarity between each answer and each source chunk. The UI plots the squared similarity.
- Every upload **replaces** the previous vectors: the Milvus collection is dropped before the new batch.
  Rows in SQLite are kept (see [Known limitations](#known-limitations)).

![Upload and query flow](docs/assets/flow.svg)

## Quick start

### Requirements

- Python 3.9 or newer (CI runs 3.10 and 3.12)
- For real answers: a running [Milvus](https://milvus.io/docs/install_standalone-docker.md) (or Milvus Lite),
  [Ollama](https://ollama.com/download) with a pulled model, and an embedding API key

### 1. Install

```bash
git clone https://github.com/PakhomLeo/Simple-RAG.git
cd Simple-RAG
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
```

Edit `.env`. At minimum set `ARK_API_KEY`, and check `MILVUS_HOST`/`MILVUS_PORT`, `OLLAMA_BASE_URL`,
`LLM_MODEL_NAME` and `MILVUS_DIMENSION` (it must match the embedding size). To use Milvus Lite instead of a
server, `pip install milvus-lite` and set `MILVUS_CONNECTION_URI=./milvus_lite.db`.

```bash
ollama pull qwen3:4b
```

### 3. Run

```bash
python -m backend.main
# or: uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000/>. The backend serves the UI on the same origin.
The interactive API docs are at <http://127.0.0.1:8000/docs>.

### Try it without any key or model

`scripts/fake_services.py` is a fake embedding API plus a fake Ollama. It returns deterministic
vectors and canned answers, so you can check the whole pipeline offline. It is for demos and tests only.

```bash
pip install -r requirements-dev.txt        # adds milvus-lite, pytest, ruff
python scripts/fake_services.py --port 18080 --dim 64 &

export ARK_API_KEY=placeholder
export EMBEDDING_API_URL=http://127.0.0.1:18080/api/v3/embeddings
export OLLAMA_BASE_URL=http://127.0.0.1:18080
export MILVUS_CONNECTION_URI=./milvus_lite.db
export MILVUS_DIMENSION=64
python -m backend.main
```

Then upload `examples/sample.md` in the UI and ask a question.

## Configuration

All settings are environment variables; `.env` in the working directory is loaded automatically.
See [`.env.example`](.env.example) for the full list.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ARK_API_KEY` | *(required)* | Bearer token sent to the embedding API |
| `EMBEDDING_API_URL` | Ark `…/api/v3/embeddings` | Embedding endpoint |
| `EMBEDDING_MODEL_NAME` | `doubao-embedding-text-240715` | Embedding model |
| `EMBEDDING_TIMEOUT_SECONDS` | `60` | Timeout for each embedding request |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server |
| `LLM_MODEL_NAME` | `qwen3:4b` | Ollama model |
| `MILVUS_HOST` / `MILVUS_PORT` | `localhost` / `19530` | Milvus server |
| `MILVUS_CONNECTION_URI` | *(empty)* | Overrides host/port; a `.db` path uses Milvus Lite |
| `MILVUS_COLLECTION_NAME` | `rag_collection` | Collection name |
| `MILVUS_DIMENSION` | `2560` | Vector size; must match the embedding model |
| `MILVUS_METRIC_TYPE` / `MILVUS_INDEX_TYPE` | `L2` / `IVF_FLAT` | Index settings |
| `MILVUS_SEARCH_MAX_DISTANCE` | `10000.0` | Drop results farther than this |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `200` / `50` | Chunking, in characters |
| `SQLITE_DATABASE_URL` | `sqlite:///./rag_sqlite.db` | Chunk text store |
| `FASTAPI_HOST` / `FASTAPI_PORT` | `127.0.0.1` / `8000` | Used by `python -m backend.main` |
| `SERVE_FRONTEND` | `true` | Serve `frontend/` at `/` |
| `SIMPLE_RAG_API_KEY` | *(empty)* | If set, `/upload/` and `/query/` need header `X-API-Key` |
| `CORS_ALLOW_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Browser origins allowed by CORS |
| `MAX_UPLOAD_FILES` | `20` | Files per upload |
| `MAX_UPLOAD_FILE_BYTES` | `5242880` | Size limit per file (5 MiB) |
| `MAX_QUESTION_CHARS` | `4000` | Question length limit |
| `MAX_TOP_K` | `50` | Upper bound for `top_k` |

## API

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `POST` | `/upload/` | `multipart/form-data`, one or more `files` | One result per file: `filename`, `message`, `total_chunks` |
| `POST` | `/query/` | JSON `{"question": "...", "top_k": 3}` | `rag_answer`, `rag_sources[]`, `llm_only_answer`, `similarity_scores[]` |
| `GET` | `/health` | – | `{"status": "healthy"}` |
| `GET` | `/` | – | The web UI |

```bash
curl -F files=@examples/sample.md http://127.0.0.1:8000/upload/
curl -H 'Content-Type: application/json' \
     -d '{"question": "Where are the vectors stored?", "top_k": 3}' \
     http://127.0.0.1:8000/query/
```

When `SIMPLE_RAG_API_KEY` is set, add `-H "X-API-Key: <key>"`. The UI has an optional API key field.

## Security notes

- The server binds to `127.0.0.1` by default. `python -m backend.main` refuses a non-loopback
  `FASTAPI_HOST` unless `SIMPLE_RAG_API_KEY` is set. If you run `uvicorn --host 0.0.0.0` yourself,
  set the key first.
- Any client that can reach `/upload/` can wipe the vector collection. Use the API key whenever
  other people can reach the server.
- Uploaded documents are inserted into the LLM prompt as-is. A document can contain instructions that
  steer the answer (prompt injection). The LLM has no tools and cannot read or change anything else,
  but do not treat answers as trusted when documents come from untrusted sources.
- The embedding API key is sent only to `EMBEDDING_API_URL` and is never logged.
- See [SECURITY.md](SECURITY.md) to report a vulnerability.

## Project layout

```
backend/
  config.py      settings from the environment / .env
  database.py    SQLite engine and the chunks table
  crud.py        chunk insert and lookup
  models.py      request and response models (pydantic)
  services.py    chunking, embeddings, Milvus, Ollama, cosine similarity
  main.py        FastAPI app, routes, CORS, API key, static UI
frontend/        index.html, script.js, style.css
scripts/
  fake_services.py   fake embedding API + fake Ollama for demos and tests
tests/           pytest suite (uses the fake services and Milvus Lite)
examples/        sample.md
Markdown/        sample documents from the original project (see note below)
docs/assets/     architecture and flow diagrams
```

## Development

```bash
pip install -r requirements-dev.txt
ruff check backend tests scripts
ruff format --check backend tests scripts
pytest
```

The tests start the fake services on a free port and use a temporary Milvus Lite file,
so they need no network, no key and no model. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Known limitations

- An upload drops the Milvus collection but does not clear SQLite, so old chunk text stays in
  `rag_sqlite.db`. Delete the file to remove it.
- The backend uses the ORM-style pymilvus API, which pymilvus 3.1 removes, so `requirements.txt`
  pins `pymilvus<3.1`.
- Chunking is by character count, not by sentence or heading.
- Chart.js is loaded from jsDelivr (pinned version with Subresource Integrity), so the chart needs internet access.

## Sample documents

`examples/sample.md` is written for this project and is MIT licensed like the rest of the code.
The files in `Markdown/` were collected from third-party public sources for testing. They are not
covered by this project's license and remain the property of their authors.

## License

[MIT](LICENSE) © 2025 PakhomLeo
