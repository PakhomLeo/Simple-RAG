"""Runtime configuration, loaded from environment variables and an optional `.env` file.

See `.env.example` for every supported variable.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def parse_bool(value, default: bool = False) -> bool:
    """`1`, `true`, `yes` and `on` (any case) are true; any other value is false; None gives the default."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_bool(name: str, default: bool = False) -> bool:
    return parse_bool(os.getenv(name), default)


def _get_list(name: str, default: str) -> list[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


ARK_API_KEY = os.getenv("ARK_API_KEY")

# Milvus. MILVUS_CONNECTION_URI takes precedence over MILVUS_HOST/MILVUS_PORT when set.
# It accepts a server URI (http://localhost:19530) or a local Milvus Lite file (./milvus_lite.db).
MILVUS_CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI", "")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_collection")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "doubao-embedding-text-240715")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://ark.cn-beijing.volces.com/api/v3/embeddings")
EMBEDDING_TIMEOUT_SECONDS = float(os.getenv("EMBEDDING_TIMEOUT_SECONDS", "60"))
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:4b")

# Bind to loopback by default. Set FASTAPI_HOST=0.0.0.0 only together with SIMPLE_RAG_API_KEY.
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", "8000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite:///./rag_sqlite.db")

# Milvus schema
MILVUS_DIMENSION = int(os.getenv("MILVUS_DIMENSION", "2560"))  # doubao-embedding-text-240715 returns 2560 dims
MILVUS_INDEX_FIELD_NAME = os.getenv("MILVUS_INDEX_FIELD_NAME", "embedding")
MILVUS_VECTOR_FIELD_NAME = os.getenv("MILVUS_VECTOR_FIELD_NAME", "embedding")
MILVUS_ID_FIELD_NAME = os.getenv("MILVUS_ID_FIELD_NAME", "chunk_id")
MILVUS_TEXT_FIELD_NAME = os.getenv("MILVUS_TEXT_FIELD_NAME", "text")

# Milvus index and search parameters
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "L2")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
MILVUS_INDEX_PARAM_NLIST = int(os.getenv("MILVUS_INDEX_PARAM_NLIST", "128"))
MILVUS_SEARCH_PARAM_NPROBE = int(os.getenv("MILVUS_SEARCH_PARAM_NPROBE", "10"))
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Strong")
# Results with an L2 distance above this value are dropped before they reach the prompt.
MILVUS_SEARCH_MAX_DISTANCE = float(os.getenv("MILVUS_SEARCH_MAX_DISTANCE", "10000.0"))

# --- Security ---
# Optional shared secret. When set, /upload/ and /query/ require the header `X-API-Key: <value>`.
SIMPLE_RAG_API_KEY = os.getenv("SIMPLE_RAG_API_KEY", "")
# Comma-separated list of origins allowed by CORS. The bundled UI is served same-origin and needs none.
CORS_ALLOW_ORIGINS = _get_list(
    "CORS_ALLOW_ORIGINS",
    "http://localhost:8000,http://127.0.0.1:8000",
)
# Serve the bundled frontend from the backend at "/".
SERVE_FRONTEND = _get_bool("SERVE_FRONTEND", True)

# Request limits
MAX_UPLOAD_FILES = int(os.getenv("MAX_UPLOAD_FILES", "20"))
MAX_UPLOAD_FILE_BYTES = int(os.getenv("MAX_UPLOAD_FILE_BYTES", str(5 * 1024 * 1024)))
MAX_QUESTION_CHARS = int(os.getenv("MAX_QUESTION_CHARS", "4000"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "50"))

# Fail fast when the embedding API key is missing.
if not ARK_API_KEY:
    raise ValueError("ARK_API_KEY is not set in the environment variables.")
