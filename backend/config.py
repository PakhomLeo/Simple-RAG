"""Runtime configuration, loaded from environment variables and an optional `.env` file.

See `.env.example` for every supported variable.
"""

import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


def parse_bool(value, default: bool = False) -> bool:
    """`1`, `true`, `yes` and `on` (any case) are true; any other value is false; None gives the default."""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _get_bool(name: str, default: bool = False) -> bool:
    return parse_bool(os.getenv(name), default)


def _get_int(name: str, default: int, minimum: Optional[int] = None, maximum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw.strip())
        except ValueError:
            raise ValueError(f"{name} must be an integer, got {raw!r}.") from None
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}.")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be at most {maximum}, got {value}.")
    return value


def _get_float(name: str, default: float, greater_than: Optional[float] = None) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw.strip())
        except ValueError:
            raise ValueError(f"{name} must be a number, got {raw!r}.") from None
    if greater_than is not None and not value > greater_than:
        raise ValueError(f"{name} must be greater than {greater_than:g}, got {value:g}.")
    return value


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
EMBEDDING_TIMEOUT_SECONDS = _get_float("EMBEDDING_TIMEOUT_SECONDS", 60.0, greater_than=0)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:4b")

# Bind to loopback by default. Set FASTAPI_HOST=0.0.0.0 only together with SIMPLE_RAG_API_KEY.
FASTAPI_HOST = os.getenv("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = _get_int("FASTAPI_PORT", 8000, minimum=1, maximum=65535)

CHUNK_SIZE = _get_int("CHUNK_SIZE", 200, minimum=1)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 50, minimum=0)
# Each chunk starts CHUNK_SIZE - CHUNK_OVERLAP characters after the previous one; that step must be positive.
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ValueError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE ({CHUNK_SIZE}).")

SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite:///./rag_sqlite.db")

# Milvus schema
MILVUS_DIMENSION = _get_int("MILVUS_DIMENSION", 2560, minimum=1)  # doubao-embedding-text-240715 returns 2560 dims
MILVUS_INDEX_FIELD_NAME = os.getenv("MILVUS_INDEX_FIELD_NAME", "embedding")
MILVUS_VECTOR_FIELD_NAME = os.getenv("MILVUS_VECTOR_FIELD_NAME", "embedding")
MILVUS_ID_FIELD_NAME = os.getenv("MILVUS_ID_FIELD_NAME", "chunk_id")
MILVUS_TEXT_FIELD_NAME = os.getenv("MILVUS_TEXT_FIELD_NAME", "text")

# Milvus index and search parameters
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "L2")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
MILVUS_INDEX_PARAM_NLIST = _get_int("MILVUS_INDEX_PARAM_NLIST", 128, minimum=1)
MILVUS_SEARCH_PARAM_NPROBE = _get_int("MILVUS_SEARCH_PARAM_NPROBE", 10, minimum=1)
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Strong")
# Results with an L2 distance above this value are dropped before they reach the prompt.
MILVUS_SEARCH_MAX_DISTANCE = _get_float("MILVUS_SEARCH_MAX_DISTANCE", 10000.0)

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
MAX_UPLOAD_FILES = _get_int("MAX_UPLOAD_FILES", 20, minimum=1)
MAX_UPLOAD_FILE_BYTES = _get_int("MAX_UPLOAD_FILE_BYTES", 5 * 1024 * 1024, minimum=1)
MAX_QUESTION_CHARS = _get_int("MAX_QUESTION_CHARS", 4000, minimum=1)
MAX_TOP_K = _get_int("MAX_TOP_K", 50, minimum=1)

# Fail fast when the embedding API key is missing.
if not ARK_API_KEY:
    raise ValueError("ARK_API_KEY is not set in the environment variables.")
