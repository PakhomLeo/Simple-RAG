from dotenv import load_dotenv
import os

load_dotenv()

ARK_API_KEY = os.getenv("ARK_API_KEY")
MILVUS_HOST = os.getenv("MILVUS_HOST", "8.140.239.23")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_COLLECTION_NAME = os.getenv("MILVUS_COLLECTION_NAME", "rag_collection")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "doubao-embedding-text-240715")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "https://ark.cn-beijing.volces.com/api/v3/embeddings")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "qwen3:4b")

FASTAPI_HOST = os.getenv("FASTAPI_HOST", "0.0.0.0")
FASTAPI_PORT = int(os.getenv("FASTAPI_PORT", 8000))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 200))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

SQLITE_DATABASE_URL = os.getenv("SQLITE_DATABASE_URL", "sqlite:///./rag_sqlite.db")

# Milvus
MILVUS_DIMENSION = int(os.getenv("MILVUS_DIMENSION", 2560)) # 根据日志更新默认值为2560
MILVUS_INDEX_FIELD_NAME = os.getenv("MILVUS_INDEX_FIELD_NAME", "embedding")
MILVUS_VECTOR_FIELD_NAME = os.getenv("MILVUS_VECTOR_FIELD_NAME", "embedding")
MILVUS_ID_FIELD_NAME = os.getenv("MILVUS_ID_FIELD_NAME", "chunk_id")
MILVUS_TEXT_FIELD_NAME = os.getenv("MILVUS_TEXT_FIELD_NAME", "text") # 虽然主要靠ID关联，但有时也会存少量文本信息

# Milvus Index and Search Parameters
MILVUS_METRIC_TYPE = os.getenv("MILVUS_METRIC_TYPE", "L2")
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")
MILVUS_INDEX_PARAM_NLIST = int(os.getenv("MILVUS_INDEX_PARAM_NLIST", 128))
MILVUS_SEARCH_PARAM_NPROBE = int(os.getenv("MILVUS_SEARCH_PARAM_NPROBE", 10))
MILVUS_CONSISTENCY_LEVEL = os.getenv("MILVUS_CONSISTENCY_LEVEL", "Strong")
MILVUS_SEARCH_MAX_DISTANCE = float(os.getenv("MILVUS_SEARCH_MAX_DISTANCE", 10000.0)) # 新增：用于过滤结果的最大L2距离

# 确保关键配置存在
if not ARK_API_KEY:
    raise ValueError("ARK_API_KEY is not set in the environment variables.") 