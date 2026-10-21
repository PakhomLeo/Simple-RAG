"""Test setup.

The backend reads its configuration at import time, so the environment is prepared here,
before `backend` is imported. External services are replaced by:
  * scripts/fake_services.py for the embedding API and Ollama (no real key, no model);
  * Milvus Lite (a local file) for the vector store.
"""

import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fake_services  # noqa: E402

DIM = 32
_tmp = Path(tempfile.mkdtemp(prefix="simple-rag-test-"))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


_port = _free_port()
_server = fake_services.ThreadingHTTPServer(("127.0.0.1", _port), fake_services.make_handler(DIM))
threading.Thread(target=_server.serve_forever, daemon=True).start()

os.environ.update(
    {
        "ARK_API_KEY": "test-placeholder-key",
        "EMBEDDING_API_URL": f"http://127.0.0.1:{_port}/api/v3/embeddings",
        "OLLAMA_BASE_URL": f"http://127.0.0.1:{_port}",
        "LLM_MODEL_NAME": "fake-llm",
        "MILVUS_CONNECTION_URI": str(_tmp / "milvus_lite.db"),
        "MILVUS_DIMENSION": str(DIM),
        "SQLITE_DATABASE_URL": f"sqlite:///{_tmp / 'rag_sqlite.db'}",
        "SIMPLE_RAG_API_KEY": "",
        "CORS_ALLOW_ORIGINS": "http://localhost:8000,http://127.0.0.1:8000",
    }
)


@pytest.fixture(scope="session")
def app_module():
    import backend.main as main

    return main


@pytest.fixture()
def client(app_module):
    from fastapi.testclient import TestClient

    with TestClient(app_module.app) as c:
        yield c
