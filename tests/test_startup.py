"""Startup behaviour: `python -m backend.main` and what happens at import time."""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _backend_env(tmp_path: Path, **extra: str) -> dict:
    """Environment for a backend subprocess: fake services from conftest, Milvus Lite and SQLite in tmp_path."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("MILVUS_", "SQLITE_", "SIMPLE_RAG_", "FASTAPI_"))}
    env.update(
        {
            "MILVUS_CONNECTION_URI": str(tmp_path / "milvus_lite.db"),
            "MILVUS_DIMENSION": os.environ["MILVUS_DIMENSION"],
            "SQLITE_DATABASE_URL": f"sqlite:///{tmp_path / 'rag_sqlite.db'}",
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(extra)
    return env


def test_import_does_not_open_milvus_or_sqlite(tmp_path):
    """Importing backend.main must not touch the stores: the reloader parent process imports it too."""
    env = _backend_env(tmp_path)
    subprocess.run(
        [sys.executable, "-c", "import backend.main"], cwd=ROOT, env=env, check=True, capture_output=True, timeout=60
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "argv,environ,expected",
    [
        ([], {}, False),
        (["--reload"], {}, True),
        ([], {"SIMPLE_RAG_RELOAD": "1"}, True),
        ([], {"SIMPLE_RAG_RELOAD": "false"}, False),
    ],
)
def test_reload_is_off_by_default(app_module, argv, environ, expected):
    assert app_module.reload_enabled(argv, environ) is expected


def _wait_healthy(base: str, proc: subprocess.Popen, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError(f"server exited with {proc.returncode}")
        try:
            if requests.get(f"{base}/health", timeout=1).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.3)
    raise AssertionError("server did not become healthy")


@pytest.mark.parametrize("reload_env", ["", "1"], ids=["default", "SIMPLE_RAG_RELOAD=1"])
def test_python_m_backend_main_works_with_milvus_lite(tmp_path, reload_env):
    """Regression: `python -m backend.main` always reloaded, and the reloader parent opened the Milvus Lite
    file first, so the server process failed with `Open local milvus failed` and uploads returned 500."""
    port = _free_port()
    env = _backend_env(tmp_path, FASTAPI_HOST="127.0.0.1", FASTAPI_PORT=str(port), SIMPLE_RAG_RELOAD=reload_env)
    log = tmp_path / "server.log"
    with open(log, "wb") as out:
        proc = subprocess.Popen(
            [sys.executable, "-m", "backend.main"],
            cwd=ROOT,
            env=env,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_healthy(base, proc)
        r = requests.post(
            f"{base}/upload/",
            files=[("files", ("a.md", ("Milvus stores the vectors. " * 20).encode(), "text/markdown"))],
            timeout=60,
        )
        assert r.status_code == 200, r.text
        assert r.json()[0]["total_chunks"] > 0
        r = requests.post(f"{base}/query/", json={"question": "Where are the vectors?", "top_k": 1}, timeout=60)
        assert r.status_code == 200, r.text
        assert r.json()["rag_sources"][0]["source_document_name"] == "a.md"
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=10)
    assert b"Open local milvus failed" not in log.read_bytes()
