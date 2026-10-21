import pytest


def _md(name: str, text: str):
    return ("files", (name, text.encode("utf-8"), "text/markdown"))


def test_health(client):
    assert client.get("/health").json() == {"status": "healthy"}


def test_frontend_is_served_same_origin(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "Simple RAG" in r.text
    assert client.get("/script.js").status_code == 200


def test_upload_and_query_end_to_end(client):
    r = client.post(
        "/upload/",
        files=[_md("notes.md", "# Milvus\n\nMilvus 是一个向量数据库。" * 5), _md("skip.txt", "not markdown")],
    )
    assert r.status_code == 200, r.text
    by_name = {item["filename"]: item for item in r.json()}
    assert by_name["notes.md"]["total_chunks"] > 0
    assert by_name["skip.txt"]["message"].startswith("Skipped")

    r = client.post("/query/", json={"question": "Milvus 是什么？", "top_k": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["rag_answer"].startswith("[fake LLM, with RAG context]")
    assert body["llm_only_answer"].startswith("[fake LLM, no context]")
    assert 1 <= len(body["rag_sources"]) <= 2
    assert body["rag_sources"][0]["source_document_name"] == "notes.md"
    assert body["similarity_scores"]


def test_upload_strips_directory_from_filename(client):
    """Filenames are metadata only; directory parts (path traversal attempts) are dropped."""
    r = client.post("/upload/", files=[_md("../../etc/evil.md", "hello world " * 30)])
    assert r.status_code == 200
    assert r.json()[0]["filename"] == "evil.md"
    r = client.post("/upload/", files=[_md("..\\..\\win\\evil2.md", "hello world " * 30)])
    assert r.json()[0]["filename"] == "evil2.md"


def test_upload_rejects_oversized_file(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_FILE_BYTES", 100)
    r = client.post("/upload/", files=[_md("big.md", "x" * 101), _md("ok.md", "small enough")])
    assert r.status_code == 200
    assert r.json()[0]["message"].startswith("Skipped: file is larger")
    assert r.json()[0]["total_chunks"] == 0
    assert r.json()[1]["total_chunks"] == 1
    # Alone, the oversized file leaves no valid file: 400 and the index is kept.
    r = client.post("/upload/", files=[_md("big.md", "x" * 101)])
    assert r.status_code == 400
    assert "big.md: Skipped: file is larger" in r.json()["detail"]


def test_upload_rejects_too_many_files(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_FILES", 2)
    r = client.post("/upload/", files=[_md(f"{i}.md", "hello") for i in range(3)])
    assert r.status_code == 413


def test_upload_non_utf8_is_reported_not_crashing(client):
    r = client.post(
        "/upload/", files=[("files", ("bad.md", b"\xff\xfe\xfa", "text/markdown")), _md("ok.md", "hello world")]
    )
    assert r.status_code == 200
    assert r.json()[0]["total_chunks"] == 0
    assert r.json()[0]["message"].startswith("Error processing document")
    r = client.post("/upload/", files=[("files", ("bad.md", b"\xff\xfe\xfa", "text/markdown"))])
    assert r.status_code == 400
    assert "bad.md: Error processing document" in r.json()["detail"]


@pytest.mark.parametrize("top_k", [0, -1, 51, 10_000])
def test_query_rejects_out_of_range_top_k(client, top_k):
    r = client.post("/query/", json={"question": "hi", "top_k": top_k})
    assert r.status_code == 422


def test_query_rejects_empty_and_oversized_question(client, app_module):
    assert client.post("/query/", json={"question": ""}).status_code == 422
    assert client.post("/query/", json={"question": "x" * 4001}).status_code == 422


def test_cors_blocks_unknown_origin(client):
    r = client.options(
        "/query/",
        headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") is None
    r = client.post("/query/", json={"question": "hi"}, headers={"Origin": "https://evil.example"})
    assert r.headers.get("access-control-allow-origin") is None


def test_cors_allows_configured_origin_without_credentials(client):
    r = client.options(
        "/query/",
        headers={"Origin": "http://localhost:8000", "Access-Control-Request-Method": "POST"},
    )
    assert r.headers.get("access-control-allow-origin") == "http://localhost:8000"
    assert r.headers.get("access-control-allow-credentials") is None


def test_api_key_required_when_configured(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "SIMPLE_RAG_API_KEY", "test-key")
    assert client.post("/query/", json={"question": "hi"}).status_code == 401
    assert client.post("/query/", json={"question": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/upload/", files=[_md("a.md", "hello")]).status_code == 401
    ok = client.post("/query/", json={"question": "hi"}, headers={"X-API-Key": "test-key"})
    assert ok.status_code == 200
    assert client.get("/health").status_code == 200


def test_unexpected_error_does_not_leak_details(client, app_module, monkeypatch):
    def boom(*a, **k):
        raise KeyError("internal-secret-path /srv/x")

    monkeypatch.setattr(app_module.services, "get_embeddings", boom)
    r = client.post("/query/", json={"question": "hi"})
    assert r.status_code == 500
    assert "internal-secret-path" not in r.text


@pytest.mark.parametrize(
    "host,expected",
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", True),
        ("0.0.0.0", False),
        ("192.168.1.2", False),
    ],
)
def test_is_loopback_host(app_module, host, expected):
    assert app_module.is_loopback_host(host) is expected


# --- Upload replaces both stores -------------------------------------------------------------


def _chunk_count(app_module):
    from backend import database

    with app_module.SessionLocal() as db:
        return db.query(database.ChunkDB).count(), {
            row.source_document_name for row in db.query(database.ChunkDB).all()
        }


def test_upload_clears_sqlite_and_milvus(client, app_module):
    r = client.post("/upload/", files=[_md("first.md", "first document about apples. " * 20)])
    assert r.status_code == 200, r.text
    first_chunks = r.json()[0]["total_chunks"]
    assert first_chunks > 0

    r = client.post("/upload/", files=[_md("second.md", "second document about pears. " * 10)])
    assert r.status_code == 200, r.text
    second_chunks = r.json()[0]["total_chunks"]

    count, names = _chunk_count(app_module)
    assert names == {"second.md"}
    assert count == second_chunks
    assert app_module.services.get_milvus_collection().num_entities == second_chunks

    body = client.post("/query/", json={"question": "apples?", "top_k": 50}).json()
    assert {s["source_document_name"] for s in body["rag_sources"]} == {"second.md"}


def test_upload_keeps_sqlite_when_milvus_clear_fails(client, app_module, monkeypatch):
    r = client.post("/upload/", files=[_md("keep.md", "keep me around please. " * 20)])
    assert r.status_code == 200
    before = _chunk_count(app_module)

    def fail():
        raise RuntimeError("milvus down")

    monkeypatch.setattr(app_module.services, "clear_milvus_data", fail)
    r = client.post("/upload/", files=[_md("new.md", "new content " * 20)])
    assert r.status_code == 500
    assert "milvus down" not in r.text
    assert _chunk_count(app_module) == before


def test_clear_milvus_data_raises_on_failure(app_module, monkeypatch):
    services = app_module.services

    def boom(*a, **k):
        raise Exception("drop failed")

    monkeypatch.setattr(services.utility, "has_collection", boom)
    with pytest.raises(RuntimeError):
        services.clear_milvus_data()


# --- Startup bind check ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "argv,environ,expected",
    [
        (["uvicorn", "backend.main:app"], {}, ("127.0.0.1", "uvicorn default")),
        (["/venv/bin/uvicorn", "backend.main:app", "--host", "0.0.0.0"], {}, ("0.0.0.0", "uvicorn --host")),
        (["uvicorn", "backend.main:app", "--host=::"], {}, ("::", "uvicorn --host")),
        (["/x/uvicorn/__main__.py", "backend.main:app", "--host", "10.0.0.5"], {}, ("10.0.0.5", "uvicorn --host")),
        (["uvicorn", "backend.main:app"], {"UVICORN_HOST": "0.0.0.0"}, ("0.0.0.0", "UVICORN_HOST")),
        (["uvicorn", "backend.main:app", "--uds", "/tmp/s.sock"], {}, ("unix-socket", "uvicorn --uds")),
        (["uvicorn", "backend.main:app", "--fd", "3"], {}, (None, "uvicorn --fd")),
        (["gunicorn", "backend.main:app"], {}, (None, "unknown launcher")),
        (["anything"], {"SIMPLE_RAG_BIND_HOST": "0.0.0.0"}, ("0.0.0.0", "SIMPLE_RAG_BIND_HOST")),
    ],
)
def test_detect_bind_host(app_module, argv, environ, expected):
    assert app_module.detect_bind_host(argv, environ) == expected


def test_bind_policy_refuses_public_address_without_key(app_module):
    with pytest.raises(RuntimeError, match="SIMPLE_RAG_API_KEY"):
        app_module.enforce_bind_policy("", ["uvicorn", "backend.main:app", "--host", "0.0.0.0"], {})


def test_bind_policy_allows_public_address_with_key(app_module):
    app_module.enforce_bind_policy("k", ["uvicorn", "backend.main:app", "--host", "0.0.0.0"], {})


def test_bind_policy_allows_loopback_without_key(app_module):
    app_module.enforce_bind_policy("", ["uvicorn", "backend.main:app"], {})
    app_module.enforce_bind_policy("", ["uvicorn", "backend.main:app", "--uds", "/tmp/s.sock"], {})


def test_bind_policy_warns_when_address_unknown(app_module, caplog):
    with caplog.at_level("WARNING"):
        app_module.enforce_bind_policy("", ["gunicorn", "backend.main:app"], {})
    assert "SECURITY WARNING" in caplog.text


def test_lifespan_refuses_to_start_on_public_address(app_module, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("SIMPLE_RAG_BIND_HOST", "0.0.0.0")
    monkeypatch.setattr(app_module, "SIMPLE_RAG_API_KEY", "")
    with pytest.raises(RuntimeError, match="Refusing to start"):
        with TestClient(app_module.app):
            pass

    monkeypatch.setattr(app_module, "SIMPLE_RAG_API_KEY", "k")
    with TestClient(app_module.app) as c:
        assert c.get("/health").status_code == 200


# --- Files are checked before the index is cleared -------------------------------------------


def _index_state(app_module):
    count, names = _chunk_count(app_module)
    return count, names, app_module.services.get_milvus_collection().num_entities


def test_upload_without_valid_files_keeps_index(client, app_module, monkeypatch):
    """Regression: the index was cleared before the files were checked, so an upload of only invalid
    files left an empty index."""
    r = client.post("/upload/", files=[_md("keep.md", "keep this index please. " * 20)])
    assert r.status_code == 200, r.text
    before = _index_state(app_module)
    assert before[0] > 0 and before[2] == before[0]

    monkeypatch.setattr(app_module, "MAX_UPLOAD_FILE_BYTES", 1000)
    r = client.post(
        "/upload/",
        files=[
            _md("notes.txt", "not markdown"),
            _md("empty.md", ""),
            ("files", ("latin1.md", b"\xff\xfe\xfa", "text/markdown")),
            _md("big.md", "x" * 1001),
        ],
    )
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert detail.startswith("No valid .md files in the upload; the index was not changed.")
    for name in ("notes.txt", "empty.md", "latin1.md", "big.md"):
        assert name in detail
    assert _index_state(app_module) == before


def test_upload_with_one_valid_file_replaces_index_and_reports_the_rest(client, app_module):
    client.post("/upload/", files=[_md("old.md", "old content here. " * 20)])
    r = client.post("/upload/", files=[_md("skip.txt", "nope"), _md("new.md", "new content here. " * 20)])
    assert r.status_code == 200, r.text
    assert [item["filename"] for item in r.json()] == ["skip.txt", "new.md"]
    count, names, vectors = _index_state(app_module)
    assert names == {"new.md"} and vectors == count
