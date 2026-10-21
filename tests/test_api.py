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
    r = client.post("/upload/", files=[_md("big.md", "x" * 101)])
    assert r.status_code == 200
    assert r.json()[0]["message"].startswith("Skipped: file is larger")
    assert r.json()[0]["total_chunks"] == 0


def test_upload_rejects_too_many_files(client, app_module, monkeypatch):
    monkeypatch.setattr(app_module, "MAX_UPLOAD_FILES", 2)
    r = client.post("/upload/", files=[_md(f"{i}.md", "hello") for i in range(3)])
    assert r.status_code == 413


def test_upload_non_utf8_is_reported_not_crashing(client):
    r = client.post("/upload/", files=[("files", ("bad.md", b"\xff\xfe\xfa", "text/markdown"))])
    assert r.status_code == 200
    assert r.json()[0]["total_chunks"] == 0


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
    monkeypatch.setattr(app_module, "SIMPLE_RAG_API_KEY", "s3cret-placeholder")
    assert client.post("/query/", json={"question": "hi"}).status_code == 401
    assert client.post("/query/", json={"question": "hi"}, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/upload/", files=[_md("a.md", "hello")]).status_code == 401
    ok = client.post("/query/", json={"question": "hi"}, headers={"X-API-Key": "s3cret-placeholder"})
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
