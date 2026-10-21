import pytest
import requests


def test_split_text_into_chunks_overlap():
    from backend import services

    chunks = services.split_text_into_chunks("abcdefghij", chunk_size=4, chunk_overlap=1)
    assert chunks == ["abcd", "defg", "ghij"]
    assert services.split_text_into_chunks("", 4, 1) == []


def test_parse_markdown_extracts_text_and_code():
    from backend import services

    text = services.parse_markdown_file("# Title\n\nHello **world**\n\n```\ncode here\n```\n")
    assert "Title" in text
    assert "Hello **world**" in text
    assert "code here" in text


def test_cosine_similarity_edge_cases():
    from backend import services

    assert services.cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert services.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert services.cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0
    assert services.cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_get_embeddings_uses_fake_api():
    from backend import services

    vectors = services.get_embeddings(["hello", "world"])
    assert len(vectors) == 2
    assert all(len(v) == services.MILVUS_DIMENSION for v in vectors)


def test_get_embeddings_connection_error_is_value_error(monkeypatch):
    """Regression: a network failure used to raise UnboundLocalError (`response` was never assigned)."""
    from backend import services

    def boom(*args, **kwargs):
        raise requests.exceptions.ConnectionError("unreachable")

    monkeypatch.setattr(services.requests, "post", boom)
    # EmbeddingError is a ValueError; its message names no host or port.
    with pytest.raises(ValueError, match="Could not reach the embedding API") as info:
        services.get_embeddings(["x"])
    assert isinstance(info.value, services.EmbeddingError)


def test_get_embeddings_sets_timeout(monkeypatch):
    """Regression: requests.post had no timeout and could hang a worker forever."""
    from backend import services

    seen = {}

    class Resp:
        text = ""

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"embedding": [0.0] * services.MILVUS_DIMENSION}]}

    def fake_post(*args, **kwargs):
        seen.update(kwargs)
        return Resp()

    monkeypatch.setattr(services.requests, "post", fake_post)
    services.get_embeddings(["x"])
    assert seen.get("timeout") == services.EMBEDDING_TIMEOUT_SECONDS


def test_default_hosts_are_local(monkeypatch):
    """Regression: MILVUS_HOST defaulted to a public IP and FASTAPI_HOST to 0.0.0.0."""
    import importlib

    from backend import config

    for name in ("MILVUS_HOST", "FASTAPI_HOST"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    try:
        fresh = importlib.reload(config)
        assert fresh.MILVUS_HOST == "localhost"
        assert fresh.FASTAPI_HOST == "127.0.0.1"
    finally:
        monkeypatch.undo()
        importlib.reload(config)
