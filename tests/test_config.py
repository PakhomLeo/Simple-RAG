"""Configuration validation at load time."""

import importlib

import pytest


def _reload_config(monkeypatch, **env):
    from backend import config

    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    return importlib.reload(config)


@pytest.fixture(autouse=True)
def _restore_config(monkeypatch):
    yield
    monkeypatch.undo()
    from backend import config

    importlib.reload(config)


@pytest.mark.parametrize(
    "env,message",
    [
        ({"CHUNK_SIZE": "200", "CHUNK_OVERLAP": "200"}, "CHUNK_OVERLAP (200) must be smaller than CHUNK_SIZE (200)"),
        ({"CHUNK_SIZE": "100", "CHUNK_OVERLAP": "150"}, "CHUNK_OVERLAP (150) must be smaller than CHUNK_SIZE (100)"),
        ({"CHUNK_SIZE": "0"}, "CHUNK_SIZE must be at least 1"),
        ({"CHUNK_SIZE": "-5"}, "CHUNK_SIZE must be at least 1"),
        ({"CHUNK_OVERLAP": "-1"}, "CHUNK_OVERLAP must be at least 0"),
        ({"CHUNK_SIZE": "abc"}, "CHUNK_SIZE must be an integer"),
        ({"MILVUS_DIMENSION": "0"}, "MILVUS_DIMENSION must be at least 1"),
        ({"FASTAPI_PORT": "70000"}, "FASTAPI_PORT must be at most 65535"),
        ({"EMBEDDING_TIMEOUT_SECONDS": "0"}, "EMBEDDING_TIMEOUT_SECONDS must be greater than 0"),
        ({"MAX_UPLOAD_FILES": "0"}, "MAX_UPLOAD_FILES must be at least 1"),
        ({"MAX_TOP_K": "0"}, "MAX_TOP_K must be at least 1"),
    ],
)
def test_invalid_config_fails_at_load_time(monkeypatch, env, message):
    """Regression: CHUNK_OVERLAP >= CHUNK_SIZE was accepted and made every upload loop forever."""
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        _reload_config(monkeypatch, **env)


def test_valid_config_loads(monkeypatch):
    fresh = _reload_config(monkeypatch, CHUNK_SIZE="10", CHUNK_OVERLAP="9")
    assert (fresh.CHUNK_SIZE, fresh.CHUNK_OVERLAP) == (10, 9)
    fresh = _reload_config(monkeypatch, CHUNK_SIZE="10", CHUNK_OVERLAP="0")
    assert fresh.CHUNK_OVERLAP == 0


@pytest.mark.parametrize("size,overlap", [(4, 4), (4, 5), (0, 0), (4, -1)])
def test_split_text_rejects_invalid_sizes(size, overlap):
    """The chunker guards itself too, so a direct call cannot loop forever."""
    from backend import services

    with pytest.raises(ValueError):
        services.split_text_into_chunks("abcdefghij", chunk_size=size, chunk_overlap=overlap)
