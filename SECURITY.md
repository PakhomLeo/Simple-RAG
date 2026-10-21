# Security policy

Simple RAG is a local, single-user tool. It is not hardened for exposure on the internet.

## Safe defaults

- The server listens on `127.0.0.1` by default and the API key is off by default.
- At startup (in the app lifespan) the server refuses to listen on a non-loopback address unless
  `SIMPLE_RAG_API_KEY` is set. This covers `python -m backend.main` and `uvicorn backend.main:app --host …`.
  When the listen address cannot be detected (gunicorn, a custom `uvicorn.run(...)`), it logs a
  prominent `SECURITY WARNING` instead; set `SIMPLE_RAG_BIND_HOST` to enable the check there.
- CORS only allows the origins in `CORS_ALLOW_ORIGINS`. The bundled UI at `/` is same-origin.
- Uploaded files are never written to disk; only their text and base file name are stored in SQLite.
- Every upload **replaces the whole index** (Milvus vectors and SQLite chunks). Anyone who can reach the API
  can do this, so protect the API with `SIMPLE_RAG_API_KEY` whenever other people can reach it.

## Known issue: prompt injection

Uploaded documents are inserted into the LLM prompt as-is, so text in a document can instruct the model
and change the answer. The model has no tools, no file or network access and no write access to the stores,
so the impact is limited to the content of the answer. Do not trust answers built from untrusted documents.
This is a known limitation and is not treated as a vulnerability.

## Reporting a vulnerability

Please open a private security advisory on GitHub
(**Security → Report a vulnerability** on the repository page) instead of a public issue.
Include the version or commit, steps to reproduce, and the impact you expect.
