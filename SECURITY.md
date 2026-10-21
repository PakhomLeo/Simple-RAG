# Security policy

Simple RAG is a local, single-user tool. It is not hardened for exposure on the internet.

## Safe defaults

- The server listens on `127.0.0.1` by default. `python -m backend.main` refuses to listen on a
  non-loopback address unless `SIMPLE_RAG_API_KEY` is set. When you start it with `uvicorn`
  yourself, that check does not run, so set the key before you pass `--host 0.0.0.0`.
- CORS only allows the origins in `CORS_ALLOW_ORIGINS`. The bundled UI at `/` is same-origin.
- Uploaded files are never written to disk; only their text and base file name are stored in SQLite.
- Every upload **deletes all vectors** in the Milvus collection. Anyone who can reach the API can do this,
  so protect the API with `SIMPLE_RAG_API_KEY` whenever other people can reach it.

## Reporting a vulnerability

Please open a private security advisory on GitHub
(**Security → Report a vulnerability** on the repository page) instead of a public issue.
Include the version or commit, steps to reproduce, and the impact you expect.
