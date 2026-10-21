# Contributing to Simple RAG

Thanks for your interest. Bug reports, fixes and small focused features are welcome.
中文贡献者也可以直接用中文提 issue 和 PR。

## Development setup

```bash
git clone https://github.com/PakhomLeo/Simple-RAG.git
cd Simple-RAG
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env      # only needed to run against real services
```

You do not need an API key, Milvus or Ollama to run the tests. `tests/conftest.py` starts
`scripts/fake_services.py` and uses a temporary Milvus Lite file.

## Before you open a pull request

```bash
ruff check backend tests scripts
ruff format --check backend tests scripts
pytest
```

- Keep each pull request focused on one change and describe what it changes and why.
- Add or update tests for behaviour changes and bug fixes.
- Update `README.md` **and** `README.zh-CN.md` when you change user-facing behaviour or configuration,
  and add new settings to `.env.example` with placeholder values only.
- Never commit `.env`, API keys, databases (`*.db`) or uploaded documents.
- Only add sample documents that you wrote yourself or that have a licence allowing redistribution.

## Reporting bugs

Open an issue with the steps to reproduce, what you expected, what happened, your OS and Python version,
and the relevant log lines (remove any keys first). For security problems, follow [SECURITY.md](SECURITY.md)
instead of opening a public issue.

## License

By contributing you agree that your contributions are licensed under the [MIT License](LICENSE).
