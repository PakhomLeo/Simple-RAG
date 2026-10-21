"""Fake embedding API and fake Ollama server for local demos and tests.

It needs no API key and no model. Embeddings are deterministic bag-of-characters vectors,
so texts that share characters are close to each other. The fake LLM echoes a short
summary of the prompt. Do not use it for real answers.

    python scripts/fake_services.py --port 18080 --dim 64

Then set, for example:
    EMBEDDING_API_URL=http://127.0.0.1:18080/api/v3/embeddings
    OLLAMA_BASE_URL=http://127.0.0.1:18080
    MILVUS_DIMENSION=64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def embed(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    for ch in text:
        if ch.isspace():
            continue
        h = int.from_bytes(hashlib.sha256(ch.encode("utf-8")).digest()[:4], "big")
        vec[h % dim] += 1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def fake_answer(prompt: str) -> str:
    if "以下是相关背景信息" in prompt:
        context = prompt.split("以下是相关背景信息：", 1)[-1].split("---\n基于以上信息", 1)[0].strip()
        snippet = context.replace("\n", " ")[:160]
        return f"[fake LLM, with RAG context] {snippet}"
    return f"[fake LLM, no context] I received a {len(prompt)}-character question."


def make_handler(dim: int):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"{}")

        def do_GET(self):  # noqa: N802
            if self.path.startswith("/api/tags"):
                self._json(200, {"models": [{"name": "fake-llm", "model": "fake-llm"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            if self.path.endswith("/embeddings"):
                data = self._body()
                inputs = data.get("input") or []
                if isinstance(inputs, str):
                    inputs = [inputs]
                self._json(
                    200,
                    {
                        "object": "list",
                        "model": data.get("model", "fake-embedding"),
                        "data": [
                            {"object": "embedding", "index": i, "embedding": embed(t, dim)}
                            for i, t in enumerate(inputs)
                        ],
                    },
                )
            elif self.path.startswith("/api/chat"):
                data = self._body()
                messages = data.get("messages") or [{}]
                prompt = messages[-1].get("content", "")
                self._json(
                    200,
                    {
                        "model": data.get("model", "fake-llm"),
                        "created_at": "1970-01-01T00:00:00Z",
                        "message": {"role": "assistant", "content": fake_answer(prompt)},
                        "done": True,
                    },
                )
            else:
                self._json(404, {"error": "not found"})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--dim", type=int, default=64)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.dim))
    print(f"fake embedding + Ollama server on http://{args.host}:{args.port} (dim={args.dim})", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
