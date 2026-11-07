"""Tiny OpenAI-compatible embedding endpoint backed by HashEmbedder.

Used by the official Mem0/GraphRAG/Letta adapters in this offline environment.
Run: ``PYTHONPATH=src python experiments/hash_embedding_server.py --port 8766``.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agora_planner.embeddings import HashEmbedder  # noqa: E402


def make_handler(embedder: HashEmbedder):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            inputs = data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            vectors = [embedder.embed(str(text)).tolist() for text in inputs]
            payload = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": idx, "embedding": vec}
                    for idx, vec in enumerate(vectors)
                ],
                "model": data.get("model", "hash-embedding"),
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path in {"/health", "/v1/health"}:
                raw = b'{"status":"ok"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, *args):
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--dim", type=int, default=384)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(HashEmbedder(dim=args.dim)))
    print(f"hash embedding server on 127.0.0.1:{args.port} dim={args.dim}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
