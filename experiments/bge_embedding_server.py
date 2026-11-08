"""Minimal OpenAI-compatible embedding server backed by a local BGE model."""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from transformers import AutoModel, AutoTokenizer


class BGEEmbedder:
    def __init__(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModel.from_pretrained(path)
        self.model.eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def embed(self, text: str):
        encoded = self.tokenizer(
            [str(text)], padding=True, truncation=True, max_length=512, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            hidden = self.model(**encoded).last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = torch.nn.functional.normalize(pooled, dim=-1)
        return pooled[0].cpu().tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8768)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "AGORA_EMBED_MODEL", "/root/autodl-tmp/models/bge-small-en-v1.5"
        ),
    )
    args = parser.parse_args()
    embedder = BGEEmbedder(args.model)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            inputs = data.get("input", [])
            if isinstance(inputs, str):
                inputs = [inputs]
            payload = {
                "object": "list",
                "data": [
                    {"object": "embedding", "index": i, "embedding": embedder.embed(text)}
                    for i, text in enumerate(inputs)
                ],
                "model": data.get("model", "bge-embedding"),
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
            }
            raw = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args):  # noqa: ANN002
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"BGE embedding server on 127.0.0.1:{args.port} ({args.model})", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
