"""Minimal OpenAI-compatible client for DeepSeek flash.

The client is deliberately small: it supports chat completions, JSON output,
retries, token accounting, and an on-disk cache.  It never prints the API key.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from .config import LLMConfig

LOGGER = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the LLM endpoint cannot produce a usable answer."""


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_calls: int = 0
    errors: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def to_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_calls": self.cached_calls,
            "errors": self.errors,
            "total_tokens": self.total_tokens,
        }


def _load_dotenv(path: str | Path = ".env") -> None:
    """Load simple KEY=VALUE lines without requiring python-dotenv."""

    path = Path(path)
    if not path.exists():
        return
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except OSError:
        LOGGER.warning("Could not read .env file at %s", path)


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response."""

    text = text.strip()
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # Fenced code block.
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if match:
        try:
            value = json.loads(match.group(1))
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    # First balanced {...}.
    start = text.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            char = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidate = text[start : idx + 1]
                        try:
                            value = json.loads(candidate)
                            if isinstance(value, dict):
                                return value
                        except json.JSONDecodeError:
                            break
    raise LLMError(f"Could not parse JSON from response: {text[:300]}")


class DeepSeekClient:
    """A thin, cache-aware client for ``https://api.deepseek.com``."""

    def __init__(self, config: LLMConfig | None = None):
        _load_dotenv()
        self.config = config or LLMConfig()
        self.api_key = (
            self.config.api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.base_url = (
            self.config.base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.model = self.config.model or os.environ.get(
            "DEEPSEEK_MODEL", "deepseek-flash"
        )
        self.usage = LLMUsage()
        self._cache: dict[str, dict[str, Any]] = {}
        self._cache_dirty = False
        self._load_cache()
        # On some Windows setups ``requests`` inherits a system proxy that
        # cannot reach api.deepseek.com while direct curl works.  Bypass the
        # system proxy by default; users who need one can set
        # ``DEEPSEEK_USE_SYSTEM_PROXY=1``.
        self.session = requests.Session()
        self.session.trust_env = os.environ.get("DEEPSEEK_USE_SYSTEM_PROXY") == "1"

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    def _load_cache(self) -> None:
        path = self.config.cache_path
        if not path:
            return
        cache_path = Path(path)
        if not cache_path.exists():
            return
        try:
            with cache_path.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                self._cache = raw
        except (OSError, json.JSONDecodeError):
            LOGGER.warning("Could not load LLM cache at %s", cache_path)

    def _save_cache(self) -> None:
        if not self.config.cache_path or not self._cache_dirty:
            return
        path = Path(self.config.cache_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as fh:
                json.dump(self._cache, fh, ensure_ascii=False, indent=2)
            self._cache_dirty = False
        except OSError:
            LOGGER.warning("Could not save LLM cache at %s", path)

    @staticmethod
    def _cache_key(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def available(self) -> bool:
        return bool(self.api_key) and self.config.enabled

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> str:
        if not self.available:
            raise LLMError(
                "DeepSeek API key is missing. Set DEEPSEEK_API_KEY or use synthetic mode."
            )
        if self.usage.calls >= self.config.max_calls:
            raise LLMError(
                f"LLM call budget exhausted ({self.config.max_calls} calls)."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": (
                self.config.temperature if temperature is None else temperature
            ),
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        key = self._cache_key(payload)
        if use_cache and not refresh_cache and key in self._cache:
            self.usage.cached_calls += 1
            return self._cache[key]["content"]

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.config.timeout,
                    proxies={"http": None, "https": None}
                    if not self.session.trust_env
                    else None,
                )
                if response.status_code in (429, 500, 502, 503, 504):
                    raise LLMError(
                        f"Transient HTTP {response.status_code}: {response.text[:200]}"
                    )
                if response.status_code >= 400:
                    raise LLMError(
                        f"HTTP {response.status_code}: {response.text[:300]}"
                    )
                data = response.json()
                message = data["choices"][0]["message"]
                content = message.get("content") or message.get(
                    "reasoning_content", ""
                )
                if not content:
                    raise LLMError("Empty response content")
                usage = data.get("usage", {})
                self.usage.calls += 1
                self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0))
                self.usage.completion_tokens += int(
                    usage.get("completion_tokens", 0)
                )
                if use_cache or refresh_cache:
                    self._cache[key] = {
                        "content": content,
                        "model": data.get("model", self.model),
                        "usage": usage,
                    }
                    self._cache_dirty = True
                    self._save_cache()
                return content
            except (requests.RequestException, KeyError, ValueError, LLMError) as exc:
                last_error = exc
                self.usage.errors += 1
                if attempt + 1 >= self.config.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        raise LLMError(f"DeepSeek request failed: {last_error}")

    def json(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> dict[str, Any]:
        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
        )
        return _extract_json(text)

    def test(self) -> dict[str, Any]:
        """A minimal health check used by ``agora-planner test-llm``."""

        started = time.time()
        text = self.chat(
            [
                {
                    "role": "system",
                    "content": "You are a terse API health-check assistant.",
                },
                {"role": "user", "content": "Reply with exactly: AGORA_OK"},
            ],
            temperature=0.0,
            max_tokens=64,
            use_cache=False,
        )
        return {
            "ok": "AGORA_OK" in text,
            "reply": text.strip(),
            "model": self.model,
            "base_url": self.base_url,
            "latency_sec": round(time.time() - started, 3),
            "usage": self.usage.to_dict(),
        }
