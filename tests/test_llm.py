from __future__ import annotations

import os

import pytest

from agora_planner.config import LLMConfig
from agora_planner.llm import DeepSeekClient, LLMError


def _has_key() -> bool:
    if os.environ.get("DEEPSEEK_API_KEY"):
        return True
    # The client loads .env lazily; check it without exposing the value.
    try:
        with open(".env", encoding="utf-8") as fh:
            return "DEEPSEEK_API_KEY=" in fh.read()
    except OSError:
        return False


@pytest.mark.skipif(not _has_key(), reason="DeepSeek API key not configured")
def test_deepseek_health_check():
    client = DeepSeekClient(LLMConfig(max_calls=1, cache_path=None))
    try:
        result = client.test()
    except LLMError as exc:
        pytest.skip(f"LLM endpoint unavailable: {exc}")
    assert result["ok"] is True
    assert result["usage"]["calls"] >= 1
