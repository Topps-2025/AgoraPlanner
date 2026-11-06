"""Official AutoGen orchestration adapter (pilot).

AutoGen is a multi-agent *orchestration* framework, not a memory system.  This
adapter uses the official ``autogen-agentchat`` package to run a small
memory-grounded QA pilot:

* each agent receives a role and a retrieved memory context;
* agents answer independently;
* a judge agent aggregates their answers.

The point is to check whether a multi-agent orchestration layer changes the
answer when the underlying memory context is private, shared, or allocated.
It is **not** a MultiAgentBench run: MultiAgentBench/MARBLE is a separate
benchmark and is handled by ``experiments/run_marble_pilot.py``.
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .embeddings import CosineRetriever
from .memory import MemoryAsset
from .text_benchmark import TextMemoryBenchmark, TextQuery, extract_code

_AUTOGEN_VENDOR = Path(__file__).resolve().parents[2] / ".vendor_autogen"
if _AUTOGEN_VENDOR.exists() and str(_AUTOGEN_VENDOR) not in sys.path:
    sys.path.insert(0, str(_AUTOGEN_VENDOR))


@dataclass
class AutoGenSettings:
    model: str = "deepseek-flash"
    api_key: str | None = None
    base_url: str = "https://api.deepseek.com"
    temperature: float = 0.0
    # DeepSeek flash emits reasoning tokens; a small max_tokens can leave the
    # final content empty.  Keep enough room for the final one-token answer.
    max_tokens: int = 512
    n_agents: int = 3
    agent_names: tuple[str, ...] = ("researcher", "engineer", "auditor")
    timeout_seconds: float = 60.0
    extra: dict[str, Any] = field(default_factory=dict)


def _load_dotenv() -> None:
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


@dataclass
class AutoGenAnswer:
    query_id: str
    agent_id: str
    gold: str
    agent_answers: dict[str, str]
    judge_answer: str
    parsed_answer: str | None
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    context_texts: dict[str, list[str]]


class AutoGenMemoryTeam:
    """Official AutoGen agents + judge over a fixed memory context."""

    def __init__(self, settings: AutoGenSettings | None = None):
        _load_dotenv()
        self.settings = settings or AutoGenSettings()
        self.settings.api_key = (
            self.settings.api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.settings.base_url = (
            self.settings.base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        )
        self.settings.model = (
            self.settings.model
            or os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
        )

    def _client(self):
        from autogen_ext.models.openai import OpenAIChatCompletionClient
        import httpx2

        return OpenAIChatCompletionClient(
            model=self.settings.model,
            api_key=self.settings.api_key or "dummy",
            base_url=self.settings.base_url,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": "unknown",
                "structured_output": True,
            },
            http_client=httpx2.AsyncClient(trust_env=False),
        )

    @staticmethod
    def _memory_context(bench: TextMemoryBenchmark, asset_ids: Iterable[str]) -> list[str]:
        texts: list[str] = []
        for mid in asset_ids:
            if mid in bench.memories:
                asset = bench.memories.get(mid)
                texts.append(f"[{asset.source_class}|q={asset.quality:.2f}] {asset.content}")
        return texts

    @staticmethod
    def _parse_answer(text: str) -> str | None:
        code = extract_code(text or "")
        if code:
            return code
        upper = (text or "").upper()
        if "UNKNOWN" in upper:
            return "UNKNOWN"
        return None

    @staticmethod
    def _result_text_and_usage(result) -> tuple[str, int, int]:
        """Return (last non-empty text, prompt tokens, completion tokens)."""

        texts: list[str] = []
        prompt_tokens = 0
        completion_tokens = 0
        for message in getattr(result, "messages", []):
            content = str(getattr(message, "content", "") or "")
            if content.strip():
                texts.append(content)
            usage = getattr(message, "models_usage", None)
            if usage is not None:
                prompt_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                completion_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return (texts[-1] if texts else "", prompt_tokens, completion_tokens)

    async def answer_query(
        self,
        bench: TextMemoryBenchmark,
        query: TextQuery,
        *,
        memory_ids_by_agent: dict[str, frozenset[str]],
        top_k: int = 3,
    ) -> AutoGenAnswer:
        from autogen_agentchat.agents import AssistantAgent

        client = self._client()
        start = time.perf_counter()
        try:
            tasks = []
            agent_names = []
            contexts: dict[str, list[str]] = {}
            for idx, agent in enumerate(bench.agents[: self.settings.n_agents]):
                name = self.settings.agent_names[idx % len(self.settings.agent_names)]
                agent_names.append(name)
                retriever = CosineRetriever()
                for mid in memory_ids_by_agent.get(agent.id, frozenset()):
                    if mid in bench.memories:
                        retriever.add(mid, bench.memories.get(mid).content)
                retrieved = [item.text for item in retriever.search(query.question, top_k=top_k)]
                # Use the original asset metadata for the prompt.
                selected_ids = []
                for item in retriever.search(query.question, top_k=top_k):
                    if item.memory_id:
                        selected_ids.append(item.memory_id)
                context = self._memory_context(bench, selected_ids)
                contexts[agent.id] = context
                context_block = "\n".join(f"- {line}" for line in context) or "- (no memory)"
                system = (
                    f"You are {name}, a team member in a memory-grounded QA task. "
                    "You may only use the memory notes below. Do not invent a code. "
                    "If the note contains the requested access code, answer with that "
                    "4-character code. If not, answer UNKNOWN.\n\n"
                    f"Your memory notes:\n{context_block}\n\n"
                    "Reply with exactly one token: the code or UNKNOWN."
                )
                agent_obj = AssistantAgent(
                    name=name,
                    model_client=client,
                    system_message=system,
                    reflect_on_tool_use=False,
                )
                tasks.append(agent_obj.run(task=query.question))
            results = await asyncio.gather(*tasks)
            agent_answers: dict[str, str] = {}
            prompt_tokens = 0
            completion_tokens = 0
            for agent, result in zip(bench.agents[: self.settings.n_agents], results):
                content, pt, ct = self._result_text_and_usage(result)
                agent_answers[agent.id] = content
                prompt_tokens += pt
                completion_tokens += ct
            answers_block = "\n".join(
                f"- {agent_id}: {answer}" for agent_id, answer in agent_answers.items()
            )
            judge_system = (
                "You are the judge of a memory-grounded QA task. You receive a "
                "question and independent answers from team members. Each answer may "
                "come from a different memory note; a wrong code is possible. Choose "
                "the single most reliable 4-character access code. If no answer is "
                "credible, answer UNKNOWN. Reply with exactly the code or UNKNOWN."
            )
            judge = AssistantAgent(
                "judge",
                model_client=client,
                system_message=judge_system,
                reflect_on_tool_use=False,
            )
            judge_result = await judge.run(
                task=f"Question: {query.question}\n\nTeam answers:\n{answers_block}"
            )
            judge_text, pt, ct = self._result_text_and_usage(judge_result)
            prompt_tokens += pt
            completion_tokens += ct
            parsed = self._parse_answer(judge_text)
            latency_ms = (time.perf_counter() - start) * 1000.0
            return AutoGenAnswer(
                query_id=query.id,
                agent_id=query.agent_id,
                gold=query.gold_answer,
                agent_answers=agent_answers,
                judge_answer=judge_text,
                parsed_answer=parsed,
                correct=parsed == query.gold_answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                context_texts=contexts,
            )
        finally:
            await client.close()


__all__ = ["AutoGenSettings", "AutoGenMemoryTeam", "AutoGenAnswer"]
