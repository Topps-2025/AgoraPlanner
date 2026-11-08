"""DeepSeek/OpenAI-compatible adapter for MARBLE's model_prompting.

MARBLE's upstream implementation routes through LiteLLM.  On this box LiteLLM
does not preserve DeepSeek tool calls reliably, so this adapter calls the
OpenAI-compatible endpoint directly and returns the same ``litellm`` Message
objects that MARBLE agents expect.  The MARBLE engine, agents, environments,
and evaluator are unchanged.
"""

import os

from beartype import beartype
from beartype.typing import Any, Dict, List, Optional
from litellm.types.utils import Message
from openai import OpenAI

from marble.llms.error_handler import api_calling_error_exponential_backoff


@beartype
@api_calling_error_exponential_backoff(retries=5, base_wait_time=1)
def model_prompting(
    llm_model: str,
    messages: List[Dict[str, str]],
    return_num: Optional[int] = 1,
    max_token_num: Optional[int] = 1024,
    temperature: Optional[float] = 0.0,
    top_p: Optional[float] = None,
    stream: Optional[bool] = None,
    mode: Optional[str] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: Optional[str] = None,
) -> List[Message]:
    model = llm_model.split("/")[-1] if "/" in llm_model else llm_model
    client = OpenAI(
        base_url=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_API_BASE"),
        api_key=os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY"),
        timeout=120,
    )
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_token_num,
        "n": return_num or 1,
        "temperature": temperature,
    }
    if top_p is not None:
        kwargs["top_p"] = top_p
    if tools:
        def _clean(value):
            if isinstance(value, dict):
                return {k: _clean(v) for k, v in value.items() if k != "optional"}
            if isinstance(value, list):
                return [_clean(item) for item in value]
            return value

        kwargs["tools"] = [_clean(tool) for tool in tools]
    # DeepSeek's thinking mode rejects explicit tool_choice values (including
    # "auto" and "required"); omitting it uses the default automatic tool
    # selection.  The prompt still asks the agent to call a tool.
    if tool_choice is not None and tool_choice not in ("auto", "required"):
        kwargs["tool_choice"] = tool_choice
    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:  # pragma: no cover - diagnostic
        body = getattr(getattr(exc, "response", None), "text", "")
        print(f"MARBLE_LLM_ERROR {type(exc).__name__}: {exc} body={body[:1000]}", flush=True)
        raise
    out: List[Message] = []
    for choice in response.choices:
        message = choice.message
        tool_calls = []
        for call in message.tool_calls or []:
            tool_calls.append(
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
            )
        out.append(
            Message(
                role="assistant",
                content=message.content or "",
                tool_calls=tool_calls or None,
            )
        )
    return out
