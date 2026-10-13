"""Stand-ins for the LiteLLM objects the suite built before the move to the Anthropic SDK.

``ModelResponse`` builds a response in the OpenAI shape that ``autorubric.llm._acompletion``
returns (the suite patches ``_acompletion``), from the same arguments LiteLLM's
``ModelResponse`` took. ``Timeout``, ``RateLimitError`` and ``APIConnectionError`` return the
Anthropic SDK's exceptions for the failures LiteLLM raised; they accept LiteLLM's
``model``/``llm_provider`` keywords and ignore them.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx

_REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def ModelResponse(  # noqa: N802 -- named as the LiteLLM class it replaces
    model: str = "claude-haiku-4-5",
    choices: list[dict[str, Any]] | None = None,
    usage: dict[str, int] | None = None,
    **_: Any,
) -> SimpleNamespace:
    out = []
    for i, choice in enumerate(choices or []):
        message = choice.get("message") or {}
        out.append(
            SimpleNamespace(
                message=SimpleNamespace(
                    content=message.get("content"),
                    reasoning_content=message.get("reasoning_content"),
                    thinking_blocks=None,
                    thinking=None,
                ),
                finish_reason=choice.get("finish_reason"),
                index=choice.get("index", i),
            )
        )
    u = usage or {}
    return SimpleNamespace(
        model=model,
        choices=out,
        usage=SimpleNamespace(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            total_tokens=u.get("total_tokens", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
        cost=None,
    )


def Timeout(message: str = "timed out", **_: Any) -> anthropic.APITimeoutError:  # noqa: N802
    exc = anthropic.APITimeoutError(request=_REQUEST)
    exc.message, exc.args = message, (message,)  # the SDK's own text otherwise
    return exc


def RateLimitError(message: str = "rate limited", **_: Any) -> anthropic.RateLimitError:  # noqa: N802
    response = httpx.Response(429, request=_REQUEST)
    return anthropic.RateLimitError(message, response=response, body=None)


def APIConnectionError(  # noqa: N802
    message: str = "Connection error.", **_: Any
) -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message=message, request=_REQUEST)
