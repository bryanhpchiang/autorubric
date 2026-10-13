"""Tests for the Anthropic transport that replaced LiteLLM (``autorubric.llm._acompletion``).

The SDK client is faked: each test checks the Messages API request built from the
OpenAI-shaped request ``LLMClient.generate`` makes, the OpenAI-shaped response returned, and
which credential each auth mode uses.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from autorubric.llm import (
    CLAUDE_CODE_IDENTITY,
    LLMClient,
    LLMConfig,
    _acompletion,
    _anthropic_client,
    anthropic_model,
    resolve_auth,
)
from autorubric.types import CriterionJudgment


class _FakeStream:
    def __init__(self, message: Any) -> None:
        self._message = message

    async def __aenter__(self) -> _FakeStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    async def get_final_message(self) -> Any:
        return self._message


class _FakeClient:
    """Records the request and answers with ``message``."""

    def __init__(self, message: Any) -> None:
        self.sent: dict[str, Any] = {}
        self.messages = SimpleNamespace(stream=self._stream)
        self._message = message

    def _stream(self, **kwargs: Any) -> _FakeStream:
        self.sent = kwargs
        return _FakeStream(self._message)

    async def __aenter__(self) -> _FakeClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


def _message(text: str, thinking: str | None = None) -> Any:
    content = [SimpleNamespace(type="text", text=text)]
    if thinking:
        content.insert(0, SimpleNamespace(type="thinking", thinking=thinking))
    usage = SimpleNamespace(
        input_tokens=10, output_tokens=5, cache_creation_input_tokens=100, cache_read_input_tokens=0
    )
    return SimpleNamespace(
        content=content, usage=usage, model="claude-opus-5-5", stop_reason="end_turn"
    )


async def _send(params: dict[str, Any], message: Any) -> tuple[dict[str, Any], Any, list[Any]]:
    fake = _FakeClient(message)
    calls: list[Any] = []

    def make(*args: Any) -> _FakeClient:
        calls.append(args)
        return fake

    with patch("autorubric.llm._anthropic_client", side_effect=make):
        response = await _acompletion(**params)
    return fake.sent, response, calls


def _params(**extra: Any) -> dict[str, Any]:
    return {
        "model": "anthropic/claude-opus-5-5",
        "messages": [{"role": "system", "content": "Judge."}, {"role": "user", "content": "Item"}],
        "timeout": 30.0,
        **extra,
    }


class TestModelIds:
    def test_drops_the_litellm_prefix(self):
        assert anthropic_model("anthropic/claude-opus-5-5") == "claude-opus-5-5"
        assert anthropic_model("claude-haiku-4-5") == "claude-haiku-4-5"

    @pytest.mark.parametrize("model", ["openai/gpt-5.2", "gemini/gemini-3-pro", "gpt-4o"])
    def test_rejects_other_providers(self, model):
        with pytest.raises(ValueError, match="Only Anthropic models"):
            anthropic_model(model)


class TestAuth:
    def test_default_is_the_subscription(self, monkeypatch):
        monkeypatch.delenv("AUTORUBRIC_AUTH", raising=False)
        assert resolve_auth(None) == "oauth"

    def test_environment_and_config(self, monkeypatch):
        monkeypatch.setenv("AUTORUBRIC_AUTH", "api_key")
        assert resolve_auth(None) == "api_key"
        assert resolve_auth("oauth") == "oauth"
        with pytest.raises(ValueError):
            resolve_auth("bearer")

    def test_oauth_never_sends_the_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-should-not-be-sent")
        monkeypatch.delenv("CLAUDE_ACCOUNT", raising=False)
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-subscription")
        client = _anthropic_client("oauth", None, None, 30.0)
        assert client.api_key is None
        assert client.auth_token == "sk-ant-oat-subscription"
        assert "oauth-2025-04-20" in client.default_headers["anthropic-beta"]

    def test_named_account_wins(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_ACCOUNT", "work")
        monkeypatch.setenv("CLAUDE_TOKEN_WORK", "sk-ant-oat-work")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-other")
        assert _anthropic_client("oauth", None, None, 30.0).auth_token == "sk-ant-oat-work"

    def test_api_key_mode(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api-key")
        client = _anthropic_client("api_key", None, None, 30.0)
        assert client.api_key == "sk-ant-api-key"
        assert client.auth_token is None or client.auth_token == ""


class TestRequest:
    @pytest.mark.asyncio
    async def test_oauth_opens_the_system_prompt_with_the_identity_line(self, monkeypatch):
        monkeypatch.delenv("AUTORUBRIC_AUTH", raising=False)
        sent, _, calls = await _send(_params(), _message("ok"))
        assert sent["system"][0]["text"] == CLAUDE_CODE_IDENTITY
        assert sent["system"][1]["text"] == "Judge."
        assert sent["messages"] == [{"role": "user", "content": "Item"}]
        assert sent["model"] == "claude-opus-5-5"
        assert calls[0][0] == "oauth"

    @pytest.mark.asyncio
    async def test_api_key_mode_sends_no_identity_line(self):
        sent, _, calls = await _send(_params(auth="api_key"), _message("ok"))
        assert sent["system"] == [{"type": "text", "text": "Judge."}]
        assert calls[0][0] == "api_key"

    @pytest.mark.asyncio
    async def test_adaptive_thinking_on_current_models(self):
        sent, _, _ = await _send(_params(reasoning_effort="high"), _message("ok"))
        assert sent["thinking"] == {"type": "adaptive"}
        assert sent["output_config"]["effort"] == "high"

    @pytest.mark.asyncio
    async def test_budget_on_older_models(self):
        params = _params(
            model="claude-sonnet-4-5", thinking={"type": "enabled", "budget_tokens": 9000}
        )
        sent, _, _ = await _send(params, _message("ok"))
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 9000}

    @pytest.mark.asyncio
    async def test_structured_output_goes_in_output_config(self):
        sent, _, _ = await _send(_params(response_format=CriterionJudgment), _message("{}"))
        schema = sent["output_config"]["format"]["schema"]
        assert sent["output_config"]["format"]["type"] == "json_schema"
        assert schema["additionalProperties"] is False
        assert "criterion_status" in schema["properties"]

    @pytest.mark.asyncio
    async def test_sampling_parameters_ride_in_the_body(self):
        sent, _, _ = await _send(_params(temperature=0.0, seed=7), _message("ok"))
        assert sent["extra_body"] == {"temperature": 0.0}  # the seed is dropped


class TestResponse:
    @pytest.mark.asyncio
    async def test_openai_shape_with_thinking_and_usage(self):
        _, response, _ = await _send(_params(), _message("verdict", thinking="weighing it"))
        message = response.choices[0].message
        assert message.content == "verdict"
        assert message.reasoning_content == "weighing it"
        assert response.usage.prompt_tokens == 110  # cache writes count as input
        assert response.usage.completion_tokens == 5
        assert response.cost is None  # a subscription call is not billed per token

    @pytest.mark.asyncio
    async def test_generate_end_to_end(self):
        fake = _FakeClient(_message('{"criterion_status": "MET", "explanation": "ok"}'))
        with patch("autorubric.llm._anthropic_client", return_value=fake):
            client = LLMClient(LLMConfig(model="claude-opus-5-5"))
            result = await client.generate("Judge.", "Item", response_format=CriterionJudgment)
        assert result.criterion_status.value == "MET"
