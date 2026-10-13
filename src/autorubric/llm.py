"""LLM calling infrastructure for Anthropic's Claude models.

This fork calls the Anthropic API directly (the ``anthropic`` SDK) instead of routing through
LiteLLM, with structured outputs, automatic retries, response caching, extended thinking and
prompt caching. It authenticates one of two ways (``LLMConfig.auth``, or ``AUTORUBRIC_AUTH``):

- ``"oauth"`` (the default): a Claude subscription (Pro/Max), with the OAuth token the
  ``claude`` CLI uses: ``CLAUDE_TOKEN_<NAME>`` when ``CLAUDE_ACCOUNT=<name>`` is set, else
  ``CLAUDE_CODE_OAUTH_TOKEN`` (from ``claude setup-token``), else the token the CLI keeps in the
  macOS Keychain. ``ANTHROPIC_API_KEY`` is never sent in this mode, so a request cannot fall
  back to API billing.
- ``"api_key"``: the paid API, with ``LLMConfig.api_key`` or ``ANTHROPIC_API_KEY``.

Model ids are Anthropic's (``claude-opus-5-5``); the LiteLLM-style ``anthropic/`` prefix is
accepted and dropped. Other providers are not supported.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from pathlib import Path, PurePath
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import diskcache
import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from autorubric.rate_limit import RateLimitPool

if TYPE_CHECKING:
    import anthropic

    from autorubric.types import TokenUsage

# Load the nearest .env at or above the working directory, in scripts and notebooks alike;
# variables already set in the environment take precedence.
load_dotenv(find_dotenv(usecwd=True))

logger = logging.getLogger(__name__)

# Type variable for structured output
T = TypeVar("T", bound=BaseModel)


# ============================================================================
# Grading error classification
# ============================================================================

ErrorCategory = Literal["infrastructure", "parse", "unknown"]
"""Category of a failure encountered while grading a single criterion.

- infrastructure: API/network failure (timeout, connection, rate limit, server error).
  Not the submission's fault; the judge never produced a usable response.
- parse: the judge responded but its output could not be parsed/validated into the
  expected schema. Also not the submission's fault.
- unknown: an unexpected error that does not fit the above categories.
"""


def classify_grading_error(exc: BaseException) -> ErrorCategory:
    """Classify an exception raised while grading a criterion.

    Reuses the Anthropic SDK's exception taxonomy: every API failure (``APITimeoutError``,
    ``APIConnectionError``, ``RateLimitError``, ``OverloadedError``, ``InternalServerError``,
    status errors, etc.) subclasses ``anthropic.APIError``, while parse/validation failures
    (``json.JSONDecodeError`` -> ``ValueError``, ``pydantic.ValidationError``) do not.

    Decision-model failures raised by the optional TypeSafe SDK (``typesafe-sdk``) are
    routed by the SDK's own exception types, ahead of the ``ValueError`` rule:

    - ``TypeSafeAPIResponseValidationError`` (the endpoint's answer did not validate),
      ``TypeSafeBadRequestError`` (400) and ``TypeSafeUnprocessableEntityError`` (422, e.g.
      a request over the model's input limit) are ``"parse"``: the endpoint answered, but
      no usable answer came back for the request that was built.
    - Every other ``TypeSafeError`` is ``"infrastructure"``: authentication, permission,
      not found, rate limit, server errors, and connection failures and timeouts (which
      subclass ``TypeSafeError`` directly, not ``TypeSafeAPIError``).

    Args:
        exc: The exception raised during a judge call.

    Returns:
        ``"infrastructure"`` for API/network errors, ``"parse"`` for JSON/validation
        errors, and ``"unknown"`` for anything else.
    """
    # The SDK is imported only when a judge calls the API (it brings its own HTTP stack), so
    # an Anthropic exception can exist only once it is in sys.modules.
    sdk = sys.modules.get("anthropic")
    if sdk is not None and isinstance(exc, sdk.APIError):
        return "infrastructure"
    typesafe_category = _classify_typesafe_error(exc)
    if typesafe_category is not None:
        return typesafe_category
    if isinstance(exc, (ValidationError, ValueError)):
        return "parse"
    return "unknown"


def _classify_typesafe_error(exc: BaseException) -> ErrorCategory | None:
    """Category of a TypeSafe SDK exception, or None when ``exc`` is not one.

    The SDK is optional and is looked up, never imported: one of its exceptions can exist
    only after ``typesafe_sdk`` was imported, so this costs nothing when the SDK is absent
    or unused.
    """
    sdk = sys.modules.get("typesafe_sdk")
    if sdk is None or not isinstance(exc, sdk.TypeSafeError):
        return None
    if isinstance(
        exc,
        (
            sdk.TypeSafeAPIResponseValidationError,
            sdk.TypeSafeBadRequestError,
            sdk.TypeSafeUnprocessableEntityError,
        ),
    ):
        return "parse"
    return "infrastructure"


# ============================================================================
# Thinking/Reasoning Configuration
# ============================================================================


class ThinkingLevel(str, Enum):
    """Standardized thinking/reasoning effort levels across LLM providers.

    LiteLLM translates these to provider-specific parameters:
    - Anthropic: thinking={type, budget_tokens} with low→1024, medium→2048, high→4096
    - OpenAI: reasoning_effort parameter for o-series and GPT-5 models
    - Gemini: thinking configuration with similar token budgets
    - DeepSeek: Standardized via LiteLLM's reasoning_effort

    Higher levels mean more "thinking" tokens/steps, trading latency for quality.
    """

    NONE = "none"  # Disable thinking (where supported, e.g., Gemini)
    LOW = "low"  # Light reasoning: ~1024 tokens budget
    MEDIUM = "medium"  # Moderate reasoning: ~2048 tokens budget
    HIGH = "high"  # Deep reasoning: ~4096 tokens budget


# String literal type for convenience
ThinkingLevelLiteral = Literal["none", "low", "medium", "high"]


# Token budget mapping for providers that support explicit budgets
THINKING_LEVEL_BUDGETS: dict[str, int] = {
    "none": 0,
    "low": 1024,
    "medium": 2048,
    "high": 4096,
}


@dataclass
class ThinkingConfig:
    """Detailed configuration for LLM thinking/reasoning.

    Provides a uniform interface across providers:
    - Anthropic: Extended thinking (claude-sonnet-4-5, claude-opus-4-5+)
    - OpenAI: Reasoning (o-series, GPT-5 models via openai/responses/ prefix)
    - Gemini: Thinking mode (2.5+, 3.0+ models)
    - DeepSeek: Reasoning content

    Attributes:
        level: High-level thinking effort. Used when budget_tokens is not set.
            Defaults to MEDIUM for a good balance of quality and latency.
        budget_tokens: Explicit token budget for thinking (provider-specific).
            When set, overrides level. Recommended: 10000-50000 for complex tasks.
            Providers that don't support explicit budgets will map this to the
            nearest level.

    Examples:
        # Simple: use a thinking level
        ThinkingConfig(level=ThinkingLevel.HIGH)

        # Fine-grained: specify exact token budget
        ThinkingConfig(budget_tokens=32000)  # For complex reasoning tasks

        # Disable thinking
        ThinkingConfig(level=ThinkingLevel.NONE)
    """

    level: ThinkingLevel | ThinkingLevelLiteral = ThinkingLevel.MEDIUM
    budget_tokens: int | None = None

    def __post_init__(self) -> None:
        """Normalize level to enum."""
        if isinstance(self.level, str):
            self.level = ThinkingLevel(self.level)

    def get_effective_budget(self) -> int:
        """Get the effective token budget based on level or explicit budget."""
        if self.budget_tokens is not None:
            return self.budget_tokens
        return THINKING_LEVEL_BUDGETS.get(self.level.value, 2048)

    def get_reasoning_effort(self) -> str:
        """Get the reasoning_effort string for LiteLLM."""
        return self.level.value


# Union type for flexible thinking parameter in LLMConfig
ThinkingParam = ThinkingConfig | ThinkingLevel | ThinkingLevelLiteral | int | None
"""Type for the thinking parameter in LLMConfig.

Accepts:
- ThinkingConfig: Full configuration object
- ThinkingLevel: Enum value (e.g., ThinkingLevel.HIGH)
- str: Level as string ("low", "medium", "high", "none")
- int: Direct token budget (e.g., 32000)
- None: Disable thinking
"""


def _normalize_thinking_param(thinking: ThinkingParam) -> ThinkingConfig | None:
    """Convert various thinking parameter formats to ThinkingConfig."""
    if thinking is None:
        return None
    if isinstance(thinking, ThinkingConfig):
        return thinking
    if isinstance(thinking, ThinkingLevel):
        return ThinkingConfig(level=thinking)
    if isinstance(thinking, str):
        return ThinkingConfig(level=ThinkingLevel(thinking))
    if isinstance(thinking, int):
        return ThinkingConfig(budget_tokens=thinking)
    raise TypeError(f"Invalid thinking parameter type: {type(thinking)}")


@dataclass
class GenerateResult:
    """Result from LLM generation including content, thinking, and usage statistics.

    Attributes:
        content: The main response content from the LLM (raw string).
        thinking: The thinking/reasoning trace if thinking was enabled.
            None if thinking was not enabled or the provider doesn't support it.
        raw_response: The raw LiteLLM response object for advanced use cases.
        usage: Token usage statistics from this LLM call.
        cost: Completion cost in USD for this LLM call, calculated using
            LiteLLM's completion_cost() function. None if cost calculation fails.
        parsed: The parsed Pydantic model instance when response_format was provided.
            None if no response_format was used or parsing failed.
    """

    content: str
    thinking: str | None = None
    raw_response: Any = None
    usage: TokenUsage | None = None
    cost: float | None = None
    parsed: Any = None


def _extract_thinking_content(message: Any) -> str | None:
    """Extract thinking/reasoning content from LLM response message.

    LiteLLM standardizes reasoning content across providers:
    - `reasoning_content`: Unified field across all providers (preferred)
    - `thinking_blocks`: Anthropic-specific list of thinking blocks (fallback)
    - `thinking`: Legacy Anthropic field (fallback)

    Args:
        message: The message object from response.choices[0].message

    Returns:
        The thinking/reasoning content as a string, or None if not present.
    """
    # Primary: LiteLLM's standardized reasoning_content field
    if hasattr(message, "reasoning_content") and message.reasoning_content:
        return message.reasoning_content

    # Fallback: Anthropic-specific thinking_blocks
    if hasattr(message, "thinking_blocks") and message.thinking_blocks:
        blocks = message.thinking_blocks
        if isinstance(blocks, list):
            thinking_parts = []
            for block in blocks:
                if isinstance(block, dict) and block.get("thinking"):
                    thinking_parts.append(block["thinking"])
                elif hasattr(block, "thinking") and block.thinking:
                    thinking_parts.append(block.thinking)
            if thinking_parts:
                return "\n".join(thinking_parts)

    # Fallback: Legacy thinking field
    if hasattr(message, "thinking") and message.thinking:
        return message.thinking

    return None


def _extract_usage_from_response(response: Any) -> TokenUsage:
    """Extract token usage from LiteLLM response.

    LiteLLM provides an OpenAI-compatible usage object:
    - prompt_tokens: Number of tokens in the prompt
    - completion_tokens: Number of tokens in the completion
    - total_tokens: Sum of prompt + completion tokens

    Additional fields may be present for specific providers:
    - cache_creation_input_tokens: Tokens used to create cache (Anthropic)
    - cache_read_input_tokens: Tokens read from cache (Anthropic)

    Args:
        response: The LiteLLM response object

    Returns:
        TokenUsage object with usage statistics. Returns zeros if usage not available.
    """
    # Import here to avoid circular import
    from autorubric.types import TokenUsage

    if not hasattr(response, "usage") or response.usage is None:
        return TokenUsage()

    usage = response.usage

    # Extract standard OpenAI-compatible fields
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or 0

    # Extract Anthropic prompt caching fields if present
    cache_creation = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cache_creation_input_tokens=cache_creation,
        cache_read_input_tokens=cache_read,
    )


# USD per million tokens (input, output) for API-key billing; cache reads cost a tenth of
# input and cache writes 1.25x. A model not listed has no cost, and nor does any call on a
# subscription (auth "oauth"), which is not billed per token.
API_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-5": (5.0, 25.0),
    "claude-opus-4-1": (15.0, 75.0),
    "claude-sonnet-4-5": (3.0, 15.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def _calculate_completion_cost(response: Any) -> float | None:
    """Cost in USD of one call, as ``_acompletion`` left it on the response (``None`` when
    unknown or not billed per token)."""
    return getattr(response, "cost", None)


_debug_console = Console()


def _print_debug_prompt(system_prompt: str, user_prompt: str, model: str) -> None:
    """Print the fully constructed LLM prompt using Rich formatting."""
    _debug_console.print()
    _debug_console.rule(f"[bold cyan]DEBUG PROMPT — {model}[/bold cyan]")
    _debug_console.print(
        Panel(system_prompt, title="[bold green]System Prompt[/bold green]", border_style="green")
    )
    _debug_console.print(
        Panel(
            Syntax(user_prompt, "xml", theme="monokai", word_wrap=True),
            title="[bold yellow]User Prompt[/bold yellow]",
            border_style="yellow",
        )
    )
    _debug_console.rule("[bold cyan]END DEBUG PROMPT[/bold cyan]")
    _debug_console.print()


def _provider_response_format(response_format: type[BaseModel]) -> type[BaseModel] | dict[str, Any]:
    """Build the ``response_format`` payload sent to the provider for structured output.

    The judgment schemas carry an optional ``reasoning`` field that is never read from the
    model's output: it is injected after the fact from the provider's separate thinking
    channel (see ``LLMClient.generate``). Under OpenAI/Groq *strict* structured output, every
    property is forced into ``required``, so leaving ``reasoning`` in the schema makes models
    that follow the prompt — which never asks for ``reasoning`` — fail schema validation and
    abstain (observed for Llama and Qwen on Groq, where the verdict collapses to
    ``CANNOT_ASSESS``). Strip ``reasoning`` from what the provider must emit; parsing still
    uses the full Pydantic model, so reasoning injection when thinking is enabled is unaffected.

    Models without a ``reasoning`` field are returned unchanged. Any unexpected schema shape
    falls back to the original model so behaviour never regresses.
    """
    if "reasoning" not in getattr(response_format, "model_fields", {}):
        return response_format
    try:
        param = _json_schema_param(response_format)
    except Exception:
        param = None
    # Unexpected schema shape: keep current behaviour (pass the model through).
    if not isinstance(param, dict):
        return response_format
    schema = param.get("json_schema", {}).get("schema")
    if not isinstance(schema, dict):
        return response_format
    schema.get("properties", {}).pop("reasoning", None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [key for key in required if key != "reasoning"]
    return param


def _json_schema_param(model: type[BaseModel]) -> dict[str, Any]:
    """A Pydantic model as a strict ``json_schema`` response format: every object forbids
    extra properties and requires all of its own, and ``default: null`` is dropped (what
    LiteLLM's ``type_to_response_format_param`` produced, via the OpenAI SDK's strict-schema
    pass, so cache keys and prompts are unchanged by the move off LiteLLM)."""
    schema = model.model_json_schema()
    _ensure_strict(schema, schema)
    name = re.sub(r"[^a-zA-Z0-9_-]", "_", model.__name__) or "response"
    return {"type": "json_schema", "json_schema": {"schema": schema, "name": name, "strict": True}}


def _ensure_strict(node: Any, root: dict[str, Any]) -> Any:
    """Make a JSON schema strict in place (see ``_json_schema_param``)."""
    if not isinstance(node, dict):
        return node
    for key in ("$defs", "definitions"):
        for sub in (node.get(key) or {}).values():
            _ensure_strict(sub, root)
    if node.get("type") == "object" and "additionalProperties" not in node:
        node["additionalProperties"] = False
    properties = node.get("properties")
    if isinstance(properties, dict):
        node["required"] = list(properties)
        for sub in properties.values():
            _ensure_strict(sub, root)
    if isinstance(node.get("items"), dict):
        _ensure_strict(node["items"], root)
    for key in ("anyOf", "allOf"):
        for sub in node.get(key) or []:
            _ensure_strict(sub, root)
    if len(node.get("allOf") or []) == 1:
        node.update(_ensure_strict(node.pop("allOf")[0], root))
    if "default" in node and node["default"] is None:
        node.pop("default")
    ref = node.get("$ref")
    if ref and len(node) > 1:  # a $ref with siblings is inlined, as strict mode requires
        target: Any = root
        for part in ref.removeprefix("#/").split("/"):
            target = target[part]
        node.pop("$ref")
        node.update({**target, **node})
        return _ensure_strict(node, root)
    return node


class _LLMConfigDumper(yaml.SafeDumper):
    """SafeDumper that writes enums and paths as plain YAML scalars.

    ``asdict()`` leaves ``ThinkingLevel`` members (bare, or inside a ``ThinkingConfig``)
    and ``Path`` values in place, and ``yaml.SafeDumper`` refuses both. Writing their plain
    values keeps ``LLMConfig.to_yaml`` output loadable by ``yaml.safe_load``.
    """


_LLMConfigDumper.add_multi_representer(
    Enum, lambda dumper, value: dumper.represent_data(value.value)
)
_LLMConfigDumper.add_multi_representer(
    PurePath, lambda dumper, value: dumper.represent_str(str(value))
)


@dataclass
class LLMConfig:
    """Configuration for LLM calls.

    Attributes:
        model: An Anthropic model id (e.g., "claude-opus-5-5", "claude-sonnet-4-5"); the
               LiteLLM-style "anthropic/" prefix is accepted. REQUIRED - no default.
        auth: How to authenticate: "oauth" (a Claude subscription's OAuth token, the
            default) or "api_key" (the paid API). None (default) reads ``AUTORUBRIC_AUTH``,
            else uses "oauth". See the module docstring for where each credential is read.
        temperature: Sampling temperature. None (default) omits it from the request so
            the provider's own default applies; this is recommended for reasoning models
            such as GPT-5.x and for Gemini 3, and required for Anthropic extended
            thinking. An explicit value, including 0.0, is sent unchanged. Temperature
            0.0 reduces but does not guarantee determinism on most providers.
            Note: the default changed from 0.0 to None in v1.6.0. To keep the previous
            behavior, pass ``temperature=0.0`` explicitly (it keeps the same cache keys).
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling parameter.
        timeout: Request timeout in seconds.
        max_retries: Maximum retry attempts for transient failures.
        retry_min_wait: Minimum wait between retries (seconds).
        retry_max_wait: Maximum wait between retries (seconds).
        max_parallel_requests: Maximum concurrent requests to this model's provider.
            When set, a global per-provider semaphore limits parallel requests. The limit
            applies within one event loop; each ``asyncio.run`` call has its own.
            None (default) means unlimited parallel requests.
        cache_enabled: Default caching behavior (can be overridden per-request).
        cache_dir: Directory for response cache.
        cache_ttl: Cache time-to-live in seconds (None = no expiration).
        api_key: Optional API key override (otherwise uses environment variables).
        api_base: Optional API base URL override.
        thinking: Enable thinking/reasoning mode (unified across providers). Accepts
            multiple formats:
            - ThinkingLevel enum: ThinkingLevel.HIGH, ThinkingLevel.MEDIUM, etc.
            - String: "low", "medium", "high", "none"
            - Int: Direct token budget (e.g., 32000)
            - ThinkingConfig: Full configuration with level and/or budget_tokens
            - None: Disable thinking (default)

            Provider support:
            - Anthropic: Extended thinking (claude-sonnet-4-5, claude-opus-4-5+)
            - OpenAI: Reasoning for o-series and GPT-5 models
            - Gemini: Thinking mode (2.5+, 3.0+ models)
            - DeepSeek: Reasoning content
        prompt_caching: Mark the system prompt for Anthropic's prompt cache (default: True).
            Applies only to model ids that start with ``anthropic/`` or ``claude``: the
            system message is sent with ``cache_control: {"type": "ephemeral"}``. Prompt
            caching is generally available, so no beta header is added, and an
            ``anthropic-beta`` value in ``extra_headers`` is sent unchanged. Other model
            ids, Claude models served through Bedrock or Vertex AI included, are sent
            unchanged. Providers that cache long prompts on their own (OpenAI, DeepSeek) do
            so whatever this setting is.
        seed: Random seed for reproducible outputs (OpenAI, some other providers).
        extra_headers: Additional HTTP headers for provider-specific features.
        extra_params: Additional provider-specific parameters passed to LiteLLM.

    Examples:
        # Basic usage without thinking (provider-default temperature)
        config = LLMConfig(model="claude-opus-5-5")

        # Explicit temperature
        config = LLMConfig(model="claude-haiku-4-5", temperature=0.0)  # explicit; sent as-is

        # Enable thinking with a level
        config = LLMConfig(model="anthropic/claude-sonnet-4-5-20250929", thinking="high")
        config = LLMConfig(model="claude-opus-5-5", thinking=ThinkingLevel.HIGH)

        # Enable thinking with explicit token budget
        config = LLMConfig(model="anthropic/claude-opus-4-5-20251101", thinking=32000)

        # Full control with ThinkingConfig
        config = LLMConfig(
            model="gemini/gemini-2.5-pro",
            thinking=ThinkingConfig(level=ThinkingLevel.HIGH, budget_tokens=50000)
        )
    """

    model: str  # REQUIRED - no default, must always be specified
    auth: Literal["oauth", "api_key"] | None = None  # None: AUTORUBRIC_AUTH, else "oauth"
    temperature: float | None = None  # None = provider default (not sent)
    max_tokens: int | None = None
    top_p: float | None = None
    timeout: float = 60.0
    max_retries: int = 3
    retry_min_wait: float = 1.0
    retry_max_wait: float = 60.0
    max_parallel_requests: int | None = None
    cache_enabled: bool = False
    cache_dir: str | Path = ".autorubric_cache"
    cache_ttl: int | None = None  # None = no expiration
    api_key: str | None = None
    api_base: str | None = None

    # Thinking/Reasoning (unified across providers)
    thinking: ThinkingParam = None

    # Other provider-specific features
    prompt_caching: bool = True  # Enable prompt caching by default for supported models
    seed: int | None = None  # OpenAI reproducibility
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)

    def get_thinking_config(self) -> ThinkingConfig | None:
        """Get normalized thinking configuration."""
        return _normalize_thinking_param(self.thinking)

    @classmethod
    def from_yaml(cls, path: str | Path) -> LLMConfig:
        """Load LLMConfig from a YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            LLMConfig instance with values from the YAML file.

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If required fields are missing or invalid.

        Example YAML file (llm_config.yaml):
            model: claude-opus-5-5
            # temperature omitted: the provider's default applies
            max_tokens: 1024
            cache_enabled: true
            cache_ttl: 3600

        ``thinking`` accepts a level string (``high``), an int token budget (``32000``),
        or a mapping of ``ThinkingConfig`` fields (``{level: high, budget_tokens: 9000}``),
        which is loaded as a ``ThinkingConfig``.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"LLM config file not found: {path}")

        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Invalid YAML config: expected dict, got {type(data).__name__}")

        if "model" not in data:
            raise ValueError("LLM config YAML must specify 'model' field")

        # Every dataclass field is a known key (derived, so new fields can't be missed);
        # anything else is a provider-specific param and goes to extra_params.
        known_fields = {f.name for f in fields(cls)}
        extra = {k: v for k, v in data.items() if k not in known_fields}
        if extra:
            data.setdefault("extra_params", {}).update(extra)
            for k in extra:
                del data[k]

        # to_yaml writes a ThinkingConfig as a mapping of its fields; rebuild it. Level
        # strings and int budgets are already valid ThinkingParam values.
        thinking = data.get("thinking")
        if isinstance(thinking, dict):
            try:
                data["thinking"] = ThinkingConfig(**thinking)
            except TypeError as e:
                raise ValueError(f"Invalid 'thinking' mapping in LLM config YAML: {e}") from e

        return cls(**data)

    def to_yaml(self, path: str | Path) -> None:
        """Save LLMConfig to a YAML file.

        The file holds only plain YAML (``yaml.safe_load`` reads it): enums are written as
        their values and paths as strings (so a ``Path`` ``cache_dir`` loads back as a
        ``str``). ``thinking`` is written so that ``from_yaml`` restores an equal value
        with the same ``get_thinking_config()``:

        - A level string or an int budget is written unchanged.
        - A ``ThinkingConfig`` is written as a mapping of its fields
          (``{level: high, budget_tokens: 9000}``) and loads back as a ``ThinkingConfig``.
        - A ``ThinkingLevel`` member is written as its string value (``high``) and loads
          back as that plain string, not the enum. Restoring the enum would need a custom
          YAML tag, which ``yaml.safe_load`` and hand-written files do not use, and it
          would add nothing: ``ThinkingLevel`` is a ``str`` enum, so the string compares
          equal to the member and normalizes to the same ``ThinkingConfig``.

        Args:
            path: Path to write YAML configuration file.
        """
        path = Path(path)
        data = asdict(self)

        # Remove None values and empty dicts for cleaner YAML
        data = {k: v for k, v in data.items() if v is not None and v != {}}

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, Dumper=_LLMConfigDumper, default_flow_style=False, sort_keys=False)


def _open_response_cache(cache_dir: str | Path) -> diskcache.Cache:
    """Open the on-disk response cache at ``cache_dir``, creating the directory if needed.

    The one place the response store is opened: LLM responses (``LLMClient``) and
    decision-model responses (``autorubric.decision``) share it whenever their configs
    name the same ``cache_dir``. Keys never collide, because each client hashes its own
    request description with SHA-256.
    """
    path = Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return diskcache.Cache(directory=str(path))


# ============================================================================
# The Anthropic transport (in place of litellm.acompletion)
# ============================================================================

OAUTH_BETAS = "oauth-2025-04-20,claude-code-20250219"
# A request on a subscription's OAuth token is accepted only when its system prompt opens
# with the Claude Code identity line.
CLAUDE_CODE_IDENTITY = "You are Claude Code, Anthropic's official CLI for Claude."
KEYCHAIN_SERVICE = "Claude Code-credentials"
DEFAULT_MAX_TOKENS = 16000
# Models that take only adaptive thinking ({"type": "adaptive"} with an effort level): a
# thinking budget_tokens is rejected (or deprecated) on these.
_ADAPTIVE_ONLY = re.compile(r"claude-(opus|sonnet|haiku|fable)-(5|4-[6-9])")


def resolve_auth(auth: str | None) -> str:
    """The auth mode to use: ``auth``, else ``AUTORUBRIC_AUTH``, else ``"oauth"``."""
    mode = auth or os.environ.get("AUTORUBRIC_AUTH") or "oauth"
    if mode not in ("oauth", "api_key"):
        raise ValueError(f"auth must be 'oauth' or 'api_key', not {mode!r}")
    return mode


def oauth_token() -> str:
    """A Claude subscription's OAuth token: ``CLAUDE_TOKEN_<NAME>`` when ``CLAUDE_ACCOUNT`` names
    an account, else ``CLAUDE_CODE_OAUTH_TOKEN``, else the token the ``claude`` CLI keeps in the
    macOS Keychain (which must not have expired; running ``claude`` once refreshes it)."""
    if name := os.environ.get("CLAUDE_ACCOUNT"):
        key = f"CLAUDE_TOKEN_{name.upper()}"
        if token := os.environ.get(key):
            return token
        raise RuntimeError(f"CLAUDE_ACCOUNT={name} but {key} is not set")
    if token := os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
        return token
    try:
        raw = subprocess.check_output(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        creds = json.loads(raw)["claudeAiOauth"]
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        creds = None
    if creds is None:
        raise RuntimeError(
            "No Claude subscription token: set CLAUDE_CODE_OAUTH_TOKEN "
            "(from `claude setup-token`), or CLAUDE_ACCOUNT and CLAUDE_TOKEN_<NAME>, "
            "or log in with `claude`; "
            "or use auth='api_key' with ANTHROPIC_API_KEY"
        )
    if creds["expiresAt"] / 1000 < time.time() + 60:
        raise RuntimeError(
            "The Claude CLI's Keychain token has expired; run `claude` once to refresh it"
        )
    return creds["accessToken"]


def _anthropic_client(
    auth: str, api_key: str | None, api_base: str | None, timeout: float
) -> anthropic.AsyncAnthropic:
    """An async client for one call. Retries are left to the caller (tenacity), so the SDK
    makes none of its own."""
    import anthropic

    if auth == "oauth":
        client = anthropic.AsyncAnthropic(
            auth_token=oauth_token(),
            base_url=api_base,
            timeout=timeout,
            max_retries=0,
            default_headers={"anthropic-beta": OAUTH_BETAS},
        )
        client.api_key = None  # the SDK reads ANTHROPIC_API_KEY on its own; never send it here
        return client
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("auth='api_key' needs LLMConfig.api_key or ANTHROPIC_API_KEY")
    return anthropic.AsyncAnthropic(api_key=key, base_url=api_base, timeout=timeout, max_retries=0)


def anthropic_model(model: str) -> str:
    """An Anthropic model id from ``model``, dropping a LiteLLM-style ``anthropic/`` prefix."""
    name = model.removeprefix("anthropic/")
    if "/" in name or not name.startswith("claude"):
        raise ValueError(f"Only Anthropic models are supported in this fork, not {model!r}")
    return name


def _thinking_params(
    model: str, params: dict[str, Any]
) -> tuple[dict[str, Any] | None, str | None]:
    """The request's ``thinking`` and effort level from a ``thinking`` budget or a
    ``reasoning_effort`` level. Models that take only adaptive thinking get it with the
    nearest effort level; older models get an explicit budget."""
    budget = (params.get("thinking") or {}).get("budget_tokens")
    level = params.get("reasoning_effort")
    if budget is None and level is None:
        return None, None
    if level == "none":
        return None, None
    if _ADAPTIVE_ONLY.search(model):
        if level is None:
            level = "low" if budget <= 2048 else "medium" if budget <= 8192 else "high"
        return {"type": "adaptive"}, level
    if budget is None:
        budget = THINKING_LEVEL_BUDGETS.get(level, 2048)
    return {"type": "enabled", "budget_tokens": budget}, None


def _system_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [dict(block) for block in content]
    return [{"type": "text", "text": content}]


async def _acompletion(**params: Any) -> Any:
    """One Messages API call from the request ``LLMClient.generate`` builds (OpenAI-shaped:
    system and user messages, ``response_format``, ``thinking`` or ``reasoning_effort``),
    returning an OpenAI-shaped response: ``choices[0].message.content`` (and
    ``.reasoning_content``, the thinking), ``usage`` with ``prompt_tokens`` (all input,
    cached included), ``completion_tokens``, the cache counts, and ``cost`` (USD, or None)."""
    params = dict(params)
    model = anthropic_model(params.pop("model"))
    auth = resolve_auth(params.pop("auth", None))
    timeout = params.pop("timeout", 60.0)
    messages = params.pop("messages")
    system: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    for m in messages:
        if m["role"] == "system":
            system += _system_blocks(m["content"])
        else:
            turns.append({"role": m["role"], "content": m["content"]})
    if auth == "oauth":
        system = [{"type": "text", "text": CLAUDE_CODE_IDENTITY}, *system]
    thinking, effort = _thinking_params(model, params)
    params.pop("thinking", None)
    params.pop("reasoning_effort", None)
    output_config: dict[str, Any] = {"effort": effort} if effort else {}
    response_format = params.pop("response_format", None)
    if response_format is not None:
        if isinstance(response_format, dict):
            schema = response_format["json_schema"]["schema"]
        else:
            schema = _json_schema_param(response_format)["json_schema"]["schema"]
        output_config["format"] = {"type": "json_schema", "schema": schema}
    api_key = params.pop("api_key", None)
    api_base = params.pop("api_base", None)
    extra_headers = params.pop("extra_headers", None)
    params.pop("seed", None)  # not an Anthropic parameter
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": params.pop("max_tokens", None) or DEFAULT_MAX_TOKENS,
        "system": system,
        "messages": turns,
    }
    if thinking:
        kwargs["thinking"] = thinking
    if output_config:
        kwargs["output_config"] = output_config
    if extra_headers:
        kwargs["extra_headers"] = extra_headers
    if params:  # temperature, top_p and any extra_params go as they are
        kwargs["extra_body"] = params
    client = _anthropic_client(auth, api_key, api_base, timeout)
    async with client:
        async with client.messages.stream(**kwargs) as stream:
            msg = await stream.get_final_message()
    text = "".join(block.text for block in msg.content if block.type == "text")
    trace = "\n".join(
        block.thinking
        for block in msg.content
        if block.type == "thinking" and getattr(block, "thinking", None)
    )
    u = msg.usage
    cache_write = getattr(u, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(u, "cache_read_input_tokens", 0) or 0
    prompt_tokens = (u.input_tokens or 0) + cache_write + cache_read
    cost = None
    if auth == "api_key" and (price := API_PRICES_PER_MTOK.get(re.sub(r"-\d{8}$", "", model))):
        cost = ((u.input_tokens or 0) + 1.25 * cache_write + 0.1 * cache_read) * price[0] / 1e6 + (
            u.output_tokens or 0
        ) * price[1] / 1e6
    message = SimpleNamespace(
        content=text, reasoning_content=trace or None, thinking_blocks=None, thinking=None
    )
    return SimpleNamespace(
        model=msg.model,
        choices=[SimpleNamespace(message=message, finish_reason=msg.stop_reason, index=0)],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=u.output_tokens or 0,
            total_tokens=prompt_tokens + (u.output_tokens or 0),
            cache_creation_input_tokens=cache_write,
            cache_read_input_tokens=cache_read,
        ),
        cost=cost,
    )


class LLMClient:
    """Unified LLM client with retries, caching, and structured output support.

    Uses diskcache for efficient, thread-safe response caching.
    """

    def __init__(self, config: LLMConfig, *, cache_namespace: str | None = None):
        """Initialize LLM client.

        Args:
            config: LLMConfig instance. The model field is required.
            cache_namespace: Keeps this client's response-cache entries apart from those of
                other clients that send identical requests, such as several judges polling
                one model, whose answers are independent samples. It becomes part of the
                cache key; ``None`` (the default) leaves the key exactly as without it.

        Raises:
            ValueError: If config.model is not specified.
        """
        if not config.model:
            raise ValueError("LLMConfig.model is required and cannot be empty")

        self.config = config
        self._cache_namespace = cache_namespace
        self._cache: diskcache.Cache | None = None

        if self.config.cache_enabled:
            self._init_cache()

    def _init_cache(self) -> None:
        """Initialize diskcache instance."""
        self._cache = _open_response_cache(self.config.cache_dir)

    def _ensure_cache(self) -> diskcache.Cache:
        """Ensure cache is initialized, creating it if needed."""
        if self._cache is None:
            self._init_cache()
        return self._cache  # type: ignore[return-value]

    def _cache_key(
        self,
        model: str,
        system_prompt: str,
        user_prompt: str,
        response_format: type | None = None,
        overrides: dict[str, Any] | None = None,
    ) -> str:
        """Generate a unique cache key for the request.

        Includes sampling parameters so that different configurations
        (temperature, thinking, top_p, seed) produce distinct cache entries, and the
        client's ``cache_namespace`` when it has one.

        Args:
            model: Model identifier sent with the request.
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            response_format: Optional Pydantic model class for structured output.
            overrides: The per-call overrides passed to ``generate``. A ``temperature``
                override replaces ``config.temperature`` in the key, so an override never
                reuses a response cached at a different temperature. None (provider
                default) is rendered as ``None``; explicit values keep the same key format
                as before, so existing cache entries still match.
        """
        temperature = (overrides or {}).get("temperature", self.config.temperature)
        schema_name = response_format.__name__ if response_format else "str"
        thinking_config = self.config.get_thinking_config()
        thinking_str = str(thinking_config) if thinking_config else "none"
        content = (
            f"{model}:{system_prompt}:{user_prompt}:{schema_name}"
            f":temp={temperature}"
            f":top_p={self.config.top_p}"
            f":max_tokens={self.config.max_tokens}"
            f":thinking={thinking_str}"
            f":seed={self.config.seed}"
        )
        if self._cache_namespace is not None:
            content += f":namespace={self._cache_namespace}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_retry_decorator(self) -> Any:
        """Build tenacity retry decorator from config."""
        import anthropic

        return retry(
            retry=retry_if_exception_type(
                (
                    anthropic.RateLimitError,
                    anthropic.OverloadedError,
                    anthropic.ServiceUnavailableError,
                    anthropic.InternalServerError,
                    anthropic.APIConnectionError,  # APITimeoutError is one
                )
            ),
            stop=stop_after_attempt(self.config.max_retries),
            wait=wait_exponential(
                min=self.config.retry_min_wait,
                max=self.config.retry_max_wait,
            ),
            reraise=True,
        )

    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: type[T] | None = None,
        use_cache: bool | None = None,
        return_thinking: bool = False,
        return_result: bool = False,
        **kwargs: Any,
    ) -> str | T | GenerateResult:
        """Generate LLM response with optional structured output.

        Args:
            system_prompt: System message for the LLM.
            user_prompt: User message for the LLM.
            response_format: Optional Pydantic model class for structured output.
                When provided, LiteLLM uses the model's JSON schema to constrain
                the LLM output and returns a validated Pydantic instance.
            use_cache: Whether to use caching for this request.
                - None (default): Use config.cache_enabled setting
                - True: Force cache usage (initializes cache if needed)
                - False: Skip cache for this request
            return_thinking: If True and thinking is enabled, return a GenerateResult
                with both content and thinking. If False (default), only return content.
                Note: When response_format is provided, thinking is injected into the
                'reasoning' field if it exists, regardless of this setting.
            return_result: If True, always return a GenerateResult with full details
                including usage statistics and completion cost. This is useful when
                you need to track token usage. When True, takes precedence over the
                default return behavior.
            **kwargs: Override any LLMConfig parameters for this call. In particular,
                ``temperature`` accepts a float to send for this call, or None to omit it
                so the provider's default applies (regardless of ``config.temperature``).

        Returns:
            If return_result=True or return_thinking=True: GenerateResult with content,
                thinking, usage, cost, and parsed (if response_format was provided).
            If response_format is None: String response from the LLM.
            If response_format is provided: Validated Pydantic model instance.
                If thinking is enabled and the response_format has a 'reasoning'
                field, it will be populated with the model's thinking trace.

        Raises:
            anthropic.APIError: If all retries fail
            pydantic.ValidationError: If response doesn't match schema
        """
        # Determine caching behavior for this request
        should_cache = use_cache if use_cache is not None else self.config.cache_enabled

        # Effective temperature: a per-call override wins; None means "omit from the request".
        temperature: float | None = kwargs.get("temperature", self.config.temperature)

        # Check cache first
        cache_key: str | None = None
        if should_cache:
            cache = self._ensure_cache()
            cache_key = self._cache_key(
                self.config.model,
                system_prompt,
                user_prompt,
                response_format,
                kwargs,
            )
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug(f"Cache hit for {cache_key[:8]}...")
                return cached  # type: ignore[return-value]

        # Build request parameters
        model = kwargs.get("model", self.config.model)

        # Check if this is an Anthropic model that supports prompt caching
        # Anthropic requires cache_control on message content; other providers
        # handle caching automatically (OpenAI, Deepseek) or don't support it
        is_anthropic = model.startswith("anthropic/") or model.startswith("claude")
        use_prompt_caching = self.config.prompt_caching and is_anthropic
        if use_prompt_caching:
            # Anthropic requires cache_control on message content
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                },
                {"role": "user", "content": user_prompt},
            ]
        else:
            # Standard message format for other providers
            # OpenAI/Deepseek: Caching is automatic for prompts ≥1024 tokens
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

        params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            # Only send temperature when set, so the provider default applies otherwise
            # (0.0 is a real value and is sent).
            **({"temperature": temperature} if temperature is not None else {}),
            "timeout": kwargs.get("timeout", self.config.timeout),
            **self.config.extra_params,
        }

        if self.config.max_tokens:
            params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        if self.config.top_p:
            params["top_p"] = kwargs.get("top_p", self.config.top_p)
        if self.config.auth:
            params["auth"] = self.config.auth
        if self.config.api_key:
            params["api_key"] = self.config.api_key
        if self.config.api_base:
            params["api_base"] = self.config.api_base

        # Thinking/Reasoning configuration (unified across providers)
        thinking_config = self.config.get_thinking_config()
        if thinking_config is not None:
            # Determine whether to use reasoning_effort or explicit thinking dict
            # Use explicit budget_tokens when specified for fine-grained control
            # Otherwise use reasoning_effort for better cross-provider compatibility
            if thinking_config.budget_tokens is not None:
                # Explicit token budget - use thinking dict (Anthropic/Gemini style)
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": thinking_config.budget_tokens,
                }
            else:
                # Level-based - use reasoning_effort for cross-provider support
                # LiteLLM translates this to provider-specific parameters
                params["reasoning_effort"] = thinking_config.get_reasoning_effort()

        # Extra headers configuration
        extra_headers = dict(self.config.extra_headers)
        if extra_headers:
            params["extra_headers"] = extra_headers

        if self.config.seed is not None:
            params["seed"] = self.config.seed

        # Enable structured output if Pydantic model provided. The schema sent to the
        # provider drops the post-hoc ``reasoning`` slot so strict-mode backends don't force
        # non-OpenAI models (e.g. Llama/Qwen on Groq) to emit it; see
        # _provider_response_format. Parsing below still uses the full Pydantic model.
        if response_format is not None:
            params["response_format"] = _provider_response_format(response_format)

        # Print debug prompt if debug mode is enabled
        import autorubric

        if autorubric.debug:
            _print_debug_prompt(system_prompt, user_prompt, model)

        # Make request with retries
        retry_decorator = self._get_retry_decorator()
        thinking_content: str | None = None
        raw_response: Any = None

        @retry_decorator
        async def _call() -> str:
            nonlocal thinking_content, raw_response
            response = await _acompletion(**params)
            raw_response = response

            message = response.choices[0].message

            # Extract thinking/reasoning content (_acompletion puts it on reasoning_content)
            thinking_content = _extract_thinking_content(message)

            return message.content  # type: ignore[return-value]

        # Apply rate limiting if configured
        semaphore = await RateLimitPool.get_instance().get_semaphore(
            model, self.config.max_parallel_requests
        )
        if semaphore is not None:
            async with semaphore:
                response_content = await _call()
        else:
            response_content = await _call()

        # Extract usage and cost from the raw response
        usage = _extract_usage_from_response(raw_response)
        cost = _calculate_completion_cost(raw_response)

        # Parse structured output if requested
        parsed_response: T | None = None
        if response_format is not None:
            # The reply is a JSON string when response_format is set; parse it into the model
            data = json.loads(response_content)

            # Inject thinking content into the reasoning field if available
            if thinking_content and "reasoning" in response_format.model_fields:
                data["reasoning"] = thinking_content

            parsed_response = response_format.model_validate(data)

        # Determine what to return
        result: str | T | GenerateResult
        if return_result or return_thinking:
            # Return full GenerateResult with all details
            result = GenerateResult(
                content=response_content,
                thinking=thinking_content,
                raw_response=raw_response,
                usage=usage,
                cost=cost,
                parsed=parsed_response,
            )
        elif response_format is not None:
            # Return just the parsed Pydantic model
            result = parsed_response  # type: ignore[assignment]
        else:
            # Return just the string content
            result = response_content

        # Cache the response (cache the parsed object for structured outputs)
        if should_cache and cache_key:
            cache = self._ensure_cache()
            cache.set(
                cache_key,
                result,
                expire=self.config.cache_ttl,
            )
            logger.debug(f"Cached response for {cache_key[:8]}...")

        return result

    def clear_cache(self) -> int:
        """Clear all cached responses.

        Returns:
            Number of entries cleared.
        """
        if self._cache is None:
            return 0
        count = len(self._cache)
        self._cache.clear()
        return count

    def cache_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with 'size', 'count', and 'directory' keys.
        """
        if self._cache is None:
            return {"size": 0, "count": 0, "directory": None}
        return {
            "size": self._cache.volume(),
            "count": len(self._cache),
            "directory": str(self._cache.directory),
        }

    def close(self) -> None:
        """Close the on-disk cache, releasing its file handles.

        diskcache keeps the underlying SQLite database open for the lifetime of
        the cache object. Call this when done with the client so the cache
        directory can be removed (notably on Windows, which refuses to delete
        files that are still open). Safe to call when no cache was initialized.
        """
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    def __enter__(self) -> LLMClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


# Convenience function for simple usage
async def generate(
    system_prompt: str,
    user_prompt: str,
    model: str,
    response_format: type[T] | None = None,
    **kwargs: Any,
) -> str | T:
    """Simple one-shot generation function.

    For repeated calls, prefer creating an LLMClient instance.

    Args:
        system_prompt: System message for the LLM.
        user_prompt: User message for the LLM.
        model: Model identifier (REQUIRED).
        response_format: Optional Pydantic model for structured output.
        **kwargs: Additional LLMConfig parameters.

    Example:
        # Simple string response
        response = await generate(
            "You are a helpful assistant.",
            "What is 2+2?",
            model="claude-haiku-4-5"
        )

        # Structured output
        from pydantic import BaseModel

        class MathAnswer(BaseModel):
            result: int
            explanation: str

        answer = await generate(
            "You are a math tutor.",
            "What is 2+2?",
            model="claude-haiku-4-5",
            response_format=MathAnswer
        )
        print(answer.result)  # 4
    """
    config = LLMConfig(model=model, **kwargs)
    client = LLMClient(config)
    return await client.generate(system_prompt, user_prompt, response_format=response_format)
