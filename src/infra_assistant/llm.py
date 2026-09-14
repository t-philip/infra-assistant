"""Provider-neutral LLM types and the ordered fallback chain (capability 15).

The chain does two things and nothing else:

1. **Failover.** Try each configured provider in order; on any error or timeout,
   move to the next. Only when every entry has failed does it raise.
2. **Cost-window avoidance.** An entry with a ``skip_between`` window is skipped
   while the clock is inside that window, and the chain moves on. This is the
   replacement for a "refuse during peak hours" plugin: route to the next
   provider (typically a local model) instead of declining to answer.

There is no difficulty-based routing — that needs a classifier to maintain and
buys little. The chain is failover plus window-skipping, full stop.

Messages are OpenAI-shaped dicts (``role`` / ``content`` / ``tool_calls`` /
``tool_call_id``); tools are OpenAI ``{"type": "function", "function": {...}}``
dicts. Both are the de-facto interchange format and what ``litellm`` speaks
natively.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

from .config import LLMConfig, ProviderSpec
from .errors import AllProvidersFailed, ConfigError, ProviderError

logger = logging.getLogger("infra_assistant.llm")


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class LLMResponse:
    """One model turn: free text, or a batch of tool calls, or both.

    The ``prompt_tokens``/``completion_tokens``/``total_tokens``/``cost_usd``
    fields are best-effort (Tier B observability) — a provider that
    doesn't report usage, or an unpriced model litellm has no rate for (e.g. a
    local Ollama model), leaves them ``None`` rather than guessing. Never used
    for anything functional; purely so the agent can log what one turn cost.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    provider_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class Completer(Protocol):
    """Anything the agent can ask for one model turn — a provider or the chain."""

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
    ) -> LLMResponse: ...


@runtime_checkable
class Provider(Completer, Protocol):
    """A single LLM endpoint the chain can call."""

    name: str

    async def aclose(self) -> None: ...


ProviderFactory = Callable[[ProviderSpec, dict[str, str], float, bool], Provider]


def _default_now() -> datetime:
    return datetime.now(UTC)


def parse_tool_arguments(raw: Any) -> dict[str, Any]:
    """Coerce a provider's ``function.arguments`` into a dict.

    Providers send either a JSON string or (some litellm paths) an object. A
    malformed value is a :class:`ProviderError`, not a crash.
    """
    if isinstance(raw, dict):
        return raw
    if raw in (None, ""):
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderError(f"tool call arguments are not valid JSON: {raw!r}") from exc
        if not isinstance(parsed, dict):
            raise ProviderError(f"tool call arguments must be a JSON object, got {parsed!r}")
        return parsed
    raise ProviderError(f"unsupported tool call arguments type: {type(raw).__name__}")


class ProviderChain:
    """Ordered failover across the configured providers, with window-skipping."""

    def __init__(
        self,
        config: LLMConfig,
        secrets: dict[str, str],
        *,
        provider_factory: ProviderFactory,
        now_fn: Callable[[], datetime] = _default_now,
    ) -> None:
        self._config = config
        self._secrets = secrets
        self._factory = provider_factory
        self._now = now_fn
        self._instances: dict[int, Provider] = {}

    def _provider_for(self, index: int, spec: ProviderSpec) -> Provider:
        cached = self._instances.get(index)
        if cached is None:
            cached = self._factory(
                spec,
                self._secrets,
                self._config.per_call_timeout_seconds,
                self._config.use_litellm,
            )
            self._instances[index] = cached
        return cached

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
    ) -> LLMResponse:
        """Return the first provider's answer; raise once all have failed/skipped.

        A failure of *any* kind on one entry — a bad response, a timeout, or a
        ``ConfigError`` from constructing the provider (missing base_url, missing
        key) — is caught and the chain moves on. One misconfigured or unhealthy
        entry never disables the entries after it. ``asyncio.CancelledError`` is
        re-raised: cancellation is not a provider failure.
        """
        now = self._now()
        last_error = "no provider was tried"
        skipped_all = True

        for index, spec in enumerate(self._config.chain):
            if spec.skipped_at(now):
                logger.info(
                    "provider %d (%s/%s) skipped — inside a peak window",
                    index,
                    spec.provider,
                    spec.model,
                )
                continue
            skipped_all = False
            try:
                provider = self._provider_for(index, spec)
                response = await provider.complete(messages, tools)
                # The chain, not the provider, is the single source of truth for
                # which entry actually answered — stamped from the spec itself
                # so a bare Completer test double need not carry a `.name`.
                response = dataclasses.replace(
                    response, provider_name=f"{spec.provider}:{spec.model}"
                )
                if index > 0:
                    logger.info(
                        "provider %d (%s) answered after %d earlier failure(s)",
                        index,
                        spec.provider,
                        index,
                    )
                return response
            except asyncio.CancelledError:
                raise
            except (ProviderError, ConfigError) as exc:
                last_error = type(exc.__cause__ or exc).__name__
                logger.warning(
                    "provider %d (%s/%s) failed: %s", index, spec.provider, spec.model, exc
                )
            except Exception as exc:  # an unforeseen provider bug must not kill the turn
                last_error = type(exc).__name__
                logger.warning(
                    "provider %d (%s/%s) raised an unexpected %s: %s",
                    index,
                    spec.provider,
                    spec.model,
                    type(exc).__name__,
                    exc,
                )

        if skipped_all:
            raise AllProvidersFailed("every provider is inside a peak window right now")
        raise AllProvidersFailed(last_error)

    async def aclose(self) -> None:
        for provider in self._instances.values():
            try:
                await provider.aclose()
            except Exception:  # pragma: no cover - best-effort cleanup
                logger.warning("error closing provider %s", provider.name, exc_info=True)


def tool_defs_to_openai(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap ``{name, description, input_schema}`` tool dicts in OpenAI's envelope."""
    wrapped: list[dict[str, Any]] = []
    for tool in tools:
        wrapped.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema")
                    or tool.get("parameters")
                    or {"type": "object", "properties": {}},
                },
            }
        )
    return wrapped


def parse_openai_choice(
    message: dict[str, Any],
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    total_tokens: int | None = None,
    cost_usd: float | None = None,
) -> LLMResponse:
    """Turn an OpenAI-shaped ``choices[0].message`` into an :class:`LLMResponse`.

    The usage/cost kwargs are the caller's best-effort extraction from its own
    response shape — this function itself stays shape-agnostic.
    """
    text = message.get("content")
    if isinstance(text, str):
        text = text or None
    raw_calls = message.get("tool_calls") or []
    calls: list[ToolCall] = []
    for entry in raw_calls:
        fn = entry.get("function", {})
        calls.append(
            ToolCall(
                id=str(entry.get("id") or f"call_{len(calls)}"),
                name=str(fn.get("name", "")),
                arguments=parse_tool_arguments(fn.get("arguments")),
            )
        )
    return LLMResponse(
        text=text,
        tool_calls=tuple(calls),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cost_usd=cost_usd,
    )


def usage_field(usage: Any, key: str) -> int | None:
    """Read one token-count field from a provider's ``usage`` blob.

    Handles both a dict (``HTTPProvider``'s parsed JSON) and an attribute-style
    object (litellm's ``Usage``) — never raises on a missing/odd-shaped value.
    """
    if usage is None:
        return None
    value = usage.get(key) if isinstance(usage, dict) else getattr(usage, key, None)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
