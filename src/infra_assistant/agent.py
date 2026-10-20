"""The tool-calling loop.

Given a user message and the retained history, run the model → tools → model
loop until it produces a text answer or hits a bound:

* **tool-call cap** (``[llm].max_tool_calls_per_turn``, default 12): once
  reached, the model is asked once more *with no tools* so it must answer from
  what it already has.
* **wall-clock cap** (``[llm].turn_wall_time_seconds``, default 60): checked
  before every model call; exceeding it raises :class:`TurnTimeout`, and the
  bot sends a clean "couldn't finish in time" message — never a half-built one.

The provider fallback chain sits underneath every model call, so a provider
outage or a peak window is handled transparently here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from .config import LLMConfig
from .errors import TurnTimeout
from .llm import Completer, LLMResponse, ToolCall, tool_defs_to_openai
from .mcp_client import ToolBackend
from .memory import Message, as_provider_messages

logger = logging.getLogger("infra_assistant.agent")

_EMPTY_ANSWER = "(the model returned an empty response)"
_BUDGET_NOTICE = "\n[tool-output budget for this turn reached — further results are trimmed]"


class Agent:
    def __init__(
        self,
        *,
        system_prompt: str,
        chain: Completer,
        llm_config: LLMConfig,
        mcp_pool: ToolBackend | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._system_prompt = system_prompt
        self._chain = chain
        self._cfg = llm_config
        self._pool = mcp_pool
        self._clock = clock
        self._openai_tools = (
            tool_defs_to_openai(mcp_pool.tools()) if mcp_pool and mcp_pool.tool_count else None
        )

    async def answer(self, user_message: str, history: Sequence[Message] = ()) -> str:
        """Run one full turn and return the assistant's text answer."""
        turn_started = self._clock()
        deadline = turn_started + self._cfg.turn_wall_time_seconds
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._system_prompt},
            *as_provider_messages(history),
            {"role": "user", "content": user_message},
        ]

        calls_made = 0
        tool_output_chars = 0
        seen_calls: dict[tuple[str, str], str] = {}
        metrics = _TurnMetrics()
        while True:
            tools_offered = (
                self._openai_tools if calls_made < self._cfg.max_tool_calls_per_turn else None
            )
            response = await self._complete(messages, tools_offered, deadline)
            metrics.add_model_call(response)

            if not response.wants_tools:
                metrics.log(self._clock() - turn_started, tool_calls=calls_made)
                return response.text or _EMPTY_ANSWER

            if tools_offered is None:
                # The model asked for tools after we stopped offering them. Give
                # it one plain nudge, then take whatever text it returns.
                final = await self._complete(
                    [*messages, {"role": "user", "content": _NUDGE}], None, deadline
                )
                metrics.add_model_call(final)
                metrics.log(self._clock() - turn_started, tool_calls=calls_made)
                return final.text or _EMPTY_ANSWER

            messages.append(_assistant_tool_turn(response))
            # The calls the model asked for in one turn are independent reads —
            # run them concurrently (each still bounded by its own
            # tool_call_timeout in MCPPool.call), but append the results in the
            # model's original order and apply the budgets in that order.
            remaining = max(0, self._cfg.max_tool_calls_per_turn - calls_made)
            to_run = list(response.tool_calls[:remaining])
            over_budget = list(response.tool_calls[remaining:])
            calls_made += len(to_run)
            results = await asyncio.gather(*(self._run_tool(c, seen_calls) for c in to_run))
            for call, result in zip(to_run, results, strict=True):
                if tool_output_chars >= self._cfg.max_tool_output_chars:
                    result = result[:200] + _BUDGET_NOTICE
                tool_output_chars += len(result)
                messages.append(_tool_result(call, result))
            for call in over_budget:
                messages.append(
                    _tool_result(call, "ERROR: tool-call budget for this turn is exhausted")
                )

    # -- internals --------------------------------------------------

    async def _complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
        deadline: float,
    ) -> LLMResponse:
        """One model turn, hard-bounded by the wall clock.

        The remaining-budget check handles the "already over" case
        deterministically; ``asyncio.timeout`` is the backstop that guarantees
        the turn still ends if the underlying call hangs with no timeout of its
        own (the between-calls check alone cannot bound a stuck ``await``).
        """
        remaining = deadline - self._clock()
        if remaining <= 0:
            raise TurnTimeout(f"turn exceeded its {self._cfg.turn_wall_time_seconds:g}s budget")
        try:
            async with asyncio.timeout(remaining):
                return await self._chain.complete(messages, tools)
        except TimeoutError as exc:
            raise TurnTimeout(
                f"turn exceeded its {self._cfg.turn_wall_time_seconds:g}s budget"
            ) from exc

    async def _run_tool(self, call: ToolCall, seen_calls: dict[tuple[str, str], str]) -> str:
        if self._pool is None:  # pragma: no cover - no tools => model never calls one
            return f"ERROR: no MCP tools are configured; cannot run {call.name!r}"
        key = _call_key(call)
        cached = seen_calls.get(key)
        if cached is not None:
            # A weak model can re-issue an identical call instead of reading the
            # result it already has — serve the cached result rather
            # than paying for another real round-trip, and say so explicitly so
            # the model sees it already has this answer.
            logger.info(
                "tool call: %s(%s) — repeated this turn, reusing cached result",
                call.name,
                ", ".join(sorted(call.arguments)),
            )
            return _REPEAT_PREFIX + cached
        # Values can carry client IPs / query fragments — keys at INFO, full at DEBUG.
        logger.info("tool call: %s(%s)", call.name, ", ".join(sorted(call.arguments)))
        logger.debug("tool call args: %s(%s)", call.name, _short_args(call.arguments))
        result = await self._pool.call(call.name, call.arguments)
        seen_calls[key] = result
        return result


class _TurnMetrics:
    """Accumulates usage across every model call in one turn.

    A turn can involve several model round-trips (tool calls, the tool-cap
    nudge) — this sums what's reported and logs one line at the end, in a
    plain ``key=value`` shape a log-scraping exporter can parse without a JSON
    dependency. Any field a provider didn't report stays absent, never ``0``,
    so "unknown" and "zero" are never confused downstream.
    """

    def __init__(self) -> None:
        self.model_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost_usd = 0.0
        self._have_tokens = False
        self._have_cost = False
        self.last_provider: str | None = None

    def add_model_call(self, response: LLMResponse) -> None:
        self.model_calls += 1
        if response.provider_name:
            self.last_provider = response.provider_name
        if response.prompt_tokens is not None:
            self.prompt_tokens += response.prompt_tokens
            self._have_tokens = True
        if response.completion_tokens is not None:
            self.completion_tokens += response.completion_tokens
            self._have_tokens = True
        if response.cost_usd is not None:
            self.cost_usd += response.cost_usd
            self._have_cost = True

    def log(self, elapsed_seconds: float, *, tool_calls: int) -> None:
        parts = [
            f"latency_ms={round(elapsed_seconds * 1000)}",
            f"model_calls={self.model_calls}",
            f"tool_calls={tool_calls}",
            f"provider={self.last_provider or 'unknown'}",
        ]
        if self._have_tokens:
            parts.append(f"prompt_tokens={self.prompt_tokens}")
            parts.append(f"completion_tokens={self.completion_tokens}")
            parts.append(f"total_tokens={self.prompt_tokens + self.completion_tokens}")
        if self._have_cost:
            parts.append(f"cost_usd={self.cost_usd:.6f}")
        logger.info("turn complete: %s", " ".join(parts))


_NUDGE = (
    "You have used the tool-call budget for this turn. Answer now using the "
    "information you already have, and note anything you could not check."
)

_REPEAT_PREFIX = (
    "NOTE: you already called this tool with these exact arguments earlier in this "
    "turn. The result is unchanged and repeated below — answer from it rather than "
    "calling this tool again.\n\n"
)


def _call_key(call: ToolCall) -> tuple[str, str]:
    return call.name, json.dumps(call.arguments, sort_keys=True, default=str)


def _assistant_tool_turn(response: LLMResponse) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": response.text or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in response.tool_calls
        ],
    }


def _tool_result(call: ToolCall, content: str) -> dict[str, object]:
    return {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": content}


def _short_args(arguments: dict[str, object]) -> str:
    text = json.dumps(arguments, default=str)
    return text if len(text) <= 200 else text[:200] + "…"
