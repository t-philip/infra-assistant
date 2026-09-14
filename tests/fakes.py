"""Test doubles for the LLM chain, the MCP tool backend, and Telegram.

All data here is fictitious: chat ids like ``111`` / ``222``, host names
``node-a``, addresses in 192.0.2.0/24, free text ``"Test Data ..."``.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Any

from infra_assistant.llm import LLMResponse


class FakeCompleter:
    """Plays a scripted list of responses/exceptions, one per ``complete`` call."""

    def __init__(self, script: Sequence[LLMResponse | Exception]) -> None:
        self._script = list(script)
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
    ) -> LLMResponse:
        self.calls.append((list(messages), list(tools) if tools is not None else None))
        if not self._script:
            raise AssertionError("FakeCompleter ran out of scripted responses")
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    @property
    def last_tools(self) -> list[dict[str, Any]] | None:
        return self.calls[-1][1] if self.calls else None


ToolFn = Callable[[dict[str, Any]], Any]  # returns str, or an awaitable of str


class FakeToolBackend:
    """Structurally satisfies ``infra_assistant.mcp_client.ToolBackend``."""

    def __init__(
        self,
        tools: Sequence[dict[str, Any]] | None = None,
        results: dict[str, str | ToolFn] | None = None,
    ) -> None:
        self._tools = list(tools or [])
        self._results = dict(results or {})
        self.invocations: list[tuple[str, dict[str, Any]]] = []

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def tools(self) -> list[dict[str, Any]]:
        return list(self._tools)

    async def call(self, prefixed_name: str, arguments: dict[str, Any]) -> str:
        self.invocations.append((prefixed_name, arguments))
        result = self._results.get(prefixed_name)
        if result is None:
            return f"ERROR: no such tool {prefixed_name!r}"
        if callable(result):
            out = result(arguments)
            resolved = await out if inspect.isawaitable(out) else out
            return str(resolved)
        return result


def tool_def(name: str, description: str = "Test Data tool") -> dict[str, Any]:
    return {
        "name": name,
        "description": description,
        "input_schema": {"type": "object", "properties": {}},
    }


# -- Telegram doubles ----------------------------------------------------------


class FakeMessage:
    def __init__(self, text: str | None) -> None:
        self.text = text
        self.replies: list[str] = []

    async def reply_text(self, text: str, **_: Any) -> None:
        self.replies.append(text)


class FakeChat:
    def __init__(self, chat_id: int) -> None:
        self.id = chat_id


class FakeBot:
    def __init__(self) -> None:
        self.chat_actions: list[tuple[int, str]] = []
        self.messages: list[tuple[int, str]] = []

    async def send_chat_action(self, chat_id: int, action: str, **_: Any) -> None:
        self.chat_actions.append((chat_id, str(action)))

    async def send_message(self, chat_id: int, text: str, **_: Any) -> None:
        self.messages.append((chat_id, text))


class FakeUpdate:
    def __init__(self, chat_id: int, text: str | None) -> None:
        self.effective_chat = FakeChat(chat_id)
        self.effective_message = FakeMessage(text)


class FakeContext:
    def __init__(self, bot_data: dict[str, Any], *, error: BaseException | None = None) -> None:
        self.bot_data = bot_data
        self.bot = FakeBot()
        self.error = error
