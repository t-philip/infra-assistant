from __future__ import annotations

from typing import Any

import pytest

from infra_assistant.agent import Agent
from infra_assistant.config import LLMConfig, ProviderSpec
from infra_assistant.errors import AllProvidersFailed, TurnTimeout
from infra_assistant.llm import LLMResponse
from infra_assistant.memory import ChatMemory
from infra_assistant.telegram_bot import (
    _split_for_telegram,
    _utf16_len,
    on_error,
    on_ignored,
    on_message,
    on_reset,
)
from tests.fakes import FakeCompleter, FakeContext, FakeUpdate

ALLOWED = 111
BLOCKED = 999


def _bot_data(chain: FakeCompleter, *, history_turns: int = 4) -> dict[str, Any]:
    cfg = LLMConfig(chain=(ProviderSpec("p", "m"),), turn_wall_time_seconds=15.0)
    memory = ChatMemory(history_turns)
    agent = Agent(system_prompt="sys", chain=chain, llm_config=cfg)
    return {
        "agent": agent,
        "memory": memory,
        "allow_from": frozenset({ALLOWED}),
        "turn_wall_time": cfg.turn_wall_time_seconds,
    }


async def test_non_allowlisted_chat_is_ignored() -> None:
    chain = FakeCompleter([LLMResponse(text="should not run")])
    data = _bot_data(chain)
    update: Any = FakeUpdate(BLOCKED, "hello?")
    context: Any = FakeContext(data)
    await on_message(update, context)
    assert update.effective_message.replies == []
    assert not chain.calls
    assert data["memory"].history(BLOCKED) == []


async def test_happy_path_replies_and_records() -> None:
    chain = FakeCompleter([LLMResponse(text="node-a is active")])
    data = _bot_data(chain)
    update: Any = FakeUpdate(ALLOWED, "is node-a up?")
    context: Any = FakeContext(data)
    await on_message(update, context)
    assert update.effective_message.replies == ["node-a is active"]
    assert [m.content for m in data["memory"].history(ALLOWED)] == [
        "is node-a up?",
        "node-a is active",
    ]
    assert context.bot.chat_actions == [(ALLOWED, "typing")]


async def test_all_providers_failed_is_a_clean_message_not_recorded() -> None:
    chain = FakeCompleter([AllProvidersFailed("ConnectionError")])
    data = _bot_data(chain)
    update: Any = FakeUpdate(ALLOWED, "status?")
    context: Any = FakeContext(data)
    await on_message(update, context)
    assert len(update.effective_message.replies) == 1
    assert update.effective_message.replies[0].startswith("⚠️")
    assert "ConnectionError" in update.effective_message.replies[0]
    assert data["memory"].history(ALLOWED) == []  # a failure is not remembered


async def test_turn_timeout_is_a_clean_message() -> None:
    chain = FakeCompleter([TurnTimeout("too slow")])
    data = _bot_data(chain)
    update: Any = FakeUpdate(ALLOWED, "big question")
    context: Any = FakeContext(data)
    await on_message(update, context)
    assert update.effective_message.replies[0].startswith("⚠️")
    assert "15s" in update.effective_message.replies[0]


async def test_oversized_reply_is_chunked() -> None:
    big = "\n".join(f"line {i} " + "x" * 80 for i in range(400))  # well over 4096
    chain = FakeCompleter([LLMResponse(text=big)])
    data = _bot_data(chain)
    update: Any = FakeUpdate(ALLOWED, "dump it")
    context: Any = FakeContext(data)
    await on_message(update, context)
    replies = update.effective_message.replies
    assert len(replies) > 1
    assert all(len(chunk) <= 4096 for chunk in replies)
    assert "".join(r.replace("\n", "") for r in replies).count("line") == 400


async def test_reset_clears_history() -> None:
    chain = FakeCompleter([LLMResponse(text="hi")])
    data = _bot_data(chain)
    data["memory"].record(ALLOWED, "old", "older")
    update: Any = FakeUpdate(ALLOWED, "/reset")
    context: Any = FakeContext(data)
    await on_reset(update, context)
    assert data["memory"].history(ALLOWED) == []
    assert update.effective_message.replies == ["History cleared for this chat."]


async def test_on_ignored_is_quiet() -> None:
    update: Any = FakeUpdate(BLOCKED, "hi")
    context: Any = FakeContext({"allow_from": frozenset({ALLOWED})})
    await on_ignored(update, context)  # must not raise
    assert update.effective_message.replies == []


@pytest.mark.parametrize(
    "length, expected_chunks",
    [(10, 1), (4096, 1), (4097, 2), (9000, 3)],
)
def test_split_for_telegram(length: int, expected_chunks: int) -> None:
    chunks = _split_for_telegram("a" * length)
    assert len(chunks) == expected_chunks
    assert all(len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == "a" * length


def test_split_counts_utf16_units_not_code_points() -> None:
    # 2500 emoji = 2500 code points but 5000 UTF-16 units -> must be split
    text = "\U0001f7e2" * 2500
    assert len(text) < 4096  # under the limit by code points...
    chunks = _split_for_telegram(text)
    assert len(chunks) > 1  # ...but split, because Telegram counts units
    assert all(_utf16_len(c) <= 4096 for c in chunks)
    assert "".join(chunks) == text


async def test_on_error_alerts_the_owner() -> None:
    context: Any = FakeContext(
        {"allow_from": frozenset({ALLOWED, 222})}, error=RuntimeError("boom")
    )
    await on_error(FakeUpdate(ALLOWED, "x"), context)
    assert context.bot.messages and context.bot.messages[0][0] == ALLOWED
    assert "internal error" in context.bot.messages[0][1]


async def test_on_error_no_owner_is_quiet() -> None:
    context: Any = FakeContext({"allow_from": frozenset()}, error=RuntimeError("boom"))
    await on_error(FakeUpdate(ALLOWED, "x"), context)
    assert context.bot.messages == []
