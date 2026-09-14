from __future__ import annotations

import asyncio

import pytest

from infra_assistant.agent import Agent
from infra_assistant.config import LLMConfig, ProviderSpec
from infra_assistant.errors import AllProvidersFailed, TurnTimeout
from infra_assistant.llm import LLMResponse, ToolCall
from infra_assistant.memory import Message
from tests.fakes import FakeCompleter, FakeToolBackend, tool_def


def _cfg(*, max_calls: int = 3, wall: float = 30.0, max_tool_output: int = 40_000) -> LLMConfig:
    return LLMConfig(
        chain=(ProviderSpec("p", "m"),),
        max_tool_calls_per_turn=max_calls,
        turn_wall_time_seconds=wall,
        max_tool_output_chars=max_tool_output,
    )


def _tc(name: str, call_id: str = "c1", **args: object) -> ToolCall:
    return ToolCall(id=call_id, name=name, arguments=dict(args))


async def test_plain_answer_no_tools() -> None:
    chain = FakeCompleter([LLMResponse(text="node-a is up")])
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg())
    assert await agent.answer("is node-a up?") == "node-a is up"
    # system + user only
    assert [m["role"] for m in chain.calls[0][0]] == ["system", "user"]


async def test_turn_metrics_logged(caplog: pytest.LogCaptureFixture) -> None:
    """One structured 'turn complete' line per answered turn.

    Tokens/cost are absent (not zero) when the fake response didn't report
    them; a real provider fills them in (see test_providers.py).
    """
    chain = FakeCompleter([LLMResponse(text="ok", provider_name="deepseek:deepseek-chat")])
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg())
    with caplog.at_level("INFO", logger="infra_assistant.agent"):
        await agent.answer("hi")
    [line] = [r.message for r in caplog.records if r.message.startswith("turn complete:")]
    assert "model_calls=1" in line
    assert "tool_calls=0" in line
    assert "provider=deepseek:deepseek-chat" in line
    assert "latency_ms=" in line
    # not reported by this fake response — must not silently show as 0
    assert "prompt_tokens" not in line
    assert "cost_usd" not in line


async def test_turn_metrics_sum_across_tool_round_trips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    tools = FakeToolBackend([tool_def("homelab__x")], results={"homelab__x": "42"})
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("homelab__x"),), prompt_tokens=100, completion_tokens=10),
            LLMResponse(text="done", prompt_tokens=150, completion_tokens=5),
        ]
    )
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg(), mcp_pool=tools)
    with caplog.at_level("INFO", logger="infra_assistant.agent"):
        result = await agent.answer("do the thing")
    assert result == "done"  # the metrics accumulation must not change behaviour
    [line] = [r.message for r in caplog.records if r.message.startswith("turn complete:")]
    assert "model_calls=2" in line
    assert "tool_calls=1" in line
    assert "prompt_tokens=250" in line  # 100 + 150
    assert "completion_tokens=15" in line  # 10 + 5
    assert "total_tokens=265" in line


async def test_history_is_replayed() -> None:
    chain = FakeCompleter([LLMResponse(text="ok")])
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg())
    await agent.answer("now", [Message("user", "before"), Message("assistant", "earlier")])
    roles = [m["role"] for m in chain.calls[0][0]]
    assert roles == ["system", "user", "assistant", "user"]


async def test_one_tool_call_then_answer() -> None:
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("lab__node_status", host="node-a"),)),
            LLMResponse(text="node-a is active"),
        ]
    )
    backend = FakeToolBackend(
        tools=[tool_def("lab__node_status")],
        results={"lab__node_status": '{"active": true}'},
    )
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg(), mcp_pool=backend)
    answer = await agent.answer("is node-a up?")
    assert answer == "node-a is active"
    assert backend.invocations == [("lab__node_status", {"host": "node-a"})]
    # second model call saw the tool result
    second_call_roles = [m["role"] for m in chain.calls[1][0]]
    assert "tool" in second_call_roles


async def test_tool_call_cap_forces_a_final_answer() -> None:
    # Model keeps asking for tools; cap is 2. After 2 runs, tools are withdrawn
    # and one more tool-request triggers the plain nudge -> final text.
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("lab__t", "a", n=1),)),
            LLMResponse(tool_calls=(_tc("lab__t", "b", n=2),)),
            LLMResponse(tool_calls=(_tc("lab__t", "c", n=3),)),  # asked again, no tools offered
            LLMResponse(text="best effort answer"),
        ]
    )
    backend = FakeToolBackend(tools=[tool_def("lab__t")], results={"lab__t": "ok"})
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg(max_calls=2), mcp_pool=backend)
    answer = await agent.answer("go")
    assert answer == "best effort answer"
    assert len(backend.invocations) == 2  # never exceeded the cap
    assert chain.calls[2][1] is None  # tools withdrawn on the 3rd model call


async def test_wall_time_raises_turn_timeout() -> None:
    ticks = iter([0.0, 5.0, 999.0])
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("lab__t"),)),
            LLMResponse(text="never reached"),
        ]
    )
    backend = FakeToolBackend(tools=[tool_def("lab__t")], results={"lab__t": "ok"})
    agent = Agent(
        system_prompt="sys",
        chain=chain,
        llm_config=_cfg(wall=10.0),
        mcp_pool=backend,
        clock=lambda: next(ticks),
    )
    with pytest.raises(TurnTimeout):
        await agent.answer("go")


async def test_tool_output_budget_trims_later_results() -> None:
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("lab__t", "a"),)),
            LLMResponse(tool_calls=(_tc("lab__t", "b"),)),
            LLMResponse(text="done"),
        ]
    )
    big = "y" * 5_000
    backend = FakeToolBackend(tools=[tool_def("lab__t")], results={"lab__t": big})
    agent = Agent(
        system_prompt="sys",
        chain=chain,
        llm_config=_cfg(max_tool_output=4_000),
        mcp_pool=backend,
    )
    assert await agent.answer("go") == "done"
    # third model call's message list: first tool result full, second trimmed
    tool_msgs = [m for m in chain.calls[2][0] if m["role"] == "tool"]
    assert len(tool_msgs[0]["content"]) == 5_000
    assert "budget for this turn reached" in tool_msgs[1]["content"]
    assert len(tool_msgs[1]["content"]) < 500


async def test_multiple_tool_calls_in_one_turn_run_concurrently_in_order() -> None:
    calls = (_tc("lab__a", "a"), _tc("lab__b", "b"), _tc("lab__c", "c"))
    chain = FakeCompleter([LLMResponse(tool_calls=calls), LLMResponse(text="done")])
    delay = 0.1

    async def _slow(_args: dict[str, object]) -> str:
        await asyncio.sleep(delay)
        return "ok"

    backend = FakeToolBackend(
        tools=[tool_def("lab__a"), tool_def("lab__b"), tool_def("lab__c")],
        results={"lab__a": _slow, "lab__b": _slow, "lab__c": _slow},
    )
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg(), mcp_pool=backend)

    loop = asyncio.get_running_loop()
    started = loop.time()
    assert await agent.answer("go") == "done"
    elapsed = loop.time() - started

    assert [name for name, _ in backend.invocations] == ["lab__a", "lab__b", "lab__c"]
    tool_msgs = [m for m in chain.calls[1][0] if m["role"] == "tool"]
    assert [m["name"] for m in tool_msgs] == ["lab__a", "lab__b", "lab__c"]  # order preserved
    assert elapsed < delay * 3  # ran concurrently, not one after another


async def test_wall_time_bounds_a_hanging_model_call() -> None:
    """The between-calls check can't bound a stuck await; asyncio.timeout does."""
    from collections.abc import Sequence
    from typing import Any

    class Hanger:
        async def complete(
            self,
            messages: Sequence[dict[str, Any]],
            tools: Sequence[dict[str, Any]] | None,
        ) -> LLMResponse:
            await asyncio.sleep(10)
            return LLMResponse(text="never")

    agent = Agent(system_prompt="sys", chain=Hanger(), llm_config=_cfg(wall=0.05))
    with pytest.raises(TurnTimeout):
        await agent.answer("go")


async def test_all_providers_failed_propagates() -> None:
    chain = FakeCompleter([AllProvidersFailed("ConnectionError")])
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg())
    with pytest.raises(AllProvidersFailed):
        await agent.answer("hi")


async def test_empty_model_reply_has_a_placeholder() -> None:
    chain = FakeCompleter([LLMResponse(text=None)])
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg())
    assert "empty response" in await agent.answer("hi")


async def test_no_tools_means_none_offered() -> None:
    chain = FakeCompleter([LLMResponse(text="hi")])
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg())  # no mcp_pool
    await agent.answer("hi")
    assert chain.calls[0][1] is None


async def test_repeated_identical_tool_call_reuses_cached_result() -> None:
    """A weak model re-issuing the same (tool, args) pair must not pay
    for a second real round-trip — the cached result is served back instead,
    with a note telling the model it already has this answer."""
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("lab__node_status", "c1", host="node-a"),)),
            LLMResponse(tool_calls=(_tc("lab__node_status", "c2", host="node-a"),)),  # repeat
            LLMResponse(text="node-a is active"),
        ]
    )
    backend = FakeToolBackend(
        tools=[tool_def("lab__node_status")],
        results={"lab__node_status": '{"active": true}'},
    )
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg(max_calls=5), mcp_pool=backend)
    answer = await agent.answer("is node-a up?")
    assert answer == "node-a is active"
    # only one real call reached the backend, despite two identical requests
    assert backend.invocations == [("lab__node_status", {"host": "node-a"})]
    # the model was told the second one was a repeat, and still got the data
    third_call_messages = chain.calls[2][0]
    tool_msgs = [m for m in third_call_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2
    assert tool_msgs[0]["content"] == '{"active": true}'
    assert "already called this tool" in tool_msgs[1]["content"]
    assert '{"active": true}' in tool_msgs[1]["content"]


async def test_repeated_call_with_different_arguments_is_not_treated_as_cached() -> None:
    chain = FakeCompleter(
        [
            LLMResponse(tool_calls=(_tc("lab__node_status", "c1", host="node-a"),)),
            LLMResponse(tool_calls=(_tc("lab__node_status", "c2", host="node-b"),)),
            LLMResponse(text="both checked"),
        ]
    )
    backend = FakeToolBackend(
        tools=[tool_def("lab__node_status")],
        results={"lab__node_status": "ok"},
    )
    agent = Agent(system_prompt="sys", chain=chain, llm_config=_cfg(max_calls=5), mcp_pool=backend)
    assert await agent.answer("check both") == "both checked"
    assert backend.invocations == [
        ("lab__node_status", {"host": "node-a"}),
        ("lab__node_status", {"host": "node-b"}),
    ]
