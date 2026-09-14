from __future__ import annotations

from datetime import UTC, datetime

import pytest

from infra_assistant.config import LLMConfig, ProviderSpec, TimeWindow
from infra_assistant.errors import AllProvidersFailed, ConfigError, ProviderError
from infra_assistant.llm import (
    LLMResponse,
    ProviderChain,
    parse_openai_choice,
    parse_tool_arguments,
    tool_defs_to_openai,
)
from tests.fakes import FakeCompleter


def _chain(config: LLMConfig, factory) -> ProviderChain:  # type: ignore[no-untyped-def]
    return ProviderChain(
        config, {}, provider_factory=factory, now_fn=lambda: datetime(1999, 6, 1, 9, tzinfo=UTC)
    )


def _two_providers(**kw: object) -> LLMConfig:
    return LLMConfig(
        chain=(
            ProviderSpec(provider="primary", model="a", **kw),  # type: ignore[arg-type]
            ProviderSpec(provider="fallback", model="b"),
        )
    )


async def test_first_provider_wins() -> None:
    primary = FakeCompleter([LLMResponse(text="from primary")])
    secondary = FakeCompleter([LLMResponse(text="unused")])
    chain = _chain(
        _two_providers(), lambda spec, *_: primary if spec.provider == "primary" else secondary
    )
    result = await chain.complete([{"role": "user", "content": "hi"}], None)
    assert result.text == "from primary"
    assert not secondary.calls


async def test_failover_to_second() -> None:
    primary = FakeCompleter([ProviderError("boom")])
    secondary = FakeCompleter([LLMResponse(text="from fallback")])
    chain = _chain(
        _two_providers(), lambda spec, *_: primary if spec.provider == "primary" else secondary
    )
    result = await chain.complete([], None)
    assert result.text == "from fallback"


async def test_chain_stamps_provider_name() -> None:
    """The chain, not the provider, names which entry answered.

    A bare Completer test double (no `.name` attribute) must still work; the
    name comes from the configured ProviderSpec, not the provider instance.
    """
    secondary = FakeCompleter([LLMResponse(text="from fallback")])
    chain = _chain(
        _two_providers(),
        lambda spec, *_: (
            FakeCompleter([ProviderError("boom")]) if spec.provider == "primary" else secondary
        ),
    )
    result = await chain.complete([], None)
    assert result.provider_name == "fallback:b"


def _wrap(cause: Exception) -> ProviderError:
    """A ProviderError with an underlying cause, as a real provider raises."""
    err = ProviderError(str(cause))
    err.__cause__ = cause
    return err


async def test_provider_construction_error_does_not_kill_the_chain() -> None:
    """A ConfigError from building an earlier entry is a per-entry failure, not fatal.

    Regression: `_provider_for` used to run outside the try, so one misconfigured
    entry disabled every entry after it.
    """
    good = FakeCompleter([LLMResponse(text="fallback answered")])

    def factory(spec: ProviderSpec, *_: object) -> FakeCompleter:
        if spec.provider == "primary":
            raise ConfigError("deepseek entry needs api_key_env")
        return good

    chain = _chain(_two_providers(), factory)
    result = await chain.complete([], None)
    assert result.text == "fallback answered"


async def test_unexpected_provider_exception_fails_over() -> None:
    primary = FakeCompleter([RuntimeError("kaboom")])  # not a ProviderError
    secondary = FakeCompleter([LLMResponse(text="ok")])
    chain = _chain(
        _two_providers(), lambda spec, *_: primary if spec.provider == "primary" else secondary
    )
    result = await chain.complete([], None)
    assert result.text == "ok"


async def test_all_fail_raises_with_last_error_class() -> None:
    class ConnectionReset(Exception):
        pass

    primary = FakeCompleter([ProviderError("first failed")])
    secondary = FakeCompleter([_wrap(ConnectionReset("dropped"))])
    chain = _chain(
        _two_providers(),
        lambda spec, *_: primary if spec.provider == "primary" else secondary,
    )
    with pytest.raises(AllProvidersFailed) as exc:
        await chain.complete([], None)
    assert exc.value.last_error == "ConnectionReset"


async def test_peak_window_skips_entry() -> None:
    cfg = _two_providers(skip_windows=(TimeWindow.parse("08:00-10:00Z"),))
    primary = FakeCompleter([LLMResponse(text="should be skipped")])
    secondary = FakeCompleter([LLMResponse(text="took over")])
    chain = _chain(cfg, lambda spec, *_: primary if spec.provider == "primary" else secondary)
    result = await chain.complete([], None)
    assert result.text == "took over"
    assert not primary.calls


async def test_all_skipped_raises_peak_window() -> None:
    cfg = LLMConfig(
        chain=(ProviderSpec("primary", "a", skip_windows=(TimeWindow.parse("00:00-23:59Z"),)),)
    )
    chain = _chain(cfg, lambda *_: FakeCompleter([LLMResponse(text="x")]))
    with pytest.raises(AllProvidersFailed, match="peak window"):
        await chain.complete([], None)


async def test_provider_instance_is_cached() -> None:
    built: list[str] = []

    def factory(spec: ProviderSpec, *_: object) -> FakeCompleter:
        built.append(spec.provider)
        return FakeCompleter([LLMResponse(text="1"), LLMResponse(text="2")])

    chain = _chain(_two_providers(), factory)
    await chain.complete([], None)
    await chain.complete([], None)
    assert built == ["primary"]  # only the first entry, built once


def test_parse_tool_arguments() -> None:
    assert parse_tool_arguments('{"a": 1}') == {"a": 1}
    assert parse_tool_arguments("") == {}
    assert parse_tool_arguments({"b": 2}) == {"b": 2}
    with pytest.raises(ProviderError):
        parse_tool_arguments("not json")
    with pytest.raises(ProviderError):
        parse_tool_arguments("[1, 2]")


def test_parse_openai_choice_text_and_tools() -> None:
    msg = {
        "content": None,
        "tool_calls": [{"id": "c1", "function": {"name": "homelab__x", "arguments": '{"n": 3}'}}],
    }
    resp = parse_openai_choice(msg)
    assert resp.wants_tools
    assert resp.tool_calls[0].name == "homelab__x"
    assert resp.tool_calls[0].arguments == {"n": 3}

    plain = parse_openai_choice({"content": "hello"})
    assert plain.text == "hello" and not plain.wants_tools


def test_tool_defs_to_openai() -> None:
    wrapped = tool_defs_to_openai(
        [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    )
    assert wrapped[0]["type"] == "function"
    assert wrapped[0]["function"]["name"] == "t"
    assert wrapped[0]["function"]["parameters"] == {"type": "object"}
