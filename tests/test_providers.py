from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from infra_assistant.config import ProviderSpec
from infra_assistant.errors import ConfigError, ProviderError
from infra_assistant.providers import build_provider
from infra_assistant.providers.http_provider import HTTPProvider, _resolve_endpoint
from infra_assistant.providers.litellm_provider import LiteLLMProvider

_CHAT: dict[str, Any] = {
    "choices": [{"message": {"content": "node-a is active", "tool_calls": []}}]
}


@pytest.mark.parametrize(
    "spec, expected",
    [
        (ProviderSpec("ollama", "llama3.1:8b"), "http://localhost:11434/v1/chat/completions"),
        (
            ProviderSpec("ollama", "m", base_url="http://ollama.example.invalid:11434"),
            "http://ollama.example.invalid:11434/v1/chat/completions",
        ),
        (ProviderSpec("deepseek", "deepseek-chat"), "https://api.deepseek.com/v1/chat/completions"),
        (
            ProviderSpec("openai-compatible", "m", base_url="https://llm.example.invalid/v1"),
            "https://llm.example.invalid/v1/chat/completions",
        ),
        # an operator who pasted a base_url that already ends in /v1 for a
        # provider whose branch adds its own /v1 — must not double it
        (
            ProviderSpec("deepseek", "m", base_url="https://api.deepseek.com/v1"),
            "https://api.deepseek.com/v1/chat/completions",
        ),
        (
            ProviderSpec("ollama", "m", base_url="http://ollama.example.invalid:11434/v1"),
            "http://ollama.example.invalid:11434/v1/chat/completions",
        ),
    ],
)
def test_endpoint_resolution(spec: ProviderSpec, expected: str) -> None:
    assert _resolve_endpoint(spec) == expected


def test_openai_compatible_needs_base_url() -> None:
    with pytest.raises(ConfigError, match="needs a base_url"):
        _resolve_endpoint(ProviderSpec("openai-compatible", "m"))


def test_deepseek_http_needs_key() -> None:
    with pytest.raises(ConfigError, match="api_key_env"):
        HTTPProvider(ProviderSpec("deepseek", "deepseek-chat"), api_key=None, timeout=5)


@respx.mock
async def test_http_provider_happy_path() -> None:
    route = respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=_CHAT)
    )
    provider = HTTPProvider(ProviderSpec("deepseek", "deepseek-chat"), api_key="k", timeout=5)
    try:
        resp = await provider.complete([{"role": "user", "content": "status?"}], None)
    finally:
        await provider.aclose()
    assert resp.text == "node-a is active"
    assert route.calls.last.request.headers["authorization"] == "Bearer k"
    # no "usage" key in _CHAT — absent, not zero
    assert resp.prompt_tokens is None
    assert resp.cost_usd is None


@respx.mock
async def test_http_provider_reports_usage_when_present() -> None:
    """Tier B observability: token counts flow through when the backend reports them."""
    chat_with_usage = {**_CHAT, "usage": {"prompt_tokens": 120, "completion_tokens": 30}}
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=chat_with_usage)
    )
    provider = HTTPProvider(ProviderSpec("deepseek", "m"), api_key="k", timeout=5)
    try:
        resp = await provider.complete([], None)
    finally:
        await provider.aclose()
    assert resp.prompt_tokens == 120
    assert resp.completion_tokens == 30
    # HTTPProvider has no litellm pricing table to consult — cost stays unknown.
    assert resp.cost_usd is None


@respx.mock
async def test_http_provider_error_body_not_leaked() -> None:
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(500, text="internal detail SECRET-TOKEN")
    )
    provider = HTTPProvider(ProviderSpec("deepseek", "m"), api_key="k", timeout=5)
    try:
        with pytest.raises(ProviderError) as exc:
            await provider.complete([], None)
    finally:
        await provider.aclose()
    assert "SECRET-TOKEN" not in str(exc.value)
    assert "HTTP 500" in str(exc.value)


@respx.mock
async def test_http_provider_rejects_oversized_by_content_length() -> None:
    from infra_assistant.providers.http_provider import _RESPONSE_BYTE_CAP

    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            headers={"content-length": str(_RESPONSE_BYTE_CAP + 1)},
            json=_CHAT,
        )
    )
    provider = HTTPProvider(ProviderSpec("deepseek", "m"), api_key="k", timeout=5)
    try:
        with pytest.raises(ProviderError, match="exceeded"):
            await provider.complete([], None)
    finally:
        await provider.aclose()


@respx.mock
async def test_http_provider_rejects_oversized_while_streaming() -> None:
    from infra_assistant.providers.http_provider import _RESPONSE_BYTE_CAP

    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, content=b"x" * (_RESPONSE_BYTE_CAP + 10))
    )
    provider = HTTPProvider(ProviderSpec("deepseek", "m"), api_key="k", timeout=5)
    try:
        with pytest.raises(ProviderError, match="exceeded"):
            await provider.complete([], None)
    finally:
        await provider.aclose()


@respx.mock
async def test_http_provider_bad_shape() -> None:
    respx.post("https://api.deepseek.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"nope": True})
    )
    provider = HTTPProvider(ProviderSpec("deepseek", "m"), api_key="k", timeout=5)
    try:
        with pytest.raises(ProviderError, match="unexpected response shape"):
            await provider.complete([], None)
    finally:
        await provider.aclose()


async def test_litellm_provider_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> dict[str, Any]:
        captured.update(kwargs)
        return _CHAT

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    provider = LiteLLMProvider(
        ProviderSpec("deepseek", "deepseek/deepseek-chat"), api_key="k", timeout=7
    )
    resp = await provider.complete(
        [{"role": "user", "content": "hi"}],
        [{"type": "function", "function": {"name": "t", "parameters": {}}}],
    )
    assert resp.text == "node-a is active"
    assert captured["model"] == "deepseek/deepseek-chat"
    assert captured["api_key"] == "k"
    assert captured["tools"]  # tools were forwarded


async def test_litellm_provider_reports_usage_and_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tier B observability: usage + litellm's own cost estimate flow through."""
    import litellm

    chat_with_usage = {**_CHAT, "usage": {"prompt_tokens": 200, "completion_tokens": 50}}

    async def fake_acompletion(**_: object) -> dict[str, Any]:
        return chat_with_usage

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", lambda completion_response: 0.001234)
    provider = LiteLLMProvider(
        ProviderSpec("deepseek", "deepseek/deepseek-chat"), api_key="k", timeout=5
    )
    resp = await provider.complete([{"role": "user", "content": "hi"}], None)
    assert resp.prompt_tokens == 200
    assert resp.completion_tokens == 50
    assert resp.cost_usd == pytest.approx(0.001234)


async def test_litellm_provider_cost_lookup_failure_is_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unpriced model (e.g. local Ollama) makes completion_cost raise — cost stays None."""
    import litellm

    async def fake_acompletion(**_: object) -> dict[str, Any]:
        return _CHAT

    def boom(*, completion_response: object) -> float:
        raise Exception("model not in litellm's pricing table")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)
    monkeypatch.setattr(litellm, "completion_cost", boom)
    provider = LiteLLMProvider(ProviderSpec("ollama", "llama3.1:8b"), api_key=None, timeout=5)
    resp = await provider.complete([], None)
    assert resp.text == "node-a is active"
    assert resp.cost_usd is None


async def test_litellm_provider_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    async def boom(**_: object) -> dict[str, Any]:
        raise RuntimeError("rate limited")

    monkeypatch.setattr(litellm, "acompletion", boom)
    provider = LiteLLMProvider(ProviderSpec("deepseek", "m"), api_key="k", timeout=5)
    with pytest.raises(ProviderError, match="litellm call"):
        await provider.complete([], None)


def test_build_provider_selects_backend() -> None:
    spec = ProviderSpec("deepseek", "deepseek/deepseek-chat", api_key_env="DK")
    secrets = {"DK": "value"}
    assert isinstance(build_provider(spec, secrets, 5.0, True), LiteLLMProvider)
    assert isinstance(build_provider(spec, secrets, 5.0, False), HTTPProvider)
