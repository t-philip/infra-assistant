"""The zero-extra-dependency provider: a direct ``httpx`` POST.

Speaks the OpenAI ``/chat/completions`` schema, which DeepSeek, most local
servers, and Ollama (via its ``/v1`` shim) all accept. Selected with
``use_litellm = false`` for a deployment that wants nothing beyond ``httpx``.

Endpoint resolution:

* ``provider = "ollama"`` -> ``{base_url}/v1/chat/completions``, no auth needed;
  ``base_url`` defaults to ``http://localhost:11434``.
* ``provider = "deepseek"`` -> ``{base_url or https://api.deepseek.com}/v1/chat/completions``,
  ``api_key_env`` required.
* anything else (``"openai-compatible"``, ``"openai"``, ...) ->
  ``{base_url}/chat/completions``; ``base_url`` is required and should already
  include any ``/v1`` the server expects.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import Any

import httpx

from ..config import ProviderSpec
from ..errors import ConfigError, ProviderError
from ..llm import LLMResponse, parse_openai_choice, usage_field

logger = logging.getLogger("infra_assistant.providers.http")

_RESPONSE_BYTE_CAP = 8 * 1024 * 1024


def _resolve_endpoint(spec: ProviderSpec) -> str:
    base = (spec.base_url or "").rstrip("/")
    provider = spec.provider.lower()
    if provider == "ollama":
        # tolerate an operator who pasted a base_url that already ends in /v1
        return f"{base.removesuffix('/v1') or 'http://localhost:11434'}/v1/chat/completions"
    if provider == "deepseek":
        return f"{base.removesuffix('/v1') or 'https://api.deepseek.com'}/v1/chat/completions"
    if not base:
        raise ConfigError(
            f"[[llm.chain]] provider {spec.provider!r} needs a base_url when use_litellm = false"
        )
    return f"{base}/chat/completions"


class HTTPProvider:
    def __init__(self, spec: ProviderSpec, *, api_key: str | None, timeout: float) -> None:
        self._spec = spec
        self._api_key = api_key
        self._timeout = timeout
        self._endpoint = _resolve_endpoint(spec)
        self._client: httpx.AsyncClient | None = None
        self.name = f"{spec.provider}:{spec.model}"
        if spec.provider.lower() == "deepseek" and not api_key:
            raise ConfigError(
                "[[llm.chain]] deepseek entry needs api_key_env when use_litellm = false"
            )

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
    ) -> LLMResponse:
        payload: dict[str, Any] = {"model": self._spec.model, "messages": list(messages)}
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        try:
            raw = await self._post_capped(payload, headers)
        except httpx.HTTPError as exc:
            raise ProviderError(f"HTTP call to {self.name} failed: {exc}") from exc

        try:
            body = json.loads(raw)
            message = body["choices"][0]["message"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"{self.name} returned an unexpected response shape") from exc
        if not isinstance(message, dict):
            raise ProviderError(f"{self.name} returned a non-object message")
        # Usage (optional usage and cost observability) — best-effort; no cost figure
        # here, since that needs litellm's pricing table (LiteLLMProvider only).
        usage = body.get("usage") if isinstance(body, dict) else None
        return parse_openai_choice(
            message,
            prompt_tokens=usage_field(usage, "prompt_tokens"),
            completion_tokens=usage_field(usage, "completion_tokens"),
            total_tokens=usage_field(usage, "total_tokens"),
        )

    async def _post_capped(self, payload: dict[str, Any], headers: dict[str, str]) -> bytes:
        """POST and stream the body, refusing to buffer more than the cap.

        A non-streaming ``.post()`` reads the whole body into memory before
        returning, so an oversized (or hostile) response would OOM the process
        before any size check. Streaming lets us stop reading past the limit.
        """
        async with self._http().stream(
            "POST", self._endpoint, json=payload, headers=headers
        ) as response:
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > _RESPONSE_BYTE_CAP:
                raise ProviderError(f"{self.name} response exceeded {_RESPONSE_BYTE_CAP} bytes")
            chunks = bytearray()
            async for chunk in response.aiter_bytes():
                chunks += chunk
                if len(chunks) > _RESPONSE_BYTE_CAP:
                    raise ProviderError(f"{self.name} response exceeded {_RESPONSE_BYTE_CAP} bytes")
            if response.status_code >= 400:
                # Log the body for diagnosis; never fold it into the raised
                # message, which can surface to the user.
                logger.warning(
                    "%s returned HTTP %d: %s",
                    self.name,
                    response.status_code,
                    bytes(chunks[:2000]).decode(errors="replace"),
                )
                raise ProviderError(f"{self.name} returned HTTP {response.status_code}")
            return bytes(chunks)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
