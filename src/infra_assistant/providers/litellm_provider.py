"""The default provider: one ``litellm.acompletion`` call.

``litellm`` is an optional dependency. It is imported lazily so that
``use_litellm = false`` deployments never need it installed, and so a missing
install produces a clear :class:`ConfigError` at first use rather than an
``ImportError`` from deep in a request.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from ..config import ProviderSpec
from ..errors import ConfigError, ProviderError
from ..llm import LLMResponse, parse_openai_choice, usage_field

logger = logging.getLogger("infra_assistant.providers.litellm")


class LiteLLMProvider:
    def __init__(self, spec: ProviderSpec, *, api_key: str | None, timeout: float) -> None:
        self._spec = spec
        self._api_key = api_key
        self._timeout = timeout
        self.name = f"{spec.provider}:{spec.model}"

    def _litellm(self) -> Any:
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - exercised via a stubbed import in tests
            raise ConfigError(
                "use_litellm is true but the 'litellm' package is not installed — "
                "install infra-assistant[litellm], or set [llm].use_litellm = false"
            ) from exc
        # litellm chatters on stdout/stderr by default; keep a 24/7 service quiet.
        # Idempotent module flag — cheap to re-assert, and litellm is only
        # imported the first time a litellm-backed entry is actually used.
        litellm.suppress_debug_info = True
        return litellm

    async def complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None,
    ) -> LLMResponse:
        litellm = self._litellm()
        acompletion = litellm.acompletion
        kwargs: dict[str, Any] = {
            "model": self._spec.model,
            "messages": list(messages),
            "timeout": self._timeout,
            "drop_params": True,  # silently drop params a given backend rejects
        }
        if tools:
            kwargs["tools"] = list(tools)
            kwargs["tool_choice"] = "auto"
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._spec.base_url:
            kwargs["api_base"] = self._spec.base_url

        try:
            response = await acompletion(**kwargs)
        except Exception as exc:  # litellm raises a wide zoo of provider errors
            # Bound the interpolated text: litellm exception strings can be long
            # and, on some error paths, echo request detail. The chain logs this
            # at WARNING; keep it short and typed.
            detail = str(exc).replace("\n", " ")[:300]
            raise ProviderError(
                f"litellm call to {self.name} failed: {type(exc).__name__}: {detail}"
            ) from exc

        try:
            choice = response["choices"][0]["message"]
            message = choice if isinstance(choice, dict) else choice.model_dump()
        except (KeyError, IndexError, AttributeError, TypeError) as exc:
            raise ProviderError(f"{self.name} returned an unexpected response shape") from exc

        # Usage/cost (Tier B observability) — best-effort, never fatal.
        # A local Ollama model or one litellm has no rate for raises inside
        # completion_cost(); that just means "unpriced", not an error.
        usage = response["usage"] if isinstance(response, dict) and "usage" in response else None
        if usage is None:
            usage = getattr(response, "usage", None)
        cost_usd: float | None
        try:
            cost_usd = float(litellm.completion_cost(completion_response=response))
        except Exception:
            cost_usd = None
        return parse_openai_choice(
            message,
            prompt_tokens=usage_field(usage, "prompt_tokens"),
            completion_tokens=usage_field(usage, "completion_tokens"),
            total_tokens=usage_field(usage, "total_tokens"),
            cost_usd=cost_usd,
        )

    async def aclose(self) -> None:
        return None
