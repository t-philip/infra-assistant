"""LLM provider implementations and the factory the chain uses.

Two back ends, same :class:`~infra_assistant.llm.Provider` interface:

* :class:`~infra_assistant.providers.litellm_provider.LiteLLMProvider` — the
  default. One dependency, one call, handles DeepSeek / Ollama / OpenAI /
  Anthropic and dozens more by model-string prefix.
* :class:`~infra_assistant.providers.http_provider.HTTPProvider` — selected by
  ``use_litellm = false`` / ``--no-litellm``. A direct ``httpx`` call to an
  OpenAI-compatible ``/chat/completions`` endpoint (Ollama included, via its
  ``/v1`` shim). Zero extra dependencies.
"""

from __future__ import annotations

from ..config import ProviderSpec
from ..llm import Provider
from .http_provider import HTTPProvider
from .litellm_provider import LiteLLMProvider

__all__ = ["HTTPProvider", "LiteLLMProvider", "build_provider"]


def build_provider(
    spec: ProviderSpec,
    secrets: dict[str, str],
    timeout: float,
    use_litellm: bool,
) -> Provider:
    """Construct the provider for one chain entry."""
    api_key = secrets.get(spec.api_key_env) if spec.api_key_env else None
    if use_litellm:
        return LiteLLMProvider(spec, api_key=api_key, timeout=timeout)
    return HTTPProvider(spec, api_key=api_key, timeout=timeout)
