"""Exception hierarchy for infra-assistant."""

from __future__ import annotations


class InfraAssistantError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(InfraAssistantError):
    """The configuration file is missing, malformed, or incomplete.

    Raised at startup. A missing bot token, an empty provider chain, or an
    enabled feature whose secret env var is absent is a ``ConfigError`` — the
    bot fails closed rather than starting half-configured.
    """


class ProviderError(InfraAssistantError):
    """One LLM provider failed (transport error, timeout, bad response).

    The provider chain catches this and moves to the next entry. Only when
    *every* entry has failed does it propagate.
    """


class AllProvidersFailed(InfraAssistantError):
    """Every entry in the provider chain failed or was skipped for this turn.

    Carries the class name of the last underlying error so the bot can tell the
    user *why* without leaking a stack trace.
    """

    def __init__(self, last_error: str) -> None:
        super().__init__(f"every LLM provider in the chain failed (last error: {last_error})")
        self.last_error = last_error


class MCPClientError(InfraAssistantError):
    """An MCP server could not be reached, or a tool call against it failed."""


class TurnTimeout(InfraAssistantError):
    """A single turn exceeded its wall-clock budget before the model finished.

    Raised instead of returning a partially-assembled answer — the bot renders
    a clean "couldn't finish in time" message rather than a truncated one.
    """
