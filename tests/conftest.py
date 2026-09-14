"""Shared fixtures.

All test data is fictitious: chat ids ``111`` / ``222``, host names ``node-a``,
addresses in 192.0.2.0/24, free text ``"Test Data ..."``, and any absolute date
before 2002.
"""

from __future__ import annotations

import pytest

from infra_assistant.config import LLMConfig, ProviderSpec

_MINIMAL_CONFIG = """
[telegram]
bot_token_env = "BOT_TOKEN"
owner_allow_from = [111]

[[llm.chain]]
provider = "primary"
model = "test/model-a"
api_key_env = "PRIMARY_KEY"
"""


@pytest.fixture
def minimal_config_text() -> str:
    return _MINIMAL_CONFIG


@pytest.fixture
def llm_config() -> LLMConfig:
    return LLMConfig(
        chain=(
            ProviderSpec(provider="primary", model="test/a"),
            ProviderSpec(provider="fallback", model="test/b"),
        ),
        max_tool_calls_per_turn=3,
        turn_wall_time_seconds=30.0,
    )
