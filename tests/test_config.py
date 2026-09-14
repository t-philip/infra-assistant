from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from infra_assistant.config import TimeWindow, load_config, load_config_text
from infra_assistant.errors import ConfigError

_FULL = """
[telegram]
bot_token_env = "BOT_TOKEN"
owner_allow_from = [111, 222]

[llm]
use_litellm = false
per_call_timeout_seconds = 20
max_tool_calls_per_turn = 5
turn_wall_time_seconds = 45

[[llm.chain]]
provider = "deepseek"
model = "deepseek/deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"
skip_between = ["01:00-04:00Z"]

[[llm.chain]]
provider = "ollama"
model = "ollama/llama3.1:8b"
base_url = "http://ollama.example.invalid:11434"

[[mcp.servers]]
name = "homelab"
transport = "stdio"
command = "homelab-mcp"
args = ["run", "-c", "/etc/homelab-mcp.toml"]
forward_env = ["NETBOX_API_TOKEN", "ADGUARD_USER"]

[[mcp.servers]]
name = "metrics"
transport = "streamable-http"
url = "http://mcp.example.invalid:9000/mcp"
token_env = "METRICS_MCP_TOKEN"

[memory]
history_turns = 4
log_dir = "/tmp/transcripts"
"""


def test_full_config_parses() -> None:
    cfg = load_config_text(_FULL)
    assert cfg.telegram.owner_allow_from == frozenset({111, 222})
    assert cfg.llm.use_litellm is False
    assert cfg.llm.max_tool_calls_per_turn == 5
    # new knobs default when absent
    assert cfg.llm.tool_call_timeout_seconds == 30.0
    assert cfg.llm.max_tool_output_chars == 40_000
    assert [p.provider for p in cfg.llm.chain] == ["deepseek", "ollama"]
    assert cfg.llm.chain[0].skip_windows[0] == TimeWindow(
        TimeWindow._parse_hhmm("01:00"), TimeWindow._parse_hhmm("04:00")
    )
    assert [s.name for s in cfg.mcp_servers] == ["homelab", "metrics"]
    assert cfg.mcp_servers[0].command == "homelab-mcp"
    assert cfg.mcp_servers[0].forward_env == ("NETBOX_API_TOKEN", "ADGUARD_USER")
    assert cfg.mcp_servers[1].forward_env == ()  # defaults empty
    assert cfg.mcp_servers[1].url and cfg.mcp_servers[1].token_env == "METRICS_MCP_TOKEN"
    assert cfg.memory.history_turns == 4 and cfg.memory.logging_enabled


def test_missing_telegram_table() -> None:
    with pytest.raises(ConfigError, match=r"\[telegram\]"):
        load_config_text('[llm]\n[[llm.chain]]\nprovider="x"\nmodel="y"\n')


def test_missing_llm_table() -> None:
    with pytest.raises(ConfigError, match=r"\[llm\]"):
        load_config_text('[telegram]\nbot_token_env="B"\nowner_allow_from=[1]\n')


def test_empty_chain_rejected() -> None:
    with pytest.raises(ConfigError, match="at least one"):
        load_config_text('[telegram]\nbot_token_env="B"\nowner_allow_from=[1]\n[llm]\nchain=[]\n')


def test_owner_allow_from_must_be_ints() -> None:
    with pytest.raises(ConfigError, match="integer chat id"):
        load_config_text(
            '[telegram]\nbot_token_env="B"\nowner_allow_from=["me"]\n'
            '[[llm.chain]]\nprovider="x"\nmodel="y"\n'
        )


def test_unknown_top_level_key() -> None:
    with pytest.raises(ConfigError, match="unknown top-level"):
        load_config_text(_FULL + "\n[nonsense]\nx = 1\n")


def test_llm_new_knobs_parse_and_validate() -> None:
    text = (
        '[telegram]\nbot_token_env="B"\nowner_allow_from=[1]\n'
        "[llm]\ntool_call_timeout_seconds = 12\nmax_tool_output_chars = 5000\n"
        '[[llm.chain]]\nprovider="x"\nmodel="y"\n'
    )
    cfg = load_config_text(text)
    assert cfg.llm.tool_call_timeout_seconds == 12.0
    assert cfg.llm.max_tool_output_chars == 5000


@pytest.mark.parametrize(
    "line, message",
    [
        ("tool_call_timeout_seconds = 0", "tool_call_timeout_seconds"),
        ('max_tool_output_chars = "big"', "max_tool_output_chars"),
        ("max_tool_output_chars = 10", "max_tool_output_chars"),
        ("bogus_key = 1", "unknown key"),
    ],
)
def test_llm_knob_rejections(line: str, message: str) -> None:
    text = (
        '[telegram]\nbot_token_env="B"\nowner_allow_from=[1]\n'
        f"[llm]\n{line}\n"
        '[[llm.chain]]\nprovider="x"\nmodel="y"\n'
    )
    with pytest.raises(ConfigError, match=message):
        load_config_text(text)


@pytest.mark.parametrize(
    "server_block, message",
    [
        ('name="a"\ntransport="stdio"\n', "requires 'command'"),
        ('name="a"\ntransport="stdio"\ncommand="c"\nurl="u"\n', "must not set 'url'"),
        ('name="a"\ntransport="streamable-http"\n', "requires 'url'"),
        ('name="Bad_Name"\ntransport="stdio"\ncommand="c"\n', "must be lowercase"),
        (
            'name="a"\ntransport="stdio"\ncommand="c"\nforward_env=["ok", "no-dashes"]\n',
            "forward_env must be a list of environment-variable names",
        ),
        (
            'name="a"\ntransport="stdio"\ncommand="c"\nforward_env="NOT_A_LIST"\n',
            "forward_env must be a list",
        ),
    ],
)
def test_bad_mcp_server(server_block: str, message: str) -> None:
    text = (
        '[telegram]\nbot_token_env="B"\nowner_allow_from=[1]\n'
        '[[llm.chain]]\nprovider="x"\nmodel="y"\n'
        f"[[mcp.servers]]\n{server_block}"
    )
    with pytest.raises(ConfigError, match=message):
        load_config_text(text)


def test_duplicate_mcp_server_name() -> None:
    text = (
        '[telegram]\nbot_token_env="B"\nowner_allow_from=[1]\n'
        '[[llm.chain]]\nprovider="x"\nmodel="y"\n'
        '[[mcp.servers]]\nname="dup"\ntransport="stdio"\ncommand="a"\n'
        '[[mcp.servers]]\nname="dup"\ntransport="stdio"\ncommand="b"\n'
    )
    with pytest.raises(ConfigError, match="used more than once"):
        load_config_text(text)


def test_resolve_secrets_reports_every_gap_at_once() -> None:
    cfg = load_config_text(_FULL)
    with pytest.raises(ConfigError) as exc:
        cfg.resolve_secrets({})
    text = str(exc.value)
    assert "BOT_TOKEN" in text
    assert "DEEPSEEK_API_KEY" in text
    assert "METRICS_MCP_TOKEN" in text


def test_resolve_secrets_ok() -> None:
    cfg = load_config_text(_FULL)
    resolved = cfg.resolve_secrets(
        {"BOT_TOKEN": "t", "DEEPSEEK_API_KEY": "k", "METRICS_MCP_TOKEN": "m"}
    )
    assert resolved == {"BOT_TOKEN": "t", "DEEPSEEK_API_KEY": "k", "METRICS_MCP_TOKEN": "m"}


def test_example_toml_parses_and_lists_its_secrets() -> None:
    example = Path(__file__).resolve().parent.parent / "infra-assistant.example.toml"
    cfg = load_config_text(example.read_text(encoding="utf-8"))
    assert cfg.llm.use_litellm is True
    with pytest.raises(ConfigError) as exc:
        cfg.resolve_secrets({})
    assert "TELEGRAM_BOT_TOKEN" in str(exc.value)
    assert "DEEPSEEK_API_KEY" in str(exc.value)


def test_load_config_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


# -- TimeWindow ------------------------------------------------------------


def _utc(h: int, m: int = 0) -> datetime:
    return datetime(1999, 6, 1, h, m, tzinfo=UTC)


def test_time_window_simple() -> None:
    w = TimeWindow.parse("01:00-04:00Z")
    assert w.contains(_utc(2)) and not w.contains(_utc(5)) and not w.contains(_utc(0))


def test_time_window_wraps_midnight() -> None:
    w = TimeWindow.parse("22:00-02:00")
    assert w.contains(_utc(23)) and w.contains(_utc(1)) and not w.contains(_utc(12))


@pytest.mark.parametrize("bad", ["nonsense", "25:00-01:00", "01:00", "01:00-01:00"])
def test_time_window_rejects_garbage(bad: str) -> None:
    with pytest.raises(ConfigError):
        TimeWindow.parse(bad)


def test_provider_skipped_at() -> None:
    cfg = load_config_text(_FULL)
    deepseek = cfg.llm.chain[0]
    assert deepseek.skipped_at(_utc(2))
    assert not deepseek.skipped_at(_utc(9))
