"""Configuration loading and validation.

The config file is TOML. Secrets are never stored in it: a field whose name
ends in ``_env`` holds the *name* of an environment variable, and the value is
read from the process environment at startup. A missing bot token, an empty
provider chain, or any ``*_env`` variable that is named but absent is a fatal
:class:`ConfigError` — the bot fails closed.
"""

from __future__ import annotations

import os
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Any

from .credentials import default_credential_source
from .errors import ConfigError
from .secret_redaction import SecretStr

CONFIG_ENV_VAR = "INFRA_ASSISTANT_CONFIG"
VALID_MCP_TRANSPORTS = ("stdio", "sse", "streamable-http")
_ALLOWED_TOP_LEVEL = frozenset({"telegram", "llm", "mcp", "memory"})
# A server name is used as a `<name>__<tool>` prefix, so it must keep the
# combined name inside the usual function-name rule (^[A-Za-z0-9_-]+$).
_MCP_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class TimeWindow:
    """A daily UTC time window, e.g. ``01:00-04:00``. May wrap past midnight."""

    start: time
    end: time

    def contains(self, moment: datetime) -> bool:
        """Is *moment* (any tz; compared in UTC) inside this daily window?"""
        t = moment.astimezone(UTC).timetz().replace(tzinfo=None)
        if self.start <= self.end:
            return self.start <= t < self.end
        # Wraps midnight: e.g. 22:00-02:00 is "after 22:00 OR before 02:00".
        return t >= self.start or t < self.end

    @classmethod
    def parse(cls, spec: str) -> TimeWindow:
        """Parse ``"HH:MM-HH:MM"`` with an optional trailing ``Z`` (UTC, the only tz).

        v1 only understands UTC windows — a provider peak window is quoted
        in UTC in every provider's docs, and a tz-aware parser is a maintenance
        cost with no caller. A trailing ``Z`` is accepted and ignored; anything
        else after the range is rejected.
        """
        raw = spec.strip()
        if raw.endswith(("Z", "z")):
            raw = raw[:-1]
        try:
            lhs, rhs = raw.split("-", 1)
            start = cls._parse_hhmm(lhs.strip())
            end = cls._parse_hhmm(rhs.strip())
        except ValueError as exc:
            raise ConfigError(
                f"skip_between entry {spec!r} is not a 'HH:MM-HH:MM' UTC range"
            ) from exc
        if start == end:
            raise ConfigError(f"skip_between entry {spec!r} is a zero-length window")
        return cls(start=start, end=end)

    @staticmethod
    def _parse_hhmm(value: str) -> time:
        hh_str, mm_str = value.split(":", 1)
        hh, mm = int(hh_str), int(mm_str)
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            raise ValueError(value)
        return time(hh, mm)


@dataclass(frozen=True)
class ProviderSpec:
    """One entry in the ordered fallback chain."""

    provider: str
    model: str
    api_key_env: str | None = None
    base_url: str | None = None
    skip_windows: tuple[TimeWindow, ...] = ()

    def skipped_at(self, moment: datetime) -> bool:
        return any(window.contains(moment) for window in self.skip_windows)


@dataclass(frozen=True)
class LLMConfig:
    chain: tuple[ProviderSpec, ...]
    use_litellm: bool = True
    per_call_timeout_seconds: float = 30.0
    max_tool_calls_per_turn: int = 12
    turn_wall_time_seconds: float = 60.0
    #: Hard ceiling on a single tool call; a wedged tool becomes a tool error.
    tool_call_timeout_seconds: float = 30.0
    #: Cumulative cap on tool output folded back into one turn's context — keeps
    #: a chatty turn from overflowing a small local fallback model's window.
    max_tool_output_chars: int = 40_000


@dataclass(frozen=True)
class MCPServerSpec:
    name: str
    transport: str
    command: str | None = None
    args: tuple[str, ...] = ()
    env: Mapping[str, str] = field(default_factory=dict)
    url: str | None = None
    token_env: str | None = None
    #: For a ``stdio`` server: names of secrets to hand the child. Each is
    #: resolved from this process's credential source and written as a 0600 file
    #: under a private directory the child sees as ``$CREDENTIALS_DIRECTORY`` —
    #: never as an environment value, so it stays out of the child's
    #: ``/proc/<pid>/environ``. The child otherwise gets only a minimal base
    #: environment plus ``env``, never the bot's whole ``os.environ`` (bot token,
    #: LLM keys, …). List only the vars the server's own config resolves.
    forward_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class TelegramConfig:
    bot_token_env: str
    owner_allow_from: frozenset[int]


@dataclass(frozen=True)
class MemoryConfig:
    history_turns: int = 8
    #: Directory for per-chat append-only transcripts. Empty string disables it.
    log_dir: str = ""

    @property
    def logging_enabled(self) -> bool:
        return bool(self.log_dir)


@dataclass(frozen=True)
class Config:
    telegram: TelegramConfig
    llm: LLMConfig
    mcp_servers: tuple[MCPServerSpec, ...]
    memory: MemoryConfig
    source: Path | None = None

    def secret_env_names(self) -> dict[str, str]:
        """Map a human field path -> env var name for every ``*_env`` in play."""
        names: dict[str, str] = {"telegram.bot_token_env": self.telegram.bot_token_env}
        for i, entry in enumerate(self.llm.chain):
            if entry.api_key_env:
                names[f"llm.chain[{i}].api_key_env"] = entry.api_key_env
        for server in self.mcp_servers:
            if server.token_env:
                names[f"mcp.servers.{server.name}.token_env"] = server.token_env
        return names

    def resolve_secrets(self, environ: Mapping[str, str] | None = None) -> dict[str, str]:
        """Resolve every ``*_env`` field against *environ*.

        Returns ``{env_var_name: value}``. Raises :class:`ConfigError` listing
        every missing variable at once, so the operator sees the whole gap.

        With no explicit *environ*, the process-wide credential source is used:
        a value comes from ``$CREDENTIALS_DIRECTORY/<name>`` (or another creds
        directory) before ``os.environ``, so a secret need never sit in the
        process environment.
        """
        env: Mapping[str, str] = default_credential_source() if environ is None else environ
        resolved: dict[str, str] = {}
        missing: list[str] = []
        for field_path, var_name in self.secret_env_names().items():
            value = env.get(var_name, "")
            if value == "":
                missing.append(f"{field_path} -> ${var_name}")
            else:
                resolved[var_name] = SecretStr(value)
        if missing:
            raise ConfigError(
                "missing secret environment variable(s): " + ", ".join(sorted(missing))
            )
        return resolved


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _require_table(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{label} must be a table")
    return raw


def _str(raw: dict[str, Any], key: str, label: str, *, default: str | None = None) -> str:
    value = raw.get(key, default)
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label}.{key} must be a non-empty string")
    return value


def _opt_str(raw: dict[str, Any], key: str, label: str) -> str | None:
    if key not in raw:
        return None
    value = raw[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{label}.{key} must be a non-empty string when set")
    return value


def _number(raw: dict[str, Any], key: str, label: str, *, default: float, minimum: float) -> float:
    value = raw.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{label}.{key} must be a number >= {minimum}")
    return float(value)


def _int(raw: dict[str, Any], key: str, label: str, *, default: int, minimum: int) -> int:
    value = raw.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ConfigError(f"{label}.{key} must be an integer >= {minimum}")
    return value


def _parse_telegram(raw: Any) -> TelegramConfig:
    table = _require_table(raw, "[telegram]")
    unknown = set(table) - {"bot_token_env", "owner_allow_from"}
    if unknown:
        raise ConfigError(f"[telegram] has unknown key(s): {', '.join(sorted(unknown))}")
    bot_token_env = _str(table, "bot_token_env", "[telegram]")

    allow_raw = table.get("owner_allow_from", [])
    if not isinstance(allow_raw, list) or not allow_raw:
        raise ConfigError("[telegram].owner_allow_from must be a non-empty list of chat ids")
    allow: set[int] = set()
    for entry in allow_raw:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise ConfigError(
                f"[telegram].owner_allow_from entry {entry!r} must be an integer chat id"
            )
        allow.add(entry)
    return TelegramConfig(bot_token_env=bot_token_env, owner_allow_from=frozenset(allow))


def _parse_provider(raw: Any, index: int) -> ProviderSpec:
    label = f"[[llm.chain]] #{index}"
    table = _require_table(raw, label)
    unknown = set(table) - {"provider", "model", "api_key_env", "base_url", "skip_between"}
    if unknown:
        raise ConfigError(f"{label} has unknown key(s): {', '.join(sorted(unknown))}")

    windows_raw = table.get("skip_between", [])
    if not isinstance(windows_raw, list) or not all(isinstance(w, str) for w in windows_raw):
        raise ConfigError(f"{label}.skip_between must be a list of 'HH:MM-HH:MM' strings")

    return ProviderSpec(
        provider=_str(table, "provider", label),
        model=_str(table, "model", label),
        api_key_env=_opt_str(table, "api_key_env", label),
        base_url=_opt_str(table, "base_url", label),
        skip_windows=tuple(TimeWindow.parse(w) for w in windows_raw),
    )


def _parse_llm(raw: Any) -> LLMConfig:
    table = _require_table(raw, "[llm]")
    unknown = set(table) - {
        "chain",
        "use_litellm",
        "per_call_timeout_seconds",
        "max_tool_calls_per_turn",
        "turn_wall_time_seconds",
        "tool_call_timeout_seconds",
        "max_tool_output_chars",
    }
    if unknown:
        raise ConfigError(f"[llm] has unknown key(s): {', '.join(sorted(unknown))}")

    chain_raw = table.get("chain", [])
    if not isinstance(chain_raw, list) or not chain_raw:
        raise ConfigError("[llm].chain must have at least one [[llm.chain]] entry")

    use_litellm = table.get("use_litellm", True)
    if not isinstance(use_litellm, bool):
        raise ConfigError("[llm].use_litellm must be a boolean")

    return LLMConfig(
        chain=tuple(_parse_provider(entry, i) for i, entry in enumerate(chain_raw)),
        use_litellm=use_litellm,
        per_call_timeout_seconds=_number(
            table, "per_call_timeout_seconds", "[llm]", default=30.0, minimum=1.0
        ),
        max_tool_calls_per_turn=_int(
            table, "max_tool_calls_per_turn", "[llm]", default=12, minimum=1
        ),
        turn_wall_time_seconds=_number(
            table, "turn_wall_time_seconds", "[llm]", default=60.0, minimum=1.0
        ),
        tool_call_timeout_seconds=_number(
            table, "tool_call_timeout_seconds", "[llm]", default=30.0, minimum=1.0
        ),
        max_tool_output_chars=_int(
            table, "max_tool_output_chars", "[llm]", default=40_000, minimum=1_000
        ),
    )


def _parse_mcp(raw: Any) -> tuple[MCPServerSpec, ...]:
    if raw is None:
        return ()
    table = _require_table(raw, "[mcp]")
    unknown = set(table) - {"servers"}
    if unknown:
        raise ConfigError(f"[mcp] has unknown key(s): {', '.join(sorted(unknown))}")
    servers_raw = table.get("servers", [])
    if not isinstance(servers_raw, list):
        raise ConfigError("[mcp].servers must be a list of [[mcp.servers]] entries")

    servers: list[MCPServerSpec] = []
    seen: set[str] = set()
    for i, entry in enumerate(servers_raw):
        label = f"[[mcp.servers]] #{i}"
        srv = _require_table(entry, label)
        allowed = {"name", "transport", "command", "args", "env", "url", "token_env", "forward_env"}
        extra = set(srv) - allowed
        if extra:
            raise ConfigError(f"{label} has unknown key(s): {', '.join(sorted(extra))}")

        name = _str(srv, "name", label)
        if not _MCP_NAME_RE.match(name):
            raise ConfigError(
                f"{label}.name {name!r} must be lowercase [a-z0-9_-], starting alphanumeric"
            )
        if name in seen:
            raise ConfigError(f"[[mcp.servers]] name {name!r} is used more than once")
        seen.add(name)

        transport = srv.get("transport", "stdio")
        if transport not in VALID_MCP_TRANSPORTS:
            raise ConfigError(
                f"{label}.transport {transport!r} is not one of {', '.join(VALID_MCP_TRANSPORTS)}"
            )

        args_raw = srv.get("args", [])
        if not isinstance(args_raw, list) or not all(isinstance(a, str) for a in args_raw):
            raise ConfigError(f"{label}.args must be a list of strings")

        forward_raw = srv.get("forward_env", [])
        if not isinstance(forward_raw, list) or not all(
            isinstance(v, str) and _ENV_NAME_RE.match(v) for v in forward_raw
        ):
            raise ConfigError(
                f"{label}.forward_env must be a list of environment-variable names "
                "(^[A-Za-z_][A-Za-z0-9_]*$)"
            )

        env_raw = srv.get("env", {})
        if not isinstance(env_raw, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env_raw.items()
        ):
            raise ConfigError(f"{label}.env must be a table of string -> string")

        command = _opt_str(srv, "command", label)
        url = _opt_str(srv, "url", label)
        if transport == "stdio":
            if not command:
                raise ConfigError(f"{label} (stdio) requires 'command'")
            if url:
                raise ConfigError(f"{label} (stdio) must not set 'url'")
        else:
            if not url:
                raise ConfigError(f"{label} ({transport}) requires 'url'")
            if command:
                raise ConfigError(f"{label} ({transport}) must not set 'command'")

        servers.append(
            MCPServerSpec(
                name=name,
                transport=transport,
                command=command,
                args=tuple(args_raw),
                env=dict(env_raw),
                url=url,
                token_env=_opt_str(srv, "token_env", label),
                forward_env=tuple(forward_raw),
            )
        )
    return tuple(servers)


def _parse_memory(raw: Any) -> MemoryConfig:
    if raw is None:
        return MemoryConfig()
    table = _require_table(raw, "[memory]")
    unknown = set(table) - {"history_turns", "log_dir"}
    if unknown:
        raise ConfigError(f"[memory] has unknown key(s): {', '.join(sorted(unknown))}")
    log_dir = table.get("log_dir", "")
    if not isinstance(log_dir, str):
        raise ConfigError("[memory].log_dir must be a string ('' disables logging)")
    return MemoryConfig(
        history_turns=_int(table, "history_turns", "[memory]", default=8, minimum=0),
        log_dir=log_dir,
    )


def load_config_data(data: Mapping[str, Any], *, source: Path | None = None) -> Config:
    """Build a :class:`Config` from an already-parsed mapping (used by tests)."""
    unknown = set(data) - _ALLOWED_TOP_LEVEL
    if unknown:
        raise ConfigError(f"config has unknown top-level key(s): {', '.join(sorted(unknown))}")
    if "telegram" not in data:
        raise ConfigError("config is missing the required [telegram] table")
    if "llm" not in data:
        raise ConfigError("config is missing the required [llm] table")
    return Config(
        telegram=_parse_telegram(data["telegram"]),
        llm=_parse_llm(data["llm"]),
        mcp_servers=_parse_mcp(data.get("mcp")),
        memory=_parse_memory(data.get("memory")),
        source=source,
    )


def load_config_text(text: str, *, source: Path | None = None) -> Config:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"config is not valid TOML: {exc}") from exc
    return load_config_data(data, source=source)


def load_config(
    path: str | os.PathLike[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Config:
    """Load config from *path*, or ``$INFRA_ASSISTANT_CONFIG`` if *path* is ``None``."""
    env = os.environ if environ is None else environ
    if path is None:
        env_path = env.get(CONFIG_ENV_VAR)
        if not env_path:
            raise ConfigError(f"no config path given and ${CONFIG_ENV_VAR} is not set")
        path = env_path
    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    return load_config_text(config_path.read_text(encoding="utf-8"), source=config_path)
