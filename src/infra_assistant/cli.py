"""Command-line entry point: ``infra-assistant run`` and ``infra-assistant check``."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import dataclasses
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from . import __version__, credentials, secret_redaction
from .config import Config, ProviderSpec, load_config
from .errors import InfraAssistantError, ProviderError
from .llm import LLMResponse
from .mcp_client import MCPPool
from .prompt import assemble_system_prompt
from .providers import build_provider

_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_DEFAULT_PROMPT_DIR = "prompts"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infra-assistant",
        description="A minimal Telegram assistant for infrastructure Q&A.",
    )
    parser.add_argument("--version", action="version", version=f"infra-assistant {__version__}")
    parser.add_argument("--log-level", choices=_LOG_LEVELS, default="INFO")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("run", "run the bot (long-poll Telegram)"),
        ("check", "validate config, probe providers and MCP servers, then exit"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument(
            "-c", "--config", help="path to the config file (default: $INFRA_ASSISTANT_CONFIG)"
        )
        p.add_argument(
            "--prompt-dir",
            default=_DEFAULT_PROMPT_DIR,
            help=f"directory holding persona.md / user.md / context.md (default: {_DEFAULT_PROMPT_DIR})",
        )
        p.add_argument("--no-litellm", action="store_true", help="force the httpx provider path")
        if name == "check":
            p.add_argument(
                "--probe-llm",
                action="store_true",
                help="also make one live 1-token call to each eligible provider",
            )
    return parser


#: Third-party loggers that emit credentials in the clear and must never drop
#: below WARNING regardless of ``--log-level``: ``httpx`` logs the full request
#: URL at INFO, and the Telegram bot token rides in that URL's path. A debug
#: flag must not double as a secret-exfil switch. (The value-based redaction
#: filter (see secret_redaction.py) is the belt to this suspenders.)
_MUZZLED_LOGGERS = ("httpx", "httpcore", "telegram.request")


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    for name in _MUZZLED_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    # Attach the value-redaction filter now (empty); it is seeded with the
    # resolved secret values as soon as the config is loaded. Belt to the
    # _MUZZLED_LOGGERS suspenders.
    secret_redaction.install()


def _load(config_path: str | None, *, no_litellm: bool) -> Config:
    config = load_config(config_path)
    if no_litellm and config.llm.use_litellm:
        llm = dataclasses.replace(config.llm, use_litellm=False)
        config = dataclasses.replace(config, llm=llm)
    return config


async def _check(config: Config, prompt_dir: str, *, probe_llm: bool) -> int:
    problems = 0
    secrets = config.resolve_secrets()
    secret_redaction.redact_secrets(secrets.values())
    print(f"secrets     : {len(secrets)} variable(s) resolved")

    system_prompt = assemble_system_prompt(prompt_dir)
    print(f"system prompt: {len(system_prompt)} chars from {prompt_dir}/")

    now = datetime.now(UTC)
    eligible_now = [s for s in config.llm.chain if not s.skipped_at(now)]
    print(
        f"providers   : {len(config.llm.chain)} in the chain "
        f"({len(eligible_now)} eligible now, use_litellm={config.llm.use_litellm})"
    )
    if not eligible_now:
        print("  WARN: every provider is inside a peak window right now — the bot cannot answer")
        problems += 1
    for i, spec in enumerate(config.llm.chain):
        skipped = spec.skipped_at(now)
        state = "SKIPPED now (peak window)" if skipped else "eligible"
        detail = ""
        if probe_llm and not skipped:
            detail = " — " + await _probe_provider(spec, config, secrets)
            if "FAIL" in detail:
                problems += 1
        print(f"  [{i}] {spec.provider}/{spec.model}: {state}{detail}")

    if config.mcp_servers:
        pool = MCPPool(
            config.mcp_servers, secrets, tool_call_timeout=config.llm.tool_call_timeout_seconds
        )
        try:
            await pool.open()
        except InfraAssistantError as exc:
            print(f"mcp servers : FAIL — {exc}")
            return 1
        try:
            for name, count in pool.server_summary():
                mark = "ok  " if count else "WARN"
                if not count:
                    problems += 1
                print(f"mcp server  : {mark}{name} — {count} tool(s)")
        finally:
            await pool.aclose()
    else:
        print("mcp servers : none configured")

    print(f"\n{'OK' if problems == 0 else f'{problems} issue(s) found'}")
    return 0 if problems == 0 else 1


async def _probe_provider(spec: ProviderSpec, config: Config, secrets: dict[str, str]) -> str:
    """One tiny live call. Returns 'ok (…)' or 'FAIL (…)'."""
    try:
        provider = build_provider(
            spec, secrets, config.llm.per_call_timeout_seconds, config.llm.use_litellm
        )
    except InfraAssistantError as exc:
        return f"FAIL (construct: {exc})"
    try:
        resp: LLMResponse = await provider.complete([{"role": "user", "content": "ping"}], None)
        return f"ok (replied {len(resp.text or '')} chars)"
    except ProviderError as exc:
        return f"FAIL ({type(exc.__cause__ or exc).__name__})"
    except Exception as exc:
        return f"FAIL (unexpected {type(exc).__name__})"
    finally:
        await provider.aclose()


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _configure_logging(args.log_level)
    # Secrets are resolved file-first from here on: $CREDENTIALS_DIRECTORY, then
    # $INFRA_ASSISTANT_CREDENTIALS_DIR (start.sh points this at the unit's
    # RuntimeDirectory, where fetch-secrets.sh drops the fetched values), then
    # os.environ. Keeps the plaintext out of /proc/<pid>/environ.
    credentials.configure(extra_dir=os.environ.get("INFRA_ASSISTANT_CREDENTIALS_DIR"))
    prompt_dir = str(Path(args.prompt_dir).expanduser().resolve())
    config: Config | None = None
    try:
        config = _load(args.config, no_litellm=args.no_litellm)
        if args.command == "check":
            return asyncio.run(_check(config, prompt_dir, probe_llm=args.probe_llm))
        if args.command == "run":
            secrets = config.resolve_secrets()
            secret_redaction.redact_secrets(secrets.values())
            from .runner import serve

            asyncio.run(serve(config, secrets, prompt_dir))
            return 0
    except InfraAssistantError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if args.command == "run":
            _best_effort_startup_alert(config, str(exc))
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130
    return 2  # pragma: no cover - argparse requires a subcommand


def _best_effort_startup_alert(config: Config | None, reason: str) -> None:
    """If we can, DM the owner that startup failed — so a crash loop is loud.

    Silent on any problem: this runs only when the process is already exiting
    non-zero, and journald still has the real error.
    """
    if config is None:
        return
    with contextlib.suppress(Exception):
        from telegram import Bot

        token = credentials.default_credential_source().get(config.telegram.bot_token_env, "")
        owner = next(iter(sorted(config.telegram.owner_allow_from)), None)
        if not token or owner is None:
            return

        async def _send() -> None:
            async with Bot(token) as bot:
                await bot.send_message(
                    chat_id=owner,
                    text=f"⚠️ infra-assistant failed to start: {reason[:300]}",
                )

        asyncio.run(_send())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
