"""Wire the pieces together and run the bot.

`build_runtime` assembles the provider chain, the MCP pool, the agent, and the
per-chat memory from a validated :class:`Config` — it is shared by ``run`` and
``check``. `serve` opens the MCP pool, drives ``python-telegram-bot``'s
long-poll loop, and closes everything on exit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass

from .agent import Agent
from .config import Config
from .llm import ProviderChain
from .mcp_client import MCPPool
from .memory import ChatMemory
from .prompt import assemble_system_prompt
from .providers import build_provider
from .telegram_bot import build_application

logger = logging.getLogger("infra_assistant.runner")


@dataclass
class Runtime:
    chain: ProviderChain
    pool: MCPPool
    agent: Agent
    memory: ChatMemory
    system_prompt: str

    async def aclose(self) -> None:
        await self.chain.aclose()
        await self.pool.aclose()


async def build_runtime(config: Config, secrets: dict[str, str], prompt_dir: str) -> Runtime:
    """Assemble every runtime object. Opens the MCP pool (fails closed).

    If any step after the pool is open raises (a bad ``log_dir``, say), the pool
    — and its stdio subprocesses — is closed before the exception propagates.
    """
    system_prompt = assemble_system_prompt(prompt_dir)
    pool = MCPPool(
        config.mcp_servers,
        secrets,
        tool_call_timeout=config.llm.tool_call_timeout_seconds,
    )
    await pool.open()
    try:
        chain = ProviderChain(config.llm, secrets, provider_factory=build_provider)
        agent = Agent(
            system_prompt=system_prompt,
            chain=chain,
            llm_config=config.llm,
            mcp_pool=pool,
        )
        memory = ChatMemory(
            config.memory.history_turns,
            log_dir=config.memory.log_dir or None,
        )
    except BaseException:
        await pool.aclose()
        raise
    return Runtime(chain=chain, pool=pool, agent=agent, memory=memory, system_prompt=system_prompt)


async def serve(config: Config, secrets: dict[str, str], prompt_dir: str) -> None:
    """Run the bot until interrupted."""
    runtime = await build_runtime(config, secrets, prompt_dir)
    try:
        app = build_application(
            config,
            agent=runtime.agent,
            memory=runtime.memory,
            token=secrets[config.telegram.bot_token_env],
        )
        logger.info(
            "infra-assistant ready — %d MCP tool(s), %d provider(s) in the chain",
            runtime.pool.tool_count,
            len(config.llm.chain),
        )
        async with app:
            await app.start()
            if app.updater is None:  # pragma: no cover - always present unless disabled
                raise RuntimeError("PTB application has no updater; cannot long-poll")
            await app.updater.start_polling()
            await _wait_for_stop()
            await app.updater.stop()
            await app.stop()
    finally:
        await runtime.aclose()


async def _wait_for_stop() -> None:
    """Block until SIGINT/SIGTERM (or KeyboardInterrupt on platforms without them)."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    installed = False
    with contextlib.suppress(NotImplementedError, ImportError):
        import signal

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, stop.set)
        installed = True
    if installed:
        await stop.wait()
    else:  # Windows: no signal handlers on the proactor loop
        with contextlib.suppress(KeyboardInterrupt):
            await asyncio.Event().wait()
