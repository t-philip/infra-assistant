"""The Telegram front end: long-poll, owner allowlist, per-chat memory.

`python-telegram-bot` handles the transport (long-poll, retries, rate limits,
command routing). Everything this module adds is thin:

* an **owner allowlist** — a message from a chat id not in
  ``[telegram].owner_allow_from`` is ignored (checked here *and* as a PTB filter);
* wiring each message through :class:`~infra_assistant.agent.Agent`;
* splitting a reply that exceeds Telegram's 4096-character limit;
* ``/start`` · ``/help`` · ``/reset`` (clear this chat's history).

The agent, memory, and allowlist live in ``application.bot_data`` so the
handlers are plain functions and easy to test with a fake update/context.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

from telegram import Update
from telegram.constants import ChatAction
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .agent import Agent
from .config import Config
from .errors import AllProvidersFailed, TurnTimeout
from .memory import ChatMemory

logger = logging.getLogger("infra_assistant.telegram")

# Telegram counts message length in UTF-16 code units, not Unicode code points.
TELEGRAM_MESSAGE_LIMIT = 4096

_START_TEXT = (
    "infra-assistant is online. Ask me about the state of the infrastructure — "
    "service status, alerts, metrics, DNS, uptime. I can only read; for anything "
    "that changes state I'll give you the command to run.\n\n"
    "/help — what I can do\n/reset — forget this chat's history"
)
_HELP_TEXT = (
    "Ask in plain language, e.g.:\n"
    "• is the reverse-proxy container up?\n"
    "• any firing alerts right now?\n"
    "• top DNS clients today\n"
    "• is the uptime monitor healthy?\n\n"
    "I use read-only tools only. I will not restart, delete, or change anything — "
    "I'll tell you how to do it yourself.\n\n"
    "/reset clears the short rolling history I keep for this chat."
)


def build_application(
    config: Config,
    *,
    agent: Agent,
    memory: ChatMemory,
    token: str,
) -> Application[Any, Any, Any, Any, Any, Any]:
    """Assemble the PTB application with handlers and shared state."""
    # concurrent_updates so a slow turn in one chat does not freeze every other
    # chat and every command; a per-chat lock still serialises one chat's turns.
    app = ApplicationBuilder().token(token).concurrent_updates(True).build()
    app.bot_data["agent"] = agent
    app.bot_data["memory"] = memory
    app.bot_data["allow_from"] = frozenset(config.telegram.owner_allow_from)
    app.bot_data["turn_wall_time"] = config.llm.turn_wall_time_seconds
    app.bot_data["chat_locks"] = defaultdict(asyncio.Lock)

    owner_filter = filters.Chat(chat_id=list(config.telegram.owner_allow_from))
    app.add_handler(CommandHandler("start", on_start, filters=owner_filter))
    app.add_handler(CommandHandler("help", on_help, filters=owner_filter))
    app.add_handler(CommandHandler("reset", on_reset, filters=owner_filter))
    app.add_handler(MessageHandler(owner_filter & filters.TEXT & ~filters.COMMAND, on_message))
    # Anything that fell through the owner filter: note it, do nothing else.
    app.add_handler(MessageHandler(filters.ALL, on_ignored), group=1)
    app.add_error_handler(on_error)
    return app


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Last-resort handler for anything a handler let escape.

    Logs with a traceback and best-effort DMs the first owner, so a bug in the
    bot surfaces in the chat rather than only in the log.
    """
    logger.error("unhandled exception in a handler", exc_info=context.error)
    allow = context.bot_data.get("allow_from") or frozenset()
    owner = next(iter(sorted(allow)), None)
    if owner is None:
        return
    try:
        await context.bot.send_message(
            chat_id=owner,
            text="⚠️ infra-assistant hit an internal error handling a message — see the bot log.",
        )
    except Exception:  # the alert itself failing must not recurse
        logger.warning("could not send the error alert to the owner", exc_info=True)


def _allowed(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    chat = update.effective_chat
    allow = context.bot_data.get("allow_from", frozenset())
    return chat is not None and chat.id in allow


async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update, context) or update.effective_message is None:
        return
    await update.effective_message.reply_text(_START_TEXT)


async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update, context) or update.effective_message is None:
        return
    await update.effective_message.reply_text(_HELP_TEXT)


async def on_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update, context) or update.effective_message is None:
        return
    memory: ChatMemory = context.bot_data["memory"]
    assert update.effective_chat is not None
    memory.reset(update.effective_chat.id)
    await update.effective_message.reply_text("History cleared for this chat.")


async def on_ignored(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat = update.effective_chat
    if chat is not None and not _allowed(update, context):
        logger.info("ignoring message from non-allowlisted chat %s", chat.id)


async def on_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _allowed(update, context):
        return
    message = update.effective_message
    chat = update.effective_chat
    if message is None or chat is None or not message.text:
        return

    agent: Agent = context.bot_data["agent"]
    memory: ChatMemory = context.bot_data["memory"]
    wall_time = context.bot_data.get("turn_wall_time", 60.0)
    locks: dict[int, asyncio.Lock] = context.bot_data.setdefault(
        "chat_locks", defaultdict(asyncio.Lock)
    )

    try:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    except Exception:  # a failed typing indicator must never sink the turn
        logger.debug("send_chat_action failed", exc_info=True)

    text = message.text
    ok = True
    async with locks[chat.id]:  # one in-flight turn per chat
        try:
            reply = await agent.answer(text, memory.history(chat.id))
        except AllProvidersFailed as exc:
            ok = False
            reply = (
                f"⚠️ I couldn't reach any language model just now "
                f"(last error: {exc.last_error}). Try again shortly."
            )
        except TurnTimeout:
            ok = False
            reply = f"⚠️ That took longer than {wall_time:g}s so I stopped. Try a narrower question."
        except Exception:
            ok = False
            logger.exception("unhandled error answering a message in chat %s", chat.id)
            reply = "⚠️ Something went wrong handling that. Details are in the bot log."

        if ok:
            memory.record(chat.id, text, reply)
        await reply_chunked(message, reply)


async def reply_chunked(message: object, text: str) -> None:
    """Send *text* as one or more replies, each within Telegram's size limit."""
    for chunk in _split_for_telegram(text):
        await message.reply_text(chunk)  # type: ignore[attr-defined]


def _utf16_len(text: str) -> int:
    """Length as Telegram counts it — UTF-16 code units, so non-BMP chars are 2."""
    return len(text.encode("utf-16-le")) // 2


def _split_for_telegram(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    text = text or "(empty reply)"
    if _utf16_len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while _utf16_len(remaining) > limit:
        # Largest code-point prefix that still fits `limit` UTF-16 units, then
        # back off to the last newline inside it if there is one.
        hi = min(len(remaining), limit)
        while _utf16_len(remaining[:hi]) > limit:
            hi -= 1
        cut = remaining.rfind("\n", 0, hi)
        if cut <= 0:
            cut = hi
        chunks.append(remaining[:cut].rstrip("\n"))
        remaining = remaining[cut:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks
