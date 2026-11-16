"""A local Unix-socket front end: ask the agent a question without Telegram.

``infra-assistant serve`` runs the same agent, memory and MCP pool as ``run`` but
takes its questions from a Unix domain socket instead of long-polling Telegram.
This lets one process own the Telegram connection (for example a command
dispatcher that answers fixed commands itself) and hand every other message to
the agent. **No Telegram code runs in this path and no Telegram credential is
loaded.**

Protocol — one request and one reply per connection, each a single UTF-8 JSON
object terminated by a newline:

* request: ``{"chat_id": 111, "text": "is node-a up?"}``
  or ``{"chat_id": 111, "reset": true}`` to clear that chat's history;
* reply: ``{"ok": true, "reply": "..."}``
  or ``{"ok": false, "error": "<code>", "reply": "<message for the user>"}``.

Error codes: ``bad_request``, ``not_allowed``, ``all_providers_failed``,
``turn_timeout``, ``internal``. A reply is never split: the caller owns
chunking to its transport's size limit.

Access control, in three independent layers:

1. the socket file is created mode ``0600`` (set via ``umask`` before ``bind``,
   then asserted), so only the service user can connect;
2. where the platform supports it (Linux ``SO_PEERCRED``) the connecting
   process's uid must equal the server's uid;
3. ``chat_id`` must be in ``[telegram].owner_allow_from`` — the same allowlist
   the Telegram front end enforces, checked again here.

Requests are bounded: a request line over :data:`MAX_REQUEST_BYTES`, or one
that does not arrive within :data:`READ_TIMEOUT_SECONDS`, is rejected.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import socket
import stat
import struct
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .agent import Agent
from .errors import AllProvidersFailed, InfraAssistantError, TurnTimeout
from .memory import ChatMemory

logger = logging.getLogger("infra_assistant.socket")

#: A request line longer than this is refused. Sized so a question at the
#: character limit still fits when a client escapes every non-ASCII character
#: (``\uXXXX``, 6 bytes each, which is the default for most JSON encoders).
MAX_REQUEST_BYTES = 32 * 1024
#: The question text itself may not exceed this many characters.
MAX_TEXT_CHARS = 4000
#: How long a client has to deliver its request line.
READ_TIMEOUT_SECONDS = 10.0


@dataclass
class ServerState:
    """What a request handler needs; built once at start-up."""

    agent: Agent
    memory: ChatMemory
    allow_from: frozenset[int]
    turn_wall_time: float = 60.0
    chat_locks: dict[int, asyncio.Lock] = field(default_factory=lambda: defaultdict(asyncio.Lock))


def _error(code: str, reply: str) -> dict[str, Any]:
    return {"ok": False, "error": code, "reply": reply}


def parse_request(raw: bytes) -> tuple[int, str | None, bool]:
    """Validate one request line. Returns ``(chat_id, text, reset)``.

    Raises :class:`ValueError` with a short reason for anything malformed.
    """
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("request is not valid UTF-8 JSON") from exc
    if not isinstance(obj, dict):
        raise ValueError("request must be a JSON object")
    chat_id = obj.get("chat_id")
    if isinstance(chat_id, bool) or not isinstance(chat_id, int):
        raise ValueError("chat_id must be an integer")
    if obj.get("reset") is True:
        return chat_id, None, True
    text = obj.get("text")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError(f"text is longer than {MAX_TEXT_CHARS} characters")
    return chat_id, text, False


async def handle_request(raw: bytes, state: ServerState) -> dict[str, Any]:
    """Turn one request line into a reply object. Never raises."""
    try:
        chat_id, text, reset = parse_request(raw)
    except ValueError as exc:
        return _error("bad_request", f"Bad request: {exc}.")

    if chat_id not in state.allow_from:
        logger.info("refusing a request for non-allowlisted chat %s", chat_id)
        return _error("not_allowed", "That chat is not allowed.")

    if reset:
        # Under the chat lock, so a turn already in flight cannot re-add its
        # exchange after the history was reported cleared.
        async with state.chat_locks[chat_id]:
            state.memory.reset(chat_id)
        return {"ok": True, "reply": "History cleared for this chat."}

    assert text is not None
    async with state.chat_locks[chat_id]:  # one in-flight turn per chat
        try:
            reply = await state.agent.answer(text, state.memory.history(chat_id))
        except AllProvidersFailed as exc:
            return _error(
                "all_providers_failed",
                f"⚠️ I couldn't reach any language model just now "
                f"(last error: {exc.last_error}). Try again shortly.",
            )
        except TurnTimeout:
            return _error(
                "turn_timeout",
                f"⚠️ That took longer than {state.turn_wall_time:g}s so I stopped. "
                "Try a narrower question.",
            )
        except Exception:
            logger.exception("unhandled error answering a request for chat %s", chat_id)
            return _error(
                "internal", "⚠️ Something went wrong handling that. Details are in the log."
            )
        state.memory.record(chat_id, text, reply)
        return {"ok": True, "reply": reply}


def _peer_uid(sock: socket.socket | None) -> int | None:
    """The connecting process's uid via ``SO_PEERCRED``; ``None`` if unsupported."""
    option = getattr(socket, "SO_PEERCRED", None)
    if sock is None or option is None:
        return None
    try:
        creds = sock.getsockopt(socket.SOL_SOCKET, option, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", creds)
    except OSError:
        return None
    return int(uid)


async def handle_connection(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, state: ServerState
) -> None:
    """Serve one connection: read a line, answer it, close."""
    try:
        peer = _peer_uid(writer.get_extra_info("socket"))
        geteuid = getattr(os, "geteuid", None)
        if peer is not None and geteuid is not None and peer != geteuid():
            logger.warning("refusing a connection from uid %s", peer)
            response = _error("not_allowed", "Connection refused.")
        else:
            try:
                raw = await asyncio.wait_for(reader.readline(), READ_TIMEOUT_SECONDS)
            except (TimeoutError, ValueError, asyncio.LimitOverrunError):
                response = _error("bad_request", "Bad request: no complete request line received.")
            else:
                if not raw.endswith(b"\n") or len(raw) > MAX_REQUEST_BYTES:
                    response = _error(
                        "bad_request", "Bad request: no complete request line received."
                    )
                else:
                    response = await handle_request(raw, state)
        writer.write(json.dumps(response, ensure_ascii=False).encode("utf-8") + b"\n")
        await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        logger.debug("client went away mid-request", exc_info=True)
    finally:
        writer.close()
        with contextlib.suppress(Exception):
            await writer.wait_closed()


def _file_mode(path: Path) -> int | None:
    """``st_mode`` of *path* without following a symlink; ``None`` if absent."""
    try:
        return path.lstat().st_mode
    except FileNotFoundError:
        return None


def remove_socket_file(path: Path) -> None:
    """Delete *path* if present (used at shutdown and to clear a stale socket)."""
    with contextlib.suppress(FileNotFoundError):
        path.unlink()


async def _prepare_socket_path(path: Path) -> None:
    """Remove a stale socket file; refuse to touch anything else or a live server."""
    mode = _file_mode(path)
    if mode is None:
        return
    if not stat.S_ISSOCK(mode):
        raise InfraAssistantError(f"{path} exists and is not a socket; refusing to remove it")
    try:
        _r, w = await asyncio.wait_for(
            asyncio.open_unix_connection(str(path)),  # type: ignore[attr-defined,unused-ignore]
            2.0,
        )
    except (ConnectionRefusedError, FileNotFoundError):
        remove_socket_file(path)  # stale: nothing is listening
        return
    except TimeoutError as exc:
        # A server that is alive but slow to accept is not stale; deleting its
        # socket would leave it running and unreachable.
        raise InfraAssistantError(
            f"{path} did not answer a connection probe; not removing it"
        ) from exc
    w.close()
    raise InfraAssistantError(f"another infra-assistant is already listening on {path}")


async def start_unix_server(path: str | Path, state: ServerState) -> asyncio.AbstractServer:
    """Bind the socket (mode 0600) and start accepting connections."""
    if not hasattr(asyncio, "start_unix_server"):  # Windows
        raise InfraAssistantError("Unix sockets are not available on this platform")
    sock_path = Path(path)
    await _prepare_socket_path(sock_path)
    old_umask = os.umask(0o177)  # the file is born 0600, never briefly wider
    try:
        server = await asyncio.start_unix_server(  # type: ignore[attr-defined,unused-ignore]
            lambda r, w: handle_connection(r, w, state),
            path=str(sock_path),
            limit=MAX_REQUEST_BYTES,
        )
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(_file_mode(sock_path) or 0)
    if mode & 0o077:
        server.close()
        raise InfraAssistantError(f"{sock_path} has mode {mode:o}; expected 600")
    result: asyncio.AbstractServer = server
    return result
