"""The local Unix-socket front end (``infra-assistant serve``).

All data is fictitious: chat ids ``111`` / ``999``, host ``node-a``, free text
``"Test Data ..."``. The protocol and handler tests run everywhere; the tests
that bind a real socket need ``AF_UNIX`` support in asyncio (Linux/macOS).
"""

from __future__ import annotations

import asyncio
import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from infra_assistant.agent import Agent
from infra_assistant.config import LLMConfig, ProviderSpec
from infra_assistant.errors import AllProvidersFailed, InfraAssistantError, TurnTimeout
from infra_assistant.llm import LLMResponse
from infra_assistant.memory import ChatMemory
from infra_assistant.socket_server import (
    MAX_REQUEST_BYTES,
    MAX_TEXT_CHARS,
    ServerState,
    handle_connection,
    handle_request,
    parse_request,
    start_unix_server,
)
from tests.fakes import FakeCompleter

ALLOWED = 111
BLOCKED = 999

needs_unix = pytest.mark.skipif(
    sys.platform == "win32" or not hasattr(asyncio, "start_unix_server"),
    reason="needs AF_UNIX sockets in asyncio",
)


def _state(chain: FakeCompleter) -> ServerState:
    cfg = LLMConfig(chain=(ProviderSpec("p", "m"),), turn_wall_time_seconds=15.0)
    return ServerState(
        agent=Agent(system_prompt="sys", chain=chain, llm_config=cfg),
        memory=ChatMemory(4),
        allow_from=frozenset({ALLOWED}),
        turn_wall_time=15.0,
    )


def _req(**fields: Any) -> bytes:
    return json.dumps(fields).encode() + b"\n"


# ---- parse_request ---------------------------------------------------------


def test_parse_valid_question() -> None:
    assert parse_request(_req(chat_id=ALLOWED, text="Test Data question")) == (
        ALLOWED,
        "Test Data question",
        False,
    )


def test_parse_reset_needs_no_text() -> None:
    assert parse_request(_req(chat_id=ALLOWED, reset=True)) == (ALLOWED, None, True)


@pytest.mark.parametrize(
    "raw",
    [
        b"not json\n",
        b"\xff\xfe\n",
        b"[1, 2]\n",
        _req(text="no chat id"),
        _req(chat_id="111", text="chat id must be an int"),
        _req(chat_id=True, text="a bool is not a chat id"),
        _req(chat_id=ALLOWED),
        _req(chat_id=ALLOWED, text="   "),
        _req(chat_id=ALLOWED, text=123),
        _req(chat_id=ALLOWED, text="x" * (MAX_TEXT_CHARS + 1)),
    ],
)
def test_parse_rejects_malformed(raw: bytes) -> None:
    with pytest.raises(ValueError):
        parse_request(raw)


# ---- handle_request --------------------------------------------------------


async def test_happy_path_answers_and_records() -> None:
    chain = FakeCompleter([LLMResponse(text="node-a is active")])
    state = _state(chain)
    reply = await handle_request(_req(chat_id=ALLOWED, text="is node-a up?"), state)
    assert reply == {"ok": True, "reply": "node-a is active"}
    assert [m.content for m in state.memory.history(ALLOWED)] == [
        "is node-a up?",
        "node-a is active",
    ]


async def test_history_is_passed_to_the_next_turn() -> None:
    chain = FakeCompleter([LLMResponse(text="first"), LLMResponse(text="second")])
    state = _state(chain)
    await handle_request(_req(chat_id=ALLOWED, text="one"), state)
    await handle_request(_req(chat_id=ALLOWED, text="two"), state)
    second_call_messages = chain.calls[1][0]
    assert any(m.get("content") == "one" for m in second_call_messages)


async def test_non_allowlisted_chat_is_refused_and_never_reaches_the_model() -> None:
    chain = FakeCompleter([LLMResponse(text="must not run")])
    state = _state(chain)
    reply = await handle_request(_req(chat_id=BLOCKED, text="hello?"), state)
    assert reply["ok"] is False
    assert reply["error"] == "not_allowed"
    assert not chain.calls
    assert state.memory.history(BLOCKED) == []


async def test_reset_clears_history_but_only_for_allowlisted_chats() -> None:
    chain = FakeCompleter([LLMResponse(text="a")])
    state = _state(chain)
    await handle_request(_req(chat_id=ALLOWED, text="q"), state)
    assert state.memory.history(ALLOWED)
    refused = await handle_request(_req(chat_id=BLOCKED, reset=True), state)
    assert refused["error"] == "not_allowed"
    assert state.memory.history(ALLOWED)
    ok = await handle_request(_req(chat_id=ALLOWED, reset=True), state)
    assert ok["ok"] is True
    assert state.memory.history(ALLOWED) == []


async def test_bad_request_is_a_clean_error_not_an_exception() -> None:
    state = _state(FakeCompleter([]))
    reply = await handle_request(b"{broken\n", state)
    assert reply["ok"] is False and reply["error"] == "bad_request"


async def test_all_providers_failed_is_reported_and_not_recorded() -> None:
    state = _state(FakeCompleter([AllProvidersFailed("ConnectionError")]))
    reply = await handle_request(_req(chat_id=ALLOWED, text="status?"), state)
    assert reply["ok"] is False and reply["error"] == "all_providers_failed"
    assert reply["reply"].startswith("⚠️")
    assert state.memory.history(ALLOWED) == []


async def test_turn_timeout_is_reported() -> None:
    state = _state(FakeCompleter([TurnTimeout()]))
    reply = await handle_request(_req(chat_id=ALLOWED, text="slow"), state)
    assert reply["ok"] is False and reply["error"] == "turn_timeout"
    assert "15" in reply["reply"]


async def test_unexpected_error_does_not_leak_details() -> None:
    state = _state(FakeCompleter([RuntimeError("Test Data internal detail")]))
    reply = await handle_request(_req(chat_id=ALLOWED, text="boom"), state)
    assert reply["ok"] is False and reply["error"] == "internal"
    assert "internal detail" not in reply["reply"]


async def test_one_turn_at_a_time_per_chat() -> None:
    running = 0
    peak = 0

    class SlowCompleter(FakeCompleter):
        async def complete(self, messages: Any, tools: Any) -> LLMResponse:
            nonlocal running, peak
            running += 1
            peak = max(peak, running)
            await asyncio.sleep(0.02)
            running -= 1
            return LLMResponse(text="ok")

    state = _state(SlowCompleter([]))
    await asyncio.gather(
        *(handle_request(_req(chat_id=ALLOWED, text=f"q{i}"), state) for i in range(3))
    )
    assert peak == 1


async def test_reset_waits_for_a_turn_in_flight() -> None:
    gate = asyncio.Event()

    class GatedCompleter(FakeCompleter):
        async def complete(self, messages: Any, tools: Any) -> LLMResponse:
            await gate.wait()
            return LLMResponse(text="late answer")

    state = _state(GatedCompleter([]))
    turn = asyncio.create_task(handle_request(_req(chat_id=ALLOWED, text="q"), state))
    await asyncio.sleep(0.01)
    reset = asyncio.create_task(handle_request(_req(chat_id=ALLOWED, reset=True), state))
    await asyncio.sleep(0.01)
    assert not reset.done()  # blocked behind the running turn
    gate.set()
    await asyncio.gather(turn, reset)
    assert state.memory.history(ALLOWED) == []  # the reset ran last


async def test_question_at_the_limit_fits_when_every_character_is_escaped() -> None:
    state = _state(FakeCompleter([LLMResponse(text="ok")]))
    text = "é" * MAX_TEXT_CHARS
    payload = json.dumps({"chat_id": ALLOWED, "text": text}, ensure_ascii=True).encode() + b"\n"
    assert len(payload) > 16 * 1024  # would have been refused by the old cap
    assert len(payload) <= MAX_REQUEST_BYTES
    out = await _round_trip(payload, state)
    assert out["ok"] is True


# ---- handle_connection (in-memory streams) ----------------------------------


class _FakeWriter:
    def __init__(self) -> None:
        self.data = b""
        self.closed = False

    def get_extra_info(self, name: str) -> Any:
        return None

    def write(self, data: bytes) -> None:
        self.data += data

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


async def _round_trip(payload: bytes, state: ServerState, *, eof: bool = True) -> dict[str, Any]:
    reader = asyncio.StreamReader()
    reader.feed_data(payload)
    if eof:
        reader.feed_eof()
    writer = _FakeWriter()
    await handle_connection(reader, writer, state)  # type: ignore[arg-type]
    assert writer.closed
    assert writer.data.endswith(b"\n")
    result: dict[str, Any] = json.loads(writer.data)
    return result


async def test_connection_round_trip() -> None:
    state = _state(FakeCompleter([LLMResponse(text="fine")]))
    out = await _round_trip(_req(chat_id=ALLOWED, text="status?"), state)
    assert out == {"ok": True, "reply": "fine"}


async def test_connection_without_newline_is_rejected() -> None:
    chain = FakeCompleter([LLMResponse(text="must not run")])
    out = await _round_trip(json.dumps({"chat_id": ALLOWED, "text": "x"}).encode(), _state(chain))
    assert out["error"] == "bad_request"
    assert not chain.calls


async def test_connection_with_oversized_line_is_rejected() -> None:
    chain = FakeCompleter([LLMResponse(text="must not run")])
    big = b'{"chat_id": 111, "text": "' + b"x" * (MAX_REQUEST_BYTES + 10) + b'"}\n'
    reader = asyncio.StreamReader(limit=MAX_REQUEST_BYTES)
    reader.feed_data(big)
    reader.feed_eof()
    writer = _FakeWriter()
    await handle_connection(reader, writer, _state(chain))  # type: ignore[arg-type]
    assert json.loads(writer.data)["error"] == "bad_request"
    assert not chain.calls


async def test_non_ascii_reply_round_trips() -> None:
    state = _state(FakeCompleter([LLMResponse(text="Test Data — ünïcode ✓")]))
    out = await _round_trip(_req(chat_id=ALLOWED, text="hi"), state)
    assert out["reply"] == "Test Data — ünïcode ✓"


# ---- a real socket (Linux/macOS) --------------------------------------------


async def _ask(path: Path, payload: bytes) -> dict[str, Any]:
    reader, writer = await asyncio.open_unix_connection(str(path))  # type: ignore[attr-defined,unused-ignore]
    writer.write(payload)
    await writer.drain()
    line = await reader.readline()
    writer.close()
    await writer.wait_closed()
    result: dict[str, Any] = json.loads(line)
    return result


@needs_unix
async def test_real_socket_is_mode_600_and_answers(tmp_path: Path) -> None:
    sock = tmp_path / "ia.sock"
    server = await start_unix_server(sock, _state(FakeCompleter([LLMResponse(text="pong")])))
    try:
        assert stat.S_IMODE(sock.lstat().st_mode) == 0o600
        out = await _ask(sock, _req(chat_id=ALLOWED, text="ping"))
        assert out == {"ok": True, "reply": "pong"}
    finally:
        server.close()
        await server.wait_closed()


@needs_unix
async def test_real_socket_ignores_a_loose_umask(tmp_path: Path) -> None:
    previous = os.umask(0o000)
    try:
        server = await start_unix_server(tmp_path / "ia.sock", _state(FakeCompleter([])))
    finally:
        os.umask(previous)
    try:
        assert stat.S_IMODE((tmp_path / "ia.sock").lstat().st_mode) == 0o600
    finally:
        server.close()
        await server.wait_closed()


@needs_unix
async def test_stale_socket_file_is_replaced(tmp_path: Path) -> None:
    import socket as socket_module

    sock = tmp_path / "ia.sock"
    dead = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)  # type: ignore[attr-defined,unused-ignore]
    dead.bind(str(sock))
    dead.close()  # leaves the file behind with nothing listening
    assert sock.exists()
    server = await start_unix_server(sock, _state(FakeCompleter([LLMResponse(text="ok")])))
    try:
        assert (await _ask(sock, _req(chat_id=ALLOWED, text="x")))["ok"] is True
    finally:
        server.close()
        await server.wait_closed()


@needs_unix
async def test_a_slow_live_server_is_not_treated_as_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sock = tmp_path / "ia.sock"
    first = await start_unix_server(sock, _state(FakeCompleter([])))

    async def _hang(*_a: Any, **_k: Any) -> Any:
        raise TimeoutError

    monkeypatch.setattr("infra_assistant.socket_server.asyncio.wait_for", _hang)
    try:
        with pytest.raises(InfraAssistantError, match="did not answer"):
            await start_unix_server(sock, _state(FakeCompleter([])))
        assert sock.exists()  # the live server's socket was left alone
    finally:
        monkeypatch.undo()
        first.close()
        await first.wait_closed()


@needs_unix
async def test_refuses_to_remove_a_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "not-a-socket"
    path.write_text("Test Data keep me")
    with pytest.raises(InfraAssistantError, match="not a socket"):
        await start_unix_server(path, _state(FakeCompleter([])))
    assert path.read_text() == "Test Data keep me"


@needs_unix
async def test_refuses_a_second_server_on_a_live_socket(tmp_path: Path) -> None:
    sock = tmp_path / "ia.sock"
    first = await start_unix_server(sock, _state(FakeCompleter([])))
    try:
        with pytest.raises(InfraAssistantError, match="already listening"):
            await start_unix_server(sock, _state(FakeCompleter([])))
    finally:
        first.close()
        await first.wait_closed()
