from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from infra_assistant import credentials
from infra_assistant.config import MCPServerSpec
from infra_assistant.errors import MCPClientError
from infra_assistant.mcp_client import MCPPool, _render_result, _transport


@pytest.fixture(autouse=True)
def _reset_credential_source() -> Iterator[None]:
    credentials._reset()
    yield
    credentials._reset()


def _server() -> MCPServer:
    srv = MCPServer(name="test-lab")

    async def node_status(host: str) -> dict[str, object]:
        """Status of a host.

        Args:
            host: the host name, e.g. 'node-a'.
        """
        return {"host": host, "active": host == "node-a"}

    async def always_fails() -> str:
        """A tool that always errors."""
        raise ToolError("Test Data failure")

    async def crashes() -> str:
        """A tool that raises an unexpected (non-ToolError) exception."""
        raise RuntimeError("secret-internal-detail-42")

    async def sleeps() -> str:
        """A tool that never returns in time."""
        await asyncio.sleep(5)
        return "unreachable"

    srv.add_tool(node_status, name="node_status", description="status of a host")
    srv.add_tool(always_fails, name="always_fails", description="errors on purpose")
    srv.add_tool(crashes, name="crashes", description="raises")
    srv.add_tool(sleeps, name="sleeps", description="hangs")
    return srv


def _spec(name: str = "lab") -> MCPServerSpec:
    return MCPServerSpec(name=name, transport="stdio", command="unused")


async def test_pool_lists_and_namespaces_tools() -> None:
    srv = _server()
    async with MCPPool([_spec()], {}, transport_factory=lambda s, t: srv) as pool:
        names = {t["name"] for t in pool.tools()}
        assert {"lab__node_status", "lab__always_fails"} <= names
        assert all(n.startswith("lab__") for n in names)
        assert pool.tool_count == len(names)
        assert pool.server_summary() == [("lab", pool.tool_count)]


async def test_pool_routes_a_call() -> None:
    srv = _server()
    async with MCPPool([_spec()], {}, transport_factory=lambda s, t: srv) as pool:
        out = await pool.call("lab__node_status", {"host": "node-a"})
        assert json.loads(out) == {"host": "node-a", "active": True}


async def test_pool_unknown_tool_is_an_error_string() -> None:
    srv = _server()
    async with MCPPool([_spec()], {}, transport_factory=lambda s, t: srv) as pool:
        assert (await pool.call("lab__nope", {})).startswith("ERROR: no such tool")


async def test_pool_tool_error_is_returned_not_raised() -> None:
    srv = _server()
    async with MCPPool([_spec()], {}, transport_factory=lambda s, t: srv) as pool:
        out = await pool.call("lab__always_fails", {})
        assert out.startswith("ERROR:")
        assert "Test Data failure" in out


async def test_pool_unexpected_tool_exception_does_not_leak_detail() -> None:
    """A non-ToolError raised in a handler comes back sanitised by the SDK."""
    srv = _server()
    async with MCPPool([_spec()], {}, transport_factory=lambda s, t: srv) as pool:
        out = await pool.call("lab__crashes", {})
        assert out.startswith("ERROR:")
        assert "secret-internal-detail-42" not in out


async def test_pool_call_that_raises_returns_generic_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If client.call_tool itself raises (transport/protocol), no raw text leaks."""
    srv = _server()
    async with MCPPool([_spec()], {}, transport_factory=lambda s, t: srv) as pool:

        async def _raise(*_a: object, **_k: object) -> object:
            raise RuntimeError("secret-transport-frame-99")

        monkeypatch.setattr(pool._clients["lab"], "call_tool", _raise)
        out = await pool.call("lab__node_status", {"host": "node-a"})
        assert out == "ERROR: tool 'lab__node_status' failed (see bot log)"


async def test_pool_tool_call_times_out() -> None:
    srv = _server()
    async with MCPPool(
        [_spec()], {}, transport_factory=lambda s, t: srv, tool_call_timeout=0.05
    ) as pool:
        out = await pool.call("lab__sleeps", {})
        assert out.startswith("ERROR:")
        assert "timed out" in out


async def test_pool_open_failure_is_fail_closed() -> None:
    def bad_factory(spec: MCPServerSpec, token: str | None) -> object:
        raise RuntimeError("no route to host")

    pool = MCPPool([_spec()], {}, transport_factory=bad_factory)
    with pytest.raises(MCPClientError, match="could not connect"):
        await pool.open()
    assert pool.tool_count == 0  # nothing left half-open


def test_transport_shapes() -> None:
    stdio = _transport(MCPServerSpec("a", "stdio", command="c", args=("run",)), None)
    assert stdio.command == "c" and stdio.args == ["run"]


def test_stdio_transport_child_env_is_narrowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "the-bot-token")  # must NOT leak
    monkeypatch.setenv("NETBOX_API_TOKEN", "nb-token")  # forwarded by name
    monkeypatch.setenv("OVERRIDE_ME", "parent-value")
    monkeypatch.delenv("RUNTIME_DIRECTORY", raising=False)
    stdio = _transport(
        MCPServerSpec(
            "a",
            "stdio",
            command="c",
            env={"OVERRIDE_ME": "spec-value"},
            forward_env=("NETBOX_API_TOKEN", "ABSENT_VAR"),
        ),
        None,
    )
    assert stdio.env is not None
    # base allow-list is carried
    assert stdio.env["PATH"] == "/usr/bin"
    # an explicit per-server literal in the TOML still wins
    assert stdio.env["OVERRIDE_ME"] == "spec-value"
    # NO secret value is in the child's environment — not the forwarded one,
    # not the bot token, nothing.
    assert "NETBOX_API_TOKEN" not in stdio.env
    assert "TELEGRAM_BOT_TOKEN" not in stdio.env
    assert "nb-token" not in stdio.env.values()

    # a forwarded var is handed over as a 0600 file under $CREDENTIALS_DIRECTORY
    creds_dir = Path(stdio.env["CREDENTIALS_DIRECTORY"])
    assert creds_dir.is_dir()
    forwarded = creds_dir / "NETBOX_API_TOKEN"
    assert forwarded.read_text(encoding="utf-8") == "nb-token"
    if os.name == "posix":  # Windows has no POSIX mode bits
        assert (forwarded.stat().st_mode & 0o777) == 0o600
    # a named-but-absent forward var produces no file
    assert not (creds_dir / "ABSENT_VAR").exists()

    url = _transport(
        MCPServerSpec("b", "streamable-http", url="http://mcp.example.invalid/mcp"), None
    )
    assert url == "http://mcp.example.invalid/mcp"

    authed = _transport(
        MCPServerSpec("c", "streamable-http", url="http://mcp.example.invalid/mcp"), "tok"
    )
    assert authed != "http://mcp.example.invalid/mcp"  # a transport object, not the bare URL


def test_transport_rejects_missing_command_or_url() -> None:
    with pytest.raises(MCPClientError, match="no command"):
        _transport(MCPServerSpec("a", "stdio", command=None), None)
    with pytest.raises(MCPClientError, match="no url"):
        _transport(MCPServerSpec("b", "streamable-http", url=None), None)


def test_transport_needs_the_http_helper_for_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    import infra_assistant.mcp_client as mc

    monkeypatch.setattr(mc, "create_mcp_http_client", None)
    with pytest.raises(MCPClientError, match="header-capable HTTP client"):
        mc._transport(
            MCPServerSpec("c", "streamable-http", url="http://mcp.example.invalid/mcp"), "tok"
        )


def test_render_result_structured() -> None:
    result = SimpleNamespace(structured_content={"a": 1}, content=[], is_error=False)
    assert _render_result(result) == '{"a": 1}'


def test_render_result_text_blocks() -> None:
    result = SimpleNamespace(
        structured_content=None,
        content=[SimpleNamespace(text="line one"), SimpleNamespace(text="line two")],
    )
    assert _render_result(result) == "line one\nline two"


def test_render_result_truncates() -> None:
    huge = "x" * 20_000
    result = SimpleNamespace(structured_content=huge, content=[], is_error=False)
    rendered = _render_result(result)
    assert len(rendered) < 20_000
    assert "truncated" in rendered
