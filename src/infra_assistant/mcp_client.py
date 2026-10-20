"""Connect to one or more MCP servers and expose their tools as one flat set.

Tool names are namespaced ``<server>__<tool>`` so two servers can't collide and
a call always routes back to the right one. The namespace is stripped again on
dispatch. Server names are constrained (in ``config``) to
``[a-z0-9][a-z0-9_-]*`` so the prefixed name still satisfies the usual
function-name rule (``^[A-Za-z0-9_-]+$``).

Every tool here is assumed read-only — that guarantee lives in the MCP server
(``homelab-mcp`` enforces it four ways). This client does not re-check it; it
just never *offers* a write path of its own.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Callable, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import mcp
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters
from mcp.client.streamable_http import streamable_http_client

from . import credentials
from .config import MCPServerSpec
from .errors import MCPClientError

try:  # a private SDK helper — the only way to pass auth headers to the HTTP transport
    from mcp.shared._httpx_utils import create_mcp_http_client
except ImportError:  # pragma: no cover - SDK layout changed
    create_mcp_http_client = None  # type: ignore[assignment]

logger = logging.getLogger("infra_assistant.mcp")

NAMESPACE_SEP = "__"
#: The only variables a spawned stdio MCP server inherits unconditionally — the
#: bare minimum for a Python process to run. Everything else it needs must be
#: named in the server's ``forward_env`` (see :class:`MCPServerSpec`). This keeps
#: a co-located server out of the bot's secret set: it never sees the Telegram
#: token, the LLM keys, a cache auth password, or anything else it did not ask for.
_CHILD_ENV_ALLOW = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "TMPDIR",
    "PYTHONHOME",
    "PYTHONPATH",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "SYSTEMROOT",  # Windows: some stdlib calls need it
)
#: Cap on a single tool result folded back into the conversation, in characters.
#: The agent additionally enforces a cumulative per-turn budget.
_RESULT_CHAR_CAP = 16_000
_DEFAULT_TOOL_CALL_TIMEOUT = 30.0


@runtime_checkable
class ToolBackend(Protocol):
    """The slice of :class:`MCPPool` the agent depends on (kept narrow for tests)."""

    @property
    def tool_count(self) -> int: ...

    def tools(self) -> list[dict[str, Any]]: ...

    async def call(self, prefixed_name: str, arguments: dict[str, Any]) -> str: ...


@dataclass(frozen=True)
class RemoteTool:
    server: str
    bare_name: str
    prefixed_name: str
    description: str
    input_schema: dict[str, Any]


#: ``(spec, token) -> whatever mcp.Client accepts`` (StdioServerParameters, a
#: URL string, or a transport). Injectable so tests connect in-process.
TransportFactory = Callable[[MCPServerSpec, str | None], Any]


#: Sub-directory of the unit's ``RuntimeDirectory`` that holds one credential
#: directory per stdio child.
_CHILD_CREDS_DIRNAME = "mcp-child-creds"


def _child_creds_root() -> Path:
    """Where per-child credential directories are created: the unit's
    ``RuntimeDirectory`` (a 0700 ``tmpfs``, cleared on stop) under systemd, and
    a self-cleaning temp directory otherwise (tests, a foreground run)."""
    runtime = os.environ.get("RUNTIME_DIRECTORY")
    if runtime:
        root = Path(runtime) / _CHILD_CREDS_DIRNAME
    else:
        root = Path(tempfile.mkdtemp(prefix="ia-mcp-creds-"))
        atexit.register(shutil.rmtree, root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    return root


def _materialise_child_credentials(spec: MCPServerSpec) -> Path | None:
    """Write the ``spec.forward_env`` secrets this process can resolve into a
    fresh 0700 directory of 0600 ``NAME`` files, and return it. The child reads
    them via ``$CREDENTIALS_DIRECTORY`` (the same contract systemd uses for
    ``LoadCredential=``), so the values never enter its environment. Returns
    ``None`` if the server forwards nothing, or nothing resolved."""
    if not spec.forward_env:
        return None
    source = credentials.default_credential_source()
    resolved = {name: source[name] for name in spec.forward_env if name in source}
    if not resolved:
        return None
    child_dir = _child_creds_root() / spec.name
    if child_dir.exists():
        shutil.rmtree(child_dir)
    child_dir.mkdir(parents=True)
    child_dir.chmod(0o700)
    for name, value in resolved.items():
        fd = os.open(str(child_dir / name), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, value.encode("utf-8"))
        finally:
            os.close(fd)
    return child_dir


def _child_env(spec: MCPServerSpec) -> dict[str, str]:
    """The environment for a spawned stdio MCP server: a minimal base, a
    ``$CREDENTIALS_DIRECTORY`` pointer to a directory of 0600 credential files
    for the vars named in ``spec.forward_env``, plus literal ``spec.env``
    overrides.

    Deliberately **not** ``os.environ`` — a co-located server must not inherit
    the bot's Telegram token, LLM keys or any other resolved secret it did not
    name. And the forwarded secrets go as files, never as environment values, so
    they never reach the child's ``/proc/<pid>/environ``. The
    SDK does not merge anything onto what we pass, so the base must be here.
    """
    env = {name: os.environ[name] for name in _CHILD_ENV_ALLOW if name in os.environ}
    creds_dir = _materialise_child_credentials(spec)
    if creds_dir is not None:
        env["CREDENTIALS_DIRECTORY"] = str(creds_dir)
    env.update(spec.env)  # explicit TOML literals win
    return env


def _transport(spec: MCPServerSpec, token: str | None) -> Any:
    """Build the argument to hand to :class:`mcp.Client` for one server."""
    if spec.transport == "stdio":
        if spec.command is None:  # config guarantees this; belt-and-braces
            raise MCPClientError(f"stdio server {spec.name!r} has no command")
        return StdioServerParameters(
            command=spec.command,
            args=list(spec.args),
            env=_child_env(spec),
        )

    if spec.url is None:  # config guarantees this; belt-and-braces
        raise MCPClientError(f"{spec.transport} server {spec.name!r} has no url")
    headers = {"Authorization": f"Bearer {token}"} if token else None
    if spec.transport == "sse":
        return sse_client(spec.url, headers=headers)
    # streamable-http
    if headers:
        if create_mcp_http_client is None:
            raise MCPClientError(
                f"server {spec.name!r} needs a bearer token over streamable-http, but this "
                "MCP SDK build no longer exposes the header-capable HTTP client helper"
            )
        return streamable_http_client(spec.url, http_client=create_mcp_http_client(headers=headers))
    return spec.url


class MCPPool:
    """Manages the lifecycle of every configured MCP client connection."""

    def __init__(
        self,
        servers: Sequence[MCPServerSpec],
        secrets: dict[str, str],
        *,
        transport_factory: TransportFactory = _transport,
        tool_call_timeout: float = _DEFAULT_TOOL_CALL_TIMEOUT,
    ) -> None:
        self._servers = list(servers)
        self._secrets = secrets
        self._transport_factory = transport_factory
        self._tool_call_timeout = tool_call_timeout
        self._stack: AsyncExitStack | None = None
        self._clients: dict[str, mcp.Client] = {}
        self._tools: dict[str, RemoteTool] = {}

    # -- lifecycle ----------------------------------------------------

    async def open(self) -> None:
        """Connect to every server and enumerate its tools. Fails closed."""
        if self._stack is not None:
            raise MCPClientError("MCP pool is already open")
        stack = AsyncExitStack()
        try:
            for spec in self._servers:
                token = self._secrets.get(spec.token_env) if spec.token_env else None
                try:
                    client = mcp.Client(self._transport_factory(spec, token))
                    await stack.enter_async_context(client)
                    listed = await client.list_tools()
                except Exception as exc:
                    raise MCPClientError(
                        f"could not connect to MCP server {spec.name!r}: {exc}"
                    ) from exc
                self._clients[spec.name] = client
                self._register_tools(spec.name, listed.tools)
            logger.info(
                "connected to %d MCP server(s), %d tool(s) total",
                len(self._clients),
                len(self._tools),
            )
        except BaseException:
            await stack.aclose()
            self._clients.clear()
            self._tools.clear()
            raise
        self._stack = stack

    async def aclose(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._clients.clear()
        self._tools.clear()

    async def __aenter__(self) -> MCPPool:
        await self.open()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    # -- introspection ----------------------------------------------

    def _register_tools(self, server: str, tools: Sequence[Any]) -> None:
        for tool in tools:
            prefixed = f"{server}{NAMESPACE_SEP}{tool.name}"
            schema = getattr(tool, "input_schema", None) or {"type": "object", "properties": {}}
            self._tools[prefixed] = RemoteTool(
                server=server,
                bare_name=tool.name,
                prefixed_name=prefixed,
                description=getattr(tool, "description", "") or "",
                input_schema=dict(schema),
            )

    def tools(self) -> list[dict[str, Any]]:
        """Every tool, as ``{name, description, input_schema}`` dicts."""
        return [
            {
                "name": t.prefixed_name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def server_summary(self) -> list[tuple[str, int]]:
        counts: dict[str, int] = dict.fromkeys(self._clients, 0)
        for tool in self._tools.values():
            counts[tool.server] = counts.get(tool.server, 0) + 1
        return sorted(counts.items())

    # -- dispatch --------------------------------------------------

    async def call(self, prefixed_name: str, arguments: dict[str, Any]) -> str:
        """Invoke a tool by its namespaced name; return a text rendering.

        A tool error or an unexpected exception is returned as an ``ERROR: …``
        string rather than raised — the agent feeds it back to the model, which
        must not crash the turn.
        """
        tool = self._tools.get(prefixed_name)
        if tool is None:
            return f"ERROR: no such tool {prefixed_name!r}"
        client = self._clients[tool.server]
        try:
            result = await asyncio.wait_for(
                client.call_tool(tool.bare_name, arguments), timeout=self._tool_call_timeout
            )
        except TimeoutError:
            logger.warning("tool %s timed out after %ss", prefixed_name, self._tool_call_timeout)
            return f"ERROR: tool {prefixed_name!r} timed out after {self._tool_call_timeout:g}s"
        except Exception:
            # Log the detail; never fold a raw exception (which may carry a
            # subprocess dump or an upstream body) back into the conversation.
            logger.warning("tool %s raised", prefixed_name, exc_info=True)
            return f"ERROR: tool {prefixed_name!r} failed (see bot log)"

        rendered = _render_result(result)
        if getattr(result, "is_error", False):
            return f"ERROR: {rendered}"
        return rendered


def _render_result(result: Any) -> str:
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        text = json.dumps(structured, default=str, ensure_ascii=False)
    else:
        parts = [
            getattr(block, "text", "")
            for block in getattr(result, "content", []) or []
            if getattr(block, "text", "")
        ]
        text = "\n".join(parts) if parts else "(no content)"
    if len(text) > _RESULT_CHAR_CAP:
        text = text[:_RESULT_CHAR_CAP] + f"\n… (truncated at {_RESULT_CHAR_CAP} chars)"
    return text
