"""Bridge to Model Context Protocol (MCP) servers.

MCP is an open standard for exposing tools, resources and prompts to LLMs.
This module connects a Teff graph to an MCP server and exposes its tools as
regular :class:`~teff.tool.Tool` instances, so they work anywhere the built-in
tools do — LLM nodes, the ReAct agent, tool registries.

    The ``mcp`` SDK is optional (install with ``teff[mcp]``); it is imported
    lazily so a plain ``import teff`` stays light and fast.

Usage::

    async with mcp_tools(url="http://localhost:8000/mcp") as tools:
        result = await graph.run(state, tools={t.name: t for t in tools})

    async with mcp_tools(command=["uvx", "mcp-server-git"]) as tools:
        result = await graph.run(state, tools={t.name: t for t in tools})
"""

from __future__ import annotations

import contextlib
import shutil
import sys
import typing
from collections.abc import AsyncIterator
from functools import partial

from teff.tool.mcp.presets import McpPreset
from teff.tool.tool import Tool

if typing.TYPE_CHECKING:
    from mcp.client.session import ClientSession
    from mcp.types import Tool as McpToolSpec

__all__ = ["McpTool", "McpToolGroup", "mcp_tools", "open_tools"]


def _format_mcp_content(result) -> str:
    """Flatten an MCP ``CallToolResult`` into a single string."""
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content.model_dump_json()
    parts: list[str] = []
    for block in getattr(result, "content", []):
        if getattr(block, "type", "") == "text":
            parts.append(block.text)
        else:
            parts.append(block.model_dump_json())
    return "\n".join(parts)


def _ensure_runtime(launcher: str, server_id: str) -> None:
    """Verify the stdio *launcher* (e.g. ``npx``/``uvx``) is installed.

    Presets like the Google servers are launched via ``npx`` (Node) while
    others use ``uvx`` (Python).  Fail with a clear message before
    spawning the subprocess, so a missing toolchain reads as a config
    problem instead of an opaque child-process error.

    Raises:
        RuntimeError: If *launcher* is not on ``PATH``.
    """
    if shutil.which(launcher):
        return
    if launcher in ("npx", "npm", "node"):
        hint = (
            f"server '{server_id}' is launched via '{launcher}' (a Node.js "
            "toolchain). Install Node.js (https://nodejs.org) so npx is on "
            "PATH, or override the preset with a python-based `command:`."
        )
    else:
        hint = (
            f"server '{server_id}' is launched via '{launcher}' but it is not on PATH."
        )
    raise RuntimeError(
        f"cannot start MCP server '{server_id}': '{launcher}' is not installed. {hint}"
    )


def _lookup_preset(name: str) -> type[McpPreset]:
    """Resolve *name* to a preset class.

    ``MCP_PRESETS`` (in the package ``__init__``) snapshots the presets
    known at import time; this also picks up subclasses defined later
    (e.g. in a user module) by scanning ``McpPreset.__subclasses__``.
    """
    from teff.tool.mcp import MCP_PRESETS

    try:
        return MCP_PRESETS[name]
    except KeyError:
        pass
    for cls in McpPreset.__subclasses__():
        if cls.name == name:
            return cls
    known = ", ".join(sorted(MCP_PRESETS))
    raise KeyError(f"unknown MCP preset {name!r} (known: {known})") from None


class McpTool(Tool):
    """A :class:`~teff.tool.Tool` that forwards calls to an MCP server.

    Instances are created by :func:`mcp_tools`.  The tool's JSON schema
    comes from the server's tool definition instead of being inferred
    from type hints.
    """

    def __init__(self, session: "ClientSession", spec: "McpToolSpec"):
        super().__init__()
        self._session = session
        self._server_name = spec.name
        self.name = spec.name
        self.description = spec.description or ""
        # SDK stubs expose `inputSchema`, runtime uses `input_schema`.
        self.schema = spec.input_schema  # type: ignore[attr-defined]

    async def arun(self, **kwargs):
        result = await self._session.call_tool(self._server_name, kwargs)
        content = _format_mcp_content(result)
        if getattr(result, "is_error", False):
            raise RuntimeError(
                content or f"MCP tool '{self._server_name}' returned an error"
            )
        return content


class McpToolGroup:
    """A lazily-opened MCP server connection, exposing its tools.

    Holds the connection *config* (url or command, env, cwd) without any
    live connection.  The session is opened on first :meth:`open` and kept
    open until :meth:`aclose`, so several ``graph.run`` calls (daemon ticks,
    conversation turns, resumes) share a single connection instead of
    re-spawning the server each time.

    Instances are created from a workflow's ``tools:`` block (``type: mcp``)
    or directly::

        group = McpToolGroup(id="drive", command=["uvx", "mcp-server-google-drive"])
        tools = await group.open()      # -> [McpTool, ...] named ``drive__<tool>``
        ...
        await group.aclose()

    Ready-made presets for known servers give the launch command and its
    env-var keys in one shot; overrides merge on top::

        group = McpToolGroup.from_preset(
            "google_drive",
            env={"GOOGLE_DRIVE_REFRESH_TOKEN": os.environ["GDRIVE"]},
        )

    Args:
        id: Server id; member tools are prefixed ``<id>__<name>`` so tools
            from different servers never collide.
        url: Streamable HTTP endpoint (mutually exclusive with *command*).
        command: Stdio server command, a list of argv tokens.
        env: Optional extra environment variables for stdio servers.
        cwd: Optional working directory for stdio servers.
        client_info: Optional dict overrides for the client
            ``Implementation`` advertised to the server.
    """

    def __init__(
        self,
        *,
        id: str = "mcp",
        url: str | None = None,
        command: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        client_info: dict | None = None,
    ):
        if (url is None) == (command is None):
            raise ValueError("McpToolGroup requires exactly one of 'url' or 'command'")
        self.id = id
        self._url = url
        self._command = command
        self._env = env
        self._cwd = cwd
        self._client_info = client_info
        self._stack: contextlib.AsyncExitStack | None = None
        self._tools: list[McpTool] | None = None
        self.is_mcp_group = self

    @classmethod
    def from_preset(
        cls,
        name: str,
        *,
        env: dict[str, str] | None = None,
        **overrides,
    ) -> "McpToolGroup":
        """Build a group from a named :data:`MCP_PRESETS` entry.

        The preset class supplies its canonical ``name`` (used as the
        default ``id``), the launch ``command`` (or ``url``) and its default
        ``env`` keys.  *env* entries merge over the preset's defaults (same
        key overrides); ``command``/``url``/``id``/``cwd`` overrides fully
        replace the preset's value.

        Raises:
            KeyError: If *name* is not a known preset.
        """
        preset: type[McpPreset] = _lookup_preset(name)
        if (preset.url is None) == (preset.command is None):
            raise ValueError(
                f"MCP preset {name!r} must define exactly one of 'url' or 'command'"
            )
        cfg: dict = {"id": preset.name}
        if preset.url is not None:
            cfg["url"] = preset.url
        else:
            cfg["command"] = preset.command
        if preset.env:
            cfg["env"] = dict(preset.env)
        cfg.update(overrides)
        merged_env = {**(cfg.get("env") or {}), **(env or {})}
        if merged_env:
            cfg["env"] = merged_env
        return cls(**cfg)

    async def open(self) -> list[McpTool]:
        """Return the server's tools, opening the connection on first use.

        Repeated calls return the cached member tools; the underlying
        session stays open until :meth:`aclose`.
        """
        if self._tools is not None:
            return self._tools
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamable_http_client

        stack = contextlib.AsyncExitStack()
        try:
            if self._url is not None:
                read_stream, write_stream = await stack.enter_async_context(  # type: ignore[misc]
                    streamable_http_client(self._url)
                )
            else:
                assert self._command is not None
                _ensure_runtime(self._command[0], self.id)
                params = StdioServerParameters(
                    command=self._command[0],
                    args=self._command[1:],
                    env=self._env,
                    cwd=self._cwd,
                )
                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(params)
                )
            session, member_tools = await _connect_tools(
                read_stream, write_stream, self._client_info
            )
        except Exception:
            await stack.aclose()
            raise
        stack.push_async_callback(partial(session.__aexit__, None, None, None))
        self._stack = stack
        for tool in member_tools:
            tool.name = f"{self.id}__{tool.name}"
        self._tools = member_tools
        return self._tools

    async def aclose(self) -> None:
        """Close the connection if it was opened.  Idempotent."""
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._tools = None


@contextlib.asynccontextmanager
async def open_tools(
    tools: list[Tool] | list[McpToolGroup],
) -> AsyncIterator[list[Tool]]:
    """Expand :class:`McpToolGroup` entries in *tools* into their members.

    Groups are opened on entry (so an ``async with open_tools(...)`` around
    a whole daemon loop keeps every server connected for its duration) and
    closed on exit.  Plain tools pass through untouched.

    Yields:
        A flat list of ready-to-call tools (group members replacing their
        groups).
    """
    opened: list[McpToolGroup] = []
    ready: list[Tool] = []
    try:
        for tool in tools:
            if isinstance(tool, McpToolGroup):
                opened.append(tool)
                ready.extend(await tool.open())
            else:
                ready.append(tool)
        yield ready
    finally:
        for group in opened:
            await group.aclose()


async def _list_all_tools(session: "ClientSession"):
    """List all tools from *session*, following pagination."""
    import mcp.types as types

    tools: list[McpToolSpec] = []
    cursor: str | None = None
    while True:
        params = (
            types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
        )
        result = await session.list_tools(params=params)
        tools.extend(result.tools)
        # SDK stubs omit `next_cursor`; runtime exposes it.
        cursor = result.next_cursor  # type: ignore[attr-defined]
        if not cursor:
            return tools


async def _connect_tools(read_stream, write_stream, client_info: dict | None = None):
    """Open a client session over *streams* and return ``(session, tools)``.

    The caller owns *session* and must close it (``await
    session.__aexit__(None, None, None)``) when done.
    """
    try:
        import mcp
        from mcp.client.session import ClientSession
    except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
        raise ModuleNotFoundError(
            "mcp support requires the optional dependency; install with "
            "'pip install teff[mcp]'"
        ) from exc

    impl_name = "teff"
    impl_version = "unknown"
    try:
        import importlib.metadata

        impl_version = importlib.metadata.version("teff")
    except Exception:
        pass
    info = mcp.types.Implementation(name=impl_name, version=impl_version)
    if client_info:
        info = info.model_copy(update=client_info)

    session = ClientSession(read_stream, write_stream, client_info=info)
    await session.__aenter__()
    try:
        await session.initialize()
        tools = [McpTool(session, spec) for spec in await _list_all_tools(session)]
    except Exception:
        await session.__aexit__(*sys.exc_info())
        raise
    return session, tools


@contextlib.asynccontextmanager
async def mcp_tools(
    url: str | None = None,
    command: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    client_info: dict | None = None,
) -> AsyncIterator[list[McpTool]]:
    """Connect to an MCP server and yield its tools as Teff :class:`Tool`\\s.

    Exactly one of *url* or *command* must be given:

    - ``url``: Streamable HTTP endpoint of an MCP server, e.g.
      ``http://localhost:8000/mcp``.
    - ``command``: Subprocess invocation for a stdio server, e.g.
      ``["uvx", "mcp-server-git"]`` or
      ``["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]``.

    The session stays open for the duration of the ``async with`` block;
    tools keep working until it exits, after which the connection is closed.

    Args:
        url: Streamable HTTP endpoint.
        command: Stdio server command (list of argv tokens).
        env: Optional extra environment variables for stdio servers.
        cwd: Optional working directory for stdio servers.
        client_info: Optional dict overrides for the client
            ``Implementation`` advertised to the server.

    Yields:
        A list of :class:`McpTool` instances, one per server tool.
    """
    from mcp import StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    if (url is None) == (command is None):
        raise ValueError("mcp_tools requires exactly one of 'url' or 'command'")

    stack = contextlib.AsyncExitStack()
    async with stack:
        if url is not None:
            # SDK stubs declare a wider tuple than runtime actually yields.
            read_stream, write_stream = await stack.enter_async_context(  # type: ignore[misc]
                streamable_http_client(url)
            )
        else:
            assert command is not None
            params = StdioServerParameters(
                command=command[0],
                args=command[1:],
                env=env,
                cwd=cwd,
            )
            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(params)
            )

        session, tools = await _connect_tools(read_stream, write_stream, client_info)
        stack.push_async_callback(partial(session.__aexit__, None, None, None))
        yield tools
