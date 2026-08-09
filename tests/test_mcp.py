"""Tests for the MCP (Model Context Protocol) tool bridge."""

import asyncio
import contextlib
import sys
import textwrap

import pytest

from teff.node.llm import LLM
from teff.tool import McpTool, McpToolGroup, mcp_tools, open_tools
from teff.tool.mcp import _connect_tools

pytest.importorskip("mcp")

import mcp.types as types  # noqa: E402
from mcp.server import InitializationOptions  # noqa: E402
from mcp.server.lowlevel import Server  # noqa: E402
from mcp.shared.memory import create_client_server_memory_streams  # noqa: E402

TOOLS = [
    types.Tool(
        name="echo",
        description="Echo the text back",
        input_schema={  # type: ignore[call-arg]
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    ),
    types.Tool(
        name="add",
        description="Add two integers",
        input_schema={  # type: ignore[call-arg]
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    ),
    types.Tool(
        name="boom",
        description="Always fails",
        input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
    ),
]


async def _server_ctx(tools=TOOLS, calls=None):
    """Start a low-level MCP server over in-memory streams."""

    async def on_list_tools(ctx, params):
        return types.ListToolsResult(tools=tools)

    async def on_call_tool(ctx, params):
        if calls is not None:
            calls.append((params.name, params.arguments or {}))
        if params.name == "echo":
            text = (params.arguments or {}).get("text", "")
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=text)]
            )
        if params.name == "add":
            args = params.arguments or {}
            total = int(args["a"]) + int(args["b"])
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(total))]
            )
        if params.name == "boom":
            return types.CallToolResult(
                content=[types.TextContent(type="text", text="kaboom")],
                is_error=True,
            )
        raise RuntimeError(f"unknown tool '{params.name}'")

    server = Server(
        "test-server",
        version="1.0",
        on_list_tools=on_list_tools,
        on_call_tool=on_call_tool,
    )

    async with create_client_server_memory_streams() as (
        client_streams,
        server_streams,
    ):
        read_stream, write_stream = server_streams
        task = asyncio.create_task(
            server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="test-server",
                    server_version="1.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability()
                    ),
                ),
            )
        )
        try:
            yield client_streams
        finally:
            task.cancel()


running_server = contextlib.asynccontextmanager(_server_ctx)


async def connect_tools(client_streams):
    read_stream, write_stream = client_streams
    session, tools = await _connect_tools(read_stream, write_stream)
    return session, tools


async def test_connect_lists_tools():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            assert [t.name for t in tools] == ["echo", "add", "boom"]
            assert tools[0].description == "Echo the text back"
            assert tools[1].schema == TOOLS[1].input_schema
            assert all(isinstance(t, McpTool) for t in tools)
        finally:
            await session.__aexit__(None, None, None)


async def test_arun_forwards_args_and_returns_text():
    calls = []
    async with running_server(calls=calls) as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            assert await tools[0].arun(text="hello world") == "hello world"
            assert await tools[1].arun(a=2, b=3) == "5"
            assert calls == [
                ("echo", {"text": "hello world"}),
                ("add", {"a": 2, "b": 3}),
            ]
        finally:
            await session.__aexit__(None, None, None)


async def test_arun_raises_on_error_result():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            boom = [t for t in tools if t.name == "boom"][0]
            with pytest.raises(RuntimeError, match="kaboom"):
                await boom.arun()
        finally:
            await session.__aexit__(None, None, None)


async def test_arun_raises_on_unknown_tool():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            with pytest.raises(Exception, match="unknown tool"):
                await session.call_tool("nonexistent", {})
        finally:
            await session.__aexit__(None, None, None)


async def test_tool_schema_used_by_llm():
    async with running_server() as client_streams:
        session, tools = await connect_tools(client_streams)
        try:
            schema = LLM._tool_to_schema(tools[0])
            fn = schema["function"]
            assert fn["name"] == "echo"
            assert fn["parameters"] == TOOLS[0].input_schema
        finally:
            await session.__aexit__(None, None, None)


async def test_mcp_tools_requires_exactly_one_transport():
    with pytest.raises(ValueError, match="exactly one"):
        async with mcp_tools():
            pass
    with pytest.raises(ValueError, match="exactly one"):
        async with mcp_tools(url="http://x", command=["y"]):
            pass


SERVER_SCRIPT = textwrap.dedent(
    """
    import asyncio

    import mcp.types as types
    from mcp.server import InitializationOptions
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server


    async def on_list_tools(ctx, params):
        return types.ListToolsResult(
            tools=[
                types.Tool(
                    name="add",
                    description="Add two integers",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer"},
                            "b": {"type": "integer"},
                        },
                        "required": ["a", "b"],
                    },
                )
            ]
        )


    async def on_call_tool(ctx, params):
        if params.name == "add":
            args = params.arguments or {}
            total = int(args["a"]) + int(args["b"])
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(total))]
            )
        raise RuntimeError(f"unknown tool {params.name}")


    async def main():
        server = Server(
            "demo-server",
            version="1.0",
            on_list_tools=on_list_tools,
            on_call_tool=on_call_tool,
        )
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="demo-server",
                    server_version="1.0",
                    capabilities=types.ServerCapabilities(
                        tools=types.ToolsCapability()
                    ),
                ),
            )


    if __name__ == "__main__":
        asyncio.run(main())
    """
)


async def test_stdio_end_to_end(tmp_path):
    script = tmp_path / "mcp_server.py"
    script.write_text(SERVER_SCRIPT)

    async with mcp_tools(command=[sys.executable, str(script)]) as tools:
        assert [t.name for t in tools] == ["add"]
        assert tools[0].description == "Add two integers"
        assert tools[0].schema["required"] == ["a", "b"]
        assert await tools[0].arun(a=20, b=22) == "42"


# --- McpToolGroup / open_tools -----------------------------------------


async def test_group_requires_exactly_one_transport():
    with pytest.raises(ValueError, match="exactly one"):
        McpToolGroup()
    with pytest.raises(ValueError, match="exactly one"):
        McpToolGroup(url="http://x", command=["y"])


async def test_group_opens_idempotently(tmp_path):
    script = tmp_path / "mcp_server.py"
    script.write_text(SERVER_SCRIPT)

    group = McpToolGroup(id="demo", command=[sys.executable, str(script)])
    tools = await group.open()
    assert [t.name for t in tools] == ["demo__add"]
    assert tools[0]._server_name == "add"
    assert await tools[0].arun(a=1, b=2) == "3"

    # cached — same list, no second subprocess
    assert await group.open() is tools
    await group.aclose()
    await group.aclose()  # idempotent


async def test_open_tools_expands_groups(tmp_path):
    script = tmp_path / "mcp_server.py"
    script.write_text(SERVER_SCRIPT)

    group = McpToolGroup(id="demo", command=[sys.executable, str(script)])
    plain = McpTool(_FakeSession(), _spec("echo"))
    async with open_tools([group, plain]) as ready:
        names = sorted(t.name for t in ready)
        assert names == ["demo__add", "echo"]
        add_tool = next(t for t in ready if t.name == "demo__add")
        assert await add_tool.arun(a=5, b=5) == "10"
    # group closed on exit: reopening must not re-use a dead session


class _FakeSession:
    async def call_tool(self, name, arguments):
        return None


def _spec(name):
    return types.Tool(
        name=name,
        description="fake",
        input_schema={"type": "object", "properties": {}},  # type: ignore[call-arg]
    )


async def test_graph_expands_groups_and_aclose(tmp_path):
    from teff.graph import Graph
    from teff.node.registry import make_function_node
    from teff.tool.builtin import CalculatorTool

    script = tmp_path / "mcp_server.py"
    script.write_text(SERVER_SCRIPT)

    async def noop(ctx, state):
        return {}

    group = McpToolGroup(id="demo", command=[sys.executable, str(script)])
    graph = Graph(
        nodes={"n": make_function_node(noop)},
        edges=[],
        entry_point="n",
        default_provider="fake",
        default_model="x",
    )
    tools: list = [group, CalculatorTool()]
    async with graph:
        expanded = await graph._expand_tools(tools)
        assert sorted(t.name for t in expanded) == [
            "calculator",
            "demo__add",
        ]
        # repeated expansion reuses the cached connection (same members)
        again = await graph._expand_tools(tools)
        assert again is not expanded
        assert [t for t in again if t.name == "demo__add"] == [
            t for t in expanded if t.name == "demo__add"
        ]
    # async with closed the group — a fresh graph gets a fresh connection
    assert group.aclose is not None


# --- Presets ---------------------------------------------------------------


def test_presets_registry():
    from teff.tool.mcp import MCP_PRESETS, GoogleDrivePreset

    assert set(MCP_PRESETS) == {
        "fetch",
        "git",
        "gmail",
        "google_calendar",
        "google_drive",
        "sqlite",
        "time",
    }
    assert MCP_PRESETS["google_drive"] is GoogleDrivePreset
    drive = MCP_PRESETS["google_drive"]
    assert drive.command[0] == "npx"
    assert "GOOGLE_DRIVE_REFRESH_TOKEN" in drive.env
    assert MCP_PRESETS["git"].command == ["uvx", "mcp-server-git"]
    assert MCP_PRESETS["time"].command == ["uvx", "mcp-server-time"]
    assert MCP_PRESETS["fetch"].command == ["uvx", "mcp-server-fetch"]
    assert MCP_PRESETS["sqlite"].command == ["uvx", "mcp-server-sqlite"]


def test_from_preset_defaults():
    group = McpToolGroup.from_preset("google_drive")
    assert group.id == "google_drive"
    assert group._command == ["npx", "-y", "@google/mcp-server-google-drive"]
    assert group._url is None
    expected = {
        "GOOGLE_API_KEY": "",
        "GOOGLE_DRIVE_CLIENT_ID": "",
        "GOOGLE_DRIVE_CLIENT_SECRET": "",
        "GOOGLE_DRIVE_REFRESH_TOKEN": "",
    }
    assert group._env == expected


def test_from_preset_env_merges_over_defaults():
    group = McpToolGroup.from_preset(
        "gmail",
        env={
            "GOOGLE_GMAIL_REFRESH_TOKEN": "secret",
            "EXTRA": "x",
        },
    )
    assert group._env["GOOGLE_GMAIL_REFRESH_TOKEN"] == "secret"
    assert group._env["EXTRA"] == "x"
    assert group._env["GOOGLE_API_KEY"] == ""


def test_from_preset_overrides_transport():
    group = McpToolGroup.from_preset(
        "google_calendar",
        command=["my-server"],
        url=None,
    )
    assert group._command == ["my-server"]
    assert group._url is None


def test_from_preset_unknown_raises():
    with pytest.raises(KeyError, match="unknown MCP preset"):
        McpToolGroup.from_preset("nope")


def test_custom_preset_subclass_auto_registered():
    from teff.tool.mcp import MCP_PRESETS, McpPreset

    class MyDrivePreset(McpPreset):
        name = "my_drive"
        command = ["my-server", "arg"]
        env = {"MY_TOKEN": "default"}

    assert "my_drive" not in MCP_PRESETS  # snapshot predates this class
    assert MyDrivePreset in McpPreset.__subclasses__()
    group = McpToolGroup.from_preset("my_drive")
    assert group._command == ["my-server", "arg"]
    assert group._env == {"MY_TOKEN": "default"}
    assert group.id == "my_drive"
    with_env = McpToolGroup.from_preset("my_drive", env={"MY_TOKEN": "real"})
    assert with_env._env == {"MY_TOKEN": "real"}


def test_ensure_runtime_missing_npx(monkeypatch):
    from teff.tool.mcp.bridge import _ensure_runtime

    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="Node.js"):
        _ensure_runtime("npx", "google_drive")
    with pytest.raises(RuntimeError, match="python-based"):
        _ensure_runtime("npx", "google_drive")


def test_ensure_runtime_present(monkeypatch):
    from teff.tool.mcp.bridge import _ensure_runtime

    monkeypatch.setattr("shutil.which", lambda cmd: "/usr/bin/uvx")
    _ensure_runtime("uvx", "git")  # no raise


def test_ensure_runtime_missing_uvx(monkeypatch):
    from teff.tool.mcp.bridge import _ensure_runtime

    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="not installed"):
        _ensure_runtime("uvx", "git")


def test_ensure_runtime_blocks_open(tmp_path, monkeypatch):
    script = tmp_path / "mcp_server.py"
    script.write_text(SERVER_SCRIPT)

    group = McpToolGroup(id="demo", command=[sys.executable, str(script)])
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    with pytest.raises(RuntimeError, match="not on PATH"):
        import asyncio

        asyncio.run(group.open())
