"""Model Context Protocol (MCP) bridge and ready-made server presets.

Public surface of the MCP integration:

- :class:`~teff.tool.mcp.bridge.McpToolGroup` — a lazily-opened connection
  to one MCP server whose tools become ``<id>__<tool>`` members.
- :func:`~teff.tool.mcp.bridge.mcp_tools` /
  :func:`~teff.tool.mcp.bridge.open_tools` — connection helpers.
- :class:`~teff.tool.mcp.presets.McpPreset` and :data:`MCP_PRESETS` —
  ready-made launch configs for known servers; subclass ``McpPreset`` and
  it is registered automatically (each subclass needs a unique ``name``).
"""

from teff.tool.mcp.bridge import (
    McpTool,
    McpToolGroup,
    _connect_tools,
    mcp_tools,
    open_tools,
)
from teff.tool.mcp.presets import (
    FetchPreset,
    GitPreset,
    GmailPreset,
    GoogleCalendarPreset,
    GoogleDrivePreset,
    McpPreset,
    SQLitePreset,
    TimePreset,
)

MCP_PRESETS: dict[str, type[McpPreset]] = {
    cls.name: cls for cls in McpPreset.__subclasses__() if cls.name
}

__all__ = [
    "MCP_PRESETS",
    "FetchPreset",
    "GitPreset",
    "GoogleCalendarPreset",
    "GoogleDrivePreset",
    "GmailPreset",
    "McpPreset",
    "McpTool",
    "McpToolGroup",
    "SQLitePreset",
    "TimePreset",
    "_connect_tools",
    "mcp_tools",
    "open_tools",
]
