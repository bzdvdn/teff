"""Ready-made launch configs for known MCP servers.

A :class:`McpPreset` fixes the launch ``command`` (stdio) or ``url``
(streamable HTTP) of a well-known server together with the env-var keys
that server expects.  ``McpToolGroup.from_preset`` / a YAML ``preset:``
build a group from one, then merge caller-provided ``env`` over the
preset's defaults.

The :data:`~teff.tool.mcp.MCP_PRESETS` registry (built in the package
``__init__``) collects every subclass automatically, so defining your own
preset here — or in your project by importing ``McpPreset`` — makes it
available as ``preset: <name>`` in YAML.
"""

from __future__ import annotations

from typing import ClassVar

__all__ = [
    "FetchPreset",
    "GitPreset",
    "GoogleCalendarPreset",
    "GoogleDrivePreset",
    "GmailPreset",
    "McpPreset",
    "SQLitePreset",
    "TimePreset",
]


class McpPreset:
    """Base class for launch configs of known MCP servers.

    Subclasses declare the canonical registry ``name`` plus either the
    stdio ``command`` (or streamable-http ``url``) and the ``env`` keys the
    server expects.  ``env`` values are overridable defaults;
    ``from_preset`` merges any caller-provided env over them (same key
    wins), and an explicit ``command``/``url`` fully replaces the preset's
    transport.
    """

    #: Canonical name used for ``preset:``/``from_preset``.
    name: ClassVar[str] = ""
    #: Stdio launch command (or set ``url`` instead).
    command: ClassVar[list[str] | None] = None
    #: Streamable-http endpoint (or set ``command`` instead).
    url: ClassVar[str | None] = None
    #: Env-var keys the server expects; values act as defaults.
    env: ClassVar[dict[str, str]] = {}
    #: One-line description of the server.
    description: ClassVar[str | None] = None


class GoogleDrivePreset(McpPreset):
    """Google Drive: file search, read, create, upload."""

    name = "google_drive"
    command = ["npx", "-y", "@google/mcp-server-google-drive"]
    env = {
        "GOOGLE_API_KEY": "",
        "GOOGLE_DRIVE_CLIENT_ID": "",
        "GOOGLE_DRIVE_CLIENT_SECRET": "",
        "GOOGLE_DRIVE_REFRESH_TOKEN": "",
    }
    description = "Google Drive file search, read, create, upload."


class GmailPreset(McpPreset):
    """Gmail: search, read, draft and send messages."""

    name = "gmail"
    command = ["npx", "-y", "@google/mcp-server-gmail"]
    env = {
        "GOOGLE_API_KEY": "",
        "GOOGLE_GMAIL_CLIENT_ID": "",
        "GOOGLE_GMAIL_CLIENT_SECRET": "",
        "GOOGLE_GMAIL_REFRESH_TOKEN": "",
    }
    description = "Gmail search, read, draft and send messages."


class GoogleCalendarPreset(McpPreset):
    """Google Calendar: read, create and update events."""

    name = "google_calendar"
    command = ["npx", "-y", "@google/mcp-server-google-calendar"]
    env = {
        "GOOGLE_API_KEY": "",
        "GOOGLE_CALENDAR_CLIENT_ID": "",
        "GOOGLE_CALENDAR_CLIENT_SECRET": "",
        "GOOGLE_CALENDAR_REFRESH_TOKEN": "",
    }
    description = "Google Calendar read, create and update events."


class GitPreset(McpPreset):
    """Git: read and edit Git repositories."""

    name = "git"
    command = ["uvx", "mcp-server-git"]
    description = "Read and edit Git repositories (via uvx)."


class FetchPreset(McpPreset):
    """Fetch: download web content as Markdown."""

    name = "fetch"
    command = ["uvx", "mcp-server-fetch"]
    description = "Download web content as Markdown (via uvx)."


class TimePreset(McpPreset):
    """Time / Timezone: current time and timezone conversion."""

    name = "time"
    command = ["uvx", "mcp-server-time"]
    description = "Current time and timezone conversion (via uvx)."


class SQLitePreset(McpPreset):
    """SQLite: inspect and query SQLite databases."""

    name = "sqlite"
    command = ["uvx", "mcp-server-sqlite"]
    description = "Inspect and query SQLite databases (via uvx)."
