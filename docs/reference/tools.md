# Tools reference

Tools implement `Tool` (or use `@tool`) and are shared across nodes. Agents
receive them through `graph.run(state, tools=tools)` or YAML `tools:`. A
library of built-in tools registers itself when `teff.tool.builtin` is
imported (the YAML helpers and examples do this for you). Marked tools need
`pip install teff[tools]`.

## Registry

```python
from teff.tool.registry import default_tool_registry

print(default_tool_registry.list())  # all registered names
```

## Built-in tools

| Name | Class | Deps | What it does |
| ---- | ----- | ---- | ------------ |
| `calculator` | `CalculatorTool` | — | AST-based safe math evaluation |
| `shell` | `ShellTool` | — | Run shell commands behind a block/whitelist sandbox |
| `read_file` | `ReadFileTool` | — | Read a file's contents |
| `write_file` | `WriteFileTool` | — | Write content to a file |
| `edit_file` | `EditFileTool` | — | Replace text in a file |
| `list_dir` | `ListDirTool` | — | List files/dirs (optionally recursive) |
| `glob` | `GlobTool` | — | Find files matching a glob pattern |
| `getenv` | `GetEnvTool` | — | Read an env var (secret values masked) |
| `current_time` | `CurrentTimeTool` | — | Current date/time in an IANA timezone |
| `json_parse` | `JsonParseTool` | — | Parse and pretty-print JSON |
| `yaml_parse` | `YamlParseTool` | — | Parse YAML, dump as JSON |
| `kv_store` | `KVStoreTool` | — | Persistent JSON key-value store |
| `python_eval` | `PythonEvalTool` | — | Safe AST-whitelist evaluation of Python expressions |
| `web_search` | `WebSearchTool` | — | DuckDuckGo search, no API key |
| `fetch_url` | `WebFetchTool` | `beautifulsoup4` | Fetch a URL and extract its text |
| `read_pdf` | `PDFReadTool` | `pypdf` | Extract text from a PDF, page by page |
| `pdf` | `PDFTool` | `pypdf` | RAG-pageable PDF text extraction (RAG) |
| `image` | `ImageTool` | `httpx` | OCR via an OpenAI-compatible vision model |
| `s3_list` / `s3_get` / `s3_put` | `S3Tool`/`S3GetTool`/`S3PutTool` | `boto3` | S3 object ops |
| `slack_send` | `SlackSendTool` | `slack-sdk` | Send a message to a Slack channel |
| `sql_query` | `SQLQueryTool` | sqlite3/`psycopg` | Read-only SELECT (SQLite/PostgreSQL) |
| `sql_list_tables` | `SQLListTablesTool` | sqlite3/`psycopg` | List tables |
| `sql_describe` | `SQLDescribeTool` | sqlite3/`psycopg` | Describe a table's columns |
| `http_request` | `HttpRequestTool` | `httpx` | Arbitrary HTTP requests |
| `send_email` | `SendEmailTool` | smtplib | Send email via SMTP |
| `send_telegram` | `SendTelegramTool` | `httpx` | Send a message via a Telegram bot |
| `csv_query` | `CSVTool` | — | Query/aggregate a CSV |
| `git` | `GitTool` | — | git status/log operations |
| `github_list_open_prs` etc. | `GitHubTool` family | `httpx` | GitHub PR list/changes/comment/approve |
| `gitlab_list_open_mrs` etc. | `GitLabTool` family | `httpx` | GitLab MR list/changes/comment/approve |
| `lock` | `LockTool` | — | Distributed-ish lock acquire/release |
| `redis` | `RedisTool` | `redis` | Redis get/set/hash/list ops |
| `wait_for` | `WaitForTool` | — | Poll until a condition holds |
| `rag` | `RAGTool` | `teff[stores-*]` | Retrieval over a vector store |
| `rag_ingest` | `RAGIngestTool` | `teff[stores-*]` | Add documents to a vector store (chunk+embed+store) |
| `memory` | `MemoryTool` | `teff[stores-*]` | Long-term memory (remember/recall/forget) |
| `mcp` | `McpToolGroup` | `teff[mcp]` | Expose every tool of an MCP server (streamable-http or stdio); ready-made `preset`s for Google Drive / Gmail / Calendar |

## Model Context Protocol (MCP) servers

`type: mcp` is a special tool: it doesn't map onto a registry class but onto
a lazily-opened [MCP](https://modelcontextprotocol.io) server, whose tools
become available to agents as `Tool` instances named `<server_id>__<tool>`.

```yaml
tools:
  - type: mcp
    config:
      id: drive            # member tools become drive__ <tool>
      command: [uvx, mcp-server-google-drive]   # stdio server
  - type: mcp
    config:
      id: git
      url: http://localhost:8000/mcp             # streamable-http server
```

Exactly one of `command` (stdio argv, resolved) or `url` (streamable HTTP
endpoint) must be given; optional `env`, `cwd` and `client_info` keys are
accepted too and map 1:1 onto `McpToolGroup`'s constructor.

Instead of spelling out a server's launch command and env keys, use a
**preset** for a known server — the preset supplies the command (or url)
and its env-var keys, and your `env:` entries merge over the defaults
(same key overrides):

```yaml
tools:
  - type: mcp
    config:
      preset: google_drive
      env: {GOOGLE_DRIVE_REFRESH_TOKEN: "${GDRIVE_TOKEN}"}
```

Known presets (`MCP_PRESETS`) split into two launchers:

- **`npx` (Node.js) — Google servers.** `google_drive`, `gmail`,
  `google_calendar` run `npx @google/mcp-server-*` with the corresponding
  `GOOGLE_DRIVE_*`, `GOOGLE_GMAIL_*`, `GOOGLE_CALENDAR_*` env keys. Google
  distributes these only as npm packages, so Node.js must be installed.
- **`uvx` (Python) — self-hosted servers.** `git`, `fetch`, `time`,
  `sqlite` run `uvx mcp-server-*` (Python packages, no Node needed).

If the launcher (`npx`/`uvx`) is missing from `PATH`, opening the group
fails with a clear error instead of an opaque subprocess failure — see
`graph.aclose()`-adjacent errors, e.g.:

```
RuntimeError: cannot start MCP server 'google_drive': 'npx' is not
installed. ... Install Node.js ... or override the preset with a
python-based `command:`.
```

A `preset` plus explicit `command:`/`url:`/`id:`/`cwd:` overrides
fully replaces the preset's value for that key; `id` defaults to the preset
name. Validation accepts a config with either `preset` *or* one of
`url`/`command`.

The connection is opened lazily on first use and cached for the lifetime of
the graph, so daemon ticks and conversation turns reuse a single connection.
Close everything with `graph.aclose()` — or equivalently `async with
graph:`:

```python
graph, tools, state, reducers = load_workflow("app/flow.yaml")
async with graph:
    result = await graph.run(state, tools=tools)
```

In plain Python, `McpToolGroup` and `open_tools` give the same primitives
without YAML:

```python
from teff.tool import McpToolGroup, open_tools

async with open_tools(
    [
        McpToolGroup(id="git", command=["uvx", "mcp-server-git"]),
        McpToolGroup.from_preset(
            "google_drive",
            env={"GOOGLE_DRIVE_REFRESH_TOKEN": os.environ["GDRIVE_TOKEN"]},
        ),
    ]
) as tools:
    result = await graph.run(state, tools=tools)
```

## Configuring tools

Tools are plain classes, so you can construct them directly with keyword
arguments — `ShellTool(root_dir=..., allowed_commands=[...])`,
`WebSearchTool(provider="google")`, `SQLQueryTool({"db_type": "sqlite",
"path": "./v.db"})`. The registry (used by YAML `tools:` blocks) maps a config
dict onto the constructor: a dict passed to constructors that take a `config`
dict, or keyword arguments for keyword constructors:

```yaml
tools:
  - type: sql_query
    config: {db_type: sqlite, path: ./vectors.db}
  - type: shell
    config: {root_dir: /tmp, allowed_commands: [echo, ls]}
  - type: s3_list
    config: {bucket: my-bucket, region: eu-central-1, verify: false}
```

The same works in Python:

```python
from teff.tool.registry import default_tool_registry

sql = default_tool_registry.create("sql_query", {"db_type": "sqlite", "path": "./v.db"})
shell = default_tool_registry.create(
    "shell", {"root_dir": "/tmp", "allowed_commands": ["echo"]}
)
```

## Writing a custom tool

```python
from teff.tool.tool import Tool
from teff.tool.registry import tool


@tool("slugify", "Convert a string to a lowercase URL slug")
def slugify(text: str = "") -> str:
    return "-".join(text.lower().split())


# or a subclass
class Search(Tool):
    name = "search"
    description = "Search a local index"

    def run(self, query: str = "") -> str:
        return f"results for {query}"
```

See [Plugins](../guide/plugins.md) for discovery of custom tools.

## Security notes

- `shell` enforces a blocklist of dangerous commands plus an optional
  whitelist. It executes via `execve` (no `/bin/sh`), so shell operators
  (`&&`, `;`, pipes, backticks, `$(…)`) are never interpreted — any token
  containing shell metacharacters is rejected outright.
- `memory` fixes its namespace at construction time (`namespace=(...)` over
  the `tools:` config); an agent cannot switch namespaces mid-call, so one
  tool instance is scoped to one owner/tenant.
- `getenv` masks values whose names hint at credentials (`TOKEN`,
  `API_KEY`, `PASSWORD`, `DSN`, …) unless configured with
  `mask_secrets: false`.
- `sql_query` and the other SQL tools are read-only and reject
  `INSERT`/`UPDATE`/`DELETE`/DDL.
- `python_eval` only allows a whitelisted AST subset (`math.*`, builtins
  like `len`/`abs`/`sum`, comparisons).