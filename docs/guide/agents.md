# Agents (ReAct loop)

Build a tool-calling agent loop with `flow.react()` — the LLM and tool
executor stay visible as graph topology, so the loop is inspectable and can
be followed by more nodes:

```python
from teff.flow import Flow
from teff.node import Transform
from teff.provider import ProviderRegistry
from teff.tool import Tool


class Search(Tool):
    name = "search"
    description = "Search a local index"

    def run(self, query: str = "") -> str:  # type: ignore[override]
        return f"results for {query}"


flow = Flow(
    "agent",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.react(system="Answer using tools.", input_key="query", output_key="answer")
flow.step(Transform(action="uppercase", input_key="answer", output_key="result"))

graph = flow.compile()
result = await graph.run({"query": "teff"}, tools=[Search()], max_iterations=10)
```

When the model requests tools, the executor runs them **all in parallel** in
one round (`asyncio.gather`) and loops back. When it answers without a tool
call, the response lands at `output_key` and execution continues. `max_iterations`
on `graph.run()` caps the loop.

## `flow.harness()`

The full API (an alias, `flow.react()` is kept for backwards compatibility):

- `max_tool_rounds` — max model calls per graph visit (default 10).
- `tool_error_mode` — `"message"` (default: a failed tool becomes a `tool`
  message the model can react to) or `"raise"` (the failure propagates, so
  you can route it with a `Graph` `__error__` edge to a fallback).
- `parse_text_tool_calls` — decode tool calls embedded in plain text, for
  local models that skip the structured `tool_calls` field (default True).
- `temperature` / `max_tokens` / `response_format` — sampling knobs.
- `tool_timeout` / `tool_retries` — bound each tool call with a timeout and
  re-attempt it on failure.
- `tool_approval` — gate risky tools. A callable (or `"interactive"`) can
  `approve`/`deny`; returning `"pause"` raises a `GraphInterrupt` so a human
  can sign off and resume with `resume={"tool_approval": value}`.
- `max_retries` / `fallbacks` / `retry_on` — retry failed model calls with
  backoff and fail over to backup models.
- `max_total_tokens` — stop the agent once the token budget is spent.
- `max_context_tokens` / `trim_messages` — trim the conversation before each
  model call to stay under a context limit.
- `stream` / `on_token` — stream LLM tokens.

```python
flow.harness(
    model="llama3.1:8b",
    input_key="query",
    output_key="answer",
    max_tool_rounds=5,
    tool_error_mode="raise",
    temperature=0.2,
    tool_timeout=30,
    tool_approval="interactive",
    max_retries=3,
    fallbacks=["llama3.1:8b"],
)
```

## MCP tools

Connect any [Model Context Protocol](https://modelcontextprotocol.io) server
and its tools become ordinary `Tool` instances — no `graph.run` changes.
Requires the `mcp` package (bundled with the core install, imported lazily):

```python
from teff.flow import Flow
from teff.provider import ProviderRegistry
from teff.tool import mcp_tools

flow = Flow(
    "agent",
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.react(input_key="query", output_key="answer")
graph = flow.compile()

async with mcp_tools(command=["uvx", "mcp-server-git"]) as tools:
    # or mcp_tools(url="http://localhost:8000/mcp")
    result = await graph.run(
        {"query": "What changed in the last commit?"},
        tools=tools,
        max_iterations=10,
    )
```

`command` starts a stdio server (split into argv), `url` connects to a
Streamable HTTP endpoint. The session stays open for the `async with` block.
A runnable pair lives in [`examples/mcp/`](https://github.com/bzdvdn/teff/tree/master/examples/mcp/).

## Multi-agent supervisors

For routing between *several* specialist agents under one decider, use
`flow.route()` with the `agent_step()` helper — see
[Multi-agent supervisors](supervisors.md).