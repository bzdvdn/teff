# Reference: `graph.yaml`

This page is the complete field reference for the low-level `graph.yaml`
document — every node and every arrow explicit, with each key, its type, and
its default. It is the compiled form of everything the
[`flow.yaml`](flow-yaml.md) authoring layer (and the `Flow` builder) produces.

Workflows are loaded with `load_workflow`, validated with `teff validate`, and
serialized back out with `workflow_to_yaml` / `Flow.to_yaml()`.

> **Two formats, auto-detected.** A `flow.yaml` uses the idiom surface
> (`llm:`, `team:`, `map:`); a `graph.yaml` spells out `id`/`type`/`config` and
> `edges:`. The CLI and loaders detect the format with `looks_like_flow`, so
> `teff run` / `teff validate` / `teff graph` work on either without a flag.
> Prefer `flow.yaml` when writing by hand; treat this `graph.yaml` reference as
> the canonical vocabulary both formats compile to.

## Top-level keys

| Key | Type | Default | Purpose |
| --- | ---- | ------- | ------- |
| `name` | string | `""` | Optional label. |
| `description` | string | — | Optional free-text description. |
| `include` | string/list | — | Merge steps/edges/tools/state from other files (path or `{path, prefix}`), recursively. |
| `steps` | list | — | **Required.** Nodes, each `{id, type, config?, retry?}` (see [steps](#steps-nodes)). |
| `edges` | list | `[]` | Routing, each `{from, to, condition?}` (see [edges](#edges-routing)). |
| `tools` | list | `[]` | Tool instances made available to agents (see [tools](#tools-tool-registry)). |
| `state` | object | `{}` | `{schema, initial}` — seed values and per-key reducers (see [state](#state-initial-values-and-reducers)). |
| `plugins` | string/list | — | Extra node/tool modules to import (paths). |
| `plugins_folder` | string | `"plugins"` | Auto-loaded plugin folder. |
| `checkpoint` | object | — | Durable-runs block (see [checkpoint](#durable-runs-checkpoint)). |
| `hooks` | object | — | Named hook callbacks (see [hooks](#hook-events-hooks)). |
| `observability` | object | — | Full-run tracing block (see [tracing](#tracing-a-workflow-observability)). |
| `providers` | list | `[]` | Declared provider endpoints (see [providers](#custom-providers)). |
| `default_provider` | string | — | Provider for every step that doesn't name one. |
| `default_model` | string | — | Model for every step that doesn't name one. |

Every value is interpolated against the process environment — `${ENV_VAR}`
references are replaced; a variable that is not set stays as a literal
placeholder (see [interpolation](#the-env-interpolation)).

## Structure

```yaml
name: my-workflow            # optional label
description: ...             # optional

state:
  initial: {title: hello}    # seed values

plugins: [nodes]             # optional: extra node/tool modules to import
plugins_folder: plugins      # optional auto-loaded folder (default: plugins)

tools:                       # tool instances made available to agents
  - type: web_search

steps:                       # nodes
  - id: search
    type: web_search
    config: {query_key: q, output_key: results}
  - id: answer
    type: llm_chat
    config: {model: llama3.1:8b, input_key: results, output_key: reply}

edges:                       # routing
  - from: search
    to: answer
  - from: answer
    condition: results==""   # conditional: route on a state expression
    to: fallback
```

## `steps:` — nodes

Every step is `{id, type, config, retry}`. `id` and `type` are required;
`config` is node-specific (see the [node reference](../reference/nodes.md) for
every key); `retry` wraps the node (see [retrying](#retrying-failing-steps)).

Built-in `type` values:

| `type` | Purpose | Reference |
| ------ | ------- | --------- |
| `transform` | String/data transforms | [transform](../reference/nodes.md#transform) |
| `llm_chat` | One model call | [llm_chat](../reference/nodes.md#llm_chat) |
| `react_agent` | Tool-calling agent loop | [react_agent](../reference/nodes.md#react_agent-tool_exec) |
| `tool_exec` | Execute tool calls signalled by an agent | [tool_exec](../reference/nodes.md#react_agent-tool_exec) |
| `tool_call` | Invoke a registered tool with fixed args | [tool_call](../reference/nodes.md#tool_call) |
| `context_builder` | Compose a scratch prompt from state + conversation | [context_builder](../reference/nodes.md#context_builder-append_assistant) |
| `append_assistant` | Append the result as an assistant message | [append_assistant](../reference/nodes.md#context_builder-append_assistant) |
| `interrupt` | Pause for human input; resume via checkpoint | [interrupt](../reference/nodes.md#interrupt) |
| `supervisor` | Ask a model "which agent next" + guards | [supervisor](../reference/nodes.md#supervisor) |
| `gate` | Verdict → loop decider + retry budget | [gate](../reference/nodes.md#gate) |
| `validate` | Decode an interrupt answer into a loop decider | [validate](../reference/nodes.md#validate) |
| `command` | Declarative `goto`/`STOP` routing from state | [command](../reference/nodes.md#command) |
| `loop` | Repeat a body chain until a state condition holds | [loop](../reference/nodes.md#loop) |
| `parallel` | Concurrent branches, merged by reducers | [parallel](../reference/nodes.md#parallel-map) |
| `map` | Dynamic fan-out of a state list | [map](../reference/nodes.md#parallel-map) |
| `subflow` | Embed a complete inner graph as one node | [subflows](#nested-subflows-composite-agents) |
| `fallback` | Fill a field a model left empty | [fallback](../reference/nodes.md#fallback) |

Plugin nodes registered via `plugins:`/`plugins_folder:` are valid `type`
values too. An unknown type fails validation.

## `edges:` — routing

Each edge is `{from, to, condition}`. `from`/`to` name node ids (required);
`condition` routes on the state:

| `condition` | Routes when |
| ----------- | ----------- |
| *(omitted)* | always (the edge fires unconditionally) |
| `"key=value"` | `state["key"] == value` |
| `"key!=value"` | `state["key"] != value` |
| `"key=a,b"` | `state["key"]` is `a` or `b` (comma = OR) |
| `"key>=N"` / `"key<=N"` / `"key>N"` / `"key<N"` | numeric comparison |
| `"__error__"` | the source node raised an exception |

Edges reference existing step ids (validation error otherwise). The entry
point is the first step; an edge to a step already targeted is fine — first
matching edge wins during execution.

## Custom providers

Declare every provider exactly as it is configured in a top-level
`providers:` list. The block is the **single source of truth** — a provider
used by any step's `provider:`, by `default_provider:`, or by `default_model:`
must be declared here, and there is no implicit built-in fallback. Each entry
is a `{name, ...}` mapping that spells out the endpoint:

```yaml
providers:
  - name: vllm
    base_url: http://vllm:8000/v1           # type defaults to openai_compatible
  - name: claude-proxy
    type: anthropic_compatible              # Anthropic wire protocol
    base_url: http://proxy
    chat_path: /v1/messages
    api_key_env: CLAUDE_PROXY_KEY
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat

steps:
  - id: answer
    type: llm_chat
    config:
      model: meta-llama/Llama-3.3-70B-Instruct
      provider: vllm                        # must be declared in providers:
```

Bare preset-name strings are rejected — every provider is spelled out, so the
file says exactly what is configured. Each `name` must be unique, and only the
recognised provider fields may appear (a stray key is an error). Referencing a
provider name that is not declared here raises `ConfigError`.

The block round-trips through `workflow_to_yaml` / `Flow.to_yaml()`, and the
graph exposes it as `graph.providers` (a `{name: Provider}` map). In code you
can pass the same map straight to `graph.run(state, providers=...)`, which
overrides `graph.providers` for that run.

### Default provider for the whole workflow

A top-level `default_provider:` picks the default for every `llm_chat` /
`react_agent` step that doesn't name one — the YAML equivalent of
`Flow("...", default_provider=...)`:

```yaml
default_provider: ollama
default_model: llama3.1:8b
providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat
name: chat
steps:
  - id: answer
    type: llm_chat
    config: {}
```

`default_model:` supplies the model for steps that omit their own `model:`
(`LLM(model=...)` still wins). Neither `default_provider` nor `model` /
`default_model` resolved? The step raises `ConfigError`. Steps may still
override the default with their own `provider:` / `model:`.

### Provider keys

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `name` | string | — | Unique provider key referenced by `provider:`/`default_provider:`. |
| `type` | string | `openai_compatible` | `openai_compatible`, `anthropic_compatible`, or `ollama`. |
| `base_url` | string | — | Endpoint base URL. |
| `chat_path` | string | — | Chat endpoint path override. |
| `api_key_env` | string | — | Env var holding the API key. |
| `auth_header` / `auth_prefix` | string | — | Custom auth header / prefix. |
| `timeout` | number | — | Request timeout. |

## `state:` — initial values and reducers

The `state:` block seeds the graph state and declares how concurrent or
accumulating writers merge:

```yaml
state:
  initial: {topic: "…", messages: []}
  schema:
    messages:
      reducer: append
      type: list
    status:
      reducer: keep
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `initial` | mapping | `{}` | Seed values. Validated against `schema` (`ConfigError` on mismatch). |
| `schema` | mapping | `{}` | Per-key reducer declarations. |

### Reducers

Each `schema.<key>` is a mapping with a `reducer:` field (a bare string is
accepted too). Keys without a `reducer` default to `override`.

| `reducer` | Semantics |
| --------- | --------- |
| `override` | (default) The node's returned value replaces the key. |
| `append` | The node's returned value is appended into the existing list. |
| `keep` | The first write wins; later writes are ignored. |

Use `append` for shared lists (e.g. `messages`) so parallel branches and
supervisor turns accumulate instead of clobbering one another.

## `tools:` — tool registry

Tool instances made available to agents (and `tool_call` steps):

```yaml
tools:
  - type: web_search
  - type: python_eval
  - type: rag
    config: {store: sqlite, collection: docs}
```

Each entry is `{type, config}`; `config` is tool-specific. See the
[tools reference](../reference/tools.md) for every built-in tool and its
config keys. RAG tools resolve relative store paths against the workflow file.
An unknown tool type fails validation.

## Tracing a workflow (`observability:`)

A top-level `observability:` block turns on full-run tracing — topology,
per-node spans and the complete LLM prompt/response — without writing any
code. `teff run` and `teff daemon` pick it up automatically:

```yaml
name: my-workflow

observability:
  db: ./data/traces.db            # local SQLite store (our trace dashboard)
  export:                          # optional: also push to remote sinks
    - type: webhook               # any HTTP endpoint (e.g. our obs-server)
      url: http://obs:8001/obs/ingest
    - type: langfuse              # langfuse public API (Basic auth)
      host: https://cloud.langfuse.com
      public_key_env: LANGFUSE_PUBLIC_KEY
      secret_key_env: LANGFUSE_SECRET_KEY
    - type: langsmith             # langsmith runs API (x-api-key)
      api_key_env: LANGCHAIN_API_KEY
      project: my-project

steps:
  - id: answer
    type: llm_chat
    config: {model: llama3.1:8b, output_key: reply}
```

- `db:` resolves relative to the workflow file; `data/` is created if needed.
- Sinks are fanned out to **all** exporters at once (`CompositeExporter`); a
  failing sink is retried and then logged, never crashes the run.
- Secrets come from environment variables (`*_env`), never from the file.
- Browse the local store in the browser:
  `teff obs-server --db ./data/traces.db --port 8001` → `http://localhost:8001/obs/ui`.
- A remote sink that targets `teff obs-server` needs no API at all — pure YAML
  workflows push their traces over HTTP and the server renders the dashboard.

The same wiring is available in code via
`teff.observability.build_observability` / `build_observer_factory`.

## The `${ENV}` interpolation

Every value in the document is interpolated against the process environment.
A variable that is not set stays as a literal placeholder — nothing crashes:

```yaml
steps:
  - type: llm_chat
    config:
      api_key_env: ${OPENAI_API_KEY}
```

## Retrying failing steps

Any step can be wrapped with retry logic via a `retry:` block next to its
`config:`.  The block supports ``max_retries`` (attempts, default 3),
``delay`` (seconds between attempts, default 0), ``backoff`` (multiplier
per retry, default 1.0), ``timeout`` (per-attempt timeout), and
``retry_on`` — a list of exception type names or HTTP status codes; by
default every exception is retried.

| `retry:` key | Type | Default | Notes |
| ------------ | ---- | ------- | ----- |
| `enabled` | bool | `true` | `false` keeps the schema valid but disables the wrapper. |
| `max_retries` | int | `3` | Total attempts (including the first). |
| `delay` | number | `0` | Seconds before each retry. |
| `backoff` | number | `1.0` | Multiplier applied to `delay` per retry. |
| `timeout` | number | — | Per-attempt timeout. |
| `retry_on` | list | *all* | Exception type names or HTTP status codes; only these are retried. |

```yaml
steps:
  - id: search
    type: web_search
    config: {query_key: q, output_key: results}
    retry:
      max_retries: 4
      delay: 0.5
      backoff: 2.0        # delays: 0.5s, 1s, 2s, 4s
      timeout: 30
      retry_on: ["httpx.HTTPStatusError", 429]
```

Use ``retry: {enabled: false}`` to keep the schema valid but disable the
wrapper, and ``retry_on: [429]`` to only retry on that status code.  The
retry wrapper preserves the inner node's normal success/failure behaviour,
so ``__error__`` edges still fire after the final failed attempt.

## Nested subflows (composite agents)

A `subflow` step embeds a complete inner graph — the composite-agent pattern.
The inner graph is declared with the same `steps`/`edges` vocabulary; the outer
graph maps state into it with `input_map` and pulls results back with
`output_map`:

```yaml
steps:
  - id: greet
    type: transform
    config: {action: trim, input_key: text, output_key: text}
  - id: inner
    type: subflow
    config:
      input_map: {text: x}      # outer key → inner key
      output_map: {y: result}   # inner key → outer key
      max_iterations: 50
      graph:                    # the nested graph
        steps:
          - id: up
            type: transform
            config: {action: uppercase, input_key: x, output_key: y}
edges:
  - from: greet
    to: inner
```

Nested graphs validate against the same node registry (any built-in or plugin
type), support `retry:` per inner step, and round-trip through
`workflow_to_yaml`.  Without `input_map`/`output_map` the whole parent state is
passed through.

Alternatively, `config.build` reuses the `agent_step` recipe (context builder →
ReAct harness → append assistant) as a composite agent:

```yaml
steps:
  - id: chat
    type: subflow
    config:
      id_prefix: chat
      build:
        type: agent_step
        system: You are a helpful assistant
        output_key: answer
        model: llama3.1:8b
        messages_key: messages
        use_tools: all
```

## Parallel fan-out

A `parallel` step runs independent branches concurrently and merges their
results back via the state reducers.  Each branch is a single step mapping
_or_ a list of step mappings (run sequentially within the branch):

```yaml
steps:
  - id: fanout
    type: parallel
    config:
      branches:
        - {type: transform, config: {action: uppercase, input_key: q, output_key: web}}
        - [{type: transform, config: {action: count_lines, input_key: q, output_key: upper}},
           {type: transform, config: {action: uppercase, input_key: q, output_key: n}}]
edges:
  - from: fanout
    to: finish
```

Branches receive an isolated copy of the state and may run on the same
or different provider endpoints.  Use `state.schema ... reducer: append`
when branches should accumulate (e.g. collecting `messages`) instead of
overwriting a key.

| `parallel` config | Type | Default | Notes |
| ----------------- | ---- | ------- | ----- |
| `branches` | list | — | Each branch: a step mapping or a list of step mappings (sequential inside the branch). |
| `converge` | mapping | — | A single `{type: transform, config: ...}` step that re-joins the branches (only `transform` is supported in YAML). |

## Dynamic fan-out (`map`)

A `map` step fans a runtime state list out across parallel branches — branch
count comes from the data, not the file:

```yaml
steps:
  - id: summarize
    type: map
    config:
      input_keys: [chunks]
      output_key: summaries
      max_concurrency: 4
      processor:
        type: llm_chat
        config: {model: llama3.1:8b, input_key: chunk, output_key: summary}
```

| `map` config | Type | Default | Notes |
| ------------ | ---- | ------- | ----- |
| `input_keys` | string/list | — | State list key(s); multiple are zipped per index. |
| `output_key` | string | — | State key receiving the list of per-item results. |
| `processor` | mapping | — | The node applied per item (inline `{type, config}`). |
| `chunk_size` | int | `1` | Items per branch. |
| `max_concurrency` | int | — | Cap on concurrent branches. |

## Composing workflows (`include:`)

An `include:` block merges steps, edges, tools and state from other workflow
files — recursively, since an included file may include others.  Paths are
relative to the including file:

```yaml
name: composed
include:
  - path: ../shared/workflow.yaml
    prefix: shared_
  - path: ./retry.yaml
```

A `prefix:` (prepended to every included step id and edge endpoint) lets you
compose the same file into several places without id collisions; it is also
applied to `command` node `goto` targets.  Included steps run first, then the
including workflow's own steps.  Without a prefix, ids must not collide.

## Declarative routing (`command`)

A `command` step routes the graph from state — dynamic `goto` / `STOP`
without code.  The `when` expressions use the same language as `edges:`
conditions, and the first match wins:

```yaml
steps:
  - id: route
    type: command
    config:
      routes:
        - {when: "score >= 0.8", goto: approve}
        - {when: "score < 0.3", goto: reject}
      goto: review
      update: {routed: true}
```

`goto: STOP` terminates the run.

| `command` config | Type | Default | Notes |
| ---------------- | ---- | ------- | ----- |
| `routes` | list | `[]` | `{when, goto}` pairs; first match wins. |
| `goto` | string | — | Fallback target, or `STOP` to end the run. |
| `update` | mapping | — | State keys merged after routing. |

## Loops (`loop`)

A `loop` step repeats a `body` chain until `state[key]` equals `until` —
everything in one node, no decider edges.  `body` is a node or list of nodes
given as inline `type: ...` specs (like `map`'s processor):

```yaml
steps:
  - id: refine
    type: loop
    config:
      key: approved
      until: "да"
      max_rounds: 3
      body:
        - {type: transform, config: {action: value, value: "нет", output_key: approved}}
```

`max_rounds` (default 10) bounds the repetition; the condition uses the edges
expression language, so `until: "да"` matches `"Да"` or `"да."`.

| `loop` config | Type | Default | Notes |
| ------------- | ---- | ------- | ----- |
| `key` | string | — | State key the condition reads. |
| `until` | string | — | Value of `key` that stops the loop. |
| `max_rounds` | int | `10` | Maximum body rounds before giving up. |
| `body` | node/list | — | Chain run each round (inline `{type, config}` specs). |

## Validated interrupts (`strategy:`)

An `interrupt` step can validate the operator's answer with a `strategy:`
mapping instead of comparing it verbatim.  The loader expands it into the
classifier + `validate` chain (`{id}-validate`), the YAML counterpart of
`flow.interrupt(key, prompt, accept=...)`:

```yaml
steps:
  - id: gate
    type: interrupt
    config:
      key: approved
      prompt: "Approve the report? (yes / no)"
      strategy: {equals: да}          # or: any_of: [да, ок] | regex: "^[A-Z0-9]{4}$"
  - id: ship
    type: transform
    config: {action: value, value: shipped, output_key: status}
edges:
  - {from: gate, to: ship, condition: "decision=да"}
```

An `llm` strategy needs `model` and `provider`:

```yaml
      strategy:
        llm:
          system: Classify the answer as approval or rejection.
          user: "Answer: {approved}"
          schema:
            type: object
            properties: {ok: {type: boolean}}
          model: llama3.1:8b
          provider: ollama
```

Edges that would have sourced from the interrupt now source from
`{id}-validate`, where the decision key (`decision` by default) is written.

| `interrupt` config | Type | Default | Notes |
| ------------------ | ---- | ------- | ----- |
| `key` | string | — | State key receiving the resume value. |
| `prompt` | string | — | Question shown to the operator. |
| `strategy` | mapping | — | `{equals}` \| `{any_of: [...]}` \| `{regex}` \| `{llm: {system, user, schema, model, provider}}`, plus passthrough keys (`decision_key`, `pass_value`, `fail_value`, `value_key`, `verdict_key`, `ok_field`, `rounds_key`, `max_rounds`). |

Pausing raises `GraphInterrupt`; resume with the same `checkpoint_id` and
`resume={key: answer}`. Requires a checkpointer. See
[durable execution](durable.md).

## Durable runs (`checkpoint:`)

A top-level `checkpoint:` block enables durable runs whose state is saved
before every node, so an interrupted or crashed workflow resumes instead of
restarting:

```yaml
checkpoint:
  type: sqlite              # file | sqlite | sqlite_history | pg | pg_history
  path: data/checkpoints.db
```

| `checkpoint:` key | Type | Notes |
| ----------------- | ---- | ----- |
| `type` | string | `file` \| `sqlite` \| `sqlite_history` \| `pg` \| `pg_history`. |
| `path` | string | Store path, resolved relative to the workflow file (SQLite/file). |
| `dsn` | string | PostgreSQL connection string (PG variants). |
| `table` | string | — | Optional table name override (PG variants). |

`path` is resolved relative to the workflow file.  PG variants require
`dsn:` (+ optional `table:`).  Use `teff run --checkpoint-id <id>` (or the
`--checkpoint '...'` JSON flag to override the block per invocation).  This
is the same durable machinery the `teff` CLI and the conversational
turn/`Assistant` layer use.

## Hook events (`hooks:`)

Hooks observe node execution. Because they're Python callbacks, a workflow
_names_ hooks registered in a plugin — declare the plugin under `plugins:`,
register them with the `@hooks.hook` decorator, then reference by name:

```python
# plugins/telemetry.py
from teff import hooks


@hooks.hook("tick")
def tick(node_id, node, state, **kwargs):
    metrics.counter("graph.node", node_id=node_id, type=node.type)
```

```yaml
plugins: [plugins/telemetry.py]
hooks:
  on_node_start: tick          # (node_id, node, state)
  on_node_end: [tick, finalize] # also passes the node result
  on_node_error: on_error        # also passes the exception
```

Each key takes a hook-name string or a list; an unknown name fails
validation with a clear message.  Sync and async hooks are both supported
(`graph.run` awaits async ones).  The same `hooks=` mapping can be passed
programmatically.

| `hooks:` key | Callback signature |
| ------------ | ------------------ |
| `on_node_start` | `(node_id, node, state)` |
| `on_node_end` | `(node_id, node, state)` + the node result |
| `on_node_error` | `(node_id, node, state)` + the exception |

## Inspecting a graph

Render the topology back to YAML or as a Mermaid diagram:

```bash
teff graph workflow.yaml          # YAML topology
teff graph workflow.yaml --mermaid # Mermaid flowchart
```

The Mermaid output marks the entry point, annotates edges with their
conditions, and styles ``__error__`` edges distinctly — useful for
docs and review.

## Loading & validating

```python
from teff.yaml import load_workflow
from teff.yaml_schema import validate_workflow_file, format_errors

errors = validate_workflow_file("workflow.yaml")  # [] when ok
if errors:
    print(format_errors(errors, source="workflow.yaml"))

graph, tools, state, reducers = load_workflow("workflow.yaml")
result = await graph.run(state, tools=tools, reducers=reducers)
```

Both loaders auto-detect the document layer: a `flow.yaml`-style document is
routed through the flow compiler/validator, a low-level graph document through
the graph path (`validate_workflow_file` → `validate_flow` vs
`validate_workflow`; `load_workflow` → `load_flow` vs `load_workflow`).

The CLI wraps both:

```bash
teff validate workflow.yaml
teff -f workflow.yaml
```

## Exporting a code-built graph

Build with `Flow`, then serialize to a deployable workflow:

```python
from teff.yaml import workflow_to_yaml, graph_to_yaml

yaml_text = workflow_to_yaml(graph, tools=tools, initial=state, reducers=reducers)
# graph_to_yaml(graph) is a shorthand when you only need the graph itself
```

ReAct edges (`_tool_call_name !=`) round-trip correctly, so an agent built in
code can be emitted as declarative YAML.

## See also

- [`flow.yaml` reference](flow-yaml.md) — the authoring-layer document this
  format compiles from.
- [Choosing your abstraction](choosing-your-abstraction.md) — when to use which
  layer.
- [Node reference](../reference/nodes.md) — every `steps:` `type` and its config.
- [Tools reference](../reference/tools.md) — every `tools:` type and its config.
- [Providers reference](../reference/providers.md) — how provider resolution works.