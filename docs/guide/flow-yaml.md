# Reference: `flow.yaml`

The authoring-layer document. A `flow.yaml` describes *how* the app should
behave — teams, chains, gates — without spelling out every node and edge. It
compiles to a `Flow` and then to a regular `Graph`, and can be exported as the
compiled `graph.yaml` artifact:

```
flow.yaml ──compile()──► Flow ──compile()──► Graph
                            └── to_yaml() ──► graph.yaml
```

Choose `flow.yaml` when you want the concise idiom surface (`team:`, `map:`,
`loop:`); choose [`graph.yaml`](yaml-workflows.md) when you want every node and
arrow explicit. Both formats are auto-detected by the CLI and the loaders
(`looks_like_flow`), so `teff run`, `teff validate` and `teff graph` work on
either without any flag.

This page is the complete field reference. Read it top to bottom once, then use
it to find *where to change what*: each key below names the exact YAML key, its
type, its default, and which code path consumes it.

---

## Top-level keys

| Key | Type | Default | Purpose |
| --- | ---- | ------- | ------- |
| `name` | string | `""` | Optional label; used as the graph name when exporting. |
| `description` | string | — | Optional free-text description. |
| `default_provider` | string | — | Provider name every LLM/agent step falls back to when it has no own `provider:`. Must be declared in `providers:`. |
| `default_model` | string | — | Model name every LLM/agent step falls back to when it has no own `model:`. A step's own `model:` always wins. |
| `providers` | list | `[]` | Declared provider endpoints (see [providers](#providers)). |
| `tools` | list | `[]` | Tool instances available to agents (see [tools](#tools)). |
| `state` | object | `{}` | `{schema, initial}` — initial values and per-key reducers (see [state](#state)). |
| `steps` | list | — | **Required.** The idiom steps (see below). Empty → `ConfigError`. |
| `plugins` | string/list | — | Extra node/tool modules to import (paths). |
| `plugins_folder` | string | `"plugins"` | Auto-loaded plugin folder. |
| `checkpoint` | object | — | Durable-runs block (see [checkpoint](#checkpoint)). |
| `hooks` | object | — | Named hook callbacks (see [hooks](#hooks)). |
| `observability` | object | — | Full-run tracing block (see [observability](#observability)). |
| `include` | string/list | — | Merge steps/edges/tools/state from other files. Each entry is a path or `{path, prefix}`. Relative to the including file. |

Every value in the document is interpolated against the process environment —
`${ENV_VAR}` references are replaced; a variable that is not set stays as a
literal placeholder.

The `state:`, `tools:`, `providers:`, `plugins:`, `checkpoint:`, `hooks:`,
`observability:` and `include:` blocks are **pass-through** — they behave
exactly as they do in [`graph.yaml`](yaml-workflows.md).

---

## `steps:` — the idiom surface

Every entry in `steps:` is a **single-key mapping**: the key is the idiom name,
the value is a mapping of that idiom's options:

```yaml
steps:
  - llm: {id: replier, system: "...", output_key: answer}
  - transform: {id: shout, action: uppercase, input_key: answer, output_key: shout}
```

The full idiom set:

| Idiom | What it builds | See |
| ----- | -------------- | --- |
| `llm:` | One model call | [llm](#llm) |
| `transform:` | String/data transform | [transform](#transform) |
| `context_builder:` | Compose a plain-text agent input from state | [context_builder](#context_builder) |
| `append_assistant:` | Append a reply back to the conversation | [append_assistant](#append_assistant) |
| `agent:` / `agent_step:` | One routed ReAct agent (context builder → harness → append) | [agent](#agent-agent_step) |
| `team:` | Supervised team: leader decider + routed roles | [team](#team) |
| `supervisor:` | Native supervisor decider (with optional `agents:` loop wiring) | [supervisor](#supervisor) |
| `supervise:` | Route an existing decider to agent chains (advanced) | [supervise](#supervise) |
| `parallel:` | Concurrent branches + optional converge | [parallel](#parallel) |
| `map:` | Dynamic fan-out over a state list | [map](#map) |
| `loop:` | Repeat a body until a state value matches | [loop](#loop) |
| `interrupt:` | Pause for human input, optional validation | [interrupt](#interrupt) |
| `branch:` | Conditional routing on a state value | [branch](#branch) |
| `route:` | Declarative `when/goto` routing from state | [route](#route) |
| `type:` | An arbitrary registered node type as one step | [type](#type) |

An unknown idiom raises `ConfigError` listing the supported set. LLM-bearing
idioms need a `model:`/`provider:` — per step, or via `default_model:` /
`default_provider:` at the top (otherwise `ConfigError`).

---

## `llm:`

Compiles to an [`LLM`](../reference/nodes.md#llm_chat) node. All `LLM` constructor keys are
accepted verbatim.

```yaml
- llm:
    id: replier
    system: "You are a helpful assistant."
    prompt: "User said: {input}"
    output_key: answer
    model: llama3.1:8b          # optional if default_model set
    provider: ollama            # optional if default_provider set
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `llm_N` | Node id in the compiled graph. |
| `model` | string | `default_model` | Required when no `default_model`. |
| `provider` | string | `default_provider` | Required when no `default_provider`. |
| `system` | string | — | System prompt; supports `{key}` templates. |
| `prompt` | string | — | User prompt with `{key}` templates. |
| `input_key` | string | — | Read a single state key as the user message. |
| `output_key` | string | `"output"` | Where the reply lands. |
| `json_schema` / `output_type` | dict/type | — | Structured output validation. |
| `parse` | bool | `false` | Parse the reply as JSON into a dict (no validation). |
| `use_tools` | bool/list | — | Tool scope for the model. |
| `skills` / `skill_dir` | list/str | — | Mount skills onto the call. |
| `temperature` / `max_tokens` | float/int | — | Sampling knobs. |
| `max_retries` / `fallbacks` | int/list | — | Retry + model failover. |
| `cache` | bool | `false` | Dedupe identical calls. |
| `memory` | object | — | Long-term memory config. |
| any other `LLM` key | — | — | Forwarded to `LLM(**config)`. |

---

## `transform:`

Compiles to a [`Transform`](../reference/nodes.md#transform) node. All `Transform` keys are
accepted verbatim.

```yaml
- transform:
    id: count
    action: count_lines
    input_key: text
    output_key: lines
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `transform_N` | Node id. |
| `action` | string | — | One of the transform actions (uppercase, lowercase, trim, count_lines, value, render, json_get, append, contains, compare, split, join, replace, coalesce, pick, to_int, to_float, now). |
| `input_key` | string | `""` | State key to read from. |
| `output_key` | string | `""` | State key to write to. |
| `value` | string | — | Literal value (`action: value`), needle (`contains`), RHS (`compare`), `coalesce` fallback. |
| `field` | string | — | Field for `json_get` / `pick`. |
| `template` | string | — | `{key}` template for `render` / `append`. |
| `raw` | bool | `false` | Keep `json_get`/`pick` values without stringifying. |
| `sep` / `op` / `old` / `new` | — | — | Action-specific options (see [transform](../reference/nodes.md#transform)). |

---

## `context_builder:`

Compiles to a [`ContextBuilder`](../reference/nodes.md#context_builder-append_assistant)
node. Renders each configured section as `<label>:\n<value>` plus the latest
`user` message into a plain-text `output_key`, and clears scratch keys so a
routed agent starts clean. This is the first half of the `agent_step` turn
routing pair.

```yaml
- context_builder:
    id: compose
    sections:
      plan: "Plan"
      summary: "Summary"
    messages_key: messages
    output_key: input
    reset_keys: [scratch]
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `context_builder_N` | Node id. |
| `sections` | mapping | `{}` | State key → section label, rendered as `<label>:\n<value>`. List values are joined with newlines. |
| `messages_key` | string | `messages` | State key holding the conversation; its latest `user` message is appended as `User: ...`. |
| `output_key` | string | `input` | State key receiving the composed text. |
| `reset_keys` | list | `[]` | Scratch state keys to clear (reset to `[]`) before the agent runs. |

## `append_assistant:`

Compiles to an [`AppendAssistant`](../reference/nodes.md#context_builder-append_assistant)
node — the second half of the turn routing pair. Copies `state[output_key]`
back into the shared conversation as an `assistant` message.

```yaml
- append_assistant: {output_key: poem, messages_key: messages}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `append_assistant_N` | Node id. |
| `output_key` | string | `draft` | State key holding the agent's reply. |
| `messages_key` | string | `messages` | State key of the conversation to append to. |

---

## `agent:` / `agent_step:`

Compiles to a [`SubFlow`](../reference/nodes.md) running the `agent_step` recipe:
**context builder → ReAct harness → append assistant**. One routed agent whose
final answer lands in `output_key` and is copied into the shared conversation.
`agent:` and `agent_step:` are aliases.

```yaml
- agent:
    id: coder
    system: "You write code."
    output_key: code
    model: llama3.1:8b
    provider: ollama
    use_tools: [python_eval, read_file]   # optional
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `agent-N` | Outer node id (the SubFlow). Inner nodes are `agent-<id>/...`. |
| `system` | string | `""` | System prompt for the agent. |
| `output_key` | string | `id` or `"output"` | State key that receives the final answer. |
| `model` | string | `default_model` | Required when no `default_model`. |
| `provider` | string | `default_provider` | Required when no `default_provider`. |
| `sections` | map | `{output_key: Capitalized}` | Shared state key → label rendered into the agent's context. |
| `messages_key` | string | `"messages"` | State key holding the shared conversation. |
| `use_tools` | `null`/list/`"all"` | `null` | Tool scope for the agent. `tools:` is the legacy alias. |
| `stream` | bool | `true` | Emit tokens as stream events. |
| any other key | — | — | Forwarded to the ReAct harness / `ToolExec` (e.g. `max_tool_rounds`, `tool_error_mode`). |

`agent_step` is also valid as a *node-level* step (see [node-level steps](#node-level-steps)).

---

## `team:`

Compiles a supervised team in one step: a [`Supervisor`](../reference/nodes.md#supervisor)
decider plus one routed agent per role, wired into the supervisor loop. The
programmatic twin of `Flow.team(...)`.

```yaml
- team:
    id: lead                       # names the supervisor node
    leader:
      system: "Route to planner or coder, then finish."
      model: llama3.1:8b
      provider: ollama
    roles:
      planner: {system: "You plan.", output_key: plan}
      coder:   {system: "You code.", output_key: code, use_tools: [python_eval]}
    fallback: planner
    max_rounds: 6
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `supervisor` | Supervisor node id. |
| `leader` | mapping | — | **Required.** Decider options (below). |
| `roles` | mapping | — | **Required, non-empty.** Role name → role spec (below). |
| `fallback` | string | `""` | Role routed to when `finish` is picked before anything is produced. |
| `max_rounds` | int | `6` | Force `finish` once the counter reaches it. |
| `finish` | step/list | — | Chain run when the decider replies `finish`. |

### `leader:`

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `system` | string | `""` | Decider system prompt (list the reply values + `finish`). |
| `model` | string | `default_model` | Decider model. |
| `provider` | string | `default_provider` | Decider provider. |
| `messages_key` | string | `"messages"` | Shared conversation key. |
| `route_keys` | map | `{role: output_key}` | Route value → output slot; a role whose slot already has content is not re-routed. |
| `done_keys` | list | `[]` | When these slots are filled, return `finish` without a model call. |
| `done_mode` | string | `"all"` | `"all"` (every `done_keys` filled) or `"any"`. |
| `fallback` | string | — | Same as the team-level `fallback:`. |
| `max_rounds` | int | `6` | Same as the team-level `max_rounds:`. |

### `roles.<name>:` (role spec)

Each role is an [`AgentRole`](../api/teff.flow.agent.md) mapping — a dict
recipe accepted for YAML parity. A role may also be a plain node/list of
nodes for that route.

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `system` | string | `""` | Role system prompt. |
| `output_key` | string | role name | State key receiving the role's final answer. |
| `model` / `provider` | string | team default | Per-role overrides. |
| `sections` | map | `{output_key: Capitalized}` | Context sections. |
| `messages_key` | string | `"messages"` | Shared conversation key. |
| `use_tools` | `null`/list/`"all"` | `null` | Tool scope. `tools:` accepted as legacy alias. |
| `stream` | bool | `true` | Emit tokens as stream events. |
| any other key | — | — | Forwarded to the role's ReAct harness. |

---

## `supervisor:`

Compiles a native [`Supervisor`](../reference/nodes.md#supervisor) decider node — **without**
the team wrapper. With an `agents:` mapping plus `finish:`/`key:` it wires the
whole supervisor loop in one step (the `team:` shape without the sugar).

```yaml
# Just the decider (pair with a `supervise:` step):
- supervisor:
    id: lead
    system: "Route to coder or finish."
    model: llama3.1:8b
    provider: ollama
    route_keys: {coder: code}
    done_keys: [code]

# Decider + loop wiring in one step:
- supervisor:
    id: lead
    system: "Route to coder or finish."
    route_keys: {coder: code}
    done_keys: [code]
    agents:
      coder: [agent_step: {id: coder, system: "You code.", output_key: code}]
    finish:
      - transform: {id: done, action: now, output_key: delivered_at}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `supervisor_N` | Decider node id. |
| `system` | string | `""` | Decider system prompt. |
| `model` / `provider` | string | default | Decider model/provider. |
| `output_key` | string | `"next_agent"` | State key receiving the chosen route. |
| `sections` | map | `{}` | State key → label map rendered into the prompt. |
| `route_keys` | map | `{}` | Route value → output slot; filled slots are not re-routed. |
| `done_keys` / `done_mode` | list/str | `{}` / `"all"` | When filled → `finish` with no model call (`"any"` = just one). |
| `fallback` | string | `""` | Alias for `fallback_agent`: route to this agent when `finish` is picked early. |
| `rounds_key` / `max_rounds` | str/int | `"supervisor_rounds"` / `6` | Force `finish` once the counter reaches `max_rounds`. |
| `messages_key` | string | `"messages"` | Source of the user message; `""` always consults the model. |
| `agents` | map | — | Route value → chain (node/list). Present → the full loop is wired. |
| `key` | string | `output_key` | The route key for the loop (defaults to `output_key` when `agents:` present). |
| `finish` | step/list | — | Chain run on `finish` (requires `agents:`). |

---

## `supervise:`

Routes an existing decider to agent chains — the low-level twin of the route
wiring `team:` does implicitly. Use it when the decider and its agent groups
are defined separately (full control over `fill_order`, `sections`, …). The
decider (the node added just before) writes `key`; each `agents` entry maps a
value of `key` to the chain run for it, after which control returns to the
decider. When `key` equals `finish` the loop exits through `finish`.

```yaml
- supervisor: {id: lead, system: "Route to coder or finish.", route_keys: {coder: code}}
- supervise:
    key: next_agent
    agents:
      coder: [agent_step: {id: coder, system: "You code.", output_key: code}]
    finish:
      - transform: {id: done, action: now, output_key: delivered_at}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `key` | string | — | **Required.** Route key written by the decider. |
| `agents` | map | — | **Required, non-empty.** Route value → chain (node/list). |
| `finish` | step/list | — | Chain run when `key == "finish"`. Omit to terminate on `finish`. |

---

## `parallel:`

Runs independent branches concurrently and merges their results via the state
reducers. Each branch is a node-level step or a list of them (run sequentially
within the branch). An optional `converge:` re-joins the branches.

```yaml
- parallel:
    id: fanout
    branches:
      - transform: {id: web, action: uppercase, input_key: q, output_key: web}
      - [transform: {id: upper, action: uppercase, input_key: q, output_key: upper},
         transform: {id: n, action: count_lines, input_key: q, output_key: n}]
    converge:
      transform: {id: join, action: value, value: done, output_key: status}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `parallel` | Parallel node id. |
| `branches` | list | — | **Required, non-empty.** Branch chains (node-level step, or list of node-level steps). |
| `converge` | mapping | — | Single-key `transform:` step that merges branch ends. Only `Transform` is supported in this release. |

Branches run on isolated copies of the state; use
`state.schema … reducer: append` when branches should accumulate (e.g.
collecting `messages`) instead of overwriting a key.

---

## `map:`

Dynamically fans a state **list** out across parallel branches — branch count
is derived from the data at runtime. Each item is passed to the `processor`
node, and the per-item results are gathered into a list at `output_key`.

```yaml
- map:
    id: summerize
    processor:
      llm: {model: llama3.1:8b, system: "Summarize the chunk.", input_key: chunk, output_key: summary}
    input_keys: [chunks]
    output_key: summaries
    max_concurrency: 4
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `map_N` | Map node id. |
| `processor` | mapping | — | **Required.** A single-key node-level step (`llm:`, `transform:`, `agent:`, …). |
| `input_keys` | string/list | `""` | State list key(s); multiple are zipped per index. |
| `output_key` | string | `""` | State key receiving the list of per-item results. |
| `max_concurrency` | int | — | Cap on concurrent branches. |
| `chunk_size` | int | — | Items per branch (default 1). |

---

## `loop:`

Repeats a `body` chain until `state[key]` equals `until`, then runs `done`
and continues after the loop. The condition uses the edges expression language
(so `until: "да"` matches `"Да"` or `"да."`).

```yaml
- loop:
    key: verdict
    until: pass
    max_rounds: 3
    body:
      - llm: {id: revise, system: "Rewrite based on feedback.", prompt: "{feedback}", output_key: draft}
    done:
      - transform: {id: final, action: now, output_key: delivered_at}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `key` | string | — | **Required.** State key the condition reads. |
| `until` | string | — | **Required.** Value of `key` that stops the loop. |
| `body` | list | — | **Required.** Chain repeated while the loop continues. |
| `done` | step/list | — | **Required.** Chain run when the loop terminates. |
| `max_rounds` | int | — | Maximum body re-runs before giving up (bounded-loop node form). Without it the loop is free-flow until the runtime gives up. |
| `label` | string | — | Route label attached to the loop decider, so a later `route:`/`command:` `goto` can jump back to the loop. |

---

## `interrupt:`

Pauses the run for human input at this point. `graph.run()` raises
`GraphInterrupt`; resume with the same `checkpoint_id` and `resume={key: answer}`.
Requires a checkpointer.

```yaml
- interrupt:
    id: approve
    key: decision
    prompt: "Approve the report? (yes / no)"
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `interrupt_N` | Node id; also prefixes the classifier/validate ids (`{id}-classifier`, `{id}-validate`). |
| `key` | string | — | **Required.** State key receiving the resume value. |
| `prompt` | string | `""` | Question shown to the operator. |
| `strategy` | mapping | — | Validate the answer instead of comparing verbatim (below). |

### `strategy:` — validated interrupts

Without `strategy:` the raw resume value is stored verbatim in `key`. With a
strategy the loader expands the interrupt into the classifier + `validate`
chain (`{id}-classifier`, `{id}-validate`) — the YAML counterpart of
`flow.interrupt(key, prompt, accept=Ask(...))` — and the decision key
(`decision` by default) is written for edges to route on.

| Strategy | Value | Meaning |
| -------- | ----- | ------- |
| `equals:` | string | Exact match on the normalized answer. |
| `any_of:` | list | Answer must be one of the values. |
| `regex:` | string | Answer must match the pattern. |
| `llm:` | mapping | An LLM classifier normalizes free-form answers into a verdict (needs `model`/`provider`). |

Common strategy passthrough keys: `decision_key` (`"decision"`), `pass_value`
(`"да"`), `fail_value` (`"нет"`), `value_key`, `verdict_key` (`"verdict"`),
`ok_field` (`"ok"`), `clear_field`, `clarify_value`, `rounds_key`,
`max_rounds`.

```yaml
- interrupt:
    id: gate
    key: approved
    prompt: "Approve the report? (yes / no)"
    strategy: {equals: да}                  # or any_of: [да, ок] | regex: "^[A-Z0-9]{4}$"
- transform: {id: ship, action: value, value: shipped, output_key: status}
```

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
`{id}-validate`, where the decision key is written.

---

## `branch:`

Conditional routing from the preceding node based on a state key. Each case
creates an edge `key=<value>`; an optional `default` catches unmatched values
(`key!=<all case values>`); an optional `converge` re-joins the branches.

```yaml
- branch:
    key: sentiment
    cases:
      - {value: positive, steps: [transform: {action: value, value: glad, output_key: reply}]}
      - {value: negative, steps: [transform: {action: value, value: sorry, output_key: reply}]}
    default:
      transform: {action: value, value: neutral, output_key: reply}
    converge:
      transform: {action: uppercase, input_key: reply, output_key: result}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `key` | string | — | **Required.** State key to evaluate. |
| `cases` | list | — | **Required, non-empty.** Each case is `{value, steps}` where `steps` is a list of node-level steps chained for that case. |
| `default` | step/list | — | Chain run when `key` matches no case. A list is wrapped as a sub-flow. |
| `converge` | mapping | — | Single-key node step merging branch ends. |

---

## `route:`

Declarative `goto`/`STOP` routing from state — a
[`CommandNode`](../reference/nodes.md#command) that routes the graph without code. The
`when` expressions use the same language as `edges:` conditions; the first
match wins.

```yaml
- route:
    id: route
    routes:
      - {when: "score >= 0.8", goto: approve}
      - {when: "score < 0.3", goto: reject}
    goto: STOP
    update: {routed: true}
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `command_N` | Node id. |
| `key` | string | — | **Required.** State key whose value is evaluated (used by the `when` expressions). |
| `routes` | list | `[]` | `{when, goto}` pairs; first match wins. `goto` may be a label or node id. |
| `goto` | string | — | Fallback target when no route matches; `STOP` ends the run. |
| `update` | mapping | — | State keys merged after routing. |

`goto` targets are resolved through [`label()`](../api/teff.flow.flow.md) labels,
so a route can jump back to a loop's decision point (pair with a `loop:` that
declares `label:`).

---

## `type:`

Compile an arbitrary registered node type as one step — the escape hatch to any
node the [registry](../reference/nodes.md#registering-custom-types) knows, without an idiom.

```yaml
- type:
    id: parse
    type: csv
    config:
      path: data.csv
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `id` | string | `{type}_N` | Node id. |
| `type` | string | — | **Required.** Registered node type name. |
| `config` | mapping | `{}` | Node config forwarded to the node's constructor. |

---

## Node-level steps

Idioms inside *containers* (loop `body`/`done`, parallel `branches`, map
`processor`, branch cases) are given as inline single-key steps and support a
slightly smaller surface:

`llm:`, `transform:`, `context_builder:`, `append_assistant:`,
`agent:`/`agent_step:`, `interrupt:`, `map:`, `type:`

```yaml
- loop:
    body:
      - llm: {model: llama3.1:8b, system: "Revise.", output_key: draft}
```

An `interrupt` with a `strategy:` used inside a container expands to the same
classifier + `validate` chain as at the top level.

A `supervisor:`/`supervise:`/`team:` step with its own `agents:` also embeds
node-level chains (e.g. `agent_step:`).

---

## `providers:`

Exactly as in `graph.yaml` — the single source of truth for model endpoints.
See [providers](../reference/providers.md) and the [graph.yaml reference](yaml-workflows.md#custom-providers).

```yaml
providers:
  - name: ollama
    type: ollama
    base_url: http://localhost:11434
    chat_path: /api/chat
  - name: vllm
    base_url: http://vllm:8000/v1           # type defaults to openai_compatible
```

| Key | Type | Default | Notes |
| --- | ---- | ------- | ----- |
| `name` | string | — | Unique provider key referenced by `provider:`/`default_provider:`. |
| `type` | string | `openai_compatible` | `openai_compatible`, `anthropic_compatible`, or `ollama`. |
| `base_url` | string | — | Endpoint base URL. |
| `chat_path` | string | — | Chat endpoint path override. |
| `api_key_env` | string | — | Env var holding the API key. |
| `auth_header` / `auth_prefix` | string | — | Custom auth header / prefix. |
| `timeout` | number | — | Request timeout. |

---

## `tools:`

Exactly as in `graph.yaml`. Tool instances made available to agents (and
`tool_call` nodes). See [tools](../reference/tools.md) for the full list.

```yaml
tools:
  - type: web_search
  - type: python_eval
  - type: mcp
    config:
      id: drive
      command: [uvx, mcp-server-google-drive]
```

`type: mcp` is a special tool — it declares an [MCP](https://modelcontextprotocol.io)
server (streamable-http `url:` or stdio `command:`) whose tools are exposed
to agents as `<id>__<tool>`. The connection opens lazily on first use and
lives for the whole graph, so it is shared across daemon ticks and
conversation turns. Exactly one of `url`/`command` is required; optional
`env`, `cwd` and `client_info` are passed through. Run the flow with
`teff run file.yaml` (the CLI closes the connection when the run ends) or
wrap your own call in `async with graph:`.

For known servers, a `preset:` supplies the launch command and env keys;
your `env:` merges over the defaults:

```yaml
tools:
  - type: mcp
    config:
      preset: google_drive   # npx: google_drive | gmail | google_calendar
      env: {GOOGLE_DRIVE_REFRESH_TOKEN: "${GDRIVE_TOKEN}"}
  - type: mcp
    config:
      preset: git            # uvx (Python): git | fetch | time | sqlite
```

A `preset` config needs no `url`/`command`. Explicit `command:`/`url:`,
`id:` or extra `env:` keys override the preset (env merges key-by-key).
Google presets launch via `npx` and need Node.js; the `uvx` presets are
pure Python. A missing launcher fails the connection with a clear error.

---

## `state:`

Exactly as in `graph.yaml`. `initial:` seeds the state; `schema:` declares
per-key [reducers](state.md) (`override` / `append` / `keep`).

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

`state.initial` is validated against `state.schema` (`ConfigError` on
mismatch); keys without a `reducer` default to `override`.

---

## `checkpoint:`

Exactly as in `graph.yaml`. Enables durable runs. See [durable](durable.md).

```yaml
checkpoint:
  type: sqlite              # file | sqlite | sqlite_history | pg | pg_history
  path: data/checkpoints.db
```

`path` resolves relative to the workflow file; PG variants require `dsn:`
(+ optional `table:`).

---

## `hooks:`

Exactly as in `graph.yaml` — named hook callbacks registered in a plugin.
See [yaml-workflows.md](yaml-workflows.md#hook-events-hooks).

```yaml
plugins: [plugins/telemetry.py]
hooks:
  on_node_start: tick
  on_node_end: [tick, finalize]
  on_node_error: on_error
```

---

## `observability:`

Exactly as in `graph.yaml` — full-run tracing without code. See
[yaml-workflows.md](yaml-workflows.md#tracing-a-workflow-observability).

```yaml
observability:
  db: ./data/traces.db
  export:
    - type: webhook
      url: http://obs:8001/obs/ingest
```

---

## Validate, compile, export

```bash
teff validate flow.yaml           # flow-specific validation (auto-detected)
teff graph flow.yaml              # compiled topology as YAML
teff graph flow.yaml --mermaid    # compiled topology as Mermaid
teff -f flow.yaml                 # run it
```

```python
from teff.flow.compiler import load_flow, compile_flow_file

graph, tools, initial, reducers = load_flow("flow.yaml")
graph2 = compile_flow_file("flow.yaml")  # just the graph

from teff.flow.compiler import build_flow_to_yaml

text = build_flow_to_yaml("flow.yaml", output="graph.yaml")  # compile artifact
```

The compiled `graph.yaml` round-trips through `teff validate` and the low-level
loaders — the authoring layer never produces anything a hand-written
`graph.yaml` could not express.

## See also

- [`graph.yaml` reference](yaml-workflows.md) — the low-level format, key by key.
- [Choosing your abstraction](choosing-your-abstraction.md) — when to use which layer.
- [`Flow` API](../api/teff.flow.flow.md) — the programmatic twin of every idiom.
- [`teff.flow.compiler`](../api/teff.flow.compiler.md) — the compiler itself.
