# Flow builder (full API)

`Flow` is the ergonomic, chainable way to assemble a `Graph` in Python. Every
method returns `self`, so a whole workflow compiles in one expression. The
result of `flow.compile()` is a `Graph` you can run or stream, and
`flow.to_yaml()` exports it as a deployable workflow.

```python
from teff.flow import Flow, Case
from teff.node import LLM, Transform
from teff.provider import ProviderRegistry

flow = Flow(
    "my-flow",  # optional name
    providers=ProviderRegistry.from_presets("ollama"),
    default_provider="ollama",
    default_model="llama3.1:8b",
)
flow.step(LLM(output_key="answer"))
result = await flow.compile().run(state={"..."})
```

LLM nodes fall back to the graph-level `default_provider=` / `default_model=`
when they don't set one — every provider a graph uses must be declared in
`providers=` (see [Providers](../reference/providers.md)).

## Linear chain

### `step(node)`

Append a node to the linear chain. Accepts any `Node` — built-in or custom:

```python
flow.step(Transform(action="uppercase"))
flow.step(LLM(model="gpt-4"))
flow.step(custom_node)
```

### Conditional steps

`step(..., when=...)` guards a node behind a predicate on the state: it only runs
when the callable (or string condition) matches. Follow it with `default(node)`:

```python
flow.step(decider)
flow.step(handle_yes, when=lambda s: s["ok"])
flow.default(handle_no)  # runs when the guard fails
```

The guarded step's edge wins on a match, so `default()` fires only when the
guard misses. `default()` requires a preceding guarded step to be useful;
for branch fallbacks use `branch(..., default=node)` instead.

### `llm(...)` and `transform(...)`

Shorthands for the two most common node classes. Accept either a pre-built
instance (to reuse a shared node) or keyword config:

```python
flow.llm(model="gpt-4", system="You are helpful", output_key="answer")
flow.llm(LLM(model="gpt-4", parse=True, output_key="data"))

flow.transform(action="uppercase", input_key="text", output_key="shout")
flow.transform(Transform(action="value", value="done", output_key="status"))
```

Passing both an instance and kwargs raises `TypeError`.

### `context_builder(...)` and `append_assistant(...)`

Shorthands for the turn-routing pair used by `agent_step`: compose a
plain-text `input` from state (each section as `<label>:\n<value>` plus the
latest user message), then append the agent's reply back to the shared
conversation:

```python
flow.context_builder(
    sections={"plan": "Plan", "summary": "Summary"},
    messages_key="messages",
    output_key="input",
)
flow.append_assistant(output_key="draft", messages_key="messages")
```

`context_builder` accepts `sections`, `messages_key`, `output_key` and
`reset_keys` (scratch state cleared before the agent runs); both also accept
a pre-built instance instead of kwargs.

### `add_flow(flow, ...)` / `SubFlow`

Embed a whole sub-flow as a single node. The inner `Flow` is compiled and
wrapped in a `SubFlow`:

```python
inner = Flow("inner")
inner.step(LLM(model="gpt-4", output_key="tmp"))

outer = Flow("outer")
outer.step(prepare_node)
outer.add_flow(inner, input_map={"user": "question"}, output_map={"tmp": "result"})
outer.step(use_result_node)
```

`SubFlow` keys:

| Key | Description |
| --- | ----------- |
| `input_map` | Parent state key → sub-graph key (default: full passthrough). |
| `output_map` | Sub-graph key → parent state key (default: full passthrough). |
| `max_iterations` | Max node executions inside the sub-graph (`None` = unlimited). |

The sub-graph runs on isolated copies and streams its node/token/llm events
through (its own `run_start`/`run_end` are stripped). Tools from the outer
context are forwarded to the inner run.

## Branching

### `branch(key, *cases, default=...)` + `Case`

Conditional routing from the last added node, based on a state key:

```python
flow.branch(
    "sentiment",  # state key to inspect
    Case("positive").add(on_pos_llm),  # when sentiment == "positive"
    Case("negative").add(on_neg_llm),  # when sentiment == "negative"
    default=fallback_llm,  # anything else
).converge(Transform(action="uppercase", input_key="reply", output_key="result"))
```

- Each `Case(value)` produces an edge `key=value`; a `Case` can hold several
  chained nodes via repeated `.add(node)`.
- `default=` adds an edge `key!=<all case values>`; it is the only way to
  add a branch fallback. The separate `default(node)` method is reserved
  for guarded steps (see [`step(..., when=...)`](#conditional-steps) below).

### `converge(node)`

Merge every branch end into a single node. Call it after `branch()` or
`parallel()` to rejoin paths before continuing the chain.

### `loop(key, until, done=..., body=...)`

Repeat a `body` chain until `state[key] == until`, then run `done` and
continue after the loop:

```python
flow.step(draft_llm)
flow.interrupt("approved", "Одобрить? (да / правки)")  # decider
flow.loop(
    key="approved",
    until="да",
    done=final_llm,
    body=edit_llm,
)
```

Wires `decider --key=until--> done` (stop) and
`decider --key!=until--> body -> ... -> decider` (repeat). The decider is any
node that writes `key` — an `Interrupt`, an `LLM`, a `Transform`, etc.

### `route(key, *, finish=..., **agents)`

The supervisor loop — route between agent chains, looping back to a decider
until it says `finish`. See [Multi-agent supervisors](supervisors.md).

### Dynamic routing with `Command`

Edges are static — but any node can **pick its own successor at runtime** by
returning a `Command` with an explicit `goto`:

```python
from teff.node import Command


async def classify(ctx, state):
    if "bad" in state["text"]:
        return Command(update={"blocked": True}, goto=Command.STOP)
    if "trusted" in state["text"]:
        return Command(update={"cleared": True}, goto="deliver")
    return {"cleared": True}  # fall through to normal edges
```

`goto` may target *any* node id — it does not need an edge — and
`Command.STOP` ends the run. `Command(update=...)` without `goto` keeps the
normal edge routing. See [Command routing](commands.md) and the
[`command_routing` example](../examples.md).

## Concurrency

### `parallel(*branches)`

Run branch chains concurrently from the last node via `asyncio.gather`, on
isolated copies of state. Per-key reducers merge their updates back:

```python
flow.parallel(
    [Transform(action="uppercase", input_key="a", output_key="a")],
    [Transform(action="uppercase", input_key="b", output_key="b")],
).converge(Transform(action="value", value="done", output_key="status"))
```

A branch is a single `Node`, a list of nodes (sequential inside the branch),
or a `Flow` (embedded as `SubFlow`). The underlying node works directly too:
`Parallel([[node1], [node2]])`.

### `map(processor, *, input_keys, output_key, chunk_size=..., max_concurrency=...)`

Dynamically fan a state **list** out across parallel branches — branch count
comes from the data at runtime:

```python
flow.map(
    LLM(model="llama3.1:8b", input_key="chunk", output_key="summary"),
    input_keys=["chunks"],  # list key(s); multiple are zipped per index
    output_key="summaries",  # list of per-item results
    max_concurrency=2,
)
```

`chunk_size` batches items per branch, `result_key` overrides which per-item
key is collected.

## Human input

### `interrupt(key, prompt="", accept=None)`

Pause for a human. `graph.run()` raises `GraphInterrupt`; resume with the same
`checkpoint_id` and `resume={key: answer}`. Requires a checkpointer. See
[Durable execution](durable.md).

With `accept` (an [`Ask`](../reference/nodes.md#ask) strategy) the raw answer
is validated instead of compared verbatim — and an arbitrary value can be
captured. `flow.interrupt(key, prompt, accept=Ask.regex(...))` validates a
single answer; `flow.interrupt_loop(key, accept=Ask(...), body=..., done=...)`
re-asks until it passes:

```python
from teff.node import Ask

flow.interrupt_loop(
    key="code",
    prompt="Введите промокод (формат XX-1234):",
    accept=Ask.regex(
        r"^[A-Z]{2}-[0-9]{4}$", decision_key="code_ok", value_key="discount_code"
    ),
    body=Transform(action="value", value="неверный код", output_key="total"),
    done=Transform(action="value", value="скидка применена", output_key="total"),
)
```

`Ask.llm(...)` inserts an LLM classifier so free-form answers ("конечно",
"ок", "хорошо") all count as approval. See the
[`ask_strategies` example](../examples.md) and the
[`validate` reference](../reference/nodes.md#validate).

## Agents

### `harness(...)` / `react(...)`

Build the ReAct loop (agent node ↔ tool executor) inside the flow. See
[Agents](agents.md) and [Supervisors](supervisors.md).

### Custom agent node

Pass a pre-built `ReActAgent` instance or subclass to override behaviour:

```python
flow.react(agent=MyAgent(model="gpt-4", system="..."))
flow.react(agent=MyAgentClass, model="gpt-4", system="...")
```

## Export & run

### `compile()`

Compile into a `Graph` ready for `run()`/`stream()`. Raises `ValueError` if
no nodes were added.

### `to_yaml(tools=..., initial=..., reducers=...)`

Export the compiled flow as a `workflow.yaml` document — including the ReAct
loop wiring. `Flow` does not track tools/state, so pass them explicitly:

```python
from teff.flow import Flow
from teff.provider import ProviderRegistry
from teff.tool.builtin.git import GitTool

yaml_text = (
    Flow(
        "repo",
        providers=ProviderRegistry.from_presets("ollama"),
        default_provider="ollama",
        default_model="llama3.1:8b",
    )
    .react(use_tools="all")
    .to_yaml(tools=[GitTool()])
)
```

The result validates with `teff validate` and round-trips through
`teff.yaml.load_workflow`.

## Quick reference

| Method | Builds | See |
| ------ | ------ | --- |
| `step(node)` | one node in the chain | [Nodes](../reference/nodes.md) |
| `llm(...)` / `transform(...)` | shorthand for `LLM`/`Transform` | [Nodes](../reference/nodes.md) |
| `context_builder(...)` / `append_assistant(...)` | turn routing pair | [Nodes](../reference/nodes.md) |
| `supervisor(...)` | supervisor decider | [Supervisors](supervisors.md) |
| `add_flow(flow)` | nested `SubFlow` node | — |
| `branch(key, *cases, default=...)` | conditional edges | [State](state.md) |
| `default(node)` / `converge(node)` | guard fallback / rejoin | — |
| `loop(key, until, ...)` | repeat-until cycle | [Durable](durable.md) |
| `route(key, **agents)` | supervisor loop | [Supervisors](supervisors.md) |
| `return Command(goto=...)` | dynamic per-node routing | [Command routing](commands.md) |
| `parallel(*branches)` | concurrent branches | [State](state.md) |
| `map(processor, ...)` | dynamic fan-out | [State](state.md) |
| `interrupt(key, prompt, accept=...)` | human-in-the-loop | [Durable](durable.md) |
| `interrupt_loop(key, accept, body, done)` | re-ask until the answer passes | [Durable](durable.md) |
| `react(...)` / `harness(...)` | ReAct agent loop | [Agents](agents.md) |
| `compile()` | runnable `Graph` | — |
| `to_yaml(...)` | deployable workflow | [YAML workflows](yaml-workflows.md) |