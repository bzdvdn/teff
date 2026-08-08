# Core entities — how to use them

A developer-facing map of the objects you'll touch every day: what each entity
*is*, where to import it, and the minimal way to use it. Terms you meet here are
defined together at the end in a short glossary.

## The mental model in one line

> A **workflow is a directed graph of nodes**, moving a flat **state** forward.
> The **Flow** builder and **YAML** are two ways to describe that graph; both
> compile to a **`Graph`** — the spectator is nothing but a graph runner.

Everything below hangs off that sentence.

## Entities at a glance

| Entity        | What it is                                    | Import                    | Used in          |
| ------------- | --------------------------------------------- | ------------------------- | ---------------- |
| `Node`        | one step: `async fn(ctx, state) -> dict`      | `teff` (built-ins) / subclass or `@node` | all graphs      |
| `Edge`        | a directed arrow (optionally `condition`-guarded) | `teff.graph` / `teff`  | graph definition |
| `Graph`       | nodes + edges + entry point; the runnable     | `teff.graph` / `teff`   | the runner      |
| `State`       | typed dict with per-key merge **(reducers)**  | `teff` / `teff.state`   | every run      |
| `Flow`        | chainable builder that **compiles to a `Graph`** | `teff.flow` / `teff`  | authoring      |
| `Tool`        | an external capability (web, sql, slack, your own) | `teff`   | agent/tool nodes |
| `Checkpointer`| durable persistence for pause/resume          | `teff.checkpoint`       | durable runs   |
| `Provider`    | an LLM backend (ollama, openai, …) + a `model` | `teff` (registry)       | `LLM`/agent nodes |
| `Command`     | a node-to-node routing directive at runtime   | `teff`                  | dynamic routing |
| `Reducer`     | how a state key merges (e.g. `append`)       | `teff.state`             | state schema |

## How the pieces fit together

```
                ┌── writing (choose one) ──────────────┐
   flow = Flow(...).transform(...).branch(...)   OR   workflow.yaml
                └──────────────┬───────────────────────┘
                               ▼  (compile / load)
                         GRAPH (nodes + edges + entry)
                               │ *.run(state, tools=…,
                               │        checkpointer=…, providers=…)
          ┌────────┬───────────┴────────────┬────────────┐
          ▼        ▼                        ▼            ▼
       nodes    edges(filters)       reducers/State   tools providers
```

Three authoring layers produce the same runtime `Graph`:

| Layer     | What you write                    | Compile call              |
| --------- | --------------------------------- | ------------------------- |
| YAML      | data (`steps:`, `edges:`)         | `load_workflow(...)` / `teff run` |
| Flow DSL  | chainable methods (`transform`, `branch`, `route`, `loop`) | `flow.compile()` |
| Low-level | `Graph(nodes=…, edges=[…], entry_point=…)` | nothing — it *is* the graph |

## Working with each entity

### `Node`
A step. Built-in types: `Transform`, `LLM` (`llm_chat`), `ReActAgent` /
`Harness` (`react_agent`), `ToolExec`, `Interrupt`, `Map`, `Parallel`,
`Supervisor`, `Retry`, `Ask`/`Validate`, `Extract`/`Fallback`. A node maps its
`output_key`s onto state.

**Your own node** (a plain async function) — auto-wrapped with
`make_function_node` when passed to `Flow`/`Graph`:
```python
async def greet(ctx, state):
    return {"greeting": f"Hello {state.get('name', 'world')}"}
```

### `Edge`
```python
Edge("count", "single", "lines=1")  # conditional
Edge("single", "status")  # unconditional
```
`__error__` is a special condition that fires when a node raises (see
[errors](errors.md)).

### `State` and reducers
In-memory run state is a plain dict at runtime. For typed, reducible state use
`State`:
```python
class TriageState(TypedDict):
    messages: Annotated[list, add_messages]
    status: str


state = State(TriageState, {"status": "open"})
state.merge({"messages": ["hi"]})
```
The reducers here **are** the merge strategy — pass `reducers=` to `run` for a
plain dict, or to a *set* state's own reducers for a `State`. See
[state](state.md).

### `Flow`
```python
flow = (
    Flow("hello")
    .transform(
        action="count_lines",
        input_key="text",
        output_key="lines",
    )
    .branch(
        "lines",
        Case("1").add(Transform(action="value", value="single", output_key="note")),
        default=Transform(action="value", value="multi", output_key="note"),
    )
    .converge(Transform(action="value", value="done", output_key="status"))
)

graph = flow.compile()
```

### `Tool`
```python
from teff.tool.registry import tool


@tool("my_lookup")
async def my_lookup(query: str, limit: int = 10) -> str: ...
```
Pass to the run with `tools=[my_lookup]` (each agent/tool node can use them).
See [plugins](plugins.md) and [agents](agents.md).

### `Checkpointer` + `Interrupt`
Make a run durable and pausable for a human:
```python
from teff.checkpoint import JSONFileCheckpointer

cp = JSONFileCheckpointer("data/cp")
try:
    result = await graph.run(state=state, checkpointer=cp, checkpoint_id="s1")
except GraphInterrupt as exc:
    result = await graph.run(
        state=state,
        checkpointer=cp,
        checkpoint_id="s1",
        resume={"decision": input(exc.prompt)},
    )
```
First `run` save a checkpoint before **every** node; the second resumes from
the pending `Interrupt`. See [durable](durable.md).

### `Provider` resolution
LLM nodes look up a provider + model. Supply them on the `Graph`/`Flow`
(defaults), per node, or at `run` time via `providers=` / `default_provider=`.
See [providers](../reference/providers.md).

## Glossary

| Term   | Mean                                       |
| ------ | ------------------------------------------ |
| graph  | me nodes + edges (a DAG up to intentional cycles) |
| state  | the flat, JSON-serializable info a run carries forward |
| reducer | per-key merge rule (e.g. append) applied when a node returns |
| durable     | uses a checkpointer across interrupts/crashes |
| resume | continue a durable run from a checkpoint (e.g. after an interrupt) |
| route | have a decision node pick the next branch of nodes |
| tool  | an external capability a node can call |
| provider/model | the backend + name for LLM nodes |

## Where to go next

- Map node types to jobs: [nodes reference](../reference/nodes.md) and
  [CLI cheat-sheet](../getting-started/cli-cheatsheet.md).
- Build a real one end-to-end: [tutorial](../getting-started/tutorial.md) → the
  runnable [hello_workflow](https://github.com/bzdvdn/teff/tree/master/examples/hello_workflow/).
- Add your own node/tool: [plugins](plugins.md).
- Lifecycle of a run: [anatomy of a run](../getting-started/anatomy-of-a-run.md).